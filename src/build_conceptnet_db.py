"""
Build a local ConceptNet SQLite DB (replacement for the flaky public API).

Pipeline:
  1. Download conceptnet-assertions-5.7.0.csv.gz (~1.2 GB) if not present
  2. Stream-decompress; keep only English<->English edges whose relation is
     in RELATION_WHITELIST (matches the RA-KG-T2I filtering policy)
  3. Insert into data/conceptnet.db with an index on the start concept

Resulting DB is ~100-300 MB and answers lookups in <1 ms — far better for
latency benchmarks than a remote API.

Usage:
    python src/build_conceptnet_db.py            # download + build
    python src/build_conceptnet_db.py --query dog  # test a lookup
"""
import argparse
import csv
import gzip
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

DUMP_URL = ("https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
            "conceptnet-assertions-5.7.0.csv.gz")
DUMP_FILE = DATA / "conceptnet-assertions-5.7.0.csv.gz"
DB_FILE = DATA / "conceptnet.db"

RELATION_WHITELIST = {
    "IsA", "PartOf", "AtLocation", "UsedFor",
    "HasA", "RelatedTo", "Synonym", "CapableOf", "MadeOf",
}


def download_dump():
    if DUMP_FILE.exists():
        print(f"Dump already present: {DUMP_FILE} "
              f"({DUMP_FILE.stat().st_size / 1e9:.2f} GB)")
        return

    def hook(blocks, block_size, total):
        done = blocks * block_size
        pct = done / total * 100 if total > 0 else 0
        if blocks % 2000 == 0:
            print(f"  {done / 1e6:8.1f} MB  ({pct:5.1f}%)", flush=True)

    print(f"Downloading {DUMP_URL} ...")
    urllib.request.urlretrieve(DUMP_URL, DUMP_FILE, reporthook=hook)
    print("Download complete.")


def parse_uri(uri: str):
    """'/c/en/dog/n' -> ('en', 'dog') ; returns (None, None) if not a concept."""
    parts = uri.split("/")
    if len(parts) >= 4 and parts[1] == "c":
        return parts[2], parts[3]
    return None, None


def build_db():
    if DB_FILE.exists():
        print(f"DB already exists: {DB_FILE} — delete it to rebuild.")
        return
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE edges (
            start TEXT NOT NULL,
            rel   TEXT NOT NULL,
            end   TEXT NOT NULL,
            weight REAL NOT NULL
        )""")

    kept = total = 0
    batch = []
    csv.field_size_limit(sys.maxsize)
    with gzip.open(DUMP_FILE, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            total += 1
            if total % 2_000_000 == 0:
                print(f"  scanned {total / 1e6:.0f}M edges, kept {kept}", flush=True)
            # row: uri, rel_uri, start_uri, end_uri, json_info
            rel = row[1].rsplit("/", 1)[-1]
            if rel not in RELATION_WHITELIST:
                continue
            s_lang, s_label = parse_uri(row[2])
            e_lang, e_label = parse_uri(row[3])
            if s_lang != "en" or e_lang != "en":
                continue
            try:
                weight = json.loads(row[4]).get("weight", 1.0)
            except json.JSONDecodeError:
                weight = 1.0
            batch.append((s_label, rel, e_label, weight))
            kept += 1
            if len(batch) >= 50_000:
                con.executemany("INSERT INTO edges VALUES (?,?,?,?)", batch)
                batch.clear()
    if batch:
        con.executemany("INSERT INTO edges VALUES (?,?,?,?)", batch)
    print(f"Kept {kept:,} of {total:,} edges. Building index ...")
    con.execute("CREATE INDEX idx_start ON edges(start)")
    con.execute("CREATE INDEX idx_end ON edges(end)")
    con.commit()
    con.close()
    print(f"Done: {DB_FILE} ({DB_FILE.stat().st_size / 1e6:.0f} MB)")


def query(concept: str, limit: int = 15):
    concept = concept.strip().lower().replace(" ", "_")
    con = sqlite3.connect(DB_FILE)
    rows = con.execute(
        """SELECT rel, end, weight FROM edges WHERE start = ?
           UNION ALL
           SELECT rel, start, weight FROM edges WHERE end = ?
           ORDER BY weight DESC LIMIT ?""",
        (concept, concept, limit)).fetchall()
    con.close()
    print(f"Neighbors of '{concept}':")
    for rel, other, w in rows:
        print(f"  {w:6.2f}  [{rel:<10}]  {other}")


PARQUET_DIR = DATA / "conceptnet_parquet"
CONCEPT_PREFIX = "http://conceptnet.io/c/"
RELATION_PREFIX = "http://conceptnet.io/r/"


def build_db_from_parquet():
    """Build the SQLite DB from HF parquet shards (CleverThis/conceptnet).

    NOTE: this RDF export has no edge weights -> weight defaults to 1.0;
    scoring downstream uses relation priors + embedding similarity instead.
    """
    import pyarrow.parquet as pq

    if DB_FILE.exists():
        print(f"DB already exists: {DB_FILE} — delete it to rebuild.")
        return
    shards = sorted(PARQUET_DIR.glob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet shards in {PARQUET_DIR}")

    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE edges (
            start TEXT NOT NULL,
            rel   TEXT NOT NULL,
            end   TEXT NOT NULL,
            weight REAL NOT NULL
        )""")

    kept = total = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=100_000,
                                     columns=["subject", "predicate", "object"]):
            subs = batch.column(0).to_pylist()
            preds = batch.column(1).to_pylist()
            objs = batch.column(2).to_pylist()
            rows = []
            for s, p, o in zip(subs, preds, objs):
                total += 1
                if not (p and p.startswith(RELATION_PREFIX)):
                    continue
                rel = p[len(RELATION_PREFIX):]
                if rel not in RELATION_WHITELIST:
                    continue
                if not (s and o and s.startswith(CONCEPT_PREFIX + "en/")
                        and o.startswith(CONCEPT_PREFIX + "en/")):
                    continue
                # '.../c/en/dog/n' -> 'dog'
                s_term = s[len(CONCEPT_PREFIX) + 3:].split("/")[0]
                o_term = o[len(CONCEPT_PREFIX) + 3:].split("/")[0]
                if s_term and o_term and s_term != o_term:
                    rows.append((s_term, rel, o_term, 1.0))
            if rows:
                con.executemany("INSERT INTO edges VALUES (?,?,?,?)", rows)
                kept += len(rows)
        print(f"  {shard.name}: cumulative kept {kept:,} / scanned {total:,}",
              flush=True)

    print("Building indexes ...")
    con.execute("CREATE INDEX idx_start ON edges(start)")
    con.execute("CREATE INDEX idx_end ON edges(end)")
    con.commit()
    con.close()
    print(f"Done: {DB_FILE} ({DB_FILE.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="test-query a concept against the built DB")
    ap.add_argument("--from-csv", action="store_true",
                    help="use original S3 csv.gz dump (slow to download in China)")
    args = ap.parse_args()
    if args.query:
        query(args.query)
    elif args.from_csv:
        download_dump()
        build_db()
    else:
        build_db_from_parquet()
