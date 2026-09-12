"""
ConceptNet KG lookup (RA-KG-T2I building block #1).

Primary source: local SQLite DB (data/conceptnet.db, built by
build_conceptnet_db.py). The public api.conceptnet.io is kept only as an
optional fallback — it has been returning 502 for a while.

Scoring (first version): relation-type prior. The thesis scoring function
extends this with dataset co-occurrence and CLIP embedding similarity.

Run directly to test:  python src/kg_lookup.py dog
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data" / "conceptnet.db"

# Relation filtering policy from the proposal (Section 4.1, component 2)
RELATION_PRIORS = {
    "IsA": 1.0,
    "PartOf": 0.9,
    "AtLocation": 0.8,
    "UsedFor": 0.8,
    "HasA": 0.7,
    "MadeOf": 0.7,
    "CapableOf": 0.6,
    "Synonym": 0.6,
    "RelatedTo": 0.4,   # low prior: broad, noisy relation
}


def _connect() -> sqlite3.Connection:
    if not DB_FILE.exists():
        sys.exit(f"{DB_FILE} not found — run: python src/build_conceptnet_db.py")
    return sqlite3.connect(DB_FILE)


def neighbors(concept: str, con: sqlite3.Connection | None = None) -> list[tuple]:
    """All (relation, other_concept, direction) edges touching `concept`."""
    concept = concept.strip().lower().replace(" ", "_")
    own = con is None
    con = con or _connect()
    rows = con.execute(
        """SELECT rel, end, 'out' FROM edges WHERE start = ?
           UNION ALL
           SELECT rel, start, 'in' FROM edges WHERE end = ?""",
        (concept, concept)).fetchall()
    if own:
        con.close()
    return rows


def extract_expansions(concept: str, top_k: int = 10,
                       con: sqlite3.Connection | None = None) -> list[dict]:
    """Relation-filtered, prior-scored neighbor concepts for one seed concept.

    Outgoing edges score full prior; incoming edges are slightly discounted
    (e.g. 'puppy IsA dog' should expand 'dog' less than 'dog IsA pet' does).
    """
    seed = concept.strip().lower().replace(" ", "_")
    candidates = {}
    for rel, other, direction in neighbors(seed, con):
        prior = RELATION_PRIORS.get(rel, 0.0)
        if prior <= 0 or other == seed:
            continue
        score = prior if direction == "out" else prior * 0.8
        key = other.lower()
        if key not in candidates or score > candidates[key]["score"]:
            candidates[key] = {
                "concept": other.replace("_", " "),
                "relation": rel,
                "score": round(score, 3),
            }
    ranked = sorted(candidates.values(), key=lambda c: -c["score"])
    return ranked[:top_k]


if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "dog"
    print(f"Seed concept: {seed}")
    expansions = extract_expansions(seed)
    if not expansions:
        print("No expansions found (check spelling / rebuild DB).")
    for e in expansions:
        print(f"  {e['score']:6.3f}  [{e['relation']:<10}]  {e['concept']}")
