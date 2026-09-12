"""
RA-KG-T2I query expansion (thesis milestone 2026.04-05, first working version).

Pipeline per query:
  1. Seed extraction: caption unigrams/bigrams that exist in the local
     ConceptNet DB (stopword-filtered). Offline; spaCy NER can replace later.
  2. KG expansion: relation-filtered neighbors via kg_lookup (IsA, PartOf,
     AtLocation, UsedFor, ... with relation priors).
  3. Drift control: score = relation_prior x cosine(CLIP(caption), CLIP(concept));
     keep concepts above --sim-threshold, take global top-K.
  4. Enriched text: '{caption}, a scene with {c1}, {c2}, {c3}'
     (concise template keeps CLIP text encoder in-distribution).

Also exposes expand_batch() for the evaluation script.

Demo:  python src/expand_query.py "a man riding a horse on a beach"
"""
import re
import sqlite3
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_lookup import DB_FILE, extract_expansions  # noqa: E402

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with",
    "is", "are", "was", "were", "be", "being", "been", "by", "for", "from",
    "as", "his", "her", "its", "their", "this", "that", "these", "those",
    "there", "here", "it", "he", "she", "they", "we", "you", "i", "up",
    "down", "out", "over", "under", "near", "next", "while", "during",
    "who", "which", "what", "some", "two", "three", "four", "man", "woman",
    "men", "women", "people", "person", "young", "old", "one", "another",
}
# 'man/woman/people' excluded as seeds: too generic, expansions add noise.

_token_re = re.compile(r"[a-z]+")


def _connect():
    return sqlite3.connect(DB_FILE)


class QueryExpander:
    def __init__(self, model=None, device="cpu", top_k=3,
                 sim_threshold=0.5, max_seeds=4):
        self.con = _connect()
        self.model = model
        self.device = device
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.max_seeds = max_seeds
        # vocabulary of concepts that actually have edges (for seed matching)
        self._has_edges_cache: dict[str, bool] = {}

    def _in_kg(self, term: str) -> bool:
        if term not in self._has_edges_cache:
            row = self.con.execute(
                "SELECT 1 FROM edges WHERE start=? OR end=? LIMIT 1",
                (term, term)).fetchone()
            self._has_edges_cache[term] = row is not None
        return self._has_edges_cache[term]

    def extract_seeds(self, caption: str) -> list[str]:
        toks = _token_re.findall(caption.lower())
        content = [t for t in toks if t not in STOPWORDS and len(t) > 2]
        seeds, seen = [], set()
        # bigrams first (more specific), then unigrams
        for i in range(len(content) - 1):
            bg = f"{content[i]}_{content[i+1]}"
            if bg not in seen and self._in_kg(bg):
                seeds.append(bg); seen.add(bg)
        for t in content:
            if t not in seen and self._in_kg(t):
                seeds.append(t); seen.add(t)
        return seeds[:self.max_seeds]

    @torch.no_grad()
    def _clip_sim(self, caption: str, concepts: list[str]) -> list[float]:
        """cosine(CLIP(caption), CLIP(concept)) for drift filtering."""
        if self.model is None:
            return [1.0] * len(concepts)
        import clip
        texts = [caption] + concepts
        tok = clip.tokenize(texts, truncate=True).to(self.device)
        f = self.model.encode_text(tok).float()
        f /= f.norm(dim=-1, keepdim=True)
        return (f[1:] @ f[0]).tolist()

    def expand(self, caption: str) -> dict:
        seeds = self.extract_seeds(caption)
        cands = []
        for s in seeds:
            for e in extract_expansions(s, top_k=8, con=self.con):
                cands.append({**e, "seed": s})
        if not cands:
            return {"caption": caption, "seeds": seeds,
                    "concepts": [], "enriched": caption}

        # noise filters (thesis Problem 1: junk crowdsourced nodes)
        def is_clean(c):
            words = c["concept"].split()
            if len(words) > 3:                       # long phrases drift
                return False
            seed_word = c["seed"].replace("_", " ")
            if seed_word in c["concept"] and c["concept"] != seed_word:
                return False                          # 'horse of course' etc.
            return True
        cands = [c for c in cands if is_clean(c)]

        # dedupe (a concept reachable from 2 seeds keeps best prior)
        best = {}
        for c in cands:
            k = c["concept"]
            if k not in best or c["score"] > best[k]["score"]:
                best[k] = c
        cands = list(best.values())

        sims = self._clip_sim(caption, [c["concept"] for c in cands])
        for c, s in zip(cands, sims):
            c["sim"] = round(s, 3)
            c["final"] = round(c["score"] * s, 3)
        kept = [c for c in cands if c["sim"] >= self.sim_threshold]
        kept.sort(key=lambda c: -c["final"])
        kept = kept[:self.top_k]

        enriched = caption
        if kept:
            enriched = f"{caption}, a scene with " + ", ".join(
                c["concept"] for c in kept)
        return {"caption": caption, "seeds": seeds,
                "concepts": kept, "enriched": enriched}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "a man riding a horse on a beach"
    device = ("mps" if torch.backends.mps.is_available() else "cpu")
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    ex = QueryExpander(model=model, device=device)
    r = ex.expand(text)
    print(f"caption : {r['caption']}")
    print(f"seeds   : {r['seeds']}")
    for c in r["concepts"]:
        print(f"  final={c['final']:5.3f} (prior={c['score']:.2f} sim={c['sim']:.2f})"
              f" [{c['relation']:<10}] {c['concept']}  <- {c['seed']}")
    print(f"enriched: {r['enriched']}")
