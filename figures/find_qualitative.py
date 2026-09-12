"""Find Flickr queries where relation-aware expansion fixes the top-1,
then render a KTIR-Fig.7-style qualitative panel (baseline vs ours, top-5)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import baseline_retrieval as br                      # noqa: E402
from ablations import load_queries                   # noqa: E402
from kg_expanded_eval import DATASETS                # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank,   # noqa: E402
                                build_text, ranks_of)

cfg = DATASETS["flickr30k"]
br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
model, preprocess = br.load_model()
names, img_feats = br.encode_gallery(model, preprocess, limit=1000)
img_feats = img_feats.float()
queries, gt = load_queries(cfg, names, 5)

expansions = json.loads(
    (ROOT / "results/expansions_flickr30k_1000_5.json").read_text())["expansions"]
print(f"{len(queries)} queries, {len(names)} gallery images")

f_base = br.encode_texts(model, queries).float()
re_concepts = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
texts_C = [build_text(q, c) for q, c in zip(queries, re_concepts)]
f_C = br.encode_texts(model, texts_C).float()
fC = 0.9 * f_base + 0.1 * f_C
fC /= fC.norm(dim=-1, keepdim=True)

r_base = ranks_of(f_base, img_feats, gt)
r_ours = ranks_of(fC, img_feats, gt)

fixed = [i for i in range(len(queries))
         if r_base[i] > 0 and r_ours[i] == 0 and re_concepts[i]]
print(f"top-1 fixed by expansion: {len(fixed)} queries")

sims_b = (f_base @ img_feats.T)
sims_o = (fC @ img_feats.T)
out = []
for i in fixed:
    top_b = sims_b[i].topk(5).indices.tolist()
    top_o = sims_o[i].topk(5).indices.tolist()
    out.append({
        "qidx": i, "query": queries[i], "gt_name": names[gt[i]],
        "base_rank": int(r_base[i]), "concepts": re_concepts[i],
        "top5_base": [names[j] for j in top_b],
        "top5_ours": [names[j] for j in top_o],
    })
Path(__file__).with_name("qualitative_candidates.json").write_text(
    json.dumps(out, indent=1))
for c in out[:25]:
    cs = ", ".join(f'{x["concept"]}({x["relation"]})' for x in c["concepts"])
    print(f'[{c["qidx"]}] rank {c["base_rank"]}->0 | {c["query"][:70]} | + {cs}')
