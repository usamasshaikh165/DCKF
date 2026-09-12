"""
E31 — Does fp16 normalization of cached features bias the baseline?

encode_gallery/encode_texts normalize in the model dtype (fp16 on CUDA) and
cache the result; fused configs (B/C/D) re-normalize in fp32. If fp16 norm
rounding alone moves R@1, part of every reported delta is an artefact.
This script measures: baseline with cached (fp16-normalized) features vs the
same features re-normalized in fp32, for image side, text side, and both,
and then re-derives B/C/D deltas against the fp32-renormalized baseline.
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br
from ablations import load_queries
from kg_expanded_eval import DATASETS
from advanced_expansion import REWEIGHTED_PRIORS, rerank, build_text, ranks_of, paired_bootstrap
ROOT = Path(__file__).resolve().parent.parent
def norm(x): return x / x.norm(dim=-1, keepdim=True)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--cap-suffix", default="")
    a = ap.parse_args(); cfg = DATASETS[a.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, pre = br.load_model()
    names, img16 = br.encode_gallery(model, pre)
    queries, gt = load_queries(cfg, names, 5)
    key = f"{a.dataset}_{len(names)}_5"
    t16 = br.encode_texts(model, queries)                  # fp16-normalized, as the paper's baseline uses
    print("dtypes:", img16.dtype, t16.dtype, "| img norm dev (fp32): %.2e" % (img16.float().norm(dim=-1) - 1).abs().max().item(),
          "| txt norm dev: %.2e" % (t16.float().norm(dim=-1) - 1).abs().max().item())
    img32, t32 = norm(img16.float()), norm(t16.float())
    r = {}
    r["A cached fp16 (paper baseline)"] = ranks_of(t16.float(), img16.float(), gt)
    r["A img renorm"] = ranks_of(t16.float(), img32, gt)
    r["A txt renorm"] = ranks_of(t32, img16.float(), gt)
    r["A both renorm"] = ranks_of(t32, img32, gt)
    exp = json.loads((ROOT/"results"/f"expansions_{key}.json").read_text())["expansions"]
    rew = [rerank(e, REWEIGHTED_PRIORS) for e in exp]
    f_e = norm(br.encode_texts(model, [build_text(q, c) for q, c in zip(queries, rew)]).float())
    fq = norm(0.9 * t32 + 0.1 * f_e)
    r["B KG (vs both-renorm A)"] = ranks_of(fq, img32, gt)
    suf = f"_{a.cap_suffix}" if a.cap_suffix else ""
    cap_p = ROOT/"results"/f"blip_captions_{key}{suf}.json"
    if cap_p.exists():
        cm = json.loads(cap_p.read_text()); caps = [cm[n] for n in names]
        fc = norm(br.encode_texts(model, caps).float()); fg = norm(0.9 * img32 + 0.1 * fc)
        r["C captions (vs both-renorm A)"] = ranks_of(t32, fg, gt)
        r["D both (vs both-renorm A)"] = ranks_of(fq, fg, gt)
    base16, base32 = r["A cached fp16 (paper baseline)"], r["A both renorm"]
    print(f"\n{'config':<34} {'R@1':>8} | vs fp16 A            | vs fp32 A")
    for k, v in r.items():
        d1, c1, p1 = paired_bootstrap(base16, v); d2, c2, p2 = paired_bootstrap(base32, v)
        print(f"{k:<34} {(v<1).mean():8.4f} | {d1*100:+.2f} [{c1[0]*100:+.2f},{c1[1]*100:+.2f}] p={p1:.3f} | {d2*100:+.2f} [{c2[0]*100:+.2f},{c2[1]*100:+.2f}] p={p2:.3f}")
if __name__ == "__main__": main()
