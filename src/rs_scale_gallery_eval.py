"""
E33 — Gallery channel under in-domain distractors (reviewer point 5).

Table VIII tested only the query channel at scale. Here every gallery item,
real RSICD image or Million-AID distractor, gets the SAME treatment, using
the E30 arrays (raw CLIP embedding + BLIP-large aerial caption embedding per
distractor). Configs, all t2i R@1/R@5 on the 5,465 RSICD queries:
  A  baseline               raw image embeddings everywhere
  B  DCKF-Q                 A gallery, KG-expanded queries
  C  captions (alpha)       alpha*I + (1-alpha)*C for real AND distractor items
  D  C + DCKF-Q
  T  caption-only           alpha = 0 (text-to-text), for completeness
Distractor scales 0 / 10k / 50k / 100k. Paired bootstrap vs A at each scale.

Usage (dell3):  python src/rs_scale_gallery_eval.py --alphas 0.9 --scales 0,10000,50000,100000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br
from ablations import load_queries
from kg_expanded_eval import DATASETS
from advanced_expansion import REWEIGHTED_PRIORS, rerank, build_text, paired_bootstrap
ROOT = Path(__file__).resolve().parent.parent
def norm(x): return x / x.norm(dim=-1, keepdim=True)

def ranks(txt, gallery, gt, chunk=1024):
    gt_t = torch.tensor(gt, device=txt.device); out = torch.empty(len(gt), device=txt.device)
    for i in range(0, len(gt), chunk):
        s = txt[i:i+chunk] @ gallery.T
        g = s.gather(1, gt_t[i:i+chunk, None]); out[i:i+chunk] = (s > g).sum(1).float()
    return out.cpu().numpy()

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.9"); ap.add_argument("--scales", default="0,10000,50000,100000")
    ap.add_argument("--cap-suffix", default="large_aerial"); ap.add_argument("--target", type=int, default=100000)
    a = ap.parse_args(); alphas = [float(x) for x in a.alphas.split(",")]; scales = [int(x) for x in a.scales.split(",")]
    cfg = DATASETS["rsicd"]; br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, pre = br.load_model(); names, img = br.encode_gallery(model, pre); dev = img.device
    queries, gt = load_queries(cfg, names, 5); key = f"rsicd_{len(names)}_5"
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    f_q = br.encode_texts(model, queries)
    pools = json.loads((ROOT/"results"/f"expansions_{key}.json").read_text())["expansions"]
    rew = [rerank(e, REWEIGHTED_PRIORS) for e in pools]
    f_kg = norm(0.9 * f_q + 0.1 * br.encode_texts(model, [build_text(q, c) for q, c in zip(queries, rew)]))
    cm = json.loads((ROOT/"results"/f"blip_captions_{key}_{a.cap_suffix}.json").read_text())
    f_cap = br.encode_texts(model, [cm[n] for n in names])
    res = ROOT/"results"; meta = json.loads((res/f"distractors_millionaid_{a.target}_bliplarge.meta.json").read_text())
    n_avail = meta["filled"]; print(f"{len(queries)} queries | {len(names)} real | {n_avail:,} captioned distractors")
    d_raw = np.load(res/f"distractors_millionaid_{a.target}_raw_v2.fp16.npy", mmap_mode="r")
    d_cap = np.load(res/f"distractors_millionaid_{a.target}_cap_bliplarge.fp16.npy", mmap_mode="r")
    out = {"alphas": alphas, "scales": [], "cap_suffix": a.cap_suffix}
    print(f"\n{'N dist.':>8} {'config':<22} {'R@1':>7} {'R@5':>7}   dR@1 vs A   CI               p")
    for n in scales:
        if n > n_avail: print(f"skip {n}: only {n_avail} captioned"); continue
        dr = norm(torch.from_numpy(np.ascontiguousarray(d_raw[:n])).float().to(dev)) if n else None
        dc = norm(torch.from_numpy(np.ascontiguousarray(d_cap[:n])).float().to(dev)) if n else None
        gal_A = torch.cat([img, dr]) if n else img
        rA = ranks(f_q, gal_A, gt); rB = ranks(f_kg, gal_A, gt)
        rows = {"A baseline": rA, "B DCKF-Q": rB}
        for al in alphas + [0.0]:
            fused_real = norm(al * img + (1 - al) * f_cap)
            gal = torch.cat([fused_real, norm(al * dr + (1 - al) * dc)]) if n else fused_real
            tag = "T caption-only" if al == 0.0 else f"C captions a={al}"
            rows[tag] = ranks(f_q, gal, gt)
            if al != 0.0: rows[f"D both a={al}"] = ranks(f_kg, gal, gt)
        rec = {"n": n}
        for tag, r in rows.items():
            d, ci, p = paired_bootstrap(rA, r); r1, r5 = (r < 1).mean(), (r < 5).mean()
            rec[tag] = {"R@1": float(r1), "R@5": float(r5), "dR1": float(d), "ci": [float(ci[0]), float(ci[1])], "p": float(p)}
            print(f"{n:>8,} {tag:<22} {r1:7.2%} {r5:7.2%}   {d*100:+6.2f}   [{ci[0]*100:+.2f},{ci[1]*100:+.2f}]  {p:.4f}")
        out["scales"].append(rec); print()
        del gal_A, dr, dc; torch.cuda.empty_cache()
    (res/f"rs_scale_gallery_{a.cap_suffix}.json").write_text(json.dumps(out, indent=1)); print("wrote results/rs_scale_gallery_*.json")
if __name__ == "__main__": main()
