"""
E27 — LLM query-rewrite baseline (the "why not just ask an LLM?" reviewer row).

Training-free comparison at parity with the KG channel: a small instruction
LLM (Qwen2.5-1.5B-Instruct, frozen, greedy decoding) rewrites each caption
into a richer retrieval query. Two usages:
  L-replace  rewritten text replaces the original query
  L-fused    0.9*f(query) + 0.1*f(rewrite)  — same gentle fusion as ours
Reference rows: A baseline, B relation-aware KG (from cached expansions).

Interpretability contrast for the paper: a ConceptNet edge is an auditable
reason; an LLM rewrite is not — and the LLM adds a 1.5B-param model to the
query path, while the KG lookup is a SQLite hit.

Rewrites cached to results/llm_rewrites_{key}.json (query-hash keyed).
Usage (dell3):
  HF_ENDPOINT=https://hf-mirror.com python src/llm_rewrite_baseline.py \
      --dataset flickr30k_k1k --limit 0 --captions-per-image 5
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                          # noqa: E402
from ablations import load_queries                       # noqa: E402
from kg_expanded_eval import DATASETS                    # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                paired_bootstrap)
from bidirectional_eval import eval_config               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = ("Rewrite the following image-search caption as one richer "
          "descriptive sentence for retrieving the matching photo. Keep all "
          "original facts, add only visually plausible detail, and output "
          "the sentence only.\nCaption: {q}")


@torch.no_grad()
def llm_rewrites(queries, cache_file, batch_size=64, max_new_tokens=50):
    qhash = hashlib.md5("\n".join(queries).encode()).hexdigest()[:10]
    if cache_file.exists():
        blob = json.loads(cache_file.read_text())
        if blob["qhash"] == qhash:
            print(f"Loaded cached rewrites ({cache_file.name})")
            return blob["rewrites"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_ID, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_ID, torch_dtype=torch.float16).to("cuda").eval()
    out, t0 = [], time.time()
    for i in range(0, len(queries), batch_size):
        chunk = queries[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(q=q)}],
            tokenize=False, add_generation_prompt=True) for q in chunk]
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=256).to("cuda")
        gen = llm.generate(**inputs, max_new_tokens=max_new_tokens,
                           do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts = tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        out.extend(t.strip().split("\n")[0].strip() or q
                   for t, q in zip(texts, chunk))
        if (i // batch_size) % 10 == 0:
            print(f"  rewritten {min(i+batch_size, len(queries))}/"
                  f"{len(queries)} ({(time.time()-t0)/max(i+batch_size,1)*1000:.0f} ms/q)",
                  flush=True)
    del llm
    torch.cuda.empty_cache()
    cache_file.write_text(json.dumps({"qhash": qhash, "rewrites": out}))
    print(f"Rewrites done ({(time.time()-t0)/len(queries)*1000:.0f} ms/q), cached.")
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k_k1k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha-q", type=float, default=0.9)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    n_images = len(names)
    key = f"{args.dataset}_{n_images}_{args.captions_per_image}"
    print(f"{len(queries)} captions over {n_images} images ({args.dataset})")

    rewrites = llm_rewrites(
        queries, ROOT / "results" / f"llm_rewrites_{key}.json")
    print("example:", queries[0], "->", rewrites[0])

    f_q = br.encode_texts(model, queries).float()
    f_rw = br.encode_texts(model, rewrites).float()
    f_fused = args.alpha_q * f_q + (1 - args.alpha_q) * f_rw
    f_fused /= f_fused.norm(dim=-1, keepdim=True)

    configs = {"A baseline": f_q,
               "L-replace (LLM rewrite)": f_rw,
               f"L-fused aq={args.alpha_q}": f_fused}
    exp_file = ROOT / "results" / f"expansions_{key}.json"
    if exp_file.exists():
        expansions = json.loads(exp_file.read_text())["expansions"]
        rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
        f_exp = br.encode_texts(
            model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
        fq_kg = args.alpha_q * f_q + (1 - args.alpha_q) * f_exp
        fq_kg /= fq_kg.norm(dim=-1, keepdim=True)
        configs[f"B relation-KG aq={args.alpha_q}"] = fq_kg

    results = {t: eval_config(f, img_feats, gt, n_images)
               for t, f in configs.items()}

    hdr = (f"\n{'config':<26} | {'t2i R@1':>8} {'R@5':>7} {'R@10':>7}"
           f" | {'i2t R@1':>8} {'R@5':>7} {'R@10':>7} | {'mR':>6}")
    print(hdr); print("-" * len(hdr))
    for tag, m in results.items():
        t, i = m["t2i"], m["i2t"]
        print(f"{tag:<26} | {t['R@1']:8.2%} {t['R@5']:7.2%} {t['R@10']:7.2%} "
              f"| {i['R@1']:8.2%} {i['R@5']:7.2%} {i['R@10']:7.2%} "
              f"| {m['mR']:6.2%}")

    boots = {}
    for tag in list(results)[1:]:
        d, (lo, hi), p = paired_bootstrap(results["A baseline"]["_ranks_t2i"],
                                          results[tag]["_ranks_t2i"])
        boots[f"{tag} vs A"] = {"delta_R@1": d, "ci95": [lo, hi], "p": p}
        print(f"bootstrap {tag} vs A, t2i R@1: {d*100:+.2f} "
              f"[{lo*100:+.2f}, {hi*100:+.2f}], p={p:.4f}")
    if any(t.startswith("B ") for t in results):
        btag = next(t for t in results if t.startswith("B "))
        for ltag in (t for t in results if t.startswith("L-")):
            d, (lo, hi), p = paired_bootstrap(results[ltag]["_ranks_t2i"],
                                              results[btag]["_ranks_t2i"])
            boots[f"{btag} vs {ltag}"] = {"delta_R@1": d, "ci95": [lo, hi], "p": p}
            print(f"bootstrap KG vs {ltag}, t2i R@1: {d*100:+.2f} "
                  f"[{lo*100:+.2f}, {hi*100:+.2f}], p={p:.4f}")

    out = {"dataset": args.dataset, "n_images": n_images,
           "n_captions": len(queries), "alpha_q": args.alpha_q,
           "llm": LLM_ID, "device": br.DEVICE,
           "configs": {t: {k: v for k, v in m.items() if not k.startswith("_")}
                       for t, m in results.items()},
           "bootstraps": boots}
    out_file = ROOT / "results" / f"llm_baseline_{key}.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
