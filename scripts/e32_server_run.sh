#!/bin/bash
# E32 — re-derive every paper table with fp32-normalized features (E31 fix in baseline_retrieval._unit32).
# Old JSONs are preserved in results/pre_fp32/ for a before/after diff.
cd ~/RA-KG-T2I
export CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com
PY=~/miniconda3/envs/rakg/bin/python
step(){ echo; echo "### $(date +%H:%M) $*"; }
mkdir -p results/pre_fp32
cp results/bidirectional_*.json results/baselines_*.json results/llm_baseline_*.json results/reviewer_*.json results/pre_fp32/ 2>/dev/null
step "bidirectional (Tables II, VIII base)"
for ds in flickr30k_k1k coco5k rsicd rsitmd ucm nwpu; do step "bidirectional $ds"; $PY src/bidirectional_eval.py --dataset $ds --limit 0 2>&1 | grep -v Warning; done
step "rsicd captioner dose-response (Table V)"; $PY src/rsicd_domain_caption.py 2>&1 | grep -v Warning; $PY src/rsicd_blip_large.py 2>&1 | grep -v Warning
step "pooled RS test (Table IV)"; $PY src/pooled_rs_test.py 2>&1 | grep -v Warning
step "selection parity (Table VI)"
for ds in flickr30k_k1k rsicd; do for m in fusion text; do step "parity $ds $m"; $PY src/knowledge_baselines.py --dataset $ds --limit 0 --mode $m 2>&1 | grep -v Warning; done; done
for ds in coco5k rsitmd ucm nwpu; do step "parity $ds fusion"; $PY src/knowledge_baselines.py --dataset $ds --limit 0 2>&1 | grep -v Warning; done
step "LLM rewriting (Table VII)"
for ds in flickr30k_k1k rsicd; do $PY src/llm_rewrite_baseline.py --dataset $ds --limit 0 --captions-per-image 5 2>&1 | grep -v Warning; done
step "backbone ViT-L/14 (IV-F4)"; $PY src/backbone_ablation.py 2>&1 | grep -v Warning
step "RemoteCLIP (Table III, IV-F4)"; $PY src/remoteclip_rs.py 2>&1 | grep -v Warning
step "tau sensitivity (Fig 8)"; $PY src/tau_sensitivity.py --dataset flickr30k --limit 1000 --captions-per-image 5 2>&1 | grep -v Warning; $PY src/tau_sensitivity.py --dataset rsicd --limit 1093 --captions-per-image 5 2>&1 | grep -v Warning
step "multi-hop (IV-F3)"; $PY src/multihop_expansion.py --dataset rsicd --captions-per-image 5 2>&1 | grep -v Warning; $PY src/multihop_expansion.py --dataset flickr30k --limit 1000 --captions-per-image 5 2>&1 | grep -v Warning
step "gallery KG rerank (IV-F2)"; $PY src/gallery_kg_rerank.py --dataset flickr30k --limit 1000 --captions-per-image 5 2>&1 | grep -v Warning
step "RS scale (Table VIII)"; $PY src/rs_scale_eval.py 2>&1 | grep -v Warning
step "reviewer baselines (E29 rerun on fp32 footing)"
for ds in flickr30k_k1k coco5k rsicd rsitmd ucm nwpu; do step "reviewer $ds"; $PY src/reviewer_baselines.py --dataset $ds 2>&1 | grep -v Warning; done
step "reviewer rsicd large_aerial"; $PY src/reviewer_baselines.py --dataset rsicd --cap-suffix large_aerial 2>&1 | grep -v Warning
step "reviewer rsicd aerial"; $PY src/reviewer_baselines.py --dataset rsicd --cap-suffix aerial 2>&1 | grep -v Warning
echo; echo "E32_DONE $(date)"
