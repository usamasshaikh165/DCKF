#!/bin/bash
# E40 — image-cluster paired bootstrap (CLUSTER_BOOT=1) for every t2i p-value in the paper.
# Scripts overwrite their usual JSONs, so: back up, run, move new JSONs to results/imgboot/, restore originals.
cd ~/RA-KG-T2I
export CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com CLUSTER_BOOT=1
PY=~/miniconda3/envs/rakg/bin/python
mkdir -p results/imgboot results/capboot_backup
cp results/*.json results/capboot_backup/ 2>/dev/null
step(){ echo; echo "### $(date +%H:%M) $*"; }
run(){ stamp=$(mktemp); sleep 1; "$@" 2>&1 | grep -v Warning
  for f in $(find results -maxdepth 1 -name "*.json" -newer $stamp); do b=$(basename $f); mv $f results/imgboot/$b; [ -f results/capboot_backup/$b ] && cp results/capboot_backup/$b results/$b; done; rm -f $stamp; }
for ds in flickr30k_k1k coco5k rsicd rsitmd ucm nwpu; do step "bidirectional $ds"; run $PY src/bidirectional_eval.py --dataset $ds --limit 0; done
step "pooled RS"; run $PY src/pooled_rs_test.py
for ds in flickr30k_k1k rsicd; do for m in fusion text; do step "parity $ds $m"; run $PY src/knowledge_baselines.py --dataset $ds --limit 0 --mode $m; done; done
for ds in coco5k rsitmd ucm nwpu; do step "parity $ds fusion"; run $PY src/knowledge_baselines.py --dataset $ds --limit 0; done
for ds in flickr30k_k1k rsicd; do step "LLM $ds"; run $PY src/llm_rewrite_baseline.py --dataset $ds --limit 0 --captions-per-image 5; done
for ds in flickr30k_k1k coco5k rsicd rsitmd ucm nwpu; do step "reviewer $ds"; run $PY src/reviewer_baselines.py --dataset $ds; done
for ds in rsicd rsitmd ucm nwpu; do for suf in aerial large_aerial; do step "reviewer $ds $suf"; run $PY src/reviewer_baselines.py --dataset $ds --cap-suffix $suf; done; done
for ds in flickr30k_k1k rsicd; do step "gate ablation $ds"; run $PY src/gate_ablation.py --dataset $ds; done
step "backbone ViT-L/14"; run $PY src/backbone_ablation.py
step "RS scale (Table X)"; run $PY src/rs_scale_eval.py
step "RS scale gallery (Table XI)"; run $PY src/rs_scale_gallery_eval.py
step "Flickr scale gallery (Table XI)"; run $PY src/flickr_scale_gallery_eval.py
echo; echo "E40_DONE $(date)"
