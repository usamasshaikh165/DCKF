#!/bin/bash
# E37 gate isolation (review item 1) + E38 caption repair on sibling RS datasets (review item 2)
cd ~/RA-KG-T2I
export CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com
PY=~/miniconda3/envs/rakg/bin/python
step(){ echo; echo "### $(date +%H:%M) $*"; }
step "E37 gate ablation flickr30k_k1k"; $PY src/gate_ablation.py --dataset flickr30k_k1k 2>&1 | grep -v Warning
step "E37 gate ablation rsicd";         $PY src/gate_ablation.py --dataset rsicd 2>&1 | grep -v Warning
for ds in rsitmd ucm nwpu; do
  step "E38 captions $ds base aerial";  $PY src/rs_large_aerial_captions.py --dataset $ds --captioner base  2>&1 | grep -v Warning
  step "E38 captions $ds large aerial"; $PY src/rs_large_aerial_captions.py --dataset $ds --captioner large 2>&1 | grep -v Warning
done
for ds in rsitmd ucm nwpu; do for suf in aerial large_aerial; do
  step "E38 reviewer $ds $suf"; $PY src/reviewer_baselines.py --dataset $ds --cap-suffix $suf 2>&1 | grep -v Warning
done; done
echo; echo "E37_E38_DONE"
