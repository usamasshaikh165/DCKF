#!/bin/bash
# E37b — rerun gate ablation with the paper-pipeline (top-5 prefilter) variant; waits for E37/E38 to finish (GPU shared)
cd ~/RA-KG-T2I
export CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com
PY=~/miniconda3/envs/rakg/bin/python
while ! grep -q E37_E38_DONE results/e37_e38.log; do sleep 30; done
for ds in flickr30k_k1k rsicd; do echo; echo "### $(date +%H:%M) E37b gate ablation $ds"; $PY src/gate_ablation.py --dataset $ds 2>&1 | grep -v Warning; done
echo; echo "E37B_DONE"
