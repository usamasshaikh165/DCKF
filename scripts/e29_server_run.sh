#!/bin/bash
# E29 reviewer baselines: PRF / caption-only / alpha,beta,k sweeps / caption gate. GPU 0, 4 GB free is enough.
cd ~/RA-KG-T2I
export CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com
PY=~/miniconda3/envs/rakg/bin/python
for ds in flickr30k_k1k coco5k rsicd rsitmd ucm nwpu; do
  echo "=== $ds ===" ; $PY src/reviewer_baselines.py --dataset $ds 2>&1 | grep -v Warning
done
echo "=== rsicd large_aerial ===" ; $PY src/reviewer_baselines.py --dataset rsicd --cap-suffix large_aerial 2>&1 | grep -v Warning
echo "=== rsicd aerial (BLIP-base prompted) ===" ; $PY src/reviewer_baselines.py --dataset rsicd --cap-suffix aerial 2>&1 | grep -v Warning
echo E29_DONE
