# DCKF: Drift-Controlled Knowledge Fusion for Training-Free Text-to-Image Retrieval

Code, experiment log and result files for the manuscript

> U. A. Shaikh, Z. Chen, M. T. Riaz, M. Y. Raja. *DCKF: Drift-Controlled Knowledge Fusion for Training-Free Text-to-Image Retrieval.* Submitted to Knowledge-Based Systems, 2026.

DCKF injects external knowledge into a frozen CLIP retriever without any training. On the query side it expands the
query with relation-weighted ConceptNet concepts; on the gallery side it fuses a frozen captioner's description into each
image embedding at index time. Both channels pass through one gentle fusion step (weight 0.1 on the injected signal), and
the paper's parity and gate studies show that this fusion weight, not the similarity gate or the knowledge source, is
what makes injection safe.

## What is in this repository

| Path | Contents |
|---|---|
| `src/` | All evaluation code (PyTorch, inference only). Entry points are listed below. |
| `scripts/` | The shell runners used on the GPU server for the review re-runs (E29, E32, E37, E37b, E40). |
| `results/` | Every result file the paper's tables and p-values are read from. `results/imgboot/` is the source of record for all text-to-image p-values (paired bootstrap over image clusters). `results/e37_e38/` holds the gate-isolation and caption-repair runs, `results/fp32_rederivation/` the float32 re-derivation of every table. |
| `results/expansions_*.json`, `results/rsicd_blip_large.json` | Cached ConceptNet candidate pools and BLIP captions, so the query-side and caption experiments can be re-run without a GPU. |
| `EXPERIMENTS.md` | The complete, dated experiment log (E1 to E41). The predictions cited in the paper's Discussion were written into this log before the runs that tested them. |
| `figures/`, `paper_figures/` | Scripts that produce every figure in the paper from the files in `results/`. |

Large artifacts that exceed GitHub's file limits (the 100k Million-AID distractor index and its BLIP-large caption
embeddings, about 210 MB, and the 1M DataComp web-distractor embeddings, about 1 GB) are deposited separately; the
Zenodo DOI will be added here and in the paper's data-availability statement.

## Setup

Python 3.12 or 3.13. Install the dependencies in `requirements.txt`, plus OpenAI CLIP from its repository:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

RemoteCLIP experiments need `open_clip_torch`; the captioning and LLM baselines need `transformers`. A CUDA GPU is
required for captioning, LLM rewriting and the 100k-distractor studies; everything else runs on a laptop (the
development machine was an Apple M3 with the MPS backend).

## Data

The benchmark datasets are not redistributed. Obtain them from their original sources and place them under `data/`:
Flickr30k and MS-COCO with the Karpathy test splits, RSICD, RSITMD, UCM-Captions and NWPU-Captions.
`src/convert_rs_datasets.py` and `src/convert_nwpu.py` convert the remote-sensing datasets to the common caption
format. ConceptNet 5.7 is built into a local SQLite store with `src/build_conceptnet_db.py` (2,219,798 English edges
over nine relations).

## Reproducing the paper

| Paper item | Script |
|---|---|
| Table III (natural-image benchmarks, both directions) and RSICD i2t | `src/bidirectional_eval.py` |
| Tables VI, VIII, XII (caption channel, knowledge-free PRF, hyperparameters) | `src/reviewer_baselines.py` |
| Table VII (selection strategies at parity) | `src/knowledge_baselines.py` |
| Table IX (LLM rewriting vs. KG) | `src/llm_rewrite_baseline.py` |
| Table V (pooled RS test) | `src/pooled_rs_test.py` |
| Tables X, XI (gallery growth) | `src/rs_scale_eval.py`, `src/rs_scale_gallery_eval.py`, `src/flickr_scale_gallery_eval.py` |
| Table XIII and Fig. 8 (gate isolation, tau sweep) | `src/gate_ablation.py`, `paper_figures/make_fig_tau_v3.py` |
| Backbone ablation (ViT-L/14, RemoteCLIP) | `src/backbone_ablation.py`, `src/remoteclip_rs.py` |
| Multi-hop ablation | `src/multihop_expansion.py` |
| Query-expansion latency (Sec. IV-H) | `src/expansion_latency.py` |
| Captions (BLIP-base, prompted, BLIP-large) | `src/rsicd_domain_caption.py`, `src/rs_large_aerial_captions.py`, `src/rsicd_blip_large.py` |
| Distractor sets | `src/fetch_distractors.py`, `src/stream_millionaid.py`, `src/stream_millionaid_captions.py` |
| Figures 1, 3, 6 | `paper_figures/make_figs_v2.py`; Fig. 2 `make_pipeline_svg.py`; Fig. 4 `make_fig_dose.py`; Fig. 5 `figures/render_qualitative.py`; Fig. 7 `make_fig_embed.py` |

All bootstrap tests use 10,000 resamples; set `CLUSTER_BOOT=1` to resample image clusters for text-to-image (the setting
used for every p-value in the paper), as in `scripts/e40_server_run.sh`.

## License

MIT, see `LICENSE`. The datasets, CLIP, BLIP, RemoteCLIP, Qwen2.5 and ConceptNet keep their own licences.

## Contact

Usama Ali Shaikh, School of Software Technology, Dalian University of Technology (usama7097@mail.dlut.edu.cn).
