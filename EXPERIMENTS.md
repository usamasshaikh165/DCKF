# RA-KG-T2I Experiment Log

Hardware: MacBook (Apple M3, 16GB RAM), device=MPS. Model: CLIP ViT-B/32 (frozen).
All runs: text→image retrieval, cosine similarity, chunked exact ranking.

## 2026-07-08 — Session 1: infrastructure + baselines + first KG experiment

### E1. Flickr30k, 1,000-image gallery, 5 captions/image (5,000 queries)
| method | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| baseline CLIP | 59.42% | 84.60% | 90.66% | 1 |

Reference: CLIP paper zero-shot Flickr30k 1k test ≈ 58.8 / 83.5 / 90.1 → **reproduced ✓**

### E2. COCO Karpathy 5k test, 5,000-image gallery, 5 captions/image (25,010 queries)
| method | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| baseline CLIP | 30.57% | 56.17% | 66.96% | 4 |

Reference: CLIP paper zero-shot COCO 5k ≈ 30.4 / 56.0 / 66.9 → **reproduced ✓**

### E3. First KG-expansion experiment — COCO 5k, 1 caption/image (5,000 queries)
Expansion v1: seed n-grams matched against local ConceptNet → relation-prior ×
CLIP-similarity scoring → noise filters (seed-containment, ≤3 words) → top-3
concepts → template ", a scene with c1, c2, c3". 4,831/5,000 queries enriched.
Expansion cost: **223 ms/query** (KG lookup + CLIP drift scoring).

| method | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| A baseline | 29.58% | 54.06% | 65.32% | 4 |
| B KG text-expanded | 24.92% | 48.00% | 58.86% | 6 |
| C fused α=0.6 (pilot method) | 29.34% | 53.12% | 64.60% | 4 |

**Finding F1 (key negative result): unselective KG expansion degrades retrieval
on descriptive-caption benchmarks (−4.66 pts R@1); embedding fusion is ~neutral.**
Consistent with proposal Problem 1 ("naive expansion increases noise and reduces
precision"). Motivates: selective triggering, per-relation weighting, better
templates → see ablation plan.

Note: pilot study's earlier "+16.5 pts from KG" (31.5→48.0 R@1) did NOT
replicate under the standard protocol; treat pilot numbers as superseded.

### Infrastructure notes (reproducibility)
- api.conceptnet.io down (502). Replaced by local SQLite build from ConceptNet 5.7
  (HF `CleverThis/conceptnet` parquet via hf-mirror.com): 2,219,798 English edges,
  whitelist relations, 168 MB, <1 ms lookups. RDF export lacks edge weights →
  scoring = relation priors × CLIP similarity.
- Amazon S3 unusable from this network (~3 KB/s); HF via hf-mirror.com fast.
- Flickr30k images were iCloud-evicted (96% dataless) → dataset relocated to
  ~/RA-KG-T2I/data/flickr30k_local (outside iCloud sync scope).

### E4. Flickr30k FULL gallery — 31,783 images, 158,914 queries (all captions)
| method | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| baseline CLIP | 21.68% | 41.44% | 50.97% | 10 |

**Finding F2: gallery scale dominates difficulty** — R@1 falls 59.4%→21.7%
as gallery grows 1k→31.8k (same model/queries). Motivates both semantic
improvements (KG) and efficient large-gallery search (ANN/indexing).
Image encoding: 31,783 imgs in ~3 min wall on M3/MPS (~170 img/s incl. JPEG decode).

### E5. Ablation matrix — Flickr30k 1k gallery, 5,000 queries
Expansion cache: top-5 concepts/query, 238 ms/q. CSV: ablations_flickr30k_1000_5.csv

| config | R@1 | ΔR@1 |
|---|---|---|
| baseline | 59.26% | — |
| expand-all T1 (3 concepts, verbose) | 54.42% | −4.84 |
| expand-all T2 (1 concept) | 58.40% | −0.86 |
| expand-all T3 (2 concepts) | 56.74% | −2.52 |
| selective <5 content words (n=1403) | 58.88% | −0.38 |
| only-IsA (n=4146) | 58.22% | −1.04 |
| only-AtLocation (n=217) | 59.22% | −0.04 |
| only-UsedFor (n=209) | 59.00% | −0.26 |
| only-PartOf (n=102) | 59.22% | −0.04 |
| fused α=0.7 | 59.42% | +0.16 |
| fused α=0.8 | 59.60% | +0.34 |
| **fused α=0.9** | **59.68%** | **+0.42** (R@10 also +0.18 at α=0.7) |

**Finding F3: template verbosity is a first-order factor.** 3-concept template
costs −4.84 pts; 1-concept only −0.86. Long appendages push CLIP text encoding
out of distribution / dilute the query.

**Finding F4: IsA is the main noise source** on descriptive captions — generic
hypernyms (place, transport) dilute specificity. Contradicts the naive prior
IsA=1.0 ≥ others; relation utility is query-context dependent. This is a
core thesis insight: relation weights must be learned/adapted, not fixed.

**Finding F5: gentle embedding fusion (α≈0.9) gives the first positive delta
(+0.42 R@1).** Expansion signal exists but must be injected weakly; heavy
injection destroys precision. Suggests direction: per-query adaptive α
(more expansion for short/ambiguous queries), relation-aware concept choice.

### E6. FAISS latency stress test — 100k→2M vectors (dim=512, M3 CPU)
Gallery = 31,783 real Flickr30k features + distribution-matched synthetic
distractors (mixup+noise, renormalized). Synthetic = latency benchmarking ONLY.
200 queries, top-10. CSV: scale_benchmark.csv, plot: scale_benchmark.png

| scale | FlatIP exact | IVF-SQfp16 np=8 | np=32 | np=32 recall-vs-exact@10 |
|---|---|---|---|---|
| 100k | 4.65 ms | 0.20 ms | 0.54 ms | 0.958 |
| 500k | 25.28 ms | 0.66 ms | 1.19 ms | 0.978 |
| 1M | 56.44 ms | 0.84 ms | 1.44 ms | 0.996 |
| 2M | 97.42 ms | 1.20 ms | 1.98 ms | 0.999 |

**Finding F6: exact search scales linearly (~48 ms/M vectors); IVF+fp16 stays
<2 ms at 2M with 99.9% top-10 agreement (49× speedup).** Peak RSS 10.7 GB at
2M (fp32 gallery + index) — the 16 GB RAM ceiling; larger scales need fp16
galleries, on-disk indices, or lab server.

Engineering note: faiss and torch cannot share a process on macOS/arm64
(duplicate libomp → SIGSEGV / OMP Error #15); stress test is torch-free,
reading features from .npy. KMP_DUPLICATE_LIB_OK does not help.

## 2026-07-09 — Session 2: adaptive expansion + cross-dataset + remote sensing

### E7. Adaptive, relation-reweighted expansion — Flickr30k 1k, 5,000 queries
Configs re-rank cached candidates: reweighted priors (IsA 1.0→0.4,
AtLocation/UsedFor→1.0), adaptive α(q)=clamp(0.70+0.04·len, 0.75, 0.95).

| config | R@1 | ΔR@1 |
|---|---|---|
| baseline | 59.26% | — |
| B fused α=0.9 (E5 best) | 59.64% | +0.38 |
| C relation-reweighted | 59.50% | +0.24 |
| **D adaptive α** | **59.66%** | **+0.40** |
| E selective (n=1980) | 59.50% | +0.24 |

**Finding F7: D is the first statistically significant improvement:
+0.40 pts R@1, paired bootstrap 95% CI [+0.12, +0.70], p = 0.0031.**

### E8. Cross-dataset check — COCO 5k, 5,000 queries (1 cap/img)
Full ablation matrix replicates all Flickr patterns (T1 −4.66; only-IsA −0.92;
fusion peak α≈0.8 +0.18). Adaptive configs: +0.16..+0.22, bootstrap
p = 0.088 → **directionally consistent, not significant** on dense captions.
Expansion cost improved to ~200 ms/q (warm KG cache).

### E9. RSICD (remote sensing, target domain) — test split 1,093 imgs, 5,465 queries
| method | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| baseline CLIP | 5.51% | 17.90% | 28.07% | 26 |

Matches published CLIP zero-shot RSICD (~5.8/17.7/27.8) → **third benchmark
reproduced ✓**. 10× weaker than natural images = the domain-gap headroom the
thesis targets.

Ablations (5,465 queries): expand-all still hurts (T1 −0.73) but far less than
natural images; IsA still the noise source (−0.27); **fusion peak shifts to
α=0.7 (+0.29 R@1)** vs α=0.9 on Flickr.
**Finding F8: optimal expansion strength grows with domain gap** — consistent
with the theory that KG context matters more where the encoder is weaker.
Expansion cost: 163 ms/q (warm cache).

Adaptive configs + bootstrap: B fused α=0.9 → **+0.22 R@1, 95% CI [+0.00, +0.44],
p = 0.028 (significant)**. Length-adaptive α does NOT transfer (+0.07): RSICD
captions are uniformly short, so query length carries no routing signal there.

### Consolidated cross-dataset result (best gentle-fusion config vs baseline, R@1)
| dataset | Δ R@1 | bootstrap p | verdict |
|---|---|---|---|
| Flickr30k 1k | +0.40 | 0.003 | significant |
| COCO 5k | +0.22 | 0.088 | directional |
| RSICD test | +0.22 | 0.028 | significant |

**Finding F9 (honest strategic read): training-free text-side KG expansion
yields consistent but SMALL gains (+0.2..+0.4 R@1) across three datasets —
2/3 statistically significant. The proposal's +10–15 pt absolute target is not
reachable by this mechanism alone on standard benchmarks.** Candidate
mechanisms with more headroom: (a) gallery-side offline expansion (enrich
image-side text/captions at index time — zero query latency), (b) the
proposal's domain-mapping module for RS vocabulary, (c) confidence-based
routing (expand only low-margin queries). Discuss target revision with
supervisor before 中期 (mid-term) check.

### E10. Accuracy vs gallery scale with 1M REAL distractors — Flickr 1k-test queries
Distractors: 1,000,000 precomputed OpenAI CLIP ViT-B/32 image embeddings from
DataComp-small (HF `mlfoundations/datacomp_small` via hf-mirror), fetched by
`fetch_distractors.py` WITHOUT downloading images: npz shards are ZIP_STORED,
so only the `b32_img.npy` member is range-requested (518 MB/shard vs 3.1 GB).
Verified same embedding space (unit-norm; cross-set cosine 0.41 vs within-set
0.45–0.50). Total footprint: 977 MB fp16 memmap. Eval: `scale_eval.py`,
exact streaming ranking in numpy, 5,000 queries × 1.03M gallery in ~4 min,
<3 GB RAM. GT similarity read from the SAME matmul as gallery sims (1-ulp
elementwise/BLAS mismatch otherwise makes the true image outscore itself,
silently shifting every rank by +1 — cost R@1 ~27 pts before the fix).

| gallery | base R@1 | D R@1 | ΔR@1 | bootstrap p | base R@5 | D R@5 |
|---|---|---|---|---|---|---|
| 1,000 | 59.24% | 59.64% | +0.40 | 0.0035 | 84.56% | 84.60% |
| 31,783 | 25.46% | 25.74% | +0.28 | 0.0112 | 46.26% | 46.62% |
| 100,000 | 24.70% | 24.94% | +0.24 | 0.0313 | 45.30% | 45.74% |
| 500,000 | 22.78% | 22.82% | +0.04 | 0.41 | 42.22% | 42.56% |
| 1,000,000 | 21.08% | 21.14% | +0.06 | 0.34 | 39.68% | 40.08% |

**Finding F10: in-domain distractors dominate difficulty, not raw count.**
Baseline falls 59.2→25.5 adding 30.8k same-dataset images, but only
25.5→21.1 adding 968k out-of-domain web images (~34× more items, ~1/7 the
damage). "Web-scale gallery" headlines overstate difficulty when the
distractors are off-distribution; per-item hardness is what matters.

**Finding F11: the KG-expansion gain persists at moderate scale but washes
out at 1M.** ΔR@1 stays significant through 100k (+0.24, p=0.031) — the E7
gain is NOT an artifact of the tiny 1k protocol — but decays to noise at
500k+ (+0.04..+0.06, ns). At R@5 a residual +0.40 remains at 1M. Reading:
gentle fusion nudges borderline rankings; as distractor density grows, the
margin a nudge must overcome shrinks faster than the nudge helps. Stronger
semantic signal (gallery-side expansion, domain mapping) is needed at scale.

### E12. Gallery-side expansion pilot v1 — self-derived tags (Flickr30k 1k, 5,000 queries)
Ran on dell3 (V100, cu126); baseline here 59.24% (fp32 sims; 59.38% in the
fp16 baseline script — platform/precision noise band ±0.15).

Pipeline: zero-shot tag each gallery image (top-3 of 10k-concept ConceptNet
degree-ranked vocab, "a photo of X"), expand tags via reweighted-prior ×
CLIP-sim(image, neighbor) (top-2), fuse text into image side. 34 ms/image
offline, zero query latency. Cache: `gallery_expansions_flickr30k_1000_5.json`.

| config (best of sweep) | R@1 | ΔR@1 | p |
|---|---|---|---|
| A baseline | 59.24% | — | — |
| C emb-fusion tags+KG α=.95 | 59.06% | −0.18 | 0.80 |
| E score-fusion tags+KG β=.05 | 58.94% | −0.30 | — |
| KG increment (best KG vs best tags-only) | | +0.02 | 0.49 |

Both fusion levels hurt monotonically as weight grows (α=0.7: −8.9;
β=0.2: −4.6). Embedding fusion additionally suffers the CLIP modality gap,
but score-level fusion failing too shows the problem is deeper:

**Finding F12: self-derived gallery-side expansion is information-free.**
Tags extracted from the image embedding via the same CLIP space are a lossy
re-projection of that embedding — a data-processing chain adds no signal,
only noise, so ANY fusion weight > 0 can only hurt on average. Gallery-side
enrichment requires an EXTERNAL information source (frozen captioner,
detector from a different model family, or metadata). Query-side expansion
never had this problem because the query text is external to the image.

### E13. Gallery-side expansion v2 — frozen BLIP captions (Flickr30k 1k, 5,000 queries)
F12 fix: external semantic source. Frozen BLIP-base (`Salesforce/
blip-image-captioning-base`, NO fine-tuning — contrast KTIR) captions each
gallery image at index time (23 ms/img); captions optionally KG-expanded with
the existing QueryExpander (naive config: top-3, sim≥0.5, original priors,
18 ms/img); text fused into image side. Zero query-side latency. No leakage
(BLIP sees pixels only). Cache: `blip_captions_flickr30k_1000_5.json`.
Ran on dell3 V100.

| config | R@1 | R@5 | R@10 | ΔR@1 |
|---|---|---|---|---|
| A baseline | 59.24% | 84.48% | 90.60% | — |
| **B emb-fusion caption α=0.9** | **61.60%** | **86.48%** | **91.90%** | **+2.36** |
| C emb-fusion caption+KG α=0.9 | 61.16% | 85.72% | 91.58% | +1.92 |
| D score-fusion caption β=0.1 | 60.94% | 85.84% | 91.64% | +1.70 |

Bootstrap C vs A: +1.91, 95% CI [+1.10, +2.70], p<10⁻⁴ (B vs A is larger).
Bootstrap KG increment (C vs B): −0.45, CI [−1.02, +0.10], p=0.94.

**Finding F13: gallery-side enrichment with an external frozen captioner
gives +2.36 R@1 — ~6× the query-side ceiling (F9), significant, zero query
latency.** Embedding-level fusion beats score-level here (new information
outweighs the modality-gap cost that sank E12).

**Finding F13b: the naive KG increment ON TOP of captions is currently
negative (−0.45, ns)** — expansion config was the pre-F4 naive one; the
E7 treatment (IsA demotion, selectivity, reranking) has not yet been applied
to gallery captions. Open question, not a dead end: same pattern as query
side, where naive expansion hurt (−4.8) before relation reweighting fixed it.

### E14. E7 treatment applied to gallery-caption KG expansion (Flickr30k 1k)
Image fusion fixed a=0.9 (E13 best). Expansion pools cached with relation+sim
meta (`gallery_cap_expansions_flickr30k_1000_5.json`, 14 ms/img).

| config | R@1 | ΔR@1 vs cap-only |
|---|---|---|
| B cap-only (ref) | 61.60% | — |
| C naive KG replace | 60.70% | −0.90 |
| D reweighted top-2 | 61.08% | −0.52 |
| E channel fusion g=0.8 | 61.58% | −0.02 (CI [−0.26,+0.22], p=0.60) |
| F selective t=0.65 (n=68) | 61.56% | −0.04 |

**Finding F14: on natural images the KG increment over BLIP captions is
exactly zero.** The E7 treatment recovers the naive loss (−0.90 → −0.02) but
never crosses into positive: BLIP captions are already complete, visually
grounded, CLIP-in-distribution text; ConceptNet's generic associations add
nothing Flickr needs. Per F8 (domain gap shifts fusion), the KG's chance to
contribute is where the captioner's vocabulary FAILS — remote sensing.
→ decisive test is RSICD (and COCO as natural-image control).

### E15. Gallery-side caption+KG, cross-dataset (COCO 5k 1cap/img 5,000 q; RSICD 5,465 q)
Same pipeline as E13/E14 on dell3. COCO baseline 29.66% (1 cap/img protocol,
5k gallery). Offline cost ~19 ms/img caption + 14 ms/img expansion.
Logs: `e15.log`, `e15_coco.log` on dell3; caches `blip_captions_*`,
`gallery_cap_expansions_*` for all three datasets.

Consolidated gallery-side picture (ΔR@1 vs baseline, paired bootstrap):

| dataset | cap-only best | best KG config | KG increment vs cap-only |
|---|---|---|---|
| Flickr30k 1k | **+2.36 (p<10⁻⁴)** | +2.34 | −0.02 (p=0.60) |
| COCO 5k | **+2.94 (p<10⁻⁴)** α=0.8 | +2.60 | +0.08 (p=0.26) |
| RSICD | −0.18 (ns; −0.84 at α=.9) | −0.60 | **+0.24 (p=0.084)** |

**Finding F15: frozen-captioner gallery enrichment is a large, significant,
zero-query-latency win on natural images (+2.4..+2.9 R@1) — but the captioner
is the bottleneck out-of-domain: on RSICD BLIP's natural-image captions are a
net liability and no fusion weight rescues them.**

**Finding F15b: the KG increment is zero where captions are strong (Flickr
−0.02, COCO +0.08, both ns) and directionally positive exactly where captions
are weak (RSICD +0.24, p=0.084).** Pattern across F1-F15: relation-aware KG
expansion is a REPAIR mechanism for deficient text channels (short queries,
domain-mismatched captions), not a general enhancer of good ones. This
reframes the thesis mechanism — and points at RSICD-domain captioning + KG
domain mapping as the place the KG can win outright.

### E16. Multi-hop (2-hop) expansion pilot — proposal Problem 3 (RSICD 5,465 q; Flickr 1k 5,000 q)
Query-side. Path score = Π(reweighted priors) × direction discounts × 0.7
decay × CLIP drift sim (min 0.55), top-2 kept, fused α=0.9. Pool: top-8
1-hop + 5 2-hop through each top-3 1-hop node (~66 candidates/q, 14 ms/q).
Strategies: pooled / backoff (2-hop only when 1-hop empty) / short-query-only.
Parity ablation: 2-hop rescored by last edge alone (no path product/decay) —
"best possible shot". Caches: `multihop_pools_*.json`. Logs `e16*.log`.

| dataset | scoring | 2-hop entered kept | ΔR@1 vs 1-hop |
|---|---|---|---|
| RSICD | path (proposal) | 23 / 5,465 | **+0.00** (CI [0,0]) |
| Flickr 1k | path (proposal) | ≤61 / 5,000 | **+0.00** (CI [0,0]) |
| RSICD | parity | 3,821 | −0.07 |
| Flickr 1k | parity | 2,831 | −0.10 (CI [−0.06,0.00], p=1.0) |

**Finding F16: 2-hop reasoning contributes nothing on either dataset, in
either regime.** Under the thesis's own relation-aware scoring, path-score
products + CLIP drift control suppress 2-hop candidates almost entirely
(self-protecting: exact-zero effect). Forced in at parity, 2-hop concepts
slightly degrade both datasets — the F4 noise mechanism compounds along
paths, so hops add noise faster than context. Answer to proposal Problem 3:
**no** — and the Problem 1/2 machinery (relation filtering + scoring) is
precisely what protects the pipeline from multi-hop noise. 1-hop is not a
limitation of the method; it is where the method's own noise controls
conclude the useful signal ends.

### E17. Additivity: query-side E7 + gallery-side caption fusion (3 datasets)
Query side: reweighted top-2 of cached expansions, fused a_q=0.9. Gallery
side: BLIP caption fusion, a_g ∈ {0.8, 0.9}. All caches reused; log `e17.log`.

| dataset | ΔB (query) | ΔC (gallery) | ΔB+ΔC | ΔD (both, best) | query incr. on enriched gallery |
|---|---|---|---|---|---|
| Flickr30k 1k | +0.22 | +2.36 | +2.58 | +2.32 | −0.04 (p=0.67) |
| **COCO 5k** | +0.08 | +2.94 | +3.02 | **+3.14 (p<10⁻⁴)** | +0.20 (p=0.055) |
| RSICD | +0.11 | −0.84 | −0.73 | −0.75 | +0.09 (p=0.13) |

Headline: **COCO 29.66 → 32.80 R@1 (+3.14, 95% CI [+2.14,+4.14], p<10⁻⁴)**,
training-free, zero query latency for the gallery component.

**Finding F17: the two channels are additive-to-slightly-synergistic on COCO,
overlapping on Flickr (caption channel absorbs the query-side gain), and
independent on RSICD (query-side +0.11 survives; gallery captions stay
harmful).** Consistent with F15b's repair-mechanism theory: both channels
repair text-image mismatch, so they stack only when repairing different
queries. Best-per-dataset policy: COCO/Flickr → gallery captions (+ query-KG
on COCO); RSICD → query-KG only, pending domain captioning.

### E18. Domain-conditioned RSICD gallery captions (1,093 imgs, 5,465 q)
BLIP-base conditional generation, prompt prefixes as zero-cost domain
adaptation. Caches `blip_captions_rsicd_1093_5_{aerial,satellite}.json`.
Qualitative: "aerial" prompt removes hallucinations (uncond/satellite invent
place names — "university of notre-lou", "london"); captions become correct
but GENERIC ("an aerial photograph of the stadium").

| config | R@1 | ΔR@1 |
|---|---|---|
| A baseline | 5.45% | — |
| B uncond α=.9 (E15 ref) | 4.61% | −0.84 |
| B aerial α=.95 | 5.54% | +0.09 (ns) |
| B satellite α=.95 | 5.32% | −0.13 |
| C aerial + KG channel g=.9 | 5.54% | +0.09 (ns); KG increment −0.00 |

**Finding F18: prompt conditioning repairs the caption channel from harmful
to neutral (−0.84 → +0.09) but not to positive, and the KG has nothing to
amplify (increment 0.00).** BLIP-base's ceiling on RS imagery is correct-but-
generic captions; RSICD's difficulty is fine-grained discrimination among
similar aerial scenes, which generic text cannot provide. Gallery-side gains
on RS need a stronger/RS-specific captioner (BLIP2, RS-tuned) — or the
domain-mapping module on the query side. Query-side KG (+0.11..0.22) remains
the only working mechanism on RSICD.

### E19. Backbone-strength ablation — ViT-L/14, all 3 datasets
Identical pipeline, expansion texts held fixed (B/32-selected concepts), only
encoders swapped. Feature caches `*_l14_*.pt` on dell3. Log `e19.log`.
L/14 baselines: Flickr 65.58, COCO 34.80, RSICD 5.05 R@1 (RSICD R@1 slightly
below B/32's 5.45 but R@5/10 higher — domain gap persists at L/14).

| dataset | Δ query-KG (B/32 → L/14) | Δ gallery+query best (B/32 → L/14) |
|---|---|---|
| Flickr 1k | +0.22 → **+0.30 (p=0.011)** | +2.32 → **+2.46 (p<10⁻⁴)** |
| COCO 5k | +0.08 → −0.06 (ns) | +3.14 → **+2.68 (p<10⁻⁴)** |
| RSICD | +0.11 → +0.13 (ns) | (aerial captions) +0.09 → +0.11 (ns) |

**Finding F19: the enrichment gains are NOT a weak-backbone artifact.** The
caption-channel headline survives L/14 essentially undiminished (+2.5..2.7,
p<10⁻⁴), and on Flickr the query-side KG gain is *more* significant under
L/14 (+0.30, p=0.011) than under B/32. COCO's tiny query-side gain fades to
noise, consistent with F14's strong-channel prediction. RSICD stays neutral
across backbones — the domain gap is a data problem, not a capacity problem.

### E20. KTIR benchmark completion — RSITMD + UCM-Captions acquired & evaluated
Acquisition: HF has no usable copies; Baidu/GDrive unreachable → ModelScope
mirrors (`YepingZhao/RSITMD`, `YepingZhao/UCM-Captions`), canonical
dataset_*.json + image archives, authors' standard splits. Converted by
`src/convert_rs_datasets.py` (RSITMD: 452 test imgs / 2,260 q; UCM: 210 /
1,050). TIFF support added to encode_gallery. We now cover KTIR's full
benchmark trio + Flickr30k + COCO. Logs `e20.log`, `e20b.log`.

| dataset | baseline R@1/5/10 | best query-KG ΔR@1 | gallery-cap ΔR@1 | KG incr. |
|---|---|---|---|---|
| RSITMD | 8.81 / 27.88 / 43.14 | +0.13 (reweighted, ns) | −0.13 (ns) | +0.13 (ns) |
| UCM | 8.67 / 36.38 / 60.19 | +0.19 (reweighted, ns) | 0.00; **R@5 +2.5, R@10 +3.1** | −0.09 (ns) |

**Finding F20: both new RS datasets replicate the established pattern** —
small positive query-side KG effect under the reweighted treatment
(+0.13/+0.19, same band as RSICD's +0.11..0.22, underpowered individually),
natural-image BLIP captions neutral-to-harmful at R@1 (F15/F18 replicated on
2 more RS datasets), KG increment ≈ 0 with slight positive lean on RSITMD.
UCM novelty: caption fusion helps deeper ranks (R@5/R@10 +2.5..3.1) even
where R@1 is flat. RS query-side effects are consistent in DIRECTION across
all three RS datasets — a pooled/meta significance test across RSICD+RSITMD+
UCM is the right next statistical step if a defense-grade claim is needed.

### E21. NWPU-Captions — 4th remote-sensing benchmark (3,150 test imgs, 15,750 q)
Acquired via ModelScope (`YepingZhao/NWPU-Captions`), converted by
`src/convert_nwpu.py` (class-keyed JSON, raw..raw_4 caption fields). The
largest RS test set — 7× RSICD — brings real statistical power. Log `e21.log`.

| config | R@1 | R@5 | R@10 | ΔR@1 |
|---|---|---|---|---|
| A baseline | 2.38% | 9.90% | 16.86% | — |
| C reweighted query-KG | 2.43% | 9.87% | 16.91% | **+0.05, CI [−0.03,+0.13], p=0.107** |
| best gallery-caption | 2.47% (α=.95) | 9.43% | 16.17% | +0.09 (ns) |

**Finding F21: 4th consecutive RS dataset with a positive-direction
query-side KG effect** (RSICD +0.11, RSITMD +0.13, UCM +0.19, NWPU +0.05) —
and 4th replication of captions-don't-transfer (F15/F18). NWPU's baseline
(2.4% R@1 over 3,150 same-domain images) is also the clearest illustration of
F10's in-domain-difficulty claim on a real benchmark. → pooled test E22.

### E22. Pooled RS-domain significance test (stratified paired bootstrap)
Config: reweighted top-2, fused α=0.9 (same as per-dataset config C).
Stratified resampling within each dataset, combined weighted by query count.
Script `src/pooled_rs_test.py`, log `e22.log`.

| dataset | queries | ΔR@1 |
|---|---|---|
| RSICD | 5,465 | +0.110 |
| RSITMD | 2,260 | +0.133 |
| UCM | 1,050 | +0.190 |
| NWPU | 15,750 | +0.051 |
| **POOLED** | **24,525** | **+0.077, 95% CI [−0.000, +0.155], p = 0.0267** |

(R@5 +0.037 p=0.28, R@10 +0.082 p=0.12 — R@1 is where the effect lives.)

**Finding F22: query-side relation-aware KG expansion yields a small but
statistically significant R@1 gain across the remote-sensing domain
(p=0.027, 4 datasets, 24.5k queries, same-direction on every dataset).**
This is the defense-grade version of the RS query-side claim: individually
underpowered, jointly significant. Honest framing: the effect is ~+0.08 pts —
real, cheap (training-free), but small; its value is mechanistic (it locates
WHERE knowledge helps) rather than headline-sized.

### E23. RS scale test — in-domain (Million-AID) vs web (DataComp) distractors
100k Million-AID aerial embeddings stream-encoded on dell3 WITHOUT storing
images (`src/stream_millionaid.py`, ~35 min; `distractors_millionaid_100000`).
RSICD 5,465 queries; KG config = reweighted fused α=0.9. Log `e23.log`.

| distractors | N | base R@1 | KG ΔR@1 (p) |
|---|---|---|---|
| — | 0 | 5.45% | +0.11 (0.18) |
| in-domain | 10k | 2.63% | −0.04 (0.77) |
| in-domain | 100k | 1.06% | −0.07 (0.91) |
| web | 10k | 4.70% | +0.13 (0.12) |
| web | 100k | 3.28% | +0.05 (0.27) |

**Finding F23a: F10 replicates in RS, amplified — 10k in-domain aerial
distractors halve R@1 (5.45→2.63) while 10k web distractors cost only 0.75;
at 100k the in-domain gallery is 3× harder (1.06 vs 3.28).** In-domain
density is THE difficulty driver in both natural and RS domains.

**Finding F23b (honest boundary): the pooled-significant RS query-side KG
effect (F22) does NOT survive in-domain scale** — it fades to noise/slightly
negative once ≥10k aerial distractors are present (mirrors F11 exactly:
nudge-sized gains die when margins shrink). The F22 claim holds for
benchmark-sized galleries; at realistic scale, training-free text-side
expansion — query OR gallery — is not enough in-domain. Thesis framing: this
is the cleanest statement of the open problem the thesis leaves behind
(discriminative, domain-adapted enrichment), not a failure of the analysis.

### E24. Domain-tuned frozen backbone — RemoteCLIP ViT-B-32, all 4 RS datasets
Answers "RS-tuned CLIPs exist, why not use them?" with data (run 2 days
before 中期). Frozen published checkpoint `chendelong/RemoteCLIP` (TGRS 2024,
via hf-mirror, 605 MB) loaded through open_clip; identical E22 protocol,
expansion texts held fixed (B/32-selected, cached) — only the encoder
changes. Still zero training on our side. Script `src/remoteclip_rs.py`,
log `e24.log`, feature caches `*_remoteclip`. NOTE: pip install
open_clip_torch upgraded pillow and broke conda PIL (GLIBCXX) — fixed by
`pip install --force-reinstall --no-deps pillow`.

| dataset | base R@1 (CLIP → RemoteCLIP) | query-KG ΔR@1 (CLIP → RemoteCLIP) |
|---|---|---|
| RSICD | 5.51 → 10.50 | +0.11 → +0.18 (p=0.051) |
| RSITMD | 8.81 → 20.00 | +0.13 → +0.04 (ns) |
| UCM | 8.67 → 17.52 | +0.19 → +0.10 (ns) |
| NWPU | 2.38 → 3.66 | +0.05 → −0.08 (ns) |
| **POOLED (24,525 q)** | — | **+0.077 (p=0.027) → −0.004 (p=0.545)** |

**Finding F24a: RemoteCLIP roughly doubles every RS baseline (2×–2.3×)** —
the RS domain gap is a data problem, fixed by domain-tuned weights
(consistent with F19's "not a capacity problem").

**Finding F24b: the pooled query-KG effect vanishes on the domain-tuned
backbone (+0.077 sig → −0.004 ns) — third backbone family confirming the
repair mechanism (F15b) by prediction:** a domain-adapted text tower is no
longer a deficient channel, so there is nothing left to repair. Only RSICD
(weakest RemoteCLIP baseline of the double-digit trio) retains a borderline
positive increment. Deployment guidance this yields: use KG expansion when
stuck with a general backbone; a domain-tuned backbone supersedes it —
knowledge injection and domain adaptation are substitutes, not complements,
on the query side.

## 2026-07-25 — Session: KAIS/KBS paper preparation (supervisor direction)

#### E24 addendum (2026-09-05): absolute R@1 values used in paper Table III
Paper Table III reports absolutes; the log above keeps baseline and Δ
separately. Derivation (t2i R@1, one matched run, ≤0.06 protocol footnote in
the paper):

| dataset | CLIP zero-shot | + DCKF-Q | RemoteCLIP frozen | + DCKF-Q |
|---|---|---|---|---|
| RSICD  | 5.51 | 5.51+0.11 = **5.62**  | 10.50 | 10.50+0.18 = **10.68** |
| RSITMD | 8.81 | 8.81+0.13 = **8.94**  | 20.00 | 20.00+0.04 = **20.04** |
| UCM    | 8.67 | 8.67+0.19 = **8.86**  | 17.52 | 17.52+0.10 = **17.62** |
| NWPU   | 2.38 | 2.38+0.05 = **2.43**  | 3.66  | 3.66−0.08 = **3.58**  |

KTIR context row (italic in Table III): t2i R@1 20.55 (RSICD) / 31.46
(RSITMD) / 19.81 (UCM), transcribed from Mi et al., TGRS 2024, Tables V / VI /
IV (PDF page 9 render, 2026-07-27); KTIR reports no NWPU. Full audit of all
paper tables II–VIII against JSONs/log performed 2026-09-05: no discrepancies.

### E29–E36. Reviewer-driven re-run campaign (2026-09-05, dell3) — see results/REVIEW_RERUN_2026-09-05.md
E29 reviewer baselines (src/reviewer_baselines.py): embedding-space PRF, caption-only gallery, alpha/beta/k sweeps,
per-item caption gate, all six datasets. E30 100k Million-AID distractors re-streamed WITH BLIP-large aerial captions
(src/stream_millionaid_captions.py). **E31 fp16-normalization bias found (baseline fp16-normalized, fused configs
fp32-renormalized; Flickr baseline moves +0.16, p<.001, from renormalization alone) → fixed in
baseline_retrieval._unit32; E32 re-derived every table (e32_server_run.sh).** E33 RSICD gallery channel at scale
(src/rs_scale_gallery_eval.py): +0.48/+0.66 at 1,093 → ≤0 (n.s.) at ≥10k in-domain distractors. E34 Table VIII fp32.
E35 legacy-subset re-derivations (IV-F1/F2/F3, E7). E36 Flickr gallery channel at scale
(src/flickr_scale_gallery_eval.py): +2.26/+2.52/+2.56/+2.38 at 1k/6k/11k/31,783 in-domain gallery, all p<1e-4.

**F29: PRF (knowledge-free) hurts natural images (-0.4..-1.5) and ties the KG on RS; KG+PRF additive on RSICD (+0.37).**
**F29b: per-item caption gating breaks score comparability (hurts everywhere) — caption channel cannot be gated.**
**F29c: caption-only retrieval on RSICD with prompted BLIP-base captions +1.21 (p=.004) at benchmark scale; collapses at scale.**
**F31: fp16 rounding of cached norms is the same size as the small query-side effects; all numbers now fp32.**
**F33/F36: the gallery channel is robust to a 32x in-domain gallery on natural images and vanishes at 10x on RSICD —
the repair account's predicted asymmetry.**
**F35: the IV-F1 "+0.40" was the adaptive-alpha config, not the paper's fixed-beta strategy (+0.26 on legacy subset).**
**F35b: the E15 broken-channel KG increment (+0.24) does not reproduce (-0.07, n.s.).**

### Addendum 2026-09-05 (figure overhaul): embedding-geometry numbers behind Fig. 6
Similarity matrices for one caption-image pair per RSICD test category
(30 categories) are persisted in `results/embed_geometry/{S0,S1}.npy`
(baseline / DCKF query expansion, top-2, tau=0.55, beta=0.9) with
`cats.json`; producer `paper/make_fig_embed_matrices.py` (same computation
as `make_fig_embed.py`, heatmap part). Verified statistics:
- 30x30: matched-pair mean 0.2761 -> 0.2775; unmatched mean 0.2152 -> 0.2165;
  max |delta| over 900 cells 0.0069; matched pair is the ROW MAXIMUM in only
  15/30 (baseline) and 14/30 (DCKF) rows. The former paper sentence "matched
  pairs dominate their rows" was therefore an overclaim and has been removed.
- 10 largest categories (airport, beach, bridge, denseresidential,
  industrial, parking, pond, river, storagetanks, viaduct; same ten as the
  t-SNE figure): matched mean 0.282 -> 0.284; unmatched 0.217 -> 0.219;
  matched pair is the row maximum in 9/10 rows in both panels; max |delta|
  0.0052. These are the numbers now stated in Sec. IV-G and Fig. 6.
- Fig. 4 (dose-response) values: captions-only 4.61 / 5.54 / 5.86 and
  captions+KG 4.85 / 5.54 / 6.06 for BLIP-base, BLIP-base+aerial,
  BLIP-large+aerial (E15 -0.84 & KG +0.24 p=.084; E18 +0.09 & KG 0.00;
  E28 +0.41 & KG +0.20 p=.016; combined +0.61 p=.0053).
- Fig. 5 provenance: `figures/find_qualitative.py` uses the first 1,000
  Flickr30k images (development subset), so the caption's "development
  split" is correct; the old image footer said "1k test" and was wrong.

### E37–E39. Review-round-3 controls (2026-09-05): gate isolation, caption repair on sibling RS datasets, expansion latency
Scripts: `src/gate_ablation.py` (E37), `src/rs_large_aerial_captions.py` + `src/reviewer_baselines.py --cap-suffix {aerial,large_aerial}` (E38),
`src/expansion_latency.py` (E39, local M3/MPS). Logs/JSONs: `results/e37_e38/` (e37_e38.log, e37b.log, gate_ablation_*.json,
reviewer_{rsitmd,ucm,nwpu}_*_{aerial,large_aerial}.json, blip_captions_*_{aerial,large_aerial}.json), `results/expansion_latency_rsicd.json`.

**E37 (gate isolation).** The paper had never run fusion without a gate: knowledge_baselines.py applied MIN_SIM=0.55 in both
operating points and the expansion cache carried a 0.5 floor (QueryExpander top_k=5, sim_threshold=0.5). Rebuilt the pool with
no floor, top_k=50 (mean 24 cands/query; 27% of Flickr / 1.5% of RSICD candidates below 0.5). Discovered a hidden stage: the
cache keeps the TOP-5 by original RELATION_PRIORS x cos (+ direction discount), then rerank picks 2 of those 5. Factorial at parity
(pool top5|deep x scoring typed|prior-only x gate .55|off x fusion|text), official splits:

| Flickr30k (base 58.86) | fusion gate on | fusion gate off | text gate on | text gate off |
|---|---|---|---|---|
| top-5, typed | +0.38 (p1=.008) | **+0.50 (p1=.0007)**; vs gate-on +0.12 CI[0,+.26] p1=.032 | −2.28 | −2.39 |
| top-5, prior-only | +0.40 | +0.44 | −2.34 | −2.55 |
| deep, typed | +0.14 (n.s.) | +0.06 | −3.32 | −4.56 |
| deep, prior-only | +0.14 | +0.10 | −3.74 | −5.26 |

| RSICD (base 5.40) | fusion on | fusion off | text on | text off |
|---|---|---|---|---|
| top-5, typed | +0.18 (p1=.048) | +0.20 (p1=.034) | −0.93 | −0.93 |
| top-5, prior-only | +0.20 | +0.22 (p1=.023) | −0.84 | −0.84 |
| deep, typed | +0.02 | +0.02 | −1.19 | −1.22 |
| deep, prior-only | +0.06 | +0.06 | −1.15 | −1.18 |

Gate changes the k=2 selection for 28.5% (Flickr) / 1.2% (RSICD) of queries (top-5 pool). Parity with the paper cache: identical
tau=.55 selections for 100% / 98.9% of queries.

**Finding F37: the drift gate is NOT the enabling component.** Fusion is positive with or without the gate (gate-off is >= gate-on
on both datasets); substitution is destructive with or without it; cos in the score is inert once the pool is fixed. The query-side
gain is carried by the top-5 pool cap (edge prior x cos): reranking the deep pool drops Flickr +0.38 -> +0.14 (p1=.96 vs top5),
RSICD +0.18 -> +0.02. Paper reframed: contribution 1, III-B, III-C (pool cap disclosed), Algorithm 1 (Pool step), new IV-F7 +
Table XIII, Fig 1 caption, Fig 8 regenerated from these JSONs (official splits, tau from 0), Discussion, Conclusion, abstract.

**E38 (caption repair on RSITMD/UCM/NWPU).** BLIP-base+aerial and BLIP-large+aerial captions generated for the three galleries
(23 ms/img V100); reviewer_baselines at parity. Caption channel (C vs A, alpha=.9) / query-KG increment (D vs C) / both (D vs A):

| dataset | uncond | base+aerial | large+aerial |
|---|---|---|---|
| RSICD (5.40) | −0.79 / +0.07 / −0.71 | +0.01 / +0.09 / +0.10 | **+0.48 (p1=.020)** / +0.18 (p1=.027) / **+0.66 (p1=.0025)** |
| RSITMD (8.81) | −1.29 / +0.27 / −1.02 | −0.31 / +0.04 / −0.26 | +0.18 / +0.08 / +0.26 |
| UCM (8.67) | −0.68 / −0.28 / −0.96 | +0.28 / −0.38 / −0.09 | +0.38 / −0.38 / −0.00 |
| NWPU (2.37) | −0.04 / −0.05 / −0.09 | +0.06 / +0.07 / +0.13 | **+0.29 (p1=.002)** / −0.01 / **+0.28 (p1=.003)** |

**Finding F38: the caption repair generalizes in sign on 4/4 RS datasets (significant on RSICD and NWPU); the KG increment does
not (RSICD-only, +0.18, two-sided p=.054).** NOTE: old Table VI rows 1–2 mixed alpha=.95 and the E15 gallery-caption-KG
mechanism (+0.16/−0.02, −0.79/−0.07); now all rows from reviewer_baselines at alpha=.9 with query-side KG. Fig 4 regenerated.

**E39 (latency, M3/MPS, 500 RSICD queries).** bare query encode 8.9 ms median; expansion as deployed 186 ms (p90 226; matches
the paper's 160–230); ConceptNet lookup 3.1 ms; with a concept-embedding cache (1,916 unique concepts, 12.5 s offline) 32.9 ms
(p90 36.9); 28 candidates/query. V100 uncached: 14 ms/q (E37 pool build). Ported to III-C and IV-H, reconciled with the intro.

**Statistics policy change (review item 4):** paper now reports two-sided bootstrap p = 2·min(P(Δ≤0),P(Δ≥0)); stars = two-sided
p<.05. Cells that lost stars: Table VII Flickr WordNet +0.24, RSICD fixed +0.18, RSICD relation-aware +0.18; Table VIII RSICD KG
+0.18, NWPU PRF +0.11, NWPU KG+PRF +0.11; Table IX RSICD DCKF +0.18; Table XI RSICD Q +0.18; Table XII alpha=.95 RSICD +0.31,
beta=.9 RSICD, k=2 RSICD, k=3 COCO +0.13; Fig 3 RSICD Q; Fig 4 KG +0.18. 41 starred claims: all pass BH 5%; 21 pass Holm
(natural-image gallery effects + Flickr k=1 + Flickr gate-off).

### E41. RSICD bidirectional eval with BLIP-large aerial captions, fp32 + image-cluster bootstrap (2026-09-05 evening; review R6 item 3g)
The paper's RSICD i2t sentence (5.40->6.50 R@1, mR 16.44->17.39) traced to E28, a PRE-fp32 run (baseline 5.45).
`src/bidirectional_eval.py` gained `--caption-suffix` (e.g. `_large_aerial`); run on dell3 with `CLUSTER_BOOT=1`.
Output `results/imgboot/bidirectional_rsicd_1093_5_large_aerial.json` + `e41_bidir_rsicd_large.log`.

| config | t2i R@1/5/10 | i2t R@1/5/10 | mR | t2i p2 | i2t p2 |
|---|---|---|---|---|---|
| A baseline | 5.40/17.73/27.94 | 5.40/16.01/26.17 | 16.44 | | |
| B query-KG | 5.58/18.02/28.16 | 6.04/16.10/25.71 | 16.60 | .157 | .060 |
| C large-cap a=0.9 | 5.87/17.79/29.17 | 6.50/18.02/26.99 | 17.39 | .118 | .072 |
| D both | 6.06/17.73/28.82 | 6.13/17.47/26.26 | 17.08 | .030 | .266 |

**F41: the E28 i2t numbers reproduce exactly under fp32 (i2t was never touched by the fp16 asymmetry); no i2t gain on
RSICD is significant.** Paper IV-C now names the configuration (captions alone) and says so.

### E40. Image-cluster paired bootstrap for every t2i p-value (2026-09-05 night; review R5 item 3a)
Five captions share one target image, so caption-level resampling overstates precision. `advanced_expansion.paired_bootstrap`
now resamples IMAGE CLUSTERS when `CLUSTER_BOOT=1` and the script has registered `gt` via `set_boot_clusters(gt)` (10 scripts
patched; `pooled_rs_test` cluster-aware; i2t already used images). Runner `e40_server_run.sh`; outputs `results/imgboot/*.json`
+ `e40_imgboot.log` + `pooled_rs_imgboot.log`; caption-level originals kept in `results/capboot_backup/`. **`results/imgboot/`
is now the source of record for every t2i p / CI in the paper**; deltas unchanged (bootstrap-mean rounding aside).
Compare script `results/e40_compare.py`: 680 rows, 13 significance flips at two-sided .05. Flips that touch the paper:

| cell | Δ | p2 caption | p2 image | paper action |
|---|---|---|---|---|
| RSICD BLIP-large captions alone (C) | +0.48 | .040 | .118 | unstar in Tables VI, XI, XII; IV-C reworded; Fig 4 label |
| RSICD KG+PRF | +0.37 | .019 | .051 | unstar Table VIII; "additive in direction" |
| RSICD caption-only α=0 (aerial base) | +1.21 | .007 | .080 | "not significant" in IV-D + Table VIII caption |
| Table XIII RSICD top-5 prior gate-off | +0.22 | .046 | .084 | unstar |
| parity RSICD ours vs WordNet | +0.22 | .049 | .089 | IV-D: "none of eighteen" separates |
| pooled RS (stratified, image clusters) | +0.094 | CI[+.016,+.167] p=.016 | CI[+.012,+.175] p=.026 | Table V, IV-C, Discussion |
Unchanged in significance: RSICD D +0.66 (p .030), NWPU C +0.29 (.018), NWPU D +0.28 (.024), all Flickr/COCO gallery
cells (<1e-4), Flickr Q +0.38 (.016), Flickr 6k/11k Q (.0026/.011), LLM fused (.009), ViT-L Q (.038). Other quoted p's
updated: RSICD Q .096->.157, KG increment .054->.102, gate-off vs on .065->.092, deep vs top5 .081->.083, Table X column.
BH/Holm recount: 37 starred claims, all pass BH; 22 pass Holm (natural-image gallery effects + Flickr k=1, gate-off, 6k Q).
IV-A now states the resampling unit. Fig 8 bands regenerated from imgboot gate JSONs.

### E25. Bidirectional evaluation (i2t + mR) — implementation + external validation
Motivation: KAIS/KBS (supervisor's target venues) and the KTIR-comparable
literature report BOTH directions + mR; all prior entries are t2i-only.
New `src/bidirectional_eval.py`: i2t rank = #captions scoring strictly higher
than the image's best GT caption (standard "any of 5 in top-K"), same-matmul
GT gather (E10 rule), mR = mean of 6 recalls; paired bootstrap in both
directions (units: captions for t2i, images for i2t). Configs A/B/C/D
constructed identically to E17. Full metrics dumped to
`results/bidirectional_{key}.json`.

External anchors (clip_benchmark, OpenAI ViT-B/32, Karpathy splits,
github.com/mlfoundations/open_clip retrieval results):

| protocol (local Mac/MPS) | t2i R@1/5/10 | i2t R@1/5/10 | anchor t2i | anchor i2t |
|---|---|---|---|---|
| COCO 5k Karpathy | 30.43/55.97/66.85 | 50.16/75.02/83.52 | 30.44/55.94/66.87 | 50.12/75.00/83.52 |
| Flickr30k Karpathy 1k (NEW `flickr30k_k1k`) | 58.84/83.52/90.00 | 78.80/94.90/98.20 | 58.78/83.56/90.02 | 78.90/94.90/98.20 |
| Flickr30k first-1000 (legacy internal) | 59.26/84.56/90.60 | 77.10/95.70/98.40 | — | — |

**Finding F25a: both directions validated against an independent external
implementation to ≤0.1 pt on both benchmarks.** COCO exact to ±0.04.

**Finding F25b (protocol catch, paper-critical): our legacy "Flickr 1k" is
the first 1,000 images alphabetically, NOT the Karpathy 1k test split**
(explains the 1.8-pt i2t gap vs anchor; t2i happens to sit 0.5 high).
E1–E24 remain valid as internally-consistent comparisons, but PAPER numbers
on Flickr must use the new `flickr30k_k1k` dataset entry (official split
from mehdidc/retrieval_annotations — the file clip_benchmark itself uses;
1,000 images verified present locally, symlinked in `data/flickr30k_k1k`).
COCO was already the true Karpathy 5k test — nothing to fix there.
Flickr KG/caption caches must be regenerated for the k1k query set on dell3
(expansions keyed by query hash; BLIP captions per image ≈ 25 s).

First bidirectional data point (legacy Flickr 1k, config B reweighted KG,
a_q=0.9): t2i +0.24 (p=0.044), i2t +0.20 (p=0.29, ns over 1,000 images) —
query-side KG does not hurt the reverse direction.

Remaining E25 matrix (GPU server): all 6 datasets ×
{A,B,C,D} both directions + regenerated k1k caches + knowledge baselines
(naive, random-triplet, WordNet) for the KAIS comparative-study requirement.

### E25 (cont.) — Full bidirectional matrix on dell3 (V100, cpi=5, all datasets)
Caches generated: expansions_flickr30k_k1k_1000_5 (5k q), expansions_coco5k_5000_5
(25k q), blip_captions_flickr30k_k1k_1000_5 (21 ms/img; needed HF_HUB_OFFLINE=1 —
transformers HEAD-checks huggingface.co even with a warm cache and the lab network
can't reach it). blip_captions_coco5k_5000_5 = alias of the _1 file (same images).
JSONs: results/bidirectional_{ds}_{n}_5.json. Configs as E17 (α_q=0.9, α_g=0.9).

Headline (flickr30k_k1k = OFFICIAL Karpathy split, dell3):
| config | t2i R@1/5/10 | i2t R@1/5/10 | mR |
|---|---|---|---|
| A baseline | 58.70/83.50/89.98 | 78.80/94.90/98.20 | 84.01 |
| B +query-KG | 59.16/83.72/90.14 | 78.70/95.00/98.30 | 84.17 |
| C +captions | 61.12/84.92/90.44 | 77.30/94.80/97.90 | 84.41 |
| D both | 61.14/84.96/90.68 | 77.80/94.70/98.00 | 84.55 |
t2i bootstraps: B +0.46 (p=.0023), C +2.42 (p<1e-4), D +2.44 (p<1e-4).
i2t bootstraps: B −0.10 ns, C −1.50 ns, D −0.99 ns.
COCO cpi=5: C +2.82**, D +2.93** t2i; B t2i +0.03 ns but **B i2t +0.44
(p=.0125)** — KG-expanding the caption gallery helps image→text on COCO.
UCM: D i2t +3.79 (p=.055, borderline; note UCM n=210 imgs). RS t2i: captions
negative everywhere (RSICD −0.84, RSITMD −1.24, dir. robust but ns), F15/F18
replicated bidirectionally; NWPU all ns.

**F25c: the headline survives the official split and is bidirectionally
honest** — query-KG is stronger on Karpathy-Flickr than legacy (+0.46* vs
+0.24*); caption channel +2.42**/+2.82** t2i on Flickr/COCO while i2t stays
within noise (caption fusion perturbs image-as-query geometry; mR still rises
84.01→84.55 / 60.32→62.14). **F25d: the repair mechanism holds in reverse**
— the one significant i2t gain (+0.44*, COCO) appears exactly where the KG
enriches the *gallery-side text channel*, mirroring the t2i logic.

### E26. Knowledge-SELECTION baselines at exact parity (src/knowledge_baselines.py)
All strategies share pools/sim-gate(0.55)/k=2/template/fusion(α=0.9); only
selection differs: random (KTIR's strategy), original priors, WordNet-synonym
source swap, relation-aware (ours). 6 datasets, both directions.
JSONs: results/baselines_{ds}_{n}_5.json.

**F26 (honest, reframes the mechanism): at the embedding-fusion operating
point, selection strategy is second-order.** On k1k ALL four strategies gain
+0.32..+0.48 over baseline (each individually significant); ours vs
random/original/wordnet is ns everywhere except ours>random on RSITMD
(+0.36, p=.042 — 1 of ~18 tests, do not oversell). Flickr pools are an IsA
monoculture (16,205/17k candidates ≥ sim .55) so strategies mostly pick the
same concepts (selection differs on 7.5% of queries; top-1 flips 8/377, all
between wrong images). ⇒ The robust never-harmful gain is delivered by the
DRIFT-CONTROLLED FUSION machinery (CLIP-sim gate + gentle α + k=2), not by
which candidate survives selection. Relation-aware weighting still earns its
place where it demonstrably protects: text-level expansion (F3/F4: naive
templates −4.8), IsA demotion, and 2-hop self-suppression (F16) — but paper
wording must credit the gate+fusion, and present relation priors as the
protective component, not the source of the headline delta.
Paper baseline table: ours ties-or-beats every alternative on every dataset
(never loses), which is the defensible claim.

### E27. LLM query-rewrite baseline (src/llm_rewrite_baseline.py; Qwen2.5-1.5B-Instruct, greedy, frozen)
The "why not just ask an LLM?" reviewer row, at parity (same gentle fusion).
Rewrites cached: results/llm_rewrites_{key}.json; results: llm_baseline_*.json.

| config | k1k t2i R@1 (Δ, p) | rsicd t2i R@1 (Δ, p) |
|---|---|---|
| A baseline | 58.70 | 5.45 |
| L-replace (rewrite alone) | 50.74 (−7.97, p≈1) | 4.78 (−0.68, p=.98) |
| L-fused α=0.9 | 59.30 (+0.60, p=.0012) | 5.60 (+0.15, ns) |
| B relation-KG α=0.9 | 59.16 (+0.46, p=.0023) | 5.56 (+0.11, ns) |
KG vs L-fused: ns on both (−0.14 / −0.04). KG vs L-replace: +8.43** / +0.79*.

**F27: the drift-controlled fusion recipe is source-agnostic and is THE
enabling component.** Naive LLM rewriting is catastrophic (−8 R@1 on k1k);
under our gate+gentle-fusion the LLM only matches ConceptNet (ns diff both
datasets). KG advantages at equal effect: no 1.5B-param model in the query
path (SQLite lookup, <2 ms), deterministic, auditable edges. Paper answer
to the LLM objection is now measured, not argued.

### E28. Stronger captioner on RSICD gallery channel (src/rsicd_blip_large.py; BLIP-large + "an aerial photograph of")
Completes the E15→E18→E28 captioner-quality dose-response: base harmful
(−0.84) → base+prompt neutral (+0.09 ns) → large+prompt POSITIVE.

| config | t2i R@1/5/10 | i2t R@1 | mR |
|---|---|---|---|
| A baseline | 5.45/17.79/27.98 | 5.40 | 16.45 |
| C large-cap α=0.9 | 5.86/17.79/29.17 | 6.50 | 17.39 |
| D both α=0.9 | 6.06/17.73/28.86 | 6.13 | 17.08 |
Bootstraps: C +0.41 (p=.042, CI grazes 0), **D +0.61 (p=.0053)**,
**KG increment on the large-caption channel +0.20 (p=.016)** (α=0.95 variants
weaker but same direction).

**F28: the repair-mechanism prediction closes on RSICD — fix the caption
channel and the KG increment turns positive AND significant** (predicted in
E18: "stronger captioner → KG has a working channel to amplify"). First
significant knowledge gain on an RS dataset: 5.45 → 6.06 (+11% relative),
training-free. RS section of the paper now has a positive result whose
mechanism was predicted in advance.

### E26b. Selection baselines at the TEXT operating point (--mode text)
Expanded text REPLACES the query (no fusion/gate). k1k: every strategy hurts
(random −2.80, original −2.20, wordnet −1.98, ours −2.16, all p≈1); ours is
the least-bad vs random (+0.64, p=.041). RSICD: all hurt (−0.7..−0.95).

**F26b: no selection strategy makes raw text-level expansion safe — the
protection lives entirely in the fusion gate.** Sharpens F26: the paper must
credit the gate/fusion for the gains; relation-awareness contributes the
protective margin vs random selection, interpretability (edge = auditable
reason), and the 2-hop self-suppression (F16) — not the headline delta.

## Next session
- Selective + relation-weighted expansion targeting F4/F5 (adaptive α, learned
  relation priors, expand-only-ambiguous-queries policy)
- RSICD/RSITMD/UCM acquisition (remote sensing = domain where expansion should
  actually help; short queries + vocabulary mismatch)
- spaCy NER seed extraction (replace n-gram matching)
- Statistical significance tests (paired bootstrap) once a config beats baseline
