import re, json, sys
from pathlib import Path
P = Path("/Users/home/RA-KG-T2I/paper/main.tex"); s = P.read_text()
R = Path("/Users/home/RA-KG-T2I/results/fp32_rederivation")
FAIL=[]
def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n: FAIL.append((c, old[:100])); return
    s = s.replace(old, new)

# ---------------- Abstract (keep <= 250 words)
rep(r"""a vision--language backbone with external knowledge, which is costly,
dataset-specific, and inapplicable when the backbone is frozen or a black
box.""", r"""a vision--language backbone with external knowledge, which is costly and
inapplicable when the backbone is frozen or a black box.""")
rep(r"""gallery enrichment improves \ratk{1} by up to $+2.9$ points, query
expansion gives small but consistent gains where queries are
domain-mismatched, and upgrading the captioner produces, to our knowledge,
the first statistically significant training-free knowledge gain on a
remote-sensing retrieval benchmark.""", r"""gallery enrichment improves \ratk{1} by up to $+2.9$ points and holds at
$+2.4$ when the in-domain gallery grows 32-fold, query expansion gives
small but consistent gains where queries are domain-mismatched, and
upgrading the captioner produces, to our knowledge, the first statistically
significant training-free knowledge gain on a remote-sensing benchmark,
a gain confined to benchmark scale.""")

# ---------------- Fig 1 caption
rep(r"""this costs
  $2.16$ \ratk{1}.""", r"""this costs
  $2.28$ \ratk{1}.""")
rep(r"""same knowledge source yields $+0.46$ \ratk{1} ($^{*}p<0.05$;""",
    r"""same knowledge source yields $+0.38$ \ratk{1} ($^{*}p<0.05$;""")

# ---------------- Contributions
rep(r"""and yields the first statistically significant knowledge gain on an RS
  benchmark, an 11\% relative improvement on RSICD that
  our mechanism predicted before the experiment.""",
    r"""and yields the first statistically significant knowledge gain on an RS
  benchmark, a 12\% relative improvement on RSICD that
  our mechanism predicted before the experiment and that, unlike the
  natural-image gain, does not survive in-domain gallery growth.""") if "an 11\\% relative improvement on RSICD that\n  our mechanism predicted before the experiment." in s else None
if "an 11\\%" in s:
    i = s.index("an 11\\%"); print("CONTRIB2 context:", repr(s[i-120:i+140])); sys.exit("fix contribution 2 pattern")
rep(r"""and both retrieval directions, and both channels fade under realistic
  in-domain gallery scale, which we measure with a new 100k aerial
  distractor set.""", r"""and both retrieval directions. Under realistic in-domain gallery
  growth, measured with a new 100k aerial distractor set and the full
  31{,}783-image Flickr30k gallery, the query-side gains and the RS gallery
  gain fade while the natural-image gallery gain holds.""")

# ---------------- Related work: DTIR
rep(r"""pre-trains with event-structure knowledge. All of these require gradient
updates to absorb knowledge noise.""", r"""pre-trains with event-structure knowledge. Most recently in this journal,
DTIR \cite{lin2026dtir} retrieves images by reasoning over LLM-built
digital-twin descriptions and names knowledge-graph integration in its
offline perception phase as future work; the present study asks what such
integration can deliver when no training is possible. All of these require
gradient updates to absorb knowledge noise.""")

# ---------------- III-B: framing + ranges
rep(r"""The same three steps are followed by every knowledge injection in this
paper.""", r"""Every query-side knowledge injection in this paper follows three steps;
the gallery channel (Sec.~\ref{sec:gallery}) uses the third alone, for a
reason given there.""")
rep(r"""$1.98$ to $2.80$ \ratk{1} and RSICD by $0.69$ to $0.95$, regardless of""",
    r"""$2.18$ to $2.96$ \ratk{1} and RSICD by $0.67$ to $0.95$, regardless of""")

# ---------------- III-C: weights table, drop "adaptive"
rep(r"""The relation weights $w(r)$ are \emph{adaptive} rather than intuitive: our
ablations (Sec.~\ref{sec:abl-rel}) show hypernymic IsA edges are the
dominant noise source, so IsA is demoted ($w{=}0.4$) while scene-grounding
relations (AtLocation, UsedFor; $w{=}1.0$) are promoted.""",
r"""The relation weights $w(r)$ (Table~\ref{tab:weights}) were fixed once on
the development subset (Sec.~\ref{sec:abl-rel}), where hypernymic IsA edges
are the dominant noise source, and used unchanged on every dataset,
including remote sensing: IsA is demoted ($w{=}0.4$) while scene-grounding
relations (AtLocation, UsedFor; $w{=}1.0$) are promoted.

\begin{table}[t]
  \centering
  \caption{Relation weights $w(r)$ for the nine whitelisted ConceptNet
  relations, fixed on the Flickr30k development subset and never re-tuned.}
  \label{tab:weights}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lc lc lc}
    \toprule
    Relation & $w$ & Relation & $w$ & Relation & $w$ \\
    \midrule
    AtLocation & 1.0 & Synonym   & 0.8 & HasA      & 0.7 \\
    UsedFor    & 1.0 & MadeOf    & 0.8 & IsA       & 0.4 \\
    PartOf     & 0.9 & CapableOf & 0.7 & RelatedTo & 0.3 \\
    \bottomrule
  \end{tabular}
\end{table}""")

# ---------------- III-D: label, contamination, why no gate
rep(r"""\subsection{Gallery-Side Caption Enrichment}""", r"""\subsection{Gallery-Side Caption Enrichment}
\label{sec:gallery}""")
rep(r"""Because the captioner sees only pixels, no ground-truth caption information
leaks into the index.""", r"""Because the captioner sees only pixels, no ground-truth caption information
leaks into the index at retrieval time. The BLIP captioning checkpoints were
fine-tuned on the COCO Karpathy training captions and pre-trained on
corpora that follow the same split, so the Karpathy test images used here
were unseen; the RS datasets lie outside BLIP's training data entirely, and
the RSICD results are therefore the cleaner test of the channel. The
gallery channel is not drift-gated per item: when only some gallery
vectors are fused, fused and unfused items are no longer score-comparable,
and gating the 10\% of captions least similar to their image costs $3.0$
\ratk{1} on Flickr30k and $1.7$ on MS-COCO relative to fusing all of them
(Sec.~\ref{sec:knowledgefree}); uniform fusion is the gate-free design that
keeps every gallery score on one scale.""")

# ---------------- IV-A: precision + statistics conventions
rep(r"""\textbf{Significance.} Paired bootstrap \cite{efron1993bootstrap}
(10{,}000 resamples; units are
captions for t2i and images for i2t); the mean difference, 95\% CI, and
$p(\Delta\!\le\!0)$ are reported for every comparison.""",
r"""\textbf{Significance.} Paired bootstrap \cite{efron1993bootstrap}
(10{,}000 resamples; units are
captions for t2i and images for i2t); the mean difference, a two-sided 95\%
CI, and the one-sided $p(\Delta\!\le\!0)$ are reported for every
comparison, so a CI that touches zero with $p$ slightly below $0.05$ is
consistent, not contradictory. Across the 27 positive claims marked
significant in this paper, all survive Benjamini--Hochberg control at 5\%
false-discovery rate; only the gallery-channel effects on natural images
(Table~\ref{tab:main}, Sec.~\ref{sec:abl-backbone}) survive Holm
family-wise control at $0.05$. \textbf{Precision.} Every feature is
$\ell_2$-normalized in float32 before any comparison. An earlier version of
our pipeline normalized cached features in fp16 while re-normalizing fused
vectors in float32; that asymmetry alone moved the Flickr30k baseline by
$0.16$ points ($p<0.001$), larger than several query-side effects, so all
numbers here come from one float32 run.""")

# ---------------- IV-B text
rep(r"""is the dominant effect on t2i: $+2.42$ \ratk{1} on Flickr30k and $+2.82$
on MS-COCO (both $p<10^{-4}$); the combined system reaches $+2.44$ and
$+2.93$ ($30.45\%\!\rightarrow\!33.38\%$ on COCO). Query expansion adds $+0.46$
($p=0.0023$) on Flickr30k; on MS-COCO, whose 5-caption queries are already
dense, it adds nothing to t2i ($+0.03$, n.s.), the first appearance of""",
r"""is the dominant effect on t2i: $+2.26$ \ratk{1} on Flickr30k and $+2.81$
on MS-COCO (both $p<10^{-4}$); the combined system reaches $+2.24$ and
$+2.93$ ($30.45\%\!\rightarrow\!33.38\%$ on COCO). Query expansion adds $+0.38$
($p=0.008$) on Flickr30k; on MS-COCO, whose 5-caption queries are already
dense, it adds nothing to t2i ($+0.04$, n.s.), the first appearance of""")
rep(r"""baselines; DCKF's $+2.44$ on ViT-B/32 is larger, and it lifts a frozen
ViT-B/32 index to within $1.1$ points of zero-shot ViT-B/16 ($61.14$ vs.\
$62.24$)""", r"""baselines; DCKF's $+2.24$ on ViT-B/32 is larger, and it lifts a frozen
ViT-B/32 index to within $1.1$ points of zero-shot ViT-B/16 ($61.10$ vs.\
$62.24$)""")
rep(r"""published in this journal, frozen DCKF at $61.14$ on Flickr30K sits above""",
    r"""published in this journal, frozen DCKF at $61.10$ on Flickr30K sits above""")
rep(r"""t2i, and i2t moves within noise (every i2t CI crosses zero on Flickr30k),
while $\mR$ still improves ($84.01\!\rightarrow\!84.55$ Flickr30k;
$60.32\!\rightarrow\!62.14$ COCO). One i2t cell is significantly
\emph{positive}: on MS-COCO, expanding the caption gallery with the KG
improves image-to-text retrieval ($+0.44$, $p=0.013$), which is the repair""",
r"""t2i, and on Flickr30k the caption channel costs i2t \ratk{1} $1.6$ points
($77.20$ vs.\ $78.80$; CI $[-3.7,+0.4]$, not significant but consistently
negative across C and D), while $\mR$ still improves
($84.03\!\rightarrow\!84.54$ Flickr30k; $60.32\!\rightarrow\!62.14$ COCO).
One i2t cell is significantly \emph{positive}: on MS-COCO, expanding the
caption gallery with the KG improves image-to-text retrieval ($+0.42$,
$p=0.018$), which is the repair""")

# ---------------- Table II ours rows
rep(r"""    A\; zero-shot baseline & 58.70 & 83.50 & 89.98 & 78.80 & 94.90 & 98.20 & 84.01 & 30.45 & 55.97 & 66.88 & 50.12 & 74.96 & 83.54 & 60.32 \\
    B\; DCKF-Q & 59.16$^{*}$ & 83.72 & 90.14 & 78.70 & 95.00 & 98.30 & 84.17 & 30.48 & 56.00 & 66.91 & 50.56$^{*}$ & \textbf{75.46} & 83.28 & 60.45 \\
    C\; DCKF-G & \textbf{61.12}$^{\ddag}$ & 84.92 & 90.44 & 77.30 & 94.80 & 97.90 & 84.41 & 33.27$^{\ddag}$ & 59.39 & 70.09 & 50.80 & 74.94 & \textbf{83.92} & 62.07 \\
    D\; DCKF & \textbf{61.14}$^{\ddag}$ & \textbf{84.96} & \textbf{90.68} & 77.80 & 94.70 & 98.00 & \textbf{84.55} & \textbf{33.38}$^{\ddag}$ & \textbf{59.49} & \textbf{70.21} & \textbf{50.86} & 75.18 & 83.74 & \textbf{62.14} \\""",
r"""    A\; zero-shot baseline & 58.86 & 83.48 & 89.94 & \textbf{78.80} & 94.90 & 98.20 & 84.03 & 30.45 & 55.95 & 66.87 & 50.14 & 74.98 & 83.52 & 60.32 \\
    B\; DCKF-Q & 59.24$^{*}$ & 83.74 & 90.12 & 78.70 & \textbf{95.00} & \textbf{98.30} & 84.18 & 30.49 & 55.99 & 66.90 & 50.56$^{*}$ & \textbf{75.50} & 83.28 & 60.45 \\
    C\; DCKF-G & \textbf{61.12}$^{\ddag}$ & 84.92 & 90.42 & 77.20 & 94.60 & 97.90 & 84.36 & 33.26$^{\ddag}$ & 59.36 & 70.11 & 50.76 & 74.96 & \textbf{83.94} & 62.06 \\
    D\; DCKF & 61.10$^{\ddag}$ & \textbf{84.96} & \textbf{90.66} & 77.80 & 94.70 & 98.00 & \textbf{84.54} & \textbf{33.38}$^{\ddag}$ & \textbf{59.47} & \textbf{70.22} & \textbf{50.86} & 75.18 & 83.74 & \textbf{62.14} \\""")

# ---------------- IV-C text
rep(r"""yields $+0.077$ \ratk{1}, 95\% CI $[-0.000,+0.155]$, $p=0.0267$ over
24{,}525 queries, every dataset contributing in the same direction.""",
r"""yields $+0.094$ \ratk{1}, 95\% CI $[+0.016,+0.167]$, $p=0.008$ over
24{,}525 queries, every dataset contributing in the same direction. A
knowledge-free alternative, embedding-space pseudo-relevance feedback,
matches this gain on RS and is compared at parity in
Sec.~\ref{sec:knowledgefree}.""")
rep(r"""out-of-domain ($-0.84$ \ratk{1} on RSICD; $-1.24$ on RSITMD)""",
    r"""out-of-domain ($-0.79$ \ratk{1} on RSICD; $-1.29$ on RSITMD)""")
rep(r"""channel from harmful to neutral ($-0.84\!\rightarrow\!+0.09$, n.s.);
upgrading to BLIP-large with the same aerial prompt makes it positive
($+0.41$, $p=0.042$); and on the repaired channel the KG increment is
significant ($+0.20$, $p=0.016$), for a combined
$5.45\!\rightarrow\!6.06$ \ratk{1} ($+0.61$, $p=0.0053$),
to our knowledge the first statistically significant training-free
knowledge gain on an RS retrieval benchmark. The i2t direction improves in
parallel ($5.40\!\rightarrow\!6.50$ \ratk{1}; $\mR$
$16.45\!\rightarrow\!17.39$).""",
r"""channel from harmful to neutral ($-0.79\!\rightarrow\!+0.16$, n.s.);
upgrading to BLIP-large with the same aerial prompt makes it positive
($+0.48$, $p=0.020$); and on the repaired channel the KG increment is
significant ($+0.18$, $p=0.027$), for a combined
$5.40\!\rightarrow\!6.06$ \ratk{1} ($+0.66$, $p=0.0025$),
to our knowledge the first statistically significant training-free
knowledge gain on an RS retrieval benchmark. The i2t direction improves in
parallel ($5.40\!\rightarrow\!6.50$ \ratk{1}; $\mR$
$16.44\!\rightarrow\!17.39$). The gain is a benchmark-scale result: it
does not survive in-domain gallery growth (Sec.~\ref{sec:scale}).""")

# ---------------- Table III
rep(r"""  best training-free result. CLIP/RemoteCLIP rows come from one matched
  run and differ from Tables~\ref{tab:pooled}--\ref{tab:rsicd} baselines
  by $\le$0.06 due to feature-cache regeneration.}""",
    r"""  best training-free result.}""")
rep(r"""    CLIP ViT-B/32, zero-shot        & 5.51  & 8.81  & 8.67  & 2.38 \\
    \quad + DCKF-Q                  & 5.62  & 8.94  & 8.86  & 2.43 \\
    RemoteCLIP (frozen)             & 10.50 & 20.00 & 17.52 & \textbf{3.66} \\
    \quad + DCKF-Q                  & \textbf{10.68} & \textbf{20.04} & \textbf{17.62} & 3.58 \\""",
r"""    CLIP ViT-B/32, zero-shot        & 5.40  & 8.81  & 8.67  & 2.37 \\
    \quad + DCKF-Q                  & 5.58  & 8.94  & 8.86  & 2.42 \\
    RemoteCLIP (frozen)             & 10.50 & 20.00 & 17.52 & \textbf{3.66} \\
    \quad + DCKF-Q                  & \textbf{10.69} & \textbf{20.04} & \textbf{17.62} & 3.58 \\""")

# ---------------- Table IV
rep(r"""    RSICD  & 5{,}465  & $+0.110$ \\""", r"""    RSICD  & 5{,}465  & $+0.183$ \\""")
rep(r"""      $\mathbf{+0.077}$, CI $[-0.000,+0.155]$, $p=0.0267$ \\""",
    r"""      $\mathbf{+0.094}$, CI $[+0.016,+0.167]$, $p=0.008$ \\""")

# ---------------- Table V
rep(r"""  (t2i \ratk{1} deltas vs.\ the 5.45 baseline; paired bootstrap). Combined
  best: $5.45\!\rightarrow\!6.06$ \ratk{1} ($+0.61$, $p=0.0053$); i2t
  $5.40\!\rightarrow\!6.50$.}""",
    r"""  (t2i \ratk{1} deltas vs.\ the 5.40 baseline; paired bootstrap). Combined
  best: $5.40\!\rightarrow\!6.06$ \ratk{1} ($+0.66$, $p=0.0025$); i2t
  $5.40\!\rightarrow\!6.50$.}""")
rep(r"""    BLIP-base, unconditioned      & $-0.84$ & --- \\
    BLIP-base + aerial prompt     & $+0.09$ (n.s.) & $\approx 0$ \\
    BLIP-large + aerial prompt    & $+0.41$ ($p{=}.042$) &
                                    $+0.20$ ($p{=}.016$) \\""",
r"""    BLIP-base, unconditioned      & $-0.79$ & $-0.07$ (n.s.) \\
    BLIP-base + aerial prompt     & $+0.16$ (n.s.) & $-0.02$ (n.s.) \\
    BLIP-large + aerial prompt    & $+0.48$ ($p{=}.020$) &
                                    $+0.18$ ($p{=}.027$) \\""")

# ---------------- Fig 4 caption
rep(r"""  moves from harmful, through neutral, to positive, and the knowledge
  increment on top of it (blue arrows) is not significant on the broken
  channel, zero on the neutral one, and significant only once the channel
  works ($+0.20$, $p{=}0.016$), giving the combined $6.06$.""",
r"""  moves from harmful, through neutral, to positive, and the knowledge
  increment on top of it (blue arrow) is zero or slightly negative on the
  broken and neutral channels and significant only once the channel works
  ($+0.18$, $p{=}0.027$), giving the combined $6.06$.""")

# ---------------- IV-D text + Table VI
rep(r"""strategies converge: on Flickr30k every strategy gains $+0.32$ to $+0.48$
over the baseline, with no consistent significant separation between them
(relation-aware never loses to any alternative on any of the six datasets).""",
r"""strategies converge: on Flickr30k every strategy gains $+0.24$ to $+0.42$
over the baseline, and no strategy separates significantly from any other
on any of the six datasets.""")
rep(r"""is safe: every one degrades Flickr30k by $-1.98$ to $-2.80$ \ratk{1}
(relation-aware is the least damaging vs.\ random, $+0.64$, $p=0.041$) and
RSICD by $-0.69$ to $-0.95$.""",
r"""is safe: every one degrades Flickr30k by $-2.18$ to $-2.96$ \ratk{1}
(relation-aware is the least damaging vs.\ random, $+0.68$, $p=0.033$) and
RSICD by $-0.67$ to $-0.95$.""")
rep(r"""    random (KTIR's)   & $+0.48^{*}$ & $-2.80$ & $+0.07$ & $-0.82$ \\
    fixed priors      & $+0.48^{*}$ & $-2.20$ & $+0.09$ & $-0.69$ \\
    WordNet synonyms  & $+0.32^{*}$ & $-1.98$ & $-0.04$ & $-0.87$ \\
    relation-aware (DCKF) & $+0.46^{*}$ & $-2.16$ & $+0.11$ & $-0.95$ \\""",
r"""    random (KTIR's)   & $+0.34^{*}$ & $-2.96$ & $+0.13$ & $-0.74$ \\
    fixed priors      & $+0.42^{*}$ & $-2.32$ & $+0.18^{*}$ & $-0.67$ \\
    WordNet synonyms  & $+0.24^{*}$ & $-2.18$ & $-0.04$ & $-0.82$ \\
    relation-aware (DCKF) & $+0.38^{*}$ & $-2.28$ & $+0.18^{*}$ & $-0.95$ \\""")

# ---------------- NEW: knowledge-free baselines (end of IV-D, before IV-E)
rows = {}
for ds, fn in [("Flickr30k","reviewer_flickr30k_k1k_1000_5.json"),("MS-COCO","reviewer_coco5k_5000_5.json"),
               ("RSICD","reviewer_rsicd_1093_5.json"),("RSITMD","reviewer_rsitmd_452_5.json"),
               ("UCM","reviewer_ucm_210_5.json"),("NWPU","reviewer_nwpu_3150_5.json")]:
    d = json.load(open(R/fn)); get = {}
    for r_ in d["rows"]:
        get[r_["label"]] = r_
    def cell(label_prefix, star=True):
        lp = label_prefix.strip(); k = next(k for k in get if k == lp or k.startswith(lp + " "))
        v = get[k]["dR1_vs_baseline"]*100; p = get[k]["p_le0"]
        return f"${v:+.2f}$" + ("$^{*}$" if (star and p < 0.05) else "")
    rows[ds] = [cell("PRF k=3 beta=0.9"), cell("PRF k=10 beta=0.9"), cell("DCKF-Q beta=0.9 k=2"),
                cell("DCKF-Q + PRF k=3"), cell("gallery alpha=0.0", star=False)]
tab_rows = "\n".join(f"    {ds} & " + " & ".join(v) + r" \\" for ds, v in rows.items())
knowledgefree = r"""
\subsubsection*{Knowledge-free enrichment at the same operating point}
\label{sec:knowledgefree}
Whether the query-side gain needs a knowledge source at all is tested with
embedding-space pseudo-relevance feedback (PRF), the classical
knowledge-free expansion \cite{rocchio1971relevance}: the query is fused,
with the same $\beta{=}0.9$, with the centroid of its top-$k$ baseline
retrievals. Table~\ref{tab:knowledgefree} reports it at parity. On natural
images PRF hurts at every $k$ (Flickr30k $-0.68$ to $-1.48$, MS-COCO
$-0.38$ to $-1.16$), and the KG beats it significantly on both
($p<10^{-4}$). On the four RS datasets PRF and the KG are statistically
indistinguishable, and on RSICD they are complementary ($+0.37$, $p=0.010$,
against $+0.18$ for either alone). Two consequences follow. The RS
query-side gain is not specific to knowledge graphs: any safe enrichment of
a weak query helps. And PRF reproduces the repair pattern of
Sec.~\ref{sec:discussion}, helping weak channels and hurting strong ones,
which makes the pattern a property of the channel rather than of the
source. The last column tests the gallery channel's opposite extreme,
replacing each image embedding by its caption embedding ($\alpha{=}0$): it is
catastrophic on natural images and with BLIP-large captions on RSICD
($-3.1$), but with prompt-conditioned BLIP-base captions on RSICD it gives
$+1.21$ ($p=0.004$), larger than any fusion setting on that caption set.
This text-to-text effect, the mechanism behind Beyond Pixels
\cite{xiao2025beyondpixels}, does not survive in-domain distractors
(Sec.~\ref{sec:scale}).

\begin{table}[t]
  \centering
  \caption{Knowledge-free enrichment at parity (t2i $\dR$ vs.\ baseline,
  official splits). PRF: query fused ($\beta{=}0.9$) with the centroid of its
  top-$k$ baseline images. Caption-only: gallery image embeddings replaced
  by BLIP-base caption embeddings ($\alpha{=}0$). $^{*}p<0.05$ vs.\
  baseline.}
  \label{tab:knowledgefree}
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{l rr r r r}
    \toprule
    Dataset & PRF $k{=}3$ & PRF $k{=}10$ & DCKF-Q & DCKF-Q+PRF & caption-only \\
    \midrule
""" + tab_rows + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
rep(r"""\subsection{Knowledge Graph vs.\ LLM Query Rewriting}""", knowledgefree + r"""
\subsection{Knowledge Graph vs.\ LLM Query Rewriting}""")

# ---------------- IV-E + Table VII
rep(r"""retrieval: $-7.97$
\ratk{1} on Flickr30k and $-0.68$ on RSICD, despite fluent, plausible
rewrites. Passed through our fusion strategy (the rewrite is drift-gated and gently
fused), the same LLM becomes positive ($+0.60$, $p=0.0012$) and
statistically indistinguishable from the ConceptNet channel on both
datasets ($\Delta$ vs.\ KG: $-0.14$ and $-0.04$, both n.s.).""",
r"""retrieval: $-8.15$
\ratk{1} on Flickr30k and $-0.60$ on RSICD, despite fluent, plausible
rewrites. Passed through our fusion strategy (the rewrite is drift-gated and gently
fused), the same LLM becomes positive ($+0.52$, $p=0.004$) and
statistically indistinguishable from the ConceptNet channel on both
datasets ($\Delta$ vs.\ KG: $-0.14$ and $+0.02$, both n.s.).""")
rep(r"""    A baseline                    & 58.70 & 5.45 \\
    LLM rewrite, replaces query   & 50.74 ($-7.97$) & 4.78 ($-0.68$) \\
    LLM rewrite, our fusion       & \textbf{59.30} ($+0.60^{*}$) & \textbf{5.60} ($+0.15$) \\
    DCKF (KG expansion)           & 59.16 ($+0.46^{*}$) & 5.56 ($+0.11$) \\
    \bottomrule
    \multicolumn{3}{l}{\footnotesize $^{*}p<0.005$. KG vs.\ LLM-fused:
    n.s.\ on both datasets.} \\""",
r"""    A baseline                    & 58.86 & 5.40 \\
    LLM rewrite, replaces query   & 50.72 ($-8.15$) & 4.79 ($-0.60$) \\
    LLM rewrite, our fusion       & \textbf{59.38} ($+0.52^{*}$) & 5.56 ($+0.16$) \\
    DCKF (KG expansion)           & 59.24 ($+0.38^{*}$) & \textbf{5.58} ($+0.18^{*}$) \\
    \bottomrule
    \multicolumn{3}{l}{\footnotesize $^{*}p<0.05$. KG vs.\ LLM-fused:
    n.s.\ on both datasets.} \\""")

# ---------------- IV-F1..F4
rep(r"""three concepts as text costs $-4.84$ \ratk{1}. Per-relation ablations show
the damage is concentrated: expanding only through IsA costs $-1.04$, while""",
    r"""three concepts as text costs $-4.80$ \ratk{1}. Per-relation ablations show
the damage is concentrated: expanding only through IsA costs $-1.00$, while""")
rep(r"""channel positive ($+0.40$, CI $[+0.12,+0.70]$, $p=0.0031$; $+0.46$ on the
official split).""", r"""channel positive ($+0.26$, CI $[+0.00,+0.54]$, $p=0.027$; $+0.38$ on the
official split).""")
rep(r"""fusion weight (best $-0.18$ \ratk{1}); score-level fusion fails equally.""",
    r"""fusion weight (best $-0.16$ \ratk{1}); score-level fusion fails equally.""")
rep(r"""5{,}465 RSICD queries (and $\le$61 of 5{,}000 on Flickr30k), leaving
rankings bit-identical to one-hop. Removing the path penalty entirely
floods 2{,}831--3{,}821 queries with two-hop concepts and \emph{degrades}
both datasets ($-0.07$ to $-0.10$).""",
r"""5{,}465 RSICD queries (and 62 of 5{,}000 on Flickr30k), leaving
rankings bit-identical to one-hop on RSICD and within $0.02$ \ratk{1} on
Flickr30k. Removing the path penalty entirely floods 2{,}826--3{,}823
queries with two-hop concepts and \emph{degrades} both datasets ($-0.05$
to $-0.10$).""")
rep(r"""caption-channel gain essentially undiminished (Flickr $+2.46$, COCO
$+2.68$, both $p<10^{-4}$) and makes the Flickr query-side gain \emph{more}
significant ($+0.30$, $p=0.011$).""",
r"""caption-channel gain essentially undiminished (Flickr $+2.44$, COCO
$+2.74$, both $p<10^{-4}$) and keeps the Flickr query-side gain
significant ($+0.28$, $p=0.019$).""")
rep(r"""$5.51\!\rightarrow\!10.50$) and makes the pooled query-side KG effect
vanish ($+0.077\!\rightarrow\!-0.004$, n.s.)""",
    r"""$5.40\!\rightarrow\!10.50$) and makes the pooled query-side KG effect
vanish ($+0.094\!\rightarrow\!-0.004$, n.s.)""")

# ---------------- IV-F5 scale: rewrite paragraph + Table VIII + new Table IX
rep(r"""We stream-encode 100k aerial distractors from Million-AID
\cite{long2021millionaid} (no images stored) and reuse 1M web distractors
from DataComp \cite{gadre2023datacomp}. Two results
(Table~\ref{tab:scale}): first, in-domain density dominates difficulty:
10k aerial distractors halve RSICD \ratk{1} ($5.45\!\rightarrow\!2.63$)
while 10k web distractors cost only $0.75$ points. Second, the pooled
query-side gain does not survive in-domain scale: it fades to noise once
${\ge}10$k aerial distractors are present. Gains of this magnitude vanish
as inter-item margins shrink; benchmark-scale claims should be read
accordingly, and we release the distractor set so that future work can be
evaluated at scale by default.""",
r"""We stream-encode 100k aerial distractors from Million-AID
\cite{long2021millionaid}, captioning each with the same BLIP-large aerial
prompt so that the gallery channel can be applied uniformly to real and
distractor items, and reuse 1M web distractors from DataComp
\cite{gadre2023datacomp}. For natural images, the remaining 30{,}783
Flickr30k images (never queried) serve as in-domain distractors, each
captioned with BLIP-base. Three results follow. First, in-domain density
dominates difficulty (Table~\ref{tab:scale}): 10k aerial distractors halve
RSICD \ratk{1} ($5.40\!\rightarrow\!2.63$) while 10k web distractors cost
$0.73$ points. Second, the query-side gain does not survive in-domain
scale: on RSICD it fades to noise once ${\ge}10$k aerial distractors are
present, and on Flickr30k it stays significant to an 11k gallery
($+0.32$, $p=0.007$) and fades at 31{,}783 ($+0.16$, n.s.). Third, the
two gallery channels part ways (Table~\ref{tab:scale-gallery}): the
natural-image caption channel holds at $+2.2$ to $+2.6$ \ratk{1} across a
32-fold larger in-domain gallery (all $p<10^{-4}$), whereas the repaired
RSICD channel, $+0.48$ at benchmark scale, is at or below zero from 10k
distractors on. The repair account predicts this asymmetry: captions are a
strong channel on natural images and a weak one on aerial scenes, and only
a strong channel has margin to spare when inter-item distances shrink.
Benchmark-scale claims should be read accordingly, and we release the
captioned distractor set so that future work can be evaluated at scale by
default.""")
rep(r"""    ---        & 0    & 5.45 & $+0.11$ (.18) \\
    in-domain  & 10k  & 2.63 & $-0.04$ (.77) \\
    in-domain  & 100k & 1.06 & $-0.07$ (.91) \\
    web        & 10k  & 4.70 & $+0.13$ (.12) \\
    web        & 100k & 3.28 & $+0.05$ (.27) \\""",
r"""    ---        & 0    & 5.40 & $+0.18$ (.048) \\
    in-domain  & 10k  & 2.63 & $-0.05$ (.88) \\
    in-domain  & 100k & 1.04 & $-0.05$ (.86) \\
    web        & 10k  & 4.67 & $+0.16$ (.047) \\
    web        & 100k & 3.24 & $+0.09$ (.12) \\""")
scale_gallery = r"""
\begin{table}[t]
  \centering
  \caption{The gallery channel under in-domain gallery growth (t2i \ratk{1};
  $\dR$ vs.\ the baseline at the same gallery size, paired bootstrap).
  Every gallery item, real or distractor, is fused with its own caption
  ($\alpha{=}0.9$). Flickr30k: 1k test images plus the other Flickr30k
  images, BLIP-base captions. RSICD: 1{,}093 test images plus Million-AID
  distractors, BLIP-large aerial captions. $^{\ddag}p<10^{-4}$,
  $^{*}p<0.05$.}
  \label{tab:scale-gallery}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{l r r r r}
    \toprule
    Gallery & Base & +captions & +both & query-KG only \\
    \midrule
    Flickr30k 1{,}000  & 58.86 & $+2.26^{\ddag}$ & $+2.24^{\ddag}$ & $+0.38^{*}$ \\
    Flickr30k 6{,}000  & 33.86 & $+2.52^{\ddag}$ & $+2.48^{\ddag}$ & $+0.42^{*}$ \\
    Flickr30k 11{,}000 & 27.22 & $+2.56^{\ddag}$ & $+2.54^{\ddag}$ & $+0.32^{*}$ \\
    Flickr30k 31{,}783 & 19.34 & $+2.38^{\ddag}$ & $+2.24^{\ddag}$ & $+0.16$ \\
    \midrule
    RSICD 1{,}093      & 5.40 & $+0.48^{*}$ & $+0.66^{*}$ & $+0.18^{*}$ \\
    RSICD +10k         & 2.63 & $-0.17$ & $-0.07$ & $-0.05$ \\
    RSICD +50k         & 1.41 & $-0.11$ & $-0.05$ & $-0.02$ \\
    RSICD +100k        & 1.04 & $-0.04$ & $-0.04$ & $-0.06$ \\
    \bottomrule
  \end{tabular}
\end{table}
"""
rep(r"""\subsubsection{Drift-threshold sensitivity}
\label{sec:abl-tau}""", scale_gallery + r"""
\subsubsection{Hyperparameter sensitivity}
\label{sec:abl-tau}""")

# ---------------- IV-F6 tau text + hyperparameter table
def sens(fn):
    d = json.load(open(R/fn)); g = {r_["label"]: r_ for r_ in d["rows"]}
    def v(prefix):
        pp = prefix.strip(); k = next(k for k in g if k == pp or k.startswith(pp + " ")); r_ = g[k]
        return f"${r_['dR1_vs_baseline']*100:+.2f}$" + ("$^{*}$" if r_["p_le0"] < 0.05 else "")
    return v
vF, vC, vR = sens("reviewer_flickr30k_k1k_1000_5.json"), sens("reviewer_coco5k_5000_5.json"), sens("reviewer_rsicd_1093_5_large_aerial.json")
def row(label, prefixes):
    return f"    {label} & " + " & ".join(f(p) for f, p in zip((vF, vC, vR), prefixes)) + r" \\"
hyper = "\n".join([
 row(r"$\alpha{=}0.7$",  ["gallery alpha=0.7 ", "gallery alpha=0.7 ", "gallery alpha=0.7 "]),
 row(r"$\alpha{=}0.8$",  ["gallery alpha=0.8 ", "gallery alpha=0.8 ", "gallery alpha=0.8 "]),
 row(r"$\alpha{=}0.9$ (used)", ["gallery alpha=0.9 ", "gallery alpha=0.9 ", "gallery alpha=0.9 "]),
 row(r"$\alpha{=}0.95$", ["gallery alpha=0.95", "gallery alpha=0.95", "gallery alpha=0.95"]),
 r"    \midrule",
 row(r"$\beta{=}0.8$",  ["DCKF-Q beta=0.8 k=2", "DCKF-Q beta=0.8 k=2", "DCKF-Q beta=0.8 k=2"]),
 row(r"$\beta{=}0.9$ (used)", ["DCKF-Q beta=0.9 k=2", "DCKF-Q beta=0.9 k=2", "DCKF-Q beta=0.9 k=2"]),
 row(r"$\beta{=}0.95$", ["DCKF-Q beta=0.95 k=2", "DCKF-Q beta=0.95 k=2", "DCKF-Q beta=0.95 k=2"]),
 r"    \midrule",
 row(r"$k{=}1$", ["DCKF-Q beta=0.9 k=1", "DCKF-Q beta=0.9 k=1", "DCKF-Q beta=0.9 k=1"]),
 row(r"$k{=}2$ (used)", ["DCKF-Q beta=0.9 k=2", "DCKF-Q beta=0.9 k=2", "DCKF-Q beta=0.9 k=2"]),
 row(r"$k{=}3$", ["DCKF-Q beta=0.9 k=3", "DCKF-Q beta=0.9 k=3", "DCKF-Q beta=0.9 k=3"]),
])
rep(r"""The gate threshold $\tau{=}0.55$ was fixed once and never tuned per
dataset; a post-hoc sweep confirms it sits in a stable region rather than
at a fragile optimum. Re-thresholding the cached expansion candidates over
$\tau\in[0.50,0.75]$ and re-running retrieval leaves the query-side gain on
the Flickr30k ablation subset essentially unchanged across
$\tau\in[0.50,0.65]$ ($+0.24$ to $+0.30$ \ratk{1}, each individually
significant or borderline), with RSICD flat over the same range ($+0.15$
to $+0.16$); at $\tau\ge0.70$ the gate discards too many candidates and
the effect weakens toward noise on both datasets. Values below $0.50$ are
not evaluable post hoc, as the candidate cache itself applies a $0.5$
similarity floor. Fig.~\ref{fig:tau} plots the sweep.""",
r"""All four hyperparameters were fixed once and never tuned per dataset;
post-hoc sweeps show each sits in a stable region rather than at a fragile
optimum. Re-thresholding the cached expansion candidates over
$\tau\in[0.50,0.75]$ and re-running retrieval leaves the query-side gain on
the Flickr30k ablation subset essentially unchanged across
$\tau\in[0.50,0.65]$ ($+0.26$ to $+0.34$ \ratk{1}, each individually
significant), with RSICD flat over the same range ($+0.18$ to $+0.20$);
at $\tau\ge0.70$ the gate discards too many candidates and the effect
weakens toward noise on both datasets. Values below $0.50$ are not
evaluable post hoc, as the candidate cache itself applies a $0.5$
similarity floor. Fig.~\ref{fig:tau} plots the sweep.
Table~\ref{tab:hyper} sweeps the fusion weights and the concept budget:
the gallery weight is the only sensitive one (heavier mixing than
$\alpha{=}0.8$ destroys discrimination, Sec.~\ref{sec:gentle}), and its
optimum is $0.9$ on Flickr30k and RSICD but $0.8$ on MS-COCO ($+3.24$
vs.\ $+2.81$), a gain we leave on the table rather than tune per dataset;
$\beta$ and $k$ move the query-side gain by at most $0.15$ points.

\begin{table}[t]
  \centering
  \caption{Fusion-weight and concept-budget sensitivity (t2i $\dR$ vs.\
  baseline). RSICD uses the BLIP-large aerial captions for $\alpha$; the
  query-side rows are caption-independent. $^{*}p<0.05$.}
  \label{tab:hyper}
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{l rrr}
    \toprule
    Setting & Flickr30k & MS-COCO & RSICD \\
    \midrule
""" + hyper + r"""
    \bottomrule
  \end{tabular}
\end{table}""")
rep(r"""\subsection{Why Gentle Fusion}""", r"""\subsection{Why Gentle Fusion}
\label{sec:gentle}""")

# ---------------- Discussion
rep(r"""of rich captions is destructive ($-4.8$); relation-weighted expansion of
short, domain-mismatched RS queries is significantly positive (pooled
$p=0.027$); external captions transform natural-image galleries ($+2.4$ to
$+2.9$) precisely because a bare image embedding is itself a deficient
``text channel''; and the KG increment on top of captions appears exactly
where the captions are weak: absent on Flickr30k and COCO, significant
on repaired RSICD ($+0.20$, $p=0.016$).""",
r"""of rich captions is destructive ($-4.8$); relation-weighted expansion of
short, domain-mismatched RS queries is significantly positive (pooled
$p=0.008$), and so is knowledge-free feedback on the same queries;
external captions transform natural-image galleries ($+2.2$ to $+2.9$)
precisely because a bare image embedding is itself a deficient ``text
channel''; and the KG increment on top of captions appears exactly where
the captions are weak: absent on Flickr30k and COCO, significant on
repaired RSICD ($+0.18$, $p=0.027$).""")
rep(r"""The account has now made four correct predictions in advance of the
experiments that tested them:""", r"""The account has made four correct predictions, each recorded in the
released experiment log before the run that tested it:""")
rep(r"""receives the KG (MS-COCO, $+0.44$, $p=0.013$).""", r"""receives the KG (MS-COCO, $+0.42$, $p=0.018$). A fifth prediction was
tested last: under in-domain gallery growth the strong natural-image caption
channel should keep its margin and the weak RSICD channel should lose it,
which is what Table~\ref{tab:scale-gallery} shows.""")
rep(r"""noise (with mR still improving). Both channels fade under realistic
in-domain scale (Sec.~\ref{sec:scale}); UCM ($n{=}210$) and RSITMD""",
    r"""noise (with mR still improving). Query-side effects and the RS gallery
gain fade under realistic in-domain scale, and the query-side RS gain is
matched by knowledge-free feedback (Sec.~\ref{sec:knowledgefree}); only
the natural-image gallery gain holds at scale (Sec.~\ref{sec:scale}).
UCM ($n{=}210$) and RSITMD""")

# ---------------- Conclusion
rep(r"""$+2.4$ to $+2.9$ \ratk{1} on the official natural-image splits at zero
query latency, and a captioner upgrade produces the first significant
training-free knowledge gain on a remote-sensing benchmark, with a
knowledge-graph increment the mechanism predicted in advance.""",
r"""$+2.2$ to $+2.9$ \ratk{1} on the official natural-image splits at zero
query latency and holds when the Flickr30k gallery grows to 31{,}783
images, and a captioner upgrade produces the first significant
training-free knowledge gain on a remote-sensing benchmark, with a
knowledge-graph increment the mechanism predicted in advance, although
that gain does not survive in-domain gallery growth.""")

if FAIL:
    for f in FAIL: print("MISMATCH", f)
    sys.exit("not written")
P.write_text(s); print("main.tex patched")

# ---------------- bib
B = Path("/Users/home/RA-KG-T2I/paper/references.bib"); b = B.read_text()
if "lin2026dtir" not in b:
    b += r"""
@article{lin2026dtir,
  author={Lin, Zexu and Zhang, Dell and Shen, Yiqing and Li, Xuelong},
  title={Reasoning Text-to-Image Retrieval with Large Language Models and Digital Twin Representations},
  journal={Knowledge-Based Systems},
  volume={336},
  pages={115313},
  year={2026},
  doi={10.1016/j.knosys.2026.115313}
}
"""
    B.write_text(b); print("bib: lin2026dtir added")
# abstract word count
a = s[s.index("\\begin{abstract}"):s.index("\\end{abstract}")]
print("abstract words:", len(re.sub(r"\\[a-zA-Z]+|[{}$]","",a).split()))
