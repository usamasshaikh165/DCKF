"""
Millions-scale retrieval stress test (efficiency objective, proposal §4.2-3).

Real Recall@K quality is measured on real benchmarks (Flickr30k, COCO 5k).
This script answers a DIFFERENT question: how do latency / memory / index
choice behave as the gallery grows to millions of vectors?

Gallery = real Flickr30k CLIP features + SYNTHETIC distractors
(mixup of random real feature pairs + Gaussian noise, re-normalized, so they
match the real embedding distribution on the unit hypersphere).
Synthetic vectors are used ONLY for scale benchmarking — never for
quality claims.

Indices compared at each scale:
  - FlatIP        exact search (brute force, the baseline)
  - IVF+SQfp16    inverted lists + fp16 scalar quantizer (sub-linear, half RAM)

Outputs results/scale_benchmark.csv and results/scale_benchmark.png
"""
import csv
import gc
import resource
import time
from pathlib import Path

# NOTE: torch and faiss cannot coexist in one process on macOS/arm64 (duplicate
# libomp -> SIGSEGV or OMP Error #15). This script is deliberately torch-free:
# features are read from a .npy exported by baseline_retrieval's cache.
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent
FEATS = ROOT / "results" / "image_features_31783.npy"
OUT_CSV = ROOT / "results" / "scale_benchmark.csv"
OUT_PNG = ROOT / "results" / "scale_benchmark.png"

# 2M cap: at 5M, gallery(10GB fp32) + FlatIP copy(10GB) exceeds 16GB RAM.
# Exact-search latency is linear in N -> extrapolate beyond 2M if needed.
SCALES = [100_000, 500_000, 1_000_000, 2_000_000]
N_QUERIES = 200
TOP_K = 10
RNG = np.random.default_rng(42)


def load_real_features() -> np.ndarray:
    x = np.load(FEATS)          # pre-normalized float32, exported from .pt cache
    print(f"Real features: {x.shape}")
    return x


def make_distractors(real: np.ndarray, n: int, batch: int = 500_000) -> np.ndarray:
    """Mixup two random real vectors + noise, renormalize -> stays in-distribution."""
    out = np.empty((n, real.shape[1]), dtype=np.float32)
    for i in range(0, n, batch):
        m = min(batch, n - i)
        a = real[RNG.integers(0, len(real), m)]
        b = real[RNG.integers(0, len(real), m)]
        lam = RNG.uniform(0.2, 0.8, (m, 1)).astype(np.float32)
        v = lam * a + (1 - lam) * b
        v += RNG.normal(0, 0.02, v.shape).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        out[i:i + m] = v
    return out


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def bench_index(index, queries: np.ndarray, ground_truth: np.ndarray | None):
    lat = []
    hits = 0
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        _, idx = index.search(q[None, :], TOP_K)
        lat.append((time.perf_counter() - t0) * 1000)
        if ground_truth is not None:
            hits += len(set(idx[0]) & set(ground_truth[i]))
    lat = np.array(lat)
    recall_vs_exact = hits / (len(queries) * TOP_K) if ground_truth is not None else 1.0
    return lat.mean(), np.percentile(lat, 95), recall_vs_exact


def main():
    real = load_real_features()
    dim = real.shape[1]
    queries = real[RNG.choice(len(real), N_QUERIES, replace=False)]

    rows = []
    for n_total in SCALES:
        n_syn = n_total - len(real)
        if n_syn < 0:
            continue
        print(f"\n=== scale {n_total:,} ===")
        gallery = np.vstack([real, make_distractors(real, n_syn)])

        # ---- exact ----
        t0 = time.perf_counter()
        flat = faiss.IndexFlatIP(dim)
        flat.add(gallery)
        t_build_flat = time.perf_counter() - t0
        mean_ms, p95_ms, _ = bench_index(flat, queries, None)
        exact_gt = np.vstack([flat.search(q[None, :], TOP_K)[1][0] for q in queries])
        print(f"  FlatIP      build {t_build_flat:6.1f}s  "
              f"mean {mean_ms:7.2f}ms  p95 {p95_ms:7.2f}ms")
        rows.append([n_total, "FlatIP-exact", t_build_flat, mean_ms, p95_ms, 1.0])
        del flat; gc.collect()

        # ---- ANN: IVF + fp16 scalar quantizer ----
        nlist = int(4 * np.sqrt(n_total))
        quant = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFScalarQuantizer(
            quant, dim, nlist, faiss.ScalarQuantizer.QT_fp16,
            faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter()
        train_n = min(n_total, 50 * nlist)
        ivf.train(gallery[RNG.choice(n_total, train_n, replace=False)])
        ivf.add(gallery)
        t_build = time.perf_counter() - t0
        for nprobe in (8, 32):
            ivf.nprobe = nprobe
            mean_ms, p95_ms, rec = bench_index(ivf, queries, exact_gt)
            print(f"  IVF np={nprobe:<3}  build {t_build:6.1f}s  "
                  f"mean {mean_ms:7.2f}ms  p95 {p95_ms:7.2f}ms  "
                  f"recall-vs-exact@{TOP_K} {rec:.3f}")
            rows.append([n_total, f"IVF-SQfp16-np{nprobe}", t_build,
                         mean_ms, p95_ms, rec])
        del ivf, gallery; gc.collect()
        print(f"  peak RSS so far: {peak_rss_gb():.1f} GB")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gallery_size", "index", "build_s",
                    "mean_ms", "p95_ms", "recall_vs_exact@10"])
        w.writerows(rows)
    print(f"\nSaved {OUT_CSV}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by = {}
    for n, name, _, mean_ms, _, _ in rows:
        by.setdefault(name, []).append((n, mean_ms))
    for name, pts in by.items():
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gallery size (vectors)")
    ax.set_ylabel("mean query latency (ms)")
    ax.set_title("CLIP retrieval latency vs gallery scale (M3, ViT-B/32 dim=512)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
