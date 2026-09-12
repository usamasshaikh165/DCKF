"""E40 — compare image-cluster bootstrap p-values (results/imgboot/) with the caption-level originals
(results/fp32_rederivation/ or results/e37_e38/). Lists every row whose two-sided significance at 0.05 flips."""
import json, glob, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
two = lambda p: 2*min(p, 1-p)
def rows(d, path=""):
    if isinstance(d, dict):
        lab = d.get("label") or d.get("config") or d.get("name")
        for pk in ("p_le0", "p"):
            if pk in d and isinstance(d[pk], (int, float)):
                dk = next((k for k in ("dR1_vs_baseline", "delta_R@1", "dR1", "delta") if k in d), None)
                yield (str(lab) if lab else path, d[dk] if dk else None, d[pk]); break
        for k, v in d.items(): yield from rows(v, f"{path}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d): yield from rows(v, f"{path}[{i}]")
flips, same, missing = [], 0, []
for f in sorted(glob.glob(f"{ROOT}/imgboot/*.json")):
    b = os.path.basename(f)
    old = next((c for c in (f"{ROOT}/fp32_rederivation/{b}", f"{ROOT}/e37_e38/{b}", f"{ROOT}/capboot_backup/{b}") if os.path.exists(c)), None)
    if not old: missing.append(b); continue
    new_rows = {lab: (d, p) for lab, d, p in rows(json.load(open(f)))}
    for lab, d, p in rows(json.load(open(old))):
        if lab not in new_rows: continue
        d2, p2 = new_rows[lab]
        s_old, s_new = two(p) < 0.05 and (d or 0) > 0, two(p2) < 0.05 and (d2 or 0) > 0
        if s_old != s_new: flips.append((b, lab, d, two(p), two(p2)))
        else: same += 1
print(f"{same} rows unchanged in significance; {len(flips)} flips; {len(missing)} files without a caption-level counterpart: {missing}")
print("\nfile | label | dR1 | p_two caption | p_two image")
for b, lab, d, po, pn in flips:
    print(f"{b} | {lab[:55]} | {('' if d is None else f'{d*100:+.2f}')} | {po:.3f} | {pn:.3f}")
