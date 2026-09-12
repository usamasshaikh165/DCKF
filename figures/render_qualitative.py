"""Render KTIR-Fig.7-style qualitative panel: baseline vs relation-aware KG."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data/images"
SCR = Path(__file__).parent; OUTD = ROOT / "paper"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_I = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

GREEN, RED, GRAY = "#2e8b3a", "#c23b3b", "#b9bfc7"
BLUE, DARK = "#1F4E9C", "#26313B"

cands = {c["qidx"]: c for c in json.loads(
    (SCR / "qualitative_candidates.json").read_text())}

EXAMPLES = [
    (3102, 'Query: "The students are practicing martial arts"',
     "expansion kept:  students —AtLocation→ school, university   "
     "(w=1.0, sim .80)"),
    (2159, 'Query: "People enjoying drinks in a crowded bar"',
     "expansion kept:  bar —IsA→ pub   (IsA demoted to w=0.4; survives the gate on sim .81)"
     ),
]

S = 1.25   # KBS artwork rule: full-width halftones >= 2244 px (was 1976 px at S=1)
TH, PAD, BORDER = int(300*S), int(26*S), int(10*S)
f_q = ImageFont.truetype(FONT_B, int(40*S))
f_c = ImageFont.truetype(FONT_I, int(30*S))
f_lab = ImageFont.truetype(FONT_B, int(32*S))
f_cap = ImageFont.truetype(FONT, int(26*S))
f_badge = ImageFont.truetype(FONT_B, int(30*S))

def thumb(name, color, badge=None):
    im = Image.open(IMG_DIR / name).convert("RGB")
    im = ImageOps.fit(im, (TH, TH))
    im = ImageOps.expand(im, border=BORDER, fill=color)
    if badge:   # colour-independent cue for colour-blind readers
        d = ImageDraw.Draw(im)
        bw, bh = int(64*S), int(40*S)
        d.rectangle([BORDER, BORDER, BORDER+bw, BORDER+bh], fill=color)
        d.text((BORDER+int(8*S), BORDER+int(2*S)), badge, font=f_badge, fill="white")
    return im

def row(names, gt, wrong_top1):
    imgs = []
    for i, n in enumerate(names):
        color = GREEN if n == gt else (RED if (i == 0 and wrong_top1) else GRAY)
        badge = "GT" if n == gt else ("X" if (i == 0 and wrong_top1) else None)
        imgs.append(thumb(n, color, badge))
    return imgs

LABW = int(220*S)
W = LABW + 5 * (TH + 2 * BORDER) + 4 * PAD + 2 * PAD
blocks = []
for qidx, qtext, ctext in EXAMPLES:
    c = cands[qidx]
    header_h = int(130*S)
    row_h = TH + 2 * BORDER + int(40*S)
    block = Image.new("RGB", (W, header_h + 2 * row_h + int(30*S)), "white")
    d = ImageDraw.Draw(block)
    d.text((PAD, int(10*S)), qtext, font=f_q, fill=DARK)
    d.text((PAD, int(68*S)), ctext, font=f_c, fill=BLUE)
    for r, (label, names, wrong) in enumerate([
            ("CLIP\nbaseline", c["top5_base"], True),
            ("+ KG\n(ours)", c["top5_ours"], False)]):
        y = header_h + r * row_h
        d.multiline_text((PAD, y + TH // 2 - int(30*S)), label, font=f_lab,
                         fill=DARK if r == 0 else BLUE, spacing=int(8*S))
        for i, im in enumerate(row(names, c["gt_name"], wrong)):
            x = LABW + i * (TH + 2 * BORDER + PAD)
            block.paste(im, (x, y))
            d.text((x + int(4*S), y + TH + 2 * BORDER + int(4*S)), f"#{i+1}",
                   font=f_cap, fill="#7a828c")
    blocks.append(block)

legend_h = int(70*S)
H = sum(b.height for b in blocks) + legend_h + int(20*S)
canvas = Image.new("RGB", (W, H), "white")
y = 0
for b in blocks:
    canvas.paste(b, (0, y))
    y += b.height
d = ImageDraw.Draw(canvas)
d.text((PAD, y + int(14*S)),
       "green frame + GT = ground-truth image   red frame + X = wrong top-1   "
       "(Flickr30k development subset, frozen CLIP ViT-B/32, gentle fusion β=0.9)",
       font=f_cap, fill="#5a6470")
canvas.save(OUTD / "qualitative_panel_v2.png"); canvas.convert("RGB").save(OUTD / "qualitative_panel_v2.jpg", quality=92)
print("saved", canvas.size)
