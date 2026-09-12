"""Plain-language deployment flow diagram: Step 1 offline / Step 2 online."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SCR = Path(__file__).parent
AR = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARI = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
UNI = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

BLUE = "#1F4E9C"; SOFT = "#E8EEF7"; DARK = "#26313B"; GRAY = "#5A6470"
GREEN = "#2E7D4F"; GSOFT = "#E6F1EA"; LINE = "#8FA6C8"

W, H = 2600, 1140
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

f_band = ImageFont.truetype(ARB, 44)
f_head = ImageFont.truetype(ARB, 36)
f_sub = ImageFont.truetype(AR, 29)
f_note = ImageFont.truetype(ARI, 33)
f_snow = ImageFont.truetype(UNI, 36)

def box(x, y, w, h, title, subs, fill=SOFT, border=BLUE, frozen=False):
    d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=fill,
                        outline=border, width=4)
    ty = y + 22
    t = title
    if frozen:
        d.text((x + 24, ty), "❄", font=f_snow, fill=BLUE)
        d.text((x + 70, ty), t, font=f_head, fill=DARK)
    else:
        tw = d.textlength(t, font=f_head)
        d.text((x + (w - tw) / 2, ty), t, font=f_head, fill=DARK)
    ty += 52
    for s in subs:
        tw = d.textlength(s, font=f_sub)
        d.text((x + (w - tw) / 2, ty), s, font=f_sub, fill=GRAY)
        ty += 40

def arrow(x1, y, x2):
    d.line([x1, y, x2 - 26, y], fill=LINE, width=6)
    d.polygon([(x2, y), (x2 - 30, y - 15), (x2 - 30, y + 15)], fill=LINE)

def band(y, text, color, softc):
    d.rounded_rectangle([40, y, W - 40, y + 66], radius=14, fill=softc)
    d.text((70, y + 10), text, font=f_band, fill=color)

# ---------------- STEP 1 ----------------
band(30, "STEP 1  —  SETUP: runs ONCE, offline  (the user brings only a folder of images — no labels)", BLUE, SOFT)
y1, bh = 140, 190
bw, gap = 560, 70
xs = [60, 60 + (bw + gap), 60 + 2 * (bw + gap), 60 + 3 * (bw + gap)]
box(xs[0], y1, bw, bh, "Your image collection",
    ["photo archive, e-commerce,", "satellite images — anything"])
box(xs[1], y1, bw, bh, "CLIP + BLIP read each image",
    ["both frozen - used exactly as downloaded", "CLIP: image -> vector (its look)", "BLIP: caption -> vector (its words)"], frozen=True)
box(xs[2], y1, bw, bh, "Blend 90 / 10",
    ["one enriched vector per image", "look + words, both searchable"])
box(xs[3], y1, bw, bh, "FAISS index",
    ["fast search database", "new images added anytime,", "23 ms each — no rebuild"])
for a, b in zip(xs, xs[1:]):
    arrow(a + bw + 6, y1 + bh // 2, b - 6)

# ---------------- STEP 2 ----------------
band(400, "STEP 2  —  EVERY SEARCH: online, adds ~200 ms  (what a person actually experiences)", GREEN, GSOFT)
y2 = 510
bw2, gap2 = 480, 56
xs2 = [60 + i * (bw2 + gap2) for i in range(4)]
box(xs2[0], y2, bw2, bh, "User types a sentence",
    ["“two ships docked", "near the port”"], fill=GSOFT, border=GREEN)
box(xs2[1], y2, bw2, bh, "Find key words",
    ["ship, port", "(no LLM, no cloud —", "local ConceptNet, 2.2M facts)"], fill=GSOFT, border=GREEN)
box(xs2[2], y2, bw2, bh, "Score the suggestions",
    ["harbor - KEPT (AtLocation, w=1.0)", "vehicle - dropped (IsA, w=0.4)", "similarity gate ≥ 0.55, keep top-2"], fill=GSOFT, border=GREEN)
box(xs2[3], y2, bw2, bh, "Nudge the query: 90 / 10",
    ["original stays dominant;", "knowledge only tips", "the close calls"], fill=GSOFT, border=GREEN)
for a, b in zip(xs2, xs2[1:]):
    arrow(a + bw2 + 6, y2 + bh // 2, b - 6)

# second row of step 2
y3 = y2 + bh + 90
box(xs2[0], y3, bw2, bh, "Search the index",
    ["nudged query vector vs", "enriched image vectors", "< 2 ms even at 2M images"], fill=GSOFT, border=GREEN)
box(xs2[1], y3, bw2, bh, "Ranked images back",
    ["right image first more often:", "+2.4 to +3.1 R@1 (p < 0.0001)"], fill=GSOFT, border=GREEN)
# connector from last box row1 down to row0 of row2
cx = xs2[3] + bw2 // 2
d.line([cx, y2 + bh + 6, cx, y2 + bh + 45], fill=LINE, width=6)
d.line([cx, y2 + bh + 45, xs2[0] + bw2 // 2, y2 + bh + 45], fill=LINE, width=6)
d.line([xs2[0] + bw2 // 2, y2 + bh + 45, xs2[0] + bw2 // 2, y3 - 26], fill=LINE, width=6)
d.polygon([(xs2[0] + bw2 // 2, y3 - 2), (xs2[0] + bw2 // 2 - 15, y3 - 30),
           (xs2[0] + bw2 // 2 + 15, y3 - 30)], fill=LINE)
arrow(xs2[0] + bw2 + 6, y3 + bh // 2, xs2[1] - 6)

# takeaway box on the right of row 2
tx = xs2[2]
d.rounded_rectangle([tx, y3, W - 60, y3 + bh], radius=18, fill="white",
                    outline=BLUE, width=5)
d.text((tx + 30, y3 + 18), "Why this matters", font=f_head, fill=BLUE)
for i, line in enumerate([
        "Nothing is ever trained — both models used exactly as downloaded.",
        "No labels, no GPUs at query time, works on API/black-box models.",
        "Every improvement is explainable: a named ConceptNet edge."]):
    d.text((tx + 30, y3 + 72 + i * 40), line, font=f_sub, fill=DARK)

d.text((60, H - 90),
       "The person searching never sees any of this — they type a sentence and better images come back first.",
       font=f_note, fill=GRAY)

img.save(SCR / "deployment_flow.png")
print("saved", img.size)
