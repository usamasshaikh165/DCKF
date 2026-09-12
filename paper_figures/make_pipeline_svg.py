"""Redrawn DCKF pipeline figure (Fig. 2 in the revised numbering).
viewBox 1000x600 -> printed at 7.0 in wide: 1 px = 0.504 pt, so 14 px = 7.1 pt, 15 px = 7.6 pt, 18 px = 9.1 pt.
No internal experiment IDs, no result numbers, no em-dashes."""
from pathlib import Path
import cairosvg

W, H = 1000, 624
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
SERIF = "DejaVu Serif, Georgia, serif"
o = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">')
a('''<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#3a3a3a"/></marker></defs>
<style>
 .ttl{font-size:18px;font-weight:bold;fill:#1c1c1c}
 .lbl{font-size:15px;fill:#222}
 .sm{font-size:14px;fill:#555}
 .xs{font-size:13px;fill:#444}
 .mono{font-family:%s;font-size:13px;fill:#222}
 .frm{font-family:%s;font-size:14px;fill:#333}
 .emb{font-family:%s;font-style:italic;font-size:15px;fill:#222}
 .ln{stroke:#3a3a3a;stroke-width:1.4;fill:none;marker-end:url(#arr)}
 .edge{stroke:#888;stroke-width:1.1;fill:none}
 .halo{fill:#fff}
</style>''' % (MONO, MONO, SERIF))
a(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff"/>')

def text(x, y, s, cls="lbl", anchor="middle", extra=""):
    a(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}" {extra}>{s}</text>')
def trap(x, y, w, h, fill, label, fw_dir="right"):
    """encoder trapezoid: wide at input side"""
    if fw_dir == "right":
        pts = f"{x},{y} {x+w},{y+h*0.2} {x+w},{y+h*0.8} {x},{y+h}"
    else:  # pointing down (input on top)
        pts = f"{x},{y} {x+w},{y} {x+w*0.85},{y+h} {x+w*0.15},{y+h}"
    a(f'<polygon points="{pts}" fill="{fill}" stroke="#3a3a3a" stroke-width="1"/>')
    cx, cy = x + w/2, y + h/2
    l1, l2 = label
    text(cx, cy-3, l1); text(cx, cy+15, l2)
def box(x, y, w, h, fill, stroke, rx=0):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
def emb(x, y, s, sub):
    a(f'<text class="emb" x="{x}" y="{y}" text-anchor="middle">{s}<tspan font-size="10" dy="3">{sub}</tspan></text>')
def column(x, y0, fill, stroke, sym, w=50, h=24, gap=30):
    ys = [y0, y0+gap, y0+2*gap]
    for yy, sub in zip(ys, "123"):
        box(x, yy, w, h, fill, stroke); emb(x+w/2, yy+17, sym, sub)
    text(x+w/2, y0+3*gap+6, "&#8942;", "lbl")
    box(x, y0+3*gap+14, w, h, fill, stroke); emb(x+w/2, y0+3*gap+31, sym, "N")
    return y0+3*gap+14+h   # bottom
def T_tilde(x, y, cls="emb", size=15):
    a(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="middle" font-size="{size}">T</text>')
    a(f'<text x="{x}" y="{y-6}" text-anchor="middle" font-family="{SERIF}" font-size="{size}">&#732;</text>')
def img_icon(x, y, w=70, h=52):
    box(x, y, w, h, "#e8efdc", "#555"); box(x+6, y+h-18, w-12, 8, "#b9c4ad", "none")
    box(x+8, y+8, 14, 11, "#8fae86", "none"); box(x+28, y+10, 11, 9, "#8fae86", "none")
def arrow(d): a(f'<path class="ln" d="{d}"/>')
SNOW = ""

# ============ (1) OFFLINE, top band ============
text(20, 30, "(1) Offline: gallery-side enrichment (index time)", "ttl", "start")
# gallery stack
box(44, 96, 70, 52, "#eef2e6", "#999"); box(37, 103, 70, 52, "#eef2e6", "#999"); img_icon(30, 110)
text(108, 186, "gallery images", "sm", "end")
# image encoder
text(240, 52, "frozen CLIP", "sm")
arrow("M100,120 C140,120 140,92 176,92")
trap(180, 62, 120, 60, "#BCDCB2", ("Image", "Encoder"))
arrow("M300,92 L636,92")
# captioner branch
arrow("M100,140 C132,140 128,222 168,222")
box(172, 194, 124, 56, "#DFE7DF", "#3a3a3a", rx=8); text(234, 217, "BLIP captioner"); text(234, 238, "frozen", "sm")
arrow("M296,222 L316,222")
box(320, 194, 158, 56, "#fff", "#444", rx=3); text(399, 217, "an aerial photograph", "mono", extra='font-size="11.5"'); text(399, 237, "of a stadium .", "mono", extra='font-size="11.5"')
text(399, 272, "synthetic caption, computed once", "sm")
arrow("M478,222 L500,222")
text(557, 184, "frozen CLIP", "sm")
trap(504, 192, 106, 60, "#CBB7DC", ("Text", "Encoder"))
arrow("M610,222 L636,222")
# I column and C column
column(640, 44, "#DFEEDC", "#6f9a6b", "I")
column(640, 176, "#EDE6F5", "#8a6fae", "C")
# fusion
arrow("M690,116 C720,116 745,140 745,150")
arrow("M690,224 C720,224 745,196 745,186")
a('<circle cx="745" cy="168" r="14" fill="#fff" stroke="#3a3a3a" stroke-width="1.4"/>')
text(745, 175, "&#8853;", "lbl", extra='font-size="19"')
arrow("M759,168 L786,168")
bottom = column(790, 96, "#D9E8F7", "#6b8cba", "&#296;")
text(815, bottom+24, "enriched index", "sm"); text(815, bottom+42, "(precomputed)", "sm")
text(980, 332, "&#296;&#8342; = &#945;&#183;I&#8342; + (1&#8722;&#945;)&#183;C&#8342;,  &#945; = 0.9   (Eq. 2)", "frm", "end")

# separator
a('<line x1="20" y1="346" x2="980" y2="346" stroke="#d0d4d8" stroke-width="1"/>')

# ============ (2) ONLINE, bottom-left ============
text(20, 366, "(2) Online: relation-aware query expansion", "ttl", "start")
box(20, 392, 152, 52, "#fff", "#444", rx=3); text(96, 414, "two ships docked", "mono", extra='font-size="12.5"'); text(96, 434, "near the port .", "mono", extra='font-size="12.5"')
text(96, 466, "user query", "sm")
arrow("M172, 418 L198, 418")
# ConceptNet box with mini graph
box(202, 380, 190, 158, "#fff", "#444", rx=8)
text(297, 400, "ConceptNet", "lbl", extra='font-weight="bold"')
# nodes
def node(cx, cy, rx, label, fill, stroke, op=1.0):
    a(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="13" fill="{fill}" stroke="{stroke}" opacity="{op}"/>')
    a(f'<text class="xs" x="{cx}" y="{cy+4}" text-anchor="middle" fill="#222" opacity="{op}">{label}</text>')
a('<line class="edge" x1="268" y1="452" x2="326" y2="440"/>')   # ship -> harbor
a('<line class="edge" x1="268" y1="466" x2="334" y2="500"/>')   # ship -> dock
a('<line class="edge" x1="246" y1="472" x2="246" y2="498" opacity="0.6"/>')   # ship -> vessel (demoted)
node(246, 458, 27, "ship", "#EDE6F5", "#8a6fae")
node(352, 434, 30, "harbor", "#DFEEDC", "#6f9a6b")
node(360, 506, 26, "dock", "#DFEEDC", "#6f9a6b")
node(246, 512, 27, "vessel", "#f0f0f0", "#aaa", 0.6)
def edge_label(x, y, s, w, op=1.0):
    a(f'<rect class="halo" x="{x-w/2}" y="{y-11}" width="{w}" height="14" opacity="0.9"/>')
    a(f'<text class="xs" x="{x}" y="{y}" text-anchor="middle" opacity="{op}">{s}</text>')
edge_label(276, 424, "AtLocation 1.0", 92)
edge_label(290, 494, "UsedFor 1.0", 82)
edge_label(220, 490, "IsA 0.4", 48, 0.7)
text(297, 558, "score = w(r)&#183;cos(T, c); keep if cos &#8805; &#964;", "xs")
arrow("M392,418 L418,418")
# expanded text
box(422, 380, 178, 76, "#fff", "#444", rx=3)
text(509, 402, "two ships docked near", "mono", extra='font-size="12.3"'); text(509, 422, "the port , a scene with", "mono", extra='font-size="12.3"')
a(f'<text class="mono" x="509" y="442" text-anchor="middle" fill="#6b4d8f" font-size="12.3">harbor , dock</text>')
arrow("M509,456 L509,474")
text(521, 548, "frozen CLIP", "xs", "start")
trap(452, 478, 114, 56, "#CBB7DC", ("Text", "Encoder"), fw_dir="down")
arrow("M509,534 L509,552")
box(484, 556, 50, 30, "#EDE6F5", "#8a6fae"); T_tilde(509, 580)
T_tilde(186, 592, "frm", 14); text(196, 592, "= &#946;&#183;T + (1&#8722;&#946;)&#183;E,  &#946; = 0.9  (Eq. 1)", "frm", "start")

# ============ (3) RETRIEVAL, bottom-right ============
text(622, 366, "(3) Retrieval: no added latency", "ttl", "start")
xs = [712, 772, 832, 892, 952]
text(686, 388, "from offline index", "xs", "start")
for i, (x, sub) in enumerate(zip(xs, ["1", "2", "3", None, "N"])):
    if sub is None:
        box(x-26, 396, 52, 26, "#fff", "#bbb"); text(x, 415, "&#8230;", "lbl")
    else:
        box(x-26, 396, 52, 26, "#D9E8F7", "#6b8cba"); emb(x, 414, "&#296;", sub)
        arrow(f"M{x},422 L{x},448")
# query arrives from panel 2
arrow("M534,570 L618,570 L618,465 L630,465")
box(634, 450, 44, 30, "#EDE6F5", "#8a6fae"); T_tilde(656, 474)
arrow("M678,465 L684,465")
a('<g font-family="%s" font-style="italic" font-size="14">' % SERIF)
for x, sub in zip(xs, ["1", "2", "3", None, "N"]):
    if sub is None:
        box(x-26, 450, 52, 30, "#fff", "#bbb"); a(f'<text x="{x}" y="473" text-anchor="middle" font-style="normal" font-family="{FONT}">&#8230;</text>')
    else:
        hl = sub == "3"
        box(x-26, 450, 52, 30, "#A8CBEA" if hl else "#D9E8F7", "#4a6f9d" if hl else "#6b8cba")
        a(f'<text x="{x}" y="473" text-anchor="middle">T&#183;&#296;<tspan font-size="10" dy="3">{sub}</tspan></text>')
        a(f'<text x="{x-12}" y="467" text-anchor="middle" font-size="14">&#732;</text>')
a('</g>')
arrow("M832,480 L832,500")
img_icon(800, 504, 64, 46); text(832, 572, "rank-1 image", "sm")
text(832, 600, "FAISS IVF + fp16 index: under 2 ms at 2M vectors", "xs")
text(20, 614, "frozen = weights exactly as released; no parameters are trained anywhere in the pipeline", "sm", "start")
a('</svg>')
svg = "\n".join(o)
out = Path(__file__).parent
(out/"fig_pipeline.svg").write_text(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(out/"fig_pipeline.pdf"))
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out/"fig_pipeline.png"), output_width=2400)
print("ok")
