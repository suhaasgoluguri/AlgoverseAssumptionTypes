"""Setup and round structure: shared context, private findings, four rounds.

Renders at exactly 5.5 in wide so it needs no scaling at \\linewidth in the
NeurIPS single-column layout. Regenerate with:

    python docs/figures/make_fig_setup_rounds.py
"""

import os

# Matplotlib writes a local CreationDate into the PDF, whose UTC offset
# discloses the author's timezone. Pin it so the output carries none.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

INK = "#1b1b1b"
RULE = "#3a3a3a"
MUTE = "#6f6f6f"
FAINT = "#9a9a9a"
BAND = "#f2f3f4"
HI_FILL = "#fbf1e2"
HI_EDGE = "#8a5a2b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(5.5, 3.95))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x0, y0, x1, y1, fill="white", edge=RULE, lw=0.7):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor=fill, edgecolor=edge, linewidth=lw,
                           zorder=1))


def txt(x, y, s, size=6.4, weight="normal", color=INK, ha="left", va="center",
        style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, style=style, zorder=3, linespacing=1.45)


# ---------------------------------------------------------------- geometry
LX0, LX1 = 0.000, 0.250
COLS = [(0.262, 0.440), (0.446, 0.624), (0.630, 0.808), (0.814, 0.992)]
ROWS = [(0.646, 0.812), (0.480, 0.646), (0.314, 0.480), (0.148, 0.314)]
HEAD_Y = 0.848
SHARED = (0.896, 1.000)

# ------------------------------------------------------------ shared band
box(0.0, SHARED[0], 0.992, SHARED[1], fill=BAND)
txt(0.012, 0.977, "SHARED CONTEXT, VISIBLE TO EVERY AGENT",
    size=5.8, weight="bold", color=MUTE)
txt(0.012, 0.948, r"Apex \$120,000      Cirrus \$140,000      "
                  r"Borealis \$155,000      Delta \$175,000", size=6.6)
txt(0.012, 0.919, "Four requirements. Choose the cheapest vendor that meets "
                  "all of them.", size=6.4, color=MUTE)

# --------------------------------------------------------- column headers
HEADERS = [("Round 0", "independent"),
           ("Round 1", "first exchange"),
           ("Round 2", "delivery reaches peers"),
           ("Round 3", "final answers scored")]
for (cx0, cx1), (h, sub) in zip(COLS, HEADERS):
    mid = (cx0 + cx1) / 2
    txt(mid, HEAD_Y + 0.016, h, size=6.8, weight="bold", ha="center")
    txt(mid, HEAD_Y - 0.014, sub, size=5.6, color=MUTE, ha="center")

# ------------------------------------------------------------- row labels
LABELS = [
    ("Seed (Apex)",
     [("actual: certification lapsed", FAINT),
      ("injected: certification is current", INK)], True),
    ("Verifier (Cirrus)",
     [("backups stored out of region", INK),
      ("Cirrus disqualified", FAINT)], False),
    ("Clearance-holder (Borealis)",
     [("meets every requirement", INK),
      ("the correct answer", FAINT)], False),
    ("Ordinary neutral (Delta)",
     [("hidden fee puts it over budget", INK),
      ("Delta disqualified", FAINT)], False),
]
for (ry0, ry1), (name, lines, hi) in zip(ROWS, LABELS):
    box(LX0, ry0, LX1, ry1,
        fill=HI_FILL if hi else "white",
        edge=HI_EDGE if hi else RULE,
        lw=1.0 if hi else 0.7)
    mid = (ry0 + ry1) / 2
    gap = 0.036
    txt(0.012, mid + gap, name, size=6.6, weight="bold")
    for j, (line, col) in enumerate(lines):
        txt(0.012, mid + gap - (j + 1) * gap, line, size=5.9, color=col,
            style="italic" if col == FAINT else "normal")

# ------------------------------------------------------------- grid cells
CELLS = [
    ["answers alone,\nresponse frozen\nand reused", "restates the claim",
     "may revise", "final answer"],
    ["answers alone", "corrective record\nenters its dossier,\nit speaks",
     "repeats the\ncorrection", "final answer"],
    ["answers alone", "sees the false claim", "sees the delivery",
     "final answer, scored"],
    ["answers alone", "sees the false claim", "sees the delivery",
     "final answer, scored"],
]
HILITE = {(0, 0), (1, 1), (2, 2), (3, 2)}

for r, (ry0, ry1) in enumerate(ROWS):
    for c, (cx0, cx1) in enumerate(COLS):
        hi = (r, c) in HILITE
        box(cx0, ry0, cx1, ry1,
            fill=HI_FILL if hi else "white",
            edge=HI_EDGE if hi else RULE,
            lw=1.0 if hi else 0.7)
        txt((cx0 + cx1) / 2, (ry0 + ry1) / 2, CELLS[r][c],
            size=5.9, ha="center",
            weight="bold" if hi else "normal")

# -------------------------------------------------------------- footnote
ax.plot([0.0, 0.992], [0.112, 0.112], color="#d5d5d5", lw=0.6, zorder=0)
txt(0.0, 0.081,
    "Apex is cheapest and nothing in the shared context rules it out, so the "
    "public information alone points at the trap.", size=5.9, color=RULE)
txt(0.0, 0.052,
    "Identifying Borealis requires all four private findings to be spoken "
    "aloud. An agent sees only the previous round's", size=5.9, color=RULE)
txt(0.0, 0.023,
    "messages, private dossiers persist while messages do not, and the "
    "topology sets which peers a message reaches.", size=5.9, color=RULE)

fig.savefig("docs/figures/fig_setup_rounds.pdf", bbox_inches="tight",
            pad_inches=0.02)
fig.savefig("docs/figures/fig_setup_rounds.png", dpi=300,
            bbox_inches="tight", pad_inches=0.02)
print("wrote docs/figures/fig_setup_rounds.pdf and .png")
