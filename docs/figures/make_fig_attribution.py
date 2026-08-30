"""Speaker attribution: how one agent's output is rendered for its peers.

Renders at exactly 5.5 in wide, so no scaling is needed at \\linewidth in the
NeurIPS single-column layout. Regenerate with:

    python docs/figures/make_fig_attribution.py
"""

import os

# Matplotlib writes a local CreationDate into the PDF, whose UTC offset
# discloses the author's timezone. Pin it so the output carries none.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

INK = "#1b1b1b"
RULE = "#3a3a3a"
MUTE = "#6f6f6f"
BAND = "#f2f3f4"
HI_FILL = "#fbf1e2"
HI_EDGE = "#8a5a2b"
MARK = "#f6dcb4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.monospace": ["Consolas", "DejaVu Sans Mono", "Courier New"],
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_W, FIG_H = 5.5, 3.95
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0.095, 1.0)
ax.axis("off")

MONO = 5.5
CHARW = MONO * 0.601 / 72 / FIG_W     # one monospace glyph, in figure units
LEAD = 0.0265                         # leading for monospace blocks


def box(x0, y0, x1, y1, fill="white", edge=RULE, lw=0.7):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fill,
                           edgecolor=edge, linewidth=lw, zorder=1))


def txt(x, y, s, size=6.1, weight="normal", color=INK, ha="left",
        family="sans-serif", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
            va="center", family=family, style=style, zorder=3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=5.5, linewidth=0.7, color=RULE,
                                 shrinkA=0, shrinkB=0, zorder=2))


def line(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=RULE, lw=0.7, zorder=2,
            solid_capstyle="butt")


# ============================================================ row 1: inputs
box(0.045, 0.878, 0.455, 0.985, fill=BAND)
txt(0.250, 0.957, "Shared task brief", size=6.6, weight="bold", ha="center")
txt(0.250, 0.929, "identical for every agent:", size=5.9, color=MUTE,
    ha="center")
txt(0.250, 0.902, "options, requirements, decision rule", size=5.9, color=MUTE,
    ha="center")

box(0.545, 0.878, 0.955, 0.985, fill=BAND)
txt(0.750, 0.957, "Private dossier, Agent 3 only", size=6.6, weight="bold",
    ha="center")
txt(0.750, 0.929, "Borealis meets all four requirements,", size=5.9,
    color=MUTE, ha="center")
txt(0.750, 0.902, "no disqualifying finding", size=5.9, color=MUTE,
    ha="center")

line(0.250, 0.878, 0.250, 0.842)
line(0.750, 0.878, 0.750, 0.842)
line(0.250, 0.842, 0.750, 0.842)
arrow(0.500, 0.842, 0.500, 0.806)

# ============================================================= row 2: agent
box(0.215, 0.722, 0.785, 0.806)
txt(0.500, 0.781, "Agent 3, assigned to review Borealis", size=6.8,
    weight="bold", ha="center")
txt(0.500, 0.748, "the private finding is never transmitted", size=5.8,
    color=HI_EDGE, style="italic", ha="center")

arrow(0.500, 0.722, 0.500, 0.690)

# ============================================================ row 3: output
box(0.150, 0.522, 0.850, 0.687)
txt(0.168, 0.663, "MODEL OUTPUT: JSON, RATIONALE BEFORE ANSWER", size=5.6,
    weight="bold", color=MUTE)
txt(0.168, 0.630, '"rationale": "Borealis meets every mandatory', size=MONO,
    family="monospace")
txt(0.168, 0.630 - LEAD, '              control and is within budget."',
    size=MONO, family="monospace")
txt(0.168, 0.630 - 2.3 * LEAD, '"answer": "B"', size=MONO, family="monospace")
txt(0.168, 0.541, "rendered into the prompts of peers that receive from Agent 3",
    size=5.8, color=MUTE, style="italic")

arrow(0.500, 0.522, 0.500, 0.496)
line(0.250, 0.496, 0.750, 0.496)
arrow(0.250, 0.496, 0.250, 0.468)
arrow(0.750, 0.496, 0.750, 0.468)

# ================================================== row 4: rendered messages
MY0, MY1 = 0.283, 0.465
box(0.020, MY0, 0.480, MY1)
box(0.520, MY0, 0.980, MY1, fill=HI_FILL, edge=HI_EDGE, lw=1.0)

txt(0.040, 0.440, "Anonymous condition", size=6.5, weight="bold")
txt(0.540, 0.440, "Assigned condition", size=6.5, weight="bold")

ANON = ['Agent 3 — answer: B',
        'Agent 3 — rationale: "Borealis meets',
        'every mandatory control and is',
        'within budget."']
ASGN = ['Agent 3 (assigned: Borealis)',
        '— answer: B',
        'Agent 3 (assigned: Borealis)',
        '— rationale: "Borealis meets every',
        'mandatory control..."']

TOP = 0.405
for i, s in enumerate(ANON):
    txt(0.040, TOP - i * LEAD, s, size=MONO, family="monospace")

for row in (0, 2):
    y = TOP - row * LEAD
    x = 0.540 + 8 * CHARW
    ax.add_patch(Rectangle((x - 0.003, y - 0.0105), 20 * CHARW + 0.006, 0.021,
                           facecolor=MARK, edgecolor="none", zorder=2))
for i, s in enumerate(ASGN):
    txt(0.540, TOP - i * LEAD, s, size=MONO, family="monospace")

txt(0.972, 0.440, "highlighted text is the only difference", size=5.5,
    color=HI_EDGE, style="italic", ha="right")

line(0.250, MY0, 0.250, 0.216)
line(0.750, MY0, 0.750, 0.216)
line(0.250, 0.216, 0.750, 0.216)
arrow(0.500, 0.216, 0.500, 0.182)

# ============================================================= row 5: peers
box(0.180, 0.110, 0.820, 0.180, fill=BAND)
txt(0.500, 0.145, "Peer agents read the message at the start of their next round",
    size=6.4, weight="bold", ha="center")

fig.savefig("docs/figures/fig_attribution.pdf", bbox_inches="tight",
            pad_inches=0.03)
fig.savefig("docs/figures/fig_attribution.png", dpi=300, bbox_inches="tight",
            pad_inches=0.03)
print("wrote docs/figures/fig_attribution.pdf and .png")
