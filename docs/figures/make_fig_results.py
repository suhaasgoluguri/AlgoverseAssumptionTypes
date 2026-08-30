"""Results figure: the delivery ladder across models, and the attribution effect.

Renders at exactly 5.5 in wide so it needs no scaling at \\linewidth in the
NeurIPS single-column layout, which is why the declared font sizes are the
sizes that reach the page. Values are the primary cell (fully connected mesh)
with both ordinary agents pooled, as reported in Section 4. Regenerate with:

    python docs/figures/make_fig_results.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["No\ncorrection", "Testimony", "Exact\nrelay", "Tool\nreceipt",
        "Prompt\nreceipt", "Tool\nplacebo"]

# Panel A: ordinary-agent accuracy, mesh, assigned attribution.
GPT_ACC = [36.1, 68.7, 93.9, 94.7, 95.5, 72.7]
DS_ACC = [3.4, 10.8, 74.9, 100.0, 97.6, 3.4]

# Panel B: adoption of the injected claim, GPT-5.4-mini, mesh.
TRAP_ANON = [27.0, 10.9, 3.5, 2.3, 2.3, 9.1]
TRAP_ASGN = [61.9, 24.0, 4.5, 1.8, 2.3, 18.7]

INK = "#1a1a1a"
GRID = "#dcdcdc"
STEEL = "#5b6b78"
LIGHT = "#d4d4d4"
ACCENT = "#9c5a1c"
BLUE = "#2e6f9e"
GREEN = "#1e7a63"
PURPLE = "#7a5aa6"
RED = "#c4551f"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.75))
x = np.arange(len(ARMS))
w = 0.36


def values(ax, xs, vals, pad, against=None, lift=6.0):
    """Label each bar, lifting the second of a close pair clear of the first.

    Two labels above bars of similar height collide horizontally however small
    the type, so the fix has to be vertical rather than a smaller font.
    """
    for i, (xi, v) in enumerate(zip(xs, vals)):
        dy = lift if against and abs(v - against[i]) < 12 else 0.0
        ax.text(xi, v + pad + dy, f"{v:.1f}", ha="center", va="bottom",
                fontsize=5.5)


def bracket(ax, x0, x1, y, label, color, drop, gap):
    """Span the two bars a contrast compares, and label the gap.

    The intervals are on paired differences rather than on the plotted levels,
    so the annotation belongs on the gap, not on either bar.
    """
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop],
            color=color, lw=1.1, clip_on=False, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y + gap, label, ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color=color)


def frame(ax, title, ylabel, ymax, ticks):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_ylim(0, ymax)
    ax.set_yticks(ticks)
    ax.set_xticks(x)
    ax.set_xticklabels(ARMS, fontsize=6.5)
    ax.set_xlim(-0.6, 5.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, labelsize=6.5)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=24)
    ax.legend(frameon=False, fontsize=7, ncol=2, handlelength=1.1,
              columnspacing=1.2, handletextpad=0.5,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))


# ---- Panel A -------------------------------------------------------------
axA.bar(x - w / 2, GPT_ACC, w, label="GPT-5.4-mini",
        color=STEEL, edgecolor=INK, linewidth=0.6)
axA.bar(x + w / 2, DS_ACC, w, label="DeepSeek-V4-Pro",
        color="white", edgecolor=INK, linewidth=0.6, hatch="////")
values(axA, x - w / 2, GPT_ACC, 2.0)
values(axA, x + w / 2, DS_ACC, 2.0, against=GPT_ACC, lift=7.0)

bracket(axA, -w / 2, 1 - w / 2, 108, "C1 +32.6 pp", BLUE, 5.0, 3.0)
bracket(axA, 1 - w / 2, 2 - w / 2, 140, "C2 +25.3 pp", GREEN, 5.0, 3.0)
bracket(axA, 3 - w / 2, 4 - w / 2, 124, "Tool $-$ prompt  $-$0.8 pp",
        PURPLE, 5.0, 3.0)

frame(axA, "A  Delivering the record", "Ordinary-agent accuracy (%)",
      168, [0, 25, 50, 75, 100])

# ---- Panel B -------------------------------------------------------------
axB.bar(x - w / 2, TRAP_ANON, w, label="Anonymous",
        color=LIGHT, edgecolor=INK, linewidth=0.6)
axB.bar(x + w / 2, TRAP_ASGN, w, label="Explicitly assigned",
        color=ACCENT, edgecolor=INK, linewidth=0.6)
values(axB, x - w / 2, TRAP_ANON, 1.3)
values(axB, x + w / 2, TRAP_ASGN, 1.3, against=TRAP_ANON, lift=4.0)

bracket(axB, -w / 2, w / 2, 70, "+34.8 pp", RED, 3.5, 2.0)

frame(axB, "B  Naming the speaker (GPT-5.4-mini)",
      "False-claim adoption (%)", 88, [0, 20, 40, 60])

fig.tight_layout(pad=0.5, w_pad=1.8)
fig.savefig("docs/figures/fig_results.pdf", bbox_inches="tight",
            pad_inches=0.02)
fig.savefig("docs/figures/fig_results.png", dpi=300, bbox_inches="tight",
            pad_inches=0.02)
print("wrote docs/figures/fig_results.pdf and .png")
