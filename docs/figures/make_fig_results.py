"""Results figure: the delivery ladder across models, and the attribution trade-off.

Values are the primary cell (fully connected mesh) with both
neutral agents pooled, as reported in Section 4. Regenerate with:

    python docs/figures/make_fig_results.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["No\ncorrection", "Testimony", "Exact\nrelay", "Direct\ntool",
        "Prompt\nreceipt", "Tool\nplacebo"]

# Panel A: ordinary-agent accuracy, mesh, assigned attribution.
GPT_ACC = [36.1, 68.7, 93.9, 94.7, 95.5, 72.7]
DS_ACC = [3.4, 10.8, 74.9, 100.0, 97.6, 3.4]

# Panel B: adoption of the injected claim, GPT-5.4-mini, mesh.
TRAP_ANON = [27.0, 10.9, 3.5, 2.3, 2.3, 9.1]
TRAP_ASGN = [61.9, 24.0, 4.5, 1.8, 2.3, 18.7]

INK = "#1a1a1a"
GREY = "#8c8c8c"
LIGHT = "#c9c9c9"
ACCENT = "#8a5a2b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
x = np.arange(len(ARMS))
w = 0.38

# ---- Panel A -------------------------------------------------------------
axA.bar(x - w / 2, GPT_ACC, w, label="GPT-5.4-mini",
        color=GREY, edgecolor=INK, linewidth=0.6)
axA.bar(x + w / 2, DS_ACC, w, label="DeepSeek-V4-Pro",
        color="white", edgecolor=INK, linewidth=0.6, hatch="////")
for xi, v in zip(x - w / 2, GPT_ACC):
    axA.text(xi, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)
for xi, v in zip(x + w / 2, DS_ACC):
    axA.text(xi, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)

axA.set_ylabel("Ordinary-agent accuracy (%)")
axA.set_ylim(0, 112)
axA.set_yticks([0, 25, 50, 75, 100])
axA.set_xticks(x)
axA.set_xticklabels(ARMS, fontsize=6.8)
axA.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.4)
axA.set_title("A  Delivering the record", loc="left", fontsize=8.5, fontweight="bold")
axA.spines[["top", "right"]].set_visible(False)
axA.tick_params(length=2.5)

# ---- Panel B -------------------------------------------------------------
axB.bar(x - w / 2, TRAP_ANON, w, label="Anonymous",
        color=LIGHT, edgecolor=INK, linewidth=0.6)
axB.bar(x + w / 2, TRAP_ASGN, w, label="Attributed",
        color=ACCENT, edgecolor=INK, linewidth=0.6)
for xi, v in zip(x - w / 2, TRAP_ANON):
    axB.text(xi, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)
for xi, v in zip(x + w / 2, TRAP_ASGN):
    axB.text(xi, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)

axB.axvspan(-0.55, 1.5, color="#f2f2f2", zorder=0)
axB.axvspan(4.5, 5.55, color="#f2f2f2", zorder=0)
axB.text(0.5, 70, "record never reaches\nordinary agents", ha="center",
         fontsize=6.2, color=GREY, style="italic")

axB.set_ylabel("Adoption of the false claim (%)")
axB.set_ylim(0, 78)
axB.set_yticks([0, 20, 40, 60])
axB.set_xticks(x)
axB.set_xticklabels(ARMS, fontsize=6.8)
axB.set_xlim(-0.55, 5.55)
axB.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.4)
axB.set_title("B  Naming the speaker (GPT-5.4-mini)", loc="left",
              fontsize=8.5, fontweight="bold")
axB.spines[["top", "right"]].set_visible(False)
axB.tick_params(length=2.5)

fig.tight_layout(pad=0.6)
fig.savefig("docs/figures/fig_results.pdf", bbox_inches="tight")
fig.savefig("docs/figures/fig_results.png", dpi=220, bbox_inches="tight")
print("wrote docs/figures/fig_results.pdf and .png")
