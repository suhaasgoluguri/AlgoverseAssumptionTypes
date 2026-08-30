"""Emit the LaTeX tables from the frozen analysis files.

Every number is read from the run's analysis JSON, so a re-analysis changes
the tables without anyone retyping a cell. Regenerate with:

    python docs/figures/make_tables.py

Writes docs/figures/tab_*.tex, each a self-contained table environment.
"""

import collections
import json
import os

RUN = "outputs/raw/20260810_merged_confirmatory_scope33_gpt54mini_v32"
SEC = json.load(open(f"{RUN}/analysis/statistics_secondary.json"))
CONF = json.load(open(f"{RUN}/analysis/statistics_confirmatory.json"))

OUT = "docs/figures"

ARMS = [
    ("injection_tool_none", "No correction"),
    ("injection_tool_testimony", "Testimony"),
    ("injection_tool_relay", "Exact relay"),
    ("injection_tool_direct", "Tool receipt"),
    ("injection_prompt_receipt_control", "Prompt receipt"),
    ("injection_tool_placebo", "Tool placebo"),
]


def pp(x):
    """Proportion to percentage points, one decimal, LaTeX minus sign."""
    return f"{100 * x:+.1f}".replace("-", "$-$")


def pct(x):
    return f"{100 * x:.1f}"


def ci(d):
    return f"[{pp(d['ci_low'])}, {pp(d['ci_high'])}]"


def cell(d):
    return f"{pp(d['difference'])} {{\\scriptsize {ci(d)}}}"


def write(name, body):
    # Table 2 sits in the body and wants the top of a page. The appendix
    # tables must stay where they are put: seven queued [t] floats overflow
    # forward and land in the middle of the NeurIPS checklist.
    if name != "tab_attribution.tex":
        body = body.replace(r"\begin{table}[t]", r"\begin{table}[!ht]", 1)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print("wrote", path)


def cluster_boot(by_scen, draws=10000, seed=20260820):
    """Paired cluster bootstrap of assigned - anonymous, scenario as cluster.

    ``by_scen`` maps scenario id to {"assigned": [0/1, ...],
    "anonymous": [0/1, ...]}. Matches the pipeline's own procedure: resample
    scenarios with replacement, pool observations within the resample.
    """
    import random
    keys = sorted(by_scen)

    def stat(sample):
        a = [v for k in sample for v in by_scen[k]["assigned"]]
        b = [v for k in sample for v in by_scen[k]["anonymous"]]
        return sum(a) / len(a) - sum(b) / len(b)

    rng = random.Random(seed)
    out = sorted(stat([rng.choice(keys) for _ in keys]) for _ in range(draws))
    return {"difference": stat(keys),
            "ci_low": out[int(0.025 * draws)],
            "ci_high": out[int(0.975 * draws)]}


# ------------------------------------------------- one pass over the records
# Two things the analysis files do not carry: agent-level deferral (Table 2)
# and which of the two ordinary agents is which (Table A.1). Roles are not
# recorded per row -- orchestrator.neutral_ordinal_of gives neutral_briefs in
# ascending agent-id order among the non-seed, non-corrector agents, and
# neutral_briefs[0] is the brief that clears the correct option.
SNAP = {s["scenario_id"]: s for s in
        json.load(open(f"{RUN}/scenario_snapshot.json", encoding="utf-8"))}

TRAP = {}
for line in open(f"{RUN}/analysis/conversations.jsonl", encoding="utf-8"):
    c = json.loads(line)
    TRAP[(c["scenario_id"], c["seed_instance_id"], c["condition"],
          c["topology"], c["attribution"])] = c.get("favored_option")

hits = collections.defaultdict(lambda: [0, 0, 0, 0])   # accK accN trapK trapN
defer = {k: collections.defaultdict(
    lambda: {"anonymous": [], "assigned": []}) for k, _ in ARMS}

for line in open(f"{RUN}/records.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["round"] != 3 or r["topology"] != "fully_connected_mesh":
        continue
    seed, corr, me = r["seed_agent_id"], r["corrector_agent_id"], r["agent_id"]
    if me in (seed, corr):
        continue
    if r["condition"] in defer:
        defer[r["condition"]][r["scenario_id"]][r["attribution"]].append(
            1.0 if r.get("answer") == "E" else 0.0)
    if r["attribution"] != "assigned":
        continue
    neutrals = [a for a in (0, 1, 2, 3) if a not in (seed, corr)]
    role = "clearance" if neutrals.index(me) == 0 else "ordinary"
    slot = hits[(r["condition"], role)]
    slot[1] += 1
    if r.get("answer") == SNAP[r["scenario_id"]]["correct_answer"]:
        slot[0] += 1
    trap = TRAP.get((r["scenario_id"], r["seed_instance_id"], r["condition"],
                     r["topology"], r["attribution"]))
    if trap:
        slot[3] += 1
        if r.get("answer") == trap:
            slot[2] += 1

DEFER = {k: cluster_boot(defer[k]) for k, _ in ARMS}


# ---------------------------------------------------------------- Table 2
C = SEC["attribution_effects_within_mesh"]["contrasts"]

rows = []
for key, label in ARMS:
    c = C[key]
    rows.append(
        f"{label} & {cell(c['ordinary_neutral_accuracy'])} & "
        f"{cell(DEFER[key])} & "
        f"{cell(c['ordinary_neutral_final_trap_share'])} & "
        f"{cell(c['group_accuracy'])} \\\\"
    )

write("tab_attribution.tex", r"""\begin{table}[t]
\centering
\caption{Effect of speaker attribution within each delivery arm, fully
connected mesh, GPT-5.4-mini. Each entry is the attributed condition minus
the anonymous condition in percentage points, with a 95\% paired cluster
bootstrap interval over 33 scenarios (10,000 draws). The first three columns
are round-3 answers from the two ordinary agents ($n=396$), where deferral is
the share declining to choose a candidate; group accuracy is the plurality
answer of all four agents ($n=198$). Deferral falls in every arm; trap
adoption rises only in the three arms where ordinary agents never receive the
record.}
\label{tab:attribution}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{3}{c}{Ordinary agents} & Group \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}
Arm & $\Delta$ Accuracy & $\Delta$ Deferral & $\Delta$ Trap & $\Delta$ Accuracy \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------- Table A.3: levels
L = SEC["by_attribution"]["assigned"]["by_topology"]["fully_connected_mesh"]["levels"] \
    if "by_topology" in SEC["by_attribution"]["assigned"] \
    else SEC["scope_ladder"]["levels"]

rows = []
for key, label in ARMS:
    d = L[key]
    rows.append(
        f"{label} & {pct(d['ordinary_neutral_accuracy'])} & "
        f"{pct(d['group_accuracy'])} & {pct(d['final_trap_share'])} & "
        f"{pct(d['deferral'])} & {pct(d['tie_rate'])} & "
        f"{pct(d['pre_scope_trap_to_final_correct_recovery_rate'])} \\\\"
    )

write("tab_outcomes.tex", r"""\begin{table}[t]
\centering
\caption{All measured outcomes by delivery arm in the primary
cell (fully connected mesh, assigned attribution, GPT-5.4-mini), in percent.
Ordinary accuracy and trap adoption pool the two ordinary agents' round-3
answers; group accuracy, deferral and tie describe the group's plurality
answer. Recovery is the share of conversations in which the ordinary agents
held the trap answer before the correction and reached the correct answer by
round 3. Accuracy is associated more strongly with receiving the record than
with the tool interaction alone.}
\label{tab:outcomes}
\small
\begin{tabular}{lcccccc}
\toprule
Arm & Ordinary acc. & Group acc. & Trap & Group defers & Tie & Recovery \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# --------------------------------------------------- Table A.4: contrasts
AC = SEC["scope_ladder"]["adjacent_contrasts"]
HOLM = SEC["scope_ladder"]["primary_inference"]["holm_adjusted_p_values"]

CONTRASTS = [
    ("testimony_minus_none", "C1", "Testimony $-$ no correction"),
    ("relay_minus_testimony", "C2", "Exact relay $-$ testimony"),
    ("direct_tool_minus_relay", "C3", "Tool receipt $-$ exact relay"),
    ("direct_tool_minus_prompt_receipt", "", "Tool receipt $-$ prompt receipt"),
    ("placebo_minus_testimony", "", "Tool placebo $-$ testimony"),
    ("direct_tool_minus_placebo", "", "Tool receipt $-$ tool placebo"),
]


def pval(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}"[1:]


rows = []
for key, tag, label in CONTRASTS:
    acc = AC[key]["ordinary_neutral_accuracy"]
    grp = AC[key]["group_accuracy"]
    name = f"{tag} $\\cdot$ {label}" if tag else f"\\quad {label}"
    p = HOLM.get(key)
    pcol = pval(p) if p is not None else "---"
    rows.append(
        f"{name} & {pp(acc['difference'])} & {ci(acc)} & {pcol} & "
        f"{pp(grp['difference'])} & {pval(grp['p_value'])} \\\\"
    )

write("tab_contrasts.tex", r"""\begin{table}[t]
\centering
\caption{Contrasts and mechanism controls on ordinary-agent accuracy,
primary cell, in percentage points with 95\% paired cluster bootstrap
intervals over 33 scenarios. Holm correction is applied across the three
contrasts C1--C3 only; the mechanism controls were not part of that family
and are reported uncorrected.}
\label{tab:contrasts}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{Ordinary accuracy} & \multicolumn{2}{c}{Group accuracy} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Contrast & $\Delta$ & 95\% CI & Holm $p$ & $\Delta$ & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# =================================================== Table A.1: role split
# Computed in the records pass above, because the pipeline pools both
# ordinary agents everywhere and never reports them separately.
rows = []
for key, label in ARMS + [("no_injection_baseline", "Honest baseline")]:
    cl, od = hits[(key, "clearance")], hits[(key, "ordinary")]
    pooled = 100 * (cl[0] + od[0]) / (cl[1] + od[1])
    rows.append(
        f"{label} & {100*cl[0]/cl[1]:.1f} & {100*od[0]/od[1]:.1f} & {pooled:.1f} & "
        f"{100*cl[2]/cl[3]:.1f} & {100*od[2]/od[3]:.1f} \\\\"
    )

write("tab_roles.tex", r"""\begin{table}[t]
\centering
\caption{What pooling the two ordinary agents hides. Round-3 accuracy and
adoption of the injected claim, fully connected mesh under assigned
attribution, in percent. The clearance-holder privately holds the finding
that clears the correct option; the ordinary neutral holds a disqualification
for a losing option. Every accuracy reported elsewhere in this paper is the
pooled column. $n=198$ per agent per arm ($n=66$ for the baseline).}
\label{tab:roles}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{Accuracy} & \multicolumn{2}{c}{Trap adoption} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Arm & Clearance & Ordinary & Pooled & Clearance & Ordinary \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ============================================ Table A.2: authority control
# A separate mesh collection with the shared brief's authority sentence
# removed. The two collections are never differenced: the brief changed, so
# the seed prompts changed, so the frozen seeds are not the main run's. Every
# contrast below is attributed vs anonymous WITHIN one collection.
CTRL_RUN = "outputs/raw/20260817_merge_scenarios_noauthority_gpt54mini"
CTRL = json.load(open(f"{CTRL_RUN}/analysis/statistics_noauthority.json",
                      encoding="utf-8"))

MAIN_C = SEC["attribution_effects_within_mesh"]["contrasts"]
CTRL_C = CTRL["attribution_effects_within_mesh"]["contrasts"]


def group_acc_by_scenario(run_dir, condition, topology="fully_connected_mesh"):
    by_scen = collections.defaultdict(lambda: {"anonymous": [], "assigned": []})
    for line in open(f"{run_dir}/results.jsonl", encoding="utf-8"):
        c = json.loads(line)
        if c["condition"] != condition or c["topology"] != topology:
            continue
        by_scen[c["scenario_id"]][c["attribution"]].append(
            1.0 if c["final_answer_correct"] else 0.0)
    return by_scen


def boot_baseline(run_dir):
    return cluster_boot(group_acc_by_scenario(run_dir, "no_injection_baseline"))


_TRAP_CACHE = {}


def trap_by_scenario(run_dir, condition, topology="fully_connected_mesh"):
    """Ordinary-agent adoption of the injected claim, by scenario and header.

    The trap option is per conversation, so it has to be read from
    conversations.jsonl and joined back onto the round-3 records. Same
    quantity as Table 2's trap column, but computed for either run.
    """
    if run_dir not in _TRAP_CACHE:
        t = {}
        for line in open(f"{run_dir}/analysis/conversations.jsonl",
                         encoding="utf-8"):
            c = json.loads(line)
            t[(c["scenario_id"], c["seed_instance_id"], c["condition"],
               c["topology"], c["attribution"])] = c.get("favored_option")
        _TRAP_CACHE[run_dir] = t
    trap = _TRAP_CACHE[run_dir]

    by_scen = collections.defaultdict(lambda: {"anonymous": [], "assigned": []})
    for line in open(f"{run_dir}/records.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if (r["round"] != 3 or r["topology"] != topology
                or r["condition"] != condition):
            continue
        seed, corr, me = r["seed_agent_id"], r["corrector_agent_id"], r["agent_id"]
        if me in (seed, corr):
            continue
        key = (r["scenario_id"], r["seed_instance_id"], r["condition"],
               r["topology"], r["attribution"])
        want = trap.get(key)
        if not want:
            continue
        by_scen[r["scenario_id"]][r["attribution"]].append(
            1.0 if r.get("answer") == want else 0.0)
    return by_scen


rows = []
for key, label in ARMS:
    m, c = MAIN_C[key], CTRL_C[key]
    rows.append(
        f"{label} & {pp(m['group_accuracy']['difference'])} {{\\scriptsize "
        f"{ci(m['group_accuracy'])}}} & {pp(c['group_accuracy']['difference'])} "
        f"{{\\scriptsize {ci(c['group_accuracy'])}}} & "
        f"{pp(m['ordinary_neutral_final_trap_share']['difference'])} "
        f"{{\\scriptsize {ci(m['ordinary_neutral_final_trap_share'])}}} & "
        f"{pp(c['ordinary_neutral_final_trap_share']['difference'])} "
        f"{{\\scriptsize {ci(c['ordinary_neutral_final_trap_share'])}}} \\\\"
    )

bm, bc = boot_baseline(RUN), boot_baseline(CTRL_RUN)
rows.append(
    f"Honest baseline & {pp(bm['difference'])} {{\\scriptsize {ci(bm)}}} & "
    f"{pp(bc['difference'])} {{\\scriptsize {ci(bc)}}} & --- & --- \\\\"
)

write("tab_authority.tex", r"""\begin{table}[t]
\centering
\caption{The attribution effect with and without the shared brief's authority
rule, fully connected mesh, GPT-5.4-mini. Each entry is attributed minus
anonymous in percentage points, with a 95\% paired cluster bootstrap interval
over 33 scenarios. The control collection removes the brief's sentence stating
that a reviewer's report about their own assignment is authoritative and
sufficient. Because the brief changed, the seed responses were newly
generated, so the two collections are reported side by side rather than
differenced; every contrast is estimated within one collection.}
\label{tab:authority}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{$\Delta$ Group accuracy} & \multicolumn{2}{c}{$\Delta$ Trap adoption} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Arm & Main & Control & Main & Control \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ================================================== Table A.5: cross-model
DS_RUN = "outputs/raw/20260807_193819_azure_confirmatory_scope33_deepseekv4pro_v32"

ds = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
for line in open(f"{DS_RUN}/analysis/aggregates.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["topology"] != "fully_connected_mesh" or r["attribution"] != "assigned":
        continue
    s = ds[r["condition"]]
    on_n = r["ordinary_neutral_final_assigned_n"]
    s[0] += round(r["ordinary_neutral_final_accuracy"] * on_n)
    s[1] += on_n
    s[2] += round(r["final_plurality_accuracy"] * r["n_conversations"])
    s[3] += r["n_conversations"]
    s[4] += round(r["ordinary_neutral_favored_adoption_final"] * on_n)
    s[5] += on_n

rows = []
for key, label in ARMS:
    g, d = L[key], ds[key]
    rows.append(
        f"{label} & {pct(g['ordinary_neutral_accuracy'])} & "
        f"{pct(g['group_accuracy'])} & {pct(g['final_trap_share'])} & "
        f"{100*d[0]/d[1]:.1f} & {100*d[2]/d[3]:.1f} & {100*d[4]/d[5]:.1f} \\\\"
    )

write("tab_crossmodel.tex", r"""\begin{table}[t]
\centering
\caption{The six delivery arms in both models, fully connected mesh under
assigned attribution, in percent. DeepSeek-V4-Pro ran the same corpus, grid
and protocol over 189 of the 198 seed instances. The two models sit at very
different operating points, so the ordering and direction of the arms are
comparable but the size of the gaps between them is not. Both models improve
with the record and neither improves with the tool channel alone.}
\label{tab:crossmodel}
\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{GPT-5.4-mini} & \multicolumn{3}{c}{DeepSeek-V4-Pro} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Arm & Ordinary & Group & Trap & Ordinary & Group & Trap \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ======================================== Table A.6: honest baseline, topology
# Supports the baseline paragraph of Section 4.2, which cites eight rates
# across the three topologies. Decomposed because the failure mode matters:
# without attribution groups do not choose the wrong option, they fail to
# choose at all.
TOPOS = [("fully_connected_mesh", "Fully connected mesh"),
         ("looped_chain", "Looped chain"),
         ("neutral_star", "Neutral star")]

base = collections.defaultdict(collections.Counter)
for line in open(f"{RUN}/results.jsonl", encoding="utf-8"):
    c = json.loads(line)
    if c["condition"] != "no_injection_baseline":
        continue
    k = (c["topology"], c["attribution"])
    if c["no_consensus"]:
        base[k]["tie"] += 1
    elif c["plurality_answer"] == "E":
        base[k]["declined"] += 1
    elif c["plurality_answer"] == SNAP[c["scenario_id"]]["correct_answer"]:
        base[k]["correct"] += 1
    else:
        base[k]["wrong"] += 1

rows = []
for topo, label in TOPOS:
    for mode, mlabel in [("anonymous", "Anonymous"), ("assigned", "Attributed")]:
        d = base[(topo, mode)]
        n = sum(d.values())
        name = f"{label} & {mlabel}" if mode == "anonymous" else f" & {mlabel}"
        rows.append(name + " & " + " & ".join(
            f"{100*d[x]/n:.1f}" for x in ["correct", "tie", "declined", "wrong"]
        ) + r" \\")
    if topo != TOPOS[-1][0]:
        rows.append(r"\addlinespace")

write("tab_baseline.tex", r"""\begin{table}[t]
\centering
\caption{The honest baseline by topology and speaker attribution,
GPT-5.4-mini, in percent. No claim is injected and no correction is needed:
every agent reports truthfully and the findings that identify the correct
option are distributed across the four dossiers. Groups are scored on their
plurality answer, decomposed into the correct option, no plurality (tie), a
plurality declining to choose, and a wrong option. $n=66$ conversations per
row. The wrong-option column is zero everywhere: without attribution honest
groups fail by deadlocking or declining, never by erring.}
\label{tab:baseline}
\small
\begin{tabular}{llcccc}
\toprule
Topology & Header & Correct & Tie & Declined & Wrong \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# =================================== Table A.7: attribution effect, by model
# Section 4.3 claims the attribution result does not replicate. The DeepSeek
# run folder carries no statistics file, so its intervals are bootstrapped
# here with the same procedure used for the primary model.
ds_boot = {k: cluster_boot(group_acc_by_scenario(DS_RUN, k)) for k, _ in ARMS}
# The trap column is what makes the cross-model comparison honest: only the
# benefit of attribution is GPT-specific, the cost shows up in both models.
gt_trap = {k: cluster_boot(trap_by_scenario(RUN, k)) for k, _ in ARMS}
ds_trap = {k: cluster_boot(trap_by_scenario(DS_RUN, k)) for k, _ in ARMS}

rows = []
for key, label in ARMS:
    g, d = MAIN_C[key]["group_accuracy"], ds_boot[key]
    gt, dt = gt_trap[key], ds_trap[key]
    rows.append(
        f"{label} & {pp(g['difference'])} {{\\scriptsize {ci(g)}}} & "
        f"{pp(d['difference'])} {{\\scriptsize {ci(d)}}} & "
        f"{pp(gt['difference'])} {{\\scriptsize {ci(gt)}}} & "
        f"{pp(dt['difference'])} {{\\scriptsize {ci(dt)}}} \\\\"
    )
bm = cluster_boot(group_acc_by_scenario(RUN, "no_injection_baseline"))
bd = cluster_boot(group_acc_by_scenario(DS_RUN, "no_injection_baseline"))
rows.append(r"\addlinespace")
rows.append(
    f"Honest baseline & {pp(bm['difference'])} {{\\scriptsize {ci(bm)}}} & "
    f"{pp(bd['difference'])} {{\\scriptsize {ci(bd)}}} & --- & --- \\\\"
)

write("tab_attribution_models.tex", r"""\begin{table}[t]
\centering
\caption{The attribution effect in both models, fully connected mesh, in
percentage points with 95\% paired cluster bootstrap intervals over 33
scenarios. Each entry is attributed minus anonymous within one model. Group
accuracy is the plurality answer; trap adoption is the two ordinary agents'
round-3 adoption of the injected claim. The two halves of the effect
separate: naming speakers raises group accuracy for GPT-5.4-mini in five of
six arms and for DeepSeek-V4-Pro in none, but it raises adoption of the
injected claim in both models.}
\label{tab:attributionmodels}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{$\Delta$ Group accuracy} & \multicolumn{2}{c}{$\Delta$ Trap adoption} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Arm & GPT-5.4-mini & DeepSeek-V4-Pro & GPT-5.4-mini & DeepSeek-V4-Pro \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ================================== Table A.8: the looped chain (appendix B)
# Two arms only, both models, chain under assigned attribution. The point of
# the table is the gap between the ordinary and group columns, so they sit
# next to each other.
CHAIN_RUNS = [("GPT-5.4-mini", RUN), ("DeepSeek-V4-Pro", DS_RUN)]
CHAIN_ARMS = [("injection_tool_none", "No correction"),
              ("injection_tool_testimony", "Testimony")]

rows = []
for mlabel, run in CHAIN_RUNS:
    acc = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for line in open(f"{run}/analysis/aggregates.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["topology"] != "looped_chain" or r["attribution"] != "assigned":
            continue
        if r["condition"] not in dict(CHAIN_ARMS):
            continue
        s = acc[r["condition"]]
        n, c = r["ordinary_neutral_final_assigned_n"], r["n_conversations"]
        s[0] += round(r["ordinary_neutral_final_accuracy"] * n)
        s[1] += n
        s[2] += round(r["final_plurality_accuracy"] * c)
        s[3] += c
        s[4] += round(r["final_no_consensus"] * c)
        s[5] += round(r["ordinary_neutral_favored_adoption_final"] * n)
    for i, (key, alabel) in enumerate(CHAIN_ARMS):
        s = acc[key]
        name = f"{mlabel} & {alabel}" if i == 0 else f" & {alabel}"
        rows.append(name + " & " + " & ".join([
            f"{100*s[0]/s[1]:.1f}", f"{100*s[2]/s[3]:.1f}",
            f"{100*s[4]/s[3]:.1f}", f"{100*s[5]/s[1]:.1f}"]) + r" \\")
    if mlabel != CHAIN_RUNS[-1][0]:
        rows.append(r"\addlinespace")

write("tab_chain.tex", r"""\begin{table}[t]
\centering
\caption{The looped chain under assigned attribution, in percent. Ordinary
accuracy is the two ordinary agents' round-3 answers; the remaining columns
describe the group's plurality answer. Without correction the agents are
individually almost always right and the group almost never is, because most
conversations reach no plurality at all. One verifier's testimony closes the
group column while barely moving the individual one.}
\label{tab:chain}
\small
\begin{tabular}{llcccc}
\toprule
Model & Arm & Ordinary acc. & Group acc. & No plurality & Trap \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
