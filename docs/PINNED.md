# Pinned decisions

Deferred, not dropped. Each entry records what was being decided, why it was
deferred, and what would settle it. Nothing is removed — resolved items get a
resolution line so the reasoning stays readable later.

---

## 2026-08-03 — Should H2's outcome be `no_decision` (defer OR tie) instead of deferral alone?

**Status:** RESOLVED — historical `prompt_v1` question; not a `tool_v2`
primary endpoint.

**Resolution (2026-08-04).** Explicit option-E deferral and 2–2 ties remain
separate secondary outcomes. The combined post-hoc `defer_or_tie` measure is
not confirmatory. In the eight-scenario `tool_v2` pilot, explicit deferral was
0% in every arm while relay produced a 50% tie rate, confirming that the two
events should not be silently merged.

**What prompted it.** The exploratory scope-ladder run (10 of 33 scenarios in at
the time) showed deferral *falling* at level 4, contradicting the predicted
increase — but tie rate jumped from 3–7% at levels 1–3 to 27% at level 4. Under
broadcast the group does not defer, it fragments. Combining the two:

| level | deferral | ties | defer OR tie |
|---|---|---|---|
| 1 no correction | 0.405 | 0.071 | 0.476 |
| 2 one verifier | 0.464 | 0.030 | 0.494 |
| 3 relayed w/ source | 0.452 | 0.042 | 0.494 |
| 4 broadcast to all | 0.327 | 0.274 | **0.601** |

Deferral alone is non-monotonic; `defer OR tie` is monotonic.

**The argument for.** The crowding-out account says corrective evidence
eliminates a candidate without licensing one. Both an E vote and a deadlocked
vote are ways of failing to license a winner, so splitting them splits one
construct across two measures.

**The argument against.** The measure was chosen *after* seeing these data. That
is legitimate in an exploratory run whose purpose is to select outcomes, but it
must be declared as such, and the confirmatory run must commit to it in advance.

**What would settle it.** Whether the pattern holds across all 33 scenarios once
the run completes, and whether the tie spike at level 4 reflects genuine
disagreement or a parse/coding artifact — check the vote distributions of tied
level-4 conversations by hand.

**Update 2026-08-03, 11/33 scenarios.** The artifact half is answered: the run
has **zero parse failures and zero generation failures** across 13,092 records,
so the ties are genuine model behaviour. Further, all 50 level-4 ties are 2–2
splits with E as one block — `{C:2,E:2}`×19, `{D:2,E:2}`×14, `{B:2,E:2}`×10,
`{A:2,E:2}`×7. The group is not fragmenting; it is deadlocked over *whether to
defer at all*, which strengthens the case for treating defer and tie as one
construct. Still open: whether it holds across all 33 scenarios, and the
interpretation problem raised in the next entry.

**Affects.** `docs/PREREGISTRATION.md` §1 H2 and §3; `scripts/analyze_statistics.py`
(`build_conversation_frame`, `scope_ladder_report`).

---

## 2026-08-03 — The honest baseline has the HIGHEST no-decision rate, which inverts the account

**Status:** RESOLVED — historical `prompt_v1` baseline account; superseded by
the direct-delivery question.

**Resolution (2026-08-04).** The uncorrupted condition is a reference, not a
scope arm or primary contrast. It receives one repetition per scenario/type
and is described precisely as an uncorrupted team with the standard
corroborating verifier record. No crowding-out claim is based on it.

**What prompted it.** Defer-or-tie by arm, 11/33 scenarios:

| arm | no-decision | accuracy |
|---|---|---|
| 1 no correction | 47.6% | 22.2% |
| 2 one verifier | 51.3% | 36.0% |
| 3 relayed w/ source | 53.4% | 35.4% |
| 4 broadcast to all | 63.4% | 33.9% |
| **no-injection baseline** | **80.0%** | 20.0% |

Groups with no false belief at all fail to decide four times in five — more
often than any injected arm.

**Why it matters.** The crowding-out account said corrective evidence causes
indecision by eliminating without licensing. The baseline says otherwise: the
injection gives the group something to converge on, and widening the correction
returns it toward its uninjected behaviour (47.6 → 63.4 → 80.0). That is close
to the opposite framing, and it is the one the data currently supports.

**What would settle it.** More baseline conversations. The baseline is n=60
against n≈189 per injected arm, because `--baseline-reps` defaults to 1 while
injected arms get 3 seeds. The baseline was a minor comparison when that default
was set and is now load-bearing. Raise it for the confirmatory run.

**Affects.** `docs/PREREGISTRATION.md` §1 (the account in the block quote), §2
sample size, §3.

---

## 2026-08-03 — Should group accuracy stay a primary hypothesis given the floor effect?

**Status:** RESOLVED — group accuracy is secondary.

**Resolution (2026-08-04).** Ordinary-neutral final accuracy is the primary
outcome because it measures downstream recovery without mechanically including
the treated seed or evidence-holding verifier. Group accuracy, trap wins, and
ties remain key secondary outcomes. The eight-scenario `tool_v2` pilot showed
why: relay reached 43.8% ordinary-neutral accuracy but only 18.8% group accuracy
because half of groups tied.

**What prompted it.** Accuracy is 20% at baseline and 22–36% across all four
ladder arms, against a 25% chance rate and a 99.4% oracle ceiling. H3 is stated
as `accuracy(L4) ≤ accuracy(L3)`; the observed difference is −0.021
[−0.125, +0.073].

**Why it matters.** A hypothesis measured inside a 16-point band that straddles
chance is weakly informative whichever way it lands. The no-pooling fix — which
was correct and necessary — appears to have made the task hard enough that
groups mostly decline to decide rather than decide wrongly. That is a property
of the instrument, not of corrective-evidence scope.

**Options.** (a) Keep H3 and report the floor effect as a stated limitation;
(b) demote H3 to secondary and promote `no_decision` to the second primary;
(c) keep H3 but power it against the no-decision outcome instead.

**What would settle it.** The full 33-scenario run — if accuracy spreads wider
across scenarios than it does across arms, the floor is a corpus-average
artifact and (a) is defensible. Otherwise prefer (b).

**Affects.** `docs/PREREGISTRATION.md` §1 H3, §3 primary outcomes and
multiplicity correction (Holm is currently across three tests).

---

## 2026-08-03 — Corrector answer/rationale incoherence: hand-labelling still needed?

**Status:** RESOLVED for primary analysis; rationale validation remains
secondary quality assurance.

**Resolution (2026-08-04).** The primary outcome is the parsed answer letter.
Rationale consistency flags and judge labels cannot override or exclude a
valid primary answer. Human rationale labels are still useful before making
qualitative evidence-use claims, but they are no longer a blocker for the
answer-based confirmatory run.

**What prompted it.** Two regexes disagreed on the incoherence rate (5% vs 30%)
and neither was trusted. On this run the flag fires at **3.1% for correctors at
round 2** (25/813), the worst round, and 1.5% across all 13,092 records.

**Why it is still open.** The number comes from the same flag whose validity is
in question, so it cannot validate itself. But 3.1% is the low end of the
disputed range, which lowers the risk that the primary outcome's coding is
compromised.

**What would settle it.** The hand-labelled sample in `data/human_labels/`
(~100 corrector rationales at round 2, labelled by someone not working on the
analysis code). Still required before the prereg is committed per §7, but it now
looks like a confirmation step rather than a possible show-stopper.

**Affects.** `docs/PREREGISTRATION.md` §7 measurement-validity precondition.

---

## 2026-08-03 — `AsyncOpenAI` has no timeout, so a wedged request stalls the run silently

**Status:** RESOLVED — the Azure client uses an explicit 120-second SDK timeout
and disables hidden SDK retries; the existing six-attempt tenacity policy is
the only retry layer. Confirmatory runners set `PYTHONUNBUFFERED=1`.

**What prompted it.** The exploratory run froze for ~11 minutes with no output
and no writes to `records.jsonl`. The process was alive with flat CPU, blocked
on a network read. `AzureChatClient` constructs `AsyncOpenAI` with no `timeout`
and no `max_retries`, so a stuck request hangs for the SDK default of 600s
before `APITimeoutError` is raised and tenacity retries. It self-healed.

**Why it matters.** On a multi-hour confirmatory run this costs 10 minutes per
occurrence, and with stdout piped through `Tee-Object` the stall is invisible.

**Fix.** Set `timeout=120` on the client and let tenacity handle the retry —
turning a 10-minute silent stall into a 2-minute logged one. Also run with
`python -u` so a pipe does not block-buffer progress output.

**Affects.** `src/assumption_propagation/clients.py` (`AzureChatClient.__init__`).
