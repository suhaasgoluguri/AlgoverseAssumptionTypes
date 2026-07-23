"""Deterministic authoring audit for ``data/scenarios_fixed.json``.

This audit intentionally checks only properties that can be established from
the JSON itself.  It does not ask a model to decide whether prose is plausible
or whether an option is substantively correct.  Every finding identifies the
field that an author should inspect, and the command exits non-zero when any
finding remains.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_PATH = REPO_ROOT / "data" / "scenarios_fixed.json"
LETTERS = ("A", "B", "C", "D", "E")
SEMANTIC_TYPES = {"factual", "causal", "framing"}
EXPECTED_VALIDITY = {
    "factual": "contradicted",
    "causal": "contradicted",
    "framing": "misaligned",
}

WEAK_OR_SELF_UNDERMINING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("likely", re.compile(r"\blikely\b", re.IGNORECASE)),
    ("probably", re.compile(r"\bprobably\b", re.IGNORECASE)),
    ("preliminary", re.compile(r"\bpreliminary\b", re.IGNORECASE)),
    ("inferred", re.compile(r"\binferred\b", re.IGNORECASE)),
    ("may not", re.compile(r"\bmay\s+not\b", re.IGNORECASE)),
    ("not confirmed", re.compile(r"\bnot\s+(?:yet\s+)?confirmed\b", re.IGNORECASE)),
    ("not verified", re.compile(r"\bnot\s+(?:yet\s+)?verified\b", re.IGNORECASE)),
    ("not checked", re.compile(r"\bnot\s+(?:yet\s+)?checked\b", re.IGNORECASE)),
)

# These deliberately require both an artifact and an attribution verb.  A bare
# mention of a public portal or booking task is not enough to claim fabricated
# private evidence.
FAKE_EVIDENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "artifact attribution",
        re.compile(
            r"\b(?:reservation|booking|email|message|memo|record|log|report|ticket|"
            r"database|portal)\s+(?:entry\s+)?(?:says|shows|states|indicates|confirms|"
            r"lists|records|reports)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "according to artifact",
        re.compile(
            r"\baccording\s+to\s+(?:an?\s+|the\s+)?(?:reservation|booking|email|"
            r"message|memo|record|log|report|ticket|database|portal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "artifact identifier",
        re.compile(
            r"\b(?:reservation|booking|ticket|record|confirmation)\s*"
            r"(?:number|no\.?|#|id)\s*"
            r"[A-Z0-9-]+",
            re.IGNORECASE,
        ),
    ),
    (
        "fabricated artifact noun",
        re.compile(
            r"\b(?:email|screenshot|manifest|private\s+log|confirmation\s+receipt)\b",
            re.IGNORECASE,
        ),
    ),
)

HARD_CODED_OPTION_PATTERN = re.compile(r"\bOption\s+[A-E]\b", re.IGNORECASE)

# A deterministic audit cannot prove semantic contradiction.  Requiring both
# target-language overlap and an explicit refutation marker catches accidental
# corrections about a different proposition while leaving the final logical
# judgment to the documented manual review.
CORRECTION_REFUTATION_PATTERN = re.compile(
    r"\b(?:no|not|zero|none|cannot|can't|doesn't|didn't|disabled|manual|denied|"
    r"unavailable|insufficient|fails?|exceeds?|outside|excludes?|remains?|"
    r"requires?|expired|late|only|but|instead|under\s+maintenance|occupies?|exits?)\b",
    re.IGNORECASE,
)

DOCUMENTARY_CONSTRAINT_PATTERN = re.compile(
    r"\b(?:confirm(?:ed|ation)?|verif(?:y|ied|ication)|document(?:ed|ation)|"
    r"record(?:ed|s)?|on\s+file)\b",
    re.IGNORECASE,
)

UNKNOWN_GAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:missing|omitted|omission|absent|unlisted|not\s+listed|no\s+(?:entry|"
        r"listing|status|detail|information|update))\b.{0,180}\b(?:unknown|unresolved|"
        r"undetermined|inconclusive|does\s+not\s+(?:mean|establish|indicate|show|imply)|"
        r"cannot\s+be\s+(?:assumed|determined|inferred|treated))\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:unknown|unresolved|undetermined|inconclusive)\b.{0,180}\b(?:missing|"
        r"omitted|omission|absent|unlisted|not\s+listed|no\s+(?:entry|listing|status|"
        r"detail|information|update))\b",
        re.IGNORECASE | re.DOTALL,
    ),
)

CORRECT_COMMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bOption\s+([A-E])\s+(?:is\s+the\s+correct|proves\s+optimal\s+execution|"
        r"is\s+the\s+fully\s+compliant,?\s+lowest-cost)\b",
        re.IGNORECASE,
    ),
)
TRAP_COMMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bOption\s+([A-E])\s+(?:isolates|captures)\s+(?:agents\s+experiencing\s+)?"
        r"assumption[- ]propagation\s+failure\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bOption\s+([A-E])\s+is\s+(?:the\s+)?(?:favou?red\s+distractor|trap)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bterminates\s+on\s+Option\s+([A-E])\b", re.IGNORECASE),
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
OPTION_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "book",
    "choose",
    "company",
    "cost",
    "day",
    "for",
    "from",
    "in",
    "is",
    "itinerary",
    "non",
    "nonpreferred",
    "of",
    "on",
    "option",
    "per",
    "preferred",
    "provider",
    "purchase",
    "rate",
    "select",
    "service",
    "the",
    "to",
    "use",
    "vendor",
    "via",
    "with",
}


@dataclass(frozen=True, order=True)
class AuditIssue:
    """One deterministic scenario-authoring failure."""

    location: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: [{self.code}] {self.message}"


def _issue(issues: list[AuditIssue], location: str, code: str, message: str) -> None:
    issues.append(AuditIssue(location=location, code=code, message=message))


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _comment_fields(record: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for key, value in record.items():
        if key.startswith("__comment") and isinstance(value, str):
            yield key, value


def _contains_explicit_unknown_rule(text: str) -> bool:
    return any(pattern.search(text) for pattern in UNKNOWN_GAP_PATTERNS)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
        if token not in OPTION_MATCH_STOPWORDS and len(token) > 1
    ]


def _comparison_bullets(context: str) -> list[str]:
    """Return dash-led comparison entries, excluding explanatory prose bullets."""
    return [line[2:].strip() for line in context.splitlines() if line.startswith("- ")]


def _match_option_to_bullet(option: str, bullets: Sequence[str]) -> int | None:
    """Map an option to a public-list position using distinctive shared tokens.

    Tokens repeated in every bullet contribute little, while names, model
    numbers, and entity codes contribute strongly.  A tie or weak match returns
    ``None`` so the audit never invents a position.
    """
    if not bullets:
        return None

    bullet_token_sets = [set(_tokens(bullet)) for bullet in bullets]
    option_tokens = set(_tokens(option))
    if not option_tokens:
        return None

    frequencies = Counter(token for tokens in bullet_token_sets for token in tokens)
    scores: list[float] = []
    for tokens in bullet_token_sets:
        shared = option_tokens & tokens
        score = sum(1.0 / frequencies[token] for token in shared)
        scores.append(score)

    best = max(scores)
    if best < 1.0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _check_comment_letters(
    issues: list[AuditIssue],
    location: str,
    comment: str,
    correct: str,
    favored: str,
) -> None:
    for pattern in CORRECT_COMMENT_PATTERNS:
        for match in pattern.finditer(comment):
            claimed = match.group(1).upper()
            if claimed != correct:
                _issue(
                    issues,
                    location,
                    "comment.stale_correct_letter",
                    f"comment calls Option {claimed} correct/optimal, but "
                    f"correct_answer is {correct}",
                )
    for pattern in TRAP_COMMENT_PATTERNS:
        for match in pattern.finditer(comment):
            claimed = match.group(1).upper()
            if claimed != favored:
                _issue(
                    issues,
                    location,
                    "comment.stale_favored_letter",
                    f"comment calls Option {claimed} the trap, but favored_option is {favored}",
                )


def audit_scenarios(data: object) -> list[AuditIssue]:
    """Return all deterministic findings for a decoded scenario JSON value."""
    issues: list[AuditIssue] = []
    if not isinstance(data, list):
        return [AuditIssue("$", "root.not_list", "top-level JSON value must be a list")]

    if len(data) != 20:
        _issue(
            issues,
            "$",
            "scenario.count",
            f"expected exactly 20 scenarios, found {len(data)}",
        )

    scenario_ids: list[str] = []
    assumption_ids: list[str] = []
    correct_letters: list[str] = []
    favored_letters: list[str] = []
    correct_favored_pairs: list[tuple[str, str]] = []
    public_positions: list[tuple[str, int, int]] = []
    semantic_type_counts: Counter[str] = Counter()
    semantic_word_counts: dict[str, list[int]] = {
        semantic_type: [] for semantic_type in SEMANTIC_TYPES
    }

    for scenario_index, raw_scenario in enumerate(data):
        base_location = f"$[{scenario_index}]"
        if not isinstance(raw_scenario, dict):
            _issue(issues, base_location, "scenario.not_object", "scenario must be an object")
            continue
        scenario = raw_scenario
        scenario_id_value = scenario.get("scenario_id")
        scenario_id = (
            scenario_id_value.strip()
            if isinstance(scenario_id_value, str) and scenario_id_value.strip()
            else f"index_{scenario_index}"
        )
        location = f"scenario[{scenario_id}]"
        if not _nonempty_string(scenario_id_value):
            _issue(issues, base_location, "scenario.id_missing", "scenario_id must be populated")
        else:
            scenario_ids.append(scenario_id)

        options = scenario.get("options")
        if not isinstance(options, dict) or set(options) != set(LETTERS):
            keys = sorted(options) if isinstance(options, dict) else []
            _issue(
                issues,
                f"{location}.options",
                "options.letters",
                f"expected exactly options A-E, found {keys}",
            )
            options = options if isinstance(options, dict) else {}
        for letter in LETTERS:
            if letter in options and not _nonempty_string(options[letter]):
                _issue(
                    issues,
                    f"{location}.options.{letter}",
                    "options.empty",
                    "option text must be populated",
                )

        correct = scenario.get("correct_answer")
        if correct not in LETTERS:
            _issue(
                issues,
                f"{location}.correct_answer",
                "answer.invalid",
                f"expected one of {LETTERS}, found {correct!r}",
            )
        else:
            correct_letters.append(correct)

        injected_level = scenario.get("injected_epistemic_level")
        if not _nonempty_string(injected_level) or "categorical" not in injected_level.lower():
            _issue(
                issues,
                f"{location}.injected_epistemic_level",
                "injection.not_categorical",
                "injected epistemic level must be populated and categorical",
            )

        context = scenario.get("common_context")
        context_text = context if isinstance(context, str) else ""
        if not _contains_explicit_unknown_rule(context_text):
            _issue(
                issues,
                f"{location}.common_context",
                "context.no_unknown_rule",
                "state explicitly that a missing/omitted status is unknown or unresolved, "
                "not evidence of failure",
            )

        constraints = scenario.get("hard_constraints")
        if not isinstance(constraints, list) or not constraints:
            _issue(
                issues,
                f"{location}.hard_constraints",
                "constraints.missing",
                "hard_constraints must be a non-empty list",
            )
            constraints = []
        for constraint_index, constraint in enumerate(constraints):
            if not isinstance(constraint, str):
                _issue(
                    issues,
                    f"{location}.hard_constraints[{constraint_index}]",
                    "constraint.not_string",
                    "constraint must be text",
                )
                continue
            match = DOCUMENTARY_CONSTRAINT_PATTERN.search(constraint)
            if match:
                _issue(
                    issues,
                    f"{location}.hard_constraints[{constraint_index}]",
                    "constraint.documentary_gate",
                    f"replace documentary-status criterion {match.group(0)!r} with the "
                    "substantive property the option must actually satisfy",
                )

        variants = scenario.get("assumption_variants")
        if not isinstance(variants, list):
            _issue(
                issues,
                f"{location}.assumption_variants",
                "variants.not_list",
                "assumption_variants must be a list",
            )
            variants = []
        if len(variants) != 3:
            _issue(
                issues,
                f"{location}.assumption_variants",
                "variants.count",
                f"expected exactly 3 variants, found {len(variants)}",
            )

        semantic_values = [
            variant.get("semantic_type")
            for variant in variants
            if isinstance(variant, dict)
        ]
        if Counter(semantic_values) != Counter(SEMANTIC_TYPES):
            _issue(
                issues,
                f"{location}.assumption_variants",
                "variants.semantic_set",
                "expected exactly one factual, one causal, and one framing variant; "
                f"found {semantic_values}",
            )

        scenario_favored: list[str] = []
        scenario_seed_lengths: dict[str, int] = {}
        for variant_index, raw_variant in enumerate(variants):
            variant_location = f"{location}.assumption_variants[{variant_index}]"
            if not isinstance(raw_variant, dict):
                _issue(
                    issues,
                    variant_location,
                    "variant.not_object",
                    "variant must be an object",
                )
                continue
            variant = raw_variant
            assumption_id_value = variant.get("assumption_id")
            assumption_id = (
                assumption_id_value.strip()
                if isinstance(assumption_id_value, str) and assumption_id_value.strip()
                else f"variant_{variant_index}"
            )
            variant_location = f"variant[{scenario_id}/{assumption_id}]"
            if not _nonempty_string(assumption_id_value):
                _issue(
                    issues,
                    f"{location}.assumption_variants[{variant_index}].assumption_id",
                    "variant.id_missing",
                    "assumption_id must be populated",
                )
            else:
                assumption_ids.append(assumption_id)

            semantic_type = variant.get("semantic_type")
            if semantic_type in SEMANTIC_TYPES:
                semantic_type_counts[semantic_type] += 1
            expected_validity = EXPECTED_VALIDITY.get(str(semantic_type))
            if expected_validity and variant.get("global_validity") != expected_validity:
                _issue(
                    issues,
                    f"{variant_location}.global_validity",
                    "variant.validity",
                    f"{semantic_type} must be {expected_validity!r}, found "
                    f"{variant.get('global_validity')!r}",
                )

            favored = variant.get("favored_option")
            if favored not in LETTERS:
                _issue(
                    issues,
                    f"{variant_location}.favored_option",
                    "favored.invalid",
                    f"expected one of {LETTERS}, found {favored!r}",
                )
            else:
                scenario_favored.append(favored)
                if favored == correct:
                    _issue(
                        issues,
                        f"{variant_location}.favored_option",
                        "favored.equals_correct",
                        "favored option must differ from correct_answer",
                    )

            for metadata_key in ("initial_provenance", "explicitness"):
                if not _nonempty_string(variant.get(metadata_key)):
                    _issue(
                        issues,
                        f"{variant_location}.{metadata_key}",
                        "metadata.empty",
                        "authoring metadata must be populated",
                    )
            if variant.get("initial_provenance") != "unsupported_private_assessment":
                _issue(
                    issues,
                    f"{variant_location}.initial_provenance",
                    "metadata.provenance_mismatch",
                    "initial_provenance must be 'unsupported_private_assessment'",
                )
            if variant.get("explicitness") != "explicit_categorical_assumption":
                _issue(
                    issues,
                    f"{variant_location}.explicitness",
                    "metadata.explicitness_mismatch",
                    "explicitness must be 'explicit_categorical_assumption'",
                )

            canonical = variant.get("canonical_assumption")
            if not _nonempty_string(canonical):
                _issue(
                    issues,
                    f"{variant_location}.canonical_assumption",
                    "canonical.empty",
                    "canonical_assumption must be populated",
                )

            if variant.get("seed_local_evidence"):
                _issue(
                    issues,
                    f"{variant_location}.seed_local_evidence",
                    "seed.legacy_evidence",
                    "use seed_private_assessment only; do not inject an evidence artifact",
                )
            assessment = variant.get("seed_private_assessment")
            if not isinstance(assessment, dict):
                _issue(
                    issues,
                    f"{variant_location}.seed_private_assessment",
                    "seed.assessment_missing",
                    "seed_private_assessment object is required",
                )
            else:
                content = assessment.get("content")
                content_text = content if isinstance(content, str) else ""
                if not content_text.startswith("Private assessment:"):
                    _issue(
                        issues,
                        f"{variant_location}.seed_private_assessment.content",
                        "seed.prefix",
                        "content must begin exactly with 'Private assessment:'",
                    )
                if HARD_CODED_OPTION_PATTERN.search(content_text):
                    _issue(
                        issues,
                        f"{variant_location}.seed_private_assessment.content",
                        "seed.hard_coded_option_letter",
                        "name the target entity instead of hard-coding an option letter",
                    )
                for label, pattern in WEAK_OR_SELF_UNDERMINING_PATTERNS:
                    if pattern.search(content_text):
                        _issue(
                            issues,
                            f"{variant_location}.seed_private_assessment.content",
                            "seed.weak_language",
                            f"remove weak or self-undermining phrase {label!r}",
                        )
                for label, pattern in FAKE_EVIDENCE_PATTERNS:
                    if pattern.search(content_text):
                        _issue(
                            issues,
                            f"{variant_location}.seed_private_assessment.content",
                            "seed.fake_artifact",
                            f"remove apparent fabricated private evidence ({label})",
                        )
                for metadata_key in ("provenance", "epistemic_marker", "verification_gap"):
                    if not _nonempty_string(assessment.get(metadata_key)):
                        _issue(
                            issues,
                            f"{variant_location}.seed_private_assessment.{metadata_key}",
                            "metadata.empty",
                            "seed-assessment metadata must be populated",
                        )
                marker = assessment.get("epistemic_marker")
                if marker != "categorical":
                    _issue(
                        issues,
                        f"{variant_location}.seed_private_assessment.epistemic_marker",
                        "seed.marker_not_categorical",
                        "epistemic_marker must be exactly 'categorical'",
                    )
                if assessment.get("provenance") != "private_assessment":
                    _issue(
                        issues,
                        f"{variant_location}.seed_private_assessment.provenance",
                        "seed.provenance_mismatch",
                        "assessment provenance must be exactly 'private_assessment'",
                    )
                if semantic_type in SEMANTIC_TYPES and content_text:
                    word_count = len(TOKEN_PATTERN.findall(content_text))
                    scenario_seed_lengths[semantic_type] = word_count
                    semantic_word_counts[semantic_type].append(word_count)
                if _nonempty_string(canonical) and content_text:
                    canonical_overlap = set(_tokens(canonical)) & set(_tokens(content_text))
                    if len(canonical_overlap) < 2:
                        _issue(
                            issues,
                            f"{variant_location}.seed_private_assessment.content",
                            "seed.canonical_target_mismatch",
                            "seed content has too little target-language overlap with its "
                            "canonical assumption",
                        )

            correction = variant.get("corrective_evidence")
            if not isinstance(correction, dict):
                _issue(
                    issues,
                    f"{variant_location}.corrective_evidence",
                    "correction.missing",
                    "corrective_evidence object is required",
                )
            else:
                if correction.get("available_round") != 2:
                    _issue(
                        issues,
                        f"{variant_location}.corrective_evidence.available_round",
                        "correction.release_round",
                        f"expected staged release in round 2, found "
                        f"{correction.get('available_round')!r}",
                    )
                for metadata_key in ("content", "recipient_role"):
                    if not _nonempty_string(correction.get(metadata_key)):
                        _issue(
                            issues,
                            f"{variant_location}.corrective_evidence.{metadata_key}",
                            "correction.empty",
                            "corrective-evidence field must be populated",
                        )
                if correction.get("recipient_role") != "corrector_agent":
                    _issue(
                        issues,
                        f"{variant_location}.corrective_evidence.recipient_role",
                        "correction.recipient",
                        "corrective evidence must be private to 'corrector_agent'",
                    )
                correction_text = correction.get("content")
                if _nonempty_string(canonical) and _nonempty_string(correction_text):
                    target_overlap = set(_tokens(canonical)) & set(_tokens(correction_text))
                    if not target_overlap:
                        _issue(
                            issues,
                            f"{variant_location}.corrective_evidence.content",
                            "correction.target_mismatch",
                            "correction has no target-language overlap with the canonical "
                            "assumption",
                        )
                    if not CORRECTION_REFUTATION_PATTERN.search(correction_text):
                        _issue(
                            issues,
                            f"{variant_location}.corrective_evidence.content",
                            "correction.no_refutation_marker",
                            "correction lacks an explicit lexical refutation of the injected "
                            "proposition",
                        )

            if isinstance(correct, str) and isinstance(favored, str):
                for comment_key, comment in _comment_fields(variant):
                    _check_comment_letters(
                        issues,
                        f"{variant_location}.{comment_key}",
                        comment,
                        correct,
                        favored,
                    )

        unique_favored = set(scenario_favored)
        if set(scenario_seed_lengths) == SEMANTIC_TYPES:
            shortest = min(scenario_seed_lengths.values())
            longest = max(scenario_seed_lengths.values())
            if longest - shortest > 10:
                _issue(
                    issues,
                    f"{location}.assumption_variants",
                    "treatment.length_imbalance",
                    "factual, causal, and framing seed assessments differ by more than "
                    f"10 words: {scenario_seed_lengths}",
                )
        if len(scenario_favored) == 3 and len(unique_favored) != 1:
            _issue(
                issues,
                f"{location}.assumption_variants",
                "favored.inconsistent",
                f"all variants must favor one option, found {scenario_favored}",
            )
        if len(unique_favored) == 1 and correct in LETTERS:
            favored = next(iter(unique_favored))
            favored_letters.append(favored)
            correct_favored_pairs.append((correct, favored))

            bullets = _comparison_bullets(context_text)
            correct_option = options.get(correct)
            favored_option = options.get(favored)
            correct_position = (
                _match_option_to_bullet(correct_option, bullets)
                if isinstance(correct_option, str)
                else None
            )
            favored_position = (
                _match_option_to_bullet(favored_option, bullets)
                if isinstance(favored_option, str)
                else None
            )
            if correct_position is None or favored_position is None:
                _issue(
                    issues,
                    f"{location}.common_context",
                    "public_list.position_unresolved",
                    "could not deterministically match both correct and trap options to "
                    "dash-led public comparison entries; align entity names in the option and list",
                )
            else:
                public_positions.append((scenario_id, correct_position, favored_position))

            for comment_key, comment in _comment_fields(scenario):
                _check_comment_letters(
                    issues,
                    f"{location}.{comment_key}",
                    comment,
                    correct,
                    favored,
                )

    for duplicate_id, count in sorted(Counter(scenario_ids).items()):
        if count > 1:
            _issue(
                issues,
                "$",
                "scenario.id_duplicate",
                f"scenario_id {duplicate_id!r} occurs {count} times",
            )
    for duplicate_id, count in sorted(Counter(assumption_ids).items()):
        if count > 1:
            _issue(
                issues,
                "$",
                "variant.id_duplicate",
                f"assumption_id {duplicate_id!r} occurs {count} times",
            )

    expected_balance = {letter: 4 for letter in LETTERS}
    expected_semantic_counts = {semantic_type: 20 for semantic_type in SEMANTIC_TYPES}
    if dict(semantic_type_counts) != expected_semantic_counts:
        _issue(
            issues,
            "$",
            "variants.semantic_balance",
            f"expected 20 variants of each semantic type; found "
            f"{dict(semantic_type_counts)}",
        )
    semantic_means = {
        semantic_type: sum(counts) / len(counts)
        for semantic_type, counts in semantic_word_counts.items()
        if counts
    }
    if len(semantic_means) == len(SEMANTIC_TYPES):
        if max(semantic_means.values()) - min(semantic_means.values()) > 3.0:
            _issue(
                issues,
                "$",
                "treatment.mean_length_imbalance",
                "mean seed-assessment word lengths differ by more than 3 words: "
                f"{semantic_means}",
            )
    correct_balance = {letter: correct_letters.count(letter) for letter in LETTERS}
    favored_balance = {letter: favored_letters.count(letter) for letter in LETTERS}
    if correct_balance != expected_balance:
        _issue(
            issues,
            "$",
            "balance.correct",
            f"correct answers must use each letter exactly four times; found {correct_balance}",
        )
    if favored_balance != expected_balance:
        _issue(
            issues,
            "$",
            "balance.favored",
            f"favored options must use each letter exactly four times; found {favored_balance}",
        )
    pair_counts = Counter(correct_favored_pairs)
    duplicates = sorted(pair for pair, count in pair_counts.items() if count > 1)
    if duplicates:
        _issue(
            issues,
            "$",
            "balance.pair_duplicate",
            f"all 20 (correct, favored) pairs must be unique; duplicates: {duplicates}",
        )

    if len(public_positions) >= 2:
        correct_positions = {correct_pos for _, correct_pos, _ in public_positions}
        favored_positions = {favored_pos for _, _, favored_pos in public_positions}
        position_pairs = {
            (correct_pos, favored_pos) for _, correct_pos, favored_pos in public_positions
        }
        if len(correct_positions) < 2 or len(favored_positions) < 2 or len(position_pairs) < 2:
            _issue(
                issues,
                "$",
                "public_list.position_constant",
                "correct and trap entities must not occupy globally constant public-list "
                f"positions; resolved positions: {public_positions}",
            )

    return sorted(issues)


def audit_file(path: Path) -> list[AuditIssue]:
    """Load and audit a JSON file, returning parse/read failures as findings."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [AuditIssue(str(path), "json.load", str(exc))]
    return audit_scenarios(data)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help=f"scenario JSON to audit (default: {DEFAULT_SCENARIO_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = args.path.resolve()
    issues = audit_file(path)
    if issues:
        print(f"Scenario audit FAILED: {len(issues)} finding(s) in {path}")
        print("Findings by check:")
        counts = Counter(finding.code for finding in issues)
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count:>3}  {code}")
        print("Detailed findings:")
        for finding in issues:
            print(f"- {finding}")
        return 1
    print(f"Scenario audit PASSED: 20 scenarios / 60 variants in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
