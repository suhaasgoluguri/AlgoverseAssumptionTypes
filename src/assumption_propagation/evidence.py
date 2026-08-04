"""Canonical corrective records for the evidence-scope experiment.

Protocol v2 keeps the evidence bytes fixed while varying how an agent obtains
them. A relay embeds the canonical payload in a peer message; direct access
forces a real model function call and returns those same bytes in a ``tool``
message. The record store is deterministic and recipient-scoped. Tool use is
forced by the orchestrator so the treatment does not also measure whether an
agent decides to seek verification. Legacy prompt-delivery helpers continue to
consume the same record type for backward-compatible analyses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


AUTHORITATIVE_RECORD_SOURCE = "Authoritative case-record service"
PLACEBO_RECORD_STATUS = "no_additional_record"


def sha256_text(text: str) -> str:
    """Full, stable SHA-256 digest for prompt and record provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Full, stable SHA-256 digest for already-encoded protocol payloads."""
    return hashlib.sha256(value).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a protocol payload as sorted, minified, Unicode JSON.

    The UTF-8 encoding of this string is the wire identity used for both a
    relayed record and a genuine tool result.  In particular, ``ensure_ascii``
    stays false so hashing cannot depend on whether a producer escaped Unicode.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class CorrectiveRecord:
    """Canonical record shared by relay and direct-delivery conditions."""

    record_id: str
    source: str
    content: str
    sha256: str

    def as_payload(self) -> dict[str, str]:
        """Return the complete, transport-independent record payload."""
        return {
            "record_id": self.record_id,
            "source": self.source,
            "content": self.content,
            "sha256": self.sha256,
        }

    def canonical_json(self) -> str:
        """Canonical JSON used without alteration by relay and tool routes."""
        return _canonical_json(self.as_payload())

    def canonical_json_bytes(self) -> bytes:
        """Canonical UTF-8 wire bytes used when hashing a record result."""
        return self.canonical_json().encode("utf-8")

    def canonical_json_sha256(self) -> str:
        """Digest of the complete canonical wire payload, not just its content."""
        return sha256_bytes(self.canonical_json_bytes())

    @classmethod
    def from_variant(cls, variant) -> "CorrectiveRecord":
        content = variant.corrective_evidence.content
        digest = sha256_text(content)
        return cls(
            record_id=f"REC-{digest[:12].upper()}",
            source=AUTHORITATIVE_RECORD_SOURCE,
            content=content,
            sha256=digest,
        )


@dataclass(frozen=True)
class DirectRecordReceipt:
    """Auditable receipt returned for one direct record-access event."""

    record: CorrectiveRecord
    recipient_agent_id: int
    round_index: int


class CorrectiveRecordStore:
    """A one-record, read-only service scoped to the active assumption.

    The store deliberately exposes only the designated corrective record.  It
    cannot reveal the other candidates' private briefs, so using it does not
    collapse the hidden-profile pooling task.
    """

    def __init__(self, record: CorrectiveRecord) -> None:
        self._record = record

    @classmethod
    def from_variant(cls, variant) -> "CorrectiveRecordStore":
        return cls(CorrectiveRecord.from_variant(variant))

    @property
    def record(self) -> CorrectiveRecord:
        """The corrector's canonical source record for a verbatim relay."""
        return self._record

    def retrieve(
        self, record_id: str, *, recipient_agent_id: int, round_index: int
    ) -> DirectRecordReceipt:
        """Retrieve the sole authorized record or fail closed."""
        if record_id != self._record.record_id:
            raise KeyError(f"record {record_id!r} is not available in this case")
        if recipient_agent_id not in range(4):
            raise ValueError("recipient_agent_id must be between 0 and 3")
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        return DirectRecordReceipt(
            record=self._record,
            recipient_agent_id=recipient_agent_id,
            round_index=round_index,
        )

    def retrieve_canonical_json(
        self, record_id: str, *, recipient_agent_id: int, round_index: int
    ) -> str:
        """Return the exact canonical JSON that a genuine tool call exposes.

        Callers relaying a record must use ``record.canonical_json()``; callers
        executing the bounded tool use this method.  Both therefore receive
        byte-identical serialized evidence while the route remains distinct.
        """
        receipt = self.retrieve(
            record_id,
            recipient_agent_id=recipient_agent_id,
            round_index=round_index,
        )
        return receipt.record.canonical_json()


def canonical_placebo_payload() -> dict[str, object]:
    """Non-diagnostic result for an extra-tool-turn placebo condition.

    The keys deliberately overlap the corrective-record result, but the record
    fields are null and the explicit status says that no additional record was
    available.  It contains neither the correction nor a candidate-specific
    hint.
    """
    return {
        "record_id": None,
        "source": AUTHORITATIVE_RECORD_SOURCE,
        "content": None,
        "sha256": None,
        "status": PLACEBO_RECORD_STATUS,
    }


def canonical_placebo_json() -> str:
    """Stable, minified JSON returned by the placebo tool."""
    return _canonical_json(canonical_placebo_payload())


def canonical_placebo_json_bytes() -> bytes:
    """Stable UTF-8 bytes returned by the placebo tool."""
    return canonical_placebo_json().encode("utf-8")


def canonical_placebo_sha256() -> str:
    """Digest of the complete canonical placebo result."""
    return sha256_bytes(canonical_placebo_json_bytes())
