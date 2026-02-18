"""Privacy pack for PII detection, redaction, and safe exports.

This module provides comprehensive privacy protection for transcripts:

- PIIDetector: Detect personal identifiable information (PII) in text
- Redactor: Redact detected PII with various modes
- SafeExporter: Export transcripts with privacy-preserving transformations
- EncryptedStore: Optional AES-256-GCM encryption for stored files

All components are designed to be composable and work with the existing
Transcript/Segment model structures.

Example usage:
    from transcription.privacy import PIIDetector, Redactor, SafeExporter

    # Detect PII
    detector = PIIDetector()
    matches = detector.detect("Call me at john@example.com or 555-1234")

    # Redact transcript
    redactor = Redactor(mode="mask")
    redacted = redactor.redact_transcript(transcript)

    # Safe export
    exporter = SafeExporter(detector, redactor)
    exporter.export(transcript, mode="redacted", output_path=Path("output.json"))
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .exceptions import SlowerWhisperError
from .models import Segment, Transcript

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class PrivacyError(SlowerWhisperError):
    """Base error for privacy operations."""


class PIIDetectionError(PrivacyError):
    """Raised when PII detection fails."""


class RedactionError(PrivacyError):
    """Raised when redaction fails."""


class EncryptionError(PrivacyError):
    """Raised when encryption/decryption fails."""


# =============================================================================
# PII Types and Data Classes
# =============================================================================


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    NAME = "NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    IP_ADDRESS = "IP_ADDRESS"
    BANK_ACCOUNT = "BANK_ACCOUNT"


@dataclass
class PIIMatch:
    """A single PII detection match.

    Attributes:
        type: The type of PII detected.
        start: Start character index in the original text.
        end: End character index in the original text.
        value: The matched PII value.
        confidence: Confidence score (0.0-1.0) for the detection.
        source: Detection source ("regex" or "ner").
    """

    type: PIIType
    start: int
    end: int
    value: str
    confidence: float = 1.0
    source: Literal["regex", "ner"] = "regex"

    def to_dict(self) -> dict[str, Any]:
        """Serialize match to a JSON-serializable dict."""
        return {
            "type": self.type.value,
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PIIMatch:
        """Deserialize match from a dict."""
        return cls(
            type=PIIType(d["type"]),
            start=d["start"],
            end=d["end"],
            value=d["value"],
            confidence=d.get("confidence", 1.0),
            source=d.get("source", "regex"),
        )


@dataclass
class RedactedText:
    """Result of redacting text.

    Attributes:
        original: The original text.
        redacted: The redacted text.
        matches: List of PII matches that were redacted.
        redaction_map: Mapping of original values to redacted values.
    """

    original: str
    redacted: str
    matches: list[PIIMatch]
    redaction_map: dict[str, str] = field(default_factory=dict)


@dataclass
class RedactedTranscript:
    """Result of redacting a transcript.

    Attributes:
        transcript: The redacted Transcript object.
        report: The redaction report.
    """

    transcript: Transcript
    report: RedactionReport


@dataclass
class RedactionReport:
    """Report of redaction operations performed.

    Attributes:
        original_hash: SHA-256 hash of the original transcript text.
        redacted_hash: SHA-256 hash of the redacted transcript text.
        entities_found: Total number of PII entities found.
        entities_redacted: Total number of PII entities redacted.
        timestamp: ISO 8601 timestamp of when redaction was performed.
        entity_details: Per-entity details of what was redacted.
    """

    original_hash: str
    redacted_hash: str
    entities_found: int
    entities_redacted: int
    timestamp: str
    entity_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-serializable dict."""
        return {
            "original_hash": self.original_hash,
            "redacted_hash": self.redacted_hash,
            "entities_found": self.entities_found,
            "entities_redacted": self.entities_redacted,
            "timestamp": self.timestamp,
            "entity_details": self.entity_details,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RedactionReport:
        """Deserialize report from a dict."""
        return cls(
            original_hash=d["original_hash"],
            redacted_hash=d["redacted_hash"],
            entities_found=d["entities_found"],
            entities_redacted=d["entities_redacted"],
            timestamp=d["timestamp"],
            entity_details=d.get("entity_details", []),
        )


# =============================================================================
# PII Detector
# =============================================================================


class PIIDetector:
    """Detect PII in text using regex patterns and optional spaCy NER.

    The detector uses a combination of regex patterns for structured PII
    (emails, phone numbers, SSNs, etc.) and optionally spaCy NER for
    names and addresses.

    Example:
        detector = PIIDetector(entity_types={PIIType.EMAIL, PIIType.PHONE})
        matches = detector.detect("Contact: john@example.com or 555-1234")
    """

    # Regex patterns for various PII types
    # These patterns are designed to catch common formats with high precision
    _PATTERNS: dict[PIIType, list[re.Pattern[str]]] = {
        PIIType.EMAIL: [
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        ],
        PIIType.PHONE: [
            # US phone numbers with optional parentheses around area code
            re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            # US phone with +1 prefix
            re.compile(r"\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            # International format
            re.compile(r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
        ],
        PIIType.SSN: [
            # US Social Security Number
            re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        ],
        PIIType.CREDIT_CARD: [
            # Major credit card formats (Visa, MC, Amex, Discover)
            re.compile(
                r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
            ),
            # Amex format (15 digits)
            re.compile(r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b"),
        ],
        PIIType.DATE_OF_BIRTH: [
            # Common date formats that might be DOB
            # MM/DD/YYYY or MM-DD-YYYY
            re.compile(r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"),
            # YYYY-MM-DD (ISO format)
            re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"),
            # Written dates like "January 15, 1990" or "15 January 1990"
            re.compile(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
                r",?\s+(?:19|20)\d{2}\b",
                re.IGNORECASE,
            ),
        ],
        PIIType.IP_ADDRESS: [
            # IPv4
            re.compile(
                r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
            ),
        ],
        PIIType.BANK_ACCOUNT: [
            # Generic bank account patterns (US routing + account)
            re.compile(r"\b\d{9}[-\s]?\d{4,17}\b"),  # Routing number + account
        ],
        PIIType.ADDRESS: [
            # US street addresses
            re.compile(
                r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s?)+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl)\.?(?:\s*(?:#|Apt|Suite|Ste|Unit)\s*\d+[A-Za-z]?)?\b",
                re.IGNORECASE,
            ),
        ],
    }

    # Common name patterns (less reliable, lower confidence)
    _NAME_PATTERNS: list[re.Pattern[str]] = [
        # Salutations followed by names
        re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
        # "My name is X" pattern
        re.compile(
            r"\b(?:my name is|I am|I'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", re.IGNORECASE
        ),
    ]

    def __init__(
        self,
        entity_types: set[PIIType] | None = None,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        """Initialize PII detector.

        Args:
            entity_types: Set of PII types to detect. If None, detects all types.
            use_spacy: Whether to use spaCy NER for name/address detection.
            spacy_model: Name of the spaCy model to use (default: en_core_web_sm).
        """
        self.entity_types = entity_types or set(PIIType)
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self._nlp: Any = None  # Lazy-loaded spaCy model

    def _load_spacy(self) -> Any:
        """Lazy-load spaCy model."""
        if self._nlp is not None:
            return self._nlp

        if not self.use_spacy:
            return None

        try:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
            return self._nlp
        except ImportError:
            logger.warning(
                "spaCy not installed. Name/address NER will be limited to regex patterns."
            )
            self.use_spacy = False
            return None
        except OSError:
            logger.warning(
                f"spaCy model '{self.spacy_model}' not found. "
                f"Install with: python -m spacy download {self.spacy_model}"
            )
            self.use_spacy = False
            return None

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect PII in the given text.

        Args:
            text: The text to scan for PII.

        Returns:
            List of PIIMatch objects for all detected PII.
        """
        matches: list[PIIMatch] = []

        # Regex-based detection
        for pii_type, patterns in self._PATTERNS.items():
            if pii_type not in self.entity_types:
                continue

            for pattern in patterns:
                for match in pattern.finditer(text):
                    pii_match = PIIMatch(
                        type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        value=match.group(),
                        confidence=0.9,  # High confidence for regex matches
                        source="regex",
                    )
                    matches.append(pii_match)

        # Name detection via regex (lower confidence)
        if PIIType.NAME in self.entity_types:
            for pattern in self._NAME_PATTERNS:
                for match in pattern.finditer(text):
                    # Get the captured group (the actual name)
                    name = match.group(1) if match.groups() else match.group()
                    pii_match = PIIMatch(
                        type=PIIType.NAME,
                        start=match.start(1) if match.groups() else match.start(),
                        end=match.end(1) if match.groups() else match.end(),
                        value=name,
                        confidence=0.6,  # Lower confidence for pattern-based names
                        source="regex",
                    )
                    matches.append(pii_match)

        # spaCy NER-based detection
        nlp = self._load_spacy()
        if nlp is not None:
            doc = nlp(text)
            for ent in doc.ents:
                ner_pii_type: PIIType | None = None
                confidence = 0.8

                if ent.label_ == "PERSON" and PIIType.NAME in self.entity_types:
                    ner_pii_type = PIIType.NAME
                    confidence = 0.85
                elif ent.label_ in ("GPE", "LOC", "FAC") and PIIType.ADDRESS in self.entity_types:
                    # Geographic/Location entities (partial address info)
                    ner_pii_type = PIIType.ADDRESS
                    confidence = 0.7

                if ner_pii_type is not None:
                    pii_match = PIIMatch(
                        type=ner_pii_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        value=ent.text,
                        confidence=confidence,
                        source="ner",
                    )
                    matches.append(pii_match)

        # Remove duplicates (prefer higher confidence)
        matches = self._deduplicate_matches(matches)

        # Sort by position
        matches.sort(key=lambda m: (m.start, m.end))

        return matches

    def _deduplicate_matches(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove duplicate/overlapping matches, keeping highest confidence."""
        if not matches:
            return []

        # Sort by start position, then by length (longer first)
        sorted_matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))

        deduplicated: list[PIIMatch] = []
        for match in sorted_matches:
            # Check if this match overlaps with any existing match
            is_overlap = False
            for existing in deduplicated:
                if match.start < existing.end and match.end > existing.start:
                    # Overlap detected - keep the one with higher confidence
                    is_overlap = True
                    break

            if not is_overlap:
                deduplicated.append(match)

        return deduplicated


# =============================================================================
# Redactor
# =============================================================================


class RedactionMode(str, Enum):
    """Redaction modes."""

    MASK = "mask"  # Replace with [REDACTED:TYPE]
    HASH = "hash"  # Replace with deterministic hash
    PLACEHOLDER = "placeholder"  # Replace with type-appropriate placeholder


class Redactor:
    """Redact PII from text and transcripts.

    Supports multiple redaction modes:
    - mask: Replace with [REDACTED:TYPE] (e.g., [REDACTED:EMAIL])
    - hash: Replace with deterministic hash (same value -> same hash)
    - placeholder: Replace with type-appropriate placeholder (e.g., [EMAIL], john@example.com -> [EMAIL])

    Example:
        redactor = Redactor(mode="mask")
        result = redactor.redact("Email: john@example.com", matches)
        print(result.redacted)  # "Email: [REDACTED:EMAIL]"
    """

    # Placeholders for each PII type
    _PLACEHOLDERS: dict[PIIType, str] = {
        PIIType.NAME: "[NAME]",
        PIIType.EMAIL: "[EMAIL]",
        PIIType.PHONE: "[PHONE]",
        PIIType.SSN: "[SSN]",
        PIIType.CREDIT_CARD: "[CREDIT_CARD]",
        PIIType.ADDRESS: "[ADDRESS]",
        PIIType.DATE_OF_BIRTH: "[DOB]",
        PIIType.IP_ADDRESS: "[IP_ADDRESS]",
        PIIType.BANK_ACCOUNT: "[BANK_ACCOUNT]",
    }

    def __init__(
        self,
        mode: RedactionMode | str = RedactionMode.MASK,
        hash_salt: str | None = None,
        hash_length: int = 8,
    ):
        """Initialize redactor.

        Args:
            mode: Redaction mode (mask, hash, or placeholder).
            hash_salt: Salt for deterministic hashing (auto-generated if not provided).
            hash_length: Length of hash output (default: 8 characters).
        """
        self.mode = RedactionMode(mode) if isinstance(mode, str) else mode
        self.hash_salt = hash_salt or secrets.token_hex(16)
        self.hash_length = hash_length
        self._hash_cache: dict[str, str] = {}

    def _compute_hash(self, value: str, pii_type: PIIType) -> str:
        """Compute deterministic hash for a value."""
        cache_key = f"{pii_type.value}:{value}"
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]

        # Use HMAC-SHA256 for deterministic but secure hashing
        salted = f"{self.hash_salt}:{pii_type.value}:{value}"
        hash_bytes = hashlib.sha256(salted.encode()).digest()
        hash_str = base64.urlsafe_b64encode(hash_bytes).decode()[: self.hash_length]

        self._hash_cache[cache_key] = hash_str
        return hash_str

    def _get_replacement(self, match: PIIMatch) -> str:
        """Get the replacement string for a PII match."""
        if self.mode == RedactionMode.MASK:
            return f"[REDACTED:{match.type.value}]"
        elif self.mode == RedactionMode.HASH:
            hash_val = self._compute_hash(match.value, match.type)
            return f"[{match.type.value}:{hash_val}]"
        elif self.mode == RedactionMode.PLACEHOLDER:
            return self._PLACEHOLDERS.get(match.type, f"[{match.type.value}]")
        else:
            raise RedactionError(f"Unknown redaction mode: {self.mode}")

    def redact(self, text: str, matches: list[PIIMatch]) -> RedactedText:
        """Redact PII from text.

        Args:
            text: Original text to redact.
            matches: List of PII matches to redact.

        Returns:
            RedactedText with the redacted string and mapping.
        """
        if not matches:
            return RedactedText(original=text, redacted=text, matches=[], redaction_map={})

        # Sort matches by position (reverse order for safe replacement)
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        redacted = text
        redaction_map: dict[str, str] = {}

        for match in sorted_matches:
            replacement = self._get_replacement(match)
            redacted = redacted[: match.start] + replacement + redacted[match.end :]
            redaction_map[match.value] = replacement

        return RedactedText(
            original=text,
            redacted=redacted,
            matches=matches,
            redaction_map=redaction_map,
        )

    def redact_transcript(
        self,
        transcript: Transcript,
        detector: PIIDetector | None = None,
    ) -> RedactedTranscript:
        """Redact PII from a transcript.

        Args:
            transcript: The transcript to redact.
            detector: PIIDetector to use (creates default if not provided).

        Returns:
            RedactedTranscript with redacted transcript and report.
        """
        if detector is None:
            detector = PIIDetector()

        # Collect original text for hashing
        original_texts = [seg.text for seg in transcript.segments]
        original_hash = hashlib.sha256("\n".join(original_texts).encode()).hexdigest()

        # Detect and redact each segment
        all_matches: list[PIIMatch] = []
        redacted_segments: list[Segment] = []
        entity_details: list[dict[str, Any]] = []

        for seg in transcript.segments:
            matches = detector.detect(seg.text)
            all_matches.extend(matches)

            if matches:
                result = self.redact(seg.text, matches)
                redacted_text = result.redacted

                for match in matches:
                    entity_details.append(
                        {
                            "segment_id": seg.id,
                            "type": match.type.value,
                            "start": match.start,
                            "end": match.end,
                            "original_value": match.value,
                            "redacted_value": result.redaction_map.get(match.value, ""),
                            "confidence": match.confidence,
                            "source": match.source,
                        }
                    )
            else:
                redacted_text = seg.text

            # Create new segment with redacted text
            redacted_seg = Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=redacted_text,
                speaker=seg.speaker,
                tone=seg.tone,
                audio_state=seg.audio_state,
                words=seg.words,
            )
            redacted_segments.append(redacted_seg)

        # Create redacted transcript
        redacted_transcript = Transcript(
            file_name=transcript.file_name,
            language=transcript.language,
            segments=redacted_segments,
            meta=transcript.meta,
            annotations=transcript.annotations,
            speakers=transcript.speakers,
            turns=transcript.turns,
            speaker_stats=transcript.speaker_stats,
            chunks=transcript.chunks,
        )

        # Compute redacted hash
        redacted_texts = [seg.text for seg in redacted_segments]
        redacted_hash = hashlib.sha256("\n".join(redacted_texts).encode()).hexdigest()

        # Create report
        report = RedactionReport(
            original_hash=original_hash,
            redacted_hash=redacted_hash,
            entities_found=len(all_matches),
            entities_redacted=len(all_matches),
            timestamp=datetime.now(UTC).isoformat(),
            entity_details=entity_details,
        )

        return RedactedTranscript(transcript=redacted_transcript, report=report)


# =============================================================================
# Safe Exporter
# =============================================================================


class ExportMode(str, Enum):
    """Safe export modes."""

    RAW = "raw"  # No changes
    REDACTED = "redacted"  # PII redacted
    HASHED = "hashed"  # Speaker IDs + entities pseudonymized
    MINIMAL = "minimal"  # Text only, no metadata


class SafeExporter:
    """Export transcripts with privacy-preserving transformations.

    Supports multiple export modes:
    - raw: No changes to the transcript
    - redacted: PII redacted from text
    - hashed: Speaker IDs and PII pseudonymized with consistent hashes
    - minimal: Text only, no metadata (speaker, audio_state, etc.)

    Example:
        exporter = SafeExporter()
        exporter.export(transcript, mode="redacted", output_path=Path("output.json"))
    """

    def __init__(
        self,
        detector: PIIDetector | None = None,
        redactor: Redactor | None = None,
    ):
        """Initialize safe exporter.

        Args:
            detector: PIIDetector to use (creates default if not provided).
            redactor: Redactor to use (creates default if not provided).
        """
        self.detector = detector or PIIDetector()
        self.redactor = redactor or Redactor(mode=RedactionMode.MASK)

    def _hash_speaker_id(self, speaker_id: str, salt: str) -> str:
        """Hash a speaker ID for pseudonymization."""
        salted = f"{salt}:speaker:{speaker_id}"
        hash_bytes = hashlib.sha256(salted.encode()).digest()
        return f"spk_{base64.urlsafe_b64encode(hash_bytes).decode()[:8]}"

    def _export_raw(self, transcript: Transcript) -> dict[str, Any]:
        """Export transcript without modifications."""
        import tempfile

        from .writers import write_json

        # Use existing writer to get consistent format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            write_json(transcript, temp_path)
            with open(temp_path, encoding="utf-8") as f:
                result: dict[str, Any] = json.load(f)
                return result
        finally:
            temp_path.unlink(missing_ok=True)

    def _export_redacted(self, transcript: Transcript) -> tuple[dict[str, Any], RedactionReport]:
        """Export transcript with PII redacted."""
        result = self.redactor.redact_transcript(transcript, self.detector)
        data = self._export_raw(result.transcript)
        return data, result.report

    def _export_hashed(self, transcript: Transcript) -> tuple[dict[str, Any], RedactionReport]:
        """Export transcript with speaker IDs and PII pseudonymized."""
        # Use hash mode for redaction
        hash_redactor = Redactor(mode=RedactionMode.HASH, hash_salt=self.redactor.hash_salt)
        result = hash_redactor.redact_transcript(transcript, self.detector)

        # Also hash speaker IDs
        salt = hash_redactor.hash_salt
        speaker_map: dict[str, str] = {}

        for seg in result.transcript.segments:
            if seg.speaker is not None:
                original_id = (
                    seg.speaker.get("id") if isinstance(seg.speaker, dict) else str(seg.speaker)
                )
                if original_id and original_id not in speaker_map:
                    speaker_map[original_id] = self._hash_speaker_id(original_id, salt)

                if isinstance(seg.speaker, dict) and "id" in seg.speaker:
                    seg.speaker["id"] = speaker_map.get(seg.speaker["id"], seg.speaker["id"])

        # Update speakers list if present
        if result.transcript.speakers:
            for speaker in result.transcript.speakers:
                if isinstance(speaker, dict) and "id" in speaker:
                    original_id = speaker["id"]
                    if original_id in speaker_map:
                        speaker["id"] = speaker_map[original_id]

        # Update turns if present
        if result.transcript.turns:
            for turn in result.transcript.turns:
                if hasattr(turn, "speaker_id") and turn.speaker_id in speaker_map:
                    turn.speaker_id = speaker_map[turn.speaker_id]
                elif isinstance(turn, dict) and "speaker_id" in turn:
                    if turn["speaker_id"] in speaker_map:
                        turn["speaker_id"] = speaker_map[turn["speaker_id"]]

        data = self._export_raw(result.transcript)
        return data, result.report

    def _export_minimal(self, transcript: Transcript) -> dict[str, Any]:
        """Export transcript with text only, no metadata."""
        return {
            "file": transcript.file_name,
            "language": transcript.language,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcript.segments
            ],
        }

    def export(
        self,
        transcript: Transcript,
        mode: ExportMode | str,
        output_path: Path,
        include_report: bool = True,
    ) -> RedactionReport | None:
        """Export transcript with the specified privacy mode.

        Args:
            transcript: The transcript to export.
            mode: Export mode (raw, redacted, hashed, minimal).
            output_path: Path to write the exported JSON.
            include_report: Whether to include redaction report (default: True).

        Returns:
            RedactionReport if mode is redacted or hashed and include_report is True,
            otherwise None.
        """
        mode = ExportMode(mode) if isinstance(mode, str) else mode
        report: RedactionReport | None = None

        if mode == ExportMode.RAW:
            data = self._export_raw(transcript)
        elif mode == ExportMode.REDACTED:
            data, report = self._export_redacted(transcript)
        elif mode == ExportMode.HASHED:
            data, report = self._export_hashed(transcript)
        elif mode == ExportMode.MINIMAL:
            data = self._export_minimal(transcript)
        else:
            raise PrivacyError(f"Unknown export mode: {mode}")

        # Write main output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Write report if applicable
        if report is not None and include_report:
            report_path = output_path.with_suffix(".redaction_report.json")
            report_path.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return report


# =============================================================================
# Encrypted Store (Optional)
# =============================================================================


class EncryptedStore:
    """AES-256-GCM encryption for stored files.

    Provides encryption at rest for sensitive transcript files using
    AES-256-GCM with key derivation from password (PBKDF2).

    Example:
        store = EncryptedStore()
        store.encrypt_file(Path("transcript.json"), password="secret")  # pragma: allowlist secret
        store.decrypt_file(Path("transcript.json.enc"), password="secret")
    """

    # Encryption parameters
    SALT_LENGTH = 16
    NONCE_LENGTH = 12
    TAG_LENGTH = 16
    KEY_LENGTH = 32  # 256 bits
    PBKDF2_ITERATIONS = 100000

    # File extension for encrypted files
    ENCRYPTED_EXTENSION = ".enc"

    def __init__(self):
        """Initialize encrypted store."""
        self._crypto_available = self._check_crypto()

    def _check_crypto(self) -> bool:
        """Check if cryptography library is available."""
        try:
            from cryptography.hazmat.primitives import hashes  # noqa: F401
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa: F401

            return True
        except ImportError:
            return False

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if not self._crypto_available:
            raise EncryptionError(
                "cryptography library not installed. Install with: pip install cryptography"
            )

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_LENGTH,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
        )
        return kdf.derive(password.encode())

    def encrypt_file(self, path: Path, password: str, output_path: Path | None = None) -> Path:
        """Encrypt a file with AES-256-GCM.

        Args:
            path: Path to the file to encrypt.
            password: Password for encryption.
            output_path: Output path (default: original path + .enc).

        Returns:
            Path to the encrypted file.

        Raises:
            EncryptionError: If encryption fails.
        """
        if not self._crypto_available:
            raise EncryptionError(
                "cryptography library not installed. Install with: pip install cryptography"
            )

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if not path.exists():
            raise EncryptionError(f"File not found: {path}")

        # Generate salt and nonce
        salt = os.urandom(self.SALT_LENGTH)
        nonce = os.urandom(self.NONCE_LENGTH)

        # Derive key from password
        key = self._derive_key(password, salt)

        # Read plaintext
        plaintext = path.read_bytes()

        # Encrypt
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Write encrypted file: salt + nonce + ciphertext (includes tag)
        output = output_path or path.with_suffix(path.suffix + self.ENCRYPTED_EXTENSION)
        output.write_bytes(salt + nonce + ciphertext)

        return output

    def decrypt_file(self, path: Path, password: str, output_path: Path | None = None) -> Path:
        """Decrypt a file encrypted with AES-256-GCM.

        Args:
            path: Path to the encrypted file.
            password: Password for decryption.
            output_path: Output path (default: original path without .enc).

        Returns:
            Path to the decrypted file.

        Raises:
            EncryptionError: If decryption fails.
        """
        if not self._crypto_available:
            raise EncryptionError(
                "cryptography library not installed. Install with: pip install cryptography"
            )

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if not path.exists():
            raise EncryptionError(f"File not found: {path}")

        # Read encrypted file
        data = path.read_bytes()

        if len(data) < self.SALT_LENGTH + self.NONCE_LENGTH + self.TAG_LENGTH:
            raise EncryptionError("Invalid encrypted file format")

        # Extract components
        salt = data[: self.SALT_LENGTH]
        nonce = data[self.SALT_LENGTH : self.SALT_LENGTH + self.NONCE_LENGTH]
        ciphertext = data[self.SALT_LENGTH + self.NONCE_LENGTH :]

        # Derive key from password
        key = self._derive_key(password, salt)

        # Decrypt
        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            raise EncryptionError(f"Decryption failed (wrong password?): {e}") from e

        # Write decrypted file
        if output_path is None:
            # Remove .enc extension if present
            if path.suffix == self.ENCRYPTED_EXTENSION:
                output = path.with_suffix("")
            else:
                output = path.with_suffix(".decrypted" + path.suffix)
        else:
            output = output_path

        output.write_bytes(plaintext)

        return output


# =============================================================================
# CLI Integration
# =============================================================================


def build_privacy_parser(subparsers: Any) -> None:
    """Build the privacy subcommand parser.

    Args:
        subparsers: The subparsers object from argparse.
    """
    p_privacy = subparsers.add_parser(
        "privacy",
        help="Privacy tools for PII detection, redaction, and safe exports.",
    )

    privacy_subparsers = p_privacy.add_subparsers(dest="privacy_action", required=True)

    # =========================================================================
    # privacy detect
    # =========================================================================
    p_detect = privacy_subparsers.add_parser(
        "detect",
        help="Detect PII in a transcript.",
    )
    p_detect.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON file.",
    )
    p_detect.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in PIIType],
        default=None,
        help="PII types to detect (default: all).",
    )
    p_detect.add_argument(
        "--no-spacy",
        action="store_true",
        help="Disable spaCy NER (regex only).",
    )
    p_detect.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    # =========================================================================
    # privacy redact
    # =========================================================================
    p_redact = privacy_subparsers.add_parser(
        "redact",
        help="Redact PII from a transcript.",
    )
    p_redact.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON file.",
    )
    p_redact.add_argument(
        "--mode",
        choices=["mask", "hash", "placeholder"],
        default="mask",
        help="Redaction mode (default: mask).",
    )
    p_redact.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>_redacted.json).",
    )
    p_redact.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in PIIType],
        default=None,
        help="PII types to redact (default: all).",
    )
    p_redact.add_argument(
        "--no-report",
        action="store_true",
        help="Do not generate redaction report.",
    )

    # =========================================================================
    # privacy export
    # =========================================================================
    p_export = privacy_subparsers.add_parser(
        "export",
        help="Export transcript with privacy mode.",
    )
    p_export.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON file.",
    )
    p_export.add_argument(
        "--mode",
        choices=["raw", "redacted", "hashed", "minimal"],
        default="redacted",
        help="Export mode (default: redacted).",
    )
    p_export.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path.",
    )
    p_export.add_argument(
        "--no-report",
        action="store_true",
        help="Do not generate redaction report.",
    )

    # =========================================================================
    # privacy encrypt
    # =========================================================================
    p_encrypt = privacy_subparsers.add_parser(
        "encrypt",
        help="Encrypt a file with AES-256-GCM.",
    )
    p_encrypt.add_argument(
        "file",
        type=Path,
        help="Path to file to encrypt.",
    )
    p_encrypt.add_argument(
        "--password",
        required=True,
        help="Encryption password.",
    )
    p_encrypt.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>.enc).",
    )

    # =========================================================================
    # privacy decrypt
    # =========================================================================
    p_decrypt = privacy_subparsers.add_parser(
        "decrypt",
        help="Decrypt a file encrypted with AES-256-GCM.",
    )
    p_decrypt.add_argument(
        "file",
        type=Path,
        help="Path to encrypted file.",
    )
    p_decrypt.add_argument(
        "--password",
        required=True,
        help="Decryption password.",
    )
    p_decrypt.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input> without .enc).",
    )


def handle_privacy_command(args: Any) -> int:
    """Handle privacy subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """

    try:
        if args.privacy_action == "detect":
            return _handle_detect(args)
        elif args.privacy_action == "redact":
            return _handle_redact(args)
        elif args.privacy_action == "export":
            return _handle_export(args)
        elif args.privacy_action == "encrypt":
            return _handle_encrypt(args)
        elif args.privacy_action == "decrypt":
            return _handle_decrypt(args)
        else:
            print(f"Unknown privacy action: {args.privacy_action}", file=__import__("sys").stderr)
            return 1
    except PrivacyError as e:
        print(f"Privacy error: {e}", file=__import__("sys").stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=__import__("sys").stderr)
        return 1


def _handle_detect(args: Any) -> int:
    """Handle privacy detect command."""

    from .writers import load_transcript_from_json

    transcript = load_transcript_from_json(args.transcript)

    # Configure entity types
    entity_types = None
    if args.types:
        entity_types = {PIIType(t) for t in args.types}

    detector = PIIDetector(
        entity_types=entity_types,
        use_spacy=not args.no_spacy,
    )

    # Detect PII in all segments
    all_matches: list[dict[str, Any]] = []
    for seg in transcript.segments:
        matches = detector.detect(seg.text)
        for match in matches:
            all_matches.append(
                {
                    "segment_id": seg.id,
                    "segment_text": seg.text,
                    **match.to_dict(),
                }
            )

    if args.json:
        print(json.dumps(all_matches, indent=2))
    else:
        if not all_matches:
            print("No PII detected.")
        else:
            print(f"Found {len(all_matches)} PII instance(s):\n")
            for m in all_matches:
                print(
                    f'  Segment {m["segment_id"]}: {m["type"]} = "{m["value"]}" (confidence: {m["confidence"]:.2f})'
                )

    return 0


def _handle_redact(args: Any) -> int:
    """Handle privacy redact command."""
    from .writers import load_transcript_from_json, write_json

    transcript = load_transcript_from_json(args.transcript)

    # Configure entity types
    entity_types = None
    if args.types:
        entity_types = {PIIType(t) for t in args.types}

    detector = PIIDetector(entity_types=entity_types)
    redactor = Redactor(mode=args.mode)

    result = redactor.redact_transcript(transcript, detector)

    # Determine output path
    output_path = args.output
    if output_path is None:
        stem = args.transcript.stem
        output_path = args.transcript.parent / f"{stem}_redacted.json"

    # Write redacted transcript
    write_json(result.transcript, output_path)
    print(f"Redacted transcript written to: {output_path}")

    # Write report unless disabled
    if not args.no_report:
        report_path = output_path.with_suffix(".redaction_report.json")
        report_path.write_text(
            json.dumps(result.report.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"Redaction report written to: {report_path}")

    print(f"\nSummary: {result.report.entities_found} PII entities found and redacted.")

    return 0


def _handle_export(args: Any) -> int:
    """Handle privacy export command."""
    from .writers import load_transcript_from_json

    transcript = load_transcript_from_json(args.transcript)

    exporter = SafeExporter()
    report = exporter.export(
        transcript,
        mode=args.mode,
        output_path=args.output,
        include_report=not args.no_report,
    )

    print(f"Exported transcript to: {args.output}")
    if report is not None and not args.no_report:
        report_path = args.output.with_suffix(".redaction_report.json")
        print(f"Redaction report written to: {report_path}")
        print(f"\nSummary: {report.entities_found} PII entities processed.")

    return 0


def _handle_encrypt(args: Any) -> int:
    """Handle privacy encrypt command."""
    store = EncryptedStore()
    output = store.encrypt_file(args.file, args.password, args.output)
    print(f"Encrypted file written to: {output}")
    return 0


def _handle_decrypt(args: Any) -> int:
    """Handle privacy decrypt command."""
    store = EncryptedStore()
    output = store.decrypt_file(args.file, args.password, args.output)
    print(f"Decrypted file written to: {output}")
    return 0
