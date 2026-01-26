"""Tests for privacy pack (PII detection, redaction, and safe exports)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from transcription.models import Segment, Transcript
from transcription.privacy import (
    EncryptedStore,
    EncryptionError,
    ExportMode,
    PIIDetector,
    PIIMatch,
    PIIType,
    RedactionMode,
    RedactionReport,
    Redactor,
    SafeExporter,
)
from transcription.writers import load_transcript_from_json, write_json


class TestPIIDetector:
    """Test PIIDetector class."""

    def test_detect_email(self) -> None:
        """Detect email addresses."""
        detector = PIIDetector(entity_types={PIIType.EMAIL}, use_spacy=False)
        matches = detector.detect("Contact me at john.doe@example.com for more info.")

        assert len(matches) == 1
        assert matches[0].type == PIIType.EMAIL
        assert matches[0].value == "john.doe@example.com"
        assert matches[0].confidence >= 0.8

    def test_detect_multiple_emails(self) -> None:
        """Detect multiple email addresses."""
        detector = PIIDetector(entity_types={PIIType.EMAIL}, use_spacy=False)
        matches = detector.detect("Email john@example.com or jane@test.org")

        assert len(matches) == 2
        values = {m.value for m in matches}
        assert "john@example.com" in values
        assert "jane@test.org" in values

    def test_detect_phone_us_format(self) -> None:
        """Detect US phone numbers."""
        detector = PIIDetector(entity_types={PIIType.PHONE}, use_spacy=False)
        text = "Call me at (555) 123-4567 or 555.987.6543"
        matches = detector.detect(text)

        assert len(matches) >= 2
        # Verify both formats were detected (may include parentheses or not)
        combined_text = " ".join(m.value for m in matches)
        assert "123-4567" in combined_text
        assert "987.6543" in combined_text

    def test_detect_phone_international(self) -> None:
        """Detect international phone numbers."""
        detector = PIIDetector(entity_types={PIIType.PHONE}, use_spacy=False)
        matches = detector.detect("International: +1-555-123-4567")

        assert len(matches) >= 1
        # Should detect the phone number (with or without +1 prefix)
        combined_text = " ".join(m.value for m in matches)
        assert "555-123-4567" in combined_text or "+1" in combined_text

    def test_detect_ssn(self) -> None:
        """Detect Social Security Numbers."""
        detector = PIIDetector(entity_types={PIIType.SSN}, use_spacy=False)
        matches = detector.detect("SSN: 123-45-6789")

        assert len(matches) == 1
        assert matches[0].type == PIIType.SSN
        assert matches[0].value == "123-45-6789"

    def test_detect_ssn_without_dashes(self) -> None:
        """Detect SSN without dashes."""
        detector = PIIDetector(entity_types={PIIType.SSN}, use_spacy=False)
        matches = detector.detect("SSN: 123 45 6789")

        assert len(matches) == 1
        assert matches[0].value == "123 45 6789"

    def test_detect_credit_card_visa(self) -> None:
        """Detect Visa credit card numbers."""
        detector = PIIDetector(entity_types={PIIType.CREDIT_CARD}, use_spacy=False)
        matches = detector.detect("Card: 4111-1111-1111-1111")

        assert len(matches) == 1
        assert matches[0].type == PIIType.CREDIT_CARD
        assert "4111" in matches[0].value

    def test_detect_date_of_birth_formats(self) -> None:
        """Detect various date of birth formats."""
        detector = PIIDetector(entity_types={PIIType.DATE_OF_BIRTH}, use_spacy=False)

        # MM/DD/YYYY format
        matches = detector.detect("DOB: 01/15/1990")
        assert len(matches) == 1
        assert matches[0].value == "01/15/1990"

        # ISO format
        matches = detector.detect("Born: 1990-01-15")
        assert len(matches) == 1
        assert matches[0].value == "1990-01-15"

        # Written format
        matches = detector.detect("Birthday: January 15, 1990")
        assert len(matches) == 1
        assert "January" in matches[0].value

    def test_detect_ip_address(self) -> None:
        """Detect IP addresses."""
        detector = PIIDetector(entity_types={PIIType.IP_ADDRESS}, use_spacy=False)
        matches = detector.detect("Server IP: 192.168.1.100")

        assert len(matches) == 1
        assert matches[0].type == PIIType.IP_ADDRESS
        assert matches[0].value == "192.168.1.100"

    def test_detect_address_street(self) -> None:
        """Detect street addresses."""
        detector = PIIDetector(entity_types={PIIType.ADDRESS}, use_spacy=False)
        matches = detector.detect("Address: 123 Main Street")

        assert len(matches) == 1
        assert matches[0].type == PIIType.ADDRESS
        assert "123 Main Street" in matches[0].value

    def test_detect_name_pattern(self) -> None:
        """Detect names using regex patterns."""
        detector = PIIDetector(entity_types={PIIType.NAME}, use_spacy=False)
        matches = detector.detect("My name is John Smith")

        assert len(matches) >= 1
        assert any(m.type == PIIType.NAME for m in matches)
        assert any("John Smith" in m.value for m in matches)

    def test_detect_name_with_salutation(self) -> None:
        """Detect names with salutations."""
        detector = PIIDetector(entity_types={PIIType.NAME}, use_spacy=False)
        matches = detector.detect("Please contact Mr. John Doe")

        assert len(matches) >= 1
        assert any("John Doe" in m.value for m in matches)

    def test_no_false_positives_on_clean_text(self) -> None:
        """Clean text should not trigger PII detection."""
        detector = PIIDetector(use_spacy=False)
        matches = detector.detect("The weather is nice today.")

        # Should be empty or very few low-confidence matches
        high_confidence = [m for m in matches if m.confidence >= 0.8]
        assert len(high_confidence) == 0

    def test_entity_type_filtering(self) -> None:
        """Only detect specified entity types."""
        text = "Email: john@example.com, Phone: 555-123-4567"

        # Only detect emails
        detector = PIIDetector(entity_types={PIIType.EMAIL}, use_spacy=False)
        matches = detector.detect(text)
        assert all(m.type == PIIType.EMAIL for m in matches)

        # Only detect phones
        detector = PIIDetector(entity_types={PIIType.PHONE}, use_spacy=False)
        matches = detector.detect(text)
        assert all(m.type == PIIType.PHONE for m in matches)

    def test_detect_all_types_by_default(self) -> None:
        """Detect all PII types when not specified."""
        detector = PIIDetector(use_spacy=False)
        text = "Email: john@example.com, SSN: 123-45-6789"
        matches = detector.detect(text)

        types_found = {m.type for m in matches}
        assert PIIType.EMAIL in types_found
        assert PIIType.SSN in types_found

    def test_pii_match_serialization(self) -> None:
        """PIIMatch can be serialized to dict and back."""
        match = PIIMatch(
            type=PIIType.EMAIL,
            start=0,
            end=20,
            value="test@example.com",
            confidence=0.95,
            source="regex",
        )

        d = match.to_dict()
        assert d["type"] == "EMAIL"
        assert d["value"] == "test@example.com"

        restored = PIIMatch.from_dict(d)
        assert restored.type == match.type
        assert restored.value == match.value
        assert restored.confidence == match.confidence


class TestRedactor:
    """Test Redactor class."""

    def test_mask_mode(self) -> None:
        """Test mask redaction mode."""
        detector = PIIDetector(entity_types={PIIType.EMAIL}, use_spacy=False)
        redactor = Redactor(mode=RedactionMode.MASK)

        text = "Email me at john@example.com"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)

        assert "[REDACTED:EMAIL]" in result.redacted
        assert "john@example.com" not in result.redacted

    def test_hash_mode(self) -> None:
        """Test hash redaction mode."""
        redactor = Redactor(mode=RedactionMode.HASH, hash_salt="test_salt")

        matches = [
            PIIMatch(type=PIIType.EMAIL, start=12, end=28, value="john@example.com"),
        ]
        result = redactor.redact("Email me at john@example.com", matches)

        assert "[EMAIL:" in result.redacted
        assert "john@example.com" not in result.redacted

    def test_hash_mode_deterministic(self) -> None:
        """Same value produces same hash."""
        redactor = Redactor(mode=RedactionMode.HASH, hash_salt="test_salt")

        matches1 = [PIIMatch(type=PIIType.EMAIL, start=0, end=16, value="john@example.com")]
        matches2 = [PIIMatch(type=PIIType.EMAIL, start=0, end=16, value="john@example.com")]

        result1 = redactor.redact("john@example.com", matches1)
        result2 = redactor.redact("john@example.com", matches2)

        assert result1.redacted == result2.redacted

    def test_placeholder_mode(self) -> None:
        """Test placeholder redaction mode."""
        redactor = Redactor(mode=RedactionMode.PLACEHOLDER)

        matches = [
            PIIMatch(type=PIIType.EMAIL, start=12, end=28, value="john@example.com"),
        ]
        result = redactor.redact("Email me at john@example.com", matches)

        assert result.redacted == "Email me at [EMAIL]"

    def test_multiple_matches(self) -> None:
        """Redact multiple PII instances."""
        redactor = Redactor(mode=RedactionMode.MASK)

        matches = [
            PIIMatch(type=PIIType.EMAIL, start=7, end=23, value="john@example.com"),
            PIIMatch(type=PIIType.PHONE, start=40, end=52, value="555-123-4567"),
        ]
        text = "Email: john@example.com, Phone number: 555-123-4567"
        result = redactor.redact(text, matches)

        assert "[REDACTED:EMAIL]" in result.redacted
        assert "[REDACTED:PHONE]" in result.redacted
        assert "john@example.com" not in result.redacted
        assert "555-123-4567" not in result.redacted

    def test_empty_matches(self) -> None:
        """No matches returns original text."""
        redactor = Redactor(mode=RedactionMode.MASK)
        text = "No PII here"
        result = redactor.redact(text, [])

        assert result.redacted == text
        assert result.original == text
        assert len(result.matches) == 0

    def test_redact_transcript(self) -> None:
        """Redact PII from transcript."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="My email is john@example.com"),
                Segment(id=1, start=2.0, end=4.0, text="Call me at 555-123-4567"),
            ],
        )

        detector = PIIDetector(use_spacy=False)
        redactor = Redactor(mode=RedactionMode.MASK)
        result = redactor.redact_transcript(transcript, detector)

        # Check redacted transcript
        assert "[REDACTED:EMAIL]" in result.transcript.segments[0].text
        assert "[REDACTED:PHONE]" in result.transcript.segments[1].text

        # Check report
        assert result.report.entities_found >= 2
        assert result.report.entities_redacted >= 2
        assert len(result.report.entity_details) >= 2

    def test_redaction_report_serialization(self) -> None:
        """RedactionReport can be serialized to dict and back."""
        report = RedactionReport(
            original_hash="abc123",
            redacted_hash="def456",
            entities_found=5,
            entities_redacted=5,
            timestamp="2024-01-01T00:00:00Z",
            entity_details=[{"type": "EMAIL", "value": "test@example.com"}],
        )

        d = report.to_dict()
        assert d["entities_found"] == 5

        restored = RedactionReport.from_dict(d)
        assert restored.entities_found == report.entities_found
        assert restored.original_hash == report.original_hash


class TestSafeExporter:
    """Test SafeExporter class."""

    def _create_test_transcript(self) -> Transcript:
        """Create a test transcript with PII."""
        return Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=2.0,
                    text="My email is john@example.com",
                    speaker={"id": "spk_0", "confidence": 0.9},
                ),
                Segment(
                    id=1,
                    start=2.0,
                    end=4.0,
                    text="My SSN is 123-45-6789",
                    speaker={"id": "spk_1", "confidence": 0.85},
                ),
            ],
            speakers=[
                {"id": "spk_0", "label": "Speaker 1"},
                {"id": "spk_1", "label": "Speaker 2"},
            ],
        )

    def test_export_raw(self) -> None:
        """Export raw mode preserves all data."""
        transcript = self._create_test_transcript()
        exporter = SafeExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.json"
            report = exporter.export(transcript, ExportMode.RAW, output)

            assert output.exists()
            assert report is None  # No report for raw mode

            data = json.loads(output.read_text())
            assert "john@example.com" in data["segments"][0]["text"]
            assert "123-45-6789" in data["segments"][1]["text"]

    def test_export_redacted(self) -> None:
        """Export redacted mode removes PII."""
        transcript = self._create_test_transcript()
        exporter = SafeExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.json"
            report = exporter.export(transcript, ExportMode.REDACTED, output)

            assert output.exists()
            assert report is not None

            data = json.loads(output.read_text())
            assert "john@example.com" not in data["segments"][0]["text"]
            assert "123-45-6789" not in data["segments"][1]["text"]
            assert "[REDACTED:" in data["segments"][0]["text"]

    def test_export_hashed(self) -> None:
        """Export hashed mode pseudonymizes speaker IDs and PII."""
        transcript = self._create_test_transcript()
        exporter = SafeExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.json"
            report = exporter.export(transcript, ExportMode.HASHED, output)

            assert output.exists()
            assert report is not None

            data = json.loads(output.read_text())
            # PII should be hashed
            assert "john@example.com" not in data["segments"][0]["text"]
            assert "[EMAIL:" in data["segments"][0]["text"]

    def test_export_minimal(self) -> None:
        """Export minimal mode removes all metadata."""
        transcript = self._create_test_transcript()
        exporter = SafeExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.json"
            report = exporter.export(transcript, ExportMode.MINIMAL, output)

            assert output.exists()
            assert report is None  # No report for minimal mode

            data = json.loads(output.read_text())
            # Should only have basic fields
            assert "speakers" not in data
            assert "meta" not in data
            # Segments should only have id, start, end, text
            seg = data["segments"][0]
            assert "speaker" not in seg
            assert "audio_state" not in seg

    def test_export_creates_report_file(self) -> None:
        """Export creates redaction report file."""
        transcript = self._create_test_transcript()
        exporter = SafeExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.json"
            exporter.export(transcript, ExportMode.REDACTED, output, include_report=True)

            report_path = output.with_suffix(".redaction_report.json")
            assert report_path.exists()

            report_data = json.loads(report_path.read_text())
            assert "original_hash" in report_data
            assert "entities_found" in report_data


class TestEncryptedStore:
    """Test EncryptedStore class."""

    @pytest.fixture
    def temp_file(self, tmp_path: Path) -> Path:
        """Create a temporary file for testing."""
        file_path = tmp_path / "test_data.json"
        file_path.write_text('{"key": "sensitive data"}')
        return file_path

    def test_encrypt_decrypt_roundtrip(self, temp_file: Path) -> None:
        """Encrypt and decrypt produces original content."""
        pytest.importorskip("cryptography")

        store = EncryptedStore()
        password = "test_password_123"

        # Read original content
        original_content = temp_file.read_bytes()

        # Encrypt
        encrypted_path = store.encrypt_file(temp_file, password)
        assert encrypted_path.exists()
        assert encrypted_path != temp_file

        # Encrypted content should be different
        encrypted_content = encrypted_path.read_bytes()
        assert encrypted_content != original_content

        # Decrypt
        decrypted_path = store.decrypt_file(encrypted_path, password)
        assert decrypted_path.exists()

        # Decrypted content should match original
        decrypted_content = decrypted_path.read_bytes()
        assert decrypted_content == original_content

    def test_wrong_password_fails(self, temp_file: Path) -> None:
        """Decryption with wrong password fails."""
        pytest.importorskip("cryptography")

        store = EncryptedStore()

        encrypted_path = store.encrypt_file(temp_file, "correct_password")

        with pytest.raises(EncryptionError, match="Decryption failed"):
            store.decrypt_file(encrypted_path, "wrong_password")

    def test_custom_output_path(self, temp_file: Path, tmp_path: Path) -> None:
        """Encryption with custom output path."""
        pytest.importorskip("cryptography")

        store = EncryptedStore()
        custom_output = tmp_path / "custom_encrypted.bin"

        encrypted_path = store.encrypt_file(temp_file, "password", output_path=custom_output)
        assert encrypted_path == custom_output
        assert encrypted_path.exists()

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Encrypting non-existent file raises error."""
        pytest.importorskip("cryptography")

        store = EncryptedStore()
        missing_file = tmp_path / "nonexistent.json"

        with pytest.raises(EncryptionError, match="File not found"):
            store.encrypt_file(missing_file, "password")

    def test_invalid_encrypted_file(self, tmp_path: Path) -> None:
        """Decrypting invalid file raises error."""
        pytest.importorskip("cryptography")

        store = EncryptedStore()
        invalid_file = tmp_path / "invalid.enc"
        invalid_file.write_bytes(b"too short")

        with pytest.raises(EncryptionError, match="Invalid encrypted file"):
            store.decrypt_file(invalid_file, "password")


class TestConsistentHashing:
    """Test that hashing is consistent and useful for linking."""

    def test_same_email_same_hash(self) -> None:
        """Same email produces same hash for linking."""
        redactor = Redactor(mode=RedactionMode.HASH, hash_salt="fixed_salt")

        email = "recurring@example.com"
        match = PIIMatch(type=PIIType.EMAIL, start=0, end=len(email), value=email)

        result1 = redactor.redact(email, [match])
        result2 = redactor.redact(email, [match])

        assert result1.redacted == result2.redacted

    def test_different_emails_different_hashes(self) -> None:
        """Different emails produce different hashes."""
        redactor = Redactor(mode=RedactionMode.HASH, hash_salt="fixed_salt")

        match1 = PIIMatch(type=PIIType.EMAIL, start=0, end=15, value="user1@test.com")
        match2 = PIIMatch(type=PIIType.EMAIL, start=0, end=15, value="user2@test.com")

        result1 = redactor.redact("user1@test.com", [match1])
        result2 = redactor.redact("user2@test.com", [match2])

        assert result1.redacted != result2.redacted

    def test_different_salts_different_hashes(self) -> None:
        """Different salts produce different hashes."""
        email = "same@example.com"
        match = PIIMatch(type=PIIType.EMAIL, start=0, end=len(email), value=email)

        redactor1 = Redactor(mode=RedactionMode.HASH, hash_salt="salt1")
        redactor2 = Redactor(mode=RedactionMode.HASH, hash_salt="salt2")

        result1 = redactor1.redact(email, [match])
        result2 = redactor2.redact(email, [match])

        assert result1.redacted != result2.redacted


class TestTranscriptRoundtrip:
    """Test that redacted transcripts survive JSON roundtrip."""

    def test_redacted_transcript_json_roundtrip(self) -> None:
        """Redacted transcript can be written and read back."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="Email: john@example.com"),
            ],
        )

        detector = PIIDetector(use_spacy=False)
        redactor = Redactor(mode=RedactionMode.MASK)
        result = redactor.redact_transcript(transcript, detector)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            write_json(result.transcript, json_path)
            loaded = load_transcript_from_json(json_path)

            assert "[REDACTED:EMAIL]" in loaded.segments[0].text
            assert "john@example.com" not in loaded.segments[0].text
        finally:
            json_path.unlink(missing_ok=True)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_transcript(self) -> None:
        """Handle empty transcript gracefully."""
        transcript = Transcript(file_name="empty.wav", language="en", segments=[])

        detector = PIIDetector(use_spacy=False)
        redactor = Redactor(mode=RedactionMode.MASK)
        result = redactor.redact_transcript(transcript, detector)

        assert len(result.transcript.segments) == 0
        assert result.report.entities_found == 0

    def test_segment_with_no_pii(self) -> None:
        """Segment without PII passes through unchanged."""
        transcript = Transcript(
            file_name="clean.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="The weather is nice today."),
            ],
        )

        detector = PIIDetector(use_spacy=False)
        redactor = Redactor(mode=RedactionMode.MASK)
        result = redactor.redact_transcript(transcript, detector)

        assert result.transcript.segments[0].text == "The weather is nice today."

    def test_overlapping_matches(self) -> None:
        """Handle overlapping PII matches."""
        detector = PIIDetector(use_spacy=False)
        # Text that might produce overlapping matches
        text = "Call 123-45-6789 or 123 45 6789"
        matches = detector.detect(text)

        # Should deduplicate overlapping matches
        for i, m1 in enumerate(matches):
            for m2 in matches[i + 1 :]:
                # No two matches should overlap
                assert m1.end <= m2.start or m2.end <= m1.start

    def test_unicode_text(self) -> None:
        """Handle Unicode text properly."""
        detector = PIIDetector(entity_types={PIIType.EMAIL}, use_spacy=False)
        text = "Email: john@example.com, Unicode: Hello"
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].value == "john@example.com"

    def test_redaction_preserves_segment_metadata(self) -> None:
        """Redaction preserves other segment metadata."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=1.5,
                    end=3.0,
                    text="Email: john@example.com",
                    speaker={"id": "spk_0"},
                    tone="neutral",
                    audio_state={"pitch": 150.0},
                ),
            ],
        )

        detector = PIIDetector(use_spacy=False)
        redactor = Redactor(mode=RedactionMode.MASK)
        result = redactor.redact_transcript(transcript, detector)

        seg = result.transcript.segments[0]
        assert seg.id == 0
        assert seg.start == 1.5
        assert seg.end == 3.0
        assert seg.speaker == {"id": "spk_0"}
        assert seg.tone == "neutral"
        assert seg.audio_state == {"pitch": 150.0}
