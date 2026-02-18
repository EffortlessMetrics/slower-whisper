"""Tests for the receipt contract module.

Tests cover:
- Receipt construction with all required fields
- Config hash computation (deterministic)
- Run ID generation (unique)
- Git commit detection (optional)
- Receipt validation
- Integration with writers.add_receipt_to_meta
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from slower_whisper.pipeline.ids import is_valid_run_id
from slower_whisper.pipeline.models import SCHEMA_VERSION
from slower_whisper.pipeline.receipt import (
    RECEIPT_REQUIRED_FIELDS,
    Receipt,
    build_receipt,
    compute_config_hash,
    generate_run_id,
    get_git_commit,
    get_tool_version,
    validate_receipt,
)
from slower_whisper.pipeline.writers import add_receipt_to_meta


class TestReceiptRequiredFields:
    """Test that receipts contain all required fields."""

    def test_build_receipt_has_all_required_fields(self) -> None:
        """build_receipt() should produce a receipt with all required fields."""
        receipt = build_receipt(
            model="large-v3",
            device="cuda",
            compute_type="float16",
        )
        receipt_dict = receipt.to_dict()

        missing = RECEIPT_REQUIRED_FIELDS - set(receipt_dict.keys())
        assert not missing, f"Missing required fields: {missing}"

    def test_receipt_dataclass_to_dict_has_required_fields(self) -> None:
        """Receipt.to_dict() should include all required fields."""
        receipt = Receipt(
            tool_version="2.1.0",
            schema_version=2,
            model="tiny",
            device="cpu",
            compute_type="int8",
            config_hash="abcd1234efgh",
            run_id="run-20260128-120000-abc123",
            created_at="2024-01-01T00:00:00Z",
        )
        receipt_dict = receipt.to_dict()

        for field in RECEIPT_REQUIRED_FIELDS:
            assert field in receipt_dict, f"Required field {field} missing from to_dict()"

    def test_validate_receipt_accepts_valid_receipt(self) -> None:
        """validate_receipt() should return empty list for valid receipt."""
        receipt = build_receipt(
            model="large-v3",
            device="cuda",
            compute_type="float16",
        )
        errors = validate_receipt(receipt.to_dict())
        assert errors == [], f"Unexpected validation errors: {errors}"


class TestConfigHash:
    """Test config hash computation."""

    def test_config_hash_is_deterministic(self) -> None:
        """Same config should produce same hash."""
        config = {"model": "large-v3", "device": "cuda", "compute_type": "float16"}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2

    def test_config_hash_is_12_chars(self) -> None:
        """Config hash should be exactly 12 characters."""
        config = {"model": "tiny", "device": "cpu"}
        hash_val = compute_config_hash(config)
        assert len(hash_val) == 12

    def test_config_hash_is_hex(self) -> None:
        """Config hash should be hexadecimal."""
        config = {"model": "tiny", "device": "cpu"}
        hash_val = compute_config_hash(config)
        assert re.match(r"^[a-f0-9]{12}$", hash_val)

    def test_config_hash_order_independent(self) -> None:
        """Config hash should be independent of key order."""
        config1 = {"model": "large-v3", "device": "cuda", "compute_type": "float16"}
        config2 = {"device": "cuda", "compute_type": "float16", "model": "large-v3"}
        assert compute_config_hash(config1) == compute_config_hash(config2)

    def test_different_configs_produce_different_hashes(self) -> None:
        """Different configs should produce different hashes."""
        config1 = {"model": "large-v3", "device": "cuda"}
        config2 = {"model": "tiny", "device": "cpu"}
        assert compute_config_hash(config1) != compute_config_hash(config2)


class TestRunId:
    """Test run ID generation."""

    def test_run_id_format(self) -> None:
        """Generated run_id should match the expected format."""
        run_id = generate_run_id()
        # Format: run-YYYYMMDD-HHMMSS-XXXXXX
        assert is_valid_run_id(run_id), f"Invalid run_id format: {run_id}"
        assert run_id.startswith("run-")

    def test_run_ids_are_unique(self) -> None:
        """Each call to generate_run_id should produce a unique value."""
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_build_receipt_generates_unique_run_ids(self) -> None:
        """Each receipt should have a unique run_id."""
        receipts = [
            build_receipt(model="tiny", device="cpu", compute_type="int8") for _ in range(10)
        ]
        run_ids = {r.run_id for r in receipts}
        assert len(run_ids) == 10


class TestCreatedAt:
    """Test created_at timestamp."""

    def test_created_at_is_iso8601(self) -> None:
        """created_at should be a valid ISO 8601 timestamp."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
        # Should parse without error
        dt = datetime.fromisoformat(receipt.created_at.replace("Z", "+00:00"))
        assert dt.tzinfo is not None or receipt.created_at.endswith("Z")

    def test_created_at_is_recent(self) -> None:
        """created_at should be close to current time."""
        before = datetime.now(UTC)
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
        after = datetime.now(UTC)

        created = datetime.fromisoformat(receipt.created_at.replace("Z", "+00:00"))
        assert before <= created <= after

    def test_created_at_can_be_overridden(self) -> None:
        """created_at should accept override value."""
        custom_time = "2020-01-01T12:00:00Z"
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
            created_at=custom_time,
        )
        assert receipt.created_at == custom_time


class TestGitCommit:
    """Test git commit detection."""

    def test_git_commit_is_optional(self) -> None:
        """git_commit should be optional in the receipt."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
            include_git_commit=False,
        )
        assert receipt.git_commit is None

    def test_git_commit_format(self) -> None:
        """git_commit should be a short hash if available."""
        commit = get_git_commit()
        if commit is not None:
            # Short commit is typically 7-12 characters
            assert 7 <= len(commit) <= 12
            assert re.match(r"^[a-f0-9]+$", commit)

    def test_receipt_without_git_commit_is_valid(self) -> None:
        """Receipt without git_commit should pass validation."""
        receipt_dict = {
            "tool_version": "2.1.0",
            "schema_version": 2,
            "model": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "config_hash": "abcd1234efgh",
            "run_id": "run-20260128-120000-abc123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        errors = validate_receipt(receipt_dict)
        assert errors == []


class TestToolVersion:
    """Test tool version detection."""

    def test_tool_version_is_string(self) -> None:
        """Tool version should be a non-empty string."""
        version = get_tool_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_tool_version_in_receipt(self) -> None:
        """Receipt should include tool_version."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
        assert receipt.tool_version == get_tool_version()


class TestSchemaVersion:
    """Test schema version handling."""

    def test_schema_version_matches_models(self) -> None:
        """Receipt schema_version should match models.SCHEMA_VERSION by default."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
        assert receipt.schema_version == SCHEMA_VERSION

    def test_schema_version_can_be_overridden(self) -> None:
        """schema_version should accept override value."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
            schema_version=99,
        )
        assert receipt.schema_version == 99


class TestReceiptValidation:
    """Test receipt validation."""

    def test_validate_missing_required_field(self) -> None:
        """Validation should fail when required field is missing."""
        incomplete = {
            "tool_version": "2.1.0",
            "schema_version": 2,
            # Missing: model, device, compute_type, config_hash, run_id, created_at
        }
        errors = validate_receipt(incomplete)
        assert len(errors) > 0
        assert "Missing required fields" in errors[0]

    def test_validate_wrong_type_tool_version(self) -> None:
        """Validation should fail when tool_version is not a string."""
        bad_receipt = {
            "tool_version": 123,  # Should be string
            "schema_version": 2,
            "model": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "config_hash": "abcd1234efgh",
            "run_id": "run-20260128-120000-abc123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        errors = validate_receipt(bad_receipt)
        assert any("tool_version must be a string" in e for e in errors)

    def test_validate_wrong_type_schema_version(self) -> None:
        """Validation should fail when schema_version is not an int."""
        bad_receipt = {
            "tool_version": "2.1.0",
            "schema_version": "2",  # Should be int
            "model": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "config_hash": "abcd1234efgh",
            "run_id": "run-20260128-120000-abc123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        errors = validate_receipt(bad_receipt)
        assert any("schema_version must be an integer" in e for e in errors)

    def test_validate_wrong_config_hash_length(self) -> None:
        """Validation should fail when config_hash is wrong length."""
        bad_receipt = {
            "tool_version": "2.1.0",
            "schema_version": 2,
            "model": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "config_hash": "abc",  # Should be 12 chars
            "run_id": "run-20260128-120000-abc123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        errors = validate_receipt(bad_receipt)
        assert any("config_hash must be exactly 12 characters" in e for e in errors)

    def test_validate_accepts_uuid_run_id_for_backward_compatibility(self) -> None:
        """Validation should accept legacy UUID run_id for backward compatibility."""
        receipt_dict = {
            "tool_version": "2.1.0",
            "schema_version": 2,
            "model": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "config_hash": "abcd1234efgh",
            "run_id": "550e8400-e29b-41d4-a716-446655440000",  # Legacy UUID format
            "created_at": "2024-01-01T00:00:00Z",
        }
        errors = validate_receipt(receipt_dict)
        assert errors == [], f"UUID run_id should be accepted: {errors}"


class TestReceiptSerialization:
    """Test Receipt serialization and deserialization."""

    def test_to_dict_roundtrip(self) -> None:
        """Receipt should survive to_dict/from_dict roundtrip."""
        original = build_receipt(
            model="large-v3",
            device="cuda",
            compute_type="float16",
        )
        receipt_dict = original.to_dict()
        restored = Receipt.from_dict(receipt_dict)

        assert restored.tool_version == original.tool_version
        assert restored.schema_version == original.schema_version
        assert restored.model == original.model
        assert restored.device == original.device
        assert restored.compute_type == original.compute_type
        assert restored.config_hash == original.config_hash
        assert restored.run_id == original.run_id
        assert restored.created_at == original.created_at
        assert restored.git_commit == original.git_commit

    def test_to_dict_excludes_none_git_commit(self) -> None:
        """to_dict should not include git_commit key when it's None."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
            include_git_commit=False,
        )
        receipt_dict = receipt.to_dict()
        assert "git_commit" not in receipt_dict

    def test_to_dict_includes_git_commit_when_present(self) -> None:
        """to_dict should include git_commit when it has a value."""
        receipt = Receipt(
            tool_version="2.1.0",
            schema_version=2,
            model="tiny",
            device="cpu",
            compute_type="int8",
            config_hash="abcd1234efgh",
            run_id="run-20260128-120000-abc123",
            created_at="2024-01-01T00:00:00Z",
            git_commit="abc1234",
        )
        receipt_dict = receipt.to_dict()
        assert receipt_dict["git_commit"] == "abc1234"


class TestAddReceiptToMeta:
    """Test integration with writers.add_receipt_to_meta."""

    def test_add_to_empty_meta(self) -> None:
        """add_receipt_to_meta should work with None meta."""
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
        result = add_receipt_to_meta(None, receipt)
        assert "receipt" in result
        assert result["receipt"]["model"] == "tiny"

    def test_add_to_existing_meta(self) -> None:
        """add_receipt_to_meta should preserve existing meta fields."""
        existing_meta = {
            "model_name": "large-v3",
            "audio_duration_sec": 120.5,
        }
        receipt = build_receipt(
            model="large-v3",
            device="cuda",
            compute_type="float16",
        )
        result = add_receipt_to_meta(existing_meta, receipt)

        # Original fields preserved
        assert result["model_name"] == "large-v3"
        assert result["audio_duration_sec"] == 120.5
        # Receipt added
        assert "receipt" in result
        assert result["receipt"]["device"] == "cuda"

    def test_add_receipt_does_not_mutate_original(self) -> None:
        """add_receipt_to_meta should not mutate the original meta dict."""
        original = {"key": "value"}
        original_copy = dict(original)
        receipt = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
        add_receipt_to_meta(original, receipt)
        assert original == original_copy


class TestBuildReceiptWithCustomConfig:
    """Test build_receipt with custom config for hashing."""

    def test_custom_config_produces_different_hash(self) -> None:
        """Custom config should produce different hash than default."""
        default_receipt = build_receipt(
            model="large-v3",
            device="cuda",
            compute_type="float16",
        )
        custom_receipt = build_receipt(
            model="large-v3",
            device="cuda",
            compute_type="float16",
            config={
                "model": "large-v3",
                "device": "cuda",
                "compute_type": "float16",
                "extra_param": "value",
            },
        )
        assert default_receipt.config_hash != custom_receipt.config_hash

    def test_same_custom_config_produces_same_hash(self) -> None:
        """Same custom config should produce same hash."""
        config = {"model": "tiny", "device": "cpu", "setting": 42}
        r1 = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
            config=config,
        )
        r2 = build_receipt(
            model="tiny",
            device="cpu",
            compute_type="int8",
            config=config,
        )
        assert r1.config_hash == r2.config_hash
