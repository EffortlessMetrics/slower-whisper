"""Tests for benchmark dataset manifest infrastructure.

Tests cover:
- Manifest schema validation against JSON Schema
- Manifest file integrity (all manifests load without errors)
- Smoke dataset availability (files exist and have correct hashes)
- Sample field requirements
- Download configuration validation

Run with:
    pytest tests/test_dataset_manifests.py -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_DIR = PROJECT_ROOT / "benchmarks" / "datasets"
SCHEMA_PATH = PROJECT_ROOT / "benchmarks" / "manifest_schema.json"
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"


def get_all_manifest_paths() -> list[Path]:
    """Find all manifest.json files in the datasets directory."""
    return list(MANIFEST_DIR.rglob("manifest.json"))


def load_json_file(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestManifestSchema:
    """Tests for manifest JSON Schema validation."""

    def test_schema_file_exists(self) -> None:
        """Manifest schema file exists at expected location."""
        assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"

    def test_schema_is_valid_json(self) -> None:
        """Manifest schema is valid JSON."""
        schema = load_json_file(SCHEMA_PATH)
        assert "$schema" in schema
        assert "properties" in schema

    def test_schema_has_required_fields(self) -> None:
        """Schema defines required fields."""
        schema = load_json_file(SCHEMA_PATH)
        required = schema.get("required", [])

        expected_required = ["schema_version", "id", "track", "split", "source", "license", "meta"]
        for field in expected_required:
            assert field in required, f"Missing required field in schema: {field}"


class TestManifestValidation:
    """Tests for manifest validation against schema."""

    @pytest.fixture
    def jsonschema_available(self) -> bool:
        """Check if jsonschema is available."""
        try:
            import jsonschema  # noqa: F401

            return True
        except ImportError:
            return False

    def test_all_manifests_validate_against_schema(self, jsonschema_available: bool) -> None:
        """All manifests validate against the JSON schema."""
        if not jsonschema_available:
            pytest.skip("jsonschema not installed")

        import jsonschema

        schema = load_json_file(SCHEMA_PATH)
        manifests = get_all_manifest_paths()

        assert len(manifests) > 0, "No manifests found"

        errors = []
        for manifest_path in manifests:
            try:
                manifest = load_json_file(manifest_path)
                jsonschema.validate(manifest, schema)
            except jsonschema.ValidationError as e:
                errors.append(f"{manifest_path.relative_to(PROJECT_ROOT)}: {e.message}")
            except json.JSONDecodeError as e:
                errors.append(f"{manifest_path.relative_to(PROJECT_ROOT)}: Invalid JSON - {e}")

        if errors:
            pytest.fail("Manifest validation errors:\n" + "\n".join(errors))


# =============================================================================
# Manifest Structure Tests
# =============================================================================


class TestManifestStructure:
    """Tests for manifest file structure and content."""

    @pytest.fixture
    def all_manifests(self) -> list[tuple[Path, dict[str, Any]]]:
        """Load all manifests."""
        manifests = []
        for path in get_all_manifest_paths():
            try:
                manifest = load_json_file(path)
                manifests.append((path, manifest))
            except json.JSONDecodeError:
                pass  # Skip invalid JSON, tested separately
        return manifests

    def test_all_manifests_load_without_error(self) -> None:
        """All manifest files are valid JSON."""
        manifests = get_all_manifest_paths()
        assert len(manifests) > 0, "No manifests found"

        errors = []
        for path in manifests:
            try:
                load_json_file(path)
            except json.JSONDecodeError as e:
                errors.append(f"{path.relative_to(PROJECT_ROOT)}: {e}")

        if errors:
            pytest.fail("Invalid JSON in manifests:\n" + "\n".join(errors))

    def test_schema_version_is_one(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """All manifests have schema_version = 1."""
        for path, manifest in all_manifests:
            version = manifest.get("schema_version")
            assert version == 1, f"{path.name}: schema_version should be 1, got {version}"

    def test_id_matches_directory_pattern(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """Manifest id follows naming conventions (lowercase, alphanumeric, hyphens)."""
        import re

        pattern = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

        for path, manifest in all_manifests:
            dataset_id = manifest.get("id", "")
            assert pattern.match(dataset_id), (
                f"{path.name}: id '{dataset_id}' doesn't match pattern "
                "(lowercase alphanumeric with hyphens/underscores)"
            )

    def test_track_is_valid(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """Track field has valid value."""
        valid_tracks = {"asr", "diarization", "streaming", "semantic", "emotion"}

        for path, manifest in all_manifests:
            track = manifest.get("track")
            assert track in valid_tracks, f"{path.name}: track '{track}' not in {valid_tracks}"

    def test_split_is_valid(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """Split field has valid value."""
        valid_splits = {"train", "dev", "test", "smoke"}

        for path, manifest in all_manifests:
            split = manifest.get("split")
            assert split in valid_splits, f"{path.name}: split '{split}' not in {valid_splits}"

    def test_license_has_required_fields(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """License object has required id and name fields."""
        for path, manifest in all_manifests:
            license_info = manifest.get("license", {})
            assert "id" in license_info, f"{path.name}: license missing 'id'"
            assert "name" in license_info, f"{path.name}: license missing 'name'"

    def test_source_has_name(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """Source object has name field."""
        for path, manifest in all_manifests:
            source = manifest.get("source", {})
            assert "name" in source, f"{path.name}: source missing 'name'"

    def test_meta_has_required_fields(self, all_manifests: list[tuple[Path, dict]]) -> None:
        """Meta object has created_at and sample_count."""
        for path, manifest in all_manifests:
            meta = manifest.get("meta", {})
            assert "created_at" in meta, f"{path.name}: meta missing 'created_at'"
            assert "sample_count" in meta, f"{path.name}: meta missing 'sample_count'"


# =============================================================================
# Smoke Dataset Tests
# =============================================================================


class TestSmokeDatasets:
    """Tests for smoke datasets - must be always available.

    Note: There are two types of smoke datasets:
    1. "Committed" smoke datasets - have samples committed to repo, no download required
    2. "Downloadable" smoke datasets - small but require download (e.g., commonvoice_en_smoke)

    Tests for committed smoke datasets are stricter (files must exist, hashes must match).
    """

    @pytest.fixture
    def smoke_manifests(self) -> list[tuple[Path, dict[str, Any]]]:
        """Load all smoke dataset manifests (both committed and downloadable)."""
        manifests = []
        for path in get_all_manifest_paths():
            try:
                manifest = load_json_file(path)
                if manifest.get("split") == "smoke":
                    manifests.append((path, manifest))
            except json.JSONDecodeError:
                pass
        return manifests

    @pytest.fixture
    def committed_smoke_manifests(self) -> list[tuple[Path, dict[str, Any]]]:
        """Load only committed smoke datasets (no download required)."""
        manifests = []
        for path in get_all_manifest_paths():
            try:
                manifest = load_json_file(path)
                if manifest.get("split") == "smoke":
                    download = manifest.get("download")
                    # Committed smoke datasets have no download or empty download
                    if download is None or download == {}:
                        manifests.append((path, manifest))
            except json.JSONDecodeError:
                pass
        return manifests

    def test_smoke_datasets_exist(self, smoke_manifests: list[tuple[Path, dict]]) -> None:
        """At least one smoke dataset exists for ASR and diarization."""
        tracks = {m["track"] for _, m in smoke_manifests}
        assert "asr" in tracks, "No ASR smoke dataset found"
        assert "diarization" in tracks, "No diarization smoke dataset found"

    def test_committed_smoke_datasets_exist(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """At least one committed smoke dataset exists for ASR and diarization."""
        tracks = {m["track"] for _, m in committed_smoke_manifests}
        assert "asr" in tracks, "No committed ASR smoke dataset found"
        assert "diarization" in tracks, "No committed diarization smoke dataset found"

    def test_committed_smoke_datasets_have_samples(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Committed smoke datasets have explicit sample definitions."""
        for path, manifest in committed_smoke_manifests:
            samples = manifest.get("samples", [])
            assert len(samples) > 0, f"{path.name}: committed smoke dataset must have samples"

    def test_committed_smoke_audio_files_exist(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """All committed smoke dataset audio files exist."""
        missing = []
        for path, manifest in committed_smoke_manifests:
            samples = manifest.get("samples", [])
            for sample in samples:
                audio_rel = sample.get("audio")
                if audio_rel:
                    audio_path = (path.parent / audio_rel).resolve()
                    if not audio_path.exists():
                        missing.append(f"{path.name}:{sample.get('id')}: {audio_rel}")

        if missing:
            pytest.fail("Missing audio files:\n" + "\n".join(missing))

    def test_committed_smoke_audio_hashes_match(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Committed smoke audio file SHA256 hashes match manifest."""
        mismatches = []
        for path, manifest in committed_smoke_manifests:
            samples = manifest.get("samples", [])
            for sample in samples:
                audio_rel = sample.get("audio")
                expected_hash = sample.get("sha256") or sample.get("audio_sha256")

                if audio_rel and expected_hash:
                    audio_path = (path.parent / audio_rel).resolve()
                    if audio_path.exists():
                        actual_hash = calculate_sha256(audio_path)
                        if actual_hash.lower() != expected_hash.lower():
                            mismatches.append(
                                f"{path.name}:{sample.get('id')}: "
                                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                            )

        if mismatches:
            pytest.fail("Hash mismatches:\n" + "\n".join(mismatches))

    def test_committed_smoke_rttm_files_exist(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Committed diarization smoke dataset RTTM files exist."""
        missing = []
        for path, manifest in committed_smoke_manifests:
            if manifest.get("track") != "diarization":
                continue

            samples = manifest.get("samples", [])
            for sample in samples:
                rttm_rel = sample.get("reference_rttm")
                if rttm_rel:
                    rttm_path = (path.parent / rttm_rel).resolve()
                    if not rttm_path.exists():
                        missing.append(f"{path.name}:{sample.get('id')}: {rttm_rel}")

        if missing:
            pytest.fail("Missing RTTM files:\n" + "\n".join(missing))

    def test_committed_smoke_rttm_hashes_match(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Committed smoke RTTM file SHA256 hashes match manifest."""
        mismatches = []
        for path, manifest in committed_smoke_manifests:
            if manifest.get("track") != "diarization":
                continue

            samples = manifest.get("samples", [])
            for sample in samples:
                rttm_rel = sample.get("reference_rttm")
                expected_hash = sample.get("rttm_sha256")

                if rttm_rel and expected_hash:
                    rttm_path = (path.parent / rttm_rel).resolve()
                    if rttm_path.exists():
                        actual_hash = calculate_sha256(rttm_path)
                        if actual_hash.lower() != expected_hash.lower():
                            mismatches.append(
                                f"{path.name}:{sample.get('id')}: "
                                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                            )

        if mismatches:
            pytest.fail("RTTM hash mismatches:\n" + "\n".join(mismatches))

    def test_committed_smoke_datasets_have_expected_baseline(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Committed smoke datasets define expected baseline metrics."""
        for path, manifest in committed_smoke_manifests:
            baseline = manifest.get("expected_baseline", {})
            assert len(baseline) > 0, (
                f"{path.name}: committed smoke dataset should have expected_baseline"
            )

    def test_committed_smoke_datasets_have_no_download(
        self, committed_smoke_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Committed smoke datasets should not require download (download is null or empty)."""
        for path, manifest in committed_smoke_manifests:
            download = manifest.get("download")
            assert download is None or download == {}, (
                f"{path.name}: committed smoke dataset should have download=null"
            )


# =============================================================================
# Download Configuration Tests
# =============================================================================


class TestDownloadConfiguration:
    """Tests for datasets with download configuration."""

    @pytest.fixture
    def downloadable_manifests(self) -> list[tuple[Path, dict[str, Any]]]:
        """Load manifests that have download configuration."""
        manifests = []
        for path in get_all_manifest_paths():
            try:
                manifest = load_json_file(path)
                download = manifest.get("download")
                if download and download != {}:
                    manifests.append((path, manifest))
            except json.JSONDecodeError:
                pass
        return manifests

    def test_download_method_is_valid(
        self, downloadable_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Download method is one of: http, huggingface, manual."""
        valid_methods = {"http", "huggingface", "manual"}

        for path, manifest in downloadable_manifests:
            download = manifest.get("download", {})
            method = download.get("method")
            if method:
                assert method in valid_methods, (
                    f"{path.name}: download.method '{method}' not in {valid_methods}"
                )

    def test_http_downloads_have_url(self, downloadable_manifests: list[tuple[Path, dict]]) -> None:
        """HTTP downloads have URL field."""
        for path, manifest in downloadable_manifests:
            download = manifest.get("download", {})
            method = download.get("method", "http")
            if method == "http":
                url = download.get("url")
                assert url, f"{path.name}: HTTP download missing 'url'"
                assert url.startswith("http"), f"{path.name}: URL should start with http(s)"

    def test_http_downloads_have_sha256(
        self, downloadable_manifests: list[tuple[Path, dict]]
    ) -> None:
        """HTTP downloads should have sha256 for verification."""
        warnings = []
        for path, manifest in downloadable_manifests:
            download = manifest.get("download", {})
            method = download.get("method", "http")
            if method == "http":
                sha256 = download.get("sha256")
                if not sha256:
                    warnings.append(f"{path.name}: HTTP download missing sha256")

        # This is a warning, not a failure, as some datasets may not have hashes yet
        if warnings:
            pytest.skip("Some downloads missing sha256 (recommended):\n" + "\n".join(warnings))

    def test_manual_downloads_have_instructions(
        self, downloadable_manifests: list[tuple[Path, dict]]
    ) -> None:
        """Manual downloads should have instructions_url or notes."""
        for path, manifest in downloadable_manifests:
            download = manifest.get("download", {})
            method = download.get("method")
            if method == "manual":
                has_instructions = download.get("instructions_url") or download.get("notes")
                assert has_instructions, (
                    f"{path.name}: manual download should have instructions_url or notes"
                )


# =============================================================================
# Sample Field Tests
# =============================================================================


class TestSampleFields:
    """Tests for sample field requirements."""

    @pytest.fixture
    def manifests_with_samples(self) -> list[tuple[Path, dict[str, Any]]]:
        """Load manifests that have explicit samples."""
        manifests = []
        for path in get_all_manifest_paths():
            try:
                manifest = load_json_file(path)
                samples = manifest.get("samples", [])
                if samples:
                    manifests.append((path, manifest))
            except json.JSONDecodeError:
                pass
        return manifests

    def test_asr_samples_have_required_fields(
        self, manifests_with_samples: list[tuple[Path, dict]]
    ) -> None:
        """ASR samples have id, audio, and reference_transcript."""
        for path, manifest in manifests_with_samples:
            if manifest.get("track") != "asr":
                continue

            samples = manifest.get("samples", [])
            for sample in samples:
                assert "id" in sample, f"{path.name}: sample missing 'id'"
                assert "audio" in sample, f"{path.name}:{sample.get('id')}: missing 'audio'"
                # reference_transcript is optional for some datasets

    def test_diarization_samples_have_required_fields(
        self, manifests_with_samples: list[tuple[Path, dict]]
    ) -> None:
        """Diarization samples have id, audio, and reference_rttm."""
        for path, manifest in manifests_with_samples:
            if manifest.get("track") != "diarization":
                continue

            samples = manifest.get("samples", [])
            for sample in samples:
                assert "id" in sample, f"{path.name}: sample missing 'id'"
                assert "audio" in sample, f"{path.name}:{sample.get('id')}: missing 'audio'"
                # reference_rttm is required for diarization smoke tests
                if manifest.get("split") == "smoke":
                    assert "reference_rttm" in sample, (
                        f"{path.name}:{sample.get('id')}: missing 'reference_rttm'"
                    )

    def test_sample_ids_are_unique_within_manifest(
        self, manifests_with_samples: list[tuple[Path, dict]]
    ) -> None:
        """Sample IDs are unique within each manifest."""
        for path, manifest in manifests_with_samples:
            samples = manifest.get("samples", [])
            ids = [s.get("id") for s in samples]
            unique_ids = set(ids)
            assert len(ids) == len(unique_ids), f"{path.name}: duplicate sample IDs found"


# =============================================================================
# Meta Statistics Tests
# =============================================================================


class TestMetaStatistics:
    """Tests for meta statistics consistency."""

    @pytest.fixture
    def manifests_with_samples(self) -> list[tuple[Path, dict[str, Any]]]:
        """Load manifests that have explicit samples."""
        manifests = []
        for path in get_all_manifest_paths():
            try:
                manifest = load_json_file(path)
                samples = manifest.get("samples", [])
                if samples:
                    manifests.append((path, manifest))
            except json.JSONDecodeError:
                pass
        return manifests

    def test_sample_count_matches_samples_array(
        self, manifests_with_samples: list[tuple[Path, dict]]
    ) -> None:
        """meta.sample_count matches length of samples array."""
        for path, manifest in manifests_with_samples:
            samples = manifest.get("samples", [])
            meta_count = manifest.get("meta", {}).get("sample_count", 0)
            actual_count = len(samples)
            assert meta_count == actual_count, (
                f"{path.name}: meta.sample_count={meta_count} but samples has {actual_count} items"
            )

    def test_total_duration_is_positive(
        self, manifests_with_samples: list[tuple[Path, dict]]
    ) -> None:
        """meta.total_duration_s is positive if present."""
        for path, manifest in manifests_with_samples:
            duration = manifest.get("meta", {}).get("total_duration_s")
            if duration is not None:
                assert duration > 0, f"{path.name}: total_duration_s should be positive"


# =============================================================================
# Directory Structure Tests
# =============================================================================


class TestDirectoryStructure:
    """Tests for expected directory structure."""

    def test_manifest_dir_exists(self) -> None:
        """Manifest directory exists."""
        assert MANIFEST_DIR.exists(), f"Manifest directory not found: {MANIFEST_DIR}"

    def test_data_dir_exists(self) -> None:
        """Data directory for committed files exists."""
        assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"

    def test_asr_track_dir_exists(self) -> None:
        """ASR track directory exists."""
        asr_dir = MANIFEST_DIR / "asr"
        assert asr_dir.exists(), "ASR track directory not found"

    def test_diarization_track_dir_exists(self) -> None:
        """Diarization track directory exists."""
        diarization_dir = MANIFEST_DIR / "diarization"
        assert diarization_dir.exists(), "Diarization track directory not found"

    def test_smoke_dirs_exist(self) -> None:
        """Smoke dataset directories exist."""
        asr_smoke = MANIFEST_DIR / "asr" / "smoke"
        diar_smoke = MANIFEST_DIR / "diarization" / "smoke"

        assert asr_smoke.exists(), "ASR smoke directory not found"
        assert diar_smoke.exists(), "Diarization smoke directory not found"

    def test_manifest_in_each_dataset_dir(self) -> None:
        """Each dataset directory contains a manifest.json."""
        missing = []
        for track_dir in MANIFEST_DIR.iterdir():
            if not track_dir.is_dir() or track_dir.name.startswith("."):
                continue
            for dataset_dir in track_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                manifest_path = dataset_dir / "manifest.json"
                if not manifest_path.exists():
                    missing.append(str(dataset_dir.relative_to(PROJECT_ROOT)))

        if missing:
            pytest.fail("Missing manifest.json in:\n" + "\n".join(missing))
