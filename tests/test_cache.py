from __future__ import annotations

from transcription import cache
from transcription.cache import CachePaths, configure_global_cache_env


def test_cachepaths_from_env_and_ensure_dirs(tmp_path):
    env = {
        "SLOWER_WHISPER_CACHE_ROOT": str(tmp_path / "root"),
        "HF_HOME": str(tmp_path / "custom_hf"),
        "TORCH_HOME": str(tmp_path / "custom_torch"),
    }

    paths = CachePaths.from_env(env)

    assert paths.root == tmp_path / "root"
    assert paths.hf_home == tmp_path / "custom_hf"
    assert paths.torch_home == tmp_path / "custom_torch"
    assert paths.whisper_root == paths.root / "whisper"
    assert paths.emotion_root == paths.root / "emotion"
    assert paths.diarization_root == paths.root / "diarization"
    assert paths.samples_root == paths.root / "samples"
    assert paths.benchmarks_root == paths.root / "benchmarks"

    created = paths.ensure_dirs()
    for path in [
        created.root,
        created.hf_home,
        created.torch_home,
        created.whisper_root,
        created.emotion_root,
        created.diarization_root,
        created.samples_root,
        created.benchmarks_root,
    ]:
        assert path.exists()
        assert path.is_dir()


def test_configure_global_cache_env_idempotent(monkeypatch, tmp_path):
    """Configure env once, then ensure reconfiguration does not override user values."""
    monkeypatch.setattr(cache, "_CACHE_ENV_CONFIGURED", False)

    env: dict[str, str] = {"SLOWER_WHISPER_CACHE_ROOT": str(tmp_path / "cache")}
    paths = configure_global_cache_env(env)

    assert env["HF_HOME"] == str(paths.hf_home)
    assert env["TORCH_HOME"] == str(paths.torch_home)
    assert env["HF_HUB_CACHE"] == str(paths.hf_home / "hub")

    # Subsequent calls should not clobber user overrides
    env["HF_HOME"] = "custom_hf_path"
    configure_global_cache_env(env)
    assert env["HF_HOME"] == "custom_hf_path"
