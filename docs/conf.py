"""Sphinx configuration for slower-whisper documentation."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "slower-whisper"
author = "Effortless Metrics"
copyright = f"{datetime.now(UTC).year}, {author}"  # noqa: A001

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_snippets", "archive"]

autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Keep docs buildable without installing GPU/ML stacks.
autodoc_mock_imports = [
    "faster_whisper",
    "ctranslate2",
    "torch",
    "torchaudio",
    "pyannote",
    "librosa",
    "parselmouth",
    "numba",
    "soundfile",
    "sklearn",
    "spacy",
    "speechbrain",
    "resemblyzer",
    "pyarrow",
    "sentence_transformers",
    "anthropic",
    "openai",
    "langchain_core",
    "llama_index",
    "fastapi",
    "uvicorn",
    "transformers",
    "claude_agent_sdk",
]
