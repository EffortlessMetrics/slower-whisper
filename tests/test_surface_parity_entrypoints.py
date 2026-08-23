"""Directory and CLI parity owners must remain explicit and executable."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "surface_parity_entrypoints.json"


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def declared_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.add(node.name)
        elif isinstance(node, ast.ClassDef):
            result.update(
                f"{node.name}.{member.name}"
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return result


def test_console_help_and_candidate_sets_are_executable() -> None:
    contract = load_contract()
    assert contract["contract_version"] == 1
    assert "slower-whisper" in contract["project_scripts"]
    assert contract["root_help"]["returncode"] == 0
    assert contract["transcribe_help"]["returncode"] == 0
    assert contract["batch_candidates"]
    assert contract["cli_candidates"]


def test_discovered_owners_resolve_in_live_source() -> None:
    contract = load_contract()
    for record in [
        *contract["batch_candidates"],
        *contract["cli_candidates"],
    ]:
        path = ROOT / record["path"]
        assert path.is_file(), record
        assert record["symbol"] in declared_symbols(path), record
        assert isinstance(record["constructs_engine"], bool)
        assert isinstance(record["canonical_calls"], list)
        assert isinstance(record["writer_calls"], list)
