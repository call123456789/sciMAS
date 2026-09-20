"""Tests for the AST-based tool-spec extractor.

Run with:
    cd /Users/a123/Documents/games/sciMAS
    python -m pytest tests/test_tool_spec.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tool_spec import (  # noqa: E402
    ToolSpec,
    build_tool_spec_index,
    extract_tool_specs,
)


def test_extract_tool_specs_environmental() -> None:
    """Real-world extraction: waste_calorific_value from chemistry/environmental."""
    server_file = ROOT / "tools" / "chemistry" / "environmental_server.py"
    if not server_file.is_file():
        # Skip silently — repo might be partially vendored.
        return
    specs = extract_tool_specs(server_file)
    by_name = {s.name: s for s in specs}
    assert "waste_calorific_value" in by_name, (
        "waste_calorific_value should be extracted from environmental_server.py"
    )
    spec = by_name["waste_calorific_value"]
    assert spec.server == "chemistry-environmental"
    assert spec.mcp_name == "mcp__chemistry-environmental__waste_calorific_value"
    assert spec.signature == "(components: list[dict]) -> str", (
        f"unexpected signature: {spec.signature!r}"
    )
    assert spec.parameters == (("components", "list[dict]"),), (
        f"unexpected parameters: {spec.parameters!r}"
    )
    assert spec.description, "description should be non-empty"
    assert "calorific" in spec.description.lower()


def test_build_tool_spec_index_synthetic(tmp_path: Path) -> None:
    """Round-trip: synthesize a server file and verify extraction."""
    # build_tool_spec_index walks tools/{chemistry,physics,biology,mathematics,
    # pharma,web,biomni}/*_server.py, so the synthetic file must live under a
    # recognized discipline directory.
    sub = tmp_path / "chemistry"
    sub.mkdir()
    server_file = sub / "demo_server.py"
    server_content = "\n".join(
        [
            "from mcp.server.mcpserver import MCPServer",
            "",
            'mcp = MCPServer("demo-server")',
            "",
            "@mcp.tool(",
            "    description=(",
            '        "Sum two floats and return JSON {sum: float}."',
            "    )",
            ")",
            "async def add_two(a: float, b: float) -> str:",
            '    return "{}"',
            "",
            '@mcp.tool(description="Echo the input string.")',
            "def echo(message: str = \"hi\") -> str:",
            "    return message",
            "",
        ]
    )
    server_file.write_text(server_content, encoding="utf-8")

    idx = build_tool_spec_index(tmp_path)
    assert "mcp__demo-server__add_two" in idx
    assert "mcp__demo-server__echo" in idx
    add_spec = idx["mcp__demo-server__add_two"]
    assert isinstance(add_spec, ToolSpec)
    assert add_spec.signature == "(a: float, b: float) -> str"
    assert "Sum two floats" in add_spec.description
    echo_spec = idx["mcp__demo-server__echo"]
    assert echo_spec.signature == '(message: str = \'hi\') -> str'
    assert echo_spec.parameters == (("message", "str"),)


def test_build_tool_spec_index_handles_syntax_error(tmp_path: Path) -> None:
    """A SyntaxError in one server file must not break the indexer."""
    sub = tmp_path / "chemistry"
    sub.mkdir()
    good = sub / "good_server.py"
    good.write_text(
        "\n".join(
            [
                "from mcp.server.mcpserver import MCPServer",
                "",
                'mcp = MCPServer("demo")',
                "",
                '@mcp.tool(description="Returns 42.")',
                "def answer() -> str:",
                '    return "42"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    bad = sub / "bad_server.py"
    bad.write_text("def broken(:\n  pass\n", encoding="utf-8")

    idx = build_tool_spec_index(tmp_path)
    assert "mcp__demo__answer" in idx, "good file's tools should still be indexed"
    assert len(idx) == 1, f"only one tool expected, got: {list(idx)}"


def test_build_tool_spec_index_missing_dir(tmp_path: Path) -> None:
    """Missing tools_dir returns {} without raising."""
    idx = build_tool_spec_index(tmp_path / "does-not-exist")
    assert idx == {}


def test_one_liner_signature() -> None:
    """Compact form for the skill-selection catalog."""
    spec = ToolSpec(
        name="x",
        server="srv",
        mcp_name="mcp__srv__x",
        signature="(a: float, b: float = 1.0) -> str",
        description="",
        parameters=(("a", "float"), ("b", "float")),
    )
    assert spec.one_liner() == "x(a: float, b: float = 1.0) -> str"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))