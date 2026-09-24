"""Command-line entry point for sciMAS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from orchestrator import SciMASOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="sciMAS scientific multi-agent runner")
    parser.add_argument("problem", nargs="?", help="Scientific problem to solve.")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Read the scientific problem from a text file.",
    )
    parser.add_argument(
        "--roles",
        nargs="*",
        default=None,
        help="Optional role whitelist for the planner.",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=None,
        help="Directory containing role markdown prompts.",
    )
    parser.add_argument(
        "--skill-dir",
        type=Path,
        default=None,
        help="Directory containing sciMAS skill packages (default: scimas_skills).",
    )
    parser.add_argument(
        "--no-skill-routing",
        action="store_true",
        help="Disable two-pass skill selection and use full role tool lists.",
    )
    parser.add_argument(
        "--allow-role-tool-fallback",
        action="store_true",
        help=(
            "When skill routing is enabled but no skill is selected, allow "
            "the broad role to receive its full discipline tool list."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save run reports.",
    )
    parser.add_argument(
        "--claude-bin",
        default="claude",
        help="Claude CLI binary name or path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model flag passed through to Claude CLI.",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Optional model override for planner and planner-reviewer calls.",
    )
    parser.add_argument(
        "--agent-model",
        default=None,
        help="Optional model override for worker and synthesizer calls.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=None,
        help="ANTHROPIC_BASE_URL override for planner and planner-reviewer calls.",
    )
    parser.add_argument(
        "--agent-base-url",
        default=None,
        help="ANTHROPIC_BASE_URL override for worker and synthesizer calls.",
    )
    parser.add_argument(
        "--no-auto-synthesize",
        action="store_true",
        help="Disable the automatic synthesizer pass.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["python-dsl", "legacy-json"],
        default="python-dsl",
        help=(
            "Planner output mode. python-dsl uses restricted async workflow "
            "source; legacy-json preserves the old JSON topology planner."
        ),
    )
    parser.add_argument(
        "--workflow-file",
        type=Path,
        default=None,
        help=(
            "Execute this precomputed Python DSL workflow directly, without "
            "calling the planner or planner-reviewer."
        ),
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "Path to an .mcp.json file exposing custom tools to the model. "
            "Defaults to auto-detecting config/mcp.json in the sciMAS "
            "package directory. Use --no-mcp to disable."
        ),
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP auto-detection even if config/mcp.json exists.",
    )
    parser.add_argument(
        "--allowed-tools",
        nargs="*",
        default=None,
        help="Space-separated tool names the spawned claude -p may call. "
        "MCP tools are named 'mcp__<server>__<tool>'.",
    )
    parser.add_argument(
        "--permission-mode",
        default="bypassPermissions",
        choices=["default", "acceptEdits", "bypassPermissions", "plan"],
        help="Claude CLI permission mode (default: bypassPermissions).",
    )
    parser.add_argument(
        "--no-skip-permissions",
        action="store_true",
        help="Do NOT pass --dangerously-skip-permissions to the Claude CLI.",
    )
    parser.add_argument(
        "--append-system-prompt",
        default=None,
        help="Optional extra text appended to the system prompt for every role.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-call timeout in seconds (default: 600).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.file:
        problem = args.file.read_text(encoding="utf-8").strip()
    elif args.problem:
        problem = args.problem.strip()
    else:
        problem = sys.stdin.read().strip()

    if not problem:
        parser.error("Provide a problem with a positional argument, --file, or stdin.")

    orchestrator = SciMASOrchestrator(
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        skill_dir=args.skill_dir,
        use_skill_routing=not args.no_skill_routing,
        strict_skill_routing=not args.allow_role_tool_fallback,
        claude_bin=args.claude_bin,
        model=args.model,
        planner_model=args.planner_model,
        agent_model=args.agent_model,
        planner_env_overrides=(
            {"ANTHROPIC_BASE_URL": args.planner_base_url}
            if args.planner_base_url
            else None
        ),
        agent_env_overrides=(
            {"ANTHROPIC_BASE_URL": args.agent_base_url}
            if args.agent_base_url
            else None
        ),
        mcp_config_path="" if args.no_mcp else args.mcp_config,
        allowed_tools=args.allowed_tools,
        permission_mode=args.permission_mode,
        dangerously_skip_permissions=not args.no_skip_permissions,
        extra_system_prompt=args.append_system_prompt,
        timeout=args.timeout,
        planner_mode=args.planner_mode.replace("-", "_"),
    )
    workflow_file = args.workflow_file.expanduser().resolve() if args.workflow_file else None
    workflow_source = (
        workflow_file.read_text(encoding="utf-8") if workflow_file else None
    )
    report = orchestrator.run(
        problem=problem,
        roles=args.roles,
        auto_synthesize=not args.no_auto_synthesize,
        workflow_source=workflow_source,
        workflow_path=str(workflow_file) if workflow_file else None,
    )

    print(report.final_answer or report.runs[-1].result)
    print(f"\nSaved run report to: {report.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
