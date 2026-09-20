"""Consistency tests for the generated DrugSDA proxy server.

``load_scimas_skills`` validates nothing. A typo in a skill's
``x-scimas-tools`` yields an ``mcp__<server>__<tool>`` name that no server
can resolve, and the failure only shows up at run time as a tool the agent
cannot call. These tests pin the four places that must agree — the cached
remote schemas, the stdio server's registration, ``tools/pharma/_manifest.json``,
and the skills on disk — so drift surfaces here instead.

Nothing here touches the network: the remote endpoint is only ever read
through the schema cache the codegen wrote.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    ExceptionGroup
except NameError:  # Python 3.10 — the deployment interpreter
    # Present wherever mcp is, since anyio requires it below 3.11.
    from exceptiongroup import ExceptionGroup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator import (  # noqa: E402
    PHARMA_SERVER_TOOLS,
    ROLE_ALLOWED_PHARMA_SERVERS,
    allowed_tools_for_role,
    load_scimas_skills,
    normalize_role_name,
)

from tools.pharma import drug_sda_server as server  # noqa: E402
from tools.pharma._sda_config import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    HEAVY_TIMEOUT_SECONDS,
    HEAVY_TOOLS,
    default_timeout_for,
    is_heavy,
)

PHARMA_DIR = REPO_ROOT / "tools" / "pharma"
SCHEMAS_PATH = PHARMA_DIR / "_tool_schemas.json"
MANIFEST_PATH = PHARMA_DIR / "_manifest.json"
SKILL_DIR = REPO_ROOT / "scimas_skills"

SERVER_NAME = "pharma-drug-sda"
SKILL_PREFIX = "pharma-drug-sda-"
# The official DrugSDA-Tool inventory. Pinned so a codegen run against a
# changed endpoint, or a half-written manifest, cannot pass unnoticed.
EXPECTED_TOOL_COUNT = 81

SCHEMAS = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
SDA = MANIFEST["drug-sda"]
SKILLS = load_scimas_skills(SKILL_DIR)


def _base_type(schema: object) -> str | None:
    """The single concrete JSON type a property declares, or None.

    Unwraps ``anyOf``/``oneOf`` with a ``null`` branch and ``type`` given
    as a list, so a nullable string and a plain string compare equal.
    """
    if not isinstance(schema, dict):
        return None
    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list):
        concrete = [v for v in variants if (v or {}).get("type") != "null"]
        return _base_type(concrete[0]) if len(concrete) == 1 else None
    kind = schema.get("type")
    if isinstance(kind, list):
        concrete = [k for k in kind if k != "null"]
        return concrete[0] if len(concrete) == 1 else None
    return kind


def _sample_value(schema: dict):
    """A placeholder value that satisfies one property's declared type."""
    base = _base_type(schema)
    if base == "string":
        return "sample"
    if base == "integer":
        return 1
    if base == "number":
        return 1.0
    if base == "boolean":
        return True
    if base == "array":
        return []
    if base == "object":
        return {}
    return "sample"


def _required_arguments(schema: dict) -> dict:
    properties = schema.get("properties") or {}
    return {
        name: _sample_value(properties.get(name) or {})
        for name in schema.get("required") or ()
    }


class PharmaSchemaCacheTests(unittest.TestCase):
    def test_cache_is_present_and_complete(self) -> None:
        self.assertEqual(
            len(SCHEMAS),
            EXPECTED_TOOL_COUNT,
            msg="schema cache does not match the official tool count",
        )

    def test_every_entry_has_a_description_and_schema(self) -> None:
        for name, spec in SCHEMAS.items():
            with self.subTest(tool=name):
                self.assertTrue(spec.get("description", "").strip())
                self.assertIsInstance(spec.get("input_schema"), dict)


class PharmaRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exposed = {
            tool.name: tool for tool in asyncio.run(server.mcp.list_tools())
        }

    def test_registration_reported_no_warning(self) -> None:
        self.assertIsNone(server._REGISTRATION_WARNING)
        self.assertEqual(server._TOOL_COUNT, EXPECTED_TOOL_COUNT)

    def test_server_exposes_exactly_the_cached_tools(self) -> None:
        self.assertEqual(sorted(self.exposed), sorted(SCHEMAS))

    def test_tools_are_also_module_globals(self) -> None:
        # Standalone scripts call e.g. `drug_sda_server.pred_mol_admet(...)`.
        for name in SCHEMAS:
            with self.subTest(tool=name):
                self.assertTrue(callable(getattr(server, name, None)))


class PharmaSchemaFidelityTests(unittest.TestCase):
    """The advertised schema must still match the remote's own contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.exposed = {
            tool.name: tool for tool in asyncio.run(server.mcp.list_tools())
        }

    def test_properties_survive_the_round_trip(self) -> None:
        for name, spec in SCHEMAS.items():
            with self.subTest(tool=name):
                remote = set((spec.get("input_schema") or {}).get("properties") or ())
                advertised = set(self.exposed[name].input_schema["properties"])
                self.assertEqual(
                    advertised - remote,
                    {"timeout_seconds"},
                    msg="unexpected extra parameters were advertised",
                )
                self.assertEqual(
                    remote - advertised,
                    set(),
                    msg="remote parameters were dropped",
                )

    def test_required_sets_match(self) -> None:
        for name, spec in SCHEMAS.items():
            with self.subTest(tool=name):
                remote = set((spec.get("input_schema") or {}).get("required") or ())
                advertised = set(
                    self.exposed[name].input_schema.get("required") or ()
                )
                self.assertEqual(remote, advertised)

    def test_property_types_match(self) -> None:
        for name, spec in SCHEMAS.items():
            remote_props = (spec.get("input_schema") or {}).get("properties") or {}
            advertised = self.exposed[name].input_schema["properties"]
            for prop, remote_schema in remote_props.items():
                with self.subTest(tool=name, prop=prop):
                    self.assertEqual(
                        _base_type(advertised.get(prop)),
                        _base_type(remote_schema),
                    )

    def test_timeout_defaults_track_tool_weight(self) -> None:
        for name in SCHEMAS:
            with self.subTest(tool=name):
                prop = self.exposed[name].input_schema["properties"][
                    "timeout_seconds"
                ]
                expected = HEAVY_TIMEOUT_SECONDS if name in HEAVY_TOOLS else (
                    DEFAULT_TIMEOUT_SECONDS
                )
                self.assertEqual(prop["default"], expected)
                self.assertEqual(default_timeout_for(name), expected)


class PharmaArgumentForwardingTests(unittest.TestCase):
    """Only what the agent asked for should reach the remote."""

    def setUp(self) -> None:
        self.captured: dict = {}
        self._original = server._proxy

        class _Capturing:
            async def call_tool(self, tool_name, arguments, timeout_seconds=None):
                self_captured = test_case.captured
                self_captured["tool"] = tool_name
                self_captured["arguments"] = arguments
                self_captured["timeout"] = timeout_seconds
                return {"ok": True}

        test_case = self
        server._proxy = _Capturing()

    def tearDown(self) -> None:
        server._proxy = self._original

    def test_omitted_optionals_are_not_sent_as_null(self) -> None:
        # The MCP layer fills every declared default before dispatch, so a
        # parameter the agent never mentioned arrives here as None. Passing
        # it on would override the remote's own default with an explicit null.
        for name, spec in SCHEMAS.items():
            schema = spec.get("input_schema") or {}
            with self.subTest(tool=name):
                asyncio.run(
                    server.mcp.call_tool(name, _required_arguments(schema))
                )
                sent = self.captured["arguments"]
                self.assertNotIn(None, sent.values(), msg=f"null sent to {name}")

    def test_timeout_seconds_routes_to_the_budget_not_the_remote(self) -> None:
        arguments = _required_arguments(SCHEMAS["hdock_tool"]["input_schema"])
        arguments["timeout_seconds"] = 42
        asyncio.run(server.mcp.call_tool("hdock_tool", arguments))
        self.assertEqual(self.captured["timeout"], 42.0)
        self.assertNotIn("timeout_seconds", self.captured["arguments"])

    def test_heavy_tools_default_to_the_heavy_budget(self) -> None:
        arguments = _required_arguments(
            SCHEMAS["pred_protein_structure_esmfold"]["input_schema"]
        )
        asyncio.run(
            server.mcp.call_tool("pred_protein_structure_esmfold", arguments)
        )
        self.assertEqual(self.captured["timeout"], HEAVY_TIMEOUT_SECONDS)


class PharmaManifestTests(unittest.TestCase):
    def test_manifest_covers_the_whole_inventory(self) -> None:
        self.assertEqual(SDA["server"], SERVER_NAME)
        self.assertEqual(sorted(SDA["tools"]), sorted(SCHEMAS))
        self.assertEqual(SDA.get("skipped_unbundled"), [])

    def test_bundles_partition_the_inventory(self) -> None:
        bundled = [tool for spec in SDA["skills"].values() for tool in spec["tools"]]
        self.assertEqual(
            len(bundled), len(set(bundled)), msg="a tool is bundled twice"
        )
        self.assertEqual(sorted(bundled), sorted(SDA["tools"]))

    def test_bundle_roles_are_declared_targets(self) -> None:
        roles = set(SDA["role_targets"])
        for skill_id, spec in SDA["skills"].items():
            with self.subTest(skill=skill_id):
                self.assertIn(spec["role"], roles)


class PharmaSkillFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sda_skills = {
            skill_id: skill
            for skill_id, skill in SKILLS.items()
            if skill_id.startswith(SKILL_PREFIX)
        }

    def test_every_bundle_has_a_loadable_skill(self) -> None:
        self.assertEqual(sorted(self.sda_skills), sorted(SDA["skills"]))

    def test_frontmatter_agrees_with_the_manifest(self) -> None:
        for skill_id, spec in SDA["skills"].items():
            with self.subTest(skill=skill_id):
                skill = self.sda_skills.get(skill_id)
                self.assertIsNotNone(skill)
                self.assertEqual(skill.server, SERVER_NAME)
                self.assertEqual(skill.role, normalize_role_name(spec["role"]))
                self.assertEqual(sorted(skill.tools), sorted(spec["tools"]))


class PharmaOrchestratorWiringTests(unittest.TestCase):
    def test_orchestrator_inventory_reads_the_manifest(self) -> None:
        self.assertEqual(
            sorted(PHARMA_SERVER_TOOLS.get(SERVER_NAME, ())),
            sorted(SDA["tools"]),
        )

    def test_every_target_role_can_reach_the_server(self) -> None:
        for role in SDA["role_targets"]:
            with self.subTest(role=role):
                self.assertIn(
                    SERVER_NAME,
                    ROLE_ALLOWED_PHARMA_SERVERS.get(normalize_role_name(role), ()),
                )

    def test_skill_routing_yields_resolvable_tool_names(self) -> None:
        # The anti-typo check: a skill naming a tool the server does not
        # register would silently hand the agent an uncallable tool.
        declared = set(PHARMA_SERVER_TOOLS.get(SERVER_NAME, ()))
        for skill_id, spec in SDA["skills"].items():
            with self.subTest(skill=skill_id):
                allowed = allowed_tools_for_role(
                    spec["role"],
                    selected_skills=[skill_id],
                    scimas_skills=SKILLS,
                    allow_role_fallback=False,
                )
                self.assertTrue(allowed, msg="skill routed to no tools at all")
                for entry in allowed:
                    prefix = f"mcp__{SERVER_NAME}__"
                    self.assertTrue(entry.startswith(prefix), msg=entry)
                    self.assertIn(entry[len(prefix) :], declared)

    def test_skill_tool_lists_stay_small(self) -> None:
        # Focused skills are what keep a step's tool list affordable; the
        # repo's widest skill is 13 tools.
        for skill_id, spec in SDA["skills"].items():
            with self.subTest(skill=skill_id):
                self.assertLessEqual(len(spec["tools"]), 20)


class PharmaPlannerRoutingTests(unittest.TestCase):
    """Registration is worthless if the planner never picks the role.

    The planner chooses roles from `prompts/planner.md`, so a role holding
    these tools but absent from that prompt — or a capability routed to a
    role that does not hold it — leaves the tools unreachable no matter how
    well they are wired.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = (REPO_ROOT / "prompts" / "planner.md").read_text(
            encoding="utf-8"
        )

    def test_every_target_role_is_offered_to_the_planner(self) -> None:
        for role in SDA["role_targets"]:
            with self.subTest(role=role):
                self.assertIn(role, self.planner)

    def test_routing_guide_names_each_stage_owner(self) -> None:
        # One keyword per pipeline stage: docking (computational-chemist),
        # MD / free energy (physical-chemist), structure prediction
        # (structural-biologist), generation + ADMET (drug-discovery-scientist).
        for keyword in ("HDOCK", "MM/PBSA", "ESMFold", "REINVENT", "ADMET"):
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, self.planner)

    def test_capabilities_are_not_routed_to_roles_without_them(self) -> None:
        # `analytical-chemist` holds no DrugSDA tools; routing drug-likeness
        # there (as an earlier revision did) sends the step to a role that
        # cannot call a single one of them.
        holders = {
            role
            for role, servers in ROLE_ALLOWED_PHARMA_SERVERS.items()
            if SERVER_NAME in servers
        }
        for absent in ("analytical-chemist", "organic-chemist", "environmental-chemist"):
            with self.subTest(role=absent):
                self.assertNotIn(absent, holders)
        self.assertTrue(holders)

    def test_broad_roles_reach_the_tools_through_their_sub_disciplines(self) -> None:
        # `_skill_role_scope` widens a broad role to its whole discipline, so
        # `chemist` genuinely can dock and `biologist` genuinely can predict
        # structures — planner.md documents this, so it needs pinning.
        for role, skill_id, expected in (
            ("chemist", "pharma-drug-sda-docking", 10),
            ("chemist", "pharma-drug-sda-md-mmpbsa", 13),
            ("biologist", "pharma-drug-sda-structure-prediction-design", 7),
        ):
            with self.subTest(role=role, skill=skill_id):
                allowed = allowed_tools_for_role(
                    role, [skill_id], SKILLS, allow_role_fallback=False
                )
                self.assertEqual(len(allowed), expected)

    def test_unrelated_disciplines_reach_nothing(self) -> None:
        for role, skill_id in (
            ("physicist", "pharma-drug-sda-docking"),
            ("mathematician", "pharma-drug-sda-md-mmpbsa"),
            ("literature-searcher", "pharma-drug-sda-mol-descriptors"),
        ):
            with self.subTest(role=role):
                self.assertEqual(
                    allowed_tools_for_role(
                        role, [skill_id], SKILLS, allow_role_fallback=False
                    ),
                    [],
                )


class PharmaTimeoutErrorShapeTests(unittest.TestCase):
    """A timeout must reach the agent as advice, not as an ExceptionGroup."""

    def test_plain_timeout_is_recognised(self) -> None:
        self.assertTrue(server._is_timeout(TimeoutError()))
        self.assertTrue(server._is_timeout(asyncio.TimeoutError()))

    def test_wrapped_timeout_is_recognised(self) -> None:
        # This is the shape anyio's task group produces when a call is
        # cancelled from inside it, and a bare `except TimeoutError` misses it.
        group = ExceptionGroup("unhandled errors in a TaskGroup", [TimeoutError()])
        self.assertTrue(server._is_timeout(group))
        nested = ExceptionGroup("outer", [ExceptionGroup("inner", [TimeoutError()])])
        self.assertTrue(server._is_timeout(nested))

    def test_timeout_found_through_the_cause_chain(self) -> None:
        wrapper = RuntimeError("transport failed")
        wrapper.__cause__ = TimeoutError()
        self.assertTrue(server._is_timeout(wrapper))

    def test_other_failures_are_not_timeouts(self) -> None:
        self.assertFalse(server._is_timeout(ValueError("nope")))
        self.assertFalse(
            server._is_timeout(ExceptionGroup("g", [ValueError("nope")]))
        )

    def test_leaf_error_digs_out_the_real_failure(self) -> None:
        inner = ValueError("root cause")
        wrapped = ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])
        self.assertIs(server._leaf_error(wrapped), inner)

    def test_leaf_error_terminates_on_self_referential_cause(self) -> None:
        looped = RuntimeError("loops")
        looped.__cause__ = looped
        self.assertIs(server._leaf_error(looped), looped)

    def test_group_detection_is_duck_typed_not_builtin(self) -> None:
        # The deployment interpreter is 3.10, where `BaseExceptionGroup` does
        # not exist, so referencing that name here would raise NameError from
        # inside the error handler. Detection must go through `.exceptions`.
        class LooksLikeAGroup(Exception):
            def __init__(self, members):
                super().__init__("group")
                self.exceptions = members

        self.assertTrue(server._is_timeout(LooksLikeAGroup([TimeoutError()])))
        self.assertFalse(server._is_timeout(LooksLikeAGroup([ValueError()])))
        self.assertIs(
            server._leaf_error(LooksLikeAGroup([ValueError("leaf")])).args[0],
            "leaf",
        )


class PharmaConfigTests(unittest.TestCase):
    def test_heavy_tools_get_the_longer_budget(self) -> None:
        self.assertTrue(is_heavy("pred_protein_structure_esmfold"))
        self.assertFalse(is_heavy("calculate_mol_drug_chemistry"))
        self.assertEqual(
            default_timeout_for("calculate_mol_drug_chemistry"),
            DEFAULT_TIMEOUT_SECONDS,
        )

    def test_every_heavy_tool_exists_remotely(self) -> None:
        # A stale name here would silently fall back to the short budget.
        for name in sorted(HEAVY_TOOLS):
            with self.subTest(tool=name):
                self.assertIn(name, SCHEMAS)

    def test_budgets_stay_under_the_runner_ceiling(self) -> None:
        self.assertLess(HEAVY_TIMEOUT_SECONDS, server.RUNNER_CEILING_SECONDS)


class PharmaAnnotationTests(unittest.TestCase):
    """`_annotation` is the schema-to-signature mapping; pin its edge cases."""

    def test_nullable_scalar_is_optional(self) -> None:
        annotation = server._annotation(
            {"anyOf": [{"type": "string"}, {"type": "null"}]}, "x"
        )
        self.assertTrue(server._is_optional(annotation))

    def test_type_given_as_a_list_is_optional(self) -> None:
        annotation = server._annotation({"type": ["string", "null"]}, "x")
        self.assertTrue(server._is_optional(annotation))

    def test_array_of_scalars(self) -> None:
        annotation = server._annotation(
            {"type": "array", "items": {"type": "string"}}, "x"
        )
        self.assertEqual(annotation, list[str])

    def test_array_of_objects(self) -> None:
        annotation = server._annotation(
            {"type": "array", "items": {"type": "object"}}, "x"
        )
        self.assertEqual(annotation, list[dict])

    def test_unknown_shape_degrades_to_any(self) -> None:
        self.assertIs(server._annotation({}, "x"), Any)
        self.assertIs(server._annotation({"type": "wat"}, "x"), Any)
        self.assertIs(server._annotation("not a schema", "x"), Any)

    def test_multi_variant_anyof_degrades_to_any(self) -> None:
        annotation = server._annotation(
            {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "x"
        )
        self.assertIs(annotation, Any)


class PharmaScriptLaunchTests(unittest.TestCase):
    """The server must work when launched the way ``config/mcp.json`` does.

    Every other test here imports ``drug_sda_server`` as a module, which puts
    the repo root on ``sys.path`` for free. The CLI instead runs
    ``python tools/pharma/drug_sda_server.py``, where ``sys.path[0]`` is
    ``tools/pharma/`` — so the ``from tools.pharma._sda_config import ...``
    line raised ``ModuleNotFoundError`` and the server died at import. The
    CLI reported the server as merely ``pending``, so 246 passing tests and a
    clean dashboard start both missed it; only a real launch catches this.
    """

    def test_every_configured_server_bootstraps_sys_path(self) -> None:
        """Repo-wide: any server importing ``tools.*`` needs the bootstrap.

        Checked statically across all of ``config/mcp.json`` rather than for
        this server alone, since the failure mode is identical for every
        subdirectory server and the fix is one three-line preamble.
        """
        config_path = REPO_ROOT / "config" / "mcp.json"
        servers = json.loads(config_path.read_text(encoding="utf-8"))
        servers = servers.get("mcpServers") or servers

        offenders: list[str] = []
        checked = 0
        for name, spec in sorted(servers.items()):
            script = next(
                (
                    arg
                    for arg in (spec.get("args") or [])
                    if isinstance(arg, str) and arg.endswith(".py")
                ),
                None,
            )
            if script is None:
                continue
            path = REPO_ROOT / script
            self.assertTrue(path.exists(), f"{name}: script {script} is missing")
            source = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"^\s*(from tools[.\s]|import tools\b)", source, re.M)
            if match is None:
                continue  # self-contained server, nothing to bootstrap
            checked += 1
            # The insert must precede the first `tools.*` import, or it is
            # no help: the import runs and raises before the line is reached.
            if "sys.path.insert" not in source[: match.start()]:
                offenders.append(f"{name} ({script})")

        self.assertEqual(offenders, [], "servers missing the sys.path bootstrap")
        # Guards against the regex silently matching nothing, which would make
        # this test pass vacuously after a refactor.
        self.assertGreaterEqual(checked, 10)

    def test_server_starts_as_a_script_and_lists_every_tool(self) -> None:
        """A cold script-path launch must complete the stdio handshake.

        Speaks JSON-RPC to the real subprocess over stdin/stdout, which is
        the only way to observe the import-time crash: an in-process import
        cannot reproduce it.
        """
        proc = subprocess.Popen(
            [sys.executable, str(PHARMA_DIR / "drug_sda_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            text=True,
        )
        try:
            self._handshake(proc)
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def _handshake(self, proc: subprocess.Popen) -> None:
        def exchange(request: dict, want_id: int | None) -> dict:
            """Write `request` and read until the matching reply arrives."""
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            if want_id is None:
                return {}
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                line = self._readline(proc, deadline)
                if not line:
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # not a reply (a stray print, say)
                if message.get("id") == want_id:
                    return message
            stderr = proc.stderr.read() if proc.stderr else ""
            self.fail(f"no reply to {request['method']} within 60s; stderr={stderr!r}")

        reply = exchange(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
            want_id=1,
        )
        self.assertNotIn("error", reply, f"initialize failed: {reply}")
        exchange(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            want_id=None,
        )
        reply = exchange(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            want_id=2,
        )
        tools = (reply.get("result") or {}).get("tools") or []
        self.assertEqual(len(tools), EXPECTED_TOOL_COUNT)
        self.assertIn("calculate_mol_drug_chemistry", {t["name"] for t in tools})

    @staticmethod
    def _readline(proc: subprocess.Popen, deadline: float) -> str:
        """`readline` with a deadline.

        A blocking `readline` on a dead server would hang the suite forever,
        and the reply to `tools/list` is ~113KB on one line, so the read
        cannot simply be capped by size.
        """
        box: list[str] = []
        reader = threading.Thread(
            target=lambda: box.append(proc.stdout.readline()), daemon=True
        )
        reader.start()
        reader.join(timeout=max(0.0, deadline - time.monotonic()))
        return box[0] if box else ""


if __name__ == "__main__":
    unittest.main()
