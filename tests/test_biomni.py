"""Tests for the BiOMNI migration:

1. Every generated server file imports without error.
2. Tool names exposed by each server match
   ``orchestrator.BIOMNI_SERVER_TOOLS`` exactly (no drift between
   generated code and the static allow-list).
3. Every skill under ``scimas_skills/biomni-*`` declares only tools
   that the matching server actually exposes.
4. The static ``ROLE_ALLOWED_BIOMNI_SERVERS`` map agrees with the
   codegen manifest.
5. Smoke tests on representative Phase-1 tools exercise the wrapper
   end-to-end (BiOMNI module load → call → JSON envelope) and prove
   that a missing dep / bad input surfaces as JSON rather than
   crashing the stdio loop.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestrator import (  # noqa: E402
    BIOMNI_SERVER_TOOLS,
    DEFAULT_PLANNER_ROLES,
    ROLE_ALLOWED_BIOMNI_SERVERS,
    allowed_tools_for_role,
    load_scimas_skills,
    normalize_role_name,
    skills_for_role,
)


# Categories generated for Phase 1. Keep in sync with PHASE_1_BUNDLES
# in scripts/biomni_codegen.py.
PHASE_1_CATEGORIES = (
    "biochemistry",
    "literature",
    "protocols",
    "database",
    "pharmacology",
)


def _server_module(category: str):
    return importlib.import_module(f"tools.biomni.{category}_server")


async def _list_tools(category: str) -> list[str]:
    mod = _server_module(category)
    tools = await mod.mcp.list_tools()
    return sorted(t.name for t in tools)


class BiomniServerImportTests(unittest.TestCase):
    def test_every_phase1_server_imports(self) -> None:
        for cat in PHASE_1_CATEGORIES:
            with self.subTest(category=cat):
                mod = importlib.import_module(f"tools.biomni.{cat}_server")
                self.assertTrue(hasattr(mod, "mcp"))
                self.assertTrue(callable(getattr(mod, "mcp").run_stdio_async))


class BiomniInventoryParityTests(unittest.TestCase):
    """Server-exposed tools must equal orchestrator's static allow-list."""

    def test_server_tools_match_orchestrator_inventory(self) -> None:
        for server, declared in BIOMNI_SERVER_TOOLS.items():
            with self.subTest(server=server):
                category = server.replace("biomni-", "", 1)
                exposed = asyncio.run(_list_tools(category))
                self.assertEqual(
                    sorted(declared),
                    exposed,
                    msg=f"{server} tools diverged from BIOMNI_SERVER_TOOLS",
                )

    def test_orchestrator_role_keys_match_manifest(self) -> None:
        manifest_path = REPO_ROOT / "tools/biomni/_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        role_targets = {cat: meta["role_target"] for cat, meta in manifest.items()}
        # Every category in the manifest should be served by at least one role.
        served_categories = {
            cat
            for cat, servers in ROLE_ALLOWED_BIOMNI_SERVERS.items()
            for server in servers
            if server.startswith("biomni-")
        }
        for cat in manifest:
            with self.subTest(category=cat):
                self.assertIn(f"biomni-{cat}", set(BIOMNI_SERVER_TOOLS.keys()))
        # Each declared role_target should actually appear as a key.
        targets = set(role_targets.values())
        declared_roles = set(ROLE_ALLOWED_BIOMNI_SERVERS.keys())
        self.assertTrue(
            targets.issubset(declared_roles),
            msg=f"unmapped role targets: {targets - declared_roles}",
        )


class BiomniManifestInvariantTests(unittest.TestCase):
    """Manifest is the single source of truth.

    The orchestrator loads ``BIOMNI_SERVER_TOOLS`` from
    ``tools/biomni/_manifest.json`` at import time, and the codegen
    merges with the existing manifest so re-running ``--phase N``
    doesn't drop earlier phases. These tests pin the invariants so
    silent drift (e.g. a hand-edited server file losing a wrapper)
    fails CI rather than corrupting the runtime allow-list.
    """

    @classmethod
    def setUpClass(cls) -> None:
        manifest_path = REPO_ROOT / "tools/biomni/_manifest.json"
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # server -> tool set, derived from the manifest
        cls.server_tools: dict[str, set[str]] = {
            meta["server"]: set(meta.get("tools", ()))
            for meta in cls.manifest.values()
        }
        # role_target -> server list, derived from the manifest
        cls.role_servers: dict[str, set[str]] = {}
        for meta in cls.manifest.values():
            cls.role_servers.setdefault(meta["role_target"], set()).add(
                meta["server"]
            )

    def test_orchestrator_inventory_matches_manifest(self) -> None:
        """Every BIOMNI_SERVER_TOOLS entry must come from the manifest."""
        for server, declared in BIOMNI_SERVER_TOOLS.items():
            with self.subTest(server=server):
                self.assertIn(server, self.server_tools)
                self.assertEqual(
                    sorted(declared),
                    sorted(self.server_tools[server]),
                    msg=(
                        f"{server} tools in orchestrator differ from manifest. "
                        "Re-run scripts/biomni_codegen.py."
                    ),
                )
        # And the reverse: nothing in the manifest should be missing
        # from the orchestrator.
        for server in self.server_tools:
            with self.subTest(server=server):
                self.assertIn(server, set(BIOMNI_SERVER_TOOLS.keys()))

    def test_role_allowlist_servers_exist_in_inventory(self) -> None:
        """Every server in ROLE_ALLOWED_BIOMNI_SERVERS must be exposed."""
        for role, servers in ROLE_ALLOWED_BIOMNI_SERVERS.items():
            for server in servers:
                with self.subTest(role=role, server=server):
                    self.assertIn(
                        server,
                        set(BIOMNI_SERVER_TOOLS.keys()),
                        msg=(
                            f"{role} references {server} but the orchestrator "
                            "inventory (loaded from manifest) does not know "
                            "about it. Did the codegen miss a category?"
                        ),
                    )

    def test_manifest_servers_resolve_to_files(self) -> None:
        """Every server in the manifest must have a generated _server.py."""
        for server in self.server_tools:
            with self.subTest(server=server):
                category = server.replace("biomni-", "", 1)
                path = REPO_ROOT / f"tools/biomni/{category}_server.py"
                self.assertTrue(
                    path.exists(),
                    msg=f"{path} missing — codegen for {server} incomplete",
                )

    def test_no_orphan_biomni_skills_on_disk(self) -> None:
        """Every biomni skill on disk must be declared in the manifest.

        Orphan skills are hand-written skills that point at BiOMNI tools
        but were never registered. They cause confusing routing and
        duplicate the maintenance burden, so we reject them.
        """
        manifest_skills = {
            skill_id
            for meta in self.manifest.values()
            for skill_id in meta.get("skills", [])
        }
        on_disk: set[str] = set()
        for entry in (REPO_ROOT / "scimas_skills").iterdir():
            if not entry.is_dir() or not entry.name.startswith("biomni-"):
                continue
            if (entry / "SKILL.md").exists():
                on_disk.add(entry.name)
        orphans = sorted(on_disk - manifest_skills)
        self.assertEqual(
            orphans,
            [],
            msg=(
                "biomni skills on disk that are NOT in _manifest.json: "
                f"{orphans}. Either add them to PHASE_N_BUNDLES in "
                "scripts/biomni_codegen.py and rerun, or delete them."
            ),
        )

    def test_role_target_in_manifest_maps_to_known_role(self) -> None:
        """Every role_target in the manifest must be wired up."""
        targets = {meta["role_target"] for meta in self.manifest.values()}
        declared = set(ROLE_ALLOWED_BIOMNI_SERVERS.keys())
        self.assertTrue(
            targets.issubset(declared),
            msg=(
                f"manifest role_targets not wired into "
                f"ROLE_ALLOWED_BIOMNI_SERVERS: {targets - declared}"
            ),
        )

    def test_skill_tools_are_subset_of_server_tools(self) -> None:
        """x-scimas-tools must be a subset of the server's exposed tools."""
        skills = load_scimas_skills(REPO_ROOT / "scimas_skills")
        for skill in skills.values():
            if not skill.server.startswith("biomni-"):
                continue
            with self.subTest(skill=skill.skill_id):
                exposed = self.server_tools.get(skill.server, set())
                stale = [t for t in skill.tools if t not in exposed]
                self.assertEqual(
                    stale,
                    [],
                    msg=(
                        f"{skill.skill_id} lists tools that {skill.server} "
                        f"does not expose: {stale}"
                    ),
                )


class BiomniSkillTests(unittest.TestCase):
    SKILLS = load_scimas_skills(REPO_ROOT / "scimas_skills")

    def test_biomni_skills_known_to_orchestrator(self) -> None:
        biomni_skills = [
            s for s in self.SKILLS.values() if s.server.startswith("biomni-")
        ]
        self.assertGreaterEqual(
            len(biomni_skills), 4,
            msg="expected at least 4 biomni skills after Phase 1",
        )

    def test_every_biomni_skill_tool_exists_on_its_server(self) -> None:
        cache: dict[str, set[str]] = {}

        async def tools_for(server: str) -> set[str]:
            if server not in cache:
                category = server.replace("biomni-", "", 1)
                cache[server] = set(await _list_tools(category))
            return cache[server]

        for skill in self.SKILLS.values():
            if not skill.server.startswith("biomni-"):
                continue
            with self.subTest(skill=skill.skill_id):
                exposed = asyncio.run(tools_for(skill.server))
                missing = [t for t in skill.tools if t not in exposed]
                self.assertEqual(
                    missing,
                    [],
                    msg=f"{skill.skill_id} declares tools missing from "
                        f"{skill.server}: {missing}",
                )

    def test_role_targeting_round_trips(self) -> None:
        # molecular-biologist role must see the biomni-biochemistry skill.
        mab_skills = {
            s.skill_id
            for s in skills_for_role("molecular-biologist", self.SKILLS)
        }
        self.assertIn("biomni-biochemistry-rna-secondary", mab_skills)
        self.assertIn("biomni-biochemistry-cd-kinetics", mab_skills)

        # literature-searcher must see biomni-literature skills.
        lit_skills = {
            s.skill_id
            for s in skills_for_role("literature-searcher", self.SKILLS)
        }
        self.assertIn("biomni-literature-arxiv-pubmed", lit_skills)

        # pharma-data-specialist must see biomni-database skills.
        pds_skills = {
            s.skill_id
            for s in skills_for_role("pharma-data-specialist", self.SKILLS)
        }
        self.assertIn("biomni-database-uniprot-pdb-pubchem", pds_skills)

        # drug-discovery-scientist must see biomni-pharmacology skills.
        dds_skills = {
            s.skill_id
            for s in skills_for_role("drug-discovery-scientist", self.SKILLS)
        }
        self.assertIn("biomni-pharmacology-rdkit-properties", dds_skills)

    def test_default_planner_roles_include_biomni_subroles(self) -> None:
        for role in (
            "molecular-biologist",
            "literature-searcher",
            "pharma-data-specialist",
            "drug-discovery-scientist",
        ):
            with self.subTest(role=role):
                self.assertIn(role, DEFAULT_PLANNER_ROLES)


class BiomniRoleToolAllowListTests(unittest.TestCase):
    SKILLS = load_scimas_skills(REPO_ROOT / "scimas_skills")

    def test_role_fallback_exposes_only_server_tools(self) -> None:
        tools = allowed_tools_for_role(
            "molecular-biologist",
            selected_skills=None,
            scimas_skills=self.SKILLS,
            allow_role_fallback=True,
        )
        # Must include the new biomni tools alongside the existing
        # biology-molecular ones.
        self.assertIn(
            "mcp__biomni-biochemistry__analyze_rna_secondary_structure_features",
            tools,
        )
        self.assertIn("mcp__biology-molecular__analyze_qrt_pcr", tools)

        # Must NOT include tools from servers the role cannot see.
        self.assertFalse(
            any(tool.startswith("mcp__biomni-database__") for tool in tools)
        )

    def test_strict_skill_selection_narrows_to_skill_tools(self) -> None:
        tools = allowed_tools_for_role(
            "molecular-biologist",
            selected_skills=["biomni-biochemistry-rna-secondary"],
            scimas_skills=self.SKILLS,
            allow_role_fallback=False,
        )
        self.assertTrue(tools)
        self.assertTrue(
            all(tool.startswith("mcp__biomni-biochemistry__") for tool in tools)
        )
        self.assertIn(
            "mcp__biomni-biochemistry__analyze_rna_secondary_structure_features",
            tools,
        )


class BiomniWrapperSmokeTests(unittest.TestCase):
    """Each wrapper must return a JSON string, never raise or hang."""

    def test_rna_secondary_structure_returns_json(self) -> None:
        from tools.biomni.biochemistry_server import (
            analyze_rna_secondary_structure_features,
        )

        payload = analyze_rna_secondary_structure_features("((..))", "")
        data = json.loads(payload)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["tool"], "analyze_rna_secondary_structure_features")
        # BiOMNI's own log surfaced inside the envelope.
        self.assertIn("log", data)

    def test_cd_spectrum_returns_json(self) -> None:
        from tools.biomni.biochemistry_server import (
            analyze_circular_dichroism_spectra,
        )

        payload = analyze_circular_dichroism_spectra(
            "Znf706",
            "protein",
            [190.0, 195.0, 215.0, 220.0],
            [5.0, 8.0, -2.0, -1.0],
        )
        data = json.loads(payload)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["tool"], "analyze_circular_dichroism_spectra")
        self.assertIn("Circular Dichroism Analysis Report", data["log"])

    def test_protocols_list_local_protocols_returns_json(self) -> None:
        from tools.biomni.protocols_server import list_local_protocols

        # Empty / missing directory should not crash; BiOMNI returns a
        # research-log error string and the wrapper envelopes it.
        payload = list_local_protocols()
        data = json.loads(payload)
        # The wrapper must always produce a JSON envelope. Whether the
        # result is ``ok`` (directory configured) or ``error`` (no
        # protocols directory on this machine) is environment-dependent.
        self.assertIn(data["status"], {"ok", "error"})
        self.assertEqual(data["tool"], "list_local_protocols")

    def test_literature_skill_wrapper_handles_missing_dep(self) -> None:
        """If a tool needs a heavy library that's not installed, the
        wrapper must still return JSON, not raise."""
        from tools.biomni.literature_server import query_arxiv

        try:
            payload = query_arxiv("cancer immunotherapy", max_papers=1)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"query_arxiv must not raise, got {type(exc).__name__}: {exc}")
        # JSON parse must succeed regardless of network / dep state.
        data = json.loads(payload)
        self.assertIn("status", data)
        self.assertEqual(data["tool"], "query_arxiv")

    def test_pharmacology_rdkit_wrapper_handles_missing_dep(self) -> None:
        from tools.biomni.pharmacology_server import (
            calculate_physicochemical_properties,
        )

        try:
            payload = calculate_physicochemical_properties("CCO")
        except Exception as exc:  # noqa: BLE001
            self.fail(
                f"calculate_physicochemical_properties must not raise, "
                f"got {type(exc).__name__}: {exc}"
            )
        data = json.loads(payload)
        self.assertIn("status", data)
        self.assertEqual(data["tool"], "calculate_physicochemical_properties")


class BiomniCommonTests(unittest.TestCase):
    def test_normalize_output_dir_default(self) -> None:
        from tools.biomni.common import normalize_output_dir

        path = normalize_output_dir("", "test_tool")
        self.assertIn("scimas_biomni", path)
        self.assertTrue(Path(path).is_dir())

    def test_normalize_output_dir_explicit(self) -> None:
        from tools.biomni.common import normalize_output_dir

        path = normalize_output_dir("/tmp/scimas_test_custom", "test_tool")
        self.assertTrue(Path(path).is_dir())

    def test_ok_envelope_shape(self) -> None:
        from tools.biomni.common import ok

        payload = json.loads(ok("foo", bar=1, baz="two"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["tool"], "foo")
        self.assertEqual(payload["bar"], 1)
        self.assertEqual(payload["baz"], "two")

    def test_err_envelope_shape(self) -> None:
        from tools.biomni.common import err

        payload = json.loads(err("boom", tool="foo"))
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["message"], "boom")
        self.assertEqual(payload["tool"], "foo")

    def test_load_biomni_module_missing(self) -> None:
        from tools.biomni.common import load_biomni_module

        with self.assertRaises(ImportError):
            load_biomni_module("nonexistent_category_xyz")


class BiOMNIToolDirResolutionTests(unittest.TestCase):
    """The BiOMNI checkout location must not be hard-coded to one machine.

    Regression: ``common.py`` used to default to the original author's
    absolute macOS path, so on any other host all 34 biomni-* MCP servers
    registered their tools fine and then failed on every call.
    """

    def setUp(self) -> None:
        import tools.biomni.common as common

        self.common = common
        self._saved = os.environ.pop("BIOMNI_TOOL_DIR", None)

    def tearDown(self) -> None:
        if self._saved is not None:
            os.environ["BIOMNI_TOOL_DIR"] = self._saved
        else:
            os.environ.pop("BIOMNI_TOOL_DIR", None)

    def test_env_var_wins(self) -> None:
        os.environ["BIOMNI_TOOL_DIR"] = "/tmp/definitely-not-biomni"
        self.assertEqual(
            self.common._biomni_tool_dir_candidates(),
            [Path("/tmp/definitely-not-biomni")],
        )

    def test_candidates_follow_the_checkout_location(self) -> None:
        """Candidates are computed from the checkout path, not baked in.

        The discarded implementation returned a literal macOS path, which
        happened to sit next to the original author's clone — so it worked
        there and nowhere else. Relocating the recorded repo root must move
        the candidates with it; a hard-coded literal cannot.
        """
        original = self.common._SCIMAS_ROOT
        try:
            self.common._SCIMAS_ROOT = Path("/opt/somewhere/else/sciMAS")
            moved = self.common._biomni_tool_dir_candidates()
        finally:
            self.common._SCIMAS_ROOT = original
        self.assertEqual(
            moved,
            [
                Path("/opt/somewhere/else/Biomni/biomni/tool"),
                Path("/opt/somewhere/else/sciMAS/Biomni/biomni/tool"),
            ],
        )

    def test_missing_biomni_raises_with_every_path_tried(self) -> None:
        """Absent BiOMNI → ImportError listing the candidates, not a mystery."""
        os.environ["BIOMNI_TOOL_DIR"] = "/tmp/nope-not-here"
        with self.assertRaises(ImportError) as ctx:
            self.common.biomni_tool_dir()
        message = str(ctx.exception)
        self.assertIn("/tmp/nope-not-here", message)
        self.assertIn("BIOMNI_TOOL_DIR", message)

    def test_importing_common_does_not_require_biomni(self) -> None:
        """Import stays side-effect free — servers import it to list tools.

        Checked in a subprocess so it reflects a cold import rather than
        whatever this test session already has in ``sys.modules``: if the
        module resolved BiOMNI eagerly, ``tools.biomni.<cat>_server`` could
        never register its tools on a host without BiOMNI.
        """
        env = dict(os.environ, BIOMNI_TOOL_DIR="/tmp/nope-not-here")
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tools.biomni.common as c; print(callable(c.biomni_tool_dir))",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "True")


if __name__ == "__main__":
    unittest.main()