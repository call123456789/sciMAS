from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestrator import (
    DEFAULT_PLANNER_ROLES,
    allowed_tools_for_role,
    load_scimas_skills,
    normalize_role_name,
    skills_for_role,
)

SKILLS = load_scimas_skills(REPO_ROOT / "scimas_skills")


class SkillRoutingTests(unittest.TestCase):
    def test_pharma_aliases_normalize(self) -> None:
        self.assertEqual(normalize_role_name("medical"), "pharmacist")
        self.assertEqual(normalize_role_name("drug discovery scientist"), "drug-discovery-scientist")
        self.assertEqual(normalize_role_name("medicinal chemist"), "drug-discovery-scientist")

    def test_generic_roles_can_select_discipline_skills(self) -> None:
        chemist_skills = {skill.skill_id for skill in skills_for_role("chemist", SKILLS)}
        physicist_skills = {skill.skill_id for skill in skills_for_role("physicist", SKILLS)}
        math_skills = {skill.skill_id for skill in skills_for_role("mathematician", SKILLS)}
        pharmacist_skills = {skill.skill_id for skill in skills_for_role("pharmacist", SKILLS)}

        self.assertIn("computational-3d-conformers", chemist_skills)
        self.assertIn("cm-band-structure", physicist_skills)
        self.assertIn("statistical-tests-and-ci", math_skills)
        self.assertIn("pharma-molecule-generation", pharmacist_skills)

    def test_selected_skill_narrows_generic_role_tools(self) -> None:
        tools = allowed_tools_for_role(
            "physicist",
            selected_skills=["cm-band-structure"],
            scimas_skills=SKILLS,
            allow_role_fallback=False,
        )

        self.assertTrue(tools)
        self.assertTrue(all(tool.startswith("mcp__physics-condensed-matter__") for tool in tools))
        self.assertIn("mcp__physics-condensed-matter__tight_binding_1d", tools)

    def test_selected_pharma_skill_narrows_pharmacist_tools(self) -> None:
        tools = allowed_tools_for_role(
            "pharmacist",
            selected_skills=["pharma-public-affinity-data"],
            scimas_skills=SKILLS,
            allow_role_fallback=False,
        )

        self.assertTrue(tools)
        self.assertTrue(all(tool.startswith("mcp__pharma-data__") for tool in tools))
        self.assertIn("mcp__pharma-data__fetch_chembl_activities", tools)

    def test_pharma_molecule_skill_includes_local_checkpoint_tools(self) -> None:
        tools = allowed_tools_for_role(
            "drug discovery scientist",
            selected_skills=["pharma-molecule-generation"],
            scimas_skills=SKILLS,
            allow_role_fallback=False,
        )

        self.assertIn("mcp__pharma-drug-discovery__list_madd_local_checkpoints", tools)
        self.assertIn("mcp__pharma-drug-discovery__predict_with_local_madd_checkpoint", tools)
        self.assertIn("mcp__pharma-drug-discovery__generate_molecules_with_local_madd", tools)

    def test_strict_skill_routing_does_not_fallback_to_whole_role(self) -> None:
        tools = allowed_tools_for_role(
            "mathematician",
            selected_skills=[],
            scimas_skills=SKILLS,
            allow_role_fallback=False,
        )

        self.assertEqual(tools, [])

    def test_role_fallback_can_be_enabled_explicitly(self) -> None:
        tools = allowed_tools_for_role(
            "mathematician",
            selected_skills=[],
            scimas_skills=SKILLS,
            allow_role_fallback=True,
        )

        self.assertGreater(len(tools), 50)

    def test_default_planner_roles_include_sub_experts(self) -> None:
        self.assertIn("statistical-mathematician", DEFAULT_PLANNER_ROLES)
        self.assertIn("condensed-matter-physicist", DEFAULT_PLANNER_ROLES)
        self.assertIn("analytical-chemist", DEFAULT_PLANNER_ROLES)
        self.assertIn("pharmacist", DEFAULT_PLANNER_ROLES)
        self.assertIn("drug-discovery-scientist", DEFAULT_PLANNER_ROLES)


if __name__ == "__main__":
    unittest.main()
