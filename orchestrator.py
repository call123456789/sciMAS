"""Orchestration logic for sciMAS."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from claude_runner import ClaudeRunner
from plan import AgentResult, AgentRun, ExecutionPlan, PlanStep, PlannerRun, RunReport
from tool_spec import ToolSpec, build_tool_spec_index
from workflow_dsl import (
    WorkflowDSLValidationError,
    WorkflowValue,
    coerce_agent_result,
    collect_workflow_dependencies,
    collect_workflow_values,
    extract_json_object,
    make_json_safe,
    parse_and_validate_workflow,
    unwrap_workflow_value,
)


ROLE_ALIASES = {
    # The five sub-domain chemists. "chemist" still works as a
    # generic all-of-chemistry fallback (see ROLE_ALLOWED_CHEMISTRY_SERVERS).
    "chemist": "chemist",
    "chemistry": "chemist",
    "analytical-chemist": "analytical-chemist",
    "analyticalchemist": "analytical-chemist",
    "analytical_chemist": "analytical-chemist",
    "analytical-chemistry": "analytical-chemist",
    "computational-chemist": "computational-chemist",
    "computationalchemist": "computational-chemist",
    "computational_chemist": "computational-chemist",
    "computational-chemistry": "computational-chemist",
    "environmental-chemist": "environmental-chemist",
    "environmentalchemist": "environmental-chemist",
    "environmental_chemist": "environmental-chemist",
    "environmental-chemistry": "environmental-chemist",
    "organic-chemist": "organic-chemist",
    "organicchemist": "organic-chemist",
    "organic_chemist": "organic-chemist",
    "organic-chemistry": "organic-chemist",
    "physical-chemist": "physical-chemist",
    "physicalchemist": "physical-chemist",
    "physical_chemist": "physical-chemist",
    "physical-chemistry": "physical-chemist",
    # Legacy physics role.
    "physics": "physicist",
    "physicist": "physicist",
    # Physics sub-experts.
    "classical-mechanicist": "classical-mechanicist",
    "classical_mechanicist": "classical-mechanicist",
    "classical-mechanics": "classical-mechanicist",
    "classical_mechanics": "classical-mechanicist",
    "mechanicist": "classical-mechanicist",
    "em-optics-physicist": "em-optics-physicist",
    "em_optics_physicist": "em-optics-physicist",
    "em-optics": "em-optics-physicist",
    "em_optics": "em-optics-physicist",
    "waves-fluid-physicist": "waves-fluid-physicist",
    "waves_fluid_physicist": "waves-fluid-physicist",
    "waves-fluid": "waves-fluid-physicist",
    "waves_fluid": "waves-fluid-physicist",
    "condensed-matter-physicist": "condensed-matter-physicist",
    "condensed_matter_physicist": "condensed-matter-physicist",
    "condensed-matter": "condensed-matter-physicist",
    "condensed_matter": "condensed-matter-physicist",
    "quantum-atomic-physicist": "quantum-atomic-physicist",
    "quantum_atomic_physicist": "quantum-atomic-physicist",
    "quantum-atomic": "quantum-atomic-physicist",
    "quantum_atomic": "quantum-atomic-physicist",
    # Legacy biology aliases.
    "biology": "biologist",
    "biological": "biologist",
    "biologist": "biologist",
    # Biology sub-experts.
    "molecular-biologist": "molecular-biologist",
    "molecular_biologist": "molecular-biologist",
    "molecular-biology": "molecular-biologist",
    "molecular_biology": "molecular-biologist",
    "geneticist": "geneticist",
    "geneticists": "geneticist",
    "genetics": "geneticist",
    "cell-biologist": "cell-biologist",
    "cell_biologist": "cell-biologist",
    "cell-biology": "cell-biologist",
    "cell_biology": "cell-biologist",
    "structural-biologist": "structural-biologist",
    "structural_biologist": "structural-biologist",
    "structural-biology": "structural-biologist",
    "structural_biology": "structural-biologist",
    "mass-spectrometrist": "mass-spectrometrist",
    "mass_spectrometrist": "mass-spectrometrist",
    "mass-spec": "mass-spectrometrist",
    "mass_spec": "mass-spectrometrist",
    "mass-spectrometry": "mass-spectrometrist",
    "mass_spectrometry": "mass-spectrometrist",
    # Legacy mathematics aliases.
    "math": "mathematician",
    "mathematics": "mathematician",
    "mathematician": "mathematician",
    # Math sub-experts.
    "algebraic-mathematician": "algebraic-mathematician",
    "algebraic_mathematician": "algebraic-mathematician",
    "algebra": "algebraic-mathematician",
    "linear-algebra": "algebraic-mathematician",
    "linear_algebra": "algebraic-mathematician",
    "statistical-mathematician": "statistical-mathematician",
    "statistical_mathematician": "statistical-mathematician",
    "statistics": "statistical-mathematician",
    "statistician": "statistical-mathematician",
    "geometric-mathematician": "geometric-mathematician",
    "geometric_mathematician": "geometric-mathematician",
    "geometry": "geometric-mathematician",
    "optimization-mathematician": "optimization-mathematician",
    "optimization_mathematician": "optimization-mathematician",
    "optimization": "optimization-mathematician",
    "numerical-mathematician": "numerical-mathematician",
    "numerical_mathematician": "numerical-mathematician",
    "numerical": "numerical-mathematician",
    # Pharmaceutical / medical drug-discovery aliases.
    "pharma": "pharmacist",
    "pharmaceutical": "pharmacist",
    "pharmaceuticalscience": "pharmacist",
    "pharmaceutical-science": "pharmacist",
    "pharmaceutical_science": "pharmacist",
    "medicine": "pharmacist",
    "medical": "pharmacist",
    "pharmacy": "pharmacist",
    "pharmacist": "pharmacist",
    "drugdiscovery": "drug-discovery-scientist",
    "drug-discovery": "drug-discovery-scientist",
    "drug_discovery": "drug-discovery-scientist",
    "drugdiscoveryscientist": "drug-discovery-scientist",
    "drug-discovery-scientist": "drug-discovery-scientist",
    "drug_discovery_scientist": "drug-discovery-scientist",
    "medicinalchemist": "drug-discovery-scientist",
    "medicinal-chemist": "drug-discovery-scientist",
    "medicinal_chemist": "drug-discovery-scientist",
    "pharmadata": "pharma-data-specialist",
    "pharma-data": "pharma-data-specialist",
    "pharma_data": "pharma-data-specialist",
    "pharmadataspecialist": "pharma-data-specialist",
    "pharma-data-specialist": "pharma-data-specialist",
    "pharma_data_specialist": "pharma-data-specialist",
    "pharmaml": "pharma-ml-engineer",
    "pharma-ml": "pharma-ml-engineer",
    "pharma_ml": "pharma-ml-engineer",
    "pharmaautoml": "pharma-ml-engineer",
    "pharma-automl": "pharma-ml-engineer",
    "pharma_automl": "pharma-ml-engineer",
    "pharmamlengineer": "pharma-ml-engineer",
    "pharma-ml-engineer": "pharma-ml-engineer",
    "pharma_ml_engineer": "pharma-ml-engineer",
    # Web / literature retrieval aliases.
    "literature": "literature-searcher",
    "literaturesearch": "literature-searcher",
    "literature-search": "literature-searcher",
    "literature_search": "literature-searcher",
    "literaturesearcher": "literature-searcher",
    "literature-searcher": "literature-searcher",
    "literature_searcher": "literature-searcher",
    "papersearch": "literature-searcher",
    "paper-search": "literature-searcher",
    "paper_search": "literature-searcher",
    "scholar": "literature-searcher",
    # Other roles.
    "generalist": "generalist",
    "synthesizer": "synthesizer",
    "planner": "planner",
}


def normalize_role_name(role: str) -> str:
    key = role.strip().lower().replace(" ", "")
    return ROLE_ALIASES.get(key, key)


# Per-role allow-list of chemistry MCP servers. Roles not in this map
# (biologist, physicist, mathematician, planner, generalist,
# synthesizer) receive NO chemistry tools. The five sub-chemists each
# get exactly one server. The legacy "chemist" role keeps access to
# all five chemistry servers for backward compatibility.
ROLE_ALLOWED_CHEMISTRY_SERVERS: dict[str, tuple[str, ...]] = {
    "analytical-chemist":     ("chemistry-analytical",),
    "computational-chemist":  ("chemistry-computational",),
    "environmental-chemist": ("chemistry-environmental",),
    "organic-chemist":        ("chemistry-organic",),
    "physical-chemist":       ("chemistry-physical",),
    "chemist":                (
        "chemistry-analytical",
        "chemistry-computational",
        "chemistry-environmental",
        "chemistry-organic",
        "chemistry-physical",
    ),
}


# Static inventory of every tool exposed by each chemistry MCP server.
# Used to translate the per-role server allow-list into the explicit
# `mcp__<server>__<tool>` names that Claude CLI accepts via
# `--allowedTools`. Update this map when adding/removing chemistry
# tools; keep it in sync with tools/chemistry/*_server.py.
CHEMISTRY_SERVER_TOOLS: dict[str, tuple[str, ...]] = {
    "chemistry-analytical": (
        "absorbance_to_concentration",
        "aromatic_ring_count",
        "baseline_correction",
        "calculate_precipitation_pH",
        "chem_visualizer",
        "convert_concentration_units",
        "curve_fit_lm",
        "element_composition_rdkit",
        "h_bond_counts",
        "heavy_atom_count",
        "horwitz_trumpet",
        "inchi_from_smiles",
        "inchikey_from_smiles",
        "intra_laboratory_rsd",
        "lipinski_properties",
        "linear_calibration",
        "logS_rdkit",
        "logp_rdkit",
        "molecular_formula_from_smiles",
        "molecular_formula_rdkit",
        "molecular_weight",
        "molecular_weight_rdkit",
        "peak_finder",
        "pca_decomposition",
        "ph_strong_acid_solution",
        "qed_druglikeness",
        "rotatable_bond_count",
        "serial_dilution",
        "titration_strong_acid_base",
        "topological_polar_surface_area",
    ),
    "chemistry-computational": (
        "chemical_name_to_smiles",
        "compute_planarity",
        "detect_c2_perp_axes",
        "detect_c3_axis",
        "detect_sigma_h",
        "generate_3d_conformer",
        "generate_multiple_conformers_with_optimization",
        "get_3d_properties",
        "init_local_symmetry_db",
        "maxwell_boltzmann_speed",
        "mol_analyzer_generate_3d",
        "numerical_quadrature",
        "optimize_geometry",
        "potential_energy_morse",
        "principal_moment_of_inertia",
        "pubchem_cid_by_name",
        "pubchem_smiles_by_cid",
        "point_group_lookup",
        "query_local_symmetry",
        "rdkit_generate_3d",
        "rotational_partition_function",
        "translational_partition_function",
        "vibrational_partition_function",
    ),
    "chemistry-environmental": (
        "analyze_nitrogen_removal",
        "analyze_waste_composition",
        "apply_dilution_factor",
        "batch_calculate_bod5",
        "bod5_from_titration",
        "calculate_component_contribution",
        "calculate_mass_balance",
        "calculate_oxygen_depletion",
        "calculate_removal_efficiency",
        "calculate_total_nitrogen",
        "dissolved_oxygen_saturation_curve",
        "dissolved_oxygen_winkler",
        "estimate_optimal_dilution",
        "evaluate_nitrification_denitrification",
        "first_order_decay_model",
        "linear_regression_with_residuals",
        "load_json_file",
        "moving_average",
        "normal_distribution_density",
        "save_calculation_report",
        "sum_contributions",
        "t_test_two_sample",
        "validate_bod5_measurement",
        "validate_total_percentage",
        "validate_waste_component",
        "waste_calorific_value",
    ),
    "chemistry-organic": (
        "amide_hydrolysis_product",
        "ester_hydrolysis_product",
        "generate_product_via_smarts",
        "parse_smiles_rdkit",
        "predict_hydrolysis_product",
        "predict_logp_rdkit",
        "predict_organic_reaction_products",
        "predict_smarts_product",
        "sn_product_predict",
    ),
    "chemistry-physical": (
        "analyze_cathode_reaction",
        "arrhenius",
        "beer_lambert",
        "boiling_point_elevation",
        "clausius_clapeyron",
        "compute_bond_energy_sum",
        "compute_products_from_mixture",
        "equilibrium_constant_from_gibbs",
        "first_order_half_life",
        "gibbs_free_energy",
        "henderson_hasselbalch",
        "ideal_gas",
        "ideal_gas_calculation",
        "kinetics_solver",
        "langmuir_isotherm",
        "molality_to_molarity",
        "nernst_equation",
        "partial_pressure",
        "phase_equilibrium_calculator",
        "quantum_energy_levels",
        "reaction_kinetics_solver",
        "reaction_quotient",
        "rdkit_all_descriptors",
        "solve_partition_problem",
        "solve_remaining_concentration",
        "thermodynamic_properties",
        "vant_hoff",
    ),
}


# ---- Physics ---------------------------------------------------------
# Sub-experts and their single pinned server. The generic "physicist"
# role keeps access to every physics server for backward compatibility
# (analogous to the legacy "chemist" all-of-chemistry fallback).
ROLE_ALLOWED_PHYSICS_SERVERS: dict[str, tuple[str, ...]] = {
    "classical-mechanicist":    ("physics-classical",),
    "em-optics-physicist":      ("physics-electromagnetism",),
    "waves-fluid-physicist":    ("physics-waves-fluid",),
    "condensed-matter-physicist": ("physics-condensed-matter",),
    "quantum-atomic-physicist": ("physics-quantum-atomic",),
    "physicist": (
        "physics-classical",
        "physics-electromagnetism",
        "physics-waves-fluid",
        "physics-condensed-matter",
        "physics-quantum-atomic",
    ),
}


# Per-server inventory of physics tools. Populated as the
# `tools/physics/*_server.py` files are written.
PHYSICS_SERVER_TOOLS: dict[str, tuple[str, ...]] = {
    "physics-classical": (
        'kinematics_displacement',
        'projectile_motion',
        'newtons_second_law',
        'kinetic_energy',
        'gravitational_potential_energy',
        'spring_potential_energy',
        'energy_conservation',
        'torque',
        'static_equilibrium',
        'centripetal_force',
        'conical_pendulum',
        'atwood_machine',
        'inclined_plane',
        'mass_spring_frequency',
        'damping_coefficient_to_quality',
        'driven_oscillator_amplitude',
        'lagrangian',
        'generalized_momentum',
        'hamiltonian',
        'lorentz_factor',
        'time_dilation',
        'length_contraction',
        'relativistic_energy_momentum',
        'slider_crank_kinematics',
        'final_speed_from_drop',
        'pendulum_period',
        'plot_trajectory',
        'plot_energy_vs_time',
    ),
    "physics-electromagnetism": (
        'series_resistance',
        'parallel_resistance',
        'wheatstone_bridge',
        'ohms_law',
        'infinite_wire_field',
        'solenoid_field',
        'parallel_wire_force',
        'magnetic_moment_loop',
        'relative_permeability',
        'capacitor_energy',
        'capacitor_charge',
        'electric_force_on_charge',
        'photon_energy_from_wavelength',
        'photon_energy_from_frequency',
        'wavelength_to_color',
        'fresnel_reflectance',
        'snells_law',
        'thin_film_optical_path_difference',
        'thin_film_interference_wavelengths',
        'thin_film_reflectance_spectrum',
        'minimum_antireflection_thickness',
        'single_slit_min_angle',
        'double_slit_max_angle',
        'plot_b_field_vs_distance',
        'plot_thin_film_spectrum',
    ),
    "physics-waves-fluid": (
        'spl_from_pressure',
        'pressure_from_spl',
        'spl_add_two',
        'spl_total',
        'spl_at_distance',
        'doppler_shift',
        'doppler_blood_velocity',
        'stark_first_order_shift',
        'hydrogen_transition_wavelength',
        'rabi_frequency',
        'hydrostatic_pressure',
        'fluid_tilt_angle',
        'buoyant_force',
        'continuity_equation',
        'contact_angle_hysteresis',
        'wenzel_roughness_estimate',
        'young_dupre',
        'manning_velocity',
        'design_pipe_diameter',
        'pipe_velocity',
        'total_variation_coefficient',
        'plot_spl_vs_distance',
        'plot_doppler_profile',
    ),
    "physics-condensed-matter": (
        'ising_1d_partition',
        'ising_mean_field_magnetization',
        'heisenberg_magnetization_vs_T',
        'heisenberg_staggered_magnetization',
        'anderson_1d',
        'anderson_localization_length',
        'ssh_gap',
        'ssh_winding_number',
        'ssh_edge_state_energy',
        'hubbard_double_occupancy',
        'hubbard_charge_gap',
        'bernoulli_function',
        'thermal_voltage',
        'scharfetter_gummel_electron_current',
        'tight_binding_1d',
        'tight_binding_2d_square',
        'diamond_bond_angles',
        'curie_weiss_susceptibility',
        'plot_tight_binding_1d',
        'plot_magnetisation_vs_temperature',
    ),
    "physics-quantum-atomic": (
        'parse_matrix_string',
        'is_hermitian',
        'is_unitary',
        'matrix_trace',
        'matrix_exponential',
        'pauli_matrices',
        'spin_density_matrix',
        'expectation_value',
        'hydrogen_dipole_matrix_element',
        'check_dipole_selection_rule',
        'invariant_mass_squared',
        'threshold_energy',
        'cnot_gate',
        'anti_cnot_gate',
        'quantum_fisher_information_pure',
        'saha_ionization_ratio',
        'mhd_pipe_flow',
        'axial_stress',
        'hoop_stress',
        'torsion_angle_per_length',
        'euler_critical_load',
        'cantilever_deflection_point_load',
        'simply_supported_center_moment',
        'ideal_gas',
        'maxwell_boltzmann_avg_ke',
        'maxwell_boltzmann_most_probable_speed',
        'carnot_efficiency',
        'clausius_clapeyron',
        'nucleation_critical_radius',
        'fourier_heat_flux',
        'stark_ionization_change_factor',
        'plot_hydrogen_energy_levels',
        'plot_maxwell_boltzmann_speed_distribution',
    ),
}


# ---- Biology ---------------------------------------------------------
ROLE_ALLOWED_BIOLOGY_SERVERS: dict[str, tuple[str, ...]] = {
    "molecular-biologist":   ("biology-molecular",),
    "geneticist":            ("biology-genetics",),
    "cell-biologist":        ("biology-cell",),
    "structural-biologist":  ("biology-structural",),
    "mass-spectrometrist":   ("biology-mass-spec",),
    "biologist": (
        "biology-molecular",
        "biology-genetics",
        "biology-cell",
        "biology-structural",
        "biology-mass-spec",
    ),
}


BIOLOGY_SERVER_TOOLS: dict[str, tuple[str, ...]] = {
    "biology-molecular": (
        "analyze_qrt_pcr",
        "query_restriction_enzyme",
        "simulate_restriction_digest",
        "analyze_digest_pattern",
        "check_methylation_sensitivity",
        "calculate_dna_quality_metrics",
        "query_plasmid_properties",
        "calculate_plasmid_compatibility",
        "calculate_transformation_difficulty",
        "recommend_alternative_plasmid",
        "analyze_exon_frame_shift",
        "simulate_morpholino_binding",
        "analyze_dmd_mutation_therapy",
        "query_rna_structure_involvement",
        "calculate_rna_concentration",
        "translate_dna_to_protein",
        "plot_qpcr_curve",
        "plot_gel_simulation",
    ),
    "biology-genetics": (
        "calculate_epistasis_coefficient",
        "analyze_single_mutant_effects",
        "determine_epistatic_relationship",
        "identify_transcription_factor",
        "analyze_gene_redundancy",
        "calculate_spindle_tension",
        "simulate_chromosome_segregation",
        "analyze_mitotic_behavior",
        "evaluate_chromosome_stability_factors",
        "fetch_snp_from_ncbi",
        "fetch_flanking_sequence_ensembl",
        "analyze_sequence_composition",
        "format_sequence_with_spacing",
        "chromosome_metrics",
        "plot_epistasis_heatmap",
        "plot_spindle_tension",
    ),
    "biology-cell": (
        "calculate_atp_concentration",
        "calculate_complex_activity",
        "assess_glucose_uptake_relevance",
        "comprehensive_mitochondrial_assessment",
        "experimental_design_validator",
        "calculate_polycomb_enrichment_score",
        "analyze_enhancer_promoter_distance_distribution",
        "simulate_polycomb_knockout_effect",
        "calculate_crosslinking_efficiency",
        "simulate_chipseq_signal",
        "predict_peak_locations",
        "analyze_disappearing_peaks",
        "plot_mitochondrial_assays",
        "plot_chipseq_profile",
    ),
    "biology-structural": (
        "parse_rna_sequence",
        "detect_base_pairs",
        "calculate_structure_complexity",
        "classify_rna_type",
        "predict_catalytic_activity",
        "get_amino_acid_properties",
        "calculate_mutation_effect_score",
        "analyze_binding_site_residue",
        "compare_mutation_candidates",
        "analyze_active_site_composition",
        "parse_pdb_structure",
        "classify_residue_type",
        "count_ligand_chains",
        "analyze_structure_composition",
        "query_nsp_protein_info",
        "analyze_protein_complex_formation",
        "comprehensive_nsp_analysis",
        "plot_mutation_impact_ranking",
        "plot_rna_arc_diagram",
    ),
    "biology-mass-spec": (
        "calculate_theoretical_isotope_pattern",
        "determine_chlorine_number_from_ratio",
        "extract_peaks_from_spectrum",
        "find_isotope_cluster",
        "analyze_spectrum_for_chlorine",
        "calculate_molecular_properties",
        "predict_fragmentation_pattern",
        "match_spectrum_to_structure",
        "analyze_spectrum_characteristics",
        "batch_structure_screening",
        "plot_mass_spectrum",
        "plot_isotope_pattern",
    ),
}


# ---- Mathematics -----------------------------------------------------
ROLE_ALLOWED_MATH_SERVERS: dict[str, tuple[str, ...]] = {
    "algebraic-mathematician":     ("math-algebraic",),
    "statistical-mathematician":   ("math-statistical",),
    "geometric-mathematician":     ("math-geometric",),
    "optimization-mathematician":  ("math-optimization",),
    "numerical-mathematician":     ("math-numerical",),
    "mathematician": (
        "math-algebraic",
        "math-statistical",
        "math-geometric",
        "math-optimization",
        "math-numerical",
    ),
}


MATH_SERVER_TOOLS: dict[str, tuple[str, ...]] = {
    "math-algebraic": (
        "parse_matrix_string",
        "is_hermitian",
        "is_unitary",
        "matrix_properties",
        "matrix_exponential",
        "hermitian_eigendecomposition",
        "compute_degree_matrix",
        "compute_graph_laplacian",
        "compute_random_walk_laplacian",
        "compute_laplacian_eigenpairs",
        "compute_gaussian_affinity",
        "compute_sandwich_covariance",
        "solve_linear_system",
        "plot_matrix_heatmap",
        "plot_eigenvalue_spectrum",
    ),
    "math-statistical": (
        "estimate_wls_parameters",
        "compute_sandwich_covariance_estimator",
        "simulate_stationary_time_series",
        "two_sample_z_test",
        "welch_t_test",
        "confidence_interval_for_mean",
        "estimate_eif_psi",
        "estimate_mediation_parameter_psi",
        "bootstrap_standard_error",
        "correlation_with_ci",
        "plot_sample_distribution",
        "plot_bootstrap_distribution",
    ),
    "math-geometric": (
        "sample_sphere",
        "sample_torus",
        "build_graph_laplacian_from_samples",
        "sphere_laplacian_eigenvalue",
        "evaluate_sphere_eigenfunction",
        "compute_sampling_vector",
        "compute_alignment_sign",
        "compute_euclidean_error",
        "theoretical_error_bound",
        "compute_spectral_convergence_error",
        "analyze_bandwidth_scaling",
        "plot_sphere_samples",
        "plot_bandwidth_scaling",
    ),
    "math-optimization": (
        "quantum_fisher_information_pure",
        "bisection_root",
        "newton_raphson_step",
        "fixed_point_iteration",
        "gradient_descent_step",
        "project_box",
        "quadratic_gradient",
        "quadratic_hessian",
        "convexity_check",
        "linear_program_2d",
        "armijo_step_size",
        "plot_descent_trajectory",
        "plot_convexity_landscape",
    ),
    "math-numerical": (
        "numerical_midpoint_quadrature",
        "numerical_trapezoid_quadrature",
        "numerical_simpson_quadrature",
        "gauss_legendre_2pt",
        "rk4_integrate",
        "euler_integrate",
        "matrix_condition_number",
        "finite_difference_derivative",
        "running_mean_smooth",
        "richardson_extrapolation",
        "adaptive_simpson",
        "spectral_radius",
        "plot_quadrature_overlay",
        "plot_ode_trajectory",
        "plot_quadrature_convergence",
    ),
}


# ---- Pharmaceutical / medical drug discovery ------------------------
# ``pharma-drug-sda`` proxies 81 tools on a remote SCP server. Which role
# may reach which of them is decided by the ``pharma-drug-sda-*`` skills
# (see ``scripts/drug_sda_codegen.py``), not by this map: skill routing is
# on by default, so a role here with matching skills gets exactly its own
# skills' tools. The four roles below are the ones the codegen assigns
# skills to; ``pharmacist`` is the legacy broad role that spans them.
ROLE_ALLOWED_PHARMA_SERVERS: dict[str, tuple[str, ...]] = {
    "drug-discovery-scientist": ("pharma-drug-discovery", "pharma-drug-sda"),
    "structural-biologist": ("pharma-drug-sda",),
    "computational-chemist": ("pharma-drug-sda",),
    "physical-chemist": ("pharma-drug-sda",),
    "pharma-data-specialist": ("pharma-data",),
    "pharma-ml-engineer": ("pharma-automl",),
    "pharmacist": (
        "pharma-drug-discovery",
        "pharma-data",
        "pharma-automl",
        "pharma-drug-sda",
    ),
}


PHARMA_SERVER_TOOLS: dict[str, tuple[str, ...]] = {
    "pharma-drug-discovery": (
        "evaluate_druglikeness",
        "list_madd_local_checkpoints",
        "predict_with_local_madd_checkpoint",
        "generate_molecules_with_local_madd",
        "draw_molecules",
        "generate_molecules_by_case",
        "predict_properties_by_smiles",
    ),
    # "pharma-drug-sda" is generated: its entry is merged in below from
    # tools/pharma/_manifest.json, next to the BIOMNI inventory.
    "pharma-data": (
        "inspect_dataset_columns",
        "filter_dataset_columns",
        "fetch_bindingdb_affinity",
        "fetch_chembl_activities",
    ),
    "pharma-automl": (
        "madd_server_state",
        "madd_case_state",
        "start_madd_ml_training",
        "start_madd_generative_training",
    ),
}


# ---- Web / literature retrieval -------------------------------------
ROLE_ALLOWED_WEB_SERVERS: dict[str, tuple[str, ...]] = {
    "literature-searcher": ("litsearch",),
}


WEB_SERVER_TOOLS: dict[str, tuple[str, ...]] = {
    "litsearch": (
        "search_literature",
    ),
}


# ---- BiOMNI wrappers --------------------------------------------------
# Migration is staged across Phase 1/2/3. Each server is generated by
# ``scripts/biomni_codegen.py`` and maps BiOMNI's tool categories onto
# existing sciMAS roles (no new role names are introduced). The inventory
# below is loaded from ``tools/biomni/_manifest.json`` (the codegen writes
# that file alongside the servers) so there is exactly one source of
# truth. To migrate another batch, add a category to the appropriate
# ``PHASE_N_BUNDLES`` dict in the codegen and rerun.
ROLE_ALLOWED_BIOMNI_SERVERS: dict[str, tuple[str, ...]] = {
    # Phase 1 — light deps.
    "molecular-biologist": (
        "biomni-biochemistry",
        "biomni-protocols",
        # Phase 2 — molecular-biology + synthetic-biology land on the
        # same role (cloning / gene-design workflow).
        "biomni-molecular_biology",
        "biomni-synthetic_biology",
    ),
    "literature-searcher": ("biomni-literature",),
    "pharma-data-specialist": ("biomni-database",),
    "drug-discovery-scientist": ("biomni-pharmacology",),
    "geneticist": (
        # Phase 2 — genetics (CRISPR / phylogeny / demographic sim).
        "biomni-genetics",
        # Phase 3 — genomics (scRNA-seq / ChIP-seq / ESM embeddings /
        # variant annotation).
        "biomni-genomics",
    ),
    "structural-biologist": (
        # Phase 2 — systems-biology (FBA + signaling dynamics).
        "biomni-systems_biology",
    ),
    "cell-biologist": (
        # Phase 3 — microscopy, FACS, cancer, immunology.
        "biomni-bioimaging",
        "biomni-cell_biology",
        "biomni-cancer_biology",
        "biomni-immunology",
    ),
    # Legacy broad-role fallbacks.
    "biologist": (
        "biomni-biochemistry",
        "biomni-protocols",
        "biomni-molecular_biology",
        "biomni-synthetic_biology",
    ),
    "pharmacist": (
        "biomni-database",
        "biomni-pharmacology",
    ),
}


def _load_generated_inventory(relative_path: str) -> dict[str, tuple[str, ...]]:
    """Read a codegen-written ``_manifest.json`` into {server: tools}.

    The codegen is the only writer of these files, so reading them here
    means there is exactly one source of truth for which tools each
    generated server exposes. If a manifest is missing (e.g. running
    before the codegen has been executed) we fall back to an empty
    dict; that triggers an obvious ImportError the first time an agent
    tries to call one of its tools rather than silently drifting.
    """
    manifest_path = Path(__file__).resolve().parent / relative_path
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory: dict[str, tuple[str, ...]] = {}
    for category, meta in payload.items():
        server = meta.get("server") or category
        inventory[server] = tuple(meta.get("tools", ()))
    return inventory


BIOMNI_SERVER_TOOLS: dict[str, tuple[str, ...]] = _load_generated_inventory(
    "tools/biomni/_manifest.json"
)

# The DrugSDA proxy server's 81 tools come from the same codegen pattern.
PHARMA_SERVER_TOOLS.update(
    _load_generated_inventory("tools/pharma/_manifest.json")
)


@dataclass(frozen=True)
class ScimasSkill:
    skill_id: str
    name: str
    description: str
    role: str
    server: str
    tools: tuple[str, ...]
    path: Path
    body: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def allowed_tools(self) -> list[str]:
        return [f"mcp__{self.server}__{tool}" for tool in self.tools]


def _parse_frontmatter_list(lines: list[str], start: int) -> tuple[list[str], int]:
    values: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if not line.startswith((" ", "\t", "-")) and ":" in line:
            break
        if stripped.startswith("- "):
            values.append(stripped[2:].strip().strip('"').strip("'"))
        i += 1
    return values, i


def _parse_skill_file(path: Path) -> ScimasSkill | None:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end >= 0:
            frontmatter = text[4:end].strip("\n")
            body = text[end + 4 :].lstrip("\n")
            lines = frontmatter.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if not line.strip() or line.lstrip().startswith("#"):
                    i += 1
                    continue
                if ":" not in line:
                    i += 1
                    continue
                key, raw_value = line.split(":", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                if raw_value:
                    metadata[key] = raw_value.strip('"').strip("'")
                    i += 1
                else:
                    values, next_i = _parse_frontmatter_list(lines, i + 1)
                    metadata[key] = values
                    i = next_i

    skill_id = str(metadata.get("name") or path.parent.name)
    role = normalize_role_name(str(metadata.get("x-scimas-role", "")))
    server = str(metadata.get("x-scimas-server", ""))
    tools_value = metadata.get("x-scimas-tools", [])
    tools = tuple(str(item) for item in tools_value if str(item).strip())
    if not role or not server or not tools:
        return None
    return ScimasSkill(
        skill_id=skill_id,
        name=skill_id,
        description=str(metadata.get("description", "")).strip(),
        role=role,
        server=server,
        tools=tools,
        path=path,
        body=body.strip(),
        metadata=metadata,
    )


def load_scimas_skills(skill_dir: Path) -> dict[str, ScimasSkill]:
    if not skill_dir.exists():
        return {}
    skills: dict[str, ScimasSkill] = {}
    for path in sorted(skill_dir.glob("*/SKILL.md")):
        skill = _parse_skill_file(path)
        if skill is not None:
            skills[skill.skill_id] = skill
    return skills


# Backwards-compatible alias for external code that imported the old name.
load_chemistry_skills = load_scimas_skills


def _skill_role_scope(role_name: str) -> set[str]:
    role_name = normalize_role_name(role_name)
    broad_skill_roles = {
        "chemist": tuple(r for r in ROLE_ALLOWED_CHEMISTRY_SERVERS if r != "chemist"),
        "physicist": tuple(r for r in ROLE_ALLOWED_PHYSICS_SERVERS if r != "physicist"),
        "biologist": tuple(r for r in ROLE_ALLOWED_BIOLOGY_SERVERS if r != "biologist"),
        "mathematician": tuple(r for r in ROLE_ALLOWED_MATH_SERVERS if r != "mathematician"),
        "pharmacist": tuple(r for r in ROLE_ALLOWED_PHARMA_SERVERS if r != "pharmacist"),
    }
    return {role_name, *broad_skill_roles.get(role_name, ())}


def skills_for_role(
    role: str, scimas_skills: dict[str, ScimasSkill]
) -> list[ScimasSkill]:
    role_name = normalize_role_name(role)
    scoped_roles = _skill_role_scope(role_name)
    return sorted(
        (skill for skill in scimas_skills.values() if skill.role in scoped_roles),
        key=lambda item: item.skill_id,
    )


# All per-discipline allow-list maps and tool inventories. Each
# sub-expert role belongs to exactly one of these disciplines and is
# pinned to exactly one server. The legacy "all-of-discipline" fallback
# roles (chemist, physicist, biologist, mathematician, pharmacist) get
# every server in their discipline. Roles outside these disciplines get
# nothing.
ROLE_ALLOWED_BY_DISCIPLINE: tuple[tuple[str, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]], ...] = (
    ("chemistry", ROLE_ALLOWED_CHEMISTRY_SERVERS, CHEMISTRY_SERVER_TOOLS),
    ("physics",   ROLE_ALLOWED_PHYSICS_SERVERS,   PHYSICS_SERVER_TOOLS),
    ("biology",   ROLE_ALLOWED_BIOLOGY_SERVERS,   BIOLOGY_SERVER_TOOLS),
    ("math",      ROLE_ALLOWED_MATH_SERVERS,      MATH_SERVER_TOOLS),
    ("pharma",    ROLE_ALLOWED_PHARMA_SERVERS,    PHARMA_SERVER_TOOLS),
    ("web",       ROLE_ALLOWED_WEB_SERVERS,       WEB_SERVER_TOOLS),
    ("biomni",    ROLE_ALLOWED_BIOMNI_SERVERS,    BIOMNI_SERVER_TOOLS),
)


DEFAULT_PLANNER_ROLES: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            "generalist",
            *ROLE_ALLOWED_CHEMISTRY_SERVERS.keys(),
            *ROLE_ALLOWED_PHYSICS_SERVERS.keys(),
            *ROLE_ALLOWED_BIOLOGY_SERVERS.keys(),
            *ROLE_ALLOWED_MATH_SERVERS.keys(),
            *ROLE_ALLOWED_PHARMA_SERVERS.keys(),
            *ROLE_ALLOWED_WEB_SERVERS.keys(),
            *ROLE_ALLOWED_BIOMNI_SERVERS.keys(),
        ]
    )
)


def _discipline_allow_list(role_name: str) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """Return (allowed_servers, tool_inventory) for a role across all disciplines."""
    allowed_servers: list[str] = []
    tool_inventory: dict[str, tuple[str, ...]] = {}
    for _name, allow_map, tool_map in ROLE_ALLOWED_BY_DISCIPLINE:
        servers = allow_map.get(role_name, ())
        if servers:
            allowed_servers.extend(servers)
            for server in servers:
                tool_inventory[server] = tool_map.get(server, ())
    return allowed_servers, tool_inventory


def allowed_tools_for_role(
    role: str,
    selected_skills: Optional[Iterable[str]] = None,
    scimas_skills: Optional[dict[str, ScimasSkill]] = None,
    allow_role_fallback: bool = True,
) -> list[str]:
    """Return the per-role `--allowedTools` list.

    For roles that own MCP servers (chemistry, physics, biology,
    mathematics), returns the explicit `mcp__<server>__<tool>` names.
    For roles without MCP access, returns an empty list — they retain
    built-in tools (Read, Bash, etc.) but cannot invoke any MCP tool.

    When ``allow_role_fallback`` is false, an empty skill selection means
    no MCP tools rather than the whole broad-role tool inventory. This
    keeps skill-routed runs specialized: first select one or two skills,
    then expose only the tools declared by those skills.
    """
    role_name = normalize_role_name(role)
    if selected_skills and scimas_skills:
        allowed: list[str] = []
        scoped_roles = _skill_role_scope(role_name)
        for skill_id in selected_skills:
            skill = scimas_skills.get(str(skill_id))
            if skill is None:
                continue
            if skill.role not in scoped_roles:
                continue
            allowed.extend(skill.allowed_tools())
        if allowed:
            return sorted(dict.fromkeys(allowed))

    if not allow_role_fallback:
        return []

    servers, tool_map = _discipline_allow_list(role_name)
    if not servers:
        return []
    tools: list[str] = []
    for server in servers:
        for tool in tool_map.get(server, ()):
            tools.append(f"mcp__{server}__{tool}")
    return tools


def _truncate(text: str, n: int) -> str:
    """Truncate text to n chars, appending a marker if cut."""
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[:n] + f"\n... [truncated, {len(text) - n} more chars]"


def _format_workflow_result(value: Any) -> str:
    value = unwrap_workflow_value(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(make_json_safe(value), ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _upstream_public_outputs(input_value: Any) -> list[dict[str, Any]]:
    return [
        {
            "node_id": item.node_id,
            "role": item.role,
            "public_output": make_json_safe(item.value),
        }
        for item in collect_workflow_values(input_value)
    ]


def _history_by_step_id(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("step_id")): item for item in history}


def _explicit_history_for_step(
    history: list[dict[str, Any]],
    step: PlanStep,
) -> list[dict[str, Any]]:
    if not step.depends_on:
        return []
    lookup = _history_by_step_id(history)
    return [
        lookup[step_id]
        for step_id in step.depends_on
        if step_id in lookup
    ]


class _WorkflowRuntime:
    """Restricted runtime bindings exposed to a validated workflow."""

    def __init__(
        self,
        orchestrator: "SciMASOrchestrator",
        problem: str,
        plan: ExecutionPlan,
    ) -> None:
        self.orchestrator = orchestrator
        self.problem = problem
        self.plan = plan
        self.runs: list[AgentRun] = []
        self.history: list[dict[str, Any]] = []
        self.session_ids: dict[str, str] = {}
        self._lock = threading.Lock()
        self._agent_counter = 0
        self._parallel_counter = 0
        self._parallel_group: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar("workflow_parallel_group", default=None)
        )

    async def run(self, source: str) -> Any:
        program = parse_and_validate_workflow(
            source,
            role_check=self.orchestrator._workflow_role_check,
            allowed_roles=self.orchestrator.active_roles,
        )
        self.plan.workflow_source = program.source
        self.plan.workflow_static_trace = program.static_trace
        namespace: dict[str, Any] = {
            "__builtins__": {},
            "agent": self.agent,
            "parallel": self.parallel,
            "range": range,
        }
        code = compile(program.tree, "<sciMAS-workflow>", "exec")
        exec(code, namespace, namespace)
        workflow = namespace.get("workflow")
        if workflow is None:
            raise RuntimeError("Validated workflow did not define workflow().")
        return await workflow(self.problem)

    async def agent(
        self,
        *,
        role: str,
        instruction: str,
        input: Any = None,
    ) -> WorkflowValue:
        return await asyncio.to_thread(
            self._run_agent_sync,
            role,
            instruction,
            input,
        )

    async def parallel(self, *awaitables: Any) -> tuple[Any, ...]:
        group_id = self._next_parallel_id()
        token = self._parallel_group.set(group_id)
        try:
            results = await asyncio.gather(*awaitables)
        finally:
            self._parallel_group.reset(token)
        children = [
            item.node_id for item in results if isinstance(item, WorkflowValue)
        ]
        self._append_trace(
            {
                "event": "parallel",
                "id": group_id,
                "children": children,
            }
        )
        self.orchestrator._emit_event(
            {
                "event": "parallel_finished",
                "parallel_id": group_id,
                "children": children,
            }
        )
        return tuple(results)

    def _next_agent_id(self) -> str:
        with self._lock:
            self._agent_counter += 1
            return f"node-{self._agent_counter}"

    def _next_parallel_id(self) -> str:
        with self._lock:
            self._parallel_counter += 1
            return f"parallel-{self._parallel_counter}"

    def _append_trace(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.plan.workflow_trace.append(event)

    def _history_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.history)

    def _run_agent_sync(
        self,
        role: str,
        instruction: str,
        input_value: Any = None,
    ) -> WorkflowValue:
        node_id = self._next_agent_id()
        role_name = self.orchestrator.normalize_role(role)
        dependencies = collect_workflow_dependencies(input_value)
        upstream_outputs = _upstream_public_outputs(input_value)
        dsl_input = make_json_safe(input_value)
        step = PlanStep(
            id=node_id,
            role=role_name,
            objective=instruction,
            depends_on=dependencies,
            expected_output="Result requested by the Python workflow DSL.",
        )
        with self._lock:
            self.plan.steps.append(step)
            self.plan.execution_order.append(node_id)
        self.orchestrator._emit_event(
            {
                "event": "agent_started",
                "node_id": node_id,
                "role": role_name,
                "instruction": instruction,
                "received_from": dependencies,
                "parallel_group": self._parallel_group.get(),
            }
        )

        history_snapshot = [
            {
                "step_id": item["node_id"],
                "role": item["role"],
                "objective": "",
                "result": item["public_output"],
                "public_output": item["public_output"],
            }
            for item in upstream_outputs
        ]
        selected_skills, skill_selection_result, had_skill_candidates = (
            self.orchestrator._select_skills_for_step(
                role_name,
                self.problem,
                self.plan,
                step,
                history_snapshot,
            )
        )
        selected_skill_ids = [skill.skill_id for skill in selected_skills]
        step_prompt = self.orchestrator._step_prompt(
            role_name,
            step,
            self.plan,
            history_snapshot,
            selected_skills=selected_skills,
        )
        context = self.orchestrator._context_bundle(
            self.problem,
            self.plan,
            step,
            history_snapshot,
            dsl_input=dsl_input,
            received_upstream_outputs=upstream_outputs,
        )
        parallel_group = self._parallel_group.get()
        previous_session = None
        step_allowed_tools = allowed_tools_for_role(
            role_name,
            selected_skills=selected_skill_ids,
            scimas_skills=self.orchestrator.skills,
            allow_role_fallback=not (
                self.orchestrator.use_skill_routing
                and self.orchestrator.strict_skill_routing
                and had_skill_candidates
            ),
        )
        result = self.orchestrator.runner.run(
            prompt=step_prompt,
            stdin_text=context,
            session_id=previous_session,
            output_format="stream-json",
            allowed_tools=step_allowed_tools,
        )
        if result.session_id:
            with self._lock:
                self.session_ids[node_id] = result.session_id

        stdin_for_log = "" if self.orchestrator.runner.mcp_config_path else context
        public_output = coerce_agent_result(result.result)
        agent_result = AgentResult(
            output=public_output,
            node_id=node_id,
            role=role_name,
            received_from=dependencies,
        )
        run = AgentRun(
            role=role_name,
            step_id=node_id,
            prompt_path=str(self.orchestrator.prompt_dir / f"{role_name}.md"),
            assembled_prompt=step_prompt,
            context_bundle=context,
            stdin_text=stdin_for_log,
            selected_skills=selected_skill_ids,
            skill_selection_result=skill_selection_result,
            allowed_tools=step_allowed_tools,
            session_id=result.session_id,
            resumed_from_session=previous_session,
            result=result.result,
            raw_stdout=result.raw_stdout,
            raw_json=result.raw_json,
            started_at=result.started_at,
            duration_ms=result.duration_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.total_cost_usd,
            stderr=result.stderr,
            public_output=make_json_safe(agent_result.output),
            received_from=agent_result.received_from,
            tool_calls=list(result.tool_calls),
        )
        value = WorkflowValue(
            node_id=node_id,
            value=agent_result.output,
            role=role_name,
        )
        with self._lock:
            self.runs.append(run)
            self.history.append(
                {
                    "step_id": node_id,
                    "role": role_name,
                    "objective": instruction,
                    "input_dependencies": dependencies,
                    "received_from": dependencies,
                    "result": make_json_safe(agent_result.output),
                    "public_output": make_json_safe(agent_result.output),
                    "session_id": result.session_id,
                }
            )
            self.plan.workflow_trace.append(
                {
                    "event": "agent",
                    "id": node_id,
                    "role": role_name,
                    "instruction": instruction,
                    "input_dependencies": dependencies,
                    "received_from": dependencies,
                    "received_upstream_outputs": upstream_outputs,
                    "public_output": make_json_safe(agent_result.output),
                    "parallel_group": parallel_group,
                }
            )
        self.orchestrator._emit_event(
            {
                "event": "agent_finished",
                "node_id": node_id,
                "role": role_name,
                "instruction": instruction,
                "received_from": dependencies,
                "received_upstream_outputs": upstream_outputs,
                "public_output": make_json_safe(agent_result.output),
                "parallel_group": parallel_group,
            }
        )
        return value


class SciMASOrchestrator:
    def __init__(
        self,
        prompt_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        skill_dir: Optional[Path] = None,
        use_skill_routing: bool = True,
        strict_skill_routing: bool = True,
        claude_bin: str = "claude",
        model: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        permission_mode: str = "bypassPermissions",
        dangerously_skip_permissions: bool = True,
        extra_system_prompt: Optional[str] = None,
        timeout: Optional[float] = 600.0,
        planner_mode: str = "python_dsl",
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        max_review_attempts: int = 3,
    ):
        if planner_mode not in {"python_dsl", "legacy_json"}:
            raise ValueError("planner_mode must be 'python_dsl' or 'legacy_json'.")
        if max_review_attempts < 1:
            raise ValueError("max_review_attempts must be at least 1.")
        self.max_review_attempts = max_review_attempts
        # Roles permitted for this run, set by `run()`. Drives workflow role
        # validation so an invented role is caught before execution rather
        # than as a FileNotFoundError from a missing prompt file.
        self.active_roles: tuple[str, ...] = DEFAULT_PLANNER_ROLES
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.output_dir = output_dir or Path(__file__).resolve().parent / "runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skill_dir = skill_dir or Path(__file__).resolve().parent / "scimas_skills"
        self.use_skill_routing = use_skill_routing
        self.strict_skill_routing = strict_skill_routing
        self.planner_mode = planner_mode
        self.event_callback = event_callback
        self.skills = load_scimas_skills(self.skill_dir) if use_skill_routing else {}
        # Build a per-tool spec index from `tools/*/*_server.py`. Eager; one
        # AST walk per process. Used by the skill-selection catalog and the
        # agent-execution prompt to surface tool signatures inline so the
        # model never has to read server source files mid-run.
        self.tool_specs: Dict[str, ToolSpec] = build_tool_spec_index(
            Path(__file__).resolve().parent / "tools"
        )
        self.runner = ClaudeRunner(
            binary=claude_bin,
            model=model,
            mcp_config_path=mcp_config_path,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            dangerously_skip_permissions=dangerously_skip_permissions,
            extra_system_prompt=extra_system_prompt,
            timeout=timeout,
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        if not self.event_callback:
            return
        try:
            self.event_callback(event)
        except Exception:
            # Monitoring must never change MAS execution behavior.
            pass

    def _load_prompt(self, role: str) -> str:
        role_name = self.normalize_role(role)
        path = self.prompt_dir / f"{role_name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt file for role '{role_name}': {path}")
        return path.read_text(encoding="utf-8")

    def _review_and_fix_workflow(
        self,
        workflow_source: str,
        validation_error: Optional[str] = None,
        execution_error: Optional[str] = None,
        attempt_number: int = 1,
        problem: str = "",
        roles: Optional[Iterable[str]] = None,
        previous_reply: str = "",
    ) -> dict[str, Any]:
        """Ask the planner-reviewer agent to validate and fix workflow code.

        `previous_reply` carries a reviewer response that failed to parse, so
        one corrective retry can be made before giving up. Without it a single
        malformed reply ended the run.
        """
        reviewer_prompt = self._load_prompt("planner-reviewer")

        available_roles = sorted(
            {self.normalize_role(role) for role in (roles or self.active_roles or [])}
        )
        context = {
            "workflow_source": workflow_source,
            "validation_error": validation_error,
            "execution_error": execution_error,
            "attempt_number": attempt_number,
            "problem_summary": problem[:500] if problem else "",
            "available_roles": available_roles,
        }
        if previous_reply:
            context["previous_reply"] = previous_reply[:4000]
            context["correction_notice"] = (
                "Your previous reply could not be parsed as JSON. Reply with a single "
                "JSON object and nothing else. Inside string values, escape newlines "
                "as \\n."
            )

        result = self.runner.run(
            prompt=reviewer_prompt,
            stdin_text=json.dumps(context, ensure_ascii=False, indent=2),
            output_format="stream-json",
            allowed_tools=[],
        )

        # Parse the reviewer's response
        review_data = self._extract_json_block(result.result)

        if not review_data and not previous_reply:
            # One corrective retry, echoing the reply that failed to parse.
            # An empty reply still gets a non-empty sentinel: `previous_reply`
            # is both the retry guard and the correction hint, so passing ""
            # back would recurse forever on a reviewer that returns nothing.
            return self._review_and_fix_workflow(
                workflow_source=workflow_source,
                validation_error=validation_error,
                execution_error=execution_error,
                attempt_number=attempt_number,
                problem=problem,
                roles=roles,
                previous_reply=(result.result or "").strip()
                or "(the reviewer returned an empty reply)",
            )

        if not review_data:
            # Still unparseable. Keep the raw reply attached: it is the one
            # datum that makes this failure diagnosable, and it used to be
            # discarded here.
            return {
                "is_valid": False,
                "issues": [
                    "Reviewer did not return valid JSON. Raw reply: "
                    f"{(result.result or '')[:600]!r}"
                ],
                "fixed_source": workflow_source,
                "changes_made": [],
                "confidence": 0.0,
                "raw_reply": result.result or "",
                "stderr": result.stderr or "",
            }

        return review_data

    @staticmethod
    def normalize_role(role: str) -> str:
        return normalize_role_name(role)

    def _workflow_role_check(self, role: str) -> bool:
        """Is `role` one a workflow may use in this run?

        `generalist` and `synthesizer` are always permitted: they have prompts
        and no tool bindings, so a narrow `--roles` list cannot newly break a
        legitimate workflow that falls back to them.
        """

        normalized = self.normalize_role(role)
        return normalized in self.active_roles or normalized in {
            "generalist",
            "synthesizer",
        }

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any]:
        """Parse a JSON object out of a model reply.

        Delegates to `workflow_dsl.extract_json_object`, which tolerates the
        prose, fences and unescaped newlines that models actually emit. This
        used to require the reply to be exactly JSON after stripping a leading
        fence, which silently discarded most real reviewer replies.
        """
        return extract_json_object(text)

    def _planner_prompt(self, problem: str, roles: Iterable[str]) -> str:
        if self.planner_mode == "legacy_json":
            return self._planner_json_prompt(problem, roles)
        prompt = self._load_prompt("planner")
        role_list = ", ".join(sorted({self.normalize_role(role) for role in roles}))
        return (
            f"{prompt}\n\n"
            "Available roles:\n"
            f"{role_list}\n\n"
            "The stdin payload contains the scientific problem.\n"
            "Return only Python source code for async def workflow(task)."
        )

    def _planner_json_prompt(self, problem: str, roles: Iterable[str]) -> str:
        path = self.prompt_dir / "planner_json.md"
        prompt = (
            path.read_text(encoding="utf-8")
            if path.exists()
            else self._legacy_json_planner_prompt_text()
        )
        role_list = ", ".join(sorted({self.normalize_role(role) for role in roles}))
        return (
            f"{prompt}\n\n"
            "Available roles:\n"
            f"{role_list}\n\n"
            "The stdin payload contains the scientific problem.\n"
            "Return strict JSON only."
        )

    @staticmethod
    def _legacy_json_planner_prompt_text() -> str:
        return (
            "You are the planning agent for sciMAS. Inspect the scientific "
            "problem and design a JSON cooperation topology for downstream "
            "specialists. Output strict JSON only with this shape:\n"
            "{\n"
            '  "problem_summary": "short summary",\n'
            '  "topology_rationale": "why this topology fits",\n'
            '  "assumptions": ["..."],\n'
            '  "risks": ["..."],\n'
            '  "steps": [\n'
            "    {\n"
            '      "id": "step-1",\n'
            '      "role": "physicist",\n'
            '      "objective": "what this role should determine",\n'
            '      "depends_on": [],\n'
            '      "expected_output": "what the next role needs",\n'
            '      "handoff_to": ["step-2"],\n'
            '      "confidence": 0.9\n'
            "    }\n"
            "  ],\n"
            '  "execution_order": ["step-1"],\n'
            '  "final_role": "synthesizer",\n'
            '  "termination_criterion": "when the final answer is ready"\n'
            "}\n"
            "Choose the smallest useful set of available roles and keep "
            "dependencies explicit."
        )

    def _step_prompt(
        self,
        role: str,
        step: PlanStep,
        plan: ExecutionPlan,
        history: List[Dict[str, Any]],
        selected_skills: Optional[list[ScimasSkill]] = None,
    ) -> str:
        prompt = self._load_prompt(role)
        skill_block = ""
        if selected_skills:
            parts = [
                "Selected sciMAS skills. Use these instructions and "
                "the tools exposed for them; if the chosen skills are "
                "insufficient, explain the limitation and hand off rather than "
                "inventing unavailable tool behavior."
            ]
            for skill in selected_skills:
                parts.append(
                    f"\n## Skill: {skill.skill_id}\n"
                    f"Source: {skill.path}\n"
                    f"Tools: {', '.join(skill.tools)}\n\n"
                    f"{skill.body}"
                )
                sig_block = self._skill_tool_signatures_block(skill)
                if sig_block:
                    parts.append(sig_block)
            skill_block = "\n\n" + "\n".join(parts)
        return (
            f"{prompt}{skill_block}\n\n"
            "The stdin payload contains a JSON context bundle with the problem, "
            "the planner topology, the current step, and prior outputs.\n"
            "Stay focused on the current scientific subtask.\n"
            "Return concise, useful work only."
        )

    def _skill_tool_signatures_block(self, skill: ScimasSkill) -> str:
        """Render a `### Tool signatures` section listing every tool with its
        full signature and description. Returns '' if no specs resolve — caller
        skips the section entirely (no half-orphaned header)."""
        specs: list[ToolSpec] = []
        for tool in skill.tools:
            mcp_name = f"mcp__{skill.server}__{tool}"
            spec = self.tool_specs.get(mcp_name)
            if spec is None:
                return ""
            specs.append(spec)
        if not specs:
            return ""
        lines = ["### Tool signatures"]
        for spec in specs:
            desc = _truncate(spec.description, 600) if spec.description else "(no description)"
            lines.append(f"- `{spec.mcp_name}` → {spec.one_liner()}")
            if desc:
                lines.append(f"  {desc}")
        return "\n\n".join(lines)

    def _skill_selection_prompt(
        self,
        role: str,
        available_skills: list[ScimasSkill],
    ) -> str:
        prompt = self._load_prompt(role)
        catalog_lines = []
        for skill in available_skills:
            head_line = (
                f"- {skill.skill_id}: {skill.description} "
                f"(tools: {', '.join(skill.tools)})"
            )
            sig_lines = self._skill_tool_signature_lines(skill, max_tools=5)
            if sig_lines:
                head_line += "\n  Tools:\n    " + "\n    ".join(sig_lines)
            catalog_lines.append(head_line)
        return (
            f"{prompt}\n\n"
            "You are choosing which sciMAS skill packages are needed "
            "for the current step. Do not solve the problem yet.\n\n"
            "Available skill packages:\n"
            + "\n".join(catalog_lines)
            + "\n\n"
            "Return strict JSON only, with this shape:\n"
            "{\n"
            '  "selected_skills": ["skill-id"],\n'
            '  "reason": "brief reason",\n'
            '  "confidence": 0.0\n'
            "}\n"
            "Select the smallest useful set, usually one skill and at most two. "
            "Use an empty list only if none fit."
        )

    def _skill_tool_signature_lines(
        self,
        skill: ScimasSkill,
        max_tools: int = 5,
    ) -> list[str]:
        """Render compact one-line `name(params) -> ret` lines for the catalog.

        Returns an empty list if no specs resolve for the skill — caller should
        treat that as "no spec block" and fall back to today's behavior.
        """
        specs: list[ToolSpec] = []
        for tool in skill.tools:
            mcp_name = f"mcp__{skill.server}__{tool}"
            spec = self.tool_specs.get(mcp_name)
            if spec is None:
                return []  # partial coverage → drop the section to avoid confusing two
            specs.append(spec)
        lines = [spec.one_liner() for spec in specs[:max_tools]]
        if len(specs) > max_tools:
            lines.append(f"(+{len(specs) - max_tools} more)")
        return lines

    def _skill_selection_context(
        self,
        problem: str,
        plan: ExecutionPlan,
        step: PlanStep,
        history: list[dict[str, Any]],
    ) -> str:
        payload = {
            "problem": problem,
            "plan_summary": {
                "problem_summary": plan.problem_summary,
                "topology_rationale": plan.topology_rationale,
            },
            "current_step": {
                "id": step.id,
                "role": step.role,
                "objective": step.objective,
                "expected_output": step.expected_output,
            },
            "prior_outputs": [
                {
                    "step_id": item.get("step_id"),
                    "role": item.get("role"),
                    "objective": item.get("objective"),
                    "result": _truncate(str(item.get("result", "")), 1200),
                }
                for item in history
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _select_skills_for_step(
        self,
        role: str,
        problem: str,
        plan: ExecutionPlan,
        step: PlanStep,
        history: list[dict[str, Any]],
    ) -> tuple[list[ScimasSkill], str, bool]:
        if not self.use_skill_routing:
            return [], "", False
        available = skills_for_role(role, self.skills)
        if not available:
            return [], "", False

        prompt = self._skill_selection_prompt(role, available)
        context = self._skill_selection_context(problem, plan, step, history)
        try:
            result = self.runner.run(
                prompt=prompt,
                stdin_text=context,
                output_format="stream-json",
                allowed_tools=[],
            )
        except Exception as exc:
            return [], f"skill selection failed: {exc}", True

        data = self._extract_json_block(result.result)
        selected_raw = data.get("selected_skills", []) if data else []
        if isinstance(selected_raw, str):
            selected_raw = [selected_raw]
        available_by_id = {skill.skill_id: skill for skill in available}
        selected: list[ScimasSkill] = []
        for item in selected_raw:
            skill = available_by_id.get(str(item))
            if skill is not None and skill not in selected:
                selected.append(skill)
            if len(selected) >= 2:
                break

        return selected, result.result, True

    def _context_bundle(
        self,
        problem: str,
        plan: ExecutionPlan,
        step: PlanStep,
        history: List[Dict[str, Any]],
        dsl_input: Any = None,
        received_upstream_outputs: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        prior_outputs = [
            {
                "step_id": item.get("step_id"),
                "role": item.get("role"),
                "objective": item.get("objective", ""),
                "public_output": item.get("public_output", item.get("result", "")),
            }
            for item in history
        ]
        payload = {
            "original_task": problem,
            "problem": problem,
            "plan": {
                "mode": plan.mode,
                "problem_summary": plan.problem_summary,
                "topology_rationale": plan.topology_rationale,
                "steps": [
                    {
                        "id": item.id,
                        "role": item.role,
                        "objective": item.objective,
                        "depends_on": item.depends_on,
                    }
                    for item in plan.steps
                ],
                "execution_order": plan.execution_order,
            },
            "current_step": {
                "id": step.id,
                "role": step.role,
                "objective": step.objective,
                "depends_on": step.depends_on,
                "expected_output": step.expected_output,
                "handoff_to": step.handoff_to,
                "confidence": step.confidence,
            },
            "current_role": step.role,
            "current_instruction": step.objective,
            "received_upstream_outputs": (
                received_upstream_outputs if received_upstream_outputs is not None else prior_outputs
            ),
            "prior_outputs": prior_outputs,
        }
        if dsl_input is not None:
            payload["workflow_input"] = dsl_input
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def plan(
        self, problem: str, roles: Iterable[str]
    ) -> tuple[ExecutionPlan, PlannerRun]:
        # Record the roles this run may use, so workflow validation can reject
        # an invented role here instead of letting it fail mid-execution.
        self.active_roles = tuple(
            sorted({self.normalize_role(role) for role in roles if role})
        ) or DEFAULT_PLANNER_ROLES
        if self.planner_mode == "python_dsl":
            return self._plan_python_dsl(problem, roles)
        return self._plan_legacy_json(problem, roles)

    def _plan_legacy_json(
        self, problem: str, roles: Iterable[str]
    ) -> tuple[ExecutionPlan, PlannerRun]:
        prompt = self._planner_prompt(problem, roles)
        result = self.runner.run(
            prompt=prompt,
            stdin_text=problem,
            output_format="stream-json",
            allowed_tools=[],
        )
        plan_dict = self._extract_json_block(result.result)
        if not plan_dict:
            raise RuntimeError(
                "Planner did not return valid JSON. Raw output:\n"
                f"{result.raw_stdout}"
            )
        plan = ExecutionPlan.from_dict(plan_dict)
        plan.mode = "legacy_json"
        prompt_path = self.prompt_dir / "planner_json.md"
        planner = PlannerRun(
            prompt_path=str(prompt_path if prompt_path.exists() else self.prompt_dir / "planner.md"),
            assembled_prompt=prompt,
            problem_text=problem,
            raw_stdout=result.raw_stdout,
            raw_json=result.raw_json,
            parsed_plan=plan.to_dict(),
            session_id=result.session_id,
            started_at=result.started_at,
            duration_ms=result.duration_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.total_cost_usd,
            stderr=result.stderr,
            tool_calls=list(result.tool_calls),
        )
        return plan, planner

    def _plan_python_dsl(
        self, problem: str, roles: Iterable[str]
    ) -> tuple[ExecutionPlan, PlannerRun]:
        """Generate a Python DSL workflow with reviewer validation and fix loop."""
        max_attempts = self.max_review_attempts
        roles = list(roles)

        # Stage 1: Planner generates initial workflow
        prompt = self._planner_prompt(problem, roles)
        result = self.runner.run(
            prompt=prompt,
            stdin_text=problem,
            output_format="stream-json",
            allowed_tools=[],
        )

        # `extract_workflow_source` (inside parse_and_validate_workflow) strips
        # the prose and markdown fences planners habitually add, so a merely
        # wrapped-but-valid workflow never reaches the reviewer at all.
        workflow_source = result.result
        validation_error = None
        attempt = 0
        reviewer_raw_reply = ""

        # Stage 2: Review and fix loop. Every iteration re-validates at the
        # top, so a reviewer that reports "valid" while echoing the same
        # source is re-checked rather than treated as a hard failure.
        # `attempt` counts reviewer repair rounds, so max_review_attempts=3
        # buys exactly three reviewer calls.
        while True:
            try:
                program = parse_and_validate_workflow(
                    workflow_source,
                    role_check=self._workflow_role_check,
                    allowed_roles=self.active_roles,
                )
                # Success - workflow is valid
                break
            except WorkflowDSLValidationError as exc:
                validation_error = str(exc)
                attempt += 1
                self._emit_event({
                    "event": "workflow_validation_failed",
                    "attempt": attempt,
                    "error": validation_error,
                })

                if attempt > max_attempts:
                    # Give up after max attempts
                    plural = "" if max_attempts == 1 else "s"
                    raise RuntimeError(
                        f"Workflow validation failed after {max_attempts} reviewer "
                        f"attempt{plural}. Last error: {validation_error}\n"
                        f"Reviewer's last reply:\n{reviewer_raw_reply[:2000]}\n"
                        f"Last source:\n{workflow_source}"
                    ) from exc

                # Ask reviewer to fix it
                review_result = self._review_and_fix_workflow(
                    workflow_source=workflow_source,
                    validation_error=validation_error,
                    execution_error=None,
                    attempt_number=attempt,
                    problem=problem,
                    roles=roles,
                )
                reviewer_raw_reply = review_result.get("raw_reply", "") or ""

                if not review_result.get("is_valid"):
                    issues = review_result.get("issues", [])
                    self._emit_event({
                        "event": "reviewer_found_issues",
                        "attempt": attempt,
                        "issues": issues,
                        "raw_reply": reviewer_raw_reply[:2000],
                    })

                fixed_source = review_result.get("fixed_source", "")
                if not fixed_source or fixed_source == workflow_source:
                    # Reviewer made no progress. Loop back: the top of the
                    # loop re-validates, and the attempt counter decides when
                    # to give up. A reviewer that keeps echoing the source
                    # while claiming it is valid now costs attempts instead of
                    # ending the run on the first reply.
                    continue

                workflow_source = fixed_source
                changes = review_result.get("changes_made", [])
                confidence = review_result.get("confidence", 0.0)

                self._emit_event({
                    "event": "workflow_fixed_by_reviewer",
                    "attempt": attempt,
                    "changes": changes,
                    "confidence": confidence,
                })

        # Build the plan with the validated workflow
        plan = ExecutionPlan(
            problem_summary="Planner-produced Python DSL workflow.",
            topology_rationale=(
                "Topology is encoded by async workflow control flow and "
                "recovered from runtime trace dependencies."
            ),
            final_role="workflow",
            termination_criterion="workflow(task) returns a final value.",
            mode="python_dsl",
            workflow_source=program.source,
            workflow_static_trace=program.static_trace,
        )
        planner = PlannerRun(
            prompt_path=str(self.prompt_dir / "planner.md"),
            assembled_prompt=prompt,
            problem_text=problem,
            raw_stdout=result.raw_stdout,
            raw_json=result.raw_json,
            parsed_plan=plan.to_dict(),
            session_id=result.session_id,
            started_at=result.started_at,
            duration_ms=result.duration_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.total_cost_usd,
            stderr=result.stderr,
            tool_calls=list(result.tool_calls),
        )
        return plan, planner

    def run(
        self,
        problem: str,
        roles: Optional[List[str]] = None,
        auto_synthesize: bool = True,
    ) -> RunReport:
        from datetime import datetime, timezone

        run_started_at = datetime.now(timezone.utc).isoformat()
        roles = roles or list(DEFAULT_PLANNER_ROLES)
        self._emit_event(
            {
                "event": "planner_started",
                "problem": problem,
                "roles": roles,
                "planner_mode": self.planner_mode,
            }
        )
        plan, planner = self.plan(problem, roles)
        self._emit_event(
            {
                "event": "planner_finished",
                "plan": plan.to_dict(),
                "planner": {
                    "session_id": planner.session_id,
                    "duration_ms": planner.duration_ms,
                    "cost_usd": planner.cost_usd,
                },
            }
        )

        if plan.mode == "python_dsl":
            return self._run_python_dsl_workflow(
                problem=problem,
                plan=plan,
                planner=planner,
                started_at=run_started_at,
            )

        ordered_steps = plan.ordered_steps()
        session_ids: Dict[str, str] = {}
        history: List[Dict[str, Any]] = []
        runs: List[AgentRun] = []

        for step in ordered_steps:
            role_name = self.normalize_role(step.role)
            step_history = _explicit_history_for_step(history, step)
            self._emit_event(
                {
                    "event": "agent_started",
                    "node_id": step.id,
                    "role": role_name,
                    "instruction": step.objective,
                    "received_from": list(step.depends_on),
                }
            )
            selected_skills, skill_selection_result, had_skill_candidates = self._select_skills_for_step(
                role_name, problem, plan, step, step_history
            )
            step_prompt = self._step_prompt(
                role_name, step, plan, step_history, selected_skills=selected_skills
            )
            context = self._context_bundle(problem, plan, step, step_history)
            previous_session = None
            selected_skill_ids = [skill.skill_id for skill in selected_skills]
            step_allowed_tools = allowed_tools_for_role(
                role_name,
                selected_skills=selected_skill_ids,
                scimas_skills=self.skills,
                allow_role_fallback=not (
                    self.use_skill_routing
                    and self.strict_skill_routing
                    and had_skill_candidates
                ),
            )
            result = self.runner.run(
                prompt=step_prompt,
                stdin_text=context,
                session_id=previous_session,
                output_format="stream-json",
                allowed_tools=step_allowed_tools,
            )
            if result.session_id:
                session_ids[step.id] = result.session_id

            # If the runner inlined stdin_text into the prompt because
            # an MCP stdio server is attached, capture the effective
            # prompt we sent (input for logging). Otherwise capture the
            # raw stdin_text.
            stdin_for_log = (
                ""
                if self.runner.mcp_config_path
                else context
            )
            run = AgentRun(
                role=role_name,
                step_id=step.id,
                prompt_path=str(self.prompt_dir / f"{role_name}.md"),
                assembled_prompt=step_prompt,
                context_bundle=context,
                stdin_text=stdin_for_log,
                selected_skills=selected_skill_ids,
                skill_selection_result=skill_selection_result,
                allowed_tools=step_allowed_tools,
                session_id=result.session_id,
                resumed_from_session=previous_session,
                result=result.result,
                raw_stdout=result.raw_stdout,
                raw_json=result.raw_json,
                started_at=result.started_at,
                duration_ms=result.duration_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.total_cost_usd,
                stderr=result.stderr,
                public_output=coerce_agent_result(result.result),
                received_from=list(step.depends_on),
                tool_calls=list(result.tool_calls),
            )
            runs.append(run)
            public_output = coerce_agent_result(result.result)
            history.append(
                {
                    "step_id": step.id,
                    "role": role_name,
                    "objective": step.objective,
                    "result": result.result,
                    "public_output": public_output,
                    "received_from": list(step.depends_on),
                    "session_id": result.session_id,
                }
            )
            self._emit_event(
                {
                    "event": "agent_finished",
                    "node_id": step.id,
                    "role": role_name,
                    "instruction": step.objective,
                    "received_from": list(step.depends_on),
                    "public_output": make_json_safe(public_output),
                }
            )

        final_answer = ""
        if auto_synthesize and self.normalize_role(plan.final_role) != "synthesizer":
            plan.final_role = "synthesizer"

        if auto_synthesize and (not ordered_steps or self.normalize_role(ordered_steps[-1].role) != "synthesizer"):
            synth_step = PlanStep(
                id="final",
                role="synthesizer",
                objective="Synthesize the scientific solution from all prior role outputs.",
                depends_on=[step.id for step in ordered_steps],
                expected_output="A coherent final answer.",
            )
            self._emit_event(
                {
                    "event": "agent_started",
                    "node_id": synth_step.id,
                    "role": "synthesizer",
                    "instruction": synth_step.objective,
                    "received_from": list(synth_step.depends_on),
                }
            )
            synth_history = _explicit_history_for_step(history, synth_step)
            synth_prompt = self._step_prompt("synthesizer", synth_step, plan, synth_history)
            synth_context = self._context_bundle(problem, plan, synth_step, synth_history)
            previous_synth_session = None
            synth_allowed_tools = allowed_tools_for_role("synthesizer")
            synth_result = self.runner.run(
                prompt=synth_prompt,
                stdin_text=synth_context,
                session_id=previous_synth_session,
                output_format="stream-json",
                allowed_tools=synth_allowed_tools,
            )
            if synth_result.session_id:
                session_ids["final"] = synth_result.session_id
            final_answer = synth_result.result
            synth_stdin_for_log = "" if self.runner.mcp_config_path else synth_context
            runs.append(
                AgentRun(
                    role="synthesizer",
                    step_id="final",
                    prompt_path=str(self.prompt_dir / "synthesizer.md"),
                    assembled_prompt=synth_prompt,
                    context_bundle=synth_context,
                    stdin_text=synth_stdin_for_log,
                    allowed_tools=synth_allowed_tools,
                    session_id=synth_result.session_id,
                    resumed_from_session=previous_synth_session,
                    result=synth_result.result,
                    raw_stdout=synth_result.raw_stdout,
                    raw_json=synth_result.raw_json,
                    started_at=synth_result.started_at,
                    duration_ms=synth_result.duration_ms,
                    input_tokens=synth_result.input_tokens,
                    output_tokens=synth_result.output_tokens,
                    cost_usd=synth_result.total_cost_usd,
                    stderr=synth_result.stderr,
                    public_output=coerce_agent_result(synth_result.result),
                    received_from=list(synth_step.depends_on),
                    tool_calls=list(synth_result.tool_calls),
                )
            )
            history.append(
                {
                    "step_id": "final",
                    "role": "synthesizer",
                    "objective": synth_step.objective,
                    "result": synth_result.result,
                    "public_output": coerce_agent_result(synth_result.result),
                    "received_from": list(synth_step.depends_on),
                    "session_id": synth_result.session_id,
                }
            )
            self._emit_event(
                {
                    "event": "agent_finished",
                    "node_id": synth_step.id,
                    "role": "synthesizer",
                    "instruction": synth_step.objective,
                    "received_from": list(synth_step.depends_on),
                    "public_output": make_json_safe(coerce_agent_result(synth_result.result)),
                }
            )
        elif ordered_steps:
            final_answer = runs[-1].result

        report = RunReport(
            problem=problem,
            plan=plan,
            planner=planner,
            runs=runs,
            final_answer=final_answer,
            session_ids=session_ids,
            started_at=run_started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        report.output_dir = str(self._save_report(report))
        self._emit_event(
            {
                "event": "run_finished",
                "output_dir": report.output_dir,
                "final_answer": report.final_answer,
                "plan": report.plan.to_dict(),
            }
        )
        return report

    def _run_python_dsl_workflow(
        self,
        *,
        problem: str,
        plan: ExecutionPlan,
        planner: PlannerRun,
        started_at: str,
    ) -> RunReport:
        """Execute a Python DSL workflow with error recovery."""
        from datetime import datetime, timezone

        MAX_EXECUTION_RETRIES = 2
        runtime = _WorkflowRuntime(self, problem, plan)

        for attempt in range(1, MAX_EXECUTION_RETRIES + 1):
            try:
                final_value = asyncio.run(runtime.run(plan.workflow_source))
                # Success - workflow executed
                break
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
                self._emit_event({
                    "event": "workflow_execution_failed",
                    "attempt": attempt,
                    "error": execution_error,
                })

                if attempt >= MAX_EXECUTION_RETRIES:
                    # Give up after max retries
                    raise RuntimeError(
                        f"Workflow execution failed after {MAX_EXECUTION_RETRIES} attempts. "
                        f"Last error: {execution_error}"
                    ) from exc

                # Try to fix the workflow with the reviewer
                try:
                    review_result = self._review_and_fix_workflow(
                        workflow_source=plan.workflow_source,
                        validation_error=None,
                        execution_error=execution_error,
                        attempt_number=attempt,
                        problem=problem,
                        roles=self.active_roles,
                    )

                    fixed_source = review_result.get("fixed_source", "")
                    if not fixed_source or fixed_source == plan.workflow_source:
                        # Reviewer couldn't help
                        raise RuntimeError(
                            f"Reviewer could not fix execution error: {execution_error}\n"
                            f"Issues: {review_result.get('issues', [])}\n"
                            "Reviewer's raw reply:\n"
                            f"{(review_result.get('raw_reply') or '')[:2000]}"
                        ) from exc

                    # Validate the fixed source
                    try:
                        program = parse_and_validate_workflow(
                            fixed_source,
                            role_check=self._workflow_role_check,
                            allowed_roles=self.active_roles,
                        )
                        plan.workflow_source = program.source
                        plan.workflow_static_trace = program.static_trace

                        # Create new runtime with fixed workflow
                        runtime = _WorkflowRuntime(self, problem, plan)

                        changes = review_result.get("changes_made", [])
                        confidence = review_result.get("confidence", 0.0)
                        self._emit_event({
                            "event": "workflow_fixed_after_execution_error",
                            "attempt": attempt,
                            "changes": changes,
                            "confidence": confidence,
                        })
                    except WorkflowDSLValidationError as val_exc:
                        # Fixed source is still invalid
                        raise RuntimeError(
                            f"Reviewer produced invalid workflow: {val_exc}"
                        ) from exc
                except Exception as review_exc:
                    # Reviewer itself failed
                    raise RuntimeError(
                        f"Reviewer failed to fix execution error: {review_exc}"
                    ) from exc

        report = RunReport(
            problem=problem,
            plan=plan,
            planner=planner,
            runs=runtime.runs,
            final_answer=_format_workflow_result(final_value),
            session_ids=runtime.session_ids,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        report.output_dir = str(self._save_report(report))
        self._emit_event(
            {
                "event": "run_finished",
                "output_dir": report.output_dir,
                "final_answer": report.final_answer,
                "plan": report.plan.to_dict(),
            }
        )
        return report

    def _save_report(self, report: RunReport) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.output_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=False)

        # 1. Full structured log: every planner/role input + output +
        #    metadata. Single file, machine-readable.
        (run_dir / "report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 2. JSONL event stream: one event per planner/role invocation
        #    with full I/O. Easier to grep, tail -f, or load into a
        #    notebook than the nested report.json.
        self._write_trace(run_dir, report)

        # 3. Human-readable Markdown with per-step inputs and outputs.
        (run_dir / "report.md").write_text(
            self._render_markdown(report), encoding="utf-8"
        )

        return run_dir

    @staticmethod
    def _write_trace(run_dir: Path, report: RunReport) -> None:
        events: list[dict] = []

        if report.planner is not None:
            p = report.planner
            events.append(
                {
                    "event": "planner",
                    "started_at": p.started_at,
                    "duration_ms": p.duration_ms,
                    "input": {
                        "problem_text": p.problem_text,
                        "assembled_prompt": p.assembled_prompt,
                        "prompt_path": p.prompt_path,
                    },
                    "output": {
                        "session_id": p.session_id,
                        "result": p.raw_json.get("result"),
                        "parsed_plan": p.parsed_plan,
                    },
                    "usage": {
                        "input_tokens": p.input_tokens,
                        "output_tokens": p.output_tokens,
                        "cost_usd": p.cost_usd,
                    },
                    "stderr": p.stderr,
                    "tool_calls": list(p.tool_calls),
                }
            )

        if report.plan.mode == "python_dsl":
            events.append(
                {
                    "event": "workflow",
                    "mode": report.plan.mode,
                    "source": report.plan.workflow_source,
                    "static_trace": report.plan.workflow_static_trace,
                    "runtime_trace": report.plan.workflow_trace,
                }
            )

        for run in report.runs:
            events.append(
                {
                    "event": "agent",
                    "role": run.role,
                    "step_id": run.step_id,
                    "started_at": run.started_at,
                    "duration_ms": run.duration_ms,
                    "input": {
                        "assembled_prompt": run.assembled_prompt,
                        "context_bundle": run.context_bundle,
                        "stdin_text": run.stdin_text,
                        "resumed_from_session": run.resumed_from_session,
                        "prompt_path": run.prompt_path,
                        "selected_skills": run.selected_skills,
                        "skill_selection_result": run.skill_selection_result,
                        "allowed_tools": run.allowed_tools,
                    },
                    "output": {
                        "session_id": run.session_id,
                        "result": run.result,
                        "public_output": run.public_output,
                    },
                    "information_flow": {
                        "received_from": run.received_from,
                        "public_output": run.public_output,
                    },
                    "usage": {
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "cost_usd": run.cost_usd,
                    },
                    "stderr": run.stderr,
                    "tool_calls": list(run.tool_calls),
                }
            )

        with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    @staticmethod
    def _render_markdown(report: RunReport) -> str:
        lines: list[str] = []
        lines.append("# sciMAS Run Report")
        lines.append("")
        lines.append(f"- **Started:** {report.started_at}")
        lines.append(f"- **Finished:** {report.finished_at}")
        lines.append(f"- **Problem:** {report.problem}")
        if report.session_ids:
            lines.append("- **Session IDs:**")
            for role, sid in report.session_ids.items():
                lines.append(f"  - `{role}`: `{sid}`")
        lines.append("")

        # ---- MAS structure / plan ----
        lines.append("## MAS Structure (Plan)")
        lines.append("")
        plan = report.plan
        lines.append(f"- **Mode:** `{plan.mode}`")
        lines.append(f"- **Problem summary:** {plan.problem_summary}")
        if plan.topology_rationale:
            lines.append(f"- **Topology rationale:** {plan.topology_rationale}")
        if plan.assumptions:
            lines.append("- **Assumptions:**")
            for a in plan.assumptions:
                lines.append(f"  - {a}")
        if plan.risks:
            lines.append("- **Risks:**")
            for r in plan.risks:
                lines.append(f"  - {r}")
        lines.append(f"- **Final role:** `{plan.final_role}`")
        if plan.termination_criterion:
            lines.append(f"- **Termination:** {plan.termination_criterion}")
        lines.append("")
        lines.append("### Steps")
        lines.append("")
        lines.append("| # | id | role | depends_on | objective |")
        lines.append("|---|---|---|---|---|")
        for i, step in enumerate(plan.steps, 1):
            obj = (step.objective or "").replace("|", "\\|")
            lines.append(
                f"| {i} | `{step.id}` | `{step.role}` | "
                f"{', '.join(step.depends_on) or '—'} | {obj} |"
            )
        if plan.execution_order:
            lines.append("")
            lines.append(f"**Execution order:** {' → '.join(plan.execution_order)}")
        if plan.mode == "python_dsl":
            lines.append("")
            lines.append("### Workflow Source")
            lines.append("")
            lines.append("```python")
            lines.append(plan.workflow_source)
            lines.append("```")
            if plan.workflow_trace:
                lines.append("")
                lines.append("### Workflow Trace")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(plan.workflow_trace, ensure_ascii=False, indent=2))
                lines.append("```")
        lines.append("")

        # ---- Planner I/O ----
        if report.planner is not None:
            p = report.planner
            lines.append("## Planner")
            lines.append("")
            lines.append(
                f"- started_at: `{p.started_at}` "
                f"| duration_ms: `{p.duration_ms}` "
                f"| session_id: `{p.session_id}` "
                f"| cost_usd: `{p.cost_usd}`"
            )
            lines.append(f"- prompt_path: `{p.prompt_path}`")
            lines.append("- **Input problem:**")
            lines.append("```")
            lines.append(p.problem_text)
            lines.append("```")
            lines.append("- **Assembled prompt (truncated to 2000 chars):**")
            lines.append("```")
            lines.append(_truncate(p.assembled_prompt, 2000))
            lines.append("```")
            lines.append("- **Output (parsed plan):**")
            lines.append("```json")
            lines.append(json.dumps(p.parsed_plan, ensure_ascii=False, indent=2))
            lines.append("```")
            if p.stderr:
                lines.append("- **stderr:**")
                lines.append("```")
                lines.append(_truncate(p.stderr, 2000))
                lines.append("```")
            if p.tool_calls:
                lines.append(f"- **Tool calls ({len(p.tool_calls)}):**")
                lines.append("```json")
                lines.append(
                    json.dumps(p.tool_calls, ensure_ascii=False, indent=2)
                )
                lines.append("```")
            lines.append("")

        # ---- Per-agent I/O ----
        for run in report.runs:
            lines.append(
                f"## Agent: `{run.role}` / step `{run.step_id}`"
            )
            lines.append("")
            meta = [
                f"started_at: `{run.started_at}`",
                f"duration_ms: `{run.duration_ms}`",
                f"session_id: `{run.session_id}`",
                f"resumed_from_session: `{run.resumed_from_session}`",
                f"cost_usd: `{run.cost_usd}`",
                f"in_tok: `{run.input_tokens}`",
                f"out_tok: `{run.output_tokens}`",
            ]
            lines.append("- " + " | ".join(meta))
            lines.append(f"- prompt_path: `{run.prompt_path}`")
            if run.selected_skills:
                lines.append(f"- selected_skills: `{', '.join(run.selected_skills)}`")
            if run.allowed_tools:
                lines.append(
                    f"- allowed_tools: `{', '.join(run.allowed_tools)}`"
                )
            if run.tool_calls:
                lines.append(f"- **Tool calls ({len(run.tool_calls)}):**")
                lines.append("```json")
                lines.append(
                    json.dumps(run.tool_calls, ensure_ascii=False, indent=2)
                )
                lines.append("```")
            if run.received_from:
                lines.append(f"- received_from: `{', '.join(run.received_from)}`")
            if run.skill_selection_result:
                lines.append("- **Skill selection result (truncated to 1000 chars):**")
                lines.append("```")
                lines.append(_truncate(run.skill_selection_result, 1000))
                lines.append("```")
            lines.append("- **Assembled prompt (truncated to 2000 chars):**")
            lines.append("```")
            lines.append(_truncate(run.assembled_prompt, 2000))
            lines.append("```")
            lines.append("- **Context bundle (truncated to 2000 chars):**")
            lines.append("```json")
            lines.append(_truncate(run.context_bundle, 2000))
            lines.append("```")
            lines.append("- **Public output:**")
            lines.append("```")
            lines.append(_truncate(_format_workflow_result(run.public_output), 4000))
            lines.append("```")
            lines.append("- **Result:**")
            lines.append("```")
            lines.append(_truncate(run.result, 4000))
            lines.append("```")
            if run.stderr:
                lines.append("- **stderr (truncated):**")
                lines.append("```")
                lines.append(_truncate(run.stderr, 2000))
                lines.append("```")
            lines.append("")

        if report.final_answer:
            lines.append("## Final Answer")
            lines.append("")
            lines.append(report.final_answer)
            lines.append("")
        return "\n".join(lines)
