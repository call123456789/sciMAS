async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify pancreatitis genes that could modulate inflammatory injury and fibrotic remodeling. Distinguish acute from chronic disease and retrieve cell-specific mechanisms, patient evidence, perturbation studies, interventions, and negative results.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify target identities and intervention modalities, and classify evidence as human, animal, or unrelated-fibrosis extrapolation. Record selectivity and translational limits.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Compare candidates for early inflammatory injury and later stromal activation or matrix deposition, including required direction, timing, repair liabilities, and evidence for functional benefit.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes with complementary inflammatory and fibrotic roles, modulation direction, acute-versus-chronic relevance, evidence maturity, and safety caveats. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
