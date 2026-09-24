async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify genes that can modulate immune activity and granuloma formation in sarcoidosis. Retrieve disease-specific human biology, organ-specific studies, interventions, refractory-disease outcomes, negative findings, and infection liabilities.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify target identity, direct modality, sarcoidosis-specific clinical status, organ scope, and whether evidence derives from another granulomatous disease. Separate biomarkers from therapeutic targets.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Compare roles in macrophage and lymphocyte interactions, granuloma initiation or maintenance, required modulation direction, patient evidence, and protective-immunity risks.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes with immune and granuloma mechanisms, intervention direction, sarcoidosis-specific evidence strength, organ scope, and infection caveats. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
