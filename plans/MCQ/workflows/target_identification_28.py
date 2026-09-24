async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify genes that could modulate both inflammatory and granulomatous responses in sarcoidosis. Retrieve cell-specific mechanisms, causal perturbations, patient evidence, intervention studies, and discordant findings.",
        input=task,
    )

    mechanism_review, translation_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Compare roles in immune-cell recruitment, macrophage activation, lymphocyte signaling, and granuloma persistence. Distinguish systemic inflammatory-marker changes from effects on granulomatous pathology.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify canonical target identity, available selective interventions, sarcoidosis-specific outcomes, off-label or investigational status, and clinically relevant toxicities.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes that best connect modifiable inflammation with granuloma biology, giving direction, evidence level, and organ-specific limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, mechanism_review, translation_review],
    )
    return final
