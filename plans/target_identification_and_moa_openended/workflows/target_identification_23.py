async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify gene targets capable of therapeutically modulating the immune response in SLE. Compare human genetic evidence, pathway dysregulation, causal perturbation, druggability, and patient intervention results.",
        input=task,
    )

    genetic_review, translation_review = await parallel(
        agent(
            role="geneticist",
            instruction="Assess whether associations and functional variants support causal immune regulation, identify relevant cell types, and infer modulation direction only from perturbation evidence rather than association alone.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify target identity, selective modality, direct versus indirect action, SLE-specific clinical evidence, and practical limitations for difficult target classes.",
            input=[task, discovery],
        ),
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Integrate genetic, mechanistic, and translational evidence to rank actionable candidates. Compare complementary immune mechanisms, context-dependent protective functions, and safety tradeoffs.",
        input=[task, discovery, genetic_review, translation_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes with protein identity, modulation strategy, SLE mechanism, evidence strength, and the most material limitation for each. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, genetic_review, translation_review, ranking],
    )
    return final
