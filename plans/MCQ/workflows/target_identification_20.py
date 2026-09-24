async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify modifiable systemic-sclerosis targets from human tissue/pathway evidence and intervention studies. Cover immune activation, fibrosis, and vasculopathy while preserving skin, lung, and other organ-specific outcomes.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify gene/protein identity, direct drug-target mechanism, systemic-sclerosis or associated-organ indication, trial outcomes, and whether a broad drug effect validates the individual target.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Rank candidates by causal disease biology, pathway dysregulation in patients, feasible modulation, effects on immune-stromal interactions, organ scope, and safety. Distinguish inflammatory biomarker changes from altered fibrosis progression.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two targets with modulation direction, pathogenic mechanism, druggability, evidence maturity, organ scope, and principal limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
