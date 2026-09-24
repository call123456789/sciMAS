async def workflow(task):
    evidence = await agent(
        role="literature-searcher",
        instruction="Retrieve primary domain, mutagenesis, interaction, localization, and signaling studies of GAB1 in PI3K-to-ERK pathway crosstalk. Extract receptor context, binding sites, perturbations, downstream readouts, feedback, and contradictory results.",
        input=task,
    )

    mechanism = await agent(
        role="molecular-biologist",
        instruction="Build a stepwise mechanistic model of GAB1 as a scaffold, distinguishing direct binding from recruitment, localization, phosphorylation, enzymatic activity, and downstream correlation. Identify checkpoints that determine whether PI3K activity influences ERK signaling in a given context.",
        input=[task, evidence],
    )

    final = await agent(
        role="synthesizer",
        instruction="Explain how GAB1 can connect PI3K activation to ERK signaling and identify the key molecular and contextual checkpoints, with each link tied to evidence. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, evidence, mechanism],
    )
    return final
