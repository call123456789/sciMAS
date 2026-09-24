async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify SLE gene targets with human mechanistic evidence and target-directed outcomes for disease activity, flares, organ involvement, or steroid sparing. Include negative trials and distinguish ligands, receptors, and downstream pathways.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify canonical target identity, selective intervention mechanism, SLE-specific clinical status, eligible subgroups, and safety. Do not transfer approval or efficacy from another autoimmune disease without SLE evidence.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Compare effects on autoreactive-cell maintenance, autoantibody production, innate and effector inflammation, required modulation direction, infection risk, and clinical evidence. Rank without assuming that an upregulated immune factor should be inhibited.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes with protein identity, intervention direction, SLE immune mechanism, evidence strength, and subgroup or safety caveats. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
