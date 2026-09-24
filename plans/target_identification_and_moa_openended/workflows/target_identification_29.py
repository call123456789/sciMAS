async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify rheumatoid-arthritis targets supported by patient pathway dysregulation, causal synovial biology, and target-directed clinical trials. Retrieve effects on disease activity, function, structural damage, and progression.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify exact gene/protein target, selective biologic or small-molecule mechanism, rheumatoid-arthritis indication, and evidence for structural or functional outcomes. Keep biomarkers distinct from target genes.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Compare effects on immune-stromal interactions, synovitis, cartilage or bone damage, intervention direction, patient evidence, and infection or systemic safety.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two targets with gene/protein identities, modulation direction, pathogenic role, verified modality, progression evidence, and key caveats. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
