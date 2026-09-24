async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify therapeutic genes for reducing inflammation in sarcoidosis. Compare clinical intervention evidence, steroid-sparing outcomes, organ manifestations, mechanistic studies, and failed or heterogeneous results.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify each candidate's gene/protein identity, selective modality, clinical status, and distinction between an activity biomarker and a disease-modifying target.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Rank the smallest set justified by translational evidence, while accounting for granuloma biology, immune-cell recruitment, organ context, infection risk, and paradoxical effects. Do not infer a required count.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Give a concise ranked target list with modulation direction, inflammatory mechanism, clinical evidence, and limitations. State why the selected number is justified. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
