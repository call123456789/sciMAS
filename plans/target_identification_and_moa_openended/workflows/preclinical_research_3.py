async def workflow(task):
    evidence = await agent(
        role="literature-searcher",
        instruction="Retrieve current method specifications and comparative studies for label-free, SILAC, TMT, iTRAQ, and PRM/SRM phosphoproteomic quantification. Extract measurement principle, sample compatibility, instrument requirements, quantification performance, and version-dependent capabilities.",
        input=task,
    )

    comparison = await agent(
        role="analytical-chemist",
        instruction="Compare the five methods by labeling and readout mechanism, discovery versus targeted scope, multiplexing, precision, missingness, interference, dynamic range, sample needs, cost, and typical applications. Keep phosphopeptide enrichment and phosphorylation occupancy distinct from quantification.",
        input=[task, evidence],
    )

    final = await agent(
        role="synthesizer",
        instruction="Provide a concise five-method comparison covering mechanistic basis, strengths, limitations, and suitable applications, with version- and experiment-dependent claims clearly qualified. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, evidence, comparison],
    )
    return final
