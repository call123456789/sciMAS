async def workflow(task):
    survival_evidence = await agent(
        role="literature-searcher",
        instruction="Retrieve reproducible colorectal-cancer survival analyses relating CASP1 expression to recurrence-free survival from authoritative publications or documented portals. Preserve database version, cohort, platform or probe mapping, filters, endpoint definition, cutoff method, sample size, and statistics without presuming direction.",
        input=task,
    )

    biology_review, statistics_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Interpret CASP1 expression across tumor and immune compartments and identify purity, stage, treatment, and inflammatory-state confounders. Keep bulk-expression association separate from a causal intervention claim.",
            input=[task, survival_evidence],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Audit group orientation, cutoff selection, censoring, hazard ratio uncertainty, sample and event counts, multiple testing, and reasons portals may disagree.",
            input=[task, survival_evidence],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="State the verified CASP1-recurrence-free-survival relationship, direction, and exact analysis context, or report that it cannot be reproduced. Avoid causal extrapolation. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, survival_evidence, biology_review, statistics_review],
    )
    return final
