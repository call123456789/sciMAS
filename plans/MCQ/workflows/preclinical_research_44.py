async def workflow(task):
    survival_evidence = await agent(
        role="literature-searcher",
        instruction="Retrieve reproducible colorectal-cancer survival analyses relating PRF1 expression to overall survival from authoritative publications or documented portals. Preserve database version, access date, cohort, platform or probe mapping, filters, endpoint, cutoff method, sample size, and reported statistics without presuming direction.",
        input=task,
    )

    biology_review, statistics_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Interpret PRF1 expression in tumor, immune, and stromal contexts and identify purity, stage, treatment, and cell-composition confounders. Separate prognostic association from a causal tumor-cell mechanism.",
            input=[task, survival_evidence],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Audit high-versus-low orientation, cutoff selection, hazard ratio and confidence interval, event count, proportional-hazards assumptions, multiple testing, and cross-portal comparability.",
            input=[task, survival_evidence],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="State the exact supported PRF1-overall-survival relationship, including direction and analysis settings only after verification, and qualify it as cohort- and method-specific rather than causal. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, survival_evidence, biology_review, statistics_review],
    )
    return final
