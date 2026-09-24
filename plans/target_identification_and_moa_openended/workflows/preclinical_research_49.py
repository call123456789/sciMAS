async def workflow(task):
    correlation_evidence = await agent(
        role="literature-searcher",
        instruction="Retrieve documented analyses of HSF1 expression and CD8-positive T-cell infiltration across cancer cohorts from authoritative immune-deconvolution portals or publications. Preserve database version, access date, cancer code, estimator, purity adjustment, sample set, and statistics without presuming sign or tumor type.",
        input=task,
    )

    biology_review, statistics_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Interpret HSF1 expression across tumor, immune, and stromal compartments and identify purity, subtype, and bulk-expression confounding. Separate a correlation from a mechanism controlling infiltration.",
            input=[task, correlation_evidence],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Audit correlation sign and magnitude, uncertainty, multiple testing, and comparability across deconvolution algorithms and cancer cohorts. Do not infer biological importance from significance alone.",
            input=[task, correlation_evidence],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Report the strongest reproducible HSF1-CD8-infiltration relationship with tumor type, direction, estimator, and settings, while emphasizing algorithm dependence and lack of causal proof. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, correlation_evidence, biology_review, statistics_review],
    )
    return final
