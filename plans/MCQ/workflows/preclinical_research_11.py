async def workflow(task):
    paper_search = await agent(
        role="literature-searcher",
        instruction="Identify the primary study that introduced Tstr cells and retrieve its experimental and computational design, cohorts, defining evidence, tissue context, functional evidence, and treatment analyses with figure-level traceability.",
        input=task,
    )

    identity_review, context_review = await parallel(
        agent(
            role="geneticist",
            instruction="Evaluate how Tstr cells were defined, their molecular identity, lineage composition, robustness across cohorts, and possible technical artifacts. Distinguish an analysis-derived population label from a stable lineage.",
            input=[task, paper_search],
        ),
        agent(
            role="cell-biologist",
            instruction="Assess tissue localization, disease-context distribution, treatment-associated changes, and outcome associations using only features recovered from the primary study. Separate observation and correlation from demonstrated function or causality.",
            input=[task, paper_search],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="State the study-supported descriptions of Tstr cells, including discovery context, molecular identity, spatial evidence, and treatment association, while reconciling apparently conflicting findings. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, paper_search, identity_review, context_review],
    )
    return final
