async def workflow(task):
    primary_source = await agent(
        role="literature-searcher",
        instruction="Retrieve the original Consensus Molecular Subtype study for colorectal cancer and its accessible supplements, plus later validation only where needed. Extract classification derivation, cohorts, subtype-defining expression programs, associated alterations, pathway analyses, and precisely defined clinical outcomes.",
        input=task,
    )

    molecular_review, outcome_review = await parallel(
        agent(
            role="geneticist",
            instruction="Map genetic alterations and pathway programs to CMS subtypes using the primary source. Distinguish subtype definitions from enrichment, common colorectal-cancer events, and later refinements.",
            input=[task, primary_source],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Audit prognosis statements by endpoint, time origin, comparison group, statistical model, and uncertainty. Keep relapse-free survival, overall survival, and survival after relapse distinct.",
            input=[task, primary_source],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="State which claims about CMS classification, molecular features, and outcomes are supported, qualified, or unsupported by the primary study. If no options are supplied, provide a verified CMS fact set rather than inventing option letters. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, primary_source, molecular_review, outcome_review],
    )
    return final
