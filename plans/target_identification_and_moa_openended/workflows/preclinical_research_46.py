async def workflow(task):
    standards = await agent(
        role="literature-searcher",
        instruction="Retrieve best-practice guidance for bulk RNA-seq studies comparing primary tumors with paired liver metastases. Cover study design, quality control, sample identity, patient pairing, batch and clinical covariates, tumor composition, functional interpretation, and validation.",
        input=task,
    )

    analysis_design, statistics_design = await parallel(
        agent(
            role="geneticist",
            instruction="Develop complementary analytical perspectives from raw counts through molecular comparison, pathway and regulatory interpretation, cell-composition assessment, subtype change, and integration with genomic or clinical features. State the input required for each and do not pretend to analyze unavailable files.",
            input=[task, standards],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Specify models that respect within-patient pairing, effect-size and uncertainty reporting, covariate handling, false-discovery control, sensitivity analysis, and external validation.",
            input=[task, standards],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Present a prioritized end-to-end set of analytical perspectives and the research question answered by each, emphasizing the paired design and limitations on causal interpretation. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, standards, analysis_design, statistics_design],
    )
    return final
