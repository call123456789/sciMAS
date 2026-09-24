async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently retrieve accessible-sample biomarkers evaluated for both diagnosis and longitudinal prognosis in type 2 diabetes. Extract specimen, analyte, assay, diagnostic standards, complication or progression outcomes, external validation, and major confounders.",
        input=task,
    )

    assay_review, validity_review = await parallel(
        agent(
            role="analytical-chemist",
            instruction="Evaluate the discovered biomarkers for analyte identity, specimen compatibility, assay standardization, biological stability, analytical interference, and feasibility in routine blood-based measurement.",
            input=[task, discovery],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Compare diagnostic discrimination and prognostic validity, distinguishing established thresholds from exploratory cutoffs and correlation from incremental prediction. Assess cohort design, calibration, validation, and uncertainty.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Choose one best-supported biomarker and state what is measured, the specimen, diagnostic and prognostic uses, validated thresholds only when supported, and important limitations. Do not force the result to be a gene. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, assay_review, validity_review],
    )
    return final
