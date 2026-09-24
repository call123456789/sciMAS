async def workflow(task):
    standards = await agent(
        role="literature-searcher",
        instruction="Retrieve best-practice guidance for analyses that follow differential-expression testing between paired primary tumors and liver metastases. Cover functional interpretation, regulatory hypotheses, composition, clinical integration, reproducibility, and validation.",
        input=task,
    )

    biology_plan, statistics_plan = await parallel(
        agent(
            role="geneticist",
            instruction="Propose evidence-based analytical angles for the supplied DEG result and map each to a distinct research aim. Preserve gene direction, patient labels, cancer context, and identifier provenance, and avoid inventing findings without the actual gene list.",
            input=[task, standards],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Explain the null hypotheses, background sets, multiple-testing controls, sensitivity analyses, and validation needed for each proposed downstream analysis. Separate exploratory interpretation from prediction and causality.",
            input=[task, standards],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Describe the strongest downstream analytical angles and corresponding research aims without claiming particular pathways or processes in the absence of the DEG list. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, standards, biology_plan, statistics_plan],
    )
    return final
