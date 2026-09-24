async def workflow(task):
    search = await agent(
        role="literature-searcher",
        instruction="Systematically retrieve colorectal-cancer liver-metastasis studies relevant to the unspecified statements implied by the question, prioritizing paired tissue, proteomic, transcriptomic, and multi-omics analyses. Extract exact claims, cohorts, comparisons, methods, enrichment outputs, and supplementary-table locations.",
        input=task,
    )

    audit = await agent(
        role="mass-spectrometrist",
        instruction="Audit the retrieved proteomic claims for measurement type, differential-protein definition, annotation category, rank order, platform dependence, and study-specific scope. Separate directly reported findings from plausible but unreported interpretations.",
        input=[task, search],
    )

    final = await agent(
        role="synthesizer",
        instruction="Because no answer choices are provided, identify any concrete claim that cannot be substantiated only if the search exposes it; otherwise state that the missing statements prevent a unique selection and summarize the evidence boundary. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, search, audit],
    )
    return final
