async def workflow(task):
    evidence = await agent(
        role="literature-searcher",
        instruction="Retrieve authoritative EGFR biology and oncology sources covering receptor activation, normal function, alteration classes, tumor distribution, and therapeutic contexts. Build a fact set broad enough to assess candidate statements, without assuming which one is incorrect.",
        input=task,
    )

    mechanism = await agent(
        role="molecular-biologist",
        instruction="Explain EGFR structure, activation, and downstream signaling, and distinguish mutation, amplification, overexpression, ligand-driven activity, and tumor-type prevalence.",
        input=[task, evidence],
    )

    clinical_context = await agent(
        role="pharma-data-specialist",
        instruction="Verify representative tumor- and variant-specific EGFR alterations and target-directed treatments. Distinguish protein-directed therapy from mutation-selective inhibition and avoid treating every alteration as predictive.",
        input=[task, evidence, mechanism],
    )

    final = await agent(
        role="synthesizer",
        instruction="Identify an incorrect statement only if it is present in the task or recovered from an authoritative source; because choices are absent, otherwise state the verified EGFR facts and explain that a unique option cannot be selected. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, evidence, mechanism, clinical_context],
    )
    return final
