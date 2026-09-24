async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify Alzheimer-disease targets involved in core pathogenic pathways and evaluate target-directed preclinical and clinical evidence. Include negative trials, safety findings, target engagement, biomarker effects, cognition, and progression outcomes.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify target identities, direct intervention mechanisms, clinical-development status, and distinctions among reducing production, clearing pathology, preventing propagation, and altering downstream signaling.",
        input=[task, discovery],
    )

    assessment = await agent(
        role="molecular-biologist",
        instruction="Compare pathogenic centrality, intervention direction, physiological functions, human genetic support, and whether molecular or biomarker effects translated into clinical benefit.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two targets with mechanisms, modulation directions, evidence tiers, clinical successes or failures, and present-day limitations on a disease-modifying recommendation. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, assessment],
    )
    return final
