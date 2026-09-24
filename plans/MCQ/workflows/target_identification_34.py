async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify regulatory nodes capable of improving multiple components of metabolic syndrome at once. Retrieve network, perturbation, patient, and intervention evidence, and compare system-wide reach with tissue-specific effects.",
        input=task,
    )

    network_review, translation_review = await parallel(
        agent(
            role="structural-biologist",
            instruction="Compare pathway position, downstream breadth, tissue specificity, feedback, and whether each candidate is a single protein, multisubunit complex, or pathway label. Do not infer centrality without network evidence.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify precise molecular identity, direct versus indirect modulators, human evidence across insulin resistance, dyslipidemia, and hypertension, and safety or tissue-specific tradeoffs.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select one best-supported regulatory node, define its molecular identity precisely, map supported effects to each requested syndrome component, and distinguish mechanistic potential from demonstrated human benefit. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, network_review, translation_review],
    )
    return final
