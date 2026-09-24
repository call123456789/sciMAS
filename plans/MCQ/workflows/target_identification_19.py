async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify therapeutic targets in triple-negative breast cancer using functional-dependency studies, tumor-versus-normal expression evidence, subtype-specific mechanisms, and target-directed treatment studies. Preserve patient-selection requirements and negative evidence.",
        input=task,
    )

    dependency_review, translation_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Assess whether each candidate is required for tumor-cell survival or proliferation, whether perturbation and rescue establish causality, how heterogeneous the dependency is, and whether normal-tissue expression permits a therapeutic window.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify target identity, available modality, direct target engagement, clinical evidence, molecular eligibility, and toxicity. Separate a drug's binding target from payload or combination effects.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Choose one target that best satisfies all three criteria, with eligible tumor context, mechanism, modality, evidence level, and any criterion that remains incompletely met. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, dependency_review, translation_review],
    )
    return final
