async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify Parkinson-disease targets with potential to alter progression. Retrieve human genetic and pathological evidence, target-directed preclinical studies, biomarker engagement, clinical outcomes, failed trials, and CNS-delivery or safety limits.",
        input=task,
    )

    causal_review, translation_review = await parallel(
        agent(
            role="geneticist",
            instruction="Assess causal genetic or functional links, patient-subgroup relevance, pathogenic direction, and whether inhibition, restoration, or clearance is justified by perturbation evidence.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify target identity, modality, CNS exposure, trial phase, target engagement, and clinical progression outcomes. Keep symptomatic targets separate from disease-modifying candidates.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly three molecular targets with gene/protein identities, pathogenic pathways, intervention directions, evidence tiers, eligible populations, and unproven claims. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, causal_review, translation_review],
    )
    return final
