async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify Huntington-disease therapeutic targets supported by causal disease biology, neuronal dysfunction, target-directed interventions, and progression outcomes. Include delivery, safety, failed trials, and symptomatic-versus-disease-modifying distinctions.",
        input=task,
    )

    causal_review, translation_review = await parallel(
        agent(
            role="geneticist",
            instruction="Assess causal genetic evidence, toxic gain or loss of function, neuronal and circuit effects, patient applicability, and the justified direction and specificity of genetic or molecular modulation.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify target identity, available modality, CNS delivery, target engagement, trial status, functional progression endpoints, and effects on normal protein function.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two targets with complementary disease mechanisms, intervention directions, evidence maturity, delivery constraints, and disease-modification uncertainty. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, causal_review, translation_review],
    )
    return final
