async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify colorectal-cancer targets satisfying pathogenic involvement, practical modulation, and patient pathway dysregulation. Retrieve subtype and variant context, causal evidence, direct or indirect interventions, clinical outcomes, and resistance.",
        input=task,
    )

    driver_review, translation_review = await parallel(
        agent(
            role="geneticist",
            instruction="Classify each candidate's alteration and causal role, infer the correct intervention direction, and separate causal importance from direct druggability. Preserve variant and disease-stage context.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify target identity, direct binding versus pathway modulation, molecular eligibility, approved or investigational combinations, and colorectal-cancer-specific outcomes.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two targets with gene/protein identity, alteration and intervention direction, pathogenic mechanism, druggability, eligible subgroup, and limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, driver_review, translation_review],
    )
    return final
