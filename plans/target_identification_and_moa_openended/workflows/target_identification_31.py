async def workflow(task):
    etiology_review = await agent(
        role="literature-searcher",
        instruction="Independently map the major etiologic and physiological mechanisms of peptic ulcer disease to evidence-based interventions. Retrieve effects on healing, recurrence, bleeding, cause removal, and mucosal protection without presupposing a molecular target.",
        input=task,
    )

    target_review = await agent(
        role="pharma-data-specialist",
        instruction="For each intervention class, identify the direct molecular or organism-level target where one is well defined, verify clinical use, and keep host proteins, microbial factors, biomarkers, and therapeutic strategies distinct.",
        input=[task, etiology_review],
    )

    mechanism_review = await agent(
        role="molecular-biologist",
        instruction="Compare how the validated interventions alter causation, secretion, injury, defense, or repair. Explain required modulation direction and etiology-dependent relevance, and avoid inventing a single gene when the intervention acts on a complex or an organism.",
        input=[task, etiology_review, target_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Present a compact comparison of the most important intervention targets or target classes, their basic functions, treatment effects, etiologic scope, and limitations. Do not impose a count absent from the question. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, etiology_review, target_review, mechanism_review],
    )
    return final
