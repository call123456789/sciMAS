async def workflow(task):
    causal_evidence = await agent(
        role="literature-searcher",
        instruction="Independently determine the causal gene and mutation classes in Duchenne muscular dystrophy, then retrieve gene-editing studies that restore protein expression and muscle function. Separate preclinical models from human evidence and compare editing with non-editing genetic modalities.",
        input=task,
    )

    editing_review = await agent(
        role="geneticist",
        instruction="Evaluate mutation-dependent editing feasibility, reading-frame or precise-correction logic, delivery to skeletal and cardiac muscle, mosaicism, durability, off-target and on-target risks, and functional rescue. Do not invent a guide or edit without a patient variant.",
        input=[task, causal_evidence],
    )

    translation_review = await agent(
        role="pharma-data-specialist",
        instruction="Verify gene/protein identity and modality-specific clinical or regulatory status. Keep genome editing, RNA splicing, and gene addition distinct, and distinguish expression from functional improvement.",
        input=[task, causal_evidence, editing_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select one primary gene-editing target and explain direct causality, feasible editing strategies at a high level, functional evidence, patient-variant dependence, and translational limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, causal_evidence, editing_review, translation_review],
    )
    return final
