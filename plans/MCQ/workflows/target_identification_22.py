async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify central immune-regulatory gene targets in SLE across innate sensing, lymphocyte tolerance, costimulation, and effector pathways. Retrieve causal perturbation, human genetics, patient dysregulation, and intervention evidence.",
        input=task,
    )

    regulatory_review, translation_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Compare each candidate's fundamental regulatory role, relevant cell types, pathway position, feedback, and beneficial modulation direction. Identify where suppressing a pathway could also remove protective immune control.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify direct targetability, exact ligand/receptor or pathway component affected by available modalities, SLE trial outcomes, and safety. Separate mechanistic centrality from clinical validation.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes that best balance fundamental immune-regulatory importance and therapeutic feasibility, with intervention direction and evidence limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, regulatory_review, translation_review],
    )
    return final
