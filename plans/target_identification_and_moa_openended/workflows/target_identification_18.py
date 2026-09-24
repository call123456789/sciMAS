async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify therapeutic gene targets for tuberculosis. Explicitly separate pathogen genes from host-directed targets, and retrieve evidence for essentiality, virulence, persistence, intervention effects, resistance, and host-defense liabilities.",
        input=task,
    )

    identity_review = await agent(
        role="pharma-data-specialist",
        instruction="Resolve organism, gene and protein identity for each candidate and verify any direct drug or modality relationship. Keep pathogen annotations distinct from human identifiers and distinguish antigens, biomarkers, and drug targets.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="molecular-biologist",
        instruction="Compare candidates under a clearly stated host-versus-pathogen scope. Rank them by causal relevance, feasible intervention direction, selectivity, validation level, and risk of impairing protective immunity.",
        input=[task, discovery, identity_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly three genes, giving organism, protein function, intervention direction, tuberculosis-specific evidence, and the main resistance or host-safety caveat. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, identity_review, ranking],
    )
    return final
