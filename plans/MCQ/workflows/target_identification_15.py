async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify therapeutic gene candidates for controlling inflammation in ulcerative colitis. Retrieve disease-specific human mechanistic studies, guidelines, intervention trials, negative results, and clinically meaningful outcomes. Build a broad candidate table without privileging any candidate in advance.",
        input=task,
    )

    validation = await agent(
        role="pharma-data-specialist",
        instruction="For the discovered shortlist, verify canonical human gene/protein identities, direct versus indirect drug-target relationships, ulcerative-colitis indication status, trial maturity, and whether the intervention direction is supported.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Compare the validated candidates for causal inflammatory control, epithelial effects, intervention evidence, and immune-safety tradeoffs. Distinguish pathogenic drivers from protective responses and rank candidates using explicit evidence tiers.",
        input=[task, discovery, validation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes that best satisfy the question, with protein identity, modulation direction, ulcerative-colitis mechanism, evidence maturity, and the principal caveat for each. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, validation, ranking],
    )
    return final
