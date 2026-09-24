async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify strategic gene targets in psoriasis using human immune-pathway evidence and selective intervention trials. Compare effects on immune-cell maintenance, effector signaling, keratinocyte responses, skin clearance, relapse, and safety.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify canonical target identities and the exact ligand, receptor, or shared subunit affected by each intervention. Preserve approval status, efficacy endpoints, and disease-specific safety limitations.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="cell-biologist",
        instruction="Rank candidates by causal skin immunology, specificity, therapeutic efficacy, durability, and safety. Distinguish upstream maintenance signals from downstream effector activity and avoid conflating related family members.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two genes with precise protein identities, psoriasis mechanisms, intervention directions, evidence maturity, and principal limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
