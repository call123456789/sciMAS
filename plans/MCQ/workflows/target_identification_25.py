async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify schizophrenia therapeutic target genes that modulate neural signaling. Retrieve human pharmacology, target-directed clinical outcomes, genetic and functional evidence, symptom-domain effects, and negative findings.",
        input=task,
    )

    translation = await agent(
        role="pharma-data-specialist",
        instruction="Verify canonical gene/protein subtype, direct drug action, selectivity, indication, and the contribution of polypharmacology. Distinguish established treatment targets from research associations.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="molecular-biologist",
        instruction="Rank a minimal evidence-supported set by neural mechanism, intervention direction, clinical validation, symptom scope, and adverse effects. The question gives no fixed count, so justify the selected number.",
        input=[task, discovery, translation],
    )

    final = await agent(
        role="synthesizer",
        instruction="Present a concise ranked set of target genes with protein identities, modulation directions, therapeutic evidence, symptom scope, and disease-modification limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, translation, ranking],
    )
    return final
