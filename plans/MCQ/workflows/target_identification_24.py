async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify molecular targets for schizophrenia with fundamental roles in neural signaling or synaptic function and evidence from human pharmacology, genetics, or target-directed trials. Separate symptom domains and negative evidence.",
        input=task,
    )

    mechanism_review, clinical_review = await parallel(
        agent(
            role="molecular-biologist",
            instruction="Evaluate synaptic localization, signaling function, required agonism or antagonism, circuit context, and adverse-effect mechanisms. Separate disease association from a demonstrated therapeutic direction.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify canonical target subtype, drug selectivity, schizophrenia indication and trial outcomes. Account for polypharmacology and distinguish symptomatic efficacy from evidence of disease modification.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly two molecular targets with canonical gene/protein names, intervention direction, neural or synaptic rationale, clinical evidence, and limitations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, mechanism_review, clinical_review],
    )
    return final
