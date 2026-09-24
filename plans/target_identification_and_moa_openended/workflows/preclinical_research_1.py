async def workflow(task):
    paper_search = await agent(
        role="literature-searcher",
        instruction="Retrieve primary studies of CYH33 in head and neck squamous cell carcinoma, including accessible main and supplementary results. Extract target-engagement, signaling, genetic perturbation, viability, apoptosis, and in-vivo evidence with figure or table traceability.",
        input=task,
    )

    mechanism_review = await agent(
        role="molecular-biologist",
        instruction="Construct an evidence-linked mechanism from direct target engagement to downstream signaling and antitumor phenotype. Distinguish core causal mechanisms from pathway crosstalk, correlations, effects shown only in other tumor types, and untested immune or combination claims.",
        input=[task, paper_search],
    )

    final = await agent(
        role="synthesizer",
        instruction="Describe only the CYH33 antitumor mechanisms demonstrated in HNSCC, linking each claim to the relevant experiment and explicitly labeling weak, indirect, or untested mechanisms. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, paper_search, mechanism_review],
    )
    return final
