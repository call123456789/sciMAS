async def workflow(task):
    records = await agent(
        role="literature-searcher",
        instruction="Retrieve authoritative cell-line records and primary publications for FaDu. Extract species, donor and tissue origin, histology, authentication identifiers, genomic or protein characteristics, common research uses, and source-specific drug-response evidence.",
        input=task,
    )

    interpretation = await agent(
        role="cell-biologist",
        instruction="Separate stable identity facts from variable phenotypes caused by assay, culture, passage, or subline. Evaluate which descriptions are consistently supported and which require source-specific qualification.",
        input=[task, records],
    )

    final = await agent(
        role="synthesizer",
        instruction="Provide the verified description of FaDu and its research use, distinguishing authoritative identity facts from context-dependent molecular or drug-response traits. If choices are absent, do not invent them. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, records, interpretation],
    )
    return final
