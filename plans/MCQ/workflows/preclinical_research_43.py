async def workflow(task):
    records = await agent(
        role="literature-searcher",
        instruction="Retrieve authoritative cell-line records and foundational publications for MCF7. Extract species, tissue and histology, authentication identifiers, molecular phenotype, common experimental uses, culture dependencies, subline drift, and xenograft conditions.",
        input=task,
    )

    interpretation = await agent(
        role="cell-biologist",
        instruction="Separate stable identity and well-replicated functional characteristics from receptor, growth, or tumorigenicity traits that vary by assay or subline. Compare candidate descriptions only from the evidence.",
        input=[task, records],
    )

    final = await agent(
        role="synthesizer",
        instruction="Provide the best-supported description of MCF7 and its principal research use, with concise caveats about variable phenotypes. If answer choices are absent, state the verified fact set. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, records, interpretation],
    )
    return final
