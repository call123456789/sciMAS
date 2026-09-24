async def workflow(task):
    standards = await agent(
        role="literature-searcher",
        instruction="Retrieve authoritative scRNA-seq data-format and preprocessing documentation explaining how common analysis objects represent data at different processing stages and how data provenance can be established.",
        input=task,
    )

    diagnosis = await agent(
        role="geneticist",
        instruction="Develop a non-destructive checklist for classifying an unspecified expression matrix from the retrieved standards and the observable properties of the matrix or analysis object. Explain the evidentiary strength, exceptions, and ambiguity of each independently chosen diagnostic.",
        input=[task, standards],
    )

    final = await agent(
        role="synthesizer",
        instruction="Give an ordered procedure for deciding whether the matrix contains raw counts or processed values, explain what each diagnostic supports, and state when provenance or an additional data layer is required. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, standards, diagnosis],
    )
    return final
