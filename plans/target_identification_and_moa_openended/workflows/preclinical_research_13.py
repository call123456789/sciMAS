async def workflow(task):
    search = await agent(
        role="literature-searcher",
        instruction="Retrieve studies specifically comparing primary breast tumors with liver metastases. Extract study design, cohort composition, analytical modalities, molecular and cellular measurements, and exact reported conclusions without prespecifying the expected findings.",
        input=task,
    )

    genomic_review, microenvironment_review = await parallel(
        agent(
            role="geneticist",
            instruction="Assess the molecular findings and disease classifications recovered from each study. Keep cohort-specific alteration frequencies separate and exclude evidence from other metastatic organs unless explicitly used as a comparator.",
            input=[task, search],
        ),
        agent(
            role="cell-biologist",
            instruction="Evaluate the cellular and tissue-context findings recovered from each study, their measurement methods, and whether observations support association or a metastasis-driving mechanism.",
            input=[task, search],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Summarize which findings about primary and liver-metastatic breast tumors are explicitly present in the retrieved literature, preserving study and cohort provenance and noting unsupported generalizations. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, search, genomic_review, microenvironment_review],
    )
    return final
