async def workflow(task):
    data_review = await agent(
        role="pharma-data-specialist",
        instruction="Retrieve authoritative lestaurtinib pharmacology and public cancer-cell-line response data stratified by PIK3R1 status. Preserve compound identifiers, dataset release, response metric and orientation, mutation definition, sample counts, and tissue composition without presuming the effect direction.",
        input=task,
    )

    mechanism_review, statistics_review = await parallel(
        agent(
            role="molecular-biologist",
            instruction="Explain PIK3R1 biology and verify lestaurtinib's direct target profile. Assess mechanistic hypotheses only after the response association is established, and separate direct pharmacology from indirect pathway effects.",
            input=[task, data_review],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Determine the direction, magnitude, uncertainty, significance, multiple-testing burden, and lineage sensitivity of the mutant-versus-wild-type comparison. Distinguish association from biological resistance.",
            input=[task, data_review],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Report whether and how PIK3R1 status is associated with lestaurtinib response in the identified dataset, while keeping response direction, target pharmacology, and mechanism separate. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, data_review, mechanism_review, statistics_review],
    )
    return final
