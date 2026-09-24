async def workflow(task):
    data_review = await agent(
        role="pharma-data-specialist",
        instruction="Locate authoritative pharmacogenomic records or publications for NSC-87877 across cancer cell lines. Preserve compound identifiers, target profile, dataset release, response metric and direction convention, genomic annotations, cell-line counts, and tissue labels. Do not assume which genomic subgroup is associated with sensitivity.",
        input=task,
    )

    biology_review, statistics_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Assess plausible determinants of NSC-87877 response using only variants or expression features discovered in the upstream evidence. Distinguish compound-target pharmacology, lineage effects, association, and a causal dependency.",
            input=[task, data_review],
        ),
        agent(
            role="statistical-mathematician",
            instruction="Audit any genomic subgroup comparison for response direction, group size, distribution, effect size, uncertainty, multiple testing, and tissue confounding. Explain when the evidence is insufficient without fabricating values.",
            input=[task, data_review],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="State the supported cell-line drug-sensitivity relationship, if reproducible, with the exact genomic feature, response-metric direction, dataset scope, and limits on causal interpretation. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, data_review, biology_review, statistics_review],
    )
    return final
