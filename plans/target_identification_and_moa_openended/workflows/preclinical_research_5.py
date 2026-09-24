async def workflow(task):
    clinical_review = await agent(
        role="literature-searcher",
        instruction="Independently retrieve current guideline and trial evidence for HNSCC treatment by anatomic site, stage, resectability, recurrence or metastasis, and clinically relevant subgroups. Extract efficacy, durability, toxicity, and limitations of local and systemic strategies.",
        input=task,
    )

    mechanism_review, therapy_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Analyze tumor-intrinsic and microenvironmental mechanisms that limit durable control, including heterogeneity, treatment adaptation, immune escape, and resistance to local or systemic therapy. Preserve disease-subsite and subgroup context.",
            input=[task, clinical_review],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify drug targets, approved indications, biomarker restrictions, line of therapy, and trial outcomes for targeted and immune treatments. Distinguish response rate, durable response, and survival.",
            input=[task, clinical_review],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Summarize major HNSCC treatment strategies by setting and connect each principal mechanistic or clinical bottleneck to the affected therapy and supporting evidence. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, clinical_review, mechanism_review, therapy_review],
    )
    return final
