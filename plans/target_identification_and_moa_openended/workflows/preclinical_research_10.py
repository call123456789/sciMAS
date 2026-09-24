async def workflow(task):
    registry_review = await agent(
        role="pharma-data-specialist",
        instruction="Search authoritative trial and drug records for CYH33 and verified synonyms. Return dated trial identifiers, phase, sponsor, status, indications, molecular eligibility, arms, combinations, endpoints, and listed biomarkers.",
        input=task,
    )

    publication_review = await agent(
        role="literature-searcher",
        instruction="Retrieve clinical publications or conference records for CYH33. Extract study population, dose-finding status, molecular selection, response evidence, pharmacodynamic measurements, safety, and publication date, and identify registered plans without public results.",
        input=[task, registry_review],
    )

    mechanism_review = await agent(
        role="molecular-biologist",
        instruction="Interpret predictive and pharmacodynamic biomarkers only where measured, and assess biological support for each combination. Separate preliminary activity from confirmed efficacy and target engagement from patient benefit.",
        input=[task, registry_review, publication_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Give a dated, trial-linked account of CYH33 clinical development, indications, combinations, biomarkers, reported outcomes, and regulatory status, explicitly resolving inconsistent sources. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, registry_review, publication_review, mechanism_review],
    )
    return final
