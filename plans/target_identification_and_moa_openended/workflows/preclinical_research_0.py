async def workflow(task):
    identity_search = await agent(
        role="literature-searcher",
        instruction="Resolve the unnamed compound described in the question using independent discovery publications, provenance records, and first-in-human literature. Establish synonyms, developer, chronology, and the exact evidence reported in the primary preclinical study without assuming identity from nearby tasks.",
        input=task,
    )

    registry_review = await agent(
        role="pharma-data-specialist",
        instruction="Verify compound identity across authoritative drug and trial records. Report sponsor, registration IDs, dated phase and status, indications, and whether the registry and discovery paper concern the same molecule.",
        input=[task, identity_search],
    )

    pharmacology_review = await agent(
        role="molecular-biologist",
        instruction="Evaluate target selectivity, downstream target engagement, model-dependent antitumor effects, exposure, and safety claims using the retrieved studies. Distinguish preclinical optimization from demonstrated clinical superiority.",
        input=[task, identity_search, registry_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Identify the compound only if the sources converge, then summarize the independently verified pharmacological, preclinical, development, and current clinical facts while reporting identity or source mismatches. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, identity_search, registry_review, pharmacology_review],
    )
    return final
