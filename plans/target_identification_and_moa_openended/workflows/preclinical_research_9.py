async def workflow(task):
    evidence = await agent(
        role="literature-searcher",
        instruction="Independently retrieve medicinal-chemistry and oncology evidence on how dual-target inhibitors are designed, validated, and used. Extract target-pair rationale, molecular architecture, binding evidence, pathway effects, resistance outcomes, and toxicity.",
        input=task,
    )

    design_review, systems_review = await parallel(
        agent(
            role="organic-chemist",
            instruction="Compare the principal high-level medicinal-chemistry strategies for achieving balanced activity at two targets. Discuss binding-site compatibility, selectivity, physicochemical constraints, and target-engagement evidence without providing a synthesis recipe.",
            input=[task, evidence],
        ),
        agent(
            role="cell-biologist",
            instruction="Assess the systems-level rationale and resistance consequences of simultaneously inhibiting two oncogenic nodes. Distinguish demonstrated synergy from theoretical coverage and include toxicity tradeoffs.",
            input=[task, evidence],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Explain which statements about dual-target inhibitor design and action are generally supported, which are conditional, and which overgeneralize, using concrete evidence criteria. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, evidence, design_review, systems_review],
    )
    return final
