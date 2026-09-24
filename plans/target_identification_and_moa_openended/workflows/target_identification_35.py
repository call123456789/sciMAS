async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify therapeutic gene candidates for idiopathic pulmonary fibrosis using human lung dysregulation, causal fibroblast or epithelial biology, genetic or pharmacological perturbation, and intervention evidence. Include negative clinical translation and safety constraints.",
        input=task,
    )

    mechanism_review, translation_review = await parallel(
        agent(
            role="cell-biologist",
            instruction="Compare candidates for roles in epithelial injury, fibroblast activation, myofibroblast persistence, oxidative or growth-factor signaling, and matrix deposition. Distinguish expression association from causal perturbation.",
            input=[task, discovery],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify exact target identity, direct versus indirect pharmacology, selectivity, trial status, IPF-specific outcomes, and whether evidence comes from another fibrotic disease.",
            input=[task, discovery],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Select one most promising gene with intervention direction, IPF mechanism, human and preclinical evidence tiers, druggability, and the main translational limitation. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, mechanism_review, translation_review],
    )
    return final
