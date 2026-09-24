async def workflow(task):
    discovery = await agent(
        role="literature-searcher",
        instruction="Independently identify type 2 diabetes therapeutic targets supported by patient pathway dysregulation, causal disease biology, and intervention studies. Cover complementary aspects of glucose homeostasis and retain unsuccessful or safety-limited candidates.",
        input=task,
    )

    tractability = await agent(
        role="pharma-data-specialist",
        instruction="Verify the shortlist's canonical target identities, direct modulation by small molecules or biologics, selectivity, clinical status, and whether evidence concerns glycemic control or longer-term disease progression.",
        input=[task, discovery],
    )

    ranking = await agent(
        role="molecular-biologist",
        instruction="Rank candidates by pathogenic involvement, druggability, patient dysregulation, tissue mechanism, required modulation direction, and safety. Avoid treating pathway membership or a downstream drug effect as direct target validation.",
        input=[task, discovery, tractability],
    )

    final = await agent(
        role="synthesizer",
        instruction="Select exactly three targets with gene/protein identity, modulation direction, mechanism, translational evidence, and limitations on disease-modification claims. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, discovery, tractability, ranking],
    )
    return final
