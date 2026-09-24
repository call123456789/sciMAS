async def workflow(task):
    methods_review = await agent(
        role="literature-searcher",
        instruction="Retrieve methodological guidance for determining whether two molecules that activate the same pathway are physically linked, ordered in a signaling hierarchy, functionally dependent, or independent parallel inputs. Focus on evidence standards and common confounders.",
        input=task,
    )

    design = await agent(
        role="molecular-biologist",
        instruction="Design a compact set of complementary experiments chosen independently from the evidence review. For each, state the relationship it can establish, expected patterns under competing models, essential controls, rescue or orthogonal confirmation, and limitations. The identities of molecules A and B are unspecified, so keep the design general.",
        input=[task, methods_review],
    )

    final = await agent(
        role="synthesizer",
        instruction="Present the complementary experimental approaches needed to distinguish direct association, upstream-downstream order, mediation, and parallel activation, explaining what each can and cannot prove. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, methods_review, design],
    )
    return final
