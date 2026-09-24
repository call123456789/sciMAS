async def workflow(task):
    evidence = await agent(
        role="literature-searcher",
        instruction="Independently retrieve mechanistic and clinical evidence explaining why MEK inhibitors have limited broad use across solid tumors. Cover adaptive, intrinsic, and acquired resistance, toxicity, pharmacokinetics, tumor-genotype dependence, combinations, and successful exceptions.",
        input=task,
    )

    mechanism_review, clinical_review = await parallel(
        agent(
            role="molecular-biologist",
            instruction="Organize the discovered biological limitations into a causal taxonomy without prespecifying the mechanisms. Distinguish perturbation evidence from association and MEK-specific effects from general kinase-inhibitor limitations.",
            input=[task, evidence],
        ),
        agent(
            role="pharma-data-specialist",
            instruction="Verify compound- and indication-specific dose limitations, adverse effects, pharmacokinetic constraints, response durability, and approved uses. Avoid transferring one drug's properties to the entire class.",
            input=[task, evidence],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Summarize the major mechanistic and clinical limitations with mechanism-to-outcome links, representative contexts, and exceptions where MEK inhibition is effective. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, evidence, mechanism_review, clinical_review],
    )
    return final
