async def workflow(task):
    standards = await agent(
        role="literature-searcher",
        instruction="Retrieve current cell-bank, pharmacopoeial, or peer-reviewed guidance on detecting mycoplasma contamination in mammalian cell cultures. Compare validated assay classes by detection principle, sensitivity, specificity, controls, turnaround, and regulatory acceptance.",
        input=task,
    )

    biological_review, assay_review = await parallel(
        agent(
            role="molecular-biologist",
            instruction="Explain which mycoplasma-specific molecular or morphological features can support direct detection and what positive, negative, inhibition, and contamination controls are needed. Keep the discussion at method-principle level.",
            input=[task, standards],
        ),
        agent(
            role="analytical-chemist",
            instruction="Compare direct and indirect assay readouts, host-cell interference, false-positive and false-negative mechanisms, validation requirements, and when orthogonal confirmation is necessary.",
            input=[task, standards],
        ),
    )

    final = await agent(
        role="synthesizer",
        instruction="Identify the reliable detection method classes, their mechanistic bases, strengths, limitations, controls, and an appropriate confirmation strategy without giving organism-cultivation instructions. Use only evidence produced by upstream agents, preserve source identifiers and uncertainty, and answer the original question directly. Do not assume a candidate or conclusion before comparing the evidence.",
        input=[task, standards, biological_review, assay_review],
    )
    return final
