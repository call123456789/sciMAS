async def workflow(task):
    result = await agent(
        role="cell-biologist",
        instruction="Work as the sole specialist for this multiple-choice problem. First parse the exact question, requested answer count, and every option, including negations such as \"NOT supported\", \"incorrect\", and \"all of the above\". Independently establish the relevant facts using appropriate authoritative sources or domain tools when available; prioritize primary studies, registries, curated datasets, and official documentation, and distinguish direct evidence from inference. Evaluate each option against the question rather than selecting by familiarity, reject unsupported claims, then check that the chosen set has the requested size and answers the exact wording. Return the option letter(s), the option text, a concise evidence-based justification for each selection, and brief reasons for rejecting close alternatives. Include source names or identifiers and dates when retrievable; state uncertainty or unavailable evidence instead of inventing details. Do not assume any candidate or answer in advance. Verify the precise tumor cohort, HSF1 expression measure, CD8 infiltration estimate, named tool/database version, and correlation direction. Distinguish correlation from causation and check whether tool-specific outputs are being conflated.",
        input=task,
    )
    return result
