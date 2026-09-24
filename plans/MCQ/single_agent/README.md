# MCQ Single-Agent Workflows

One force-single-agent sciMAS workflow for each of the 50 questions in `dataset/MCQ.jsonl`. Each file has exactly one agent call; its instruction assigns that worker the full analysis, evidence gathering, option-by-option evaluation, self-check, and final response.

The manifest records public IDs, order, task type, workflow path, and selected role. It contains no answer key, rationale, or hidden evidence. Workflow instructions do not state a preferred option or introduce answer-specific facts absent from the runtime task.
