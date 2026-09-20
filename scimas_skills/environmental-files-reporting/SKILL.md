---
name: environmental-files-reporting
description: Use for environmental calculation JSON file loading and report saving.
x-scimas-role: environmental-chemist
x-scimas-server: chemistry-environmental
x-scimas-tools:
  - load_json_file
  - save_calculation_report
---
# Environmental Files And Reporting

Use this skill only when the step explicitly requires loading a JSON input file or saving a calculation report.

Do not use file tools for ordinary calculations when the needed values are already in the prompt or context bundle.
