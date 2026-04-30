# Codex Survey Paper Handoff Prompt

You are working in `/home/lingwei/CITS4010` on an honours survey paper about multi-page visually rich document understanding (MP-VRDU).

Before making any edits, inspect the repository and infer the current state yourself. In particular, read:

- `AGENTS.md` if present or otherwise follow any repository instructions supplied by the session.
- `survey_source/latex/acl_latex.tex`
- `survey_source/latex/custom.bib`
- the generated model summary tables under `survey_source/latex/`
- the relevant CSV summaries under `survey_paper/`
- the original model markdown papers under `models_main_md/`
- the general MLLM VRDU survey notes in `survey_paper/Genera_MLLM_VRDU_Survey.md`

Your first response should not edit files. Instead, report:

1. The current structure and progress of the survey paper.
2. Which sections appear polished, drafted, incomplete, or still bullet-point-like.
3. Any obvious consistency issues, citation issues, duplicated content, or overlap with the general MLLM VRDU survey.
4. A concise recommendation for what section should be worked on next.

After that report, ask me what I want to do next. Do not assume the next task. When I give the next task, continue by reading the relevant model papers and CSV summaries before writing, and keep the writing concise, academic, and focused on the multi-page setting rather than general VRDU.

Important writing preferences:

- The survey should focus specifically on MP-VRDU, especially architectures, trends, methods, and challenges.
- Avoid unnecessary overlap with the general MLLM-based VRDU survey.
- When discussing models, do not merely list what each model does. Generalise into modelling patterns, trends, and trade-offs, with citations interleaved near the relevant method.
- Prefer concise article-ready LaTeX prose over bullet points.
- When revising a section, preserve the paper’s existing style where possible.
- Use citation keys from `survey_source/latex/custom.bib`; if a needed paper is missing, use a clear placeholder citation key and mention it.
