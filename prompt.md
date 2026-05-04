# Codex Survey Paper Handoff Prompt

You are working in `/home/lingwei/CITS4010` on an honours survey paper about multi-page visually rich document understanding (MP-VRDU).

Before making any edits, inspect the repository and infer the current state yourself. In particular, read:

- `AGENTS.md` if present or otherwise follow any repository instructions supplied by the session.
- current progress of the paper in `survey_paper/main.tex`
- the bibliography in `survey_paper/custom.bib`
- the relevant CSV summaries under `survey_paper/`
- the original model markdown papers under `models_main_md/`
- the general MLLM VRDU survey notes in `survey_paper/General_MLLM_VRDU_Survey.md`

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
- Use citation keys from `survey_paper/custom.bib`; if a needed paper is missing, use a clear placeholder citation key and mention it.

Writing style to mimic the general MLLM VRDU survey:

- Write in compact academic survey prose. Each sentence should add a concrete definition, mechanism, comparison, trend, benefit, limitation, or transition.
- Do not use the word "because". Prefer more academic causal constructions such as "as", "since", "thereby", "due to", "which", or a separate sentence stating the consequence.
- Avoid colons in prose. Prefer complete sentences over sentence fragments joined by ":". Use colons only where LaTeX syntax, table captions, or unavoidable technical notation requires them.
- Prefer full, declarative sentences. Avoid conversational phrasing, rhetorical questions, and informal connectors.
- Avoid wording that only smooths the reading experience without adding information. Remove filler such as "in this context", "it is important to note", "this design is effective when", "the retrieved units", "the main point is", and similar scaffolding.
- Use the survey's paragraph structure where suitable: define the category, describe representative mechanisms or modelling choices, then state strengths and limitations.
- Use taxonomy-level phrasing before model-level detail. Prefer "Some frameworks...", "Other approaches...", "Recent work...", and "These methods..." over repeatedly naming individual models as sentence subjects.
- Keep citations close to the relevant method or claim. Use citations to support modelling patterns, not as a standalone list.
- Prefer concise mechanism descriptions over exhaustive implementation detail. Include only details needed to distinguish architectural trends, trade-offs, or MP-VRDU-specific challenges.
- Use contrastive academic transitions sparingly, especially "However", "Moreover", "Additionally", "In contrast", "In sum", and "Overall". Do not overuse any one transition.
- State limitations directly and specifically. Avoid vague limitations such as "scalability issues" unless the sentence identifies the source, such as context length, visual token growth, OCR error propagation, retrieval recall, compression loss, or tool latency.
- Avoid overclaiming. Use cautious formulations such as "often", "typically", "may", "can", and "tend to" when summarising a trend across heterogeneous models.
- Avoid direct model-by-model chronology unless the user explicitly requests a logical progression. For overview sections, synthesise models into architecture patterns and trade-offs.
- Keep MP-VRDU focus explicit. Mention multi-page evidence aggregation, cross-page reasoning, page-level structure, long-context constraints, or document-scale evidence whenever relevant.
- Do not repeat the same limitation in adjacent sentences. Merge overlapping statements about memory, computation, context length, or evidence bottlenecks.
- Avoid broad general VRDU background unless needed to position an MP-VRDU-specific argument.
