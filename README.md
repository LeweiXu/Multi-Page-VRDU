# Multi-Page Visually Rich Document Understanding

This repo contains various documents and code for my Honours project. Project Proposal available here:
https://www.overleaf.com/read/ntdnhjhxkgqt#386d92

## Progress

- [X] Project Proposal
- [x] Gather Models
- [X] Gather Datasets
- [X] Background & General Knowledge Catch-up
- [ ] Write Literature Review (in progress)
- [ ] Write Survey Paper (optional)
- [ ] Explore potential MP-VRDU solutions
- [ ] Implement code
- [ ] Experiment Setup
- [ ] Write Dissertation

## Folders

- `dependencies_app`: a web application to help visualise background/backbone works for general LLM/MLLMs and MP-VRDU. Available [here](https://leweixu.github.io/MP-VRDU/). Note that the information is just a visualisation, and may not be entirely accurate.
- `models_main`: contains the articles/papers for multi-page VRDU, in pdf format downloaded from either arxiv or directly from the conference.
- `models_main_md`: contains .md conterpart to `models_main` using a pdf-parser, specifically `pymupdf4llm` which preserves the structure of tables and text exceedingly well. For academic papers, this is enough for a LLM such as Claude/GPT to perform well on understanding the documents.
- `datasets_main`: contains the articles/papers for multi-page VRDU datasets, as well as .md conterparts to plug in to LLMs.
- `seminar`: contains seminar documents.
- `survey_paper`: contains code/documents for the survey paper / literature review.