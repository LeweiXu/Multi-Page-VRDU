# Practice Seminar Presentation

## Slide 1: Project Description

### Slide Contents

Training-Free Agentic RAG Framework for Multi-Page Visually Rich Document Understanding

- Document Understanding: tasks such as question answering, information extraction, summarisation
- Visually Rich: not just text, but also charts, plots, images, tables etc.
- Multi-page: quadratic increase in compute and memory required, multi-hop reasoning, answer page/passage retrieval

Possibly add diagram here.

### Slide notes

- Go through project title (in reverse, as makes more sense that way)
- Document understanding involves allowing computers to perform various tasks on documents.
- In reality, all tasks are just an offshoot of question answering, KIE, summarisation is just answering a question in a specific way
- Visually rich documents also contain visual elements (examples on slide), need to reason about layout, of the page as well now e.g. how does the table relate to the plot?
- Single or <10 page document VRDU is very good with SOTA MLLMs, but with larger documents, scanned images, documents >100, run into context window problem. 

## Slide 2: Motivaion

### Slide contents

Cut costs
Automate pipelines that would take humans a long long time
old documents that are paper only and scanned into image pdfs can be mass processed and sorted this way
Add other slide contents here.

### Slide notes

Not too sure, write some good notes for me

## Slide 3: Solutions / Current Approaches

### Slide contents:
3 main approaches:
1. (most intuitive but also worst performance): scale single page VRDU models/approaches to multi-page setting
2. (ultimately not addressing the fundamental problem): try to compress document representation to fit within context window, possibly try only keep information relevant to query using cross-attention
3. (also not addressing fundamental problem of context window, but most promising): RAG to retrieve answer passage(s)/pages(s), possibly using agentic framework to iteratively retrieve/refine selection.

## Slide Notes
Write some notes for the above slide contents that I can look off during the seminar.
Also talk loop back to first slide about "Training free agentic RAG framework"
    - Training models from scratch is expensive, use existing powerful LLM backbone to generate answer from answer passages/pages