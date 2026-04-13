# Practice Seminar — Speaker Notes
## Training-Free Agentic RAG Framework for MP-VRDU
**Lewei Xu | BACS (Honours)**

---

## Slide 1: Project Description

**Opening — work backwards through the title**

> "I'll walk through the title in reverse order, because it actually makes more logical sense that way."

**"Document Understanding"**
- Computers performing tasks on documents: question answering, information extraction, summarisation.
- In practice, most document tasks are variants of QA — even summarisation is just "answer the question: *what is this document about?*"
- This is a well-studied area, and short, clean documents are largely *solved*.

**"Visually Rich"**
- Real-world documents are not plain text files. They contain *tables, charts, figures, photographs, scanned handwriting*.
- To understand these documents fully, a model must reason about **layout**: how does the table on page 3 relate to the bar chart on page 4? Which caption belongs to which figure?
- This is qualitatively harder than pure text QA.

**"Multi-Page"**
- Single-page VRDU is now very good — state-of-the-art multimodal LLMs handle it well.
- But when documents are long (think: a 200-page legal contract or a full annual report), three problems emerge:
  1. **Context window** — feeding 200 page images into a model simultaneously is computationally infeasible or exceeds the model's limit entirely.
  2. **Multi-hop reasoning** — the answer to a question might require combining information from page 12 and page 87.
  3. **Retrieval** — before you can answer, you need to *find* the right page(s).

*(Point to the diagram)* The pipeline is: a long scanned document comes in, a question is asked, and we need to extract a precise answer — that's the MP-DocVQA task.

---

## Slide 2: Motivation

**Opening — set the scene**

> "Why does this actually matter outside of academia?"

**Cost reduction**
- Document review is one of the most expensive professional tasks in law, finance, medicine, and government.
- A junior lawyer or analyst might spend weeks manually reviewing contracts looking for specific clauses. If we can automate even part of that, the cost savings are enormous.

**Automation of slow pipelines**
- Many organisations have workflows — invoice processing, regulatory filing review, scientific literature screening — that currently require human experts reading every page. These pipelines are bottlenecks.
- Automated document understanding can run at scale, 24/7, without fatigue.

**Legacy scanned documents**
- Decades of paper records have been scanned into image-PDFs. They're stored but essentially *unsearchable* — you can't Ctrl+F a scanned image.
- A system that can reason over scanned documents at scale would unlock the value of enormous historical archives: medical records, court filings, historical newspapers.

**Scalability gap**
- Current tools (even GPT-4o, Claude, Gemini) perform well on short documents. Show them a 5-page invoice — great. Show them a 300-page insurance policy — the context window breaks, performance degrades sharply.
- *This* is the gap this project targets.

---

## Slide 3: Solutions / Current Approaches

**Framing — three families**

> "Researchers have tried three broad strategies. I'll walk through them in order of increasing promise."

**Approach 1 — Scale single-page models**
- The simplest idea: just take a model that works on one page and feed it all pages at once.
- Problem: context windows are finite. Even with very long contexts, performance degrades — models "forget" early pages by the time they process late ones. This is sometimes called the *lost-in-the-middle* problem.
- Benchmark results consistently show this is the weakest approach.

**Approach 2 — Context compression**
- Instead of feeding everything, try to compress the document so it *fits* in the context window.
- One technique is cross-attention: score each page/passage by its relevance to the query, keep only the top-scoring pieces.
- The problem: you're still fundamentally fighting against the context-window constraint rather than solving it. And compression loses visual information — tables and figures are especially hard to compress without losing meaning.

**Approach 3 — RAG with an agentic loop** *(this is the project's direction)*
- RAG = Retrieval-Augmented Generation. Instead of stuffing everything in, *retrieve* the relevant pages first, then answer.
- **Agentic** means the model doesn't just retrieve once — it can iteratively refine: "I found page 14 and page 67, but I need one more piece of evidence — let me search again." This loop continues until the model is confident.
- This is the most promising direction in current literature.

**Loop back to the title**
> "And this is exactly what 'Training-Free Agentic RAG' means in my project title."
- **Training-free** — we don't train or fine-tune a new model from scratch (which is extremely expensive, requires massive GPU clusters, months of compute).
- Instead, we *use* an existing powerful MLLM (like a vision-language model) as our backbone and build the agentic retrieval framework *around* it.
- The model's visual and language understanding is already there — our contribution is the *framework* that tells it which pages to look at and when to stop retrieving.

---

*End of notes.*