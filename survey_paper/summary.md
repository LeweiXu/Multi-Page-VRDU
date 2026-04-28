# MP-VRDU OCR Dependency and Architecture Categorization

This revision applies a stricter OCR-dependency rule: a model is **Hybrid Multimodal** only when OCR is explicitly used during inference and combined with visual or multimodal evidence. OCR used only for training, pretraining, supervision, or baselines is noted in the rationale but does not determine the OCR-dependency label.

## Category Legend

### OCR Dependency

- **OCR-Only**: inference depends on explicit OCR text or OCR layout tokens, without page images or visual evidence used by the answerer.
- **Hybrid Multimodal**: inference explicitly uses OCR text/tokens/layout and also uses page images, visual features, crops, screenshots, or VLM reasoning.
- **OCR-Free**: no explicit external OCR step is used during inference. This includes visual-only pipelines, VLM-internal reading, and methods that use PDF parsing or extracted text without naming OCR.

### Architecture

- **Retriever-Generator Architecture**: retrieves pages, chunks, images, graph nodes, or other evidence before answer generation.
- **Agentic Architecture**: uses agents, tools, iterative search, reflection, verification, or role-specialized reasoning stages.
- **Hierarchical/Global Transformer Architecture**: models page-to-document structure with global-local attention, page summaries, document tokens, or recurrent memory.
- **Backbone-Centric Multimodal Adaptation Architecture**: adapts the core LLM/MLLM through compression, resampling, layout tokens, instruction tuning, RL, adapters, or long-context streaming rather than adding a separate external reasoning pipeline.

## Model-Level Categorization

| # | Model | OCR dependency | Architecture | Rationale / key technique |
|---:|---|---|---|---|
| 1 | LayTokenLLM | OCR-Only | Backbone-Centric Multimodal Adaptation Architecture | Inference is centered on OCR text segments and their coordinates; the compact layout token injects spatial position into the LLM without using page images as answer evidence. |
| 2 | AVIR | OCR-Free | Retriever-Generator Architecture | Retrieves evidence from page images with a lightweight visual retriever, then sends selected pages to a frozen VLM generator; no explicit OCR inference step is required. |
| 3 | DFVC | OCR-Free | Retriever-Generator Architecture | Fuses neighboring-page context into visual page embeddings for retrieval and answer generation over page images, avoiding an OCR-based evidence pipeline. |
| 4 | Arctic-TILT | Hybrid Multimodal | Hierarchical/Global Transformer Architecture | Uses OCR text/layout together with image features in a TILT-style encoder-decoder, with sparse long-context processing and layer-wise multimodal fusion. |
| 5 | CREAM | Hybrid Multimodal | Retriever-Generator Architecture | Explicitly runs an OCR engine at inference to extract text chunks, retrieves/reranks those chunks, and combines them with corresponding document images in the MLLM. |
| 6 | Doc-React | OCR-Free | Agentic Architecture | Performs iterative ReAct-style retrieval and reasoning over multi-page document images and multimodal evidence; OCR appears only in baseline comparisons, not as an explicit inference component. |
| 7 | Doc-V* | OCR-Free | Agentic Architecture | Uses OCR-free page thumbnails, visual fetch actions, and working memory; the key technique is coarse-to-fine agentic evidence aggregation trained with SFT and GRPO. |
| 8 | DocAgent | Hybrid Multimodal | Agentic Architecture | Builds an outline from extracted document content and uses tools for text, image, page, and table evidence; its prompt explicitly warns that document content is obtained using OCR, so OCR is part of inference evidence. |
| 9 | DocDancer | OCR-Free | Agentic Architecture | Uses MinerU2.5 layout analysis/extraction plus search/read tools over text, images, tables, and screenshots; OCR is discussed for baselines, but the proposed inference pipeline does not explicitly state OCR. |
| 10 | DocLens | Hybrid Multimodal | Agentic Architecture | Uses OCR/layout parsing, screenshots, crops, and visual element localization through page navigator, element localizer, answer sampler, and adjudicator agents. |
| 11 | DocR1 | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Fine-tunes a VLM with evidence-page-guided GRPO so supporting pages and answers are predicted from visual document inputs without external OCR. |
| 12 | DocSLM | Hybrid Multimodal | Backbone-Centric Multimodal Adaptation Architecture | Uses a lightweight OCR module to produce word-box tokens, then fuses OCR, layout, and visual crop features through hierarchical compression and streaming inference. |
| 13 | Docopilot | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Inference uses rendered page images and PDF-parser-derived interleaved text-image inputs rather than an explicit OCR module. OCR is mentioned as a capability strengthened by training/high-resolution data and in baselines, not as external inference OCR. |
| 14 | Chain-of-Reading | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Directly consumes PDF page images and trains locate-then-read reasoning; the "OCR" step is model-internal reading over localized content, not an explicit external OCR inference pipeline. |
| 15 | RAG-DocVQA | Hybrid Multimodal | Retriever-Generator Architecture | Includes textual RAG variants that segment OCR-extracted tokens and feed retrieved chunks to generators, alongside visual OCR-free variants; the model family therefore explicitly includes OCR-at-inference multimodal RAG. |
| 16 | GRAM | Hybrid Multimodal | Hierarchical/Global Transformer Architecture | Extends OCR/text-layout document VQA backbones with global document tokens and local-global reasoning to fuse page-level and document-level multimodal context. |
| 17 | Hi-VT5 | Hybrid Multimodal | Hierarchical/Global Transformer Architecture | Encodes per-page OCR/text-layout and visual information, then reasons over page summary tokens for multi-page answer generation. |
| 18 | InstructDr | Hybrid Multimodal | Backbone-Centric Multimodal Adaptation Architecture | Processes document images with an image encoder and explicitly runs an OCR engine to supply OCR tokens and bounding boxes to the Document-former/LLM at inference. |
| 19 | Knowledge Graph Prompting | OCR-Free | Retriever-Generator Architecture | Builds a knowledge graph from already available passages, pages, and tables; no explicit OCR inference step is described, so it is OCR-independent rather than OCR-based. |
| 20 | Leopard | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Adapts a text-rich multi-image VLM through high-resolution visual encoding and instruction tuning; OCR-like reading is handled internally by the model, not by external OCR. |
| 21 | M3DocRAG | OCR-Free | Retriever-Generator Architecture | Uses visual page-image retrieval, such as ColPali-style embeddings, followed by VLM answer generation and explicitly avoids text extraction/OCR as the retrieval basis. |
| 22 | MDocAgent | Hybrid Multimodal | Agentic Architecture | Its document preprocessing explicitly extracts text via OCR/PDF parsing while preserving page images, then coordinates text, image, critical, and summarizer agents. |
| 23 | MHier-RAG | OCR-Free | Retriever-Generator Architecture | Uses Docling PDF parsing, hierarchical text indexes, LVLM-generated visual descriptions, and raw visual evidence; OCR appears in comparison baselines, not as an explicit proposed inference step. |
| 24 | MLDocRAG | Hybrid Multimodal | Retriever-Generator Architecture | Defines document pages and chunks via OCR and MinerU parsing, including OCR-derived table Markdown, then retrieves multimodal chunks for LVLM generation. |
| 25 | mPLUG-DocOwl2 | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Compresses high-resolution page images into compact visual tokens and explicitly targets OCR-free multi-page document understanding. |
| 26 | M2RAG | Hybrid Multimodal | Agentic Architecture | Its parsing tool explicitly combines OCR with PDF parsing to extract text and page images, then BM25, visual retrieval, and modal-fuser agents coordinate answer generation. |
| 27 | MoLoRAG | OCR-Free | Retriever-Generator Architecture | Constructs a multimodal logic-aware retrieval graph over document images/pages; OCR is discussed for text-RAG baselines and advanced OCR ablations, not as the proposed inference pipeline. |
| 28 | Self-Attention Scoring | OCR-Free | Retriever-Generator Architecture | Uses Pix2Struct-style visual page representations and self-attention scoring to select relevant page images before answering, without explicit OCR extraction. |
| 29 | Recurrent Memory Transformer | Hybrid Multimodal | Hierarchical/Global Transformer Architecture | Consumes OCR word/spatial embeddings together with visual page features and carries recurrent memory tokens across pages for document-level reasoning. |
| 30 | MultiDocFusion | Hybrid Multimodal | Retriever-Generator Architecture | Performs OCR over detected document regions and combines OCR text with visual parsing/layout to build structure-aware multimodal chunks for RAG. |
| 31 | PDF-WuKong | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Uses a PDF parser to create interleaved text and image evidence plus an end-to-end sparse sampler; the inference pipeline describes parsing structured PDF data, not explicit OCR. |
| 32 | SimpleDoc | OCR-Free | Agentic Architecture | Combines visual page retrieval, VLM-generated page summaries, memory, and iterative query refinement; it uses extracted page text, but the paper does not explicitly define an OCR inference module. |
| 33 | TextHawk2 | OCR-Free | Backbone-Centric Multimodal Adaptation Architecture | Trains a high-resolution VLM with strong OCR and grounding ability, but text reading is internal to the model rather than supplied by an external OCR inference pipeline. |
| 34 | VDocRAG | OCR-Free | Retriever-Generator Architecture | Performs visual document retrieval and generation over page images. OCR text is used only as a pretraining pseudo-target/signal, so the inference label remains OCR-Free. |
| 35 | ViDoRAG | Hybrid Multimodal | Agentic Architecture | Combines OCR-text chunk retrieval with visual retrieval and uses dynamic iterative seeker, inspector, and answer agents to merge textual and visual evidence. |
| 36 | MACT | OCR-Free | Agentic Architecture | Uses VLM planning/execution agents and LLM judgment/answer agents over visual document inputs, with adaptive test-time scaling and no explicit external OCR module. |

## Overall Pattern

Under the stricter inference-only definition, several systems previously treated as hybrid move to OCR-Free because they use parser text, built-in VLM reading, or OCR-related training objectives rather than explicit external OCR at inference. Hybrid systems remain those that explicitly expose OCR text/tokens/layout to the inference pipeline while also preserving visual evidence.
