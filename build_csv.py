import csv

header = [
    "Name","Bibtex","Venue","Dependencies","Relationship","Approach","OCR","OCR Engine",
    "Prompt Strategies","RAG Strategies","Agentic","Modality","Vision Encoder","LLM/MLLM",
    "Downstream Task","LLM Role","Key Technique","Model Architecture",
    "Pretraining (No Instruct)","Instruct-Tuning","Inference",
    "Pretraining","Instruct / Fine Tuning","Domain"
]

# Each model has 2 rows: the data row, followed by an evidence row that cites supporting quotes.
# Evidence row uses the same 24 columns; cells hold citations/quotes from the paper for that column.

rows = []

# =========================================================================
# 1. RM-T5
# =========================================================================
rows.append([
    "RM-T5", "dong2024multi", "DAS 2024",
    "T5; DiT; ViT",
    "seq2seq backbone; document image encoder; patch encoder",
    "Scaling",
    "Yes",
    "Amazon Textract (dataset-provided)",
    "N/A",
    "N/A",
    "N/A",
    "T, V, L",
    "DiT",
    "T5-base",
    "QA, PageID",
    "T5 encoder-decoder: encoder produces per-page memory tokens from OCR+visual+question; decoder consumes concatenated memories from all pages to generate the final answer",
    "Recurrent Memory Transformer (RMT) within an encoder-decoder framework. m=100 learnable memory tokens propagated across pages; memories from all pages concatenated for decoder to integrate global document context. Auxiliary page prediction module",
    "T5-base encoder-decoder + DiT visual patch encoder + spatial embedding module + page prediction module",
    "N/A (T5-base Hugging Face pretrained weights used for initialization)",
    "MP-DocVQA training set (curriculum learning: 1 page -> 3 pages, then full 20-page doc)",
    "MP-DocVQA test",
    "N/A",
    "Curriculum learning SFT: start with 1 page, grow to 3 pages; then fix encoder and fine-tune decoder + page selector on full 20-page documents. 10 epochs, batch size 4, NVIDIA A40 GPU",
    "Multi-page VRD"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln9 'we use DiT [13] to extract the features'; Ln74 'Inspired by Hi-VT5 and RMT, we introduce m learnable Memory cells'; Ln45 cites RMT [6]",
    "Ln66 'we integrate a T5 model at each timestep'; Ln74 'm learnable Memory cells'; Ln72 'we use DiT'",
    "Ln49 Table 1 classifies as 'Recurrent method: Our proposed (2024)' vs Single-page adapted / Hierarchical / Local-global methods",
    "Ln72 'We summarize the OCR ok and the spatial embedding sk'; Ln100 'OCR annotations extracted by Amazon Textract2'",
    "Ln100 'OCR annotations extracted by Amazon Textract2'",
    "No CoT/ReAct/other prompting strategies referenced",
    "Not a RAG framework - abstract: 'integrate recurrent memory mechanisms'",
    "Not an agent - model is a single end-to-end transformer",
    "Ln72 OCR token + spatial embedding + visual patches from DiT",
    "Ln72 'we use DiT [13] to extract the features of the document image'",
    "Ln66 'we integrate a T5 model at each timestep'; Ln112 'Hugging Face T5 base model pretrained weights were used for initialization'",
    "Abstract 'specialized for multi-page document VQA'; Ln102 'accuracy (ACC) ... ANLS ... page accuracy (Page ACC)'",
    "Decoder is T5 LM head (standard)",
    "Ln66 'memory cells from Recurrent Neural Networks (RNN) to selectively retain or forget information over time'; Ln74 'm learnable Memory cells to store the current summary information'",
    "Ln66 architecture description; Ln72 'we use DiT [13]'; Ln114 'We fixed the number of memory tokens to 100'",
    "Ln112 'Hugging Face T5 base model pretrained weights were used for initialization' (no additional pretraining)",
    "Ln100 'The MP-DocVQA dataset' - sole training data",
    "Ln120 Table 2 evaluation on MP-DocVQA",
    "N/A",
    "Ln94 'curriculum learning method to pretrain the model'; Ln112 'increase it to 3 pages at most. Then the parameters of the encoder are fixed, and the decoder and page selector are fine-tuned with the full 20-page documentation'; Ln112 '10 epochs ... batch size of 4 ... NVIDIA A40 GPU'",
    "Ln100 'MP-DocVQA dataset ... Documents in the dataset have a maximum of 20 pages'"
])

# =========================================================================
# 2. DocLens
# =========================================================================
rows.append([
    "DocLens", "zhu2025doclens", "preprint (2025-11-17)",
    "MinerU; Gemini 2.5 Pro/Flash; Claude-4-Sonnet",
    "document parsing tool; primary VLM backbones; alternative VLM backbone",
    "RAG (agentic)",
    "Yes",
    "MinerU (Niu et al. 2025)",
    "CoT",
    "Joint",
    "Open (sampling-adjudication within iteration-bounded loop)",
    "T, V, L",
    "N/A (frontier VLMs consume page screenshots + cropped visual elements directly)",
    "Gemini-2.5-Pro, Gemini-2.5-Flash, Claude-4-Sonnet",
    "QA",
    "4 specialised agents: Page Navigator (retrieves evidence pages via OCR-augmented VLM), Element Localizer (crops figures/tables via MinerU), Answer Sampler (Ta=8 CoT candidates at T=0.7), Adjudicator (synthesises best answer)",
    "Tool-augmented 'Lens' approach: Page Navigator achieves ~97.3% evidence-page recall by prompting LLM with all page screenshots + OCR text interleaved and sampling Te=8 times; Element Localizer uses MinerU to crop visual elements; Sampling-adjudication mitigates hallucination",
    "Training-free pipeline: Lens Module (Page Navigator + Element Localizer with MinerU) feeds evidence set S={page screenshot + OCR text + cropped elements} to Reasoning Module (Answer Sampler + Adjudicator)",
    "N/A (uses pretrained Gemini-2.5-Pro/Flash and Claude-4-Sonnet)",
    "N/A (training-free)",
    "MMLongBench-Doc, FinRAGBench-V",
    "N/A",
    "N/A (training-free)",
    "VRD, long visual documents, financial reports"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln561 'We employ MinerU ... to perform OCR, layout detection, and cropping'; Ln123 'Gemini-2.5-Pro, Gemini-2.5-Flash, and Claude-4-Sonnet'",
    "Same as above",
    "Ln23 'DocLens, a multi-agent framework that overcomes these challenges by strategically leveraging document-parsing tools'; Ln123 compared as 'VLM-based Agentic Framework'",
    "Ln69 'First, it uses an OCR tool to extract the text'; Ln561 'MinerU ... to perform OCR'",
    "Ln561 'We employ MinerU'",
    "Ln101 'a reasoning process R (e.g., a chain-of-thought trace)'",
    "Ln79 'sample multiple times' with Te iterations joint over page screenshots + OCR text (interleaved input). Not a fixed/dense retriever",
    "Ln23 sampling-adjudication with iterative Page Navigator chunking when context insufficient (Ln575 chunking strategy)",
    "Ln23 'page screenshots, text, and cropped visual elements' - three modalities aggregated",
    "Ln123 frontier VLMs (Gemini/Claude) - no separate vision encoder exposed",
    "Ln123 Gemini-2.5-Pro/Flash, Claude-4-Sonnet",
    "Abstract 'state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V'",
    "Ln23 'Page Navigator ... augment VLMs for page-level retrieval; Element Localizer ... locate visual elements ... Answer Sampler ... Adjudicator agent'",
    "Ln23 'multi-agent framework ... strategically leveraging document-parsing tools'",
    "Ln23 + Ln563 'Ta are set to 8, with a temperature 0.7'",
    "N/A (proprietary frontier VLMs)",
    "N/A (training-free agentic framework)",
    "Ln121 'MMLongBench-Doc ... FinRAGBench-V'",
    "N/A",
    "N/A (training-free)",
    "Ln121 'FinRAGBench-V ... documents with dense, newspaper-like layouts'; MMLongBench-Doc long multi-domain documents"
])

# =========================================================================
# 3. SimpleDoc
# =========================================================================
rows.append([
    "SimpleDoc", "jain2025simpledoc", "EMNLP 2025",
    "ColQwen-2.5 (ColPali); Qwen2.5-VL-32B-Ins; Qwen3-30B-A3B",
    "visual retrieval embedding model; VLM for summarisation/reasoning; text-only LLM for summary re-ranking",
    "RAG (agentic)",
    "No",
    "N/A",
    "CoT (iterative refinement with working memory)",
    "Dense",
    "Open (reasoner decides answer / unanswerable / refine query and iterate; max L iterations)",
    "T, V",
    "ColQwen-2.5 (ColPali-strategy trained)",
    "Qwen2.5-VL-32B-Instruct (VLM reasoner + per-page summariser); Qwen3-30B-A3B (text LLM for summary re-ranking)",
    "QA",
    "Qwen2.5-VL-32B-Ins as VLM reasoner agent and per-page summariser; Qwen3-30B-A3B as text-only LLM for summary-based re-ranking and decision making",
    "Dual-cue page retrieval: offline indexes each page via ColQwen-2.5 dense embedding + VLM-generated semantic summary; online retrieves top-k=30 by embedding, re-ranks via LLM over summaries; reasoner iteratively refines query via working memory if evidence insufficient",
    "Two-stage pipeline: (1) offline pre-processing (ColQwen-2.5 embeddings + Qwen2.5-VL-32B summaries per page) + (2) online iterative QA loop with dual-cue retriever + memory-guided reasoner",
    "N/A (uses pretrained ColQwen-2.5 and Qwen models)",
    "N/A (training-free; no additional fine-tuning)",
    "MMLongBench, LongDocURL, PaperTab, FetaTab",
    "N/A (training-free)",
    "N/A (training-free)",
    "Multi-page VRD, tabular"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln131 'For visual embedding model, we use ColQwen-2.5 for all methods ... we use Qwen2.5-VL-32B-Ins whenever a VLM is needed ... Qwen3-30B-A3B to for page retrieval'",
    "Same as above",
    "Ln7 'lightweight yet powerful retrieval augmented framework ... A single VLM-based reasoner agent repeatedly invokes this dual-cue retriever'",
    "No OCR mentioned; retrieval is purely via dense visual embedding on page images",
    "N/A",
    "Ln93 'Self-reflection has been proven an effective method in LLMs' - iterative reasoning with working memory",
    "Ln7 'first retrieving candidates through embedding similarity and then filtering and re-ranking these candidates based on page summaries'",
    "Ln85-92 reasoner outputs one of: Answer / Not Answerable / Query Update; Ln97 iterative process, max L iterations",
    "Ln7 'pages as images' and 'summaries' (text) - two modalities used at retrieval",
    "Ln131 ColQwen-2.5 as visual embedder",
    "Ln131 'Qwen2.5-VL-32B-Ins whenever a VLM is needed' (also tested Qwen2.5-VL-7B-Instruct); Qwen3-30B-A3B for retrieval",
    "Ln7 'Document Visual Question Answering (DocVQA) is a practical yet challenging task'",
    "Ln7 'A single VLM-based reasoner agent repeatedly invokes this dual-cue retriever'",
    "Ln27 'dual-cue retriever that first shortlists pages via embedding similarity and then asks an LLM, operating solely over the summaries, to decide which of those pages are pertinent'",
    "Ln43 figure caption: 'two stages: (1) offline extraction of visual embeddings and LLM-generated summaries ... (2) online reasoning loop'",
    "N/A",
    "N/A (training-free)",
    "Ln115 'MMLongBench ... LongDocURL ... PaperTab ... FetaTab'",
    "N/A",
    "N/A",
    "Ln117-125 covers long PDFs with text/tables/images including research papers and Wikipedia tables"
])

# =========================================================================
# 4. MACT
# =========================================================================
rows.append([
    "MACT", "yu2025visual", "arXiv preprint (2025)",
    "Qwen2.5-VL / MiMo-VL / InternVL3 (VLM); Qwen2.5 / MiMo / InternVL3 (LLM); GRPO; VisualPRM; Skywork-VL-Reward",
    "VLM backbones (plan+execute agents); LLM backbones (judgment+answer agents); RL algorithm; process reward model; outcome reward model",
    "Scaling (Multi-Agent Collaboration with Test-Time Scaling) - NOTE not retrieval-based",
    "No",
    "N/A",
    "CoT (analogical prompting in Planning Agent)",
    "N/A (not a RAG framework - no external retrieval)",
    "Open (4-agent procedural reasoning with up to Nc=3 judgment-triggered corrections)",
    "T, V",
    "Qwen2.5-VL-7B-Instruct / MiMo-VL-7B-SFT / InternVL3-9B (VLM for plan+execute agents)",
    "Plan/Exec: Qwen2.5-VL-7B or MiMo-VL-7B-SFT or InternVL3-9B; Judge/Answer: Qwen2.5-7B/3B or MiMo-7B-SFT or InternVL3-8B/2B",
    "QA",
    "Planning Agent (VLM) analyses question and produces Np high-level plans via analogical prompting; Execution Agent (VLM) executes step-by-step with Ne candidates per step scored by process reward model; Judgment Agent (LLM) verifies correctness and redirects to prior agents (Nc<=3); Answer Agent (LLM) summarises final answer",
    "Four-agent collaboration with agent-wise hybrid test-time scaling: Planning (parallel Np), Execution (sequential Ne candidates with PRM), Judgment (budget forcing internal scaling), Answer (no scaling). Mixed reward = per-agent rewards + global outcome reward via GRPO",
    "Three MACT variants: Qwen2.5-VL-Series-24B, MiMo-VL-Series-28B, InternVL3-Series-28B (each with VLM for plan/exec + LLM for judge/answer)",
    "N/A (uses pretrained base VLMs/LLMs)",
    "Stage 1 SFT: VLM tuned on document/non-document datasets with/without CoT; Judgment LLM tuned on GPT-4o-generated labels + rule-based verifications; Answer LLM tuned on ground-truth summaries",
    "DocVQA, DUDE, SlideVQA, MMLongBench-Doc, VisualMRC, InfographicVQA, ChartQA, CharXiv, TableVQA-Bench, TableBench, ScienceQA, RealWorldQA, MathVista, Math-Vision, MathVerse",
    "N/A",
    "Two-stage pipeline: (1) SFT per agent on targeted task; (2) RL with GRPO using VisualPRM (process reward for plan/execute) + Skywork-VL-Reward (outcome reward for judgment/answer) with mixed agent-specific + global reward",
    "VRD (text, webpage, chart, table) + general + mathematical reasoning"
])
rows.append([
    "EVIDENCE", "", "",
    "Training Pipeline: 'Qwen2.5-VL Series Based: Qwen2.5-VL-7B-Instruct and Qwen2.5-7B/3B-Instruct; MiMo-VL Series Based: MiMo-VL-7B-SFT and MiMo-7B-SFT; InternVL3 Series Based: InternVL3-9B/8B/2B-Instruct'; 'optimize our model via GRPO'; 'Process reward model VisualPRM ... Skywork-VL-Reward'",
    "Same",
    "Abstract: 'MACT, a Multi-Agent Collaboration framework with Test-Time scaling'. Note: author list compares MACT alongside M3DocRAG/MDocAgent in related work (Multi-Agent Models section) - treated here as multi-agent reasoning + TTS rather than strict RAG.",
    "No OCR tool referenced in methodology. VLMs consume visual inputs directly",
    "No OCR engine mentioned",
    "Methodology Planning Agent: 'Largely following the analogical prompting principles (Yasunaga et al. 2024)'",
    "Paper does not describe a retrieval mechanism or external vector store",
    "Methodology: 'four distinct small-scale agents, i.e., planning, execution, judgment, and answer agents' with 'To prevent infinite correction loops, the maximum number of corrections Nc is set to 3'",
    "Abstract 'tailored for visual document understanding and visual question answering (VQA)' + Table 1 covers text/webpage/chart/table",
    "Training Pipeline: plan+execute agents use VLM (Qwen2.5-VL-7B / MiMo-VL-7B-SFT / InternVL3-9B)",
    "Training Pipeline: plan+execute use VLM; judgment+answer use LLM. Specific sizes listed.",
    "Abstract 'tailored for visual document understanding and visual question answering (VQA)'",
    "Methodology: 'planning agent ... analysis and decomposition of the original question'; 'execution agent ... execute plans step by step'; 'judgment agent ... assess the correctness of steps'; 'answer agent ... summarize prior information and generate the final answer'",
    "Methodology 'agent-wise hybrid test-time scaling' + 'Mixed reward modeling'",
    "Methodology Agent-Wise Hybrid Test-Time Scaling: Planning (Np parallel), Execution (Ne candidates scored by PRM), Judgment (budget forcing), Answer (no scaling); Mixed Reward = agent-specific + global",
    "Training Pipeline: 'we initiate with pretrained models' - no separate pretraining",
    "Training Pipeline: 'In the first SFT stage, we initially train a tuned 11B/7B/7B VLM on the selected document-based or non-document-based datasets ... fine-tune an 8B/7B/7B LLM on judgment labels generated via GPT-4o and rule-based verifications ... fine-tune another 3B/3B/7B LLM on the outputs from preceding agents and ground-truths'",
    "Datasets section lists all 15 benchmarks",
    "Pretrained only",
    "Training Pipeline: 'two-stage SFT and RL pipeline ... optimize our model via GRPO. Process reward model VisualPRM ... for step-by-step reward signals of A_plan and A_exe, while Skywork-VL-Reward is used for ... A_judg and A_ans'",
    "Datasets: four document types (Text / Webpage / Chart / Table) + General + Mathematical"
])

# =========================================================================
# 5. Doc-V*
# =========================================================================
rows.append([
    "Doc-V*", "zheng2026docv", "preprint (15/04/2026 release)",
    "Qwen-2.5-VL; ColQwen; GRPO; GPT-4o (teacher)",
    "VLM backbone; external visual retriever; RL algorithm; teacher for SFT trajectory distillation",
    "RAG (agentic)",
    "No",
    "N/A",
    "CoT (ReAct-style think-action protocol with <analysis>/<plan>/<summary> blocks)",
    "Dense (ColQwen visual retrieval on page images)",
    "Open (agent decides retrieval_page / fetch_page / answer; T=8 max interaction steps)",
    "V",
    "ColQwen (external retriever)",
    "Qwen2.5-VL-7B-Instruct (agent backbone)",
    "QA",
    "Single OCR-free agent initialised from Qwen2.5-VL-7B-Instruct performs ReAct-style think-action reasoning; three actions: retrieval_page (query-based semantic retrieval via ColQwen), fetch_page (direct index-based page fetch), answer (terminate). Structured <think> with <analysis>/<plan>/<summary> blocks; working memory accumulates summaries",
    "Sequential evidence aggregation via active perception: Global Thumbnail Overview (256x256 per page in grid) gives structural prior; agent alternates between structured visual reasoning and document navigation; two-stage training: SFT on 9019 GPT-4o-distilled trajectories + GRPO RL with reward = 0.6*answer + 0.3*evidence + 0.1*format",
    "Qwen-2.5-VL architecture (ViT + MLP projector + LLM backbone); pre-computes and caches visual tokens per page at up to 1024x768 native resolution; agent dynamically requests pages via retrieval/fetch actions; working memory concatenates historical summaries",
    "N/A (uses pretrained Qwen-2.5-VL-7B-Instruct)",
    "Stage 1 SFT: 9,019 GPT-4o-distilled trajectories from MP-DocVQA + DUDE. Stage 2 GRPO: 2,048 filtered non-overlapping examples with easy/medium/hard stratification",
    "In-Domain: MP-DocVQA, DUDE. Out-of-Domain: SlideVQA, LongDocURL, MMLongBench-Doc",
    "N/A",
    "Two-stage training: (1) SFT on 9019 high-quality trajectories (format validity + answer correctness + evidence sanity filtering) distilled from GPT-4o; (2) GRPO with composite reward (answer 0.6 + evidence 0.3 + format 0.1)",
    "VRD, multi-page (academic papers, financial reports, industrial manuals, slides)"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln105 'Our agent is initialized from Qwen-2.5-VL-7B-Instruct ... we employ ColQwen as the external retriever'; Ln93 'supervised fine-tuning ... 9,019 high-quality trajectories'; Ln95 'GRPO reinforcement learning'; Ln41 'trajectories synthesized by GPT-4o'",
    "Same",
    "Abstract 'Doc-V*, an OCR-free agentic framework that casts multi-page DocVQA as sequential evidence aggregation ... Trained by imitation learning ... and further optimized with Group Relative Policy Optimization'",
    "Abstract 'OCR-free agentic framework'",
    "No OCR tool referenced",
    "Ln85 'a ReAct reasoning style with visual feedback ... <think>...</think><action>...</action>' with <analysis>/<plan>/<summary> blocks",
    "Ln77-78 'retrieval action ... environment then calls an external multimodal retriever (e.g., ColQwen) ... fetch action requests specific pages by absolute indices'",
    "Ln61 'closed-loop formulation enables selective evidence acquisition'; Ln81 agent decides when to 'answer' to terminate; Ln105 T=8 max steps",
    "Ln67 'not fed to the agent all at once but are dynamically requested by the agent'; Ln69 pure-vision thumbnails",
    "Ln105 'ColQwen (Faysse et al. 2025) as the external retriever'",
    "Ln105 'Our agent is initialized from Qwen-2.5-VL-7B-Instruct'",
    "Abstract 'OCR-free agentic framework that casts multi-page DocVQA as sequential evidence aggregation'",
    "Ln61 'an OCR-free MLLM-based agent πθ interacts with the document environment for up to T steps'",
    "Abstract 'Trained by imitation learning from expert trajectories and further optimized with Group Relative Policy Optimization'; Ln69 'Global Thumbnail Overview'",
    "Ln65-67 Qwen-2.5-VL architecture (ViT + MLP projection + LLM); Ln67 '1024x768' cap; Ln69 thumbnails 256x256",
    "Ln67 'pretrained' Qwen-2.5-VL assumed",
    "Ln93 '9,019 high-quality trajectories constructed from MP-DocVQA and DUDE'; Ln95 '2,048 non-overlapping training examples' for GRPO",
    "Ln101 'In-Domain evaluation ... MP-DocVQA and DUDE ... Out-of-Domain ... SlideVQA, LongDocURL ... MMLongBench-Doc'",
    "N/A",
    "Ln93-95 'two-stage optimization strategy ... supervised fine-tuning ... distilled closed-loop interaction trajectories ... we apply GRPO reinforcement learning ... weighted reward that combines answer correctness, evidence retrieval quality, and format validity'; Ln105 'ω_ans = 0.6, ω_evi = 0.3, ω_struct = 0.1'",
    "Abstract 'academic papers, financial reports, and industrial manuals'"
])

# =========================================================================
# 6. ORCA
# =========================================================================
rows.append([
    "ORCA", "lassoued2026orca", "CVPR? (2026)",
    "Qwen3VL-8B-Instruct; Qwen2.5-VL-7B; GLM-4.5V-9B; InternVL3-8B; Qwen3-1.7B",
    "specialist/debate/thesis/sanity backbones; fine-tuned router backbone; thinker backbone; antithesis backbone; evaluator/judge LLMs",
    "RAG (agentic) - NOTE: single-page DocVQA in this work; authors list multi-page as future work",
    "Yes (OCR is one of nine specialist agents)",
    "N/A (no specific external OCR tool; OCR is an internal agent)",
    "CoT (reasoning-guided agent selection)",
    "N/A (no external retrieval; framework operates on single-page document)",
    "Open (5-stage pipeline with conditional debate activation ~8.3% + up to Nc corrections; thesis-antithesis debate)",
    "T, V",
    "N/A (VLM backbones used directly; no separate vision encoder)",
    "Specialists/Debate/Thesis/Sanity: Qwen3VL-8B-Instruct; Thinker: GLM-4.5V-9B; Antithesis: InternVL3-8B-hf; Router: Qwen2.5-VL-7B (fine-tuned); Evaluator/Judge: Qwen3-1.7B",
    "QA",
    "Thinker (GLM-4.5V-9B) generates reasoning path and initial answer; Router (Qwen2.5-VL-7B, fine-tuned) selects specialist agents via Turbo DFS constrained generation; 9 Qwen3VL-8B specialists (OCR, Layout, Table/List, Figure/Diagram, Form, Free Text, Image/Photo, Yes/No, General); Orchestrator sequences agents; Debate agent stress-tests when thinker and expert disagree; Thesis (Qwen3VL-8B) + Antithesis (InternVL3-8B) debate under Judge (Qwen3-1.7B); Sanity Checker refines final answer",
    "5-stage pipeline: (1) Context Understanding (thinker + reasoning path); (2) Collaborative Agent Execution (router + orchestrator + 9 specialists); (3) Stress Testing (debate when aE != aT); (4) Multi-turn Conversation (thesis-antithesis under judge); (5) Answer Refinement (sanity checker). Reasoning-path masking prevents confirmation bias. Conditional activation = ~8.3% of instances",
    "Multi-agent collaborative framework. Router trained on Single-Page DocVQA using Qwen2.5-VL-7B + Multilabel Stratified K-Fold (n=8) + Turbo DFS decoding + vocabulary shrinking + Unsloth + Flash Attention 2",
    "N/A (uses pretrained VLMs; only router is fine-tuned)",
    "Router SFT on Single-Page DocVQA with ground-truth agent annotations; specialist agents 'fine-tuned for their respective tasks' (supplementary)",
    "Single-Page DocVQA, InfographicsVQA, OCRBench-v2 (en). NOTE: Single-page benchmarks - multi-page extension is future work",
    "N/A",
    "Router SFT via Multilabel Stratified K-Fold (n=8) on Qwen2.5-VL-7B with vocabulary shrinking + Flash Attention 2 + Unsloth; Turbo DFS decoding. Specialist agents fine-tuned per task",
    "Single-page VRD (forms, tables, figures, handwriting) - future work: multi-page"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln101 'All specialized agents are based on variants of Qwen3VL-8B, fine-tuned'; Ln191 'thinker agent uses GLM-4.5V-9B ... specialized agents ... Qwen3VL-8B-Instruct ... antithesis agent is based on InternVL3-8B-hf ... evaluation and judge agents use Qwen3-1.7B'; Ln105 'We employ Qwen2.5-VL-7B as the base architecture for A_route'",
    "Same",
    "Abstract 'novel multi-agent framework ... iterative refinement'; Ln55 'ORCA employs conditional activation, engaging debate in only 8.3% of instances'",
    "Ln99 'A_ocr: Recognizes handwritten and difficult text' is one of nine specialist agent types",
    "Ln99 lists OCR as one of 9 specialist agents but no specific OCR engine named",
    "Ln71 'The reasoning path R = {r1, r2, ..., rn}'",
    "Not a retrieval-based framework (single-page DocVQA); Ln313 multi-page extension listed as future work",
    "Multi-agent 5-stage pipeline; Ln55 conditional debate at ~8.3% of cases",
    "Ln45 'text, tables, figures, handwritten content'",
    "N/A - VLM backbones are used directly",
    "Ln191 'thinker agent uses GLM-4.5V-9B ... Qwen3VL-8B-Instruct ... InternVL3-8B-hf ... Qwen3-1.7B'",
    "Ln67 'Document Visual Question Answering' (single-page)",
    "Ln33 describes 5 stages and the role of each agent",
    "Ln49 'ORCA introduces a VLM trained specifically for document-type routing via constrained generation with Turbo DFS decoding'",
    "Ln49 + Ln494-534 implementation/training details in Appendix",
    "Ln101 specialists 'fine-tuned for their respective tasks' - no mention of additional pretraining",
    "Ln530 'Multilabel Stratified K-Fold cross-validation with n splits = 8'; Ln534 'Qwen2.5-VL-7B as the base architecture for A_route, fine-tuned on our augmented dataset'",
    "Ln494 'three challenging document understanding benchmarks: (1) Single-Page DocVQA; (2) InfographicsVQA; (3) OCRBench-v2 (en)'",
    "N/A",
    "Ln105 'We train the router on the Single-Page Document VQA dataset'; Ln101 specialists fine-tuned per task; Ln313 future work 'extend ORCA to multi-page document understanding'",
    "Ln19 'single-page document images'; Ln313 future multi-page"
])

# =========================================================================
# 7. MLDocRAG
# =========================================================================
rows.append([
    "MLDocRAG", "zhang2026mldocrag", "preprint",
    "MinerU; BGE-m3; Qwen2.5-VL; FAISS/ElasticSearch; Neo4j",
    "document parsing tool; dense text encoder; VLM for query generation and answer; vector database; graph database",
    "RAG (fixed)",
    "Yes",
    "MinerU (for PDF parsing including OCR for tables/equations)",
    "CoT",
    "Dense (query-centric KNN search on MCQG)",
    "Fixed (no iterative/agentic refinement; single-pass retrieval + answer)",
    "T, V, L",
    "N/A (LVLM Qwen2.5-VL used directly on multimodal chunks)",
    "Qwen2.5-VL-32B-Instruct (default LVLM for MDoc2Query and final answer); Qwen2.5-VL-7B (MDoc2Query optimisation); Qwen2.5-72B (LLM-as-a-Judge evaluator only)",
    "QA",
    "Qwen2.5-VL-32B generates query-answer pairs per chunk during MCQG construction and produces final answers during inference; optional parametric fine-tuning of LVLM on curated chunk-to-query exemplars",
    "Multimodal Chunk-Query Graph (MCQG): MDoc2Query extends Doc2Query to multimodal via LVLM-generated queries from heterogeneous chunks; Chunk-Query edges + Query-Query KNN edges; query-centric KNN retrieval aggregates linked source chunks",
    "MinerU PDF parser extracts paragraphs/figures/tables/equations into JSON -> LVLM generates answerable queries per chunk -> BGE-m3 embeds query-answer pairs (stored in FAISS/ElasticSearch) -> MCQG stored in Neo4j -> KNN retrieval -> chunk ranking -> LVLM answer generation",
    "N/A (uses pretrained BGE-m3 and Qwen2.5-VL)",
    "N/A by default (training-free); optional parametric optimisation via fine-tuning LVLM on chunk-to-query exemplars",
    "MMLongBench-Doc, LongDocURL",
    "N/A",
    "N/A by default; optional SFT on curated chunk-to-query exemplars for parametric optimisation",
    "VRD (multimodal long documents: research papers, reports, books)"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln85 'We adopt an existing multimodal PDF parsing tool, such as MinerU'; Ln207 'we employ the BGE-m3 encoder ... ElasticSearch ... For query generation and final answer generation, we use Qwen2.5-VL-32B by default, while Qwen2.5-VL-7B is adopted for MDoc2Query optimization. For evaluation, we adopt an LLM-as-a-Judge setup using Qwen2.5-72B'; Ln117 'FAISS ... Neo4j'",
    "Same",
    "Ln39 'MLDocRAG (Multimodal Long-Context Document Retrieval-Augmented Generation)'",
    "Ln85 'MinerU' + Ln89 'table images accompanied by textual captions and OCR-converted Markdown text'",
    "Ln85 'MinerU'",
    "Ln91 'answerable query-answer pairs' + Ln155 LVLM reasoning over retrieved context",
    "Ln79 'user query is embedded and matched against nodes in the MCQG using KNN-based retrieval'",
    "Ln79 'KNN-based retrieval' + Ln155 single-pass final answer generation - no agent loop",
    "Abstract 'multimodal chunks spanning text, images, and tables'",
    "N/A - LVLM used directly on multimodal chunks",
    "Ln207 'Qwen2.5-VL-32B by default, while Qwen2.5-VL-7B is adopted for MDoc2Query optimization ... Qwen2.5-72B' (evaluator only)",
    "Abstract 'multimodal long-context document question answering'",
    "Ln155 'selected multimodal context Crel is provided to a Large Vision-Language Model (LVLM), together with the original user query'; Ln91 LVLM generates Q-A pairs",
    "Abstract 'leverages a Multimodal Chunk-Query Graph (MCQG)'",
    "Ln85-117 four steps: Document Parsing (MinerU); MDoc2Query (LVLM-generated Q-A pairs); Graph Assembly (Chunk-Query + Query-Query KNN edges); Storage (FAISS + Neo4j)",
    "N/A",
    "Ln167 'we further explore parametric optimization by fine-tuning a pretrained Large Vision–Language Model (LVLM) on a curated set of high-quality multimodal chunk-to-query exemplars'",
    "Ln181 'MMLongBench-Doc'; Ln187 'LongDocURL'",
    "N/A",
    "Ln165-167 'Non-Parametric Optimization' via Page-Context-Aware Generation (default); 'Parametric Optimization' as optional fine-tuning",
    "Ln15 'research papers, reports, and books'"
])

# =========================================================================
# 8. MultiDocFusion
# =========================================================================
rows.append([
    "MultiDocFusion", "shin2025multidocfusion", "preprint (14/04/2026 in CSV)",
    "DETR / VGT (DP); Tesseract / EasyOCR / TrOCR; BGE / E5 / BM25; Mistral-8B / Llama-3.2-3B / Qwen-2.5-3B (DSHP-LLM); LoRA",
    "layout detection models; OCR engines; text embedders and sparse retriever; instruction-tuned chunk-hierarchy LLMs; efficient fine-tuning",
    "RAG (fixed)",
    "Yes",
    "Tesseract, EasyOCR, TrOCR",
    "N/A",
    "Dense (BGE / E5) or Sparse (BM25)",
    "Fixed (no agent loop; static pipeline chunking + retrieval + LLM generation)",
    "T, V, L (text + layout detected from images; no visual encoder in retrieval/QA model)",
    "N/A (retrieval operates on detected text + layout; DP via DETR/VGT produces bounding boxes but output is text-centric chunks)",
    "DSHP-LLM backbone: Mistral-8B (selected), Llama-3.2-3B, Qwen-2.5-3B, Qwen-2.5-7B. Final answer generator: 'Llama-based models' (results reported for Llama-3.2-3B, Mistral-8B, Qwen-2.5-7B)",
    "QA",
    "DSHP-LLM (LoRA-tuned Mistral-8B) reconstructs document section hierarchy as JSON Header Tree then attaches general nodes; final answer LLM (Llama-based) generates answer from retrieved hierarchical chunks",
    "Four-stage pipeline: (1) Document Parsing (DETR/VGT detects Titles, Section Headers, text blocks, tables, figures with bounding boxes); (2) OCR (Tesseract/EasyOCR/TrOCR extracts text); (3) DSHP-LLM (LoRA-tuned Mistral-8B parses Header Tree and attaches general nodes via All Segments list to build Document Hierarchical Tree); (4) DFS-based Grouping with max_len threshold assembles Hierarchical Chunks marked with Markdown headers",
    "DP (vision detection) + OCR (classical engines) + DSHP-LLM (LoRA-tuned LLM for hierarchy) + DFS Grouping; top-k=4 retrieval via BGE/E5/BM25 + LLM answer generation",
    "N/A (uses pretrained LLM/embedder/OCR/DP components)",
    "DSHP-LLM instruction-tuned via LoRA on DocHieNet + HRDH for hierarchical parsing; downstream answer LLMs evaluated: Llama-3.2-3B, Mistral-8B, Qwen-2.5-7B",
    "DUDE, MPVQA (MP-DocVQA), CUAD, MOAMOB",
    "N/A",
    "DSHP-LLM LoRA-instruction-tuned on DocHieNet + HRDH (hierarchical parsing training data); downstream generator LLMs used without explicit further fine-tuning for each VQA dataset",
    "Multi-page industrial / academic VRD (financial reports, scientific reports, scanned forms, legal contracts (CUAD), nuclear/specialised domain (MOAMOB))"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln109 'DP is performed with object detection models such as DETR and VGT, while OCR text extraction uses Tesseract, EasyOCR, and TrOCR. The DSHP-LLM ... is trained via instruction tuning on LLMs such as Llama-3.2-3B, Qwen-2.5-3B, and Mistral-8B ... chunk embeddings are generated using BGE, E5, and BM25'; Ln77 'we employ LoRA-based parameter-efficient fine-tuning (PEFT)'",
    "Same",
    "Abstract 'MultiDocFusion, a multimodal chunking pipeline that integrates ... document section hierarchical parsing (DSHP-LLM) ... DFS-based Grouping' - RAG-style pipeline without agent loop",
    "Ln109 'OCR text extraction uses Tesseract, EasyOCR, and TrOCR'",
    "Ln109 'Tesseract, EasyOCR, and TrOCR'",
    "No CoT/ReAct prompt strategy mentioned; focus is on chunking pipeline",
    "Ln109 'chunk embeddings are generated using BGE, E5, and BM25'",
    "Abstract 'RAG-based QA'; Ln107 'we index all test documents jointly and retrieve top-k chunks ... k = 4' (fixed top-k, no iterative loop)",
    "Abstract 'integrates: (i) detection of document regions using vision-based document parsing, (ii) text extraction from these regions via OCR'",
    "DP via DETR/VGT produces layout bounding boxes; retrieval/QA operate on detected text + structure (not raw images)",
    "Ln109 + Ln138 'we selected the fine-tuned Mistral-8B model as the backbone of our DSHP-LLM' + Ln150 Table 3 reports Llama-3.2-3B, Mistral-8B, Qwen-2.5-7B as final answer generators",
    "Abstract 'RAG-based QA' / Ln15 'RAG-based QA has emerged as a powerful method'",
    "Ln77 'DSHP-LLM is built upon an LLM backbone and is instruction-tuned ... LoRA'; Ln109 'Top-k retrieved chunks are then fed into LLMs (e.g., Llama-based models) for final answer generation'",
    "Abstract pipeline description: DP + OCR + DSHP-LLM + DFS Grouping",
    "Ln53 'four stages: (a) DP (Document Parsing), (b) OCR (Optical Character Recognition), (c) DSHP-LLM ... (d) DFS-based Grouping'",
    "N/A",
    "Ln107 'For DSHP-LLM training and testing, we combine documents from DocHieNet and HRDH'; Ln77 'LoRA'",
    "Ln107 'four datasets: DUDE, MPVQA (Tito et al. 2023), CUAD, and MOAMOB'",
    "N/A",
    "Ln77 'instruction-tuned on public datasets of document hierarchies ... LoRA-based parameter-efficient fine-tuning'",
    "Ln33 'financial statements, scientific reports, scanned forms'; Ln146 MOAMOB 'specialized nuclear domain'"
])

# =========================================================================
# 9. RAG-DocVQA
# =========================================================================
rows.append([
    "RAG-DocVQA", "lopez2025enhancing", "ICDAR 2025",
    "VT5 / Pix2Struct / Qwen2.5-VL-7B-Instruct; bge-en-small-v1.5; bge-reranker-v2-m3; DIT (dit-base-finetuned-rvlcdip); LoRA",
    "generator backbones; bi-encoder; cross-encoder reranker; document image transformer for VT5 visual crops; efficient fine-tuning",
    "RAG (fixed)",
    "Yes (textual variants) / No (visual variant)",
    "Not specified (uses dataset-provided OCR tokens; MP-DocVQA ships Amazon Textract)",
    "N/A",
    "Dense (bi-encoder + cross-encoder reranker for textual variants; multi-vector late-interaction via Pix2Struct encoder for visual variant)",
    "Fixed (no agent loop; single-pass RAG pipeline)",
    "T, V (textual variants use T, V, L; visual variant uses V only)",
    "DIT (dit-base-finetuned-rvlcdip) for VT5 visual crops; Pix2Struct vision transformer for visual RAG",
    "VT5 (T5-based, ~223M), Qwen2.5-VL-7B-Instruct (~7B), Pix2Struct (~282M)",
    "QA",
    "VT5 decoder generates answers from fused text + bbox + DIT-encoded image crops; Qwen2.5-VL-7B-Instruct generates answers from concatenated chunk text + question + prompt with each chunk image embedded via Qwen's vision encoder; Pix2Struct encodes tiled mini-patches with 2D positional indices and generates the answer",
    "Three-stage RAG pipeline (indexing, retrieval, generation) with two variants: (1) Textual RAG - chunks OCR tokens (L=60, overlap 10, tolerance 0.2), fine-tuned bge-en-small-v1.5 bi-encoder retrieves top-k'=20, bge-reranker-v2-m3 cross-encoder reranks to top-k=10, concatenates chunk text+boxes+image crop into generator (VT5 or Qwen2.5-VL-7B); (2) Visual RAG - vertical patches (P=512, overlap 256), Pix2Struct encodes patches + query (rendered as image), ColBERT-style late interaction selects top-k=5 patches, tiled as mini-patches for Pix2Struct generator",
    "Bi-encoder (bge-en-small-v1.5, ~33M) + Cross-encoder (bge-reranker-v2-m3, ~568M) + Generator (VT5 ~223M / Qwen2.5-VL-7B ~7000M / Pix2Struct ~282M); DIT for VT5 image crops",
    "N/A (uses pretrained backbones)",
    "Bi-encoder contrastive fine-tuned per dataset on query-chunk pairs (Multiple-Negatives Ranking Loss); VT5 fully fine-tuned (AdamW, lr=2e-4, 4 epochs); Qwen2.5-VL-7B LoRA (alpha=16, rank=8, dropout=0.05) on Q/V projections; Pix2Struct uses pre-existing dataset-fine-tuned weights",
    "MP-DocVQA, DUDE, InfographicVQA",
    "N/A",
    "Bi-encoder: MNR Loss contrastive with synthetic anchor-positive pairs (ANLS > 0.8 threshold); VT5: full fine-tuning; Qwen2.5-VL-7B: LoRA on Q/V; Pix2Struct: dataset-pretrained weights used directly",
    "Multi-page VRD"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln27 'textual RAG variants for two representative Vision Language Models: VT5 and Qwen2.5-VL-7B-Instruct, along with a visual counterpart for Pix2Struct'; Ln181 'For the bi-encoder, we used bge-en-small-v1.5 ... for the cross-encoder we used bge-reranker-v2-m3'; Ln87 'Document Image Transformer (DIT), specifically dit-base-finetuned-rvlcdip'; Ln195 LoRA details",
    "Same",
    "Abstract 'systematically evaluate the impact of incorporating RAG into Document VQA through different retrieval variants'",
    "Ln7 'text-based retrieval using OCR tokens and purely visual retrieval without OCR'",
    "Ln61 'OCR-extracted token sequences' (source not specified in-paper)",
    "No explicit CoT/ReAct prompt strategy",
    "Ln57 'three-stage pipeline: indexing, retrieval, and generation' with textual + visual variants; Ln63 bi-encoder + cross-encoder reranker for textual variants; Ln95 ColBERT-style late interaction for visual variant",
    "Fixed k'=20 then k=10 (or k=5 for visual) - no iterative agent loop",
    "Ln7 'text-based retrieval using OCR tokens and purely visual retrieval without OCR'",
    "Ln87 'Document Image Transformer (DIT), specifically dit-base-finetuned-rvlcdip'; Ln93 'Pix2Struct vision transformer encoder'",
    "Ln27 'VT5 and Qwen2.5-VL-7B-Instruct, along with ... Pix2Struct'",
    "Ln7 'Document Visual Question Answering (Document VQA)'",
    "Ln87 'E_O(Oi) is the semantic embedding of OCR token Oi, produced by a T5 language backbone'; Ln89 'Qwen's decoder, autoregressively generating the final answer'",
    "Abstract 'careful evidence selection consistently boosts accuracy across multiple model sizes'",
    "Ln57 pipeline + Ln61-63 bi-encoder+cross-encoder + VT5/Qwen/Pix2Struct generator",
    "N/A",
    "Ln193-195 'For VT5, we perform full fine-tuning of all layers ... For Qwen2.5-VL-7B-Instruct, we employ parameter-efficient Low-Rank Adaptation (LoRA) ... α = 16, rank = 8 and dropout = 0.05 ... For Pix2Struct, we use versions already fine-tuned on each target dataset'",
    "Ln7 'multi-page datasets MP-DocVQA, DUDE, and InfographicVQA'",
    "N/A",
    "Ln187 'The bi-encoder embedding model is fine-tuned using contrastive learning ... Multiple Negatives Ranking Loss'; Ln193-195 generator fine-tuning per-model",
    "Ln15 'manuals, scientific papers, and technical reports'"
])

# =========================================================================
# 10. M2RAG
# =========================================================================
rows.append([
    "M2RAG", "duan2025m2rag", "ICONIP 2025 (CCIS 2753)",
    "BM25; VisRAG-Ret (MiniCPM-V); Qwen2.5-7B-Instruct; Qwen2.5-VL-7B-Instruct; Qwen2.5-VL-32B (teacher); LoRA",
    "sparse text retriever; visual retriever; text filter LLM; fine-tuned visual extractor and modal fuser VLM; teacher VLM for knowledge distillation; efficient fine-tuning",
    "RAG (agentic)",
    "Yes",
    "OCR + PDF parsing tools (unspecified engines)",
    "CoT (reasoning + evidence chain)",
    "Joint (dual-tower: BM25 text + VisRAG-Ret visual)",
    "Fixed (3-agent pipeline without iterative loop)",
    "T, V",
    "VisRAG-Ret (MiniCPM-V based)",
    "Qwen2.5-7B-Instruct (Text Filter agent); Qwen2.5-VL-7B-Instruct (LoRA-tuned; Visual Extractor + Modal Fuser); Qwen2.5-VL-32B (teacher for distillation only)",
    "QA",
    "Text Filter (Qwen2.5-7B-Instruct) generates preliminary text-based answer a_T + evidence chain E_T from retrieved text segments; Visual Extractor (LoRA-tuned Qwen2.5-VL-7B-Instruct) generates preliminary visual answer a_V + evidence E_V from retrieved pages; Modal Fuser (same LoRA-tuned VLM) integrates text/visual answers and evidence into final answer a_F",
    "Dual-tower retrieval (BM25 text + VisRAG-Ret visual) + 3-agent multi-agent generation (Text Filter + Visual Extractor + Modal Fuser); Knowledge-distillation LoRA fine-tuning of Qwen2.5-VL-7B student using Qwen2.5-VL-32B teacher on MP-DocVQA",
    "BM25 (sparse text) + VisRAG-Ret (visual, MiniCPM-V based) retrieval; top-k=1/3/5 retrieval; Text Filter (Qwen2.5-7B) + LoRA-tuned Visual Extractor + Modal Fuser (Qwen2.5-VL-7B)",
    "N/A (uses pretrained backbones)",
    "Knowledge-distillation LoRA fine-tuning of Qwen2.5-VL-7B student on MP-DocVQA with Qwen2.5-VL-32B as teacher; 3 epochs, batch size 16, NVIDIA A800 80GB GPU",
    "DocBench, MMLongBench, LongDocURL",
    "N/A",
    "LoRA fine-tuning via knowledge distillation on MP-DocVQA (student=Qwen2.5-VL-7B-Instruct, teacher=Qwen2.5-VL-32B-Instruct)",
    "VRD / DocQA (open-domain and closed-domain, long and short documents, textual and visual content)"
])
rows.append([
    "EVIDENCE", "", "",
    "Ln121 'a dual-tower retrieval architecture, utilizing BM25 for text retrieval and VisRAG-Ret for image-based retrieval ... the Text Filter is based on Qwen2.5-7B-Instruct, while the Visual Extractor and Modal Fuser are built upon a fine-tuned version of Qwen2.5-VL-7B-Instruct ... Qwen2.5-VL-7B-Instruct as the student model and Qwen2.5-VL-32B-Instruct as the teacher model'",
    "Same",
    "Abstract 'dual-tower retrieval and multi-agent generation mechanism ... multimodal RAG framework'",
    "Ln65 'utilize a parsing tool that combines Optical Character Recognition (OCR) technology with PDF parsing techniques'",
    "Ln65 'OCR technology' (specific engine not named)",
    "Ln97 'fusing and logically extracting text from different sources thereby improving the interpretability and traceability of the question-answering process' + evidence chain",
    "Ln75 'dual-tower hybrid retrieval ... BM25 ... VisRAG-Ret'",
    "Abstract multi-agent generation; Ln75 top-k fixed retrieval without iterative refinement loop",
    "Abstract 'dual-tower retrieval' (text + visual) + multi-agent generation",
    "Ln121 'VisRAG-Ret' - VisRAG-Ret uses MiniCPM-V as its visual encoder",
    "Ln121 'Text Filter is based on Qwen2.5-7B-Instruct, while the Visual Extractor and Modal Fuser are built upon a fine-tuned version of Qwen2.5-VL-7B-Instruct'",
    "Abstract 'document question answering'",
    "Ln97 Text Filter; Ln101 Visual Extractor; Ln105 Modal Fuser roles",
    "Abstract 'multimodal RAG framework ... dual-tower retrieval with multi-agent generation mechanism'",
    "Ln55 dual-tower retrieval + multi-agent generation pipeline (Fig 2)",
    "N/A",
    "Ln83 'we construct a high quality distillation-based training dataset using the publicly available MP-DocVQA dataset. Given an input x, we prompt a teacher model MT with task instruction I and visual context c to generate an output y with reasoning traces ... perform supervised fine tuning on a student model MS'; Ln121 '3 epochs on a single NVIDIA A800 80GB GPU with a batch size of 16'",
    "Ln123 'DocBench, MMLongBench, and LongDocURL'",
    "N/A",
    "Ln83 distillation-based fine-tuning on MP-DocVQA",
    "Ln123 'open-domain and closed-domain settings, long and short documents, as well as both textual and visual content'"
])

# -------------------------------------------------------------------------
# Write CSV - csv module will correctly quote any field containing commas / quotes
# -------------------------------------------------------------------------

with open('/home/claude/new_models_with_evidence.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for r in rows:
        writer.writerow(r)

# Validate
with open('/home/claude/new_models_with_evidence.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    read_rows = list(reader)

print(f"Total rows: {len(read_rows)}")
print(f"Header cols: {len(read_rows[0])}")
all_ok = True
for i, row in enumerate(read_rows):
    if len(row) != 24:
        print(f"  ROW {i} HAS {len(row)} COLS (should be 24)")
        all_ok = False
print("All rows have 24 columns!" if all_ok else "COLUMN MISMATCH DETECTED")