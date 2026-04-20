# **MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation** 

Yongyue Zhang Independent Researcher Singapore yongyue002@gmail.com 

## **Abstract** 

Understanding multimodal long-context documents that comprise multimodal chunks such as paragraphs, figures, and tables is challenging due to (1) cross-modal heterogeneity to localize relevant information across modalities, (2) cross-page reasoning to aggregate dispersed evidence across pages. To address these challenges, we are motivated to adopt a query-centric formulation that projects cross-modal and cross-page information into a unified query representation space, with queries acting as abstract semantic surrogates for heterogeneous multimodal content. In this paper, we propose a Multimodal Long-Context Document Retrieval Augmented Generation (MLDocRAG) framework that leverages a Multimodal ChunkQuery Graph (MCQG) to organize multimodal document content around semantically rich, answerable queries. MCQG is constructed via a multimodal document expansion process that generates finegrained queries from heterogeneous document chunks and links them to their corresponding content across modalities and pages. This graph-based structure enables selective, query-centric retrieval and structured evidence aggregation, thereby enhancing grounding and coherence in multimodal long-context question answering. Experiments on datasets MMLongBench-Doc and LongDocURL demonstrate that MLDocRAG consistently improves retrieval quality and answer accuracy, demonstrating its effectiveness for multimodal long-context understanding. 

## **ACM Reference Format:** 

Yongyue Zhang and Yaxiong Wu. 2018. MLDocRAG: Multimodal LongContext Document Retrieval Augmented Generation. In _Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX)._ ACM, New York, NY, USA, 15 pages. https://doi. org/XXXXXXX.XXXXXXX 

## **1 Introduction** 

Multimodal long-context documents, such as research papers, reports, and books, often span tens to hundreds of pages and contain diverse multimodal components/chunks including text, images, and tables [7, 21, 28, 33, 38]. Understanding such lengthy multimodal documents presents two central challenges [7]: (1) _cross-modal heterogeneity_ , which requires identifying and localizing relevant 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _Conference acronym ’XX, Woodstock, NY_ 

© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX 

Yaxiong Wu Independent Researcher Singapore wuyashon@gmail.com 

information across heterogeneous modalities; and (2) _cross-page reasoning_ , which demands integrating evidence scattered across multiple pages to support coherent inference. Addressing these challenges necessitates the ability of _multimodal long-context association_ —accurately identifying, connecting, and integrating semantically relevant information across modalities and segments. 

Large Vision-Language Models (LVLMs) have shown strong cross-modal understanding capabilities of effectively aligning and interpreting multimodal information within localized short contexts [9, 19, 46]. Representative examples include GPT-4o [20], Gemini [35], Qwen2.5-VL [3], and InternVL [6], which demonstrate impressive performance on short-range multimodal reasoning benchmarks [23, 43]. However, they often struggle to maintain consistent semantic modeling within limited context windows when applied to long-document scenarios [24]. In particular, their performance deteriorates when evidence is sparsely distributed across pages and modalities by overlooking the relevant information, leading to the so-called “needle in a haystack” problem [41, 42]. 

Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for overcoming the limited context windows of large models, such as Large Language Models (LLMs) and Large VisionLanguage Models (LVLMs), by incorporating external retrieval [1, 26, 50]. In multimodal document scenarios, existing RAG approaches typically follow five strategies (as shown in Figure 1): (a) _𝑅𝐴𝐺𝐷𝑒𝑛𝑠𝑒[𝑡𝑥𝑡]_[:] converting multimodal content into plain text via Optical Character Recognition (OCR) [32] and applying textual chunk-based RAG with LLMs [48] and dense retrievers (e.g., BGE-m3 [5]); (b) _𝑅𝐴𝐺[𝑡𝑥𝑡]_[+] _[𝑖]_[2] _[𝑡] 𝐷𝑒𝑛𝑠𝑒_[: generating image descriptions (e.g., Image2Text (i2t) [][18][])] and fusing them with OCR text using document content extraction toolkits (e.g., MinerU [40]) for textual chunk-based RAG; (c) _𝑀𝑅𝐴𝐺𝐶𝐿𝐼𝑃[𝑡𝑥𝑡]_[+] _[𝑖𝑚𝑔]_ : encoding text and images into a shared embedding space via multimodal encoders (e.g., CLIP [29], SigLIP [37, 47]) for retrieval followed by LVLM-based generation; (d) _𝑉𝑅𝐴𝐺𝐶𝑜𝑙𝑃𝑎𝑙𝑖[𝑝𝑎𝑔𝑒]_[:] rendering document pages as images and retrieving relevant pagelevel content using vision-based methods (e.g., ColPali [14]) prior to LVLM decoding; and (e) _𝐺𝑟𝑎𝑝ℎ𝑅𝐴𝐺𝐷𝑒𝑛𝑠𝑒[𝑡𝑥𝑡]_[+] _[𝑖]_[2] _[𝑡]_[: constructing knowledge] graphs (KGs) from document content and applying Graph-based RAG [11, 13]. However, these approaches often struggle to capture fine-grained cross-modal and cross-page associations, leading to incomplete grounding and suboptimal retrieval. 

Document expansion methods such as Doc2Query [27] provide a principled way to map document content into a unified query representation space, enhancing retrieval by generating synthetic queries that capture a document’s latent information needs. Building upon this idea, QCG-RAG [44] constructs a query-centric graph that explicitly links generated queries to their corresponding textual document chunks, enabling query-aware indexing and multi-hop retrieval over long textual contexts, thereby improving evidence 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

**Figure 1: Illustration of RAG for multimodal long-context documents, comparing (a)–(e) baselines with (f) our MLDocRAG.** 

aggregation and grounding in long-document question answering. However, these approaches remain largely unexplored in the setting of multimodal long-context documents, where information is distributed across heterogeneous modalities and pages. 

Building on this idea, we extend query-centric formulation to the multimodal setting and propose MLDocRAG (Multimodal LongContext Document Retrieval-Augmented Generation), a framework for multimodal long-context document understanding. MLDocRAG leverages a unified retrieval structure—the Multimodal ChunkQuery Graph (MCQG)—constructed via MDoc2Query, which extends Doc2Query [27] to multimodal scenarios and generates semantically rich, answerable queries from heterogeneous chunks spanning text, images, and tables. The resulting MCQG links each query to its corresponding multimodal content, enabling selective retrieval and structured evidence aggregation across modalities and pages. This design effectively addresses cross-modal and crosspage associations, improving retrieval accuracy and grounding in multimodal long-document question answering (QA). Figure 1(f) illustrates the overall pipeline of our proposed MLDocRAG framework. Our main contributions are as follows: 

• We propose **MLDocRAG (Multimodal Long- Context Document Retrieval-Augmented Generation)** , a unified framework for multimodal long-document QA that integrates multimodal document expansion with query-centric, graph-based retrieval for finegrained and interpretable evidence selection. 

• We introduce the **MCQG (Multimodal Chunk-Query Graph)** , which links generated queries to corresponding multimodal chunks and connects semantically related information across pages. 

• To construct MCQG, we leverage **MDoc2Query** , a multimodal document expansion process that generates answerable queries from multimodal chunks. 

• Extensive experiments on MMLongBench-Doc [25] and LongDocURL [8] show that MLDocRAG consistently improves QA accuracy, advancing multimodal long-context document understanding. 

## **2 Related Work** 

_Long-Context Document Understanding._ Understanding multimodal long-context documents, such as research papers or technical reports, requires resolving both cross-modal heterogeneity and long-range cross-page reasoning—capabilities still limited in current models [8, 25]. Despite the strong local alignment abilities of 

Large Vision-Language Models (LVLMs) like GPT-4o [20], Qwen2.5VL [3], and Gemini [35], their fixed context windows limit their effectiveness in capturing globally relevant evidence dispersed across pages and modalities. This limitation leads to degraded performance in complex reasoning tasks, where key information is sparsely located, as highlighted in benchmarks such as MMLongBenchDoc [25] and LongDocURL [8]. Retrieval-Augmented Generation (RAG) offers partial relief by introducing external memory, yet struggles with retrieving semantically aligned multimodal content at scale. Retrieval-Augmented Generation (RAG) partly mitigates this by introducing external memory, but struggles to retrieve and integrate semantically aligned multimodal information at scale. These challenges call for new retrieval and representation strategies tailored to multimodal long-document understanding. 

_Multimodal RAG._ Recent efforts in multimodal Retrieval Augmented Generation (RAG) have explored various strategies to adapt long-document understanding to the multimodal setting. Common approaches include OCR-based text extraction, image captioning fused with text chunks, shared embedding retrieval via multimodal encoders (e.g., CLIP [29], SigLIP [37, 47]), and vision-based page retrieval using rendered document images (e.g., ColPali [14]). Some integrate structured knowledge via document-derived graphs to enhance reasoning [1, 26, 49, 50]. However, existing pipelines operate at coarse granularity, overlooking fine-grained cross-modal and cross-page associations, leading to incomplete grounding and retrieval mismatches. This motivates developing semantically aligned and structurally informed retrieval for multimodal contexts. 

_Document Expansion._ Document expansion has been widely adopted in RAG settings to improve retrieval coverage by generating synthetic contents that anticipate potential information needs [12, 31, 34, 39]. Methods such as Doc2Query [27] and its enhanced variant Doc2Query-- [15] generate diverse, semantically meaningful queries from document content, effectively enriching the retrieval index. Building on this idea, QCG-RAG [44] introduces a query-centric graph structure that connects queries to their source textual chunks, enabling multi-hop retrieval and structured evidence aggregation in long-document scenarios. Yet, these methods remain unimodal, lacking explicit modeling of cross-modal relationships. Extending query-centric expansion to multimodal documents remains an open problem for capturing fine-grained, heterogeneous associations. 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

## **3 Methodology** 

## **3.1 Preliminaries** 

We consider the task of multimodal long-context document question answering, where the goal is to answer a user natural language question _𝑞[𝑢]_ based on a multimodal long-context document _𝐷_ that spans multiple pages and contains heterogeneous content. Each document _𝐷_ originates from a PDF file and is processed into a sequence of pages _𝐷_ = { _𝑃_ 1 _, 𝑃_ 2 _, . . . , 𝑃𝑋_ } via OCR. Each page _𝑃𝑥_ consists of a set of modality-specific chunks _𝐶𝑖_ = { _𝑐_ 1 _,𝑐_ 2 _, . . . ,𝑐𝑁_ }, where each chunk _𝑐_ is associated with a modality (text ( _𝑡𝑥𝑡_ ) or image ( _𝑖𝑚𝑔_ )) and a content type (e.g., paragraph, figure, table, or equation). 

Text chunks are contiguous paragraphs or extracted titles; image chunks consist of a visual region paired with an associated caption; table chunks include a caption, a rendered table image, and an OCRconverted Markdown-style textual representation [40]. The QA task follows a retrieval-augmented generation (RAG) paradigm: given a query _𝑞_ , a retriever selects the top- _𝐾_ relevant multimodal chunks { _𝑐_ 1[′] _[, . . . ,𝑐] 𝐾_[′][}][from] _[ 𝐷]_[,][and][a][LVLM][conditions][on][these][chunks][to] generate the final answer _𝑎_ ˆ. The primary challenge lies in retrieving semantically aligned and cross-modally grounded chunks from the long, heterogeneous document to support accurate and coherent generation. 

## **3.2 Framework Overview** 

To tackle the challenges of cross-modal heterogeneity and longrange reasoning in multimodal long-document QA, we propose a _Multimodal Long-Context Document Retrieval Augmented Generation (MLDocRAG)_ framework based on the construction and usage of a _Multimodal Chunk-Query Graph (MCQG)_ that organizes multimodal document content around semantically rich, answerable queries. To construct MCQG, we move beyond conventional chunkbased retrieval paradigms that treat multimodal content as flat and independent units, and instead hypothesize that organizing multimodal long-context document understanding around generated, answerable queries enables more fine-grained, semantically aligned, and interpretable retrieval. Inspired by prior work on document expansion via query generation (e.g., Doc2Query [27]), we extend this idea to the multimodal setting by proposing MDoc2Query—a multimodal document expansion framework that generates semantically rich queries from heterogeneous document chunks. These generated queries act as _retrieval anchors_ that bridge the gap between user information needs and multimodal document content. Figure 2 illustrates the overall architecture of our proposed MLDocRAG framework which consists of two main stages: _MCQG Construction_ and _MCQG Usage_ . 

_MCQG Construction._ In this stage, a multimodal long document parsed from a PDF is decomposed into modality-specific chunks, where text-modality chunks correspond to paragraph text (including equations), and image-modality chunks correspond to figures or tables, each associated with its caption and any OCR-derived structured textual content. We then apply the MDoc2Query process to generate a set of answerable queries for each chunk using a Large Vision-Language Model (LVLM). These queries are explicitly 

linked to their source chunks and further connected to semantically similar queries across the document, forming the _Multimodal Chunk-Query Graph (MCQG)_ . This graph captures both intra-modal and inter-modal associations, and provides a structured retrieval index that aligns semantically meaningful questions with relevant multimodal evidence. 

_MCQG Usage._ At inference time, a user query is embedded and matched against nodes in the MCQG using KNN-based retrieval in the query embedding space. By retrieving semantically similar generated queries and aggregating their linked source chunks, the system collects a compact yet relevant set of multimodal content. The retrieved chunks are further ranked based on their semantic similarity to the user query and provided as context to a Large Language or Vision-Language Model (LLM/LVLM) for final answer generation. This query-centric retrieval strategy enables interpretable, structured evidence aggregation over multimodal long contexts. 

## **3.3 MCQG Construction** 

The MCQG Construction stage aims to transform a multimodal long-context document into a query-centric retrieval structure that supports fine-grained, semantically aligned, and cross-modal evidence retrieval. Specifically, we convert the long document into a collection of multimodal chunks, generate answerable queries from each chunk, and construct a graph that encodes chunk-query and inter-query associations. This process comprises four steps: (1) Document Parsing, (2) MDoc2Query, (3) Graph Assembly, and (4) Vector & Graph Storage. 

_(1) Document Parsing._ We adopt an existing multimodal PDF parsing tool, such as MinerU [40], to extract structured information from the document. Given a PDF document _𝐷_ = { _𝑃_ 1 _, 𝑃_ 2 _, . . . , 𝑃𝑋_ } of _𝑋_ pages, we extract a layout-preserving sequence of heterogeneous document components, including: (i) paragraphs (text only); (ii) figures (images with captions); (iii) tables (table images accompanied by textual captions and OCR-converted Markdown text); and (iv) equations (parsed into Markdown text). The extracted content is stored in a JSON format ordered by visual layout position. Based on this structured output, we define a set of modality-specific chunks: 

**==> picture [198 x 9] intentionally omitted <==**

Paragraph text is segmented into overlapping spans using a sliding window with a maximum token length and a fixed stride. Equations are treated as part of the regular text content. In contrast, each figure or table is treated as an image-modality chunk, while preserving its associated caption and any structured OCR-derived textual content. Visual noise filtering can be further applied to remove uninformative images (e.g., blank pages, string-only images, or logos; see Appendix B) via visual chunk classification using zero-shot CLIP inference. 

_(2) MDoc2Query._ To bridge document content with potential information needs, we extend the Doc2Query paradigm to multimodal settings. For each chunk _𝑐𝑖_ ∈C, we employ a Large Vision-Language Model (LVLM) to generate a set of answerable query–answer pairs: 

**==> picture [216 x 17] intentionally omitted <==**

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

**Figure 2: Overview of our proposed MLDocRAG framework, consisting of (a)** _**MCQG Construction**_ **for building a multimodal chuck-query graph and (b)** _**MCQG Usage**_ **for retrieving relevant multimodal chunks in long-document QA.** 

Here, _𝑞𝑖_[(] _[𝑗]_[)] denotes a generated query and _𝑎𝑖_[(] _[𝑗]_[)] its corresponding answer, both grounded in chunk _𝑐𝑖_ . The number of pairs per chunk is adaptively determined by the richness of its content, up to a global cap _𝑀_ max per chunk. Each query–answer pair ( _𝑞𝑖_[(] _[𝑗]_[)] _,𝑎𝑖_[(] _[𝑗]_[)] ) is embedded into a dense vector representation using a pretrained text encoder _𝜙_ (·) (e.g., BGE-m3 [5]): 

**==> picture [172 x 14] intentionally omitted <==**

where [ _𝑞_ ; _𝑎_ ] denotes concatenation of the query and answer as the retrieval unit. We simplify notation in the remainder of the paper by referring to _𝑞𝑖_[(] _[𝑗]_[)] as a shorthand for the full query–answer pair. 

_(3) Graph Assembly._ We build the Multimodal Chunk-Query Graph (MCQG) as a heterogeneous graph G = (V _,_ E) with nodes: 

**==> picture [182 x 26] intentionally omitted <==**

Nodes in V include **Chunk Nodes** C and **Query Nodes** Q. Edges in E include: (1) **Chunk–Query (C-Q) Edges** : Each query _𝑞𝑖_[(] _[𝑗]_[)] is connected to its originating chunk _𝑐𝑖_ via a directed anchor edge. (2) **Query–Query (Q-Q) Edges** : We compute semantic similarity between queries using inner product of embeddings and connect each query to its top- _𝑘_ nearest neighbors (KNN): 

**==> picture [193 x 10] intentionally omitted <==**

Here, _𝑞_ and _𝑞_[′] denote two generated queries. ⟨· _,_ ·⟩ denotes the inner product. All query embeddings are _ℓ_ 2-normalized prior to similarity computation. The constant offset _𝜖_ (e.g., _𝜖_ = 1 _._ 0) is added to ensure non-negative similarity scores for stable KNN construction. This dual-edge structure allows the graph to capture both local chunk associations and global semantic neighborhoods across queries, enabling multi-hop traversal and cross-modal reasoning during retrieval. 

_(4) Storage._ To support scalable and efficient retrieval, we decouple vector retrieval from graph traversal by storing: (1) Query–Answer vectors {v _𝑖_[(] _[𝑗]_[)] } in a dense **vector database** (e.g., FAISS [10], ElasticSearch [22]), supporting fast approximate nearest neighbor (ANN) search [2]. (2) The full MCQG graph structure in a **graph database** (e.g., Neo4j [17]), preserving chunk–query and query–query relationships. 

Importantly, we do _not_ perform vectorization of multimodal chunks (e.g., figure or table content) directly. Instead, all retrieval operates in the query–answer space, thereby avoiding the need for complex multimodal embedding alignment and reducing storage and computation overhead. The generated queries serve as interpretable, semantically rich anchors that effectively summarize and index the multimodal document content. 

## **3.4 MCQG Usage** 

The MCQG Usage stage aims to retrieve semantically relevant multimodal content in response to a user query by leveraging the structure of the Multimodal Chunk-Query Graph (MCQG). Rather than retrieving from raw document chunks, our MLDocRAG approach retrieves and aggregates content via generated queries that serve as semantically aligned retrieval anchors. This process involves four main steps: (1) Query Node Retrieval, (2) Chunk Node Ranking, (3) Context Collection, and (4) Answer Generation. 

_(1) Query Node Retrieval._ Given a user query _𝑞[𝑢]_ , we first compute its vector embedding _𝜙_ ( _𝑞[𝑢]_ ) using the same query encoder used during MCQG construction. We perform approximate nearest neighbor (ANN) search over the query–answer vectors in the vector database to retrieve the top- _𝑛_ semantically similar generated queries: 

**==> picture [190 x 10] intentionally omitted <==**

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

In practice, the number of retrieved queries is jointly constrained by the maximum node budget _𝑛_ and a similarity threshold _𝛼_ ∈[0 _,_ 2] (e.g., _𝛼_ = 1 _._ 0), such that only queries with sim( _𝑞[𝑢] ,𝑞_ ) ≥ _𝛼_ are retained. To capture broader contextual evidence, we expand each retrieved query node _𝑞_ ∈Qret in the MCQG via multi-hop neighbor expansion. For each query _𝑞_ ∈Qret, we traverse _ℎ_ hops in the graph (through Query–Query Edges) and collect its neighboring queries: 

**==> picture [173 x 22] intentionally omitted <==**

where Nbr _ℎ_ ( _𝑞_ ) denotes the set of _ℎ_ -hop neighbors of _𝑞_ in the MCQG, and Qexp represents the expanded query set that augments the initially retrieved queries with semantically related queries discovered through multi-hop graph traversal. 

_(2) Chunk Node Ranking._ Each query _𝑞_ ∈Qexp is linked to a source multimodal chunk _𝑐_ ∈C. We collect all chunks associated with the expanded query set: 

**==> picture [194 x 12] intentionally omitted <==**

To prioritize the most relevant evidence, we assign a relevance score to each candidate chunk _𝑐𝑖_ ∈Ccand based on the maximum semantic similarity between the user query _𝑞[𝑢]_ and any query linked to _𝑐𝑖_ : 

**==> picture [177 x 25] intentionally omitted <==**

Here, { _𝑞_ | ( _𝑞,𝑐𝑖_ ) ∈E} denotes the subset of expanded queries in Qexp that are connected to chunk _𝑐𝑖_ in the MCQG. 

_(3) Context Collection._ Based on the relevance scores, we select the top- _𝐾_ (e.g., _𝐾_ = 5) ranked multimodal chunks: 

**==> picture [167 x 10] intentionally omitted <==**

where the ranking is determined by score( _𝑐𝑖_ ). The selected chunks are concatenated to form the multimodal retrieval context for the LVLM. These chunks may include text blocks, image regions with captions, and tables augmented with structured and OCR-derived textual content. 

_(4) Answer Generation._ Finally, the selected multimodal context Crel is provided to a Large Vision–Language Model (LVLM), together with the original user query _𝑞[𝑢]_ , to generate the final answer: 

**==> picture [159 x 10] intentionally omitted <==**

This generation step benefits from the query-centric retrieval strategy and graph-based evidence aggregation, which together improve grounding, coverage, and factual consistency when answering complex questions over multimodal long-context documents. 

## **3.5 MDoc2Query Optimization** 

The effectiveness of MLDocRAG largely depends on the quality of the Multimodal Chunk–Query Graph (MCQG), which is constructed via MDoc2Query. In particular, the quality and granularity of the generated _answerable queries_ produced by MDoc2Query are critical, as they directly determine the semantic coverage, retrievability, and grounding fidelity of the overall pipeline. To this end, we explore optimization strategies for MDoc2Query from both _non-parametric_ and _parametric_ perspectives. 

_Non-Parametric Optimization._ By default, MDoc2Query in MLDocRAG employs a LVLM to generate a set of answerable queries from parsed document chunks, such as cropped images or parsed table segments. However, document parsing tools (e.g., MinerU [40]) often strip away essential contextual information—including figure captions, table headers, and hierarchical section titles—resulting in isolated chunks that can be semantically ambiguous and consequently degrade the quality of the generated queries. To mitigate this issue, we adopt a _Page-Context-Aware Generation_ strategy. Specifically, for a given chunk _𝑐𝑖_ located on page _𝑥_ with the corresponding page rendering image _𝑃𝑥_[page] , we construct the LVLM input as a tuple ( _𝑐𝑖, 𝑃𝑥_[page] ). Incorporating page-level visual context enables the model to resolve ambiguities arising from incomplete local information, such as coreference resolution (e.g., linking a chunk labeled “Table 3” to its corresponding description on the same page). 

_Parametric Optimization._ In addition to non-parametric strategies, we further explore parametric optimization by fine-tuning a pretrained Large Vision–Language Model (LVLM) on a curated set of high-quality multimodal _chunk-to-query_ exemplars. Each training instance consists of a multimodal chunk _𝑐_ (including text, an image with its caption, or a table with structured content) paired with a set of human-curated or automatically synthesized answerable query–answer ( _𝑞,𝑎_ ) pairs. The LVLM is trained using standard teacher forcing, where the model conditions on the input chunk to generate the corresponding answer and subsequently autoregressively decodes the associated queries. Through parametric adaptation, the model learns to produce more semantically precise, context-aware, and structurally grounded queries, thereby improving the expressiveness and reliability of MDoc2Query for downstream MCQG construction. 

## **4 Experimental Setup** 

In this section, we evaluate the effectiveness of the proposed MLDocRAG framework for multimodal long-context document question answering (QA), and compare it with existing approaches (as illustrated in Figure 1). Specifically, our experimental study is designed to address the following research questions: 

• **RQ1:** How does MLDocRAG perform on multimodal long-context document QA compared with baseline methods? 

• **RQ2:** What are the effects on MLDocRAG of different MCQG node variants, including query node choices (query vs. answer), chunk ranking strategies (max vs. mean), and visual noise filtering? • **RQ3:** How do key hyperparameters of MCQG usage affect the performance of MLDocRAG, such as expansion hops _ℎ_ , KNN neighbors _𝑘_ , and max nodes _𝑛_ ? 

• **RQ4:** What is the impact of MDoc2Query optimization from both non-parametric and parametric perspectives? 

## **4.1 Datasets & Metrics** 

_Datasets._ We evaluate our MLDocRAG on two multimodal longcontext document QA benchmarks: (1) **MMLongBench-Doc** [25]: A curated benchmark for multimodal long-document understanding, consisting of documents in PDF format with diverse content including text, images, tables, and charts. Questions are paired with answerable evidence spread across pages and modalities. (2) 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

**LongDocURL** [8]: A newly collected dataset containing web-based scientific and technical documents. Each instance includes a document (in PDF or HTML format), a natural language question, and annotated answerable segments across multimodal content. 

_Metrics._ We adopt exact match Accuracy as the primary evaluation metric to measure the factual correctness of generated answers. To ensure consistent and scalable judgment across modalities and formats, we follow recent work and employ an _LLM-as-a-Judge_ protocol [16], using a strong LLM (e.g., Qwen2.5-72B [36, 45]) to verify whether the predicted answer matches the gold reference answer. The evaluation prompt is shown in Appendix A. 

## **4.2 Baselines** 

We compare MLDocRAG with representative baselines from five categories: 

• _**Text Only (txt).**_ Use only text chunks with or without basic retrieval (BM25 [30] / BGE-m3 [5]): LC _[𝑡𝑥𝑡]_ with textual long-context reasoning, RAG _[𝑡𝑥𝑡]_ BM25[with sparse retrieval,][ RAG] _[𝑡𝑥𝑡]_ BGE-m3[with dense] retrieval. 

• _**Image2Text (txt+i2t).**_ Augment text with LVLM-generated image captions, treated as plain text: LC _[𝑡𝑥𝑡]_[+] _[𝑖]_[2] _[𝑡]_ , RAG _[𝑡𝑥𝑡]_ BM25[+] _[𝑖]_[2] _[𝑡]_[, RAG] _[𝑡𝑥𝑡]_ BGE-m3[+] _[𝑖]_[2] _[𝑡]_[.] • _**Multimodal (txt+img).**_ Encode both text and image chunks using multimodal embedding models for dense retrieval: LC _[𝑡𝑥𝑡]_[+] _[𝑖𝑚𝑔]_ for multimodal long-context reasoning, MRAG _[𝑡𝑥𝑡]_ CLIP[+] _[𝑖𝑚𝑔]_ with CLIP [29], MRAG _[𝑡𝑥𝑡]_ SigLIP[+] _[𝑖𝑚𝑔]_ with SigLIP [37, 47], MRAG _[𝑡𝑥𝑡]_ ColPali[+] _[𝑖𝑚𝑔]_ with ColPali [14]. • _**Page-level (page).**_ Render full document pages as images and perform vision-only retrieval followed by VQA: LC _[𝑝𝑎𝑔𝑒]_ for visual long-context reasoning, VRAG _[𝑝𝑎𝑔𝑒]_ CLIP[, VRAG] _[𝑝𝑎𝑔𝑒]_ SigLIP[, VRAG] _[𝑝𝑎𝑔𝑒]_ ColPali[.] • _**Graph.**_ Construct knowledge graphs from extracted multimodal entities with the augmented text (i.e., _txt+i2t_ ) for graph-based retrieval: GraphRAG _[𝑡𝑥𝑡] 𝐵𝐺𝐸_[+] _[𝑖]_ −[2] _𝑚[𝑡]_ 3[extended with efficient and lightweight] MiniRAG [13]. 

In addition, we evaluate several variants of the proposed MLDocRAG framework under different MCQG construction and usage settings to analyze the effects of key components and hyperparameters: **(1) Node Variants.** (i) _MLDocRAG w/ Query_ to use queries only as nodes; (ii) _MLDocRAG w/ Answer_ to use answers only as nodes; (iii) _MLDocRAG w/ Mean_ to apply mean semantic similarity for chunk node ranking instead of the max operation; (iv) _MLDocRAG w/o Filter_ to disable visual noise filtering during document parsing. **(2) Hyperparameters.** (i) _Hops ℎ_ ∈{0 _,_ 1 _,_ 2 _,_ 3} for multihop neighbor expansion; (ii) _KNN 𝑘_ ∈{1 _,_ 2 _,_ 3 _,_ 4 _,_ 5} for query-query edges; (iii) _Max Nodes 𝑛_ ∈{5 _,_ 10 _,_ 15 _,_ 20} for query node retrieval. We additionally compare different query node retrieval backends (BGE-m3 vs. BM25) and similarity thresholds _𝛼_ ∈{1 _._ 0 _,_ 1 _._ 2}. Note that MLDocRAG performs chunk-only query generation by default, whereas MLDocRAG _[𝑃]_ and MLDocRAG _[𝑃] 𝑃_[additionally incorporate] page context during query generation and final answer generation, respectively. 

## **4.3 Setup Details** 

_Document Parsing Setting._ Multimodal long-context documents are parsed using MinerU [40], which extracts layout-ordered elements from PDF into structured JSON files. Chunks are constructed 

as follows: (1) Text: Paragraphs are segmented using a maximum length of 1200 tokens with an overlap of 100 tokens. Equations parsed as Markdown are treated as regular text. (2) Image: Figures and tables, together with their captions and OCR-derived content, are treated as individual image-modality chunks. In addition, document pages are rendered as images for page-level methods and MDoc2Query optimization. We further apply visual noise filtering to image chunks via zero-shot classification using CLIP[1] , with details in Appendix B. 

_Model Configuration._ We employ the BGE-m3 encoder to generate dense embeddings for queries, which are stored and indexed in ElasticSearch [22] as the core vector database to support efficient similarity search. For query generation and final answer generation, we use Qwen2.5-VL-32B[2] [4] by default, while Qwen2.5-VL-7B[3] [4] is adopted for MDoc2Query optimization. For evaluation, we adopt an LLM-as-a-Judge setup using Qwen2.5-72B[4] [36, 45]. All LLMs/LVLMs are deployed via the SGLang [51] framework on NVIDIA H20 GPUs to enable high-throughput inference. 

_Default Hyperparameters._ For experiments on both MMLongBenchDoc and LongDocURL, we adopt a unified set of default hyperparameters. Specifically, we set the KNN neighborhood size to _𝑘_ = 3 for query–query edge construction, retrieve up to _𝑛_ = 10 query nodes with a similarity threshold _𝛼_ = 1 _._ 2, and perform _ℎ_ = 2-hop query expansion. For answer generation, the top- _𝐾_ = 5 ranked multimodal chunks are selected as the retrieval context. 

## **5 Experimental Results** 

In this section, we analyse the experimental results with respect to the four research questions stated in Section 4 to gauge the effectiveness of our proposed MLDocRAG. 

## **5.1 MLDocRAG vs. Baselines (RQ1)** 

Table 1 reports the performance of MLDocRAG and representative baselines on MMLongBench-Doc and LongDocURL in terms of accuracy (%). Overall, MLDocRAG achieves the best overall performance on both datasets, with an accuracy of 47.9% on MMLongBenchDoc and 50.8% on LongDocURL, consistently outperforming all baseline methods. In particular, compared to text-only and imageto-text baselines, MLDocRAG benefits from explicitly modeling multimodal evidence without collapsing visual information into flat textual descriptions, thereby preserving fine-grained visual semantics that are critical for layout-, chart-, and figure-centric questions. In contrast to multimodal dense retrieval methods that independently retrieve text and image chunks, MLDocRAG organizes multimodal content around generated, answerable queries and performs query-centric multi-hop expansion, enabling effective aggregation of semantically related evidence scattered across pages. This advantage is particularly evident in multi-page settings, where simple chunk-level retrieval or page-level visual reasoning fails to capture long-range dependencies. Moreover, in contrast to the text-only graph-based baseline, MLDocRAG explicitly models crossmodal associations through the Multimodal Chunk–Query Graph, 

1https://huggingface.co/openai/clip-vit-base-patch32 

2https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct 

3https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct 

4https://huggingface.co/Qwen/Qwen2.5-72B-Instruct 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

|Method|Model<br>Retriever<br>GEN|Modality<br>TXT IMG|**MMLongBench-Doc**<br>Evidence Source<br>Evidence Page<br>Overall<br>TXT LAY CHA TAB<br>FIG<br>SIN MUL UNA<br>ACC|**MMLongBench-Doc**<br>Evidence Source<br>Evidence Page<br>Overall<br>TXT LAY CHA TAB<br>FIG<br>SIN MUL UNA<br>ACC|**MMLongBench-Doc**<br>Evidence Source<br>Evidence Page<br>Overall<br>TXT LAY CHA TAB<br>FIG<br>SIN MUL UNA<br>ACC|**LongDocURL**<br>Evidence Source<br>Evidence Page<br>Overall<br>TXT LAY FIG TAB<br>SP<br>MP<br>CE<br>ACC|**LongDocURL**<br>Evidence Source<br>Evidence Page<br>Overall<br>TXT LAY FIG TAB<br>SP<br>MP<br>CE<br>ACC|**LongDocURL**<br>Evidence Source<br>Evidence Page<br>Overall<br>TXT LAY FIG TAB<br>SP<br>MP<br>CE<br>ACC|
|---|---|---|---|---|---|---|---|---|
|LC_𝑡𝑥𝑡_<br>RAG_𝑡𝑥𝑡_<br>_𝐵𝑀_25<br>RAG_𝑡𝑥𝑡_<br>_𝐵𝐺𝐸_−_𝑚_3|-<br>LLM<br>BM25<br>LLM<br>BGE-m3<br>LLM|✓<br>✗<br>✓<br>✗<br>✓<br>✗|44.3 31.9<br>20.8<br>12.8<br>22.0<br>44.9 26.1<br>20.2<br>14.2<br>17.1<br>40.0 31.1<br>20.2<br>10.6<br>18.4|32.8<br>21.9<br>69.3<br>31.6<br>18.9<br>76.3<br>29.2<br>19.4<br>77.2|36.9<br>36.8<br>36.0|59.1 38.8<br>26.0 41.5<br>60.2 37.4 24.9 37.3<br>59.8 34.7 19.3 31.5|43.3 45.7 29.2<br>43.1 45.9 26.8<br>41.4 44.6 21.0|40.2<br>39.5<br>36.8|
|LC_𝑡𝑥𝑡_+_𝑖_2_𝑡_<br>-<br>LLM<br>✓<br>✗<br>48.5<br>32.8<br>40.5<br>41.7<br>28.6<br>44.1<br>32.5<br>54.8<br>42.5<br>51.1 31.0 29.1 33.6<br>42.2 49.0 15.7<br>37.3<br>RAG_𝑡𝑥𝑡_+_𝑖_2_𝑡_<br>_𝐵𝑀_25<br>BM25<br>LLM<br>✓<br>✗<br>43.6 31.9<br>38.2<br>38.1<br>27.0<br>45.3<br>26.9<br>66.7<br>43.7<br>60.6<br>36.5 45.7 40.3<br>61.9<br>54.1 22.4<br>48.1<br>RAG_𝑡𝑥𝑡_+_𝑖_2_𝑡_<br>_𝐵𝐺𝐸_−_𝑚_3<br>BGE-m3<br>LLM<br>✓<br>✗<br>42.3 26.1<br>37.6<br>45.41 31.3<br>49.0<br>27.2<br>71.5<br>46.5<br>59.7 32.8 47.0<br>42.3<br>61.9<br>**57.4** 17.1<br>47.8|||||||||
|_𝐵𝐺𝐸𝑚_3<br>LC_𝑡𝑥𝑡_+_𝑖𝑚𝑔_<br>-<br>LVLM<br>✓<br>✓<br>**53.8** 34.5<br>**45.5**<br>45.9<br>**35.9**<br>52.4<br>**37.8**<br>45.6<br>46.1<br>57.7 34.7 30.9 39.8<br>46.0 48.9 23.4<br>40.7<br>MRAG_𝑡𝑥𝑡_+_𝑖𝑚𝑔_<br>_𝐶𝐿𝐼𝑃_<br>CLIP<br>LVLM<br>✓<br>✓<br>39.0 27.7<br>27.0<br>13.8<br>21.4<br>34.0<br>19.2<br>70.2<br>36.7<br>51.7 32.4 22.6 30.7<br>35.6 44.0 21.0<br>34.5<br>MRAG_𝑡𝑥𝑡_+_𝑖𝑚𝑔_<br>_𝑆𝑖𝑔𝐿𝐼𝑃_<br>SigLIP<br>LVLM<br>✓<br>✓<br>23.9 15.1<br>16.9<br>15.1<br>18.1<br>21.9<br>14.7<br>**82.0**<br>32.2<br>23.2 22.7<br>5.9<br>14.1<br>8.0<br>21.2 17.9<br>15.5<br>MRAG_𝑡𝑥𝑡_+_𝑖𝑚𝑔_<br>_𝐶𝑜𝑙𝑃𝑎𝑙𝑖_<br>ColPali<br>LVLM<br>✓<br>✓<br>34.1 29.4<br>28.1<br>25.7<br>25.3<br>35.8<br>20.3<br>71.1<br>38.1<br>51.0 31.8 32.9 34.0<br>45.2 44.8 23.4<br>38.9|||||||||
|LC_𝑝𝑎𝑔𝑒_<br>VRAG_𝑝𝑎𝑔𝑒_<br>_𝐶𝐿𝐼𝑃_<br>VRAG_𝑝𝑎𝑔𝑒_<br>_𝑆𝑖𝑔𝐿𝐼𝑃_<br>VRAG_𝑝𝑎𝑔𝑒_<br>_𝐶𝑜𝑙𝑃𝑎𝑙𝑖_|-<br>LVLM<br>CLIP<br>LVLM<br>SigLIP<br>LVLM<br>ColPali<br>LVLM|✗<br>✓<br>✗<br>✓<br>✗<br>✓<br>✗<br>✓|41.0 26.9<br>31.5<br>30.3<br>27.6<br>28.9 27.7<br>24.7<br>22.5<br>28.3<br>16.1 13.5<br>13.5<br>9.2<br>9.5<br>40.0 **37.8** 32.0<br>30.7<br>32.2|39.7<br>23.6<br>53.1<br>34.8<br>16.7<br>75.4<br>15.4<br>9.2<br>77.2<br>44.1<br>23.9<br>63.2|37.2<br>37.3<br>26.3<br>41.4|40.2 29.7 23.7 30.3<br>33.8 26.2 20.1 26.1<br>21.2 25.3 17.0 19.1<br>56.8 **39.4** 45.8 **53.1**|34.0 39.1 17.9<br>24.1 34.2 18.9<br>17.4 17.7 25.8<br>52.1 50.3 **36.4**|31.3<br>26.2<br>19.9<br>47.1|
|GraphRAG_𝑡𝑥𝑡_+_𝑖_2_𝑡_<br>_𝐵𝐺𝐸_−_𝑚_3|BGE-m3<br>LLM|✓<br>✗|48.1 25.5<br>36.4<br>**51.5**<br>30.2|51.3<br>27.1<br>48.5|42.6|58.2 33.9 32.8 36.1|59.2 54.4 15.7|45.3|
|MLDocRAG|BGE-m3 LVLM|✓<br>✓|47.2 **37.8** 42.7<br>41.3<br>31.9|**52.6** 26.4<br>71.5|**47.9**|**65.1 39.4 48.3** 41.1|**66.9** 56.3<br>23.4|**50.8**|



**Table 1: Performance comparison on MMLongBench-Doc and LongDocURL measured by Accuracy (%). For MMLongBench-Doc, five formats are considered: text (TXT), layout (LAY), chart (CHA), table (TAB), and image (FIG), with scopes including singlepage (SIN), multi-page (MUL), and unanswerable (UNA). For LongDocURL, four formats are evaluated: text (TXT), layout (LAY), table (TAB), and figures (FIG), with evidence spans categorized as single-page (SP), multi-page (MP), or cross-element (CE), where CE denotes the integration of multiple modalities. Best results are shown in bold, and second-best results are underlined.** 

**==> picture [180 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) MMLongBench-Doc (b) LongDocURL<br>**----- End of picture text -----**<br>


**Figure 3: Ablation on node variants across both datasets.** 

allowing structured evidence propagation and ranking. As a result, MLDocRAG delivers consistent gains across different evidence sources and document scopes, demonstrating superior grounding and robustness for multimodal long-context document question answering. 

## **5.2 Impact of Node Variants (RQ2)** 

Figure 3 analyzes the impact of different MCQG node variants on the performance of MLDocRAG. **(1) Query Node Choices (Query vs. Answer).** Generated queries play a dominant role in performance. While _MLDocRAG w/ Query_ remains relatively stable, _MLDocRAG w/ Answer_ exhibits substantial performance degradation (dropping below 39% on MMLongBench-Doc), indicating that answers alone are insufficient as graph node representations. The default MLDocRAG, which combines queries with answers, achieves the best results by leveraging answers as complementary contextual signals. **(2) Chunk Node Ranking Strategies (Max vs. Mean).** The Max strategy consistently outperforms Mean aggregation. Using averaged similarity scores ( _MLDocRAG w/ Mean_ ) degrades performance compared to the default Max-based ranking, suggesting that a chunk’s relevance is better captured by its most relevant query 

**==> picture [198 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Hops  ℎ (b) KNN  𝑘 (c) Max Nodes  𝑛<br>Figure 4: Ablation on hyperparameters (MMLongBench-Doc).<br>eo Kanes eo 8GEm3 me<br>=e kNN=2 =e eM2s we-<br>S osco. Los) A”<br>7 h z 3 ame z k3 a 3 ers 0 n rs<br>(a) Hops  ℎ (b) KNN  𝑘 (c) Max Nodes  𝑛<br>**----- End of picture text -----**<br>


**Figure 4: Ablation on hyperparameters (MMLongBench-Doc).** 

**Figure 5: Ablation on hyperparameters (LongDocURL).** 

rather than by an average over all associated queries. **(3) Visual Noise Filtering.** Visual noise filtering is crucial for effective retrieval. Removing this component ( _MLDocRAG w/o Filter_ ) leads to noticeable performance drops, confirming that filtering out irrelevant image chunks (e.g., blank pages or logos) is necessary to prevent noise from interfering with retrieval. 

## **5.3 Impact of Hyperparameters (RQ3)** 

Figures 4 and 5 illustrate the impact of key hyperparameters on the performance of MLDocRAG. **(1) Expansion Hops (** _ℎ_ **).** Multi-hop expansion ( _ℎ_ = 1 _,_ 2) consistently outperforms zero-hop retrieval ( _ℎ_ = 0), highlighting the benefit of query node expansion through 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

**==> picture [180 x 7] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) MMLongBench-Doc (b) LongDocURL<br>**----- End of picture text -----**<br>


**Figure 6: Ablation on MDoc2Query optimization.** 

graph traversal. However, performance degrades when _ℎ_ = 3, indicating that excessive expansion introduces semantic noise that outweighs the gains from additional context. **(2) KNN Neighbors (** _𝑘_ **).** Accuracy generally improves as the number of nearest neighbors increases, peaking around _𝑘_ = 3. This suggests that a moderate graph density effectively bridges semantic gaps between queries, while denser connections beyond this point yield diminishing returns or introduce noise. **(3) Max Nodes (** _𝑛_ **).** Performance exhibits a clear optimum at _𝑛_ = 10. Retrieving too few nodes ( _𝑛_ = 5) limits evidence coverage, whereas retrieving too many nodes ( _𝑛 >_ 10) introduces irrelevant information, leading to notable performance drops. In addition, a stricter similarity threshold ( _𝛼_ = 1 _._ 2) consistently outperforms a looser threshold ( _𝛼_ = 1 _._ 0), emphasizing the importance of prioritizing high-quality entry nodes over quantity. 

## **5.4 Impact of MDoc2Query Optimization (RQ4)** 

Figure 6 illustrates the impact of non-parametric and parametric optimization strategies for MDoc2Query on the performance of MLDocRAG. **(1) Non-Parametric Optimization.** Providing each chunk with its corresponding page image during query generation consistently improves performance. Specifically, _MLDocRAG[𝑃]_ outperforms the chunk-only MLDocRAG across both datasets, demonstrating that incorporating page-level context enhances query quality by mitigating semantic fragmentation caused by isolated chunks. Representative query–answer examples generated by MDoc2Query with and without page context are provided in Appendix C. Furthermore, _MLDocRAG[𝑃] 𝑃_[integrates page-level information at both the] query generation and answer generation stages, achieving the best overall performance. This result confirms that supplying reference page context throughout both retrieval indexing and QA stages maximizes evidence utilization. **(2) Parametric Optimization.** MLDocRAG remains effective when deployed with smaller LVLMs. Although the 32B model achieves the strongest overall performance, the 7B model with supervised fine-tuning ( _MLDocRAG w/ 7B SFT_ ) attains 50.1% accuracy on LongDocURL, closely matching the 32B baseline (50.8%). These results demonstrate that MLDocRAG can be efficiently adapted to resource-constrained models through parametric optimization. 

## **5.5 Use Case** 

Figure 7 illustrates a representative use case from MMLongBenchDoc that requires cross-modal reasoning across distant scientific diagrams (Pages 3 and 14). In contrast to baseline methods such as _𝑅𝐴𝐺_ BGE-m3 _[𝑡𝑥𝑡]_[+] _[𝑖]_[2] _[𝑡]_[and] _[ 𝑉𝑅𝐴𝐺]_ ColPali _[𝑝𝑎𝑔𝑒]_[, which fail to bridge this long-context] 

**Question & Answer** ✓ Question: Which step in Figure 1 maps to the content of Figure 10? **Evidence Pages:** [3, 14] **Ground Truth:** Deletion/duplication/rearrangement of the genetic material and Genetic diseases. **RAG[txt+i2t] BGE-m3** Response: The question asks which step in **Figure 1** (Pathogenesis of genetic diseases) maps to the content of **Figure 10** . However, the provided context does not include any information about Figure 10. Since the content of Figure 10 is not described in the given text, it is impossible to determine which step in Figure 1 corresponds to it. **Final Answer: I don’t know. VRAG[page] ColPali** Response: The question asks which step in **Figure 1** maps to the content of **Figure 10** . However, the provided images and text do not include either Figure 1 or Figure 10. Without these figures, it is impossible to determine the mapping between them. **Final Answer: I don’t know. MLDocRAG** ✓ Response: To determine which step in Figure 1 maps to the content of Figure 10, let’s analyze both figures step by step. **Figure 1: Pathogenesis of Genetic Diseases.** Figure 1 outlines a sequence of events leading to genetic diseases: (1) **Mutation** (2) **Deletion / Duplication / Rearrangement of Genetic Material** (3) **Deficient / Defective Transcription of mRNA** (4) **Deficient / Defective Post-Transcriptional Modifications of mRNA** (5) **Deficient / Defective Translation of mRNA** (6) **Deficient / Defective Production of Proteins / Regulatory Factors** (7) **Deficient / Defective Post-Translational Modifications of Proteins** (8) **Disturbed Function(s) of Metabolic Networks** (9) **Patho-Physiological Alterations of Cellular Functions** (10) **Genetic Disease Figure 10: Pathological Consequences of Genomic Structural and Functional Mutations.** Figure 10 illustrates the consequences of genomic imprinting defects and chromosomal abnormalities, specifically: • **Structural and numerical chromosomal mutations** leading to altered genetic material. • The resulting phenotypes (e.g., normal zygote, gynogenetic constitution, and androgenetic constitution) are shown. **Mapping the Content.** Figure 10 focuses on the consequences of **structural and numerical chromosomal mutations** , which directly correspond to the second step in Figure 1: **Deletion / Duplication / Rearrangement of Genetic Material** . The phenotypes shown in Figure 10 (e.g., normal zygote, gynogenetic constitution, and androgenetic constitution) are the downstream effects of these chromosomal mutations, aligning with the pathogenesis outlined in Figure 1. **Final Answer: Deletion / Duplication / Rearrangement of Genetic Material.** 

**Figure 7: Use Case on MMLongBench-Doc.** 

gap, MLDocRAG successfully retrieves the dispersed evidence and performs the required multi-hop reasoning. Specifically, MLDocRAG accurately aligns the “chromosomal mutations” visually depicted in “Figure 10” on Page 14 with the corresponding “Deletion / Duplication / Rearrangement” stage in “Figure 1” on Page 3, demonstrating its superior capability in fine-grained information extraction and cross-page visual alignment. More use cases are provided in Appendix D. 

## **6 Conclusion** 

We presented MLDocRAG, a framework for multimodal long-document QA built on the Multimodal Chunk-Query Graph (MCQG), which enables unified, query-centric retrieval. By extending document expansion to the multimodal setting via MDoc2Query, MCQG organizes heterogeneous chunks and their generated queries into a structured graph that captures cross-modal and cross-page associations. This query-centric representation supports selective, multi-hop retrieval and semantically grounded evidence aggregation, achieving consistent gains on MMLongBench-Doc and LongDocURL. Our results demonstrate the effectiveness of query-based multimodal retrieval and the scalability of graph-structured organization for multimodal long-context understanding. 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

## **7 Limitations** 

While MLDocRAG achieves promising results, several limitations remain. First, it currently supports only text and image, limiting generalization to richer modalities such as video or audio. Second, its effectiveness depends on the quality of generated queries—noisy or incomplete queries may reduce retrieval accuracy. Finally, constructing large multimodal graphs can be computationally expensive, posing challenges for scaling to massive document collections. 

## **References** 

- [1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari. 2025. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. _arXiv preprint arXiv:2502.08826_ (2025). 

- [2] Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANNBenchmarks: A benchmarking tool for approximate nearest neighbor algorithms. _Information Systems_ 87 (2020), 101374. 

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2. 5-vl technical report. _arXiv preprint arXiv:2502.13923_ (2025). 

- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_ (2025). 

- [5] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. _arXiv preprint arXiv:2402.03216_ 4, 5 (2024). 

- [6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ . 24185–24198. 

- [7] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ (2024). 

- [8] Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, et al. 2025. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . 1135–1159. 

- [9] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. 2024. Internlmxcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _Advances in Neural Information Processing Systems_ 37 (2024), 42566–42592. 

- [10] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025. The faiss library. _IEEE Transactions on Big Data_ (2025). 

- [11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. _arXiv preprint arXiv:2404.16130_ (2024). 

- [12] Miles Efron, Peter Organisciak, and Katrina Fenlon. 2012. Improving retrieval of short texts through document expansion. In _Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval_ . 911– 920. 

- [13] Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. 2025. MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation. _arXiv preprint arXiv:2501.06713_ (2025). 

- [14] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ (2024). 

- [15] Mitko Gospodinov, Sean MacAvaney, and Craig Macdonald. 2023. Doc2Query–: when less is more. In _European Conference on Information Retrieval_ . Springer, 414–422. 

- [16] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. 2024. A survey on llm-as-a-judge. _The Innovation_ (2024). 

- [17] José Guia, Valéria Gonçalves Soares, and Jorge Bernardino. 2017. Graph databases: Neo4j analysis.. In _ICEIS (1)_ . 351–356. 

- [18] MD Zakir Hossain, Ferdous Sohel, Mohd Fairuz Shiratuddin, and Hamid Laga. 2019. A comprehensive survey of deep learning for image captioning. _ACM Computing Surveys (CsUR)_ 51, 6 (2019), 1–36. 

- [19] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. 2024. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ . 3096–3120. 

- [20] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. _arXiv preprint arXiv:2410.21276_ (2024). 

- [21] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. 2023. Financebench: A new benchmark for financial question answering. _arXiv preprint arXiv:2311.11944_ (2023). 

- [22] Rafal Kuc and Marek Rogozinski. 2013. _Elasticsearch server_ . Packt Publishing Ltd. 

- [23] Jian Li, Weiheng Lu, Hao Fei, Meng Luo, Ming Dai, Min Xia, Yizhang Jin, Zhenye Gan, Ding Qi, Chaoyou Fu, et al. 2024. A survey on benchmarks of multimodal large language models. _arXiv preprint arXiv:2408.08632_ (2024). 

- [24] Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. 2025. A comprehensive survey on long context language modeling. _arXiv preprint arXiv:2503.17407_ (2025). 

- [25] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems_ 37 (2024), 95963–96010. 

- [26] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. 2025. A survey of multimodal retrieval-augmented generation. _arXiv preprint arXiv:2504.08748_ (2025). 

- [27] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. _arXiv preprint arXiv:1904.08375_ (2019). 

- [28] Jiwon Park, Seohyun Pyeon, Jinwoo Kim, Rina Carines Cabal, Yihao Ding, and Soyeon Caren Han. 2025. DocHop-QA: Towards Multi-Hop Reasoning over Multimodal Document Collections. _arXiv preprint arXiv:2508.15851_ (2025). 

- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ . PmLR, 8748–8763. 

- [30] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. _Foundations and trends® in information retrieval_ 3, 4 (2009), 333–389. 

- [31] Amit Singhal and Fernando Pereira. 1999. Document expansion for speech retrieval. In _Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval_ . 34–41. 

- [32] Ray Smith. 2007. An overview of the Tesseract OCR engine. In _Ninth international conference on document analysis and recognition (ICDAR 2007)_ , Vol. 2. IEEE, 629– 633. 

- [33] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 37. 13636–13645. 

- [34] Tao Tao, Xuanhui Wang, Qiaozhu Mei, and ChengXiang Zhai. 2006. Language model information retrieval with document expansion. In _Proceedings of the Human Language Technology Conference of the NAACL, Main Conference_ . 407– 414. 

- [35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ (2023). 

- [36] Qwen Team. 2024. Qwen2.5: A Party of Foundation Models. https://qwenlm. github.io/blog/qwen2.5/ 

- [37] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. 2025. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. _arXiv preprint arXiv:2502.14786_ (2025). 

- [38] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. 2023. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 19528–19540. 

- [39] Xiaojun Wan, Jianwu Yang, and Jianguo Xiao. 2007. Single document summarization with document expansion. In _AAAI_ . 931–936. 

- [40] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. 2024. Mineru: An open-source solution for precise document content extraction. _arXiv preprint arXiv:2409.18839_ (2024). 

- [41] Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. 2025. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. In _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies_ 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

## _(Volume 1: Long Papers)_ . 3221–3241. 

- [42] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. 2024. Needle in a multimodal haystack. _Advances in Neural Information Processing Systems_ 37 (2024), 20540–20565. 

- [43] Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and Hao Fei. 2025. Multimodal chain-of-thought reasoning: A comprehensive survey. _arXiv preprint arXiv:2503.12605_ (2025). 

- [44] Yaxiong Wu, Jianyuan Bo, Yongyue Zhang, Sheng Liang, and Yong Liu. 2025. Query-Centric Graph Retrieval Augmented Generation. _arXiv preprint arXiv:2509.21237_ (2025). 

- [45] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2 Technical Report. _arXiv preprint arXiv:2407.10671_ (2024). 

- [46] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2024. A survey on multimodal large language models. _National Science Review_ 11, 12 (2024), nwae403. 

- [47] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In _Proceedings of the IEEE/CVF international conference on computer vision_ . 11975–11986. 

- [48] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. 2025. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented generation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 17443–17453. 

- [49] Rui Zhang, Chen Liu, Yixin Su, Ruixuan Li, Xuanjing Huang, Xuelong Li, and Philip S Yu. 2025. A Comprehensive Survey on Multimodal RAG: All Combinations of Modalities as Input and Output. _Authorea Preprints_ (2025). 

- [50] Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding, Xiaobao Guo, Minzhi Li, Xingxuan Li, et al. 2023. Retrieving multimodal information for augmented generation: A survey. _arXiv preprint arXiv:2303.10868_ (2023). 

- [51] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2024. Sglang: Efficient execution of structured language model programs. _Advances in neural information processing systems_ 37 (2024), 62557–62583. 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

## **A LLM Prompt** 

The following are some LLM prompts for query generation and answer evaluation, including MDoc2Query Prompt, Page-ContextAware MDoc2Query Prompt, Response Generation Prompt, PageContext-Aware Response Generation and Evaluation Prompt. 

## **A.1 MDoc2Query Prompt** 

## **MDoc2Query Prompt** 

- "query": the generated question 

- "answer": the exact answer 

Adjust the number of queries according to the information richness of the text and image (0–20 queries). 

[ 

- {"index": 0, "query": "Where did Alice and Bob meet?", "answer": " Central Cafe"}, 

- {"index": 1, "query": "When did the meeting take place?", "answer": "Tuesday"} 

- ] 

## **—Role—** 

You are a **Doc2Query** assistant. 

## **—Goal—** 

Given a text chunk and image, generate no more than 20 distinct user queries that can be directly answered by that provided text chunk and image. For each query, also provide an exact answer and a relevance score. 

- **—Generation Rules—** 

   - (1) **Answerability:** Every query must be answerable using _only_ information in the text chunk and image. 

   - (2) **Comprehensive coverage:** Collectively, all generated queries should cover all key ideas in the text chunk and image from different viewpoints or levels of detail. 

   - (3) **Adaptive quantity:** Adjust the number of generated queries (0-20) according to the semantic richness and information value of both text and image. 

   - (4) **Diversity requirements:** Ensure diversity along the following dimensions: 

      - _Question-style variety:_ Mix interrogative forms (who/what/why/how/where/when/did), imperative prompts ("List...", "Summarize..."), comparative questions, conditional or speculative forms, etc. 

      - _Content-perspective variety:_ Include queries on facts, definitions, methods, reasons, outcomes, examples, comparisons, limitations, and so on. 

      - _Information granularity:_ Combine macro (overall purpose, highlevel summary) and micro (specific figures, terms, steps) queries. 

      - _User-intent variety:_ Simulate intents such as confirmation, evaluation, usability, diagnosis, and decision-making (e.g., "Is this approach more efficient than...?"). 

      - _Linguistic expression variety:_ Vary wording, syntax (active ↔ passive), and synonyms; avoid repeating near-identical phrases. 

      - _No redundancy:_ Each query must be meaningfully distinct; eliminate trivial rephrases that offer no new angle. 

   - _Chunk-grounded specificity:_ Queries must be grounded in specific factual points from the text chunk and image. Avoid vague or generic formulations such as "What did X say?" or "Tell me more about Y" that lack anchoring in actual content. 

   - (5) **Required fields:** Each output item must be based on the given text chunk and image, including the following fields: 

      - query: A question or search phrase a user might ask. 

      - answer: A concise answer taken verbatim (or nearly verbatim) from the text chunk and image. 

## **—Example—** 

## **1. Input Chunk and Images** 

- [ 

- {"type": "images", "image": "/path/image1"}, 

- {"type": "images", "image": "/path/image2"}, 

- {"type": "text", "text": "Alice met with Bob at the Central Cafe on Tuesday to discuss their upcoming collaborative research project..."} 

- ] 

## **2. Generated Queries** 

   - Where did Alice and Bob meet? 

   - When did the meeting take place? 

   - What was the main topic discussed during the meeting? 

   - Who suggested incorporating advanced AI methodologies? 

   - ... (more queries) 

- **—Output Format—** Return only a JSON array of objects. Each object must include: • "index": a zero-based integer 

## **A.2 Page-Context-Aware MDoc2Query Prompt** 

## **Page-Context-Aware MDoc2Query Prompt** 

**—Role—** 

You are an expert **Document Understanding AI** designed to build a high-precision Retrieval Graph. 

## **—Inputs—** 

- (1) **Target Chunk:** A specific segment of content (Text, Cropped Image, or Table Data) extracted from a document. 

- (2) **Source Page Image:** The original full-page screenshot where this chunk is located. 

## **—Workflow (Strict Execution Order)—** 

**Step 1: Visual Localization & Context Recovery (Mental Scratchpad)** Locate: First, look at the Source Page Image and identify exactly where the Target Chunk is located. 

Scan Surroundings: Look immediately above, below, and to the sides of the chunk location in the full page. 

- Identify Missing Context: Find any information _not_ inside the chunk but essential for understanding it. This includes: 

   - Headers: Section Titles, Page Headers, Chapter Names. 

   - Captions: Figure Titles (e.g., "Figure 3: Revenue Trend"), Table Headers, or Axis Labels that were cut off. 

   - Pre-text: Preceding sentences that define what "this table" or "the data below" refers to. 

**Step 2: Adaptive Query Generation** 

Based on the chunk’s content and the recovered context from Step 1, generate a set of 5 to 20 Query-Answer pairs. 

_Quantity Rule:_ If the chunk is dense (e.g., a complex table or dense text), generate closer to 20. If it is sparse, generate closer to 5. 

- _Coverage Rule:_ You must generate queries for all 4 Levels defined below. The distribution ratio is up to you based on the content type. 

## **—Query Level Definitions—** 

**Level 1: Integrated Entity Relationships (Dense & Comprehensive)** Goal: Instead of asking about single entities one by one, generate complex queries that link multiple entities found in the chunk. 

Constraint: Do not generate simple questions like "What is X?". Instead, ask: "How does [Entity A] interact with [Entity B] regarding [Topic]?" or "What are the key specifications and performance metrics of [Product X]?" Why: To create strong semantic connections between entities in the graph without flooding it with simple queries. 

**Level 2: Detailed Content Description (The "Core Message")** 

Goal: Paraphrase and summarize the detailed information provided inside the chunk. 

Instruction: Imagine a user asks "What specific details does this paragraph/chart provide?". Cover the key arguments, data trends, or descriptive points. 

Why: To ensure the chunk is retrievable via natural language descriptions of its content. 

**Level 3: Macro Hierarchy (Navigation)** 

Goal: Anchor this chunk to the document structure. 

Instruction: Extract the Section Title, Chapter Name, or Page Header from the Source Page Image. Generate queries that link this specific chunk content to that high-level topic. 

Example: "What information does the section ’[Section Title]’ provide regarding [Chunk Topic]?" 

**Level 4: Context Restoration (Immediate Surroundings)** Goal: Fix "context loss" caused by chunking. Instruction: Explicitly incorporate the missing headers, captions, or pre-text 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

you found in Step 1 into the query. 

**==> picture [218 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
Critical: If the chunk is a table/chart without a title, use the title found in<br>the Page Image to formulate the query.<br>Example: "According to [Figure Title found in Page Image], what trend is<br>shown in the data regarding [Chunk Content]?" or "As detailed in [Table<br>Caption], what are the specific values for [Row Name]?"<br>—Output Format (JSON Only)—<br>[<br>{<br>"index": 0,<br>"level": "level_1_entity_integrated",<br>"query": "Comprehensive query connecting Entity A and B...",<br>"answer": "Detailed answer..."<br>},<br>{<br>"index": 1,<br>"level": "level_2_detailed_content",<br>"query": "...",<br>"answer": "..."<br>},<br>{<br>"index": 2,<br>"level": "level_3_macro_hierarchy",<br>"query": "...",<br>"answer": "..."<br>},<br>{<br>"index": 3,<br>"level": "level_4_context_restoration",<br>"query": "Query incorporating the external Figure Title/Table<br>Header...",<br>"answer": "..."<br>}<br>// ... generate 5 to 20 pairs total<br>]<br>**----- End of picture text -----**<br>


## (1) **Visual Verification (Grounding):** 

      - Use the **Reference Page Image** to verify the context of the **Text Chunk** . 

      - • _Example:_ If a chunk is a row of numbers, look at the Page Image to identify the **Column Headers** and **Row Labels** to ensure you interpret the numbers correctly. 

   - _Example:_ If a chunk describes a chart trend, look at the Page Image to confirm the **Axis Labels** , **Units** , and **Legends** . 

   - (2) **Contextual Synthesis:** 

      - The Text Chunk might be stripped of its section title. Use the Page Image to see which **Section Header** (e.g., "2023 Q4 Results" vs "2022 Q4 Results") the chunk belongs to. 

      - Combine information from multiple chunks if necessary to form a complete answer. 

   - (3) **Strict Evidence-Based:** 

      - Answer **ONLY** using the provided information. Do not use outside knowledge. 

      - If the text chunk and the image contradict each other, trust the **Visual Evidence (Image)** for raw data (numbers, charts) and the **Text Chunk** for semantic explanations. 

      - If the answer is not present in the provided contexts, explicitly state: "Based on the provided documents, I cannot answer this question." 

   - (4) **Conciseness:** 

      - Get straight to the point. Do not start with "Based on the context..." or "The image shows...". Just state the answer. 

- **—Output Format—** 

**Analysis:** (Briefly map the chunk text to the visual location in the page image to confirm headers/legends) **Final Answer:** [Your direct answer] 

## **A.5 Evaluation Prompt** 

## **A.3 Response Generation Prompt** 

## **Response Generation Prompt** 

You are a knowledgeable assistant that answers questions based on the given text and image data. 

## **—Guidelines—** 

- (1) Carefully reason through the provided information before answering, but only use evidence **explicitly supported** by the text or image. 

- (2) If the answer cannot be determined from the provided data, clearly say you don’t know. 

- (3) Avoid unnecessary explanations—respond concisely and directly. (4) Present the final output in the format: **Final Answer: [your answer]** 

## **A.4 Page-Context-Aware Response Generation Prompt** 

## **Page-Context-Aware Response Generation Prompt** 

- You are an expert Multimodal QA Assistant. You will be provided with a user question and a set of **Retrieved Contexts** . Each Context consists of: 

   - (1) **Text/Data Chunk:** A specific segment of text, a table row, or a data point retrieved from a document. 

   - (2) **Reference Page Image:** The full document page where this chunk is located. 

## **—Goal—** 

Answer the user’s question accurately by synthesizing information from the provided Text Chunks and verifying it against the Reference Page Images. 

**—Reasoning Guidelines—** 

## **Evaluation Prompt** 

You are a strict and precise evaluation assistant. You will be given a question, a reference answer, and a candidate answer generated via retrievalaugmented generation (RAG). 

## **—Goal—** 

Evaluate the candidate answer against the reference answer based on factual accuracy and completeness. Slight differences in phrasing are acceptable as long as the meaning is the same. If the candidate answer includes an analysis followed by "Final Answer:", only evaluate the content after "Final Answer:". If "Final Answer:" is missing, treat the entire candidate as the final answer. 

## **—Normalization for Unanswerable/None—** 

- Treat the following expressions as **equivalent to "Not answerable / No answer / None"** when they appear as the candidate’s final answer: 

   - _Explicit statements:_ "I don’t know", "Not answerable", "Not enough information", "Insufficient information", "Not mentioned", "Unknown", "Cannot be determined", "No information provided", "N/A", "Not applicable", "None". 

   - _Negative-existence statements:_ Assertions of absence for list/type questions, e.g., "No stages require a cooler.", "No such category exists.", "There are none.", "No [items] are present.", "None of the stages...". 

**Note:** This normalization applies **only when the reference answer itself is unanswerable/none/empty** . Normalization **takes precedence over all other rules** when determining equivalence. 

## **—Important Rules—** 

- If the candidate answer fails to provide an answer (e.g., says "I don’t know") **when the reference is answerable** , it must receive a **score of 0** . 

- If the reference answer is unanswerable/none/empty, the candidate answer must produce an unanswerable-equivalent expression (as normalized above) to receive **score 1** . If the candidate gives any substantive or fabricated specific content, assign **score 0** . 

- If the candidate answer overgeneralizes, omits key elements, or adds unrelated information not supported by the reference, the score must be **0** , even if part of it is correct. 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

   - Only when the candidate answer covers **all essential factual elements** of the reference answer without introducing unrelated content, should the score be **1** . 

- **—Scoring Criteria— Score = 1 (Correct):** 

   - The candidate answer accurately matches the factual content and level of detail of the reference answer. Minor wording differences are acceptable if the meaning is equivalent. 

- If the reference answer is "Unable to answer" / "Not answerable" / "None" / empty, and the candidate provides an unanswerableequivalent expression (after normalization). 

- **Score = 0 (Incorrect):** 

   - The reference answer is answerable, but the candidate says any unanswerable-equivalent expression (including negativeexistence). 

   - The reference answer is answerable, but the candidate gives an incorrect, incomplete, or irrelevant answer. 

   - Corresponding Page Image: 

- The reference answer is unanswerable/none/empty, but the candidate provides any substantive or fabricated answer. 

## **B Details of Visual Noise Filtering** 

We employ clip-vit-base-patch32 to classify visual chunks via zero-shot inference. Based on their semantic relevance to the RAG task, visual elements are partitioned into two sets: those retained for indexing and those filtered as decorative noise. Table 2 lists the specific labels for each action. 

|**Retain**<br>table, chart, graph, diagram, map, infographic, equation,<br>flow chart, scatter plot, bar chart, form|**Retain**<br>table, chart, graph, diagram, map, infographic, equation,<br>flow chart, scatter plot, bar chart, form|
|---|---|
|**Filter**|logo, banner, advertisement, poster, cover, illustration,<br>background, icon, photo, texture|



## **Table 2: Visual chunk classification and filtering actions.** 

**Rationale:** Elements labeled for filtering are pruned to eliminate redundant visual noise (e.g., repeating logos) and enhance the precision of the embedding space during retrieval. 

## **C MDoc2Query Example** 

There are some query-answer pair examples generated by MDoc2Query and Page-Context-Aware MDoc2Query. The multimodal chunk and corresponding page image as follows: 

## **Multimodal Chunk with Page Image** 

**==> picture [184 x 70] intentionally omitted <==**

**----- Start of picture text -----**<br>
{<br>"type": "image",<br>"text": "",<br>"image": [<br>"./images/9bc7afb9a74e615b67303104c415d.jpg"<br>],<br>"visual_context": "./visual_context/a325667e4e534f96.png"<br>}<br>Chunk Image:<br>**----- End of picture text -----**<br>


## **Doc2Query Example** 

**==> picture [210 x 291] intentionally omitted <==**

**----- Start of picture text -----**<br>
[<br>{<br>"index": 0,<br>"query": "Which app store had more apps in 2015?",<br>"answer": "Google Play Store",<br>"q_id": "reportq32015-151009093138-lva1-app6891_95-6-0"<br>},<br>{<br>"index": 1,<br>"query": "How many apps were in the Apple App Store in 2015?",<br>"answer": "1,5 million",<br>"q_id": "reportq32015-151009093138-lva1-app6891_95-6-1"<br>},<br>{<br>"index": 2,<br>"query": "What was the number of apps in the Google Play Store<br>in 2015?",<br>"answer": "1,6 million",<br>"q_id": "reportq32015-151009093138-lva1-app6891_95-6-2"<br>},<br>{<br>"index": 3,<br>"query": "In which year did the number of apps in both stores<br>start to increase significantly?",<br>"answer": "2013",<br>"q_id": "reportq32015-151009093138-lva1-app6891_95-6-3"<br>},<br>{<br>"index": 4,<br>"query": "How many apps were in the Apple App Store in 2012?",<br>"answer": "0,5 million",<br>"q_id": "reportq32015-151009093138-lva1-app6891_95-6-4"<br>},<br>{<br>"index": 5,<br>"query": "What was the number of apps in the Google Play Store<br>in 2012?",<br>"answer": "0,35 million",<br>"q_id": "reportq32015-151009093138-lva1-app6891_95-6-5"<br>},<br>...<br>]<br>**----- End of picture text -----**<br>


Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

Yongyue Zhang and Yaxiong Wu 

- **Page-Context-Aware Example** [ { "index": 0, "level": "level_1_entity_integrated", "query": "How do the growth trends of apps in the Apple App 

- Store and Google Play Store compare from 2012 to 2015, and what does this indicate about their market dynamics?", "answer": "From 2012 to 2015, both the Apple App Store and 

- Google Play Store experienced significant growth in the number of apps. The Google Play Store showed a more rapid increase, starting at 0.35 million in 2012 and reaching 1.6 million in 2015. The Apple App Store started at 0.5 million in 2012 and reached 1.5 million in 2015. This indicates that while both platforms grew, Google Play Store expanded at a faster rate, leading to a larger number of apps by 2015.", "q_id": "reportq32015-151009093138-lva1-app6891_95-6-0" 

- }, { "index": 1, "level": "level_2_detailed_content", "query": "What specific details does the chart provide about 

- the number of apps in the Apple App Store and Google Play Store from 2012 to 2015?", "answer": "The chart shows the number of apps in the Apple App Store and Google Play Store from 2012 to 2015, measured in 

- millions. In 2012, the Apple App Store had 0.5 million apps, while the Google Play Store had 0.35 million. By 2013, the numbers increased to 0.8 million for the Apple App Store and 0.37 million for the Google Play Store. In 2014, the Apple App Store reached 1.25 million apps, and the Google Play Store reached 1.3 million. By 2015, the Apple App Store had 1.5 million apps, and the Google Play Store had 1.6 million apps.", "q_id": "reportq32015-151009093138-lva1-app6891_95-6-1" 

- }, { "index": 2, "level": "level_3_macro_hierarchy", "query": "What information does the section 'Global Mobile 

- Apps by Number' provide regarding the growth of apps in the Apple App Store and Google Play Store?", "answer": "The section 'Global Mobile Apps by Number' provides information on the growth of apps in the Apple App Store and 

- Google Play Store from 2012 to 2015. It highlights that the number of apps in the Google Play Store grew by more than 50% last year, reaching over 1.6 million, compared to 1.5 million for the Apple App Store. This section emphasizes the rapid expansion of mobile apps globally and the competitive landscape between the two major app stores.", "q_id": "reportq32015-151009093138-lva1-app6891_95-6-2" 

- }, { "index": 3, "level": "level_4_context_restoration", "query": "According to the 'Global Mobile Apps by Number' 

- section, what trend is shown in the data regarding the growth of apps in the Apple App Store and Google Play Store?", "answer": "The 'Global Mobile Apps by Number' section shows a 

- consistent upward trend in the number of apps in both the Apple App Store and Google Play Store from 2012 to 2015. The Google 

- Play Store experienced a more significant growth rate, surpassing the Apple App Store by 2015. This trend indicates the rapid expansion of the mobile app market and the increasing competition between the two major app stores.", "q_id": "reportq32015-151009093138-lva1-app6891_95-6-3" 

- }, ... 

- ] 

**==> picture [171 x 11] intentionally omitted <==**

**----- Start of picture text -----**<br>
QA Example 1 on MMLongBench-Doc<br>**----- End of picture text -----**<br>


Souce evidence images: 

## **D MLDocRAG QA Example** 

There is a use case from the MMLongBench-Doc dataset to evaluate the multi-page and cross-modal reasoning capabilities on baseline method and our MLDocRAG. 

**Question:** Among the adults conducted the survey on April 25 - May 1 2018, how many adults rated Trump’s government ethical standards as poor? **Evidence Pages:** [3, 4] **Ground Truth:** 541 

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY 

MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

**==> picture [200 x 351] intentionally omitted <==**

**----- Start of picture text -----**<br>
RAG [txt+i2t] BGE-m3<br>Response:<br>To answer the question, let’s break down the information provided:<br>(1) Overall Ratings of Trump’s Government Ethical Standards:<br>• The survey conducted from April 25 to May 1, 2018, asked respondents to rate the ethical<br>standards of top Trump administration officials.<br>• According to the text, 58% of adults rated the ethical standards of top Trump administration<br>officials as “not good” or “poor.”<br>(2) Specific Breakdown:<br>• Poor: The text does not provide a specific percentage for those who rated the ethical<br>standards as “poor” alone.<br>• However, it does state that 36% of adults rated the ethical standards as “poor.”<br>(3) Conclusion:<br>• The question asks for the number of adults who rated Trump’s government ethical stan-<br>dards as “poor.”<br>• Based on the information provided, 36% of adults rated the ethical standards as “poor.”<br>Final Answer: 36%<br>VRAG [page] ColPali<br>Response:<br>To answer the question, let’s analyze the relevant information from the provided images:<br>(1) The survey was conducted from April 25 to May 1, 2018, as stated in the text.<br>(2) The relevant chart is the one titled “Most rate top Trump administration officials’ ethics as<br>‘not good’ or ‘poor”’ (Image 2).<br>(3) This chart shows the percentage of adults who rated the ethical standards of top Trump<br>administration officials as “Poor,” “Not good,” “Good,” or “Excellent.”<br>(4) The “Poor” category is represented by the dark blue bar, which shows 58% of the total respon-<br>dents.<br>Final Answer: 58% of adults rated Trump’s government ethical standards as poor.<br>MLDocRAG  ✓<br>Response:<br>To answer the question, let’s break down the information provided:<br>(1) The survey was conducted from April 25 to May 1, 2018.<br>(2) The question asks about the number of adults who rated Trump’s government ethical standards<br>as “poor.”<br>(3) From the text, we know:<br>• About four-in-ten Americans rate the ethical standards of Trump administration officials<br>as excellent (9%) or good (30%).<br>• 58% say they are not good (21%) or poor (36%).<br>(4) Specifically, the text states: “58% say they are not good (21%) or poor (36%).” This means 36% of<br>the adults rated Trump’s government ethical standards as “poor.”<br>(5) The survey sample size is 1,503 adults.<br>(6) To find the number of adults who rated Trump’s government ethical standards as “poor,” we<br>calculate:<br>• 36% of 1,503 = 0.36 × 1,503 = 541.08<br>Since we can’t have a fraction of a person, we round to the nearest whole number.<br>Final Answer: 541<br>**----- End of picture text -----**<br>


Section 5.5 presents a QA example on MMLongBench-Doc with detailed result comparisons; in this section, we further supplement the example with two evidence page images. 

## **QA Example 2 on MMLongBench-Doc** 

Souce evidence images: 

**Question:** Which step in Figure 1 maps to the content of Figure 10? **Evidence Pages:** [3, 14] **Ground Truth:** Deletion/duplication/rearrangement of the genetic material and Genetic diseases. 

