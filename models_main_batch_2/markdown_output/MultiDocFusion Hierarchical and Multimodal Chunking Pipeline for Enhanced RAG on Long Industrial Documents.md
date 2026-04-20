# **MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents** 

**Joongmin Shin[1] , Chanjun Park[3] , Jeongbae Park[1] , Jaehyung Seo[1,2]** _[‡]_ **, Heuiseok Lim[1,2]** _[‡]_ 

1Human-inspired AI Research, Korea University 

2Department of Computer Science and Engineering, Korea University 

3School of Software, Soongsil University 

{tlswndals13, insmile, seojae777, limhseok}@korea.ac.kr bcj1210nlp@ssu.ac.kr 

## **Abstract** 

RAG-based QA has emerged as a powerful method for processing long industrial documents. However, conventional text chunking approaches often neglect the complex structures of long industrial documents, causing information loss and reduced answer quality. To address this, we introduce **MultiDocFusion** , a multimodal chunking pipeline that integrates: (i) detection of document regions using visionbased document parsing, (ii) text extraction from these regions via OCR, (iii) reconstruction of document structure into a hierarchical tree using large language model (LLM)based document section hierarchical parsing (DSHP-LLM), and (iv) construction of hierarchical chunks through DFS-based Grouping. Extensive experiments across industrial benchmarks demonstrate that **MultiDocFusion** improves retrieval precision by 8–15% and ANLS QA scores by 2–3% compared to baselines, emphasizing the critical role of explicitly leveraging document hierarchy for multimodal document-based QA. These significant performance gains underscore the necessity of structure-aware chunking in enhancing the fidelity of RAG-based QA systems. 

## **1 Introduction** 

The emergence of retrieval-augmented generation (RAG) has significantly advanced the capabilities of large language models (LLMs) in handling long and information-dense documents (Lewis et al., 2021; Jeong, 2023; Ge et al., 2023). Central to the success of RAG pipelines is the document chunking strategy, which segments source documents into manageable and semantically coherent units. Despite its importance, existing chunking methods remain predominantly text-centric, relying on fixed-length splits or shallow semantic cues, and fail to account for the rich visual and structural at- 

tributes inherent in real-world documents (Gong et al., 2020; Gao et al., 2024). 

This limitation becomes especially problematic in industrial and academic domains where documents often take the form of scanned images, multi-page PDFs, or reports with intricate visual and hierarchical layouts. For instance, visual elements such as tables, figures, and section headers may span multiple pages, while hierarchical section structures encode critical semantic relationships that are lost under naive chunking. Optical Character Recognition (OCR) artifacts further exacerbate this issue by introducing noise and misalignments in the extracted text, thereby degrading both retrieval and QA performance (Tito et al., 2023; Hong et al., 2024). As a result, general RAG systems frequently fail to preserve the documents’ semantic continuity, leading to information fragmentation and suboptimal generation quality. 

While recent advances in vision-based document parsing (DP) and OCR techniques enable the extraction of visually coherent regions such as tables and text blocks (Dosovitskiy et al., 2021; Pfitzmann et al., 2022), these approaches lack an explicit representation of logical structure, particularly the parent-child relationships embedded in hierarchical sectioning (Xing et al., 2024). This structural gap limits their effectiveness in tasks that depend on accurate context reconstruction and long-range reasoning. 

To bridge this gap, we introduce **MultiDocFusion** , a multimodal chunking pipeline that explicitly incorporates both visual layout and a document’s structural hierarchy into the chunking process. Our framework integrates four key components: (i) detection of document regions and layout structure using vision-based DP, (ii) text extraction from these regions via OCR, (iii) section hierarchical parsing with large language models (DSHPLLM), and (iv) depth-first search (DFS)-based chunk assembly. By reconstructing a document’s 

_‡_ Co-corresponding authors 

semantic hierarchy and aligning it with visual segmentation, **MultiDocFusion** produces structurally faithful and semantically coherent chunks that are better suited for downstream RAG-based QA. 

We evaluate our approach across diverse document types, such as financial statements, scientific reports, scanned forms, and visually intricate multi-page documents, consistently demonstrating improvements in both retrieval precision and answer accuracy. Our results highlight that explicitly modeling the document’s hierarchical structure is essential for robust and context-aware question answering. Our main contributions are summarized as follows: 

- **MultiDocFusion** : A novel pipeline that systematically integrates DP, OCR, DSHP-LLM, and DFS-based Grouping, effectively handling the structural complexities unique to industrial documents that conventional approaches typically overlook. 

- **DSHP-LLM** : We introduce DSHP-LLM, an instruction-tuned LLM that robustly reconstructs hierarchical section structures from diverse and complex documents, enabling precise context preservation for downstream retrieval and QA. 

- **Comprehensive Experiments** : We conduct extensive validation across various industrial and academic domains, including financial reports, technical documents, scanned images, and documents with complex layouts, and demonstrate consistent improvements in both retrieval and QA performance (retrieval precision by 8–15% and ANLS QA scores by 2–3%). 

## **2 Related Work** 

**Chunking for QA on Long Industrial Documents** Chunking has emerged as an essential strategy for effectively handling long, multi-page documents (Gao et al., 2024). Traditionally, documents have been segmented using Length chunking (Gong et al., 2020) or Semantic chunking (Qu et al., 2025). However, these methods often fail to adequately reflect hierarchical relationships among sections or incorporate visual layout elements such as tables and figures. Recent approaches leveraging LLMs, such as LumberChunker (Duarte et al., 2024) and Perplexity chunking (Zhao et al., 2024), still suffer from contextual 

fragmentation because they lack explicit modeling of document hierarchies (Hong et al., 2024). StyleDFS, which constructs hierarchical trees using font size and style, struggles with scanned documents lacking text layers or irregular layouts (Hong et al., 2024). While end-to-end multimodal models combining textual and visual information have been proposed to mitigate these limitations (Hu et al., 2024; Wang et al., 2024a; Fujitake, 2024), most methods face challenges due to limited context lengths, making it difficult to process entire multi-page documents at a time. Consequently, there is a growing need for chunking methods that comprehensively capture document structure and context (Saad-Falcon et al., 2023; Kang et al., 2024). 

**Document Parsing and Hierarchical Parsing** Recent studies in Visual Question Answering (VQA) have increasingly focused on document parsing (DP), aiming to segment PDFs and imagebased documents into visual components such as tables, figures, and text blocks (Dosovitskiy et al., 2021; Pfitzmann et al., 2022). However, these object-detection-centric approaches inherently lack the capability to fully reconstruct semantic hierarchical relationships, such as the relationship between sections "1.2" and "1.2.1" (Rausch et al., 2023; Wang et al., 2024a). Document hierarchical parsing (DHP) methods have been proposed to address these issues (Rausch et al., 2021, 2023; Zhang et al., 2024c), but their applicability remains limited to structured templates or certain document types, struggling with scanned or irregularly formatted documents (Wang et al., 2024b; Xing et al., 2024). However, LLMs have emerged as promising candidates for DHP tasks due to their advanced text understanding and long-context handling capabilities (Fujitake, 2024). LLMs still face challenges in inferring hierarchical semantic connections between sections, necessitating additional fine-tuning or specialized instructiontuning methods (Zhang et al., 2024b; Wang et al., 2024a; Tabatabaei et al., 2025). While existing approaches may capture either visual layout or textual content, they fail to unify structural and semantic hierarchies. The **MultiDocFusion** pipeline addresses this gap by systematically combining visual region detection, OCR, and LLM-based hierarchical parsing to enable accurate, context-aware chunking of long industrial documents. 

Figure 1: The pipeline for **MultiDocFusion** . The figure illustrates the step-by-step process for handling a long industrial document. (a) DP extracts layout structures; (b) OCR recognizes and annotates text; (c) DSHP-LLM constructs a hierarchical tree from identified section headers and general nodes; (d) DFS-based Grouping constructs coherent hierarchical chunks for retrieval tasks. The color-coded blocks represent document elements: yellow for Root and Title, red for Section Headers, and green for general nodes (tables, figures, and text blocks). 

## **3 MultiDocFusion** 

**MultiDocFusion** ( _Multimodal Document Structure Fusion_ ) is a pipeline designed to effectively integrate visual layouts and hierarchical semantic structures of long industrial documents, enhancing chunking and retrieval performance. The term "MultiDoc" emphasizes the pipeline’s capability to handle diverse document formats frequently encountered in industrial settings such as PDFs, scanned images, and documents with complex layouts, and to support corpus-level multidocument RAG scenarios, enabling retrievalaugmented generation across large collections of documents. Meanwhile, "Fusion" highlights the integration of visual information, textual content, and hierarchical document analysis to produce refined and contextually accurate chunks. The pipeline consists of four stages: (a) DP (Document Parsing), (b) OCR (Optical Character Recognition), (c) DSHP-LLM (Document Section Hierarchical Parsing with LLM), and (d) DFS-based Grouping. Figure 1 provides an overview of this four-stage process and the resulting hierarchical chunks. Below, we detail each stage using the components and terminology shown in the figure. 

## **3.1 DP (Document Parsing)** 

As shown in (a), DP examines each page of a long industrial document to identify and extract its _Layout Structure_ . 

**Process** Advanced vision models detect Titles, Section Headers, text blocks, tables, figures, etc. Each detected segment is assigned bounding-box coordinates and segment type. The pipeline constructs a page-by-page _Layout Structure_ that captures the spatial arrangement of all segments. 

**Output** For each page, DP generates metadata including page numbers, segment IDs, segment types, and bounding box coordinates. This _Layout Structure_ is passed to the OCR stage. 

## **3.2 OCR (Optical Character Recognition)** 

As described in (b), OCR processes the _Layout Structure_ from DP to extract text from each bounding box, resulting in an _Annotated Layout_ . 

**Input** The page-by-page _Layout Structure_ with bounding boxes and segment information from DP. 

**Process** Each segment image is sent to OCR engines tailored to the document’s languages and fonts. The recognized text is then linked back to the corresponding bounding box. 

**Output** The _Annotated Layout_ merges bounding boxes, segment types, and recognized text into structured metadata, preparing the necessary inputs for the subsequent processing stage. 

## **3.3 DSHP-LLM** 

As depicted in (c), DSHP-LLM constructs a _Document Hierarchical Tree_ by identifying, ordering, and attaching section headers along with other nodes based on _Parent–Child_ relationships. 

**Model Setup** DSHP-LLM is built upon an LLM backbone and is instruction-tuned on public datasets of document hierarchies (Zhang et al., 2024a). To improve training efficiency, we employ LoRA-based parameter-efficient fine-tuning (PEFT) (Hu et al., 2021; Han et al., 2024). Hyperparameters and further details are provided in Appendix A.2. 

**Input** The DSHP-LLM receives a _Header List_ , which consists of candidate section headers extracted during the DP and OCR stages from the _Annotated Layout_ . 

**Process** The DSHP-LLM initially performs _Header Tree_ construction by analyzing the _Header List_ and assigning each header a unique identifier and parent reference (e.g., ID:3 Parent:2), resulting in an initial hierarchical structure (e.g., Root → Title → Section 1 → Section 1.1). Next, it proceeds to link general nodes, utilizing the _All Segments list_ maintained from the DP stage. This list includes tables, figures, text blocks, and other document elements sorted by spatial coordinates, such as page number and bounding box position. As the DSHP-LLM traverses the Header Tree, it sequentially scans through the _All Segments list_ . General segments encountered before reaching the next header from the _Header List_ are attached as child nodes of the current header node. This ensures accurate grouping of tables, figures, and text blocks, preserving both logical and spatial document structures. 

**Output** The output is a fully _Document Hierarchical Tree_ explicitly detailing the hierarchical placement of section headers and associated general nodes (e.g., Root → Title → Section 

1 → Section 1.1 → Text). By integrating LLM-identified headers with spatially sorted child nodes, the pipeline maintains coherent logical and visual relationships. For example prompts and outputs, refer to Table 10. 

## **3.4 DFS-based Grouping** 

As illustrated in (d), DFS-based Grouping performs a depth-first traversal of the _Document Hierarchical Tree_ to construct coherent _Hierarchical Chunks_ (e.g., Chunk1, Chunk2, Chunk3, ...). During this stage, the hierarchical structure is explicitly reflected within each chunk using Markdown headers, where each chunk’s depth corresponds directly to the heading level. Detailed algorithms are provided in Appendix A.5. 

**Input** The _Document Hierarchical Tree_ from DSHP-LLM and text corresponding to each node in the tree. 

**Process** In this process, a virtual node called FAKE_ROOT is created, which points directly to the actual root node. The algorithm performs a recursive traversal of the nodes following a depthfirst approach, aggregating the text content from parent nodes along with their child nodes to preserve the contextual information. When the aggregated text length surpasses a predefined threshold (max_len), the algorithm splits the chunk at that specific point. 

**Output** A list of _Hierarchical Chunks_ that encapsulate entire sections or sub-sections, thereby minimizing token waste. The resulting chunks explicitly represent the document’s hierarchical structure via Markdown headers corresponding to each node’s depth. For example, if “1” is a parent of “1.1,” both might be combined into "Chunk4" to preserve continuity in retrieval/QA tasks. An illustrative example is shown below: 

# Document Title ## Section 1 {name} ### Section 1.1 {name} Section 1.1 {Text Content...} 

By combining _Layout Structure_ (from DP) with recognized text (from OCR) into an _Annotated Layout_ , and then applying the DSHP-LLM model to build a _Header Tree_ , **MultiDocFusion** captures both spatial and semantic relationships in long industrial documents. The final DFS-based Grouping stage yields _Hierarchical Chunks_ that main- 

tain these relationships, clearly marked by Markdown headers, outperforming traditional text-only chunking in real-world retrieval and QA scenarios. 

## **4 Experimental Settings** 

This section briefly describes the experimental settings for training and evaluating the DSHP-LLM model and the RAG-based VQA system for multipage documents. We utilize various datasets and model configurations, with additional details (e.g., dataset statistics, hyperparameters, and model setups) provided in the Appendix A. 

**Datasets** For DSHP-LLM training and testing, we combine documents from DocHieNet (Xing et al., 2024) and HRDH (Ma et al., 2023). These datasets include diverse domains and complex layouts, making them suitable for evaluating the generalization of hierarchical parsing models. For multi-page RAG-based VQA performance evaluation, we use four datasets: DUDE (Landeghem et al., 2023), MPVQA (Tito et al., 2023), CUAD (Hendrycks et al., 2021), and MOAMOB (Hong et al., 2024). These datasets encompass financial reports, contracts, scanned documents, and various structures, allowing comprehensive evaluation of chunking and retrieval performance. For each dataset, we index all test documents jointly and retrieve top- _k_ chunks from the entire corpus (not restricted to a gold document). Unless stated otherwise, _k_ = 4. This corpus-level setup reflects realistic deployment and stresses cross-document disambiguation. 

**Models** DP is performed with object detection models such as DETR (Carion et al., 2020) and VGT (Da et al., 2023), while OCR text extraction uses Tesseract (Smith, 2007), EasyOCR (Vedhaviyassh et al., 2022), and TrOCR (Li et al., 2022). The DSHP-LLM, which infers hierarchical parent-child relationships among document section headers, is trained via instruction tuning on LLMs such as Llama-3.2-3B (Grattafiori et al., 2024), Qwen-2.5-3B (Yang et al., 2024), and Mistral-8B (AI, 2024) to predict JSON-structured hierarchies. In the retrieval stage, chunk embeddings are generated using BGE (Chen et al., 2024), E5 (Wang et al., 2024c), and BM25 (Robertson and Zaragoza, 2009). Top- _k_ retrieved chunks are then fed into LLMs (e.g., Llama-based models) for final answer generation. 

||**Model**|**DocHieNet**<br>**F1**<br>**TEDS**|**HRDH**<br>**F1**<br>**TEDS**|
|---|---|---|---|
||**GPT-4**|0.5139<br>0.6961|0.2594<br>0.3342<br>0.4389<br>0.4904<br>**0.8664**<br>**0.8459**<br>0.3299<br>0.3734<br>**0.8856**<br>**0.8658**<br>0.3445<br>0.3974<br>**0.9321**<br>**0.9199**<br>0.2962<br>0.3807<br>**0.6330**<br>**0.6381**|
||Llama-3.2-3B<br>_�→_**DSHP-LLM**<br>Qwen-2.5-3B<br>_�→_**DSHP-LLM**<br>Mistral-8B<br>_�→_**DSHP-LLM**<br>Qwen-2.5-7B<br>_�→_**DSHP-LLM**|0.2558<br>0.5464<br>**0.4894**<br>**0.7549**<br>0.4122<br>0.6995<br>**0.4808**<br>0.6957<br>0.3907<br>0.6559<br>**0.6291**<br>**0.8230**<br>0.5230<br>0.7356<br>**0.5565**<br>**0.8104**||



Table 1: Performance on DHP datasets (DocHieNet + HRDH) for DSHP-LLM (section headers). _�→_ : DSHPLLM applied. **Bold** : improvement over baseline. 

**Evaluation** We evaluate the DSHP-LLM performance using accuracy, F1, and TEDS (Zhong et al., 2020) metrics. Retrieval quality is measured using Precision, Recall, and nDCG (Järvelin and Kekäläinen, 2002), while generated VQA answers are quantitatively assessed via ANLS (Biten et al., 2019), ROUGE-L (Lin, 2004), and METEOR (Banerjee and Lavie, 2005). 

## **5 Experimental Results** 

In this section, we comprehensively evaluate the performance of the proposed **MultiDocFusion** pipeline using the experimental setup. The evaluation consists of: (1) DSHP-LLM performance comparison across different fine-tuned LLMs, (2) retrieval performance comparison among different chunking methods, (3) QA performance analysis, and (4) retrieval robustness analysis under various DP, OCR, and embedding model combinations. To provide objective comparative benchmarks, we include several baseline chunking methodologies, such as Length chunking (Gong et al., 2020), Semantic Chunking (Qu et al., 2025), LumberChunker (Duarte et al., 2024), Perplexity chunking (Zhao et al., 2024), and Structure-based Chunking (W/O DSHP-LLM). Detailed chunking methods are explained in Appendix A.3. 

## **5.1 DSHP-LLM Performance** 

Table 1 summarizes the performance results of section hierarchy parsing on the DocHieNet and HRDH datasets. Each dataset has distinct characteristics: DocHieNet comprises documents from diverse domains, including reports, academic papers, and industrial documents, with complex 

|**Chunking Method**|**DUDE**<br>Recall Precision nDCG|**MPVQA**<br> Recall Precision nDCG|**CUAD**<br> Recall Precision nDCG|**MOAMOB**<br> Recall Precision nDCG|
|---|---|---|---|---|
|Length chunking<br>0.2628<br>0.1686<br>0.2166 0.2523<br>0.1587<br>0.1933 0.9011<br>0.8537<br>0.8776 0.6462<br>0.5676<br>0.6209<br>Semantic chunking<br>0.0956<br>0.0549<br>0.0775 0.0939<br>0.0524<br>0.0680 0.7684<br>0.6719<br>0.7181 0.2737<br>0.1950<br>0.2453<br>LumberChunker<br>0.2395<br>0.1533<br>0.1986 0.2152<br>0.1298<br>0.1609 **0.9031**<br>0.8576<br>0.8800 0.6130<br>0.5205<br>0.5692<br>Perplexity chunking<br>0.2428<br>0.1559<br>0.2020 0.2159<br>0.1318<br>0.1629 0.8869<br>0.8395<br>0.8603 0.6173<br>0.5241<br>0.5785<br>Structure-based chunking 0.2219<br>0.1450<br>0.1862 0.2036<br>0.1230<br>0.1524 0.8844<br>0.8311<br>0.8581 0.5544<br>0.4662<br>0.5149<br>**MultiDocFusion**<br>**0.2927**<br>**0.2001**<br>**0.2505 0.2705**<br>**0.1759**<br>**0.2131** 0.9021<br>**0.8651**<br>**0.8819 0.6758**<br>**0.6184**<br>**0.6554**|||||



Table 2: Retrieval performance by Chunking Method (Average Recall, Precision, nDCG for top- _k_ = 1 _∼_ 4), Best scores are in **bold** . 

scanned images, while HRDH focuses on academic papers characterized by intricate layouts. The experimental results show that GPT-4, used without any fine-tuning, demonstrated limited performance on both DocHieNet (TEDS 0.6961) and HRDH (TEDS 0.3342), indicating similar deficiencies across other general-purpose LLMs. This suggests that general pre-training alone is insufficient for effective section hierarchy parsing. Conversely, applying our proposed DSHP-LLM approach significantly improved performance (measured by TEDS) across both datasets, with varying degrees of improvement depending on model and dataset characteristics. Specifically, for the diverse domains and layout complexities in DocHieNet, Mistral-8B +16.71% and Llama-3.2-3B +20.85% showed substantial improvements. For HRDH, characterized by complex yet relatively regular academic document structures, Mistral-8B +52.25% and Qwen-2.5-3B +49.24% achieved the most significant enhancements. These results clearly indicate that general-purpose LLMs have inherent limitations when performing section hierarchy parsing tasks, underscoring the necessity for dataset-specific fine-tuning. Furthermore, the results emphasize the importance of selecting appropriate models and training strategies tailored to the unique characteristics of each dataset. Based on these findings, we selected the fine-tuned Mistral8B model as the backbone of our DSHP-LLM for integration into the **MultiDocFusion** chunking pipeline, and subsequently evaluated its performance in various multi-page VQA scenarios against other chunking methods. 

## **5.2 Retrieval Performance in Different Chunking Methods** 

Table 2 presents the average Recall, Precision, and nDCG values for top- _k_ = 1 _∼_ 4 retrieval re- 

sults, comparing various chunking methods across four multi-page VQA datasets: DUDE, MPVQA, CUAD, and MOAMOB. 

**MultiDocFusion** consistently achieved the best overall retrieval performance across most datasets. In particular, **MultiDocFusion** demonstrated significant advantages in retrieval accuracy on DUDE (Recall 0.2927, Precision 0.2001, nDCG 0.2505) and MPVQA (Recall 0.2705, Precision 0.1759, nDCG 0.2131), clearly outperforming other methods. These results underscore **MultiDocFusion** ’s effectiveness even under challenging conditions such as the diverse domains and complex document structures inherent in the DUDE dataset, and the varied layouts characteristic of the MPVQA dataset. On CUAD, while LumberChunker attained the highest Recall (0.9031), **MultiDocFusion** showed superior Precision (0.8651) and nDCG (0.8819), confirming its capability for precise retrieval in specialized legal documents. Moreover, in MOAMOB, an extreme scenario characterized by highly intricate document structures and challenging questions within a specialized nuclear domain, **MultiDocFusion** (Recall 0.6758, Precision 0.6184, nDCG 0.6554) markedly outperformed other approaches across all evaluation metrics, demonstrating robust and superior performance even with a limited dataset. 

Additionally, compared to methods solely dependent on LLM-based chunking (e.g., LumberChunker, Perplexity chunking), **MultiDocFusion** significantly enhanced retrieval performance by explicitly capturing and utilizing the hierarchical structure of documents. Specifically, in datasets with complex document structures such as DUDE and MPVQA, simple LLM-based chunking methods failed to sufficiently incorporate structural relationships or context between sections, thus limiting retrieval performance. Conversely, **MultiDoc-** 

|**Chunking Method**|DUDE<br>ANLS ROUGE-L METEOR|MPVQA<br> ANLS ROUGE-L METEOR|CUAD<br> ANLS ROUGE-L METEOR|MOAMOB<br> ANLS ROUGE-L METEOR|
|---|---|---|---|---|
|Length chunking<br>Semantic chunking<br>LumberChunker<br>Perplexity chunking<br>Structure-based chunking <br>**MultiDocFusion**|0.1611<br>0.1444<br>0.1988<br>0.1548<br>0.1261<br>0.1657<br>0.1531<br>0.1284<br>0.1752<br>0.1653<br>0.1390<br>0.1855<br> 0.1751<br>0.1489<br>0.1921<br>**0.1859**<br>**0.1692**<br>**0.2285**|0.1398<br>0.0966<br>0.1408<br>0.1332<br>0.0805<br>0.0978<br>0.1307<br>0.0769<br>0.0993<br>0.1344<br>0.0751<br>0.0950<br>0.1537<br>0.0980<br>0.1278<br>**0.1615**<br>**0.1316**<br>**0.1850**|0.2585<br>0.1677<br>**0.1662**<br>0.2593<br>0.1491<br>0.1468<br>0.2657<br>0.1630<br>0.1650<br>0.2641<br>0.1646<br>0.1524<br>0.2498<br>0.1556<br>0.1591<br>**0.2738**<br>**0.1762**<br>0.1650|0.2497<br>0.0823<br>0.1115<br>0.2455<br>0.0846<br>0.1043<br>0.2536<br>0.0848<br>0.1167<br>0.2532<br>0.0894<br>0.1190<br>0.2501<br>**0.0979**<br>0.1114<br>**0.2596**<br>0.0916<br>**0.1257**|



Table 3: Average QA performance (ANLS, ROUGE-L, METEOR) of six chunking strategies on DUDE, MPVQA, CUAD, and MOAMOB datasets, for top- _k ∈{_ 1 _,_ 4 _}_ . Results are averaged over Llama-3.2-3B, Mistral-8B, and Qwen-2.5-7B models. Best scores are in **bold** . 

**Fusion** effectively captured hierarchical and semantic relationships among sections, significantly improving chunking quality and retrieval performance. 

Furthermore, in comparison to Structure-based Chunking, **MultiDocFusion** continuously demonstrated superior performance by incorporating DSHP-LLM, enhancing Recall by 7.08% and Precision by 5.51% on the DUDE dataset. This clearly indicates that explicitly recognizing hierarchical structures and semantic contexts of sections provides more robust and accurate retrieval performance than approaches based solely on physical document structures. Overall, these results affirm the efficacy and practical utility of **MultiDocFusion** ’s chunking strategy across various document types (e.g., financial reports, legal contracts, multipage documents) within multi-page VQA scenarios. 

## **5.3 Impact on QA Performance in Chunking Methods** 

Table 3 presents a comparative analysis of QA performance (ANLS, ROUGE-L, METEOR) across four multi-page VQA datasets, DUDE, MPVQA, CUAD, and MOAMOB, using various chunking methods. 

**MultiDocFusion** consistently achieved the best QA performance across most datasets. Particularly, it demonstrated significant advantages on MPVQA (ANLS 0.1615, ROUGE-L 0.1316, METEOR 0.1850) and DUDE (ANLS 0.1859, ROUGE-L 0.1692, METEOR 0.2285), clearly outperforming other chunking approaches. These results indicate **MultiDocFusion** ’s robustness and effectiveness even under challenging conditions such as the diverse document layouts in MPVQA and the broad range of domains and complex document types characteristic of DUDE. On the 

CUAD dataset, **MultiDocFusion** achieved the highest ANLS (0.2738) and ROUGE-L (0.1762), though its METEOR score was slightly lower than that of Length chunking (0.1662). This outcome highlights the positive impact of hierarchical structure information in enhancing the coherence and consistency of QA responses. Furthermore, **MultiDocFusion** also recorded the highest scores on MOAMOB in terms of ANLS (0.2596) and METEOR (0.1257), confirming its capability to effectively improve QA quality even under limited and highly complex document scenarios. 

Compared to existing LLM-based chunking methods (LumberChunker, Perplexity chunking) as well as simple Length chunking and Semantic chunking (Length chunking, Semantic chunking), **MultiDocFusion** significantly improved retrieval precision by more comprehensively capturing the structural context of documents, thereby enhancing both the accuracy and consistency of QA responses. Particularly notable is that compared to Structure-based Chunking, the additional integration of DSHP-LLM within **MultiDocFusion** substantially elevated RAG-based QA performance. Overall, these findings confirm that **MultiDocFusion** , by explicitly utilizing hierarchical document structures, consistently provides superior QA performance across diverse document scenarios. 

## **5.4 Robustness to Pipeline Components (DP/OCR/Embeddings)** 

This section provides an in-depth analysis of the robustness of retrieval performance across different chunking methods when varying DP, OCR, and embedding models. All results are compared based on the average nDCG for top- _k_ = 1 _∼_ 4 retrieval outcomes. 

**(1) Comparison across DP Models** Table 4 shows the average nDCG performance of differ- 

|**Chunking Method**|DETR|DiT|VGT|Avg|
|---|---|---|---|---|
|Length chunking|0.4952|0.4863|0.4497|0.4771|
|Semantic chunking|0.3247|0.3263|0.2297|0.2936|
|LumberChunker|0.4620|0.4533|0.4412|0.4522|
|Perplexity chunking|0.4636|0.4460|0.4436|0.4511|
|Structure-based chunking|0.4396|0.4171|0.4269|0.4279|
|**MultiDocFusion**|**0.5014**|**0.4976**|**0.5061**|**0.5017**|



Table 4: Average performance of Chunking Methods by DP model (top- _k_ = 1 _∼_ 4 nDCG). 

|**Chunking Method**|EasyOCR|Tesseract|TrOCR|Avg|
|---|---|---|---|---|
|Length chunking|0.5369|0.4799|**0.4144**|0.4771|
|Semantic chunking|0.3213|0.2757|0.2423|0.2798|
|LumberChunker|0.5057|0.4546|0.3963|0.4522|
|Perplexity chunking|0.5115|0.4674|0.3739|0.4509|
|Structure-based chunking|0.5194|0.4650|0.2993|0.4279|
|**MultiDocFusion**|**0.5681**|**0.5068**|0.4097|**0.4949**|



Table 5: Average performance of Chunking Methods by OCR model (top- _k_ = 1 _∼_ 4 nDCG). 

ent chunking methods across three DP model environments: DETR, DiT, and VGT. Overall, **MultiDocFusion** achieved the highest average performance (0.5017) and consistently delivered superior results across all individual DP models. Particularly noteworthy is its substantial improvement of up to +27.64% over the lowest-performing Semantic chunking method in the VGT environment. This clearly demonstrates that **MultiDocFusion** , by explicitly incorporating hierarchical document structures, consistently maintains robust and superior performance regardless of variations in DP models. 

**(2) Comparison across OCR Models** Table 5 compares the average nDCG scores of various chunking methods across different OCR models, namely EasyOCR, Tesseract, and TrOCR. **MultiDocFusion** consistently achieved the highest performance (avg 0.4949) and notably outperformed other methods, particularly in EasyOCR (avg 0.5681) and Tesseract (avg 0.5068) settings. Even with TrOCR, where the overall performance was lower, **MultiDocFusion** maintained a relatively high score (avg 0.4097), demonstrating that hierarchical structure-based chunking remains robust and provides stable retrieval performance despite variations in OCR quality. 

**(3) Comparison across Embedding Models** Table 6 shows the average nDCG performance across various embedding models (BGE, E5, and BM25) for each chunking method. On average, **MultiDocFusion** achieved the highest overall 

|**Chunking Method**|BGE|E5|BM25|Avg|
|---|---|---|---|---|
|Length chunking|0.4834|0.4715|0.4764|0.4771|
|Semantic chunking|0.3114|0.3378|0.1825|0.2772|
|LumberChunker|0.4708|0.4319|0.4539|0.4522|
|Perplexity chunking|0.4715|0.4318|0.4495|0.4509|
|Structure-based chunking|0.4679|0.4040|0.4118|0.4279|
|**MultiDocFusion**|**0.5213**|**0.4884**|**0.5085**|**0.5061**|



Table 6: Average performance of Chunking Methods by embedding model (top- _k_ = 1 _∼_ 4 nDCG). 

performance (0.5061), consistently outperforming other chunking methods across all embedding environments. In particular, **MultiDocFusion** recorded the best performance (0.5213) in the BGE embedding environment. These results demonstrate that chunking methods leveraging hierarchical document structure are robust and effective in enhancing retrieval accuracy, irrespective of the embedding model utilized. 

## **6 Conclusion** 

This work targets a core bottleneck in RAG over long industrial documents: context fragmentation caused by text-only chunking that ignores visual layout and explicit section hierarchy. We formalize this problem and introduce **MultiDocFusion** , a structured multimodal pipeline that (i) parses page-level layout regions (DP), (ii) extracts text with OCR, (iii) reconstructs an explicit section hierarchy with DSHP-LLM, and (iv) assembles hierarchical chunks via DFS to preserve both spatial and semantic context. 

Evaluated under a corpus-level setting on four multi-page VQA benchmarks, MultiDocFusion consistently outperforms baseline chunking methods in retrieval and QA. DSHP-LLM, fine-tuned for hierarchical parsing, accurately reconstructs complex section structures and surpasses general-purpose LLMs (e.g., GPT-4) on DHP datasets. The gains hold across diverse domains and layouts and remain stable under different DP, OCR, and embedding choices, underscoring the pipeline’s practical reliability. 

Taken together, these results support a clear conclusion: _Hierarchy-aware, visually grounded chunking should be a first-class design principle for RAG on long, complex, and often scanned industrial documents_ . By aligning visual segmentation with an explicit document tree and reflecting it in chunk boundaries, MultiDocFusion reduces contextual breakage and yields more faithful re- 

trieval and answers. 

## **Limitations** 

Although the **MultiDocFusion** chunking pipeline effectively incorporates document hierarchy to improve retrieval and QA, several limitations remain. 

**Limited visual grounding of DSHP-LLM** Our DSHP-LLM was trained on DHP datasets, which do not provide fine-grained layout signals such as font size/style, color, whitespace, alignment/ruling lines, or column structure. As the model is inherently LLM-centric and primarily conditioned on OCR text with coarse bounding boxes, it underutilizes these visual cues that are often decisive for reliable hierarchy reconstruction in scanned or visually complex pages. Future work should incorporate detailed layout features and, more broadly, visually ground DSHP-LLM via multimodal document encoders or VLM backbones to enable more accurate structural analysis. 

**Graph-structured retrieval not evaluated** Because **MultiDocFusion** induces an explicit hierarchical _document graph_ (headers _→_ subheaders _→_ content blocks) with typed relations (e.g., parent–child, reading order), it can naturally be instantiated as a _GraphRAG_ pipeline that retrieves and reasons over nodes and paths. This formulation is likely to better support multi-hop and other reasoning-intensive tasks. However, we did not systematically validate this direction, as the present work focuses on verifying multimodal hierarchical chunking under standard RAG settings. Future research should rigorously evaluate graphaugmented retrieval and reasoning on benchmarks requiring multi-hop, compositional reasoning, and cross-page evidence aggregation. 

**Error propagation and end-to-end alternatives** While a serial, multi-module pipeline is pragmatic and familiar in industrial settings, such a design is inherently susceptible to error propagation: mistakes in earlier-stage components (DP/OCR) can cascade into DSHP-LLM, DFS-based chunking, retrieval, and ultimately QA. To mitigate this risk, future work should investigate the substitutability of VLM-based end-to-end models as drop-in replacements or hybrid components that jointly optimize visual parsing, hierarchical structuring, chunking, and retrieval/answering. Such end-toend formulations may reduce the accumulation 

of earlier-stage noise and provide stronger crossmodal consistency, albeit with trade-offs in controllability and interpretability that warrant careful study. 

**Computational overhead** Hierarchical chunking duplicates parent context across multiple children to preserve coherence, which can increase index size, retrieval latency, and storage costs. Budget-aware chunking, graph pruning, and nodelevel caching/deduplication are practical mitigations to explore. 

## **Ethical Considerations** 

The primary objective of this research is to enhance multimodal document parsing and questionanswering capabilities; however, ethical considerations must be carefully addressed when applying this technology. First, documents processed by the pipeline may contain sensitive information such as personal data, copyrighted materials, or proprietary business content. Thus, meticulous care must be exercised in data collection, processing, and usage to ensure strict adherence to privacy regulations and data security standards. 

Second, despite aiming to provide accurate information, the proposed system could inadvertently generate incorrect or biased responses, potentially misleading users. When deploying the system in practical settings, clear guidelines for accountability and measures against misuse should be implemented. 

## **Acknowledgments** 

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (RS-2024-00398115, Research on the reliability and coherence of outcomes produced by Generative AI). This research was supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Education(NRF-2021R1A6A1A03045425). This work was supported by the Commercialization Promotion Agency for R&D Outcomes(COMPA) grant funded by the Korea government(Ministry of Science and ICT)(2710086166) 

## **References** 

Mistral AI. 2024. Ministral-8b-instruct-2410. https://huggingface.co/mistralai/ Ministral-8B-Instruct-2410. Accessed: 2024-05-16. 

- Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72. 

- Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marcal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. 2019. Scene text visual question answering. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 4291–4301. 

- Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-end object detection with transformers. In _Computer Vision – ECCV 2020_ , pages 213–229. 

- Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. _Preprint_ , arXiv:2402.03216. 

- Cheng Da, Chuwei Luo, Qi Zheng, and Cong Yao. 2023. Vision grid transformer for document layout analysis. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ . 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, and et al. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In _Proceedings of the 9th International Conference on Learning Representations (ICLR)_ . 

- André V Duarte, João Marques, Miguel Graça, Miguel Freire, Lei Li, and Arlindo L Oliveira. 2024. Lumberchunker: Long-form narrative document segmentation. _arXiv preprint arXiv:2406.17526_ . 

- Masato Fujitake. 2024. LayoutLLM: Large language model instruction tuning for visually rich document understanding. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LRECCOLING 2024)_ , pages 10219–10224, Torino, Italia. ELRA and ICCL. 

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. _Preprint_ , arXiv:2312.10997. 

- J. Ge, Steve Sun, Joseph Owens, Victor Galvez, O. Gologorskaya, Jennifer C Lai, Mark J Pletcher, and 

Ki Lai. 2023. Development of a liver diseasespecific large language model chat interface using retrieval augmented generation. _medRxiv_ . 

- Hongyu Gong, Yelong Shen, Dian Yu, Jianshu Chen, and Dong Yu. 2020. Recurrent chunking mechanisms for long-text machine reading comprehension. pages 6751–6761. 

- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, and Abhishek Kadian. 2024. The llama 3 herd of models. _Preprint_ , arXiv:2407.21783. 

- Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. 2024. Parameter-efficient finetuning for large models: A comprehensive survey. _Preprint_ , arXiv:2403.14608. 

- Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. 2021. Cuad: An expert-annotated nlp dataset for legal contract review. _NeurIPS_ . 

- Seongtae Hong, Joong Min Shin, Jaehyung Seo, Taemin Lee, Jeongbae Park, Cho Man Young, Byeongho Choi, and Heuiseok Lim. 2024. Intelligent predictive maintenance RAG framework for power plants: Enhancing QA with StyleDFS and domain specific instruction tuning. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track_ , pages 805–820, Miami, Florida, US. Association for Computational Linguistics. 

- Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. 2024. mPLUG-DocOwl 1.5: Unified structure learning for OCR-free document understanding. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pages 3096– 3120, Miami, Florida, USA. Association for Computational Linguistics. 

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. _Preprint_ , arXiv:2106.09685. 

- Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of ir techniques. _ACM Transactions on Information Systems (TOIS)_ , 20(4):422–446. 

- CheonSu Jeong. 2023. A study on the implementation of generative ai services using an enterprise databased llm application architecture. _Adv. Artif. Intell. Mach. Learn._ , 3:1588–1618. 

- Lei Kang, Rubèn Tito, Ernest Valveny, and Dimosthenis Karatzas. 2024. Multi-page document visual question answering using self-attention scoring mechanism. In _Document Analysis and Recognition - ICDAR 2024: 18th International Conference, Athens, Greece, August 30–September 4, 2024, Proceedings, Part VI_ , page 219–232, Berlin, Heidelberg. Springer-Verlag. 

- Jordy Van Landeghem, Rafał Powalski, Rubèn Tito, Dawid Jurkiewicz, Matthew Blaschko, Łukasz Borchmann, Mickaël Coustaty, Sien Moens, Michał Pietruszka, Bertrand Ackaert, Tomasz Stanisławek, Paweł Józiak, and Ernest Valveny. 2023. Document understanding dataset and evaluation (dude). In _2023 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 19471–19483. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks. _Preprint_ , arXiv:2005.11401. 

- Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. 2022. Trocr: Transformer-based optical character recognition with pre-trained models. _Preprint_ , arXiv:2109.10282. 

- Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81. 

- Jiefeng Ma, Jun Du, Pengfei Hu, Zhenrong Zhang, Jianshu Zhang, Huihui Zhu, and Cong Liu. 2023. Hrdoc: dataset and baseline method toward hierarchical reconstruction of document structures. In _Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence_ , AAAI’23/IAAI’23/EAAI’23. AAAI Press. 

- Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter Staar. 2022. Doclaynet: A large human-annotated dataset for documentlayout segmentation. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , KDD ’22, page 3743–3751, New York, NY, USA. Association for Computing Machinery. 

- Renyi Qu, Ruixuan Tu, and Forrest Sheng Bao. 2025. Is semantic chunking worth the computational cost? In _Findings of the Association for Computational Linguistics: NAACL 2025_ , pages 2155–2177, Albuquerque, New Mexico. Association for Computational Linguistics. 

- Johannes Rausch, Octavio Martinez, Fabian Bissig, Ce Zhang, and Stefan Feuerriegel. 2021. Docparser: Hierarchical document structure parsing from renderings. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 35:4328–4338. 

- Johannes Rausch, Gentiana Rashiti, Maxim Gusev, Ce Zhang, and Stefan Feuerriegel. 2023. Dsg: An end-to-end document structure generator. 

- Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. _Found. Trends Inf. Retr._ , 3(4):333–389. 

- Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon, Ryan A. Rossi, and Franck Dernoncourt. 2023. Pdftriage: Question answering over long, structured documents. _Preprint_ , arXiv:2309.08872. 

- R. Smith. 2007. An overview of the tesseract ocr engine. In _Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)_ , volume 2, pages 629–633. 

- Seyed Amin Tabatabaei, Sarah Fancher, Michael Parsons, and Arian Askari. 2025. Can large language models serve as effective classifiers for hierarchical multi-label classification of scientific documents at industrial scale? In _Proceedings of the 31st International Conference on Computational Linguistics: Industry Track_ , pages 163–174, Abu Dhabi, UAE. Association for Computational Linguistics. 

- Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multi-page docvqa. _Preprint_ , arXiv:2212.05935. 

- D.R. Vedhaviyassh, R. Sudhan, G. Saranya, M. Safa, and D. Arun. 2022. Comparative analysis of easyocr and tesseractocr for automatic license plate recognition using deep learning algorithm. In _2022 6th International Conference on Electronics, Communication and Aerospace Technology_ , pages 966–971. 

- Prashant Verma. 2025. S2 chunking: A hybrid framework for document segmentation through integrated spatial and semantic analysis. _Preprint_ , arXiv:2501.05485. 

- Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. 2024a. DocLLM: A layout-aware generative language model for multimodal document understanding. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8529–8548, Bangkok, Thailand. Association for Computational Linguistics. 

- Jiawei Wang, Kai Hu, Zhuoyao Zhong, Lei Sun, and Qiang Huo. 2024b. Detect-order-construct: A tree construction based approach for hierarchical document structure analysis. _Pattern Recognition_ , 156:110836. 

- Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024c. Multilingual e5 text embeddings: A technical report. _Preprint_ , arXiv:2402.05672. 

- Hangdi Xing, Changxu Cheng, Feiyu Gao, Zirui Shao, Zhi Yu, Jiajun Bu, Qi Zheng, and Cong Yao. 2024. Dochienet: A large and diverse dataset for document hierarchy parsing. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, and Zhihao Fan. 2024. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ . 

- Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, and Renyu Li. 2024. Financial report chunking for effective retrieval augmented generation. _Preprint_ , arXiv:2402.05131. 

- Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin Wang. 2024a. Instruction tuning for large language models: A survey. _Preprint_ , arXiv:2308.10792. 

- Yizhuo Zhang, Heng Wang, Shangbin Feng, Zhaoxuan Tan, Xiaochuang Han, Tianxing He, and Yulia Tsvetkov. 2024b. Can LLM graph reasoning generalize beyond pattern memorization? In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pages 2289–2305, Miami, Florida, USA. Association for Computational Linguistics. 

- Yue Zhang, Zhihao Zhang, Wenbin Lai, Chong Zhang, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024c. PDF-to-tree: Parsing PDF text blocks into a tree. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pages 10704–10714, Miami, Florida, USA. Association for Computational Linguistics. 

- Jihao Zhao, Zhiyuan Ji, Pengnian Qi, Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu li. 2024. Metachunking: Learning efficient text segmentation via logical perception. 

- Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. 2020. Image-based table recognition: data, model, and evaluation. In _European Conference on Computer Vision (ECCV)_ , pages 564–580. Springer. 

## **A Appendix** 

This appendix complements the main text and provides a concise roadmap for reproduction and inspection of results. 

- **Datasets:** scope, splits, and language coverage (Sec. A.1; Table 7). 

- **Pipeline & Hyperparameters:** DP, OCR, DSHP-LLM, embeddings, and QA LLMs with default settings (Sec. A.2). 

- **Compared Chunkers:** definitions and assumptions (Sec. A.3). 

- **Detailed Results:** retrieval by _k_ , OCR/DP/Embedding ablations, and QA metrics (Tables 12–15). 

- **Server Evaluations:** DUDE/MPVQA ANLS on official test servers (Table 9). 

- **Algorithm & Prompts:** DFS-based chunking algorithm and DSHP-LLM prompts/examples (Sec. A.5; Sec. B.1; Figures 2, 3). 

## **A.1 Dataset Details** 

|**Dataset**|**Type/Domain**|**#Documents**|**Avg. Pages**|**#QA pairs**|
|---|---|---|---|---|
|**DocHieNet**|Mixed (Reports/Papers/Industrial)|1,673|5.3|-|
|**HRDH**|Academic Papers (arXiv)|1,500|7.1|-|
|**MPVQA**|General Documents (Multi-page)|17,000|3.4|48,000+|
|**CUAD**|Legal Documents (Contracts)|510|6.2|13,000+|
|**DUDE**|Mixed (Financial Reports/Manuals)|3,000+|4.9|7,000+|
|**MOAMOB**|Industrial Technical Documents|2|35.5|71|



Table 7: Summary of key datasets used in MultiDocFusion. 

**(A) Datasets for DSHP-LLM Training and Evaluation** DocHieNet and HRDH include annotations of hierarchical section structures (parent-child relationships) within documents, making them suitable for training the DSHP-LLM model. 

- **DocHieNet** (Xing et al., 2024): Comprising 1,673 PDF documents (average 5.3 pages/document), this dataset covers diverse domains, including reports, academic papers, and industrial documents, with many scanned images. Each document is annotated with hierarchical JSON structures (parent-child relationships among titles, paragraphs, tables, figures, etc.), making it suitable for training and evaluating models on diverse structural layouts and domains. 

- **HRDH** (Ma et al., 2023): Consisting of approximately 1,500 PDF academic papers sourced from arXiv (average 7.1 pages/document), HRDH is a carefully selected subset of the HRDoc dataset featuring particularly complex layouts (HRDoc-Hard). It includes more than 30 types of complicated layouts ranging from single-column to specialized templates. Each line is labeled with its corresponding parent section, making it ideal for training and evaluating hierarchical parsing models. 

**(B) Multi-page VQA Datasets** The datasets _MPVQA_ , _CUAD_ , _DUDE_ , and _MOAMOB_ were utilized for practical RAG-based Question Answering (QA) experiments. These datasets include various document formats and layouts, such as indus- 

trial reports, legal contracts, and financial documents. 

- **DUDE** (Landeghem et al., 2023): Over 3,000 documents spanning various domains such as financial reports and user manuals, with more than 7,000 annotated QA pairs. This dataset enables broad semantic understanding and structural evaluation across diverse document types. Because the official serverevaluated test set does not release groundtruth answers, retrieval metrics cannot be computed, and we cannot directly assess how test-set QA gains are driven by retrieval improvements. Accordingly, for our joint retrieval+QA analysis we report results on the validation split in Table 2 and Table 3, while the official test results[1] are provided in Table 9(approximately 700 documents and 1,500 QA pairs). 

- **MPVQA** (Tito et al., 2023): Contains approximately 17,000 documents with more than 48,000 questions (average 2.8 questions per document). As with DUDE, the official server-evaluated test set does not release ground-truth answers, which precludes computing retrieval metrics. Therefore, we use the validation split for the retrieval+QA results reported in Table 2 and Table 3, and include the official test results[2] in Table 9 (around 2,000 documents and 6,000 questions). Its multi-page documents with varied layouts enable assessment of stability and robustness in chunking and retrieval processes. 

- **CUAD** (Hendrycks et al., 2021): Comprises 510 legal contracts annotated with over 13,000 QA pairs, primarily targeting specific contractual clauses and legal details. This study uses only the test set (approximately 50 documents and 1,200 QA pairs), making it suitable for verifying the effectiveness of RAG approaches in specialized domains such as law. 

- **MOAMOB** (Hong et al., 2024): A small dataset containing just two documents with 71 challenging QA pairs. This study utilizes the entire dataset. Despite its limited size, the dataset’s complex document structures 

> 1 https://rrc.cvc.uab.es/?ch=23 

2 https://rrc.cvc.uab.es/?ch=17 

and challenging questions provide a rigorous evaluation under constrained conditions. 

Language Information: DocHieNet consists of English and Chinese documents, MOAMOB contains Korean documents, and all other datasets are in English. 

## **A.2 Model and Implementation Settings** 

In our experiments, we cross-applied multiple models for each pipeline component to verify the robustness of the proposed **MultiDocFusion** pipeline across realistic scenarios with varied performance. 

**(1) Document Parsing (DP) Models** To identify layout components (e.g., tables, figures, text blocks) from PDF or scanned images, we utilized several object detection-based models, including DETR, DiT, and VGT, each fine-tuned on the DocLayNet dataset (Pfitzmann et al., 2022). These models generated page-level segment information (segment ID, segment type, bounding box) used for subsequent steps. 

**(2) OCR Models** For text extraction within identified segments, we employed multiple OCR models such as EasyOCR, Tesseract, and TrOCR. The accuracy varied significantly depending on document quality, font types, and languages. 

**(3) DSHP-LLM (Document Hierarchical Parsing) Models** We fine-tuned various LLMs—Llama-3.2-3B, Qwen-2.5-3B, Mistral8B, and Qwen-2.5-7B—using instruction tuning on hierarchical section structures (represented in JSON) derived from the DocHieNet and HRDH datasets. To enhance parameter efficiency, we combined LoRA (Hu et al., 2021) and 4-bit quantization (QLoRA). 

**(4) Embedding Models** For chunk embedding in the retrieval phase, we compared multiple methods including BGE, E5, and traditional BM25. For _top-k_ retrieval, we used _k_ = 4, selecting the topranked chunks as input context for the LLM to generate the final answers. 

**(5) QA Generation (LLM) Models** Answer generation was performed using various LLMs such as Llama-3.2-3B, Mistral-8B, and Qwen-2.57B. These models produced responses based on the top-ranked chunks retrieved in the previous step, following a RAG-based approach. 

**Hardware and Software Environment** Experiments were conducted on a single NVIDIA A100 40GB GPU, with an Intel Xeon 32-core CPU and 256GB RAM. Model training and inference utilized PyTorch 2.0 and the Transformers library. Embedding inference was batch-processed in a CPU/GPU hybrid environment. 

**Hyperparameters** For DSHP-LLM training, the baseline hyperparameters were set as epochs=5, batch size=16, learning rate=1 _×_ 10 _[−]_[5] , with further tuning via grid search. Retrieval utilized a default _top-k_ of 4, with BM25 parameters _k_ _1=1.2, b=0.75. To maintain experimental consistency and adhere to the embedding model’s context length constraints, the maximum chunk length (max_len) was fixed at 550 tokens, following prior studies (Duarte et al., 2024; Hong et al., 2024; Yepes et al., 2024). 

## **A.3 Detailed Descriptions of Compared Chunking Methods** 

This section provides comprehensive descriptions of the chunking methodologies compared against our proposed **MultiDocFusion** pipeline. 

**Length chunking (Gong et al., 2020)** This method divides documents into chunks based on a fixed token length limit. Each chunk is created uniformly, without considering semantic or structural boundaries. While simple and computationally efficient, it risks splitting important contexts, leading to potential information loss and degraded performance in retrieval and QA tasks. 

**Semantic chunking (Qu et al., 2025)** Semantic chunking leverages encoder-based language models to maintain semantic consistency. Chunks are formed by grouping sentences based on semantic similarity scores derived from language models (e.g., E5 embeddings). Although effective in maintaining semantic coherence, it tends to produce shorter, numerous chunks, potentially impacting retrieval efficiency. Following prior work (Hong et al., 2024), we employed the E5 model for consistency in our experiments. 

**LumberChunker (Duarte et al., 2024)** LumberChunker employs Large Language Models (LLMs) to dynamically partition documents by identifying topical shifts between sentences or paragraphs. It effectively captures the semantic independence of textual segments, resulting in 

chunks of variable sizes optimized for dense retrieval tasks. For experimental consistency across LLM-based methods, we employed the Mistral8B model as the base model. 

**Perplexity chunking (Zhao et al., 2024)** Based on the concept of Meta-Chunking, Perplexity chunking identifies optimal chunk boundaries by analyzing the perplexity distribution of sentences and paragraphs. It dynamically merges or splits textual segments at a fine-grained level, effectively balancing granularity and computational efficiency. To ensure fairness among LLM-based methods, we also used the Mistral-8B model for these experiments. 

**Structure-based Chunking (W/O DSHP-LLM)** This approach partitions documents solely based on their structural layouts, such as section headers, tables, and figures. Similar methodologies have been explored in recent works (Yepes et al., 2024; Verma, 2025). In our experiments, Structure-based Chunking served as a baseline to clearly isolate and demonstrate the impact of the proposed DSHP-LLM. Specifically, chunks were created by ordering structural elements obtained via DP (Document Parsing), without explicitly considering hierarchical parent-child relationships identified by DSHP-LLM. Segment types were included in the resulting chunks. 

_**MultiDocFusion**_ Our proposed multimodal chunking pipeline integrates hierarchical document structure into the chunking process. It utilizes the best-performing DSHP-LLM model (fine-tuned Mistral-8B) identified from our previous experiments to explicitly reconstruct section hierarchies, significantly enhancing the semantic and structural coherence of document chunks and thus improving retrieval and QA outcomes. 

## **A.4 Detailed experimental results** 

## **A.4.1 Chunking Statistics and Examples** 

Table 8 summarizes the chunking statistics for the six evaluated chunking methods. Length chunking consistently generates chunks close to the predefined maximum token length. Semantic chunking tends to produce the shortest and highest number of chunks. LumberChunker and Perplexity methods yield intermediate chunk sizes and counts, whereas Structure-based chunking produces relatively longer chunks by explicitly including segment types. 

|**Method**|**Metric**|**Avg. Length (Characters / Tokens)**|**Number of Chunks**|
|---|---|---|---|
|Length chunking|Characters<br>Tokens|789.25<br>548.30|6,807|
|Semantic chunking|Characters<br>Tokens|289.10<br>201.54|18,498|
|LumberChunker|Characters<br>Tokens|702.45<br>483.82|10,650|
|Perplexity|Characters<br>Tokens|503.12<br>350.90|10,615|
|Structure-based|Characters<br>Tokens|719.50<br>498.30|9,478|
|**MultiDocFusion**|Characters<br>Tokens|766.85<br>521.65|20,773|



Table 8: Chunk statistics (average length and total number) for Length chunking, Semantic chunking, LumberChunker, Perplexity, Structure-based, and **MultiDocFusion** chunking methods (max_len=550 tokens) 

The proposed **MultiDocFusion** generates the highest number of chunks (20,773), each of which tends to approach the maximum token length (averaging 766.85 characters and 521.65 tokens). This increase results from the hierarchical approach where chunks with identical parent headers include duplicated content. Despite generating more chunks, **MultiDocFusion** consistently achieves superior retrieval performance, demonstrating the effectiveness of fine-grained, hierarchical chunking in retrieving relevant context. 

## **A.4.2 Detailed Retrieval and QA Performance Comparisons** 

Tables 2, 12, 13, 14, 15, 16 extend the summarized results presented in the main text, providing comprehensive comparisons across top- _k_ = 1 _∼_ 4, DP models, OCR models, and embedding models. Consistent with the summarized experiments, these detailed tables further confirm that the **MultiDocFusion** pipeline consistently outperforms other chunking methods across diverse datasets and industrial document scenarios, highlighting its robust chunking performance. 

## **A.4.3 Official Test-Server Results for DUDE and MPVQA** 

Official test-server result on Table 9 

## **A.5 DFS-based Grouping Algorithm** 

The DFS-based algorithm traverses the parsed hierarchy_tree in a _Depth-First Search_ manner, accumulating text from parent to child sections. When the accumulated text exceeds max_len, it splits appropriately to create new chunks. This method efficiently maintains the hierarchical document structure while managing the chunk length constraint. 

|**Chunking Method**|**DUDE**|**MPVQA**|
|---|---|---|
|Length chunking|0.1592|0.1348|
|Semantic chunking|0.1537|0.1294|
|LumberChunker|0.1573|0.1351|
|Perplexity chunking|0.1668|0.1299|
|Structure based Chunking|0.1683|0.1488|
|**MultiDocFusion**|**0.1793**|**0.1544**|



Table 9: Official test-server ANLS on DUDE and MPVQA. Ground-truth is hidden on the server, so retrieval metrics cannot be computed; main paper reports corpus-level retrieval+QA on validation splits. 

**Algorithm 1** DFS-based Hierarchical Chunking Algorithm (Conceptual Summary) 

|**Algorithm 1** DFS-based Hierarchical <br>Algorithm(Conceptual Summary)|**Algorithm 1** DFS-based Hierarchical <br>Algorithm(Conceptual Summary)|
|---|---|
|**Require:** hierarchy_tree,max_len||
|1:|**function**DFS_CHUNKING(_node_,_context_)|
|2:|_currentText ←_node.text|
|3:|_temp ←context_+_currentText_|
|4:|**if**length(_temp_)_>_max_len**then**|
|5:|Split_temp_into multiple chunks|
|6:|**else**|
|7:|Append_temp_to chunk list|
|8:|**end if**|
|9:|**for**child_∈_node.children**do**|
|10:|DFS_CHUNKING(child,_temp_)|
|11:|**end for**|
|12:|**end function**|



13: DFS_CHUNKING(root, "") 

## **B Examples** 

## **B.1 Prompt Examples for DSHP-LLM** 

Table 10 provides condensed examples of the **system prompts** , **user inputs** , and **output examples** used to instruct the DSHP-LLM model to infer the hierarchical structure of document headers. In training, hundreds or thousands of header lists paired with corresponding JSON ground truths are employed. 

## **B.2 Results of Document Chunking Using Different Methods** 

Table 11 presents the results of applying each chunking method to the document shown in Figure 2. Conventional text-based chunking approaches (Length, Semantic, LumberChunker, Perplexity) often lack clear segmentation criteria between chunks and frequently fail to maintain contextual continuity. In contrast, our proposed method includes higher-level hierarchical nodes within each chunk, thereby preserving contextual coherence and enabling the generation of wellstructured, hierarchically organized chunks. 

## **System Prompt (Common)** 

You are an expert in analyzing section headers of documents and creating a hierarchical structure. The following is a list of ’section header’ texts extracted from a document. 

For each item, determine its relationship with the parent section (parent-child relationship). 

If possible, follow standard document numbering rules, such as treating ’3.1’ as a child of ’3’ and ’3.1.1’ as a child of ’3.1’. 

Even if there is no numeric pattern, infer hierarchy based on textual context. 

If an item is a top-level heading (i.e., the root node is its parent), set ‘parent‘ to null. 

Output format: 

json only. 

DO NOT include any other explanations or text. 

[ 

{"id": "<id from the original header_list>", "parent": "<id of the parent node or null if root>"} 

] 

|**Example 1 (HRDH)**|**Example 2 (DocHieNet)**|
|---|---|
|_### User:_|_### User:_|
|{ "id": "2", "text": "1. Introduction",|{ "id": "2", "text": "CONGRESSIONAL|
|"page_number": 0, "top": 395, "left":|BUDGET OFFICE COST ESTIMATE",|
|257 },|"page_number": 1, "top": 74, "left":|
|{ "id": "3", "text": "2. The puzzle",|171 },|
|"page_number": 1, "top": 659, "left":|{ "id": "5", "text": "SUMMARY",|
|263 },|"page_number": 1, "top": 282, "left":|
|{ "id": "4", "text": "3. Next section",|71 },|
|...},|{ "id": "23", "text": "MAJOR PROVISIONS|
|{ "id": "5", "text": "4. Another|OF THE LEGISLATION", ...},|
|topic", ...},|{ "id": "37", "text": "ESTIMATED COST|
|{ "id": "6", "text": "4.1 Subtopic of|TO THE FEDERAL GOVERNMENT", ...},|
|4", ...},|{ "id": "39", "text": "BASIS OF|
|{ "id": "7", "text": "5. Additional|ESTIMATE", ...},|
|info", ...},|{ "id": "44", "text": "Use of the March|
|...|2016 Baseline", ...},|
||{ "id": "46", "text": "Methodology for|
||Estimating Effects of ...", ...},|
||...|
|_### Assistant:_|_### Assistant:_|
|{ "id": "2", "parent": null },|{ "id": "2", "parent": null },|
|{ "id": "3", "parent": "2" },|{ "id": "5", "parent": "2" },|
|{ "id": "4", "parent": "3" },|{ "id": "23", "parent": "2" },|
|{ "id": "5", "parent": "4" },|{ "id": "37", "parent": "2" },|
|{ "id": "6", "parent": "5" },|{ "id": "39", "parent": "37" },|
|{ "id": "7", "parent": "5" },|{ "id": "44", "parent": "37" },|
|...|{ "id": "46", "parent": "37" },|
||...|



Table 10: Prompt examples for DSHP-LLM model training. The common prompt (top) is used for both Example 1 (easier, with numbered sections) and Example 2 (harder, no section numbers). Lines expanded so that both examples reach a similar height. The parent value _null_ denotes the root node. The symbol ... indicates omitted content for brevity. 

Figure 2: An example of a long industrial document (MPVQA) illustrating the content structure and formatting used for guidelines and requirements in nuclear power plant operations. The document contains various sections, such as general information, application scope, and specific criteria, serving as a representative case for evaluating document chunking methods. 

**==> picture [222 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Document Parsing example  — page-level layout regions<br>(titles, headers, text blocks, tables, figures) detected by DP.<br>(c)  DSHP-LLM example  — section headers parsed into a<br>document hierarchical tree with parent–child links; general<br>nodes attached by spatial order.<br>**----- End of picture text -----**<br>


**==> picture [206 x 22] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  OCR example  — recognized text linked to bounding<br>boxes, forming an annotated layout.<br>**----- End of picture text -----**<br>


**==> picture [215 x 34] intentionally omitted <==**

**----- Start of picture text -----**<br>
(d)  Chunking example  — DFS-based Grouping assembles<br>hierarchy-aware chunks that preserve both spatial and<br>semantic context.<br>**----- End of picture text -----**<br>


Figure 3: Step-by-step illustration aligned with the **MultiDocFusion** pipeline: (a) Document Parsing (DP), (b) OCR, (c) DSHP-LLM for section hierarchy reconstruction and node attachment, and (d) DFS-based Grouping for hierarchy-aware chunking. 

|**Method**|**Chunk**|**Example Content**|
|---|---|---|
||Chunk 1|A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Pharma-|
|Length chunking||ceuticals REMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000 tember 1999|
|||PROPRIETARY AND CONFIDENTIAL . . .|
||Chunk 2|INTRODUCTION Design Write has prepared the following proposal for a comprehensive educational and com-|
|||munications program . . .|
||Chunk 3|we have been guided by the brand operating strategies for 2000. We have had the fortunate experience of working|
|||with the HRT Management Team since 1997 . . .|
||Chunk 1|A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Pharma-|
|Semantic Chunking||ceuticals PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000 tember 1999|
|||PROPRIETARY AND CONFIDENTIAL © 1999 by Design Write, Ine. EXHIBIT W_§ Source: hops vAwaaw in-|
|||dusirvvdagumens cst eduideesipbwila |?|
||Chunk 2|Medical and Scientifc Communications Plan. 2000: Premarin® Family of Products|
||Chunk 3|INTRODUCTION Design Write has prepared the following proposal for a comprehensive educational and com-|
|||munications program to support the PREMARIN® Family of Products. Design Write thanks the Women’s Health|
|||Care, HRT Management Team, for this opportunity to present its ideas to further the goal of expanding the PRE-|
|||MARIN Family of Products’ position in the marketplace, In developing this proposal, we have been guided by the|
|||brand operating strategies for 2000.|
||Chunk 1|A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Pharma-|
|LumberChunker||ceuticals PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000 tember 1999|
|||PROPRIETARY AND CONFIDENTIAL © 1999 by Design Write, Ine. EXHIBIT W_§ Source: hops vAwaaw in-|
|||dusirvvdagumens cst eduideesipbwila |? Medical and Scientifc Communications Plan. 2000: Premarin® Family|
|||of Products|
||Chunk 2|Medical and Scientifc Communications Plan 2000: Premarin® Fi amily of Products INTRODUCTION Design|
|||Write has prepared the following proposal for a comprehensive educational and communications program to sup-|
|||port the PREMARIN® Family of Products.|
||Chunk 3|Design Write thanks the Women’s Health Care, HRT Management Team, for this opportunity to present its ideas to|
|||further the goal of expanding the PREMARIN Family of Products’ position in the marketplace, In developing this|
|||proposal, we have been guided by the brand operating strategies for 2000. We have had the fortunate experience of|
|||working with the HRT Management Team since 1997, during which time we successfully developed and impleme|
||Chunk 1|A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Pharma-|
|Perplexity chunking||ceuticals PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000 tember 1999|
|||PROPRIETARY AND CONFIDENTIAL © 1999 by Design Write, Ine.EXHIBIT W_§ Source: hops vAwaaw in-|
|||dusirvvdagumens cst eduideesipbwila |?Medical and Scientifc Communications Plan.|
||Chunk 2|"2000: Premarin® Family of Products Medical and Scientifc Communications Plan 2000: Premarin® Fi amily of|
|||Products INTRODUCTION Design Write has prepared the following proposal for a comprehensive educational|
|||and communications program to support the PREMARIN® Family of Products.Design Write thanks the Women’s|
|||Health Care, HRT Management Team, for this opportunity to present its ideas to further the goal of expanding the|
|||PR|
||Chunk 3|EMARIN Family of Products’ position in the marketplace, In developing this proposal, we have been guided by|
|||the brand operating strategies for 2000.|
||Chunk 1|[Title] | A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst|
|Structure-based Chunking||Pharmaceuticals|
||Chunk 2|[Title] PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000 - [Text] ” -|
|||[Text] Medical and Scientifc Communications Plan 2000: Premarin® Fi amily of Products|
||Chunk 3|[Section] INTRODUCTION - [Text] Design Write has prepared the following proposal for a comprehensive ed-|
|||ucational and communications program to support the PREMARIN® Family of Products. Design Write thanks|
|||the Women’s Health Care, HRT Management Team, for this opportunity to present its ideas to further the goal of|
|||expanding the PREMARIN Family of Products’ position in the marketplace, In developing this proposal, we have|
|||been guided by the bra|
||Chunk 1|# | A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Phar-|
|**MultiDocFusion**||maceuticals + PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000|
|||## INTRODUCTION|
|||text_split_1 : Design Write has prepared the following proposal for a comprehensive educational and communica-|
|||tions program to support the PREMARIN® Family of Products. Design Write thanks the Women’s Health Care,|
|||HRT Management Team, for this opportunity to present its ideas to further the goal of expanding the PREMARIN|
|||Family of Products’ position in|
||Chunk 2|# | A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Phar-|
|||maceuticals + PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000|
|||## INTRODUCTION|
|||text_split_2 : the marketplace, In developing this proposal, we have been guided by the brand operating strategies|
|||for 2000. We have had the fortunate experience of working with the HRT Management Team since 1997, during|
|||which time we successfully developed and implemented a number of programs, including: PREMARIN publica-|
|||tion plan of review articl|
||Chunk 3|# | A Proposal for — Jeffrey A. Solomon _ Product Director, HRT _ Women’s Health Care “Wyeth-Ayerst Phar-|
|||maceuticals + PREMARIN” FAMILY OF PRODUCTS Medical and Scientifc Communications Plan 2000|
|||## INTRODUCTION|
|||text_split_3 : es Sales training backgrounders:and journal article responses SERMs advisory board and executive|
|||summary Internal white papers Pharmaceutical compendia surveillance program Publications management pro-|
|||gram We believe that our expertise and experience enable us:to provide the necessary marketing support in the|
|||organization and development of sci|



Table 11: Qualitative comparison of chunking methods applied to the document in Figure 2. Each method shows three chunks (1 to 3) for six approaches: Length chunking, Semantic chunking, LumberChunker, Perplexity chunking, Structure-based chunking, and **MultiDocFusion** . 

|**Chunking Method**<br>**k**|**DUDE**<br>R<br>P<br>nDCG|**MPVQA**<br>R<br>P<br>nDCG|**CUAD**<br>R<br>P<br>nDCG|**MOAMOB**<br>R<br>P<br>nDCG|
|---|---|---|---|---|
|Length chunking<br>1<br>2<br>3<br>4<br>Semantic chunking<br>1<br>2<br>3<br>4<br>LumberChunker<br>1<br>2<br>3<br>4<br>Perplexity chunking<br>1<br>2<br>3<br>4<br>Structure-based chunking<br>1<br>2<br>3<br>4<br>**MultiDocFusion**<br>1<br>2<br>3<br>4|0.1642<br>0.1642<br>0.1642<br>0.2454<br>0.1684<br>0.2105<br>0.3001<br>0.1700<br>0.2368<br>0.3416<br>0.1717<br>0.2546<br>0.0586<br>0.0586<br>0.0586<br>0.0883<br>0.0559<br>0.0750<br>0.1095<br>0.0534<br>0.0848<br>0.1258<br>0.0516<br>0.0916<br>0.1538<br>0.1538<br>0.1538<br>0.2222<br>0.1526<br>0.1922<br>0.2715<br>0.1530<br>0.2157<br>0.3107<br>0.1538<br>0.2325<br>0.1566<br>0.1566<br>0.1566<br>0.2293<br>0.1572<br>0.1975<br>0.2753<br>0.1553<br>0.2196<br>0.3099<br>0.1547<br>0.2345<br>0.1471<br>0.1471<br>0.1471<br>0.2092<br>0.1453<br>0.1818<br>0.2500<br>0.1443<br>0.2013<br>0.2814<br>0.1432<br>0.2146<br>**0.2029**<br>**0.2029**<br>**0.2029**<br>**0.2791**<br>**0.2004**<br>**0.2459**<br>**0.3277**<br>**0.1993**<br>**0.2692**<br>**0.3612**<br>**0.1978**<br>**0.2838**|0.1556<br>0.1556<br>0.1556<br>0.2367<br>0.1591<br>0.1815<br>0.2889<br>0.1599<br>0.2084<br>0.3278<br>0.1601<br>0.2278<br>0.0562<br>0.0562<br>0.0562<br>0.0856<br>0.0530<br>0.0629<br>0.1079<br>0.0508<br>0.0724<br>0.1258<br>0.0495<br>0.0803<br>0.1282<br>0.1282<br>0.1282<br>0.1979<br>0.1288<br>0.1489<br>0.2484<br>0.1308<br>0.1742<br>0.2862<br>0.1315<br>0.1924<br>0.1313<br>0.1313<br>0.1313<br>0.1991<br>0.1312<br>0.1511<br>0.2481<br>0.1322<br>0.1755<br>0.2852<br>0.1326<br>0.1935<br>0.1209<br>0.1209<br>0.1209<br>0.1902<br>0.1235<br>0.1426<br>0.2347<br>0.1239<br>0.1651<br>0.2685<br>0.1237<br>0.1813<br>**0.1773**<br>**0.1773**<br>**0.1773**<br>**0.2566**<br>**0.1762**<br>**0.2016**<br>**0.3059**<br>**0.1753**<br>**0.2275**<br>**0.3422**<br>**0.1749**<br>**0.2460**|0.8607<br>0.8607<br>0.8607<br>0.9001<br>0.8513<br>0.8770<br>0.9157<br>0.8514<br>0.8842<br>0.9278<br>0.8513<br>0.8883<br>0.6810<br>0.6810<br>0.6810<br>0.7560<br>0.6657<br>0.7100<br>0.8063<br>0.6726<br>0.7353<br>0.8304<br>0.6681<br>0.7463<br>0.8611<br>0.8611<br>0.8611<br>**0.9049**<br>0.8577<br>0.8820<br>0.9179<br>0.8568<br>0.8867<br>**0.9287**<br>0.8549<br>0.8902<br>0.8355<br>0.8355<br>0.8355<br>0.8859<br>0.8389<br>0.8602<br>0.9049<br>0.8413<br>0.8696<br>0.9212<br>0.8422<br>0.8758<br>0.8341<br>0.8341<br>0.8341<br>0.8851<br>0.8347<br>0.8597<br>0.9026<br>0.8267<br>0.8667<br>0.9160<br>0.8289<br>0.8718<br>**0.8622**<br>**0.8622**<br>**0.8622**<br>0.9021<br>**0.8655**<br>**0.8832**<br>**0.9186**<br>**0.8688**<br>**0.8902**<br>0.9254<br>**0.8638**<br>**0.8919**|0.5847<br>0.5847<br>0.5847<br>0.6435<br>0.5711<br>0.6218<br>0.6696<br>0.5605<br>0.6349<br>0.6869<br>0.5542<br>0.6423<br>0.2079<br>0.2079<br>0.2079<br>0.2598<br>0.1943<br>0.2406<br>0.2963<br>0.1903<br>0.2589<br>0.3309<br>0.1874<br>0.2738<br>0.5022<br>0.5022<br>0.5022<br>0.6242<br>0.5304<br>0.5792<br>0.6524<br>0.5272<br>0.5932<br>0.6731<br>0.5221<br>0.6022<br>0.5195<br>0.5195<br>0.5195<br>0.6262<br>0.5336<br>0.5868<br>0.6513<br>0.5225<br>0.5994<br>0.6721<br>0.5207<br>0.6083<br>0.4593<br>0.4593<br>0.4593<br>0.5457<br>0.4647<br>0.5138<br>0.5931<br>0.4700<br>0.5375<br>0.6197<br>0.4709<br>0.5490<br>**0.6153**<br>**0.6153**<br>**0.6153**<br>**0.6828**<br>**0.6281**<br>**0.6629**<br>**0.7006**<br>**0.6183**<br>**0.6718**<br>**0.7122**<br>**0.6121**<br>**0.6768**|



Table 12: Retrieval summary by Chunking Method — Comparison of VQA Datasets (top- _k_ = 1 _∼_ 4) 

|**Chunking Method**<br>**OCR**|**DUDE**<br>R<br>P<br>nDCG|**MPVQA**<br>R<br>P<br>nDCG|**CUAD**<br>R<br>P<br>nDCG|**MOAMOB**<br>R<br>P<br>nDCG|
|---|---|---|---|---|
|Length chunking<br>easyocr<br>tesseract<br>trocr<br>Semantic chunking<br>easyocr<br>tesseract<br>trocr<br>LumberChunker<br>easyocr<br>tesseract<br>trocr<br>Perplexity chunking<br>easyocr<br>tesseract<br>trocr<br>Structure-based chunking<br>easyocr<br>tesseract<br>trocr<br>**MultiDocFusion**<br>easyocr<br>tesseract<br>trocr|0.2836<br>0.1805<br>0.2289<br>0.2511<br>0.1617<br>0.2082<br>0.2538<br>**0.1636**<br>**0.2125**<br>0.1050<br>0.0593<br>0.0845<br>0.0909<br>0.0516<br>0.0739<br>0.0907<br>0.0537<br>0.0742<br>0.2543<br>0.1664<br>0.2114<br>0.2146<br>0.1351<br>0.1765<br>0.2497<br>0.1586<br>0.2078<br>0.2482<br>0.1603<br>0.2051<br>0.2422<br>0.1584<br>0.2029<br>0.2379<br>0.1491<br>0.1981<br>0.2633<br>0.1759<br>0.2231<br>0.2476<br>0.1631<br>0.2078<br>0.1548<br>0.0960<br>0.1277<br>**0.3305**<br>**0.2350**<br>**0.2871**<br>**0.2938**<br>**0.2020**<br>**0.2524**<br>**0.2539**<br>0.1633<br>0.2119|0.2701<br>0.1658<br>0.2042<br>0.2804<br>0.1805<br>0.2168<br>0.2064<br>0.1298<br>0.1589<br>0.1037<br>0.0572<br>0.0757<br>0.1014<br>0.0556<br>0.0731<br>0.0765<br>0.0443<br>0.0551<br>0.2272<br>0.1334<br>0.1669<br>0.2303<br>0.1410<br>0.1738<br>0.1881<br>0.1151<br>0.1421<br>0.2230<br>0.1331<br>0.1667<br>0.2396<br>0.1486<br>0.1816<br>0.1851<br>0.1138<br>0.1403<br>0.2439<br>0.1451<br>0.1819<br>0.2456<br>0.1542<br>0.1884<br>0.1212<br>0.0697<br>0.0871<br>**0.3034**<br>**0.1974**<br>**0.2401**<br>**0.2963**<br>**0.1949**<br>**0.2347**<br>**0.2117**<br>**0.1354**<br>**0.1645**|0.9076<br>0.8756<br>**0.8923**<br>0.8919<br>0.8469<br>0.8644<br>**0.9037**<br>**0.8386**<br>**0.8760**<br>0.7392<br>0.6517<br>0.6869<br>0.7573<br>0.6584<br>0.7057<br>0.8088<br>0.7055<br>0.7618<br>0.9021<br>0.8670<br>0.8813<br>0.9046<br>0.8637<br>0.8817<br>0.9028<br>0.8421<br>0.8770<br>0.8979<br>0.8647<br>0.8823<br>0.9000<br>0.8632<br>0.8795<br>0.8627<br>0.7904<br>0.8191<br>0.8996<br>0.8600<br>0.8789<br>0.8990<br>0.8553<br>0.8765<br>0.8547<br>0.7781<br>0.8188<br>**0.9077**<br>**0.8787**<br>0.8911<br>**0.9077**<br>**0.8794**<br>**0.8896**<br>0.8908<br>0.8370<br>0.8650|0.8511<br>0.7689<br>0.8223<br>0.6489<br>0.5919<br>0.6301<br>**0.4385**<br>0.3421<br>**0.4103**<br>0.4437<br>0.3241<br>0.4078<br>0.2785<br>0.2038<br>0.2502<br>0.0989<br>0.0571<br>0.0779<br>0.8111<br>0.7125<br>0.7631<br>0.6189<br>0.5474<br>0.5863<br>0.4089<br>0.3015<br>0.3582<br>0.8293<br>0.7370<br>0.7918<br>0.6337<br>0.5665<br>0.6057<br>0.3889<br>0.2687<br>0.3380<br>0.8318<br>0.7279<br>0.7938<br>0.6230<br>0.5464<br>0.5871<br>0.2085<br>0.1243<br>0.1637<br>**0.8859**<br>**0.8081**<br>**0.8542**<br>**0.6644**<br>**0.6107**<br>**0.6504**<br>0.4109<br>**0.3756**<br>0.3972|



Table 13: OCR performance by Chunking Method (top- _k_ = 1 _∼_ 4 average) 

|**Chunking Method**<br>**DP Model**|**DUDE**<br>R<br>P<br>nDCG|**MPVQA**<br>R<br>P<br>nDCG|**CUAD**<br>R<br>P<br>nDCG|**MOAMOB**<br>R<br>P<br>nDCG|
|---|---|---|---|---|
|Length chunking<br>detr<br>dit<br>vgt<br>Semantic chunking<br>detr<br>dit<br>vgt<br>LumberChunker<br>detr<br>dit<br>vgt<br>Perplexity chunking<br>detr<br>dit<br>vgt<br>Structure-based chunking<br>detr<br>dit<br>vgt<br>**MultiDocFusion**<br>detr<br>dit<br>vgt|0.2918<br>0.2007<br>0.2507<br>0.2852<br>0.1836<br>0.2370<br>0.2115<br>0.1214<br>0.1620<br>0.1298<br>0.0784<br>0.1069<br>0.1037<br>0.0591<br>0.0845<br>0.0532<br>0.0271<br>0.0412<br>0.2386<br>0.1612<br>0.2027<br>0.2357<br>0.1462<br>0.1920<br>0.2442<br>0.1526<br>0.2011<br>0.2527<br>0.1697<br>0.2145<br>0.2429<br>0.1556<br>0.2019<br>0.2327<br>0.1425<br>0.1898<br>0.2390<br>0.1642<br>0.2046<br>0.2210<br>0.1440<br>0.1855<br>0.2058<br>0.1268<br>0.1684<br>**0.3051**<br>**0.2192**<br>**0.2662**<br>**0.3017**<br>**0.2076**<br>**0.2588**<br>**0.2714**<br>**0.1736**<br>**0.2264**|0.2588<br>0.1680<br>0.2023<br>**0.2835**<br>0.1852<br>0.2250<br>0.2145<br>0.1227<br>0.1526<br>0.1109<br>0.0605<br>0.0790<br>0.1123<br>0.0647<br>0.0833<br>0.0585<br>0.0318<br>0.0416<br>0.2123<br>0.1301<br>0.1596<br>0.2315<br>0.1444<br>0.1784<br>0.2018<br>0.1150<br>0.1448<br>0.2098<br>0.1301<br>0.1585<br>0.2312<br>0.1447<br>0.1777<br>0.2068<br>0.1206<br>0.1524<br>0.2013<br>0.1205<br>0.1471<br>0.2103<br>0.1293<br>0.1606<br>0.1991<br>0.1192<br>0.1497<br>**0.2800**<br>**0.1864**<br>**0.2235**<br>0.2671<br>0.1758<br>0.2117<br>**0.2644**<br>**0.1654**<br>**0.2041**|0.9097<br>0.8800<br>0.8934<br>0.9047<br>0.8450<br>0.8804<br>0.8888<br>0.8361<br>0.8590<br>0.8621<br>0.7873<br>0.8226<br>0.7373<br>0.6244<br>0.6746<br>0.7059<br>0.6039<br>0.6573<br>**0.9213**<br>0.8866<br>**0.9049**<br>0.9027<br>0.8420<br>0.8701<br>0.8854<br>0.8442<br>0.8650<br>0.9055<br>0.8720<br>0.8878<br>0.8739<br>0.8093<br>0.8329<br>0.8813<br>0.8371<br>0.8602<br>0.8941<br>0.8550<br>0.8710<br>0.8785<br>0.8075<br>0.8464<br>0.8806<br>0.8308<br>0.8568<br>0.9144<br>**0.8882**<br>0.9006<br>0.8990<br>**0.8527**<br>0.8760<br>**0.8928**<br>**0.8543**<br>**0.8691**|**0.6611**<br>**0.6078**<br>**0.6345**<br>0.6278<br>0.5365<br>0.6030<br>0.6496<br>0.5585<br>0.6252<br>0.3274<br>0.2374<br>0.2903<br>0.2933<br>0.2241<br>0.2667<br>0.2004<br>0.1234<br>0.1788<br>0.6178<br>0.5509<br>0.5809<br>0.6048<br>0.5168<br>0.5727<br>0.6163<br>0.4937<br>0.5541<br>0.6315<br>0.5517<br>0.5915<br>0.5985<br>0.5089<br>0.5716<br>0.6218<br>0.5117<br>0.5724<br>0.5859<br>0.4937<br>0.5355<br>0.5130<br>0.4267<br>0.4759<br>0.5644<br>0.4782<br>0.5332<br>0.6318<br>0.5965<br>0.6152<br>**0.6711**<br>**0.5959**<br>**0.6438**<br>**0.7407**<br>**0.6773**<br>**0.7246**|



Table 14: DP performance by Chunking Method (top- _k_ = 1 _∼_ 4 average) 

|**Chunking Method**<br>**Embedding**|**DUDE**<br>R<br>P<br>nDCG|**MPVQA**<br>R<br>P<br>nDCG|**CUAD**<br>R<br>P<br>nDCG|**MOAMOB**<br>R<br>P<br>nDCG|
|---|---|---|---|---|
|Length chunking<br>bge<br>e5<br>BM25<br>Semantic chunking<br>bge<br>e5<br>BM25<br>LumberChunker<br>bge<br>e5<br>BM25<br>Perplexity chunking<br>bge<br>e5<br>BM25<br>Structure-based chunking<br>bge<br>e5<br>BM25<br>**MultiDocFusion**<br>bge<br>e5<br>BM25|0.2655<br>0.1646<br>0.2097<br>0.2644<br>0.1602<br>0.2178<br>0.2586<br>0.1810<br>0.2222<br>0.0753<br>0.0433<br>0.0619<br>0.1184<br>0.0665<br>0.0955<br>0.0929<br>0.0548<br>0.0752<br>0.2615<br>0.1651<br>0.2157<br>0.1983<br>0.1132<br>0.1564<br>0.2587<br>0.1817<br>0.2236<br>0.2673<br>0.1695<br>0.2213<br>0.2084<br>0.1204<br>0.1661<br>0.2526<br>0.1779<br>0.2188<br>0.2611<br>0.1672<br>0.2176<br>0.1723<br>0.0977<br>0.1377<br>0.2323<br>0.1700<br>0.2033<br>**0.3222**<br>**0.2202**<br>**0.2771**<br>**0.2762**<br>**0.1797**<br>**0.2318**<br>**0.2798**<br>**0.2005**<br>**0.2425**|0.2810<br>0.1742<br>0.2118<br>0.2469<br>0.1485<br>0.1855<br>0.2289<br>**0.1534**<br>**0.1827**<br>0.0876<br>0.0511<br>0.0653<br>0.1152<br>0.0650<br>0.0839<br>0.0789<br>0.0409<br>0.0547<br>0.2456<br>0.1456<br>0.1811<br>0.1892<br>0.1065<br>0.1370<br>0.2108<br>0.1373<br>0.1647<br>0.2561<br>0.1554<br>0.1938<br>0.1844<br>0.1030<br>0.1316<br>0.2073<br>0.1371<br>0.1632<br>0.2560<br>0.1570<br>0.1960<br>0.1615<br>0.0901<br>0.1166<br>0.1932<br>0.1219<br>0.1447<br>**0.3112**<br>**0.2070**<br>**0.2501**<br>**0.2682**<br>**0.1673**<br>**0.2071**<br>**0.2320**<br>0.1533<br>0.1821|**0.9152**<br>0.8778<br>0.8960<br>0.8915<br>0.8491<br>0.8651<br>0.8965<br>0.8342<br>0.8717<br>0.8663<br>0.8046<br>0.8361<br>0.8380<br>0.7628<br>0.8035<br>0.6009<br>0.4482<br>0.5148<br>0.9055<br>0.8699<br>0.8852<br>0.8937<br>0.8441<br>0.8660<br>**0.9102**<br>**0.8589**<br>**0.8889**<br>0.8858<br>0.8471<br>0.8635<br>0.8918<br>0.8485<br>0.8682<br>0.8831<br>0.8227<br>0.8492<br>0.9003<br>0.8559<br>0.8757<br>0.8681<br>0.8047<br>0.8378<br>0.8850<br>0.8327<br>0.8606<br>0.9125<br>**0.8815**<br>**0.8967**<br>**0.8952**<br>**0.8581**<br>**0.8723**<br>0.8986<br>0.8555<br>0.8767|0.6504<br>0.5677<br>0.6162<br>0.6467<br>0.5672<br>0.6175<br>0.6415<br>0.5679<br>0.6290<br>0.2956<br>0.2309<br>0.2824<br>0.4126<br>0.2966<br>0.3681<br>0.1130<br>0.0575<br>0.0853<br>0.6378<br>0.5523<br>0.6013<br>0.6100<br>0.5230<br>0.5680<br>0.5911<br>0.4862<br>0.5383<br>0.6418<br>0.5498<br>0.6073<br>0.6026<br>0.5138<br>0.5613<br>0.6074<br>0.5086<br>0.5668<br>0.6211<br>0.5236<br>0.5823<br>0.5574<br>0.4734<br>0.5237<br>0.4848<br>0.4016<br>0.4386<br>**0.6808**<br>**0.6276**<br>**0.6613**<br>**0.6663**<br>**0.6070**<br>**0.6425**<br>**0.7560**<br>**0.6859**<br>**0.7325**|



Table 15: Performance comparison by Embedding and Chunking Method (top- _k_ = 1 _∼_ 4 average) 

|Top-k<br>Chunking Method<br>1<br>Length chunking<br>Semantic chunking<br>LumberChunker<br>Perplexity chunking<br>Structure based Chunking<br>**MultiDocFusion**|**MPVQA**<br>ANLS<br>ROUGE-L<br>METEOR<br>0.1299<br>0.0679<br>0.0859<br>0.1256<br>0.0635<br>0.0794<br>0.1394<br>0.0737<br>0.0873<br>0.1254<br>0.0544<br>0.0676<br>0.1440<br>0.0858<br>0.1103<br>**0.1473**<br>**0.1021**<br>**0.1335**|**DUDE**<br>ANLS<br>ROUGE-L<br>METEOR<br>0.1573<br>0.1281<br>0.1722<br>0.1527<br>0.1126<br>0.1420<br>0.1614<br>0.1264<br>0.1642<br>0.1683<br>0.1357<br>0.1689<br>0.1615<br>0.1217<br>0.1527<br>**0.1726**<br>**0.1512**<br>**0.2001**|**CUAD**<br>ANLS<br>ROUGE-L<br>METEOR<br>0.2675<br>**0.1761**<br>0.1616<br>0.2611<br>0.1544<br>0.1409<br>0.2690<br>0.1698<br>**0.1657**<br>0.2592<br>0.1629<br>0.1457<br>0.2489<br>0.1569<br>0.1592<br>**0.2692**<br>0.1739<br>0.1578|**MOAMOB**<br>ANLS<br>ROUGE-L<br>METEOR<br>0.2498<br>0.0739<br>0.1054<br>0.2444<br>0.0737<br>0.1009<br>0.2588<br>0.0748<br>0.1189<br>0.2530<br>0.0868<br>0.1182<br>0.2477<br>**0.0892**<br>0.1108<br>**0.2642**<br>0.0826<br>**0.1239**|
|---|---|---|---|---|
|4<br>Length chunking<br>Semantic chunking<br>LumberChunker<br>Perplexity chunking<br>Structure based Chunking<br>**MultiDocFusion**|0.1398<br>0.0966<br>0.1408<br>0.1332<br>0.0805<br>0.0978<br>0.1307<br>0.0769<br>0.0993<br>0.1344<br>0.0751<br>0.0950<br>0.1537<br>0.0980<br>0.1278<br>**0.1615**<br>**0.1316**<br>**0.1850**|0.1611<br>0.1444<br>0.1988<br>0.1548<br>0.1261<br>0.1657<br>0.1531<br>0.1284<br>0.1752<br>0.1653<br>0.1390<br>0.1855<br>0.1751<br>0.1489<br>0.1921<br>**0.1859**<br>**0.1692**<br>**0.2285**|0.2495<br>0.1593<br>0.1708<br>0.2574<br>0.1438<br>0.1526<br>0.2623<br>0.1562<br>0.1643<br>0.2690<br>0.1662<br>0.1591<br>0.2507<br>0.1543<br>0.1590<br>**0.2783**<br>**0.1785**<br>**0.1721**|0.2495<br>0.0906<br>0.1176<br>0.2465<br>0.0955<br>0.1076<br>0.2483<br>0.0947<br>0.1144<br>0.2534<br>0.0919<br>0.1197<br>**0.2524**<br>**0.1065**<br>0.1120<br>0.2550<br>0.1005<br>**0.1274**|



Table 16: Average generation performance ( _ANLS_ , _ROUGE-L_ , _METEOR_ ) of six chunking strategies on MPVQA, DUDE, CUAD, and MOAMOB datasets, separated by top- _k_ settings ( _k_ = 1 and _k_ = 4). Best scores for each metric and dataset are highlighted in **bold** . 

