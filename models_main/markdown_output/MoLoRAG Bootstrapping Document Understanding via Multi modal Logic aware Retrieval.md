# **MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval** 

**Xixi Wu**[1] **, Yanchao Tan**[2] **, Nan Hou**[1] **, Ruiyang Zhang**[3] **, Hong Cheng**[1(] 

) 

1The Chinese University of Hong Kong 2Fuzhou University 3University of Macau {xxwu, nhou, hcheng}@se.cuhk.edu.hk yctan@fzu.edu.cn, yc47931@um.edu.mo 

## **Abstract** 

Document Understanding is a foundational AI capability with broad applications, and Document Question Answering (DocQA) is a key evaluation task. Traditional methods convert the document into text for processing by Large Language Models (LLMs), but this process strips away critical multi-modal information like figures. While Large Vision-Language Models (LVLMs) address this limitation, their constrained input size makes multi-page document comprehension infeasible. Retrievalaugmented generation (RAG) methods mitigate this by selecting relevant pages, but they rely solely on semantic relevance, ignoring logical connections between pages and the query, which is essential for reasoning. 

To this end, we propose **MoLoRAG** , a logicaware retrieval framework for multi-modal, multi-page document understanding. By constructing a page graph that captures contextual relationships between pages, a lightweight VLM performs graph traversal to retrieve relevant pages, including those with logical connections often overlooked. This approach combines semantic and logical relevance to deliver more accurate retrieval. After retrieval, the top- _K_ pages are fed into arbitrary LVLMs for question answering. To enhance flexibility, MoLoRAG offers two variants: a training-free solution for easy deployment and a fine-tuned version to improve logical relevance checking. Experiments on four DocQA datasets demonstrate average improvements of **9.68%** in accuracy over LVLM direct inference and **7.44%** in retrieval precision over baselines. Codes and datasets are released at https://github.com/WxxShirley/MoLoRAG. 

## **1 Introduction** 

Document Understanding is a foundational AI capability with extensive real-world applications, such as interpreting medical reports and assisting 

**==> picture [215 x 303] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question  How many days with overflow do Outfall 002A (Southwest<br>Hoboken) and Outfall 005A (Central Hoboken) have in total?<br>Top-1 Page Retrieved by<br>M3DocRAG  and  MDocAgent<br>Outfall OO1A/002A (West New York) FY<br>Overflow Volume<br>200<br>175]<br>1503=<br>gas<br>423 109) 5th largest Overflow: 8.3 MG<br>6 75]<br>50<br>ool r) 10 20 30 WADA40 Rotana30 %<br>Overflow Number<br>Ground-truth Evidence Page<br>Adams Street Combined Sewer System Performance for a Typical Year<br>ileleteieteleietetetens Adame St PS sername<br>H Outfall ' wwre i, ions<br>1 #of Days with Overflow , firs] i aaeeeeel<br>' #of Overflow Events 1 ee a<br>\ Annualan Overflow Volume! by agers ey Baldwihave PS<br>2 cr " a8" | ue<br>(co2a--3->,H et'[ oosa |! [ o06a 008A 012A 013A 015A<br>' 49 days ||) 003A ' | 116 days ' 17 days 13 days 10 days 85 days ‘53 days<br>| 34events|i{ (Closed) | | | atevents|; | 12events| | 14events| | 10 events 58 events 40 events<br>1, 4G |! 4 65MG |) 10MG 14MG 4MG 219 MG 24MG<br><s=---”. ~== =="<br>Answer   49 days + 116 days = 165 days<br>**----- End of picture text -----**<br>


Figure 1: **Illustration of a retrieval example on LongDocURL (Deng et al., 2024).** Both M3DocRAG (Cho et al., 2024) and MDocAgent (Han et al., 2025) rely solely on **semantic relevance** between the query and the page for retrieval. As a result, they retrieve a page containing keywords from the question but lacking the necessary information to answer it. In contrast, the ground-truth evidence page, successfully retrieved by our MoLoRAG, is **logically relevant** to the question, providing detailed statistics for each outfall and enabling accurate derivation of the correct answer. 

with academic literature. This ability holds significant potential to improve productivity and support decision-making (Ding et al., 2022; Ma et al., 2024a; Suri et al., 2024; Zhang et al., 2024). A key task for evaluating document understanding is Document Question Answering (DocQA), which requires models to automatically answer questions 

based on the content of a document. 

Classic approaches to DocQA typically follow a two-step pipeline: the document is first converted into text using Optical Character Recognition (OCR) (Memon et al., 2020; Wang et al., 2024a; Wei et al., 2024), and then Retrieval-augmented Generation (RAG) techniques identify relevant paragraphs to feed into Large Language Models (LLMs) for question answering. However, the text extraction process often strips away essential multimodal information, such as tables, figures, and document layouts, resulting in incomplete document understanding. Large Vision-Language Models (LVLMs) address this limitation by processing image-format document snapshots, enabling multimodal comprehension. Nevertheless, LVLMs, such as LLaMA-Vision (Grattafiori et al., 2024) and LLaVA-Next (Li et al., 2024a), are constrained to single-image inputs, rendering them ineffective for long, multi-page documents. 

Recent research has explored methods to address these challenges. For example, M3DocRAG (Cho et al., 2024) leverages a document encoder, i.e., ColPali (Faysse et al., 2024), to encode individual pages and retrieve relevant ones based on vector similarity. This approach reduces the number of input pages, alleviating the comprehension burden for LVLMs. MDocAgent (Han et al., 2025) extends this by introducing parallel pipelines for text and image retrieval, with specialized agents for each modality to enable collaborative reasoning. While effective, these methods focus primarily on **semantic relevance** , matching queries to pages based on embedding similarity. For example, as shown in Figure 1, when asked to determine the total overflow days for two outfalls, the top-1 page retrieved by both methods contains keywords from the question but lacks detailed information about each outfall. In contrast, a **logically relevant** page, such as the ground-truth evidence page, provides detailed statistics for each outfall, enabling reasoning (e.g., summing the overflow days) to derive the correct answer. 

In DocQA, accurate retrieval is critical, as it directly impacts downstream answering. Without precise retrieval, LVLMs are prone to errors or hallucinations stemming from irrelevant or incomplete inputs. Addressing this challenge requires retrieval methods that go beyond surface-level semantic matching to capture deeper logical relationships. Building on this insight, we propose **MoLoRAG** , a graph-based retrieval framework tailored for **M** ulti- 

m **o** dal **Lo** gic-aware document understanding. Document pages naturally exhibit structured relationships, e.g., cross-references and shared entities. Leveraging this property, we first construct a page graph to represent the dependencies between pages. A lightweight VLM serves as the retrieval engine, reasoning over this graph through traversal to identify logically relevant pages. Finally, both semantic and logical relevance are combined into a unified similarity score to re-rank pages, enabling a more comprehensive retrieval process. 

To further enhance its utility, MoLoRAG introduces two variants to offer flexibility for different deployments. The first is a training-free variant, which leverages a pre-trained VLM, e.g., Qwen2.5VL-3B (Bai et al., 2025), to perform graph traversal and retrieval directly, providing an off-the-shelf solution that is easy to deploy. The second is a finetuned variant, which involves training the retrieval engine on a curated dataset to improve its reasoning capabilities. This fine-tuned version functions as a more intelligent retrieval engine, capable of capturing nuanced relationships between queries and document pages. Moreover, MoLoRAG demonstrates strong compatibility with arbitrary LVLMs. Once the retrieval step is complete, only the top- _K_ page snapshots are passed to an LVLM for question answering, filtering out irrelevant content and ensuring concise, high-quality inputs. To summarize, our contributions are as follows: 

- **Logic-aware Retrieval Framework** We highlight the importance of page retrieval in DocQA and propose MoLoRAG, a novel retrieval method that incorporates logical relevance. By representing the document as a page graph and enabling a VLM to perform multi-hop reasoning through graph traversal, our method identify both semantically and logically relevant pages. 

- **Comprehensive Experiments** We conduct extensive experiments on four DocQA datasets, comparing MoLoRAG with LLMbased, LVLM-based, and Multi-agent methods. Results demonstrates its superior retrieval accuracy, significant performance improvements over baselines, and flexible compatibility with arbitrary LVLMs. 

- **Released Model and Dataset** We release the fine-tuned retriever engine model weights[1] 

- 1https://huggingface.co/xxwu/MoLoRAG-QwenVL-3B 

and the curated training dataset[2] , empowering further development of intelligent and logicaware retrieval engines. 

## **2 Related Works** 

**Document Question Answering** DocQA is a core task for evaluating document understanding. Early benchmarks (Mathew et al., 2021; Tito et al., 2023; Tanaka et al., 2023) focused on single-page or short documents with low information density, where questions targeted individual elements like text or figures. Recent benchmarks (Ma et al., 2024b; Deng et al., 2024) shift toward lengthy, informationrich documents, introducing challenges like crosspage and multi-modal reasoning. DocQA methods can be broadly categorized into two branches based on backbone models: LLM-based and LVLMbased. LLM-based methods rely on OCR techniques to extract text from the document, enabling text-based question answering. LVLM-based methods, on the other hand, leverage their inherent multi-modal capabilities to process document images directly. With the advancements in LVLMs, the latter approach now dominates recent solutions. A notable advancement is MDocAgent (Han et al., 2025), which represents a new category by combining both LLMs and LVLMs into a multi-agent framework for collaborative question answering. However, challenges like input size limitations still necessitate effective retrieval strategies to reduce input burden and enhance performance. 

**Retrieval-augmented Generation** RAG enhances LLMs by supplementing them with external knowledge, improving performance in domain-specific or knowledge-intensive tasks (Gao et al., 2024; Lewis et al., 2021; Asai et al., 2024). The emergence of LVLMs has further expanded RAG to multi-modal contexts, enabling the retrieval of relevant images to handle knowledge-seeking queries (Chen et al., 2024, 2022). Despite these advancements, existing RAG methods fail to address the unique challenges of DocQA, involving highly interleaved textual and visual elements. For page retrieval in DocQA, existing method like M3DocRAG (Cho et al., 2024) rely on document encoders for semantic-based retrieval, neglecting the logical relevance essential for accurate question answering. 

**Graph-based RAG** GraphRAG is an advanced RAG paradigm that leverages graph-structured knowledge and retrieval for improved contextual 

2https://huggingface.co/datasets/xxwu/MoLoRAG 

reasoning (Zhang et al., 2025; Xiang et al., 2025). Existing methods are categorized into two types: Knowledge-based, which constructs knowledge graphs through entity recognition and relation extraction (He et al., 2024), and Index-based, which creates a two-layer graph linking high-level topic nodes to detailed text nodes for efficient retrieval (Sarthi et al., 2024; Edge et al., 2025; Liu et al., 2025; Li et al., 2024b). However, current GraphRAG approaches are limited to text and cannot handle the document with multi-modal information. We are the first to extend GraphRAG to the document domain by constructing a page graph that enables reasoning over its structure. 

## **3 Methodology** 

In this section, we present the details of MoLoRAG, a novel graph-based retrieval framework designed to facilitate multi-modal and multi-page document understanding. The overall framework is illustrated in Figure 2. 

## **3.1 Preliminary** 

Given a question _q_ expressed in natural language and a document _D_ = _{p_ 1 _, p_ 2 _, . . . , pN }_ , where each _pi_ represents an individual page in the form of an RGB image, and _N_ is the total number of pages. The goal of DocQA is to generate an answer _a_ that accurately addresses _q_ using the information contained within _D_ . To solve this task, MoLoRAG adopts an LVLM-based two-stage framework: 

- **Retrieval:** Given the extensive nature of document _D_ , the first step involves retrieving the top- _K_ most relevant pages for the question, denoted as _P[r]_ = _{p[r]_ 1 _[, . . . , p][r] K[}]_[,][where] _K ≪ N_ (e.g., _K_ = 3). Unlike traditional retrieval methods that rely solely on semantic relevance, MoLoRAG incorporates both semantic and logical relevance to enhance retrieval accuracy and contextual understanding for effective reasoning. 

- **Generation:** The retrieved pages _P[r]_ , along with the input question _q_ , are then fed into an LVLM to generate the answer _a_ . For LVLMs that cannot directly process multiple images, we use a processing function Process( _·_ ) to prepare _P[r]_ , e.g., concatenating multiple images into a single composite one. Formally, this stage is expressed as: 

**==> picture [430 x 182] intentionally omitted <==**

**----- Start of picture text -----**<br>
Graph-based Index Graph Traversal for Retrieval Question Answering<br>Exploration Set<br>Document 𝒟 Univisited<br>··· Neighbors<br>𝑠 [sem] = 0.4 𝑠 [logi] = 0.1 ···<br>Final 𝒔 Re-rank by  Final 𝒔<br>Top-1<br>wig ty v e<br>𝑠 [sem] = 0.1 𝑠 [logi] = 0.5 Question<br>Final 𝒔<br>+<br>Question  𝒒 In Figure 12, which  /<br>··· LVLM<br>variant consistently …?  Retrieval Engine<br>𝑠 [sem] = ,  𝑠 [logi]  = VLM(  𝑞 , ) Answer<br>**----- End of picture text -----**<br>


Figure 2: **Illustration of MoLoRAG framework.** 

**==> picture [128 x 12] intentionally omitted <==**

## **3.2 Logic-aware Page Retrieval** 

In this subsection, we detail the retrieval process of MoLoRAG. We first construct a page graph as a graph-based index, depicting the relationships between pages within a document. Then, a VLM serves as the retrieval engine, performing reasoning over this graph through traversal to adaptively identify pages that are both semantically and logically relevant to the given question. 

**Graph-based Index** Firstly, each document page _pi_ is encoded into a latent embedding that captures its distinct multi-modal content, represented as _Epi_ = DocEncoder( _pi_ ) _∈_ R _[k][×][d]_ , where _k_ denotes the number of visual tokens per page and _d_ is the embedding dimension. Following Cho et al. (2024); Han et al. (2025), we choose ColPali (Faysse et al., 2024) as the document encoder due to its demonstrated effectiveness in preserving multi-modal semantics. 

Using these embeddings, we construct a page graph _G_ ( _V, E_ ) to represent relationships between pages. In this graph, each node _pi ∈V_ corresponds to a page from the document _D_ , and edges _E_ are established between pairs of pages based on their similarity. Specifically, an edge ( _pi, pj_ ) is added if the similarity between their embeddings exceeds a threshold _θ_ , expressed as: _E_ = _{_ ( _pi, pj_ ) _|⟨Epi, Epj ⟩≥ θ}_ where _⟨·, ·⟩_ denotes the inner product as the similarity measure. While such graph construction mechanism is simple, it offers the advantages of being **efficient** , **automatic** , ensuring **scalability** to large document, and lever- 

aging prior knowledge encoded in the embedding. **Graph Traversal for Retrieval** With the page graph constructed, we leverage a VLM as the retrieval engine to evaluate the relevance of each visited page in relation to the given question. This approach overcomes the limitations of traditional semantic-only retrieval by incorporating logical checking into the process. By utilizing the reasoning capabilities of the VLM, our method effectively identifies important pages that may otherwise be overlooked. The graph traversal process is outlined as follows, aligning with the pseudo-code in Algorithm 1 in the Appendix. 

> _−_ **Initialization** For the question _q_ and a page _pi_ from the document, the document encoder computes a semantic relevance score as _s_[sem] _i_ = _⟨_ DocEncoder( _q_ ) _, Epi⟩_ . Based on these scores, the top- _w_ nodes (pages) with the highest semantic scores are selected as the initial exploration set. 

_−_ **Relevance Scoring** For page _pi_ in the exploration set, the VLM assigns a logical relevance score _s_[logi] _i_ using the prompt provided in Appendix B, reflecting the deeper logical connection of the page to the question. The final relevance score _si_ is then updated as _si_ = Combine( _s_[sem] _i , s_[logi] _i_ ), where Combine( _·_ ) integrates semantic and logical relevance scores, e.g., taking their weighted average. 

_−_ **Iterative Traversal** The traversal proceeds iteratively: at each step, we define the candidate set as the unvisited neighbors of the current exploration set. Each page in this candidate set is evaluated and its relevance score is updated using the same combination of semantic and logical relevance. The pages are ranked by their final relevance scores, and only the top- _w_ nodes are retained as the new 

**==> picture [218 x 67] intentionally omitted <==**

**----- Start of picture text -----**<br>
Triplets<br>Selected<br>image Question Relevance3<br>Selected image<br>Question Predicted<br>𝒔= 𝟑 What is the reward of the revised trajectory ? Relevance: 3<br>Sampled score<br>ea t Step 1  Question Generation 6 { |} Step 2  Quality Checking 6<br>an ne<br>**----- End of picture text -----**<br>


Figure 3: **Illustration of training data generation for MoLoRAG+.** 

exploration set for the next iteration. The traversal continues until either the candidate set is empty or the maximum hop limit is reached. Both the exploration set size _w_ and the hop limit _n_ hop constrain the traversal space, ensuring efficiency by avoiding the exhaustive process of sequentially traversing every page in the document. 

Once the traversal is complete, all visited nodes are **re-ranked** based on their final relevance scores, and the top- _K_ pages are selected for the subsequent question-answering phase. 

## **3.3 Training-required Variant** 

While the MoLoRAG framework allows the use of a pre-trained VLM as an off-the-shelf solution for fast deployment, we propose an enhanced variant, **MoLoRAG+**[3] . This variant fine-tunes the VLM (retrieval engine) to bolster its reasoning capabilities during graph traversal, enabling the model to assign more accurate logical relevance scores. 

**Data Preparation (Figure 3)** The success of finetuning relies on the availability of high-quality training data (Sun et al., 2024). To achieve this, we utilize GPT-4o (OpenAI, 2024) as a data generation engine to create reliable triplets in the format _⟨_ Question _,_ Image _,_ Relevance_Score _⟩_ , where the Relevance_Score quantifies the alignment between the question and the image content. These triplets serve as supervision signals for fine-tuning, enabling the model to better estimate logical relevance. The data creation process begins by randomly selecting a page snapshot (image) from the document and sampling a relevance score from a pre-defined range. Using both the selected image and the sampled relevance score as context, GPT4o generates a question that reflects the degree to which the selected image can answer it. GPT-4o then predicts the relevance score between the generated question and the image, enabling an automated **quality-checking** : only samples where the predicted score and the target score closely match (e.g., within a tolerance of _≤_ 1) are retained. To 

- 3We use **MoLoRAG+** to denote the fine-tuned version. 

further ensure accuracy, these filtered samples undergo manual verification, ensuring that the final dataset fully aligns with the task’s requirements. Note that the data engine can be replaced with any arbitrary LVLMs instead of proprietary models like GPT-4o to reduce costs. An additional analysis is provided in Appendix C. 

**Model Training** Using the curated dataset, we fine-tune the backbone VLM with supervised finetuning (SFT) techniques (Hu et al., 2022). Detailed training configurations are provided in Appendix C. After fine-tuning, the updated VLM replaces the original pre-trained model as the retrieval engine. By incorporating enhanced logical checking capabilities, this variant is expected to deliver more accurate retrieval performance. 

## **3.4 Summary** 

In the proposed MoLoRAG framework, the top- _K_ scored pages are fed into an LVLM during the question-answering phase, ensuring that only the most relevant information is utilized. Its key strengths include compatibility with arbitrary LVLMs, making it particularly adaptable for models limited to processing a single image by transforming an otherwise infeasible task into a practical solution. By incorporating both semantic and logical relevance, the framework enhances retrieval accuracy (Section 4.3). Furthermore, the graph-based traversal mechanism effectively narrows the search space, prioritizing relevant pages and significantly accelerating the retrieval process compared to exhaustive page-by-page traversal (Appendix D.4). Collectively, these features position MoLoRAG as a powerful solution for the DocQA task. 

## **4 Experiments** 

## **4.1 Experimental Setup** 

**Datasets** We utilize four datasets from three benchmarks for evaluation, including **MMLongBench** (Ma et al., 2024b), **LongDocURL** (Deng et al., 2024), and **PaperTab** and **FetaTab** from the UDA-Benchmark (Hui et al., 2024). Dataset statistics are shown in Table 1. These datasets span a wide range of topics (e.g., administrative files, tutorials, research reports) and feature diverse multi-modal elements (e.g., chart, text, and table). Additionally, they vary in average document length and information density, ensuring a comprehensive evaluation. Other benchmarks like DocVQA (Mathew et al., 2021; Tanaka et al., 2023; Masry 

Table 1: **Statistics of experimental datasets.** 

|**Dataset**|#**Question**|#**Document**|**Avg. Pages**|**Avg. Tokens**|
|---|---|---|---|---|
|**PaperTab**<br>**FetaTab**<br>**MMLongBench**<br>**LongDocURL**|393<br>1,016<br>1,082<br>2,325|307<br>871<br>135<br>396|11.0<br>15.8<br>47.5<br>85.6|12,685.4<br>16.524.5<br>24,992.6<br>56,715.1|



et al., 2022) are omitted due to their shorter document lengths and lower information density. 

**Evaluation Metrics** For MMLongBench and LongDocURL, we follow their original evaluation protocol, using a generalized **Accuracy** with rule-based evaluation to handle various answer types. Additionally, we report **Exact Match (EM)** as a supplementary metric, as the answers in these datasets are typically short and concise. For PaperTab and FetaTab, where ground-truth answers are formulated as long sentences or multiple choices, we follow MDocAgent to employ GPT-4o as the evaluator. Specifically, it evaluates **Binary Correctness** by determining whether the generated answer matches the ground-truth answer, assigning a binary score of 0 or 1. We also evaluate retrievalstage accuracy using metrics like Recall@ _K_ , with further details provided in Appendix D.1. 

**Baselines** We consider the following baselines: (1) **LLM w. Text RAG** first converts the document into texts using OCR and then applies retrieval techniques to the text, with LLMs serving as the backbone for question answering. (2) **LVLM Direct Inference** directly feeds LVLMs with full document snapshot images for question answering. For LVLMs that only support single-image input, we follow Ma et al. (2024b) by concatenating all images into a single combined one. (3) **M3DocRAG** (Cho et al., 2024) uses ColPali as a page retriever to identify relevant pages and feeds only the retrieved pages to the LVLM for further processing. (4) **MDocAgent** (Han et al., 2025) is a strong baseline for document understanding that employs a multi-agent system. A text agent and an image agent independently handle their respective modalities and collaborate to synthesize the final answer. Due to space limits, implementation details of each method are provided in Appendix D.2. 

**Choices of LLMs** For LLMs, we consider Mistral7B-Instruct-v0.2 (Jiang et al., 2023), Qwen2.57B-Instruct (Qwen, 2025), LLaMA3.1-8B-Instruct (Grattafiori et al., 2024), GPT-4o, and DeepSeekV3 (DeepSeek-AI, 2025). These models vary in series, scales, reasoning capabilities, and open-source availability, offering a diverse evaluation of LLM- 

based methods. 

**Choices of LVLMs** We classify LVLMs into three categories based on their input capacity: (1) Large Input Size models, such as Qwen2.5-VL-3B and Qwen2.5-VL-7B, which can process extensive context sizes, e.g., 30 images. (2) Medium Input Size models, such as DeepSeek-VL-16B (Lu et al., 2024), which can handle a moderate number of inputs, e.g., 5 images. (3) Single Input models, such as LLaVA-Next-7B, which are limited to processing one image at a time. For each LVLM backbone, we assess its compatibility with various methods and sensitivity to context size, providing guidelines for effectively leveraging LVLMs in document understanding tasks. 

## **4.2 Overall Performance** 

In this subsection, we present the overall performance of MoLoRAG alongside all baseline methods. To evaluate performance under varying retrieval availability, we consider top- _K_ values of _K_ = 1 _,_ 3 _,_ 5. Results for top-3 retrieval are shown in Table 2, while additional results for _K_ = 1 and _K_ = 5 are provided in Tables 9 and 10 in Appendix, respectively. For LLM w. Text RAG, each retrieved element corresponds to a text chunk, whereas for LVLM-based methods, each retrieved element represents a document page in image format. Based on the experimental results, we summarize the key findings below: 

## **1. LLMs struggle with document understanding** 

**compared to LVLM-based methods.** Even advanced LLMs like DeepSeek-V3, fall short in performance compared to LVLM-based methods. This highlights the inherent limitations of LLMs in handling multi-modal document understanding tasks, even when paired with sophisticated retrieval methods. LVLMs, on the other hand, can natively handle multi-modal inputs, making them better suited for document understanding. A fine-grained analysis across different evidence modalities (e.g., text, tables, figures) in Appendix D.7 reveals LLMs’ weak performance with non-text modalities, while LVLMs excel across diverse modalities. 

## **2. MoLoRAG consistently boosts LVLM per-** 

**formance.** Integrating LVLMs with MoLoRAG significantly improves their question answering capabilities. For example, DeepSeek-VL-16B, which performs poorly with concatenated document images (e.g., 8.40% on MMLongBench due to content overload), achieves a substantial improvement when paired with MoLoRAG, reaching 20.43%. 

Table 2: **Overall performance comparison (in** % **) under the retrieved top-** 3 **setting.** The “Direct” mode processes up to 30 document pages, while “MoLoRAG+” refers to the variant with a fine-tuned retrieval engine. Results for the top-1 and top-5 settings are in Tables 9 and 10, respectively. The best performance is **highlighted** . 

|**Type**|**Type**|**Model**<br>**Method**|**Model**<br>**Method**|**MMLongBench**<br>**LongDocURL**<br>**PaperTab**<br>**FetaTab**|**Avg**|**.**<br>3<br>6<br>5<br>1<br>**9**<br><br>**2**<br>7<br>3<br>4<br>7<br>7<br>**9**<br>8<br>6<br>0<br>**3**<br>5<br>7<br>0<br>**7**<br>5|
|---|---|---|---|---|---|---|
|**_LLM-based_**||Mistral-7B<br>Text RAG<br>Qwen2.5-7B<br>Text RAG<br>LLaMA3.1-8B<br>Text RAG<br>GPT-4o<br>Text RAG<br>DeepSeek-V3<br>Text RAG||24.47<br>25.06<br>11.45<br>41.14<br>25.52<br>27.93<br>12.72<br>40.06<br>22.56<br>29.80<br>13.49<br>45.96<br>27.23<br>32.74<br>14.25<br>50.20<br>**29.82**<br>**34.73**<br>**17.05**<br>**52.36**|25.5<br>26.5<br>27.9<br>31.1<br>**33.4**||
|**_LVLM-based_**||LLaVA-Next-7B<br>Direct<br>7.15<br>10.78<br>3.05<br>11.61<br>8.15<br>M3DocRAG<br>**10.10**<br>**13.85**<br>5.34<br>**13.98**<br>**10.8**<br>MoLoRAG<br>9.37<br>13.49<br>4.83<br>13.78<br>10.3<br>MoLoRAG+<br>9.47<br>13.58<br>**5.60**<br>13.48<br>10.5<br>DeepSeek-VL-16B<br>Direct<br>8.40<br>14.72<br>6.11<br>16.14<br>11.3<br>M3DocRAG<br>18.12<br>29.60<br>7.89<br>27.07<br>20.6<br>MoLoRAG<br>20.43<br>29.98<br>9.67<br>38.98<br>24.7<br>MoLoRAG+<br>**25.47**<br>**37.21**<br>**10.94**<br>**41.54**<br>**28.7**<br>Qwen2.5-VL-3B<br>Direct<br>26.65<br>24.89<br>25.19<br>51.57<br>32.0<br>M3DocRAG<br>29.11<br>44.40<br>24.68<br>53.25<br>37.8<br>MoLoRAG<br>32.11<br>**45.79**<br>24.43<br>57.68<br>40.0<br>MoLoRAG+<br>**32.47**<br>45.27<br>**27.23**<br>**58.76**<br>**40.9**<br>Qwen2.5-VL-7B<br>Direct<br>32.77<br>26.38<br>29.77<br>64.07<br>38.2<br>M3DocRAG<br>36.18<br>49.03<br>28.50<br>63.78<br>44.3<br>MoLoRAG<br>39.28<br>51.71<br>**32.32**<br>69.09<br>48.1<br>MoLoRAG+<br>**41.01**<br>**51.85**<br>31.04<br>**69.19**<br>**48.2**||7.15<br>10.78<br>3.05<br>11.61<br>**10.10**<br>**13.85**<br>5.34<br>**13.98**<br>9.37<br>13.49<br>4.83<br>13.78<br>9.47<br>13.58<br>**5.60**<br>13.48|8.15<br>**10.8**<br>10.3<br>10.5||
|||||8.40<br>14.72<br>6.11<br>16.14<br>18.12<br>29.60<br>7.89<br>27.07<br>20.43<br>29.98<br>9.67<br>38.98<br>**25.47**<br>**37.21**<br>**10.94**<br>**41.54**|11.3<br>20.6<br>24.7<br>**28.7**||
|||||26.65<br>24.89<br>25.19<br>51.57<br>29.11<br>44.40<br>24.68<br>53.25<br>32.11<br>**45.79**<br>24.43<br>57.68<br>**32.47**<br>45.27<br>**27.23**<br>**58.76**|32.0<br>37.8<br>40.0<br>**40.9**||
|||||32.77<br>26.38<br>29.77<br>64.07<br>36.18<br>49.03<br>28.50<br>63.78<br>39.28<br>51.71<br>**32.32**<br>69.09<br>**41.01**<br>**51.85**<br>31.04<br>**69.19**|38.2<br>44.3<br>48.1||
||||||**48.2**||
|**_Multi-agent_**||MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)<br>38.53<br>46.91<br>30.03<br>66.34<br>45.4|||||
||||**(b) Qwen2.5-VL-3B**<br>**(c) DeepSeek-VL-16B**<br>**(d) LLaVA-Next-7B**||||
||||||||
||||||||
||||||||
||**(a) Qwen2.5-VL-7B**||||||



Figure 4: **Performance trends of MoLoRAG across different top-** _K_ **retrieval settings for LVLMs.** LVLMs with extensive context support (e.g., Qwen2.5-VL series) benefit from retrieving more pages, improving performance with higher _K_ . In contrast, LVLMs with limited context capacity (e.g., LLaVA-Next-7B) perform best with _K_ = 1. 

Similarly, high-capacity LVLMs like the Qwen2.5VL series benefit from MoLoRAG’s ability to filter and prioritize relevant pages, further improving their already strong performance. 

**3. Fine-tuned MoLoRAG+ delivers further performance gains.** The fine-tuned variant, MoLoRAG+, outperforms the training-free version, demonstrating the benefits of task-specific optimization. For example, with DeepSeek-VL16B, MoLoRAG+ achieves 5.04% improvement on MMLongBench compared to the training-free MoLoRAG. This enhancement stems from its superior ability to assess logical relevance, enabling more accurate retrieval (details in Section 4.3). 

## **4. The relationship between top-** _K_ **retrieval and performance depends on LVLM capability.** Fig- 

ure 4 illustrates how performance of MoLoRAG varies with top- _K_ retrieval settings across four datasets. For LVLMs with extensive context support, such as the Qwen2.5-VL series, increasing _K_ (i.e., providing more pages) improves performance across all scenarios. However, for LVLMs with limited context capacity, such as DeepSeek-VL16B and LLaVA-Next-7B, additional pages often exceed their processing capabilities, leading to degraded performance. For these models, _K_ = 1 is typically the optimal choice. 

## **4.3 Retrieval Performance Comparison** 

In this subsection, we evaluate the retrieval accuracy of various methods to highlight the effectiveness of MoLoRAG in identifying relevant pages. 

Table 3: **Retrieval performance comparison (in** % **) under the top-** _K_ **setting.** 

**==> picture [417 x 384] intentionally omitted <==**

**----- Start of picture text -----**<br>
||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|MMLongBench|LongDocURL|
|Top-|K|Method|
|Recall|Precision|NDCG|MRR|Recall|Precision|NDCG|MRR|
|M3DocRAG|43.31|56.67|56.67|56.67|46.84|64.66|64.66|64.66|
|MDocAgent (Text)|29.30|38.99|38.99|38.99|42.03|58.37|58.37|58.37|
|1|MDocAgent (Image)|43.79|57.49|57.49|57.49|46.80|64.57|64.57|64.57|
|MoLoRAG|45.46|59.95|59.95|59.95|48.98|67.71|67.71|67.71|
|MoLoRAG+|51.32|66.86|66.86|66.86|50.82|70.08|70.08|70.08|
|M3DocRAG|64.17|31.62|54.13|65.36|67.00|33.78|58.23|72.51|
|MDocAgent (Text)|43.21|20.77|37.13|45.26|58.53|29.33|54.12|65.28|
|3|MDocAgent (Image)|64.74|31.97|54.75|66.12|66.67|33.62|58.26|72.47|
|MoLoRAG|67.22|40.81|57.34|68.56|70.04|36.41|61.56|75.78|
|MoLoRAG+|68.87|48.67|64.49|73.50|68.92|47.53|64.90|77.14|
|M3DocRAG|72.00|22.58|54.06|66.92|74.32|23.34|58.05|73.83|
|MDocAgent (Text)|50.60|15.48|37.19|46.98|65.41|20.41|53.97|66.55|
|5|MDocAgent (Image)|71.45|22.37|54.58|67.53|74.60|23.50|58.06|73.90|
|MoLoRAG|74.13|35.83|57.29|69.63|77.14|26.13|61.30|76.88|
|MoLoRAG+|72.37|45.34|64.36|73.97|73.69|42.47|64.74|77.89|
|ponoH!Li|Question|What is the total square footage of office space for Gray Television’s newspaper publishing operations in Georgia and Indiana?|EEEno|nnEEEnn|en ones nn|on|nn nn|on|nn|nn|nn|nyII|
|Evidence Page|ogThe‘Aaay, EtveetySeesparerAtseryOA Lecotion.Herd ane|OferVerafay feadWve The produces|Index 36|Aihany|wmaeaeeOwenLense||[MeesecaCapproinatem0|||Papirationwe Date|||DP|||Qwen2.5-VL-7B DirectUnanswerable|“|based solely on the given informationIt is not possible to provide an exact total square footage|”||i||||
|cw|Gk|Othemts|owt|||amo||||1|ee!||
|Rackaae|Cen|||----|
|rerewoaeies|Newee Chon|tat|Lone|||sa|I[onan|w. M3DocRAG|nanan|nn|nnn|nnn|nnyIpon|MDocAgent|a5|55-55|=------- === --------5|
|facilityecmGuowett for|theDusty|Post|a||||was|||mae||||Retrieved Pages|[15, 42, 119]|7I|Retrieved Pages|Text [119, 2, 4]  Image [15, 42, 119]||||
|tindNewton RochdaleCotten Cason|||||||Unanswerable|x|it|Unanswerable|x|||
|Gone,|[IN]|fatayOttisGoshen hwdN|e|wnereduction|— Owned|||||en21,000|||||||“ There is insufficient information|provided to determine the total square|II1|||“ There is insufficient information provided to|determine the total square footable of office space for|||
|Mem 2|Legal Proceedings|||footage of office space for … ”|II||Gray Television’s newspaper … ”|
|Setement(nenowyof lacemimormeTat|Mater|ractanindatingtettemttemetnig |pue|
|Aligedtemaed{hechamcteriandIWS"IncomactieayesScpreciaton‘Petition mattnthe“TRS Jarry inthatthetoCompany ofthecoeframwithredationanforwah Company's204, te eectthe Cospany’s the mater recogaizedsemtumtiotaxs|Company.hnuertspurposesofthe Companyon inthe Seral enone1996 «gett1spina1996deaconas on nlnThe ha tiedsalUnited StatesCompanyacquisition 1998198fnameteating of cera fertxitionfree Fr ret Taxayrecdofcf eperting10 Courstock,leme exaheAencan witthestsln the IRS1998 axtoOn yaa los fomthe rrr JunmaryMediaeurethraghmienast caryforemns ropingreFstbx AmericanUe"The1K,Toe19%?Thesale 2002,altenloiwwomen mtaetae byofforego cern 9 oppestinanelyMedia,yar,the ecleney Companyofverecertan Sfciency aoswhch alaeeso aegedclaimedinweified cdhS16.wo1986 theatre|l|beso»||IiI||||w. MoLoRAGRetrieved Pages|Answer|Television’s newspaper publishing operations in Georgia and Indiana, we need to sum up the sizes|199,750|VY|[|36|“|, 15]To determine the total square footage of office space for Gray|VY|||||||||
|tnlion.Grcreas isDuring deterredthe ree tax ests mois by ended appctiattyDecember $8.M1,|200%lion|theofl Companyds stentrecoded|tt-cahDating 2008 hare en|I|provided for each location. From the first page: The Albany Herald (Albany, GA): 83,000 sq. ft.|||
|‘=u‘overly ste selon,$|1|wl|pothe saa Conarytere, eee!»ele reviuly|lated fdr|at|Cash tnd|of|||Conyers, GA: 20,000 sq. ft. Covington, GA: 3,750 sq. ft. Lawrenceville, GA: 72,000 sq. ft. Goshen,|||
|ter Log|receotings|wo196||I|IN: 21,000 sq. ft. Adding these together: 83,000 + 20,000 + 3,750 + 72,000 + 21,000 = 199,750.|||
|teiliowTh neeCampy cle,i mtaberay todayettw|pontUa|merce,in wihpe|the Comgunyene acon excepweae dent ta|e|d|lLI|||Therefore, the total square footage of office space for Gray Television's newspaper publishing|operations in Georgia and Indiana is 199,750 square feet.”|eeei|

**----- End of picture text -----**<br>


Figure 5: **Case study on LongDocURL.** MoLoRAG successfully **retrieves the correct evidence page** for the given question by leveraging logical relevance, enabling it to **provide the correct answer** . In contrast, both LVLM direct inference and other baseline methods fail to answer the question due to limited or irrelevant context. 

Since only MMLongBench and LongDocURL provide ground-truth evidence pages for each query, our comparison is confined to these two datasets. We employ standard metrics, including Recall, Precision, NDCG, and MRR (details in Appendix D.1), where higher values indicate better retrieval performance. We compare MoLoRAG and its fine-tuned variant, MoLoRAG+, against two baseline methods: M3DocRAG (Cho et al., 2024) and MDocAgent (Han et al., 2025). MDocAgent performs separate text- and image-based retrieval, and results for both modalities are reported. The detailed results under top- _K_ settings are presented in Table 3. MoLoRAG consistently outperforms baseline methods across metrics, with an average improvement of **9.94%** on MMLongBench and **7.16%** on LongDocURL. This advantage arises from MoLoRAG’s integration of both semantic and 

logical relevance, unlike the baselines, which focus solely on semantic relevance. The fine-tuned variant, MoLoRAG+, further improves performance by leveraging task-specific optimization. 

## **4.4 Case Study** 

Figure 5 presents a case study on LongDocURL. LVLM direct inference marks the question as “unanswerable” due to limited input context. Baselines such as M3DocRAG and MDocAgent rely solely on semantic relevance for retrieval, failing to locate the evidence page, which leads to incorrect answers. In contrast, MoLoRAG accurately retrieves the evidence page by considering logical relevance, enabling the LVLM to leverage this knowledge and correctly answer the question. Another case involving **cross-page understanding** is illustrated in Figure 8 in the Appendix. 

Due to space limits, **ablation study** and **efficiency analysis** are moved to Appendix D.6 and D.4. 

## **5 Conclusion** 

In this paper, we tackle the DocQA task by addressing the limitations of prior methods that rely only on semantic relevance for retrieval. By incorporating logical relevance, our VLM-powered retrieval engine performs multi-hop reasoning over page graph to identify key pages. Extensive experiments demonstrate that MoLoRAG delivers superior retrieval accuracy, achieves SOTA performance, and ensures seamless compatibility with LVLMs. 

## **Acknowledgments** 

This research is supported in part by project #MMTp2-23 of the Shun Hing Institute of Advanced Engineering, The Chinese University of Hong Kong, by grants from the Research Grants Council of the Hong Kong SAR, China (No. CUHK 14217622). This research is also supported in part by the National Natural Science Foundation of China (No.62302098), and the Fujian Provincial Natural Science Foundation of China (2025J01540). The authors would like to express their gratitude to the reviewers for their valuable feedback, which has improved the clarity and contribution of the paper. 

## **Limitations** 

MoLoRAG primarily focuses on closed-domain document understanding, where the relevant document is provided. Extending this approach to an **open-domain** setting, where the document corpus consists of extensive and diverse documents, is a challenge. This is because modeling relationships not only within individual document but also across different documents, as well as performing graph traversal between both document- and page-level nodes, becomes complex. 

In this paper, we did not use any non-public data, unauthorized software, or APIs, and there are no privacy or other related ethical concerns associated with our work. 

## **References** 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ . 

- Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_ . 

- Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen. 2022. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. _Preprint_ , arXiv:2210.02928. 

- Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. 2024. Mllm is a strong reranker: Advancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and noise-injected training. _arXiv preprint arXiv:2407.21439_ . 

- Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multimodal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ . 

- DeepSeek-AI. 2025. Deepseek-v3 technical report. _Preprint_ , arXiv:2412.19437. 

- Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, ZhongZhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, and Cheng-Lin Liu. 2024. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. _Preprint_ , arXiv:2412.18424. 

- Yihao Ding, Zhe Huang, Runlin Wang, YanHang Zhang, Xianru Chen, Yuzhong Ma, Hyunsuk Chung, and Soyeon Caren Han. 2022. V-doc: Visual questions answers with documents. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 21492–21498. 

- Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2025. From local to global: A graph rag approach to query-focused summarization. _Preprint_ , arXiv:2404.16130. 

- Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. _Preprint_ , arXiv:2407.01449. 

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. _Preprint_ , arXiv:2312.10997. 

- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ . 

- Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: A multi-modal multi-agent framework for document understanding. _arXiv preprint arXiv:2503.13964_ . 

- Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ . 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ . 

- Yulong Hui, Yao Lu, and Huanchen Zhang. 2024. UDA: A benchmark suite for retrieval augmented generation in real-world document analysis. In _The Thirtyeight Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ . 

- Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L’elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. _ArXiv_ , abs/2310.06825. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks. _Preprint_ , arXiv:2005.11401. 

- Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. 2024a. Llava-next: Stronger llms supercharge multimodal capabilities in the wild. 

- Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yangguang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng. 2024b. GraphReader: Building graph-based agent to enhance long-context abilities of large language models. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pages 12758–12786, Miami, Florida, USA. Association for Computational Linguistics. 

- Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and Wentao Zhang. 2025. Hoprag: Multi-hop reasoning for logic-aware retrievalaugmented generation. _Preprint_ , arXiv:2502.12442. 

- Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. 2024. Deepseek-vl: Towards real-world vision-language understanding. _Preprint_ , arXiv:2403.05525. 

- Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, and Jimmy Lin. 2024a. Visa: Retrieval augmented generation with visual source attribution. _arXiv preprint arXiv:2412.14457_ . 

- Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024b. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Preprint_ , arXiv:2407.01523. 

- Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 2263– 2279, Dublin, Ireland. Association for Computational Linguistics. 

- Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209. 

- Jamshed Memon, Maira Sami, Rizwan Ahmed Khan, and Mueen Uddin. 2020. Handwritten optical character recognition (ocr): A comprehensive systematic literature review (slr). _IEEE Access_ , 8:142642–142668. 

- OpenAI. 2024. Gpt-4o system card. _Preprint_ , arXiv:2410.21276. 

- Qwen. 2025. Qwen2.5 technical report. _Preprint_ , arXiv:2412.15115. 

- Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. Raptor: Recursive abstractive processing for tree-organized retrieval. In _International Conference on Learning Representations (ICLR)_ . 

- Jianwei Sun, Chaoyang Mei, Linlin Wei, Kaiyu Zheng, Na Liu, Ming Cui, and Tianyi Li. 2024. Dial-insight: Fine-tuning large language models with high-quality domain-specific data preventing capability collapse. _Preprint_ , arXiv:2403.09167. 

- Manan Suri, Puneet Mathur, Franck Dernoncourt, Kanika Goswami, Ryan A Rossi, and Dinesh Manocha. 2024. Visdom: Multi-document qa with visually rich elements using multimodal retrieval-augmented generation. _arXiv preprint arXiv:2412.10704_ . 

- Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In _AAAI_ . 

- Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multi-page docvqa. _Preprint_ , arXiv:2212.05935. 

- Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. 2024a. Mineru: An open-source solution for precise document content extraction. _Preprint_ , arXiv:2409.18839. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024b. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _Preprint_ , arXiv:2409.12191. 

- Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, and Xiangyu Zhang. 2024. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. _Preprint_ , arXiv:2409.01704. 

- Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and Jinsong Su. 2025. When to use graphs in rag: A comprehensive analysis for graph retrieval-augmented generation. _arXiv preprint arXiv:2506.05690_ . 

- Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. 2024. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented generation. _arXiv preprint arXiv:2412.02592_ . 

- Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, and Xiao Huang. 2025. A survey of graph retrieval-augmented generation for customized large language models. _arXiv preprint arXiv:2501.13958_ . 

## **Algorithm 1 Graph Traversal for Retrieval** 

**Require:** Question _q_ , DocEncoder( _·_ ), Page graph _G_ , Exploration size _w_ , Hop limit _n_ hop, Document _D_ 

**Ensure:** Re-ranked pages _D[r]_ 

- 1: _s_[sem] _i ←⟨_ DocEncoder( _q_ ) _,_ DocEncoder( _pi_ ) _⟩_ for each _pi ∈D ▷_ Semantic relevance scoring 

- 2: _B ←_ TopK( _{s_[sem] _i }, w_ ) _▷_ Exploration set initialization 

3: _D[r] ←∅_ 

- 4: _S ←B ▷_ Visited marking 

**==> picture [214 x 66] intentionally omitted <==**

- 9: **end for** 

10: **for** Hop = 1 to _n_ hop **do** 

**==> picture [218 x 152] intentionally omitted <==**

22: **end for** 

23: Sort _D[r]_ by descending _s ▷_ Pages re-ranking 

24: **return** _D[r]_ 

## **A Algorithm Pseudo-Code** 

The graph traversal algorithm for retrieval is presented in Algorithm 1. This algorithm operates by efficiently identifying relevant pages through a combination of semantic and logical relevance scores. By leveraging an exploration size _w_ and a hop limit _n_ hop, the traversal is restricted to exploring only the most promising paths in the page graph, ensuring scalability and avoiding the need to process all pages. The output, _D[r]_ , is a re-ranked set of document pages that are both semantically and logically relevant, with its size typically smaller than the total number of document pages _N_ . From this re-ranked set, the top- _K_ pages are selected and passed to the next stage for question answering. 

## **B Prompt** 

In this section, we present all the prompts used within the MoLoRAG framework, including querying the VLM to assign logical relevance scores, using LLMs or LVLMs for question answering, and prompting GPT-4o to curate the dataset for training MoLoRAG+. 

**Assessing Logical Relevance** The prompt for querying the VLM to assign a logical relevance score between the observed image and the question is provided below: 

## **Prompt for Assessing Logical Relevance** 

## **# GOAL #** 

You are an Retrieval Expert, and your task is to evaluate how **relevant** the input document page is to the given query. Rate the relevance on a scale of 1 to 5, where: 

- **5** Highly relevant - contains complete information needed to answer the query 

- **4** Very relevant - contains most of the information needed 

- **3** Moderately relevant - contains some useful information 

- **2** Slightly relevant - has minor connection to the query 

**1** Irrelevant - contains no information related to the query 

## **# INSTRUCTION #** 

Please first read the given query, think about what knowledge is required to answer that query, and then carefully go through the document snapshot for judgment. 

## **# QUERY #** 

_{{_ Question _}}_ 

Please generate just a single number (1-5) representing your relevance judgment. Your answer should be a single number without any extra contents. 

**LLM Question Answering** For LLM-based question answering, all retrieved text chunks are concatenated into the context, and the LLM is queried to answer the given question based on the provided context as:“ _{{_ Context _}}_ Answer the question based on the above context: _{{_ Question _}}_ ”. 

**LVLM Question Answering** For LVLM-based question answering, the input includes document images and the question: “ _{{_ Image _}}_ Based on the document, please answer the question: 

_{{_ Question _}}_ ”. 

**Curating Training Data** The prompt for guiding GPT-4o to generate training data is shown below: 

## **Prompt for Curating Training Data** 

## **# GOAL #** 

Given the input image, your task is to generate a question related to it. The relevance score is _{{_ relevance_score _}}_ , where a higher score indicates a closer connection between the question and the image. For example, a relevance score of 5 means the answer is **DIRECTLY** contained in the image, while a score below 3 indicates that the answer **CANNOT** be derived from it, with lower scores signifying less relevance. 

## **# REQUIREMENT #** 

The question must be based on the content of the input image, except when the relevance score is _≤_ 2. For relevance scores of 4 or higher, create clear and straightforward questions with answers that are explicitly present in the image. For relevance scores of 3, generate questions that may require some inference but are still somewhat related to the content. For relevance scores of 2 or lower, formulate questions that are unanswerable based on the snapshot. You may consider various elements, including text, layout, and figures. For this generation, please concentrate on _{{_ focus _}}_ if applicable and remember that the relevance score is _{{_ relevance_score _}}_ . 

Your output should be formatted as follows: { “query”: “Your generated question”, “relevance_score”: “relevance score”, “answer”: “Corresponding answer or inference” } 

## **C Supplementary Materials for MoLoRAG+** 

This section provides detailed information on the training data, learning configurations, and alternative data engine for MoLoRAG+. 

**Training Data** In the first stage, **Question Generation** , we prompt GPT-4o to generate approximately 5,500 samples using the illustrated prompt. Document snapshots are randomly selected from MMLongBench (Ma et al., 2024b) and LongDocURL (Deng et al., 2024), as these datasets contain multimodal, information-rich documents. The relevance 

score _s_ is sampled from _{_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 5 _}_ with equal distribution to ensure a balanced representation across different levels of logical relevance, preventing over-fitting to specific scores. For each sampled document snapshot _pi_ , the generated question is expressed as: _q[′]_ = GPT-4o(prompt _, s, pi_ ), where _pi_ denotes the randomly selected document snapshot, and _s_ is the sampled relevance score. To ensure data quality, each generated question _q[′]_ and its corresponding document snapshot _pi_ are fed back to GPT-4o to assign a predicted logical relevance score as: _s[′]_ = GPT-4o(prompt _, q[′] , pi_ ). Samples are retained only if the predicted score closely matches the original score, i.e., _|s − s[′] | ≤_ 1. After this automated filtering, the remaining samples undergo manual verification. This process results in a final high-quality training set containing 3,519 samples, formatted as triplets: _⟨_ Question _q[′] ,_ Image _pi,_ Relevance Score _s[′] ⟩_ . In each sample, _s[′]_ , the predicted relevance score, is the expected output for model training. 

**Learning Configuration** We utilize the LLaMAFactory package[4] to fine-tune the backbone VLM, Qwen2.5-VL-3B (Bai et al., 2025), using the LoRA technique for parameter-efficient training. The finetuning process is configured with the following hyperparameters: a LoRA Rank of 8, a learning rate of 1 _×_ 10 _[−]_[4] , a warmup ratio of 0.1, and gradient accumulation steps set to 8. The model is trained for a total of 2 epochs, ensuring efficient and effective parameter adaptation. 

**Alternative Data Engine** We primarily use GPT4o as the data engine for curating training data. However, our data generation pipeline is flexible and **can accommodate other LVLMs** . To demonstrate this flexibility, we replace the original engine with the open-source model Qwen2.5VL-32B (Bai et al., 2025), while keeping all prompts and processes consistent. This variant is denoted as MoLoRAG[+][We][compare][the] Qwen[.] retrieval performance of MoLoRAG, MoLoRAG+, and MoLoRAG[+] Qwen[on MMLongBench (Table][ 4][),] where the backbones are pre-trained Qwen2.5-VL3B, Qwen2.5-VL-3B fine-tuned with GPT-4o data, and fine-tuned with Qwen2.5-VL-32B data, respectively. The results indicate that the Qwen2.5-VL32B distilled model performs **comparably** to its GPT-4o counterparts. This is attributed to (1) the simplicity of logical relevance scoring, enabling effective high-quality data generation by Qwen2.5- 

4https://github.com/hiyouga/LLaMA-Factory/ 

Table 4: **Retrieval performance comparison (in** % **)** between MoLoRAG, MoLoRAG[+] Qwen[, and MoLoRAG+] on MMLongBench. 

|**Top-**_K_<br>**Model**|**Recall**<br>**Precision**<br>**NDCG**<br>**MRR**|
|---|---|
|1<br>MoLoRAG<br>MoLoRAG+<br>Qwen<br>MoLoRAG+|45.46<br>59.95<br>59.95<br>59.95<br>**51.62**<br>**67.56**<br>**67.56**<br>**67.56**<br>51.32<br>66.86<br>66.86<br>66.86|
|3<br>MoLoRAG<br>MoLoRAG+<br>Qwen<br>MoLoRAG+|67.22<br>40.81<br>57.34<br>68.56<br>**71.79**<br>37.94<br>64.24<br>**74.57**<br>68.87<br>**48.67**<br>**64.49**<br>73.50|
|5<br>MoLoRAG<br>MoLoRAG+<br>Qwen<br>MoLoRAG+|74.13<br>35.83<br>57.29<br>69.63<br>**78.72**<br>29.06<br>64.01<br>**75.69**<br>72.37<br>**45.34**<br>**64.36**<br>73.97|



VL-32B, and (2) the shared family of the data engine and distilled model, which facilitates capability transfer. Additionally, this variant reduces data construction costs due to its open-source nature, highlighting the effectiveness of our data construction pipeline. 

## **D Supplementary Materials for Experiments** 

## **D.1 Details of Metrics** 

**Evaluation for Question Answering** We utilize **Accuracy** and **Exact Match** as evaluation metrics for MMLongBench (Ma et al., 2024b) and LongDocURL (Deng et al., 2024). Accuracy is rulebased to accommodate various answer types, with detailed explanations provided in Appendix B.3 of Ma et al. (2024b). Exact Match measures the percentage of predictions where the generated answer **exactly** matches the ground-truth answer. For the remaining datasets, PaperTab and FetaTab (Hui et al., 2024), we employ GPT-4o as the evaluator to assign a **Binary Correctness** score _∈{_ 0 _,_ 1 _}_ , where 1 indicates that the generated answer aligns with the ground-truth answer. We report the averaged values across all test samples. 

**Evaluation for Retrieval** We employ standard retrieval metrics, including Recall@ _K_ , Precision@ _K_ , NDCG@ _K_ , and MRR@ _K_ , where _K_ represents the number of retrieved elements. For a specific data sample with ground-truth evidence pages denoted as _P_[gt] = _{p_[gt] 1 _[, . . . , p] n_[gt] _[}]_[ and the top-] _K_ retrieved pages _P[r]_ = _{p[r]_ 1 _[, . . . , p][r] K[}]_[, these met-] rics are computed as follows: 

- **Recall** measures the proportion of groundtruth pages that are successfully retrieved within the top- _K_ results: 

**==> picture [146 x 27] intentionally omitted <==**

where I( _·_ ) is the indicator function that returns 1 if the condition is true and 0 otherwise. 

- **Precision** assesses the accuracy of the retrieved pages by calculating the proportion of retrieved pages that are relevant: 

**==> picture [159 x 26] intentionally omitted <==**

- **NDCG** (Normalized Discounted Cumulative Gain) evaluates the ranking quality of the retrieved pages by considering the positions of relevant pages within the top- _K_ results. It is computed in three steps: 

**==> picture [154 x 87] intentionally omitted <==**

**==> picture [111 x 23] intentionally omitted <==**

- **MRR** (Mean Reciprocal Rank) measures the reciprocal rank of the first relevant page within the top- _K_ retrieved pages. It is defined as: 

**==> picture [213 x 33] intentionally omitted <==**

where _i_ denotes the position of the **first relevant page** that satisfies _p[r] i[∈P]_[gt][.][If no rel-] evant page is retrieved within the top- _K_ , the MRR score for that sample is 0. 

For each metric, we average the scores over all test samples and report the values in Tables 3, 7 and 8, respectively. 

## **D.2 Implementation Details** 

This subsection outlines the detailed configurations for each method to ensure clarity and reproducibility. All experiments were conducted on 3 NVIDIA A6000 48G GPUs. 

- **LLM w. Text RAG** We use the PyPDFLoader from the LangChain package[5] to extract text from the raw document. Each document is divided into chunks of 1,000 tokens. The QwenRAG API[6] is employed as the text RAG engine. Specifically, we use the text-embedding-v1 model to encode text chunks and perform retrieval. For each top- _K_ setting, the top-ranked _K_ text chunks are combined as context, which is then passed to the LLM for answering. 

- **LVLM Direct Inference** To ensure scalability, we truncate documents to retain only the first 30 pages. For Qwen2.5-VL series, as these models support extensive image contexts, all images are fed directly for processing. For DeepSeek-VL-16B, although it supports multi-image inputs, it requires significant memory for loading. Therefore, we concatenate document images into 5 larger images, ensuring compatibility with GPU memory. For LLaVA-Next-7B, as it accepts only a single image, all available pages are combined into one single image for processing. 

- **M3DocRAG** This baseline is implemented according to its original paper and official repository. The document encoder used is colpali for encoding and retrieval. While the original paper uses Qwen2-VL-7B (Wang et al., 2024b) as the backbone LVLM, we extend the evaluation by integrating the method with various LVLMs to assess compatibility. 

- **MDocAgent** This baseline is implemented following its official repository: colpaligemma-3b-mix-448-base for image retrieval and colbertv2.0 for text retrieval. For all five agents in this framework, we consistently use the original LLaMA3.1-8B (Grattafiori et al., 2024) as the LLM for the text agent, while employing a consistent LVLM, i.e., Qwen2.5-VL-7B (Bai et al., 2025), for remaining agents. 

- **MoLoRAG** For document encoding, colpali is used as the document encoder. Pages are connected in the graph if their similarity score exceeds a threshold of 0.4. During graph traversal, the number of hops _n_ hop is set to 

- 5https://python.langchain.com/ 

- 6https://dashscope.console.aliyun.com/overview 

- 4, and the exploration set size _w_ is set to 3. Semantic relevance and LVLM-generated logical relevance are combined using an average score. All visited pages are re-ranked based on this combined score for final retrieval. Qwen2.5-VL-3B is employed as the retrieval engine due to its strong performance and lightweight architecture, ensuring both effectiveness and efficiency. The retrieved top- _K_ images are fed into LVLMs based on their input format capabilities: for LLaVA-Next7B, the images are concatenated into a single composite image, while for all other LVLMs, the images are processed separately. 

## **D.3 Discussion of Advanced OCR Methods** 

We expand our discussion on the LLM w. Text RAG baseline by using more advanced OCR tools, including MinerU (Wang et al., 2024a) and GOT-OCR-2.0 (Wei et al., 2024), as alternatives to PyPDFLoader, while ensuring consistency in the retrieval engine and LLM calling process. The results across different top- _K_ settings on MMLongBench and PaperTab are presented in Table 5. Our findings indicate that advanced tools generally enhance QA performance due to improved OCR capabilities, with the benefits becoming more pronounced as _K_ increases. For instance, replacing PyPDFLoader with MinerU yields performance gains of up to 4% across various LLMs. However, **LLMs w. Text RAG baselines still show a performance gap compared to strong LVLMs** , primarily due to the inevitable loss of multi-modal information. 

## **D.4 Efficiency Analysis** 

In this subsection, we provide efficiency analysis of our MoLoRAG with baseline methods from three aspects: (1) Retrieval scalability, (2) Inference efficiency, and (3) Total time costs. 

**Retrieval Scalability** The retrieval stage of MoLoRAG requires the VLM to evaluate the logical relevance score of each visited page. To efficiently manage the traversal scope, we introduce hyper-parameters such as hop limit and exploration set size, which help narrow the querying space, ensuring scalability and accelerating the retrieval process. To illustrate the scalability of MoLoRAG, Figure 6 shows the average number of queried pages alongside the total number of pages for all testing samples across different datasets. The figure also indicates the percentage of queried pages. As the document size increases, the average percentage 

Table 5: **Comparison of various OCR methods on the performance of LLM w. Text RAG.** 

|**Mdl**<br>**OCR**|**Top-**1|**Top-**3|**Top-**5|
|---|---|---|---|
|**oe**<br>|**MMLongBench**<br>**PaperTab**|**MMLongBench**<br>**PaperTab**|**MMLongBench**<br>**PaperTab**|
|**Qwen2.5-7B**<br>PyPDFLoader<br>GOT-OCR-2.0<br>MinerU|22.11<br>5.34<br>22.66<br>6.11<br>21.99<br>7.63|25.52<br>12.72<br>25.84<br>13.74<br>24.54<br>16.28|26.09<br>16.79<br>26.70<br>16.79<br>27.53<br>19.08|
|**GPT-4o**<br>PyPDFLoader<br>GOT-OCR-2.0<br>MinerU|24.07<br>8.65<br>23.86<br>8.91<br>25.89<br>9.92|27.23<br>14.25<br>27.78<br>13.74<br>28.74<br>18.32|28.74<br>20.36<br>29.47<br>18.07<br>30.98<br>22.90|
|**DeepSeek-V3**<br>PyPDFLoader<br>GOT-OCR-2.0<br>MinerU|25.94<br>10.18<br>25.35<br>9.92<br>26.21<br>11.96|29.82<br>17.05<br>28.39<br>18.83<br>30.11<br>21.12|31.23<br>23.92<br>30.75<br>22.90<br>32.36<br>26.97|
|**Qwen2.5-VL-7B w. MoLoRAG**|34.35<br>23.92|39.28<br>32.32|39.97<br>31.04|



**==> picture [220 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
PaperTab 93.9% # Total Page# Queried Page<br>FetaTab 83.1%<br>MMLongBench 58.7%<br>LongDocURL 46.1%<br>0 20 40 60 80<br>Number of Pages<br>**----- End of picture text -----**<br>


Figure 6: **Illustration of retrieval scalability.** We present the average number of pages queried by MoLoRAG and the total number of pages across all test samples for each dataset. 

of queried pages decreases significantly. For instance, fewer than 50% of the pages are queried for LongDocURL, demonstrating MoLoRAG’s ability to effectively control graph traversal and focus on relevant pages. This reduction highlights the scalability of the method, as it maintains high retrieval accuracy while minimizing computational overhead even for large documents. 

**Inference Efficiency** After retrieval, MoLoRAG requires only **a single query** to the LVLM for question answering by providing the relevant pages, ensuring efficiency comparable to LVLM Direct Inference. In contrast, the best-performing baseline, MDocAgent (Han et al., 2025), requires **five separate queries** to both LLMs and LVLMs, as it functions as a unified multi-agent system. Figure 7 illustrates the average inference times of MoLoRAG (Qwen2.5-VL-7B as the backbone) and MDocAgent, clearly highlighting MoLoRAG’s significant advantage in inference efficiency, making it a more practical and scalable solution. 

**Total Time Costs** In addition to inference time efficiency, we present the total time (retrieval and inference) for various methods to provide a clearer illustration. Note that the retrieval time includes 

**==> picture [180 x 149] intentionally omitted <==**

**----- Start of picture text -----**<br>
MoLoRAG (Ours)<br>30 MDocAgent 29.82s<br>21.50s<br>20<br>14.70s<br>10<br>7.15s<br>6.26s<br>5.05s<br>0<br>Top-1 Top-3 Top-5<br>Avg. Inference Time (Seconds)<br>**----- End of picture text -----**<br>


Figure 7: **Inference time comparison between MoLoRAG and MDocAgent (Han et al., 2025) on the MMLongBench.** 

both indexing and retrieval processes. DeepSeekV3 and GPT-4o are invoked via APIs in a sequential manner to ensure a fair comparison, while the remaining open-source LLMs are deployed locally using vLLM on a single NVIDIA A6000-48G GPU. For LVLM Direct, the number of pages considered is up to 30, with times reported using 3 A6000 GPUs. All other time costs are measured on a single A6000 GPU, except for MDocAgent, which utilizes 2 A6000 GPUs. The results are shown in Table 6. 

For LLM with Text RAG, variations in latency occur due to differences in invocation methods and model architecture. The index stage that invokes Qwen-RAG via APIs significantly affects overall latency. In LVLM-based methods, direct inference involves managing 30 pages, which can slow processing. While M3DocRAG is the most efficient method, it focuses solely on semantic relevance, limiting retrieval accuracy. The robust baseline, MDocAgent, employs parallel text and image retrieval; however, its multi-agent framework can 

Table 6: **Total time costs comparison (in seconds) across different methods on LongDocURL.** 

**==> picture [452 x 357] intentionally omitted <==**

**----- Start of picture text -----**<br>
||||||||||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Top-|1|Top-|5|
|Type|Model|Method|
|Retrieval Time|Inference Time|Total|Retrieval Time|Inference Time|Total|
|Mistral-7B|Text RAG|48.5s|2.4s|50.9s|48.5s|4.1s|52.6s|
|Qwen2.5-7B|Text RAG|48.5s|5.1s|53.6s|48.5s|6.9s|55.4s|
|LLM-based|LLaMA3.1-8B|Text RAG|48.5s|11.8s|60.3s|48.5s|18.3s|66.8s|
|GPT-4o|Text RAG|48.5s|1.4s|49.9s|48.5s|1.5s|50.0s|
|DeepSeek-V3|Text RAG|48.5s|7.5s|56.0s|48.5s|21.4s|69.9s|
|Direct|-|34.3s|34.3s|-|34.3s|34.3s|
|Qwen2.5-VL-3B|M3DocRAG|10.7s|5.7s|16.4s|10.7s|8.3s|19.0s|
|MoLoRAG|38.4s|5.7s|44.1s|38.4s|8.2s|46.6s|
|LVLM-based|
|Direct|-|47.1s|47.1s|-|47.1s|47.1s|
|Qwen2.5-VL-7B|M3DocRAG|10.7s|6.8s|17.5s|10.7s|10.2s|20.9s|
|MoLoRAG|38.4s|6.8s|45.2s|38.4s|10.1s|48.5s|
|Multi-agent|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|31.8s|16.8s|48.6s|31.8s|35.5s|67.3s|
|peeeee|eee|+--+|+--+|+--+|
||Le|Question|How many total available positions for actors were there across Playwrights Horizons and The Public Theater in the 2018-19 season?|EEE|EE|EE|EE|EE|EE|EE|HH|EH|HH|
|Evidence Pages|—|.|.|Index 49, 51|\|B-=—\)|ae|.|\7A-—|\}|\|||||Qwen2.5-VL-7B DirectUnanswerable|Xx|
|‘goodPlaywrights‘andPlaywrights theenough only|Horizonsplayto tie written forhired frstby 64.4%place.an Indigenous BIPOCHorizonsThey actors producedplaywright in the the 2018-19 season,thisonly|playseason,written NOURA byanew\ aee)by high HeatherMENA_ and playwright a Ratfo was}|DespiteThetlas{reaofFora unconscious producing companyPublica play written thatbies when utedincsie AsianTheaterby an AsianAmericancatingplaywright, actorsto castonly 427%Theaccounted ofPublicha hadoes, oneforNey,less ofbatts the=than worsta|percentageling}|i}nnnUf|w. M3DocRAGRetrieved Pages|nnn|[62, 49, 56]|-s|||
|seatonbytheylanded Indigenous couldbackhiringimproveonwritermoreour Mostinclude LarissaPOCDiversehiring FastHorse,detigners more BIPOCist|this andhowever,year expandingafterdirectorshadbeing anthelr all-White(only bumped BIPOCone leadership.BIPOCcast.off|last rectorPlaywrightsyear.|Areaswas HorizonsInhitedwhichthis|CastingSecondinSBIPOCwathar|At287%, 61.2%writersthe hing ofits Theout roles of Publicof|16BFOC withandhad BIPOCdesigners.were morethe actors, 8POConlyWhile non-profit boardincreasingtherestil meribasfromto produce oom 83.6%than for aany the improvementplay ether written season nonprofitprior. by atheseThey Latin«end hiredfactors were|i]\I|Unanswerable|across Playwrights Horizons and The Public Theater in the 2018-19|x|“The total available positions for actors|II||
|sono|snc|s|sss|esse|sas|nnns|iggrcccc|season is not provided in the given text.”|
|setonswee|[ee]|[ee]|[ee]|[ee]|[ee]|[ee]|[ee]|Torn|watanierosmonsias|[ee]|[ee]|[ee]|[ftMee|ee|ee|eecrocee|cree|eecsssssaaceeOR|SRRee|ee|eee|M|at|oEI|
|oe|cs|
|oes|om|1.01|PO|MDocAgent|
|wena |||se|for every$1<penton|tate|[02|White actors|||Retrieved Pages|Text [39, 49, 62]  Image [56,|51|, 48]|!|
|soonnnncen m7 -|asionamercan|Bf|sax|0|||258|xX|“The text mentions that Playwrights Horizons had 111 positions|||
|om|64.4%|~|61.2%||||and The Public Theater had 65 positions. So, the total available|!|
|positions for actors across Playwrights Horizons and The Public Theater|
|!|in the 2018-|19 season would be 111 + 65 = 176 positions.”|||
|warnens|Ta|RAsceoeToNS|Li—------|
|w. MoLoRAG|
|—_—wowcone|on|we|66.7%.|manasa|"“oe|lors|33.3%BIPOCOfwriters|||iLL|Retrieved Pages|Answer|Playwrights Horizons and The Public Theater in the 2018-19 season is|192 (45 + 147)192|4|.”|“|The total available positions for actors across [39,|49, 51|]||!I|

**----- End of picture text -----**<br>


Figure 8: **Case study under the top-** 3 **setting on LongDocURL involving cross-page reasoning.** The given question requires evidence from two separate pages to derive the correct answer. LVLM direct inference fails due to its limited context capacity (e.g., processing up to 30 pages). M3DocRAG is unable to retrieve the relevant content, leading to failure. MDocAgent relies on only one evidence page and falls into hallucination, producing an incorrect answer. In contrast, MoLoRAG successfully **retrieves both relevant pages** , enabling the LVLM to leverage this information to **answer the question correctly** . 

hinder overall efficiency as context complexity increases. Despite the associated retrieval time costs, MoLoRAG maintains a notably fast inference stage and scales well with larger _K_ values, ensuring a balance between efficiency and performance. 

## **D.5 Case Study** 

We present another case involving cross-page reasoning, as shown in Figure 8. The question asks: “How many total available positions for actors were there across Playwrights Horizons and The Public Theater in the 2018–19 season?” Answering this requires evidence from two separate pages: the available positions for actors at Playwrights Horizons and The Public Theater. 

LVLM direct inference fails due to limited con- 

text capacity, as it can only process up to 30 pages, while the evidence pages are located at indices 49 and 51. Both M3DocRAG and MDocAgent fail to retrieve these evidence pages because they rely solely on semantic relevance. Specifically, MDocAgent retrieves only one evidence page, resulting in hallucination and an incorrect answer. In contrast, MoLoRAG effectively retrieves both relevant pages, enabling the LVLM to access the necessary information and correctly compute the total available positions, demonstrating its superior cross-page reasoning capability. 

## **D.6 Ablation Study** 

In this subsection, we present the ablation study to evaluate the effectiveness of individual components 

Table 7: **Retrieval performance comparison of MoLoRAGLogi and MoLoRAG+ with the same finetuned retrieval engine.** MoLoRAGLogi, which relies solely on logical relevance, consistently underperforms compared to MoLoRAG+, demonstrating the effectiveness of combining semantic and logical relevance. 

|**Dataset**<br>_K_|**Method**<br>**Recall**<br>**NDCG**<br>**MRR**|
|---|---|
|**MMLong**<br>1<br>3|MoLoRAGLogi<br>46.55<br>58.90<br>58.90|
||MoLoRAG+<br>**51.32**<br>**66.86**<br>**66.86**|
||MoLoRAGLogi<br>60.45<br>58.02<br>64.62<br>MoLoRAG+<br>**68.87**<br>**64.49**<br>**73.50**|
|**LongDocURL**<br>1<br>3|MoLoRAGLogi<br>47.65<br>65.56<br>65.56<br>MoLoRAG+<br>**50.82**<br>**70.08**<br>**70.08**|
||MoLoRAGLogi<br>62.71<br>61.27<br>71.53<br>MoLoRAG+<br>**68.92**<br>**64.90**<br>**77.14**|



within MoLoRAG, focusing on two key aspects: the combination of semantic and logical relevance, and the graph construction process. 

**Combination of Semantic and Logical Relevance** We consider a variant of MoLoRAG that relies solely on logical relevance by setting _si_ = _s_[logi] _i_ within the framework. This variant is referred to as MoLoRAGlogi. In this setup, the retrieval engine is the fine-tuned Qwen2.5-VL-3B, which already demonstrates strong task understanding. We compare the retrieval performance of MoLoRAGlogi with MoLoRAG+ in Table 7. From the results, it becomes evident that relying solely on logical relevance does not achieve optimal performance. This is because VLMs may exhibit over-confidence and, in some cases, fall into hallucinations when relying exclusively on reasoning capabilities. Additionally, logical relevance scores are discrete, often leading to multiple pages with identical scores, making it difficult to rank and distinguish between them. Consequently, it is more reliable to combine both logical and semantic relevance. 

**Effectiveness of Graph Construction** We evaluate another variant of MoLoRAG, named MoLoRAGFull, in which the VLM **traverses all pages within the document** for a given question, assigning a relevance score to each page. Each page’s relevance score is updated by combining semantic and logical relevance scores. After traversal, only the top- _K_ pages are retained as the final retrieval result. Unlike MoLoRAG, this variant **eliminates the graph-based indexing mechanism** and instead **sequentially processes all pages in the document for selection** . We compare the retrieval performance of this variant with MoLoRAG+ in Table 8. While MoLoRAGFull achieves a slight improvement due to the expanded set of queried 

Table 8: **Retrieval performance comparison between MoLoRAGFull and MoLoRAG+ using the same finetuned retrieval engine.** MoLoRAGFull sequentially queries each page of the entire document and combines their relevance scores for final re-ranking. Although this variant achieves a marginal improvement over MoLoRAG+, it requires nearly **twice the computational time** (Figure 6), significantly reducing efficiency. 

|**Dataset**<br>_K_|**Method**<br>**Recall**<br>**NDCG**<br>**MRR**|
|---|---|
|**MMLong**<br>1<br>3|MoLoRAGFull<br>**51.63**<br>**67.21**<br>**67.21**<br>MoLoRAG+<br>51.32<br>66.86<br>66.86|
||MoLoRAGFull<br>**73.64**<br>64.31<br>**75.20**<br>MoLoRAG+<br>68.87<br>**64.49**<br>73.50|
|**LongDocURL**<br>1<br>3|MoLoRAGFull<br>**51.24**<br>**70.68**<br>**70.68**<br>MoLoRAG+<br>50.82<br>70.08<br>70.08|
||MoLoRAGFull<br>**72.30**<br>64.32<br>**78.53**<br>MoLoRAG+<br>68.92<br>**64.90**<br>77.14|



pages, it requires nearly twice the computational time (Figure 6), significantly reducing efficiency. Furthermore, this marginal performance difference highlights the effectiveness of our graph construction, which provides a high-quality candidate set for retrieval. 

## **D.7 Fine-grained Analysis** 

This subsection provides a detailed performance analysis across evidence modalities (e.g., text, tables, figures) and question contexts (e.g., singlepage and multi-page understanding). The analysis focuses on the MMLongBench and LongDocURL datasets, which offer official splits based on modalities and locations. For MMLongBench, results for top-1, top-3, and top-5 settings are presented in Tables 11, 12, and 13, respectively. For LongDocURL, the corresponding results are shown in Tables 14, 15, and 16. From these results, we conclude that **LVLMs w. MoLoRAG excel across diverse modalities** . While LLMs w. Text RAG methods perform reasonably well on text-based modalities, they struggle significantly with nontextual content like figures (e.g., 9.83% on Figure). This highlights the critical need for robust multimodal understanding. In contrast, LVLMs integrated with MoLoRAG demonstrate strong and balanced performance across all modalities. For example, Qwen2.5-VL-7B with MoLoRAG+ achieves 37.43% on Text and 36.94% on Figure (MMLongBench in the top-1 setting), showing the effectiveness of retrieval-based multi-modal reasoning. 

Table 9: **Overall performance comparison (in** % **) under the retrieved top-** 1 **setting.** The best performance across all methods is **highlighted** . 

|**Type**|**Model**<br>**Method**|**MMLongBench**<br>**LongDocURL**<br>**PaperTab**<br>**FetaTab**|**Avg.**|
|---|---|---|---|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>Qwen2.5-7B<br>Text RAG<br>LLaMA3.1-8B<br>Text RAG<br>GPT-4o<br>Text RAG<br>DeepSeek-v3<br>Text RAG|21.66<br>17.63<br>5.09<br>24.11<br>22.11<br>20.75<br>5.34<br>22.64<br>18.70<br>22.57<br>7.12<br>28.25<br>24.07<br>22.84<br>8.65<br>34.15<br>**25.94**<br>**24.32**<br>**10.18**<br>**34.55**|17.12<br>17.71<br>19.16<br>22.43<br>**23.75**|
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|7.15<br>10.78<br>3.05<br>11.61<br>16.32<br>25.25<br>6.62<br>15.26<br>16.73<br>26.11<br>**6.87**<br>**18.41**<br>**17.15**<br>**27.00**<br>6.36<br>17.52|8.15<br>15.86<br>**17.03**<br>17.01|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|8.40<br>14.72<br>6.11<br>16.14<br>26.23<br>42.21<br>16.54<br>48.43<br>27.47<br>44.75<br>20.87<br>56.89<br>**28.98**<br>**45.17**<br>**21.88**<br>**58.27**|11.34<br>33.35<br>37.50<br>**38.58**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|26.65<br>24.89<br>**25.19**<br>51.57<br>26.77<br>39.82<br>19.85<br>45.77<br>29.08<br>41.95<br>21.88<br>54.72<br>**30.03**<br>**43.17**<br>23.16<br>**55.41**|32.08<br>33.05<br>36.91<br>**37.94**|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|32.77<br>26.38<br>**29.77**<br>**64.07**|38.25<br>36.48<br>41.67|
|||32.29<br>43.32<br>19.34<br>50.98<br>34.35<br>46.89<br>23.92<br>61.52||
|||**36.37**<br>**47.86**<br>27.48<br>62.50|**43.55**|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B + Qwen2.5-VL-7B)<br>31.73<br>44.42<br>21.63<br>57.78<br>38.89|||



Table 10: **Overall performance comparison (in** % **) under the retrieved top-** 5 **setting.** The best performance across all methods is **highlighted** . 

|**Type**|**Model**<br>**Method**|**MMLongBench**<br>**LongDocURL**<br>**PaperTab**<br>**FetaTab**|**Avg.**|
|---|---|---|---|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>Qwen2.5-7B<br>Text RAG<br>LLaMA3.1-8B<br>Text RAG<br>GPT-4o<br>Text RAG<br>DeepSeek-v3<br>Text RAG|23.43<br>26.43<br>13.23<br>48.62<br>26.09<br>31.36<br>16.79<br>49.21<br>24.25<br>33.27<br>17.81<br>54.53<br>28.74<br>36.98<br>20.36<br>57.78<br>**31.23**<br>**39.04**<br>**23.92**<br>**62.01**|27.93<br>30.86<br>32.47<br>35.97<br>**39.05**|
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|7.15<br>10.78<br>3.05<br>11.61<br>**10.43**<br>12.65<br>**4.58**<br>12.80<br>9.56<br>12.72<br>4.07<br>**14.07**<br>9.19<br>**13.59**<br>4.33<br>13.09|8.15<br>**10.12**<br>10.11<br>10.05|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|8.40<br>14.72<br>6.11<br>16.14<br>18.87<br>29.27<br>8.14<br>27.26<br>20.07<br>30.76<br>8.40<br>39.76<br>**24.86**<br>**38.02**<br>**9.67**<br>**41.44**|11.34<br>20.89<br>24.75<br>**28.50**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|26.65<br>24.89<br>25.19<br>51.57<br>28.38<br>44.67<br>27.48<br>55.22<br>31.43<br>**46.05**<br>26.97<br>57.48<br>**32.41**<br>45.13<br>**27.48**<br>**58.07**|32.08<br>38.94<br>40.48<br>**40.77**|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|32.77<br>26.38<br>29.77<br>64.07<br>37.19<br>50.33<br>30.53<br>64.37<br>39.97<br>52.33<br>31.04<br>68.80|38.25<br>45.61<br>48.04|
|||**40.47**<br>**52.33**<br>**31.55**<br>**69.39**|**48.44**|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B + Qwen2.5-VL-7B)|38.34<br>48.07<br>29.77<br>63.78|44.99|



Table 11: **Fine-grained performance analysis (Accuracy in** % **) across evidence modality and evidence locations on MMLongBench under the retrieved top-** 1 **setting.** “UNA” refers to unanswerable questions. 

|**T**|**Mdl**<br>**Mthd**|**Modality**|**Location**|**Overall**|
|---|---|---|---|---|
|**ype**|**oe**<br>**eo**|Text<br>Table<br>Figure<br>Chart<br>Layout|Single<br>Multiple<br>UNA|Acc<br>EM|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>Qwen2.5-7B<br>Text RAG<br>LLaMA3.1-8B<br>Text RAG<br>GPT-4o<br>Text RAG<br>DeepSeek-v3<br>Text RAG|11.32<br>6.60<br>5.20<br>6.97<br>7.21<br>13.31<br>9.56<br>6.65<br>6.18<br>8.01<br>16.53<br>10.93<br>7.80<br>9.84<br>**12.92**<br>15.04<br>12.95<br>6.39<br>8.05<br>10.33<br>**17.76**<br>**14.70**<br>**9.83**<br>**9.99**<br>10.72|10.54<br>4.26<br>75.78<br>12.09<br>5.41<br>73.09<br>13.26<br>7.96<br>47.98<br>13.34<br>5.97<br>**77.13**<br>**16.80**<br>**7.97**<br>76.68|21.66<br>19.96<br>22.11<br>20.61<br>18.70<br>16.36<br>24.07<br>22.46<br>**25.94**<br>**23.84**|
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|6.54<br>1.53<br>7.38<br>2.12<br>4.38<br>18.04<br>**10.47**<br>20.03<br>13.35<br>14.76<br>18.12<br>9.71<br>19.68<br>**13.57**<br>**16.57**<br>**18.94**<br>9.85<br>**21.61**<br>13.22<br>15.94|4.17<br>5.07<br>**16.59**<br>19.93<br>11.26<br>15.70<br>19.80<br>**12.13**<br>16.14<br>**22.92**<br>10.50<br>13.90|7.15<br>5.73<br>16.32<br>13.03<br>16.73<br>12.94<br>**17.15**<br>**13.03**|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|8.86<br>6.57<br>13.39<br>5.23<br>13.63<br>30.84<br>23.95<br>31.55<br>27.29<br>30.16<br>31.67<br>**28.54**<br>30.80<br>26.42<br>**32.84**<br>**35.14**<br>27.86<br>**35.32**<br>**27.42**<br>30.74|9.86<br>9.36<br>3.14<br>41.55<br>15.76<br>7.62<br>43.65<br>16.63<br>**8.07**<br>**47.62**<br>**17.37**<br>4.93|8.40<br>6.01<br>26.23<br>20.98<br>27.47<br>21.81<br>**28.98**<br>**22.83**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|34.11<br>23.37<br>**33.75**<br>24.72<br>**29.56**<br>31.88<br>22.70<br>27.87<br>23.85<br>22.22<br>33.07<br>**29.18**<br>29.06<br>23.44<br>24.88<br>**35.26**<br>28.11<br>32.09<br>**25.03**<br>26.66|36.30<br>**23.42**<br>9.87<br>37.90<br>15.40<br>20.63<br>41.58<br>17.02<br>**21.52**<br>**43.68**<br>18.11<br>19.73|26.65<br>20.98<br>26.77<br>21.90<br>29.08<br>23.84<br>**30.03**<br>**24.49**|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|37.14<br>25.52<br>31.95<br>28.00<br>27.26<br>31.87<br>23.88<br>30.03<br>26.48<br>26.74<br>33.80<br>**32.12**<br>29.77<br>29.23<br>30.34<br>**37.43**<br>31.34<br>**36.94**<br>**29.95**<br>**33.11**|40.21<br>**23.88**<br>30.94<br>42.66<br>12.37<br>**40.81**<br>46.61<br>15.37<br>37.22<br>**50.14**<br>18.25<br>34.08|32.77<br>27.36<br>32.29<br>27.63<br>34.35<br>29.67<br>**36.37**<br>**30.87**|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|35.08<br>30.13<br>29.61<br>27.47<br>24.72|45.26<br>14.86<br>29.15|31.73<br>27.45|



Table 12: **Fine-grained performance analysis (Accuracy in** % **) across evidence modality and evidence locations on MMLongBench under the retrieved top-** 3 **setting.** “UNA” refers to unanswerable questions. 

|**T**|**Mdl**<br>**Mthd**|**Modality**|**Location**|**Overall**|
|---|---|---|---|---|
|**ype**|**oe**<br>**eo**|Text<br>Table<br>Figure<br>Chart<br>Layout|Single<br>Multiple<br>UNA|Acc<br>EM|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>Qwen2.5-7B<br>Text RAG<br>LLaMA3.1-8B<br>Text RAG<br>GPT-4o<br>Text RAG<br>DeepSeek-v3<br>Text RAG|15.97<br>14.77<br>8.41<br>10.70<br>12.38<br>17.96<br>15.42<br>9.58<br>10.35<br>10.69<br>20.40<br>20.02<br>10.44<br>15.42<br>13.82<br>19.68<br>19.14<br>10.10<br>13.58<br>12.25<br>**25.37**<br>**22.23**<br>**13.34**<br>**19.60**<br>**17.27**|16.16<br>8.86<br>68.61<br>17.72<br>9.34<br>70.40<br>18.74<br>**13.62**<br>45.29<br>20.20<br>10.52<br>**70.40**<br>**24.85**<br>13.03<br>69.06|24.47<br>22.00<br>25.52<br>23.29<br>22.56<br>19.22<br>27.23<br>24.31<br>**29.82**<br>**26.62**|
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|6.54<br>1.53<br>7.38<br>2.12<br>4.38<br>**8.74**<br>**6.59**<br>**11.72**<br>1.87<br>**8.54**<br>6.84<br>5.55<br>10.72<br>2.15<br>7.66<br>7.49<br>2.49<br>11.24<br>**2.89**<br>8.08|4.17<br>5.07<br>16.59<br>7.27<br>**8.55**<br>**17.49**<br>7.82<br>6.43<br>16.14<br>**7.86**<br>6.25<br>16.59|7.15<br>5.73<br>**10.10**<br>**8.23**<br>9.37<br>7.49<br>9.41<br>7.30|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|8.86<br>6.57<br>13.39<br>5.23<br>13.63<br>19.75<br>14.31<br>27.55<br>18.38<br>25.02<br>21.57<br>18.79<br>29.00<br>17.55<br>24.54<br>**27.58**<br>**23.33**<br>**34.45**<br>**21.56**<br>**32.67**|9.86<br>9.36<br>3.14<br>25.91<br>15.84<br>3.59<br>29.60<br>18.15<br>2.69<br>**39.40**<br>**18.74**<br>**4.04**|8.40<br>6.01<br>18.12<br>13.49<br>20.43<br>16.27<br>**25.47**<br>**19.59**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|34.11<br>23.37<br>33.75<br>24.72<br>29.56<br>35.11<br>25.58<br>32.23<br>24.04<br>28.06<br>**38.94**<br>**30.48**<br>**36.05**<br>24.33<br>28.73<br>38.29<br>29.39<br>35.34<br>**26.48**<br>**33.53**|36.30<br>23.42<br>9.87<br>39.62<br>20.62<br>18.83<br>44.29<br>**23.52**<br>19.28<br>**45.12**<br>23.17<br>**19.28**|26.65<br>20.98<br>29.11<br>23.66<br>32.11<br>26.16<br>**32.47**<br>**26.52**|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|37.14<br>25.52<br>31.95<br>28.00<br>27.26<br>38.83<br>36.24<br>35.83<br>30.46<br>36.56<br>41.67<br>37.89<br>37.56<br>**34.15**<br>32.44<br>**42.69**<br>**38.53**<br>**40.73**<br>33.26<br>**38.79**|40.21<br>23.88<br>30.94<br>46.85<br>25.29<br>28.70<br>50.07<br>26.12<br>34.98<br>**52.90**<br>**27.59**<br>**35.87**|32.77<br>27.36<br>36.18<br>30.96<br>39.28<br>33.18<br>**41.01**<br>**34.94**|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|43.14<br>38.72<br>37.90<br>32.55<br>31.17|53.45<br>23.82<br>28.25|38.53<br>33.27|



Table 13: **Fine-grained performance analysis (Accuracy in** % **) across evidence modality and evidence locations on MMLongBench under the retrieved top-** 5 **setting.** “UNA” refers to unanswerable questions. 

|**T**|**Mdl**<br>**Mthd**|**Modality**|**Location**|**Overall**|
|---|---|---|---|---|
|**ype**|**oe**<br>**eo**|Text<br>Table<br>Figure<br>Chart<br>Layout|Single<br>Multiple<br>UNA|Acc<br>EM|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>Qwen2.5-7B<br>Text RAG<br>LLaMA3.1-8B<br>Text RAG<br>GPT-4o<br>Text RAG<br>DeepSeek-v3<br>Text RAG|17.41<br>12.03<br>8.13<br>13.32<br>16.30<br>19.98<br>19.29<br>10.06<br>12.57<br>15.31<br>24.61<br>22.71<br>12.21<br>18.42<br>**21.49**<br>22.38<br>24.50<br>12.30<br>15.42<br>16.17<br>**27.54**<br>**28.33**<br>**15.53**<br>**21.39**<br>20.44|16.16<br>10.59<br>61.43<br>19.99<br>11.34<br>64.13<br>22.60<br>**15.74**<br>41.26<br>23.37<br>13.54<br>**65.47**<br>**28.90**<br>15.67<br>62.33|23.43<br>20.43<br>26.09<br>23.57<br>24.25<br>21.07<br>28.74<br>25.51<br>**31.23**<br>**27.54**|
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|6.54<br>1.53<br>7.38<br>2.12<br>4.38<br>**8.59**<br>**6.23**<br>**12.16**<br>3.40<br>7.43<br>7.33<br>5.52<br>11.36<br>3.12<br>8.34<br>6.46<br>2.26<br>10.33<br>**3.70**<br>**8.54**|4.17<br>5.07<br>16.59<br>**8.65**<br>**7.92**<br>**17.49**<br>7.95<br>7.12<br>16.14<br>8.02<br>4.79<br>17.49|7.15<br>5.73<br>**10.43**<br>**7.95**<br>9.56<br>7.67<br>9.19<br>7.30|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|8.86<br>6.57<br>13.39<br>5.23<br>13.63<br>22.04<br>15.45<br>28.86<br>15.25<br>23.28<br>22.38<br>18.64<br>28.07<br>13.83<br>23.68<br>**26.61**<br>**22.98**<br>**34.82**<br>**19.32**<br>**32.11**|9.86<br>9.36<br>3.14<br>26.90<br>15.77<br>**4.93**<br>29.60<br>16.52<br>4.04<br>**38.27**<br>**18.58**<br>4.04|8.40<br>6.01<br>18.87<br>14.51<br>20.07<br>15.80<br>**24.86**<br>**19.41**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|34.11<br>23.37<br>33.75<br>24.72<br>29.56<br>35.79<br>26.04<br>32.06<br>24.15<br>29.33<br>38.22<br>**31.23**<br>32.84<br>**27.85**<br>30.33<br>**38.38**<br>30.99<br>**36.04**<br>26.00<br>**33.94**|36.30<br>23.42<br>9.87<br>39.08<br>22.16<br>14.35<br>43.33<br>23.60<br>17.04<br>**44.82**<br>**23.67**<br>**18.83**|26.65<br>20.98<br>28.38<br>22.46<br>31.43<br>25.60<br>**32.41**<br>**26.52**|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|37.14<br>25.52<br>31.95<br>28.00<br>27.26<br>40.17<br>34.20<br>36.39<br>**35.34**<br>31.71<br>**43.07**<br>38.10<br>38.92<br>35.22<br>35.08<br>41.57<br>**38.31**<br>**39.08**<br>31.64<br>**38.62**|40.21<br>23.88<br>30.94<br>48.36<br>24.59<br>31.39<br>**51.72**<br>**27.02**<br>33.63<br>51.16<br>26.96<br>**38.57**|32.77<br>27.36<br>37.19<br>32.16<br>39.97<br>34.20<br>**40.47**<br>**34.57**|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|41.92<br>42.00<br>34.02<br>33.45<br>29.77|49.97<br>25.25<br>32.29|38.34<br>32.99|



Table 14: **Fine-grained performance analysis (Accuracy in** % **) across evidence modality and evidence locations on LongDocURL under the retrieved top-** 1 **setting.** 

|**T**|**Mdl**<br>**Mthd**|**Modality**|**Location**|**Overall**|
|---|---|---|---|---|
|**ype**|**oe**<br>**eo**|Text<br>Table<br>Figure<br>Layout|Single<br>Multiple|Acc<br>EM|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>26.18<br>11.51<br>14.59<br>12.62<br>16.32<br>18.49<br>17.63<br>15.05<br>Qwen2.5-7B<br>Text RAG<br>29.25<br>14.98<br>20.50<br>16.10<br>20.66<br>20.53<br>20.75<br>17.46<br>LLaMA3.1-8B<br>Text RAG<br>30.63<br>16.26<br>22.18<br>16.52<br>21.32<br>23.49<br>22.57<br>18.11<br>GPT-4o<br>Text RAG<br>32.16<br>16.35<br>23.21<br>17.44<br>21.91<br>23.40<br>22.84<br>18.45<br>DeepSeek-V3<br>Text RAG<br>**33.28**<br>**18.17**<br>**25.47**<br>**19.84**<br>**24.00**<br>**24.34**<br>**24.32**<br>**20.09**||||
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|16.79<br>5.28<br>12.12<br>7.39<br>33.67<br>17.46<br>31.41<br>22.21<br>34.61<br>18.32<br>32.37<br>24.13<br>**34.87**<br>**19.99**<br>**32.83**<br>**24.47**|7.87<br>13.43<br>24.72<br>25.58<br>24.99<br>**26.99**<br>**27.09**<br>26.80|10.78<br>9.29<br>25.25<br>17.33<br>26.11<br>17.89<br>**27.00**<br>**18.28**|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|19.98<br>8.26<br>13.81<br>13.65<br>51.63<br>36.79<br>40.39<br>33.04<br>54.64<br>39.97<br>41.52<br>34.64<br>**54.91**<br>**40.11**<br>**42.90**<br>**34.90**|11.18<br>17.87<br>46.83<br>38.10<br>48.77<br>**41.17**<br>**49.88**<br>41.16|14.72<br>11.35<br>42.21<br>33.08<br>44.75<br>35.18<br>**45.17**<br>**35.61**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|31.98<br>17.43<br>23.50<br>22.86<br>49.08<br>32.91<br>38.70<br>31.24<br>50.60<br>36.93<br>38.09<br>32.20<br>**52.06**<br>**37.56**<br>**40.40**<br>**32.97**|21.60<br>27.77<br>42.77<br>37.17<br>44.66<br>39.54<br>**46.83**<br>**39.91**|24.89<br>18.67<br>39.82<br>32.22<br>41.95<br>34.15<br>**43.17**<br>**34.80**|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|32.37<br>19.88<br>27.09<br>23.25<br>52.00<br>38.18<br>41.79<br>34.86<br>55.91<br>42.77<br>44.06<br>36.58<br>**56.99**<br>**42.81**<br>**45.77**<br>**37.24**|24.20<br>28.15<br>49.67<br>37.47<br>52.88<br>41.46<br>**53.91**<br>**42.47**|26.38<br>19.74<br>43.32<br>34.19<br>46.89<br>37.25<br>**47.86**<br>**37.81**|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|54.52<br>40.41<br>43.20<br>31.25|48.88<br>40.26|44.42<br>36.30|



Table 15: **Fine-grained performance analysis (Accuracy in** % **) across evidence modality and evidence locations on LongDocURL under the retrieved top-** 3 **setting.** 

|**T**|**Mdl**<br>**Mthd**|**Modality**|**Location**|**Overall**|
|---|---|---|---|---|
|**ype**|**oe**<br>**eo**|Text<br>Table<br>Figure<br>Layout|Single<br>Multiple|Acc<br>EM|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>33.94<br>17.95<br>21.25<br>18.98<br>23.09<br>26.63<br>25.06<br>19.78<br>Qwen2.5-7B<br>Text RAG<br>36.41<br>20.55<br>25.94<br>23.77<br>26.75<br>28.73<br>27.93<br>21.94<br>LLaMA3.1-8B<br>Text RAG<br>37.22<br>22.99<br>29.64<br>22.53<br>29.42<br>30.08<br>29.80<br>22.75<br>GPT-4o<br>Text RAG<br>40.91<br>26.80<br>33.14<br>26.57<br>33.19<br>32.13<br>32.74<br>25.20<br>DeepSeek-V3<br>Text RAG<br>**41.89**<br>**30.84**<br>**35.49**<br>**28.15**<br>**35.77**<br>**33.67**<br>**34.73**<br>**26.84**||||
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|16.79<br>5.28<br>12.12<br>7.39<br>**20.64**<br>7.17<br>16.16<br>**10.75**<br>20.52<br>6.45<br>15.64<br>10.58<br>19.94<br>**7.32**<br>**17.11**<br>10.64|7.87<br>13.43<br>11.12<br>16.20<br>10.94<br>15.68<br>**11.17**<br>**15.73**|10.78<br>9.29<br>**13.85**<br>10.62<br>13.49<br>**10.75**<br>13.58<br>10.49|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|19.98<br>8.26<br>13.81<br>13.65<br>40.61<br>16.19<br>27.56<br>30.78<br>40.62<br>17.57<br>28.69<br>28.86<br>**44.28**<br>**29.89**<br>**37.81**<br>**32.84**|11.18<br>17.87<br>25.54<br>33.31<br>27.07<br>32.67<br>**39.19**<br>**35.58**|14.72<br>11.35<br>29.60<br>21.29<br>29.98<br>21.81<br>**37.21**<br>**27.74**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|31.98<br>17.43<br>23.50<br>22.86<br>54.07<br>37.97<br>42.07<br>36.97<br>**55.99**<br>38.23<br>**42.95**<br>**37.01**<br>54.24<br>**39.03**<br>41.13<br>36.62|21.60<br>27.77<br>46.39<br>42.64<br>**48.09**<br>**43.76**<br>47.49<br>43.31|24.89<br>18.67<br>44.4<br>34.97<br>**45.79**<br>**36.17**<br>45.27<br>35.53|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|32.37<br>19.88<br>27.09<br>23.25<br>58.16<br>43.75<br>46.04<br>41.24<br>**61.46**<br>45.66<br>**49.06**<br>**43.27**<br>61.43<br>**45.98**<br>49.01<br>42.56|24.20<br>28.15<br>53.35<br>45.13<br>**55.60**<br>**48.30**<br>55.01<br>49.01|26.38<br>19.74<br>49.03<br>38.88<br>51.71<br>**40.86**<br>**51.85**<br>40.13|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|56.81<br>42.25<br>44.07<br>35.48|49.46<br>44.51|46.91<br>37.63|



Table 16: **Fine-grained performance analysis (Accuracy in** % **) across evidence modality and evidence locations on LongDocURL under the retrieved top-** 5 **setting.** 

|**T**|**Mdl**<br>**Mthd**|**Modality**|**Location**|**Overall**|
|---|---|---|---|---|
|**ype**|**oe**<br>**eo**|Text<br>Table<br>Figure<br>Layout|Single<br>Multiple|Acc<br>EM|
|**_LLM-based_**|Mistral-7B<br>Text RAG<br>34.74<br>18.78<br>22.91<br>19.91<br>25.57<br>26.94<br>26.43<br>20.22<br>Qwen2.5-7B<br>Text RAG<br>39.38<br>24.63<br>29.76<br>24.72<br>30.48<br>31.92<br>31.36<br>25.03<br>LLaMA3.1-8B<br>Text RAG<br>40.04<br>26.02<br>32.74<br>26.01<br>32.56<br>33.85<br>33.27<br>25.42<br>GPT-4o<br>Text RAG<br>44.20<br>31.54<br>40.20<br>29.32<br>37.86<br>36.00<br>36.98<br>28.13<br>DeepSeek-V3<br>Text RAG<br>**45.71**<br>**34.39**<br>**41.58**<br>**31.26**<br>**40.09**<br>**38.08**<br>**39.04**<br>**29.38**||||
|**_LVLM-based_**|LLaVA-Next-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|16.79<br>5.28<br>12.12<br>7.39<br>19.20<br>5.89<br>13.58<br>9.36<br>19.39<br>5.94<br>13.51<br>10.13<br>**20.03**<br>**7.00**<br>**16.67**<br>**10.69**|7.87<br>13.43<br>8.86<br>15.93<br>9.18<br>15.79<br>**10.79**<br>**16.01**|10.78<br>9.29<br>12.65<br>10.02<br>12.72<br>10.19<br>**13.59**<br>**10.58**|
||DeepSeek-VL-16B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|19.98<br>8.26<br>13.81<br>13.65<br>40.54<br>15.44<br>26.41<br>30.35<br>42.08<br>16.97<br>28.58<br>31.09<br>**46.11**<br>**29.48**<br>**38.03**<br>**34.02**|11.18<br>17.87<br>24.50<br>33.62<br>26.40<br>34.76<br>**39.76**<br>**36.61**|14.72<br>11.35<br>29.27<br>21.03<br>30.76<br>22.19<br>**38.02**<br>**28.34**|
||Qwen2.5-VL-3B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|31.98<br>17.43<br>23.50<br>22.86<br>54.76<br>37.06<br>39.66<br>38.47<br>**55.29**<br>**39.68**<br>40.79<br>**38.88**<br>53.12<br>39.45<br>**40.95**<br>37.32|21.60<br>27.77<br>45.49<br>43.95<br>**47.07**<br>**45.25**<br>47.05<br>43.43|24.89<br>18.67<br>44.67<br>34.84<br>**46.05**<br>**35.74**<br>45.13<br>35.53|
||Qwen2.5-VL-7B<br>Direct<br>M3DocRAG<br>MoLoRAG<br>MoLoRAG+|32.37<br>19.88<br>27.09<br>23.25<br>59.52<br>45.01<br>45.25<br>42.82<br>**60.70**<br>**46.99**<br>47.95<br>44.74<br>60.54<br>46.61<br>**48.68**<br>**45.13**|24.20<br>28.15<br>53.14<br>47.71<br>**55.23**<br>49.65<br>54.86<br>**50.05**|26.38<br>19.74<br>50.33<br>39.23<br>**52.33**<br>**41.76**<br>52.33<br>40.65|
|**_Multi-agent_**|MDocAgent (LLaMA3.1-8B+Qwen2.5-VL-7B)|57.09<br>44.66<br>45.97<br>35.47|50.79<br>45.52|48.07<br>38.32|



