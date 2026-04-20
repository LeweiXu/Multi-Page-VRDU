## **M[2] RAG: A Multi-agent and Multimodal Fusion Framework for Retrieval-Augmented Document QA** 

Tongtong Duan[1,2 ] , Minghao Hu[2(][B][)] , Liang Xue[1(][B][)] , Chunming Liu[1,2] , Guotong Geng[2] , Wei Luo[2] , and Zhunchen Luo[2] 

> 1 School of Information and Electrical Engineering, Hebei University of Engineering, Handan, China 

liangxue@hebeu.edu.cn 

> 2 Center of Information Research, Academy of Military Science, Beijing, China humh573@163.com 

**Abstract.** Document Question Answering (DocQA) aims to answer textual questions based on the information contained within the document. Effective integration of textual and visual cue features remains a challenge for current retrieval augmented generation (RAG) approaches. To address this issue, we propose M[2 ] RAG, a novel multimodal RAG framework. The framework integrates dual-tower retrieval and multi-agent generation mechanism to effectively deal with multi-modal retrieval, and uses text filter, visual extractor and modal fuser to reason collaboratively to achieve a comprehensive understanding of document content. To enhance the performance of the model in DocQA, we adopt a knowledge distillation based fine-tuning strategy to train a lightweight visual language model on the constructed fine-tuning dataset as the core component of the multi-agent module. Experiments on DocBench, MMLongBench, and LongDocURL show that M[2] RAG outperforms baseline methods by an average of 7.7%, demonstrating its effectiveness in complex document question answering. 

**Keywords:** Document question answering _·_ RAG _·_ Multi-agent collaboration _·_ Multimodal fusion 

## **1 Introduction** 

In the real world, documents serve as indispensable carriers of information across various domains. These documents often exhibit diverse structures and complex content, incorporating rich semantic and visual elements such as textual paragraphs, headings, tables, and images [6]. Document Question Answering (DocQA) [8,17,26], which enables automatic understanding and interaction with documents, has emerged as a key task in numerous practical applications [15,20,25]. However, DocQA faces significant challenges. On one hand, 

> _⃝_ c The Author(s), under exclusive license to Springer Nature Singapore Pte Ltd. 2026 T. Taniguchi et al. (Eds.): ICONIP 2025, CCIS 2753, pp. 507–520, 2026. https://doi.org/10.1007/978-981-95-4088-4_35 

508 T. Duan et al. 

questions are highly diverse, potentially focusing on textual details, visual layouts, or the interaction between the two. On the other hand, document content is often unevenly distributed and redundant, requiring models to effectively filter relevant information and perform robust cross-modal understanding. 

Although Large Language Models (LLMs) and Large Vision Language Models (LVLMs) have achieved remarkable progress in natural language processing and multimodal tasks, they still encounter limitations when dealing with long and complex multimodal documents. First, the abundance of redundant information in lengthy documents can lead to information overload, causing models to hallucinate [3,10] and generate answers inconsistent with the document content [21]. Second, when critical information resides primarily in visual elements or when fine-grained interactions between text and visual components are required, existing models often fail to produce accurate responses due to their limited capacity to integrate and reason over different modalities [15,24]. 

To mitigate these issues, Retrieval-Augmented Generation (RAG) [12,19] has been proposed as an auxiliary mechanism to extract key information from external knowledge sources, thereby improving information accessibility and boosting the performance of question-answer systems. However, as illustrated in Fig. 1, most existing RAG approaches focus on unimodal retrieval, either textual or visual, and fail to effectively integrate information across different modalities. In multimodal documents where textual and visual data are often tightly coupled, unimodal retrieval is insufficient to identify relevant content. For instance, when questions target visual elements, text-based RAG struggles to locate the corresponding visual cues; conversely, vision-based RAG [5] is constrained by the reasoning capabilities of visual models and prone to hallucination [9], making it difficult to align visual data with textual segments effectively. Therefore, designing a multimodal RAG framework that supports cross-modal retrieval and joint reasoning is of critical importance for advancing DocQA in complex real-world scenarios. 

To address the aforementioned limitations, this paper proposes M[2] RAG, an innovative multimodal Retrieval-Augmented Generation framework that combines dual-tower retrieval with multi-agent generation mechanisms, aimed at solving complex document question answering tasks. Specifically, our method integrates both textual and visual retrieval to provide target text and visual context for our framework, thereby extracting multimodal evidence comprehensively. We designed a multi-agent reasoning architecture, which includes a text filter agent and a visual extractor agent, each responsible for semantically understanding the retrieved text and visual content. Simultaneously, a modality fusion agent aligns and integrates the outputs of both agents to generate the final context-aware answer. This architecture not only preserves the semantic richness of the original document but also enhances the interpretability and accuracy of the reasoning process. The main contributions of this paper can be summarized as follows: 

- We propose the M[2] RAG framework, a multimodal RAG framework that combines dual-tower retrieval with a multi-agent generation mechanism, 

M[2] RAG 509 

specifically designed for visually rich document question answering tasks. This includes a text filter, visual extractor, and modality fusion module, enabling fine-grained cross-modal reasoning and evidence fusion. 

- We construct a fine-tuning dataset using knowledge distillation to train a lightweight VLM, enhancing the efficiency of multi-agent collaboration. 

– We demonstrate the effectiveness of our approach in three benchmark experiments: DocBench [30], MMLongBench [16], and LongDocURL [7], achieving an average performance improvement of 7.7% compared to the best baseline. 

**Fig. 1.** Comparison between traditional single-modal RAG and our proposed M[2 ] RAG approach. While traditional RAG is capable of handling long documents, it exhibits limitations in fine-grained attention and cross-modal understanding. In contrast, M[2 ] RAG incorporates multi-modal retrieval and dedicated agents for text filtering, visual extraction, and modality fusion. This design not only maintains the ability to process lengthy documents but also enhances detailed comprehension and the integration of information across modalities, leading to improved performance in document question answering tasks. 

## **2 Related Works** 

**Retrieval-Augmented Generation.** RAG addresses the challenges faced by Large Language Models (LLMs) in terms of accuracy, knowledge update speed, and answer transparency by integrating external information. Early RAG retrieval systems employed OCR (Optical Character Recognition) [18,23] technology to extract text, converting multimodal data into a unified textual representation. However, this unimodal approach resulted in the loss of modalityspecific details such as visual features, limiting the system’s ability to fully leverage the potential of multimodal inputs. With the advancement of visual models and multimodal embedding technologies, multimodal RAG [24,27] has evolved by expanding the RAG framework to incorporate both multimodal retrieval and 

510 T. Duan et al. 

generation, allowing for more comprehensive and contextually relevant responses. Approaches such as VisRAG [29] preserve document page screenshots in vectorized form and establish an index, enabling the effective retrieval of relevant document screenshots based on user queries. These methods utilize vision-language models to directly process multimodal data, minimizing information loss during data transformation. However, the integration of multimodal data inputs may compromise the accuracy of traditional text-based query descriptions, and the effective fusion of textual and visual modalities remains a significant challenge. 

**Multi-agent Framework.** Multi-agent frameworks enhance the scalability and adaptability of complex workflows through specialization and coordinated collaboration [4,11,13]. By enabling agents to communicate and share intermediate results, these systems assign subtasks to specialized agents, thereby achieving task-specific processing and parallel execution. Existing works [14,27,28] adopt core agent paradigms such as reflection, planning, tool use, and multiagent collaboration to dynamically adapt to task-specific requirements. However, managing communication and coordination among multiple agents may introduce inefficiencies and increase computational overhead. Balancing task allocation, resolving conflicts among agents, and effectively integrating multimodal information remain key challenges for Document Question Answering in multiagent frameworks. 

## **3 Methodology** 

This section provides a detailed description of M[2] RAG for solving complex document question answering tasks. As shown in Fig. 2, the M[2] RAG framework employs a novel dual-tower retrieval and multi-agent generation method, aimed at achieving a comprehensive understanding of multimodal documents. 

**Problem Formulation.** Given a question . _q_ , there exists a corpus of . _M_ documents, . _C_ = _{D_ 1 _, D_ 2 _, . . . , DM }_ , where each document . _Di_ consists of a sequence of text segments . _Ti_ = _{t_ 1 _, t_ 2 _, . . . , tn}_ and a set of pages . _Pi_ = _{P_ 1 _, P_ 2 _, . . . , Pm}_ , with . _n_ representing the number of text segments in document . _Di_ . The objective is to leverage the available multimodal information in the document corpus . _C_ to accurately generate an answer . _a_ to the given question . _q_ . Each question . _q_ requires information from one or more relevant documents, distributed across different pages within . _C_ . Answer generation relies on retrieving pertinent visual or textual evidence from one or more documents as contextual support for the response. 

## **3.1 Document Pre-processing** 

Document preprocessing is a fundamental step in the multimodal retrievalaugmented generation framework and is crucial for achieving efficient retrieval 

M[2] RAG 511 

and generation. The quality of document parsing directly impacts the performance of subsequent retrieval and generation tasks. The goal is to extract the information contained within the document corpus, generating both textual and visual multimodal representations to support text and visual retrieval. In this paper, we utilize a parsing tool that combines Optical Character Recognition (OCR) technology with PDF parsing techniques to deeply analyze documents and accurately extract textual information. For the visual context, we employ high-fidelity conversion tools to save document pages as images, preserving the original layout and non-text elements. This dual representation not only maintains the visual contextual relevance of the document but also ensures the accuracy of text retrieval, providing a multimodal foundation for collaborative optimization in subsequent text-visual retrieval tasks. 

**Fig. 2.** Overview of M[2 ] RAG. A multi-agent multimodal fusion framework composed of three stages: (1) Document parsing extracts both text segments and page images using OCR and PDF tools. (2) A dual-tower retrieval strategy retrieves the top-k relevant text segments and visual pages from corresponding databases based on the input question. (3) The multi-agent reasoning module consists of a text filter that generates a textbased preliminary answer and evidence chain, a visual extractor that constructs visual reasoning paths, and a modality fuser that performs consistency checking and integrates multimodal information to synthesize the final answer. 

## **3.2 Dual-Tower Retrieval** 

In this stage, a dual-tower hybrid retrieval mechanism is employed to fully leverage the multimodal information in the documents. In the text retrieval pipeline, we first segment the parsed document content and construct an inverted index using the BM25 retrieval algorithm based on term frequency and inverse document frequency. Given a user question . _q_ , BM25 [22] retrieves the top-k most 

512 T. Duan et al. 

relevant text segments, forming the text context set . _Tq_ = _{t_ 1 _, t_ 2 _, . . . , tk}_ . Simultaneously, to preserve the visual information in the original document (such as images, charts, titles, etc.), we use VisRAG-Ret [29] to perform visual retrieval on document page images to extract visual context. Both the text query and the page images are encoded into embedding vectors, and their cosine similarity is computed to return the top-k most relevant pages . _Pq_ = _{p_ 1 _, p_ 2 _, . . . , pk}_ . This dual-tower design integrates the contextual information from both text and visual dimensions, providing a richer semantic foundation for subsequent question-answer generation. 

**Algorithm 1.** Dual-Tower Retrieval 

**Require:** Question _q_ , Document corpus _C_ , Text relevance scores _St_ , Page image relevance scores _Sp_ , Text scoring function _Rt_ , Page image scoring function _Rp_ **Ensure:** Top- _k_ text segments _Tq_ , Top- _k_ image segments _Pq_ 1: _St ← {} ▷_ Initialize text scores 2: _Sp ← {} ▷_ Initialize page image scores 3: **for** each document _D_ in _C_ **do** 4: **for** each text segment _t_ in _D_ **do** 5: _St_ [ _t_ ] _← Rt_ ( _q, t_ ) _▷_ Calculate text relevance score 6: **end for** 7: **for** each page _p_ in _D_ **do** 8: _Sp_ [ _p_ ] _← Rp_ ( _q, p_ ) _▷_ Calculate page image relevance score 9: **end for** 10: **end for** 11: _Tq ←_ Top_K( _St, k_ ) _▷_ Select top- _k_ most relevant text segments 12: _Pq ←_ Top_K( _Sp, k_ ) _▷_ Select top- _k_ most relevant page images 13: **return** _Tq, Pq_ 

## **3.3 Fine Tuning VLM Driven by Document Question Answering** 

To enhance the performance of Vision Language Models in document question answering, our proposes a dataset construction and fine tuning strategy guided by the DocQA objective. DocQA tasks require models not only to understand natural language but also to interpret visual elements in document images, such as layout structures, table formats, and graphical content, in order to perform multimodal reasoning. 

We first construct a high quality distillation-based training dataset using the publicly available MP-DocVQA [26] dataset. Given an input . _x_ , we prompt a teacher model . _MT_ with task instruction . I and visual context . _c_ to generate an output . _y_ with reasoning traces. The labeled dataset can be formulated as: 

**==> picture [276 x 13] intentionally omitted <==**

M[2] RAG 513 

We then perform supervised fine tuning on a student model . _MS_ using the distillation dataset. The objective is to align the student predictions with the teacher outputs. The training loss is defined as: 

**==> picture [276 x 13] intentionally omitted <==**

## **3.4 Multi-agent Generation Based on Modal Fusion** 

**Text Filter.** The text filter aims to further process the set of retrieved text segments . _Tq_ = _{t_ 1 _, t_ 2 _, . . . , tk}_ from the document, focusing on text segments containing key information. The core function is to perform deep processing of the text, including semantic parsing, key information extraction, and contextual relevance analysis, in order to generate a preliminary answer . _aT_ for the question . _q_ . During the answer generation process, a set of highly relevant evidence . _ET_ is filtered from the text, where . _ET ⊂ Tq_ . This evidence is presented in the form of text segments and provides strong support for the preliminary answer. The agent is capable of fusing and logically extracting text from different sources, thereby improving the interpretability and traceability of the question-answering process. 

**==> picture [223 x 12] intentionally omitted <==**

**Visual Extractor.** The visual extractor, powered by the fine tuned vision language model . _M_ LoRA, is dedicated to processing the set of retrieved page images . _Pq_ = _{p_ 1 _, p_ 2 _, . . . , pk}_ for multimodal reasoning. Its core task is to extract and semantically analyze key information embedded in visual elements (e.g., images, tables, and text blocks) within the images, thereby generating a preliminary answer . _aV_ to the query. Simultaneously, this agent selects the most relevant images from the retrieved pages as supporting evidence, which visually illustrate the answer to the query. This agent is capable of performing precise reasoning while preserving the visual context, effectively complementing the limitations of purely textual information. 

**==> picture [243 x 12] intentionally omitted <==**

**Modal Fuser.** The modality fusion agent possesses cross-modal integration and generation capabilities. Its core function is to consolidate the outputs from the text information extraction agent and the visual information extraction agent to generate the final answer . _aF_ . By fusing the textual answer . _aT_ , textual evidence . _ET_ , visual answer . _aP_ , and visual evidence . _EP_ , this agent employs a comprehensive reasoning model to perform integration and logical inference, resulting in a complete and accurate final answer. This agent is built upon the fine-tuned vision language model . _M_ LoRA, which is capable of handling multiple image inputs and performing text-vision fusion. By aligning linguistic cues with visual evidence, the agent achieves unified modeling and joint reasoning over multimodal information. It fully leverages the strengths of both textual and visual modalities, providing more informative and precise answers through multimodal integration. 

514 T. Duan et al. 

## **Algorithm 2.** Multi-Agent Modal Fusion Generation 

**Input:** Question _q_ , Top- _k_ text segments _Tq_ , Top- _k_ page images _Pq_ **Output:** Final answer _aF_ 

1: ( _aT , ET_ ) _← Θ_ text( _q, Tq_ ) 2: ( _aV , EV_ ) _← Θ_ vis( _q, Pq_ ; _M_ LoRA) 3: **if** _aT_ = "no information" **then** 4: _aF ← AF_ ( _q, aV , EV_ ) 5: **else if** _aV_ = "no information" **then** 6: _aF ← AF_ ( _q, aT , ET_ ) 7: **else** 8: _aF ← AF_ ( _q, aT , ET , aV , EV_ ) 9: **end if** 10: **return** _aF_ 

_▷_ Text Filter answer and evidence _▷_ Visual Extractor answer and evidence 

## **4 Experiments** 

## **4.1 Experimental Settings** 

**Implementation Details.** Our M[2] RAG framework employs a dual-tower retrieval architecture, utilizing BM25 [22] for text retrieval and VisRAG-Ret [29] for image-based retrieval. The generation part is powered by three agents: a Text Filter, a Visual Extractor, and a Modal Fuser. Specifically, the Text Filter is based on Qwen2.5-7B-Instruct, while the Visual Extractor and Modal Fuser are built upon a fine-tuned version of Qwen2.5-VL-7B-Instruct [2]. We retrieve the top-1, top-3, and top-5 highest-scoring text and visual segments as input context for the multi-agent generation component. The fine-tuning process involves using Qwen2.5-VL-7B-Instruct as the student model and Qwen2.5-VL-32B-Instruct as the teacher model, trained on the fine-tuning dataset we constructed. The finetuning is conducted for 3 epochs on a single NVIDIA A800 80GB GPU with a batch size of 16. 

**Datasets.** The benchmarks involve DocBench [30], MMLongBench [16], and LongDocURL [7], which cover a wide range of scenarios including open-domain and closed-domain settings, long and short documents, as well as both textual and visual content. This ensures a comprehensive and fair evaluation across different task types and modalities. 

**Evaluation Metrics.** For all benchmarks, we follow the standard evaluation protocols and use GPT-4o [1] as the evaluator to assess the consistency between the model’s output and the reference answers. A binary decision (correct/incorrect) is made for each instance, and the final metric is reported as the average accuracy across each benchmark dataset. 

## **4.2 Main Results** 

To validate the effectiveness of our proposed M[2] RAG framework, we conducted a comprehensive evaluation on multiple benchmark datasets, comparing its per- 

M[2] RAG 515 

formance with two baseline methods: a text-based RAG (BM25 + Qwen2.5-7BInstruct) and a image-based RAG (VisRAG-Ret + Qwen2.5-VL-7B-Instruct). 

**Table 1.** Performance comparison across M[2 ] RAG and existing RAG-based methods. 

|Method|DocBench|MMLongBench|LongDocUrl|Avg|
|---|---|---|---|---|
|_Top1_|||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|35.88<br>59.19<br>**65.07**|10.11<br>29.63<br>**32.11**|20.60<br>47.31<br>**49.42**|22.20<br>45.38<br>**48.87**|
|_Top3_|||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|44.99<br>72.06<br>**74.45**|13.60<br>36.01<br>**38.87**|23.35<br>54.67<br>**56.99**|37.31<br>54.25<br>**56.77**|
|_Top5_|||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|47.36<br>72.98<br>**76.29**|15.40<br>38.84<br>**39.90**|16.97<br>56.34<br>**57.68**|26.57<br>56.05<br>**57.96**|



As shown in the Table 1, our method consistently outperforms both baselines across all evaluation benchmarks. Under the top-1 retrieval setting, M[2] RAG achieves an average score of 48.87% across the three datasets. Compared to the baselines, our approach yields an average accuracy improvement of 26.67% over TextRAG and 3.49% over VisualRAG, representing a 7.7% performance enhancement. This highlights the advantages of modal fusion and multi-agent collaboration within our framework. 

With the top-3 retrieved text and visual segments as input, M[2] RAG demonstrates superior accuracy across various benchmarks compared to the baseline methods. Specifically, M[2] RAG achieves an average accuracy of 56.77%, which is 2.52% higher than the second-best performing VisualRAG, which has an average accuracy of 54.25%. This represents a performance improvement of 4.64% over VisualRAG, highlighting the effectiveness of M[2] RAG in leveraging multimodal information for enhanced reasoning capabilities. Additionally, when using the top-5 retrieved segments as input, our method continues to outperform the baseline methods, further underscoring the effectiveness of M[2] RAG in utilizing multimodal information to enhance reasoning capabilities. 

These experimental results clearly demonstrate that the combination of dualtower retrieval and multi-agent reasoning in M[2] RAG effectively leverages multimodal information fusion, leading to substantial improvements in overall performance for document question answering tasks. 

516 T. Duan et al. 

## **4.3 Ablations** 

To further validate the effectiveness of the key components in the proposed M[2] RAG framework, we conducted comprehensive ablation studies under a consistent experimental setup where the top-5 retrieved textual and visual information was used as input across all benchmarks and calculated the average accuracy. These studies specifically examine the impact of two critical components: (1) the multi-agent collaboration module and (2) the fine-tuning strategy on the overall performance. The experimental results are summarized in Table 2. 

**Table 2.** Ablation study on the effects of Multi-Agent and Fine Tuning Components. 

|Component<br>Multi-Agent<br>Fine-Tuning|Component<br>Multi-Agent<br>Fine-Tuning|DocBench|MMLongBench|LongDocUrl|Avg|
|---|---|---|---|---|---|
|||||||
|✗|✗|50.55|36.72|52.58|46.62|
|✓|✗|75.18|39.26|57.01|57.15|
|✓|✓|**76.29**|**39.90**|**57.68**|**57.96**|



In the first ablation setting, we removed both the multi-agent module and the fine-tuning. The retrieved multimodal context was directly fed into the VisionLanguage Model (Qwen2.5-VL-7B) to generate answers. Under this configuration, the model achieved an average accuracy of only 46.62%, significantly lower than the full model 57.96%. This result indicates that relying solely on a visionlanguage model to process long multimodal contexts, without structured information handling, considerably degrades performance. This performance drop may be attributed to the limitations of current vision-language models in reasoning over long contexts and visual inputs simultaneously, which hinders their ability to capture complex semantic and visual cues embedded in document data. 

When the multi-agent mechanism was introduced without applying finetuning, the performance improved markedly, with an average accuracy reaching 57.15%. This demonstrates the clear advantage of the multi-agent architecture in semantic extraction, information filtering, and modality fusion, effectively alleviating the burden imposed by long-context reasoning on vision-language models. 

Under the full configuration of M[2] RAG, our framework integrates the multiagent module with a fine-tuned vision-language model, achieving the best performance on all benchmarks. These results further validate that (1) multi-agent collaboration effectively leverage fine-tuned representations to enhance reasoning, and (2) task-adapted fine-tuning improves model capabilities in specific scenarios. 

## **4.4 Fine-Grained Performance Analysis** 

To further analyze the performance of our proposed M[2] RAG framework under different types of evidence, we conducted a fine-grained evaluation on three 

M[2] RAG 517 

benchmarks: MMLongBench, DocBench, and LongDocURL. We analyze the performance based on the top-1, top-3, and top-5 retrieved text and visual segments as input across different benchmark datasets. 

**Table 3.** Performance comparison across different evidence source on MMLongBench. 

|Method|Chart|Table|Pure-text|Layout|Figure|Avg|
|---|---|---|---|---|---|---|
|_Top1_|||||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|9.14<br>28.09<br>**29.21**|5.63<br>27.19<br>**29.50**|13.68<br>29.21<br>**31.62**|10.34<br>22.88<br>**26.27**|7.14<br>25.52<br>**28.28**|10.11<br>29.63<br>**32.11**|
|_Top3_|||||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|13.14<br>33.71<br>**35.39**|5.16<br>30.88<br>**35.94**|21.40<br>37.80<br>**39.52**|10.34<br>30.51<br>**33.05**|8.93<br>32.07<br>**34.13**|13.60<br>36.01<br>**38.87**|
|_Top5_|||||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|15.43<br>32.02<br>**34.27**|7.04<br>**32.26**<br>**32.26**|23.16<br>38.83<br>**40.89**|15.52<br>33.90<br>**33.90**|8.57<br>**35.86**<br>35.52|15.40<br>38.84<br>**39.90**|



We present the fine-grained performance comparison on the MMLongBench benchmark, as shown in Table 3. Under Top-1, Top-3, and Top-5 retrieval settings, the average accuracy of M[2] RAG is always higher than the two baseline methods TextRAG and VisualRAG. Compared with the best performing baseline method VisualRAG, the results are improved by 2.48%, 2.86%, and 1.06%, respectively. Although VisualRAG performs slightly better in the Figure category, M[2] RAG demonstrates stronger performance in Chart, Table, Layout, and Text categories, highlighting its superior capability in effectively processing diverse types of information. 

On DocBench (Table 4), our M[2] RAG framework consistently outperforms the two baseline methods across all three types of evidence, further verifying the effectiveness of the multi-agent architecture and modality fusion mechanism in document understanding tasks. Moreover, the model shows stable reasoning capability when dealing with tabular and graphical evidence. 

We present the fine-grained performance comparison on the LongDocURL benchmark, as shown in Table 5. Under Top-1, Top-3, and Top-5 retrieval Settings, the average accuracy of M[2] RAG is always higher than the two baseline methods TextRAG and VisualRAG. Compared with the best performing baseline method VisualRAG, the results are improved by 2.11%, 2.32% and 1.34%, respectively. These results verify the higher effectiveness of M[2] RAG in dealing with multi-modal evidence information in complex long documents. 

518 T. Duan et al. 

**Table 4.** Performance comparison across different evidence source on DocBench. 

|Method|Text|Table<br>|Figure|Avg|
|---|---|---|---|---|
|_Top1_|||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|56.43<br>70.36<br>**74.29**|10.77<br><br>47.67<br><br>**55.96**<br>|24.32<br>46.48<br>**53.52**|35.88<br>59.19<br>**65.07**|
|_Top3_|||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|69.64<br>81.07<br>**84.29**|14.36<br><br>64.25<br><br>**64.77**<br>|32.43<br>57.75<br>**61.97**|44.99<br>72.06<br>**74.45**|
|_Top5_|||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|71.79<br>83.57<br>**87.50**|16.41<br><br>61.66<br><br>**64.24**<br>|36.49<br>61.97<br>**64.79**|47.36<br>72.98<br>**76.29**|



**Table 5.** Performance comparison across different evidence source on LongDocURL. 

|Method|Layout|Text|Figure|Table|Others|Avg|
|---|---|---|---|---|---|---|
|_Top1_|||||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|15.40<br>36.80<br>**39.53**|32.49<br>53.72<br>**56.94**|21.40<br>50.54<br>**51.08**|11.41<br>45.12<br>**46.04**|12.50<br>**87.75**<br>62.50|20.60<br>47.31<br>**49.42**|
|_Top3_|||||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|20.15<br>43.00<br>**46.85**|36.92<br>64.29<br>**64.59**|23.56<br>57.73<br>**58.10**|11.41<br>51.66<br>**55.68**|12.50<br>**87.50**<br>75.00|23.35<br>54.67<br>**56.99**|
|_Top5_|||||||
|TextRAG<br>VisualRAG<br>**M**.**2RAG(Ours)**|20.54<br>45.70<br>**47.37**|41.35<br>65.79<br>**66.10**|28.24<br>57.55<br>**57.91**|16.53<br>53.04<br>**56.14**|12.50<br>**87.50**<br>75.00|26.97<br>56.34<br>**57.68**|



In summary, M[2] RAG effectively captures multi-modal evidence and enhances the generalization and robustness of the model in complex multimodal document question answering tasks. 

## **5 Conclusion** 

In this paper, we propose a multimodal Retrieval-Augmented Generation (RAG) framework, termed M[2] RAG, which integrates dual-tower retrieval and multiagent generation to enhance performance in document question answering 

M[2] RAG 519 

(DocQA) tasks. By employing a specialized set of agents to effectively integrate textual and visual information, our framework addresses the limitations of unimodal approaches when dealing with long texts and complex document layouts. Experimental results on multiple benchmark datasets demonstrate that M[2] RAG consistently achieves significant performance gains, underscoring the effectiveness of our multimodal fusion strategy. The framework effectively mitigates information overload and facilitates deep cross-modal understanding, enabling more accurate and comprehensive answers in DocQA tasks. Future work will focus on further optimizing the framework by exploring advanced agent communication mechanisms and incorporating external knowledge sources. 

## **References** 

1. Achiam, J., et al.: GPT-4 technical report. arXiv preprint arXiv:2303.08774 (2023) 

2. Bai, S., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 

3. Bang, Y., et al.: A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023 (2023) 

4. Chan, C.M., et al.: Chateval: towards better LLM-based evaluators through multiagent debate. arXiv preprint arXiv:2308.07201 (2023) 

5. Chen, W., Hu, H., Chen, X., Verga, P., Cohen, W.W.: Murag: multimodal retrievalaugmented generator for open question answering over images and text. arXiv preprint arXiv:2210.02928 (2022) 

6. Cho, J., Mahata, D., Irsoy, O., He, Y., Bansal, M.: M3docrag: multi-modal retrieval is what you need for multi-page multi-document understanding. arXiv preprint arXiv:2411.04952 (2024) 

7. Deng, C., et al.: Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. arXiv preprint arXiv:2412.18424 (2024) 

8. Ding, Y., Luo, S., Chung, H., Han, S.C.: VQA: a new dataset for real-world VQA on pdf documents. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 585–601. Springer (2023) 

9. Ghosh, S., et al.: VDGD: mitigating LVLM hallucinations in cognitive prompts by bridging the visual perception gap. arXiv e-prints pp. arXiv-2405 (2024) 

10. Ji, Z., et al.: Survey of hallucination in natural language generation. ACM Comput. Surv. **55** (12), 1–38 (2023) 

11. Kannan, S.S., Venkatesh, V.L., Min, B.C.: Smart-LLM: smart multi-agent robot task planning using large language models. In: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 12140–12147. IEEE (2024) 

12. Lewis, P., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. Adv. Neural. Inf. Process. Syst. **33** , 9459–9474 (2020) 

13. Li, B., Wang, Y., Gu, J., Chang, K.W., Peng, N.: Metal: a multi-agent framework for chart generation with test-time scaling. arXiv preprint arXiv:2502.17651 (2025) 

14. Li, Y., et al.: Benchmarking multimodal retrieval augmented generation with dynamic VQA dataset and self-adaptive planning agent. arXiv preprint arXiv:2411.02937 (2024) 

15. Ma, X., Zhuang, S., Koopman, B., Zuccon, G., Chen, W., Lin, J.: Visa: retrieval augmented generation with visual source attribution. arXiv preprint arXiv:2412.14457 (2024) 

520 T. Duan et al. 

16. Ma, Y., et al.: Mmlongbench-doc: benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523 (2024) 

17. Mathew, M., Karatzas, D., Jawahar, C.: Docvqa: a dataset for VQA on document images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2200–2209 (2021) 

18. Memon, J., Sami, M., Khan, R.A., Uddin, M.: Handwritten optical character recognition (OCR): a comprehensive systematic literature review (SLR). IEEE Access **8** , 142642–142668 (2020) 

19. Mialon, G., et al.: Augmented language models: a survey. arXiv preprint arXiv:2302.07842 (2023) 

20. Mishra, A., Shekhar, S., Singh, A.K., Chakraborty, A.: OCR-VQA: visual question answering by reading text in images. In: 2019 International Conference on Document Analysis and Recognition (ICDAR), pp. 947–952. IEEE (2019) 

21. Ramesh, A., et al.: Zero-shot text-to-image generation. In: International Conference on Machine Learning, pp. 8821–8831. PMLR (2021) 

22. Robertson, S.E., Walker, S., Jones, S., Hancock-Beaulieu, M.M., Gatford, M., et al.: Okapi at TREC-3. NIST Special Publication SP **109** , 109 (1995) 

23. Smith, R.: An overview of the tesseract OCR engine. In: Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), vol. 2, pp. 629–633. IEEE (2007) 

24. Suri, M., Mathur, P., Dernoncourt, F., Goswami, K., Rossi, R.A., Manocha, D.: Visdom: multi-document QA with visually rich elements using multimodal retrievalaugmented generation. arXiv preprint arXiv:2412.10704 (2024) 

25. Tanaka, R., Nishida, K., Nishida, K., Hasegawa, T., Saito, I., Saito, K.: Slidevqa: a dataset for document visual question answering on multiple images. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, pp. 13636–13645 (2023) 

26. Tito, R., Karatzas, D., Valveny, E.: Hierarchical multimodal transformers for multipage docvqa. Pattern Recogn. **144** , 109834 (2023) 

27. Wang, Q., et al.: Vidorag: visual document retrieval-augmented generation via dynamic iterative reasoning agents. arXiv preprint arXiv:2502.18017 (2025) 

28. Xu, Z., et al.: Activerag: autonomously knowledge assimilation and accommodation through retrieval-augmented agents. arXiv preprint arXiv:2402.13547 (2024) 

29. Yu, S., et al.: Visrag: vision-based retrieval-augmented generation on multimodality documents. arXiv preprint arXiv:2410.10594 (2024) 

30. Zou, A., et al.: Docbench: a benchmark for evaluating LLM-based document reading systems. arXiv preprint arXiv:2407.10701 (2024) 

