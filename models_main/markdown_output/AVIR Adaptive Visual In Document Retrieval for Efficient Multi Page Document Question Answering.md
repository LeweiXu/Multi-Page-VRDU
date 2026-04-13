# **AVIR: Adaptive Visual In-Document Retrieval for Efficient Multi-Page Document Question Answering** 

Yachuan Li[∗] 

Zongmin Li 

Lei Kang 

China University of Petroleum (East China University of Petroleum (East China) China) Qingdao, Shandong, China Qingdao, Shandong, China Shandong Xiehe University Universitat Autonoma de Barcelona Jinan, Shandong, China Cerdanyola, Barcelona, Spain lizongmin@upc.edu.cn liyachuan@s.upc.edu.cn 

Universitat Autonoma de Barcelona Cerdanyola, Barcelona, Spain lkang@cvc.uab.es 

Dimosthenis Karatzas 

Wenkang Ma China University of Petroleum (East China) Qingdao, Shandong, China s23070027@s.upc.edu.cn 

Universitat Autonoma de Barcelona Cerdanyola, Barcelona, Spain dimos@cvc.uab.cat 

## **1 Introduction** 

## **Abstract** 

Multi-page Document Visual Question Answering (MP-DocVQA) remains challenging because long documents not only strain computational resources but also reduce the effectiveness of the attention mechanism in large vision–language models (LVLMs). We tackle - these issues with an Adaptive Visual In document Retrieval (AVIR) framework. A lightweight retrieval model first scores each page for question relevance. Pages are then clustered according to the score distribution to adaptively select relevant content. The clustered pages are screened again by Top-K to keep the context compact. However, for short documents, clustering reliability decreases, so we use a relevance probability threshold to select pages. The selected pages alone are fed to a frozen LVLM for answer generation, - eliminating the need for model fine tuning. The proposed AVIR framework reduces the average page count required for question answering by 70%, while achieving an ANLS of 84.58% on the MPDocVQA dataset—surpassing previous methods with significantly lower computational cost. The effectiveness of the proposed AVIR is also verified on the SlideVQA and DUDE benchmarks. The code is available at https://github.com/Li-yachuan/AVIR-main. 

Multi-page documents, such as financial reports, slide decks, con- tracts, and scientific papers, are pervasive in real world workflows. Enabling models to read these lengthy documents and answer open-ended questions. Multi-Page Document Visual Question An- swering (MP DocVQA) is therefore a critical capability for modern Document-AI systems. 

However, our experimental analysis reveals that while current Large Vision–Language Models (LVLMs) perform well on short, single-page inputs, they encounter two fundamental limitations when extended to long documents. (1) Computation cost: Encod- ing dozens of pages with quadratic self attention quickly becomes prohibitive, making naïve end-to-end inference impractical for real-time or resource-constrained settings. (2) Context dilution: As a document grows, only a handful of pages contain information relevant to a given question, while the remaining pages introduce distracting or contradictory context that harms answer quality. 

A natural remedy is to retrieve the most relevant pages first and - run VQA only on that subset. Yet existing retrieval based pipelines still fall short because they often depend on heavy cross-encoders whose overhead rivals the LVLM itself, and they rely on a rigid Top- _𝐾_ cutoff, so any retrieval error irrevocably removes the supporting page and cascades into a wrong answer. 

## **CCS Concepts** 

## • **Computing methodologies** → **Information extraction** . 

This paper proposes an Adaptive Visual In-Document Retrieval (AVIR) framework that tackles both issues simultaneously. We de- sign a lightweight Pix2Struct based retriever to score each page, followed by an adaptive page selector that analyzes the score distribution: for short documents, it applies threshold filtering; for longer ones, it clusters pages into relevant versus irrelevant groups and keeps the Top-K pages of the relevant pages. This dynamic strategy preserves recall while aggressively pruning noise, making - the downstream LVLM, a quantized Qwen2.5 VL, both faster and - more accurate. Crucially, Qwen2.5 VL remains frozen; prompt engi- neering alone provides strong cross domain generalization without costly fine-tuning. 

## **Keywords** 

Adaptive Page Selection , Multi-Page Document Question Answering, Qwen2.5-vl 

## **ACM Reference Format:** 

Zongmin Li, Yachuan Li, Lei Kang, Dimosthenis Karatzas, and Wenkang Ma. 2025. AVIR: Adaptive Visual In-Document Retrieval for Efficient Multi-Page Document Question Answering. In _ACM Multimedia Asia (MMAsia ’25), December 9–12, 2025, Kuala Lumpur, Malaysia._ ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3743093.3771020 

Our main contributions are threefold: 

∗Corresponding author 

MMAsia ’25, December 9–12, 2025, Kuala Lumpur, Malaysia 

Zongmin Li et al. 

- (1) A systematic analysis showing how irrelevant pages impair current LVLMs and how lightweight retrieval restores efficiency. 

- (2) An adaptive page selector that mitigates retrieval errors by data-driven thresholding and clustering, outperforming rigid Top- _𝐾_ baselines. 

- - 

- (3) A complete retrieval guided MP VQA pipeline that is efficient (only selected pages are processed) yet achieves superior performance on comprehensive public benchmarks of MP-DocVQA, SlideVQA, and DUDE. 

## **2 Related Work** 

Multi-page Document Visual Question Answering (MP-DocVQA) enables models to comprehend and respond to questions based on complex, multi-page documents. Due to challenges such as lengthy content, intricate layouts, and cross-page dependencies, conventional single-page VQA methods exhibit limited effectiveness in this domain. Existing approaches to MP-VQA typically fall into three categories: sequence-based methods, structured fusion approaches, and retrieval-based frameworks. 

## **2.1 Sequence-based MP-VQA** 

- Sequence-based methods treat multi page documents as _flattened sequences_ and apply Transformer architectures originally devised for long-form language modelling. Early work such as Longformer [4] and BIGBIRD [29] extends local or sparse global attention to accommodate thousands of tokens. RM-T5 [11] introduces a recurrent - memory cache so that a fixed size core iteratively processes page chunks and accumulates context. GRAM [6] pushes the idea further by adding global reasoning tokens that attend to every page - token, enabling holistic cross page aggregation within a single encoder–decoder pass. Arctic-TILT [7] revisits the TILT architecture - with byte level tokenisation and lightweight sparse attention, scaling end-to-end processing up to 400k tokens on commodity GPUs. Such models eliminate explicit page segmentation but still incur heavy memory footprints on very long documents. 

## **2.2 Structured Fusion MP-VQA** 

Structured fusion methods explicitly model the hierarchical, spatial, and semantic relationships inherent in multi-page documents to better capture their complex structure. For instance, Hi-VT5[26] extends sequence-to-sequence models with a hierarchical encoderdecoder that processes page tokens at multiple granularities, enabling effective cross-page information integration. DocFormerv2[1] employs multi-window two-dimensional attention combined with squeezed layout tokens to achieve fine-grained layout reasoning and spatial understanding. These approaches build structured representations that jointly capture visual and textual relationships critical for multi-page document understanding. Despite their strong performance, they often introduce significant computational overhead, which motivates the development of lighter and more efficient methods. 

## **2.3 Retrieval-based MP-VQA** 

Retrieval-based approaches decompose multi-page visual question answering (MP-VQA) into two distinct stages: first, retrieving the 

**Figure 1: The influence of document length on the accuracy of MP-VQA. The experiment is conducted on the validation set of MP-DocVQA, where the answers are usually on a single page. ’w/ doc’ indicates using the entire document, while ’w/ true page’ means only the correct page is used. Due to the uneven distribution of page lengths, we have only selected the first 202 documents for each category.** 

most relevant document pages, and second, performing question answering on the selected pages. Typically, existing methods rely on a straightforward Top-K page selection strategy, where only the top-ranked pages based on retrieval scores are passed to the reader model. M3DOCRAG[9] improves retrieval quality by employing more sophisticated retrieval models that better capture relevance across pages, but these improvements often come with increased computational costs. More recently, SelfAttnScoring[16] introduces lighter-weight retrieval modules designed to enhance efficiency. Despite these gains, it still depends on a fixed Top-K selection mechanism, which leaves it vulnerable to the same retrieval error issues and limits its robustness. 

Thus, our proposed method addresses the limitations of existing retrieval-based frameworks through an **adaptive in-document retrieval guidance approach** , effectively balancing efficiency and accuracy. Specifically, we introduce a lightweight retrieval module coupled with a novel adaptive page selector, dynamically classifying relevant pages based on the distribution of retrieval scores. This adaptive mechanism significantly mitigates the impact of retrieval errors, reduces irrelevant context processing, and greatly enhances computational efficiency. 

## **3 Proposed Methodology** 

## **3.1 Question Definition** 

Although current large-scale visual-language models (LVLMs) can directly handle multi-page documents, their performance degrades significantly as the document length increases. We evaluate this problem on the MP-DocVQA validation set using Qwen2.5-VL (3B) and interVL3 (2B) as examples, and the results are shown in Fig. 1. 

When the document contains only 1-2 pages, both models achieve high accuracy when using the entire document or only relevant pages. However, when the document length exceeds 3 pages, the performance of both models degrades significantly when processing 

AVIR: Adaptive Visual In-Document Retrieval for Efficient Multi-Page Document Question Answering 

MMAsia ’25, December 9–12, 2025, Kuala Lumpur, Malaysia 

the entire document. The significant performance gap between the full document setting and the ground truth page setting confirms that irrelevant context introduces significant noise. 

The performance degradation of interVL3 worsens with increasing document length, especially when the document length exceeds 8 pages. Although Qwen2.5-VL shows higher robustness than interVL3, its performance on longer documents is still significantly degraded compared to only relevant pages Q&A. 

Despite the differences between the two models, they both show a consistent trend: LVLM performs significantly worse on long documents than on short documents. This highlights the challenges that current models face in handling long-range dependencies and context clutter in multi-page settings. 

## **3.2 Overview** 

To alleviate the performance degradation of large models on long documents, retrieval-augmented generation (RAG) has emerged as a promising paradigm. However, existing RAG-based methods are highly sensitive to retrieval errors, which can lead to significant performance degradation. 

To mitigate the impact of retrieval errors and enhance overall performance, we propose a novel adaptive visual in-document retrieval framework, termed AVIR, tailored for multi-page document question answering (MP-DQA), as illustrated in Fig. 2. Our method follows the general RAG pipeline but introduces key innovations to improve retrieval robustness and answer accuracy. Specifically, the framework consists of three sequential components: 

**Page Retrieval** : We employ a page-level retriever that estimates the semantic relevance between each document page and the given question. **Adaptive Page Selection** : Based on the initial relevance scores, an adaptive page selector—designed to consider both score distribution and inter-page relationships—is introduced to select the most informative page(s) for downstream reasoning. **Answer Generation** : Finally, the selected pages and the question are fed into a large pre-trained LVLM for answer generation. To balance accuracy and computational efficiency, we adopt Qwen2.5-VL [3] as our backbone LVLM. 

In the following sections, we provide a detailed introduction to the page retrieval module and our adaptive page selector. The architecture can significantly reduce retrieval noise and enhance the performance of long-document VQA systems. 

## **3.3 Page Retrieval Model** 

An essential component of retrieval-augmented generation (RAG) frameworks is the page retrieval module, which identifies the most relevant pages from a multi-page document in response to a given question. While several high-performance retrieval models have been proposed—such as ColBERTa-based architectures like ColPali [12] and document structure-enhanced models like DSE [19]—they typically require substantial computational resources. This becomes particularly problematic in the multi-page setting, where the retrieval model must evaluate every individual page in the document. In some cases, the computational cost of such large-scale retrieval models even exceeds that of the answer generation module itself. 

To ensure the end-to-end efficiency and practical applicability of the framework, we adopt a lightweight and efficient page retrieval 

model. Inspired by SelfAttnScoring [16], we leverage the encoder of Pix2Struct [17], a visual language model designed for document understanding, to extract visual semantic features from each page. On top of the frozen encoder, a custom relevance scoring head is built, consisting of two self-attention layers and a sigmoid activation function that maps page-question feature alignments to relevance probability space. 

Our retrieval module contains only approximately 100 million parameters, which is nearly 4% the size of the downstream LVLM used for question answering. While this lightweight design inevitably limits the model’s generalization capability, requiring task-specific fine-tuning on each dataset, we find this trade-off acceptable. In practice, the gain in inference efficiency significantly outweighs the one-time fine-tuning cost, especially when deploying the system at scale. 

## **3.4 Adaptive Page Selection** 

The adaptive page selector serves as a critical component of our proposed framework. It is specifically designed to address three major challenges in multi-page document question answering (MPDQA): 

- (1) Mitigating the performance degradation of large-scale LVLMs on long documents, as discussed in Section 3.1; 

- (2) Improving computational efficiency by reducing the number of pages passed to the answer generation module; 

- (3) Reducing the negative impact of retrieval errors introduced during the initial page relevance estimation stage. 

To this end, we propose a dynamic selection strategy that adapts to the number of pages in a document and the distribution of their relevance scores. 

For longer documents (i.e., those exceeding four pages), we first employ a relevance-based clustering strategy to categorize pages into two groups: relevant and irrelevant. If the number of relevant pages exceeds a predefined threshold (set to 8), we apply a hard cutoff by selecting the top 8 pages with the highest relevance scores. This step is essential for preserving answer quality, as supplying too many pages to the LVLM not only introduces noise but also increases the reasoning complexity, which can negatively impact performance. Empirical results show that the top 8 most relevant pages are typically sufficient to encompass the information necessary to answer the vast majority of questions. 

While the clustering strategy works well for long documents, it is less suitable for short documents (fewer than four pages), where forced binary classification may lead to the exclusion of relevant pages with marginally lower relevance scores. For instance, if a three-page document yields relevance probabilities of 0.34, 0.33, and 0.33, clustering would retain only the first page, despite all three being similarly relevant. Such close score distributions are more common in short documents. As shown in Fig. 1, the question-answering performance on short documents is generally less sensitive to the number of included pages. Therefore, we adopt a threshold-based filtering strategy for this case. Specifically, if any page has a relevance score above a predefined threshold (set to 0.6 in our experiments), only those pages are retained; otherwise, all pages are preserved. 

MMAsia ’25, December 9–12, 2025, Kuala Lumpur, Malaysia 

Zongmin Li et al. 

**==> picture [493 x 229] intentionally omitted <==**

**----- Start of picture text -----**<br>
Document<br>277 Document<br>= ue<br>True False<br>Page>4<br>E ! ‘. *<br>H 1 A A NN<br>ii | A’ N False<br>What is the name of the  H AAs Clustering P(x0)>T<br>company? t A True<br>Question ee False<br>XN<br>Retrieval Model N Page>8 Top-1<br>XN<br>XN<br>XN True<br>XN<br>XN<br>N<br>XN<br>XN Top-8 All pages<br>XN<br>__ ee || N\<br>ITC Limited LVLM<br>Pages<br>=Ta<br>Answer Question-answering Model Selected Pages Adaptive page selector<br>Encoder Sigmoid<br>Transformer Self Attention Self Attention<br>**----- End of picture text -----**<br>


**Figure 2: Overview of the proposed AVIR.** _𝑃_ ( _𝑥_ 0) **means The relevance probability assigned by the retrieval model to the most relevant page.** 

This simple yet effective approach helps prevent mistakenly discarding relevant pages in short documents, thereby improving the robustness and error tolerance of the overall system. 

In summary, this adaptive selection mechanism strikes a balance between retrieval robustness and computational efficiency. It ensures that highly relevant information is retained while excluding irrelevant or marginally relevant content, thereby enhancing the reliability and effectiveness of the downstream answer generation process. The pseudo code of the adaptive page selector is shown in Algorithm 1. 

## **4 Experiments 4.1 Dataset and Metrics** 

We evaluate our model on three multi-page document visual question answering (MP-VQA) datasets, each designed to capture different challenges associated with understanding long-form documents: SlideVQA, MP-DocVQA, and DUDE. 

**SlideVQA** [24] contains presentation-style documents averaging 20 pages. Questions reference content spread across multiple slides, requiring sequential, layout-aware integration of visual and textual information. Evaluation uses Exact Match (EM) accuracy and tokenlevel F1 score, emphasizing temporal reasoning and structured document understanding. **MP-DocVQA** [26], built on DocVQA[20], focuses on multi-page documents averaging 8.3 pages with dense content. Questions often span spatially distant regions. Evaluation metrics include Accuracy and ANLS [5], covering various formats such as manuals, reports, and forms. **DUDE** [27] comprises 3,000 real-world PDFs (up to 25 pages) across finance, healthcare, and law. It includes 23.7K questions of four types: Extractive, Abstractive, List, and Unanswerable. Evaluated with ANLS, it introduces 

**Algorithm 1:** Adaptive Page Selector with K-Means Clustering **Input:** Document pages _𝑃_ = { _𝑝_ 1 _, 𝑝_ 2 _, . . . , 𝑝𝑛_ }; Relevance scores _𝑅_ = { _𝑟_ 1 _,𝑟_ 2 _, . . . ,𝑟𝑛_ }; Threshold _𝑇_ = 0 _._ 6; Max selected pages _𝐾_ = 8 **Output:** Selected page subset _𝑃_ selected **if** _𝑛 <_ 4 **then if** _exists 𝑖 such that 𝑟𝑖_ ≥ _𝑇_ **then** _𝑃_ selected ←{ _𝑝𝑖_ | _𝑟𝑖_ ≥ _𝑇_ } ; **else** _𝑃_ selected ← _𝑃_ ; // Select all pages **else** Apply K-Means clustering ( _𝑘_ = 2) on _𝑅_ to divide pages into: _𝐶_ rel (relevant cluster), _𝐶_ irrel (irrelevant cluster) ; Let _𝑃_ rel be pages in _𝐶_ rel ; **if** | _𝑃rel_ | _> 𝐾_ **then** Sort _𝑃_ rel by relevance score in descending order ; _𝑃_ rel ← top- _𝐾_ pages in _𝑃_ rel ; _𝑃_ selected ← _𝑃_ rel ; **return** _𝑃selected_ 

domain diversity and reasoning complexity, and offers fine-grained performance analysis by question type. 

AVIR: Adaptive Visual In-Document Retrieval for Efficient Multi-Page Document Question Answering 

MMAsia ’25, December 9–12, 2025, Kuala Lumpur, Malaysia 

## **4.2 Implementation details** 

We use PyTorch, Transformers, and FlashAttention2 libraries for running models. The minimum and maximum pixels of LM are 768 × 28 × 28 and 5120 × 28 × 28, which follow the setting of Qwen2.5 [3]. All experiments are conducted with a single A40 46GB GPU. 

We adopt the pre-trained Qwen2.5-VL-3B [3] as the questionanswering model, without any task-specific fine-tuning. To improve inference efficiency, we use its AWQ-quantized version, denoted as Qwen2.5-VL-3B-AWQ. Since the adaptive page selector is non-parametric, the only trainable component is the page retrieval model, which contains approximately 0.1B parameters and is trained following the SelfAttnScoring strategy [16]. 

is closest in accuracy, trailing ours by only 0.14% ANLS. However, this marginal improvement comes at the expense of a larger model size, with a total number of parameters nearly 3 times that of the proposed AVIR. Furthermore, since both its retrieval and QA components rely on large models, its computational efficiency is substantially lower. In terms of page retrieval accuracy, our method achieves the same Top-1 accuracy as SelfAttnScoring. However, our retrieval mechanism is more tolerant to ranking noise, making it more robust: errors from the retriever have a smaller negative impact on final performance. We provide a more detailed analysis of efficiency and ablations in Section 4.4. 

**Table 2: Performance on the SlideVQA benchmark. “T/L/V” denotes the “text/layout/visual” modality of images.** 

## **4.3 Comparison with the State of the Art** 

**Table 1: Comparison with state of the art on the test set of MP-DocVQA dataset.** 

|Method|Year|OCR|Params.|Page Pred.(%) ANLS|Page Pred.(%) ANLS|
|---|---|---|---|---|---|
|BERT [10]|2018|✓|334M|71.24|0.5347|
|T5 [21]|2020|✓|223M|46.05|0.4028|
|Longformer [4]|2020|✓|148M|70.37|0.5506|
|Big Bird [29]|2020|✓|131M|72.27|0.5854|
|LayoutLMv3 [14]|2022|✓|125M|74.02|0.5513|
|Hi-VT5 [26]|2023|✓|316M|79.23|0.6201|
|ScreenAI [2]|2024|✓|5B|77.88|0.7711|
|GRAM [6]|2024|✓|859M|-|0.8032|
|Arctic-TILT [7]|2024|✓|800M|-|0.8122|
|RMT5 [11]|2024|✓|312M|**88.32**|0.6401|
|SelfAttnScoring [16]|2024|-|273M|81.55|0.6199|
|M3DOCRAG [9]|2024|-|8B|81.05|0.8444|
|Qwen2.5-3B-AWQ [3]|2025|-|3B|-|0.8405|
|**AVIR(Ours)**|-|-|3B|81.55|**0.8458**|



_4.3.1 Results on MP-DocVQA Dataset._ We systematically compare our method with both OCR-based and OCR-free approaches, as summarized in Table 1. It is evident that earlier methods relied heavily on OCR to extract textual content, while recent advances in large vision-language models (LVLMs) have enabled OCR-free methods to achieve comparable or even superior performance. Our method achieves an ANLS score of 0.8458, outperforming many strong OCR-based baselines and surpassing existing OCR-free methods in accuracy. Notably, our model maintains a relatively small parameter size. Although ARIV is larger than recent lightweight models like GRAM and Arctic-TILT, it is significantly smaller than large-scale models such as ScreenAI and M3DOCRAG. A key advantage of our approach lies in its retrieval-augmented design. The retrieval module contains only about 100 million parameters, and the question-answering model processes only a small subset of the most relevant retrieved pages. This drastically reduces the overall computational overhead, making our method both accurate and efficient. Our baseline QA model, Qwen2.5-3B-AWQ, is a fully endto-end LVLM with fewer parameters than our method. However, due to the need to encode the entire document, its actual computation cost is significantly higher. Among all baselines, M3DOCRAG 

|Model|Year|Modal|Params|EM|F1|
|---|---|---|---|---|---|
|T5 [21]|2020|T|0.2B|29.3|37.9|
|LayoutT5 [25]|2021|TLV|-|31.7|39.9|
|LayoutLMv2 [28]|2020|TLV|-|21.4|29.3|
|FID [15]|2020|T|-|30.4|38.9|
|M3D [24]|2023|TLV|-|33.5|41.7|
|BLIP-2 [18]|2023|TV|3.4B|28.3|38.8|
|InstructDr [23]|2024|TLV|3.4B|31.9|40.2|
|Arctic-TILT [7]|2024|V|0.8B|55.1|-|
|VDocRAG [22]|2025|V|8B|-|44.2|
|FRAG [13]|2025|V|7B|59.8|65.1|
|Qwen2.5-3B-AWQ [3]|2025|V|3B|56.6|65.8|
|Eagle-2.5 [8]|2025|V|8B|**63.2**|**72.3**|
|**AVIR(Ours)**|-|V|3B|60.3|68.9|



_4.3.2 Results on SlideVQA Dataset._ SlideVQA is a slide-based document dataset characterized by rich layout information and visually structured content. Questions in SlideVQA often require reasoning over content that is scattered across multiple slides, making cross-page understanding a critical capability. Early methods typically integrate textual, layout, and visual features to address this challenge. The evaluation results on the SlideVQA test set are presented in Table 2. Although earlier approaches such as LayoutT5 and InstructDr incorporate multiple modalities (text, layout, and vision), their performance has been surpassed by more recent pure vision-based models. Our method also relies solely on visual input and achieves strong results, with an Exact Match (EM) score of 60.3 and an F1 score of 68.9, outperforming our baseline Qwen2.5-3BAWQ by 3.7 and 3.1 points, respectively. These results are second only to Eagle-2.5. Eagle-2.5 achieves higher scores, but at the cost of a significant increase in model size—nearly 3 times that of AVIR. Importantly, our retrieval-based approach, AVIR, avoids processing the entire document directly with a large-scale VQA model, further speeding up inference and reducing computational cost. Overall, the results on SlideVQA further demonstrate the effectiveness and generalizability of our retrieval-augmented approach for long-document VQA, highlighting its ability to scale across document types with varying layout complexity and cross-page dependencies. 

_4.3.3 Results on DUDE Dataset._ The DUDE dataset presents a unique challenge that is not directly compatible with our method, 

MMAsia ’25, December 9–12, 2025, Kuala Lumpur, Malaysia 

Zongmin Li et al. 

**Table 3: Comparison of Models on the DUDE benchmark.** 

|**Method**<br>**ANLS**|**ANLSper Answer Type**|
|---|---|
||Extractive<br>Abstractive<br>List-of-answers<br>Unanswerable|
|Arctic-TILT [7]<br>**0.5809**<br>GPT-4&Azure OCR<br>0.5392<br>GRAM [6]<br>0.5336<br>GRAM C-Former [6]<br>0.5097<br>DocGptVQA<br>0.5002<br>**AVIR (Ours)**<br>0.4905<br>DocBlipVQA<br>0.4762<br>**Qwen2.5-3B-AWQ (Our baseline)**[3]<br>0.4575<br>T5-concat<br>0.3867<br>Multi-Modal T5 VQA<br>0.3790<br>HI-VT5 [26]<br>0.3574<br>QAP<br>0.1159|0.6271<br>0.5645<br>0.4669<br>**0.6261**<br>0.5973<br>0.5248<br>**0.5785**<br>0.5131<br>0.5683<br>0.5232<br>0.1996<br>0.6543<br>0.5515<br>0.5046<br>0.1726<br>0.6104<br>0.5186<br>0.4832<br>0.2822<br>0.6204<br>**0.6754**<br>**0.6404**<br>0.1029<br>0.0000<br>0.5069<br>0.4631<br>0.3073<br>0.5522<br>0.6319<br>0.5940<br>0.1199<br>0.0000<br>0.3727<br>0.3750<br>0.1681<br>0.5289<br><br><br><br>|
||0.4155<br>0.4024<br>0.2021<br>0.3467<br>0.2831<br>0.3298<br>0.1060<br>0.6290<br>0.0009<br>0.0007<br>0.0000<br>0.6199|



as its questions involve four distinct answer types—Extractive, Abstractive, List-of-answers, and Unanswerable—which require models to handle diverse output formats. Without targeted fine-tuning, models struggle to generalize across such heterogeneous formats. As shown in Table 3, our method, along with the baseline Qwen2.53B-AWQ, achieves near-zero performance on the List-of-answers and Unanswerable categories, while all other listed methods are fine-tuned specifically on DUDE. We show four typical mistakes of AVIR in Fig. 3: 1) The question-answering model cannot handle global questions because it does not access all pages. 2) The question answer is not comprehensive and cannot find all answers. 3) Although the answer is correct, it cannot be answered in the specified format. 4) Nonsense in questions without answers. 

Nevertheless, for the Extractive and Abstractive types—whose formats are commonly shared with other datasets—our method achieves state-of-the-art performance, outperforming the baseline by approximately 0.04 ANLS. This result highlights the effectiveness of our retrieval-augmented design even in DUDE. During evaluation, we employ a general prompt: "Answer the question using a single word or phrase." When this is modified to match DUDE’s expected format, e.g., "List all answers in one JSON array without anything else. Use ’none’ for unanswerable questions," the performance on List-of-answers and Unanswerable questions does not improve and instead causes a significant drop in performance on the Extractive and Abstractive categories. These findings suggest that DUDE is not well-suited for the Qwen-2.5-vl model without finetuning, due to its strict and diverse answer format requirements. Nonetheless, the strong results on the Extractive and Abstractive answers demonstrate the robustness and generalizability of the proposed AVIR. 

## **4.4 Ablation experiment** 

We conduct ablation experiments to investigate the impact of retrieval on MP-DocVQA performance. The results are reported in Table 4. We can observe that in the baseline, where we ask questions directly without using retrieval, the question answering model not only inefficiently processes all pages but also suffers from a large number of irrelevant pages, resulting in low accuracy. This is illustrated in Fig. 1. Following the SelfAttnScoring [16], using 

Question: Which pages show graphs? GT: ['5', '3', '6'] Prediction: 9 Question:  what is the taxable value? GT: ['$31,500', '$45,000', '$34,650'] Prediction: $31,500 Question: What are the Improving description of electron corelation? GT: ['double excitations', 'perturbative of triple excitations'] Prediction: Double excitations, Perturbative of triple excitations Question: What are the person's names on the second page? GT: [] Prediction: HEBERT, DUNIA ABDULAHAH 

**Figure 3: Four typical errors of AVIR on the DUDE dataset.** 

only the top-1 relevant pages improves the efficiency of the question answering model, but retrieval errors significantly degrade performance. As the number of Top-K pages increases, model performance gradually improves, reaching peak performance at Top-4, the strategy used by M3DOCRAG [9]. However, further increasing the Top-K results in a decrease in question answering performance. When Top-K reaches 20 (the maximum number of pages in the SlideVQA dataset), the model degenerates to non-retrieval question answering (our baseline). 

Compared to the baseline, the proposed AVIR method effectively reduces interference from irrelevant pages. In contrast to the Top-K approach, it also alleviates the negative impact of retrieval errors, leading to optimal performance. Moreover, our method requires only an average of 2.9 pages to answer each question, striking an ideal balance between effectiveness and efficiency. 

## **5 Conclusions** 

In this paper, we propose an Adaptive Visual In-Document Retrieval (AVIR) for efficient multi-page document question answering, which effectively alleviates the problems of low computational efficiency 

AVIR: Adaptive Visual In-Document Retrieval for Efficient Multi-Page Document Question Answering 

MMAsia ’25, December 9–12, 2025, Kuala Lumpur, Malaysia 

**Table 4: Ablation experiment on SlideVQA dataset. APS means the proposed Adaptive Page Selector.** 

|**Method**|Ave.page|EM|F1|
|---|---|---|---|
|Qwen2.5-VL-AWQ (baseline)|20.0|56.6|65.8|
|Retrival+Top-K (K=1)|1.0|52.7|60.2|
|Retrival+Top-K (K=2)|2.0|56.5|64.6|
|Retrival+Top-K (K=4)|4.0|58.3|66.8|
|Retrival+Top-K (K=8)|8.0|57.2|65.7|
|Retrival+APS(AVIR)|2.9|60.3|68.9|



and low accuracy of long documents in large visual-language models. The proposed adaptive page selector can effectively select highly relevant pages and alleviate the negative impact of retrieval errors. The experimental results on multiple datasets show that AVIR surpasses the baseline in both accuracy and efficiency and achieves the best performance among models of similar scale. 

**Limitation:** Our method is not able to handle non-universal answer formats, such as those found in DUDE, unless the question answering model is specifically fine-tuned to do so. Moreover, because our method leverages a retrieval mechanism that only exposes a subset of relevant pages to the question answering model, it lacks access to the full document context. As a result, it is inherently incapable of addressing global questions that require complete document-level understanding, even for seemingly simple queries like, “How many pages does the document have?”. 

## **Acknowledgments** 

This work is supported by the China Scholarship Council (CSC) (Grant no. 202406450075), the National key R&d program (Grant no. 2019YFF0301800), the National Natural Science Foundation of China (Grant no. 61379106), and the Shandong Provincial Natural Science Foundation (Grant nos.ZR2013FM036, ZR2015FM011). 

## **References** 

- [1] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R Manmatha. 2024. Docformerv2: Local features for document understanding. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol. 38. 709–718. 

- [2] Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor Cărbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. 2024. Screenai: A vision-language model for ui and infographics understanding. _arXiv preprint arXiv:2402.04615_ (2024). 

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_ (2025). 

- [4] Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The longdocument transformer. _arXiv preprint arXiv:2004.05150_ (2020). 

- [5] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. 2019. Scene text visual question answering. In _Proceedings of the IEEE/CVF international conference on computer vision_ . 4291–4301. 

- [6] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. 2024. GRAM: Global reasoning for multi-page VQA. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 15598–15607. 

- [7] Łukasz Borchmann, Michał Pietruszka, Wojciech Jaśkowski, Dawid Jurkiewicz, Piotr Halama, Paweł Józiak, Łukasz Garncarek, Paweł Liskowski, Karolina Szyndler, Andrzej Gretkowski, et al. 2024. Arctic-TILT. Business Document Understanding at Sub-Billion Scale. _arXiv preprint arXiv:2408.04632_ (2024). 

- [8] Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Tuomas Rintamaki, et al. 2025. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. _arXiv preprint arXiv:2504.15271_ (2025). 

- [9] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ (2024). 

- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)_ . 4171–4186. 

- [11] Qi Dong, Lei Kang, and Dimosthenis Karatzas. 2024. Multi-page document VQA with recurrent memory transformer. In _International Workshop on Document Analysis Systems_ . Springer, 57–70. 

- [12] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ (2024). 

- [13] De-An Huang, Subhashree Radhakrishnan, Zhiding Yu, and Jan Kautz. 2025. FRAG: Frame Selection Augmented Generation for Long Video and Long Document Understanding. _arXiv preprint arXiv:2504.17447_ (2025). 

- [14] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM international conference on multimedia_ . 4083–4091. 

- [15] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. _arXiv preprint arXiv:2007.01282_ (2020). 

- [16] Lei Kang, Rubèn Tito, Ernest Valveny, and Dimosthenis Karatzas. 2024. Multi-page document visual question answering using self-attention scoring mechanism. In _International Conference on Document Analysis and Recognition_ . Springer, 219– 232. 

- [17] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. 2023. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _International Conference on Machine Learning_ . PMLR, 18893–18912. 

- [18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ . PMLR, 19730–19742. 

- [19] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. 2024. Unifying multimodal retrieval via document screenshot embedding. _arXiv preprint arXiv:2406.11251_ (2024). 

- [20] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ . 2200–2209. 

- [21] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ 21, 140 (2020), 1–67. 

- [22] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. 2025. Vdocrag: Retrieval-augmented generation over visually-rich documents. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ . 24827–24837. 

- [23] Ryota Tanaka, Taichi Iki, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. 2024. Instructdoc: A dataset for zero-shot generalization of visual document understanding with instructions. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol. 38. 19071–19079. 

- [24] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 37. 13636–13645. 

- [25] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. 2021. Visualmrc: Machine reading comprehension on document images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 35. 13878–13888. 

- [26] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ 144 (2023), 109834. 

- [27] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. 2023. Document understanding dataset and evaluation 

   - (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 19528–19540. 

- [28] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. 2020. Layoutlmv2: Multimodal pre-training for visually-rich document understanding. _arXiv preprint arXiv:2012.14740_ (2020). 

- [29] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big bird: Transformers for longer sequences. _Advances in neural information processing systems_ 33 (2020), 17283–17297. 

