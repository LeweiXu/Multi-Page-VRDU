## _Article_ 

## **Accurate Multi-Page Document Retrieval by Effectively Fusing Context Information Across Pages** 

**Bing Qian[1] , Kaiwei Deng[2] , Yuexin Wu[2,] *, Jianming Zhang[2] , Juanjuan Sun[2] , Yanru Xue[1] and Chunxiao Fan[2]** 

- 1 China Telecom Corporation Limited, Beijing Research Institute, Beijing 102209, China 

- 2 School of Electronic Engineering, Beijing University of Posts and Telecommunications, Beijing 100876, China ***** Correspondence: wuyuexin@bupt.edu.cn 

## **Abstract** 

Visual retrievers such as visual retrieval-augmented generation (RAG) have recently emerged as a powerful model for retrieving multimodal documents without the need to convert page images into text. Existing visual retrievers typically encode every document page separately, ignoring the inherent rich context information across pages within multi-page documents. However, some crucial semantic information often spans multiple pages in a document, and should be effectively encoded for better retrieval. To address this problem, this paper proposes a novel approach utilizing dynamically fusing visual context (DFVC), which adaptively encodes the semantic information across pages. In the proposed DFVC approach, a lightweight plug-and-play adapter is designed; in addition, a contrastive loss function incorporating the positive fused embedding vectors and negative embedding vectors is designed to constrain the adapter, allowing it to learn the weights for the context pages. Together, the designed adapter and loss function allow the retriever to effectively encode useful semantic information across pages while excluding distracting noise. The proposed DFVC is validated on commonly used challenging multi-page document benchmarks. Extensive experimental results demonstrate that it significantly boosts retrieval performance. In addition, the proposed DFVC is highly parameter-efficient since it employs frozen vision-language backbones, allowing it to be easily integrated into existing visual RAG pipelines for finer document retrieval. 

**Keywords:** visual retrieval-augmented generation; multi-page document retrieval; context-aware retrieval; vision-language models; parameter-efficient fine-tuning 

## **1. Introduction** 

Academic Editors: Mehmet Bakir, Aytu˘g Onan and Huiping Cao 

Received: 1 February 2026 Revised: 1 March 2026 Accepted: 10 March 2026 Published: 25 March 2026 

**Copyright:** © 2026 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license. 

In recent years, retrieval-augmented generation (RAG) has become a cornerstone technique for enhancing large language models (LLMs) with external knowledge while mitigating hallucinations and enabling domain-specific applications [1,2]. However, in many real applications knowledge is predominantly stored in semantically rich multimodal formats such as PDFs, textbooks, or financial reports. In these formats, text is intricately interleaved with layouts, tables, figures, etc. Traditional RAG pipelines, which we refer to as TextRAG, tend to struggle with these formats, since they rely on complex ingestion pipelines involving optical character recognition (OCR) and layout analysis to extract plain text [3]. The ingestion pipeline is not only computationally expensive but also lossy, inevitably discarding critical visual cues and structural information required for accurate understanding. 

_Electronics_ **2026** , _15_ , 1353 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

2 of 15 

To address this limitation, the recently proposed VisRAG [4] and ColPali [5] treat document pages directly as images instead of parsing text. By leveraging powerful visionlanguage models (VLMs) as encoders, they map page images into vectors in a semantic embedding space, enabling end-to-end retrieval based on visual features. This “retrieval in vision space” concept eliminates the information bottleneck of OCR and has demonstrated superior performance on multimodal benchmarks. 

Despite the advantages of the approach used by VisRAG and ColPali, they encode each document page separately, meaning that they they treat pages as independent and identically distributed (i.i.d.) instances and encode each page for the purpose of generating its embedding vector by simply using the page itself, as shown in Figure 1a. This discards the rich context information available across multi-page documents, where some semantic information often spans page boundaries. For example, when a table spans two pages or a paragraph is split by a page break, the model may fail to capture the entire semantic information across pages, leading to suboptimal retrieval ranking. In text-based RAG, to preserve semantic continuity across chunk boundaries [6] it is common to use overlapping windows or surrounding context (e.g., sentence-window retrieval) for information aggregating in order to generate embedding vectors. 

However, for visual document retrievers, naively aggregating the information of neighboring pages often leads to context dilution [7,8], since the irrelevant adjacent content contaminates the precise semantics of the target page. Unlike continuous text streams, document pages are discrete units of high semantic variance, and a relevant content page might be immediately followed by a completely unrelated advertisement, new chapter title, etc. [9]. This means that direct aggregation of the embeddings from adjacent but unrelated pages may generate irrelevant noise that dampens the semantic information of the target page. Since the generated embeddings may further degrade retrieval performance due to this irrelevant noise, it is necessary to design methods for reducing noise while effectively encoding information across pages. 

To effectively encode semantic information across pages while reducing context dilution, this paper proposes a novel approach called dynamically fusing visual context (DFVC), shown in Figure 1b, that encodes the context information of a page by adaptively utilizing its neighboring pages. In DFVC, we design a lightweight and parameter-efficient adapter that operates on top of the frozen backbone of vision-language models (VLMs) such as VisRAG. A contrastive loss function is designed that takes in both the positive fused embedding vector and negative embedding vectors, helping the designed adapter to learn the appropriate weights for neighboring pages. The loss function enables selectively fusing the context only when strong semantic continuity is determined to exist across pages, allowing the proposed DFVC to filter out distracting noise. 

Because only the designed lightweight adapter needs to be trained, the proposed DFVC does not need to perform expensive fine-tuning of the underlying vision encoders. This makes the proposed DFVC highly efficient, meaning that it can be used as a plug-andplay module in various existing pipelines. 

Our contributions can be summarized as follows: 

- We propose an approach called dynamically fusing visual context (DFVC), composed of a lightweight adapter and a contrastive loss function. The proposed DFVC encodes semantic context information across pages while excluding distracting noise. 

- We design a loss function that constrains the designed adapter to learn interpage dependencies and proper weights, reducing context dilution in multi-page document retrieval. 

- Experiments on multi-page document benchmarks demonstrate that the proposed DFVC significantly improves retrieval performance, achieving superior accuracy. 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

3 of 15 

( **a** ) Retrieval pipeline when encoding pages independently 

( **b** ) Retrieval pipeline when encoding context information across pages 

**Figure 1.** Comparison between a visual retrieval pipeline that encodes pages independently and a pipeline that encodes context information across pages. ( **a** ) In the standard retrieval pipeline, the target page (page _n_ ) is encoded independently, leading to low cosine similarity and retrieval failure. ( **b** ) In the proposed framework, the context information from the previous page (page _n −_ 1) and the next page (page _n_ + 1) is fused with that of page _n_ . This context-aware encoding bridges the semantic gap, resulting in a high similarity and successful retrieval. 

## **2. Related Work** 

Traditional retrieval-augmented generation (RAG) systems predominantly rely on a “parse-then-embed” pipeline [1,2]. These systems utilize optical character recognition (OCR) engines (e.g., Tesseract [10]) or PDF parsing tools to extract textual content from documents. The extracted text is then chunked and encoded by dense retrievers such as DPR [11], ANCE [12], or Contriever [13]. While effective for plain-text corpora, this paradigm struggles with the visually rich documents (VRDs) commonly found in various fields such as finance and industry. The parsing process of these VRDs often ignores crucial structural information (layouts, fonts) and non-textual elements (charts, figures), leading to significant semantic loss [3,4]. 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

4 of 15 

To mitigate the loss of structural information, LayoutLM and its successors [14–16] integrate text, layout (bounding boxes), and image features in a unified multimodal pretraining framework. These models have shown improvements in the task of document understanding, including form understanding and receipt understanding; however, they typically require fine-grained OCR annotations during training and inference, which remains both prone to error propagation and computationally expensive. 

Recent advancements have shifted towards directly encoding page images while bypassing OCR. VisRAG [4] leverages the capabilities of large vision-language models (LVLMs), specifically MiniCPM-V [17], to map document images into embedding vectors in a dense embedding space. ColPali [5] extends the ColBERT [18] late-interaction architecture to the visual domain, representing pages as bags of visual patch embeddings derived from PaliGemma [19]. More recent studies have further optimized this paradigm, for example Light-ColPali [20] for efficiency and OmniParser [21,22] for robust screen parsing. While these pixel-to-embedding approaches achieve state-of-the-art performance on multimodal benchmarks such as ViDoRe [5], they typically assume that pages are independent and do not use other pages for encoding the embedding vector for each page. Thus, the independent encoding does not fully encode the semantic information lying in the neighboring pages around the current page and their performance degrades when dealing with the multi-page retrieval problem. Consequently, it is desirable to effectively encode the context information across neighboring pages of a page in order to improve the visual retrieval performance. 

In the field of text retrieval, context information has already been used for resolving ambiguity. One approach is to use sliding overlapped windows to preserve information at chunk boundaries [23]. Other approaches decouple the retrieval unit from the generation unit; for instance, _sentence-window retrieval_ indexes single sentences before returning a broader context window to the LLM, while _parent document retrieval_ [6] retrieves small chunks to fetch their parent documents. 

In addition to the above approaches using sliding windows, learning-based approaches have been proposed in recent years. Previous works have explored learning of context-aware embeddings in which passage representations are conditioned on their global document context [24]. Contextualized query expansion methods [25] augment queries with predicted document content. In the field of multimodal retrieval, graph-based RAG methods such as MoLoRAG [26] and mKG-RAG [27] have emerged to explicitly model cross-page relationships, though at the cost of high indexing complexity. 

It is possible to transfer these context-encoding strategies from the field of text retrieval to that of visual document retrieval; however, unlike text tokens, document pages are discretely distributed high-dimensional visual units. Thus, naively concatenating the embedding vectors of neighboring pages, which amounts to increasing the window size, leads to the context dilution problem. This is because irrelevant visual noise from adjacent pages (e.g., advertisements) is also encoded and affects the target representation. Recent efforts including VRAG-RL [28] have attempted to use reinforcement learning to filter out the visual noise, but involve complex reward engineering and often suffer from training instability due to the optimization difficulties of discrete decision-making processes. 

Observing the drawback of the above methods, this work proposed a novel approach that consists of dynamically fusing visual context (DFVC) information across neighboring pages of a page to generate a semantically rich embedding vector. In the proposed DFVC approach, a lightweight adapter is first designed that aims to learn the appropriate weights for the neighboring pages; then, a contrastive loss function is designed that incorporates the positive fused embedding vector and negative embedding vectors. The designed loss 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

5 of 15 

function constrains the adapter to generate weights that represent meaningful context while excluding distracting noise. 

The proposed adapter follows the same philosophy of parameter-efficient fine-tuning (PEFT) as Adapters [29], LoRA [30], and Prompt Tuning [31]. Unlike methods such as SigLIP [32] that insert layers inside the vision encoders, the proposed adapter works in a post hoc manner, that is, it operates on the final output embeddings. Thus, re-indexing the entire corpus (as required by existing methods such as [29–31]) is not required. This approach makes the designed adapter more efficient, allowing it to serve as a plug-and-play module; consequently the proposed DFVC can be applied to existing vector databases without re-computing the base embeddings, offering significant advantages over other internal PEFT methods. 

## **3. Proposed Method** 

This section presents the details of the proposed DFVC approach. We first begin with an overview of the proposed approach, then present the designed adapter and loss function in DFVC. Figure 2 illustrates the overall architecture of the proposed dynamically fusing visual context (DFVC) approach. DFVC mainly includes a lightweight adapter that predicts the gating weights in order to effectively fuse context information across pages, along with a contrastive loss function that constrains the adapter to distinguish meaningful context information from distracting noise. 

**Figure 2.** The proposed approach is composed of the designed lightweight adapter and contrastive loss function. In the encoding phase, the previous, current, and next page image are first encoded by a frozen vision-language model (VLM), after which the generated embedding vectors are concatenated and fed to the designed lightweight adapter to predict gating weights ( _α_ , _β_ ). With ( _α_ , _β_ ), the context information encoded in the neighboring pages is fused with the current embedding, ending with the fused embedding vector. The contrastive loss function takes in the embedding vector of the query text, the fused embedding vector as the positive sample, and the negative embedding vectors. Since the backbone is frozen, only the lightweight adapter (indicated by the flame icon) needs to be updated via backpropagation, making the proposed DFVC a computationally efficient plug-and-play module. 

## _3.1. Fusing Context Information Across Pages_ 

This section discusses fusing the context information when encoding page images. We first present the detail of predicting the fusing coefficients, then discuss the process of fusing neighboring pages with the predicted coefficients. 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

6 of 15 

## 3.1.1. The Adapter Predicting the Fusing Coefficients 

Let _D_ = _{d_ 1, _d_ 2, . . . , _dN}_ denote the corpus for a multi-page document, where _di_ represents the _i_ -th page image. Given a user query _q_ , the goal of visual document retrieval is to rank the pages in _D_ based on their similarity with _q_ , then give the top-K retrieved pages according to the similarity between _q_ and every page of each such corpus. VisRAG [4] employs a dual-encoder architecture: a vision encoder _EV_ mapping each page image _di_ , _i_ = 1, . . . , _N_ to a _D_ -dimensional embedding vector **v** _i ∈_ R _[D]_ , and a text encoder _ET_ mapping the query _q_ to a query embedding vector **q** _∈_ R _[D]_ . The score is cosine similarity, 

**==> picture [269 x 26] intentionally omitted <==**

where we used the multimodal large language model (MLLM) Qwen2-VL-2B-Instruct as the text encoder _ET_ . 

When generating the visual embedding vector **v** _i_ , _i_ = 1, . . . , _N_ used in (1), existing methods encode each page **v** _i_ = _EV_ ( _di_ ) independently, i.e., **v** _i_ is generated by only using _di_ without other pages _dj_ , _j_ = _i_ . This independent encoding mechanism ignores the context information in the neighboring pages. However, in visual retrieval some important semantic information may lie across multiple pages, which should be encoded to improve retrieval performance. To capture the inter-page dependency while reducing context dilution, this work designs a lightweight adapter to refine the embedding **v** _i_ by incorporating the information of **v** _i−_ 1 and **v** _i_ +1. 

Let **v** _i−_ 1, **v** _i_ , and **v** _i_ +1 denote the page-level semantic representations of _di−_ 1, _di_ , and _di_ +1, respectively. For dense retrievers such as VisRAG, we directly use **v** _i_ , _i_ = 1, . . . , _N_ . For multi-vector retrievers such as ColQwen, which outputs a bag of patch embeddings _Pi ∈_ R _[M][×][D]_ , we apply global average pooling to obtain a unified vector 

**==> picture [72 x 28] intentionally omitted <==**

Then, the concatenation **x** _i_ = [ **v** _i−_ 1; **v** _i_ ; **v** _i_ +1] _∈_ R[3] _[D]_ is fed to the adapter to predict the fusing coefficients. 

To determine the fusing weights of neighboring pages that reflect their contributions, the designed adapter employs a multi-layer perceptron (MLP) consisting of two linear layers with ReLU activation and LayerNorm in between. The MLP outputs the importance weights for the previous and next pages as 

**==> picture [322 x 13] intentionally omitted <==**

where _σ_ ( _·_ ) is the sigmoid function ensuring that the weights fall within the range (0, 1). The adapter is trained using the designed loss function (see Section 3.2). 

Note that because the adapter is a lightweight network, it can be efficiently trained while keeping the backbone frozen on the existing embedding vectors for generating proper weights, after which the fused embedding vector is computed (see Section 3.1.2). Thus, there is no need to re-compute the base embedding vectors for the entire corpus, making the proposed DFVC a plug-and-play module. 

## 3.1.2. Residual Context Fusion 

Having obtained the predicted weights by (2), we then fuse the context information into the embedding vector **v** _i_ . During the process of fusing context information, we want to preserve the main contribution of **v** _i_ in the fused embedding vector while incorporating 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

7 of 15 

the context information from the neighboring embedding vectors. To this end, we employ a residual-connection structure to compute the fused embedding vector **v** _i[′]_[:] 

**==> picture [258 x 13] intentionally omitted <==**

where the context information encoded in **v** _i−_ 1 and **v** _i_ is fused into the embedding vector **v** _i_ . Afterwards, **v** _i[′]_[carries richer semantic information across pages, leading to improved] retrieval performance. 

## _3.2. Contrastive List-Wise Ranking Loss Function_ 

This section discusses the loss function, which plays a key role in constraining the designed adapter (see Section 3.1.1) to generate the fusing coefficients such that meaningful information across pages can be incorporated and distracting information excluded. 

Since a single query may correspond to multiple relevant pages, we formulate the loss function as a list-wise ranking problem. Specifically, for each query _q_ , we sample a list of _K_ candidate retrieval pages _{dj}[K] j_ =1[, including the set of positive pages] _[ P]_[+][ and negative pages.] The positive pages are the correct ones with respect to query _q_ in the dataset. The negative samples are sampled randomly, and can be either from the same document containing the ground-truth page or from different documents. The ground-truth probability distribution _P_ ( _dj|q_ ) is defined to be uniform over positive pages 

**==> picture [248 x 26] intentionally omitted <==**

For each retrieval page, the fused embedding vector is generated with (3) and the predicted probability distribution is computed with the softmax function: 

**==> picture [268 x 31] intentionally omitted <==**

where _τ_ is the temperature hyperparameter. In this work, _K_ is set to 8 and _τ_ is set to 0.1. In (5), if the query _q_ corresponds to **v** _[′] j_[,] _[s]_[(] _[q]_[,] **[ v]** _[′] j_[)][will][be][large;][hence,] _[P]_[ˆ][(] _[d][j][|][q]_[)][will][be] close to _P_ ( _dj|q_ ). Based on this observation, the Kullback–Leibler (KL) divergence is used to measure the distance between the ground-truth probability distribution _P_ ( _dj|q_ ) and the predicted distribution _P_[ˆ] ( _dj|q_ ): 

**==> picture [298 x 30] intentionally omitted <==**

To reduce context dilution, in which the embedding vector is “diluted” by the neighboring embedding vectors in (3), we design a drift regularization term _Ldri f t_ for the designed loss function _Lrank_ in (6). Here, _Ldri f t_ encourages the fused embedding vector **v** _i[′]_[to remain] semantically close to the original embedding **v** _i_ . Formally, _Ldri f t_ is defined to be 

**==> picture [262 x 28] intentionally omitted <==**

where _B_ is the batch size. 

The total loss is the weighted sum of the drift regularization term _Ldri f t_ and the loss function _Lrank_ : 

**==> picture [255 x 13] intentionally omitted <==**

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

8 of 15 

where _λdri f t_ helps to balance the ranking quality in (6) and the context dilution in (7). In (8), the item _Lrank_ constrains the adapter to incorporate proper context information for generating the fused embedding vector such that a positive sample will have higher similarity with the query and a negative sample will have lower similarity. The item _Ldri f t_ prevents the fused embedding vector from being “diluted” too much from the original vector by the neighboring vectors. 

## _3.3. Training Process_ 

To ensure training stability, we initialize the bias of the final MLP layer to a negative value (e.g., _−_ 5.0) and set _α_ and _β_ to 0. This guarantees that the generated weights are close to 0 at the start of training, making the fusing process in (3) behave like an identity mapping **v** _i[′][≈]_ **[v]** _[i]_[.] 

This means that the adapter can be steadily trained from the identity mapping, which prevents its performance from degrading in the early stages. 

Because the adapter is a lightweight MLP, it can be easily trained without expensive computational requirements. The training process is only conducted on the adapter, with the backbone being frozen; therefore, the proposed DFVC comprising the adapter and loss function can act as a plug-and-play module, allowing for easy application as well as integration into existing large-scale models. 

## **4. Experiments** 

In this section, we evaluate the effectiveness of the proposed DFVC. Two challenging multi-page document retrieval benchmarks are adopted to investigate the performance of DFVC and the compared methods. The performance of the proposed DFVC is first compared with state-of-the-art baselines, after which we conduct in-depth ablation studies to validate the contribution of its two components. 

The proposed method was implemented with PyTorch (version 2.1.2). Two frozen visionlanguage backbones were employed: VisRAG [4] (MiniCPM-V based) and ColQwen [5] (Qwen2-VL based). For the proposed dynamically fusing visual context (DFVC), we employed a multi-layer perceptron (MLP) consisting of two linear layers with a hidden dimension of 512. The MP-DocVQA and MMLongBench-Doc datasets were split into two parts, with 80% used for training and 20% for validation. We trained the proposed DFVC for twenty epochs on MP-DocVQA and five epochs on MMLongBench-Doc with the backbones kept frozen. We used the AdamW optimizer (lr = 5 _×_ 10 _[−]_[4] ) with a batch size of 64 and set the drift regularization weight to _λdrift_ = 0.2. 

The commonly used normalized discounted cumulative gain (nDCG) metric is used to evaluate the performance of the different methods. The discounted cumulative gain (DCG) computes the summation of the scores of the retrieval results while applying a logarithmic discount to every position, then is normalized by the ideal discounted cumulative gain (IDCG) to obtain the nDCG, which measures the affinity of the ranking of the retrieved document pages to that of the ground truth. In particular, nDCG@K measures the normalized DCG for the top-ranked K pages. 

## _4.1. Evaluation Dataset Benchmarks_ 

We conducted the experiments on two challenging multi-page document retrieval benchmarks: 

- MMLongBench-Doc [9], a benchmark comprising 135 documents and 1091 questions across seven domains; and each document has an average of 47.5 pages. We report 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

9 of 15 

   - nDCG@5 on its retrieval subsets categorized by evidence modality: Text (TXT), Layout (LAY), Chart (CHA), Table (TAB), and Image (IMG). 

- MP-DocVQA [33], a dataset focused on the task of processing multiple pages from scanned document collections for visual question answering. Because of the high visual similarity between pages of the same document, this dataset is appropriate for testing the ability of models to distinguish semantic relevance from visual noise. 

## _4.2. Experimental Results_ 

We evaluate the effectiveness of the proposed DFVC by comparing it against three categories of methods: 

- OCR-based text retrieval: BGE-M3 [34] with OCR is used as the traditional baseline method, representing the “parse-then-embed” pipeline that relies on text extraction. 

- Independent visual retrieval: We include three state-of-the-art visual retrievers: VisRAG [4], ColPali [5], and ColQwen. ColPali first introduced the effective lateinteraction mechanism for visual documents based on PaliGemma, while ColQwen significantly enhances the ColPali by leveraging the superior multimodal understanding ability of the Qwen2-VL backbone. VisRAG, ColPali, and ColQwen are all evaluated in their original forms, with each page is encoded independently. 

- Trainable Contextualized Baselines: To demonstrate the superiority of our dynamic gating architecture over standard sequence modeling, we implement two variants on top of the frozen VLM features: (1) Concat-Linear, which concatenates [ **v** _i−_ 1; **v** _i_ ; **v** _i_ +1] followed by a linear projection, and (2) Bi-LSTM, which uses a bidirectional LSTM to aggregate information across the three-page window. 

In order to rigorously validate the robustness of our improvements and account for training variance, all experiments involving trainable modules (Concat-Linear, Bi-LSTM, and DFVC) were repeated five times using different random seeds. We report the mean and standard deviation ( _±_ std) for the overall performance. In addition, we conducted paired t-tests between the independent visual retrieval baselines and the proposed DFVC. In all major comparisons, DFVC achieved a statistically significant improvement with _p_ -value _<_ 0.05. 

## 4.2.1. Performance Comparison on MMLongBench-Doc 

Table 1 provides the retrieval performance of different methods, with nDCG@5 used to evaluate performance. Table 1 shows that our proposed DFVC significantly outperforms BGE and ColPali. For example, the “Overall” performance of DFVC using VisRAG achieves 0.5250, while DFVC using ColQwen achieves 0.6050; in comparison, BGE achieves 0.2716 and ColPali 0.3102. 

Although the sequence models such as Bi-LSTM provide an improvement over retrieval methods that encode pages independently, their performance is still inferior to that of the proposed DFVC due to the discrete heterogeneity of document pages. Unlike continuous text, discrete adjacent pages may contain irrelevant content such as ads or new chapter headings, leading to context dilution in simple aggregation models. Bi-LSTM achieves 0.5480 on the subset “TAB” indicating tables; this result is higher than vanilla VisRAG, which achieves 0.5309. In comparison, the proposed DFVC reaches 0.5620, which it accomplishes by effectively fusing the context only when semantic continuity is detected while filtering out distracting noise. For the “LAY” (Layout) subset, DFVC is also able to bridge the structural gap caused by page breaks, achieving a retrieval performance of 0.4510. Ths is higher than the vanilla VisRAG, Concat-Linear, and Bi-LSTM, which reach 0.4055, 0.4180, and 0.4320, respectively. The proposed DFVC also yields higher retrieval 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

10 of 15 

performance than the vanilla VisRAG, Concat-Linear, and Bi-LSTM on the “TXT”, “CHA”, and “IMG” subsets. 

**Table 1.** Retrieval performance (nDCG@5) of different methods on the benchmark MMLongBenchDoc. The best results are shown in **bold** . For trainable methods, we report the mean and standard deviation over five runs in the Overall column. 

|**Backbone**<br>**Method**|**Evidence Modalities**<br>**Overall**<br>**TXT**<br>**LAY**<br>**CHA**<br>**TAB**<br>**IMG**|
|---|---|
|BGE<br>OCR|0.5910<br>0.2150<br>0.1450<br>0.2950<br>0.1120<br>0.2716|
|ColPali<br>Original|0.3250<br>0.2840<br>0.3520<br>0.3150<br>0.2750<br>0.3102|
|VisRAG<br>Original<br>Concat-Linear<br>Bi-LSTM<br>**DFVC (Ours)**|0.4950<br>0.4055<br>0.6005<br>0.5309<br>0.4252<br>0.4959<br>0.5012<br>0.4180<br>0.6150<br>0.5420<br>0.4310<br>0.5014_±_0.0035<br>0.5045<br>0.4320<br>0.6210<br>0.5480<br>0.4380<br>0.5105_±_0.0042<br>**0.5085**<br>**0.4510**<br>**0.6350**<br>**0.5620**<br>**0.4450**<br>**0.5250****_±_ 0.0028**|
|ColQwen<br>Original<br>Concat-Linear<br>Bi-LSTM<br>**DFVC (Ours)**|0.5850<br>0.5136<br>0.6510<br>0.5716<br>0.5328<br>0.5750<br>0.5910<br>0.5220<br>0.6580<br>0.5850<br>0.5410<br>0.5794_±_0.0031<br>0.5980<br>0.5310<br>0.6600<br>0.5920<br>0.5480<br>0.5858_±_0.0039<br>**0.6100**<br>**0.5450**<br>**0.6650**<br>**0.6010**<br>**0.5550**<br>**0.6050****_±_ 0.0024**|



The proposed DFVC also shows improved retrieval performance when ColQwen is used as the frozen backbone. For example, on the “TAB” subset vanilla ColQwen achieves 0.5716, Concat-Linear 0.5850, and Bi-LSTM 0.5920, while the proposed DFVC achieves 0.6010. On the “TXT”, “LAY”, “CHA”, and “IMG” subsets, Concat-Linear, Bi-LSTM, and the proposed DFVC all improve over vanilla ColQwen, verifying the contribution of fusing context information across pages. 

## 4.2.2. Performance Comparison on MP-DocVQA 

Table 2 provides the retrieval results on the multi-page dataset MP-DocVQA. On MP-DocVQA, BGE and ColPali achieve 0.1095 and 0.1550, respectively, while the original VisRAG reaches 0.1873. This verifies the value of visual encoding in the field of retrieval. Concat-Linear and Bi-LSTM reach 0.1895 and 0.1942, showing improved the retrieval performance over VisRAG when fusing context information across pages. The proposed DFVC further improves over Concat-Linear and Bi-LSTM owing to effective context fusion while filtering distracting noise, which it accomplishes by learning the appropriate contribution weights of neighboring pages. 

**Table 2.** Retrieval performance (nDCG@5) on MP-DocVQA. Best results are shown in **bold** . Trainable methods report mean _±_ std over five runs. 

|**Backbone**|**Method**|**nDCG@5**|
|---|---|---|
|-|BGE (OCR)|0.1095|
|ColPali|Original|0.1550|
||Original|0.1873|
|VisRAG|Concat-Linear<br>Bi-LSTM|0.1895_±_0.0038<br>0.1942_±_0.0045|
||**DFVC (Ours)**|**0.2050****_±_ 0.0021**|
||Original|0.3301|
|ColQwen|Concat-Linear<br>Bi-LSTM|0.3350_±_0.0041<br>0.3420_±_0.0052|
||**DFVC (Ours)**|**0.3510****_±_ 0.0026**|



https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

11 of 15 

For ColQwen, the proposed DFVC also improved retrieval performance over the original ColQwen by 2.09%, verifying its robustness even when the pages of documents are visually dense and similar. The performance improvement of the proposed DFVC on both the VisRAG and ColQwen shows that it can effectively fuse context information across discrete adjacent pages. 

## _4.3. Ablation Studies_ 

To validate the contributions of each component in the proposed DFVC, we performed ablation studies using the VisRAG backbone on both MMLongBench-Doc and MP-DocVQA. The results of the ablation studies are provided in Table 3. 

**Table 3.** Results of ablation studies using VisRAG as the backbone on the MMLongBench-Doc and MP-DocVQA datasets. We report nDCG@5 on both datasets.The proposed method and the best results are shown in **bold** . 

|**Methods**|**nDCG@5**|
|---|---|
||**MMLongBench**<br>**MP-DocVQA**|
|**Full DFVC (Ours)**|**0.5250**<br>**0.2050**|
|static weights w/o dynamic gating<br>w/o drift regularization<br>w/o residual connection<br>original VisRAG|0.4998<br>0.1888<br>0.5120<br>0.1980<br>0.4980<br>0.1850<br>0.4959<br>0.1873|



Contribution of gating mechanism. We first investigated the contribution of the dynamic gating mechanism of the designed adapter. For this purpose, we used a static context with fixed weights of 0.1, 0.8, 0.1 as the baseline. In comparison with vanilla VisRAG and ColQwen, the static context provides negligible improvement on both the MMLongBenchDoc and MP-DocVQA datasets. Vanilla VisRAG reaches 0.4959 on MMLongBench-Doc and 0.1873 on MP-DocVQA. By comparison, it reaches only 0.4998 and 0.1888 when using static weights without dynamic gating. The full DFVC reaches 0.5250 and 0.2050, verifying the value of dynamic weights in controlling the contributions of adjacent pages. 

Contribution of drift regularization. Removing the drift regularization term ( _Ldri f t_ ) from the loss function leads to a performance drop on both datasets. DFVC without drift regularization reaches 0.5120 and 0.1980, while the full DFVC reaches 0.5250 and 0.2050. This indicates that the designed adapter may be prone to being overfitted on the training samples, causing the fused embedding vector to drift too far from the original embedding vector. This overfitting is detrimental for the retrieval performance, especially on the visually dense MP-DocVQA dataset. 

Contribution of residual connection. To investigate the contribution of the residual connection to the ability of the fused embedding vector encoding semantic information, we replaced the residual connection with a feed-forward output that directly predicts the coefficients for _vi−_ 1, _vi_ , and _vi_ +1. Without the residual connection, DFVC achieves 0.4980 and 0.1850, which is comparable to the original VisRAG. This shows that the model can learn a small correction term (∆ **v** ) more easily than reconstructing the entire semantic vector from scratch, validating the contribution of the residual connection. 

Impact of context window size. To investigate the choice of the context window size, we evaluate the performance of DFVC using varying window sizes: _k_ = 1 (no context, equivalent to the original VisRAG); _k_ = 3 (immediate previous and next pages, the default); _k_ = 5 ( _±_ 2 neighboring pages); and _k_ = 7 ( _±_ 3 neighboring pages). As shown in Table 4, increasing the context window from _k_ = 1 to _k_ = 3 yields the most significant performance gain by successfully capturing immediate cross-page semantics (e.g., a sentence or table 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

12 of 15 

spanning two adjacent pages). However, when the window size is further expanded to _k_ = 5 or _k_ = 7, the retrieval performance begins to saturate and even slightly decrease. This decrease is caused by the discrete and heterogeneous nature of visual documents. Unlike continuous text streams, distant pages in visually rich documents often contain completely irrelevant visual information; hence, expanding the window size may introduce excessive visual noise. 

**Table 4.** Ablation study on the context window size ( _k_ ) using the VisRAG backbone; _k_ = 1 represents the original page-independent encoding. The proposed default window size ( _k_ = 3) and the best results are shown in **bold** . 

|**Window Size**|**nDCG@5**|
|---|---|
||**MMLongBench**<br>**MP-DocVQA**|
|_k_=1 (Original VisRAG)<br>_k_=3**(DFVC, Default)**<br>_k_=5 (_±_2 pages)<br>_k_=7 (_±_3 pages)|0.4959<br>0.1873<br>**0.5250**<br>**0.2050**<br>0.5180<br>0.2010<br>0.5090<br>0.1950|



## _4.4. Efficiency Analysis_ 

We next present a quantitative comparison between the original VisRAG pipeline and our VisRAG + DFVC framework. Unlike methods that fine-tune massive vision-language models (VLMs) or require complex contextualized pretraining, the proposed DFVC is a lightweight approach that can be applied to precomputed 1D page embeddings entirely post hoc. 

As shown in Table 5, the main computational cost is needed by the frozen VLM encoder, which requires approximately 3000 MB of GPU memory. The proposed adapter introduces only 4.2 MB of additional trainable parameters, an increase of less than 0.15%. _∼_ DFVC requires only 1.3 ms ( 2% relative overhead) to the query latency, which has little impact on the indexing throughput. 

**Table 5.** Efficiency comparison between the original VisRAG and our proposed DFVC, evaluated on a single NVIDIA A40 GPU (NVIDIA, Santa Clara, CA, USA). The proposed method is shown in **bold** . Arrows ( _↓_ and _↑_ ) indicate whether lower or higher values are better, respectively. 

|**Pipeline**|**Query Latency****_↓_**|**Indexing Throughput****_↑_**|**GPU Memory Usage****_↓_**|
|---|---|---|---|
|Original VisRAG|65.2 ms|15.3 pages/s|_∼_3000 MB (Baseline)|
|**DFVC**|**66.5 ms**|**15.0 pages/s**|_∼_**3004.2 MB**|
|Relative Overhead|+1.99%|_−_1.96%|+0.14%|



## _4.5. Qualitative Analysis_ 

To understand the working rationale of the proposed DFVC, we use a query example to compare the original VisRAG baseline with DFVC. Figure 3 illustrates a challenging scenario in which the user query requires complex multi-hop reasoning: “In Q3 2015, what is the approximate range of cost in one day for installing a mobile incentive platform in Vietnam?” Answering this query requires retrieving two related pieces of evidence: the unit cost per install ($0.3–0.4 incentive) in Page _n_ (Slide 26), and the daily install volume (5–20 k installs a day) in its subsequent page, Page _n_ + 1 (Slide 27). 

The page-independent baseline (VisRAG) encodes each page separately. Page _n_ contains the price, but lacks the daily volume context of “one day”, while Page _n_ + 1 contains the daily volume but lacks the price context. Because neither page independently satisfies the full semantic condition to correctly answer the query, the VisRAG baseline generates a low similarity score to either page, leading to retrieval failure. 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

13 of 15 

In contrast, the proposed DFVC dynamically fuses the visual semantic context from Page _n_ + 1 and Page _n −_ 1 in the embedding of Page _n_ . Thus, the fused vector of Page _n_ encapsulates both the “unit cost” and “daily volume” information and successfully aligns it with the user query, significantly boosting the retrieval score for a successful retrieval. 

**Figure 3.** A success case demonstrating multi-hop cross-page semantic fusion. The query asks for the “cost in one day”, which requires fusing the unit pricing from Page _n_ (Slide 26) with the daily volume from Page _n_ + 1 (Slide 27), highlighted by red bounding boxes. The independent baseline (VisRAG, marked with _×_ ) fails due to fragmented semantics, whereas our proposed DFVC (marked with ✓) successfully retrieves the pages by integrating their adjacent contexts. 

## **5. Conclusions and Future Work** 

## _5.1. Conclusions_ 

In this paper, we have proposed an approach for visual document retrieval called dynamically fusing visual context (DFVC). The proposed DFVC consists of a lightweight adapter designed to run on top of frozen vision-language backbones. In order to constrain the adapter to generate the appropriate weights for adjacent pages, we also design a contrastive loss function incorporating the positive fused embedding vector and negative embedding vectors. Together, the adapter and loss function effectively fuse the context information of neighboring pages while excluding distracting noise across pages. Experimental results on the MMLongBench-Doc and MP-DocVQA benchmarks demonstrates that the proposed DFVC can significantly improve retrieval performance when applied to the VisRAG and ColQwen backbones. In addition, DFVC requires only negligible computational cost, allowing it to serve as a plug-and-play module that can be easily integrated into existing visual RAG pipelines. 

## _5.2. Discussion and Limitation_ 

The proposed DFVC performs better than existing methods by effectively encoding the semantic information across pages. However, while experimental results show that neighboring pages provide meaningful information, there is also a need for effective ways to encode them into a fused embedding vector. 

One limitation of the proposed DFVC is that it only uses neighboring pages. In many real scenarios, the pages that are relevant to the current page may be distributed randomly. In this case, DFVC may not obtain a better embedding vector by only considering the context pages. To effectively encode the meaningful information distributed in nonadjacent pages, one research direction would be to model the relationship between pages and utilize the model to help encode the information contained in them. 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

14 of 15 

**Author Contributions:** Conceptualization, B.Q.; methodology, K.D.; software, Y.W.; validation, J.S.; formal analysis, J.Z.; investigation, K.D.; resources, B.Q.; data curation, Y.W.; writing—original draft preparation, B.Q. and K.D.; writing—review and editing, J.S. and C.F.; visualization, J.Z. and C.F.; supervision, Y.W. and J.Z.; project administration, B.Q., J.Z. and Y.X.; funding acquisition, B.Q., J.Z. and Y.X.; All authors have read and agreed to the published version of the manuscript. 

**Funding:** This research was supported by China Telecom Research Institute Project (Grant No. HQBYG2400153GGN00) and the National Natural Science Foundation of China (Grant No. 62376034). 

**Data Availability Statement:** All data are available within the manuscript. 

**Conflicts of Interest:** Author Bing Qian and Yanru Xue were employed by the company China Telecom Corporation Limited Beijing Research Institute. The remaining authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest. 

## **References** 

1. Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.; Rocktäschel, T.; et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In _Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)_ ; Curran Associates, Inc.: Red Hook, NY, USA, 2020; Volume 33, pp. 9459–9474. 

2. Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; Chang, M.W. REALM: Retrieval-Augmented Language Model Pre-Training. In Proceedings of the 37th International Conference on Machine Learning (ICML), Virtual Event, 13–18 July 2020; Volume 119, pp. 3929–3938. 

3. Blecher, L.; Cucurull, G.; Scialom, T.; Stojnic, R.; Ai, M. Nougat: Neural optical understanding for academic documents. _arXiv_ **2023** , arXiv:2308.13418. [CrossRef] 

4. Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y.; Liu, Z.; Wang, S.; Han, X.; Liu, Z.; et al. VisRAG: Vision-based retrieval-augmented generation on multi-modality documents. _arXiv_ **2024** , arXiv:2410.10594. 

5. Faysse, M.; Sibille, H.; Wu, T.; Omrani, B.; Viaud, G.; Hudelot, C.; Colombo, P. ColPali: Efficient document retrieval with vision language models. _arXiv_ **2024** , arXiv:2407.01449. 

6. Liu, J. LlamaIndex. 2022. Available online: https://github.com/jerryjliu/llama_index (accessed on 8 December 2025). 

7. Liu, N.F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; Liang, P. Lost in the middle: How language models use long contexts. _Trans. Assoc. Comput. Linguist._ **2024** , _12_ , 157–173. [CrossRef] 

8. Wang, H.; Shi, H.; Tan, S.; Qin, W.; Wang, W.; Zhang, T.; Nambi, A.; Ganu, T.; Wang, H. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. In _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies_ ; Association for Computational Linguistics: Stroudsburg, PA, USA, 2025; Volume 1, pp. 3221–3241. 

9. Ma, Y.; Zang, Y.; Chen, L.; Chen, M.; Jiao, Y.; Li, X.; Lu, X.; Liu, Z.; Ma, Y.; Dong, X.; et al. MMLongBench-Doc: Benchmarking Long-Context Document Understanding with Visualizations. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), Vancouver, BC, Canada, 10–15 December 2024; Volume 37, pp. 95963–96010. 

10. Smith, R. An Overview of the Tesseract OCR Engine. In Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), Curitiba, Brazil, 23–26 September 2007; Volume 2, pp. 629–633. 

11. Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W. Dense Passage Retrieval for Open-Domain Question Answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ ; Association for Computational Linguistics: Stroudsburg, PA, USA, 2020; pp. 6769–6781. 

12. Xiong, L.; Xiong, C.; Li, Y.; Tang, K.F.; Liu, J.; Bennett, P.; Ahmed, J.; Overwijk, A. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In _Proceedings of the International Conference on Learning Representations (ICLR)_ ; International Conference on Machine Learning Attn: San Diego, CA, USA , 2021. 

13. Izacard, G.; Caron, M.; Hosseini, L.; Riedel, S.; Bojanowski, P.; Joulin, A.; Grave, E. Unsupervised dense information retrieval with contrastive learning. _Trans. Mach. Learn. Res._ **2022** . [CrossRef] 

14. Xu, Y.; Li, M.; Cui, L.; Huang, S.; Wei, F.; Zhou, M. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Virtual Event, 23–27 August 2020; pp. 1192–1200. 

15. Xu, Y.; Xu, Y.; Lv, T.; Cui, L.; Wei, F.; Wang, G.; Lu, Y.; Florencio, D.; Zhang, C.; Che, W.; et al. LayoutLMv2: Multi-modal Pretraining for Visually-Rich Document Understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)_ ; Association for Computational Linguistics: Stroudsburg, PA, USA, 2021; pp. 2579–2591. 

16. Huang, Y.; Lv, T.; Cui, L.; Lu, Y.; Wei, F. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In Proceedings of the 30th ACM International Conference on Multimedia, Lisbon, Portugal, 10–14 October 2022; pp. 4083–4091. 

https://doi.org/10.3390/electronics15071353 

_Electronics_ **2026** , _15_ , 1353 

15 of 15 

17. Yao, Y.; Yu, T.; Zhang, A.; Wang, C.; Cui, J.; Zhu, H.; Cai, T.; Li, H.; Zhao, W.; He, Z.; et al. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. _arXiv_ **2024** , arXiv:2408.01800. 

18. Khattab, O.; Zaharia, M. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), Xi’an, China, 25–30 July 2020; pp. 39–48. 

19. Beyer, L.; Steiner, A.; Pinto, A.S.; Kolesnikov, A.; Wang, X.; Salz, D.; Neumann, M.; Alabdulmohsin, I.; Tschannen, M.; Bugliarello, E.; et al. PaliGemma: A Versatile 3B VLM for Transfer Learning. _arXiv_ **2024** , arXiv:2407.07726. 

20. Ma, Y.; Li, J.; Zang, Y.; Wu, X.; Dong, X.; Zhang, P.; Cao, Y.; Duan, H.; Wang, J.; Cao, Y.; et al. Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings. _arXiv_ **2025** , arXiv:2506.04997. 

21. Wan, J.; Song, S.; Yu, W.; Liu, Y.; Cheng, W.; Huang, F.; Bai, X.; Yao, C.; Yang, Z. Omniparser: A unified framework for text spotting key information extraction and table recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 16–22 June 2024; pp. 15641–15653. 

22. Yu, W.; Yang, Z.; Wan, J.; Song, S.; Tang, J.; Cheng, W.; Liu, Y.; Bai, X. Omniparser v2: Structured-points-of-thought for unified visual text parsing and its generality to multimodal large language models. _arXiv_ **2025** , arXiv:2502.16161. 

23. Chen, J.; Lin, H.; Han, X.; Sun, L. Benchmarking Large Language Models in Retrieval-Augmented Generation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ ; AAAI Press: Vancouver, BC, Canada, 2024; Volume 38, pp. 17754–17762. 

24. Hofstatter, S.; Lin, S.C.; Yang, J.H.; Lin, J.; Hanbury, A. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Awareness. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), Virtual Event, 11–15 July 2021; pp. 113–122. 

25. Nogueira, R.; Yang, W.; Lin, J.; Cho, K. Document Expansion by Query Prediction. _arXiv_ **2019** , arXiv:1904.08375. [CrossRef] 

26. Wu, X.; Tan, Y.; Hou, N.; Zhang, R.; Cheng, H. Molorag: Bootstrapping document understanding via multi-modal logic-aware retrieval. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ ; Association for Computational Linguistics: Stroudsburg, PA, USA, 2025; pp. 14035–14056. 

27. Yuan, X.; Ning, L.; Fan, W.; Li, Q. mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering. _arXiv_ **2025** , arXiv:2508.05318. 

28. Wang, Q.; Ding, R.; Zeng, Y.; Chen, Z.; Chen, L.; Wang, S.; Xie, P.; Huang, F.; Zhao, F. VRAG-RL: Empower Vision-PerceptionBased RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning. _arXiv_ **2025** , arXiv:2505.22019. 

29. Houlsby, N.; Giurgiu, A.; Jastrzebski, S.; Morrone, B.; De Laroussilhe, Q.; Gesmundo, A.; Attariyan, M.; Gelly, S. ParameterEfficient Transfer Learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (ICML), Long Beach, CA, USA, 9–15 June 2019; Volume 97, pp. 2790–2799. 

30. Hu, E.J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W. LoRA: Low-Rank Adaptation of Large Language Models. In Proceedings of the International Conference on Learning Representations (ICLR), Virtual Event, 25–29 April 2022. 

31. Lester, B.; Al-Rfou, R.; Constant, N. The Power of Scale for Parameter-Efficient Prompt Tuning. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ ; Association for Computational Linguistics: Stroudsburg, PA, USA, 2021; pp. 3045–3059. 

32. Zhai, X.; Mustafa, B.; Kolesnikov, A.; Beyer, L. Sigmoid Loss for Language Image Pre-Training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 1–6 October 2023; pp. 11975–11986. 

33. Tito, R.; Karatzas, D.; Valveny, E. Hierarchical Multimodal Transformers for Multi-Page DocVQA. _Pattern Recognit._ **2023** , _144_ , 109834. [CrossRef] 

34. Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; Liu, Z. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. _arXiv_ **2024** , arXiv:2402.03216. 

**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 

https://doi.org/10.3390/electronics15071353 

