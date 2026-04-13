## **Doc-React: Multi-page Heterogeneous Document Question-answering** 

**Junda Wu**[1] **, Yu Xia**[1] **, Tong Yu**[2] **, Xiang Chen**[2] **, Sai Sree Harsha**[2] **, Akash V. Maharaj**[2] **, Ruiyi Zhang**[2] **, Victor Bursztyn**[2] **, Sungchul Kim**[2] **, Ryan Rossi**[2] **, Julian McAuley**[1] **, Yunyao Li**[2] **, Ritwik Sinha**[2] 1University of California San Diego 2Adobe Inc. 

## **Abstract** 

Answering questions over multi-page, multimodal documents, including text and figures, is a critical challenge for applications that require answers to integrate information across multiple modalities and contextual dependencies. Existing methods, such as single-turn retrieval-augmented generation (RAG), struggle to retrieve fine-grained and contextually relevant information from large, heterogeneous documents, leading to suboptimal performance. Inspired by iterative frameworks like ReAct, which refine retrieval through feedback, we propose Doc-React, an adaptive iterative framework that balances information gain and uncertainty reduction at each step. Doc-React leverages InfoNCE-guided retrieval to approximate mutual information, enabling dynamic sub-query generation and refinement. A large language model (LLM) serves as both a judge and generator, providing structured feedback to iteratively improve retrieval. By combining mutual information optimization with entropyaware selection, Doc-React systematically captures relevant multimodal content, achieving strong performance on complex QA tasks. 

## **1 Introduction** 

Answering queries over multi-page, multimodal documents with textual and visual information is a critical challenge (Ma et al., 2024a; Tanaka et al., 2023). These documents, which are typically long, often have diverse layouts and interleaved content (Nguyen et al., 2024), requiring reasoning across modalities. Specifically, as illustrated by the use case in Figure 1, the input is a user query and multi-page documents, either provided by the user or from a database. The goal is to produce an accurate and concise text answer by analyzing the relevant content across pages. 

Existing methods (Yu et al., 2024; Yang et al., 2023; Faysse et al., 2024; Yao et al., 2022; Wu et al., 2024d) retrieve multimodal content and feed 

it into single-turn RAG models. While effective in simpler scenarios, these methods fall short in multi-page settings, missing fine-grained, contextually relevant information. As illustrated in Figure 1, when applying (Yu et al., 2024; Faysse et al., 2024), the query _Q_ does not explicitly mention the region with “252M mobile broadband subscriptions.” Simply retrieving figures semantically similar to _Q_ fails to identify that the region refers to North America and further locate the figure containing North America’s “active social network users” (described in _S_ 2). This highlights the need for an iterative approach to dynamically refine queries, discover relevant figures, and gather necessary information. Inspired by iterative methods (Yao et al., 2022), we adopt a dynamic approach to refine queries iteratively, addressing the challenges of multi-page document QA. 

We propose Doc-React, an adaptive iterative framework for multi-page document retrieval and acting. Doc-React refines retrieval and query formulation in distinct steps to maximize normalized information gain, balancing information gain and uncertainty reduction with InfoNCE-guided retrieval. It addresses residual gaps through subqueries refined by an LLM, which provides feedback as both judge and generator. By maximizing step-wise information gain, Doc-React iteratively captures fine-grained, contextually relevant information, adapting queries and improving retrieval efficiency. In summary, our contributions are: 

- We propose Doc-React, an adaptive iterative framework that dynamically refines multimodal retrieval and QA performance through LLM-guided feedback. 

- We introduce a method to balance information gain and uncertainty reduction, dynamically adapting sub-queries and retrieval prompts. 

- We conduct extensive experiments showing 

67 

_Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 67–78 July 27 - August 1, 2025 ©2025 Association for Computational Linguistics 

**==> picture [397 x 159] intentionally omitted <==**

**----- Start of picture text -----**<br>
Progressive Framework (Section 4.1)<br>Information Differentiation for<br>Query Formulation (Section 4.2)<br>Query 𝑄𝑄 Sub-query 𝐴𝐴1′ Sub-query 𝐴𝐴′2<br>population in the region with What portion of the total  GPT-4o that [Search] A figure or chart shows  mobile  GPT-4o [Response] North America has  252M  mobile  GPT-4o outputs<br>252M  mobile  broadband  broadband subscriptions  broadband subscriptions.  ……<br>er subscriptions actively uses  ” and active social network  ad [Search] A figure shows  ><br>provides the input  social networks? users by region. total population in North America.<br>query 𝑄𝑄 InfoNCE-guided Information  max𝑆𝑆 [′] Δ(𝑆𝑆 [′] )  in Eq. (7)  Multimodal RAG 𝐴𝐴∖⋃𝑆𝑆𝑘𝑘∈𝜋𝜋𝑡𝑡 𝜇𝜇 𝑆𝑆𝑘𝑘 , 𝑄𝑄  in Eq. (3)  GPT-4o Multimodal RAG The Answer<br>Retrieval (Section 4.3)<br>Multi-page Document Step 1 Retrieval 𝑆𝑆1 Retrieval 𝑆𝑆2 Step 2 percentage of active<br>social<br>= ii 1) ail Mobile Broadband Subscriptions _ North America | network<br>……  users in<br>North<br>America is<br>56%.<br>**----- End of picture text -----**<br>


Figure 1: Doc-React applied to the multi-page document QA task. The framework processes a user query as input and operates on multi-page documents. It iteratively refines information retrieval and query formulation to maximize information gain and reduce uncertainty, ultimately generating an accurate and contextually relevant answer. 

significant improvements over baselines for complex multi-page QA tasks. 

## **2 Related Work** 

## **2.1 Iterative and Adaptive Retrieval** 

Iterative approaches, such as ReAct (Yao et al., 2022), Chain-of-thought (Wei et al., 2022; Wu et al., 2024c,a) and MM-React (Yang et al., 2023), leverage dynamic feedback mechanisms to refine queries and align context (Shinn et al., 2024; Huang et al., 2023; Zhang et al., 2024), effectively addressing challenges in noisy or incomplete retrieval scenarios (Zhou et al., 2024). These methods demonstrate the importance of iterative frameworks (Hu et al., 2024b; Yang et al., 2024) in achieving more accurate and context-aware retrieval. Similarly, mutual information optimization techniques, inspired by InfoNCE (Oord et al., 2018), rely on iterative strategies to balance information gain and uncertainty reduction, further highlighting the relevance of iterative methods for handling complex retrieval tasks. In our framework, InfoNCE-guided mutual information optimization is not only theoretically motivated but also serves as a practical mechanism for adaptive multi-step retrieval (detailed in Section 4). 

## **2.2 Multimodal Document Retrieval** 

Advancements in multimodal learning enable systems to jointly process text and visual information (Yan et al., 2024; Wu et al., 2025, 2024b; Yao et al., 2024; Liu et al., 2024a). Models like LayoutLM (Xu et al., 2020b,a) leverage spatial embeddings for document layout understanding, excelling 

at tasks like form and invoice parsing. These models are limited to single-page documents (Liu et al., 2024b) and cannot handle reasoning over multipage (Ma et al., 2024a), and heterogeneous layouts (Tanaka et al., 2023). 

## **2.3 Retrieval-Augmented Generation (RAG)** 

RAG (Lewis et al., 2020) and FiD (Izacard and Grave, 2020) combine retrieval and generation for open-domain QA. While effective, their reliance on static pipelines limits adaptability to complex document layouts. Retrieval methods like BM25 (Robertson and Walker, 1994) and dense models such as ColBERT (Khattab and Zaharia, 2020) efficiently match content but fail to capture the crossmodal dependencies for multimodal documents. MuRAR (Zhu et al., 2024) generates multimodal responses to convey rich cross-modal information. 

## **3 Formulation: Multi-page Doc-QA** 

We introduce a novel formulation for the multipage document question-answering (Doc-QA) task, where the input consists of heterogeneous documents that interleave images and texts in unstructured, free-form layouts. This practical task involves disentangling multimodal information embedded within the images of document pages to accurately answer a given query. The objective of a multi-page Doc-QA is to extract sufficient multimodal information from the relevant pages of a document and use it to generate a correct answer. Our novel formulation is different from the formulation of existing works (Mathew et al., 2021; Chang et al., 2022; Masry et al., 2022; Liu et al., 

68 

2024b) that we consider question-answering with a long sequence of document page images due to heterogeneous layouts of various types of documents. Different from tool-chain based agentic framework (Yang et al., 2023; Ma et al., 2024b; Kumar et al., 2024), our formulation focuses on balancing between information gain and generation uncertainty when multiple images are in the prompts. 

Given a _N_ -page document _D_ = _{p_ 1 _, p_ 2 _, . . . , pN }_ and a query _Q_ , an underlying policy is required to identify multiple groups of document pages _π_ = _{Si|Si ⊆ D}_ , where each group of document pages _Si_ provides relevant information to a specific aspect of the query _Q_ . The multimodal information necessary to derive the answer _A_ must then be extracted from each subset _Si_ , conditioned on the query _Q_ , using a multimodal large language model (MLLM) denoted as _µ_ ( _Si, Q_ ). The final output is a text-only response, generated based on the extracted multimodal content. To quantify the informativeness of this process, we measure the mutual information _I_ ( _µ_ ( _Si, Q_ ); _A_ ) as the amount of information required to determine the answer A. 

We further formalize this task as an information optimization problem, which seeks a balance between reducing uncertainty and ensuring information completeness (Alemi et al., 2016): 

**==> picture [214 x 57] intentionally omitted <==**

**==> picture [13 x 10] intentionally omitted <==**

where _H_ ( _·_ ) denotes the entropy, capturing uncertainty in the extracted information. A key practical challenge is the increasing estimation error as the number of input images grows. Empirically, we can observe that there exist subsets _S_ 1 _, S_ 2 _⊆ D_ such that _H_ ( _µ_ ( _S_ 1 _, S_ 2 _, Q_ )) _> H_ ( _µ_ ( _S_ 1 _, Q_ )) + _H_ ( _µ_ ( _S_ 2 _, Q_ )). This implies that increasing the size of the input set does not always improve answer accuracy, while the increased input size also retrains the usage of MLLMs, due to their limited perception of multimodal context. 

## **4 Method: Doc-React** 

The formulation’s major challenge is incrementally identifying and aggregating multimodal information from a multi-page document corpus (Hu et al., 2024a; Cho et al., 2024). Inspired by informationentropy optimization (Alemi et al., 2016; Still, 

**Algorithm 1 Doc-React** : Adaptive Retrieval with Normalized MI Gain Maximization 

**Require:** Document _D_ = _{p_ 1 _, p_ 2 _, . . . , pN }_ , Query _Q_ , MLLM _µ_ , Retrieval model _f_ ( _·, ·_ ), Differentiation prompt _P_ diff, Maximum iteration _T_ , Candidate pool Ω. 

1: Initialize _π_ 0 _←∅_ . 

2: **for** _t_ = 0 to _T −_ 1 **do** 

3: **Sub-query Formulation with Residual Information Approximation:** 

4: Formulate sub-queries according to Eq.(5); 5: **Candidate Evaluation:** 6: **for** each _S[′] ∈_ Ω **do** 

7: Estimate the lower bound of the mutual information gain according to Eq.(6); 

8: Compute entropy _H_ ( _µ_ ( _S[′] , Q_ )) from MLLM’s generation distributions; 

- 9: Compute normalized gain ∆( _S[′]_ ) according to Eq.(4); 

10: **end for** 

11: **Select Best Subset:** 12: _St[∗]_ +1 _[←]_[arg max] _[S][′][∈]_[Ω] _[′]_[ ∆(] _[S][′]_[)] 13: **Update:** 14: _πt_ +1 _← πt ∪{St[∗]_ +1 _[}]_ 15: **if** stopping criteria met **then** 16: **break** 

17: **end if** 

18: **end for** 

- 19: **return** Final selected subsets _πT_ , and final retrieval results _µ_ ( _{Sk}k∈πT , Q_ ). 

2009), we propose **Doc-React** , which greedily maximizes the entropy-normalized information gain for each round of retrieval and action. 

## **4.1 Progressive Framework Refining Retrieval and Query Formulation** 

At time step _t_ , we maintain a selection of document page subsets _πt_ , where each subset’s _Sk ∈ πt_ multimodal information is extracted by the MLLM _µ_ ( _Sk, Q_ ). To solve the information optimization problem in Eq.(1), we propose to maximize the _normalized information gain_ for the next step _t_ +1: 

**==> picture [227 x 46] intentionally omitted <==**

where _A_ represents the ground-truth answer to the query. However, different from conventional goaloriented optimization tasks (Sutton, 2018; Levine et al., 2016), which assume a good representation 

69 

of the goal, our task poses a challenge of lacking such _oracle information_ . 

## **4.2 Information Differentiation** 

For query formulation, we propose an information differentiation process to approximate each step’s maximum information gain, 

**==> picture [167 x 26] intentionally omitted <==**

where _A[′] t_[denotes the information residual at time] step _t_ . Takes Eq.(3) into Eq.(2), we can derive the approximation of the original information gain, 

**==> picture [180 x 27] intentionally omitted <==**

where the residual information is practically approximated by the MLLM 

**==> picture [172 x 14] intentionally omitted <==**

which aggregates retrieved information _{Sk}k∈πt_ and the query _Q_ , quantifying the information gap. 

**Theorem 1** ( **Information Gain Derivation via Information Differentiation** ) **.** _With the information differentiation enabled as A[′] t[(in Eq.]_[(][3][)] _[), the infor-] mation gain_ ∆( _St_ +1) _in Eq._ (2) _can be derived via the information differentiation in Eq._ (4) _._ 

We prove the equivalency between Eq.(2) and Eq.(4) in Appendix D. Then, this gap is addressed by leveraging the LLM-as-a-judge (Zheng et al., 2023; Rosset et al., 2024; Gu et al., 2024) to evaluate and decompose complex queries (Xia et al., 2024). Specifically, the gap is formulated into subqueries by following the in-context examples (in Appendix E). 

## **4.3 InfoNCE-guided Information Retrieval** 

To further track the information gain _I_ ( _µ_ ( _St_ +1 _, Q_ ); _A[′] t_[)][,] we introduce existing multimodal retrieval models (Faysse et al., 2024; Yu et al., 2024) as prior, which can efficiently provide such information quantification through a retrieval process. Given the multimodal retrieval model _f_ ( _·, ·_ ) which measures the distance between multimodal inputs, we can derive the variational lower bound for the mutual information estimation following InfoNCE (Oord et al., 2018), 

**==> picture [214 x 51] intentionally omitted <==**

|Dataset|Dataset|MMLongBench|SlideVQA|
|---|---|---|---|
|Method<br>Model||F1<br>Acc|F1<br>Acc|
|||||
|**VisRAG**<br>GPT-4o<br>InternVL<br>MiniCPM||29.02<br>27.78<br>11.43<br>11.03<br>8.28<br>8.17|48.37<br>46.93<br>18.13<br>17.28<br>12.62<br>11.83|
|||||
|**ColPali**<br>GPT-4o<br>InternVL<br>MiniCPM||32.17<br>30.76<br>8.86<br>8.57<br>7.84<br>7.74|53.42<br>52.33<br>16.34<br>15.23<br>13.34<br>12.48|
|||||
|**DocReact**<br>GPT-4o||**38.07**<br>**38.29**|**54.87**<br>**55.04**|



Table 1: Comparison with multimodal retrievalaugmented generation baselines. 

|Dataset|MMLongBench|SlideVQA|
|---|---|---|
|Method<br>Model|F1<br>Acc|F1<br>Acc|
||||
|**Standard**<br>**(i.e., w/o CoT)**<br>GPT-4o<br>InternVL<br>MiniCPM|**42.00**<br>**41.71**<br>4.38<br>4.31<br>5.59<br>5.55|53.58<br>53.41<br>19.00<br>18.20<br>13.85<br>13.03|
||||
|**CoT**<br>GPT-4o<br>InternVL<br>MiniCPM|41.17<br>40.77<br>10.79<br>10.47<br>6.37<br>6.13|52.12<br>51.71<br>17.45<br>16.74<br>12.45<br>11.73|
||||
|**DocReact**<br>GPT-4o|38.07<br>38.29|**54.87**<br>**55.04**|



Table 2: Comparisons with multi-image multimodal LLM baselines. 

where Ω= _{S[′] | ∀S[′] ⊆ D}_ , while in practice we select from a random subset Ω _[′]_ to reduce the computation complexity. In addition, the estimation uncertainty _H_ ( _µ_ ( _St_ +1 _, Q_ )) can be calculated based on the MLLM’s generation probabilities. 

Therefore, using Eq.(6) in Eq.(2), we optimize step-wise normalized information gain ∆( _St_ +1), by maximizing its lower-bound estimation, 

**==> picture [164 x 36] intentionally omitted <==**

for which, we can achieve the near-optimal solution to the original problem in Eq.(1). We illustrate our method in Algorithm 1. 

## **5 Experiments** 

For our comparison experiments, we use two lines of baselines, including **Multi-image MLLMs:** GPT4-o, InternVL (Chen et al., 2024), and MiniCPM (Yao et al., 2024), suitable for multipage inputs; and **Multimodal RAG:** ColPali (Faysse et al., 2024) and VisRAG-Ret (Yu et al., 2024). To evaluate baselines and DocReact on multi-page document question-answering, we apply our method on two datasets, **SlideVQA** (Tanaka et al., 2023) and **VisualWebBench** (Liu et al., 2024b). For details, please refer to Appendix C. 

70 

|Method|MMLongBench<br>F1<br>Acc|SlideVQA<br>F1<br>Acc|
|---|---|---|
|ColPali<br>DocReact (w/o ColPali)<br>DocReact (w/ ColPali)|32.17<br>30.76<br>37.22<br>37.37<br>**38.07**<br>**38.29**|53.42<br>52.33<br>54.12<br>54.29<br>**54.87**<br>**55.04**|



Table 3: Ablation study comparing Doc-React with and without ColPali retrieval. Results are reported on the MMLongBench and SlideVQA benchmarks. 

**Comparison with Multimodal RAG** We evaluate DocReact’s effectiveness against multimodal RAG methods in Table 1. While multimodal RAG methods mitigate input information overload, they struggle with complex queries requiring progressive multi-image reasoning and retrieval. In contrast, DocReact addresses these challenges through step-wise information differentiation and retrieval, achieving strong performance on both datasets. 

**Comparison with Multi-image MLLMs** We evaluate DocReact’s efficiency in processing multipage document inputs compared to multi-image MLLMs. As shown in Table 2, GPT-4o performs well on MMLongBench, where the limited number of candidate images allows it to process the full context effectively. However, _simply relying on GPT-4o (i.e., the baseline Standard) is not scalable for real-world scenarios with a larger number of candidate images_ , where DocReact’s iterative approach becomes essential for efficiently retrieving fine-grained, contextually relevant information. While CoT reasoning offers limited improvement on GPT-4o due to inefficiencies in handling multimodal inputs, DocReact achieves competitive performance on MMLongBench and surpasses GPT4o on the SlideVQA dataset, showing its scalability and effectiveness in handling multi-page QA tasks. **Case Study** We validate DocReact’s advantage by a case study (Appendix A). In this example, the query _Q_ does not explicitly mention the region with “252M mobile broadband subscriptions.” Advanced retrieval methods such as ColPali (Faysse et al., 2024) fail to infer that the region refers to North America and subsequently miss the figure containing relevant information on active social network users. In contrast, DocReact dynamically refines its search process. Through intermediate reasoning steps (Thought 2 and Thought 3), it identifies the missing region, retrieves the relevant figure, and gets the information (Observation 3), ultimately yielding the accurate result of 56% (Action 4). **Ablation Study on the Effect of Retriever** Instead of relying solely on the strong retrieval approach 

ColPali (Faysse et al., 2024; Yao et al., 2024; Chen et al., 2024), we conducted additional experiments to compare **DocReact (w/ ColPali)** and **DocReact (w/o ColPali)** . In the latter setting, we replaced ColPali with a weaker retrieval method as indicated in Table 3. This comparison allows us to better understand whether the observed improvements are attributable to our framework itself, rather than being solely due to the strong retrieval backbone. Our experiments demonstrate that even without leveraging the strong ColPali method, our approach still outperforms directly using ColPali. 

## **6 Conclusion** 

We address the challenge of answering questions over multi-page, multimodal documents by introducing Doc-React, an adaptive iterative framework for retrieval and acting. By balancing information gain and uncertainty reduction at each step, the framework leverages InfoNCE-guided retrieval and a large language model (LLM) for dynamic query refinement. Through systematic optimization and feedback, Doc-React effectively captures fine-grained and contextually relevant information, and achieves strong performance. 

## **7 Limitations** 

Our approach maximizes the normalized mutual information gain, which is achieved through an iterative framework that approximates this objective (as discussed in the theoretical analysis in Appendix D). While the framework effectively refines retrieval and query formulation, the approximation process may lead to slight inefficiencies in specific cases. Future work could focus on refining the optimization process to further streamline the approximation. 

## **References** 

Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. 2016. Deep variational information bottleneck. _arXiv preprint arXiv:1612.00410_ . 

71 

- Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. 2022. Webqa: Multihop and multimodal qa. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 16495–16504. 

- Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 24185–24198. 

- Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multimodal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ . 

- Manuel Faysse, Hugues Sibille, Tony Wu, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ . 

- Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. 2024. A survey on llm-as-a-judge. _arXiv preprint arXiv:2411.15594_ . 

- Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. 2024a. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_ . 

- Xiang Hu, Hongyu Fu, Jinge Wang, Yifeng Wang, Zhikun Li, Renjun Xu, Yu Lu, Yaochu Jin, Lili Pan, and Zhenzhong Lan. 2024b. Nova: An iterative planning and search approach to enhance novelty and diversity of llm generated ideas. _arXiv preprint arXiv:2410.14255_ . 

- Di Huang, Ziyuan Nan, Xing Hu, Pengwei Jin, Shaohui Peng, Yuanbo Wen, Rui Zhang, Zidong Du, Qi Guo, Yewen Pu, et al. 2023. Anpl: towards natural programming with interactive decomposition. _Advances in Neural Information Processing Systems_ , 36:69404– 69440. 

- Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. _arXiv preprint arXiv:2007.01282_ . 

- Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In _Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval_ , pages 39– 48. 

- Somnath Kumar, Yash Gadhia, Tanuja Ganu, and Akshay Nambi. 2024. Mmctagent: Multi-modal critical thinking agent framework for complex visual reasoning. _arXiv preprint arXiv:2405.18358_ . 

- Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. 2016. End-to-end training of deep visuomotor policies. _Journal of Machine Learning Research_ , 17(39):1–40. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024a. Visual instruction tuning. _Advances in neural information processing systems_ , 36. 

- Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. 2024b. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding? _arXiv preprint arXiv:2404.05955_ . 

- Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. 2024a. Mmlongbenchdoc: Benchmarking long-context document understanding with visualizations. _arXiv preprint arXiv:2407.01523_ . 

- Zixian Ma, Jianguo Zhang, Zhiwei Liu, Jieyu Zhang, Juntao Tan, Manli Shu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Caiming Xiong, et al. 2024b. Taco: Learning multi-modal action models with synthetic chains-of-thought-and-action. _arXiv preprint arXiv:2412.05479_ . 

- Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ . 

- Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209. 

- Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, et al. 2024. Gui agents: A survey. _arXiv preprint arXiv:2412.13501_ . 

- Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ . 

- Stephen E Robertson and Steve Walker. 1994. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In _SIGIR’94: Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University_ , pages 232–241. Springer. 

72 

- Corby Rosset, Ho-Lam Chung, Guanghui Qin, Ethan C Chau, Zhuo Feng, Ahmed Awadallah, Jennifer Neville, and Nikhil Rao. 2024. Researchy questions: A dataset of multi-perspective, decompositional questions for llm web agents. _arXiv preprint arXiv:2402.17896_ . 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2024. Reflexion: Language agents with verbal reinforcement learning. _Advances in Neural Information Processing Systems_ , 36. 

- Susanne Still. 2009. Information-theoretic approach to interactive learning. _Europhysics Letters_ , 85(2):28005. 

- Richard S Sutton. 2018. Reinforcement learning: An introduction. _A Bradford Book_ . 

- Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In _AAAI_ . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837. 

- Junda Wu, Xintong Li, Ruoyu Wang, Yu Xia, Yuxin Xiong, Jianing Wang, Tong Yu, Xiang Chen, Branislav Kveton, Lina Yao, et al. 2024a. Ocean: Offline chain-of-thought evaluation and alignment in large language models. _arXiv preprint arXiv:2410.23703_ . 

- Junda Wu, Xintong Li, Tong Yu, Yu Wang, Xiang Chen, Jiuxiang Gu, Lina Yao, Jingbo Shang, and Julian J McAuley. 2024b. Commit: Coordinated instruction tuning for multimodal large language models. _CoRR_ . 

- Junda Wu, Yuxin Xiong, Xintong Li, Yu Xia, Ruoyu Wang, Yu Wang, Tong Yu, Sungchul Kim, Ryan A Rossi, Lina Yao, et al. 2025. Mitigating visual knowledge forgetting in mllm instruction-tuning via modality-decoupled gradient descent. _arXiv preprint arXiv:2502.11740_ . 

- Junda Wu, Tong Yu, Xiang Chen, Haoliang Wang, Ryan Rossi, Sungchul Kim, Anup Rao, and Julian McAuley. 2024c. Decot: Debiasing chain-of-thought for knowledge-intensive tasks in large language models via causal intervention. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 14073–14087. 

- Junda Wu, Zhehao Zhang, Yu Xia, Xintong Li, Zhaoyang Xia, Aaron Chang, Tong Yu, Sungchul Kim, Ryan A Rossi, Ruiyi Zhang, et al. 2024d. Visual prompting in multimodal large language models: A survey. _arXiv preprint arXiv:2409.15310_ . 

- Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A Rossi, Haoliang Wang, and Julian McAuley. 2024. Knowledge-aware query expansion with large language models for textual and relational retrieval. _arXiv preprint arXiv:2410.13765_ . 

- Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. 2020a. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. _arXiv preprint arXiv:2012.14740_ . 

- Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020b. Layoutlm: Pre-training of text and layout for document image understanding. In _Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining_ , pages 1192–1200. 

- An Yan, Zhengyuan Yang, Junda Wu, Wanrong Zhu, Jianwei Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Julian McAuley, Jianfeng Gao, et al. 2024. List items one by one: A new data source and learning paradigm for multimodal llms. _arXiv preprint arXiv:2404.16375_ . 

- Yujing Yang, Boqi Chen, Kua Chen, Gunter Mussbacher, and Dániel Varró. 2024. Multi-step iterative automated domain modeling with large language models. In _Proceedings of the ACM/IEEE 27th International Conference on Model Driven Engineering Languages and Systems_ , pages 587–595. 

- Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023. Mmreact: Prompting chatgpt for multimodal reasoning and action. _arXiv preprint arXiv:2303.11381_ . 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ . 

- Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. _arXiv preprint arXiv:2408.01800_ . 

- Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. 2024. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. _arXiv preprint arXiv:2410.10594_ . 

- Xianren Zhang, Xianfeng Tang, Hui Liu, Zongyu Wu, Qi He, Dongwon Lee, and Suhang Wang. 2024. Divide-verify-refine: Aligning llm responses with complex instructions. _arXiv preprint arXiv:2410.12207_ . 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot 

73 

arena. _Advances in Neural Information Processing Systems_ , 36:46595–46623. 

- Zhanke Zhou, Rong Tao, Jianing Zhu, Yiwen Luo, Zengmao Wang, and Bo Han. 2024. Can language models perform robust reasoning in chain-of-thought prompting with noisy rationales? _arXiv preprint arXiv:2410.23856_ . 

- Zhengyuan Zhu, Daniel Lee, Hong Zhang, Sai Sree Harsha, Loic Feujio, Akash Maharaj, and Yunyao Li. 2024. Murar: A simple and effective multimodal retrieval and answer refinement framework for multimodal question answering. _arXiv preprint arXiv:2408.08521_ . 

74 

## **A Case Study** 

In this section, we illustrate a case study about the comparison between DocReact and CoT baseline. The desired answer, according to the ground truth, is 56%. We show in the use case that DocReact can successfully enable information verbal differentiation by identifying the missing information in the current observation (highlighted in green font). In addition, DocReact can further formulate sub-queries based on the information differentiation results, which can be used to retrieve more document image potentially containing such information (highlighted in orange font). 

**Question** : What percentage of the TOTAL POPULATION in the region where there are 252M mobile broadband subscriptions are ACTIVE SOCIAL NETWORK USERS? (i) The result by our DocReact 

**Thought 1** : I need to find the region where there are 252M mobile broadband subscriptions and then determine the percentage of the total population in that region who are active social network users. **Action 1** : Search[A figure or chart that shows mobile broadband subscriptions and active social network users by region.] **Observation 1** : 6,572,950,124 mobile subscriptions 

**Thought 2** : The observation does not provide the necessary information about the region with 252M mobile broadband subscriptions. I need to search again for a figure or chart that shows mobile broadband subscriptions and active social network users by region. 

**Action 2** : Search[A figure or chart that shows mobile broadband subscriptions and active social network users by region, specifically looking for 252M mobile broadband subscriptions.] **Observation 2** : North America has 252M mobile broadband subscriptions 

**Thought 3** : I need to search for a figure or chart that shows active social network users in North America. **Action 3** : Search[A figure or chart that shows active social network users in North America.] **Observation 3** : 56% North America’s share of active social network **Action 4** : Finish[56%] 

(ii) The result by the baseline Standard 

The image does not provide a direct percentage of active social network users in the region with 252M mobile broadband subscriptions. However, it does show that there are 173,871,400 active social network users in the region with 252M mobile broadband subscriptions. To calculate the percentage, we would need the total population of that region, which is not provided in the image. 

(iii) The result by the baseline CoT 

The image shows a bar graph with different regions and their respective mobile broadband subscriptions and active social network users. To find the percentage of active social network users, we need to look at the bar corresponding to the region with 252M mobile broadband subscriptions. The bar indicates that the percentage of active social network users in that region is 17%. Therefore, the answer is 17%. 

## **B Implementation Discussions** 

While our approach may involve more API calls than baseline methods due to its iterative nature, it is important to note that iterative frameworks are both necessary and common for addressing complex retrieval tasks. Prior works, such as ReAct (Yao et al., 2022), have demonstrated the effectiveness of dynamic feedback mechanisms for refining queries in noisy or incomplete scenarios. Similarly, mutual information optimization methods inspired by InfoNCE (Oord et al., 2018) rely on iterative strategies to balance information gain and uncertainty reduction effectively. 

ing an iterative query refinement framework to progressively retrieve the most relevant information. This approach significantly reduces the number of input tokens processed per API call by avoiding redundant or irrelevant content. Consequently, while iterative, Doc-React achieves a lower overall computational cost in terms of processed tokens compared to baseline methods, which often rely on processing larger, less refined multimodal contexts in a single pass. The iterative design is not only practical but also essential for tackling the complexity of multi-page, multimodal document QA tasks. Given the markedly inferior performance of opensourced MLLMs (e.g., MiniCPM and InternVL) on 

Doc-React builds on these principles by leverag- 

75 

such challenging tasks, we opt to build DocReact on the stronger baseline GPT-4o to enable a more rigorous comparison. In practice, we limit the process to a maximum of 4 iterations of retrieval and action steps. The MLLM has the flexibility to terminate early by directly outputting the final answer. However, if the MLLM does not voluntarily end the interaction, the system enforces termination after the final iteration. At this point, the answer is generated using aggregated information from all previous rounds. 

## **C Experiment Details** 

- **Hardware:** Experiments are conducted on 4 NVIDIA A6000 GPUs with multi-GPU support. 

## **D Theoretical Analysis** 

This section provides more details and the proof of Theorem 1. 

_Proof._ Denote the accumulated information as 

**==> picture [95 x 26] intentionally omitted <==**

We can assume the information differentiation by 

## **C.1 Dataset Details** 

We evaluate our method on the following datasets: 

- **SlideVQA** (Tanaka et al., 2023): Designed for structured documents like presentation slides, focusing on reasoning with visual elements such as diagrams and charts. 

- **MMLongBench-Doc** (Ma et al., 2024a): Evaluates tasks requiring long-context reasoning over multimodal documents. 

## **C.2 Baseline Details** 

We include the following baselines in our comparisons: 

- **Multi-image MLLMs:** GPT4-o, InternVL (Chen et al., 2024), and MiniCPM (Yao et al., 2024). 

- **Multimodal RAG:** ColPali (Faysse et al., 2024) and VisRAG-Ret (Yu et al., 2024). 

## **C.3 Implementation Details** 

- **Framework:** Implemented using the Hugging Face Transformers library for integration with retrieval and generative models. 

- **Query Refinement:** Iterative query refinement is driven by LLM feedback and InfoNCE-based optimization. 

**==> picture [148 x 12] intentionally omitted <==**

where _g_ ( _Z_ ) is the part of _A_ explained by _Z_ . By definition of conditional mutual information, the information gain term in Eq.(2) can be derived 

**==> picture [210 x 28] intentionally omitted <==**

Using the decomposition assumption (in Eq.(8)) of information differentiation, we show that 

**==> picture [183 x 29] intentionally omitted <==**

In addition, since the differentiated information _A[′] t_ is assumed to be independent from the currently collected information _Zt_ , we can further reduce the results as 

**==> picture [174 x 29] intentionally omitted <==**

Therefore, we could conclude with the equivalency 

**==> picture [211 x 30] intentionally omitted <==**

- **Hyperparameters:** We set generation temperature as 0 for deterministic decoding. We set the token limit of 100 for generation and the maximum number of images input as 50 for multi-image MLLM baselines. 

- **Evaluation Metrics:** We follow the evaluation metrics introduced by MMLongBenchDoc (Ma et al., 2024a) where Acc and F1 are used to meature QA performance. 

76 

## **E In-context Examples** 

We illustrate the three in-context learning examples we used to align MLLMs’ behaviors in generating thoughts of information differentiation ( **Thought** _t_ ), actions of multimodal search sub-queries ( **Action** _t_ ), and MLLM received feedback back through MLLM information retrieval ( **Observation** _t_ ). 

## **Example 1:** 

**Question** : What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into? 

**Thought 1** : I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area. 

**Action 1** : Search[A figure of the regional map of Colorado orogeny, where eastern sector is displayed.] **Observation 1** : Based on the give images, we can observe as follows. The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai progeny. 

**Thought 2** : It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector. 

**Action 2** : Lookup[Extract the eastern sector information from the given figure.] 

**Observation 2** : Based on the give images, we can observe as follows. The Colorado orogen, formerly called the Colorado province, is a >500-km-wide belt of oceanic arc rock (1.78–1.65 Ga) that extends southward into New Mexico and composes a major part of the Proterozoic provinces of southwestern United States. This transcontinental collisional event occurred during the Paleoproterozoic (Statherian Period). The Wyoming sector of the Colorado orogeny was formerly called the Medicine Bow orogeny. The eastern sector extends into the High Plains and is called the Central Plains orogeny. The boundary between the Colorado orogeny and the Wyoming craton is the Cheyenne belt, a 5-km-wide mylonitic shear zone that verges northward. The Cheyenne belt transects and cuts off the south edge of the older Trans-Hudson progeny. 

**Thought 3** : The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range. 

**Action 3** : Search[A figure of High Plains, where its elevation range is annotated.] 

**Observation 3** : Based on the give images, we can observe as follows. The High Plains are a subregion of the Great Plains, mainly in the Western United States, but also partly in the Midwest states of Nebraska, Kansas, and South Dakota, generally encompassing the western part of the Great Plains before the region reaches the Rocky Mountains. The High Plains are located in eastern Montana, southeastern Wyoming, southwestern South Dakota, western Nebraska, eastern Colorado, western Kansas, eastern New Mexico, western Oklahoma, and to just south of the Texas Panhandle. The southern region of the Western High Plains ecology region contains the geological formation known as Llano Estacado which can be seen from a short distance or on satellite maps. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m). 

**Thought 4** : High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft. **Action 4** : Finish[1,800 to 7,000 ft] 

77 

## **Example 2:** 

**Question** : Which magazine was started first Arthur’s Magazine or First for Women? 

**Thought 1** : I need to search Arthur’s Magazine and First for Women, and find which was started first. **Action 1** : Search[A wiki document page about Arthur’s Magazine.] 

**Observation 1** : Based on the give images, we can observe as follows. Arthur’s Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into Godey’s Lady’s Book. 

**Thought 2** : Arthur’s Magazine was started in 1844. I need to search First for Women next. **Action 2** : Search[A webpage introduction about First for Women.] 

**Observation 2** : Based on the give images, we can observe as follows. First for Women is a woman’s magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies. 

**Thought 3** : First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first. **Action 3** : Finish[Arthur’s Magazine] 

## **Example 3:** 

**Question** : Guitars for Wounded Warriors is an album that was recorded in the village in which New York county? 

**Thought 1** : I need to search Guitars for Wounded Warriors, then find in which New York county it was recorded. 

**Action 1** : Search[A document page about Guitars for Wounded Warriors.] 

**Observation 1** : Based on the give images, we can observe as follows. Guitars for Wounded Warriors is the sixth solo studio album by guitarist Brian Tarquin, released in August 2014 by Cleopatra Records. In 2016, it received a Best Compilation Album nomination by the Independent Music Awards. All tracks were recorded at Tarquin’s Jungle Room Studios in New Paltz (village), New York. Being moved by the lack of support for returning veterans through his life Tarquin decided to compose a heavy instrumental rock album as a way to show his appreciation to all veterans. So once again he enlisted top notch players to guest with him on the album, Billy Sheehan on Black Hawk, Ron "Bumblefoot" Thal and Reb Beach on Taliban Terror, Steve Morse on Freedom, Chris Poland on Charlie Surfs and Hunting, Chuck Loeb on Escape Kabul, Hal Lindes on Sand & Blood, Gary Hoey on 5 Klicks To Hell and Baghdad, Alex De Rosso Dokken on Broken Arrow, and The Flyin’ Ryan Brothers on Surgical Strike. The entire album was engineered, produced and composed by Tarquin especially for each one of the guest guitarists. Partial proceeds are donated to the Fisher House Foundation from sales. 

**Thought 2** : Guitars for Wounded Warriors was recorded in New Paltz (village), New York. I need to search New Paltz (village), New York and find which county it is in. 

**Action 2** : Search[A figure of map that indicates the location of New Paltz (village), New York.] 

**Observation 2** : Based on the give images, we can observe as follows. New Paltz is a village in Ulster County located in the U.S. state of New York. It is approximately 80 miles (130 km) north of New York City and 70 miles (110 km) south of Albany. The population was 7,324 at the 2020 census. 

**Thought 3** : New Paltz (village), New York is in Ulster County, so the answer is Ulster County. **Action 3** : Finish[Ulster County] 

78 

