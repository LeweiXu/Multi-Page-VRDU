# **empty PDF-WuKong : A Large Multimodal Model for Efficient Long PDF Reading with End-to-End Sparse Sampling** 

Xudong Xie[1] _[∗]_ Hao Yan[1] _[∗]_ Liang Yin[1] _[∗]_ Yang Liu[1] _[∗]_ Jing Ding[1] Minghui Liao[2] Yuliang Liu[1] Wei Chen[1(] ) Xiang Bai[1(] ) 1Huazhong University of Science and Technology 2Huawei Inc. _{_ lemuria ~~c~~ hen,xbai _}_ @hust.edu.cn 

## **Abstract** 

_Multimodal document understanding is a challenging task to process and comprehend large amounts of textual and visual information. Recent advances in Large Language Models (LLMs) have significantly improved the performance of this task. However, existing methods typically focus on either plain text or a limited number of document images, struggling to handle long PDF documents with interleaved text and images, especially for academic papers. In this paper, we introduce_ _**PDF-WuKong** , a multimodal large language model (MLLM) which is designed to enhance multimodal question-answering (QA) for long PDF documents. PDF-WuKong incorporates a sparse sampler that operates on both text and image representations, significantly improving the efficiency and capability of the MLLM. The sparse sampler is integrated with the MLLM’s image encoder and selects the paragraphs or diagrams most pertinent to user queries for processing by the language model. To effectively train and evaluate our model, we construct_ _**PaperPDF** , a dataset consisting of a broad collection of English and Chinese academic papers. Multiple strategies are proposed to automatically generate_ _**1.1 million** QA pairs along with their corresponding evidence sources. Experimental results demonstrate the superiority and high efficiency of our approach over other models on the task of long multimodal document understanding, surpassing proprietary products by an average of_ _**8.6%** on F1. Our code and dataset will be released at https://github.com/yhhust/PDF-Wukong._ 

## **1. Introduction** 

The advent of Large Language Models (LLMs) has significantly advanced the field of PDF document understanding [1, 2], where these models have demonstrated impres- 

> _∗_ Equal contribution; Corresponding author 

**==> picture [236 x 146] intentionally omitted <==**

**----- Start of picture text -----**<br>
PDF  Long-context/RAG LLM<br>Parser<br>Pure Text<br>oe<br>(a) Plain text solution<br>Vision Page-level Feature  LLM<br>Encoder Interaction<br>(b) Pure vision solution<br>[Para. 1],<br>Long PDF Document ParserPDF  [Para. 2], [Img. 1], [Para. 15], ..., SamplerSparse  [Para. 3], [Img. 2 tokens] LLM<br>[Img. 2], …    Sampled Evidence<br>(c) Ours for interleaved text and images<br>**----- End of picture text -----**<br>


Figure 1. Method comparison for long multi-page PDF document understanding. (a) Plain text solution: long-context/RAG LLMs for parsed pure text content. (b) Pure vision solution: VDU models for page-level encoding and feature interaction. (c) Our method is based on end-to-end sparse sampling for long PDFs with interleaved text and images. 

sive capabilities in processing and generating human-like text. However, they still face many challenges when it comes to lengthy PDF documents with interlaced text and images, such as academic papers. 

To handle lengthy documents, current research in multimodal document understanding with LLMs primarily follows two mainstream technical routes. The first route is based on _pure text modality understanding_ . As shown in Fig. 1(a), these approaches typically consider the parsed OCR results from PDF documents, and convert all visual elements into textual representations (e.g., captions and OCR content extracted from figures and tables). They then employ long-context LLMs [3–5] or utilize Retrieval Augmented Generation (RAG) techniques [6–9] to process the textual content. The main disadvantage of this approach is the significant loss of visual information inherent in multimodal documents, making it challenging to support answering fine-grained visual-related questions. 

1 

The second route is based on _pure visual modality understanding_ . As illustrated in Fig.1(b), this approach generally avoids parsing PDF documents and instead treats each page as an image, and utilizing multimodal LLMs for document understanding tasks[10–12]. However, high-resolution images and numerous pages generate a large number of visual tokens, significantly increasing the models’ input token length and leading to scalability issues. This makes it challenging to efficiently process multi-page long documents. While some recent multi-page VDU models handle up to 8 pages [13] or 20 pages [14], they typically encode each page separately and perform page-level visual feature interactions such as concatenation [15, 16]. Nevertheless, as the number of pages grows, computational resource consumption still escalates substantially, rendering these models inefficient for processing longer documents. 

Considering the limitations of existing methods in multimodal understanding of long PDF documents, we propose a new MLLM architecture with end-to-end sparse sampling, named **PDF-WuKong** . Since most user queries are only related to a small part of the content in a long document, the sparse sampling can significantly remove redundant noise information. It encodes text paragraphs and diagrams in the parsed PDF document, utilizing both text and image representations to identify and extract the most relevant evidence in response to a user’s query. The sampled sparse evidence significantly reduces the number of input tokens of LLM and this process is independent of the length of the input documents. Moreover, the sparse sampler and LLM can be integrated in an end-to-end manner for training and inference, optimizing the performance of multimodal representation and question answering while improving time efficiency. It is worth noting that this sparse sampler is a plug-and-play design that can be applied to any MLLMs. Another important characteristic is that it can naturally provide strong interpretability for the question answering. 

In order to simultaneously represent and understand the multimodal content of documents and further improve the ability to process long PDF documents, we construct a training dataset specifically for English and Chinese academic paper PDFs. The academic paper PDF is a kind of typical document that contains rich interleaved text and images, which can intuitively reflect the challenges of our task and the advantages of our model. The dataset contains complete PDF documents, professional academic questions, answers, and evidence sources for the answers, based on multiple construction strategies. We also provide a corresponding bilingual benchmark named **PaperPDF** . 

We train PDF-WuKong on PaperPDF dataset, complemented by general-domain document question-answering datasets. Experimental results substantiate the effectiveness and efficiency of our approach to the task of long multimodal PDF understanding. PDF-WuKong significantly out- 

performs potential open-source models that may be applied to this task. It also surpasses some proprietary products for document understanding on our proposed PaperPDF benchmark. As the number of document pages increases, its accuracy and efficiency will not decrease significantly. It also achieves competitive performance on several documentoriented VQA datasets, especially multi-page benchmarks like DUDE [17]. Besides, for the recent benchmark MMNIAH [18] of long multimodal documents, PDF-WuKong also outperforms other models with fewer parameters. Our model achieves the best performance on multimodal content with a context length of 64K. 

The **main contributions** of this paper are as follows: 

- We introduce a large multimodal model for long PDF understanding with end-to-end sparse sampling, achieving accurate and efficient PDF question answering. 

- We propose a bilingual PDF multimodal question answering dataset ( **PaperPDF** ) with 1 _._ 1 _M_ QA pairs for training and 10 _k_ QA pairs for evaluation. 

- Our model significantly surpasses existing open-source models and proprietary products (by an average of 8 _._ 6% on F1) on long multimodal PDF understanding. 

## **2. Related Works** 

## **2.1. Document Understanding Datasets** 

Early datasets focused on NLP tasks like summarization [27] and QA [28] of plain text, while visual document datasets targeted text perception tasks such as Document Layout Analysis (DLA) [29–31] and Key Information Extraction (KIE) [32–34]. Recent multimodal document QA datasets include DocVQA [35] and OCRVQA [36] for single-page documents, ChartQA [37] and ChartX [38] focusing on visual reasoning in charts. Datasets like ArXivQA [39] and InfoVQA [40] enhance MLLMs’ abilities on academic and infographic documents. However, these datasets are limited to single-page tasks, and current MLLMs [11, 41] perform well on them. 

Multi-page QA datasets like MP-DocVQA [14], DUDE [17], DocGenome [42], and MM-NIAH [18] require understanding content relationships via multi-hop reasoning. Yet, answers in these datasets lack evidence and reliable interpretability, especially for questions needing multiple pieces of evidence from long documents. 

## **2.2. Document Understanding Methods** 

Existing methods focus on plain text or limited document images. Text-based approaches convert documents into plain text using OCR and then employ long-context mechanisms like sparse attention [4], memory networks [5], or position interpolation [3]. Retrieval-augmented generation methods [6, 9, 19] also handle long texts effectively. These approaches struggle with fine-grained visual understanding. 

2 

|**Input modality**|**Type**|**Number of tokens**|**Models**|
|---|---|---|---|
|Plain text|Long-context|Linear increase|LongLoRA[4],LongLLaMA[5],YaRN[3]|
||RAG|w/o Linear increase|Graph RAG[9],DISC-LawLLM[6],RAPTOR[19]|
|Pure vision|Single-page|w/o Linear increase|UniDoc [20], DocOwl [21], Vary [12], UReader [22], TextMonkey [10],<br>LLaVA-NeXT[23],XC2-4KHD[24],InternVL-V1.5[11]|
||Multi-page|Linear increase|Hi-VT5[14],GRAM[16],Fox[13],DocOwl2[25],CREAM[26]|
|Text and images|Unlimited-page|w/o Linear increase|PDF-WuKong**(Ours)**|



Table 1. Comparison of various models for processing multi-page long documents. 

Another solution, visual document understanding, treats each page as an image. MLLMs like UniDoc [20], mPLUGDocOwl [21], and Vary [12] perform OCR-free understanding. Models such as UReader [22] and TextMonkey [10] divide high-resolution pages into patches. InternLM-XC24KHD [24] and InternVL-V1.5 [11] introduce a dynamic resolution mechanism with automatic patch configuration. However, reliance on high resolution increases token counts and isn’t scalable to multi-page documents. 

For multi-page documents, models like Hi-VT5 [14], GRAM [16], Fox [13], CREAM [26] and mPLUGDocOwl2 [25] encode pages separately and perform pagelevel interactions. More pages generate more visual tokens, increasing resource consumption and inefficiency for longer documents. Thus, we propose parsing documents into interleaved text and images, followed by sparse sampling in an end-to-end manner. Tab. 1 summarizes these methods. 

## **3. Methodology** 

## **3.1. Overview** 

Our pipeline consists of three components: a document parser, a sparse sampler, and a large language model, as shown in Fig. 2. The document parsing stage converts input PDFs into machine-readable content with interleaved text and images. The sparse sampler then encodes and caches embeddings for text blocks and images separately. When receiving a user query, it retrieves the most relevant content through similarity matching. Finally, the query and sampled tokens are fed into the LLM for answer generation. The detailed procedure is outlined in Algorithm 1 in the appendix. 

## **3.2. Document Parsing** 

Given a PDF document _D_ , the goal of document parsing is to convert it into some machine-readable text blocks _{T_ 1 _, T_ 2 _, . . . , Tn}_ and diagrams _{I_ 1 _, I_ 2 _, . . . , Im}_ according to the reading order and layout structure. By default, text blocks are organized into paragraphs, and all figures and tables are saved as images. These text and images are finally reorganized into an XML file in reading order. This process can be completed using existing open-source PDF parsing 

tools. During inference, we directly input the parsed structured full data into the subsequent stage of PDF-WuKong. 

## **3.3. Sparse Sampling** 

For a lengthy multi-page document, if it is directly input into the LLM, there will be two problems. The first is the problem of computing efficiency. The consumption of computing resources will increase dramatically. The second is the problem of inaccurate attention. Key information related to the user query is easily submerged by a large amount of irrelevant content. It is difficult for the model to accurately locate and extract important information in a huge token sequence. Therefore, sparse sampling is essential for efficiently handling lengthy multi-page documents by identifying and extracting the most relevant text chunks or diagrams based on their similarity to the user query. 

During the training, for the parsed _n_ text chunks _{T_ 1 _, T_ 2 _, . . . , Tn}_ , _m_ images _{I_ 1 _, I_ 2 _, . . . , Im}_ , and the input user query _q_ , we first extract the positive samples and the negative samples for the query. Our PaperPDF dataset has provided corresponding positive single-evidence or multievidence samples for each query-answer pair (detailed in Sec. 4). We randomly select two text blocks and two images from the remaining text blocks and images as negative samples. Then, we use a text encoder _En_ ~~_T_~~ to obtain the text embeddings _eTP , eTN_ and the query embedding _eq_ . An image encoder _En_ ~~_I_~~ is utilized to output the image features _eIP , eIN_ , which is shared with MLLM. 

Given the embeddings of the user query _eq_ , the positive samples _EP_ = _{eTP , eIP }_ and negative samples _EN_ = _{eTN , eIN }_ , we employ a contrastive learning approach to align the text and image features with the query. The goal is to enable the model to capture the document content that is most relevant to the query. The contrastive learning loss is: 

**==> picture [228 x 45] intentionally omitted <==**

where sim( _eq, ei_ ) and sim( _eq, ej_ ) represent the similarity between the query and the positive/negative samples. _τ_ is 

3 

**==> picture [386 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Text Blocks<br>Text Embed.<br>Text<br>Encoder Positive<br>4 —B Query Embed. bOL Text<br>Contrastive<br>— —_ ParserPDF  :|f Query =. o- ie Learning Oa Positive Image Language Large<br>Model<br>Image Embed. Query<br>Image<br>Instruction<br>Encoder<br>Long PDF<br>Document ! |B ! MLLM with End-to-End Sparse Sampling Answer<br>ee ee eeI IA Images M o eo<br>**----- End of picture text -----**<br>


Figure 2. The overall structure of PDF-WuKong consists of a document parser, a sparse sampler and a large language model. 

the temperature parameter that controls the scale of the similarity scores. _P_ is the number of positive samples. By maximizing this probability, the model encourages the representations of the query and positive samples to be closer while pushing the representations of the query and negative samples apart. It is worth noting that this sparse sampler is a plug-and-play design that can be applied to any MLLMs. 

During the inference, we pre-encode all text blocks and images and cache all candidate embeddings. When the user inputs a query, we calculate the similarity between query embedding and cached text/image embeddings. Then the model automatically selects the top- _k_ relevant text blocks and images as evidence to respond to this query. Therefore, this process samples out sparse document content, greatly reducing the computational burden of the subsequent LLM and alleviating the problem of attention shift when facing ultra-long sequences. Moreover, the multimodal embedding cache further optimizes inference time. The pseudocodes for training and inference are shown in the appendix. 

## **3.4. Answer Generation** 

At this stage, the large language model only receives the document content that is most relevant to the query and discards a lot of redundant information, so it can generate more accurate answers with higher efficiency. Specifically, we input the sampled top- _k_ evidence, the user query, and the task instruction into the LLM, and let it generate an answer based on the provided query and evidence. 

Considering that MLLM needs to encode images first for multimodal understanding, we directly input the image tokens obtained from the sparse sampler into the LLM, to save one image encoding process. Thus, the sparse sampler shares the same vision encoder with the MLLM. They can be integrated and trained in an end-to-end manner. 

During the training, we input the positive text _TP_ and the positive image tokens _eIP_ into the LLM. Besides, the query and instruction are also input into the LLM. Then, we calculate the cross-entropy loss _L_ QA between the output an- 

swer _a_ and the ground truth. Finally, the total optimization objective is: 

**==> picture [160 x 11] intentionally omitted <==**

PDF-WuKong is optimized end-to-end by these two loss functions for effective multimodal alignment and QA. 

## **4. PaperPDF Dataset** 

## **4.1. Overview** 

Our motivation for creating PaperPDF is threefold: (1) In long document QA contexts, answers often derive from specific segments, with other content acting as noise and complicating MLLM reasoning; (2) Existing datasets are either limited to single-page documents or lack fine-grained evidence ground truth, hampering the training of our sparse sampler; (3) There is currently a lack of a bilingual multimodal PDF QA dataset. Therefore, we introduce a method for automatically generating question-answer pairs from long documents and present PaperPDF, a dataset designed for both training and evaluation. 

## **4.2. Dataset construction strategy** 

The PaperPDF dataset is constructed through a four-step process as shown in Fig. 3: document parsing, evidence extraction, QA generation, and data filtering. First, 100 _k_ PDFs are parsed to extract text and image chunks. Then we extract some evidence from each PDF according to our specific rules. Given the evidence and the generation prompt as the input, we use Gemini Pro [43] to generate questions and answers for the training set due to its free and rapid accessibility. GPT-4V [44] is used for the test set to ensure high evaluation quality. After obtaining the preliminary dataset, we conduct automatic filtering and remove abnormal data. For the test data, we further perform manual filtering from multiple aspects to ensure its quality. According to different evidence extraction rules, the triplets in PaperPDF can be divided into two categories. 

4 

**==> picture [492 x 133] intentionally omitted <==**

**----- Start of picture text -----**<br>
[Para. 1] neural network architecture ... Abstract:  Designing a single  Single-Evidence EvidenceTraining   Remove too-short questionsAutomatic Filtering [Q]: Single-Evidence Q-E-A Triplet  How does the model incorporate<br>[Para. 2] predicting properties of a ... Introduction :  The task of  Training   Remove too-long answersRemove duplicate questionsRemove outputs with very low  handcrafted features and word embeddings? [E]: [A]:  features and word embeddings by first {The model incorporates handcrafted  Para  ;  Image }<br>… [Para. 1]Multi-Evidence[Image 1] QA  information entropy… learning word clusters from external tweets in an unsupervised manner. ...<br>Generation<br>[Image 1] Prompt Manual Filtering Multi-Evidence Q-E-A Triplet<br> Whether the answer is correct [Q]:  How does the proposed COP algorithm<br>i fe [Para. 2] [Image 2] Testing QA  Questions, evidence, and answers are interrelatedIs the question meaningful? ———— address the limitations of existing pruning methods, and …  [E]: [A]:  {The COP algorithm addresses the  Para. 1 ;  Image 1 ;  Para. 2 ;  Image 2 }<br>[Image 2]  Remove sensitive information limitations of existing pruning methods by<br>fee … [Para. 3]… [Image 3]  5) EvidenceTesting   Remove the typos… considering the computational cost of pruning different layers of a network .....<br>Document Parsing Evidence Extraction QA Generation Data Filtering Q-E-A Triplets<br>**----- End of picture text -----**<br>


Figure 3. The construction process of PaperPDF based on single evidence and multiple evidence. 

**Single-evidence Q-E-A triplets.** The questions in these triplets can be answered based on a single text chunk or a diagram. Therefore, the evidence can be categorized into _Text-only_ and _Image-only_ . These triplets enable PDFWuKong to initially acquire the capabilities of sparse sampling and long multimodal document understanding. 

**Multi-evidence Q-E-A triplets.** These triplets require reasoning across multiple chunks, involving combinations of text and images. The types include _Image-text_ (derived from a text chunk and associated images), _Section_ (generated from all chunks in a section), and _Cross-paragraph_ (involving related paragraphs across a document). For the third type, semantic summarizations for each paragraph are conducted first, followed by a selection of several related text chunks for QA generation. These multi-evidence triplets enhance our model’s multi-hop reasoning capability. 

In total, we obtained 1 _._ 1 _M_ bilingual training data and 10 _k_ testing data. The dataset statistics are shown in Tab. 2. The appendix contains more data statistics. 

|**Category**<br>~~rr~~|**Category**<br>~~rr~~|**Train (En/Zh)**<br>~~rr~~<br>~~a~~|**Test (En/Zh)**<br>~~rr~~|
|---|---|---|---|
|_S_ingle<br>Text-only<br>249k/12k<br>2939/296<br>Image-only<br>21K/40k<br>212/1018<br>~~rr~~<br>~~a~~||||
|_M_ulti|Image-text<br>Section<br>Cross-paragraph|250k/53k<br>499k/7k<br>1.2k/0|2566/2150<br>255/394<br>118/0|



Table 2. The statistics of PaperPDF in English and Chinese. 

## **5. Experiments** 

## **5.1. Implementation Details** 

The LLM and the vision encoder are initialized from IXC2VL-4KHD [24] and the maximum number of dynamic tiles is set as 16. The text encoder is initialized from 

BGE-M3 [45]. We train the model by leveraging several document datasets including PaperPDF, DocVQA [35], ChartQA [37], InfoVQA [40], MPDocVQA [46], and DUDE [17]. Before both training and testing, PaperPDF is parsed using Grobid [47] and MinerU [48], while the other datasets are processed following their default instructions. The training is conducted for one epoch using 128 Ascend 910B NPUs with a learning rate of 4e-5. We set the top 5 sampling results as input to the LLM. 

For the model evaluations on the PaperPDF dataset, we use three objective metrics ANLS, F1, and Rouge to report the quantitative results on the full test set. To evaluate the semantic correctness of the answer, we introduce GPT-Acc to determine whether the output is correct. Considering the expensive cost of GPT-4 evaluation, we randomly selected two subsets with 50 English PDFs and 30 Chinese PDFs for GPT-Acc calculation. They contain 488 and 317 QA samples, respectively. 

## **5.2. Long PDF Understanding** 

To assess the effectiveness of our model in understanding long PDF documents, we conduct comprehensive experiments comparing it with both open-source models and commercial products on the PaperPDF dataset. Due to limited capabilities of traditional document understanding models [16, 46, 54–56] and the huge resource costs in advanced MLLMs [11, 24, 49], achieving deep understanding of lengthy PDF documents remains a highly challenging task. To handle this task, we explore three approaches to input the PDF documents into these MLLMs: pure text content, page images, and parsed interleaved text-image content. For some baselines with plain text input modality, we also report results based on retrieval enhancement techniques [45]. The experimental results are reported in Tab. 3 and Tab. 4. 

Several key conclusions can be drawn from the results. Firstly, inputting parsed interleaved text-image content generally outperforms multiple page images. The main reason 

5 

|**Model**|**#Param**||**English**|**English**|||**Chinese**|**Chinese**||**Token**|
|---|---|---|---|---|---|---|---|---|---|---|
|||**ANLS**|**F1**|**Rouge **|**GPT-Acc **|**ANLS**|**F1**|**Rouge **|**GPT-Acc**||
||||_Plain Text_||_Solution_||||||
|IXC2-VL [49]|8B|27.4|30.8|32.8|37.6|21.1|28.5|28.7|30.8|4644|
|IXC2-VL-RAG [49]|8.5B|32.4|34.0|32.4|48.4|23.3|29.8|32.3|38.1|623|
|InternVL2 [11]|8B|19.5|28.0|27.6|37.5|24.4|28.2|27.7|30.8|4051|
|InternVL2-RAG [11]|8B|29.8|29.8|28.3|50.2|24.6|27.8|26.8|43.0|**583**|
||||_Pure Vision Solution_||||||||
|Hi-VT5 [46]|0.3B|13.5|3.1|3.7|15.2|-|-|-|-|11589|
|IXC2-VL [49]|8B|27.1|24.8|25.1|20.5|15.9|19.5|22.2|27.4|4712|
|InternVL2 [11]|8B|29.4|33.1|35.0|35.5|20.3|33.9|37.8|37.7|5008|
|||_Parsed Interleaved_|||_Text and Images_||||||
|IXC2-VL [49]|8B|27.8|31.2|32.6|37.7|22.5|29.5|29.2|31.4|6217|
|InternVL2 [11]|8B|33.4|36.2|36.6|54.3|28.5|40.6|42.0|54.7|6220|
|PDF-WuKong (ours)|8.5B|**41.9**|**43.5**|**40.9**|**77.5**|**40.9**|**47.8**|**48.6**|**57.8**|2107|



Table 3. Performance comparison with open-source models for long PDF understanding on PaperPDF. The best results are marked **bold** and the second results are underlined. 

|**Model**|**ANLS **|**F1 **|**Rouge **|**GPT-Acc**|
|---|---|---|---|---|
|Gemini pro [50]|26.6|29.0|29.8|67.9|
|Kimi [51]|28.5|33.6|31.1|74.7|
|ChatGLM [52]|31.2|35.4|32.0|73.5|
|Qwen [53]|36.0|40.3|35.5|**78.1**|
|PDF-WuKong (ours)|**41.8**|**43.2**|**40.7**|77.5|



Table 4. Performance comparison with commercial products (tested in Sep 2024) for long PDF understanding. The results are tested on a subset of 50 English PDFs. Note: These are the products based on the models rather than the models themselves. 

is the limited input resolution and number of tokens that the model can accept. Secondly, when handling tokens of similar scale, inputting parsed interleaved text-image content yields better performance compared to pure text content. It is obvious that diagram information in the PDF plays a crucial role for document comprehension. This also indicates that PaperPDF places rigorous requirements on the visual information within documents. Additionally, we observe that for the InternVL2 model, the approach of parsing interleaved text-image content as input outperforms InternVL2RAG. However, the opposite conclusion is drawn for IXC2VL. We hypothesize that this discrepancy may be due to the max length setting in IXC2-VL, which could cause the model to overlook some critical information. Finally, benefiting from the inclusion of the spare sampler, our proposed PDF-WuKong model not only surpasses the existing state-of-the-art open-source model InternVL2 by approxi- 

mately 7% on both the Chinese and English subsets, but also demonstrates competitive performance comparable to proprietary products. Moreover, due to the integration of the sparse sampler, PDF-WuKong maintains efficiency in inference token cost. 

|**Model**|**# param **|**ANLS**|**F1**|**ROUGE**||
|---|---|---|---|---|---|
|Qwen-VL [57]|9.6B|26.4|19.6|18.3||
|Monkey [58]|9.8B|30.0|24.4|22.3||
|mPLUG-Owl2 [25]|8.2B|19.5|20.3|22.7||
|Emu2-Chat [59]|37B|26.0|24.4|23.4||
|MiniCPM-2.5 [60]|8.5B|31.8|28.2|24.8||
|IXC2-VL [49]|8B|23.4|20.8|21.3||
|IXC2-4KHD [24]|8B|24.5|20.0|18.0||
|CogVLM2 [61]|17B|24.8|27.4|26.3||
|PDF-WuKong (ours)_†_|8.5B|36.6|35.2|31.7||
|PDF-WuKong (ours)_∗_|8.5B|**41.5**|**42.8**|**39.8**||



Table 5. Performance comparison with other DocVLMs for PDF multimodal understanding on the Single-Evidence Subset. _†_ denotes the page image input, aligning with other models in the table; * indicates that our model utilizes parsed content as input. 

To compare with more open-source document MLLMs, considering that most of these models can only handle single-page documents, we construct a subset of the PaperPDF benchmark, only containing test samples with single evidence. Therefore, we provided all models with only one page containing the evidence as their input. The pages are input in the form of images. Our PDF-WuKong can ac- 

6 

cept input in two formats. One is the page image and another is the parsed page. As shown in Tab. 5, our model’s capability on this subset is significantly better than other document models. Moreover, the document parsing-based paradigm used in our approach is superior to the purely visual paradigm. 

## **5.3. Document-oriented VQA** 

To validate the strong capability of our model in other document understanding scenarios, we conduct experiments on several public benchmarks and compare PDF-WuKong with other representative models. First, we evaluate the performance of PDF-WuKong on single-page document datasets [35, 37, 40]. As shown in Tab. 6, our model achieves comparable performance on single-page document understanding. This demonstrates that PDF-WuKong can effectively handle various types of documents and questions, showcasing its versatility in document-oriented visual question-answering tasks. 

In addition, we assess the performance of traditional specialized models and MLLMs on two existing multi-page document QA datasets. The results shown in Tab. 7 indicate that our model’s performance in multi-page document scenarios is comparable to these specialized models and far surpasses the latest document MLLM DocOwl2 [25]. Notably, on complex multi-page document datasets like DUDE [17], PDF-WuKong outperforms GPT-4V [44]. This improvement is attributed to our sparse sampler, which effectively filters out useful information from multi-page documents, enabling the model to focus on relevant content. 

Furthermore, we conduct zero-shot evaluations on a new long multimodal document understanding benchmark MMNIAH [18]. As shown in Tab. 8, our model uses the fewest parameters yet achieves the second-best performance. Although InternVL-V1-5-RAG surpasses PDF-WuKong by 2.8%, it utilizes 36.5 billion more parameters than our model. Moreover, as the context length of the multimodal documents increases, the performance of our model remains stable, while that of other models significantly decreases. At a context length of 64K, PDF-WuKong even achieves the best performance, demonstrating its robustness in handling long-context multimodal inputs. 

## **5.4. Ablation Study** 

To comprehensively evaluate the effectiveness of our contributions, we conduct ablation studies focusing on the impact of the sparse sampler, the dataset, the document length, and sampling strategies. These experiments are based on English PaperPDF for training and evaluation to avoid interference from other factors. 

## **Sparse sampler** 

To assess the effectiveness of the sparse sampler, we compared models trained with and without it. Without the 

|**Model**|**DocVQA **|**ChartQA **|**InfoVQA**|
|---|---|---|---|
|Qwen-VL [57]|65.1|65.7|35.4|
|Monkey [58]|66.5|65.1|36.1|
|Text-Monkey [10]|73.0|66.9|28.6|
|MiniCPM-V-2.5 [60]|84.8|-|-|
|Vary-base [12]|76.3|66.1|-|
|TextHawk [62]|76.4|66.6|50.6|
|IXC2-4KHD-16 [24]|84.9|**80.1**|60.8|
|DocOwl 2 [25]|80.7|70.0|46.4|
|CREAM [26]|79.4|-|53.6|
|PDF-WuKong (ours)|**85.1**|80.0|**61.3**|



Table 6. Performance comparison with other DocVLMs on singlepage document-oriented VQA benchmarks. 

|**Model**|**MP-DocVQA **|**DUDE**|
|---|---|---|
|LayoutLMv3 [54]|55.1|20.3|
|Longformer [55]|55.1|27.1|
|BigBird [56]|58.5|26.3|
|Hi-VT5 [46]|61.8|35.7|
|DocFormerv2 [63]|76.4|48.4|
|GRAM [16]|**83.0**|53.4|
|GPT-4V (2024-06) [44]|-|53.9|
|Idefcs3-8B [64]|67.2|38.7|
|DocOwl2 [25]|69.4|46.7|
|CREAM [26]|65.3|52.5|
|PDF-WuKong (ours)|76.9|**56.1**|



Table 7. Performance comparison with other DocVLMs for multipage document understanding. 

|**Model**|**#param**|**Overall **|**1K **|**4K **|**16K**|**64K**|
|---|---|---|---|---|---|---|
|Emu2-Chat [59]|37B|8.8|38.9|18.2|0.0|0.0|
|VILA1.0-13b [65]|13B|15.7|41.9|33.2|8.6|0.1|
|llava-v1.6-13b [23]|13B|16.9|43.7|34.9|13.6|0.0|
|llava-v1.6-34b [23]|34B|20.6|57.4|45.1|8.2|0.0|
|InternVL1.5 [11]|26B|41.1|**59.5**|**50.1**|41.9|16.6|
|InternVL1.5-RAG [11]|45B|**46.1**|**59.5**|**50.1**|**44.9**|39.3|
|PDF-WuKong (ours)|8.5B|43.3|53.0|43.9|43.0|**42.1**|



Table 8. Performance comparison with other DocVLMs on MMNIAH. The evaluation approach aligns with the benchmark. 

sparse sampler, the MLLM struggled to process long documents with interleaved text and images, resulting in poor performance due to the large amount of irrelevant information. Introducing the sparse sampler significantly improved the model’s accuracy, as evidenced in Tab. 9, by efficiently selecting the most relevant content for each query. Further- 

7 

more, end-to-end joint training of the sparse sampler and the MLLM led to additional performance gains compared to training them separately. This indicates that our end-toend optimization of multimodal representation and question answering can further promote the document understanding ability of MLLM. 

|~~a~~<br>||~~a~~<br>||~~a~~<br>||~~a~~<br>||~~a~~<br>||
|---|---|---|---|---|
|**Sparse Sampler End-to-End**<br>**ANLS**<br>**F1**<br>**ROUGE**<br>✗<br>✗<br>11.1<br>5.1<br>5.0<br>|<br>~~TT~~|||||
|✓|✗|40.3|42.3|39.8|
|✓|✓|**42.6**|**43.6**|**40.2**|



Table 9. Ablation study on the impact of sparse sampler 

## **Dataset** 

We retrain PDF-WuKong on various subsets of our English PaperPDF, and the results are shown in Tab. 10. Increasing the amount of training data leads to consistent improvements in the model’s accuracy, proving that our dataset follows scaling laws. In addition, we verify the effectiveness of our two data construction methods. Under the same data scale, multi-evidence data can enhance the model’s complex reasoning ability. 

|**Dataset**|**ANLS**|**F1**|**ROUGE**|
|---|---|---|---|
|100 k|38.7|40.1|37.5|
|500 k|41.6|43.5|40.8|
|1 M|42.6|43.6|40.2|
|Single (200k)|39.2|40.4|38.1|
|Multi (100k) + Single (100k)|40.0|41.6|38.9|



Table 10. Ablation study on dataset setting 

## **Document length** 

To understand the impact of document length on model performance and efficiency, we divide the test set into subsets based on the number of pages per document. Results in Fig. 4 demonstrate that our model’s performance and token counts remain relatively stable across documents of varying lengths. This stability indicates that the sparse sampler effectively reduces the input size to a reasonable level, regardless of the original document length. In contrast, the baseline MLLM without the sparse sampler is unable to handle long documents effectively. Its performance deteriorates significantly as the document length increases. The number of tokens also increased dramatically, resulting in huge resource consumption. These findings highlight the robustness of our model in processing long documents without sacrificing accuracy or incurring extra computational costs. **Sampling strategy** 

Figure 4. Ablation study of different document length 

We explore the impact of different numbers of text blocks or diagrams selected by the sparse sampler. As shown in Tab. 11, setting a small top _k_ can lead to missing crucial information needed for accurately answering queries, thus reducing performance. Conversely, a larger top _k_ introduces redundant information and increases computational costs without significantly enhancing accuracy. To strike a balance between performance and resource efficiency, we use the top 5 as the default setting. 

|**Sampling chunks ANLS**|**Sampling chunks ANLS**|**F1**|**ROUGE **|**Tokens**|
|---|---|---|---|---|
|Top 1|39.20|38.28|35.26|1186|
|Top 3|42.09|43.06|39.69|1452|
|Top 5|42.59|43.63|40.19|1789|
|Top 10|43.01|44.22|40.67|2386|
|Top 15|43.19|44.57|42.08|2704|
|Top 20|43.42|45.02|42.30|3364|



Table 11. Ablation study of different sampling strategy 

## **6. Conclusion** 

We have presented PDF-WuKong, a novel MLLM that effectively addresses the challenges of understanding long PDF documents containing interleaved text and images. By introducing an end-to-end sparse sampling mechanism, our model efficiently extracts the most relevant paragraphs and diagrams in response to user queries, significantly reducing input token size and making the process independent of document length. We also constructed PaperPDF, a bilingual dataset with 1 _._ 1 _M_ question-answer pairs for training and 10 _k_ pairs for evaluation, specifically tailored for academic PDFs. Experimental results demonstrate that PDFWuKong not only outperforms existing open-source mod- 

8 

els but also surpasses proprietary products by an average of 8.6% in F1 score on the long multimodal PDF understanding task. Our approach maintains high accuracy and efficiency even as document length increases, offering a scalable and interpretable solution for practical applications in document understanding. 

## **References** 

- [1] Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon, Ryan A Rossi, and Franck Dernoncourt. Pdftriage: Question answering over long, structured documents. _arXiv preprint arXiv:2309.08872_ , 2023. 1 

- [2] T. Prem Jacob, Beatriz Lucia Salvador Bizotto, and Mithileysh Sathiyanarayanan. Constructing the chatgpt for pdf files with langchain – ai. In _2024 International Conference on Inventive Computation Technologies (ICICT)_ , pages 835– 839, 2024. 1 

- [3] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 1, 2, 3 

- [4] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. LongloRA: Efficient fine-tuning of long-context large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 2, 3 

- [5] Szymon Tworkowski, Konrad Staniszewski, Mikoł aj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Mił o´s. Focused transformer: Contrastive training for context scaling. In _Advances in Neural Information Processing Systems_ , volume 36, pages 42661–42688, 2023. 1, 2, 3 

- [6] Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, et al. Disc-lawllm: Fine-tuning large language models for intelligent legal services. _arXiv preprint arXiv:2309.11325_ , 2023. 1, 2, 3 

- [7] Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang, Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong Xu, Xiang Bai, Xuanjing Huang, et al. Disc-finllm: A chinese financial large language model based on multiple experts finetuning. _arXiv preprint arXiv:2310.15205_ , 2023. 

- [8] Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries. _arXiv preprint arXiv:2401.15391_ , 2024. 

- [9] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to queryfocused summarization. _arXiv preprint arXiv:2404.16130_ , 2024. 1, 2, 3 

- [10] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. _arXiv preprint arXiv:2403.04473_ , 2024. 2, 3, 7 

- [11] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing 

   - the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 2, 3, 5, 6, 7 

- [12] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language model. In _European Conference on Computer Vision_ , pages 408–424. Springer, 2024. 2, 3, 7 

- [13] Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for finegrained multi-page document understanding. _arXiv preprint arXiv:2405.14295_ , 2024. 2, 3 

- [14] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834, 2023. 2, 3 

- [15] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In _AAAI_ , pages 13636–13645, 2023. 2 

- [16] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. Gram: Global reasoning for multi-page vqa. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15598–15607, 2024. 2, 3, 5, 7 

- [17] Jordy Van Landeghem, Rub`en Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Micka¨el Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19528–19540, 2023. 2, 5, 7, 13 

- [18] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. Needle in a multimodal haystack. _arXiv preprint arXiv:2406.07230_ , 2024. 2, 7 

- [19] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In _The Twelfth International Conference on Learning Representations_ , 2024. 2, 3 

- [20] Hao Feng, Zijian Wang, Jingqun Tang, Jinghui Lu, Wengang Zhou, Houqiang Li, and Can Huang. Unidoc: A universal large multimodal model for simultaneous text detection, recognition, spotting and understanding, 2023. 3 

- [21] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. _arXiv preprint arXiv:2307.02499_ , 2023. 3 

- [22] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. _arXiv preprint arXiv:2310.05126_ , 2023. 3 

- [23] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Im- 

9 

proved reasoning, ocr, and world knowledge, January 2024. 3, 7 

- [24] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv preprint arXiv:2404.06512_ , 2024. 3, 5, 6, 7 

- [25] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplugdocowl2: High-resolution compressing for ocr-free multipage document understanding, 2024. 3, 6, 7 

- [26] Jinxu Zhang, Yongqi Yu, and Yu Zhang. Cream: Coarse-tofine retrieval and multi-modal efficient tuning for document vqa. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , pages 925–934, 2024. 3, 7 

- [27] Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 1419–1436, 2021. 2 

- [28] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 4599–4610, 2021. 2 

- [29] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In _2019 International conference on document analysis and recognition (ICDAR)_ , pages 1015–1022. IEEE, 2019. 2 

- [30] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. _arXiv preprint arXiv:2006.01038_ , 2020. 

- [31] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. Doclaynet: a large human-annotated dataset for document-layout segmentation. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 3743–3751, 2022. 2 

- [32] Stˇep´an[ˇ] Simsa,[ˇ] Milan Sulc,[ˇ] Michal Uˇriˇc´aˇr, Yash Patel, Ahmed Hamdi, Matˇej Koci´an, Maty´aˇs Skalick`y, Jiˇr´ı Matas, Antoine Doucet, Micka¨el Coustaty, et al. Docile benchmark for document information localization and extraction. pages 147–166, 2023. 2 

- [33] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. Cord: A consolidated receipt dataset for post-ocr parsing. In _Workshop on Document Intelligence at NeurIPS_ , 2019. 

- [34] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In _ICDAR_ , pages 1516–1520, 2019. 2 

- [35] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In _WACV_ , pages 2200–2209, 2021. 2, 5, 7 

- [36] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _ICDAR_ , pages 947–952, 2019. 2 

- [37] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 2, 5, 7 

- [38] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. _arXiv preprint arXiv:2402.12185_ , 2024. 2 

- [39] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. _arXiv preprint arXiv:2403.00231_ , 2024. 2 

- [40] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. In _WACV_ , pages 1697–1706, 2022. 2, 5, 7 

- [41] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 2 

- [42] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open largescale scientific document benchmark for training and testing multi-modal large language models. _arXiv preprint arXiv:2406.11633_ , 2024. 2 

- [43] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 4 

- [44] OpenAI. Gpt-4v(ision) system card. https://openai. com/contributions/gpt-4v, 2023. 4, 7 

- [45] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multilingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. _arXiv preprint arXiv:2402.03216_ , 2024. 5 

- [46] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834, 2023. 5, 6, 7, 13 

- [47] Grobid. https://github.com/kermitt2/grobid, 2008–2024. 5 

- [48] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian 

10 

   - Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction, 2024. 5 

- [49] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form textimage composition and comprehension in vision-language large model. _arXiv preprint arXiv:2401.16420_ , 2024. 5, 6 

- [50] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 6 

- [51] Moonshot AI. Kimi. https://kimi.moonshot.cn, 2023. 6 

- [52] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. _arXiv preprint arXiv:2406.12793_ , 2024. 6 

- [53] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 6 

- [54] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4083–4091, 2022. 5, 7 

- [55] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_ , 2020. 7 

   - [60] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. _arXiv preprint arXiv:2404.06395_ , 2024. 6, 7 

   - [61] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. _arXiv preprint arXiv:2408.16500_ , 2024. 6 

   - [62] Ya-Qi Yu, Minghui Liao, Jihao Wu, Yongxin Liao, Xiaoyu Zheng, and Wei Zeng. Texthawk: Exploring efficient finegrained perception of multimodal large language models. _arXiv preprint arXiv:2404.09204_ , 2024. 7 

   - [63] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R Manmatha. Docformerv2: Local features for document understanding. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 709– 718, 2024. 7 

   - [64] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. _Advances in Neural Information Processing Systems_ , 36, 2024. 7 

   - [65] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 26689–26699, 2024. 7 

   - [66] Qwen. https://tongyi.aliyun.com/qianwen/. 12 

   - [67] ChatGLM. https://chatglm.cn/. 12 

   - [68] Kimi. https://kimi.moonshot.cn/. 12 

   - [69] Gemini-Pro. https://gemini.google.com/. 12 

- [56] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. _Advances in neural information processing systems_ , 33:17283–17297, 2020. 5, 7 

- [57] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_ , 2023. 6, 7 

- [58] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint arXiv:2311.06607_ , 2023. 6, 7 

- [59] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14398–14409, 2024. 6, 7 

11 

## **A. Algorithm** 

Algorithm 1 shows the detailed inference process of PDFWuKong. The training pipeline is shown in Algorithm 2. Our PDF-WuKong can achieve efficient and accurate understanding of long PDFs with end-to-end sparse sampling. 

## **Algorithm 1** Inference pipeline for PDF-WuKong 

- 1: **Input:** PDF document _D_ , user query _q_ 

- 2: **Output:** Generated answer _a_ 

- 3: **Initialize:** Text encoder En ~~T~~ , image encoder En ~~I~~ , large language model LLM 

- 4: **Stage 1: Document Parsing** 

- 5: Parse the input document _D_ into text blocks and images: 

**==> picture [196 x 11] intentionally omitted <==**

- 6: **Stage 2: Sparse Sampling** 

- 7: Encode all text blocks and images and **cache** all candidate vector embeddings: 

**==> picture [222 x 11] intentionally omitted <==**

**==> picture [210 x 12] intentionally omitted <==**

- 8: Encode the user query _q_ : 

**==> picture [59 x 12] intentionally omitted <==**

- 9: Calculate the similarity between query embedding _eq_ and cached text/image embeddings _{ET , EI }_ : 

pro [69]. The success stems from its ability to effectively sample key evidence relevant to the question and its strong multimodal understanding capabilities. This observation further validates the effectiveness of our method in enhancing the long PDF understanding capabilities of MLLMs. 

## **Algorithm 2** Training pipeline for PDF-WuKong 

- 1: **Input:** PDF document _D_ , user query _q_ , ground truth answer _gt_ 

- 2: **Output:** Final loss function _L_ total 

- 3: **Initialize:** Text encoder En ~~T~~ , image encoder En ~~I~~ , large language model LLM 

- 4: **Stage 1: Data Preparing** 

- 5: Text blocks and images: 

**==> picture [196 x 11] intentionally omitted <==**

## 6: **Stage 2: Multimodal encoding** 

- 7: Encode the user query, positive and negative samples: 

**==> picture [161 x 51] intentionally omitted <==**

- 8: Calculate the contrastive learning loss: 

**==> picture [133 x 11] intentionally omitted <==**

## 9: **Stage 3: Output prediction of MLLM** 

- 10: Input the query, the positive text, and the positive image tokens from the **shared image encoder** En ~~I~~ : 

**==> picture [158 x 11] intentionally omitted <==**

**==> picture [156 x 12] intentionally omitted <==**

- 10: Select the top- _k_ relevant text blocks and images: 

**==> picture [120 x 11] intentionally omitted <==**

- 11: **Stage 3: Answer Generation** 

- 12: Input the query _q_ and the selected tokens into the LLM: 

**==> picture [102 x 11] intentionally omitted <==**

- 13: **Return** the generated answer _a_ . 

## **B. Visualization Results** 

## **B.1. Qualitative Comparison with Other Products** 

The qualitative comparison between PDF-WuKong and other proprietary products on the long PDF understanding task is presented in Fig. 5. PDF-WuKong achieves significant performance advantages over others such as Qwen [66], ChatGLM [67], Kimi [68], and Gemini- 

**==> picture [91 x 11] intentionally omitted <==**

- 11: Calculate the cross-entropy loss: 

**==> picture [44 x 11] intentionally omitted <==**

- 12: **Stage 4: Optimize model in an end-to-end manner** 

- 13: Update model parameters according to the joint loss: 

**==> picture [79 x 11] intentionally omitted <==**

- 14: **Return** the final loss _L_ total. 

## **B.2. Visualization Results on Chinese Documents** 

In addition to the outstanding performance on English documents, PDF-WuKong also demonstrates remarkable capabilities on Chinese documents. Fig. 6 shows its qualitative performance on the Chinese long PDF document understanding task. It effectively samples relevant evidence within the PDF and demonstrates the ability to derive accurate answers based on the sampled evidence. 

12 

**==> picture [438 x 312] intentionally omitted <==**

**----- Start of picture text -----**<br>
Qwen ChatGLM Kimi Gemini PDF-WuKong<br>B 1211.6807v2.pdfPDF H B 1411.0860v1.pdfPDF<br>What field of Computer Science listed in  According to Figure 6, how does the approximation error measured<br>Table 2 has the lowest conductance value? by Frobenius norm change for CUR+ and CUR-E as α increases in<br>the Enron dataset for r = 50?<br>List often ld of Computer Science end ter umber of papers and conductance : ca - of Je<br>23 || DatabasesBate Secs Mastiand Theory S1011261 | PORT0.3429 ': - Ore eae Dexter r= 20 Form Ads r = 20 Gisetter = 20<br>5 | Hardware and Areiecare ra0r | o.tme2 ' F2d | EJ EE] ER).<br>— 6 | Human Computer Interaction 16st | 0.4527 H ’ . sf os ef<br>cord 8| Networking 1561 | 0.3086 H — St 4 — =<br>VY a1019 ||| Infomation ProseammingOperating Systems Retin 2580so722 | || 09088 0.3178 0.3736 H:'H =—VV rcEnron- r = a50. foeOeDexter r = 50 FarmrraAds r = 50. Gisette: r = 50<br>As α increases, the approximation error for CUR+ decreases while<br>Information Retrieval.<br>for CUR-E, it unexpectedly increases.<br>According to Figure 6, the approximation error measured by<br>Information Retrieval.<br>Frobenius norm decreases for CUR+ and increases for CUR-E as α<br>increases in the Enron dataset for r = 50.<br>The field of Computer Science with the<br>lowest conductance value listed in Table 2 is  The provided information does not include the actual content of the<br>Programming with a value of 0.3178. scientific article or Figure 6. To answer….<br>See the provided document for Table 2. The  For the Enron dataset with r = 50, as α increases, the approximation<br>field of Operating Systems (OS) has the  error measured by Frobenius norm for both CUR+ and CUR-E<br>lowest conductance value of 0.011 decreases<br>POFIN | [poe IS " The approximation error of CUR+ remains almost unchanged, while<br>Artificial Intelligence. the error for CUR-E increases significantly.<br>Answer :  Artificial  Intelligence  has  the  lowest  Answer : For r = 50 in the Enron dataset, the approximation error for CUR+<br>conductance value of 0.1568. remains constant while it increases for CUR-E as α increases.<br>**----- End of picture text -----**<br>


Figure 5. Qualitative comparison between PDF-WuKong with other proprietary products. The red box indicates the evidence that the correct answer depends on. 

## **C. More Dataset Details** 

The Q-E-A triplets in the PaperPDF dataset are derived from approximately 60k English and 10k Chinese documents, encompassing nearly 70 disciplines such as computer science, engineering, and materials science. 

To comprehensively illustrate the characteristics of the PaperPDF dataset, we conduct a detailed statistical analysis. The key statistics of the source documents in PaperPDF are presented in Tab. 12. Compared to previous multi-page document datasets, PaperPDF not only includes a significantly larger number of documents but also features more pages and OCR tokens. More importantly, PaperPDF encompasses both Chinese and English documents with interleaved text and images. Our dataset provides an important innovation driver and a comprehensive benchmark for the development of the multimodal long PDF document understanding task. 

For the five types of data ( _Text-only, Image-only, Imagetext, Section, Cross-paragraph_ ) in single evidence and multiple evidence, Fig. 7 - Fig. 11 show their prompt engineering and a corresponding example, respectively. 

|**MPdataset**|**Document **|**Language **|**Page **|**Image **|**Token**|
|---|---|---|---|---|---|
|DUDE [17]|5k|En|6|-|1831|
|MP-Docvqa [46]|6k|En|8|-|2026|
|PaperPDF (En)|60k|En|25|12|11371|
|PaperPDF (Zh)|10k|Zh|11|7|3413|
|PaperPDF|70k|En+Zh|23|11|10234|



Table 12. Statistics of the source documents, where **Page** , **Image** , and **Token** represent the average number of pages per document, the average number of images in the document, and the average number of OCR tokens per document, respectively. 

## **D. Limitation** 

For our proposed dataset, current documents are mainly limited to academic papers, so the layout format and subject matter of the documents are relatively simple. We will expand the dataset with more diverse documents. Besides, our model is not specifically designed for some global queries, which will become a key research problem in the future. 

13 

**==> picture [450 x 590] intentionally omitted <==**

**----- Start of picture text -----**<br>
z 基于超声波避障的仓… a’ PER acu 基于超声波避障的仓…<br>PDF PDF<br>® 图处理之后的关键步骤是什么？ 5 5 中描述的运动软件设计流程中，数据 |gS| =| Bi.__axenenEe—,——_j=eansi}ESE 根据图程序是如何在不同的模块之间切换的？ S 7 中的 LabVIEW 框图，<br>Por LS Stee<br>am 电机控制。 & of Jama (*) SeenON >ee —snrjaxeans nN<br>图 7 中，程序通过条件<br>标志切换模块。<br>aoe oF<br>Answer<br>fo 1. 电机控制。 Answer E L<br>2. 图5显示数据处理后进入“电机控制”环节... 1.通过条件标志切换不同模块。<br>用于控制电机，实现循迹与避障。 2.图7显示的LabVIEW框图中，程序通过条件标志切换不同模块。程序首先进入初始化模块…。然后进入<br>等待模块…。当触发事件发生后，程序进入数据处理模块…。最后，当停止标志（停止开关）触发时，程<br>Evidence 序进入退出模块，关闭所有指示灯、清除显示并退出 While 循环。“<br>1.底层运动系统的软件设计如图5所示，控制核<br>Hie 心是STM32单片机…进入程序后…信息采集完成 Evidence<br>后进行数据处理，控制电机相应转动… 1.设计的LabVIEW程序框图如图7所示，系统分为…，通过不同的条件标志来切换不同模块。程序开始…<br>2.图5. 底层运动软件设计流程图.jpg 然后进入等待模式…当触发停止标志时进入到退出模块…。<br>ES RRRiki REA 2.图7上位机程序设计.jpg<br>铜钨异种金属焊接工艺研究<br>县级媒体主持人如何融入媒体时代PDF . , | 8 PDF<br>根据图 4 的 SEM 扫描结果及 EDS 分析<br>谱图，焊缝特征点 1 的主要元素成<br>作者认为应如何面对变化的时代和社会？ 分是什么？<br>® ®<br>守好主阵地，进入新领域。 PDFS — =} tepetpertneerieberererrerr errr pr 主要元素成分是 Cu 和 Ag 。<br>3} Bd SHERI SEMABER REDS HTD &<br>Answer<br>Answer<br>1.2.社会、大众同步。这体现了应对变化的综合策略，使传统力量和创新能力并行不悖。保持主阵地，同时进入新领域。作者建议既要守住现有的主要阵地，同时也要进入新的领域，这样才能与时代、 1. Cu2. （银）图4和的……AgSEM。两个明显的峰值分别标记为扫描结果及EDS分析谱图清楚地显示，焊缝特征点Cu和Ag……。 1的元素成分主要为Cu（铜）和Ag<br>Evidence<br>Evidence<br>1.大众所抛弃。最好的办法就是既要守好主阵地，也要进入新领域，与时代同步， 1. 2.焊缝特征点焊缝特征点11 SEM组织电子能谱仪分析扫描结果及EDS……EDS分析谱图元素分析结果表明，特征点.jpg 1的元素组成分别为61.12%的Cu、的元素组成分别为61.12%的Cu、61.12%的Cu、的Cu、Cu、、<br>与社会同步，与大众同步。 38.88%的Ag……分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。的Ag……分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。Ag……分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。与Cu发生融合形成的共晶产物……是焊缝主要成分。Cu发生融合形成的共晶产物……是焊缝主要成分。发生融合形成的共晶产物……是焊缝主要成分。……是焊缝主要成分。是焊缝主要成分。<br>4000 bw thou 铜钨异种金属焊接工艺研究 200 480,850 | | 铜钨异种金属焊接工艺研究<br>PDF PDF<br>a 钨 - 铜二元合金在 1084°C 时 ga 200) P9) 300] 该焊接工艺的温度在什么时<br>ea)of S| () 的相组成是什么？ “=100 | of : () 间段内保持在最高点？<br>BioSwou/% ae W+Cu. PDF DS af oo 36 7 [ros 1a fs 216 252 288 324 108 分钟至 144 分钟。 Por LS<br>W-cua 7 [ey 2. seTSSHm [ey<br>Answer<br>1. W和Cu. Answer<br>2.图1显示了W-Cu二元相图。在1084 ° C时，相区标记为“W+Cu”。这意味着在这个 1. 108到144分钟。<br>温度下，钨和铜是以各自的固相形式存在的。这一点可以通过浏览图中1084 ° C线 2. 从图2可以看出，温度在108分钟时达到850℃，并持续到144分钟，此后温度开始下降。因此，温<br>下的相区标记确认。 度在108到144分钟内保持在最高点。<br>Evidence Evidence<br>1. 图1-二元相图.jpg 1. 图2焊接工艺参数图.jpg<br>4 SCM RIBIERE ee iam |<br>煤矿井下支架快速安装…<br>REE FA BE PU) PUIK ABO) tctaaih bales antes \ °<br>ZIT (grem3) (Wem'K1) (10°C) fE/MPa uA cA — & (> PDF<br>液压支架从平板车推上平<br>Ag?2Cu26Ti_ 铜钨异种金属焊接工艺研究 780 10.00 352 178 _ | 250~360 fem | all ® 台后，如何进行找正调平？<br>PDF<br>使用牵引千斤进行找正调<br>该材料的抗拉强度的范围是什么？ ) 平。 Gi<br>® [3 A \ Por LS I<br>250~360  Mpa 。<br>[sy Answer A2 RELRARESTOR<br>1. 由牵引千斤进行找正调平。<br>2. 根据图2的描述，当液压支架从平板车推上平台后，是由牵引千斤进行找正调平的。这是在详细<br>Answer 的安装过程中说明的步骤。<br>1. 250~360 MPa。<br>2. 根据表4，抗拉强度的数值范围一栏，明确指出了该材料Ag72Cu26Ti的抗拉强度范 Evidence<br>围为250~360 MPa。因此，该材料的抗拉强度在250到360 MPa之间。 1. 图2 液压支架安装平台实物图.jpg<br>2. 如图2所示，井下支架快速安装平台主要由两部牵引千斤和推移千斤组成……现场安装时，首先<br>Evidence 将运架平板车与平台对接……由牵引千斤将液压支架从平板车推上平台……牵引至无极绳绞车后运<br>1. 表4 填充材料物理性能.jpg 输……通过推移千斤将支架推下平台……完成进架操作。<br>**----- End of picture text -----**<br>


- 1的元素组成分别为61.12%的Cu、的元素组成分别为61.12%的Cu、61.12%的Cu、的Cu、Cu、、 

- 38.88%的Ag……分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。的Ag……分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。Ag……分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。分析此处为Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。Ag与Cu发生融合形成的共晶产物……是焊缝主要成分。与Cu发生融合形成的共晶产物……是焊缝主要成分。Cu发生融合形成的共晶产物……是焊缝主要成分。发生融合形成的共晶产物……是焊缝主要成分。……是焊缝主要成分。是焊缝主要成分。 

Figure 6. Examples of PDF-WuKong on Chinese documents. The red box indicates the evidence that the correct answer depends on. 

14 

## **Text-only Question Generation Prompt** 

## **Task Definition:** 

Create 2 academic questions from a given research paper paragraph. 

**Requirements:** Analyze the paragraph thoroughly,understanding its content including the study’s objectives, ethods, results,and conclusions. Focus on the paragraph,not the entire paper. If the paragraph lacks valid information,return ‘quit’. You should use English. Develop 2 questions that: 

- Are no more than 30 words. 

- Incorporate knowledge from the paragraph. 

- Are answered by text instead of one of the multiple choices. 

• Are elicit detailed responses supported by the text. **Expected Output:** (Return ‘quit’ directly if the paragraph lacks valid information.) [Q1]: question1 here [Q2]: question2 here 

## **Text-only Answer Generation Prompt** 

## **Task Definition:** 

Answer a question based on the material given. 

**Requirements:** The answers should: 

- Be comprehensive and cover all relevant aspects. 

- Accurately reflect the paragraph’s information and insights. 

You should think step by step and give you answer in the end of your generation like: [thinking procedure]: [A1/A2] **Expected Output:** 

[THINKING PROCEDURE]: ... 

[A1]: answer1 here, no more than 20 words. 

[THINKING PROCEDURE]: ... 

[A2]: answer2 here, no more than one sentence. 

## **Text-only Data Example** 

**Query:** What is the impact of using the same dataset for optimizing and measuring the performance of a model? 

**Text:** Here, again, the unfair advantage of optimizing (selecting the models for the ensemble) and measuring performance on the same dataset appears.The advantage is small but systematic for the test split of ISIC (Fig. 5a); it is much more apparent for the challenging collection of clinical images of EDRA Atlas (Fig. 5b). 

**Answer 1:** It can lead to an unfair advantage for the model. **Answer 2:** Optimizing a model involves selecting certain parameters or features that improve its performance on a given dataset. If the same dataset is used to measure the model’s performance, it may lead to an unfair advantage as the model has already been “tuned” to that specific dataset. 

Figure 7. Text-only Q-E-A triplets generation prompt and data example. 

15 

**==> picture [464 x 398] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image-only Question Generation Prompt<br>Task Definition:<br>Formulate 2 academic questions based on a provided figure or table from a research paper.<br>Requirements: The questions must directly reference and integrate information presented in the image and its cap-<br>tion, ensuring a cohesive understanding of the content depicted. You should use English.<br>Develop 2 questions that are:<br>• No more than 30 words.<br>• Specific to the unique data or details visible in the figures/tables and are answerable only based on the material<br>without inferring or speculating on details not explicitly explained by the figures/tables.<br>• Not mentioning the label of the figure/table directly or use words like ‘from the figure/table’.<br>Expected Output:<br>[Q1]: [Q2]:<br>Image-only Answer Generation Prompt<br>Task Definition:<br>Answer a question based on an image and its caption from a research paper.<br>Requirements:<br>The answers should:<br>• Always use English.<br>• Not infer or speculate on details not explicitly explained by the figures/tables.<br>You should think step by step and give you answer in the end of your generation like: [thinking procedure]: [A1/A2]<br>Expected Output:<br>[THINKING PROCEDURE]: ... [A1]: answer1 here, no more than 20 words.<br>[THINKING PROCEDURE]: ... [A2]: answer2 here, no more than one sentence.<br>Image-only Data Example<br>Query: Based on Figure 4, which fuzzing technique consistently leads in coverage across all benchmarks over the<br>24-hour period?<br>Figure: Figure 4<br>**----- End of picture text -----**<br>


**==> picture [464 x 22] intentionally omitted <==**

**----- Start of picture text -----**<br>
Caption: Figure 4: Number of benchmarks on which each technique has the lead in coverage at each hour. A<br>benchmark is counted for multiple techniques if two techniques are tied for the lead.<br>**----- End of picture text -----**<br>


**Answer 1:** FairFuzz consistently leads in coverage across all benchmarks over the 24-hour period in Figure 4. **Answer 2:** FairFuzz, highest coverage benchmark count over 24 hours. 

Figure 8. Image-only Q-E-A triplets generation prompt and data example. 

16 

## **Text-image Question Generation Prompt** 

## **Task Definition:** 

Formulate 2 academic questions based on a provided paragraph, a figure or table from a research paper. **Requirements:** The questions should not rely on the accompanying text but only the figure / table. However, you may use the provided text to understand the figure / table. You should use English. Develop 2 questions that are: 

- No more than 30 words. 

- Based on a provided figure of a research paper, without relying on accompanying text for the questions. However, you may use the provided text to understand the figure. 

- Not mentioning the label of the figure/table directly or use words like ‘from the figure/table’. 

- **Expected Output:** 

[Q1]: [Q2]: 

## **Text-image Answer Generation Prompt** 

## **Task Definition:** 

Answer a question based on the material given. **Requirements:** The answers should be: 

- Always using English. 

• Comprehensive and cover all relevant aspects, as presented in the provided text and figure. Accurately reflect the information and insights offered by the research paper. You should think step by step and give you answer in the end of your generation like: [thinking procedure]: [A1/A2] **Expected Output:** 

[THINKING PROCEDURE]: ... [A1]: answer1 here, no more than 20 words. 

[THINKING PROCEDURE]: ... [A2]: answer2 here, no more than one sentence. 

## **Text-image Data Example** 

**Query:** What is the best performing method for both detection and classification, according to the provided figure? **Text:** 3) The last one is our proposed SFCN-OPI with both sibling branches and OPI (Ours in Table 1). 

**Figure:** Table 1 

**Caption:** Experimental results of ablation analysis, ... 

## **Answer 1:** Ours. 

**Answer 2:** The best performing method for both detection and classification is Ours, as it achieves the highest F1 scores for both tasks. This can be seen in the “Ours” row of the table, where the F1 score for detection is 0.834 and the F1 score for classification is 0.742. 

Figure 9. Text-image Q-E-A triplets generation prompt and data example. 

17 

## **Section Question Generation Prompt** 

**Task Definition:** 

Formulate 2 academic questions based on a section from a research paper. **Requirements:** Carefully read and comprehend the entire provided section of the research paper to ensure a thorough understanding of its content, including key points, findings, methodologies, and conclusions. You should Always use English. Develop 2 questions that are: 

- No more than 30 words. 

- Requiring an integration of information from all paragraphs and figures/tables in the section. 

- Not mentioning the label of the figure/table directly or use words like ‘from the figure/table’. 

- Not based on common knowledge or assumptions not supported by the figures and tables. 

- **Expected Output:** 

- [Q1]: [Q2]: 

## **Section Answer Generation Prompt** 

## **Task Definition:** 

Answer a question based on the material given. **Requirements:** 

The answers should be: 

- Always using English. 

• Not inferring or speculating on details not explicitly explained by the figures/tables. You should think step by step and give you answer in the end of your generation like: [thinking procedure]: [A1/A2] **Expected Output:** 

- [THINKING PROCEDURE]: ... [A1]: answer1 here, no more than 20 words. 

- [THINKING PROCEDURE]: ... [A2]: answer2 here, no more than one sentence. 

## **Section Data Example** 

**Query:** How does the qualitative evaluation of extractive summarizers using word clouds elucidate the differences in content focus between the original documents and the summaries? 

## **Text:** 

Here we use word cloud representations to give an intuitive interpretation of the content in the generated extractive summarizers... 

Figure 3 shows a word cloud made by the aggregation of all the summaries generated by the PKUSUMSUM-Centroid method... 

The images clearly show a contrast of content... 

**Figure:** Figure 3 

**Caption:** Figure 3: The word clouds representing summaries generated by PKUSUMSUM-Centroid method (left) and original documents without the content of those summaries (right). **Answer 1:** Visual contrast in word frequency highlights content focus differences. **Answer 2:** Word clouds highlight the prominent themes in summaries versus original texts by displaying relative word frequencies visually. 

Figure 10. Section Q-E-A triplets generation prompt and data example. 

18 

## **Cross-paragraph Question Generation Prompt** 

## **Task Definition:** 

Based on the selected paragraph from a research paper that share a thematic or conceptual connection, formulate an insightful, open-ended question. This question should reflect the shared themes or concepts of your selections and relate to the broader context of the research paper. 

## **Requirements:** 

- Ascertain the underlying connection among the paragraphs and the figures/tables(if provided). 

- Subsequently, craft an insightful, open-ended question that encapsulates the identified themes or connections, aiming to foster analytical thinking and in-depth discussion on the subject matter of the paper. 

- **Expected Output:** 

- [Q]: [Your generated question based on the shared themes or information] 

## **Cross-paragraph Answer Generation Prompt** 

## **Task Definition:** 

Given some selected paragraphs from a research paper, and ensuring that these paragraphs share a certain level of association, you are to answer a question that is related to the content of these selected paragraphs. **Requirements:** 

Craft the 2 answers for the question that: 

- Are directly derived from the provided figure, excluding information not found within the material. 

- Are comprehensive and cover all relevant aspects, as presented in the provided figure. 

- **Expected Output:** 

- [A1]: [Insert a concise answer here, no more than 20 words.] 

- [A2]: [Insert a detailed answer here,including a detailed “thought chain” or reasoning process.] 

## **Cross-paragraph Data Example** 

**Query:** How can we leverage the proposed EDO approach to optimize the selection of datasets for specific algorithms, thereby enhancing the overall performance and validity of the algorithms? 

## **Text:** 

**Paragraph 7:** Figure 1: On the right: the current path for selecting some algorithm(s), ... **Paragraph 17:** Section 2 describes the structure of the proposed method including its parameters and operators. **Paragraph 20:** In this section, the details of an algorithm that generates data for which a given function, ... **Figure:** Figure 1 

**Caption:** Sample number of rows and columns 2. Sample columns and fill/trim values as needed. **Answer 1:** The EDO approach identifies optimal datasets, improving algorithm performance and validity through tailored data selection. 

**Answer 2:** We can systematically evaluate and select datasets that align closely with the specific requirements of the algorithms in use. This method analyzes various dataset characteristics, ensuring that the chosen data not only matches the algorithm’s operational parameters but also enhances its predictive accuracy. 

Figure 11. Cross-paragraph Q-E-A triplets generation prompt and data example. 

19 

