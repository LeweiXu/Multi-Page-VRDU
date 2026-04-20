# **Enhancing Document VQA Models via Retrieval-Augmented Generation** 

Eric López[1[0009] _[−]_[0003] _[−]_[3229] _[−]_[5739]] , Artemis Llabrés[1[0000] _[−]_[0002] _[−]_[6128] _[−]_[1796]] , and Ernest Valveny[1[0000] _[−]_[0002] _[−]_[0368] _[−]_[9697]] 

Computer Vision Center, Universitat Autònoma de Barcelona, Spain `Eric.LopezCe@autonoma.cat, allabres@cvc.uab.cat, ernest@cvc.uab.cat` 

**Abstract.** Document Visual Question Answering (Document VQA) must cope with documents that span dozens of pages, yet leading systems still concatenate every page or rely on very large vision-language models, both of which are memory-hungry. Retrieval-Augmented Generation (RAG) offers an attractive alternative, first retrieving a concise set of relevant segments before generating answers from this selected evidence. In this paper, we systematically evaluate the impact of incorporating RAG into Document VQA through different retrieval variants—text-based retrieval using OCR tokens and purely visual retrieval without OCR—across multiple models and benchmarks. Evaluated on the multi-page datasets MP-DocVQA, DUDE, and InfographicVQA, the text-centric variant improves the "concatenate-all-pages" baseline by up to +22.5 ANLS, while the visual variant achieves +5.0 ANLS improvement without requiring any text extraction. An ablation confirms that retrieval and reranking components drive most of the gain, whereas the layout-guided chunking strategy—proposed in several recent works to leverage page structure—fails to help on these datasets. Our experiments demonstrate that careful evidence selection consistently boosts accuracy across multiple model sizes and multi-page benchmarks, underscoring its practical value for real-world Document VQA. 

**Keywords:** Document Visual Question Answering · Multi-Page Scenario · Multi-Modal Scenario · Retrieval-Augmented Generation. 

**Project page:** `https://pikurrot.github.io/RAG-DocVQA/` 

## **1 Introduction** 

Document VQA aims to enable models to answer questions about the contents of documents, requiring multimodal reasoning over textual, spatial, and visual elements. While early research mainly addressed single-page scenarios through popular benchmarks such as DocVQA, the majority of real-world documents—including manuals, scientific papers, and technical reports—extend across multiple pages. Addressing this gap, MP-DocVQA was recently proposed as a 

2 E. López et al. 

multi-page extension of the original DocVQA dataset, posing natural language questions over sequences of up to twenty pages, accompanied by OCR-extracted text and page images. 

Existing state-of-the-art multi-page models, such as Hi-VT5, leverage hierarchical transformer architectures to summarize individual pages before synthesizing the final answer. However, their performance deteriorates as the document length grows, since cross-page reasoning significantly increases complexity. While Large Visual Language Models (LVLMs) such as Qwen2.5-VL can handle long-context inputs, they require substantial computational resources and GPU memory, making deployment costly and inefficient. 

A promising alternative is Retrieval-Augmented Generation (RAG), which has recently demonstrated effectiveness in open-domain question answering by first retrieving compact subsets of relevant evidence, thus drastically reducing the inference cost and noise introduced by irrelevant content. In the context of multi-page Document VQA, a RAG pipeline involves segmenting documents into concise textual chunks or visual patches, retrieving the most pertinent segments based on the question, and subsequently generating answers conditioned solely on these retrieved segments. 

In this paper, we hypothesize that augmenting existing small and mediumsized Document VQA models with RAG can significantly mitigate the challenges posed by multi-page scenarios. Specifically, we argue that this approach yields performance improvements in both accuracy and computational efficiency across various models and datasets. To validate this hypothesis, our contributions are as follows: 

**–** We propose and implement textual RAG variants for two representative Vision Language Models: VT5 and Qwen2.5-VL-7B-Instruct, along with a visual counterpart for Pix2Struct, an OCR-free vision transformer model. 

**–** Through comprehensive experiments on MP-DocVQA, DUDE, and InfographicVQA, we demonstrate consistent improvements in question-answering accuracy, ANLS scores, and retrieval metrics, thereby validating our hypothesis. 

## **2 Related Work** 

## **2.1 Datasets and Task Evolution** 

Document Visual Question Answering was initially formalized with the DocVQA dataset [20], comprising 50,000 questions posed over more than 12,000 industryrelated single-page document images, where answers are explicitly present in the OCR-extracted text. InfographicVQA [19] expanded the task visually, introducing infographic-rich images accompanied by questions that occasionally require numerical or categorical reasoning beyond mere text extraction. Subsequently, DocCVQA [22] introduced a retrieval-oriented scenario, where questions are answered by retrieving relevant evidence from a collection of 14,362 single-page documents. Recognizing the limitations of single-page scenarios, MP-DocVQA 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

3 

[23] extended the original DocVQA dataset to multi-page contexts by augmenting each document-question pair with preceding and following pages, forming sequences of up to 20 pages. Concurrently, DUDE [15] provided diverse multipage documents from various domains, featuring questions that require multistep reasoning and both extractive and abstractive answers. Additionally, the recent MMLongBench-Doc dataset [18] features 1,091 questions over 135 long PDF documents with rich layouts, where 33% of the questions require cross-page reasoning and 22.5% are unanswerable, which tests model robustness. 

## **2.2 From OCR-based to OCR-free Approaches** 

Initial Document VQA models largely relied on OCR-extracted text, adapting pre-trained language models such as BERT [8] with a classification head to predict the start and end indices where the answer appeared in the text, and T5 [21] for generative extractive question answering from OCR tokens. LayoutLM [28] introduced spatial understanding by augmenting textual embeddings from BERT with 2-dimensional positional information derived from OCR bounding boxes. LayoutLMv2 [27] then integrated visual embeddings from the document image and added relative positional biases to the self-attention mechanism. LayoutLMv3 [12] further refined this by encoding images as patch-level embeddings. To further leverage visual modalities, DocFormer [1] combined textual, visual, and spatial features via a novel multi-modal self-attention mechanism, using a ResNet [10] backbone for image features. Later, DocFormerv2 [2] replaced the ResNet CNN with a linear projection of image patches and adopted a T5-based encoder-decoder architecture. Addressing multi-page reasoning, Hi-VT5 [23] introduced a hierarchical transformer where the encoder summarizes each page independently, and the decoder attends to these summaries to generate the answer. GRAM proposed inserting additional document-level transformer layers between page-level encodings, using document tokens to propagate information across pages, achieving state-of-the-art performance when combined with DocFormerv2. More recently, OCR-free models emerged, eliminating explicit text extraction. Pix2Struct [16] proposed a fully end-to-end approach without explicit OCR, pretraining a vision transformer encoder to transcribe masked web screenshots into simplified HTML. Nevertheless, handling multi-page scenarios remained challenging due to growing computational complexity. 

## **2.3 Generic Vision-Language Models** 

Recently, generic vision-language models (VLMs), such as Qwen2.5-VL [3], have started being explored for Document VQA tasks. These models can directly process combined visual and textual inputs thanks to extensive pre-training on large multimodal datasets. However, a major issue with these models is their high computational cost and GPU memory usage, making practical deployment difficult. To mitigate such limitations, retrieval-based strategies like ColPali [9] have been adopted, which extends the PaliGemma [4] vision-language model with a ColBERT-style [14] late interaction mechanism, generating visual page 

4 E. López et al. 

embeddings optimized for retrieval tasks, thus improving evidence selection prior to answer generation. 

## **3 Methodology** 

We adapt Retrieval-Augmented Generation (RAG) to multi-page Document VQA through a three-stage pipeline comprising indexing, retrieval, and generation. In particular, we propose two complementary variants: a textual RAG, applied to two language-based models—a smaller encoder–decoder (VT5) and a larger vision–language model (Qwen2.5-VL-7B)—and a visual RAG, where retrieval and generation operate directly on image inputs without explicit OCR, using Pix2Struct as our baseline model. 

## **3.1 Textual RAG** 

**Indexing** We first segment the OCR-extracted token sequences of the multipage documents into textual chunks. The chunking process is controlled by three parameters: chunk size _L_ (tokens per chunk), overlap _O_ (shared tokens between neighbors), and chunk-size tolerance _τ_ , which allows the last chunk on a page to expand to (1 + _τ_ ) _× L_ tokens, so that small remainders are merged rather than isolated. Each chunk text is then encoded offline into a fixed-dimensional vector embedding using a bi-encoder architecture. Specifically, we use _bge-ensmall-v1.5_ [26], which has proven to be the most effective and efficient. We fine-tune this embedding model using contrastive learning on dataset-specific query–chunk pairs, optimizing a Multiple-Negatives Ranking Loss (details in Section 4). Finally, we store each chunk’s tokens, bounding boxes, image crop (extracted from the chunk bounding box), and embedding for fast retrieval. 

**Retrieval** At inference time, we encode the input query ( _q_ ) using the same embedding model with shared weights, obtaining an embedding vector of the same dimensionality as the chunks. We compute the cosine similarity between the query embedding and all stored chunk embeddings, retrieving an initial candidate set of the top _k[′]_ most similar chunks. To further enhance precision, we employ an additional cross-encoder reranker—a transformer-based classification model trained specifically to discriminate between relevant and irrelevant query–chunk pairs. Unlike the initial bi-encoder, the cross-encoder reranker jointly encodes each query together with each candidate chunk in a single forward pass, leveraging full self-attention across all tokens from both inputs. It outputs a refined relevance score indicating the likelihood that the chunk contains the answer. Finally, we select the top _k_ chunks with the highest reranked scores, which are then used for generation. 

**Generation** The generation step differs according to the underlying model used. For VT5, the _k_ top-ranked chunks’ text tokens are concatenated with 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

5 

**==> picture [331 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
offline Question “In which city is ITC's Watershed online<br>Development Project located?”<br>Bi<br>=<br>text boxes img<br>text Bi<br>EEs| 2= -f=<br>text boxes img text<br>boxes<br>text Bi text img<br>text boxes img<br>text<br>a| = text Bi _ text Cross CONCAT boxesimg Generator “Sehore”<br>< # 2 NN<br>k = 10<br>k’ = 20<br>text boxes img boxestext<br>text Bi text img<br>~~} HIbe<br>. . .<br>. . .<br>. . .<br>**----- End of picture text -----**<br>


**Fig. 1.** Overview of the **Textual RAG** pipeline. **Offline (pink panel):** A multi-page document is segmented into chunks consisting of OCR (text and boxes) and the image crop of the chunk. A Bi-Encoder converts the chunk text into a dense embedding for later retrieval. **Online (green panel):** the user question is encoded in the same way, and cosine-similarity is used to select the top- _k[′]_ chunks, whose text is then passed to a Cross-Encoder (Reranker) for a more refined filtering and ranking. The full information of the highest-ranked _k_ chunks is then concatenated and fed to the Generator to produce an answer. 

6 E. López et al. 

the question tokens to form the semantic input. Question tokens are assigned placeholder bounding boxes with all coordinates set to zero and concatenated with the chunks’ original boxes, guaranteeing that the semantic and spatial sequences remain aligned. The image crops of the different chunks are arranged into a compact adaptive grid, minimizing the total area and thereby optimizing input efficiency. As in Hi-VT5, the three input modalities—semantic, spatial, and visual—are first embedded separately. The semantic and spatial information of each token is fused by summing their corresponding embeddings: 

**==> picture [285 x 13] intentionally omitted <==**

where _EO_ ( _Oi_ ) is the semantic embedding of OCR token _Oi_ , produced by a T5 language backbone, and _Ex_ , _Ey_ are learned embeddings for the bounding box coordinates. 

The image crops grid is encoded into an embedding _V_ by a Document Image Transformer (DIT), specifically _dit-base-finetuned-rvlcdip_ . We then concatenate all the question embeddings _E[q]_ , the text embeddings _E[o]_ , and the visual patch embeddings _V_ to form the final input sequence [ _E[q]_ ; _E[o]_ ; _V_ ] which is fed to the VT5 decoder to generate the answer. This generation process for VT5 is illustrated in Fig. 3 (top-left). 

For Qwen2.5-VL-7B, the retrieved chunks’ texts are concatenated with the question and a prompt ( _"Directly provide only a short answer to the question."_ ) to form a single textual input sequence. Each chunk’s corresponding image crop is embedded independently by Qwen’s vision encoder and integrated within the textual input through special placeholders. The combined multimodal embeddings are then processed by Qwen’s decoder, autoregressively generating the final answer. The generation procedure for Qwen2.5-VL-7B is depicted in Fig. 3 (top-right). 

## **3.2 Visual RAG** 

**Indexing** Document images are horizontally segmented into overlapping visual patches. Each patch has a fixed vertical pixel size (patch size _P_ ), and overlaps by half of this height with adjacent patches to ensure no crucial information is lost across patch boundaries. Each visual patch is encoded independently into multi-vector embeddings _Ep_ of shape _T_ img _×_ 768 using a Visual Encoder, where _T_ img is the number of image tokens. Specifically, we use the Pix2Struct vision transformer encoder, which works with an image sequence length of _T_ img = 2048. 

**Retrieval** For retrieval, the textual query is rendered as an image on a white background and embedded into multi-vector representations _Eq_ of shape _T_ img _×_ 768 using the same Pix2Struct encoder as used for the document patches. We perform late interaction retrieval following ColBERT [14], which computes tokenlevel similarity _Sq,p_ between the query _q_ and each candidate patch _p_ , allowing fine-grained matching: 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

7 

**==> picture [336 x 210] intentionally omitted <==**

**----- Start of picture text -----**<br>
offline Question “In which city is ITC's Watershed online<br>Development Project located?”<br>Visual<br>Encoder<br>i |<br>img<br>| eom —=_—_ els b img EncoderVisual epep<br>img<br>img Visual img<br>Encoder<br>= Development Project es<br>—— atSehore’has been<br>img<br>img EncoderVisual img CONCAT img Generator “Sehore”<br>-. ry . ” ><br>=|<br>k = 5<br>img _<br>img Visual img<br>H is Encoder<br>. . . . . .<br>**----- End of picture text -----**<br>


**Fig. 2.** Overview of the **Visual RAG** pipeline. **Offline (pink panel):** A multi-page document is segmented into image patches, each of which is passed to a Visual Encoder to produce multi-vector embeddings for later retrieval. **Online (green panel):** the user question is rendered as an image, encoded in the exact same way and matched to the patch embeddings through late interaction. The top- _k_ image patches are concatenated (see Fig. 3 (bottom)) and passed to the generator. 

8 E. López et al. 

**==> picture [237 x 25] intentionally omitted <==**

The top- _k_ most relevant patches are selected based on their similarity scores. If retrieved patches overlap spatially, overlapping regions are merged into single unified patches, removing duplicate visual information. 

**Generation** The concatenation and generation process for RAG-Pix2Struct is shown in Fig. 3 (bottom). The retrieved _k_ image patches are resized and concatenated to form the input sequence for Pix2Struct [16]. Specifically, each image patch is scaled so it can be evenly divided into small, non-overlapping 16 _×_ 16 mini-patches. These mini-patches receive 2-dimensional positional indices: row indices increase continuously across the concatenated patches (from top to bottom), while column indices restart at 1 within each patch (from left to right). This indexing preserves each patch’s internal 2-D layout while forming a vertical stack of patches. Finally, the resulting sequence of _T_ img mini-patches is fed directly into Pix2Struct, which encodes them using its built-in 2-D positional encoder to generate the final answer. 

## **4 Experimental Setup** 

## **4.1 Evaluation Metrics** 

We employ metrics for two complementary aspects of the task: (i) question answering, and (ii) retrieval quality of relevant document content. 

_(i) Question Answering Metrics._ To evaluate the final output of the generator, we use standard accuracy and ANLS (Average Normalized Levenshtein Similarity), which are commonly used in Document VQA benchmarks to measure the correctness and textual similarity of predicted answers with respect to ground truth. 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

9 

**==> picture [317 x 235] intentionally omitted <==**

**----- Start of picture text -----**<br>
RAG-VT5 RAG-Qwen<br>CONCAT Generator<br>Q CONCAT Generator<br>text1 T5 Q<br>text...2 Encoder prompt<br>textk text1<br>text2<br>boxesQ 1 Spatial + VT5 text...k Qwen2.5 Answer<br>boxes... 2 Emb Decoder Answer img1 VL-7B<br>boxesk img2<br>...<br>img1 imgk<br>img...2 DIT<br>imgk<br>a h. jij<br>RAG-Pix2Struct<br>CONCAT Generator<br>img1<br>1 (1,1)<br>img2 23 1 2 3 4 5 6 7 8 9 (1,2)(1,3)(1,4)<br>4 (1,5)<br>‘ae img S k ti 76 1 2 3 4 5 Se 6 7 8 9  D Timg (9,6)(9,5) : | Pix2Struct - Answer<br>789 (9,7)(9,8)(9,9)<br>1 2 3 4 5 6 7 8 9<br>**----- End of picture text -----**<br>


**Fig. 3.** Different concatenation and generation strategies for the proposed RAG methods. For RAG-VT5, the information of different modalities is separately concatenated and fed to a specialized encoder, before aggregating and passing it to the decoder. For RAG-Qwen, the text of the chunks is concatenated along with the question and an instruction prompt, and is passed together with the separate chunk image crops to the model. For RAG-Pix2Struct, only the image crops are processed, by tiling them into mini-patches, assigning a positional coordinate to each one and feeding them to the model. 

_(ii) Retrieval Quality Metrics._ To assess the effectiveness of our retrieval pipeline, we use two additional metrics: 

_• Page-level retrieval._ We adapt the retrieval precision metric, originally used by existing multi-page models to evaluate their page-retrieval modules, to measure whether any of the retrieved chunk comes from the correct page. Specifically, we define Retrieval Precision@ _k_ as: 

**==> picture [303 x 15] intentionally omitted <==**

where _p_ gt is the ground-truth answer page and _p_[pred] _j_ is the page associated with the _j_ -th retrieved chunk ( _j_ = 1 _, . . . , k_ ). 

_• Chunk-level semantic matching._ We introduce Chunk Score@ _k_ as a smooth measure of how similar the ground-truth answer is to any part of the retrieved 

10 E. López et al. 

**Table 1.** Key hyper-parameters used for the Textual RAG models (left) and the visual RAG (right). 

## **RAG-VT5 / RAG-Qwen** 

|Parameter|Value|
|---|---|
|Chunk size _L_|60|
|Chunk size Tolerance _τ_|0.2|
|Overlap _O_|10|
|_k′_|20|
|_k_|10|



**RAG-Pix2Struct** 

|Parameter|Value|
|---|---|
|Patch size _P_|512|
|Overlap _Opix_|256|
|_k_|5|



text, even if not an exact match: 

**==> picture [275 x 25] intentionally omitted <==**

where sim( _cj, a_ ) computes the similarity between chunk _cj_ and the ground-truth answer _a_ . This similarity is defined as the log-scaled maximum inverse-editdistance between _a_ and all possible substrings of equal length within _cj_ . 

## **4.2 Implementation Details** 

The specific hyperparameter values shown in Table 1 are the ones that yield better results in our experimentation. For the bi-encoder, we used bge-en-smallv1.5 [26] to encode both the query and the document chunks, and for the crossencoder we used bge-reranker-v2-m3 [17,6] to rerank and filter the retrieved chunks. 

To enable a fair multi-page comparison between our RAG variants and their single-page baselines, we extend each baseline with a lightweight multi-page strategy. For VT5, we adopt a “concatenate” approach analogous to the T5concat method in [23], merging the OCR tokens—and, in our case, the page screenshots—of all pages into a single input context. For Pix2Struct, concatenating complete page images would over-shrink text, so we instead process each page independently, generate an answer per page, and retain the answer with the highest confidence; we refer to this variant as Pix2Struct-MaxConf. 

## **4.3 Training** 

The bi-encoder embedding model is fine-tuned using contrastive learning. For this, we create a synthetic dataset of anchor-positive pairs, where each anchor is a query and the corresponding positive is a chunk that previously yielded good retrieval performance. Specifically, we first process each document VQA dataset through our RAG pipeline with the original embedding model. For each retrieved set of _k_ chunks, we individually pass each chunk to the generator, obtaining _k_ 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

11 

separate answers. These answers are compared with the ground truth, and the chunk producing the highest ANLS is selected for that sample. Query–chunk pairs that achieve an ANLS greater than a threshold _t_ , which we empirically set to 0 _._ 8, form the positive training set. The bi-encoder is subsequently fine-tuned using a Multiple Negatives Ranking Loss [11], where for a given positive pair (query _q_ 1, chunk _c_ 1), the other chunks _c_ 2 _, ..., cb_ in the batch serve as negatives. 

Due to the sparse, discontinuous, and contextually diverse nature of retrievalaugmented inputs, we further fine-tune the generator models on each target VQA dataset, keeping the bi-encoder embedding weights frozen. For VT5, we perform full fine-tuning of all layers—including the language backbone and the spatial and visual embedding modules—using the AdamW optimizer with a linear learning-rate scheduler, an initial learning rate of 2 _e−_ 4, 1000 warm-up steps, and 4 epochs. For Qwen2.5-VL-7B-Instruct, we employ parameter-efficient LowRank Adaptation (LoRA), applying it specifically to the query and value projection matrices with parameters _α_ = 16, rank = 8 and dropout = 0 _._ 05. Training for both the embedding model and VT5 was performed on an NVIDIA TITAN Xp GPU (12 GB). The Qwen LoRA fine-tuning was conducted on an NVIDIA L40S GPU (46 GB). For Pix2Struct, we use versions already fine-tuned on each target dataset, so no further training was required. 

## **5 Results** 

## **5.1 Quantitative Results** 

Table 2 compares four publicly available models of similar parameter scale, the single-page baselines adapted to multi-page input (Pix2Struct-baseline, VT5baseline, Qwen2.5-VL-baseline), and their corresponding RAG versions. RAG consistently boosts the textual models: RAG-VT5 gains +13.2, +7.3, and +10.6 ANLS on MP-DocVQA, DUDE, and InfographicVQA, respectively, while RAGQwen improves by +22.5, +4.0, and +2.0 on the same datasets. For the visual model, RAG-Pix2Struct yields moderate gains on MP-DocVQA (+5.0 ANLS) and InfographicVQA (+3.3 ANLS) and a slight drop on DUDE (-0.8 ANLS). We attribute this to DUDE’s higher reasoning complexity—scenarios where processing each full-page image (as in the baseline) can retain crucial global context that retrieval-based pruning may omit. Apart from this outlier, the gains are consistent across models and datasets, confirming the general effectiveness of retrievalaugmented generation for multi-page Document VQA. Unlike Pix2Struct-SAretrieval, which recomputes full self-attention between the question and page features at inference time, our RAG-Pix2Struct encodes all patches offline and uses a lightweight late-interaction step. While the performance is not as high, this cuts GPU cost and latency, making real-world application more feasible. 

## **5.2 Qualitative Results** 

Fig. 4 illustrates qualitative examples comparing the performance of Textual and Visual RAG on two representative samples. In Example 1, we present a 

12 E. López et al. 

**Table 2.** Comparison of published multi-page Document VQA systems, our multi-page baselines, and their retrieval-augmented variants. 

|**Method**<br>**Params (M)**|**ANLS (%)**<br>MP-DocVQA<br>DUDE<br>InfoVQA|
|---|---|
|_Published SOTA_<br>T5-concat[23]<br>223<br>Pix2Struct-SA-retrieval[13]<br>273<br>Hi-VT5[23]<br>316<br>GRAM[5]<br>281|50.5<br>38.7<br>—<br>62.0<br>—<br>—<br>61.8<br>35.7<br>—<br>**73.9**<br>**46.2**<br>—|
|_Multi-page baselines_<br>Pix2Struct-baseline<br>282<br>VT5-baseline<br>223<br>Qwen2.5-VL-baseline<br>7 000|49.1<br>18.9<br>30.2<br>50.0<br>36.0<br>21.1<br>51.2<br>35.9<br>61.6|
|_Retrieval-augmented variants_<br>RAG-Pix2Struct (ours)<br>282<br>RAG-VT5 (ours)†<br>223 + 601<br>RAG-Qwen2.5-VL (ours)†<br>7 000 + 601|54.1<br>18.1<br>33.5<br>63.1<br>43.3<br>31.7<br>73.7<br>39.9<br>**63.6**|



> † Textual RAG totals include an additional 601M parameters (33M bi-encoder + 568M cross-encoder) used in the retrieval stack. 

scenario in which a question is posed to a 17-page document. Textual retrieval selects the top relevant text chunks, concatenates them, and passes them to the generator. Similarly, visual retrieval identifies the top image patches. Here, both models generate a correct answer, as the retrieved evidence clearly contains the necessary information. 

In contrast, Example 2 depicts a case where, despite accurate retrieval, the generated answers are incorrect. For textual retrieval, the error arises because the retrieved chunks provide ambiguous context; specifically, two different retrieved tables could individually answer the question, but only one is appropriate (the first retrieved chunk corresponds to non-smokers). For visual retrieval, the model fails to identify the correct column within the retrieved image patches, as crucial contextual information—the column headers distinguishing between smoker categories—is missing. 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

13 

**Example 1 (correct prediction)** 

**==> picture [282 x 416] intentionally omitted <==**

**----- Start of picture text -----**<br>
Textual retrieval<br>Top 3 chunks:<br>“8 june 18, 1975 the ability of dietary glycan to lower elevated blood cholesterol<br>and triglycerides in rats.  ten male rats  were fed the control diet (basal +<br>cholesterol and cholic acid) for four weeks and divided into two groups. one group<br>(2a) continued to receive the control diet. the second group (2b) received a test<br>diet of the same”<br>“second group (2b) received a test diet of the same composition as the control<br>diet except that 15 parts of glycan replaced 15 parts of sucrose. another group<br>(#1) of ten male rats were fed the basal diet for four weeks and then continued<br>Q:  “How many rats were were fed the on the basal diet. at weekly intervals, whole blood was drawn from the caudalvein for”<br>control diet?”<br>“fiber 2 2 20 - - - - glycan - - - 2 5 10 20 cholesterol + 0.2% cholic 0 1.0 1.0 1.0<br>17 pages 1.0 1.0 1.0 males: feed consumption (g) 126 124 144 126 123 121 124 014cholesterol consumption (g) 1.24 1.44 1.26 1.23 1.21 1.24 cholesterol excreted<br>(g) .046 .664 .861 .526 .603 .625 .750 % excreted”<br>A:  “ten male rats”<br>Visual retrieval<br>Top 5 image patches [†] :<br>—<br>GT: [ “ten” ,  “ten male rats” ]<br>(pag. 7)<br>A:  “ten”<br>Example 2 (wrong prediction)<br>Textual retrieval<br>Top 3 chunks:<br>“non-smokers doing enjoyable things means more to me than having a lot<br>of prized possessions 68 75  there should be less emphasis on money in<br>our society  80  82  the only meaningful measure of success is money 27 23<br>rjr712/monitor/mg/pi 29 7886 9eets”<br>“to pleasure doing enjoyable things means more to me than having<br>a lot of prized possessions 67 71 73  there should be less<br>emphasis on money in our society  27  31  33 the only meaningful<br>measure of success is money 27 25 24 rjr712/monitor/mg/pl 28<br>Q:  “What percentage of non-smokers feel 1886 9eets”<br>there should be less emphasis on money “the physical self = physical appearance trends undergoing asimilar shift in emphasis focus on looking good and being well<br>in our seciety?” groomed - over the long term - with minimum effort = signs of<br>turning away from fashion perfectionism less attention to "latest<br>20 pages fashions" less competitiveness more interest in comfort - physical -emotional = smokers are as involved in appearance as non-<br>smokers rjr712/monitor/mg/pl 32 $886 9eets”<br>A:  “31”<br>Visual retrieval<br>Top 5 image patches [†] :<br>|<br>GT: [ “82%”, “82” ]<br>(pag. 7)<br>A:  “80”<br>**----- End of picture text -----**<br>


**Fig. 4.** Qualitative examples of Question Answering on long-context multi-page samples of MP-DocVQA. Textual RAG uses VT5 as generator, while Visual RAG uses Pix2Struct. Example 1 shows a correct retrieval and correct model generation, while Example 2 shows a correct retrieval but a wrong generation. † While 5 image patches are retrieved, some may be overlapping so they are merged into a bigger one. 

14 E. López et al. 

## **6 Ablation Study** 

**Table 3.** Ablation of **RAG-VT5 on MP-DocVQA validation set** : impact of adding reranking, embedder fine-tuning, layout segmentation, spatial clustering, and layout loss. 

|Rerank<br>Train<br>Emb.<br>Layout<br>Cluster<br>Train<br>Layout|ANLS<br>Acc<br>Ret.<br>Prec.@_k_<br>Chunk<br>Score@_k_|
|---|---|
|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|58.23<br>50.74<br>**97.20**<br>**98.18**<br>61.06<br>53.57<br>95.39<br>97.78<br>**61.46 53.82**<br>96.20<br>97.97<br>57.92<br>49.01<br>95.41<br>95.55<br>58.45<br>49.54<br>96.26<br>95.37<br>59.14<br>49.99<br>96.26<br>95.37|



We conduct an ablation study to assess the contribution of individual components in our RAG pipeline, using the RAG-VT5 variant on the MP-DocVQA validation set. Alongside standard performance metrics (ANLS and accuracy), we also report retrieval-specific metrics—retrieval precision and chunk score—as defined in Section 4. The analysis is divided into two parts: (i) the impact of retrieval enhancements (reranking and embedding fine-tuning), and (ii) the effect of layout-guided chunking. 

## **6.1 Retrieval variants** 

The base retrieval system consists of chunking the document, encoding each chunk with a frozen embedding model, and selecting the top- _k[′]_ chunks via cosine similarity. To this, we add two enhancements: a cross-encoder reranker that refines the top- _k[′]_ selection into a more accurate top- _k_ subset, and contrastive fine-tuning of the embedding model to improve the quality of initial retrieval. As shown in Table 3, the base pipeline yields the highest retrieval precision, likely because it selects a larger candidate set ( _k_ ), increasing the chance of including the correct chunk. However, this broader context introduces more irrelevant content, which can hurt answer accuracy. Adding the reranker improves ANLS and accuracy by narrowing the context to the most relevant chunks. Further fine-tuning the embedding model does not significantly raise ANLS but leads to improved retrieval precision and chunk score, indicating more effective early filtering. 

## **6.2 Layout segmentation** 

While our default chunks are formed by sliding windows over the raw OCR tokens, several recent works show that layout-aware chunking—in which segments follow the page’s structural regions—can improve retrieval accuracy or 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

15 

downstream document-understanding tasks such as question answering and information extraction [24,29,25]. 

We therefore replace the sliding-window chunks with regions produced by a Document Image Transformer (DIT). Specifically, we use _cmarkea/dit-baselayout-detection_ [7], a DIT that outputs 11 fine-grained labels per page. Small boxes or boxes overlapping more than 50% with larger ones are considered noise and discarded. Labels are simplified to four categories: _title_ , _text_ , _figure_ , and _table_ . Centroids of these boxes form an undirected graph with edge weights as inverse Euclidean distances. Spectral clustering is then applied, selecting the number of clusters per page by maximizing the silhouette coefficient. Clusters are merged into rectangular layout chunks. 

During training, each OCR token receives a layout embedding scaled by a factor of 10 to match the semantic and spatial embeddings and summed with them. Additionally, a linear head attached to every encoder output predicts each token’s layout label, and the resulting cross-entropy loss is added to the main VQA loss. 

Table 3 shows that layout-based chunks hurt retrieval precision and, despite a modest recovery after clustering and the auxiliary loss, never surpass the simpler window strategy. We attribute this to over-segmentation of pages, which fragments the context and prevents the generator to answer with the enough information. 

## **7 Conclusion** 

In this work, we revisited multi-page Document VQA through the lens of RetrievalAugmented Generation. We adapted RAG pipelines—text-based for VT5 and Qwen2.5-VL, and purely visual for Pix2Struct—to efficiently select and aggregate evidence from long documents. Across three multi-page benchmarks, our approach consistently improved performance over strong baselines, achieving up to +22 ANLS gains while avoiding the memory demands of processing full document sequences. Through ablation studies, we found that reranking and contrastive embedder fine-tuning are key to these gains, whereas layout-guided chunking—despite success in other document tasks—did not yield benefits on our benchmarks. These results demonstrate that careful evidence selection enables smaller models to effectively tackle real-world, multi-page document VQA. 

## **Acknowledgments** 

This research has been supported by the Consolidated Research Group 2021 SGR 01559 from the Research and University Department of the Catalan Government, and by project PID2023-146426NB-100 funded by MCIU/AEI/10.13039/ 501100011033 and FEDER, UE. 

This work has also been funded by the European Lighthouse on Safe and Secure AI (ELSA) from the European Union’s Horizon Europe programme under grant agreement No 101070617. 

- 16 E. López et al. 

With the support of the FI SDUR predoctoral grant program from the Department of Research and Universities of the Generalitat de Catalunya and cofinancing by the European Social Fund Plus (2024FISDU_00095). 

## **References** 

1. Appalaraju, S., Jasani, B., Kota, B.U., Xie, Y., Manmatha, R.: Docformer: Endto-end transformer for document understanding. In: Proc. of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 993–1003 (2021) 

2. Appalaraju, S., Tang, P., Dong, Q., Sankaran, N., Zhou, Y., Manmatha, R.: Docformerv2: Local features for document understanding. Proceedings of the AAAI Conference on Artificial Intelligence **38** (2), 709–718 (Mar 2024). `https://doi.org/ 10.1609/aaai.v38i2.27828` , `https://ojs.aaai.org/index.php/AAAI/article/ view/27828` 

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. CoRR **abs/2502.13923** (2025), arXiv:2502.13923 

4. Beyer, L., Steiner, A., Pinto, A.S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., Unterthiner, T., Keysers, D., Koppula, S., Liu, F., Grycner, A., Gritsenko, A., Houlsby, N., Kumar, M., Rong, K., Eisenschlos, J., Kabra, R., Bauer, M., Bošnjak, M., Chen, X., Minderer, M., Voigtlaender, P., Bica, I., Balazevic, I., Puigcerver, J., Papalampidi, P., Henaff, O., Xiong, X., Soricut, R., Harmsen, J., Zhai, X.: PaliGemma: A versatile 3B VLM for transfer. CoRR **abs/2407.07726** (2024), arXiv:2407.07726 

5. Blau, T., Fogel, S., Ronen, R., Golts, A., Tsiper, S., Ben-Avraham, E., Aberdam, A., Bronstein, I., Litman, R., Mazor, S., Appalaraju, S., Manmatha, R.: Gram: Global reasoning for multi-page vqa. In: Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 15598–15607 (2024). `https://doi.org/10.1109/CVPR52733.2024.01477` 

6. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z.: Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through selfknowledge distillation (2024) 

7. Delestre, C.: (2024), `https://huggingface.co/cmarkea/ dit-base-layout-detection` 

8. Devlin, J., Chang, M., Lee, K., Toutanova, K.: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In: Proc. of NAACL-HLT (Vol. 1: Long and Short Papers). pp. 4171–4186 (2019) 

9. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P.: Colpali: Efficient document retrieval with vision language models. In: Proc. of the Int. Conf. on Learning Representations (ICLR) (2025) 

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016). `https://doi.org/10.1109/CVPR.2016.90` 

11. Henderson, M., Al-Rfou, R., Strope, B., Sung, Y., Lukacs, L., Guo, R., Kumar, S., Miklos, B., Kurzweil, R.: Efficient natural language response suggestion for smart reply. CoRR **abs/1705.00652** (2017), arXiv:1705.00652 

12. Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F.: Layoutlmv3: Pre-training for document ai with unified text and image masking. In: Proc. of the 30th ACM Int. Conf. on Multimedia (ACM-MM) (2022) 

17 

Enhancing Document VQA Models via Retrieval-Augmented Generation 

13. Kang, L., Tito, R., Valveny, E., Karatzas, D.: Multi-page document visual question answering using self-attention scoring mechanism. In: Proc. of the 17th Int. Conf. on Document Analysis and Recognition (ICDAR) (2024) 

14. Khattab, O., Zaharia, M.: Colbert: Efficient and effective passage search via contextualized late interaction over bert. In: Proc. of the 43rd Int. ACM SIGIR Conf. on Research and Development in Information Retrieval (SIGIR). pp. 39–48 (2020). `https://doi.org/10.1145/3397271.3401075` 

15. Landeghem, J.V., Tito, R., Borchmann, Ł., Pietruszka, M., Józiak, P., Powalski, R., Jurkiewicz, D., Coustaty, M., Anckaert, B., Valveny, E., Blaschko, M., Moens, S., Stanisławek, T.: Document Understanding Dataset and Evaluation (DUDE). In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 19528–19540 (2023) 

16. Lee, K., Joshi, M., Turc, I., Hu, H., Liu, F., Eisenschlos, J., Khandelwal, U., Shaw, P., Chang, M., Toutanova, K.: Pix2struct: Screenshot parsing as pretraining for visual language understanding. In: Proc. of the 40th International Conference on Machine Learning (ICML) (2023) 

17. Li, C., Liu, Z., Xiao, S., Shao, Y.: Making large language models a better foundation for dense retrieval (2023) 

18. Ma, Y., Zang, Y., Chen, L., Chen, M., Jiao, Y., Li, X., Lu, X., Liu, Z., Ma, Y., Dong, X., Zhang, P., Pan, L., Jiang, Y.G., Wang, J., Cao, Y., Sun, A.: Mmlongbenchdoc: Benchmarking long-context document understanding with visualizations. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 95963– 96010. Curran Associates, Inc. (2024) 

19. Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., Jawahar, C.V.: Infographicvqa. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1697–1706 (2022) 

20. Mathew, M., Karatzas, D., Jawahar, C.V.: Docvqa: A dataset for vqa on document images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 2199–2208 (2021). `https://doi.org/10.1109/ WACV48630.2021.00225` 

21. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research **21** (140), 1–67 (2020) 

22. Tito, R., Karatzas, D., Valveny, E.: Document collection visual question answering. In: Proc. 16th Int. Conf. on Document Analysis and Recognition (ICDAR). LNCS, vol. 12822, pp. 778–792. Springer, Cham (2021). `https://doi.org/10. 1007/978-3-030-86331-9_50` 

23. Tito, R., Karatzas, D., Valveny, E.: Hierarchical multimodal transformers for multi-page docvqa. Pattern Recognition **144** , 109834 (2023). `https://doi.org/ 10.1016/j.patcog.2023.109834` 

24. Verma, P.: S2 chunking: A hybrid framework for document segmentation through integrated spatial and semantic analysis. CoRR **abs/2501.05485** (2025), arXiv:2501.05485 

25. Wang, D., Raman, N., Sibue, M., Ma, Z., Babkin, P., Kaur, S., Pei, Y., Nourbakhsh, A., Liu, X.: Docllm: A layout-aware generative language model for multimodal document understanding. CoRR **abs/2401.00908** (2023), arXiv:2401.00908 

26. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N.: C-pack: Packaged resources to advance general chinese embedding (2023) 

18 E. López et al. 

27. Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C., Che, W., Zhang, M., Zhou, L.: Layoutlmv2: Multi-modal pre-training for visuallyrich document understanding. In: Proc. of ACL/IJCNLP (Volume 1: Long Papers). pp. 2579–2591 (2021) 

28. Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of text and layout for document image understanding. In: Proc. of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD). pp. 1192–1200 (2020). `https://doi.org/10.1145/3394486.3403172` 

29. Zhao, Z., Kang, H., Wang, B., He, C.: Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. CoRR **abs/2410.12628** (2024), arXiv:2410.12628 

