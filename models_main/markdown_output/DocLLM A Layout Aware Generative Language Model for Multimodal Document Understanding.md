## **DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding** 

## **Dongsheng Wang** _[∗]_ **, Natraj Raman** _[∗]_ **, Mathieu Sibue** _[∗]_ **Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, Xiaomo Liu** JPMorgan AI Research first.last@jpmorgan.com 

## **Abstract** 

Enterprise documents such as forms, receipts, reports, and other such records, often carry rich semantics at the intersection of textual and spatial modalities. The visual cues offered by their complex layouts play a crucial role in comprehending these documents effectively. In this paper, we present DocLLM, a lightweight extension to traditional large language models (LLMs) for reasoning over visual documents, taking into account both textual semantics and spatial layout. Our model differs from existing multimodal LLMs by avoiding expensive image encoders and focuses exclusively on bounding box information to incorporate the spatial layout structure. Specifically, the cross-alignment between text and spatial modalities is captured by decomposing the attention mechanism in classical transformers to a set of disentangled matrices. Furthermore, we devise a pre-training objective that learns to infill text segments. This approach allows us to address irregular layouts and heterogeneous content frequently encountered in visual documents. The pre-trained model is fine-tuned using a large-scale instruction dataset, covering four core document intelligence tasks. We demonstrate that our solution outperforms SotA LLMs on 14 out of 16 datasets across all tasks, and generalizes well to 4 out of 5 previously unseen datasets. 

## **1 Introduction** 

Documents with rich layouts, including invoices, contracts, and forms, constitute a significant portion of enterprise corpora, and the automatic analysis of these documents offer considerable advantages (Kunduru, 2023). Although Document AI (DocAI) has made tremendous progress, there remains a significant performance gap in real-world applications due to the complex layouts, bespoke type-setting and template diversity exhibited by 

*Equal Contribution. 

these visually rich documents. In particular, accuracy, reliability, contextual understanding and generalization to previously unseen domains continues to be a challenge (Cui et al., 2021). 

Conventional large language models (LLMs) such as GPT-3.5 (Brown et al., 2020), Llama (Touvron et al., 2023) or Falcon (Penedo et al., 2023) primarily accept text-only inputs and assume that the documents exhibit simple layouts and uniform formatting. They are not suitable for document intelligence tasks, which are inherently multi-modal, requiring the understanding of both text content and visual layout cues. Numerous vision-language frameworks (Li et al., 2022; Huang et al., 2022) that can process documents as images and capture the interactions between textual and visual modalities do exist. However, these frameworks necessitate the use of complex vision backbone architectures (Dosovitskiy et al., 2021) to encode image information, and often make use of spatial information as an auxiliary contextual signal (Xu et al., 2021; Lee et al., 2022). 

In this paper, we present DocLLM, a lightweight extension to standard LLMs that excels in several visually rich form understanding tasks. Unlike traditional LLMs, it models both spatial layouts and text semantics, and therefore is intrinsically multimodal. The spatial layout information is incorporated through bounding box coordinates of the text tokens obtained typically using optical character recognition (OCR), and does not rely on a complex vision encoder component. Consequently, our solution preserves the causal decoder architecture, introduces only a marginal increase in the model size, and has reduced processing times. We demonstrate that merely including the spatial layout structure is sufficient for various document intelligence tasks such as form understanding, table alignment and visual question answering. 

Existing efforts to incorporate spatial layout information typically involve either concatenating 

8529 

_Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8529–8548 August 11-16, 2024 ©2024 Association for Computational Linguistics 

Figure 1: Key elements of DocLLM. (1) Input documents with text tokens and bounding boxes. (2) Extended attention mechanism captures cross-alignment between text semantics and spatial layouts. (3) Infilling text blocks is used as pre-training objective. (4) Task adaptation is performed on a newly collated dataset of instructions. 

spatial and textual embeddings (Tang et al., 2023) or summing the two (Xu et al., 2020). In contrast, we treat the spatial information as a distinct modality and compute its inter-dependency with the text modality in a disentangled manner (Meng et al., 2021). Specifically, we extend the self-attention mechanism of transformers to include new attention scores that capture cross-modal relationships. There is often a correlation between the content, position and size of the fields in a form and hence representing their alignments at various abstraction levels across the transformer layers can enhance document understanding. 

Visual documents often feature heterogeneous content, irregular layouts, and disjointed text segments. A classical next token prediction in selfsupervised pre-training can be restrictive for these documents since the preceding tokens may not always be relevant due to the diverse arrangements of text. To tackle this issue, we propose two modifications to the pre-training objective: (a) adopting cohesive blocks of text that account for broader contexts, and (b) implementing an infilling approach by conditioning the prediction on both preceding and succeeding tokens. Due to these modifications, the model is better equipped to address misaligned text, contextual completions, intricate layouts, and mixed data types. Although text spans and infilling tasks have been studied before (Du et al., 2021), our solution is tailored for visual documents with an emphasis on semantically coherent blocks. 

We tune DocLLM on instruction data curated from multiple datasets for several document intelligence tasks including Key Information Extraction (KIE), Natural Language Inference (NLI), Visual Question-Answering (VQA) and document classification (CLS). The modifications introduced by DocLLM enhance the performance of Llama2-7B 

model by 15-60% in four of five datasets unseen during training. 

Our contributions include: (1) A lightweight extension to LLMs designed for understanding visual documents. (2) A disentangled spatial attention mechanism that captures cross-alignment between text and layout modalities. (3) An infilling pre-training objective tailored to address irregular layouts effectively. (4) A large instruction tuning dataset (with OCR data) specially curated towards visual document intelligence tasks. (5) Comprehensive experiments and insights into the model behavior. Fig. 1 summarizes the framework. 

## **2 Related Work** 

**General Purpose Models** . By treating a document as text content, many text based LLMs (OpenAI, 2023; Touvron et al., 2023; Anil et al., 2023) can be directly utilized for document intelligence tasks. Despite the remarkable capabilities provided by these LLMs, their lack of understanding of visual elements and layouts can be severely limiting in the DocAI context. Although multi-modal LLMs (Li et al., 2023; Zhu et al., 2023; Liu et al., 2023a; Wu et al., 2023; Ye et al., 2023c; Zhang et al., 2023a) that explicitly include image information can account for visual signals, they often struggle to recognize specific structures and patterns prevalent in enterprise documents (Liu et al., 2023c). Instead of relying on generalized training, a model tailored for visually rich document understanding (VRDU) tasks can gain a nuanced comprehension of the language, formats and data structures unique to these types of documents. 

**Document Understanding Models** . Models such as LayoutLM (Xu et al., 2020), LAMPreT (Wu et al., 2021), Pix2Struct (Lee et al., 2023), and 

8530 

UDOP (Tang et al., 2023) specifically cater towards document processing tasks. They can account for different modalities including text, image and layout information, and are trained to exploit both the structure and content of documents, often using a large corpora. However, these models require task-specific fine-tuning, may lack a flexible interface and cannot understand open-domain instructions. Recent efforts like mPLUG-DocOwl (Ye et al., 2023a) and UReader (Ye et al., 2023b) build on LLMs and perform DocAI-focused instruction tuning. We differ from these by avoiding expensive visual encoders. 

**Model Architecture** . Disentangled attention mechanisms, where different signals are represented by independent vectors, have been studied before (He et al., 2020). While we use a similar construct, our spatial position based encodings are more complex and applied in a multimodal context. Learning to infill autoregressive language models has been explored in Bavarian et al. (2022), Shen et al. (2023), and Du et al. (2021). Although we share their goal of adding fill-in-the-middle (FIM) capability, we differ in the mechanism by integrating FIM into the visual document contexts and avoiding extremely short segments. 

## **3 DocLLM Framework** 

## **3.1 Architecture Overview** 

DocLLM is constructed upon the foundation of an auto-regressive transformer language model (Touvron et al., 2023; Penedo et al., 2023) following a causal decoder structure. It integrates lightweight visual information by utilizing the spatial positions and dimensions of text tokens obtained using OCR. Instead of simply augmenting the text with bounding box information via additive positional encoding (Xu et al., 2021), separate vectors are used to represent these two distinct modalities and the self-attention mechanism of the transformer architecture is extended to compute their interdependencies in a disentangled manner. Furthermore, the traditional left-to-right next token prediction during self-supervised training is replaced by a block infilling objective that better leverages contextual information. See Figure 2 for an overview. 

## **3.2 Disentangled Spatial Attention** 

Let **x** = ( _x_ 1 _, ..., xi, ..., xT_ ) be an input sequence of length _T_ , where _xi_ is a text token. In classical transformers, using a learned embedding matrix 

based on the text vocabulary and a learned set of parameters for the token position in the sequence, the input tokens are first encoded into hidden vectors **H** _∈_ R _[T][×][d]_ . A self-attention head then computes the attention scores between tokens _i_ and _j_ as: 

**==> picture [220 x 26] intentionally omitted <==**

where **W** _[q] ∈_ R _[d][×][d]_ and **W** _[k] ∈_ R _[d][×][d]_ are projection matrices, and the superscript _t_ indicates the text modality. The attention scores **A** _∈_ R _[T][×][T]_ along with another projection matrix **W** _[v]_ are further used to compute the hidden vectors **H** _[′]_ , which are in turn used as inputs for a subsequent layer: 

**==> picture [215 x 27] intentionally omitted <==**

In DocLLM, the input is represented as **x** = _{_ ( _xi, bi_ ) _}[T] i_ =1[,] where _bi_ = (left, top, right, bottom) is the bounding box corresponding to _xi_ . To capture the new modality (i.e. spatial information), we encode the bounding boxes into hidden vectors represented by **S** _∈_ R _[T][×][d]_ . We then decompose the attention matrix computation into four different scores, namely _text-to-text_ , _text-to-spatial_ , _spatial-to-text_ and _spatial-to-spatial_ . Formally, the new attention mechanism is calculated as: 

**==> picture [189 x 51] intentionally omitted <==**

where **W** _[s,q] ∈_ R _[d][×][d]_ and **W** _[s,k] ∈_ R _[d][×][d]_ are newly introduced projection matrices corresponding to the spatial modality, and _λ_ s are hyperparameters that control the relative importance of each score. The input hidden vectors for the next layer **H** _[′]_ are computed exactly as before. However, in contrast to equation (2), the newly calculated hidden vectors rely not only on the text semantics but also on the layout information of the text tokens. 

It is important to mention that the hidden vectors **S** are reused across different layers, while each layer retains the flexibility to employ different projection matrices. We also note that the number of extra parameters required to encode the bounding box information is significantly lower compared to the overhead introduced by image based models (Li et al., 2022). By simply adding **S** to **H** similar to Xu et al. (2020), we could have avoided using **W** _[s]_ matrices altogether and further reduced 

8531 

Figure 2: DocLLM model architecture with disentangled spatial attention and infilling objective. _left_ : Input document with text tokens _xi_ and bounding boxes _bi_ . Some text blocks are randomly masked (two blocks here) and the model predicts the tokens in these text blocks autoregressively. _right_ : The infilling sequence is created by replacing the sampled blocks with [M] and prepending them with [S]. The attention mechanism is extended to account for cross-attention between text and spatial modalities. 

the number of parameters. However, it would have irreversibly coupled the layout information with the text semantics. In contrast, our disentangled representation of these modalities in the attention scores enables selective focus when appropriate (He et al., 2020), thereby providing an optimal balance between model size and effectiveness. 

## **3.3 Pretraining** 

DocLLM is first pre-trained in a self-supervised fashion on a large number of unlabeled documents. Visual documents are often sparse and irregular, featuring isolated and disconnected text fragments. It is preferable to consider coarse segments of related tokens during pre-training rather than focusing on individual tokens. Hence we use the broader context provided by multiple tokens, referred as blocks,[1] for better comprehension. Most OCR engines can provide block level information, which makes it feasible to identify coherent text blocks such as a heading or an address.[2] 

Learning to infill text, where the prediction is conditioned on both prefix and suffix tokens rather than only preceding tokens, can be beneficial for document understanding. The infilling objectives enable contextually relevant completions, provide 

> 1In Figure 2 “Name”, “John Doe” , and “Doctor” are all examples of blocks 

> 2In order to avoid any leakage of useful information, the block information is only used during pre-training, and the model is unaware of the number of tokens in a masked block. 

robustness to OCR noise or misaligned tokens, and can better handle relationships between various document fields. Hence we modify the standard pre-training objective to predict blocks of text given preceding and following text blocks. Inspired by (Du et al., 2021), we follow an autoregressive block infilling objective, where text blocks are randomly masked, and the masked blocks are shuffled and reconstructed in a sequential left-to-right fashion. 

Formally, let **c** = _{c_ 1 _, ..., cK}_ be a set of text blocks that partitions an input sequence **x** into nonoverlapping contiguous tokens such that _c_ 1 _∪ ... ∪ cK_ = **x** and _ck ∩ ck′_ = _∅_ . Let **z** = _{zm}[M] m_ =1[be] _M ≪ K_ different text blocks randomly sampled from **c** , where each block _zm_ = ( _zm,_ 1 _, ..., zm,Nm_ ) contains a consecutive series of tokens. Further, let **˜x** be a corrupted version of **x** where the contiguous tokens corresponding to a sampled text block are replaced with a special mask token [M]. To facilitate the identification of the block to be filled during text generation, each input block is augmented with a special start token [S] while the output block includes an end token [E]. For instance, a block with tokens ( _x_ 4 _, x_ 5) becomes [M] in **˜x** , ([S] _, x_ 4 _, x_ 5) when conditioned upon, and is expected to generate ( _x_ 4 _, x_ 5 _,_ [E]) as output autoregressively.[3] Let _θ_ denote all the parameters of the transformer model, including the projection matrices discussed above. The following cross-entropy 

> 3See Figure 2 for an illustration of these configurations. 

8532 

loss is then minimized for the infilling objective 

**==> picture [218 x 48] intentionally omitted <==**

## **3.4 Instruction Tuning** 

Following recent work in the field of VRDU (Tang et al., 2023; Ye et al., 2023a,b) and prior work in NLP (Wei et al., 2022; Chung et al., 2022), we instruction-tune DocLLM on a variety of instructions curated from multiple DocAI datasets using templates. We employ a total of 16 datasets with their corresponding OCRs, spanning four DocAI tasks. 

The diversity of supervised fine tuning (SFT) instructions is critical in helping zero-shot generalization (Wei et al., 2022; Chung et al., 2022; Ouyang et al., 2022). Thus, we diversify templates per task when possible, with each template asking a different question, and in some cases, expecting different types of answers. We re-use the templates introduced in Ye et al. (2023a,b) when applicable. 

We create the templates following what we believe end users would generally ask about documents (see Table 1). For KIE and CLS, we hypothesize that (1) the extraction instructions can teach DocLLM to correlate names of keys in the prompts with document fields so as to retrieve values, (2) the internal classification instructions can help the model understand what intrinsically characterizes each key or document type, and (3) the multiple choice question (MCQ) instructions can teach the model to leverage its comprehension of key names included as choices in the prompt (resp. document type names) to classify extracted values (resp. entire documents). The templates are as follows:[4] 

**Visual Question Answering** . A single template. Prompt Example: _What is the deadline for scientific abstract submission for ACOG - 51st annual clinical meeting?_ 

**Natural Language Inference** . A single template. Prompt Example: _"The UN commission on Korea include 2 Australians.", Yes or No?_ 

**Key Information Extraction** . Three templates corresponding to extraction, internal classification, and MCQ instructions. Example prompt for extraction: _What is the value for the "charity number"?_ 

> 4 Examples are derived from DocVQA (Mathew et al., 2021), TabFact (Chen et al., 2020), KLC (Stanislawek et al., 2021), RVL-CDIP (Harley et al., 2015). 

**Document Classification** . Two templates corresponding to internal classification and MCQ instructions. Example prompt for MCQ: _What type of document is this? Possible answers: [budget, form, file folder, questionnaire]._ 

See Appendix A.2 for further details. 

## **4 Experiments** 

## **4.1 Datasets** 

**Pre-training.** We gather data for pre-training from two primary sources: (1) IIT-CDIP Test Collection 1.0 (Lewis et al., 2006) and (2) DocBank (Li et al., 2020). IIT-CDIP Test Collection 1.0 encompasses a vast repository of over 5 million documents, comprising more than 16 million document pages. This dataset is derived from documents related to legal proceedings against the tobacco industry during the 1990s. DocBank consists of 500K documents, each featuring distinct layouts and a single page per document. We obtain a collection of 16.7 million pages comprising a total of 3.8 billion tokens. See Table 6 in the Appendix for detailed statistics. 

**Instruction Tuning.** To instruction-tune the model for the VQA task, we collect DocVQA (Mathew et al., 2021), WikiTableQuestions (WTQ) (Pasupat and Liang, 2015), VisualMRC (Tanaka et al., 2021), and DUDE (Landeghem et al., 2023). For NLI, we only include TabFact (Chen et al., 2020) in our instruction-tuning data mix, due to lack of additional DocAI NLI datasets available. For KIE, we gather Kleister Charity (KLC) (Stanislawek et al., 2021), CORD (Park et al., 2019), FUNSD (Jaume et al., 2019), DeepForm (Svetlichnaya, 2020), PWC (Kardas et al., 2020), SROIE (Huang et al., 2019), and VRDU ad-buy (Wang et al., 2023) (with random train-test splitting). Finally, we use RVL-CDIP (Harley et al., 2015) to build our CLS instruction-tuning data. We also downsample RVL-CDIP in the train split to avoid hindering the other datasets due to size. See Table 7 in the Appendix for detailed statistics. 

To the above datasets, we add BuDDIE (Zmigrod et al., 2024), a collection of _∼_ 1,600 business entity filings curated from state registration websites within the US. BuDDIE is annotated for three tasks – VQA, KIE, and CLS – and we therefore include it in the respective instruction-tuning collections.[5] 

> 5The instruction-tuning data is available upon request at https://www.jpmorgan.com/technology/ artificial-intelligence/initiatives/datasets 

8533 

Table 1: Prompt templates used for instruction-tuning (spatial tokens not included). 

|**Task**|**Template type**<br>**Prompt template**<br>**Expected response**|
|---|---|
|**VQA**|Extraction<br>{document} {question}<br>answerannotation|
|**NLI**|MCQ<br>{document} "{statement}", Yes or No?<br>answerannotation|
|**KIE**|Extraction<br>{document} What is the value for the "{key}"?<br>Associatedvalueannotation|
||MCQ<br>{document} What is "{value}" in the document? Possible choices: {keys}.<br>_(where_keys_is a subset of all the key names in the dataset in random order)_<br>Associatedkeyannotation|
||Internal classifcation<br>{document} What is "{value}" in the document?<br>Associatedkeyannotation|
|**CLS**|MCQ<br>{document} What type of document is this? Possible choices: {classes}.<br>_(where_classes_is a subset of all the classes in the dataset in random order)_<br>classannotation|
||Internal classifcation<br>{document} What type of document is this?<br>classannotation|



Table 2: Performance comparison in the SDDS setting against other multimodal and non-multimodal LLMs; non-multimodal LLMs are Zero-Shot (ZS) prompted while multimodal LLMs are instruction-tuned on the train split of the datasets considered. ‘*’ indicates datasets for which a designated test set was not publicly available. 

|**Dataset**|**GPT4+OCR**<br>**Llama2+OCR**<br>**mPLUG-DocOwl**<br>**UReader**<br>– (T)<br>7B (T)<br>7B (T+V)<br>7B (T+V)<br>ZS<br>ZS<br>SDDS<br>SDDS|**GPT4+OCR**<br>**Llama2+OCR**<br>**mPLUG-DocOwl**<br>**UReader**<br>– (T)<br>7B (T)<br>7B (T+V)<br>7B (T+V)<br>ZS<br>ZS<br>SDDS<br>SDDS|**DocLLM-1B**<br>**DocLLM-7B**<br>1B (T+L)<br>7B (T+L)<br>SDDS<br>SDDS|
|---|---|---|---|
|||||
|**VQA**<br>DocVQA<br>WTQ_(Accuracy)_<br>VisualMRC_(CIDEr)_<br>DUDE*<br>BuDDIE||**82.8**<br>47.4<br>62.2<br>65.4<br>**65.4**<br>25.0<br>26.9<br>29.4<br>255.1<br>115.5<br>188.8<br>221.7<br>**54.6**<br>38.1<br>-<br>-<br>76.4<br>48.8<br>-<br>-|61.4<br>69.5<br>21.9<br>27.1<br>245.0<br>**264.1**<br>42.6<br>47.2<br>84.5<br>**86.7**|
|||||
|**NLI**<br>TabFact||**77.1**<br>48.2<br>60.2<br>67.6|58.0<br>66.4|
|||||
|**KIE**<br>KLC<br>CORD<br>FUNSD<br>DeepForm<br>PWC<br>SROIE<br>VRDU a.-b.*<br>BuDDIE||45.9<br>27.8<br>30.3<br>32.8<br>58.3<br>13.8<br>-<br>-<br>37.0<br>17.8<br>-<br>-<br>42.1<br>20.5<br>42.6<br>49.5<br>18.3<br>6.8<br>-<br>-<br>90.6<br>56.4<br>-<br>-<br>43.7<br>18.7<br>-<br>-<br>66.1<br>10.8<br>-<br>-|58.9<br>**60.3**<br>66.9<br>**67.4**<br>48.2<br>**51.8**<br>71.3<br>**75.7**<br>25.7<br>**29.06**<br>91.0<br>**91.9**<br>87.6<br>**88.8**<br>95.4<br>**96.0**|
|||||
|**CLS**<br>RVL-CDIP<br>BuDDIE||68.2<br>32.8<br>-<br>-<br>84.9<br>40.9<br>-<br>-|90.9<br>**91.8**<br>98.3<br>**99.4**|



## **4.2 Evaluation Setup** 

**Model** . We train two variants **Configuration** of DocLLM: DocLLM-1B, which is based on the Falcon-1B architecture (Penedo et al., 2023), and DocLLM-7B, which is based on the Llama2-7B architecture (Touvron et al., 2023).[6] The maximum sequence length is set to 1,024 for both these models during the entire training process. See Appendix B for a detailed discussion on the model configuration and training hyper-parameters. 

**Settings** . We investigate two experimental settings: _**Same Datasets, Different Splits**_ (SDDS): Following previous work (Lee et al., 2023; Davis et al., 2022; Kim et al., 2022; Tang et al., 2023; Ye et al., 

> 6Since LLaMA2 does not come with pre-trained weights at 1B parameters, we use the Falcon-1B architecture for the smaller version of DocLLM. 

2023a,b), we first evaluate DocLLM on the unseen test split (or dev split when labeled test split is not publicly available) of each of the 16 datasets composing the instruction tuning data. The motivation behind this very typical setting is to check how DocLLM performs when tasks and domains supposedly stay the same from train to test. 

_**Same Tasks, Different Datasets**_ (STDD): Following (Wei et al., 2022; Chung et al., 2022; Dai et al., 2023; Zhang et al., 2023a), we also evaluate DocLLM on held-out datasets. More precisely, we instruction-tune the pretrained checkpoint of DocLLM on prompts from 11 of the 16 datasets considered in SDDS, then evaluate DocLLM on the test split of the remaining five datasets. The rationale behind this evaluation setting is to assess the performance of DocLLM when tasks are unchanged but domains and layouts differ from train to test. 

8534 

Table 3: Performance comparison in the STDD setting on held-out VRDU datasets against non-multimodal LLMs. 

|**Model**|**Size**|**Setting**|**DocVQA**<br>VQA|**KLC**<br>KIE|**BuDDIE**<br>VQA<br>KIE|**BuDDIE**<br>VQA<br>KIE|CLS|
|---|---|---|---|---|---|---|---|
|GPT4+OCR|–|ZS|**82.8**|45.9|**76.4**|66.1|**84.9**|
|Llama2+OCR|7B|ZS|47.4|27.8|48.4|10.8|40.9|
|||||||||
|DocLLM-1B|1B|STDD|53.5|40.1|65.5|63.0|20.8|
|DocLLM-7B|7B|STDD|63.4|**49.9**|73.3|**72.6**|31.1|



(a) Prompt: What is the value for (b) Prompt: What is written (c) Prompt: How many objectives the “advertiser”? under the heading ‘emergency are listed under at-event DocLLM: Bloomberg/D/President protein allowances’? activities? GPT4+OCR: MIKE BLOOMBERG 2020 DocLLM: Grams per person per day DocLLM: 4 GPT4+OCR: Men (70 Kg.) 50 55 ... GPT4+OCR: 5 

Figure 3: Qualitative examples of DocLLM-7B performance for KIE (Svetlichnaya, 2020) and VQA (Mathew et al., 2021) tasks. Correct answers are highlighted in blue and incorrect answers are highlighted in red. 

Table 4: Ablation study on disentangled spatial attention. T and S stands for text and spatial modality respectively. 

|**Mode**|**Cross-Modal Interactions**|**NTP Accuracy**|
|---|---|---|
|Additive|SEmbed + TEmbed|38.16|
||T2T|35.43|
||T2S + T2T|38.08|
|Disentangled|S2T + T2T<br>S2S + T2T<br>T2S + S2S + T2T|38.05<br>**39.12**<br>39.06|
||S2T + S2S + T2T|39.07|
||T2S + S2T + S2S + T2T|39.02|



Table 5: Ablation study on the block infilling objective. 

|**Pretraining Objective**|**NTP Accuracy**|
|---|---|
|Causal Learning<br>Causal Learning + Spatial<br>Block Infilling + Spatial|32.6<br>36.2<br>**39.1**|



We believe examining this setting in the DocAI field is relevant because industry use cases usually encountered in practice revolve around VQA, KIE, and CLS, while document characteristics tend to change more often in production. We specifically isolate DocVQA, KLC, and BuDDIE for STDD evaluation in order to (1) exclude at least one dataset per task from SFT when possible, (2) leave enough datapoints per task in the training split of the instruction-tuning data, (3) avoid data leakage, and (4) benchmark models on popular yet 

challenging datasets when possible. Due to the high cost of instruction-tuning, we were not able to run experiments with other held-out datasets. 

**Baselines** . In SDDS and STDD, we benchmark DocLLM against comparably-sized SotA LLMs using ZS prompts that contain the text extracted from each document using an OCR engine (excluding the spatial information) (Touvron et al., 2023; Ouyang et al., 2022). In SDDS, we also report numbers from recent DocAI LLMs evaluated in a similar setting (Ye et al., 2023a,b). As motivated in Section 2, we do not consider DocAI models that require task-specific fine-tuning such as LayoutLMv3 (Huang et al., 2022) or Pix2Struct (Lee et al., 2023), and/or dataset-specific prompts such as UDOP (Tang et al., 2023). We instead focus on LLMs with out-of-the-box instruction following capability.[7] 

**Metrics** . Following previous work (Borchmann et al., 2021; Lee et al., 2023; Ye et al., 2023b,a), we evaluate all VQA datasets using Average Normalized Levenshtein Similarity (ANLS) (Biten et al., 2019), with the exception of VisualMRC, for which we use CIDEr[8] (Vedantam et al., 2015) and WTQ, for which we use accuracy. Performance on all 

> 7Refer to Appendix C.4 for a comparison against SotA models regardless of architecture. 

> 8This is done to remain consistent with the results reported by other baselines. 

8535 

CLS and NLI datasets is measured using accuracy. We evaluate all KIE datasets with the F1 score. 

## **4.3 Results** 

**SDDS Setting** . Table 2 shows that DocLLM-7B excels in 12 out of 16 datasets, inclusively compared to ZS results of GPT4 and Llama2, and SDDS results of mPLUG-DocOwl and UReader. Among equivalent models (excluding GPT4), our model outperforms in 14 out of 16 datasets. Specifically, DocLLM demonstrates superior performance in layout-intensive tasks such as KIE and CLS. In VQA and NLI, its performance surpasses that of most multimodal language models, although it underperforms compared to GPT4. GPT4 outperforms DocLLM in VQA, possibly due to the higher complexity of reasoning and abstraction involved in VQA datasets compared to tasks like KIE or CLS.[9] DocLLM-1B demonstrates performance close to that of our larger model, suggesting that the smaller model can derive significant benefits from the architecture of DocLLM. 

**STDD Setting** . Table 3 shows that our model demonstrates superior performance compared to Llama2 across four out of five datasets, and achieves the best score overall for two of them (KIE task again). DocLLM also outperforms mPLUGDocOwl on DocVQA and both mPLUG-DocOwl and UReader on KLC, despite both baselines having been instruction-tuned on these datasets. However, it is important to note that classification accuracy is notably lower in our model. This discrepancy may stem from the fact that our model has been trained using only one CLS dataset, limiting its ability to generalize effectively to new datasets. 

**Qualitative Comparisons** . Figure 3 shows qualitative examples, comparing the outputs of DocLLM-7B and GPT4. Figure 3a corresponds to a KIE instruction, showing that DocLLM can provide correct answers when a question requires some knowledge of the semantic nuances of enterprise documents. DocLLM’s spatial reasoning abilities are demonstrated in Figure 3b, where the model correctly locates the heading _‘emergency protein allowances’_ and identifies the text immediately underneath it. Figure 3c highlights a limitation, with the model failing at a counting task, at which GPT4 succeeds. See Appendix C.1 for more examples. 

**Ablation Analysis** . We conduct ablation studies based on Next Token Prediction (NTP) accuracy 

> 9See Appendix C.2 for further details. 

to validate the main contributions of DocLLM. We observe that incorporating the spatial modality in the attention mechanism performs better over the classical text-only modality, thereby validating the utility of disentangled spatial attention (See Table 4). Furthermore, block infilling with spatial modality outperforms causal learning, highlighting the value of fill-in-the-middle objectives (See Table 5). Appendix D contains more details. 

## **5 Discussion** 

**Impact** . DocLLM enables language models to go beyond plain text settings and offers immediate utility in visually rich document understanding tasks. By accommodating complex layout structures, DocLLM allows documents with rich layouts to be included in the pre-training corpus without requiring extensive preprocessing. The explicit modeling of spatial relationships enables perceiving the documents as inherently structured knowledge. 

**Flexibility** . The support for multi-page documents, implemented through page breaks and document boundaries, enhances the model’s ability to comprehend documents with diverse lengths. This overcomes the constraints of small multimodal models that can handle only a single page and multimodal LLMs mainly designed for images. 

**Limitations** . The use of English-language datasets derived from limited enterprise domains (such as IIT-CDIP) may introduce inherent representational biases in VRDU models, including DocLLM. Also, DocLLM may be vulnerable to inaccurate bounding box information produced by an OCR engine.[10] However, several modern off-the-shelf solutions can robustly extract text from documents (Hegghammer, 2022), mitigating this issue. DocLLM’s support for long-form documents is restricted by its context length. Increasing the model size and allowing unbounded context length during inference can address this limitation. DocLLM is trained and designed to capture the characteristics of enterprise documents. Consequently, its efficacy may be restricted when applied to documents outside this domain, such as presentation decks or marketing reports. Finally, DocLLM may not excel at complex reasoning tasks, especially those requiring a deep understanding of numerical concepts. See Appendix F for additional discussion. 

> 10Appendix E studies the robustness of DocLLM to noisy input bounding boxes. 

8536 

## **6 Conclusions** 

We introduced DocLLM, a lightweight extension to traditional LLMs, tailored for generative reasoning over documents with rich layouts. DocLLM eschews expensive image encoders and instead utilizes bounding box information to capture the spatial layout structure of documents. This is achieved through a disentangled attention mechanism that models cross-alignment between text and spatial modalities. Notably, our model addresses the challenges posed by irregular layouts and heterogeneous content using a learning to infill pre-training objective. Tuning the model on a carefully curated instruction dataset provides a flexible interface for interactions. Our evaluation across various document intelligence tasks demonstrates that DocLLM surpasses equivalent models both for in-domain and out-of-domain datasets. In the future, we plan to infuse vision into DocLLM in a lightweight manner. 

## **Acknowledgments** 

This paper was prepared for information purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co and its affiliates (“JP Morgan”), and is not a product of the Research Department of JP Morgan. J.P. Morgan makes no representation and warranty whatsoever and disclaims all liability for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation, offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful. © 2024 JP Morgan Chase & Co. All rights reserved. 

## **References** 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ . 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large 

vision-language model with versatile abilities. _CoRR_ , abs/2308.12966. 

- Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, and Mark Chen. 2022. Efficient training of language models to fill in the middle. _arXiv preprint arXiv:2207.14255_ . 

- Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez, Marçal Rusiñol, Minesh Mathew, C. V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. 2019. ICDAR 2019 competition on scene text visual question answering. In _2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019_ , pages 1563–1570. IEEE. 

- Lukasz Borchmann, Michal Pietruszka, Tomasz Stanislawek, Dawid Jurkiewicz, Michal Turski, Karolina Szyndler, and Filip Gralinski. 2021. DUE: end-toend document understanding benchmark. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ . 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. 

- Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2020. Tabfact: A large-scale dataset for table-based fact verification. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ . 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. _CoRR_ , abs/2210.11416. 

8537 

- Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei. 2021. Document ai: Benchmarks, models and applications. _arXiv preprint arXiv:2111.08609_ . 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. 2023. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. _CoRR_ , abs/2305.06500. 

- Brian L. Davis, Bryan S. Morse, Brian L. Price, Chris Tensmeyer, Curtis Wigington, and Vlad I. Morariu. 2022. End-to-end document recognition and understanding with dessurt. In _Computer Vision - ECCV 2022 Workshops - Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part IV_ , volume 13804 of _Lecture Notes in Computer Science_ , pages 280–296. Springer. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ . 

- Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2021. Glm: General language model pretraining with autoregressive blank infilling. _arXiv preprint arXiv:2103.10360_ . 

- Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis. 2015. Evaluation of deep convolutional nets for document image classification and retrieval. In _13th International Conference on Document Analysis and Recognition, ICDAR 2015, Nancy, France, August 23-26, 2015_ , pages 991–995. IEEE Computer Society. 

- Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. Deberta: Decoding-enhanced bert with disentangled attention. In _International Conference on Learning Representations_ . 

- Thomas Hegghammer. 2022. Ocr with tesseract, amazon textract, and google document ai: a benchmarking experiment. _Journal of Computational Social Science_ , 5(1):861–882. 

- Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4083–4091. 

- Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. 2019. Icdar2019 competition on scanned receipt ocr and information extraction. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_ , pages 1516–1520. 

- Guillaume Jaume, Hazim Kemal Ekenel, and JeanPhilippe Thiran. 2019. FUNSD: A dataset for form understanding in noisy scanned documents. In _2nd International Workshop on Open Services and Tools for Document Analysis, OST@ICDAR 2019, Sydney, Australia, September 22-25, 2019_ , pages 1–6. IEEE. 

- Marcin Kardas, Piotr Czapla, Pontus Stenetorp, Sebastian Ruder, Sebastian Riedel, Ross Taylor, and Robert Stojnic. 2020. AxCell: Automatic extraction of results from machine learning papers. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 8580– 8594, Online. Association for Computational Linguistics. 

- Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. Ocr-free document understanding transformer. In _Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII_ , page 498–517, Berlin, Heidelberg. Springer-Verlag. 

- Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ . 

- Arjun Reddy Kunduru. 2023. From data entry to intelligence: Artificial intelligence’s impact on financial system workflows. _International Journal on Orange Technologies_ , 5(8):38–45. 

- Jordy Van Landeghem, Rubèn Tito, Lukasz Borchmann, Michal Pietruszka, Pawel Józiak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, Matthew B. Blaschko, Sien Moens, and Tomasz Stanislawek. 2023. Document understanding dataset and evaluation (DUDE). _CoRR_ , abs/2305.08455. 

- Chen-Yu Lee, Chun-Liang Li, Timothy Dozat, Vincent Perot, Guolong Su, Nan Hua, Joshua Ainslie, Renshen Wang, Yasuhisa Fujii, and Tomas Pfister. 2022. FormNet: Structural encoding beyond sequential modeling in form document information extraction. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3735–3754, Dublin, Ireland. Association for Computational Linguistics. 

- Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. 2023. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 18893–18912. PMLR. 

- D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and J. Heard. 2006. Building a test collection 

8538 

for complex document information processing. In _Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval_ , SIGIR ’06, page 665–666, New York, NY, USA. Association for Computing Machinery. 

- Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si. 2021. StructuralLM: Structural pre-training for form understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 6309– 6318. Association for Computational Linguistics. 

- Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. 2022. Dit: Self-supervised pre-training for document image transformer. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 3530–3539. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ . 

- Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. 2020. DocBank: A benchmark dataset for document layout analysis. In _Proceedings of the 28th International Conference on Computational Linguistics_ , pages 949–960, Barcelona, Spain (Online). International Committee on Computational Linguistics. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ . 

- Tianyang Liu, Fei Wang, and Muhao Chen. 2023b. Rethinking tabular data understanding with large language models. _CoRR_ , abs/2312.16702. 

- Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, and Xiang Bai. 2023c. On the hidden mystery of OCR in large multimodal models. _CoRR_ , abs/2305.07895. 

- Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. _arXiv preprint arXiv:2308.08747_ . 

- Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In _Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 2263–2279. Association for Computational Linguistics. 

- Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. 2022. Infographicvqa. In _IEEE/CVF Winter Conference_ 

_on Applications of Computer Vision, WACV 2022, Waikoloa, HI, USA, January 3-8, 2022_ , pages 2582– 2591. IEEE. 

- Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. Docvqa: A dataset for VQA on document images. In _IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021_ , pages 2199–2208. IEEE. 

- Zihang Meng, Licheng Yu, Ning Zhang, Tamara L Berg, Babak Damavandi, Vikas Singh, and Amy Bearman. 2021. Connecting what to say with where to look by modeling human attention traces. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12679–12688. 

- Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Nick Barnes, and Ajmal Mian. 2023. A comprehensive overview of large language models. _CoRR_ , abs/2307.06435. 

OpenAI. 2023. Gpt-4 technical report. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In _NeurIPS_ . 

- Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. 2019. Cord: A consolidated receipt dataset for post-ocr parsing. 

- Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing of the Asian Federation of Natural Language Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long Papers_ , pages 1470– 1480. The Association for Computer Linguistics. 

- Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. _arXiv preprint arXiv:2306.01116_ . 

- Tianxiao Shen, Hao Peng, Ruoqi Shen, Yao Fu, Zaid Harchaoui, and Yejin Choi. 2023. Film: Fill-in language models for any-order generation. _arXiv preprint arXiv:2310.09930_ . 

- Tomasz Stanislawek, Filip Gralinski, Anna Wróblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemyslaw Biecek. 2021. 

8539 

Kleister: Key information extraction datasets involving long documents with complex layouts. In _16th International Conference on Document Analysis and Recognition, ICDAR 2021, Lausanne, Switzerland, September 5-10, 2021, Proceedings, Part I_ , volume 12821 of _Lecture Notes in Computer Science_ , pages 564–579. Springer. 

- Stacey Svetlichnaya. 2020. Deepform: Understand structured documents at scale. 

- Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. 2021. Visualmrc: Machine reading comprehension on document images. In _Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021_ , pages 13878–13888. AAAI Press. 

- Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. 2023. Unifying vision, text, and layout for universal document processing. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023_ , pages 19254–19264. IEEE. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015_ , pages 4566–4575. IEEE Computer Society. 

- Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee, and Sandeep Tata. 2023. VRDU: A benchmark for 

visually-rich document understanding. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023_ , pages 5184– 5193. ACM. 

- Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned language models are zero-shot learners. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net. 

- Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. 2023. Visual chatgpt: Talking, drawing and editing with visual foundation models. _arXiv preprint arXiv:2303.04671_ . 

- Te-Lin Wu, Cheng Li, Mingyang Zhang, Tao Chen, Spurthi Amba Hombaiah, and Michael Bendersky. 2021. Lampret: Layout-aware multimodal pretraining for document understanding. _arXiv preprint arXiv:2104.08405_ . 

- Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. 2021. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 2579–2591, Online. Association for Computational Linguistics. 

- Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. Layoutlm: Pre-training of text and layout for document image understanding. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , pages 1192–1200. 

- Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. 2023a. mplug-docowl: Modularized multimodal large language model for document understanding. _CoRR_ , abs/2307.02499. 

- Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Alex Lin, and Fei Huang. 2023b. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. _CoRR_ , abs/2310.05126. 

- Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. 2023c. mplug-owl: Modularization empowers large language models with multimodality. _CoRR_ , abs/2304.14178. 

8540 

- Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. 2023d. Large language models are versatile decomposers: Decomposing evidence and questions for table-based reasoning. In _Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023_ , pages 174–184. ACM. 

- Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. 2024. Investigating the catastrophic forgetting in multimodal large language model fine-tuning. In _Conference on Parsimony and Learning_ , pages 202–227. PMLR. 

- Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. 2023a. Llavar: Enhanced visual instruction tuning for textrich image understanding. _CoRR_ , abs/2306.17107. 

- Zhenrong Zhang, Jiefeng Ma, Jun Du, Licheng Wang, and Jianshu Zhang. 2023b. Multimodal pre-training based on graph attention network for document understanding. _IEEE Trans. Multim._ , 25:6743–6755. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ . 

- Ran Zmigrod, Dongsheng Wang, Mathieu Sibue, Yulong Pei, Petr Babkin, Ivan Brugere, Xiaomo Liu, Nacho Navarro, Antony Papadimitriou, William Watson, Zhiqiang Ma, Armineh Nourbakhsh, and Sameena Shah. 2024. Buddie: A business document dataset for multi-task information extraction. _CoRR_ , abs/2404.04003. 

8541 

## **A Dataset Details** 

## **A.1 Preprocessing** 

Of the datasets used in our study, IIT-CDIP and DocBank do not provide token-level OCR output. Therefore we process both datasets using the Tesseract-OCR engine.[11] For the remaining datasets, we used the OCR output provided by each publisher. 

## **A.2 Instruction Tuning Templates** 

For the extraction template, we add a “None” answer if a key does not exist in the given document, following Ye et al. (2023a,b). As described in Section 3.4 and Table 1, to increase diversity in the training data, we derive internal classification and MCQ instructions in addition to extraction instructions from the original KIE annotations. However, to stay consistent with benchmarks from previous work (Ye et al., 2023a,b), we only keep the prompts derived from the extraction template in the test split of each KIE dataset. To avoid the cold start problem induced by potentially unseen types of documents in testing or production usage, we only keep the MCQ prompts for the test split of each CLS dataset. Note that when a prompt accepts more than one answer, we create multiple copies of the prompt with one acceptable answer assigned to each. We only perform this “flattening” operation in the training split of the dataset. 

## **A.3 Dataset Statistics** 

See Table 6 for pretraining dataset details and Table 7 for instruction tuning dataset details. 

## **B Training Details** 

DocLLM-1B is composed of 24 layers, each with 16 attention heads and a hidden size of 1,536. DocLLM-7B comprises 36 layers, 32 heads, and a hidden size of 4,096. Using pretrained weights as the backbone for the text modality, we extend the Falcon-1B and Llama2-7B models by adding the disentangled attention and block infilling objective as described in Section 3. We start directly from the pretrained weights of the backbone LLMs in order to continue their pretraining in a multimodal manner and avoid catastrophic forgetting of instruction following abilities (Luo et al., 2023; Zhai et al., 2024). 

> 11https://github.com/tesseract-ocr/tesseract 

Table 6: Pretraining dataset statistics. 

||**Dataset**|**#Docs**|**#Pages**|**#Tokens**|
|---|---|---|---|---|
||**CDIP**<br>**DocBank**<br>**Total**|5,092,636<br>499,609<br>5,592,245|16,293,353<br>499,609<br>16,792,962|3,637,551,478<br>228,362,274<br>3,865,913,752|



Table 7: Instruction tuning dataset statistics. 

||**Task**|**#Train prompts**|**#Test prompts**|
|---|---|---|---|
||VQA|145,090|24,347|
||NLI|104,360|12,720|
||KIE|236,806|38,039|
||CLS<br>**Total**|149,627<br>635,883|21,813<br>96,919|



For DocLLM-1B, we use a pre-training learning rate of 2 _×_ 10 _[−]_[4] with 1,000 warmup steps, employing a cosine scheduler, and Adam optimizer (Kingma and Ba, 2015) with _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 96 and a weight decay of 0.1. For instruction tuning we use a learning rate of 1 _×_ 10 _[−]_[4] with 500 warmup steps and a cosine scheduler, and the same parameters for weight decay and Adam optimizer as the pre-training phase. The Adam epsilon is set to 1 _×_ 10 _[−]_[5] . We pretrain for one epoch, and instruction-tune for a total of 10 epochs. 

For DocLLM-7B, pretraining involves a learning rate of 3 _×_ 10 _[−]_[4] with 1,000 warmup steps and cosine scheduler, weight decay of 0.1, and Adam optimizer with _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 95. Instruction tuning uses a learning rate of 1 _×_ 10 _[−]_[4] with 500 warmup steps and a cosine scheduler, weight decay of 0.1, and Adam optimizer with _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 95. Adam epsilon is set at 1 _×_ 10 _[−]_[6] . We conduct one epoch of pretraining, followed by three epochs of instruction tuning, considering available computing resources. 

The DocLLM-7B models are trained with 16-bit mixed precision on 8 24GB A10G GPUs using fully sharded data parallelism, implemented with the Accelerate library.[12] The DocLLM-1B model, on the other hand, is trained on a single 24GB A10G GPU. 

Table 8 provides an overview of the model configuration and training hyper-parameters that were used. 

> 12https://huggingface.co/docs/accelerate 

8542 

Table 8: Model configuration and training hyperparameters setting for DocLLM-1B and -7B. 

||**DocLLM-1B**|**DocLLM-1B**|**DocLLM-1B**|**DocLLM-7B**|**DocLLM-7B**|
|---|---|---|---|---|---|
|Backbone|Falcon-1B (|Falcon-1B (Penedo et al.,2023)||Llama2-7B (Touvron et al.,2023)||
|#Parameters|1,524,963,328|||7,853,019,136||
|Layers||24|||36|
|Attention heads||16|||32|
|Hidden size||1,536|||4,096|
|Precision||bfloat16|||bfloat16|
|Batch size||2|||5|
|Max context length||1,024|||1,024|
||**Pretraining**|**Instruction tuning**||**Pretraining**|**Instruction tuning**|
|Learning rate|2_×_10_−_4|1_×_|_×_10_−_4|3_×_10_−_4|1_×_10_−_4|
|Warmups|1,000||500|1,000|500|
|Scheduler type|cosine||cosine|cosine|cosine|
|Weight decay|0.1||0.1|0.1|0.1|
|Adam_β_s|(0.9, 0.96)|(0.9,0.96)||(0.9,0.95)|(0.9,0.95)|
|Adam epsilon|1_×_10_−_5|1_×_|_×_10_−_5|1_×_10_−_6|1_×_10_−_6|



**==> picture [432 x 267] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Prompt: What is the doctor’s (b) Prompt: What is the value for (c) Prompt: What is the value for<br>id no.? the “contract num”? the “gross amount”?<br>DocLLM: 162 DocLLM: 1328762 DocLLM: None<br>GPT4: No information provided GPT4: 09732930 GPT4: 40,000.00<br>= itt, Gives bamuse py fasts tangy guring duty Estimate for R.J. Reynolds<br>ebeces Ba2e00, MaDe MPs | aaatatant Professor es/0ws3 eepachag paereaey alten allt) EEiages ate Creditability Enhancenent’ Project<br>~ERaSiepBaltioore,JomaSSHopkinsofbaryianéruneUniversitGeddoa,avo Learnoregon EE||e]| wela.en| cowieWB || Taeecupational| ae shemeHEShe.peltia’ epieNPsUi aaaPent1:pee2a [Seesoge] *enetaaeaS agaieTecaeiatBoiattoe aE?wsteeanReteee acar BehTs Ho Hisglibeltbee.She ReeeveeMaterials-avel se archarch FocustwoFoouspeeGroupseGroups,a metiaisthreé cities $30,000$20,000z<br>{Stace seeetmenbwamo on any Pca Crear ous sen on un nara wane te avait (20) sh sat Fr Bl Wiss $67,000<br>Seiwa mee omm mm veanmee Ind. (8) Bits ‘elas at care eS<br>sroressrOn ES. 13 Hie Be Ee ie Eetiate to be reconciled against actual hours and costs incurred.<br>1979 = 1982 Internship and Residency, Iatersal Madieine, The Johns BG Ree eee at er<br>198298s --= 19851886 oontpeHopkinsJointfodicine,esearchFellowshiponeUniversstyDivisionFellowship,Hetisine [of] in [Allergy] PulsonaryThe Johns [sod] [Clinical] andHopkinsResidency [Temaoloey] School inof ma.a:ieHose.nee (enBRgpBbOy nero?BeOBE2,802,640Heals were46,696eohidBey 108138947,661,RG!ShsSeed PE2pDatettne ma.MS:EsBeaBie<br>SGhoctor Marylandor Modietne,EnvironsentelWediealResearchDirectoryFacilityThe University Bixa. anap czas.Re ricyate Achne ‘ioe fehookeor<br>1986tg87 Clintos!Young‘Hlotaré Investigatort.ZamnoiceyBlley ScholarAuard,Award,AxericanAnertcanAcadesyLung Asscedattonof Allergy and £5ewEee fh3 Rieen7yaag tHtoonseet seenRat3 tefuygHOESheHewes<br>aneSev Boopir  eerreeysansstivityDin 1985; 131:463-865.peoumenitia fron aiphenyisethane auiscoyamate, Am hertisth 83is) SSsere ISPpe IeeH wiSpo taeOBE<br>BascomsensitivityBomcocenateietion,‘mosesrespiratoryBady. R,tmR,YangIndBleeckersyaptonsHygin¥, AaaceBascomasthmaticsdm E:inRev9 %,rubber1986;RespirDistilledBaser92551-559.andinjectionDiaM, 1986;relationshipSakerwater12H:208-253.pressJ:inducedPeripheraloperators:todronchoconstriction:exercise-inducedeosinephsiiaA case-controland zeeHe:* of CigaretSHUEa"EheFeleantsNUgEiiotlstates taxshowingin SpatrateseveraiSPEgursieesiassincreaseincreased statesAInde fatstefromthevelunea"tgerteigaret7¢ perfrensamenpackagetandulySpansis, 1971Teturnedto 9.1¢HeheWashesper cereInshoiepackagee# ‘orShaeffectivein partatlneto7/1/72.politicalEoDee gan cf5=<br>ee ~ TIMS 0010003 .<br>50569673<br>(d) DocLLM: resume (e) DocLLM: budget (f) DocLLM: budget<br>GPT4: form GPT4: scientific report GPT4: invoice<br>**----- End of picture text -----**<br>


Figure 4: Qualitative examples of DocLLM-7B performance versus a SotA baseline (GPT4). Correct answers are highlighted in blue and incorrect answers are highlighted in red. (a): VQA example from the DocVQA dataset (Mathew et al., 2021). (b)-(c): KIE examples from the DeepForm dataset (Svetlichnaya, 2020). (d)-(f): CLS examples from the RVL-CDIP dataset (Harley et al., 2015). The prompt used here was: What type of document is this? Possible answers: [letter, memo, email, file folder, form, handwritten, invoice, advertisement, budget, news article, presentation, scientific publication, questionnaire, resume, scientific report, specification]. 

8543 

## **C Detailed Performance Analysis** 

## **C.1 Qualitative Examples** 

Figure 4 shows additional qualitative examples from the DocLLM-7B output, where 4a highlights a VQA example from the DocVQA dataset (Mathew et al., 2021), 4b and 4c display two KIE examples from the DeepForm dataset (Svetlichnaya, 2020), and the bottom row shows CLS examples from the RVL-CDIP dataset (Harley et al., 2015). 

As Figures 4a and 4e show, DocLLM can provide correct answers when the question requires some knowledge of the semantic nuances of enterprise documents. As an example, in Figure 4e, GPT4 mislabels a tax report issued by a local tax council as a scientific report, possibly due to the numeric contents of the table, whereas DocLLM is able to associate the content and the corresponding issuing authority with a budget report. Figure 4b demonstrates DocLLM’s spatial reasoning capability. The rightmost column of Figure 4 shows examples of failure by DocLLM. Each failure case demonstrates a limitation in the design and scope of the model. Figure 4c shows an example where DocLLM is unable to extract the gross amount. This error is due to the fact that the correct answer falls outside of the context window of the model, as it is located on the fourth page of a multi-page document. Lastly, Figure 4f shows an example for which the class predicted by DocLLM, i.e. “budget”, is semantically viable, but is nevertheless not the correct class. In future studies, we plan to address some of the above mentioned limitations, and increase the context length of the model. 

## **C.2 DocVQA Deep-Dive** 

We conduct an in-depth analysis of the performance of DocLLM-7B on the various question categories of DocVQA. Table 9 lists the categories under which the DocVQA questions are listed. The “M” column identifies the modality that is expected to be uniquely helpful to the corresponding category. As an example, the “Figure/Diagram” category includes questions over charts and diagrams the answers to which rely on visual reasoning, e.g. “What is the variable taken along the x axis?” Whereas the “Form” category includes questions the answers to which require reasoning over layout, e.g. “What is the text at the top right corner of the page??”[13] 

Table 9: DocLLM-7B scores for DocVQA categories. 

|**Category**|**M**|**ANLS**|
|---|---|---|
||||
|Figure/Diagram<br>Form<br>Table/List<br>Layout<br>Free text<br>Image/Photo<br>Handwritten<br>Yes/No<br>Other|V<br>L<br>L<br>L<br>T<br>V<br>T<br>-<br>-|41.4<br>82.2<br>66.2<br>72.4<br>64.6<br>47.8<br>62.8<br>43.9<br>56.8|



Needless to say, all modalities are often crucial to answering all question types. The modalities listed in the table are those expected to offer a uniquely important signal to the model. 

As depicted in Table 9, DocLLM exhibits strong performance on “Form” and “Layout” questions, attaining scores of 82.2 and 72.4 respectively. These results underline the model’s proficiency in understanding and processing structured document formats and layouts. Conversely, the "Image/Photo", "Figure/Diagram", and "Yes/No" questions have lower scores of 47.9, 41.4, and 43.9 respectively. The absence of integrated vision features might account for DocLLM’s lower capacity in recognizing certain visual cues. Overall, DocLLM shows the strongest performance when reasoning over layout is key, and the weakest performance when visual reasoning is key. 

Table 10: DocLLM-7B performance comparison against GPT4+OCR and GPT4V. BuDDIE KIE GPT4V results were obtained on a sample of 5K (cost & API limits). 

|**Model**<br>**Setting**|**DocVQA**<br>**BuDDIE**<br>VQA<br>VQA<br>KIE<br>CLS|
|---|---|
|||
|GPT4+OCR<br>ZS<br>GPT4V<br>ZS|82.8<br>76.4<br>66.1<br>84.9<br>**88.4**<br>67.9<br>70.0<br>86.0|
|||
|DocLLM-7B<br>SDDS<br>DocLLM-7B<br>STDD|69.5<br>**86.7**<br>**96.0**<br>**99.4**<br>63.4<br>73.3<br>72.6<br>31.1|



## **C.3 GPT4V Performance Comparison** 

Given the recent roll out of the GPT4V API[14] and the interest it has generated, we also benchmark DocLLM-7B against GPT4V on DocVQA and BuDDIE (Table 10). We select these datasets in order to include both SDDS and STDD results in the com- 

the vision modality could play a more crucial role. 

13Note that the text modality has been highlighted for the “Handwritten” category because the DocVQA dataset provides OCR output for all documents. In the absence of OCR output, 

14https://openai.com/blog/ 

new-models-and-developer-products-announced-atdevday 

8544 

parison. Moreover, as BuDDIE was not publicly released before this work, we can be certain that GPT4 and GPT4V were not trained on it. Due to cost and daily API usage limitations, we were not able to cover additional datasets. 

We first observe that GPT4V does not uniformly outperform GPT4+OCR on the datasets considered. Both models show close ZS performance in BuDDIE CLS, but GPT4+OCR beats GPT4V in BuDDIE VQA while GPT4V tops GPT4+OCR on BuDDIE KIE and DocVQA. The additional vision component of GPT4V seems to help in general, especially for datasets such as DocVQA. However, as the characteristics of these model are undisclosed, analyzing their performance differences in depth is difficult. We do note that, despite its lack of visual and spatial features, GPT4+OCR fares well on VQA, KIE, and CLS tasks, and might be able to partially model the spatial relationships in documents based on the natural ordering of OCR tokens. Its robustness to OCR token position permutations is however not guaranteed. 

Next, we observe that DocLLM-7B also outperforms GPT4V in addition to GPT4+OCR on BuDDIE SDDS. In the STDD evaluation setting, which is closer to out-of-distribution ZS inference, our model still exhibits competitive performance in VQA and KIE – although not consistently exceeding the scores of the likely larger GPT4 models. DocLLM’s lack of vision encoder appears to be mostly detrimental on DocVQA, where it particularly struggles on “Image/Photo” and “Figure/Diagram” questions, as seen in Section C.2. 

## **C.4 SotA Performance Comparison** 

In Table 11, we compare DocLLM-7B against the SotA on the datasets considered in this paper. Note that BuDDIE is not included here as it was not publicly released before this work. Similarly, DUDE and VRDU ad-buy are not considered in this section, since we used validation and bespoke splits respectively to evaluate models on them (see the caption on Table 2). FUNSD and PWC are also excluded from this study, as the prompts we built for these datasets leveraged annotations differently than previous work: our FUNSD KIE questions are based on the annotated key-value links, and our PWC KIE questions are formulated using the annotated set of Machine Learning tasks covered by the dataset. 

Table 11 offers a few notable takeaways. First, despite the recent progress in multi-modal docu- 

ment understanding, a foundation model that outranks others across a wide range of tasks and datasets does not currently exist. Most SotA models are single-task fine-tuned models that outperform others in one or a few datasets, as seen here with LayoutT5 (Tanaka et al., 2021), StructuralLM (Li et al., 2021), PASTA+DATER (Ye et al., 2023d), GPT-3.5+DP+PyAgent+MixSC (Liu et al., 2023b), and GraphDoc (Zhang et al., 2023b). The same observation applies to general NLP (Brown et al., 2020; Wang et al., 2022; Chowdhery et al., 2022; Naveed et al., 2023). While UDOP tops all models on three KIE datasets, it remains an expert model that requires dataset-specific prompts and per dataset fine-tuning (on top of its multitask supervised pretraining) in order to reach the performance reported. Similarly, Pix2Struct, a specialist model with a ViT encoder and a text decoder pretrained on 80M website screenshots, exhibits strong performance on DocVQA (76.6 for its 1.3Bparameter variant), but underperforms in chart and infographics understanding compared to generalist multimodal LLMs such as UReader. Its performance on KIE and CLS datasets is also understudied. On table-based datasets such as WTQ and TabFact, SotA models rely on large, text-only LLMs to reason over data using SQL or Pandas – thus reducing their ability to generalize to nontabular document data. The abstractive reasoning limitations of DocLLM-7B are more apparent on these table-based datasets, but our single model performs competitively in KIE and CLS (even on KLC and Deepform, despite DocLLM’s relatively short context-length). 

Second, recent multimodal LLMs such as QwenVL-Max[15] and GPT4V[16] show impressive ZS performance in VQA. These generalist models report strong performance on DocVQA and other datasets like ChartQA (Masry et al., 2022) and InfographicVQA (Mathew et al., 2022) (which we do not consider in this paper) thanks to their additional vision encoder.[17] However, the lack of transparency about their size, exact architecture, training procedure, and training data makes it hard to draw any conclusions. On DocVQA, DocLLM-7B outperforms Qwen-VL-10B (Bai et al., 2023). Moreover, as these recent multimodal LLMs were designed 

> 15https://qwenlm.github.io/blog/qwen-vl/ 

> 16https://openai.com/research/gpt-4 

> 17In future studies, we hope to equip DocLLM with access to the vision modality too — albeit in a more efficient manner than is typically implemented. 

8545 

Table 11: DocLLM-7B (SDDS) performance comparison against SotA models. 

||**Dataset**|||||**Model**|**Modality**|**SotA**|||||**DocLLM-7B**|
|---|---|---|---|---|---|---|---|---|---|
||DocVQA|||**Qwen-VL-Max**<br>(qwenlm.github.io/blog/qwen-vl)|T+V|93.1|||69.5|
|**VQA**|WTQ_(Accuracy)_|||**GPT-3.5+DP+PyAgent+MixSC**<br>(Liu et al.,2023b)|T|73.6|||27.1|
||VisualMRC_(CIDEr)_|||**LayoutT5**<br>(Tanaka et al.,2021)|T+V+L|364.2|||264.1|
|**NLI**|TabFact|||**PASTA+DATER**<br>(Ye et al.,2023d)|T|93.0|||66.4|
||KLC|||**UDOP**<br>(Tang et al.,2023)|T+V+L|82.8|||60.3|
|**KIE**|CORD<br>DeepForm|||**UDOP**<br>(Tang et al.,2023)<br>**UDOP**<br>(Tang et al.,2023)|T+V+L<br>T+V+L|97.6<br>85.5|||67.4<br>75.7|
||SROIE|||**GraphDoc**<br>(Zhang et al.,2023b)|T+V+L|98.45|||91.9|
|**CLS**|RVL-CDIP|||**StructuralLM**<br>(Li et al.,2021)|T+L|96.1|||91.8|



to tackle a wide range of tasks (e.g., image captioning) and not just DocAI, their ZS performance on certain tasks considered here (document NLI, KIE, CLS) has not been investigated – making a thorough comparison with our model even more complex. 

Finally, despite lower performance compared to the top-performing model in each category, DocLLM still shows superior performance to generalist LLMs of comparable size, as indicated in Table 2. The model also proves robust to out-ofdistribution data in ZS, as demonstrated in Table 3. 

## **D Ablation Studies** 

We conduct ablation studies to validate the three main contributions of DocLLM: (1) disentangled spatial features, (2) the block infilling pre-training objective, and (3) the masking strategy used for decoding. For all ablations, we use Next Token Prediction (NTP) out-of-sample accuracy to compare configurations at the pre-training stage. Due to resource restrictions, each experiment uses a subset of our pre-training corpus: we randomly sample 100,000 chunks and predict on 1,000 unseen documents. A chunk is a collection of documents wherein the total number of tokens across the collection is less than the maximum input context length. The hyperparameters are set consistently following Table 8 across all ablation experiments. 

**==> picture [68 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Causal decoder<br>Key<br>54 <bMask> USA <infill> Doctor<br>(b) Prefix decoder<br>**----- End of picture text -----**<br>


Figure 5: A simplified illustration of attention masks for causal-decoder and prefix-decoder for block infilling. 

**Disentangled Spatial Attention** . To measure the effect of disentangled spatial attention on cross- 

8546 

Figure 6: Performance comparison on NTP between causal decoder and prefix decoder. 

modal interactions, we train the models by setting the _λ_ hyperparameter in Eq 4 to 0 or 1. Table 4 enumerates the attention combinations, and the results suggest that keeping only the spatial-to-spatial interaction (i.e. _λs,s_ = 1) yields the highest NTP accuracy. The performance differences among other configurations, such as text-to-spatial and spatialto-text, are subtle. Notably, the vanilla text-only self-attention mechanism yields the lowest NTP accuracy, underlining the importance of incorporating spatial features for understanding documents with rich layouts. For all experiments in Section 4, we therefore set _λs,s_ = 1, _λs,t_ = 0, and _λt,s_ = 0. We opt for simplicity by choosing a hard mode over a soft one while acknowledging the potential advantage of flexibility for the latter. 

. To evaluate the ef- **Autoregressive Block Infilling** fectiveness of the proposed autoregressive block infilling objective especially comparing with the conventional left-to-right causal learning, we benchmark three configurations in our ablation study: (1) causal learning, (2) causal learning with spatial modality, and (3) block infilling with spatial modality. As highlighted in Table 5, autoregressive block infilling exhibits the best performance. Additionally, the performance gain of adding the spatial modality to the causal learning proves the advantage of the spatial modality. 

**Prefix Decoder and Causal Decoder** . For document-conditioned generation, an intuitive choice is to employ a prefix decoder with prefix masking that utilizes bidirectional attention mechanism for the entire document, as illustrated in Figure 5b. We investigate this assumption through experiments where we compare a prefix decoder against the conventional causal decoder. Specifically, we conduct experiments on these two decoders for different settings outlined in the **Disentangled Spatial Attention** ablation to study their resulting performance. 

The results in Figure 6 show marginal differences between these two decoders across the five configurations, with the causal decoder having a slight edge over the prefix. The minor difference suggests that both masking methods are comparable in modeling documents. Thus the bidirectional attention enabled by the prefix decoder may not be crucial in this context, and we consequently elect to use a causal decoder for all experiments in section 4. 

## **E Robustness to inaccurate OCR Bounding Boxes** 

To assess DocLLM’s sensitivity to inaccurate token bounding boxes, we conduct experiments on DocVQA and inject variable amounts of noise to shift the borders of the original OCR data. Each border is shifted by _ϵ ∼N_ (0 _, l_[2] _σ_[2] ), where _l_ is the length of the sides orthogonal to the border considered and _σ_ is the hyperparameter we use to control the amount of noise injected. We clip tail values beyond _±_ 2 _lσ_ and restrict _σ_ to values between 0 and 0.25 to avoid accidentally swapping bounding box borders. 

We observe in Table 12 that DocLLM-1B’s performance on DocVQA remains very stable when input OCR borders are randomly shifted, highlighting the model’s robustness to moderately inaccurate spatial coordinates. 

Table 12: DocLLM-1B robustness to inaccurate bounding box information on DocVQA 

|_σ_(noise level)|0|0.125|0.25|
|---|---|---|---|
|DocLLM-1B|61.4|60.9|60.8|



## **F Additional Discussion** 

The main concept for a cohesive block is to ensure meaningful infilling during the pretraining phase, preventing disconnected predictions. However, the choice of OCR engines to obtain such cohesive blocks remains an open area for exploration. Practical comparisons with various OCR engines and/or layout parsers are left as future work, as LayoutLMs underscore the importance of accurate OCR for improved VQA results. They leverage the Microsoft Azure API, demonstrating superior performance compared to TesseractOCR, as indicated in the DocVQA leaderboard.[18] Consequently, 

> 18https://rrc.cvc.uab.es/?ch=17&com=evaluation& task=1 

8547 

researchers are also encouraged to utilize more accurate OCR engines for potential enhancements, if such resources are available. 

We have presented a collection of SDDS results alongside zero-shot outcomes. To mitigate prompt influence in the zero-shot results, a rigorous methodology was implemented. This involved the engagement of three independent prompt engineers, each undergoing five rounds of refinement for zero-shot settings, followed by a series of postprocessing techniques to enhance result reliability. The best results are thus obtained from each of the three groups. We still acknowledge the potential for refinement and improvement. 

We share some internal training experiences, acknowledging the absence of robust validation. First, we observe that a higher weight decay (e.g., 0.1 versus 0.01) generally improves performance in both pretraining and instruction tuning. During the instruction tuning phase, a higher initial learning rate, such as 1e-4 versus 5e-5, leads to enhanced performance. Overall, we’ve observed that the cosine scheduler tends to outperform linear or constant schedulers across various settings. 

8548 

