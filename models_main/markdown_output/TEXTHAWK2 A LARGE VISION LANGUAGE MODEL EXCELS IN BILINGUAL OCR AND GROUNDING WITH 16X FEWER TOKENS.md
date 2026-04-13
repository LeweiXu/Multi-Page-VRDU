Technical Report 

# TEXTHAWK2: A LARGE VISION-LANGUAGE MODEL EXCELS IN BILINGUAL OCR AND GROUNDING WITH 16X FEWER TOKENS 

**Ya-Qi Yu, Minghui Liao, Jiwen Zhang, Jihao Wu** Huawei Technologies Co., Ltd. 

{yuyaqi5, liaominghui1}@huawei.com 

## ABSTRACT 

Reading dense text and locating objects within images are fundamental abilities for Large Vision-Language Models (LVLMs) tasked with advanced jobs. Previous LVLMs, including superior proprietary models like GPT-4o, have struggled to excel in both tasks simultaneously. Moreover, previous LVLMs with fine-grained perception cost thousands of tokens per image, making them resource-intensive. We present TextHawk2, a bilingual LVLM featuring efficient fine-grained perception and demonstrating cutting-edge performance across general-purpose, OCR, and grounding tasks with 16 times fewer image tokens. Critical improvements include: (1) Token Compression: Building on the efficient architecture of its predecessor, TextHawk2 significantly reduces the number of tokens per image by 16 times, facilitating training and deployment of the TextHawk series with minimal resources. (2) Visual Encoder Reinforcement: We enhance the visual encoder through LVLM co-training, unlocking its potential for previously unseen tasks like Chinese OCR and grounding. (3) Data Diversity: We maintain a comparable scale of 100 million samples while diversifying the sources of pre-training data. We assess TextHawk2 across multiple benchmarks, where it consistently delivers superior performance and outperforms closed-source models of similar scale, such as achieving 78.4% accuracy on OCRBench, 81.4% accuracy on ChartQA, 89.6% ANLS on DocVQA, and 88.1% accuracy@0.5 on RefCOCOg-test. 

**==> picture [395 x 68] intentionally omitted <==**

**----- Start of picture text -----**<br>
— My father had a small estate in Nottinghamshire: I was the third of five<br>7i tech eae——, sons.whereHeI residedsent methreeto Emmanuelyears, and Collegeapplied myselfin Cambridgeto my studies,during autumn,but the<br>‘ebookenc eyen charge of maintaining me, although I had a very scanty allowance, being<br>45 add me oe mae te too great for a narrow fortune, I was bound apperently to Mr. James Bates,<br>ath hat en an officer in London, in whom I continued for four years. My father now<br>bund pr e testnn fist eget ay _—i and then sending me small sums of money, I laid them out in learning<br>hoveenteddenoting whoeteryasto Pave Myfieroslaysfoneensedi es Gulliver(Tunnel anan— || navigationas well as it would be, some time or other, or fortune to do when I left Mr.and other parts of the mathematics, useful to those who travel,<br>fete tad.hen let te nb e l e d ttewoud be,sme (Encoder(&(Resampler) ua) 3 \ Bates; I went down to my father's, where by the assistance of my uncle<br>andaasinineprs iypatrai ee io <— …32 tokens  andyearly pounds a year, I remained at Leyden; I studied two years and sevenmy John, and other relations, I got forty pounds, and a promise of a<br>pices errs ops neAEE months, knowing it would be useful in long voyages.<br>448 px<br>Big(Light<br>(Decoder)<br>224 px<br>**----- End of picture text -----**<br>


Figure 1: The magic of visual token compression. In this demonstration, TextHawk2 compresses 183 words displayed on a 448 _×_ 224 image, where each character measures under 8 pixels, into 32 tokens, allowing for accurate recovery. It’s reminiscent of the futuristic gadgets in _Doraemon_ anime. 

## 1 INTRODUCTION 

Over the past few years, significant advancements have been made in the realm of Large Language Models (LLMs) (Touvron et al., 2023; Zeng et al., 2024; Yang et al., 2024; DeepSeek-AI et al., 2024; Cai et al., 2024). These breakthroughs have also driven the development of Large Vision-Language Models (LVLMs) (Li et al., 2023b; Liu et al., 2023b; Wang et al., 2023c; Bai et al., 2023; Lu et al., 2024a; Chen et al., 2024b). LVLMs effectively combine visual and linguistic modalities, allowing them to understand visual content while leveraging the instruction-following and dialogue capabilities of LLMs. Over the past year, the rapid evolution of LVLMs, incorporating larger foundational LLMs and richer datasets, has significantly improved their ability to perform complex multimodal understanding and reasoning. Consequently, state-of-the-art LVLMs have achieved outstanding re- 

1 

Technical Report 

sults across various Visual Question Answering (VQA) benchmarks. However, to apply LVLMs to real-world scenarios beyond general VQA tasks, there is a need for a more refined perception of visual details, such as Optical Character Recognition (OCR) and object localization (Zhang et al., 2024d). These advanced capabilities are essential for applications in areas like document intelligence, Graphics User Interface (GUI) agents, and visual assistance for blind and low-vision users, where accurately interpreting and responding to detailed visual information is critical. 

Recent advancements in leading LVLMs have significantly improved their ability to recognize and interpret dense text within images (Li et al., 2024a; Hu et al., 2024a; Zeng et al., 2024; Dong et al., 2024; Chen et al., 2024b). GPT-4V, as the initial multimodal version of ChatGPT, demonstrates strong OCR capabilities for English text. However, it struggles with accurately interpreting Chinese text, often leading to hallucinations. This limitation has been significantly relieved in its successor, GPT-4o, which substantially improves its performance in handling non-English text, including Chinese. In the open-source domain, significant strides have also been made with models like InternVL series (Chen et al., 2024b). For instance, InternVL 1.2 increases its image resolution from 224 to 448, allowing it to capture finer details. This improvement is complemented by co-training the visual encoder on a mix of image captioning and OCR-specific datasets, boosting the model’s ability to recognize text effectively within images. Building on this progress, InternVL 1.5 employes an image cropping strategy that enables the dynamic processing of high-resolution images. 

However, text-oriented LVLMs often require processing a large number of tokens when handling high-resolution images, which results in significant computational costs and extensive context usage. This is due to the rapid increase in image tokens, making it crucial to compress them effectively. Despite this need, previous top-performing OCR models have only achieved an image compression ratio of up to 4, which is inadequate for practical applications. This raises the first question: _Can we increase the compression ratio to 16 without losing the ability to perceive fine-grained details and achieve state-of-the-art OCR performance with limited resources?_ 

Despite the impressive visual understanding and OCR performance shown by leading models like GPT-4o and InternVL 1.5, they still face challenges in achieving basic grounding capabilities. Unlike generalist LVLMs that often employ language-supervised visual encoders, grounding-oriented models (Liu et al., 2023c; You et al., 2023) typically rely on self-supervised visual encoders like DINOv2 (Oquab et al., 2023). However, an intriguing finding is that language-supervised visual encoders actually outperform self-supervised ones, particularly on OCR tasks, where the gap is notably wide (Tong et al., 2024). Some studies (You et al., 2023; Lin et al., 2023) have suggested combining multiple visual encoders, like CLIP (Radford et al., 2021) and DINOv2, to improve performance. Nonetheless, while these models aim to improve grounding capabilities, none guarantees strong performance across general multimodal understanding and OCR tasks. Additionally, using dual encoders leads to computational redundancy. This leads to the second question: _Can we train an LVLM with a single visual encoder that excels in general multimodal understanding, OCR, and grounding simultaneously?_ 

In this study, we delve into the previously mentioned questions, aiming to provide a comprehensive analysis and innovative solutions. Our key contributions are summarized as follows: 

- We introduce TextHawk2, a versatile LVLM that accommodates visual inputs of any resolution and demonstrates outstanding performance on fine-grained benchmarks, including OCRBench, ChartQA, DocVQA, InfoVQA, RefCOCO, and others. 

- We demonstrate that our thoughtfully designed resampler can compress visual tokens by a factor of 16 without compromising fine-grained perception capabilities. 

- We establish that, through effective data curation and reinforcement of the visual encoder, it is possible to achieve state-of-the-art performance in general multimodal understanding, OCR, and grounding simultaneously with a unified visual encoder. 

## 2 RELATED WORKS 

- 2.1 TEXT-ORIENTED LVLMS 

Text recognition or document understanding is a pivotal feature of LVLMs. Consequently, numerous LVLMs dedicate their efforts not only to general image comprehension but also to text- 

2 

Technical Report 

oriented tasks. Initially, certain methodologies, such as LLaVAR (Zhang et al., 2023d) and mPLUGDocOwl (Ye et al., 2023a), enhance image resolution and incorporate text-rich data during the instruction tuning phase. For instance, mPLUG-DocOwl elevates the image resolution to 896 _×_ 896 and integrates a diverse array of text-rich data, including documents, tables, webpages, and charts, building upon the mPLUG-Owl (Ye et al., 2023c) framework. CogAgent (Hong et al., 2023), on the other hand, employs both low-resolution and high-resolution image encoders to accommodate inputs at a resolution of 1120 _×_ 1120. Subsequently, UReader (Ye et al., 2023b) introduces a shapeadaptive cropping module tailored for handling high-resolution images. mPLUG-DocOwl 1.5 (Hu et al., 2024a) adopts this cropping module and constructs an extensive dataset, DocStruct4M, to further refine its text-oriented capabilities. InternLM-XComposer2-4KHD (Dong et al., 2024) ventures into image resolutions up to 4K HD and beyond, employing a similar cropping strategy. Meanwhile, InternVL 1.5 (Chen et al., 2024b) incorporates OCR data during the pre-training phase, thereby significantly enhancing the models’ text recognition capabilities. 

TextHawk (Yu et al., 2024) adheres to the shape-adaptive cropping strategy for handeling arbitrary shape images and introduces innovative features such as Scalable Positional Embeddings (SPEs) and Query Proposal Network (QPN) to more effectively model sub-images. Additionally, it incorporates a Multi-Level Cross-Attention (MLCA) mechanism that capitalizes on the hierarchical structure and semantic relationships within the data, thereby significantly enhancing the model’s fine-grained visual perception capabilities. TextHawk2 builds upon the architecture of TextHawk, with enhancements made in both the training data and the model’s training strategies, aiming to achieve superior performance in text-oriented tasks. 

## 2.2 GROUNDING-ORIENTED LVLMS 

Grounding capabilities are critical for LVLMs to tackle complex reasoning tasks involving specific regions and objects within images. To improve interpretability and enhance user interaction, LVLMs are typically expected to accept and provide positional information in formats such as point coordinates, bounding boxes, or region masks. Shikra (Chen et al., 2023) approaches this by encoding positions as normalized plain-text coordinates, leveraging the flexibility of natural language. Conversely, models like VisionLLM (Wang et al., 2023d), Kosmos-2 (Peng et al., 2023), and Ferret (You et al., 2023) extend LVLM vocabularies by incorporating location tokens that represent normalized and quantized offsets of image dimensions. These models are trained on carefully crafted largescale visual grounding datasets to support the new tokens. LLaVA-G (Zhang et al., 2023a) adopts a different strategy, predicting segmentation masks rather than bounding boxes, using a pre-trained grounding model as its decoder, which necessitate additional alignment training with the LVLM. Meanwhile, GPT4RoI (Zhang et al., 2023c) and VolCano (Li et al., 2024c) enhance fine-grained multimodal understanding by supplementing the model with additional regional features, instead of positional information. In contrast, TextHawk series pioneer the native grounding capability of LVLMs via a detection head combined with efficient representation of bounding boxes. 

## 2.3 VISUAL TOKEN COMPRESSION 

With the support for higher image resolutions in recent LVLMs, the number of visual tokens has surged, creating a strong demand for efficient compression methods. Solutions like CogAgent and MiniGemini (Li et al., 2024b) tackle this by introducing a lightweight visual encoder specifically for high-resolution refinement, without increasing the visual token count. Their method uses lowresolution visual embeddings as queries to retrieve relevant high-resolution cues, either within the LLM or via a resampler. Qwen-VL (Bai et al., 2023) and LLaVA-UHD (Xu et al., 2024) adopt a different approach by directly compressing visual tokens of each sub-image by factors of 4 and 9, respectively, using a shared perceiver resampler layer. Meanwhile, LLaVA-PruMerge (Shang et al., 2024) implements an adaptive strategy, dynamically identifying and retaining the most critical visual tokens, then merging similar ones through clustering. TextMonkey (Liu et al., 2024c) also performs visual token compression based on token similarity. MADTP (Cao et al., 2024) introduces a Dynamic Token Pruning (DTP) module that adjusts visual token compression ratios layer by layer, adapting to varying input complexities. TextHawk stands out as the first LVLM to achieve 16 times token compression through a novel two-step process for resampling and rearrangement, each reducing the token count by a factor of 4. Building on TextHawk’s approach, TextHawk2 achieves the same compression ratio of 16, offering enhanced efficiency in visual token handling. 

3 

Technical Report 

**==> picture [382 x 180] intentionally omitted <==**

**----- Start of picture text -----**<br>
Detection Head<br>LLM<br>𝑣𝑣00 𝑣𝑣10 𝑣𝑣20 𝑣𝑣30 𝑣𝑣40 𝑡𝑡00 𝑡𝑡10 𝑡𝑡20 𝑡𝑡30 𝑡𝑡40 𝑡𝑡50 … 𝑣𝑣01 𝑣𝑣11 𝑣𝑣21 𝑡𝑡01 𝑡𝑡11 𝑡𝑡21 …<br>1 tf t ft f A recipe for making bread at home. First, … 1 tf f A cat riding on a dog in …<br>Resampler<br>1 1 1 1 1 1 1 1<br>To, Ee, Ga OG, ga, ©, oo<br>OOS| | | | ee ee |<br>OOS es es es ee ||<br>tt<br>Visual Encoder<br>**----- End of picture text -----**<br>


Figure 2: The network architecture and dataflow of TextHawk2. 

## 3 ARCHITECTURE 

The overall architecture and key components of TextHawk2 continue the design of TextHawk (Yu et al., 2024) family, including a lightweight visual encoder and an LLM, which are bridged by a meticulously designed resampler for modality adaption and token compression. The network architecture and dataflow of TextHawk2 is depicted in Fig. 2. 

## 3.1 LARGE LANGUAGE MODEL 

Recent advances in open-source LLMs have also enhanced the ability of upper-level LVLMs to understand both linguist and visual content. Noteworthy among these developments are models like LLaMA3 (AI@Meta, 2024) and Qwen2 (Yang et al., 2024). There are two major enhancements in the latest LLMs. Firstly, the integration of Grouped-Query Attention (GQA) has greatly reduced the memory requirements for key-value cache during deployment. Secondly, these models support longer context lengths, e.g., Qwen2 can handle up to 32,768 tokens during pre-training and extend up to 131,072 tokens during inference. To develop a Chinese-English bilingual LVLM, we utilize Qwen2-7B-Instruct due to its strong capability in processing Chinese data. 

## 3.1.1 DETECTION HEAD 

To improve training throughput and inference efficiency, our TextHawk series extend the vocabulary of LVLMs with special coordinate tokens. As depicted in Fig. 3, representing a bounding box with a digit string requires 25 tokens—2 trigger marks, 4 floating-point numbers (each uses 5 tokens), and 3 commas. In contrary, by replacing each floating-point number with a unique coordinate token and retaining the center comma, we significantly reduce the token count to just 7. 

digit string: [ 0 . 2 3 0 , 0 . 3 3 4 , 0 . 8 2 6 , 0 . 4 8 2 ] coordinate tokens: <box> <230> <334> , <826> <482> </box> 

Figure 3: Different representations of coordinates. For clarity, we separate the tokens by different colors and broken underlines. 

4 

Technical Report 

To facilitate the training of newly appended coordinate tokens and strengthen grounding capability, TextHawk series introduce an auxiliary training objective. This is achieved by integrating a detection head and _ℓ_ 1 loss. Specifically, the detection head consists of a 2-layer MLP and a linear projection layer, running parallel to the original output layer of the LLM. 

## 3.2 VISUAL ENCODER 

Following TextHawk, we utilize the lightweight ViT from SigLIP-SO400M (Zhai et al., 2023) as the visual encoder of TextHawk2, maintaining the original resolution of 224 _×_ 224. The effectiveness of the SigLIP family serving as visual encoders for LVLMs has also been demonstrated in concurrent studies (He et al., 2024; Laurençon et al., 2024). Our work further confirms the feasibility of transferring SigLIP to previously unseen tasks, such as Chinese OCR. 

## **Unified Visual Encoder** 

_Language-supervised models such as CLIP (Radford et al., 2021) and SigLIP are not optimized for fine-grained tasks. There was once a trend for LVLMs to use dual or even more visual encoders (Lin et al., 2023; Fan et al., 2024). In the realm of text-oriented LVLMs, some of the previous works (Wei et al., 2023; Lu et al., 2024a) use CLIP-ViT as the lowresolution visual encoder and SAM (Kirillov et al., 2023) as the high-resolution encoder encoder. As for grounding tasks, previous state-of-the-art methods (Lin et al., 2023; Zhang et al., 2024b) opt for a combination of CLIP-ViT and DINOv2 (Oquab et al., 2023)._ 

Despite the marginal advantages on academic benchmarks, using dual visual encoders is computationally expensive and lacks flexibility in building generalist models, making it impractical for real-world applications. Hence, we opt for a unified visual encoder and enhance it through feature merging (Section 3.3.4) and LVLM co-training to maximize its potential. 

## 3.2.1 DYNAMIC HIGH-RESOLUTION 

Following UReader (Ye et al., 2023b), the TextHawk family enhances a fixed-resolution ViT through a dynamic cropping strategy. This approach, widely adopted in recent research, effectively processes images with varying aspect ratios and resolutions. For further details, readers may refer to TextHawk (Yu et al., 2024). Our findings indicate that input resolution significantly impacts the accuracy in fine-grained tasks, particularly OCR tasks like image-to-markdown. For example, lowresolution images of math formulas sometimes contain small and blurry characters, leading to hallucinations. During pre-training, TextHawk2 is configured to allow a maximum area of 36 sub-images and a maximum side length of 12 sub-images per row or column. This setup yields a maximum of 1.8 million pixels and a long edge length of 2688 pixels. Unlike TextHawk, TextHawk2 expands the maximum area from 36 to 72 during supervised fine-tuning, enabling higher-resolution image inputs. To clarify, the maximum area value is relevant only for high-resolution images that surpass the specified limit. For example, an input image with dimensions of 896 _×_ 672 will be split into 4 _×_ 3 sub-images rather than 8 _×_ 6 sub-images, thereby avoiding unnecessary computational costs. 

It is important to note that increasing the maximum area can enhance high-resolution performance but introduces two notable side effects. Firstly, it demands a significantly larger amount of memory to store a large batch of ViT activations, which can strain system resources. Secondly, it leads to an unbalanced computational load across different samples due to the varying number of sub-images, which creates inefficiencies. These factors can severely impact training throughput, particularly when using pipeline parallelism with limited resources. Therefore, it is crucial to impose a proper limit on the maximum area to mitigate these issues while balancing high-resolution performance. 

## 3.3 RESAMPLER 

The resampler plays a critical role in bridging different modalities as well as compressing tokens between the visual encoder and the LLM. For the reader’s convenience, we briefly revisit several key improvements of the TextHawk resampler. 

5 

Technical Report 

## 3.3.1 SCALABLE POSITIONAL EMBEDDINGS 

Scalable Positional Embeddings (SPEs) (Yu et al., 2024) present an innovative extension of factorized positional embeddings (decomposing row and column), making them applicable to arbitrary input shapes. To ensure a fair comparison, we also modify Absolute Positional Embeddings (APEs) to accommodate dynamic shapes by slicing sections of the APEs during both training and inference phases. Due to their adaptability and training efficiency, SPEs achieve superior performance over APEs while utilizing fewer parameters. Furthermore, SPEs exhibit outstanding extrapolation capabilities to unseen input resolutions, resulting in impressive zero-shot performance. 

The concept of SPEs arises from the observation that positional embeddings, in practice, tend to distribute themselves around the surface of a hypersphere. This insight leads us to consider Spherical Linear Interpolation (Slerp) as a potential alternative to traditional interpolation methods, such as nearest-neighbor or linear interpolation. However, our initial attempts to directly apply Slerp to pre-trained APEs prove to be ineffective. We believe this ineffectiveness stems from the incomplete assumption that these embeddings are perfectly distributed on a hypersphere. To address this issue, we introduce a normalization and scaling process for the embeddings prior to interpolation, ensuring they conform to the requirements of Slerp. Moreover, given that different parts of the positional embeddings are used independently by each attention head, we apply normalization and scaling operations on a per-head basis, allowing for more precise interpolation aligned with the needs of each attention mechanism. The pseudocode of SPEs is shown in Algorithm 1. 

## **Algorithm 1** Scalable Positional Embeddings 

|**Input:** start embeddings**_e_**0|**Input:** start embeddings**_e_**0|**Input:** start embeddings**_e_**0|_∈_R_d_, end embeddings**_e_**1 _∈_R_d_, interpolation position_t ∈_[0_,_1]|
|---|---|---|---|
|**Output:** interpolated positional embeddings**_e_**(_t_)||||
|1: **initialization:** _s ←_|_√_|_d_|_▷_scaling factor|
|2: **for**_i ∈{_0_,_1_}_**do**||||
|3:<br>**_e_**_i ←_<br>**_e_**_i_<br>_∥_**_e_**_i∥_|||_▷_normalization|
|4:<br>**_e_**_i ←s ·_**_e_**_i_|||_▷_scaling|
|5: **end for**||||
|6: _θ ←_arccos<br>**_e_**0**_e_**1<br>_∥_**_e_**0_∥∥_**_e_**1_∥_||||
|7: **_e_**(_t_)_←_sin(_θ−tθ_)<br>sin_θ_<br>**_e_**0|+|sin(_tθ_)<br>sin_θ_ **_e_**1||



## 3.3.2 QUERY PROPOSAL NETWORK 

To enhance convergence and improve grounding performance, the TextHawk family incorporates the Query Proposal Network (QPN) (Yu et al., 2024) to dynamically generate resampling query tokens. Attention-based adapters, such as Quering Former (Q-Former) (Li et al., 2023b) and perceiver resampler (Alayrac et al., 2022), show promise in token compression but are challenging to train. On the other hand, MLP-based adapters, while simpler, often outperform attention-based adapters when training data is limited. We attribute the difficulty of training attention-based adapters to the fixed query tokens used in previous approaches. This observation led us to merge the strengths of both methods. Specifically, QPN utilizes a lightweight MLP-Pool-Dense architecture to efficiently transform features from the visual encoder into queries. It also offers greater adaptability by allowing a variable number of unique queries for images with different resolutions. In the QPN, we apply a 2 _×_ 2 max pooling, achieving a compression ratio of 4 during the resampling stage. 

## 3.3.3 RESAMPLING AND REARRANGEMENT 

TextHawk introduces a two-stage token compression strategy called ReSampling and ReArrangement (ReSA) (Yu et al., 2024), designed to minimize information loss and preserve critical information from visual inputs. In the first stage, resampling, a smaller set of highly informative tokens is selectively extracted from the visual encoder outputs. This is achieved through a cross-attention mechanism where query tokens, generated by the QPN, guide the selection process. For TextHawk2, these tokens are progressively refined in 4 bidirectional decoder layers. In the second stage, rearrangement, visual tokens are flattened following the image scanning order and then grouped into sets of four. Instead of arranging tokens based on the sequence of sub-images, we preserve the original 

6 

Technical Report 

**==> picture [270 x 136] intentionally omitted <==**

**----- Start of picture text -----**<br>
CapsFusion Wanjuan<br>UMG GrIT<br>WukongText IIT-CDIP<br>E2E OCR RenderedText<br>PT-100M<br>PDF arXiv<br>docx README<br>PubTables MMC<br>ChartSFT<br>Grounding41%<br>OCR<br>23%<br>36%General<br>**----- End of picture text -----**<br>


Figure 4: The 100M pre-training data mixture of TextHawk2. 

line-by-line scanning order of the entire image. The former approach leads to an input order that diverges significantly from the natural reading order of text, which is typically from left to right and top to bottom, thereby impairing the document understanding capabilities of LVLMs. In contrast, our approach arranges visual tokens from sub-images in the same row in an interleaved pattern. Furthermore, our token concatenation strategy aligns with this approach by combining four adjacent tokens within a 1 _×_ 4 window along the same row. 

## **16 Times Token Compression** 

By combining the 2 _×_ 2 subsampling window from the resampling stage with the 1 _×_ 4 subsampling window from the rearrangement stage, we achieve a total compression ratio of 16 within a 2 _×_ 8 subsampling window. This window shape may be particularly suited for text-oriented LVLMs, and efforts to explore different window shapes are discussed in a concurrent work (Hu et al., 2024a). 

## 3.3.4 MULTI-LEVEL CROSS-ATTENTION 

To address the limitations of language-supervised visual encoders on fine-grained tasks, TextHawk proposes a feature merging approach called Multi-Level Cross-Attention (MLCA) (Yu et al., 2024). The MLCA mechanism is designed to enhance feature extraction by allowing the resampler to efficiently aggregate information from multiple layers of a visual encoder. This is achieved through a predefined routing table that determines which features are to be extracted and merged at each resampler layer. One of the key findings is that a deep-to-shallow feature extraction strategy yields superior results for grounding tasks while preserving the overall performance of general visual understanding. Notably, MLCA accomplishes this without incurring any additional computational costs, making it both effective and efficient. In practical terms, the implementation of MLCA in TextHawk involves utilizing four distinct stages of the visual encoder. Features are extracted specifically from the 14th, 18th, 22nd, and 26th layers of the encoder. 

## 4 DATA 

TextHawk2 employs a one-pass pre-training approach, differing from the two-stage pre-training paradigm commonly used in prior works on LVLMs. These models undergo an initial stage where different modalities are aligned using fixed, low-resolution image-caption pairs. This is followed by a second stage of continual training on mixed-resolution image-text data from diverse sources, such as OCR and grounding datasets. In contrast, TextHawk2 skips the initial alignment stage and instead focuses on training on more detailed image captions from the beginning. 

## 4.1 PRE-TRAINING 

The 100M pre-training data are collected from diverse sources and carefully curated to enhance the OCR and grounding capabilities. The sampling ratios for various datasets are shown in Fig. 4. 

7 

Technical Report 

## 4.1.1 CONCEPTION 

To improve alignment, we utilize data from CapsFusion (Yu et al., 2023a), a framework designed for re-captioning web-crawled data. Within our previous work, the largest conceptual caption dataset, LAION-400M (Schuhmann et al., 2021), is automatically gathered from the web. This approach can result in captions containing irrelevant descriptions or lacking essential details, causing hallucinations and misalignments. CapsFusion addresses these issues by employing LVLM to generate captions that directly reflect the image content. These generated captions are then integrated with the web-sourced captions using a caption fuser, avoiding knowledge loss. 

## 4.1.2 INTERLEAVED 

Previous works have demonstrated that interleaved image-text data is beneficial for improving the multimodal in-context learning capability of LVLMs (Zhang et al., 2023b; Laurençon et al., 2024). TextHawk2 makes up for the lack of large-scale interleaved data by leveraging the image-text dataset from the Wanjuan1.0 data collection (He et al., 2023). This part comprises bilingual interleaved data sourced from Wikipedia and news outlets. 

## 4.1.3 GROUNDING 

We primarily utilize GrIT-20M (Peng et al., 2023), a synthetic caption dataset with additional location labels for major visual elements, to enhance the grounding capability of TextHawk2. Additionally, we incorporate referring and grounding data from UMG-41M (Shi et al., 2024). These data are curated from various public image-caption datasets, including CC3M (Sharma et al., 2018), CC12M (Changpinyo et al., 2021), SBU (Ordonez et al., 2011), Flickr (Young et al., 2014), VG (Krishna et al., 2017), YFCC-15M (Thomee et al., 2016), and ImageNet-21K (Ridnik et al., 2021), by jointly applying an object detector and a regional captioner. Specifically, each region is randomly assigned to either a referring task or a grounding task. In the referring task, we provide the model with a bounding box to generate a caption for that specific region, while in the grounding task, we reverse this by using the caption to predict the corresponding bounding box. We also include approximately 1/8 of the captions in Chinese, which are generated using an English-to-Chinese translation API. 

## 4.1.4 OCR 

To gather extensive OCR pre-training data, we employ a commercial OCR engine to transcribe text from images. This includes Chinese text from the Wukong (Gu et al., 2022) dataset and English text from the IIT-CDIP (Lewis et al., 2006) dataset. We also use PDFPlumber to extract text lines from Common Crawl PDFs. To improving English handwriting recognition, we incorporate RenderedText (StabilityAI & LAION, 2023). Additional end-to-end OCR datasets, including ArT (Chng et al., 2019), COCO-Text (Veit et al., 2016), CTSU (Guo et al., 2021), CTW (Yuan et al., 2019), IC15 (Karatzas et al., 2015), LSVT (Sun et al., 2019), MLT (Nayef et al., 2019), MTWI (He et al., 2018), RCTW-17 (Shi et al., 2017), ReCTS (Zhang et al., 2019), and SCUT-HCCDoc (Zhang et al., 2020) are also integrated into our training data. 

## 4.1.5 MARKDOWN 

Building upon the markup-based data pipeline introduced in Kosmos-2.5 (Lv et al., 2023), we expand our dataset by gathering more image-to-markdown pairs to enhance OCR and layout understanding capabilities. We specifically source L[A] TEX documents from arXiv, README files from GitHub, and DOCX files from Common Crawl. These files are then converted into images and subsequently translated into markdown format. 

## 4.1.6 TABLE & CHART 

Alongside the previously mentioned markdown data, we also collect data to enhance the ability to interpret tables and charts. For tables, we use the PubTables-1M (Smock et al., 2022) dataset, including both its original English version and a translated Chinese version, to gather table recognition data. For charts, we employ chart-to-table conversion and chart-based QA data from existing datasets, including MMC (Liu et al., 2024b) and ChartSFT (Meng et al., 2024). 

8 

Technical Report 

**==> picture [397 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
Mixed Dataset Mixed Dataset Mixed Dataset Mixed Dataset<br>worker 0 worker 1 worker 2 worker 3<br>1<br>2<br>3<br>Data Loader Data Loader<br>DP rank 0 DP rank 1<br>1 Sequential Loading 2 Data Concatenation 3 Pre-ordered Sampling<br>tar file and sample indices recorded pending samples recorded random seed fixed<br>**----- End of picture text -----**<br>


Figure 5: Data pipeline with sequential loading and failure recovery mechanism. 

## 4.2 SUPERVISED FINE-TUNING 

TextHawk2 enhances the mixture of TextHawk instruction data by incorporating several newly added datasets. First, it replaces all text-only data with two high-quality data collections: OpenHermes2.5 (Teknium, 2023) and COIG-CQIA (Bai et al., 2024). Next, it adds a variety of other datasets, including ShareGPT-4o (Chen et al., 2024b), LVIS-Instruct4V (Wang et al., 2023a), LAION-GPT4V, LLaVAR (Zhang et al., 2023d), Cauldron (Laurençon et al., 2024), KVQA (Shah et al., 2019), ViQuAE (Lerner et al., 2022), Geo170K (Gao et al., 2023), HME100K (Yuan et al., 2022), UniMER-1M (Wang et al., 2024a), FUNSD (Jaume et al., 2019), XFUND (Xu et al., 2022), SROIE (Huang et al., 2019b), POIE (Kuang et al., 2023), ST-VQA (Biten et al., 2019), and ESTVQA (Wang et al., 2020). Finally, it randomly samples 80K, 60K, 20K, 300K, and 80K pre-training data samples from the aforementioned OCR, Markdown, Table, and Chart categories. 

## 5 IMPLEMENTATION DETAILS 

## 5.1 INFRASTRUCTURE 

TextHawk2 is trained on Huawei Cloud, utilizing the elastic cloud computing and file system. 

## 5.1.1 STORAGE 

For storing large-scale multimodal data, we use Huawei Cloud’s Parallel File System (PFS). PFS is a high-performance file system built on Object Storage Service (OBS), offering millisecondlevel access latency, TB/s-level bandwidth, and millions of IO/s, enabling fast handling of HighPerformance Computing (HPC) workloads. 

To accelerate data preparation during training, we introduce a sequential loading method that avoids the inefficiency of random access to numerous small files. Our dataset implementation is built on WebDataset (WebDataset, 2024), a high-performance IO system that uses tar files and provides Python APIs. It supports reading files from local storage as well as files from OBS. By using the WebDataset format, we create a fully sequential IO pipeline optimized for handling large-scale data. Specifically, we divide the samples into equal-sized chunks, store them in tar files, and distribute the chunks across different data workers. Each data worker then loads all the samples from its assigned chunks sequentially, with no overlap in the data between workers. This approach is crucial for maximizing IO throughput and efficiently leveraging cloud storage during training. 

Additionally, we implement a failure recovery mechanism that ensures training can accurately resume from any checkpoint. While the native WebDataset offers fully indexed access to datasets, it 

9 

Technical Report 

introduces significant storage and IO overhead and is not suitable for training. Hence, we introduce another efficient failure recovery mechanism with three main components. First, we shuffle each dataset before training, removing the need for large cache pools during dynamic shuffling. This approach also allows for independent shuffling of any newly added datasets. Second, we log both the tar file and sample indices, enabling quick recovery by directly jumping to the specific tar file and skipping previously processed samples. Finally, we back up pending samples that dataset workers have processed but not yet sent to the data loaders, as well as the state of random generators to guarantee accurate recovery. The overall data pipeline is shown in Fig. 5. 

## 5.1.2 ACCELERATOR 

We support both NVIDIA GPU and Ascend NPU platforms. To accelerate attention computation, we adopt memory-efficient attention (Lefaudeux et al., 2022) for NVIDIA V100, and NPU fusion attention (Ascend, 2024) for Ascend 910B. TextHawk2 is trained on 16 nodes with 128 cards. 

## 5.2 EFFICIENCY 

To enhance the training efficiency of LVLMs on devices with limited memory, we present two major improvements, including 4D parallelism and data packing. 

## 5.2.1 PARALLELISM 

We employ a combination of four types of parallelisms, including Data Parallelism (DP) (Li et al., 2020), Tensor Parallelism (TP) (Shoeybi et al., 2019), Sequence Parallelism (SP) (Korthikanti et al., 2023), and Pipeline Parallelism (PP) (Huang et al., 2019a; Harlap et al., 2018; Narayanan et al., 2021). DP is the most prevalent technique for distributed training, allowing large data batches to be divided across various devices. We utilize DP alongside the DeepSpeed Zero Redundancy Optimizer (ZeRO) (Rajbhandari et al., 2020) to enhance memory efficiency. TP reduces memory usage by distributing model weights and partial activations across multiple devices, while SP further alleviates memory demands by handling activations that TP cannot manage. However, TP introduces significant communication overhead, requiring all devices to reside on the same node with highspeed connections. By default, we set TP = SP = 1, using a maximum of TP = SP = 4 only when needed. In contrast to LLM training, PP operates differently in LVLM training due to the heterogeneous characteristics of LVLMs. Here are the challenges: 

- **Computational Imbalance** : Distributing model layers evenly across multiple pipeline stages is crucial for load balancing. However, achieving this balance with LVLMs is more complex than with LLMs. The challenge arises from the requirement to place the visual encoder and resampler before the first LLM layer, complicating the even distribution of these components across pipeline stages. 

- **Memory Imbalance** : The initial pipeline stages have to store activations from the warmup micro-batches. The size of these activations is proportional to the number of pipeline stages. As PP increases, the memory required to store activations in both the visual encoder and resampler also increases, which might lead to memory overload. 

For PP, computational imbalance results in increased idle time (aka bubble size), which should be avoided. To address this and improve communication efficiency, we integrate the entire visual encoder and resampler into the initial pipeline stage. To avoid memory overload, we restrict PP, setting it to PP = 1 during LoRA training and PP = 2 during full-parameters training. To tackle computational imbalance, we divide the LLM layers into unequal segments, ensuring that the first pipeline stage contains the visual encoder, resampler, and fewer LLM layers. 

## 5.2.2 PACKING 

To achieve optimal performance, it is crucial to balance the model components with the data stream. However, LVLMs with variable resolution inputs and variable length outputs inevitably involve imbalances. To address this, we set a fixed context length and pack multiple samples to reduce padding. We also restrict the number of packed images to avoid overloading the visual encoder. Specifically, 

10 

Technical Report 

Table 1: Performance comparison on text-oriented tasks. 

|**Model**|OCRBench|ChartQA|DocVQA|InfoVQA|TabFact|WTQ|TextVQA|
|---|---|---|---|---|---|---|---|
|Low compression ratio||||||||
|Qwen-VL-Chat (Bai et al., 2023)|50.6|66.3|62.6|28.3|-|-|61.5|
|UReader (Ye et al., 2023b)|-|59.3|65.4|42.2|67.6|29.4|61.5|
|Monkey (Li et al., 2023c)|51.4|65.1|66.5|36.1|-|25.3|67.6|
|CogAgent (Hong et al., 2023)|59.0|68.4|81.6|44.5|-|-|76.1|
|DocOwl-1.5-Chat (Hu et al., 2024a)|59.9|70.2|82.2|50.7|**80.2**|40.6|68.6|
|MiniCPM-V-2.5 (Yao et al., 2024)|72.5|72.1|84.8|50.8|-|-|76.6|
|GLM-4v-9B (Zeng et al., 2024)|78.6|30.1|76.5|53.1|-|-|83.0|
|InternVL2-7B (Chen et al., 2024c)|79.4|83.3|91.6|74.8|-|-|77.4|
|Qwen2-VL-7B (Wang et al., 2024b)|**84.5**|**83.0**|**94.5**|**76.5**|-|-|**84.3**|
|High compression ratio||||||||
|TextMonkey (Liu et al., 2024c)|56.1|66.9|73.0|28.6|-|31.9|64.3|
|TextHawk (Yu et al., 2024)|-|66.6|76.4|50.6|71.1|34.7|-|
|HRVDA (Liu et al., 2024a)|-|67.6|72.1|43.5|72.3|31.2|73.3|
|DocKylin (Zhang et al., 2024c)|-|66.8|77.3|46.6|-|32.4|-|
|DocOwl-2 (Hu et al., 2024b)|-|70.0|80.7|46.4|**78.2**|36.5|66.7|
|MM1.5 (Zhang et al., 2024a)|63.5|78.6|88.1|59.5|75.9|46.0|**76.8**|
|**TextHawk2**|**78.4**|**81.4**|**89.6**|**67.8**|78.1|**46.2**|75.1|



we use a context length of 4096 and a maximum of 108 image tiles, including thumbnails and sub-images. Additionally, we apply masking to ensure that the samples remain mutually invisible. 

## 5.3 HYPERPARAMETERS 

During the pre-training phase, our focus is on training the newly initialized resampler and updating both the ViT and LLM using LoRA. Specifically, LoRA modules are applied to the query and value projection layers, with ranks of 16 for ViT and 128 for LLM. In the supervised fine-tuning stage, we unfreeze all parameters, allowing the entire model to be trained end-to-end. Our preliminary investigation of different training strategies has shown that training an LVLM for Chinese OCR with a frozen ViT is possible. However, unfreezing ViT during both pre-training and supervised finetuning significantly enhances performance, making it essential for achieving state-of-the-art results in OCR tasks. To further improve OCR robustness, we introduce manual perturbations by randomly resizing images from text-oriented datasets within a small range. 

During the pre-training phase, we utilize a global batch size of 384, where each data point is a collection of multiple packed samples. The training process spans 45,000 steps. The learning rate initiates at 0 and linearly warms up to 2 _×_ 10 _[−]_[4] within the initial 3% of the steps. Beyond this point, it follows a cosine decay schedule, tapering down to 0. We also incorporate a "late warmup" strategy for both ViT and the LLM. During the first half of the warm-up phase, the parameters of these modules remain fixed. Concurrently, only the parameters of the resampler are updated, which serves to offset the lack of a dedicated pre-training phase for the resampler alone. For the supervised fine-tuning stage, the global batch size is set to 256, and the model undergoes training for two epochs. The learning rate schedule is akin to the pre-training phase, albeit with distinct peak values: 5 _×_ 10 _[−]_[5] for both the ViT and the resampler, and 2 _×_ 10 _[−]_[5] for the LLM. 

In configuring the AdamW optimizer for stable training, we set _β_ 1 to 0.9 and _β_ 2 to 0.95. Additionally, a weight decay of 0.05 is applied to enhance model generalization. 

## 6 EXPERIMENTS 

## 6.1 OCR BENCHMARK 

We explore the native OCR capabilities of TextHawk2 across various text-oriented tasks, including nature scene text recognition, document information retrieval, chart comprehension, and table fact-checking. The benchmarks utilized are OCRBench (Liu et al., 2023e), ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), InfoVQA (Mathew et al., 2022), TabFact (Chen et al., 

11 

Technical Report 

Table 2: Performance comparison (Acc@0.5) on referring expression comprehension tasks. 

|**Model**|**RefCOCO**<br>val<br>test-A<br>test-B|**RefCOCO+**<br>val<br>test-A<br>test-B|**RefCOCOg**<br>val<br>test|
|---|---|---|---|
|Specialist||||
|G-DINO-L (Liu et al., 2023c)<br>UNINEXT-H (Yan et al., 2023)<br>ONE-PEACE (Wang et al., 2023b)|90.6<br>93.2<br>88.2<br>92.6<br>94.3<br>**91.5**<br>92.6<br>94.2<br>89.3|82.8<br>89.0<br>75.9<br>85.2<br>89.6<br>79.8<br>**88.8**<br>92.2<br>83.2|86.1<br>87.0<br>88.7<br>89.4<br>89.2<br>89.3|
|Grounding-oriented||||
|OFA-L (Wang et al., 2022)<br>Shikra (Chen et al., 2023)<br>Ferret-7B (You et al., 2023)<br>Ferret-v2-7B (Zhang et al., 2024b)<br>CogVLMGrounding (Wang et al., 2023c)|80.0<br>83.7<br>76.4<br>87.0<br>90.6<br>80.2<br>87.5<br>91.4<br>82.5<br>**92.8**<br>94.7<br>88.7<br>**92.8**<br>**94.8**<br>**89.0**|68.3<br>76.0<br>61.8<br>81.6<br>87.4<br>72.1<br>80.8<br>87.4<br>73.1<br>87.4<br>92.8<br>79.3<br>**88.7**<br>**92.9**<br>**83.4**|67.6<br>67.6<br>82.3<br>82.2<br>83.9<br>84.8<br>89.4<br>89.3<br>**89.8**<br>**90.8**|
|Generalist||||
|Qwen-VL-Chat (Bai et al., 2023)<br>TextHawk (Yu et al., 2024)<br>InternVL2-8B (Chen et al., 2024c)<br>MM1.5 (Zhang et al., 2024a)<br>Qwen2-VL-7B (Wang et al., 2024b)<br>**TextHawk2**|88.6<br>92.3<br>84.5<br>87.3<br>90.9<br>83.3<br>87.1<br>91.1<br>80.7<br>-<br>92.5<br>86.7<br>91.7<br>**93.6**<br>87.3<br>**91.9**<br>93.0<br>**87.6**|82.8<br>88.6<br>76.8<br>-<br>-<br>-<br>79.8<br>87.9<br>71.4<br>-<br>88.7<br>77.8<br>85.8<br>**90.5**<br>79.5<br>**86.2**<br>90.0<br>**80.4**|86.0<br>86.3<br>-<br>-<br>82.7<br>82.7<br>-<br>87.1<br>87.3<br>87.8<br>**88.2**<br>**88.1**|



2020), WTQ (Pasupat & Liang, 2015), and TextVQA (Singh et al., 2019). Our model is compared against multiple baseline LVLMs with different compression ratios, as illustrated in Table 3. Notably, TextHawk2 consistently outperforms other baseline LVLMs with high compression ratios by a significant margin. Among the models evaluated, MM1.5 (Zhang et al., 2024a) comes closest in performance, yet TextHawk2 exceeds it by 14.9%, 2.8%, 1.5%, 8.3%, 2.2%, and 0.2% on OCRBench, ChartQA, DocVQA, InfoVQA, TabFact, and WTQ, respectively. When compared to baseline LVLMs with low compression ratios, TextHawk2 surpasses GLM-4v-9B (Zeng et al., 2024), MiniCPM-V-2.5 (Yao et al., 2024), and other previous models. Although it falls short relative to InternVL2-8B (Chen et al., 2024c), the performance gap is small. The notable exception is Qwen2-VL-7B (Wang et al., 2024b), a member of the highly effective open-source Qwen2-VL series. Qwen2-VL-7B outperforms other leading LVLMs significantly. We attribute this advantage to its native resolution ViT and its full parameter training approach, which we plan to investigate further in future work. In summary, the results of TextHawk2 demonstrate an definite answer to our first question: It is feasible to achieve cutting-edge OCR performance with a visual token compression ratio of 16, where the keys are visual encoder reinforcement and effective data curation. 

## 6.2 GROUNDING BENCHMARK 

Following previous works (Chen et al., 2023; Bai et al., 2023; You et al., 2023), we investigate the grounding capabilities of TextHawk2 on three Referring Expression Comprehension (REC) tasks: RefCOCO, RefCOCO+, and RefCOCOg Kazemzadeh et al. (2014); Mao et al. (2016). As presented in Table 2, we compare TextHawk2 with both generalist and grounding-oriented models, as well as specialist models. Remarkably, TextHawk2 surpasses all current state-of-the-art generalist LVLMs, including the highly effective Qwen2-VL-7B (Wang et al., 2024b) and InternVL2-8B (Chen et al., 2024c). The performance of TextHawk2 on REC tasks is comparable to that of grounding-oriented LVLMs, some of which employ specialized grounding-oriented visual encoders like DINOv2. Combined with the findings in Table 3, this confirms the feasibility of training an LVLM using a unified visual encoder that excels across general multimodal understanding, OCR, and grounding tasks, apparently addressing our second question. 

## 6.3 MARKDOWN CONVERTER 

TextHawk2 demonstrates a strong capability to transcribe content from screenshots of scientific papers, README files, and DOCX documents into markdown text. Two examples are illustrated in Fig. 6. Notably, in the first example, TextHawk2 accurately extracts plain text as well as precisely captures L[A] TEX formulas for complex mathematical expressions. This demonstrates the potential of LVLMs over traditional OCR engines in layout-aware OCR tasks. Additionally, the second example 

12 

Technical Report 

**==> picture [398 x 421] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)<br>黑神话：悟空<br>《黑神话：悟空》是一款由游戏科学开发和发行的动作角色扮演游戏，被媒体誉为中国首部“3A游戏”，<br>INA 游戏于2024年8月20日登陆Windows (Steam、Epic Games、WeGame)及PlayStation 5平台，Xbox<br>Games,XSHeWRT ONCORA ACHR2024WeGameE: RIE,HRS4 )8 Ji JakH PlayStationBEALE20 1A AR cbWindows5 ¥EaR ff,IERAUF(Steam,“ANXbox aR”SeriesEpicah5 ff{ y4aSe AA Series X/S游戏内容改编自中国四大名著之一的《西游记》，在正式发布前，已在获得媒体与评论家的普遍好评，称赞其在战斗系统、视觉设计以及世界观方面的构建。游戏上线后迅速登顶多个平台的销售榜，两周内版的发售日期待定。<br>WRAPAEP,eee‘sdIDV,TISAARARBEICCEAR HEALSeeeSLRAERAG,ALES—1800eaeMIELE 75. MH Coni2OLERATEiResinMeeeEe DLC. ce Rl MA N WUKONGBzPAB ; | fei 的全球销量超过 玩法 1800万份，成为有史以来销售速度最快的游戏之一。<br>《黑神话：悟空》是一款动作角色扮演游戏，采用单人模式和第三人称视角。尽管在游戏中发布前因为<br>Bute<br>本作借鉴了不少魂类要素，曾有人讨论其是否属于魂类游戏，但其开发商游戏科学并不认为它属于这一<br>AILTERREiM.CNOWiS:ANSARIEDIFAS IENTE TRASHaka A AIMD, ERR, RLIER, PPAHATALASREELA OSCEISBCPREAAHAASR,IEEFREESE IEE HI” 类型，许多评论者也认为该游戏并不完全符合游戏的设计灵感来源于中国《四大名著》之一“——魂类古典神魔小说《西游记》，玩家将操控一名名为”的定义。 “天<br>WAREHAGANABILA “RA”RATAMER ETA, AERAPRALR)TARACCORZ———A9AR68 AMMENNsBEA EBEAL,CETEDRRMAHe«ENECHE, RHuch 命系统包括三种主要玩法：棍棒、棍棒和立棍，为战斗提供多样化选择。战斗还涉及资源管理，当计量表”的花果山灵明石猴。主角的主要武器是如来金箍棒等梯棒类武器，能在战斗中伸缩自如。游戏的战斗<br>FROLLPHGAYE.ATER ASGAit.  LAER,hia, DRUG SUPBEFRLESIM,Ut SUATULBUR “ERE” 充满时，玩家可以积累“专注点”，用于连招或蓄力攻击。冲刺、闪避和攻击都会消耗气值。<br>AUERSeFEED,"RAE,DLEiti.“HRS”TAMARARSE, SAIN AAEH JHE YeZAEROFESATMEINT EUASLJEREA: “RGEC”ULIHESCAMAMRHERAVL, Lh“AER”EL. ARIATSREBUsb,VAAL“SHEYG,SARPARAAL“Wa”ATATREE"i BOS,PLR?BEIRYAI UsGABA»UREIL” LARGEAR  RIERAEE” AAPL.ICEEE MVHEATARIESAT aPACHLSa ARE 除了物理战斗，玩家还可以使用消耗法力值，各类法术技能可再细分为下蹲以便气散气走，聚形时再行突袭；术允许角色变形为不同的生物，并拥有独特的特招和生命值。此外，玩家可以通过击败强大的敌人获得“奇术”、““定身身外身法“身法”奇术，可以在短时间内冻结敌人；”、”“的魔法技能能召唤多个分身辅助战斗，而各种变化法毫末”和“变化”等四类法术。这些法术没有冷却时间并“聚形散气”身法可让玩家<br>(riceWRIFYE SRLA.REAA SRMRARTE, HEARE AUERASAI ERRORSIBM MSS CEIBRD)  ARNAIRSERESK IB RATA EDTA MIME, “游戏的进程大多是线性的，亦有较宽的区域供玩家探索。玩家会遇到各种怪和头目作为敌人。玩家的检精魄”技能，这些技能类似于一次性的能力，例如，“幽魂”精魄可以让玩家进行头锤攻击。<br>Oko,IMVELFUERaT ABIMEAWIIRAL.HRMSAARA “Bed”& i MEU,BK,JA AAR B ILAEA ALSLRE. FERAA SELe”SAL DL CIR. NODS, INARSE:PAA, HEA 查点是世界各地的神魔（土地）。游戏地图无法由玩家调整，但会随游戏进展而变化。此外，游戏也支持周目模式，为玩家提供“再入轮回”的玩法和彩蛋，例如，进入二周目可以解锁“救命”的<br>毫末法术，在战斗中死亡可以立即复活；三周目怪物变强；四周目敌人强度大幅提升，并触发四段势彩<br>蛋等。<br>(c) (d)<br>Figure 6: Examples of image-to-markdown.<br>shows that despite the initial visual encoder not being pre-trained on a Chinese corpus, TextHawk2<br>still achieves impressive Chinese OCR performance through LVLM joint training. However, a chal-<br>lenge for Chinese OCR still stands that LVLMs are short at recognizing uncommon words. For<br>shén k¯an<br>example, in the second last paragraph, the Chinese word “ 神龛 ” (shrine) is mistakenly recognized<br>shén mó<br>as “ 神魔 ” (gods and demons), which are similar in shape but significantly different in meaning.<br>To solve these problems, greater emphasis should be placed on improving the training strategy and<br>refining the OCR data in future work.<br>13<br>**----- End of picture text -----**<br>


Technical Report 

**==> picture [393 x 197] intentionally omitted <==**

**----- Start of picture text -----**<br>
Region Caption Region Caption Region Caption Region Caption<br>A tall building with lots of. Traffic light is red. A dome on top of. Yellow bus on the street.<br>TextHawk2 TextHawk2 TextHawk2 TextHawk2<br>A tall modern building with many A traffic light on a pole, located near A large green dome on top of a A yellow double decker bus is driving<br>windows, located on the left side of the center of the image. building. down the street.<br>the image.<br>Region Caption Region Caption<br>Region Caption<br>A car on the street. Region Caption A window on a building. A black car on the street.<br>TextHawk2 A car on the street. TextHawk2 TextHawk2<br>A car with its brake lights on, stopped TextHawk2 A street light on a pole. A black car is driving on the road.<br>at a traffic light. A car is driving on the road.<br>Briefly describe this region: ……<br>Caption Describe all key information in the image as thoroughly as possible, including an overall introduction,<br>A building with a dome on top. detailed bullet points for each aspect, and a brief summary.<br>==4==<br>TextHawk2<br>The image depicts a bustling urban scene with a mix of modern and classical architecture. In the foreground, there is a busy street filled with various vehicles, including cars and a double-decker bus. The street is lined with<br>traffic lights and street signs, indicating a well-regulated traffic system. The vehicles are in motion, suggesting a lively atmosphere.<br>• Dominating the background is a large, ornate building with a prominent dome. The dome is topped with a statue, which appears to be a figure of a person, possibly a historical or significant figure. The building itself is<br>made of stone and features classical architectural elements such as columns and decorative moldings. The facade of the building is adorned with statues and intricate details, giving it an imposing and grand appearance.<br>• To the left of the classical building, there is a modern high-rise building with a glass facade. This building contrasts with the classical architecture, showcasing the blend of old and new within the cityscape. The high-rise<br>has multiple floors, and its design is sleek and contemporary.<br>• The sky above the buildings is clear, with a soft glow indicating either early morning or late afternoon light. The lighting casts shadows on the buildings, adding depth and dimension to the scene. Trees with green foliage<br>are visible on the left side of the image, providing a touch of nature amidst the urban environment.<br>Overall, the image captures a moment in a city where historical and modern elements coexist, with a focus on the grandeur of the classical building and the activity on the street. The combination of architectural styles, the<br>presence of traffic, and the natural elements contribute to a dynamic and vibrant urban scene.<br>**----- End of picture text -----**<br>


Figure 7: Comparison of captions from the UMG-41M dataset, along with those re-generated by TextHawk2. The bounding boxes are provided in the text prompts but are not visible in the images. 

## 6.4 GROUNDING CAPTIONER 

Most large-scale visual grounding datasets, such as grounding captions and referring expressions, are generated using outdated specialist models or region-based captioners like GLIP (Li et al., 2022) and GPT4RoI (Zhang et al., 2023c). However, data quality plays a critical role in the final performance of LVLMs and often becomes a bottleneck for training more advanced models. Unlike generic image captions, which can be widely collected from the web and recaptioned by proprietary commercial APIs or open-source LVLMs, visual grounding data are much harder to generate and remain under-explored. This is primarily because current proprietary models lack strong grounding capabilities, and there is no high-performing open-source foundational LVLM specifically designed for grounding tasks. One potential solution for data augmentation of visual grounding involves a data-model-iteration loop, utilizing smaller detection models. A similar approach has been explored in VILA[2] (Fang et al., 2024), which introduces the concepts of self-augment and specialist-augment. In the specialist-augment step, an LVLM is fine-tuned on a high-quality subset of grounding captions and then used to recaption the remaining large-scale image dataset. It has been shown that image caption quality improves across up to three iterations. 

To assess the effectiveness of TextHawk2 as a grounding captioner, we compare the original region captions from UMG-41M with those generated by TextHawk2, which uses bounding boxes as additional inputs, as shown in Fig. 7. Although TextHawk2 is pre-trained on UMG-41M, its re-generated captions provide more detailed descriptions and better spatial relationships compared to the original captions. Unlike the specialist-augment method from VILA[2] , our approach integrates detection results from convolutional models, which produce more accurate bounding boxes. We believe this strategy can help address distribution issues that arise in data augmentation loops, and we plan to explore this direction further in future work. 

## 6.5 COMPARISON WITH PROPRIETARY MODELS 

While TextHawk2 is designed for computational efficiency and optimized for fine-grained tasks, it also demonstrates strong performance on general VQA tasks. We conduct a comprehensive comparison with proprietary models across various benchmarks, including general multimodal understanding and OCR tasks. Grounding tasks are not shown here since they have limited support. The benchmarks we consider are MMMU (Yue et al., 2023), MMBench (Liu et al., 2023d), MME (Fu et al., 2023), MMStar (Chen et al., 2024a), BLINK (Fu et al., 2024), MMT-Bench (Ying et al., 2024), RealWorldQA (X.AI, 2024), SEED-Bench (Li et al., 2023a), AI2D (Kembhavi et al., 2016), ScienceQA (Lu et al., 2022), MathVista (Lu et al., 2024b), HallusionBench (Liu et al., 2023a), 

14 

Technical Report 

Table 3: Performance comparison with proprietary models on vision-language benchmarks. Results are evaluated in VLMEvalKit (Duan et al., 2024) using official APIs by default. 

|**Benchmark**<br>**GPT-4o-mini**|**Gemini-1.5-Flash**<br>**Claude3-Haiku**<br>**TextHawk2**|
|---|---|
|MMMUval (Yue et al., 2023)<br>**60.0**<br>MMBench-1.1test-EN (Liu et al., 2023d)<br>**77.1**<br>MMBench-1.1test-CN (Liu et al., 2023d)<br>75.0<br>MMBenchtest-EN (Liu et al., 2023d)<br>77.6<br>MMBenchtest-CN (Liu et al., 2023d)<br>75.9<br>MME (Fu et al., 2023)<br>2003.4<br>MMStar (Chen et al., 2024a)<br>54.8<br>BLINK (Fu et al., 2024)<br>53.6<br>MMT-Benchval (Ying et al., 2024)<br>61.2<br>RealWorldQA (X.AI, 2024)<br>67.1<br>SEED-Benchimg (Li et al., 2023a)<br>72.8<br>AI2Dtest (Kembhavi et al., 2016)<br>77.8<br>ScienceQAtest (Lu et al., 2022)<br>85.4<br>MathVistatestmini (Lu et al., 2024b)<br>52.4<br>HallusionBench (Liu et al., 2023a)<br>46.1|58.2<br>49.7<br>45.0<br>**77.1**<br>58.0<br>75.0<br>**76.7**<br>56.2<br>75.6<br>**79.4**<br>60.7<br>77.5<br>**78.6**<br>57.2<br>77.6<br>2077.9<br>1453.2<br>**2125.9**<br>**55.8**<br>38.1<br>54.5<br>**57.7**<br>37.5<br>48.7<br>**62.6**<br>50.0<br>56.4<br>**69.0**<br>45.5<br>66.8<br>**75.0**<br>63.3<br>74.3<br>**78.5**<br>65.6<br>75.7<br>83.3<br>-<br>**85.8**<br>51.2<br>42.2<br>**54.5**<br>48.5<br>39.2<br>**49.5**|
|TextVQAval (Singh et al., 2019)<br>-<br>OCRBench (Liu et al., 2023e)<br>**78.5**<br>ChartQAtest (Masry et al., 2022)<br>26.3<br>DocVQAtest (Mathew et al., 2021)<br>70.1|**78.7***<br>-<br>76.1<br>75.3<br>65.8<br>78.4<br>**85.4***†<br>81.7*†<br>**81.4**<br>**89.9***‡<br>88.8<br>**89.6**|



TextVQA (Singh et al., 2019), OCRBench (Liu et al., 2023e), ChartQA (Masry et al., 2022), and DocVQA (Mathew et al., 2021). As illustrated in Table 3, TextHawk2 achieves competitive results with similar-scale closed-source models in most benchmarks. Notably, TextHawk2 scores 49.5% on HallusionBench, despite lacking supervision from reinforcement learning methods like RLHFV (Yu et al., 2023b). This suggests that training on grounding tasks may help reduce hallucinations. The most significant gap between TextHawk2 and GPT-4o-mini is observed on MMMU, suggesting that TextHawk2 has limitations in advanced and complex tasks. This is likely due to insufficient knowledge and reasoning data and the disparity between foundational LLMs. In text-oriented benchmarks, TextHawk2 either matches or surpasses state-of-the-art models, which utilize OCR engines or Chain-of-Thought (CoT) prompting during inference. 

## 7 CONCLUSION AND LIMITATIONS 

In this work, we address two key questions: _Can we increase the compression ratio to 16 without losing the ability to perceive fine-grained details and achieve state-of-the-art OCR performance with limited resources?_ And _can we train an LVLM with a single visual encoder that excels in general multimodal understanding, OCR, and grounding simultaneously?_ To answer these, we introduce TextHawk2, which demonstrates state-of-the-art performance in multimodal understanding, OCR, and grounding, all while achieving a 16 times token compression ratio with a unified visual encoder. Notably, TextHawk2 is pre-trained on a relatively modest dataset of 100 million samples— fewer than comparable LVLMs—highlighting the significance of visual encoder reinforcement and data diversity. Meanwhile, we optimize the data pipeline and model parallelism to boost training throughput, allowing TextHawk2 to be trained using limited resources. 

However, our experiments face several limitations. First, the training data contains insufficient scene text, limiting the model’s ability to accurately recognize complex Chinese characters. Second, the supervised fine-tuning process lacks adequate multimodal knowledge and reasoning data, which affects performance in these areas. Third, the potential of native resolution ViT and full-parameter pre-training remains unexplored. Lastly, the current version of TextHawk2 does not incorporate Reinforcement Learning from Human Feedback (RLHF), which could help reduce hallucinations. Addressing these limitations will be essential in future work. 

> *indicates results from official reports. 

> †indicates Chain-of-Thought prompting. 

> ‡indicates using extra annotations from OCR engine. 

15 

Technical Report 

## REFERENCES 

- AI@Meta. Llama 3 model card. https://github.com/meta-llama/llama3/blob/ main/MODEL_CARD.md, 2024. 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In _NeurIPS_ , 2022. 

- Ascend. Ascend extension for pytorch. https://www.hiascend.com/document/ detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000142.html, 2024. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _CoRR_ , abs/2308.12966, 2023. 

- Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Ziqiang Liu, Junting Zhou, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, Ruibin Yuan, Haihong Wu, Hongquan Lin, Wenhao Huang, Jiajun Zhang, Wenhu Chen, Chenghua Lin, Jie Fu, Min Yang, Shiwen Ni, and Ge Zhang. COIG-CQIA: quality is all you need for chinese instruction fine-tuning. _CoRR_ , abs/2403.18058, 2024. 

- Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez i Bigorda, Marçal Rusiñol, C. V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Scene text visual question answering. In _ICCV_ , 2019. 

- Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, and et al. Internlm2 technical report. _CoRR_ , abs/2403.17297, 2024. 

- Jianjian Cao, Peng Ye, Shengze Li, Chong Yu, Yansong Tang, Jiwen Lu, and Tao Chen. MADTP: multimodal alignment-guided dynamic token pruning for accelerating vision-language transformer. _CoRR_ , abs/2403.02991, 2024. 

- Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In _CVPR_ , 2021. 

- Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. _CoRR_ , abs/2306.15195, 2023. 

- Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large visionlanguage models? _CoRR_ , abs/2403.20330, 2024a. 

- Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact: A large-scale dataset for table-based fact verification. In _ICLR_ , 2020. 

- Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _CoRR_ , abs/2404.16821, 2024b. 

16 

Technical Report 

- Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy. https://internvl.github.io/blog/2024-07-02-InternVL-2.0, 2024c. 

- Chee Kheng Chng, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, Chee Seng Chan, Lianwen Jin, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, and Junyu Han. ICDAR2019 robust reading challenge on arbitrary-shaped text - rrc-art. In _ICDAR_ , 2019. 

- DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. _CoRR_ , abs/2405.04434, 2024. 

- Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k HD. _CoRR_ , abs/2404.06512, 2024. 

- Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. _CoRR_ , abs/2407.11691, 2024. 

- Xiaoran Fan, Tao Ji, Changhao Jiang, Shuo Li, Senjie Jin, Sirui Song, Junke Wang, Boyang Hong, Lu Chen, Guodong Zheng, Ming Zhang, Caishuang Huang, Rui Zheng, Zhiheng Xi, Yuhao Zhou, Shihan Dou, Junjie Ye, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. Mousi: Poly-visual-expert vision-language models. _CoRR_ , abs/2401.17221, 2024. 

- Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jang Hyun Cho, Marco Pavone, Song Han, and Hongxu Yin. Vila[2] : VILA augmented VILA. _CoRR_ , abs/2407.17453, 2024. 

- Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. _CoRR_ , abs/2306.13394, 2023. 

- Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: multimodal large language models can see but not perceive. _CoRR_ , abs/2404.12390, 2024. 

- Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model. _CoRR_ , abs/2312.11370, 2023. 

- Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, Chunjing Xu, and Hang Xu. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. In _NeurIPS_ , 2022. 

17 

Technical Report 

Yunfei Guo, Wei Feng, Fei Yin, Tao Xue, Shuqi Mei, and Cheng-Lin Liu. Learning to understand traffic signs. In _ACM MM_ , 2021. 

- Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, and Phillip B. Gibbons. Pipedream: Fast and efficient pipeline parallel DNN training. _CoRR_ , abs/1806.03377, 2018. 

- Conghui He, Zhenjiang Jin, Chao Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, Jiaqi Wang, and Dahua Lin. Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models. _CoRR_ , abs/2308.10755, 2023. 

- Mengchao He, Yuliang Liu, Zhibo Yang, Sheng Zhang, Canjie Luo, Feiyu Gao, Qi Zheng, Yongpan Wang, Xin Zhang, and Lianwen Jin. ICPR2018 contest on robust reading for multi-type web images. In _ICPR_ , 2018. 

Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. _CoRR_ , abs/2402.11530, 2024. 

- Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. _CoRR_ , abs/2312.08914, 2023. 

- Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _CoRR_ , abs/2403.12895, 2024a. 

- Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding. _CoRR_ , abs/2409.03420, 2024b. 

- Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Xu Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. In _NeurIPS_ , 2019a. 

- Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. ICDAR2019 competition on scanned receipt OCR and information extraction. In _ICDAR_ , 2019b. 

- Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. FUNSD: A dataset for form understanding in noisy scanned documents. In _OST@ICDAR_ , 2019. 

- Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman K. Ghosh, Andrew D. Bagdanov, Masakazu Iwamura, Jiri Matas, Lukás Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, and Ernest Valveny. ICDAR 2015 competition on robust reading. In _ICDAR_ , 2015. 

- Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara L. Berg. Referitgame: Referring to objects in photographs of natural scenes. In _EMNLP_ , 2014. 

- Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In _ECCV_ , 2016. 

- Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloé Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment anything. In _ICCV_ , 2023. 

- Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. In _MLSys_ . mlsys.org, 2023. 

- Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _IJCV_ , 2017. 

18 

Technical Report 

- Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, and Xiang Bai. Visual information extraction in the wild: Practical dataset and end-to-end solution. In _ICDAR_ , 2023. 

- Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? _CoRR_ , abs/2405.02246, 2024. 

- Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022. 

- Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, José G. Moreno, and Jesús Lovón-Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In _SIGIR_ , 2022. 

- David D. Lewis, Gady Agam, Shlomo Argamon, Ophir Frieder, David A. Grossman, and Jefferson Heard. Building a test collection for complex document information processing. In _ACM MM_ , 2006. 

- Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild. https://llava-vl.github.io/blog/ 2024-05-10-llava-next-stronger-llms/, 2024a. 

- Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. _CoRR_ , abs/2307.16125, 2023a. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping languageimage pre-training with frozen image encoders and large language models. In _Proc. ICML_ , 2023b. 

- Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In _CVPR_ , 2022. 

- Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. Pytorch distributed: Experiences on accelerating data parallel training. _Proc. VLDB Endow._ , 13, 2020. 

- Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. _CoRR_ , abs/2403.18814, 2024b. 

- Zejun Li, Ruipu Luo, Jiwen Zhang, Minghui Qiu, and Zhongyu Wei. Vocot: Unleashing visually grounded multi-step reasoning in large multi-modal models. _CoRR_ , abs/2405.16919, 2024c. 

- Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In _CVPR_ , 2023c. 

- Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Qiao. SPHINX: the joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. _CoRR_ , abs/2311.07575, 2023. 

- Chaohu Liu, Kun Yin, Haoyu Cao, Xinghua Jiang, Xin Li, Yinsong Liu, Deqiang Jiang, Xing Sun, and Linli Xu. HRVDA: high-resolution visual document assistant. _CoRR_ , abs/2404.06918, 2024a. 

- Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v(ision), llava-1.5, and other multi-modality models. _CoRR_ , abs/2310.14566, 2023a. 

19 

Technical Report 

Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. MMC: advancing multimodal chart understanding with large-scale instruction tuning. In _NAACL-HLT_ , 2024b. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _NeurIPS_ , 2023b. 

- Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: marrying DINO with grounded pre-training for open-set object detection. _CoRR_ , abs/2303.05499, 2023c. 

- Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? _CoRR_ , abs/2307.06281, 2023d. 

- Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, and Xiang Bai. On the hidden mystery of OCR in large multimodal models. _CoRR_ , abs/2305.07895, 2023e. 

- Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. _CoRR_ , abs/2403.04473, 2024c. 

- Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding. _CoRR_ , abs/2403.05525, 2024a. 

- Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In _NeurIPS_ , 2022. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _ICLR_ , 2024b. 

- Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model. _CoRR_ , abs/2309.11419, 2023. 

- Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In _CVPR_ , 2016. 

- Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In _ACL_ , 2022. 

- Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for VQA on document images. In _WACV_ , pp. 2199–2208, 2021. 

- Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In _WACV_ , 2022. 

- Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. Chartassisstant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. _CoRR_ , abs/2401.02384, 2024. 

- Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on GPU clusters using megatron-lm. In _SC_ , 2021. 

- Nibal Nayef, Cheng-Lin Liu, Jean-Marc Ogier, Yash Patel, Michal Busta, Pinaki Nath Chowdhury, Dimosthenis Karatzas, Wafa Khlif, Jiri Matas, Umapada Pal, and Jean-Christophe Burie. ICDAR2019 robust reading challenge on multi-lingual scene text detection and recognition - RRCMLT-2019. In _ICDAR_ , 2019. 

20 

Technical Report 

- Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. _CoRR_ , abs/2304.07193, 2023. 

- Vicente Ordonez, Girish Kulkarni, and Tamara L. Berg. Im2text: Describing images using 1 million captioned photographs. In _NeurIPS_ , 2011. 

- Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In _ACL_ , 2015. 

- Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. _CoRR_ , abs/2306.14824, 2023. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _Proc. ICML_ , volume 139, 2021. 

- Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In _SC_ , 2020. 

- Tal Ridnik, Emanuel Ben Baruch, Asaf Noy, and Lihi Zelnik. Imagenet-21k pretraining for the masses. In _NeurIPS_ , 2021. 

- Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: open dataset of clip-filtered 400 million image-text pairs. _CoRR_ , abs/2111.02114, 2021. 

- Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. KVQA: knowledgeaware visual question answering. In _AAAI_ , 2019. 

- Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. _CoRR_ , abs/2403.15388, 2024. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _ACL_ , 2018. 

- Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge J. Belongie, Shijian Lu, and Xiang Bai. ICDAR2017 competition on reading chinese text in the wild (RCTW17). In _ICDAR_ , 2017. 

- Bowen Shi, Peisen Zhao, Zichen Wang, Yuhang Zhang, Yaoming Wang, Jin Li, Wenrui Dai, Junni Zou, Hongkai Xiong, Qi Tian, and Xiaopeng Zhang. UMG-CLIP: A unified multi-granularity vision generalist for open-world understanding. _CoRR_ , abs/2401.06397, 2024. 

- Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. _CoRR_ , abs/1909.08053, 2019. 

- Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In _CVPR_ , 2019. 

- Brandon Smock, Rohith Pesala, and Robin Abraham. Pubtables-1m: Towards comprehensive table extraction from unstructured documents. In _CVPR_ , 2022. 

- StabilityAI and LAION. Renderedtext. https://huggingface.co/datasets/ wendlerc/RenderedText, 2023. 

- Yipeng Sun, Dimosthenis Karatzas, Chee Seng Chan, Lianwen Jin, Zihan Ni, Chee Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, and Jingtuo Liu. ICDAR 2019 competition on large-scale street view text with partial labeling - RRC-LSVT. In _ICDAR_ , 2019. 

21 

Technical Report 

- Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants. https: //huggingface.co/datasets/teknium/OpenHermes-2.5, 2023. 

- Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. YFCC100M: the new data in multimedia research. _Commun. ACM_ , 2016. 

- Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. _CoRR_ , abs/2406.16860, 2024. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971, 2023. 

- Andreas Veit, Tomas Matera, Lukás Neumann, Jiri Matas, and Serge J. Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. _CoRR_ , abs/1601.07140, 2016. 

- Bin Wang, Zhuangcheng Gu, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition. _CoRR_ , abs/2404.15254, 2024a. 

- Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting GPT-4V for better visual instruction tuning. _CoRR_ , abs/2311.07574, 2023a. 

- Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. OFA: unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _Proc. ICML_ , 2022. 

- Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. ONE-PEACE: exploring one general representation model toward unlimited modalities. _CoRR_ , abs/2305.11172, 2023b. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _CoRR_ , abs/2409.12191, 2024b. 

- Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. _CoRR_ , abs/2311.03079, 2023c. 

- Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. In _NeurIPS_ , 2023d. 

- Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In _CVPR_ , 2020. 

WebDataset. Webdataset. https://webdataset.github.io/webdataset/, 2024. 

- Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. _CoRR_ , abs/2312.06109, 2023. 

- X.AI. Realworldqa. https://huggingface.co/datasets/xai-org/RealworldQA, 2024. 

- Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an LMM perceiving any aspect ratio and highresolution images. _CoRR_ , abs/2403.11703, 2024. 

22 

Technical Report 

- Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei A. F. Florêncio, Cha Zhang, and Furu Wei. XFUND: A benchmark dataset for multilingual visually rich form understanding. In _ACL_ , 2022. 

- Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In _CVPR_ , 2023. 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. _CoRR_ , abs/2407.10671, 2024. 

- Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A GPT-4V level MLLM on your phone. _CoRR_ , abs/2408.01800, 2024. 

- Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-docowl: Modularized multimodal large language model for document understanding. _CoRR_ , abs/2307.02499, 2023a. 

- Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Fei Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In _EMNLP_ , 2023b. 

- Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality. _CoRR_ , abs/2304.14178, 2023c. 

- Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI. In _Proc. ICML_ , 2024. 

- Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. _CoRR_ , abs/2310.07704, 2023. 

- Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _TACL_ , 2, 2014. 

- Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Xinlong Wang, and Jingjing Liu. Capsfusion: Rethinking image-text data at scale. _CoRR_ , abs/2310.20550, 2023a. 

- Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, and Tat-Seng Chua. RLHF-V: towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. _CoRR_ , abs/2312.00849, 2023b. 

- Ya-Qi Yu, Minghui Liao, Jihao Wu, Yongxin Liao, Xiaoyu Zheng, and Wei Zeng. TextHawk: Exploring efficient fine-grained perception of multimodal large language models. _CoRR_ , abs/2404.09204, 2024. 

- Tailing Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the wild. _JCST_ , 2019. 

23 

Technical Report 

- Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai. Syntaxaware network for handwritten mathematical expression recognition. In _CVPR_ , 2022. 

- Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. _CoRR_ , abs/2311.16502, 2023. 

- Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large language models from GLM-130B to GLM-4 all tools. _CoRR_ , abs/2406.12793, 2024. 

- Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _ICCV_ , 2023. 

- Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models. _CoRR_ , abs/2312.02949, 2023a. 

- Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, Sam Dodge, Keen You, Zhen Yang, Aleksei Timofeev, Mingze Xu, Hong-You Chen, Jean-Philippe Fauconnier, Zhengfeng Lai, Haoxuan You, Zirui Wang, Afshin Dehghan, Peter Grasch, and Yinfei Yang. Mm1.5: Methods, analysis & insights from multimodal llm fine-tuning. _CoRR_ , abs/2409.20566, 2024a. 

- Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, and Yinfei Yang. Ferret-v2: An improved baseline for referring and grounding with large language models. _CoRR_ , abs/2404.07973, 2024b. 

- Hesuo Zhang, Lingyu Liang, and Lianwen Jin. Scut-hccdoc: A new benchmark dataset of handwritten chinese text in unconstrained camera-captured documents. _PR_ , 2020. 

- Jiaxin Zhang, Wentao Yang, Songxuan Lai, Zecheng Xie, and Lianwen Jin. Dockylin: A large multimodal model for visual document understanding with efficient visual slimming. _CoRR_ , abs/2406.19101, 2024c. 

- Jiwen Zhang, Yaqi Yu, Minghui Liao, Wentao Li, Jihao Wu, and Zhongyu Wei. UI-Hawk: Unleashing the screen stream understanding for gui agents. _Preprints_ , manuscript/202408.2137, 2024d. 

- Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. _CoRR_ , abs/2309.15112, 2023b. 

- Rui Zhang, Mingkun Yang, Xiang Bai, Baoguang Shi, Dimosthenis Karatzas, Shijian Lu, C. V. Jawahar, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, and Minghui Liao. ICDAR 2019 robust reading challenge on reading chinese text on signboard. In _ICDAR_ , 2019. 

- Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. _CoRR_ , abs/2307.03601, 2023c. 

- Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. _CoRR_ , abs/2306.17107, 2023d. 

24 

