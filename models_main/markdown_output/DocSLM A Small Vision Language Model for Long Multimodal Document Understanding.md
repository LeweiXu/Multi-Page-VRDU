# **DocSLM: A Small Vision-Language Model for Long Multimodal Document Understanding** 

Tanveer Hannan[1] _[,]_[2] _[,]_[3][*] Dimitrios Mallios[1] Parth Pathak[4] Faegheh Sardari[1] Thomas Seidl[2] _[,]_[3] Gedas Bertasius[5] Mohsen Fayyaz[1] Sunando Sengupta[1] 1 Microsoft 2 LMU Munich 3 MCML 4 FAIR Meta 5 UNC Chapel Hill 

## **Abstract** 

_Large Vision–Language Models (LVLMs) have demonstrated strong multimodal reasoning capabilities on long and complex documents. However, their high memory footprint makes them impractical for deployment on resourceconstrained edge devices. We present DocSLM, an efficient Small Vision–Language Model designed for long document understanding under constrained memory resources. DocSLM incorporates a Hierarchical Multimodal Compressor that jointly encodes visual, textual, and layout information from each page into a fixed-length sequence, greatly reducing memory consumption while preserving both local and global semantics. To enable scalable processing over arbitrarily long inputs, we further introduce a Streaming Abstention mechanism that operates on document segments sequentially and filters low-confidence responses through an entropy-based uncertainty calibrator. Across multiple long multimodal document benchmarks, DocSLM matches or surpasses state-of-the-art methods while using 82% fewer visual tokens, 75% fewer parameters, and 71% lower latency—delivering reliable multimodal document understanding on lightweight edge devices. Code and Model are available in https://github.com/Tanveer81/DocSLM.git._ 

## **1. Introduction** 

Large Vision-Language Models (LVLMs) have made remarkable progress in understanding multimodal documents that integrate text, figures, and visual elements [18, 24, 38]. These capabilities are particularly important for understanding financial and technical reports, industrial documents, presentation slides, and scientific papers. Despite recent progress, scaling vision–language models to longer context lengths remains a fundamental challenge, especially under constrained memory resources. LVLMs frequently exceed 8B parameters and can reach the total memory ca- 

> *Work done during an internship at Microsoft. Correspondence to: hannan@dbs.ifi.lmu.de 

Figure 1. **Model accuracy versus token efficiency** on MMLongDoc [36]. DocSLM achieves a +9.3% gain over DocOwl2 [24] with a comparable Tokens/Image budget, while using 75% fewer parameters than large RAG-based models such as InternVL2RAG [46], and outperforming the similarly sized Docopilot2B [18] by +0.9% despite its significantly larger token budget. 

pacity of typical edge GPUs [41] during inference (Fig. 2), which makes deployment on mobile or embedded devices particularly difficult [9, 18, 24, 28, 30, 38, 54, 55]. However, model size is only part of the cost: the number of input tokens greatly increases computational and memory demands. Document-focused LVLMs must process tens of pages rather than a single image, and each page can contain dense text, tables, figures, and complex layouts that inflate token counts. The burden rises further when systems incorporate OCR signals to read textual content, since many recent approaches feed OCR cues alongside visual tokens into the model [21, 49, 53, 55]. OCR-enhanced methods improve accuracy in reading documents but also amplify memory and compute requirements, compounding the challenge of efficient scaling and practical deployment on resource-constrained hardware. 

To minimize input context, recent Retrieval-Augmented Generation (RAG) frameworks, such as [8, 10, 40, 46, 52], 

1 

shorten document length by retrieving only the most relevant pages. However, RAG methods often rely on segmented document retrieval and multi-stage query pipelines, which fragment contextual information and introduce additional retrieval latency [18]. InternVL2-RAG [46] exhibits a token generation latency of 113.4,ms, which is 3.5 _×_ slower than compact non-RAG models [18, 42], and also incurs significantly higher memory usage (Fig. 2), thereby limiting its practicality for interactive or edge-device scenarios. 

To reduce memory consumption, recent methods use Smaller Vision–Language Models (SVLMs) [9, 18, 25, 53]. Although they save memory by reducing parameter counts, they still depend on dense visual encodings (3K–9K tokens per document page) to compensate for limited modeling capacity. Such high token counts quickly exceed the context length and memory capacity of mobile or edge devices, hindering scalability to multi-page documents. For instance, the state-of-the-art small model Docopilot-2B [46] requires about 3,133 tokens per image, which nearly saturates the input capacity of edge GPUs (1,440–4,320 tokens) and thus accommodates only a single image per inference, making multi-page document processing infeasible. On the other hand, DocOwl2 [24] offers the lowest token count per image but degrades performance due to over-aggressive compression (Fig. 1). 

To this end, we introduce **DocSLM** , a lightweight Vision–Language Model designed for reliable long-document understanding under strict memory and input token-length constraints. Documents naturally consist of sequences of pages, each containing multimodal information. Consequently, addressing the memory challenges of documentunderstanding VLMs requires solutions at both the page level and the document level. At the page level, we propose a **Hierarchical Multimodal Compression** module that jointly encodes visual, textual, and layout features from each document page into a **fixed 576 tokens** —independent of the number of OCR tokens—while preserving finegrained semantic and spatial information. As shown in Fig. 2, this compression method substantially reduces memory overhead compared to similar sized models Docopilot2B [18] and InternVL2-2B [42]. 

However, reducing per-page memory footprint alone is not sufficient. Even with efficient page-level encoding, memory usage still grows rapidly as the number of pages increases (Fig. 2). To address this document-level challenge, we propose a **Streaming Abstention** mechanism that processes long documents sequentially in a segment-wise manner. The input document is divided into segments, each independently encoded to produce an intermediate prediction along with an uncertainty score. This ensures a constant memory footprint regardless of document length. While each segment is encoded independently to maintain a constant memory footprint, DocSLM preserves contextual con- 

Figure 2. **Peak GPU memory usage vs. number of document pages.** We evaluate all models using their official implementations under identical inference settings, progressively increasing the number of document pages. Peak GPU memory is measured using PyTorch profiling tools. DocSLM achieves the lowest memory footprint among all methods, maintaining a constant plateau of _∼_ **14 GB** beyond 10 pages due to its streaming mechanism, enabling scalable inference on resource-constrained devices. 

tinuity through a streaming mechanism that implicitly carries information across segments via textual cues and the model’s calibrated uncertainty. This allows the model to achieve full-document understanding without storing crosssegment activations. 

With this design, DocSLM can process up to 120-page documents from MMLongDocBench [36] with a _∼_ **14 GB peak GPU memory** . Finally, an uncertainty calibrator aggregates all valid segment-level predictions and selects the most reliable document-level answer based on uncertainty. Together, these components enable DocSLM to handle arbitrarily long documents efficiently under limited GPU or edge-device memory, while providing stable and accurate document-level understanding across segments. Our contributions can be summarized as follows: 

- We introduce DocSLM , a compact 2B-parameter Vision–Language Model that uses **82%** fewer visual tokens and **75%** fewer parameters than existing large LVLMs and retrieval-augmented models. 

- We propose a Hierarchical Multimodal Compression module that achieves a **5.6** _×_ reduction in input tokens by jointly encoding visual, textual, and layout features, while a Streaming Abstention mechanism maintains a constant memory footprint—enabling efficient inference over arbitrarily long documents across edge devices. 

- Despite its compact size, DocSLM achieves state-of-theart performance, surpassing DocOwl2-8B [24] by **+9.3%** under a comparable token budget and outperforming the similarly sized Docopilot-2B [18] by **+0.9%** , while reducing latency by **3.5** _×_ (32.1 ms vs. 113.4 ms) compared to InternVL2-RAG [46]. 

2 

## **2. Related Works** 

**Document Understanding.** OCR-free models [4, 31, 34, 42] typically process high-resolution document images or tiled patches to capture global visual context but often struggle with densely packed text regions. On the other hand, OCR-enhanced approaches [6, 7, 20, 45, 51] extract textual content and layout information, resulting in long input sequences, particularly for a multi-page document. Recent hybrid methods [21, 53] embed OCR text as language tokens for structured multimodal alignment, while layoutaware models [33, 47, 49] explicitly encode markup or spatial structures to enhance localized reasoning. Our main goal is to build an efficient document undesrstanding model that can run on resource-constrained edge devices. 

**Document Compression.** OCR-free models [4, 23, 24, 32] focus on specialized visual token compression but often fail to preserve fine-grained textual and layout cues essential for understanding text-heavy documents. In OCR-based paradigms, GRAM [7] introduces a Compression Transformer to aggregate OCR tokens across pages, improving long-document understanding at the cost of substantial model complexity. DocVLM [38] encodes textual semantics into fixed-size OCR embeddings but still exhibits linear token growth as the number of pages increases. In contrast, our hierarchical compression maintains a constant token count per page and does not incur any additional tokens from OCR inclusion, enabling scalable multimodal encoding for long documents. 

**Long Multimodal Document Understanding.** In addition to the increasing of input tokens size, long document understanding poses additional challenges due to complex inter-page dependencies. One line of existing approaches tackles this problem through long-context vision–language models [9, 18, 24, 28, 30, 38, 54, 55]. For example, DocVLM [38] mitigates input redundancy using fixed-size OCR embeddings, LayTokenLLM [55] encodes layoutaware OCR tokens without positional extrapolation, and Docopilot [17] fine-tunes off-the-shelf LVLMs on largescale instruction datasets. Meanwhile, RAG-based methods [8, 10, 40, 46, 52] retrieve relevant document pages or visual embeddings before generation, but introduce additional retrieval latency and still require thousands of input tokens per page—restricting scalability to long documents. We propose a streaming model to handle long documents with a constant input token and memory footprint. 

## **3. Method** 

Given a long multimodal document _D_ = _{d_[1] _, d_[2] _, . . . , d[N] }_ with _N_ pages where _d[n]_ is the _n[th]_ page and a natural language query _S_ , our goal is to predict a response _y_ that is consistent with the entire input. To achieve this under strict memory and context-length constraints, DocSLM in- 

Figure 3. **Hierarchical Multimodal Compressor.** The Vision Encoder produces global ( **V** _g_ ) and local ( **V** _i,j_ ) visual features, while the Grounded OCR module provides region-aligned text embeddings ( **T** _i,j_ ). ( **Bottom** ) As indicated by green boxes and dotted links, _Local OCR Compression_ performs spatially localized crossattention—each visual patch **V** _i,j_ attends only to its paired OCR tokens **T** _i,j_ —yielding compressed local features **V**[ˆ] _i,j_ . ( **Top** ) The _Global Visual Compression_ , shown with dotted red connections, aggregates these local representations by allowing the global visual **V** ˆ _i,j_ , producing the final global representationfeature **V** _g_ to attend selectively to compressed **V** ˆ _g[n]_[.] local regions 

troduces two key components: (1) a Hierarchical Multimodal Compression module that condenses visual, textual, and layout features into a compact token representation per page, and (2) a Streaming Abstention mechanism that enables reliable reasoning over arbitrarily long inputs. Overview of the method is in Fig. 4. 

## **3.1. Hierarchical Multimodal Compression** 

Our compression module performs structured token reduction through a two-stage fusion process (Fig. 3). In the first stage, local OCR compression aligns each visual region with its corresponding OCR text and layout using localized attention, merging them into compact region-level features. In the second stage, global visual compression aggregates these regional features into a fixed-length page representation that preserves both spatial alignment and overall document semantics. 

**Multimodal Feature Extraction.** The process starts with multimodal feature extraction. Specifically, for each document page _d[n]_ , we divide the image into a grid of _R × C_ spatial crops 

**==> picture [171 x 12] intentionally omitted <==**

3 

Figure 4. **Streaming Abstention.** To process long documents efficiently, DocSLM divides the input into shorter segments that can be handled sequentially or in parallel. Each segment produces an intermediate prediction that can either ( **Left** ) correctly answer the query with low uncertainty, ( **Middle** ) abstain when the answer is not present, or ( **Right** ) produce an incorrect answer with high uncertainty. A _Uncertainty Calibrator_ aggregates all valid segment predictions and selects the final document-level answer corresponding to the lowest uncertainty. In memory-limited settings, segments are processed sequentially, with memory from previous segments released before processing the next one. 

and obtain a downsampled global crop _d[n]_ g[.][Then,][a shared] vision encoder _Ev_ ( _·_ ) extracts patch-level visual features: 

**==> picture [194 x 13] intentionally omitted <==**

To incorporate textual cues, a lightweight OCR module produces _K[n]_ word–bounding-box pairs _{_ ( _s[n] k[, b][n] k_[)] _[}] k[K]_ =1 _[n]_[.] Each word token is embedded as **t** _[n] k_[=] _[ E][t]_[(] _[s][n] k_[)][ using the to-] kenizer of the Small Language Model (SLM). We then spatially associate OCR tokens with their corresponding image crops by bounding-box overlap: 

**==> picture [198 x 12] intentionally omitted <==**

where _τ_ is a fixed overlap threshold. This mapping yields region-aligned OCR sets **T** _[n] i,j_[for each visual crop] _[ d][n] i,j_[, en-] suring that subsequent local compression attends only to semantically and spatially relevant text regions. 

**Local OCR Compression.** At the local level, visual and text features within each region ( _i, j_ ) are fused using crossattention (CA): 

**==> picture [181 x 13] intentionally omitted <==**

where **V** _i,j_ serves as queries and **T** _i,j_ as keys and values. This enriches local visual tokens with corresponding OCR semantics without increasing the overall sequence length, achieving spatially aligned multimodal fusion. 

**Global Visual Compression.** The local features **V**[˜] _i, j_ are spatially aligned and preserve high-resolution details, but they result in long token sequences. To reduce the total number of tokens, these local representations **V**[˜] _i,j_ are summarized into compact global features **V** g through an additional cross-attention layer: 

**==> picture [190 x 14] intentionally omitted <==**

producing **V** ˆ _i,j_ . Subsequently,compact, allspatiallyregionalconsistentfeatures arerepresentationsconcatenated to form the final page-level representation: 

**==> picture [194 x 15] intentionally omitted <==**

This hierarchical compression ensures that each page—regardless of OCR token count—is represented by a fixed number of tokens while preserving both local fine-grained and global structural details. 

## **3.2. Streaming Abstention** 

Even after hierarchical multimodal compression, extremely long documents can still exceed device memory limits during inference. To address this, we introduce the Streaming Abstention mechanism (Fig. 4), which enables us to process arbitrarily long inputs under constant GPU memory usage. **Document Segmentation.** For a sequence of length _N_ , attention memory scales linearly with sequence length, i.e., _O_ ( _N_ ) per layer when using optimized kernels such as FlashAttention [14]. However, even linear growth becomes impractical on memory-constrained edge devices. To reduce the peak memory usage at any given time, we divide the document into _T_ smaller segments of equal length _N/T_ denoted as _{st}[T] t_ =1[.] 

Each segment can be processed with _O_ ( _N/T_ ) memory, reducing the peak GPU usage by roughly _T ×_ when processed sequentially. This segmentation strategy enables inference over extremely long sequences while keeping memory within device constraints (refer Tab. 3). While segmentwise processing guarantees constant-memory inference, it still requires an aggregation mechanism to coherently integrate information across segments. We therefore propose an uncertainty-guided aggregation approach that fuses the 

4 

|**Stage**|**Curriculum Goal**|**Trainable Modules**|**Dataset Source**[22,24]|
|---|---|---|---|
|Pretrain 1|Image–OCR Alignment|OCR Compressor|DocStruct4M (25%)|
|Pretrain 2|Image Compression|Vision Encoder, OCR & Vision Compressor|DocStruct4M (Remaining 75%)|
|Pretrain 3|Document Compression|OCR & Vision Compressor|DocStruct4M (Random 25%), MP-DocStruct1M|
|Finetune 1|Instruction Following|OCR & Vision Compressor, SLM|DocDownstream 1.0|
|Finetune 2|Streaming Abstention|OCR & Vision Compressor, SLM|DocDownstream 2.0, DocGenome12K, MP-DocReason51K|



Table 1. **Curriculum Training Stages.** The model is progressively trained from single-page pretraining to multi-document finetuning, with increasingly complex objectives and negative-pair supervision. An MLP adapter is trained in all stages. 

most confident segment outputs into a coherent documentlevel prediction—achieved in a single forward pass without any additional model calls. 

**Segment-Wise Processing.** Unlike traditional streaming models that retain key–value (KV) caches across segments [16, 48], DocSLM stores only the textual prediction and the SLM’s intrinsic uncertainty for each segment, avoiding large memory accumulation. For each segment _St_ , the Hierarchical Multimodal Compressor produces compressed embeddings: 

**==> picture [165 x 14] intentionally omitted <==**

which are fed into the SLM together with the query _S_ to generate a segment-level prediction: 

**==> picture [158 x 13] intentionally omitted <==**

Before releasing activation memory, DocSLM estimates the predictive uncertainty for each segment using token-level entropy, where _ut_ denotes the average uncertainty of the generated text distribution for segment _st_ . 

**==> picture [228 x 30] intentionally omitted <==**

After storing the prediction text and its corresponding uncertainty, all intermediate activations and KV caches from _st_ are released, maintaining a constant GPU memory. **Uncertainty-Based Aggregation.** Among the valid segment predictions _P_ valid, the final document-level answer is obtained by selecting the most confident one: 

**==> picture [156 x 16] intentionally omitted <==**

DocSLM produces calibrated uncertainty estimates through its learned abstention mechanism during Finetuning Stage 2 (Sec. 3.3). It enables robust evidence aggregation across arbitrarily long documents. Notably, sequential processing does not compromise accuracy; in fact, the uncertainty-guided aggregation enhances performance by emphasizing confident evidence (Tab. 3). We also explored hierarchical aggregation, where high-confidence predictions are reused as input to the SLM. While yielding slight accuracy gains, these methods incurred extra computation and latency, making them less suitable for edge deployment. Hence, we adopt the single-pass uncertaintyguided selection for its balance of reliability and efficiency. 

## **3.3. Training** 

Tab. 1 summarizes the multi-stage curriculum used to train DocSLM. The training progresses from low-level singlepage pretraining to high-level multi-document finetuning with gradually increasing task complexity. 

**Pretraining Stages** focus on multimodal representation learning. In Pretraining 1, only the OCR Compressor is optimized to align textual embeddings with their corresponding visual regions. Pretraining 2 extends training to the Vision Encoder and both OCR and Vision Compressors, using single-page documents to learn consistent visual–text fusion. Finally, Pretraining 3 introduces multi-page documents, encouraging the model to encode coherent representations across multiple visual contexts. 

**Finetune Stage 1** serves as the first step in training the Small Language Model (SLM). Through instruction tuning on single-image inputs, the model learns to interpret multimodal cues and follow natural language instructions, establishing a foundation for subsequent multi-image and longdocument understanding. 

**Finetune Stage 2.** This stage extends training to multiimage and multi-document settings with _negative-pair supervision_ , where question–answer pairs are randomly mismatched with unrelated document segments to simulate incomplete or irrelevant context. For each positive segment _st_ containing evidence for a query _q_ , a negative counterpart _s[−] t_ is constructed from unrelated content where _q_ cannot be answered. DocSLM is trained to detect such mismatches and _abstain_ from unsupported predictions by generating an explicit _“Not Answerable”_ token sequence. This dual supervision over _{st, s[−] t[}]_[calibrates][model][confidence][between] valid and invalid contexts, ensuring reliable streaming inference under uncertain or partial evidence. 

Across all stages, the model is trained using a standard next-token prediction loss on instruction-tuned triplets _{_ ( _st, q, y_ ) _}_ , where _st_ denotes the input segment, _q_ the instruction query, and _y_ the target response: 

**==> picture [201 x 22] intentionally omitted <==**

This curriculum progressively transitions DocSLM from learning localized visual–text compression to performing robust, long-context multimodal understanding. 

5 

|**Model**<br>**Tok/Image**_↓_<br>**Param**_↓_<br>**Latency (ms)**_↓_<br>**MMLDoc (Acc)**_↑_<br>**MP-DocVQA (ANLS)**_↑_<br>**DUDE (ANLS)**_↑_<br>**NewsVQA (ANLS)**_↑_|**Model**<br>**Tok/Image**_↓_<br>**Param**_↓_<br>**Latency (ms)**_↓_<br>**MMLDoc (Acc)**_↑_<br>**MP-DocVQA (ANLS)**_↑_<br>**DUDE (ANLS)**_↑_<br>**NewsVQA (ANLS)**_↑_|
|---|---|
|**Large**<br>**Models**<br>LayTokenLLM [55]<br>Var.<br>8B<br>InternVL2 [9]<br>_∼_3,133<br>8B<br>Docopilot [18]<br>_∼_3,133<br>8B<br>Idefcs3 [28]<br>838<br>8B<br>DocOwl2 [24]<br>**324**<br>8B<br>LongVA [54]<br>_∼_2,029<br>7B<br>LLaVA-Next [30]<br>729<br>7B<br>DocVLM [38]<br>1088<br>7B|–<br>–<br>74.3<br>**52.0**<br>–<br>81.0<br>17.4<br>79.3<br>37.0<br>53.0<br>81.0<br>**28.8**<br>81.3<br>–<br>–<br>–<br>–<br>67.2<br>38.7<br>**60.2**<br>–<br>13.4<br>69.4<br>46.8<br>–<br>–<br>–<br>60.8<br>38.4<br>50.6<br>–<br>–<br>44.9<br>28.0<br>56.7<br>–<br>–<br>**84.5**<br>47.4<br>–|
|**RAG**<br>**Models**<br>VisRAG [52]<br>Var.<br>12B<br>InternVL2+RAG [46]<br>_∼_3,133<br>8B<br>M3DocRAG [10]<br>16,384<br>7B<br>SV-RAG [8]<br>3,072<br>4B<br>VDocRAG [40]<br>**768**<br>4B<br>InternVL2+RAG [46]<br>_∼_3,133<br>2B|288.3<br>18.8<br>–<br>–<br>36.3<br>113.4<br>24.2<br>78.7<br>–<br>–<br>–<br>21.0<br>**84.4**<br>–<br>–<br>–<br>**23.0**<br>71.0<br>45.0<br>**61.0**<br>–<br>–<br>–<br>**48.5**<br>44.2<br>**82.9**<br>17.2<br>72.6<br>–<br>–|
|**Small**<br>**Models**<br>DocThinker [53]<br>9,216<br>3B<br>MiniMonkey [25]<br>3,072<br>2B<br>InternVL2 [9]<br>_∼_3,133<br>2B<br>Docopilot [18]_∗_<br>_∼_3,133<br>2B|–<br>–<br>–<br>21.3<br>–<br>–<br>10.3<br>70.3<br>–<br>–<br>35.9<br>10.5<br>71.8<br>–<br>–<br>35.9<br>21.8<br>**76.2**<br>–<br>–|
|**Ours**<br>**576**<br>2B|**32.1**<br>**22.7**<br>70.0<br>**47.6**<br>**66.2**|



Table 2. **Main results on long-document benchmarks** across large-scale, retrieval-augmented, and compact vision-language models. “Var.” denotes variable-length inputs without a fixed tokenization limit. For RAG-based models, token counts refer to the generator module only (excluding retriever overhead). Best and second-best results per column are shown in **bold** and underline, respectively. Despite operating with only 576 tokens per image and 2B parameters, **DocSLM** matches or exceeds the performance of 7B–8B models and RAG-enhanced systems across most benchmarks, while achieving the lowest measured latency. 

## **4. Experimental Setup** 

## **4.1. DocSLM Baselines** 

Long-document understanding under memory constraints remains underexplored for Small Vision–Language Models (SVLMs). For fair comparison, we evaluate two backboneconsistent variants built on SigLIP2 [44] and Qwen2.5 [39], also used in our proposed DocSLM: 

**OCR-Free Baseline.** A LLaVA-Next–style [30] model that interleaves visual and text tokens without explicit OCR fusion, relying solely on dense visual embeddings. **OCR Baseline.** Extends the above by incorporating OCR text from PaddleOCR [11], serialized and appended to the visual tokens for joint visual–text attention. DocSLM builds upon the backbone of OCR-Free Baseline. 

## **4.2. Datasets and Metrics** 

**MP-DocVQA Dataset** [43] comprises approximately 46K QA pairs derived from 60K scanned pages of around 6K industrial documents, encompassing tables, diagrams, figures, and both handwritten and printed text. 

**DUDE Dataset** [27] expands the coverage to multiple realworld domains, including medical, legal, financial, and technical reports with 41K QA pairs across 5K documents. **MMLongBench-Doc Dataset** [37] extends the evaluation scope by incorporating documents with considerably greater lengths, averaging 47.5 and a maximum of 120 pages per document, respectively. 

**NewsVideoQA Dataset** [26] focuses on text-rich broadcast videos collected from major global news outlets such as BBC and CNN, providing 8K QA pairs grounded in 

3K news clips containing dynamic, text-heavy scenes. Although primarily a video QA benchmark, its frames often contain overlaid text, captions, and layout-rich visual elements similar to real-world documents. We include this dataset to evaluate the model’s ability to generalize its document understanding capabilities to temporally varying, textcentric visual content. 

**Evaluation Metric.** We evaluate our model on the multimodal _Document Question Answering_ (DocQA) task. Following prior works, we use the _Average Normalized Levenshtein Similarity_ (ANLS) [5] as the primary evaluation metric. ANLS computes the normalized edit similarity between predicted and ground-truth answers, averaged over all samples, with scores below a threshold of _τ_ = 0 _._ 5 set to zero. For MMLongDocBench, which contains longdocument question–answer pairs, the same thresholding rule is applied, but the metric reports binary _Accuracy_ . 

## **4.3. Implementation Details.** 

In our framework, we use SigLIP2 [44] as the vision encoder, PaddleOCR [11] for OCR extraction, and Qwen2.51.5B [39] as the Small Language Model (SLM). Training is performed using Fully Sharded Data Parallel (FSDP) across 8 nodes, each equipped with 4 NVIDIA A100 (80 GB) GPUs, resulting in a total of 32 GPUs. We use a total batch size of 1024 during pretraining and 256 during finetuning. Learning rate is set to 1 _e −_ 4 for pretraining and 2 _e −_ 5 for finetuning. Training steps count and batch size for each stage are listed in Tab. 4. We use the AdamW [35] optimizer with a cosine learning rate schedule and an initial warm-up phase. More details are in the Supplementary material Section S1 and Section S2. 

6 

## **5. Experiments** 

**Comparison with Large and RAG Models.** Tab. 2 compares DocSLM with large-scale and retrieval-augmented approaches. Large LVLMs, such as DocVLM-7B [38] (1,088 input tokens per image) and Docopilot-8B [18] (3,133 input tokens per image) achieve strong accuracy but incur high memory requirements, with inference latencies of 81–113 ms. RAG-based methods, including InternVL2+RAG [46] and M3DocRAG [10] introduce additional retrieval overhead, often exceeding 110 ms per sample. In contrast, DocSLM operates with only **576** tokens per image— **5.4** _×_ fewer than large models—achieving 22.7% on MMLDoc, 70.0 ANLS on MP-DocVQA, and 47.6 ANLS on DUDE at just **32.1 ms latency** . This corresponds to a **+5.7 pp gain** over InternVL2-8B on DUDE and near-parity with the 8B Docopilot on MMLDoc despite using **75% fewer parameters** . Even compared to 8B RAG models, DocSLM retains over **95%** of their accuracy while running **3.5** _×_ **faster** , underscoring the efficiency of multimodal compression for resource-constrained reasoning. 

**Comparison with Small Vision–Language Models.** Within the 2–3B parameter range, DocSLM achieves the best trade-off between efficiency and accuracy (Tab. 2). Compared to Docopilot-2B [18] (3,133 tokens, 35.9 ms), it runs faster ( **32.1 ms** ) while improving by **+0.9 pp** on MMLDoc and **+26.3 pp** on DUDE. Relative to InternVL2-2B [9], DocSLM gains **+12.2 pp** on MMLDoc and **+47.6 pp** on DUDE, highlighting its superior multimodal reasoning and text–layout alignment under tight token budgets. 

## **5.1. Generalization to Video Question Answering.** 

Despite being trained mainly on documents, with only 8.6K video samples versus 6.75M document annotations, DocSLM generalizes effectively, achieving state-of-the-art ANLS of **66.2** (Tab. 2). It surpasses larger models including Idefics3 [28], LLaVA-Next [30], and SV-RAG [8] by **+6.0** , **+9.5** , and **+5.2 pp** , respectively, while using **5.4** _×_ fewer visual tokens and **75%** fewer parameters. These results highlight DocSLM’s strong cross-modal generalization, particularly valuable for edge devices, as it eliminates the need to load separate models for different domains. 

## **5.2. Ablation Studies** 

We report ablation on the Mp-DocVQA dataset [43] and LLaVA-NeXT [30] as baseline model. Additional experiments are in Supplementary Section S3 and Section S4. **Ablation on the Proposed Modules.** In Tab. 3, we begin with the OCR baseline, which achieves 50.2 ANLS but relies on dense tokenization. Removing OCR tokens results in a sharp performance drop to 38.5, confirming their essential role in text-grounded reasoning. Applying Visual Compression reduces the token count by **5.6×** (3210→576), 

|**Modules**<br>**Tok/Image**_↓_<br>**Memory(GB)**_↓_<br>**ANLS**_↑_|**Modules**<br>**Tok/Image**_↓_<br>**Memory(GB)**_↓_<br>**ANLS**_↑_|
|---|---|
|||
|OCR Baseline [30]<br>OCR-Free Baseline [30]<br>(+) Visual Compression<br>(+) OCR Compression(no Visual)<br>(+) Visual + OCR Compression|3210<br>29.2<br>50.2<br>2880<br>27.9<br>38.5<br>576<br>23.5<br>22.7<br>2880<br>27.9<br>68.3<br>576<br>23.5<br>67.4|
|(+) Streaming Abstention_∗_|576<br>**14.2**<br>**70.0**|



Table 3. **Cumulative Ablation on Proposed Modules** on the Mp-DocVQA dataset [43]. Each proposed module progressively enhances text-rich visual understanding under strict token constraints. The Streaming Abstention mechanism achieves the highest accuracy while requiring the lowest GPU memory usage. _∗_ Indicates the final DocSLM model. 

but slightly decreases accuracy to 36.1 due to information loss from aggressive spatial downsampling. Introducing the OCR Compression module restores text fidelity, achieving 68.3 ANLS under a modest 2880-token budget. Furthermore, combining both OCR and visual compression achieves an optimal balance—maintaining only 576 tokens per image while preserving 67.4 ANLS. Finally, the Streaming Abstention module yields the best overall performance (70.0 ANLS) under the same token constraint. This progressive improvement illustrates how hierarchical compression, combined with streaming abstention, enables efficient and reliable long document understanding. 

|**Stage**|**Training Steps**|**Data Size**|**ANLS**_↑_|
|---|---|---|---|
|Instruction Tuning|3.0K|1.00M|38.5|
|(+) Image–OCR Alignment|9.0K|3.00M|50.5|
|(+) Image Compression|2.4K|2.00M|61.8|
|(+) Document Compression|3.0K|0.58M|66.7|
|(+)Streaming Abstention|4.4K|0.18M|**70.0**|



Table 4. **Cumulative Curriculum Training** on the Mp-DocVQA dataset [43]. Each stage incrementally introduces new objectives and datasets, improving ANLS from 38.5 to 70.0. Data size gradually decreases as task complexity increases, reflecting a shift from large-scale pretraining to specialized fine-tuning. 

**Ablation on the Training Stages.** Tab. 4 demonstrates the progressive improvements achieved through our staged curriculum learning. Starting from the Instruction Tuning baseline, the model is trained without any OCR input, achieving an ANLS of 38.5%. Introducing OCR supervision restores text grounding in the subsequent Image–OCR Alignment stage, with an ANLS score of 50.5%. Adding the Image Compression stage greatly improves efficiency by reducing tokens by 5.6× (3210→576) and increases ANLS to 61.8%. Incorporating Document Compression enhances long-context reasoning, achieving 66.7%. Finally, the proposed Streaming Abstention mechanism yields the best overall accuracy of 70.0%. 

**Ablation across Document Lengths.** As seen in Table 5, the baseline model performs well on single-page inputs, but collapses on longer ones, dropping from 75.3 on singlepage to just 0.7 on documents exceeding 10 pages. Incorporating OCR improves overall accuracy by +11.7 points, 

7 

Figure 5. **Qualitative examples** of long-document reasoning with DocSLM. For each user query (bottom), DocSLM first identifies the relevant document segment via its uncertainty-based ranking mechanism before generating the final answer. Queries 1–3 correspond sequentially to Segments 1–3. For **Query 1** , the model reasons over text-heavy content (Page 2) to correctly identify the _Berlin School of Experimental Psychology_ , demonstrating effective OCR fusion. For **Query 2** , it interprets visual figures to infer the shapes _Circle and Rectangle_ , highlighting strong visual understanding. For **Query 3** , DocSLM jointly reasons over textual and visual cues in a complex map, comparing multiple numeric indicators to correctly predict _Europe_ , illustrating its multimodal fine-grained understanding. More results are in Supplementary Section S5. 

||||**MP-DocVQA (ANLS)**_↑_|**MP-DocVQA (ANLS)**_↑_|**MP-DocVQA (ANLS)**_↑_|**(a) Compression Depth**|**(a) Compression Depth**|**(a) Compression Depth**|**(b) Compression Tuning**|**(b) Compression Tuning**|**(c) OCR Source**|**(c) OCR Source**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Model|**Tok/Image**_↓_|**1**|**[2,10]**|_>_**10**|**Overall**|**OCR**_↓_|**Visual**_↓_|**ANLS**_↑_|**Tunable Params**|**ANLS**_↑_|**Source**|**Acc**_↑_|
|Baseline|2880|75.3|29.7|0.7|38.5|4|4|**70.2**|Compressor|66.5|Tesseract|69.1|
|Baseline + OCR<br>Ours|3210<br>**576**|78.6<br>**79.6**|40.5<br>**70.0**|2.2<br>**61.2**|50.2<br>**70.0**|4<br>**2**|**2**<br>4|57.3<br>70.1|(+) Linear Adapter<br>(+) Visual Encoder|68.6<br>**70.0**|PaddleOCR|**70.0**|
|||||||**2**|**2**|70.0|||||



Table 5. **Ablation across document lengths.** ANLS scores (%) for varying numbers of pages. Our model achieves the best overall robustness, particularly on longer documents. 

but the gain remains marginal for long documents. In contrast, DocSLM delivers substantial improvements across all length achieving gains of +4.3, +40.3, and +60.5 ANLS for 1, [2–10], and _>_ 10-page documents, respectively—a far smaller degradation as the input grows. 

**Ablation on Compression Design and OCR Quality.** (a) As shown in Table 6, increasing the compression depth to 4 layers for both OCR and visual branches yields the best performance (70.2 ANLS), while a lightweight 2-layer configuration attains nearly identical accuracy (70.0) with 38.6M fewer parameters. We adopt this 2-layer setup as our default for all subsequent experiments. (b) Progressive finetuning—first adapting linear adapters, then the SigLIP vision encoder—further enhances performance from 66.5 to 70.0 ANLS, demonstrating the benefit of joint optimization across modalities. (c) Finally, OCR quality plays a pivotal role: PaddleOCR improves recognition accuracy to 

Table 6. **Ablations on compression design and OCR quality** on the Mp-DocVQA dataset [43]. (a) Balanced compression depth (4 layers each) yields the best accuracy, while a 2-layer setup achieves comparable performance with 38.6M fewer parameters. (b) Gradual fine-tuning of the adapter and encoder improves ANLS. (c) High-quality OCR provides a clear accuracy boost. 70.0 compared to 69.1 with Tesseract, indicating that reliable text extraction remains a crucial factor specially under aggressive token compression. 

## **6. Conclusion, Limitations, and Future Work** 

DocSLM employs a compact architecture and feature representation for efficient, reliable multimodal understanding under strict memory constraints, enabling deployment on resource-limited devices. Although trained on limited video data, it already demonstrates strong cross-modal generalization. Future work will extend to additional modalities like audio and pursue balanced document–video training toward an omnimodal foundation model for edge deployment. 

8 

## **DocSLM: A Small Vision-Language Model for Long Multimodal Document Understanding** 

## Supplementary Material 

Our supplementary materials contain Section S1: Additional Implementation Details, Section S2: Edge Deployment, and Section S3: Efficiency Analysis, Section S4: Additional Ablation Studies, Section S5: Additional Qualitative Results. 

## **S1. Additional Implementation Details** 

## **S1.1. OCR Integration** 

Our OCR pipeline is built around a bounding-box alignment mechanism (Fig. F1) that enables consistent OCR integration under multi-crop processing [30] to handle documents of any size and shape. As illustrated in Fig. F2, each input page is first resized, padded, and subdivided into a grid of non-overlapping crops. OCR tokens detected on the original image must therefore be remapped to these crops in a geometrically consistent manner. Fig. F1 and F2 show 4 regions for simplicity; however, based on aspect ratio and resolution, the number of crops can range from 4-16 in our default setup. As visualized in Fig. F1, each OCR token is represented by a bounding box in original-image coordinates, normalized by the image width and height. A token is assigned to every crop whose bounding box overlaps with it. This supports one-to-many assignments when a word spans crop boundaries and handles empty crops. The resulting crop-aligned OCR lists are then fused with the hierarchical multimodal compression module. This alignment mechanism ensures that multimodal training receives consistent and spatially grounded OCR information, even under highly variable document layouts and multi-resolution patch configurations. 

## **S1.2. Training Details** 

Table T1 summarizes the full five-stage training pipeline used to build our 2B-parameter model. The training strategy gradually transitions from large-scale noisy pretraining to highly curated downstream finetuning, while progressively increasing task difficulty and reducing learning rates. Pretrain 1 initializes the multimodal alignment by training the MLP adapter and hierarchical compressor on 1M weakly supervised image–text pairs using cross-attention–based fusion of SigLIP2[44] visual features and PaddleOCR[29] tokens. Pretrain 2 scales the same objective to a larger 3M corpus and unlocks the vision tower and multimodal compressor for joint optimization, improving cross-modal grounding. Pretrain 3 adapts the model to high-quality single-page document datasets (2M samples), introduces 

Figure F1. **OCR-to-crop assignment.** The OCR bounding boxes (red) are tested for overlap with the crop regions. An OCR token is assigned to a crop if its bounding boxes intersect, ensuring spatially consistent OCR alignment across crops. 

|**Stage**|**Training Steps**|**Batch**|**Data Size**|**LR**|
|---|---|---|---|---|
|Pretrain 1|3.0K|1.0K|1.00M|1_×_10_−_4|
|Pretrain 2|9.0K|1.0K|3.00M|1_×_10_−_4|
|Pretrain 3|2.4K|1.0K|2.00M|2_×_10_−_5|
|Finetune 1|3.0K|256|0.58M|2_×_10_−_5|
|Finetune 2|4.4K|256|0.18M|2_×_10_−_6|



Table T1. Each stage progressively adapts to more complex tasks, while the availability of high-quality data decreases. 

early-layer OCR and visual compression, and begins tuning the language model to better handle structured document semantics. Finetune 1 transitions to the DocDownstream1.0[22] mixture (0.58M examples) and trains under longcontext settings, enabling robust reasoning over long documents while maintaining a manageable batch size via ZeRO-2 and gradient accumulation[19]. Finally, Finetune 2 introduces negative-pair supervision and multi-image document sequences, training the model to abstain on unsupported evidence and improving calibration in streaming settings. 

Across all stages, we use bf16 precision and flash-attention [14]. This staged progression allows the model to retain broad generalization from large-scale pretraining while acquiring strong long-document reasoning capabilities from high-quality downstream data. To further boost computational and memory efficiency, we incorporate Liger 

9 

Figure F2. **Multi-crop OCR decomposition. (left)** Each page is first resized and padded, then dynamically divided into an aspectratio–dependent grid of overlapping crops. **(right)** OCR tokens are spatially redistributed to their corresponding crops, enabling localized grounding and improving fine-grained multimodal alignment. 

Kernel [12], a lightweight optimization toolkit designed for large-scale model training. Liger provides highperformance fused operators and memory-aware execution strategies, such as combining sequential kernels, using in-place updates, and partitioning inputs into manageable chunks. These optimizations increase training throughput while lowering the memory footprint, enabling our multimodal model to scale more effectively under constrained GPU resources. The complete implementation details can be found in the attached codebase. 

## **S2. Edge Deployment** 

To enable fast and memory-efficient on-device inference, we convert our PyTorch-based Vision-Language Model into an optimized NPU-executable pipeline through a sequence of conversion and hardware-specific compilation steps to run on a Windows Copilot+ Laptop[1] (Fig. F3). 

**1. ONNX [2] Conversion.** The PyTorch model is first exported to the ONNX format using the standard PyTorch tracing pipeline. The exported ONNX graph preserves full model parameters, operator structure, and tensor formats required for downstream compiler optimization. This intermediate representation provides a hardware-agnostic bridge between the PyTorch runtime and the target NPU execution environment. 

**2. Weight and Activation Quantization.** We then apply post-training quantization to the full ONNX model. All model weights are quantized to **8-bit integers** using a min– max calibration scheme, while activations are quantized to **16-bit** precision. Quantization statistics, the scales and offsets of the layers are computed from a representative set of **300 document samples** . The resulting quantized model runs natively on both GPU and CPU backends in PyTorch, enabling thorough validation before hardware conversion. 

**3. NPU Compilation.** Finally, we compile the Quantized ONNX model using the Qualcomm AI Engine (QNN) compiler [3] to generate a fully NPU-executable binary. The compiler maps ONNX operators to NPU-supported kernels, performs graph-level optimizations, and produces a hardware-targeted model artifact. This step transforms the architecture into a latency-optimized, memory-efficient NPU-runnable Vision-Language model while retaining the core multimodal reasoning capabilities of the original implementation. Specifically, we use a Windows laptop equipped with a Snapdragon X Elite (X1E80100) processor featuring a 45-TOPS Hexagon-class NPU and 16 GB of unified memory. 

This deployment pipeline enables our model to run efficiently on edge devices, substantially reducing memory consumption while sustaining high throughput. It provides 

10 

Figure F3. **Local Document Understanding on Laptop.** Screenshot of our interactive on-device system for local document understanding. Users can upload PPTX files, browse slide thumbnails, and issue natural-language queries about slide content, structure, or figures. Responses are generated entirely on-device using a Windows laptop powered by a Qualcomm Snapdragon X Elite (X1E80100) with 16 GB memory. This setup demonstrates that our pipeline performs fine-grained multimodal reasoning locally on lightweight edge hardware without relying on cloud resources. Portions of the interface have been **anonymized** using solid color blocks. 

a practical path for real-world applications such as slide analysis, document assistants, and on-device multimodal agents. 

## **S3. Additional Efficiency Analysis** 

**Memory Efficiency.** Following standard memory analyses of transformer architectures [13–15], the peak VRAM during inference can be expressed using the simple approximation: 

**==> picture [223 x 31] intentionally omitted <==**

where _P_ B is the number of parameters (in billions), _b_ is the bytes per parameter, _K_ is the context length measured in units of 1k tokens, _g_ is the KV-cache cost per 1k tokens, and _O_ denotes fixed activation and workspace memory. Existing document VLMs typically emit 3k–4k visual tokens per page, which leads to a steep linear increase in the tokendependent term _Kg_ as the number of pages grows. Our model follows the same linear trend in principle; however, the crucial difference is the _slope_ of this growth. 

Figure F4. Prior methods process visual and OCR features independently, resulting in a large number of input tokens for the language model. In contrast, DocSLM fuses both modalities with a compression module, substantially reducing token count. 

DocSLM compresses OCR, visual, and layout information into a fixed **576-token** representation per page, which dramatically reduces _K_ for any given document. As a result, the contribution of the KV-cache and activation components in Eq. 12 grows much more slowly for our model, yielding a significantly lower overall memory footprint across long 

11 

documents compared to baselines whose vision encoders produce thousands of tokens or crops per page. 

**Peak GPU Memory Comparison** Table T2 reports the peak GPU memory usage of several Document understanding models as the number of document pages increases from 2 to 120. All experiments were conducted using the official implementations of each model on an NVIDIA A100–80GB GPU, using the MMLongDocBench [36] dataset. We observe that existing large and medium-scale models (InternVL2-RAG[46], Docopilot[18], DocOWL2[50]) exhibit monotonic memory growth as document length increases, with memory rising sharply between 10 and 20 pages before eventually triggering out-of-memory failures. This behavior highlights the fundamental limitation of these architectures (Fig. F4) whose token counts scale linearly with the number of pages. In contrast, our streaming 2B model maintains a strictly constant peak memory footprint of 14.2 GB across all document lengths—including the 120-page setting—due to its fixed-size per-page multimodal representation and sequential stream processing. This plateau demonstrates that our design fully decouples memory usage from document length, enabling reliable, large-scale document understanding on fixed-memory hardware such as edge GPUs, laptops, and resource-constrained servers. 

|**Model**<br>**Size**|**Peak Memory (GB) by Page Count**<br>**2**<br>**5**<br>**10**<br>**15**<br>**20**<br>**120**|
|---|---|
|InternVL2-RAG [46]<br>8B<br>Docopilot[18]<br>8B<br>InternVL2-RAG [46]<br>**2B**<br>Docopilot [18]<br>**2B**<br>DocOWL2 [24]<br>8B|22.6<br>31.7<br>47.0<br>61.9<br>76.8<br>OOM<br>21.6<br>30.5<br>45.5<br>60.4<br>75.3<br>OOM<br>10.8<br>18.2<br>30.3<br>42.9<br>55.3<br>OOM<br>9.2<br>16.2<br>27.9<br>40.3<br>52.7<br>OOM<br>17.7<br>20.0<br>24.4<br>28.6<br>34.1<br>OOM|
|**Ours**<br>**2B**|**5.2**<br>**9.2**<br>**14.2**<br>**14.2**<br>**14.2**<br>**14.2**|



Table T2. **Peak GPU memory usage (GB) under increasing document length.** Measurements were obtained on an NVIDIA A100–80GB GPU using the MMLongDocBench [36] dataset. Our streaming 2B model maintains a constant 14.2 GB memory footprint up to 120 pages. 

**Latency Vs. Accuracy** Table T3 presents a detailed comparison of inference latency and accuracy on the MMLongDoc[36] benchmark across a range of state-ofthe-art large multimodal models. Existing LVLMs, such as InternVL2-RAG[46] (2B/8B), Docopilot-2B[18], and VisRAG-12B[52] exhibit high computational overhead due to their large parameter counts and heavy visual token budgets (approximately 3K tokens per image). Even with retrieval-augmented pipelines (InternVL2+RAG), latency remains high (82–113 ms) and accuracy does not improve, 

highlighting the limitations of RAG-based pruning for longdocument reasoning. 

In contrast, our 2B model uses only 576 tokens per image through hierarchical multimodal compression, resulting in a 3–7× reduction in latency while simultaneously achieving the highest accuracy (22.7 Acc). This efficiency–accuracy trade-off demonstrates that compact models, when paired with structured compression and streaming mechanisms, can outperform much larger LVLMs both in speed and effectiveness, making our approach particularly suitable for real-time and edge-device deployment. 

|**Model**|**Size**|**Tok/Image↓**|**Latency (ms)↓**|**MMLDoc (Acc↑)**|
|---|---|---|---|---|
|InternVL2|8B|_∼_3,133|81.0|17.4|
|InternVL2+RAG|2B|_∼_3,133|82.9|17.2|
|VisRAG|12B|_>_3K|288.3|18.8|
|InternVL2|2B|_∼_3,133|35.9|10.5|
|Docopilot|2B|_∼_3,133|35.9|21.8|
|**Ours**|**2B**|**576**|**32.1**|**22.7**|



Table T3. **Latency vs. accuracy** comparison on MMLongDoc [36] (Acc). Our 2B model achieves SOTA accuracy with substantially lower latency. 

## **S4. Additional Ablation Studies** 

## **S4.1. Effect of OCR Confidence Threshold** 

To evaluate our model’s robustness to OCR noise, we apply a confidence filter to OCR tokens before fusion: 

**==> picture [191 x 11] intentionally omitted <==**

where _τ_ is the OCR confidence threshold. Table T4 reports Mp-DocVQA accuracy for thresholds ranging from 0.0 to 0.9. Performance remains extremely stable across the full range, with the best result at _τ_ = 0 _._ 0. This indicates that our hierarchical compressor effectively absorbs OCR noise, and that aggressive filtering may remove useful but lowconfidence text tokens. 

|**OCR Threshold**|0.0|0.5|0.6|0.7|0.8|0.9|
|---|---|---|---|---|---|---|
|**Mp-DocVQA**|**70.0**|69.6|69.6|69.7|69.7|69.4|



Table T4. **Ablation on OCR confidence threshold.** Performance remains consistent across all thresholds, indicating that our model is robust to OCR noise and does not rely heavily on aggressive confidence filtering. 

## **S4.2. Ablation: Effect of OCR Granularity** 

To study how OCR granularity influences model performance, we evaluate three configurations of the dynamic cropping pipeline, each corresponding directly to one entry in Table T5. Specifically: (i) fine-grained cropping (768–2304), which produces the largest number of crops 

12 

Figure F5. **Qualitative comparison of model predictions with and without OCR on a 15-page text-rich document.** With OCR (green), the model extracts the correct answers directly from the corresponding pages (highlighted). Without OCR (red), the model fails to recognize text-dense regions, instead hallucinating plausible-sounding but incorrect outputs. This illustrates that the failure arises from missing text perception rather than reasoning when processing visually complex document layouts. 

(16–100), (ii) medium cropping (384–1536), which generates a moderate number of crops (4–49), and (iii) coarse cropping (384–1152), which yields the smallest crop count (4–18). These settings differ in the density of visual patches produced by the dynamic cropping pipeline and, accordingly, the locality of OCR tokens grounded within each patch. Table T5 reports MP-DocVQA accuracy for all three configurations. The coarse configuration (384–1152) achieves the highest accuracy, while both the medium and especially the fine-grained configurations underperform despite introducing more crops and enabling more localized OCR grounding. Higher-resolution cropping grids (e.g., 768–2304) fragment each page into many small overlapping patches, forcing OCR tokens to be split across numerous local regions. 

Although this improves fine-grained text–vision alignment, it disrupts global document structure, paragraph continuity, table layout, and multi-column flow—which hinders holistic document understanding. As a result, finergrained OCR assignments do not yield performance gains and instead degrade accuracy. In contrast, the coarse configuration (384–1152) preserves global layout while still providing adequate OCR grounding for local reasoning. This balance enables the hierarchical compressor to integrate textual cues without over-fragmenting the document. 

|**Resolution**|**#Crops**|**MP-DocVQA**|
|---|---|---|
|768–2304|16–100|56.7|
|384–1536|4–49|57.9|
|384–1152|4–18|**70.0**|



Table T5. **Ablation on OCR granularity across dynamic crop configurations.** The #Crops column indicates the range of possible crops generated for each resized resolution; the exact number depends on the aspect ratio of the original document. Mid-range resolutions (384–1152) achieve the best balance between OCR locality and global structure. 

Overall, these results show that higher OCR granularity does not necessarily improve performance. Effective longdocument understanding requires a balance between local OCR grounding and global structural coherence, and the coarse 384–1152 configuration offers the most favorable trade-off. 

## **S5. Aditional Qualitative Results** 

**With vs without OCR** Fig. F5 presents a qualitative analysis of the model’s behavior on a multi-page, textheavy academic document when OCR is present versus absent. With OCR, the model consistently retrieves correct 

13 

Figure F6. **Qualitative examples of generalization to videos.** We evaluate our model on the NewsVQA [26] benchmark, which requires understanding text embedded within video frames. We show two representative cases where our model accurately identifies the temporal segment containing the answer and correctly interprets the textual cues present in the frames. These examples highlight the model’s ability to leverage multimodal signals for precise temporal localization and factually grounded answering in real video scenarios. 

information from the relevant pages, demonstrating reliable grounding across segments (Pages 3, 10, and 13). In contrast, without OCR the model is unable to parse dense textual regions and instead hallucinates answers that bear no relation to the document content (e.g., inventing course names, misreading table quantities, and guessing arbitrary deadline months). These errors highlight a fundamental limitation of vision-only processing: the model fails not due to reasoning but due to its inability to perceive fine-grained text embedded in complex layouts. This underscores the necessity of OCR for long-document understanding tasks requiring precise textual extraction. 

**Generalization to Videos** Fig. F6 presents qualitative examples illustrating our model’s ability to generalize to realworld video settings. Using the NewsVQA [26] benchmark, which demands a precise understanding of text appearing within broadcast news footage, our method successfully identifies the temporal window in which the answerrelevant information is displayed. In both cases, the model tracks the textual overlays across frames, correctly localizes the segment containing the key evidence, and produces a factually accurate answer. These results demonstrate that our approach effectively leverages fine-grained textual cues in videos, enabling robust temporal grounding and reliable question answering in dynamic, text-rich video environments. 

## **References** 

- [1] Buy 13.8-inch surface laptop, copilot+ pc with windows - microsoft store. [Online; accessed 2025-11-21]. 10 

- [2] Execution providers — onnxruntime. [Online; accessed 2025-11-20]. 10 

- [3] Qualcomm - qnn — onnxruntime. [Online; accessed 202511-20]. 10 

- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _NeurIPS_ , 35: 23716–23736, 2022. 3 

- [5] Ali Furkan Biten, Rub`en Tito, Andres Mafla, Lluis Gomez, and Dimosthenis Karatzas. Scene text visual question answering. In _ICDAR_ , 2019. 6 

- [6] Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, and R Manmatha. Latr: Layout-aware transformer for scene-text vqa. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 16548–16558, 2022. 3 

- [7] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. Gram: Global reasoning for multi-page vqa. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15598–15607, 2024. 3 

- [8] Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A Rossi, Changyou Chen, and Tong Sun. Sv-rag: Lora-contextualizing adaptation of mllms for long document understanding. _arXiv preprint arXiv:2411.01106_ , 2024. 1, 3, 6, 7 

14 

- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 1, 2, 3, 6, 7 

- [10] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ , 2024. 1, 3, 6, 7 

- [11] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. _arXiv preprint arXiv:2507.05595_ , 2025. 6 

- [12] Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, et al. Liger kernel: Efficient triton kernels for llm training. _arXiv preprint arXiv:2410.10989_ , 2024. 10 

- [13] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. _arXiv preprint arXiv:2307.08691_ , 2023. 11 

- [14] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. _NeurIPS_ , 35:16344–16359, 2022. 4, 9 

- [15] Tim Dettmers et al. Qlora: Efficient finetuning of quantized large language models. In _NeurIPS_ , 2023. 11 

- [16] Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Tao Zhong, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, and Hao Jiang. Streaming video question-answering with in-context video kv-cache retrieval. _arXiv preprint arXiv:2503.00540_ , 2025. 5 

- [17] Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, Jifeng Dai, and Wenhai Wang. Docopilot: Improving multimodal models for document-level understanding, 2025. 3 

- [18] Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, et al. Docopilot: Improving multimodal models for document-level understanding. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 4026–4037, 2025. 1, 2, 3, 6, 7, 12 

- [19] Jianwei Feng and Dong Huang. Optimal gradient checkpoint search for arbitrary computation graphs. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11433–11442, 2021. 9 

- [20] Roy Ganz, Oren Nuriel, Aviad Aberdam, Yair Kittenplon, Shai Mazor, and Ron Litman. Towards models that can see and read. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 21718–21728, 2023. 3 

- [21] Tongkun Guan, Zining Wang, Pei Fu, Zhengtao Guo, Wei Shen, Kai Zhou, Tiezhu Yue, Chen Duan, Hao Sun, Qianyi Jiang, et al. A token-level text image foundation model for document understanding. _arXiv preprint arXiv:2503.02304_ , 2025. 1, 3 

- [22] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. 

   - mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_ , 2024. 5, 9 

- [23] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocrfree multi-page document understanding. _arXiv preprint arXiv:2409.03420_ , 2024. 3 

- [24] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocrfree multi-page document understanding. _arXiv preprint arXiv:2409.03420_ , 2024. 1, 2, 3, 5, 6, 12 

- [25] Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, and Xiang Bai. Mini-monkey: Alleviate the sawtooth effect by multi-scale adaptive cropping. _arXiv preprint arXiv:2408.02034_ , 2024. 2, 6 

- [26] Soumya Jahagirdar, Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Watching the news: Towards videoqa models that can read. In _WACV_ , pages 4430–4439. IEEE, 2023. 6, 14 

- [27] Jordy Van Landeghem, Rafal Powalski, Rub`en Tito, Dawid Jurkiewicz, Matthew B. Blaschko, Lukasz Borchmann, Micka¨el Coustaty, Sien Moens, Michal Pietruszka, Bertrand Anckaert, Tomasz Stanislawek, Pawel J´oziak, and Ernest Valveny. Document understanding dataset and evaluation (DUDE). In _ICCV_ , pages 19471–19483. IEEE, 2023. 6 

- [28] Hugo Laurenc¸on, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? _CoRR_ , abs/2405.02246, 2024. 1, 3, 6, 7 

- [29] Chenxia Li, Weiwei Liu, Ruoyu Guo, Xiaoting Yin, Kaitao Jiang, Yongkun Du, Yuning Du, Lingfeng Zhu, Baohua Lai, Xiaoguang Hu, et al. Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system. _arXiv preprint arXiv:2206.03001_ , 2022. 9 

- [30] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. _CoRR_ , abs/2407.07895, 2024. 1, 3, 6, 7, 9 

- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pages 19730– 19742. PMLR, 2023. 3 

- [32] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. _arXiv preprint arXiv:2407.02392_ , 2024. 3 

- [33] Wenhui Liao, Jiapeng Wang, Hongliang Li, Chengyu Wang, Jun Huang, and Lianwen Jin. Doclayllm: An efficient multimodal extension of large language models for text-rich document understanding, 2025. 3 

- [34] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3 

- [35] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. 6 

15 

- [36] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations, 2024. 1, 2, 12 

- [37] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems_ , 37:95963–96010, 2024. 6 

- [38] Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz, Elad Ben Avraham, Alona Golts, Yair Kittenplon, Shai Mazor, and Ron Litman. Docvlm: Make your vlm an efficient reader, 2024. 1, 3, 6, 7 

- [39] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 6 

- [40] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. Vdocrag: Retrievalaugmented generation over visually-rich documents, 2025. 1, 3, 6 

- [41] Jiuqiang Tang, Raman Sorokin, Ekaterina Ignasheva, Grant Jensen, Lin Chen, Juhyun Lee, Andrei Kulik, and Matthias Grundman. Scaling on-device gpu inference for large generative models. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 6355–6364, 2025. 1 

- [42] OpenGVLab Team. Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy, 2024. 2, 3 

- [43] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multi-page docvqa. _CoRR_ , abs/2212.05935, 2022. 6, 7, 8 

- [44] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. _arXiv preprint arXiv:2502.14786_ , 2025. 6, 9 

      - haystack. _arXiv preprint arXiv:2406.07230_ , 2024. 1, 2, 3, 6, 7, 12 

   - [47] Zining Wang, Tongkun Guan, Pei Fu, Chen Duan, Qianyi Jiang, Zhentao Guo, Shan Guo, Junfeng Luo, Wei Shen, and Xiaokang Yang. Marten: Visual question answering with mask generation for multi-modal document understanding, 2025. 3 

   - [48] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. _arXiv preprint arXiv:2309.17453_ , 2023. 5 

   - [49] Han Xiao, Yina Xie, Guanxin Tan, Yinghao Chen, Rui Hu, Ke Wang, Aojun Zhou, Hao Li, Hao Shao, Xudong Lu, Peng Gao, Yafei Wen, Xiaoxin Chen, Shuai Ren, and Hongsheng Li. Adaptive markup language generation for contextuallygrounded visual document understanding, 2025. 1, 3 

   - [50] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. _arXiv preprint arXiv:2307.02499_ , 2023. 12 

   - [51] Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao. Deepsolo: Let transformer decoder with explicit points solo for text spotting. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19348–19357, 2023. 3 

   - [52] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. _arXiv preprint arXiv:2410.10594_ , 2024. 1, 3, 6, 12 

   - [53] Wenwen Yu, Zhibo Yang, Yuliang Liu, and Xiang Bai. Docthinker: Explainable multimodal large language models with rule-based reinforcement learning for document understanding, 2025. 1, 2, 3, 6 

   - [54] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. _CoRR_ , abs/2406.16852, 2024. 1, 3, 6 

   - [55] Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao, Hangdi Xing, Qi Zheng, and Ji Zhang. A simple yet effective layout token in large language models for document understanding, 2025. 1, 3, 6 

- [45] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. Docllm: A layout-aware generative language model for multimodal document understanding. _arXiv preprint arXiv:2401.00908_ , 2023. 3 

- [46] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, Xizhou Zhu, Ping Luo, Yu Qiao, Jifeng Dai, Wenqi Shao, and Wenhai Wang. Needle in a multimodal 

16 

