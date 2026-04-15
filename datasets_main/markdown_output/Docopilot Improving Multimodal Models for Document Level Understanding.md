This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

## **Docopilot: Improving Multimodal Models for Document-Level Understanding** 

Yuchen Duan[1] _[,]_[2] _[∗]_ , Zhe Chen[3] _[,]_[1] _[∗]_ , Yusong Hu[4] _[,]_[1] _[∗]_ , Weiyun Wang _[∗]_[5] _[,]_[1] , Shenglong Ye[1] , Botian Shi[1] , Lewei Lu[7] , Qibin Hou[4] , Tong Lu[3] _[,]_[1] , Hongsheng Li[2] _[,]_[1] , Jifeng Dai[6] _[,]_[1] , Wenhai Wang[2] _[,]_[1][�] 

1Shanghai AI Laboratory, 2The Chinese University of Hong Kong, 3Nanjing University, 4Nankai University, 5Fudan University, 6Tsinghua University, 7SenseTime Research 

## **Abstract** 

_Despite significant progress in multimodal large language models (MLLMs), their performance on complex, multipage document comprehension remains inadequate, largely due to the lack of high-quality, document-level datasets. While current retrieval-augmented generation (RAG) methods offer partial solutions, they suffer from issues, such as fragmented retrieval contexts, multi-stage error accumulation, and extra time costs of retrieval. In this work, we present a high-quality document-level dataset, Doc750K, designed to support in-depth understanding of multimodal documents. This dataset includes diverse document structures, extensive cross-page dependencies, and real question-answer pairs derived from the original documents. Building on the dataset, we develop a native multimodal model—Docopilot, which can accurately handle document-level dependencies without relying on RAG. Experiments demonstrate that Docopilot achieves superior coherence, accuracy, and efficiency in document understanding tasks and multi-turn interactions, setting a new baseline for document-level multimodal understanding. Data, code, and models are released at https://github.com/ OpenGVLab/Docopilot._ 

## **1. Introduction** 

In recent years, multimodal large language models (MLLMs) [6, 12, 41, 56, 60, 72, 81, 82, 84, 92] have rapidly developed, achieving remarkable performance in various visual understanding tasks [30, 77], particularly image-level tasks, such as image captioning [10, 42], optical character recognition (OCR) [45, 65], and visual question answering (VQA) [24, 44]. Despite these advances, current MLLMs still face significant challenges in document-level understanding [51, 76, 89], where models are required to identify and integrate key information across multi-page documents, setting high expectations for their long-context processing 

> * Equal contribution; 

> � Corresponding author: wangwenhai@pjlab.org.cn 

**==> picture [237 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
65.0<br>Docopilot-8B<br>60.0<br>InternVL2-26B<br>55.0<br>Docopilot-2B<br>50.0 InternVL2-8B<br>+RAG<br>45.0<br>40.0 Diameter<br>InternVL2-8B<br>35.0 2B 8B 26B<br>LLaVA-OV-8B<br>30.0 InternVL2-2B Docopilot (ours)<br>+RAG InternVL2<br>25.0 LLaVA-OV<br>InternVL2-2B InternVL2-RAG<br>20.0<br>0 5 10 15 20 25 30<br>Inference Latency (s)<br>Accuracy + 11.5%<br>< 31% Latency<br>< 31% #Param<br>Accuracy +19.9%<br>Accuracy +22.8%<br>MM-NIAH Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1. **Accuracy** _**v.s**_ **inference latency on MM-NIAH.** The proposed Docopilot-8B shows a notable improvement over baseline models [73], achieving a +19.9% accuracy gain compared to InternVL2-8B and surpassing InternVL2-26B with less than 31% of the inference latency. Additionally, Docopilot-2B uses fewer parameters (less than 10%) while exhibiting comparable performance to the 10 _×_ larger InternVL2-26B. These results suggest that our Docopilot strikes a reasonable balance between latency, model size, and performance. 

## capabilities of MLLMs. 

Current research on long-content understanding primarily focuses on text-only models [5, 9, 74], targeting specific retrieval tasks such as “Needle in a Haystack” (NIAH) [2, 31]. However, existing open-source MLLMs [12, 13, 36, 41, 62, 81, 83] are primarily trained on image-level data, lacking the long-context understanding capacity required for document-level understanding. Retrieval-augmented generation (RAG) methods [14, 22, 50, 61, 63, 95] attempt to address this by retrieving key information to fit within the limited context windows of MLLMs, but they still encounter the following challenges in document-level tasks. (1) _Fragmented Retrieval Contexts_ . Retrieved information is fragmented, lacking the overall structure of the document; (2) _Multi-Stage Error Accumulation_ . Incorrect retrieval results can affect subsequent responses, leading to errors or omissions of critical details, especially in multiturn or complex tasks; (3) _Extra Time Costs_ . The retrieval 

4026 

step increases the latency of the QA system, limiting the scalability of RAG in time-sensitive scenarios. 

To address these problems, two primary challenges need to be considered. (1) _High-Quality Multimodal Document Dataset_ . While extensive datasets [11, 91, 96, 97] exist for long-context, text-only tasks, high-quality document-level question-answering datasets remain scarce. This shortage is largely attributed to the high costs associated with annotation and the lack of streamlined construction pipelines. (2) _Native Document-Level MLLMs_ . Although RAG-based methods [14, 86, 95] provide some relief, native multimodal models with long-context processing abilities are crucial. However, training native MLLMs specifically for document-level understanding is constrained by current hardware limitations. 

In this work, we introduce a new multimodal document dataset that supports document-level understanding tasks. Compared to counterparts [33, 76, 78], this dataset has the following features: (1) _Large Scale_ . It includes a total of 758K question-answer samples, containing 5.2B text tokens and 3.1M images. It encompasses content from various sources, such as Sci-Hub, Arxiv, and OpenReview, covering a wide range of topics and document layouts. (2) _High Quality_ . Unlike existing datasets that insert irrelevant questions into documents, we collect real, in-depth question-answer pairs and construct single-page and crosspage questions based on document structure. Such highquality question-answer data accounts for 31.6% of the dataset. (3) _Multimodal_ . For the document content, we provide not only the conventional interleaved text-image context but also purely rendered image inputs, catering to the needs of different models. 

Building upon this dataset, we developed a native baseline model for document-level multimodal understanding– Docopilot. Unlike existing approaches [14, 86, 95] that rely on RAG, our model achieves efficient document-level training and testing through simple engineering optimizations, such as multimodal data packing, Ring Attention [43], and Liger Kernel [16]. Leveraging the proxy tasks carefully designed within the dataset, Docopilot can directly handle long-distance dependencies and cross-page information integration without external retrieval support. As shown in Figure 1, this approach not only enhances coherence and accuracy compared to RAG methods but also significantly reduces the response time of the entire question-answering system, delivering superior real-time performance in multiturn interactions. 

The main contributions are summarized as follows: 

(1) We develop the first large-scale, high-quality dataset for document-level multimodal understanding, consisting of 758K QA pairs from 3 sources, supporting 9 types of proxy tasks. This dataset includes 31.6% real QA pairs directly extracted from documents. 

(2) Based on the dataset, we implement Docopilot, a native MLLM designed for document-level understanding without relying on retrieval mechanisms. This approach greatly improves its ability to integrate and comprehend information across multi-page documents. 

(3) Through extensive experiments on multiple document-level benchmarks, our method demonstrates performance significantly superior to existing approaches, proving its effectiveness and generality. As shown in Figure 1, Docopilot-8B achieves a score of 61.8 on MMNIAH [86], outperforming InternVL2-8B by 19.9 points and surpassing InternVL2-26B with less than 31% of the latency. We hope this work could provide a baseline for future advancements in MLLMs for document-level tasks. 

## **2. Related Work** 

## **2.1. Multimodal Large Language Models** 

Multimodal large language models (MLLMs) have demonstrated impressive capabilities in processing image and text information, opening up new directions for applications such as visual question answering and image captioning. Early models [12, 29, 37, 59] trained with contrastive learning methods excelled in recognizing and understanding open-world semantics within an image-text matching framework. However, their limited generative abilities restricted their applicability. To leverage the powerful generation abilities of large language models (LLMs), subsequent works [13, 38, 42, 47, 83, 84] introduced a connector to align the embedding spaces of vision encoders and LLMs, allowing encoded image embeddings to serve as soft prompts for LLMs. Another series of works [1, 34, 39, 99] extended LLMs by integrating additional visual experts, reducing reliance on standalone vision encoders. More recently, models capable of both understanding and generating images have also made notable progress [20, 35, 67, 75], leveraging the insight that image generation can enhance image understanding. Despite these advancements, current MLLMs still face challenges with long-context multimodal inputs. For instance, InternVL 2.0 [13, 25] performs optimally within a token range of up to 8192, constraining its effectiveness in document-level applications. 

## **2.2. Document Understanding Models** 

Extracting key information from documents is crucial for industries and academic research. OCR-model-driven methods [3, 4, 71, 80] represent one of the primary technical approaches. These methods extract text, layout, and bounding box information from external systems and integrate it with another model. However, they are prone to error propagation and high processing times due to their reliance on multiple components. Benefitting from the rapid advancements in LLMs, OCR-free methods have also achieved 

4027 

**==> picture [468 x 147] intentionally omitted <==**

**----- Start of picture text -----**<br>
Data Source Raw Data Collection Document Content Extraction Question-Answer Pairs Construction<br>(1) Interleaved Text-Image Format (1) Documents with Reliable QA Annotations<br>Deeper neural networks are more difficult Question: Please write a review for the<br>to train. We present a residual learning provide paper: <paper><br>framework ... Answer:  <answer><br>(2) Documents with a Clear Textual Structure<br>... Driven by the significance of depth, a Question: duction section for this paper: <paper>Please write an abstract / intro-<br>question arises: Is learning better networksas easy as stacking more layers? ... Answer: <answer><br>(3) Other Documents<br>— (2) Multi-Image Format<br>: —<br>Review —— ——— ca| > © Miner G<br>= = — Question: Please read the paper: <paper>,<br>and answer the question: <question><br>Answer: <answer><br>**----- End of picture text -----**<br>


Figure 2. **Multimodal document dataset generation pipeline.** This pipeline involves three main stages: (1) Raw Data Collection: Documents are gathered from sources like Sci-Hub, arXiv, and OpenReview, available in PDF and HTML formats. (2) Document Content Extraction: Multimodal content is processed in two formats: interleaved text-image format and multi-image format. (3) Question-Answer Pairs Construction: QA pairs are generated based on the document structure or constructed using GPT-4o. 

great progress. Donut [32] is the first end-to-end training framework based on a Transformer without requiring OCR engines or APIs. Subsequent works [23, 49, 87, 88, 94] propose diverse modifications in model architectures and training algorithms. However, these models are designed for specific tasks and lack general abilities. 

## **2.3. Long-Context Large Language Models** 

With advancements in engineering, architecture, and algorithms, long-context large language models have made substantial progress. Techniques such as Flash Attention [17, 18] and Ring Attention [43] have notably reduced GPU memory usage for training on extended contexts. Additionally, various sparse attention mechanisms—including Shifted Sparse Attention [11], Dilated Attention [19], and Attention Sinks [26, 90]—have enabled efficient scaling to handle larger contexts. New positional embedding methods, like ALiBi [58], xPOS [68], and RoPE [66], further enhance the models’ generalization capabilities in length extrapolation. However, these advancements remain largely confined to natural language processing, and methods to extend the context size of MLLMs are still under-explored. Another research approach aims to reduce context size by leveraging retrieval augmented generation (RAG) [14, 22, 95], where only the most relevant passages are retrieved and fed into the generation model. However, this retrieval-based approach can disrupt the coherence of the semantic chain, particularly in complex reasoning tasks, due to fragmented information flow. In this work, we integrate the above engineering techniques into MLLMs and demonstrate that a model fine-tuned on a high-quality, long-context training corpus is a strong baseline, achieving superior performance compared to its RAG counterpart. 

## **3. Multimodal Document Dataset Generation** 

In this section, we begin by introducing the details of the data engine. Following this, we provide a comprehensive overview of the dataset—Doc-750K. 

## **3.1. Data Engine** 

The data engine operates primarily through two steps: document content extraction and question-answer (QA) pair construction. Specifically, we first extract multimodal content, including both text and images, from the documents. Based on this extracted content, we then create question-answer pairs. The document content and these constructed pairs are combined to produce conversational-style training data. The format is outlined as: 

Please read the paper: <paper>, and answer the question: <question> Answer: <answer> 

Here, the <paper>, <question>, and <answer> are the placeholder for extracted document content, the generated questions and answer, respectively. In the following, we will provide a detailed explanation of the two key steps: document content extraction and QA pair construction. **Document Content Extraction.** In practical applications, different documents have varying page layouts and content types, which poses significant challenges for content extraction. To enhance the efficiency of multimodal models, it is necessary to organize documents into a unified format for streamlined processing. In this work, we process each document into two formats as follows: 

(1) _Interleaved Text-Image Format._ Using the document content extractor MinerU [79], we segment the document content into interleaved text and image annotations, for 

4028 

**==> picture [456 x 351] intentionally omitted <==**

**----- Start of picture text -----**<br>
How Many Kilowatts are in a Negawatt? <paper><br>Verifying “Ex Post” Estimates of Utility Conservation Impacts at the Regional Level Please read the paper and answer the question: q -<br>According to this paper, how does Image-1 illustrate the conservation<br>The objective consideration of conservation policy under restructuring is proving to  impacts of various utilities in 1993? Discuss the findings and the<br>be a difficult task. One of the greatest obstacles has been the persistent uncertainty  significance of these data in the context of the research.<br>among utility planners regarding the true resource- effectiveness and cost- fF -<br>effectiveness of conservation relative<br>…<br>These figures indicate that larger utilities tend to report greater … According to the paper, ‘In 1993, the 39 companies in the sample<br>accounted for 19% of total C&I electricity sales nationwide and 40% of all<br>utility conservation program impacts, including residential impacts<br>— EST So 8 E Juel oil price 0.026 0.006 40 (Energy Information Administration 1995).’ The table in Image-1 provides<br>ID BSee 2 2:2 … fs. employment 0.20 oo 127 detailed data on the 1993 C&I sales and C&I conservation program impacts for various utilities. This image shows the sales in GWh and<br>- ES Bugs conservation 0,994 0.281 3s conservation impacts both in GWh and as a percentage of sales. For<br>ESs-- @ 2:2 SCE conervation “0261 oar 08 example, California utilities such as Southern California Edison reported<br>high sales and high conservation percentages, which is a significant outlier.<br>Early estimation attempts with individual utilities revealed that there was insufficient  These data illustrate the varying levels of conservation achievements<br>variation in the data series to produce reliable coefficient estimates for each utility  across utilities, underlining the study’s aim to verify the effectiveness and<br>taken separately. To increase the statistical power of the analysis, … accuracy of reported conservation impacts. Larger utilities tend to report<br>… greater conservation impacts relative to sales, and this detail supports the<br>Finally, since no incentives are offered, it becomes more difficult to attribute  analysis presented in the paper.<br>customer investment decisions to the utility programs as opposed to other influences.<br>ao a<br>= —rE rr SS peat Sia foe =<br>baa pomsOther OE187 0.4% OS6% re Sores uous a Analyze the importance of Image-3 in comparing the study’s findings  q p<br>San DiegoGus & Elestic Incentives we os see mew Saar at with previous empirical research on electricity price elasticities. (<br>‘oval MS 4B 100% comitaetoe9 UA nat Le 039 03 208 0m<br>…<br>Total 79% TA 100% woanasions SSWAa ars Lk kt 02 029 48 Gy<br>According to the paper, ‘The regression diagnostics offered little guidance<br>on improving the structure of the model. The addition of a dummy variable<br>for the years 1971-1973 did little to improve the residuals in subsequent<br>Towal 157 ie 100% Temty yom inal Sk 207 0083 oars amis years. Adding a set of dummy variables to account for each unstable period<br>According to California regulators, SCE’s reported impacts for service programs  did not seem reasonable.’ Image-3 presents a comparative analysis of<br>have always been viewed with skepticism during regulatory reviews, although its  electricity cross-price elasticity studies. This table showcases various<br>impacts for other types of conservation programs have not. In California, service  studies, their sample sectors, and reported elasticities for electricity, natural<br>program impacts have historically been viewed as unverifiable, so utilities have been  gas, oil, and coal across different regions and time periods. The study’s<br>largely ineligible for incentive payments related to service program expenditures.  own findings,  -0.078 for short-run own-price elasticity of electricity, align<br>Regulators in California acknowledge that they believe SCE’s reported conservation  closely with prior research, confirming its place within established<br>impacts overstate the actual savings achieved by their programs and have  literature. The context provided by other studies helps to validate the<br>communicated this skepticism to the utility. But, according to the rate regulation … paper’s conclusions regarding the robustness of its econometric model and<br>the effectiveness of conservation programs.<br>**----- End of picture text -----**<br>


Figure 3. **Visualization of an example from Doc-750K.** The left side presents the interleaved text-image format data obtained through Document Content Extraction, while the right side showcases the annotations generated via Question-Answer Pairs Construction. 

example, “<text> _\_ n<image> _\_ n<text> _\_ n<image>” This format captures the document’s textual content, making it easier to construct question-answer pairs. 

(2) _Multi-Image Format._ In this format, a document with _n_ pages is rendered as _n_ images, with each image corresponding to a single page. The structure follows the pattern “<image> _\_ n<image> _\_ n<image>”. This format preserves the original layout, enabling the model to learn the overall pagination and visual layout of the document. 

After processing the document into contexts in interleaved text-image and paginated image formats, we can not only use these contexts for next-token prediction training but also leverage the document’s content, hierarchical structure, and layout features to flexibly and precisely generate high-quality question-answer pairs. 

**Question-Answer Pairs Construction.** In this step, we create question-and-answer pairs tailored to the source, content features, and formatting structure of each document. This process is divided into the following main categories: 

(1) _For documents with reliable QA annotations,_ like the review and reply in OpenReview, we extract the QA pairs and organize them into conversation format. 

(2) _For documents with a clear textual structure_ , such as well-structured papers from Sci-Hub and Arxiv, we convert them to text and segment them, while the model is instructed to generate contents for each segment, including abstracts, experiment descriptions, and captions for figures and tables. The details of each task for structural papers are illustrated in Table 1. 

(3) _For other documents_ , in addition to using them directly for NTP pretraining, we can input text interspersed with images into MLLMs to obtain QA pairs. To ensure high-quality generated data, we use the state-of-the-art model GPT-4o [56]. 

Through our pipeline, most data has been processed into high-quality document-level question-answering data, while the remaining data is converted to plain text and used for next-token prediction tasks. Our pipelines are meticu- 

4029 

**==> picture [427 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
Source: Sci-Hub (14.4%) Multi-Page Document QA (78.8%)<br>Abstract Writing (2K) |_| Paper Titling (2K) i | Docmatix (141K) | ArxivQA (31K) | DUDE (27K)<br>: Caption Writing (2K) Experiment Writing (2K) J:oo Single-Page Document QA (10.8%)MP-DocVQA (51K) Doc-750K (750K) (ours)<br>Translation (3K) MT-QA (interleave) (49.8K) ;<br>: DocVQA (56K) DocReason (25.8K)<br>MT-QA (image) (49.8K)<br>L— InfoVQA (25.5K) ChartQA (30.2K)<br>Source: Arxiv (55.8%)<br>FP Multi-Image General QA (3.6%) MMDU (45K)<br>| MT-QA (interleave) (211K) MT-QA (image) (211K) ay<br>TeyONO — Pure-Text QA (6.8%)<br>Source: OpenReview (29.8%)<br>Mish 304 |_| LongAlign (5.9K) |_| LongCite (13.5K) LongReward (10K)<br>| Review (147.6K) Reply (77.7K) LongAlpaca (12K) | LongQLoRA (38.7K)<br>(a) Data Distribution of Doc-750K (b) Data Recipe for Supervised Fine-tuning<br>**----- End of picture text -----**<br>


Figure 4. **Data distribution of our dataset.** The outer circle shows the distribution of all data categories and the inner circle shows the distribution of data subsets. **Left:** Data distribution of Doc-750K. **Right:** Data distribution of our complete SFT training dataset. Note that the number reported in the figure represents the number of samples. “MT” is short for multi-turn. 

|Tasks|Questions|Statistics|Number|
|---|---|---|---|
|Abstract Writing|Read the full text of the paper and|Total Questions|758K|
||provide a concise summary in the|Total Images|3.1M|
||form of an abstract.|Total Conversations|251K|
|Paper Titling|Based on the provided abstract or|Multi-Turn Questions|87K|
||introduction of the research paper,|Single-Turn Questions|164K|
||please generate a concise and infor-<br>mative title|Average Text Tokens<br>Average Image Tokens|11245<br>6178|
|Caption Writing|Give the relative texts of the images|||
||or tables, please write a caption for|Table 2. **Key statistics of the Doc-750K datasets.**||
||each image or table based on the rel-|758K questions, 3.1M images, and 251K conversations, including|758K questions, 3.1M images, and 251K conversations, including|
||ative texts provided.|87K multi-turn and 164K single-turn questions. With||
|Experiment Writing|Please write the ”Experiments” sec-<br>tion based on the incomplete research|of 11,245 text tokens and 6,178 image tokens, it<br>dataset’s richness and diversity for multimodal research.||
||paper provided.|||
|Translation|Please read the full text of the follow-|ing process and improve the model’s ability||
||ing research paper and translate the<br>Experiments section into Chinese.|across different types of data inputs.<br>**Dataset Statistics.** In our Doc-750K dataset,||



Table 2. **Key statistics of the Doc-750K datasets.** It comprises 758K questions, 3.1M images, and 251K conversations, including 87K multi-turn and 164K single-turn questions. With an average of 11,245 text tokens and 6,178 image tokens, it highlights the dataset’s richness and diversity for multimodal research. 

ing process and improve the model’s ability to generalize across different types of data inputs. 

**Dataset Statistics.** In our Doc-750K dataset, the majority of the data consists of reliably annotated entries, with OpenReview and Arxiv collectively accounting for 75.4%. The remaining data, sourced from Sci-Hub, is processed using our designed tasks. The overall distribution and number of tasks are shown in Figure 4(a). Our dataset ultimately consists of 251K conversations, comprising a total of 758K questions. Additional statistical details are provided in Table 2. Compared to previous datasets, Doc-750K contains a larger number of images, with an average of four images per conversation segment. Further comparisons with other datasets are shown in Table 3. 

Table 1. **Questions format for different tasks.** For documents with a clear textual structure, we design several proxy tasks. All tasks leverage the inherent structure of the documents, with answers directly sourced from the original text. 

lously designed to ensure high data quality across all generated context. Each LLM-generated sample is explicitly marked in the metadata as model-generated. Across the entire dataset, only 4.8% of the data is LLM-generated, reinforcing the overall reliability and quality of the dataset. 

## **3.3. Data Recipe for Supervised Fine-Tuning** 

## **3.2. Multimodal Document Dataset** 

Although Doc-750K effectively covers multimodal document QA scenarios, using it directly may lead to model over-fitting on a specific document domain. Therefore, we combine it with several open-source datasets to create a mixed dataset for SFT training. As shown in Figure 4(b), these datasets are organized into 4 categories as follows: 

**Data Source.** The composition and distribution of our training data are detailed in Figure 4. Specifically, our dataset predominantly consists of academic papers, which constitute approximately 32.6% of the total data. The multimodal data, carefully selected to augment our model’s learning dimensions, makes up about 88.8% of our dataset. This strategic distribution is designed to optimize the train- 

(1) _For multi-page document QA_ , Doc-750K serves 

4030 

|Dataset|#Images|#QA Pairs|#Tokens|
|---|---|---|---|
|Docmatix [33]|2,444,750|9,500,000|390,000,000|
|DocVQA [15]|10,189|39,463|337,829|
|TextCaps [64]|21,953|21,953|389,658|
|TextVQA [65]|21,953|34,602|181,918|
|ST-VQA [8]|17,247|23,121|127,846|
|OCR-VQA [55]|165,746|801,579|6,073,824|
|VisualMRC [70]|3,027|11,988|168,828|
|DUDE[78]|147,597|23,716|11,341,228|
|Doc-750K (ours)|3,103,494|758,000|5,200,000,000|



Table 3. **Comparison with popular VQA datasets.** 

as the core dataset, specifically curated to address complex, multi-page document comprehension. Additional datasets such as MP-Docmatix [33], MP-DocVQA [53], DUDE [78], and Taesiri-ArxivQA [69] offer valuable multipage scenarios requiring inter-page reasoning and contextual retention across sequences. 

(2) _For multi-image general QA_ , MMDU-45K [48] offers a comprehensive dataset encompassing diverse realworld scenarios, such as natural environments and everyday contexts. It emphasizes multi-turn dialogues and integration of multiple images, supporting the development of systems capable of generating coherent and accurate responses from complex, lengthy inputs. 

(3) _For single-page document QA_ , We introduce DocVQA [53], DocReason [93], InfoVQA [54], and ChartQA [52] to further enhance the diversity of the SFT dataset. These datasets focus on individual pages with complex layouts, rich textual information, and, in some cases, graphical data interpretation. 

(4) _For pure-text QA_ , we add datasets including LongAlpaca [11], LongAlpaca-16K-Length [11], LongQLoRA [91], LongCite [96], LongAlign [7], and LongReward [97] to support the assessment of the model’s capabilities in QA tasks requiring long-range dependencies. 

This expanded dataset provides a balanced foundation for training and evaluating multimodal document understanding models, enhancing robustness and adaptability across diverse document-related VQA tasks. 

## **4. Enhanced Baseline for Document-Level Multimodal Understanding** 

## **4.1. Model Architecture** 

Our model architecture leverages the widely-adopted ViTMLP-LLM structure [41, 42, 73], consisting of a pre-trained Vision Transformer (ViT), a two-layer MLP projector, and a pre-trained Language Model (LLM). This combination provides a strong baseline for multimodal document analysis, effectively integrating visual and textual information within a unified framework. 

## **4.2. Optimizing Training Efficiency** 

The training efficiency of MLLMs is hindered by two key challenges: (1) _Inconsistent Sample Lengths._ Samples with different context lengths will result in excessive padding and lower training throughput; and (2) _Limited GPU Memory._ As the model scale and context length increase, GPU memory consumption becomes increasingly unsustainable. To address these issues, we have implemented the following strategies: 

(1) _Multimodal Data Packing._ To balance the computational load between the vision model (ViT) and the language model (LLM) while minimizing resource waste caused by padding, we implement a multimodal data-packing strategy. The key idea is to concatenate multiple samples into long sequences to fully utilize the model’s input capacity. Specifically, thresholds _T_ img and _T_ tok are set for the number of images and tokens, respectively. Samples are managed using a priority queue, sorted in descending order by the number of images and total tokens. A new sample _s_ attempts to combine with the sample at the front of the priority queue. If the combination meets the thresholds ( _i.e_ ., _T_ img, _T_ tok), the combined sample is pushed back into the priority queue. If _s_ cannot match with any existing sample in the queue, it is directly added to the queue. When the image number and total token number of the front sample reach one of the thresholds or the number of samples exceeds the maximum limit _M_ , the front sample is dequeued, padded as needed, and sent for training. This strategy optimizes resource utilization and ensures balanced computational workloads. The detailed pseudo-code can be found in the supplementary materials. 

(2) _Ring Attention._ We implement the Ring Attention mechanism [43] to alleviate memory constraints associated with processing long sequences. By partitioning sequences into blocks and distributing computation across multiple devices, Ring Attention allows the model to accommodate larger contexts. This approach enables overlapping communication between key-value blocks and attention computations, thereby enhancing parallel processing efficiency. Consequently, Ring Attention improves the model’s capacity to handle extended context lengths without exceeding memory limits. 

(3) _Liger Kernel._ To further improve memory and computational efficiency, we integrate the Liger Kernel [16], a specialized kernel library optimized for large-scale model training. The Liger Kernel enhances throughput and reduces memory consumption by employing techniques like kernel fusion, in-place operations, and input chunking. Leveraging the Liger Kernel thus enables higher training throughput and addresses memory limitations, allowing for efficient scaling of large multimodal models. 

4031 

|Models|MP-Doc|MMLong-Doc|DocGenome|MM-NIAH|
|---|---|---|---|---|
||ANSL_↑_|Acc_↑_<br>F1_↑_|Class Acc_↑_<br>Title ED_↓_<br>Abstract ED_↓_<br>SP Acc_↑_<br>MP Acc_↑_|Short<br>Medium<br>Long<br>Overall|
|_Proprietary Models_<br>Gemini-1.5-Pro [60]<br>GPT-4o [56]|–<br>–|28.2<br>20.6<br>42.8<br>44.9|–<br>–<br>–<br>–<br>–<br>97.6<br>9.5<br>6.5<br>71.8<br>67.6|73.8<br>65.2<br>60.8<br>67.1<br>–<br>–<br>–<br>–|
|_Open-Source Models_<br>MiniMonkey-2B [28]<br>InternVL2-2B [13]<br>InternVL2-2B + RAG [85]<br>Llama3.2-3B-Instruct_†_ [21]|70.3<br>71.8<br>72.6<br>–|10.3<br>8.6<br>10.5<br>10.8<br>17.2<br>16.7<br>23.7<br>21.2|57.4<br>16.5<br>55.0<br>40.3<br>28.9<br>60.8<br>18.4<br>54.3<br>39.4<br>28.9<br>60.8<br>18.4<br>54.3<br>39.4<br>28.4<br>85.3<br>194.7<br>51.0<br>40.2<br>34.9|40.9<br>26.9<br>23.5<br>31.0<br>36.6<br>21.2<br>19.4<br>26.4<br>36.8<br>30.2<br>34.8<br>33.8<br>15.5<br>2.2<br>0.5<br>6.6|
|Docopilot-2B (ours)|76.2|21.8<br>16.0|56.2<br>4.5<br>43.6<br>45.1<br>37.4|58.0<br>46.7<br>40.9<br>49.2|
|MiniCPM-V2.6-8B [92]<br>LLaVA-OneVision-8B [36]<br>mPLUG-DocOwl2-8B [27]<br>M3DocRAG [14]<br>VisRAG-8B [95]<br>InternLM2.5-7B-1M_†_ [9]<br>InternVL2-8B [13]<br>InternVL2-8B + RAG [85]<br>InternVL2-26B [13]|–<br>–<br>69.4<br>84.4<br>–<br>–<br>79.3<br>78.7<br>–|16.9<br>15.4<br>10.8<br>9.6<br>13.4<br>8.9<br>21.0<br>22.6<br>18.8<br>18.3<br>28.7<br>25.6<br>17.4<br>16.5<br>24.2<br>24.5<br>15.5<br>15.4|92.8<br>10.2<br>32.6<br>60.0<br>54.2<br>85.6<br>49.9<br>77.5<br>9.8<br>7.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>92.8<br>10.2<br>32.6<br>60.0<br>50.7<br>92.7<br>77.6<br>59.3<br>42.7<br>42.5<br>90.6<br>8.2<br>39.6<br>56.0<br>46.1<br>90.6<br>8.2<br>39.6<br>56.0<br>46.0<br>87.5<br>16.9<br>23.3<br>49.7<br>42.7|49.0<br>15.3<br>0.0<br>23.4<br>65.7<br>38.0<br>0.0<br>37.7<br>17.9<br>0.1<br>0.0<br>6.6<br>–<br>–<br>–<br>–<br>47.1<br>29.2<br>29.5<br>35.8<br>40.5<br>37.2<br>35.1<br>37.8<br>56.4<br>37.3<br>32.4<br>42.9<br>55.7<br>43.4<br>45.2<br>48.4<br>65.0<br>48.7<br>41.9<br>52.8|
|Docopilot-8B (ours)|81.3|28.8<br>23.0|93.8<br>2.0<br>19.7<br>53.9<br>51.9|71.2<br>57.4<br>55.3<br>61.8|



Table 4. **Evaluation on multi-page and interleaved VQA benchmarks.** We report the metrics on MP-DocVQA [76] (MP-Doc), MMLongbench-Doc [51] (MMLong-Doc), DocGenome [89], and MM-NIAH [85]. Our model outperforms document-level MLLMs and multimodal RAG methods on multi-page, medium, and long-context QA. The “Short”, “Medium”, and “Long” in MM-NIAH refer to “ input length in [0 _,_ 8k], (8k _,_ 32k], (32k _,_ 64k], respectively. _†_ ” denotes input documents are parsed by OCR models. 

## **5. Experiments** 

## **5.1. Experimental Setup** 

**Training Details.** Our model is available in two sizes: Docopilot-2B and Docopilot-8B, both of which are based on the InternVL2 [73] and fine-tuned for one epoch using the data recipe that includes Doc-750K. The training uses a batch size of 128, the AdamW optimizer with a learning rate of 1e-5, weight decay of 0.01 for the 2B variant, and 0.05 for the 8B variant, along with a cosine learning rate schedule. To speed up training, we apply multimodal data packing to reduce padding and a dynamic high-resolution strategy [13] to enhance OCR for document understanding. The maximum number of tiles for multimodal data is limited to 24, and the maximum sequence length is set to 32k tokens. 

**Baselines.** We compare our Docopilot with a series of opensource document-level MLLMs [27, 28, 36, 40, 46, 92, 98] that supports multi-image input and proprietary MLLMs, including Gemini-1.5-pro [60], GPT-4o [57]. For comparison with the commonly used RAG method for handling long documents, we selected the latest multimodal RAG methods VisRAG [95], InternVL + RAG [85], and M3DocRAG [14]. To compare with more long-context large language models, we use InternVL2-8B as the OCR model to extract texts from the documents and images and feed the parsed documents to long-context LLMs [9, 21]. 

## **5.2. Multi-Page VQA** 

**Benchmarks.** For the multi-page VQA task, we evaluate our model on three benchmarks: (1) **MP-DocVQA** [76], 

|Models|DocVQA|ChartQA|InfoVQA|
|---|---|---|---|
|Gemini-1.5-Pro [60]|93.1|87.2|81.0|
|GPT-4o [56]|92.8|85.7|–|
|MiniMonkey-2B [28]|87.4|76.5|60.1|
|InternVL2-2B [13]|86.9|76.2|58.9|
|Docopilot-2B (ours)|87.3|76.4|58.5|
|Monkey-8B [40]|66.5|65.1|36.1|
|TextMonkey-9B [46]|73.0|66.9|28.6|
|mPLUG-DocOwl2-8B [27]|80.7|70.0|46.4|
|IXC2.5-7B [98]|90.9|82.2|69.9|
|InternVL2-8B [13]|91.6|83.3|74.8|
|Docopilot-8B (ours)|92.0|83.3|73.3|



Table 5. **Results on single-page VQA benchmarks.** Our Docopilot models perform comparably to baselines [73], demonstrating enhanced long-context modeling without loss on shorter tasks. 

which is designed to evaluate the ability to handle complex questions across multiple scanned document pages. (2) **MMLongbench-Doc** [51], a benchmark for evaluating the performance of MLLMs on multi-modal documents. (3) **DocGenome** [89], a large-scale benchmark for the evaluation of scientific document comprehension. 

**Results.** As illustrated in Table 4, our model achieves consistent improvements on multi-page QA benchmarks, outperforming previous document-level MLLMs. Notably, our Docopilot-8B surpasses Gemini-1.5-Pro [60] on MMLongBench-Doc, positioning it as the closest opensource model to GPT-4o. In comparison to RAG-based methods [14, 86, 95], our model demonstrates advantages in multi-page scenarios. For example, in the Multi-Page QA of DocGenome benchmark, the RAG method shows a perfor- 

4032 

mance decline due to the disruption of document continuity while our Docopilot exhibits a significantly stable improvement compared to the baseline, with Docopilot-8B showing an increase of 12.6% over InternVL2-8B. 

## **5.3. Interleaved Long-Context QA** 

**Benchmarks.** For the interleaved long-context QA task, we evaluate our models on MM-NIAH [86], a benchmark designed for long multimodal document comprehension. **Results.** The right side of Table 4 presents the results of MM-NIAH across context lengths ranging from 1K to 64K. We categorize the context lengths into ”Short,” ”Medium,” and ”Long” based on the context window of InternVL2 (8K) and Docopilot (32K). Our Docopilot demonstrates exceptional performance in both medium- and longcontext scenarios, while maintaining high accuracy in shortcontext situations. Notably, for QA tasks with context lengths in the range of (32 _K,_ 64 _K_ ], Docopilot-2B outperforms InternVL2-2B by 110%, and Docopilot-8B surpasses InternVL2-8B by 70%. Furthermore, our model performs comparably to the state-of-the-art multimodal long-context model Gemini-1.5-Pro in contexts longer than 8K, establishing a new state-of-the-art performance among opensource long-context MLLMs. 

## **5.4. Single-Page VQA** 

**Benchmarks.** For single-page VQA tasks, we evaluate our model on three benchmarks: (1) DocVQA [53], a benchmark for the evaluation of extracting key information from an image of the given document. (2) ChartQA [52], a benchmark for evaluating the reasoning abilities for chart images. (3) InfoVQA [54], a benchmark for infographic image comprehension. 

**Results.** As shown in Table 5, our model achieves comparable performance to baseline models. Across the three benchmarks, Docopilot-2B and InternVL2-2B exhibit comparable results, while Docopilot-8B outperforms InternVL2-8B by 0.4 points in DocVQA. These results demonstrate that Doc-750K effectively enhances the model’s long-context modeling capabilities without compromising its performance on shorter documents. 

## **5.5. Ablation Study** 

**Effect of Doc-750K.** We conducted ablation studies on MMLongBench-Doc [51] to analyze the impact of our Doc750K. We divided Doc-750K into 3 parts according to the source of the data: (1) Sci-Hub data; (2) Arxiv data; and (3) OpenReview data. We demonstrate the effects of incorporating each part of the data into the SFT process, reported in Table 6. We observed that with the inclusion of different parts of Doc-750K, the model’s performance improves continuously. Utilizing only open-source data results in an inferior F1 score. 

|Models|Acc|F1|
|---|---|---|
|Baseline (InternVL2-2B [73])|10.5|10.8|
|– Variant1: SFT using data recipe w/o Doc-750K|18.4|9.4|
|– Variant2: Variant1 + our Sci-Hub data|18.5|15.2|
|– Variant3: Variant2 + our Arxiv data|20.5|15.5|
|Docopilot-2B: Variant3 + our OpenReview data|21.8|16.0|



Table 6. **Ablation study on the data recipe.** We evaluate the effectiveness of the training data from different sources on MMLongBench-Doc. Our Doc-750K can consistently enhance the ability of the model to understand multi-page documents. 

|Models|Latency|Acc|F1|
|---|---|---|---|
|MiniCPM-V2.6 [92]|225.4ms|16.9|15.4|
|VisRAG-12B [95]|288.3ms|18.8|18.3|
|InternVL2-2B [13]|35.9ms|10.5|10.8|
|InternVL2-2B + RAG [85]|82.9ms|17.2|16.7|
|Docopilot-2B (ours)|35.9ms|21.8|16.0|
|InternVL2-8B [13]|81.0ms|17.4|16.5|
|InternVL2-8B + RAG [85]|113.4ms|24.2|24.5|
|Docopilot-8B (ours)|81.0ms|28.8|23.0|



Table 7. **Latency analysis.** We evaluate the average token output latency of model outputs on MMLongBench-Doc [51]. RAGbased methods [85, 95] exhibit slower processing speeds due to their two-stage inference process, making them less efficient than document MLLMs for handling multimodal long documents. 

**Latency Analysis.** To compare the latency in inference between RAG methods and our Docopilot, we conducted a latency analysis on MMLongBench-Doc [51], as reported in Table 7. While RAG reduces the document length input to the MLLM, its own time cost remains non-negligible. For instance, InternVL2-2B + RAG is 130% slower than InternVL2-2B, and VisRAG is 28% slower than MiniCPMV2.6. Our Docopilot does not require additional processes and therefore has the same inference time as baseline models, making it more suitable for analyzing long documents. 

## **6. Conclusions** 

This work introduced a diverse document-level questionanswering dataset that covers complex structures and crosspage dependencies, providing a robust foundation for training and evaluating document understanding models. We also proposed a retrieval-free long-document understanding model that effectively integrates multi-page information, reducing reliance on external retrieval systems. Experimental results show that our model achieves state-of-the-art performance across several document-level QA benchmarks, underscoring its strength in multi-page integration and complex reasoning. Future work will focus on improving computational efficiency, extending the model to larger multimodal tasks, and adapting it to broader applications for enhanced practicality and generalization. 

4033 

## **Acknowledgments** 

This project was supported by the National Key R&D Program of China (No. 2022ZD0161300, 2022ZD0160101), the National Natural Science Foundation of China (No. 62376134, 62372223). Zhe Chen is supported by the Youth PhD Student Research Project under the National Natural Science Foundation (No. 623B2050). 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _NeurIPS_ , 35: 23716–23736, 2022. 2 

- [2] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L- eval: Instituting standardized evaluation for long context language models. _arXiv preprint arXiv:2307.11088_ , 2023. 1 

- [3] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R Manmatha. Docformerv2: Local features for document understanding. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 709–718, 2024. 2 

- [4] Haoli Bai, Zhiguang Liu, Xiaojun Meng, Wentao Li, Shuang Liu, Nian Xie, Rongfu Zheng, Liangwei Wang, Lu Hou, Jiansheng Wei, et al. Wukong-reader: Multi-modal pretraining for fine-grained visual document understanding. _arXiv preprint arXiv:2212.09621_ , 2022. 2 

- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023. 1 

- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 1 

- [7] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. Longalign: A recipe for long context alignment of large language models. _arXiv preprint arXiv:2401.18058_ , 2024. 6 

- [8] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In _ICCV_ , pages 4291–4301, 2019. 6 

- [9] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. _arXiv preprint arXiv:2403.17297_ , 2024. 1, 7 

- [10] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 1 

- [11] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient finetuning of long-context large language models. _arXiv preprint arXiv:2309.12307_ , 2023. 2, 3, 6 

- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 1, 2 

- [13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 1, 2, 7, 8 

- [14] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ , 2024. 1, 2, 3, 7 

- [15] Christopher Clark and Matt Gardner. Simple and effective multi-paragraph reading comprehension. In _ACL_ , pages 845–855, 2018. 6 

- [16] Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, et al. Liger kernel: Efficient triton kernels for llm training. _arXiv preprint arXiv:2410.10989_ , 2024. 2, 6 

- [17] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. _arXiv preprint arXiv:2307.08691_ , 2023. 3 

- [18] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. _NeurIPS_ , 35:16344–16359, 2022. 3 

- [19] Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. _arXiv preprint arXiv:2307.02486_ , 2023. 3 

- [20] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. In _ICLR_ , 2024. 2 

- [21] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 7 

- [22] Manuel Faysse, Hugues Sibille, Tony Wu, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ , 2024. 1, 3 

- [23] Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile 

4034 

document understanding. _arXiv preprint arXiv:2311.11810_ , 2023. 3 

- [24] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. _arXiv preprint arXiv:2306.13394_ , 2023. 1 

- [25] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, et al. Mini-internvl: A flexible-transfer pocket multimodal model with 5% parameters and 90% performance. _arXiv preprint arXiv:2410.16261_ , 2024. 2 

- [26] Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Simple on-the-fly length generalization for large language models. _arXiv preprint arXiv:2308.16137_ , 2023. 3 

- [27] Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocrfree multi-page document understanding. _arXiv preprint arXiv:2409.03420_ , 2024. 7, 2 

- [28] Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, and Xiang Bai. Mini-monkey: Alleviate the sawtooth effect by multi-scale adaptive cropping. _arXiv preprint arXiv:2408.02034_ , 2024. 7 

- [29] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. Zenodo. Version 0.1. https://doi.org/10. 5281/zenodo.5143773, 2021. DOI: 10.5281/zenodo.5143773. 2 

- [30] Yao Jiang, Xinyu Yan, Ge-Peng Ji, Keren Fu, Meijun Sun, Huan Xiong, Deng-Ping Fan, and Fahad Shahbaz Khan. Effectiveness assessment of recent large vision-language models. _Visual Intelligence_ , 2(1):17, 2024. 1 

- [31] Greg Kamradt. Llmtest ~~n~~ eedleinahaystack. https : / / github . com / gkamradt / LLMTest _ NeedleInAHaystack, 2024. Accessed: 2024-11-11. 1 

- [32] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision_ , pages 498–517. Springer, 2022. 3 

- [33] Hugo Laurenc¸on, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions. _arXiv preprint arXiv:2408.12637_ , 2024. 2, 6 

- [34] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. _NIPS_ , 36, 2024. 2 

- [35] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. _arXiv preprint arXiv:2307.16125_ , 2023. 2 

- [36] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , 2024. 1, 7 

- [37] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , pages 12888–12900, 2022. 2 

- [38] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ , pages 19730–19742. PMLR, 2023. 2 

- [39] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: An unified multimodal corpus of 10 billion-level images interleaved with text. _arXiv preprint arXiv:2406.08418_ , 2024. 2 

- [40] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint arXiv:2311.06607_ , 2023. 7 

- [41] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 1, 6 

- [42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _NeurIPS_ , 36, 2023. 1, 2, 6 

- [43] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. _arXiv preprint arXiv:2310.01889_ , 2023. 2, 3, 6 

- [44] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 1 

- [45] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. _arXiv preprint arXiv:2305.07895_ , 2023. 1 

- [46] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. _arXiv preprint arXiv:2403.04473_ , 2024. 7 

- [47] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. _arXiv preprint arXiv:2305.05662_ , 2023. 2 

- [48] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. _arXiv preprint arXiv:2406.11833_ , 2024. 6 

- [49] Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, Zhi Yu, and Cong Yao. Layoutllm: Layout instruction tuning with large language models for document understanding. In _Pro-_ 

4035 

_ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15630–15640, 2024. 3 

- [50] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying multimodal retrieval via document screenshot embedding. _arXiv preprint arXiv:2406.11251_ , 2024. 1 

- [51] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations, 2024. 1, 7, 8, 5 

- [52] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In _ACL_ , pages 2263–2279, 2022. 6, 8, 1, 5 

- [53] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _WACV_ , pages 2200–2209, 2021. 6, 8, 1, 5 

- [54] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In _WACV_ , pages 1697–1706, 2022. 6, 8, 1, 5 

- [55] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _ICDAR_ , pages 947–952, 2019. 6 

- [56] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf, 2023. 1, 4, 7 

- [57] OpenAI. Gpt-4o system card. https://openai.com/ index/gpt-4o-system-card/, 2024. 7 

- [58] Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. _arXiv preprint arXiv:2108.12409_ , 2021. 3 

- [59] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. _NeurIPS_ , 30, 2017. 2 

- [60] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 1, 7 

- [61] Sahel Sharifymoghaddam, Shivani Upadhyay, Wenhu Chen, and Jimmy Lin. Unirag: Universal retrieval augmentation for multi-modal large language models. _arXiv preprint arXiv:2405.10311_ , 2024. 1 

- [62] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. _arXiv preprint arXiv:2408.15998_ , 2024. 1 

- [63] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. _arXiv preprint arXiv:2301.12652_ , 2023. 1 

- [64] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In _ECCV_ , pages 742–758, 2020. 6 

- [65] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _CVPR_ , pages 8317–8326, 2019. 1, 6 

- [66] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_ , 568:127063, 2024. 3 

- [67] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. In _ICLR_ , 2024. 2 

- [68] Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable transformer. _arXiv preprint arXiv:2212.10554_ , 2022. 3 

- [69] Mohammad Reza Taesiri. Arxivqa. https://github. com/taesiri/ArXivQA, 2024. 6 

- [70] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 13878–13888, 2021. 6 

- [71] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 19254–19264, 2023. 2 

- [72] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 1 

- [73] OpenGVLab Team. Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy, 2024. 1, 6, 7, 8 

- [74] Qwen Team. Qwen2.5: A party of foundation models, 2024. 1 

- [75] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. _arXiv preprint arXiv:2401.10208_ , 2024. 2 

- [76] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834, 2023. 1, 2, 7, 5 

- [77] Xiaoguang Tu, Zhi He, Yi Huang, Zhi-Hao Zhang, Ming Yang, and Jian Zhao. An overview of large ai models and their applications. _Visual Intelligence_ , 2(1):1–22, 2024. 1 

- [78] Jordy Van Landeghem, Ruben Tito, Łukasz Borchmann, Michał Pietruszka, Paweł Joziak, Rafał Powalski, Dawid Jurkiewicz, Mickael Coustaty, Bertrand Ackaert, Ernest Val- 

4036 

veny, et al. Document understanding dataset and evaluation (dude). In _Proceedings IEEE/CVF international conference on computer vision-ICCV 2023_ , pages 19528–19540. IEEE/CVF, 2023. 2, 6 

- [79] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. _arXiv preprint arXiv:2409.18839_ , 2024. 3 

- [80] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. Docllm: A layout-aware generative language model for multimodal document understanding. _arXiv preprint arXiv:2401.00908_ , 2023. 2 

- [81] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 1 

- [82] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023. 1 

- [83] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. _arXiv preprint arXiv:2402.19474_ , 2024. 1, 2 

- [84] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In _ICLR_ , 2024. 1, 2 

- [85] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, Xizhou Zhu, Ping Luo, Yu Qiao, Jifeng Dai, Wenqi Shao, and Wenhai Wang. Needle in a multimodal haystack. _arXiv preprint arXiv:2406.07230_ , 2024. 7, 8, 3 

- [86] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. Needle in a multimodal haystack. _arXiv preprint arXiv:2406.07230_ , 2024. 2, 7, 8, 5 

- [87] Yonghui Wang, Wengang Zhou, Hao Feng, Keyi Zhou, and Houqiang Li. Towards improving document understanding: An exploration on text-grounding via mllms. _arXiv preprint arXiv:2311.13194_ , 2023. 3 

   - [90] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. _arXiv preprint arXiv:2309.17453_ , 2023. 3 

   - [91] Jianxin Yang. Longqlora: Efficient and effective method to extend context length of large language models. _arXiv preprint arXiv:2311.04879_ , 2023. 2, 6 

   - [92] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. _arXiv preprint arXiv:2408.01800_ , 2024. 1, 7, 8 

   - [93] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. _arXiv preprint arXiv:2307.02499_ , 2023. 6 

   - [94] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. _arXiv preprint arXiv:2310.05126_ , 2023. 3 

   - [95] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. _arXiv preprint arXiv:2410.10594_ , 2024. 1, 2, 3, 7, 8 

   - [96] Jiajie Zhang, Yushi Bai, Xin Lv, Wanjun Gu, Danqing Liu, Minhao Zou, Shulin Cao, Lei Hou, Yuxiao Dong, Ling Feng, et al. Longcite: Enabling llms to generate fine-grained citations in long-context qa. _arXiv e-prints_ , pages arXiv–2409, 2024. 2, 6 

   - [97] Jiajie Zhang, Zhongni Hou, Xin Lv, Shulin Cao, Zhenyu Hou, Yilin Niu, Lei Hou, Yuxiao Dong, Ling Feng, and Juanzi Li. Longreward: Improving long-context large language models with ai feedback. _arXiv preprint arXiv:2410.21252_ , 2024. 2, 6 

   - [98] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. _arXiv preprint arXiv:2407.03320_ , 2024. 7 

   - [99] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. _NIPS_ , 36, 2024. 2 

- [88] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. _arXiv preprint arXiv:2312.06109_ , 2023. 3 

- [89] Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open largescale scientific document benchmark for training and testing multi-modal large language models. _arXiv preprint arXiv:2406.11633_ , 2024. 1, 7, 2, 5 

4037 

