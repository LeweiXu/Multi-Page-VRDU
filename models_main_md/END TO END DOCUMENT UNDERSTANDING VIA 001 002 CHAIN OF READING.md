Under review as a conference paper at ICLR 2026 

## **000** - END TO-END DOCUMENT UNDERSTANDING VIA **001 002** CHAIN-OF-READING 

## **003** 

**004 005 006 007** 

## **Anonymous authors** 

Paper under double-blind review 

**008 009 010** 

## ABSTRACT 

**011** 

Intelligent Document Analysis (IDA) is a formidable task owing to documents’ complex layouts, dense tables, charts, and mixed modalities. Conventional pipelines apply OCR before large language model reasoning but suffer from error propagation. End-to-end multimodal models avoid explicit pipelines yet struggle to scale to multi-page documents, where information dilution and evidence localization remain major bottlenecks. We propose Chain-of-Reading (CoR), an endto-end framework that transforms traditional text-centric reading into a native multimodal paradigm. CoR directly consumes PDF pages as visual input, mimicking human eyes, and performs document-level question answering through a chain-ofthought process. It first localizes relevant evidence, then selectively applies OCR, and finally performs reasoning over the localized content. To further enhance comprehension of visual elements such as charts and scientific figures—which exacerbate information dilution and impede pinpointing evidence—we introduce Masked Auto-Regression (Mask-AR), a self-supervised method for multimodal grounding. CoR achieves a 14.3% improvement over the base model on the MMLongBench-Doc benchmark. We will release the CoR-Dataset and our finetuned model, Qwen2.5-VL-CoR. 

**012** 

**013** 

**014** 

**015 016 017** 

**018 019** human eyes, and performs document-level question answering through a chain-ofthought process. It first localizes relevant evidence, then selectively applies OCR, **020** and finally performs reasoning over the localized content. To further enhance **021** comprehension of visual elements such as charts and scientific figures—which **022** exacerbate information dilution and impede pinpointing evidence—we introduce **023** Masked Auto-Regression (Mask-AR), a self-supervised method for multimodal **024** grounding. CoR achieves a 14.3% improvement over the base model on the **025** MMLongBench-Doc benchmark. We will release the CoR-Dataset and our fine- **026** tuned model, Qwen2.5-VL-CoR. **027 028 029** 1 INTRODUCTION **030 031** The proliferation of Large Language Models (LLMs) has precipitated a paradigm shift in Intelli- **032** gent Document Analysis (IDA). Nonetheless, a formidable challenge persists: enabling these mod- **033** els to achieve deep semantic comprehension of complex, visually-rich documents, such as PDFs. **034** These documents, curated for human readership, fuse text, charts, and intricate layouts into a semi- **035** structured format that poses a substantial barrier to information extraction and query reasoning. The **036** key problems in this field, therefore, converge on the imperative to develop models that can accu- **037** rately and efficiently reason over information embedded within these complex visual layouts. **038** Two dominant paradigms address this challenge. The first relies on a pipeline-based approach, **039** executing tasks sequentially, such as layout analysis, OCR, and specialized recognition for tables or **040** formulas (Livathinos et al., 2025; Cui et al., 2025). Although modular, this approach suffers from **041** high complexity and maintenance overhead. More importantly, it is highly susceptible to cascading **042** errors: a single inaccuracy from an upstream module, like OCR, can propagate through the pipeline **043** and compromise the integrity of the final output. 

> **044** The second paradigm focuses on end-to-end solutions that bypass traditional OCR, including OCR- 

> **045** free Multimodal LLMs (MLLMs) (Ye et al., 2023; Wei et al., 2024) and multi-modal Retrieval- **046** Augmented Generation (RAG) systems (Faysse et al., 2024). RAG first retrieves relevant document **047** patches and then feeds them to a model for generation; however, decoupling retrieval from reasoning **048** often makes the retriever a critical bottleneck. A more promising direction involves MLLMs that **049** learn to read, localize, and reason directly from raw document pixels, integrating comprehension **050** and reasoning within a single end-to-end framework. 

> **051** Despite their potential, existing MLLMs exhibit substantial performance limitations when process- **052** ing long multi-modal documents. Their effectiveness diminishes in multi-page scenarios due to two **053** main challenges: **key information dilution** and **evidence localization difficulty** (Ma et al., 2024; Deng et al., 2024). As input sequences grow, models struggle to identify relevant passages, and they 

1 

Under review as a conference paper at ICLR 2026 

**054** 1 **055** Common Pipeline for Document QA '1 Chain of Reading (CoR) porary **056** ' Q: What's the swoon iy Q: What's the rey CoR: Okay, | need to ... navigate to ... | totalamount { — Layout Modal-specific ‘ LM 1 total amount {Plan} table<box>(0.15, 0.30, 0.70, 0.45)</box> on **057** | of X? Hi Analysisi Ingestion H} Generation 1 of X?? H: HH page 25 and locate ...ini tite"2021"Aah column. : H T H od ; Locate; I've found that value to be "3000". Then, | **058** ' H H ' ' H : need to ... turn to page 28 and locate the **059** \1 T i a |H Asmount The totalof X i 'a)H T ? H Extract :: > table<box>(....)</box>down ... to the row for(er paaneallscan"Raw Materials" and ' we) : ==» xox > S00, fa aa i} Synth look across to the column for "2021". The X **060** ! i H . it Lo is "2002". Finally, | will sum these two ' i im — | rt _ values, X = 3000 **061** 1 Nee ee cece eeeeeneeneeneeneeet” it $y CoR A: 5002 + 2002 = 5002. **062** oot1 ®& High Complexity @& OCR Error & Modal Information Loss ! End-to-end Design Interpretable CoT Information Preservation **063 064** Figure 1: Comparison of pipeline-based methods and our Chain-of-Reading (CoR) framework for **065** document understanding **066 067 068** often miss critical visual cues embedded in tables or charts. These shortcomings frequently result in **069** reasoning errors or factual hallucinations, which significantly constrain their practical utility. **070** To address these issues, we introduce the **”Chain-of-Reading” (CoR)** , a training paradigm inspired **071** by human cognitive strategies for document analysis (Figure 1). CoR guides the model to first con- **072** struct an explicit information-gathering path before performing complex reasoning. Under CoR, **073** the model learns to first _locate_ evidence—pinpointing relevant texts, charts, or pages—and then **074** performs _integrated reasoning_ upon this grounded foundation. This process mirrors the human **075** cognitive pattern of scanning for key information before conducting an in-depth analysis. Further- **076** more, given that chart comprehension presents a distinct and formidable challenge, we also designed **077 Masked Auto-Regression (Mask-AR)** , an efficient self-supervised method aimed at bolstering the model’s fine-grained comprehension of such complex visual elements. **078 079** Our main contributions are as follows: **080** • We propose **Chain-of-Reading (CoR)** , a novel training paradigm that effectively addresses evi- **081** dence localization in long PDF documents and reduces hallucination. **082** • We introduce **Masked Auto-Regression (Mask-AR)** , a self-supervised method that substantially **083** enhances fine-grained, multimodal comprehension of complex charts. **084** • We construct and release the **CoR-dataset** , the first dataset specifically designed for CoR training, **085** curated through a low-cost, high-quality data generation pipeline. **086 087** • We develop and open-source **Qwen2.5-VL-CoR** , an end-to-end document understanding model. Experiments on long-document benchmarks demonstrate that our model achieves significant **088** improvements, surpasses existing open-source methods—including agentic approaches—and **089** reaches performance comparable to leading proprietary MLLMs. **090 091 092** 2 RELATED WORK 

often miss critical visual cues embedded in tables or charts. These shortcomings frequently result in reasoning errors or factual hallucinations, which significantly constrain their practical utility. 

To address these issues, we introduce the **”Chain-of-Reading” (CoR)** , a training paradigm inspired by human cognitive strategies for document analysis (Figure 1). CoR guides the model to first construct an explicit information-gathering path before performing complex reasoning. Under CoR, the model learns to first _locate_ evidence—pinpointing relevant texts, charts, or pages—and then performs _integrated reasoning_ upon this grounded foundation. This process mirrors the human cognitive pattern of scanning for key information before conducting an in-depth analysis. Furthermore, given that chart comprehension presents a distinct and formidable challenge, we also designed **Masked Auto-Regression (Mask-AR)** , an efficient self-supervised method aimed at bolstering the model’s fine-grained comprehension of such complex visual elements. 

- We introduce **Masked Auto-Regression (Mask-AR)** , a self-supervised method that substantially enhances fine-grained, multimodal comprehension of complex charts. 

**093** 

## **094** 2.1 INTELLIGENT DOCUMENT ANALYSIS 

**095 096** Intelligent Document Analysis (IDA) is a foundational discipline for extracting and reasoning over **097** complex documents prevalent in fields like finance, law, and science. The contemporary landscape **098** of IDA is largely defined by a dichotomy between pipeline-based and end-to-end methodologies. **099** Pipeline-based methods orchestrate a sequence of specialized modules. These systems typically **100** commence with OCR engines or PDF parsers to extract raw text and layout information, which is **101** then fed into a downstream LLM for semantic processing (Xie et al., 2024; Wang et al., 2024a). This **102** modular architecture permits the integration of powerful, task-specific models for layout analysis, **103** table recognition, and formula parsing (Huang et al., 2022; Blecher et al., 2023), as exemplified by **104** systems like DocLayLLM and DocFormer (Liao et al., 2025; Appalaraju et al., 2021). However, this approach harbors a critical vulnerability: its susceptibility to cascading errors, where upstream **105** inaccuracies can irrevocably degrade downstream performance. 

**106** 

**107** To circumvent this fragility, end-to-end methods have emerged as a compelling alternative. These models employ a single, unified MLLM to process document images directly, thereby obviating 

2 

Under review as a conference paper at ICLR 2026 

**108** fragile intermediate steps. This OCR-free philosophy was pioneered by models like Donut (Kim **109** et al., 2021) and Pix2Struct (Lee et al., 2023), which reframe document understanding as a direct **110** image-to-sequence task. Recent advancements, such as mPLUG-DocOwl 1.5 and TextMonkey, have **111** further enhanced cross-page understanding and robustness in text-dense scenarios (Hu et al., 2024; **112** Liu et al., 2024). State-of-the-art models like Qwen2.5-VL now demonstrate capabilities that are **113** closing the gap with proprietary systems like GPT-4V on a spectrum of document-centric tasks (Bai **114** et al., 2025; Yang et al., 2023). Despite these advances, such models still grapple with the core **115** challenges of information dilution and evidence localization in long documents—the precise gap our work aims to address. **116** 

**117** 

## **118** 2.2 MULTIMODAL LARGE MODELS AND REASONING STRATEGIES 

**119** 

The fusion of vision and language within MLLMs has unlocked new frontiers in complex reasoning. **120** Architecturally, these models typically consist of a vision encoder, a projection layer for modality **121** alignment, and an LLM backbone for inference. The rapid evolution of open-source models, includ- **122** ing the InternVL series and MiniCPM-V, has been remarkable, steadily narrowing the performance **123** chasm with their proprietary counterparts on diverse multimodal benchmarks (Chen et al., 2024c;b; **124** Yao et al., 2024). 

**125** To elevate their reasoning capabilities from simple perception to complex cognition, strategies like **126** Chain-of-Thought (CoT)(Wei et al., 2022) have been adapted for the multimodal domain (MCoT) **127** (Wang et al., 2025). By generating explicit intermediate reasoning steps, MCoT enhances both **128** model transparency and performance, a benefit substantiated by methods such as DDCoT and Com- **129** positional CoT (Mitra et al., 2024). Such explicit cognitive pathways have been shown to not only **130** boost task performance but also to mitigate the propensity for model hallucination (Wang et al., **131** 2025). 

**132** 

However, for all their success, standard CoT variants overlook a crucial step in the human cognitive **133** process for document analysis: the distinct, sequential act of first locating relevant information **134** before engaging in reasoning. This observation forms the central motivation for our work. While **135** recent efforts have begun to touch upon similar concepts—for instance, SV-RAG employs an MLLM **136** as a retriever to first select evidence (Chen et al., 2024a)—they often remain within a retrieve-then- **137** reason paradigm rather than an integrated, trainable process. The acute challenges highlighted by **138** benchmarks like LongDocURL and MMLongBench-Doc further underscore the urgent need for a **139** more integrated paradigm (Deng et al., 2024; Ma et al., 2024). Drawing conceptual support from the **140** ”multi-paradigm collaboration” ideology in mathematical reasoning (Yu et al., 2025), our **Chain-of141 Reading (CoR)** formalizes this ”locate-then-reason” sequence into a trainable, end-to-end paradigm tailored specifically for deep document understanding. 

**142** 

**143** 

> **144** 3 METHOD **145** 

## **146** 3.1 CHAIN-OF-READING 

**147** 

- **148** 3.1.1 ANALYSIS OF FAILURE CASES 

**149** 

**150** 

**151** 

To identify the bottlenecks of existing methods, we systematically diagnose the failure cases of leading MLLMs on long-document benchmarksA.10.1. Our analysis reveale two fundamental bottlenecks in long-document processing. 

**152** 

**153** 

**Macroscopic Level: Contextual Inertia and Localization Failure.** Models exhibit significant **154** “inertia” when processing long sequences, with sharp performance degradation when critical evi- **155** dence is buried deep within a document or requires cross-page integration. This inefficient retrieval **156** is a contributing factor to localization failure.¸ 

**157** 

> **158 Microscopic Level: Shallow Comprehension.** Even when evidence is successfully located, mod- 

> **159** els could still fail due to insufficient depth of understanding. As shown in Figure 2, they struggle to **160** disambiguate nuanced concepts, perform calculations on tabular data, or interpret complex charts, **161** indicating that they merely succeed in locating evidence but fail to demonstrate genuine comprehension. 

3 

Under review as a conference paper at ICLR 2026 

**162** Tl11aleatteteiattteaiatatiatatetatatatetatatatetatated iittatetetatetetatatetetiatetetetatetetetetetetetstetetetetateteteteteteteteteetatetsatetetetetaeeteteteeieeteteateeeieeaenr|1 **163** 'Query It Result from Base Model (Qwen2.5-VL) 1+11 **164** 'What is the removal T I Based on the document, the BiMgFeO4/Bentonite composite showed a 92% removal ' **165** } efficiency of Methyl 1 efficiency in the tests, under the conditions of 1.5g of adsorbent in 100 mL of a 50 @) ' ' Orange dye using the Es ' 1 mg/L Methyl Orange solution, after 720 minutes of contact time. 4 **166** | BiMgFeO4/Bentonite rt ' | composite as an adsorbent? It Result from Qwen2.5-VL-CoR (Ours) ' **167** 1 i 1 <think> ' **168** i Evidence | | The user wants to know the removal efficiency of Methyl Orange using BiMgFeO4 ' ' " . | t nanoferrite, which requires finding a specific numerical value. | have located the '! **169** :1 "The percentagereneof the solutiondye. |:1 1 .informationol .in section. 5.2, which. contains. a key sentence directly. comparing. the!' : decolorization is higher with the on ; an . Hy **170** rn.:!' BiMgFeO4/Bentonitei composite.. It . 1i1 11 decolorizationperformancewe ofis twohighermaterials.with. theTheBiMgFeO4/Bentonite. sentence is: "The. percentagecomposite.. ofIt thereachedsolutiondye92%° for |1| **171** !' reached100 mL of92%solution,for. 1.5gwhile ofaaa adsorbentinit is 72% for «'1 '1| 1.5g of adsorbent. .in 100 mL ofa solution,. ; whileaeit is 72% for BiMgFeO4. nanoferritewe . as_ t|i **172** io. . ' 1 adsorbent with the same dose.“My task is to extract the percentage specifically forthe ; ' BiMgFeO4 nano ferrite as adsorbent 1! | oy . . _ . 1 **173** ' with the same dose.". 1 BiMgFeO4 nanoferrite", not the composite. The sentence clearly identifies this value as, ' 11 72%. Therefore, the correct value to extract is 72%. ' **174** ! 11 </think> 1' Ground-truth 141 1 <answer> 'i **175** | 72% i 1 72% ' **176** 'aed || | </answer> ' **177 178** Figure 2: Qualitative example illustrating the effectiveness of Chain-of-Reading (CoR). The CoR **179** process is segmented as follows: task planning (orange), phased & focused search (dark blue), cross- **180** modal evidence integration (purple), and synthesized reasoning & verification (blue). **181 182** 3.1.2 THE ”CHAIN-OF-READING” PARADIGM **183 184** To address both macroscopic localization and microscopic comprehension bottlenecks, we propose **185** the **Chain-of-Reading (CoR)** paradigm. CoR emulates an expert’s reading process by transforming **186** unstructured exploration into a structured reasoning chain, as exemplified in Figure 3. The process **187** consists of the following four stages. **188** First, the **Task Planning** stage, in which the model formulates a retrieval strategy based on the query **189** and document structure, such as prioritizing the “Methodology” section for technical questions. **190** Second, the **Phased & Focused Search** stage, during which the model executes a coarse-to-fine **191** iterative search, transforming the needle-in-a-haystack problem into a logical workflow of (1) scope **192** reduction, (2) snippet localization, and (3) field extraction. Third, the **Cross-modal Evidence In193 tegration** stage, in which the model aggregates all the textual and visual evidence located. Finally, **194** the **Synthesized Reasoning & Verification** stage, during which the model reasons over only the **195** integrated evidence chain to generate the answer, effectively minimizing hallucination and reducing **196** computational overhead. **197** Fine-tuning on data with explicit CoR traces significantly enhances model performance in long- **198** document scenarios. For a comprehensive gallery of detailed, step-by-step examples illustrating **199** CoR in various complex scenarios, please see the Appendix A.9. **200 201** 3.2 MASKED AUTO-REGRESSION FOR VISUAL GROUNDING **202 203** While CoR establishes a macroscopic reasoning framework, achieving a deep understanding of **204** information-dense and complex visual elements like charts and diagrams requires finer-grained per- **205** ception and reasoning abilities. However, constructing large-scale supervised datasets for such fine- **206** grained tasks is often prohibitively expensive. To address this challenge, we introduce **Masked Auto-Regression (Mask-AR)** , an efficient self-supervised learning method designed to enhance the **207** model’s capacity for in-depth analysis of complex visual artifacts. **208 209** In this method, we use scientific literature as the primary data source. Scientific articles contain **210** abundant figures and tables, whose captions or titles provide dense, expert-authored summaries of **211** the most important multimodal information. We design a pre-training task in which the MLLM **212** model receives native multimodal inputs (PDF page image sequences), while a layout model masks portions of key caption text. The Mask-AR task is formulated to reconstruct masked text segments **213** by leveraging both the visual content and the surrounding contextual information. **214 215** This task encourages the model to form deep cross-modal associations between visual features and 

In this method, we use scientific literature as the primary data source. Scientific articles contain abundant figures and tables, whose captions or titles provide dense, expert-authored summaries of the most important multimodal information. We design a pre-training task in which the MLLM model receives native multimodal inputs (PDF page image sequences), while a layout model masks portions of key caption text. The Mask-AR task is formulated to reconstruct masked text segments by leveraging both the visual content and the surrounding contextual information. 

This task encourages the model to form deep cross-modal associations between visual features and textual semantics. The self-supervised process implicitly guides the model to analyze the internal 

4 

Under review as a conference paper at ICLR 2026 

|**216**|||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**217**|||||||||||||||**Question: Enumerate the activities organized by the extension programs in the**<br>**neighborhood community**||
|**218**|||||||||||||||**Response:**||
|**219**|||||||||||||||<think><br>The user wants me to list the extension activities. I need to find the section on||
|**220**|||||||||||||||extension activities and extract the list of examples provided. I'll scan the document||
|**221**|||||||||||||||for keywords like "extension activities" or "neighborhood community".||
|**222**|||||||||||||||I've located section 3.4, "Extension Activities," on**page 35**. Within this section,<br>**subsection 3.4.1**describes the activities. It explicitly states, "Some of the activities are||
|**223**|||||||||||||||listed here" and then provides a bulleted list .I will carefully transcribe each item from||
|**224**|||||||||||||||this list to ensure accuracy. The items are:- Health Check-up Camp- Breast Cancer<br>Awareness Programme-Yoga & Meditation Time to time- Campaign against Drug,||
|**225**|||||||||||||||Tobacco and Alcohol addiction- Swachh Bharat Abhiyan- Tree Plantation.- Legal||
|**226**|||||||||||||||awareness regarding Domestic Violence & Human Rights.<br>I will format this list as the answer.||
|**227**|||||||||||||||</think>||
|**228**<br>**229**<br>**230**|||||||||||||||<answer><br>**Health Check-up Camp, Breast Cancer Awareness Programme, Yoga & Meditation**<br>**Time to time, Campaign against Drug, Tobacco and Alcohol addiction, Swachh Bharat**<br>**Abhiyan, Tree Plantation, Legal awareness regarding Domestic Violence & Human**||
|**231**||||||Single-page<br>Evidence|||||||||**Rights.**<br></answer>||
|**232**<br>**233**<br>**234**<br>**235**<br>**236**||=n|||||||||||||**Question: Which policy received the lowest percentage of 'No' responses between**<br>**GE2a, GE2b, GE3, GE4, and BE1?(A). GE2a (B). GE2b (C). GE3 (D). GE4 Choose the**<br>**letter name in front of the right option from A, B, C, D.**<br>**Response:**<br><think>||
|**237**|||||||||||||||This is a multi-step reasoning question. First, I need to find the percentage of 'No'<br>responses for each policy listed in the question. Second, I need to compare them to||
|**238**|||||||||||||||find the lowest one.Third, I need to identify the policy corresponding to that lowest||
|**239**|||||||||||||||value. I will go through the policies one by one. Policy GE2a: The table shows 'No'<br>responses are 0.69% on**page 43**.Policy GE2b: The table shows 'No' responses are||
|**240**|||||||||||||||2.07% on**page 44**.Policy GE3: The table shows 'No' responses are 2.74% on**page 45**.||
|**241**<br>**242**<br>**243**|||||||||||||||Policy GE4: The table shows 'No' responses are 0.70% on**page 47**.Policy BE1: The table<br>shows 'No' responses are 4.29% on**page 48.**Now I will compare these percentages:<br>0.69%, 2.07%, 2.74%, 0.70%, 4.29%. The lowest value is 0.69%. This value corresponds<br>to policy GE2a. The question asks for the letter name of the right option, which is**'A’**.<br></think>||
|**244**|||||||||||||||<answer>||
|**245**<br>**246**||||||Multi-page<br>Evidence|||||Multi-page||||**A**<br></answer>||
|**247**|||||||||||||||||
|**248**|Figure 3: Exemplars of the Chain-of-Reading (CoR) paradigm in action.||||||||||||Exemplars of the Chain-of-Reading (CoR) paradigm in action. Top: precise localization||||
|**249**|and extraction of a list from a specific section; Bottom: cross-page evidence retrieval and comparison||||||||and extraction of a list from a specific section; Bottom: cross-page evidence retrieval and comparison||and extraction of a list from a specific section; Bottom: cross-page evidence retrieval and comparison||||||
|**250**|to identify the minimum value across multiple tables.|||||||||||||to identify the minimum value across multiple tables.|||
|**251**|||||||||||||||||
|**252**<br>**253**<br>**254**|structure of visual elements—such as complex model architectures or multi-step flowcharts—and<br>accurately align these visual cues with their corresponding textual descriptions.||||||||||||||||
|**255**|This enhanced comprehension is crucial for complex tasks, such as identifying and rejecting ques-||||||This enhanced comprehension is crucial for complex tasks, such as identifying and rejecting ques-||||||||||
|**256**|tions based||||on||false premises, as demonstrated in Appendix A.9.11, Example 11. Implementation||||||||||
|**257**|details are provided in Appendix A.1.||details are provided in Appendix A.1.||||||||||||||
|**258**|By leveraging abundant|||||||||||||figure-caption pairs in scientific documents, Mask-AR offers a fully self-|||
|**259**|supervised, data-efficient, and scalable approach for developing advanced visual reasoning capabil-||||supervised, data-efficient, and scalable approach for developing advanced visual reasoning capabil-||||||||||||
|**260**|ities.||||||||||||||||



**261** 

## **262** 4 DATASET AND TRAINING 

**263** 

**264** 

**265** 

## 4.1 DATASET CONSTRUCTION 

## **266** 4.1.1 MOTIVATION AND THE COR-DATASET 

**267** 

**268** The advancement of long-document understanding has been critically hindered by the scarcity of ap- **269** propriate training data. Most existing VQA and document analysis datasets are confined to singlepage input (Huang et al., 2022; Masry et al., 2022), a limitation that precludes models from de- 

5 

Under review as a conference paper at ICLR 2026 

**270** veloping the cross-page reasoning and evidence aggregation capabilities essential for real-world **271** applications involving multi-page reports or scholarly articles. **272 273** To address this critical deficit, we construct the **CoR-Dataset** , a resource specifically engineered **274** following our Chain-of-Reading paradigm. The dataset was curated using a novel, low-cost semiautomated pipeline that yields high-fidelity data, as depicted in Figure 4. This process integrates **275** guided data generation with automated quality assessment and iterative refinement, ultimately yield- **276** ing 26 088 high-quality QA pairs. Each pair is annotated with an explicit reasoning trace that ma- **277** terializes the structured ”reading chain,” providing the direct supervision necessary for our training **278** approach. **A detailed statistical breakdown of the CoR-Dataset’s composition, including dis279 tributions of document types, question intents and reasoning complexity, is provided in Ap280 pendix A.8.** A detailed breakdown of each stage in our data generation pipeline is provided in **281** Appendix A.2. **282 283** nO1 **PDF Source** 1 1 **PDF Parsing Tool** I i **Q&A Generation** I ' **Q&A Scoring & Optimization** 4 1 **Sampling Verification** ' **284** ' i ' POF ' i ' j ! Hy H **285** 1 SS **…...** 1 1 = 1 1 1 1 03 ' 1 foe) ' | **Paper   Report** & i BB _ GS) ' | Gemini : | Vy **286** 1 i 1 to1 itod 11 **287288** t11'1 **Collects diverse PDFs (papers, reports, manuals, contracts)Provides broad coverage** 11i11iu1' **High-Precision PDF Content Parsing, structuring tables and figures as machine-** 11i11111 **LLM creates Q&A pairs with guiding promptsTargets complex reasoning Produces both** 111aaaii)I1 **An evaluator model refines and scores Q&A pairs on key quality criteria.Only high-scoring pairs are** i1'—_11111 **Human experts conduct random checksConfirms scoring consistency and ensures** 'i)1! **289290** 11ae **for model generalization** IH1 '11 **readable inputs.** 1HH 111 **answers and thought of chain** 1HH I1' **selected for the training dataset.** i'4 111 **final dataset quality** '1! **291** Figure 4: Overview of our data generation pipeline for the CoR-Dataset. The process involves **292** document parsing, guided Q&A generation with reasoning trace annotation, automated scoring and **293** refinement, and final human verification. Full details are in Appendix A.2. **294 295 296** 4.1.2 MASK-AR DATASET **297 298** The training data for our Mask-AR objective **299 300** arescientific documents. We note that naive extrac-also sourced from our extensive corpus of 1 | = f H **301** tion of all figure-caption pairs yields a dataset _)/ 6 rote| et: | eee© Gemini |on! **302** fraught with low-quality and irrelevant samples 1' PaperPOF. |\—>|‘ remumeerSeemeee: | 1 oe ae i; **303** (e.g., simple logos or decorative images). To f\ —_= '' 11| "Type": "caption"Stktcntttmase sn — ''| 1'' Le |e x iI **304** ensure that the self-supervised task is both chal' ' "Bboxes": {"x1"..} ' ' ' **305** lenging and semantically meaningful, we enLone!| Uni-parser | loo1 “Contents”:eee [ Fig 2..] 1 io------------4| eee | eer eiceesiateisttateiatataiatstetetatetaietetatetateteteteaiatan **306** gineer a sophisticated filtering pipeline, as de—— Mask-AR Dateset ' **307** picted in Figure 5. Following an initial PDF oie * The Uni-Parser parses literature H parsing with Uni-Parser(Team, 2025), a high \ Goss content into structured results. H **308** precision PDF parsing framework, we employ '' L |; generates the* Gemini selectsdataset.suitable titles and |' **309** a powerful MLLM (Gemini-2.5-Pro), which bee ee ew eee ee eee ee ee ee ee ee ee ee ee eee **310** acts as an expert surrogate to programmati- **311312** cally identify and select the most valuable sam-ples. This curation process is guided by criteFigure 5:self-supervisedcaption pairs,Data curation pipeline for the Mask-ARaftertask. whichUni-Parsera powerfulextractsMLLMfigure- **313** ria that prioritize pairs exhibiting high informa(Gemini-2.5-Pro) acts as an expert surrogate to **314** tion density in the caption and substantial visual filter for high-quality, information-dense samples, **315** complexity in the figure, such as architectural ensuring the effectiveness of the training set. **316** diagrams or plots of experimental results. This **317** meticulous curation is indispensable for creat- **318** ing a dataset that guides the model to develop deep visual-textual reasoning skills. The complete step-by-step methodology is further detailed in Appendix A.1. **319 320 321** 4.2 THREE-STAGE TRAINING STRATEGY **322 323** Our training recipe is progressive, in a three-stage framework, designed methodically to comprehensively enhance the model capabilities of document analysis. 

6 

Under review as a conference paper at ICLR 2026 

**324 Stage 1: Foundational Capability Enhancement.** We start by bolstering the foundational capabili- **325** ties of the base model (Qwen2.5-VL-7B). Using Low-Rank Adaptation (LoRA) Hu et al. (2022), we **326** perform parameter-efficient fine-tuning on a curated mixture of publicly available document anal- **327** ysis datasets. This foundational training is designed to enhance the model’s core competencies in **328** visual text recognition, layout understanding, and table/chart parsing. **A comprehensive list of the 329 datasets employed is detailed in Appendix A.7.** This stage focuses updates on the language model **330** components while the visual encoder remained frozen. **331 Stage 2: Task-Specific Fine-tuning.** The model then undergoes full-parameter fine-tuning on the **332** language model components using our proprietary **CoR-Dataset** and **Mask-AR dataset** . This cru- **333** cial stage deeply ingrains the CoR reasoning patterns and enhances its visual grounding abilities. **334** The training is specifically structured to remediate common failure modes identified in our analysis, **335** such as evidence hallucination, format inconsistency, and superficial content retrieval. 

**336 Stage 3: Preference Alignment.** In the final stage, we align the model’s outputs with human **337** preferences for quality, reliability, and helpfulness using Direct Preference Optimization (DPO) **338** Rafailov et al. (2023). We train the model on a custom-built preference dataset of 5,000 pairs. The **339** preferred (chosen) responses are high-quality examples from our CoR-Dataset, while the undesirable **340** (rejected) responses are synthetically generated to reflect the common error patterns identified. To **341** enhance training stability and mitigate the impact of potential label noise, we employ a hybrid loss **342** function combining the standard sigmoid loss with a robust variant. The mathematical formulation **343** and further details are available in Appendix A.3. **344** 

## **345** 5 EXPERIMENTS 

**346** 

> **347** 5.1 EXPERIMENTAL SETUP **348** 

**349** We fine-tune the Qwen2.5-VL-7B model, henceforth referred to as **Qwen2.5-VL-CoR-7B** . We con- **350** duct a comprehensive evaluation, benchmarking our model against its base version as well as series **351** of leading open-source and proprietary models. Detailed training configurations are provided in **352** Appendix A.5. 

**353** 

**354 Evaluation Benchmarks.** We evaluate model performance on two challenging public benchmarks **355** for long-document multimodal question answering: **MMLongBench-Doc** Ma et al. (2024) and **356 LongDocURL** Deng et al. (2024). These benchmarks are specifically selected as they feature lengthy, multi-page documents and complex queries that necessitate synthesizing evidence across **357** multiple pages and modalities. Consequently, they serve as an ideal testbed for evaluating the core **358** capabilities our work aims to enhance. A detailed statistical breakdown of each benchmark is avail- **359** able in Appendix A.4. **360 361 Evaluation Metrics.** For both benchmarks, we strictly adhere to their official evaluation proto- **362** cols. To facilitate a granular analysis, we report accuracy disaggregated by both the modality of **363** the required evidence and the number of pages from which information must be synthesized. Fur- **364** thermore, we report the overall **generalized accuracy** and **F1 score** to provide a holistic view of **365** performance. The main results are presented in Table 1 and Table 2. **366 367** 5.2 MAIN RESULTS AND ANALYSIS **368 369** The experimental results, presented in Table 1 and Table 2, unequivocally demonstrate the substan- 

**369** The experimental results, presented in Table 1 and Table 2, unequivocally demonstrate the substan- **370** tial performance gains conferred by our proposed framework. 

**371 372** 

**Dominant Performance on MM-LongBench-Doc.** As shown in Table 1, Qwen2.5-VL-CoR-7B establishes a new state-of-the-art among open-source end-to-end models. It achieves an overall accuracy of **37.4%** and an F1-score of **36.0%** , decisively outperforming its base model (23.1% Acc) by a remarkable margin of +14.3 percentage points. This substantial delta underscores the profound impact of our CoR training paradigm and multi-stage fine-tuning strategy. 

**373** 

**374** 

**375** 

**376** 

In a broader comparison, Qwen2.5-VL-CoR-7B not only surpasses all open-source rivals like Docopilot-8B but also outperforms formidable proprietary models such as GPT-4V (32.4%). While 

**377** 

7 

Under review as a conference paper at ICLR 2026 

|**378**<br>**379**<br>**380**<br>**381**<br>**382**<br>**383**<br>**384**<br>**385**<br>**386**<br>**387**<br>**388**<br>**389**<br>**390**<br>**391**<br>**392**<br>**393**<br>**394**<br>**395**<br>**396**<br>**397**<br>**398**<br>**399**<br>**400**<br>**401**<br>**402**<br>**403**<br>**404**<br>**405**<br>**406**<br>**407**<br>**408**<br>**409**<br>**410**<br>**411**<br>**412**<br>**413**<br>**414**<br>**415**<br>**416**<br>**417**<br>**418**<br>**419**<br>**420**<br>**421**<br>**422**<br>**423**<br>**424**<br>**425**<br>**426**<br>**427**<br>**428**<br>**429**<br>**430**<br>**431**|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|Table 1: Detailed performance on the**MM-LongBench-Doc**benchmark. The**best overall**score in<br>each column is bolded, and the best open-source<br>score is underlined. †Results are from Han et al.<br>(2025), Duan et al. (2025), or the offcial benchmark paper (Ma et al., 2024). Abbreviations: SIN<br>(single-page), MUL (multi-page), UNA (unanswerable). All scores are in percentage (%).|
|---|---|---|---|---|---|---|---|
||Method|**Overall**<br>ACC<br>F1||By Page Count (Acc.)<br>SIN<br>MUL<br>UNA||By Evidence Source (Acc.)||
|||||||TXT CHA|LAY<br>TAB<br>FIG|
||_Non-End-to-End Methods (RAG, e_<br>OCR(Tesseract)+GPT-4o†<br>MDocAgent†|_tc.)_<br>30_._5<br>30_._1<br>31_._5<br>—||35_._4<br>29_._3<br>18_._6<br>—<br>—<br>—||41_._1<br>23_._4<br>34_._7<br>32_._3|28_._5<br>38_._1<br>22_._4<br>40_._1<br>29_._4<br>32_._1|
||_End-to-End Methods (Open-source)_<br>Docopilot-8B†<br>28_._8<br>23_._0<br>Qwen2.5-VL-7B<br>23_._1<br>22_._5|||—<br>—<br>—<br>24_._3<br>16_._5<br>31_._1||—<br>—<br>27_._4<br>20_._5|—<br>—<br>—<br>25_._2<br>22_._4<br>20_._3|
||**Qwen2.5-VL-CoR-7B (Ours)**<br>37.4 **(+14.3)**<br>36.0 **(+13.5)**|||41.9<br>25.9<br>45.5||39.4<br>27.7|31.2<br>38.6<br>27.5|
||_End-to-End Methods (Proprietary)_<br>GPT-4V†<br>32_._4<br>31_._2<br>Gemini-1.5-Pro†<br>28_._2<br>20_._6<br>GPT-4o†<br>**42****_._8**<br>**44****_._9**|||36_._4<br>27_._0<br>31_._2<br>21_._1<br>11_._1<br>**69****_._2**<br>**54****_._5 41****_._5**<br>20_._2||34_._4<br>28_._3<br>21_._0<br>17_._6<br>**46****_._3 46****_._0 **|28_._2<br>32_._4<br>26_._8<br>6_._9<br>14_._5<br>15_._2<br> **45****_._3 50****_._0 44****_._1**|
||Table 2: Detailed performance on the**LongDoc-URL**benchmark. The**best overall**sc<br>and the best open-source<br>score is underlined.<br>†Results are from Han et al. (2025) o<br>benchmark paper (Deng et al., 2024). All scores are reported as Accuracy (%).|||||||
||Method|**Overall**<br>ACC|Main Task<br>UND REA LOC||Element Type<br> TXT<br>LAY<br>FIG<br>TAB|||
||_Non-End-to-End Methods (Agent-based, etc.)_<br>OCR(PyMuPDF) + GPT-4o†<br>34_._7<br>OCR(PyMuPDF) + o1-preview†<br>35_._8<br>MDocAgent†<br>51_._7||35_._3<br>28_._0<br>37_._2<br>35_._6<br>30_._6<br>38_._6<br>—<br>—<br>—||34_._3<br>33_._7<br>35_._0<br>26_._9<br>33_._2<br>36_._8<br>35_._9<br>33_._0<br>—<br>—<br>—<br>—|||
||_End-to-End Methods (Open-source)_<br>Qwen2-VL-7B†<br>30_._6<br>Qwen2.5-VL-7B<br>39_._2||36_._8<br>24_._0<br>22_._6<br>44_._5<br>31_._2<br>33_._5||33_._4<br>38_._2<br>30_._9<br>24_._3<br>42_._8<br>43_._9<br>37_._5<br>33_._3|||
||**Qwen2.5-VL-CoR-7B (Ours)**<br>51.5 **(+12.3)**||56.3<br>41.2<br>48.6||55.6<br>51.4<br>48.2<br>46.2|||
||_End-to-End Methods (Proprietary)_<br>Qwen-VL-Max†<br>49_._5<br>Gemini-1.5-Pro†<br>50_._9<br>GPT-4o†<br>**64****_._5**||58_._9<br>43_._9<br>36_._0<br>55_._6<br>42_._3<br>46_._4<br>**68****_._6 59****_._3 59****_._6 **||53_._5<br>55_._2<br>52_._5<br>46_._7<br>51_._8<br>**56****_._1**<br>52_._1<br>43_._1<br> **66****_._3 64****_._1 67****_._5 60****_._2 **|||



Crucially, our model’s performance transcends the open-source sphere and is highly competitive with top-tier proprietary systems. It is particularly noteworthy that Qwen2.5-VL-CoR-7B (51.5%) effectively matches the performance of the powerful, agent-based MDocAgent system (51.7%) and **surpasses other leading proprietary models, including Qwen-VL-Max (49.5%) and Gemini-** 

8 

Under review as a conference paper at ICLR 2026 

**432 1.5-Pro (50.9%)** . This is a remarkable achievement for a 7B-parameter model, demonstrating that **433** our targeted training approach can bridge the performance gap typically attributed to massive model **434** scale or complex external tool usage. The ability to outperform larger proprietary models under- **435** scores the efficiency and power of instilling structured reasoning directly into the model. **436 437 Summary of Experimental Findings.** In summary, our comprehensive evaluations on two de- **438** manding long-document benchmarks validate the superiority of our methodology. Qwen2.5-VL- **439** CoR-7B consistently sets a new standard for open-source models in this domain. The results furnish **440** compelling evidence that with a principled, data-centric approach to teaching structured reasoning, **441** smaller models can not only compete with but, in certain cases, surpass their much larger, proprietary **442** counterparts. **443 444** 5.3 ABLATION STUDIES **445 446** To rigorously dissect the contribution of each component within our framework, we conduct a comprehensive ablation study. We systematically evaluate the incremental impact of Supervised Fine- **447** Tuning (SFT) on our Chain-of-Reading (CoR) and Mask-AR datasets, followed by Direct Preference **448** Optimization (DPO). The results are summarized in Table 3. **449 450** Table 3: Main ablation study on overall accuracy (%). The checkmarks (✓) indicate which compo-✓) indicate which compo-) indicate which compo- **451** nents are included in each configuration. The performance gains for each step are shown relative to **452** the base model. **453 454 Components Overall Accuracy (%) Configuration 455** CoR SFT Mask-AR DPO MMLongBench LongDocURL **456** Base Model 23.1 39.2 **457** + CoR ✓ 34.0 **(+10.9)** 47.0 **(+7.8) 458** + CoR + Mask-AR ✓ ✓ 35.1 **(+12.0)** 48.1 **(+8.9) 459** + CoR + DPO ✓ ✓ 35.9 **(+12.8)** 48.9 **(+9.7) 460 Full Model (Ours)** ✓ ✓ ✓ **37.4 (+14.3) 51.5 (+12.3) 461 462 Analysis of Component Synergy.** The main ablation results in Table 3 clearly elucidate the effec- **463** tiveness of our multi-stage architecture. SFT with the CoR dataset provides a foundational perfor- **464** mance boost(+10.9% and +7.8% on the two benchmarks, respectively), establishing robust reason- **465** ing capabilities . Both Mask-AR and DPO contribute further gains on top of this foundation. Critically, the full model (Row 5), which integrates all three components, achieves the highest scores, **466** confirming a powerful synergistic effect. This indicates that enhancing visual grounding (Mask-AR) **467** and aligning with human preferences (DPO) are complementary, rather than redundant, to the core **468** reasoning patterns instilled by CoR. **469 470 Component-Specific Contributions.** To further investigate these effects, we analyzed the spe- **471** cific roles of Mask-AR and DPO. Our fine-grained analysis reveals that Mask-AR provides a tar- **472** geted boost to visual-centric questions, measurably improving accuracy on queries requiring chart **473** and figure interpretation. Concurrently, DPO proves instrumental in refining higher-level cogni- **474** tive abilities, yielding the most substantial gains in complex, multi-page reasoning tasks where nu- **475** anced judgment is paramount. A detailed breakdown substantiating these claims is provided in **476** Appendix A.6. **477 478** 6 CONCLUSION **479** 

To rigorously dissect the contribution of each component within our framework, we conduct a comprehensive ablation study. We systematically evaluate the incremental impact of Supervised FineTuning (SFT) on our Chain-of-Reading (CoR) and Mask-AR datasets, followed by Direct Preference Optimization (DPO). The results are summarized in Table 3. 

Table 3: Main ablation study on overall accuracy (%). The checkmarks (✓) indicate which compo-✓) indicate which compo-) indicate which components are included in each configuration. The performance gains for each step are shown relative to the base model. 

**Analysis of Component Synergy.** The main ablation results in Table 3 clearly elucidate the effectiveness of our multi-stage architecture. SFT with the CoR dataset provides a foundational performance boost(+10.9% and +7.8% on the two benchmarks, respectively), establishing robust reasoning capabilities . Both Mask-AR and DPO contribute further gains on top of this foundation. Critically, the full model (Row 5), which integrates all three components, achieves the highest scores, confirming a powerful synergistic effect. This indicates that enhancing visual grounding (Mask-AR) and aligning with human preferences (DPO) are complementary, rather than redundant, to the core reasoning patterns instilled by CoR. 

**Component-Specific Contributions.** To further investigate these effects, we analyzed the specific roles of Mask-AR and DPO. Our fine-grained analysis reveals that Mask-AR provides a targeted boost to visual-centric questions, measurably improving accuracy on queries requiring chart and figure interpretation. Concurrently, DPO proves instrumental in refining higher-level cognitive abilities, yielding the most substantial gains in complex, multi-page reasoning tasks where nuanced judgment is paramount. A detailed breakdown substantiating these claims is provided in Appendix A.6. 

**480** 

This paper presents Chain-of-Reading (CoR), an end-to-end paradigm for document understanding. **481** CoR enhances multimodal document QA by structuring document-level reasoning through explicit **482** reasoning paths. It further leverages Masked Auto-Regression for fine-grained visual comprehension **483** with self-supervised visual grounding. Qwen2.5-VL-CoR-7B achieves accuracy improvements of **484** 14.3% on MMLongBench-Doc and 12.3% on LongDocURL compared to Qwen2.5-VL-7B, and, **485** despite having only 7B parameters, delivers performance comparable to proprietary MLLMs such as GPT-4o. 

9 

Under review as a conference paper at ICLR 2026 

> **486** REFERENCES **487 488** Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha. Doc- **489** former: End-to-end transformer for document understanding. In _Proceedings of the IEEE/CVF_ **490** _international conference on computer vision_ , pp. 993–1003, 2021. 

- **491** Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, **492** Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. _arXiv preprint arXiv:2502.13923_ , **493** 2025. 

**494** 

- Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical 

- **495** understanding for academic documents. _arXiv preprint arXiv:2308.13418_ , 2023. 

**496** 

- **497** Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A Rossi, **498** Changyou Chen, and Tong Sun. Sv-rag: Lora-contextualizing adaptation of mllms for long doc- **499** ument understanding. _arXiv preprint arXiv:2411.01106_ , 2024a. 

- **500** Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shen- **501** glong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source **502** multimodal models with model, data, and test-time scaling. _arXiv preprint arXiv:2412.05271_ , **503** 2024b. **504** Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong 

- **505** Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning 

- **506** for generic visual-linguistic tasks. In _Proceedings of the IEEE/CVF Conference on Computer_ 

- **507** _Vision and Pattern Recognition_ , pp. 24185–24198, 2024c. **508 509** Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning **510** language models with noisy feedback. _arXiv preprint arXiv:2403.00409_ , 2024. **511** Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, **512** Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. _arXiv preprint_ **513** _arXiv:2507.05595_ , 2025. **514** Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun 

- **515** Song, Bo Zheng, et al. Longdocurl: a comprehensive multimodal long document benchmark 

- **516** integrating understanding, reasoning, and locating. _arXiv preprint arXiv:2412.18424_ , 2024. 

- **517 518** Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin **519** Hou, Tong Lu, Hongsheng Li, et al. Docopilot: Improving multimodal models for document-level **520** understanding. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pp. 4026–4037, 2025. 

- **521 522** Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre **523** Colombo. Colpali: Efficient document retrieval with vision language models. _arXiv preprint_ **524** _arXiv:2407.01449_ , 2024. **525** Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. Mdoca- 

- **526** gent: A multi-modal multi-agent framework for document understanding. _arXiv preprint_ 

- **527** _arXiv:2503.13964_ , 2025. **528 529** Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and **530** Jingren Zhou. mplug-docowl2: High-resolution compressing for ocr-free multi-page document **531** understanding. _arXiv preprint arXiv:2409.03420_ , 2024. 

- **532** Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, **533** Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _ICLR_ , 1(2):3, 2022. 

- **534 535** Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM international_ 

- **536** _conference on multimedia_ , pp. 4083–4091, 2022. 

- **537 538** Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, **539** Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Donut: Document understanding transformer without ocr. _arXiv preprint arXiv:2111.15664_ , 7(15):2, 2021. 

10 

Under review as a conference paper at ICLR 2026 

- **540** Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, 

- **541** Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screen- **542** shot parsing as pretraining for visual language understanding. In _International Conference on_ **543** _Machine Learning_ , pp. 18893–18912. PMLR, 2023. **544 545** Wenhui Liao, Jiapeng Wang, Hongliang Li, Chengyu Wang, Jun Huang, and Lianwen Jin. Do- **546** clayllm: An efficient multi-modal extension of large language models for text-rich document **547** understanding. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pp. 4038–4049, 2025. 

- **548 549** Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. **550** Textmonkey: An ocr-free large multimodal model for understanding document. _arXiv preprint_ **551** _arXiv:2403.04473_ , 2024. **552 553** Nikolaos Livathinos, Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Panos Vage- **554** nas, Cesar Berrospi Ramis, Matteo Omenetti, Kasper Dinkla, Yusik Kim, et al. Docling: An ef- **555** ficient open-source toolkit for ai-driven document conversion. _arXiv preprint arXiv:2501.17887_ , 2025. 

- **556 557** Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, **558** Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document under- **559** standing with visualizations. _Advances in Neural Information Processing Systems_ , 37:95963– **560** 96010, 2024. **561 562** Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A bench- **563** mark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 

- **564 565** Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of- 

- **566** thought prompting for large multimodal models. In _Proceedings of the IEEE/CVF Conference_ **567** _on Computer Vision and Pattern Recognition_ , pp. 14420–14431, 2024. **568 569** Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea **570** Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances_ **571** _in neural information processing systems_ , 36:53728–53741, 2023. **572** Uni-Parser Team. Uniparser, 2025. URL https://huggingface.co/UniParser. 

- **573 574** Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, **575** Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content **576** extraction. _arXiv preprint arXiv:2409.18839_ , 2024a. **577 578** Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, **579** Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. _arXiv preprint arXiv:2401.06080_ , 2024b. 

- **580 581** Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and **582** Hao Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. _arXiv preprint_ **583** _arXiv:2503.12605_ , 2025. **584 585** Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chun- **586** rui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language **587** model. In _European Conference on Computer Vision_ , pp. 408–424. Springer, 2024. 

- **588** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny 

- **589** Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in_ **590** _neural information processing systems_ , 35:24824–24837, 2022. **591** 

- **592** Xudong Xie, Hao Yan, Liang Yin, Yang Liu, Jing Ding, Minghui Liao, Yuliang Liu, Wei Chen, and **593** Xiang Bai. Wukong: A large multimodal model for efficient long pdf reading with end-to-end sparse sampling. _arXiv preprint arXiv:2410.05970_ , 2024. 

11 

Under review as a conference paper at ICLR 2026 

- **594** Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Li- 

- **595** juan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv preprint_ **596** _arXiv:2309.17421_ , 2023. 

## **597** 

- **598** Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, **599** Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. _arXiv preprint arXiv:2408.01800_ , 2024. 

## **600** 

- **601** Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, **602** Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model **603** for document understanding. _arXiv preprint arXiv:2307.02499_ , 2023. 

**604** 

- **605** Yiyao Yu, Yuxiang Zhang, Dongdong Zhang, Xiao Liang, Hengyuan Zhang, Xingxing Zhang, Mahmoud Khademi, Hany Awadalla, Junjie Wang, Yujiu Yang, et al. Chain-of-reasoning: Towards 

- **606** unified mathematical reasoning in large language models via a multi-paradigm perspective. _arXiv_ 

- **607** _preprint arXiv:2501.11110_ , 2025. 

**608 609** 

## **610** A APPENDIX 

**611** 

> **612** This appendix provides supplementary details on our methodology and dataset construction to facil- 

> **613** itate reproducibility and deeper understanding. **614** 

- **615** A.1 IMPLEMENTATION DETAILS OF MASKED AUTO-REGRESSION (MASK-AR) 

**616 617** The implementation of our Mask-AR self-supervised objective follows a structured process designed **618** to maximize its learning signal for deep, cross-modal reasoning. The process, illustrated in the main **619** text in Figure 5, consists of the following steps: 

**620** 

1. **Extraction:** We use a high-fidelity document parser (Uni-Parser) to extract all figure im- 

**621** ages and their corresponding caption texts from a large corpus of scientific and technical 

**622** documents. Each figure-caption pair is maintained with a link to its source document. 

- **623 624** 2. **Intelligent Filtering:** To create a challenging and high-quality training set, we filter the extracted pairs. Each pair, along with its full document context, is evaluated by a powerful 

- **625** MLLM (Gemini-2.5-Pro) based on predefined criteria: 

- **626 627** • **Information Density:** Captions that are rich in technical details, experimental results, **628** or key conclusions are preferred over simple descriptive labels (e.g., ”Figure 1: System Overview”). 

- **629 630** • **Visual Complexity:** Figures with multiple components, data series, complex layouts, or abstract concepts are prioritized. 

- **631 632** • **Content Relevance:** We select figures that are central to the document’s main contributions, such as model architecture diagrams or plots of primary experimental results. 

- **633 634** 3. **Sample Construction:** For each selected document, we adhere to a ”one instance per **635** document” principle. We mask the caption of only the single most representative figure **636** identified during the filtering stage. The training sample then consists of all pages of the document (with the target caption text masked out) and the target figure image. 

- **637 638** 4. **Training Objective:** The model is trained to auto-regressively generate the original, un- **639** masked caption text. This task compels the model to synthesize information from both the **640** visual data in the figure and the textual context scattered throughout the document, effectively teaching it to perform the complex cognitive process of summarizing visual evidence 

- **641** in context. 

**642** 

- **643** A.2 DATASET CONSTRUCTION DETAILS 

- **644** 

- **645** A.2.1 COR-DATASET GENERATION PIPELINE 

**646** 

**647** The CoR-Dataset was constructed using the semi-automated pipeline shown in Figure 4. The four key stages are: 

12 

Under review as a conference paper at ICLR 2026 

- **648** 1. **Document Collection and Parsing:** We first gathered a diverse collection of PDF doc- 

- **649** uments spanning scientific literature, financial reports, technical manuals, and legal con- 

- **650** tracts. Each document was processed with Uni-Parser, a high-performance tool that per- **651** forms OCR and structures content like tables and lists, providing a clean, machine-readable **652** foundation. 

**653** 

2. **Guided Q&A and CoR Generation:** The parsed document content was fed to a power- 

**654** ful teacher model (GPT-4o). We used carefully engineered prompts to guide the model to **655** generate question-answer pairs that necessitate complex reasoning (e.g., cross-page com- **656** parison, chart interpretation with text). Crucially, we also prompted the model to output a **657** detailed, step-by-step ”reading chain” that explicitly follows our CoR paradigm, serving as **658** the ground-truth reasoning path. 

**659** 

3. **Automated Quality Assessment and Refinement:** To ensure data quality, we employed 

**660** an independent evaluator model (Gemini-2.5-Pro) to score each generated sample. The **661** scoring criteria included the logical soundness of the question, the clarity of the CoR chain, **662** and the factual accuracy of the answer. Low-scoring samples were either discarded or sent **663** back to the teacher model with feedback for revision, creating a closed-loop optimization **664** process that continuously improved data quality. 

> **665** 4. **Human Verification:** The final stage involved manual review and verification by human 

> **666** annotators to filter out any remaining subtle errors and ensure the dataset’s overall reliabil- 

> **667** ity. **668** 

- **669** A.3 DETAILS OF THE DPO TRAINING STAGE **670** 

> **671** In Stage 3 of our training, we used Direct Preference Optimization (DPO) to align the model with **672** human preferences. 

**673** 

**==> picture [432 x 66] intentionally omitted <==**

> **680** where _D_ is the preference dataset, _πθ_ is the policy model being optimized, _π_ ref is a fixed reference 

> **681** model (initialized from the Stage 2 checkpoint), _β_ is a temperature hyperparameter, and _σ_ is the 

> **682** logistic sigmoid function. 

**683** 

**684** 

> **684 Preference Dataset Construction.** We constructed a high-quality preference dataset containing 

> **685** 5,000 pairs. The generation process was as follows: 

- **686 687** • **Preferred Responses (** _yw_ **):** We selected high-scoring, correct examples from a held-out **688** portion of our CoR-Dataset. These represent the ideal model outputs in terms of format, reasoning, and accuracy. 

**689 690** • **Dispreferred Responses (** _yl_ **):** We first conducted a thorough error analysis of the outputs **691** from the Stage 2 model. Based on a typology of common errors (e.g., factual inaccuracies, **692** evidence misattribution, format violations, lazy retrieval), we prompted Gemini-2.5-Pro to **693** generate corresponding dispreferred responses for each prompt _x_ and its preferred response **694** _yw_ . This ensures that the model learns to avoid specific, realistic failure modes. **695 696 Hybrid Loss Function.** To enhance training stability and robustness against potential label noise 

**Hybrid Loss Function.** To enhance training stability and robustness against potential label noise in our synthetically-aided preference dataset, we employed a hybrid loss function that combines two variants. The total loss _L_ total is a weighted sum: 

**697 698 699 700 701** 

**==> picture [270 x 11] intentionally omitted <==**

- where _w_ 1 = 0 _._ 7 and _w_ 2 = 0 _._ 3 (configured via --loss ~~t~~ ype sigmoid robust). The components are: 

13 

Under review as a conference paper at ICLR 2026 

> **702** • **Sigmoid Loss (** _L_ **sigmoid):** This is the standard loss from the original DPO paper Rafailov 

> **703** et al. (2023), equivalent to Equation 1: **704 705 706** _L_ sigmoid( _πθ_ ; _π_ ref) = _−_ E( _x,yw,yl_ ) _∼D_ �log _σ_ � _β_ log _π[π]_ ref _[θ]_[(] ( _[y] y[w] w[|] |[x] x_[)] ) _[−][β]_[ log] _π[π]_ ref _[θ]_[(] ( _[y] y[l] l[|] |[x] x_[)] ) �� _,_ (3) 

> **707** where _σ_ is the sigmoid function, fitting a Bradley-Terry model to the preferences. **708** • **Robust Loss (** _L_ **robust):** This variant is an unbiased estimator of the DPO loss that is resilient **709** to preference noise in the data Wang et al. (2024b); Chowdhury et al. (2024). It models the 

> **710** possibility of incorrect preference labels via a label smoothing hyperparameter _ε ∈_ (0 _,_ 1 _/_ 2) 

> **711** (the flip rate of preference labels). The loss is defined as: **712** _N_ **713** (1 _− ε_ ) _L_ sigmoid( _πθ_ ; _π_ ref _, xi,_ ˜ _yw,i,_ ˜ _yl,i_ ) _− εL_ sigmoid( _πθ_ ; _π_ ref _, xi,_ ˜ _yl,_ **714** _L_ robust( _πθ_ ; _π_ ref) = _N_[1] � 1 _−_ 2 _ε i_ =1 

> **715** (4) 

> **716** where ˜ _yw,i_ and ˜ _yl,i_ are the potentially noisy preferred and dispreferred responses for prompt 

> **717** _xi_ , and _N_ is the batch size. When _ε_ = 0, this reduces to the standard sigmoid loss. In our **718** experiments, we used _ε_ = 0 _._ 1 (or specify your value if different). **719** 

> **720** A.4 EVALUATION BENCHMARKS **721** 

**==> picture [392 x 41] intentionally omitted <==**

**722** Our experiments were conducted on the following standard long-document VQA benchmarks, **723** which are designed to test a model’s ability to comprehend and reason over lengthy, visually com- **724** plex documents. **725** 

- **MMLongBench-Doc** Ma et al. (2024): This benchmark consists of 135 long-form PDF 

- **726** documents, with an average of 47.5 pages and 21,214 tokens per document. It contains 

- **727** 1,082 expert-annotated questions designed to test long-context understanding. 

**728** • **LongDocURL** Deng et al. (2024): This dataset is constructed from 396 lengthy PDF docu- **729** ments, averaging 85.6 pages and 43,622.6 tokens. It includes 2,325 high-quality question- **730** answering pairs. A key challenge of this benchmark is that correct answers often require **731** synthesizing evidence from multiple modalities (e.g., text, tables, images) and across dif- **732** ferent pages. **733** 

> **734** A.5 TRAINING CONFIGURATIONS **735** 

**736** All fine-tuning was performed on a server equipped with **8 NVIDIA A100 80GB GPUs** . The **737** training utilized the PyTorch framework, along with libraries such as Hugging Face Transformers **738** and Swift. The base model for all stages is **Qwen2.5-VL-7B** . Below are the specific configurations for each of our three training stages. 

**739** 

**740** 

- **741** A.5.1 STAGE 1: FOUNDATIONAL CAPABILITY ENHANCEMENT (LORA) 

**742** 

**743** 

In this stage, we performed parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) to enhance the model’s core document understanding abilities on a mixture of public datasets. 

**744** 

**745 746 747 748 749 750 751 752 753 754** 

**755** 

- **Method:** Low-Rank Adaptation (LoRA). 

- **Trained Components:** LoRA adapters were applied to the language model’s attention (q ~~p~~ roj, k ~~p~~ roj, v ~~p~~ roj, o ~~p~~ roj) and MLP (gate ~~p~~ roj, up ~~p~~ roj, down ~~p~~ roj) layers, as well as the multimodal projector (mm ~~p~~ rojector). The visual encoder weights remained frozen. 

- **LoRA Hyperparameters:** 

   - LoRA Rank ( _r_ ): 16 

   - LoRA Alpha ( _α_ ): 32 

   - LoRA Dropout: 0.05 

- **Training Hyperparameters:** 

   - Optimizer: AdamW 

   - Learning Rate: 1 _._ 0 _×_ 10 _[−]_[4] 

   - LR Scheduler: Cosine decay with a 10% warmup ratio 

14 

Under review as a conference paper at ICLR 2026 

**756 757** 

**758** 

**759** 

**760** 

   - Global Batch Size: 64 (1 per device _×_ 8 accumulation steps _×_ 8 GPUs) 

   - Number of Epochs: 3.0 

   - Precision: bfloat16 

   - Max Sequence Length: 32,768 

- **761 –** Attention Implementation: Flash Attention 2 

- **762 –** Weight Decay: 0.05 

- **763 –** Gradient Clipping Norm: 0.3 

- **764** 

- **765** A.5.2 STAGE 2: TASK-SPECIFIC FINE-TUNING (FULL-PARAMETER) 

> **766** This stage involved full-parameter fine-tuning on our proprietary CoR-Dataset and Mask-AR dataset 

> **767** to instill the Chain-of-Reading reasoning patterns. 

**768** 

- **Method:** Full-parameter supervised fine-tuning. 

**769** 

- **Trained Components:** We updated the full weights of the language model and the multimodal 

- **770** projector. The visual encoder (vision ~~t~~ ower) remained frozen throughout this stage. 

**771** 

- **Training Hyperparameters:** 

**772** 

- Optimizer: AdamW 

**773** 

- Learning Rate: 1 _._ 0 _×_ 10 _[−]_[5] 

**774** 

- LR Scheduler: Cosine decay with a 5% warmup ratio 

**775** 

- **776 –** Global Batch Size: 16 (1 per device _×_ 2 accumulation steps _×_ 8 GPUs) 

**777** 

   - Number of Epochs: 1.0 

- **778 –** Precision: bfloat16 

- 

- **779** Max Sequence Length: 32,768 

- **780 –** Parallelism Strategy: DeepSpeed ZeRO Stage 3 

- **781 –** Attention Implementation: Flash Attention 2 

**782** 

- **783** A.5.3 STAGE 3: PREFERENCE ALIGNMENT (DPO WITH LORA) 

**784** 

In the final stage, we aligned the model with human preferences using Direct Preference Optimiza- **785** tion (DPO). For computational efficiency, this stage was also conducted using LoRA. 

**786** 

- **Method:** Direct Preference Optimization (DPO) with LoRA. 

**787** 

**788** • **Reference Model:** The reference model ( _p_ ref) for calculating the KL-divergence was the check- **789** point obtained at the end of Stage 2. • **Hybrid Loss Function:** As mentioned in Section 4.2, we employed a hybrid loss function. The **790** final loss was a weighted sum of the standard sigmoid loss and a robust loss variant: _L_ hybrid = **791** 0 _._ 7 _× L_ sigmoid + 0 _._ 3 _× L_ robust. **792** • **LoRA Hyperparameters: 793 –** LoRA Rank ( _r_ ): 8 **794 –** LoRA Alpha ( _α_ ): 32 **795 –** Target Modules: All linear layers in the language model. **796** • **Training Hyperparameters: 797 798 –** Optimizer: AdamW **799 –** Learning Rate: 5 _._ 0 _×_ 10 _[−]_[6] **800 –** LR Scheduler: Cosine decay with a 5% warmup ratio 

- Global Batch Size: 16 (1 per device _×_ 2 accumulation steps _×_ 8 GPUs) 

**801 802** 

- Number of Epochs: 1.0 

**803 –** Precision: bfloat16 **804 –** 

- Max Sequence Length: 32,767 

- Parallelism Strategy: DeepSpeed ZeRO Stage 3 

**805** 

**806** 

- **807** A.6 DETAILED BREAKDOWN OF ABLATION COMPONENT EFFECTS 

**808 809** To further dissect the results of our main ablation study (Table 3), we analyzed the specific impact of the Mask-AR and DPO stages on relevant sub-tasks. 

15 

Under review as a conference paper at ICLR 2026 

**810 Effect of Mask-AR on Visual Element Understanding.** To specifically isolate the impact of **811** the Mask-AR dataset on visual parsing, we compare performance on visually-intensive evidence **812** types before and after its inclusion, across both benchmarks. As shown in Table 4, adding Mask- **813** AR SFT consistently improves accuracy on questions related to figures and charts/tables. On **814** MMLongBench-Doc, chart-related accuracy increases by **+3.1%** , while on LongDocURL, figure **815** accuracy improves by **+3.2%** . This consistently positive impact across different benchmarks and vi- **816** sual types directly validates our hypothesis that Mask-AR enhances the model’s ability to interpret **817** and extract information from complex visual elements. **818 819** Table 4: Impact of Mask-AR on visual categories (Accuracy, %) across both benchmarks. **820 821** Benchmark Evidence Type + CoR SFT + CoR + Mask-AR SFT **822** Chart (CHA) 23 _._ 1 **26** _**.**_ **2** MMLongBench **823** Figure (FIG) 20 _._ 7 **21** _**.**_ **3 824** Figure 44 _._ 3 **47** _**.**_ **5** LongDocURL **825** Table 41 _._ 8 **42** _**.**_ **3 826 827 Effect of DPO on Higher-Level Cognitive Abilities.** We hypothesize that DPO’s primary role is **828** to refine the model’s high-level cognitive abilities. To verify this, we measured its impact on complex **829** reasoning and comprehension sub-tasks in both benchmarks. Table 5 shows that applying DPO **830** yields significant gains in these crucial areas. It boosts multi-page reasoning on MMLongBench **831** by a remarkable **+7.6%** , demonstrating an improved ability to synthesize information across long **832** contexts. Similarly, on LongDocURL, it enhances both Understanding ( **+3.2%** ) and Reasoning **833** ( **+1.6%** ). This robust evidence across two benchmarks confirms that DPO is crucial for aligning the **834** model with nuanced human expectations, fundamentally improving its ability to think and reason **835** through complex problems. **836 837** Table 5: Impact of DPO on reasoning and comprehension (Accuracy, %) across both benchmarks. **838 839** Benchmark Sub-task SFT Only (CoR+Mask-AR) + DPO (Full Model) **840** MMLongBench Multi-page (MUL) 18 _._ 3 **25** _**.**_ **9 841** Understanding (UND) 53 _._ 1 **56** _**.**_ **3 842** LongDocURL Reasoning (REA) 39 _._ 6 **41** _**.**_ **2 843 844 845** A.7 DATASETS FOR STAGE 1 FOUNDATIONAL FINE-TUNING **846 847** In the first stage of our training, we performed LoRA-based fine-tuning on a diverse collection of public and curated datasets to enhance the model’s fundamental document understanding skills. The **848** datasets were carefully selected to cover a wide range of tasks, including document-based visual **849** question answering (DocVQA), table question answering (TableQA), and chart question answering **850** (ChartQA). This mixed-data approach ensures the model develops robust capabilities across various **851** document types and formats before undergoing specialized training in Stage 2. Table 6 provides a **852** detailed summary of each dataset component. **853 854** A.8 STATISTICAL ANALYSIS OF THE COR-DATASET **855 856** The CoR-Dataset was meticulously designed to encompass a wide diversity of documents, question **857** types, and reasoning challenges, reflecting the complexity of real-world document analysis tasks. **858** In total, the dataset comprises 26 087 high-quality, annotated question-answer pairs. To ensure its **859** breadth and depth, we analyzed its composition across several key dimensions. A summary of the **860** primary statistics is presented in Table 7, while the detailed distributions for each dimension are illustrated in **Figures 6 through 9** . **861 862** The distributions highlight a focus on academic and technical documents, which provide fertile **863** ground for complex questions. The question intents are predominantly geared towards factual extraction, but with significant representation from summarization, comparison, and causal inquiries, 

**Effect of Mask-AR on Visual Element Understanding.** To specifically isolate the impact of the Mask-AR dataset on visual parsing, we compare performance on visually-intensive evidence types before and after its inclusion, across both benchmarks. As shown in Table 4, adding MaskAR SFT consistently improves accuracy on questions related to figures and charts/tables. On MMLongBench-Doc, chart-related accuracy increases by **+3.1%** , while on LongDocURL, figure accuracy improves by **+3.2%** . This consistently positive impact across different benchmarks and visual types directly validates our hypothesis that Mask-AR enhances the model’s ability to interpret and extract information from complex visual elements. 

**Effect of DPO on Higher-Level Cognitive Abilities.** We hypothesize that DPO’s primary role is to refine the model’s high-level cognitive abilities. To verify this, we measured its impact on complex reasoning and comprehension sub-tasks in both benchmarks. Table 5 shows that applying DPO yields significant gains in these crucial areas. It boosts multi-page reasoning on MMLongBench by a remarkable **+7.6%** , demonstrating an improved ability to synthesize information across long contexts. Similarly, on LongDocURL, it enhances both Understanding ( **+3.2%** ) and Reasoning ( **+1.6%** ). This robust evidence across two benchmarks confirms that DPO is crucial for aligning the model with nuanced human expectations, fundamentally improving its ability to think and reason through complex problems. 

In the first stage of our training, we performed LoRA-based fine-tuning on a diverse collection of public and curated datasets to enhance the model’s fundamental document understanding skills. The datasets were carefully selected to cover a wide range of tasks, including document-based visual question answering (DocVQA), table question answering (TableQA), and chart question answering (ChartQA). This mixed-data approach ensures the model develops robust capabilities across various document types and formats before undergoing specialized training in Stage 2. Table 6 provides a detailed summary of each dataset component. 

The CoR-Dataset was meticulously designed to encompass a wide diversity of documents, question types, and reasoning challenges, reflecting the complexity of real-world document analysis tasks. In total, the dataset comprises 26 087 high-quality, annotated question-answer pairs. To ensure its breadth and depth, we analyzed its composition across several key dimensions. A summary of the primary statistics is presented in Table 7, while the detailed distributions for each dimension are illustrated in **Figures 6 through 9** . 

16 

Under review as a conference paper at ICLR 2026 

**==> picture [432 x 641] intentionally omitted <==**

**----- Start of picture text -----**<br>
864<br>865 Distribution of Document Type<br>866<br>867<br>868 academic_paper 55.3% (14417)<br>869 government_report 20.0% (5229)<br>870<br>871 product_guide 7.6% (1988)<br>872 technical_manual 5.7% (1482)<br>873<br>874 financial_statement 3.7% (963)<br>875 marketing_brochure 2.2% (569)<br>876<br>legal_document 1.9% (497)<br>877<br>878 corporate_policy 1.8% (482)<br>879<br>presentation_slides 1.1% (294)<br>880<br>881 unknown 0.6% (145)<br>882<br>faq_document 0.1% (20)<br>883<br>884<br>0 2500 5000 7500 10000 12500 15000 17500<br>885<br>Count<br>886<br>887<br>Figure 6: Distribution of document types in the CoR-Dataset. The dataset is predominantly com-<br>888<br>posed of academic papers (55.3%) and government reports (20.0%), providing a rich source of<br>889 structured, information-dense content for training complex reasoning.<br>890<br>891<br>892<br>893 Distribution of Question Intent<br>894<br>895 factual_extraction 71.2% (18562)<br>896<br>897 definition_explanation 9.3% (2435)<br>898<br>comparative_analysis 5.6% (1468)<br>899<br>900 causal_inquiry 5.2% (1369)<br>901<br>902 procedural_instruction 2.9% (760)<br>903<br>summarization_request 2.6% (685)<br>904<br>905 existence_verification 2.0% (532)<br>906<br>907 conditional_inquiry 1.0% (264)<br>908<br>other 0.0% (8)<br>909<br>910 calculation 0.0% (4)<br>911<br>912 0 5000 10000 15000 20000<br>913 Count<br>914<br>915 Figure 7: Distribution of question intents. While factual extraction (71.2%) forms the core, the<br>916 dataset includes a significant proportion of questions requiring higher-level understanding, such as<br>917 definition/explanation (9.3%) and comparative analysis (5.6%).<br>Document Type<br>Question Intent<br>**----- End of picture text -----**<br>


17 

Under review as a conference paper at ICLR 2026 

**==> picture [432 x 641] intentionally omitted <==**

**----- Start of picture text -----**<br>
918<br>919 Distribution of Reasoning Complexity<br>920<br>921<br>922 direct_lookup 58.3% (15220)<br>923<br>924<br>925 information_synthesis 23.8% (6201)<br>926<br>927<br>928 logical_inference 7.1% (1855)<br>929<br>930<br>multi_step_reasoning 4.6% (1205)<br>931<br>932<br>933<br>structural_analysis 3.3% (862)<br>934<br>935<br>936 calculation 2.9% (744)<br>937<br>938<br>0 2500 5000 7500 10000 12500 15000 17500<br>939<br>Count<br>940<br>941<br>Figure 8: Distribution of reasoning complexity. A key feature of the dataset is that over 40% of ques-<br>942<br>tions require more than simple direct lookups, demanding skills like information synthesis (23.8%)<br>943 and multi-step reasoning (4.6%) to arrive at the correct answer.<br>944<br>945<br>946<br>947 Distribution of Answer Form<br>948<br>949<br>single_value_or_phrase 47.7% (12443)<br>950<br>951<br>952 synthesized_paragraph 30.1% (7856)<br>953<br>954 formatted_list 11.6% (3036)<br>955<br>956<br>negative_or_not_answerable 7.8% (2044)<br>957<br>958<br>959 direct_quote 2.4% (630)<br>960<br>961<br>boolean_yes_no 0.2% (57)<br>962<br>963<br>other 0.1% (21)<br>964<br>965<br>966 0 2000 4000 6000 8000 10000 12000 14000<br>967 Count<br>968<br>969 Figure 9: Distribution of expected answer forms. The dataset requires models to generate a variety<br>970 of output formats, from concise single phrases (47.7%) to comprehensive synthesized paragraphs<br>971 (30.1%), mirroring real-world application needs.<br>Reasoning Complexity<br>Answer Form<br>**----- End of picture text -----**<br>


18 

Under review as a conference paper at ICLR 2026 

|**972**<br>**973**<br>**974**<br>**975**<br>**976**<br>**977**<br>**978**<br>**979**<br>**980**<br>**981**<br>**982**<br>**983**<br>**984**<br>**985**<br>**986**<br>**987**<br>**988**<br>**989**<br>**990**<br>**991**<br>**992**<br>**993**<br>**994**<br>**995**<br>**996**<br>**997**<br>**998**<br>**999**<br>**1000**<br>**1001**<br>**1002**<br>**1003**<br>**1004**<br>**1005**<br>**1006**<br>**1007**<br>**1008**<br>**1009**<br>**1010**<br>**1011**<br>**1012**<br>**1013**<br>**1014**<br>**1015**<br>**1016**<br>**1017**<br>**1018**<br>**1019**<br>**1020**<br>**1021**<br>**1022**<br>**1023**<br>**1024**|Table 6: Datasets used for Stage 1 foundational fne-tuning. The total volume comprises over48 000<br>question-answer pairs, providing a solid foundation for the model.|
|---|---|
||**Dataset Component**<br>**Primary Source**<br>**Task Type**<br>**Size (Pairs)**<br>**Key Characteristics**|
||ChartQA (subset)<br>Open-source<br>ChartQA<br>5000<br>Short-form<br>question-<br>answering<br>pairs<br>fo-<br>cused on chart compre-<br>hension.<br>DocVQA (subset)<br>Public Benchmark<br>DocVQA<br>5349<br>Question-answering<br>on real-world scanned<br>documents with chal-<br>lenging OCR.<br>Paper+CC VQA Mix<br>Scholarly Papers, CC<br>Mixed VQA<br>2127<br>A<br>composite<br>dataset<br>blending<br>academic<br>paper content with web<br>data<br>from<br>Common<br>Crawl.<br>Curated DocQA Mix<br>Diverse Sources<br>Single-page QA<br>29 489<br>A large, diverse collec-<br>tion of QA pairs from<br>various<br>single-page<br>document types.<br>Visual QA (generic)<br>Public VQA Dataset<br>General VQA<br>6000<br>Standard open-domain<br>visual<br>question-<br>answering<br>pairs<br>to<br>bolster general visual<br>reasoning.|
||Table 7: Summary statistics of the CoR-Dataset. The dataset is intentionally skewed towards more<br>complex, multi-faceted categories to foster advanced reasoning capabilities.<br>**Dimension**<br>**Dominant Category**<br>**Count**<br>**Percentage**<br>**Document Type**<br>Academic Paper<br>14 417<br>55.3%<br>_(Top 3 total)_<br>21 634<br>82.9%<br>**Question Intent**<br>Factual Extraction<br>18 562<br>71.2%<br>_(Top 3 total)_<br>22 465<br>86.1%<br>**Reasoning Complexity**<br>Direct Lookup<br>15 220<br>58.3%<br>Information Synthesis<br>6201<br>23.8%<br>**Answer Form**<br>Single Value/Phrase<br>12 443<br>47.7%<br>Synthesized Paragraph<br>7856<br>30.1%<br>pushing models beyond simple lookups. Similarly, while direct lookups are common, over 40% of<br>the questions require more advanced cognitive skills like information synthesis and multi-step rea-<br>soning. This composition ensures that models trained on the CoR-Dataset are robustly prepared for<br>diverse and demanding document understanding scenarios.<br>A.9<br>DETAILEDCHAIN-OF-READINGAPPLICATIONEXAMPLES<br>This section provides a series of detailed, step-by-step examples to illustrate the practical application<br>and effectiveness of the Chain-of-Reading (CoR) framework. Each case breaks down a complex doc-<br>ument understanding task into the CoR stages, demonstrating how the model navigates challenges<br>like irregular layouts, repetitive content, and nested data structures. The document examples used in<br>this paper are sourced from MMlongbench-doc(Ma et al., 2024), LongDoc-URL(Deng et al., 2024),<br>and publicly available documents online intended for academic exchange.|



**1025** 

19 

Under review as a conference paper at ICLR 2026 

**1026** A.9.1 EXAMPLE 1: REASONING OVER IRREGULAR VISUAL LAYOUTS **1027** 

**1028** This first example demonstrates CoR’s capability to parse non-standard document layouts where **1029** data is presented in a visually associative manner rather than a conventional grid table. This requires **1030** the model to perform visuo-spatial reasoning to correctly link headers to data before performing any calculations. 

**1031** 

> **1032 Question:** _”What is the absolute difference in the Solar Heat Gain Coefficient between the ’STAN-_ 

> **1033** _DARD LOW E²’ and the ’ZO-E-SHIELD 5 EXTREME’ window?”_ 

**1034** 

## **1035 Generated CoR Trace:** 

**1036** 

- **1037 [Task Planning]** The user wants to calculate the difference in the Solar Heat Gain **1038** Coefficient (SHGC) for two products. I will first locate the chart, then visually **1039** associate the product names with their data columns, extract the two SHGC values, **1040** and finally compute the difference. 

   - **[Phased & Focused Search]** I have identified the performance chart. ”STANDARD LOW E²” is the first column; ”ZO-E-SHIELD 5 EXTREME” is the third column. Now locating the ”Solar Heat Gain Coefficient” row. 

## **1041** 

## **1042** 

## **1043** 

- **[Cross-modal Evidence Integration]** The value for ’Standard’ (Column 1) is 

## **1044** 

- 0.30. The value for ’Extreme’ (Column 3) is 0.19. 

## **1045** 

- **1046 [Grounded Reasoning & Verification]** The task is to calculate the absolute dif- **1047** ference: 0.30 - 0.19 = 0.11. The steps are verified, and the answer is grounded in the extracted values. 

**1048** 

## **1049** 

**Final Answer:** The absolute difference is **0.11** . 

**1050** 

**1051** 

**1052** 

**1053** 

**1054** 

**1055** 

**1056** 

**1057** 

**1058** 

**1059** 

**1060** 

**1061** 

**1062** 

**1063** 

**1064** 

**1065** 

**1066** 

**1067** 

**1068** 

**1069** 

**1070** 

**1071** 

**1072** 

**1073** 

**1074** 

**1075** 

**1076** 

**1077** 

**1078** 

**1079** 

20 

Under review as a conference paper at ICLR 2026 

**1080 1081 1082 1083** Energy SSavings. and ENERGYin ghgies STAR® RotQualified **1084** Leading Performance g aN **1085** With the majority of a window being made up of glass, picking the right **1086** glass is critical. Weather Shield offers a wide range of glass to meet the an performance that is right for you. Os, Oisat., So. Mat, **1087** STANDARD ZO-E-SHIELD 5 ZO-E-SHIELD 5 LOW E? EXTREME yhkits CPD# 084—A-313 **10881089** —_— —j — ——— TendteanenB(x)RS || 20-€5Model314" Glazing7205,022 (#2)Casement **1090** } | ENERGY PERFORMANCE RATINGS: **1091** |___SAAADDITIONALWere)PERFORMANCE RATINGS | Trnerenon sae **1092** Tinanr‘rendasterfeearee neterageranoxen ¥Cs SoD organ atantanestat **1093** ! Seta rawerne been te ae preren rorune **1094** an ! _ t oa [Waa oi scent REE TET.or sel ECE Weletivies toptemes —] Casement Total Unit Performance **1095** ee WeNFRC knowLABEL our AspireRATINGS Series windows **1096** 0.27 0.26 0.23 and patio doors are energy efficient. But Solar Heat Gain Coefficient* don't take our word for it. All Weather **10971098** Visible| 030Light Transmittance*| 020 019 VisiblelabeledShield products Transmittance,for U-Value, are rated,SolrAir LeakagecertifiedHeat Gain, andand 052 0.46 0.42 Condensation Resistance by the National **10991100** Phcetchentscntittss85% imino 95% 95% productsFenestrationBy using thecan informationRating Councileasily be compared on (NFRC). the label,within * TotalaccordanceUret caleulationswithNFRC are100.04,dorvedVahaes {ram computerinclude Argon senulabors gat that ae than verifiedby 3rd party testing the product line or against competitors. **1101 1102 1103** Real Warm-Edge® Spacer System ae **11041105** temperatures.WeathercondensationShield’s andThis Real frost improves Warm-Edge® build-up. the efficiencyStandard spacer on hasof the glass all no Weather metaland to Shield reduces transfer ; . - ==| **1106 1107** é =| **1108** METAL REAL = > SPACER WARM **1109 1110** _Pages 5 of 4068263.pdf (60 pages) in LongDocURL_ **1111** Hlee **Question: What is the difference in the solar heat gain coefficient between standard and** ee eee eeeeeEES 1 **1112** ' **extreme casement units?** 1 **! 1113** I' **Response: 11141115** '| <think>The user wants to know the difference between the solar heat gain coefficient for 'standard' 1! **1116** 'I and 'extreme' casement units.I need to find a table that compares these two types of units. I've located a table under the heading \"Casement Total Unit Performance\" on **page 4.** This !! **1117** ' table has columns for 'Standard', 'ZO-E-SHIELD 5', and 'ZO-E-SHIELD 5 EXTREME’. **11181119** '|'I It also has a row labeled 'Solar Heat Gain Coefficient*'. To find the answer, I need to extract the values from this row for both the 'Standard' and 'ZO-E-SHIELD 5 EXTREME' columns. The value for 'Standard' is **'0.30'** . The value for 'ZO-E-SHIELD 5 EXTREME' is **'0.19’** . 1!!! **1120** ' The question asks for the difference, so I will calculate: 0.30 - 0.19 = 0.11. **112111221123** '|'II </think><answer> **0.11** </answer> 1!!!1 **1124** 1SoUee’ **1125 1126** Figure 10: A product brochure snippet with a non-standard layout. **1127** ciate the product titles at the top with their corresponding data columns below. **1128 1129 1130** A.9.2 EXAMPLE 2: EVIDENCE LOCALIZATION IN REPETITIVE DOCUMENTS **1131 1132 1133** This case showcases CoR’s ability to navigate visually repetitive multi-page documents, a common 

Figure 10: A product brochure snippet with a non-standard layout. The model must visually associate the product titles at the top with their corresponding data columns below. 

This case showcases CoR’s ability to navigate visually repetitive multi-page documents, a common scenario in reports and surveys. The key challenge is to disambiguate between many similar-looking 

21 

Under review as a conference paper at ICLR 2026 

**1134** pages to pinpoint the precise evidence required by the user’s query, thereby avoiding hallucinations **1135** from incorrect context. **1136 1137 1138** Se ion 2020 NHS Staff Survey Results > Theme results > Trends > Safe environment - Bullying & harassment NHS] = =—_ = Centre England **1139 1140** : |I **1141** 8 **1142 1143 1144** ==a4 **1145 1146** 2 **1147** Il **1148** ~ ~ Best 3 **1149** Average 81 Worst 72 **1150 1151** = = | **1152** _Pages 12 of 4083860.pdf (69 pages) in LongDocURL_ **1153** oT See eee eee SS SSS SSS SS SSS SSS SSS SSS SSS SSS SSS SSS SSS SSS SSS 7S f N **1154** ! **Question: What is the highest score for 'Safe environment - Bullying & harassment** 1 **1155** !! **Response:** I **1156** ! <think> !I **1157** ! The user wants to know the "highest score" for the theme "Safe environment - Bullying & !I **1158** ! harassment". I need to find the table with this title. **1159** ! I have located it on **page 12** . I **1160** ! The table has four rows: **'Best', 'Your org', 'Average'** , and **'Worst’** . The question asks for the I ! highest score, which corresponds to the 'Best' row. I will extract the value from the 'Best' row. I **1161 1162** ! The value is **'8.7’.** I ! </think> I **1163** ! ! <answer> I **1164** ! **8.7 1165** !! </answer>" 1I **1166** ‘Nt **1167 1168 1169** Figure 11: A multi-page report where many pages share an identical visual template. The main **1170** challenge is to locate the specific page that matches the query’s context before data extraction. **1171 1172 Question:** _”What is the highest score for ’Safe environment - Bullying & harassment’?”_ **1173 1174 Generated CoR Trace: 1175 1176 [Task Planning]** The user wants the ”highest score” for a specific theme. I need to **1177** scan the document for the theme title ”Safe environment - Bullying & harassment” **1178** to find the correct page. 

**1178** 

**1179 [Phased & Focused Search]** I am scanning the document thumbnails and titles. I **1180** have located the correct chart on **page 12** . **1181 [Cross-modal Evidence Integration]** The question asks for the ”highest score.” **1182** In the table on page 12, this corresponds to the row labeled ”Best”. The value in this row is 8.7. **1183 1184 [Grounded Reasoning & Verification]** The evidence is located on the correct **1185** page and the semantic meaning of ”highest score” is correctly mapped to ”Best”. **1186** The extracted value is verified. **1187 Final Answer:** The highest score is **8.7** . 

22 

Under review as a conference paper at ICLR 2026 

## **1188** A.9.3 EXAMPLE 3: NAVIGATING COMPLEX NESTED TABLES 

**1189** 

**1190** This final example highlights CoR’s proficiency in parsing complex, nested tables commonly found **1191** in academic papers. The challenge lies not just in finding the correct table, but in navigating its **1192** hierarchical structure (Panels, Columns, Sub-columns, and Rows) to pinpoint a single data point **1193** with high precision. 

**1193** 

> **1194 Question:** _”What is the regression discontinuity (RD) effect of stimulus payments on consumer_ 

> **1195** _spending in bottom income quartile ZIP codes over the period April 1-April 30 (Table 4, Panel A,_ **1196** _Column 1)?”_ 

**1197** 

## **1198 Generated CoR Trace:** 

**1199** 

**1200 [Task Planning]** The user is asking for a very specific data point. I will decon- **1201** struct the query into a navigation path: Table 4 -¿ Panel A -¿ Column ”Bottom **1202** Income Quartile ZIP Codes” / Sub-column ”(1)” -¿ Row ”RD Effect of Stimulus:”. 

## **1203** 

**[Phased & Focused Search]** Navigating to Table 4, titled ”Causal Effect of Stimulus Payments...”. Focusing on ”Panel A: Impact... on Consumer spending”. 

## **1204** 

## **1205** 

**[Cross-modal Evidence Integration]** I have pinpointed the intersection of the specified row and column. The value at this exact location in the table is ”25.15”. 

## **1206** 

## **1207** 

**1208 [Grounded Reasoning & Verification]** The extracted value ”25.15” directly cor- **1209** responds to the multi-coordinate location specified in the query. The retrieval path is verified. 

## **1210** 

## **1211** 

**Final Answer:** The value is **25.15** . 

**1212** 

**1213** 

**1214 1215 1216 1217 1218 1219 1220 1221 1222 1223 1224 1225 1226** 

**1227 1228** 

**1229 1230 1231 1232** 

**1233 1234** 

**1235 1236 1237 1238 1239** 

**1240** 

**1241** 

23 

Under review as a conference paper at ICLR 2026 

|**1242**||
|---|---|
||Table 4]|
|**1243**|Causal Effect ofStimulus Payments on Spending and Small Business Revenue:|
|**1244**|Regression Discontinuity Estimates|
|**1245**|PanelA: Impact ofStimulusPaymentson ConsumerSpending|
|**1246**|Dep. Var.:<br>Change in Consumer Spending (%)<br>Bottom Income Quartile ZIP Codes<br>Top Income Quartile ZIP Codes|
|**1247**|Q)<br>(2)<br>(3)<br>(4)|
|**1248**<br>**1249**|RD Effect of<br>25.15<br>36.97<br>8.45<br>15.83<br>Stimulus:<br>(7.15)<br>(9.81)<br>(3.83)<br>(5.14)|
|**1250**|Window:<br>April 1 - April 30<br>April 7 - April 21<br>April 1 - April 30<br>April 7 - April 21|
|**1251**|Panel B: Impact ofStimulus Payments on SmallBusiness Revenue|
|**1252**|Dep. Var.:<br>Change in Small Business Revenue (%)|
|**1253**|Bottom Rent Quartile ZIP Codes<br>Top Rent Quartile ZIP Codes<br>Q)<br>(2)<br>(3)<br>(4)|
|**1254**||
|**1255**|RD Effect of<br>17.92<br>20.83<br>1.20<br>-T.54<br>Stimulus:<br>(9.59)<br>(16.76)<br>(6.27)<br>(10.45)|
|**1256**|Window:<br>April<br>1 - April 30<br>April 7 - April 21<br>April 1 - April 30<br>April 7 - April 21|
|**1257**||
||‘Notes: This table shows regression discontinuity estimates of changes in spending and business revenue around the date of stimulus|
|**1258**|payments on April 15, 2020. Panel A shows estimated effects of stimulus payments on consumer spending. To construct the estimates, we|
|**1259**|first express consumer spending on each day as a percentage change relative to mean daily consumer spending over the period January 4-31<br>in the corresponding calendar year. We then residualize these daily percentage changes with respect to day of week and first day of the|
|**1260**|month fixed effects. which we estimate using data from January<br>1. 2019, to May 10, 2019. We then compute OLS regressions of the|
||residualized outcome variable on an indicator variable for the date being on or after April 15 2020, using a linear control function before|
|**1261**|and after April 15, and excluding the partially treated date of April 14. The first row shows the coefficient on the indicator variable for the|
|**1262**|date being on or after April 15, which we refer to as the RD effect of stimulus; standard errors are reported in parentheses. The dependent<br>variable is scaled as a percentage change from January so that, for example, the first row of Column (1) indicates that stimulus payments|
|**1263**|increased consumer spending by 25.15 percentage points in bottom income quartile ZIP codes relative to the January 2020 level of|
||spending. In columns (1) and (2), we compute daily changes in spending restricted to cardholders in ZIP codes in the bottom quartile of the|
|**1264**|distribution ofZIP code median household income (based on data from the 2014-2018 ACS). Columns (3) and (4) replicate columns (1) and|
|**1265**|(2), computing daily changes in spending restricted to cardholders living in the top income quartile of ZIP codes. The coefficient and<br>standard error in columns (1) and<br>(3) of Panel A correspond to the specifications displayed in Figures 13b and<br>13c. Panel B shows|
|**1266**|regression discontinuity estimates for the effect ofstimulus payments on small business revenue using data from Womply. We first express|
||small business revenue on each day relative to mean daily small business revenue over the period January 4-31 of the corresponding year.|
|**1267**|We then residualize daily changes in small business revenue as in Panel A, and compute OLS regressions as in Panel A. Columns (1) and|
|**1268**|(2) restrict to ZIP codes in the bottom quartile of the distribution of ZIP code median rent for a two bedroom apartment (based on data<br>from the 2014-2018 ACS). Columns (3) and (4) restrict to businesses in the top rent quartile ZIP codes. The coefficient and standard error|
|**1269**|in columns (1) and (3) of Panel B correspond to the specifications displayed in Figures Ida and 14b. In both panels, columns (1) and (3)|
||include all of April 2020 in the regression specification, while columns (2) and (4) restrict to a narrower bandwidth, within one week on|
|**1270**|either side ofthe stimulus payment date. Data sources: Panel A: Affinity Solutions; Panel B: Womply.|
|**1271**|_Pages 70 of 4078345.pdf (109 pages) in LongDocURL_|
|**1272**|**Question: What is the regression discontinuity (RD) effect of stimulus payments on consumer**<br>ae<br>,<br>‘<br>U\|
|**1273**<br>**1274**<br>**1275**<br>**1276**<br>**1277**<br>**1278**<br>**1279**<br>**1280**<br>**1281**<br>**1282**<br>**1283**<br>**1284**|**spending in bottom income quartile ZIP codes over the period April 1-April 30 (Table 4, Panel A,**<br>**Column 1)?**<br> **Response:**<br><think><br>This is a precise data extraction question. I need to find the value at the intersection of a<br>specific row and column in Table 4.First, I'll deconstruct the question. The key components<br>are: "regression discontinuity (RD) effect," "stimulus payments," "consumer spending,"<br>"bottom income quartile ZIP codes," and "**Table 4, Panel A, Column 1**."Next, I'll narrate my<br>search. I will navigate to Table 4, which is titled "Causal Effect of Stimulus Payments on<br>Spending and Small Business Revenue: Regression Discontinuity Estimates." The question<br>specifies "Panel A: Impact of Stimulus Payments on Consumer Spending," so I will focus on<br>that panel . Now, I'll demonstrate precision. Within Panel A, I need to find the column<br>corresponding to "Bottom Income Quartile ZIP Codes" and the sub-column "(1)". Then, I must<br>locate the row labeled "RD Effect of Stimulus:".The value at the exact intersection of this row<br>I1<br>11<br>1I<br>1I<br>11<br>i}<br>**1**<br>1<br>11<br>11<br>11<br>1I<br>1I<br>1I<br>11<br>i}<br>1<br>i}<br>**1**<br>1<br>11<br>11<br>1I<br>1I<br>1I<br>11<br>11<br>i}<br>**1**<br>1|
|**1285**|and column is "25.15".<br>11<br>11|
|**1286**|</think><br>1I<br>1I|
|**1287**<br>**1288**<br>**1289**|<answer><br>**25.15**<br></answer><br>1I<br>1<br>1<br>i)<br>I<br>\d<br>N<br>4<br>.:<br>~<br>ee ee ee ee<br>ee<br>ee ee<br>ee<br>ee ee<br>eee|
|**1290**<br>**1291**<br>**1292**|Figure 12: An academic table with a complex, hierarchical structure (Panels and multi-level<br>columns). Precise extraction requires careful navigation of this structure.|
|**1293**||
|**1294**||
|**1295**||



24 

Under review as a conference paper at ICLR 2026 

## **1296** 

## A.9.4 EXAMPLE 4: MULTI-STEP CALCULATION WITH PRIOR KNOWLEDGE 

## **1297** 

**1298** This example demonstrates the model’s ability to solve complex questions that require not only **1299** precise data extraction but also the application of prior knowledge, such as a specific mathemat- **1300** ical formula. The CoR process shows how the model formulates a calculation plan, extracts the **1301** necessary data points, and then executes the multi-step reasoning to arrive at the final answer. 

**1301** 

**1302** 

> **1302 Question:** _”What was the percentage change in the Maltese lira against the euro between the aver-_ 

> **1303** _age for Q4 2004 and the average for Q4 2003?”_ 

**1304** 

## **1305 Generated CoR Trace:** 

**1306** 

- **1307 [Task Planning]** This is a multi-step calculation question. My plan is: 1. Find the **1308** exchange rates for the Maltese lira against the euro for Q4 2004 and Q4 2003. 2. **1309** Apply the percentage change formula: ((New Value - Old Value) / Old Value) * 100. 

**1310** 

- **[Phased & Focused Search]** I have located ”Table 5.3 MALTESE LIRA EXCHANGE RATES...”. I need to extract values from the ’EUR/MTL’ column for the rows ’Average for Q4 2004’ and ’Average for Q4 2003’. 

## **1311** 

## **1312** 

## **1313** 

- **1314 [Cross-modal Evidence Integration]** The value for ’Average for Q4 2004’ is **1315** 2.3181. The value for ’Average for Q4 2003’ is 2.3327. 

- **1316 [Grounded Reasoning & Verification]** Now, I will perform the calculation based **1317** on my plan: ((2.3181 - 2.3327) / 2.3327) * 100. This equals (-0.0146 / 2.3327) **1318** * 100, which is approximately -0.626%. Rounding to one decimal place gives -0.6%. The calculation is verified. 

**1319** 

**1320 Final Answer:** The percentage change was **-0.6%** . 

**1321** 

**1322** 

**1323** 

**1324** 

**1325** 

**1326** 

**1327** 

**1328** 

**1329** 

**1330** 

**1331** 

**1332** 

**1333** 

**1334** 

**1335** 

**1336** 

**1337** 

**1338** 

**1339** 

**1340** 

**1341** 

**1342** 

**1343** 

**1344** 

**1345** 

**1346** 

**1347** 

**1348** 

**1349** 

25 

Under review as a conference paper at ICLR 2026 

**1350 1351** 2008, net direct investment inflows dropped by **1352** L.m2.4reflectedmiliona smallerto 1.39.9amountmillon.of inter-companyThis essentially The Maltese lira. **1353** transactions, which include trade credits and During the fourth quarter of 2004. the Maltese lira **1354** loans between parents and subsidiarics. continued to depreciale against the cure, **1355** Moreover, net inflows of equity caprial, which had shedding | 39 (see Table 5.3). On the other hand, **1356** been boosted by heavy investment in one bank in the lira maintained its appreciation against the US. the fourth quarter of 20403, also declined. These dollar, gaining 8.5% as a resull of the continuous **1357** developments outweighed a rise in carmings retreat of the dollar on international foreign **1358** retained by foreign-owned firms operating in exchange markets. The lira also continued to **1359** Malia. Net flows related to financial derivatives, strengthen againsl sterling and the Japanese yen. **1360** which record instruments including options, appreciating by 1.5% and 0.7%, respectively. futures and swaps, also tumed negative. Throughout the first quarter of 2005, counter to **1361** the previous. trend, the lira rose marginally against **1362** Conversely, net portfolio outflows dropped the cura and depreciated against the dollar (sec **1363** sharply, from Lm 104.5 million te Lm 7.9 million. (Chart 45.2). At the same lime, the lira rose further **1364** Although the banking sector continued to invest apainel the yen, while it fell against sterling. **1365 1366** Table 5.3 **1367 1368** MALTESE LIRA EXCHANGE RATES AGAINST SELECTED **1369** CURRENCIES **1370** Period EUR/MTL USDYMTL GBP/MTL JPY/MTL **1371** Average forj4 2004 | 2.3181) 3.0053 1.6114 317.8 **1372** Average ford 2003 235327 2.7747 1.6269 302.0 **1373** % change 0.6 83 -1.0 5.2 **1374** Closing rate on 31.12.2004 243023 3.1393 1.6252 321.7 **1375** (Closing rate on 30.09.2004 24315 2.892] 1.o012 319.6 **1376** % change -l4 a4 13 0.7 **1377 1378 1379** at Comal Bank of Malia (ecrterlyRewirw MOS-7 **1380** _Pages 46 of 4064830.pdf (133 pages) in LongDocURL_ **1381 1382** i[°] **Question: What was the percentage change in the Maltese lira against the euro between** - \ nl \ **1383** i)1 **the average for Q4 2004 and the average for Q4 2003? 1 1384** 1 **Response:** 1 **1385** 1 <think> i] 1 This is a multi-step calculation question. First, I need to deconstruct it. I need to find the 1 **1386** i)1 exchange rate for the Maltese lira against the euro for two periods **: Q4 2004** and **Q4 2003.** II **1387** i)1 Second, I need to calculate the percentage change using the formula: ((New Value - Old **1 1388** 11 Value) / Old Value) * 100. I'll search for a table containing exchange rates. I've located 1 **1389** 11 \"Table 5.3 MALTESE LIRA EXCHANGE RATES AGAINST SELEcTcD CURRENCIES\". I need to be i] **1390** 11 precise and extract the values from the 'EUR/MTL' column. For the 'Average for Q4 2004' II **1391** i)1 row, the value is **2.3181** . For the 'Average for Q4 2003' row, the value is **2.3327** . Now I'll I 1 perform the calculation: ((2.3181 - 2.3327) / 2.3327) * 100. This simplifies to (-0.0146 / 1 **1392** 1 2.3327) * 100, which equals -0.6%. The answer should be rounded to one decimal place. 1 **1393** 1 </think> I **1394** i)1 <answer> II **1395** 1 **-0.6%** 1 i) ! **1396** \ </answer> v7 _ Uf **1397 1398** Figure 13: A financial data table listing currency exchange rates. Answering questions based on this **1399** table requires finding multiple values and performing calculations. **1400 1401 1402 1403** 

Figure 13: A financial data table listing currency exchange rates. Answering questions based on this table requires finding multiple values and performing calculations. 

26 

Under review as a conference paper at ICLR 2026 

**1404** A.9.5 EXAMPLE 5: CROSS-PAGE EVIDENCE SYNTHESIS FROM HETEROGENEOUS LAYOUTS **1405 1406** This case demonstrates a crucial capability for real-world document analysis: synthesizing infor- **1407** mation scattered across disparate, non-consecutive pages with heterogeneous layouts, a common **1408** feature of slide decks and reports. The model must independently locate two pieces of evidence **1409** from visually distinct contexts and then perform a comparison. **1410 1411** STATE OF MOBILE ADS IN VIETNAM BPEOIANEWS **1412 1413** Trendsin Vietnam Appota'sFacebook. fanpage “Android Store Viét Nam" became the biggest VN tech community on Instagram advertising **1414** wapengagel cafeixpensiv in Vietnam, adv **e** rtisingoninstagram bring more Total fans **1415** than other advertising platforms. weal oe % 21 64,246 **1416** - — —| 6=o— | 7} **1417 1418** “oa it E oe ile 15,681 © 125,000 **1419** Inc. **1420** _Pages 25, 31 of reportq32015-151009093138-lva1-app6891_95.pdf (34pages) in MMLongBench-Doc_ **14211422** /a **Question: As of Q3 2015, are there more active Instagram users in Vietnam, or are there** r ~~ay~~ ‘ **\ 1423** ~~I~~ ! **more fans for Appota's fan page? Directly answer 'Instagram' or 'Appota ‘.** 1 **1424** ! **Response:** 1 ! <think> **142514261427** !!! The user is asking a direct comparison question about the number of users for two entities: "Instagram" and "Appota's fan page" The document is a report titled "VIETNAM MOBILE APPS REPORT Q3 2015". This title is 111 **142814291430** !!! present on Slide 1, which is the time frame of the report.The question asks for a comparison as of Q3 2015. The document's title explicitly states this **first page** of the document. This confirms the context and 111 **1431** ! timeframe. I will search the document for the term "Instagram". **On Slide 28** , under the I **1432** ! section "Trends in Vietnam", there is a sub-section "Instagram advertising". The text within I **1433** ! this sub-section provides a specific figure: "With about 700,000 active users in Vietnam...". I **1434** ! This directly answers the part of the question concerning Instagram users.Next, I will search I **1435** ! the document for information related to "Appota's fan page". **On Slide 31** , under the I **1436 1437** !! section "APPOTA NEWS", there is a sub-section that discusses the fan page. The text reads: I ! "Appota's fanpage “Android Store Việt Nam” became the biggest VN tech community on I **1438** ! Facebook." This identifies the fan page. The slide also contains a table with the heading I **1439** ! "Total fans" and the value "1,134,462". This is the number of total fans for the fan page. 1 **1440** ! Now, I will compare the two figures found in the document: 1 **1441** ! ! *Instagram users: 700,000 (Slide 28) 1 **1442** ! *Appota's fan page total fans: 1,134,462 (Slide 31) 1 **1443** ! Comparing these two numbers, 1,134,462 is greater than 700,000. Therefore, the number 1 **1444** ! ! of fans for Appota's fan page is greater than the number of active Instagram users in 1 **1445** ! **1446** ! Vietnam as of Q3 2015.n9.  The question asks me to directly answer 'Instagram' or 'Appota'. 1 **1447** ! Based on my analysis, the correct response is 'Appota’. 1 **14481449** !!! </think><answer> 1I **1450** \ **Appota** ! \ </answer> 7 **1451** a **1452 1453** Figure 14: Evidence scattered across two different pages (Slide 28 and Slide 31) from a presentation. **1454** The pages have completely different visual styles, requiring the model to perform robust, context- **1455** independent searches. **1456 1457 Question:** _”As of Q3 2015, are there more active Instagram users in Vietnam, or are there more fans for Appota’s fan page? Directly answer ’Instagram’ or ’Appota’.”_ 

27 

Under review as a conference paper at ICLR 2026 

## **1458 Generated CoR Trace: 1459** 

> **1460 [Task Planning]** This is a comparison question requiring two data points from 

> **1461** different parts of the document. My plan is: 1. Search for the number of ”Insta- **1462** gram users in Vietnam”. 2. Search for the number of fans for ”Appota’s fan page”. **1463** 3. Compare the two numbers and provide a direct answer as requested. 

**1464 [Phased & Focused Search]** I am searching the document for ”Instagram”. On **1465** Slide 28, I found the text ”With about 700,000 active users in Vietnam...”. Next, **1466** I am searching for ”Appota’s fan page”. On Slide 31, I found a table with ”Total fans” and the value ”1,134,462”. 

## **1467** 

**1468 [Cross-modal Evidence Integration]** - Evidence 1 (Slide 28): Instagram active **1469** users = 700,000. - Evidence 2 (Slide 31): Appota’s fan page fans = 1,134,462. **1470 [Grounded Reasoning & Verification]** Now I will compare the two values. **1471** 1,134,462 is greater than 700,000. Therefore, Appota has more fans. The question **1472** requires a direct answer of ’Instagram’ or ’Appota’. Based on my reasoning, the correct answer is ’Appota’. 

**1473** 

> **1474 Final Answer: Appota 1475 1476 1477 1478 1479 1480 1481 1482 1483 1484 1485 1486 1487 1488 1489 1490 1491 1492 1493 1494 1495 1496 1497 1498 1499 1500 1501 1502 1503 1504** 

**1505 1506 1507 1508 1509 1510 1511** 

28 

Under review as a conference paper at ICLR 2026 

**1512** A.9.6 EXAMPLE 6: LOGICAL REASONING ON VISUALLY ENCODED INFORMATION **1513 1514** This final case study demonstrates a sophisticated combination of skills: understanding implicit **1515** information from visual layouts, performing robust multilingual OCR, and executing logical set **1516** operations. The model must infer rankings from a visual list, correctly read non-English text (in **1517** this case, Vietnamese), and then perform a ”set difference” comparison across two pages to find the **1518** unique item. **1519 1520 1521** TOP VIETNAMESE iOS APPS ON APPOTA PLATFORM TOP VIETNAMESE ANDROID APPS ON APPOTA PLATFORM **1522 1523** YN **1524 1525** QOQoOse SOWi **15261527** Gn ae it4 BN ~< Vo* onmeé > aA il= Vo* Ee3é“A **1528** hinh B Lién minh Hiép Khach Vua béng MobiT F r i@p Kt € a J MobiTiv wse **1529 1530** _Pages 29, 30 of reportq32015-151009093138-lva1-app6891_95.pdf (34pages) in MMLongBench-Doc_ **1531** rrr errr rrr rr rr rr ere rrr rr rr er rr rr rr er rr rr re rr er rr rrr er err rrr ee ~ **1532 Question: Which APP on APPOTA platform is top 10 Vietnam Android App, but not top 10** |/ **Vietnam iOS App?** ‘\\1 **1533** I **Response:** 1 **1534** i} I 1 <think> 1 **1535** i} i} I The user is asking for an app that is in the top 10 on the "APPOTA platform" for "Vietnam i} **1536** I Android App" but not for "Vietnam iOS App", which requires performing a logical set i} **1537** i}I difference operation. My process begins by locating the two required charts within the I1 **1538** i}1 document. I1 **1539154015411542** i}i}i}I1II I have found the "TOP VIETNAMESE iOS APPS ON APPOTA PLATFORM" chart on the "TOP VIETNAMESE ANDROID APPS ON APPOTA PLATFORM" chart onchallenge is that the rankings are not numbered but are implicitly conveyed by the top-to-bottom visual order, which I must correctly interpret. **page 30page 29** . A key  and i}i}i}IIi} **1543** 1 I i} After performing multilingual OCR, I have extracted the full iOS Top 10 list as **[Zing mp3, Tiếu** i} **1544** i}I **Ngạo Giang Hồ, NCT, I am Naruto, Đồ sát mobile, Chinh Đồ Mobile, Liên minh huyền thoại,** i}1 **1545** i}i} **Hiệp Khách, Vua bóng đá, MobiTivi]** and the full Android Top 10 list as **[Tiếu Ngạo Giang Hồ,** i}I **1546** i}I **Zing Mp3, Đồ sát mobile, Chinh Đồ Mobile, NCT, I am Naruto, Liên minh huyền thoại, Hiệp** 1i} **1547** i}1 **Khách, MobiTivi, UC Browser Tiếng Việt]** . Now, I will execute the planned set difference by Ii} **1548** i}I systematically comparing the Android list against the iOS list. My verification confirms that i} **1549** II nine of the applications appear in both lists. The only app that is present in the Android list i}i} **1550** I but is absent from the iOS list is "UC Browser Tiếng Việt“ 1 i} i} **1551** 1 Therefore, this is the unique and correct answer that satisfies all conditions of the user's I i} i} **1552** i} query. i} **1553** i}1 </think> 1i} I i} **1554** I <answer> I **1555** i}I **UC Browser Tiếng Việt** i} **1556** \‘\ N </answer> 7 4 1 SLL LLL LLL LLL LLL LL LLL LLL ee eee ee ee ee eee ee ee ee eee ee eee ee **1557 1558** Figure 15: Two top-app charts from consecutive slides (29 and 30). The rankings are implicitly en- **1559** coded by vertical position, and the app names are in Vietnamese, requiring advanced visual parsing **1560** and multilingual OCR capabilities. **1561 1562 Question:** _”Which APP on APPOTA platform is top 10 Vietnam Android App, but not top 10 Vietnam_ **1563** _iOS App?”_ **1564** 

**1565** 

## **Generated CoR Trace:** 

29 

Under review as a conference paper at ICLR 2026 

**1566 [Task Planning]** The user is asking for an app that meets two criteria: it is in the **1567** Android top 10, but not in the iOS top 10. This requires a set difference operation. **1568** My plan is to: 1. Locate and extract the list of top 10 Android apps. 2. Locate and **1569** extract the list of top 10 iOS apps. 3. Compare the two lists to find the app present **1570** only in the Android list. **1571 [Phased & Focused Search]** I have located the ”TOP VIETNAMESE iOS APPS **1572** ON APPOTA PLATFORM” chart on page 29. On page 30, I found the ”TOP **1573** VIETNAMESE ANDROID APPS ON APPOTA PLATFORM” chart. The rank- **1574** ings are presented visually, from top to bottom. **1575 [Cross-modal Evidence Integration]** The model extracts the two lists of app **1576** names from the charts. The first list contains the top 10 iOS apps, and the second **1577** list contains the top 10 Android apps, including one app not present in the first list. **1578 1579 [Grounded Reasoning & Verification]** Now I will perform the set difference **1580** operation: (Android List) - (iOS List). By systematically comparing the two ex- **1581** tracted lists, I have determined that a specific browser app is present in the Android list but is absent from the iOS list. All other apps in the Android top 10 are also **1582** present in the iOS list. Therefore, this is the unique app that satisfies the user’s **1583** condition. **1584 1585 Final Answer: UC Browser Tieng Viet** 

**1586 1587** A.9.7 EXAMPLE 7: JOINT REASONING OVER TEXT AND UNCAPTIONED IMAGES **1588** This example demonstrates the model’s ability to perform joint reasoning by synthesizing informa- **1589** tion from both surrounding text and an uncaptioned, illustrative image. To answer the question, **1590** the model must first locate a relevant text passage and then use the context provided by that text to **1591** correctly interpret the visual information in the adjacent figure. **1592 1593 Question:** _”What is the color on the watch that shows the aerobic zone?”_ **1594 1595 Generated CoR Trace: 1596 1597 [Task Planning]** The user is asking for the color corresponding to the ”aerobic zone”. My plan is to search the document for the term ”aerobic zone” and analyze **1598** the surrounding text and any associated images for color information. **1599 [Phased & Focused Search]** I have searched the document and located the term **1600** 

**[Phased & Focused Search]** I have searched the document and located the term **1600** ”aerobic” on page 19, within section ”3.1 Monitoring your heart rate”. The text **1601** nearby mentions ”Your watch will display different colors when your heart rate **1602** reaches corresponding zones...” and points to an example figure on the same page **1603** showing an outdoor run. 

**1604** 

**[Cross-modal Evidence Integration]** - **Textual Evidence (Page 19):** Confirms **1605** that different zones are represented by different colors and directs attention to the **1606** figure. - **Visual Evidence (Page 19):** The figure, though uncaptioned, shows a **1607** watch interface. I can visually identify a section highlighted in yellow with the **1608** text label ”Aerobic” directly next to it. 

> **1609 [Grounded Reasoning & Verification]** By combining the textual clue (zones **1610** have colors) with the direct visual evidence (the label ”Aerobic” is next to the **1611** color yellow), I can confidently conclude that the color for the aerobic zone is **1612** yellow. The reasoning is grounded in this direct text-to-image link. 

**1613 Final Answer: Yellow 1614 1615** A.9.8 EXAMPLE 8: LINGUISTIC AND VISUAL REASONING ON SCANNED ARCHIVES **1616 1617** This example showcases the model’s robustness and deep reasoning capabilities when dealing with **1618** low-quality, historical documents. The task requires accurate localization within the document, **1619** advanced OCR on degraded, archaic fonts, and a nuanced linguistic understanding to differentiate between singular and plural nouns in the figure captions to arrive at a correct count. 

30 

Under review as a conference paper at ICLR 2026 

**1620** Setting the heart rate zone calculation method **1621** The heart rate interval can be calculated based on the maximum heart rate **1622** percentage or HRR percentage. To set the heart rate interval calculation method, open the Huawei Health app, go to Me > Settings > Heart rate limit and zones **1623** and set Calculation method to either Maximum heart rate percentage or HRR percentage. **1624 1625** (1) NOTE « If you select Maximum heart rate percentage as the calculation method, the heart **1626** rate zone for different types of workout activities (Extreme, Anaerobic, Aerobic, Fatburning, and Warm-up) is calculated based on your maximum heart rate ("220 - your **1627** age" by default). Heart rate = Maximum heart rate x Maximum heart rate percentage. e If you select HRR percentage as the calculation method, the heart rate interval for **1628** different types of workout activities (Advanced anaerobic, Basic anaerobic, Lactic acid, Advanced aerobic, and Basic aerobic) is calculated based on your heart rate reserve **1629** (HRmax - HRrest). Heart rate = Heart rate reserve x Heart rate reserve percentage + Resting heart rate. **1630** e The heart rate zone calculation methods while you are running are not affected by the settings in the Huawei Health app. For most running courses, HRR percentage is **1631** selected by default. **1632** Your watch will display different colors when your heart rate reaches corresponding zones during a workout. The following figure shows how heart rate **1633** is displayed during an outdoor run. **1634** re| **1635 1636** c r **1637** 630" 4.03. **1638** ce(/kr **1639** 00:25:30. ° **1640** 07:16 **1641 1642** Measuring your heart rate during a workout **1643** 1. After you start a workout, swipe on the watch screen to check your real-time heart rate and heart rate zone. **1644** 2. After completing your workout, you can check your average heart rate, **1645** maximum heart rate, and heart rate zone on the workout results screen. **1646** Issue 01 (2020-09-07) Copyright © Huawei Technologies Co., Ltd. — **1647** _Pages 23 of honor_watch_gs_pro.pdf (42pages) in MMLongBench-Doc_ **1648** POT TTT TTT eee nnn mm ee ee ee em me **1649** ~~I~~ 17 **Question: What is the color on the watch that shows the aerobic zone?Response: \** ! **1650** 1I <think> ! I **1651** I The user is asking for the specific color of the "aerobic zone" on the watch. I will search the ! I **1652** I entire document for the term "aerobic zone". ! **1653** II On **page 19** , under the section “3.1 Monitoring your heart rate", there is a sub-section titled[l] **1654** II "Setting the heart rate zone calculation method“ . Within this section, the text states: "Your[!] I **1655165616571658** 11IIIIII watch will display different colors when your heart rate reaches corresponding zones during a workout."It then provides an example image: "The following figure shows how heart rate is displayed during an outdoor run." This image is on the same page (19). The image explicitly labels the "Aerobic" zone with a yellow color. The document provides a direct visual example of the !!!!! **1659** I color associated with the aerobic zone. Therefore, the information is present and can be I[!] **1660** I extracted directly from the image.The question asks for the color, which is a factual detail. " I The document provides this detail through its figures and text. The user's question is[!] **1661** 1I answerable.n9.  The answer is "Yellow" ! **1662** II </think> ! **1663** I <answer> ! I **1664** I **Yellow** ! I **1665** \ </answer>[1] \ / **1666** So LLL LLL LLL LLL LLL LLL LL LLL LLL LLL LLL LL LLL LL LLL LL LLL” **1667 1668** Figure 16: An excerpt from a user manual on page 19. The answer is not in the text alone but requires linking the description of heart rate zones to the colors shown in the unlabeled example **1669** image. **1670 1671 1672** 

**1673** 

**Question:** _”Among figure 1-4, how many figures show more than one breccia gash?”_ 

31 

Under review as a conference paper at ICLR 2026 

## **1674 Generated CoR Trace: 1675** 

> **1676 [Task Planning]** The user wants me to count how many of the first four figures 

> **1677** depict more than one ”breccia gash”. My plan is to locate the specified figures, **1678** carefully read the caption for each one, and use linguistic cues (singular vs. plural) **1679** to determine the count for each figure before summing the result. 

**1680 [Phased & Focused Search]** I have located the relevant content on page 3 of the **1681** document. This page contains four diagrams labeled Fig. 1, Fig. 2, Fig. 3, and **1682** Fig. 4, each with a descriptive caption below it. 

**1682** 

**1683 [Cross-modal Evidence Integration]** I will analyze each figure’s caption: - **Fig. 1684 1:** The caption reads ”Diagram of Breccia-Gash...” (singular). This figure shows **1685** one gash. - **Fig. 2:** The caption reads ”Diagram of Breccia Gashes...” (plural). **1686** This figure shows more than one gash. - **Fig. 3:** The caption reads ”Diagram of narrow Breccia Gashes” (plural). This figure shows more than one gash. - **Fig. 4: 1687** The caption reads ”Diagram of Breccia-Gash...” (singular). This figure shows one **1688** gash. 

**1689** 

**[Grounded Reasoning & Verification]** Based on the linguistic analysis of the **1690** captions, Figures 2 and 3 show ”more than one breccia gash”. Figures 1 and 4 **1691** show only one. Therefore, the total count of figures showing more than one gash **1692** is two. The reasoning is grounded in the grammatical distinction found in the **1693** scanned text. **1694** 

**1695 Final Answer: 2 1696 1697 1698 1699 1700 1701 1702 1703 1704 1705 1706 1707 1708 1709 1710 1711 1712 1713 1714** 

**1715 1716 1717 1718** 

**1719 1720** 

**1721 1722** 

**1723 1724 1725** 

**1726** 

**1727** 

32 

Under review as a conference paper at ICLR 2026 

**1728 1729** 16 illustrateProt Lebours paper On theBreccia. Gushes ofDurham? **1730 1731 1732** = a —_———EEO"PEED PEARED - **1733** | fas? Say \ “eee AF **1734 17351736** _.fea_—y—4¥—_—~—~ **1737 1738** withDiagramtop deniuiled of Breccia offGashuni Bottomin Caty DiagramderuuiedofofBrecciabut bottomGashes shownwithitrtap **1739 1740 1741 1742** ne _ FTA oF= |= 4 es__ **1743** _ . U4. Spe eee 4 oe —o iS ee a **1744** eT/Z poeenelLi PROU5 ee eeee 2:z } kOeii **1745** ANCLYP acme { Pa bt **1746** __— $+ i 7 **1747 1748** 8. Intheabove sketches the Cementing matter ix represented. **1749 1750** Ptoceetenas0°ET of MAM E T8838 SA A Rea Memande _Pages 3 of fd76bbefe469561966e5387aa709c482.pdf (17pages) in MMLongBench-Doc_ **1751 ‘** PTT TTT TT TT em mm me mmm mm NS N **1752 Question: Among figure 1-4, how many figures show more than one breccia gash? 1753** |I **Response:** 1 I <think> I **1754** | i The user's question asks to count the figures that depict more than one breccia gash among f **1755** || figures 1-4. I will examine the provided document images to locate the figures. I **1756** | **Page 3** contains the figures. The caption for this page reads: "Vol. XXXIII PLATE XII. To I **1757** | illustrate Prof. Lebour's paper 'On the Breccia Gashes of Durham'." The figures are numbered I **1758** I as follows: *** Fig. 1: "Diagram of Breccia Gash in CLIP with top denuded off and Bottom** I I **1759** I **concealed by the beach." * Fig. 2: "Diagram of Breccia Gashes with top denuded off but** ! **1760** I| **bottom shown in cliff." * Fig. 3: "Diagram of narrow Breccia Gashes." * Fig. 4: "Diagram of** f **1761** i| **Breccia Gash showing process of formation."** I **1762** || The question specifies "figure 1-4". This means I will analyze figures 1, 2, 3, and 4. Now, I will I **17631764** ||I examine each figure to determine if it shows more than one breccia gash. * Figure 1: This figure shows a single diagram of a "Breccia Gash". * Figure 2: This figure shows a diagram of "Breccia Gashes". The word "Gashes" implies multiple gashes are depicted. The caption II **1765** II explicitly says "...but bottom shown in cliff." This confirms it shows more than one gash. * I **1766** I| Figure 3: This figure shows diagrams of "narrow Breccia Gashes". The word "narrow" does not I **1767** i|i change the fact that they are multiple gashes. The caption explicitly says "...of narrow Breccia Gashes." This confirms it shows more than one gash. * Figure 4: This figure shows a diagram II **1768** | of a "Breccia Gash". I **1769** || It does not show more than one gash. Based on this analysis, figures 2 and 3 show more than I **1770** I one breccia gash. The question asks for the total count. Therefore, the answer is 2. I I **1771** I </think> 1 I **1772** | <answer> fl **1773** i\ **2** \s </answer> ¢ **1774** So LLL LLL LLL LLL LLL LLL LLL Lee eee ee eee ee eee eee eee ee eee eee a **1775 1776** Figure 17: A low-quality scan from a historical scientific paper, identified by the model as page 3. To answer the question, the model must accurately read the captions for Figures 1-4 and interpret **1777** the linguistic cues within them. **1778 1779 1780** 

**1781** 

33 

Under review as a conference paper at ICLR 2026 

## **1782** 

## A.9.9 EXAMPLE 9: MULTI-STEP REASONING ACROSS HETEROGENEOUS PAGES 

## **1783** 

**1784** This case study demonstrates the model’s ability to solve complex, multi-step questions by synthe- **1785** sizing clues from disparate sections of a document. The process requires decomposing the initial **1786** query, locating evidence across heterogeneous page types (a table and a chart), interpreting graphical **1787** data, and performing a final calculation. 

**1787** 

**1788** 

**Question:** _”In the year in which Palestine was added to the survey, respondents who believe traditional values are outdated increased by how many percentage points compared to 2011?”_ 

## **1789** 

## **1790** 

## **1791 Generated CoR Trace:** 

## **1792** 

> **1793 [Task Planning]** The user’s question requires a multi-step process. First, I need **1794** to identify the year Palestine was added to the survey. Second, I must find the **1795** percentage point data for that year and for 2011. Finally, I will calculate the difference. 

**1796** 

**1797 [Phased & Focused Search]** I am searching for ”Palestine”. On page 6, a table **1798** lists ”Palestine” under the heading ”New in 2014”. This establishes the target **1799** year. Next, I am locating the data on values. On page 10, a bar chart provides **1800** the percentage of respondents who believe ”Traditional values are outdated” for various years. 

## **1801** 

- **1802 [Cross-modal Evidence Integration]** - **Evidence 1 (from Table, Page 6):** The year Palestine was added is 2014. - **Evidence 2 (from Chart, Page 10):** The 

- **1803** value for 2011 is 17%. - **Evidence 3 (from Chart, Page 10):** The value for 2014 

- **1804** is 46%. 

**1805** 

**[Grounded Reasoning & Verification]** The question asks for the increase in per- **1806** centage points. The calculation is a simple subtraction: 46% (the value for 2014) **1807** minus 17% (the value for 2011). The result is 29 percentage points. The reasoning **1808** is verified, as I correctly linked the country to the year and then used that year to **1809** extract the correct data for the final calculation. 

**1810** 

**1811 Final Answer: 29 1812 1813 1814 1815 1816** 

**1817** 

**1818** 

**1819 1820** 

**1821** 

**1822** 

**1823** 

**1824** 

**1825** 

**1826** 

**1827** 

**1828** 

**1829** 

**1830** 

**1831** 

**1832** 

**1833** 

**1834** 

**1835** 

34 

Under review as a conference paper at ICLR 2026 

**1836 1837 1838** THE STUDY WAS COMMISSIONED IN DECEMBER 2013. **1839** THETO JANUARYINTERVIEWS2014.TOOKSAMPLEPLACEDETAILSFROMAREDECEMBERGIVEN BELOW:2013 Newin New in New in New in 2011 2012 2013 2014 **18401841** Seiginal Aili 1D.cosarivion **1842** 40%‘AbuDhabDeol ‘0%Muscat Ochao% Manama100% Riyadh40% 20%Kuala 50%Garo Amman50% etul60% Abit(25%) ‘Stax(25%) Benghazi(25%) Oran(25%) Fes(25%) (25%)AlHudaydsn = Gaze(50%) aon oeana onRenan donJeaisn San—AlHowall Acordia2m 2%bid aoeaida Suesh — Misia Gonstarine | abet Wie **1843** Pal gon domo oe (28%) (25%) (25%) (25%) (25%)Marrakech(25%) (25%) **1844** Farwanyason **1845** i **1846** aia ans eo @ ‘ASDICA Sinan thal **1847** VALUES AND BELIEFS VALUES AND BELIEFS BY COUNTRY **1848** Which of the following is closest to your view? Il [bbeTraditionalpreserved valuesfor meangeneraions2 lot tome,to Come.and ought [il Tradiional{fom Keen tovaluesemtrace are outdatedmodem valuesand belongand bela inthe past **1849** IB toTraditional be preservedvaluesSor meanpeneraonsa bot to me,9 coma.and ought II TinTraditionalkaon valuesto brace are outdatedmeer andvetoes belongand balesin the past 2014 **1850 1851 1852 1853** 40 48 as 0 of sf a sof ccf os **1854 1855** _Pages 5, 6, 10, 11 of asdaaburson-marstellerarabyouthsurvey2014-140407100615-phpapp01_95.pdf (42pages) in MMLongBench-Doc_ **1856** 4[<] eee ‘ \ **1857** ~~f~~ ‘ **Question: In the year in which Palestine was added to the survey, respondents who believe** \ ~~!~~ **traditional values are outdated increased by how many percentage points compared to** ! **1858** ' **2011?** ' **1859** | ' **Response:** l **1860** ' <think> l **1861** ! Identify the Year of Addition: The question asks about the year in which "Palestine was added I **1862** ! to the survey". I will search the document for this information. . l **1863** ' **On page 6** , there is a table that lists countries "New in 2011", "New in 2012", etc. This table ' **1864** ' explicitly categorizes countries based on their inclusion in the survey. In the last column of this ' ! table, under the heading "New in 2014", the country listed is "Palestine". This directly **1865** 'l answers the first part of the question: the year is 2014. Locate Data for 2011 and 2014: The **1866** ' question also requires the percentage point change between 2011 and 2014. I will examine l **1867** ' the "VALUES AND BELIEFS" section, which contains data for these years. **On page 10** , the l **1868** ! chart shows the percentage of respondents who believe "Traditional values are outdated and I **1869** ' belong in the past" for each year. For the year 2011, the value is 17%. For the year 2014, the ' **1870** '' value is 46%. Calculate the Difference: The question asks for the increase in percentage points. ' | This is a simple subtraction: 46% (2014 value) - 17% (2011 value). The calculation is 46 - 17 = 1 **1871** ' 29. l **1872** ' Verify Context: To ensure the answer is correct, I will verify the context. The table on page 6 l **1873** ! confirms that "Palestine" was a new country in 2014. The "VALUES AND BELIEFS BY I **1874** !! COUNTRY" chart on page 11 shows the percentage of respondents in "Among All" who hold l **1875** ' the "Traditional values are outdated" view. The data for "Among All" is the sum of the l **1876** ' percentages from all countries. The data for "Among All" in 2011 is 17% (17% of the total ' **187718781879** ''''! respondents held this view), and the data for "Among All" in 2014 is 46% (46% of the total respondents held this view). The difference between 46% and 17% is indeed 29 percentage points. “.Final Check: The question asks for the increase in percentage points. The calculation is 46 - 17 lll **1880** ! = 29. The document provides the necessary data points (17% for 2011 and 46% for 2014) and l **1881** ' the table on page 6 confirms that "Palestine" was a new country in 2014. Therefore, the ' **1882** '' calculation is correct and the answer is 29. ' **18831884** |'''' </think><answer> **29** fl! **1885** \N </answer> ¢ ’ **1886** seo LLL” **1887** Figure 18: Evidence required to solve the query is scattered across two different pages. The model **1888** must first use the table on page 6 to identify the target year (2014), and then use that information to **1889** extract relevant data from the bar chart on page 10. 

35 

Under review as a conference paper at ICLR 2026 

**1890** A.9.10 EXAMPLE 10: ROBUSTNESS ON A HIGH-DIFFICULTY, ERROR-PRONE TASK **1891 1892** This case study demonstrates the model’s robustness and high precision on a complex, multi- **1893** constraint query. The inherent difficulty of this task is underscored by the fact that the benchmark’s **1894** own annotations for the evidence pages were found to be incorrect, likely due to the challenge of **1895** pinpointing specific data across multiple, similarly formatted tables. Despite this, our model suc- **1896** cessfully navigated the document and independently located the correct evidence on pages 15 and 16. This success on an error-prone task highlights the model’s superior ability to deconstruct com- **1897** plex questions and maintain reasoning accuracy, a key advantage of the CoR framework. 

**1898** 

**1899 Question:** _”What is the percentage gap between the percentage of the male 65+ age group who use_ **1900** _the internet or email, and the percentage of that group who have broadband at home, according to the Pew Research Center’s Internet Project July 18-September 30, 2013 tracking survey?”_ 

**1901** 

> **1902 Generated CoR Trace:** 

## **1903** 

- **1904 [Task Planning]** The user is asking for a ”percentage gap”, which implies sub- **1905** traction. I need to find two specific percentages for the ”male 65+” group from a **1906** specific survey: their internet usage and their broadband adoption rate. My plan is to locate both values and then calculate the difference. 

**1907** 

- **1908 [Phased & Focused Search]** I am searching for tables related to the ”Pew Re- **1909** search Center... 2013 tracking survey”. I have located a table for ”Internet usage for older adults” on page 15, and another for ”Broadband adoption for older 

- **1910** adults” on page 16. In both tables, I will find the row for ”Male”. 

**1911** 

- **[Cross-modal Evidence Integration]** - **Evidence 1 (from Table, Page 15):** In 

- **1912** the ”Internet usage” table, the value for ”Male (n=612)” is 65(%). - **Evidence 2** 

- **1913 (from Table, Page 16):** In the ”Broadband adoption” table, the value for ”Male 

- **1914** (n=612)” is 53(%). 

**1915** 

- **[Grounded Reasoning & Verification]** The task is to calculate the percentage 

- **1916** gap. The calculation is 65% (internet usage) - 53% (broadband adoption). The **1917** result is 12 percentage points. The model’s ability to independently locate the **1918** correct evidence pages (15 and 16) confirms its robust and accurate understanding **1919** of the document’s content and structure. **1920** 

## **1921 Final Answer: 12 1922** 

**1923 1924** 

**1925** 

**1926** 

**1927** 

**1928** 

**1929** 

**1930** 

**1931** 

**1932** 

**1933** 

**1934** 

**1935** 

**1936** 

**1937** 

**1938** 

**1939** 

**1940** 

**1941** 

**1942** 

**1943** 

36 

Under review as a conference paper at ICLR 2026 

**1944 1945 1946 1947** Appendix—Detailed Demographic Tables Broadband adoption for older adults **19481949** nsInternet« sp thoseTotal fortranusage all 65+aterfor(n= who olderuse the adultsinternet ovement 59% % of thoseGenderTotalMale for65 andall(n=61265+older with(n'=2.52 broadbandat home C=]47% **1950** romaie 2 = age **1951** — 531 14 656970-74 (n=4014 6555 **1952** 70-7475-79 (n=401(n=264) 68a7 Lait7B-70 (ne24ata “at **1953** High80+—s schoo!(n= grador less (n=596 4037 SomeHighEducationschoolcollege grad(738%or less (n= 5727 **1954** Some college 69 College graduate (1'~ 76 College graduate 87 Household Income **1955** Household Income $30,000 (n=4 25 $30,000 (n= 39 $30,000-$49,999 («282 SL **1956** $30,000-$49,999 («282 6 $50,000-$74,999 (n=192 73 $50,000-$74,999 (n= 192 86 $75,000¢ (n=274 82 **1957** $75,000¢ (n=27 90 ‘Community Type ‘Community Type Urban 1 48 **1958** {ten Gn bed Suburban 50: senmten bad Rural (n= 355 40 **1959** Rural 50 **1960 1961** pew ReSeARcH CENTER PEW RESEARCH CENTER **1962** _Pages 15, 16 of PIP_Seniors-and-Tech-Use_040314.pdf (27pages) in MMLongBench-Doc_ **1963** ¢ ST TTT SS SS SS SE SS SS SS SS SS SS SS SS SS SS SS SS SS SS SS SS SSS SS SSS Ss A ‘ **\ 1964** / I **Question: What is the percentage gap between male 65+ age group who use internet and**[\] **1965** I1 **broadband at home in the Pew Research Center’s Internet Project July 18-September 30,** i} **1966** II **2013 tracking survey? Please write the answer into float format.** i} **1967** II **Response:** i} **1968** I <think> i} I **1969** II The user is asking for a specific calculation: the percentage gap between two groups ("male i} **1970** I 65+ age group") and two metrics ("internet" and "broadband at home") from a specific i} I i} **1971** I survey ("Pew Research Center’s Internet Project July 18-September 30, 2013 tracking **1972** II survey"). I will search the document for the keywords  "male", "65+", "internet", i} **1973** I1 "broadband", and the specified survey date range. i} **1974** II **On page 15** , under the table titled "Internet usage for older adults", I find the row i} **1975** II "Gender" and its sub-row "Male (n=612)". The value in this cell is "65a". This represents i} **1976** II the percentage of male seniors who use the internet or email. **On page 16** , under the table i} **1977** II titled "Broadband adoption for older adults", I find the row "Gender" and its sub-row i} **1978** I "Male (n=612)". The value in this cell is "53b". This represents the percentage of male i} I **1979** I seniors who have broadband at home. The question asks for the "percentage gap". This i} I i} **198019811982** II1I means I need to calculate the difference between these two percentages. .The calculation is: also specifies that the answer should be written in "float format". The number 12 is already **65% (internet usage) - 53% (broadband at home) = 12%.** The question i}i} **1983** II a float. The final check is the context. The tables are explicitly for the "Pew Research i} **1984** II Center’s Internet Project July 18-September 30, 2013 tracking survey". i} **1985** II The text on page 15 and page 16 confirms this survey period. The calculated value directly i} **1986** II answers the question. i} I </think> i} **1987** I I <answer>[1] **1988** I i} I **12 1989** 1 </answer>[I] **1990** \4 Seo Le” **1991 1992 1993** Figure 19: Evidence for the query is split across two tables on consecutive pages (15 and 16). The model must correctly parse the complex query to locate the precise cells corresponding to ”Male **1994** 65+” for both ”internet usage” and ”broadband adoption”. **1995 1996 1997** 

37 

Under review as a conference paper at ICLR 2026 

## **1998** A.9.11 EXAMPLE 11: IDENTIFYING AND REJECTING HALLUCINATION TRAPS **1999** 

**2000** This final, crucial example demonstrates the model’s advanced ability to perform meta-reasoning **2001** and resist ”hallucination traps”—questions that are based on a false premise about the document’s **2002** content. Instead of attempting to generate a factually incorrect answer, the model correctly identifies **2003** the discrepancy by performing a comprehensive document search, locating the most relevant figures, **2004** and concluding that the question’s core subject (”BERT model”) is not present. This capability is vital for building trustworthy and reliable document analysis systems. 

**2005** 

**2006 Question:** _”In the pipeline diagram of the BERT model, which type of organism is used as the input_ **2007** _case?”_ 

## **2007** 

**2008** 

## **2009 Generated CoR Trace:** 

**2010** 

**2011 [Task Planning]** The user is asking about an input case in the ”pipeline diagram of **2012** the BERT model”. My plan is to first search the entire document for any mention **2013** of a ”BERT model” pipeline diagram. If found, I will analyze it. If not found, I **2014** must verify if the document discusses BERT at all to determine if the question is answerable. 

## **2015** 

**2016 [Phased & Focused Search]** A global search for ”BERT” yields no results. However, to be thorough, I searched for ”pipeline diagram”. This led me to two key **2017** figures: Fig. 1 on page 2 and Fig. 2 on page 6. 

## **2018** 

**[Cross-modal Evidence Integration]** - **Fig. 1 (Page 2):** The caption discusses the **2019** motivation for ”CLIP and MLLM” and the proposed ”RAR” model. No mention **2020** of BERT. - **Fig. 2 (Page 6):** The caption explicitly describes the ”Pipeline of **2021** RAR”. It details a multimodal retriever and refiner. No mention of BERT. 

**2022** 

- **[Grounded Reasoning & Verification]** My comprehensive search confirms that 

- **2023** the document does not contain a pipeline diagram for the BERT model; the di- 

- **2024** agrams present are for different models (RAR, MLLM). The user’s question is **2025** based on a false premise. Therefore, the question is unanswerable from the given **2026** document. Attempting to answer it would lead to hallucination. **2027** 

**2028 Final Answer: The provided document does not contain a pipeline diagram for the BERT model. 2029 2030 2031 2032 2033 2034** 

**2035 2036 2037 2038** 

**2039 2040 2041 2042** 

**2043** 

**2044** 

**2045** 

**2046** 

**2047** 

**2048** 

**2049** 

**2050** 

**2051** 

38 

Under review as a conference paper at ICLR 2026 

**2052** 2 Ziyu Liu et al. 6 ZiyuLin et al **2053** exsification RAR (a) Multimodal Retriever **20542055** m=== Se=e saeeem we= >.>Gils— i | _ =T ome ll fe (Oa gs = —__~sen 4 mage age os **2056** .a=) hh?= . Sale atele = GE os L == a: oe Lil jeooetda7 & t- - TSReticvin TasRan **2057** Sone eee ree Mere ~~ Racine Tap Ss **2058** trp veabaey= . a— ~ pacesi acewe — **2059** Baa.+B > ; > =teal <a:: : - gag.me eG = **2060** BB -- “HoeSEpeo _|\_1TGer3)7 |a=Se~ !:7H ilaT | |]BL:Nee : theFig. imag[2:][ Pipeline] or t **e** xt embeddings[of][(A][0.][(a)] and—[We] stores[design] embeddingsa multimodalin an externalretriever— memorythat extracts M. (b) Pig. 1:: Upper a: Our anea se eee tteaoe few at nee atee aba " Forfromthe theinference memorystageand ofwe MLLMsdownsireunto refine recognitionthe retrieved! tasks, reseliswe retrieveas the lop-final predictioncategories **2061** us Sesunlssly intent st MEL Ms tosapere “ through ranking. **2062** CLIP's performance begins to wane when faced with datasets characterized by its efficiency markedly diminishes as the dataset escalates to the magnitude of vast vocabularies or fine-grained categories. As shown in the upper left of Fig. 1, millions of embeddings. ‘lo enhance the speed of retrieval, we implement an index **2063** thetions declineand theis challengeslargely attributableposed byto thesynonyms, inherentwhich ambiguitycan confound of languagethe descrip-model's system‘The adoption that uses of the the HNSWHNSW methodology(Hierarchical Navigable Smallfacilitates a significant World) algorithmdimensionality[35]. abilityto distinguish between closely related but distinct classes. reduction, thereby enabling the construction of **2064** Parallel amore condensed index. Specifhave emerged to theseas a powerful developments,class of Multi-modalgenerative Largemodels, Language Modelsexemplified by the(MLLMs)likes of ically,dimensionalvector **s** pace. in a R‘Thi **s** pacereductionof dimensin d **i** onmensionalityd are transformedplays a pivotalinto a rolereducedin en-£ **2065** GPTAV [38] and analogous advancements [1 33,6, 29, 10, 9,55, 56, 60]. MLLMs, hancing the speed of the retrieval process. **2066** markableber of candidatesproficiencyremainsin identifyingmanageable.fine-grainedNevertheless,categoriesMLLMs’whenefficacythe totalis simi-numapproachmethodologydiscussedfor extractingpreviously.image embeddingsAs presented inein,Fig.is 3, slightlywe apply differenttwo additionalfrom the **2067** larlycategorizationsing compromised(upperin scenarios leftof Fig.involving1). Theextensivecore of vocabulariesthe issue liesandin fine-grainedMLLMs facsimilar methodspre-processingsteps:in CLIP cropping like [:i,and 53].blurring.In the objectSome previousdetectionworksdataset,have anproposed: image **2068** 2kinterpretingand[tokens] significantsubtlefordistinctions.tasks LLL ch **a** thatllengesVAL.5 demand [2s}),in m **a** nagcrita nuanced **i** calng largerequirementunderstanding contextfor windowsaccurateof vast(e.g., processing vocabularies maximumand ourproposaltypicallya large object portionboundingcontainadetectionof the multipleboxprocedure image,coordinates, objectswhereasbeginsof varying withsubsequentlyothers croppingoccupy sines. Some minimalresizing the image objects space.theregions maycropped Accordingly, dominatebasedregion on **2069** our‘To address these challenges, we propose augmenting standard MLLMs with to a fixed proportion. Moreover, unlike image classification tasks the objects **2070** modelsgeneration(Al,to workflows.dynamically@ retrieving-and-rankingByincorporate augmentingexternal MLLMsaugmented knowl witht **e** externalchnique.dge into knowledgetheOur processing!(\! sources,enables and objectofMLLMs interesde understand **t** ectiongenerallydatasets are smallertheappear objectslarge to and be detected,and centrallytheir positionswepositioned, employmore varied. a blurringthe objects‘To techniquehelpwithin the we address challenges related to language ambiguity, synonym handling, and the on the non-target areas surrounding the objects of interest. ‘The blurring strategy **2071** ularies.limitationsOur methodimposed byuseslimitedthe inherentcontextstrengthwindows ofwhenMLLMsdealingin generalizingwith vast vocab- from facilitatingis designed theirto directidentificationthe MLLMs’ in objectfocus detectiontoward tasks.the relevant objects, thereby **2072** _Pages 2, 6 of RAR.pdf (28pages) in MMLongBench-Doc_ **2073 v** 7 PT TTTT N **2074 Question: In the pipeline diagram of the BERT model, which type of organism is used as** \ **2075** | **the input case?** i} **2076** ! **Response:** 1 **2077** I! <think> i} **2078** !! The user is asking about the input organism in the pipeline diagram of the BERT model. The 1 **2079** ! document is titled "RAR: RRetrieving And Ranking Augmented MLLMs for Visual 1 **2080** ! Recognition". This title does not mention "BERT" or a "pipeline diagram". The user has 1 **2081** ! provided two document images. I will examine these images to find the relevant 1 **2082** ! information. Image 1 (page 2) contains a figure caption that reads: "Fig. 1: Upper left: our 1 **2083** ! motivation about the drawbacks of CLIP and MLLM. Our RAR can seamlessly integrate into 1 **2084** ! MLLMs to improve the few-shot/zero-shot abilities on classification (upper right) and 1 **2085** ! detection (bottom) datasets." This caption explicitly refers to "CLIP and MLLM", not "BERT". 1 **2086** !! Image 2 (page 6) contains a figure caption that reads: "Fig. 2: Pipeline of RAR. (a) We 1 **2087** !! design a multimodal retriever... (b)... we retrieve top-k categories from the memory and 1 **2088** ! use MLLMs to refine the retrieved results as the final prediction through ranking." This 1 **2089** ! caption explicitly refers to "RAR" and its pipeline. The user's question asks about the 1 **209020912092** !!!! "pipeline diagram of the BERT model". The document images do not contain this information. They show figures for "CLIP and MLLM" and "RAR". hallucination trap. It asks for information about a model ("BERT") that is not mentioned in the document, while also referencing a specific diagram ("pipeline diagram") that is The user's question is a !11 **2093** ! present but incorrectly attributed to a different model. Therefore, the question is not 1 **20942095** !!! answerable because it is based on a false premise about the content of the document.</think> 11 **2096** ! <answer> 1 **2097** ! \ **Not answerable** 1 **2098** \ \ </answer> / v7 **2099** Se LL LLL LL LL LL LL LLL LL LL LLL eee ee ee ee ee ee ee eee eee eee” **2100 2101** Figure 20: The model located the two most relevant pipeline diagrams in the document (Fig. Fig. 2). By analyzing their captions and content, it correctly determined that neither of them depicts **2102** the BERT model, thus identifying the user’s question as unanswerable based on the provided text. **2103 2104 2105** 

Figure 20: The model located the two most relevant pipeline diagrams in the document (Fig. 1 and Fig. 2). By analyzing their captions and content, it correctly determined that neither of them depicts the BERT model, thus identifying the user’s question as unanswerable based on the provided text. 

39 

Under review as a conference paper at ICLR 2026 

## **2106** A.10 CASE STUDY **2107 2108** A.10.1 CASE STUDY: NEGATION BLINDNESS VS. SYSTEMATIC VERIFICATION 

**2109 2110** This example highlights a common failure mode in complex query understanding: **negation blind2111 ness** . The question requires the model to identify an option that is _not_ present in the text, a task that demands more than simple keyword matching. 

**2112** 

- **2113** A base model, lacking a structured reasoning plan, tends to exhibit this failure. It is driven by infor- **2114** mation retrieval, successfully locating a passage that confirms one of the options (D) and incorrectly **2115** presenting it as the answer, thereby failing to address the negative constraint of the query. 

> **2116** In contrast, our CoR model employs a **systematic verification strategy** . It correctly interprets the 

> **2117** task as a process of elimination and methodically checks each option against the source document. 

> **2118** This robust process allows it to ignore misleading positive matches and correctly identify the truly **2119** absent option. **2120** Figure 21 provides a side-by-side comparison of the final outputs, visually demonstrating the base 

> **2121** model’s failure and the success of our CoR-driven approach. **2122** 

**2123 Question:** _”Which of the following is NOT mentioned as a consideration for whaling-related CG_ **2124** _operations? A. Guidance for Cutters/Aviation, including D17 MMPA Guidance/D17INST/OPLAN/NEPA._ **2125** _B. The dangers of whaling, particularly its occurrence ¿40NM offshore villages during whaling_ **2126** _season._ **2127** _C. The potential impact of commercial fishing vessels on whale populations._ **2128** _D. The potential impact of research vessels on whales, including the risk of diverting westward_ **2129** _migration._ **2130** _Choose the letter name in front of the right option from A, B, C, D.”_ **2131** 

**2132 2133** 

**2134 2135** 

**2136** 

**2137** 

**2138** 

**2139** 

**2140** 

**2141** 

**2142** 

**2143** 

**2144** 

**2145** 

**2146** 

**2147** 

**2148** 

**2149 2150** 

**2151** 

**2152** 

**2153** 

**2154** 

**2155** 

**2156** 

**2157** 

**2158** 

**2159** 

40 

Under review as a conference paper at ICLR 2026 

## **2160** A.10.2 CASE STUDY: FAILURE IN NUMERICAL AND COMPARATIVE REASONING **2161** 

**2162** This case study demonstrates a failure in fine-grained numerical and comparative reasoning, a crit- **2163** ical task in document analysis. Answering the question requires not only locating the correct data **2164** but also accurately interpreting and comparing the values. 

> **2165** The base model successfully locates the correct evidence source (Figure 7) and even extracts the 

> **2166** relevant numerical data for both low-wage (-17.7%) and high-wage (-16.9%) workers. However, it 

> **2167** fails at the final, crucial reasoning step. It incorrectly concludes that a -16.9% decline is a more **2168** severe negative impact than a -17.7% decline, leading to a completely erroneous final answer. 

**2169** 

This highlights a key advantage of our CoR model. Its explicit verification and reasoning steps **2170** would force a direct numerical comparison, correctly identifying that -17.7% represents a larger **2171** magnitude of decline. The stark difference in the models’ final outputs is illustrated in Figure 22. 

**2172** 

**2173** 

**2174** 

**Question:** _”In terms of employment rates, who is more negatively impacted by ’low-wage workers’ vs ’high-wage workers’?”_ 

**2175** 

**2176 2177** 

**2178** 

**2179** 

**2180 2181** 

**2182 2183** 

**2184 2185** 

**2186 2187** 

**2188 2189 2190 2191 2192 2193 2194 2195 2196 2197 2198 2199 2200 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 2211 2212 2213** 

41 

Under review as a conference paper at ICLR 2026 

## **2214** B STATEMENT ON THE USE OF LARGE LANGUAGE MODELS (LLMS) 

**2215** 

**2216** 

During the preparation of this manuscript, the Large Language Models (LLMs) Grok and Gemini **2217** were utilized to assist with the literature review, as well as for language polishing and grammatical **2218** correction to improve clarity and readability. It is important to state that the role of these models **2219** was strictly supportive; they were not involved in the core conceptualization, experimental design, or **2220** data analysis of this research. The authors completed all academic work independently and assume **2221** full responsibility for the final content, ensuring it adheres to all scholarly standards. 

**2222** 

**2223 2224 2225 2226 2227 2228 2229 2230 2231 2232 2233 2234 2235 2236 2237 2238 2239 2240 2241 2242 2243 2244 2245 2246 2247 2248 2249 2250 2251 2252 2253 2254 2255 2256 2257 2258 2259 2260 2261 2262 2263 2264 2265 2266 2267** 

42 

Under review as a conference paper at ICLR 2026 

**2268** eS **2269 2270 2271** * Cutters/Aviation: Guidance includes — j||° Potential to “Divert” whales et FA **2272** ** D17 MMPAGuidance/D17INST/OPLAN/NEPA é migrating westward am : f sy **2273** AEWC* CG Conflict does not Avoidance sign this document, Agreement but (CAA): abides by Zspea ‘ ds e] * offshore/openFall whale hunts water — up hunt to SONM 9 mc 5 — a—— | **2274** these guidelines unless a valid need to do otherwise * Difference in “Research” vs. Sn LS **2275** * >40NM= Food offshore Security villages vs. Vessel during Ops/Research whaling season dangerousWhaling is — + AWSC“Industry” working Ops/MMPA for long-term ¥» Syey r —aeys= **2276** = Contact AEWC and/or local Whaling Captain Associations skittish animals resolution of Research issues . aa f ae ; ma i = Do not operate in whaling zones during whaling season are more * Annual “Arctic Research Matrix” = : "Ls **2277** unless SAR or other necessity: dangerous Research Operations Schedule = = **2278** * Subsistence Hunting (Maritime/SAR Awareness): | **2279** ** Skin-covered boatsSkiffs/Open-Water Hunts in ice (Spring (Fall Harvest) Harvest) = y Sigs yeeroswe. = Up to 40-60 Miles Offshore 29 Bee 2} : — **2280** _Pages 29, 32 of 4034595.pdf (52 pages) in LongDocURL_ **2281 2282** Or { rr rere\ **2283** I **Question:** Which of the following is NOT mentioned as a consideration for whaling-related CG i} **2284** I operations?[i}] **2285** I A.Guidance for Cutters/Aviation, including D17 MMPA Guidance/D17INST/OPLAN/NEPA.[i}] **2286** 1I B.The dangers of whaling, particularly its occurrence >40NM offshore villages during whaling[i}] I season. **2287** 1[i}] C.The potential impact of commercial fishing vessels on whale populations. **2288** I[i}] I D.The potential impact of research vessels on whales, including the risk of diverting westward **2289** 1[i}] I migration. **2290** 1[i}] I Choose the letter name in front of the right option from A, B, C, D. i} **2291** 1 **Answer:** "C” 1 VL _ d **2292 2293** CTT SS SS SS ES SS SS SS SS Sy **2294** 1I **Qwen2.5-VL** xX i}1 **2295** 1 After reviewing the document‘s section on “Whaling: Agency Operational Issues,” the text i} **2296** I confirms that the potential impact of research vessels on whales, including the risk of diverting I **2297** I' westward migration, is a mentioned consideration for CG operations.So the answer is D. Il **2298 2299** OS { — SS SS SS SSS SS SS SS SS SS ST SR SS TS SS SS SH SH BT BH TSH SRY1 **2300** 1 **Qwen2.5-VL-CoR-7B** i} <think> **2301** Ii} **2302** I The user wants me to identify which of four options is *not* mentioned in the document. This i} I requires me to verify the existence of each option and then report the one that is absent.First, I **2303** II need to locate the relevant section. I'll search for keywords like \"whaling\" and \"CG i} **2304** Ii} I operations\". I've found a section titled \"Whaling: Agency Operational Issues\". This seems like **2305** I the most relevant place to find information about considerations for CG operations. Now, I will i} **2306** Ii} I examine each of the four options presented in the question.Option A: \"Guidance for i} **2307 2308** I Cutters/Aviation, including D17 MMPA Guidance/D17INST/OPLAN/NEPA.\" This is explicitly i} **2309** I listed under the \"Cutters/Aviation\" heading.Option B: \"The dangers of whaling...\". This is i} **2310** II mentioned\"The potentialunderimpact of commercialthe \"Subsistence Huntingfishing vessels(Maritime/SARon whale populations.\"Awareness)\" heading.OptionI have searchedC: i} **2311** II the entire document, including the \"Whaling\" section, and there is no mention of commercial i} **2312** II fishing vessels.Option D: \"The potential impact of research vessels on whales...\”. This is i} **2313** Ii} I mentioned under the \"Research Vessel Issues\" heading. Since option C is the only one that is **2314** II not mentioned anywhere in the document, it is the correct answer. i}i} **2315** I </think> i} **2316** I <answer> i} **2317 C** Ii} **2318** I </answer> 1 Noo? **2319 2320 2321** Figure 21: A side-by-side comparison for a negation-based question. The Base Model incorrectly matches keywords for option D and outputs it as the answer. Our CoR model correctly identifies the unmentioned option C through its systematic verification process. 

43 

Under review as a conference paper at ICLR 2026 

**2322 2323 2324** FIGURE 7: Changes in Employment by Wage Quartile **2325** FIGURE 5: Changes in Small Business Revenues vs. ZIP Code Characteristics, ‘A changes n Employmentby Wage Quarte **2326** ‘A. Median income 1. Population Density _ i fot **2327 2328 2329 2330** €- dian Two Becom Ret Maan Teo Bedoom Met: nPeron vs. Tleortable £8. changes in Employmenttueby Wagean Oust,nee Rewighng Aeons ©. changes in Employment"Penbyhea W **a** geng Quariaand ConsumerNose **2331 2332 2333 2334 2335** revenueNotes: Thisin Womply figure presentsdata vs. binnedvarious scatterZIP plotscode-lovel showingcharacteris. the relationship‘The betweenbinned changeseatter inplots seasonall-adjustdare constracted smalas decribed business seriesILC.Notes: ofWeThis privateseparatefigure norfarmplotsthe samplechangesemploymentintoin employmentwageconstructedquartilesby basedwageby combiningquartileon fixedrelativethresholdsPaychex:to JanuaryIntuitof the and20194-31Earnin2020.hourlyIndata,wageeachasdistribution.panel,describedwe show inIn SectionaPaneldaly **2336** inPeriodtheor Figurewhere(populstion-weighted)fromthe3 JanuaryInvarianceeach panel,4oftonormalizeJanuaryZIP-lovelthe changes31distribution.revenueandin smallseasonallyexceedsbusinessWe900%.adjustedexcluderevenueWo datsasalsoaredefinedfromexcludeteasedZIPin ZIPPigutecodesduringcodo-by-industry4,in witsoriaingwhichMarchchanges25 tocellsatAprilthearewith9Othlargor11 averagerelativepercentilethanweeklyto200%theof {Acombinedforforecastedthethe sodsamoPaychex-Intuit-EarnintingportiondateKronos(1),ofandeachdatathelineandPaychexreprenentsseriesPaychexonweeklythedatathede-seasonalizedseriescombinedfromforfirmethreePaycher-Intuit-EarninwithpriorKronosweeklyweeksseriespeyeycin.(¢ 7)for thedata, (014) same‘To whileconstrictdate(€~21).the(¢),dashedthistheWeforecast,Paychexthenportion weweweeklyoftheregrenstheresultinglineseriestheia **23372338** tststhrrevenueHouseholdof‘two-bedroom ZIPthe codecombination combinstionof incomeless inhabitantsapartmentthan atthe **of** $4,250 FoodPod perZIPina a squaretheduring **n** coded Accommodatio2014-2018dlevel mil Assmmodatithe from inbase the 2014-18ACS.theperiod **n** 2014-2018Panel(( **NAI** ofCS ACS.CSDJanuaryreplicates **72)** ACS.In **a** nd Panelnd4In Retail Panel Panelto etalJanuaryC, Trade theCB, theTradefor scass two (31. ( a-axis variable **NAI** distinct variableInCSCSPanel 14 **4** a asectors:isthe **nd** A, 48)isthe theZIP codex-axisin-personandlogarithm teleworkablevariable modian servicesoftheisrent services number definedmedianfor a beBcoefficientsamassJanuaryin toemploymentoma countywage411aefo xtoreal industryquartile202;predictslocethisJanuaryrethethe sample(2-digitexplainedvaluealee restriction4-31 NAICS)ofSome2020combinedby diffrencescellsin excludeseach whichPayeePaychex-Intuit-Earnincountyin0.9% incehaveincasiry~ ofx nonzeroindustry worker-dayaoe amfirst-quartileqaographinx employmentcuamlamentincomefrom thequartile andecmnponiion. sample. belonebefore fourth-quartilecell,seventheWewinsorsing. We thencombinedrestricthoming employmentcalculaat acadatathethe explo wean **t** heisOBthe dalyavailable.in thepercentilein change periodPanelieIn (weighted by total employment in the period January 4-31 2020). For the top and bottom wage quartiles, we estimate overall **23392340** dailyof‘insodartion,ofGquartlePresentemploymenttopemploymentquartile employmentthe changeFinally,employmentin byJanuaryrelativetakinglevelwe etinate4-31.intotheby JanuaryJanuaryperformingweightedThesebot4-31series431mean quactiooniployeanttheininareeachtheofsamesimilaremploymentPaycheclntui-Earnincell.procedure,Inbat Panelnotchangesroweighted tobetC,identical we reatrictweightingin everyemploymento rnichPanelbottomcountytothe theA retallbecauseseriesx ledustyquartileindustry tradeandoftheemploymentand actorthecll, gungsaphieAfBaityeampleweighting (NAICSchangesrestrictionconsumer Sompoution 44-45)by theby andlevelcardandtop **2341** Spending series. Data sources: Paychex, Intuit, Earnin, Kronos, Afinity Solutions _Pages 83, 85 of 4078345.pdf (109 pages) in LongDocURL_ **2342** por en ne en ee ee ee ee ee, **2343** !! **Question:** In terms of employment rates, who is more negatively impacted by \"low-wage 1 **2344** ' workers\" vs \"high-wage workers\"? I **2345** '1 **Answer:** "\"low-wage workers\"” a **2346** t[\] **2347** !! **Qwen2.5-VL** I **2348** ! Based on the data in Figure 7, which details changes in employment rates, high-wage workers I **2349** ! were more negatively impacted. The chart shows that the top wage quartile (high-wage I **2350** !! workers) experienced a significant employment decline of -16.9%, which is a more substantial negative impact compared to the -17.7% change for the bottom wage quartile (low-wage I **2351** !l ! workers). Final Answer: "high-wage workers" **2352** \’ **23532354** !;1 pene **Qwen2.5-VL-CoR-7B** ne eo ee ee—ee ee ee ee ee ee ‘\ **2355** I <think> 1 **23562357** \\\ The question asks me to compare the impact of \"low-wage workers\" versus \"high-wage workers\" on employment rates. I need to find a table or figure that breaks down employment 11 ' changes by wage quartile. I'll scan the document for keywords like \"wage,\" \"employment,\" **2358** ' ' and \"quartile.\" I've located **Figure 7** , which is titled \"Changes in Employment by Wage **2359** ' Quartile.\" This is the correct source. Panel A of this figure shows the change in employment **2360** '1 rates for different wage quartiles. I need to read the values for the bottom and top quartiles. **2361** I\ The bottom quartile (labeled \"Bottom Wage Quarter\") has a change of -17.7%. The top 1 **23622363** \\\\ quartile (labeled \"Top Wage Quarter\") has a change of -16.9%. Comparing these two values, the bottom wage quartile (-17.7%) is more negatively impacted than the top wage quartile (- 11 ' 16.9%). \".Low-wage workers are more negatively impacted by employment rate changes, with **2364** ' a decline of -17.7% compared to high-wage workers, who experienced a decline of -16.9%.\ **2365** ' </think> **2366** ' <answer> 1 **2367** \ **Low-wage workers** 1 \ **2368** \ </answer> 1 Noe eee! **2369 2370 2371 2372** Figure 22: Figure 7 from the source document, showing changes in employment by wage quartile. Correctly answering the question depends on accurately comparing the values for the top and bottom **2373** quartiles. **2374 2375** 

44 

