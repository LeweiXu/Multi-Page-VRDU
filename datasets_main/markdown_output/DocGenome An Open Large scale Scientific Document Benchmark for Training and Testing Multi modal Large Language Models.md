# **DocGenome: An Open Large-scale Scientific Document Benchmark for Training and Testing Multi-modal Large Language Models** 

**Renqiu Xia**[1] _[,]_[2] _[,][∗]_ **, Song Mao**[1] _[,][∗]_ **, Xiangchao Yan**[1] _[,][∗]_ **, Hongbin Zhou**[1] **, Bo Zhang**[1] _[,][‡]_ **Haoyang Peng**[1] **, Jiahao Pi**[1] **Daocheng Fu**[1] **, Wenjie Wu**[1] _[,]_[2] **, Hancheng Ye**[1] **, Shiyang Feng**[4] **Bin Wang**[1] **, Chao Xu**[1] **, Conghui He**[1] **, Pinlong Cai**[1] **, Min Dou**[1] **, Botian Shi**[1] _[,][‡]_ **Sheng Zhou**[3] **, Yongwei Wang**[3] **, Bin Wang**[4] **, Junchi Yan**[1] _[,]_[2] **, Fei Wu**[3] **, Yu Qiao**[1] 

1 Shanghai Artificial Intelligence Laboratory, 2 Shanghai Jiao Tong University 3 Zhejiang University, 4 Fudan University 

## **Abstract** 

Scientific documents record research findings and valuable human knowledge, comprising a vast corpus of high-quality data. Leveraging multi-modality data extracted from these documents and assessing large models’ abilities to handle scientific document-oriented tasks is therefore meaningful. Despite promising advancements, large models still perform poorly on multi-page scientific document extraction and understanding tasks, and their capacity to process within-document data formats such as charts and equations remains under-explored. To address these issues, we present DocGenome, a structured document benchmark constructed by annotating 500K scientific documents from 153 disciplines in the arXiv open-access community, using our custom auto-labeling pipeline. DocGenome features four key characteristics: _1) Completeness_ : It is the first dataset to structure data from all modalities including 13 layout attributes along with their L[A] TEX source codes. _2) Logicality_ : It provides 6 logical relationships between different entities within each scientific document. _3) Diversity_ : It covers various document-oriented tasks, including document classification, visual grounding, document layout detection, document transformation, open-ended single-page QA and multi-page QA. _4) Correctness_ : It undergoes rigorous quality control checks conducted by a specialized team. We conduct extensive experiments to demonstrate the advantages of DocGenome and objectively evaluate the performance of large models on our benchmark. DocGenome is available at `https://unimodal4reasoning.github.io/DocGenome_page` 

## **1 Introduction** 

Extracting data from scientific documents and developing large models to understand them is crucial for advancing AI-assisted scientific exploration and discovery [19, 11, 4]. On one hand, scientific documents provide comprehensive, high-quality, logically rich corpora for training large models [31, 7, 8, 33]. On the other hand, the ability of large models [31, 7, 8, 33] to accurately understand scientific documents is considered as a crucial evaluation criterion. 

However, we observed that current Multi-modal Large Language Models (MLLMs) [22, 56, 34, 9, 45, 7, 8, 5, 1, 23, 39, 46, 47, 50, 55, 58] still struggle to understand the content of scientific documents as deeply as humans do. This challenge is primarily due to the inherently complicated multi-modal information present in scientific documents, such as multi-modal charts [52, 49], intricate equations [42, 43], and sophisticated logical relationships. Currently, MLLMs cannot effectively 

*Equal contribution, _[‡]_ Corresponding authors. 

Preprint. Under review. 

**==> picture [372 x 149] intentionally omitted <==**

**----- Start of picture text -----**<br>
DocGenome Benchmark 6 Relationships Document to Genome<br>• 4 level types:<br>. 7 ————<br>Part 2  Intro figure  1<br>» Q Ss&% sgge | of 2 | [| xxx = -referredexplicitly<br>NSaeVorOAgaa3a atemel 7.4M ace~~ [N] ¢S eeoe Part 1 of 2 identical title adjacent XxxXxxxxx | |] | Method XxxXxx tr atre1id————  1  caption  2text-eq  3<br> 2 text-eq  4<br>: Introduction Introduction 3<br>erti — eid 4 equation  5<br>sotGoaen XxX XX XX] x 5 text-eq  6<br>_ Masia | | | becooe<br>subordinate non-title adjacent text-eq  7<br>S.et A& • 2 cite types: [ghjeeg ttten ier nissan Whe ayo sr etal Ses empty. eee enn 6 title  8<br>Bighyey *fi 27.3M ey > op. As shown in Fig1 — Seer d$ 8 eere$$ 7 text-eq  9<br>4‘a/ 2$ 3g ie %, Fig 1 Experiment of Table1 XXX [PasSasete Seespsonnentan amaseeinasnengeasUmer arecamere noes areneeewiasgtaeamma 109 -referredexplicitly footnote  11text  10<br>‘a explicitly- implicitly- 11<br>referred referred<br>explicitly- referred<br>implicitly -referred<br>non-title  adjacent<br>adjacent non-title<br>non-title  adjacent<br>adjacent non-title<br>subordinate<br>adjacent non-title<br>**----- End of picture text -----**<br>


Figure 1: **Overview of the DocGenome dataset.** Our work introduces DocGenome, a multi-modal dataset of academic documents encompassing 8 primary disciplines, 153 secondary disciplines, 13 categories of component units, and 6 types of entity relationships between units. We showcase an example of the paper [41] parsing into structured graph forms, termed as the document’s genome, by leveraging the attributes and relationships of component units. 

parse and comprehend such complicated modalities and logical relationships. To alleviate this challenge, we present DocGenome, an open large-scale scientific document benchmark constructed using the designed DocParser. 

DocParser is a cutting-edge auto-labeling pipeline, which can generate both attribute information of component units and logical relationships between units by auto-annotating and structuring a large amount of unlabeled arXiv papers, with four stages: 1) data preprocessing, 2) unit segmentation, 3) attribute assignment and relation retrieval, and 4) color rendering as elaborated in Sec. 3.1. Furthermore, we utilize the proposed DocParser to label 500K scientific documents collected from the arXiv open-access community, and the resulting auto-annotated dataset is termed as DocGenome (illustrated in Fig. 1), which contains 153 scientific disciplines and 7 document-oriented tasks including: document classification, visual grounding, open-ended single-page and multi-page QA tasks, document layout detection, Equation-to-L[A] TEX transformation, Table-to-L[A] TEX transformation, which is elaborated in Sec. 4.3. Furthermore, we employ the quality grading and human validation methods to ensure the data quality as described in Sec. 3.2 and Sec. 4.2, respectively. 

We conduct extensive experiments on the proposed DocGenome benchmark to objectively evaluate many mainstream MLLMs, including QWen-VL [5], CogAgent [15], InternVL 1.5 [8], GPT-4V [33], and _etc_ . The experiments on DocGenome also verify the effectiveness of the proposed dataset, demonstrating its ability to enhance the document understanding of the existing baseline models. 

Our main contributions can be summarized as follows: 

- For the first time, we construct an open large-scale dataset that includes **500K** structured scientific documents with **13** categories of component units and **6** types of logical relationships between them. This dataset also encompasses various data types within scientific documents, such as Figure, Equation, Table, Algorithm, List, Code, Footnote, and _etc_ . 

- To construct DocGenome, we design DocParser to automatically generate rich annotation information from the source code of a wealth of arXiv papers. 

- DocGenome covers **7** document-oriented tasks, such as document layout detection, document transformation, multi-page QA, _etc_ . Besides, we conduct extensive verification and experiments based on these tasks to demonstrate that DocGenome can significantly enhance the document understanding capabilities of the existing baselines. 

## **2 Related Works** 

**Visual Document Datasets.** To comprehensively show the advantages of the proposed DocGenome dataset, we have reviewed visual document datasets and summarized them in Table 1. In earlier years, visual document datasets [22, 56, 34, 9] mainly aim to recognize the region categories of different regions from a given document, such as text region, table region, abstract region, and _etc_ . For example, 

2 

Table 1: Comparison with document-related benchmarks. “ - ” indicates that the corresponding part is not mentioned in the original paper. “ * ” means that each sample in their training set is cropped from the entire page, resulting in a total of 6.4M samples at the region level rather than the page level. 

|Datasets|# Discipline|# Category of<br>Component Units|# Pages in<br>Train-set|# Pages in<br>Test-set|# Task<br>Type|# Used Evaluation<br>Metric|Publication<br>Period|With-<br>Entity Relation|
|---|---|---|---|---|---|---|---|---|
|DocVQA [32]|-|N/A|11K|1K|1|2|1960-2000|�|
|DocLayNet [34]|-|11|80K|8K|1|1|-|�|
|DocBank [22]|-|13|0.45M|**50K**|3|1|2014-2018|�|
|PubLayNet [56]|-|5|0.34M|12K|1|1|-|�|
|VRDU [48]|-|10|7K|3K|3|1|-|�|
|DUDE [40]|-|N/A|20K|6K|3|3|1860-2022|�|
|_D_4_LA_[9]|-|**27**|8K|2K|1|3|-|�|
|Fox Benchmark [25]|-|5|N/A (No train-set)|0.2K|3|5|-|�|
|ArXivCap [21]|32|N/A|6.4M_∗_|N/A|4|3|-|�|
|**DocGenome (ours)**|**153**|13|**6.8M**|9K|**7**|**7**|2007-2022|�|



DocBank [22] constructs 500K high-quality document pages to enable the document layout model to utilize both textual and visual information. Recently, some research works [32, 51, 52, 40, 21, 25] are proposed to build a document dataset with the enhanced diversity from multiple tasks, multiple modalities, and large-scale training data. By comparison, our DocGenome demonstrates more comprehensive features, including the number of disciplines and training samples covered, types of tasks, evaluation metrics, and entity relationships. 

**Visual Document Understanding.** Research in the field of document Artificial Intelligence (AI) has made rapid progress, due to its successful applications in visual document layout analysis [44, 40, 9, 3, 30, 17, 14] and image representation learning [57, 13, 10, 6]. Inspired by Transformer [41], LayoutLMv3 [17] utilizes word-patch features to perform pre-training and designs a cross-modal alignment for document AI. UDIO [37] tries to unify multiple document-oriented vision tasks using task-specific prompting. Besides, Kosmos-2.5 [31] generates the text outputs by a shared decoder-only Transformer. mPLUG-DocOwl [54] boosts the OCR-free document understanding ability. Recently, ICL-D3IE [12] proposes an in-context-based learning framework to integrate LLM into document information extraction tasks and LayoutLLM [30] employs the layout instruction mechanism to improve the ability of document analysis. 

**Multi-modal Large Language Models (MLLMs).** The development of MLLMs has profound impacts on the Artificial General Intelligence (AGI) landscape. Recently, commercial MLLMs [33, 38, 2, 35] have experienced extremely rapid progress. GPT-4V [33] has significantly advanced the MLLMs. Google’s Genimi series [38, 35] further enhance the ability of MLLMs to process text, images, and audio. Besides, open-source MLLMs [45, 7, 8, 5, 1, 29, 23, 24, 27, 36, 39, 46, 47, 50, 55, 58] have also attracted great attention. Such MLLMs bring accessibility to the rapid development of AI, enabling widespread multi-modal applications and fostering innovation across industries. 

## **3 Data Collection Methodology For DocGenome** 

## **3.1 Introduction of Auto-labeling Pipeline** 

In this section, we present DocParser, a cutting-edge auto-labeling pipeline that streamlines the extraction of labeled source code from unlabeled arXiv data, serving as a key instrument for annotating the DocGenome dataset. As shown in Fig. 2, the annotation process of DocParser is concisely divided into four stages, mitigating the issues of data scarcity and annotation expenses. 

**Stage 1: Data Preprocessing.** Our primary focus is to improve the data quality and enhance the compilation success rate of L[A] TEX source code. Initially, we undertake an expansion of all files referenced by the `\input` and `\include` commands, followed by a series of crucial pre-processing steps. These steps encompass the integration of requisite environment packages, the exclusion of comment lines, and the removal of extraneous tokens such as `\vspace` , `\ref` , and other annotations that do not contribute to the semantic essence of the document. Subsequently, we concentrate on standardizing the figure format within the L[A] TEX source code, converting all graphical elements to the PNG format. Furthermore, we remove the color attribute from the “hyperref”, ensuring that the L[A] TEX source code is ready for targeted color rendering during annotation in stage 4. 

**Stage 2: Units Segmentation.** The objective of this phase is to automate the segmentation of content units, thereby streamlining the rendering process for distinct sections. We employ the 

3 

**==> picture [397 x 182] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original Data Stage-one: Data Preprocessing Stage-two: Units Segmentation Stage-four: Color Rendering<br>Component Units List<br>\input{method} Title 1 Title 1<br>LaTEX Source-Code \input{intro}\input{related} [   {'section': '\\section{Example}'},  '\n'. 1 Text 2Text 3 Text 3Text 2 Bounding BoxOf Title 1<br>Expand all referenced files   'Here is an example.', 2 Table 4 Table 4<br>  '\n',<br>  'As shown in Table \ref{table}.', 3<br>\listoffigures\vspace comments\ref   '\n',  {'tabular': '\\begin{tabular}{|c|c|} \n 1 & 2 \\ \hline \n 3 & 4\\ \hline \n end{tabular} \n \label{table}'},  '\n ’ 4 Text 3Text 2Title 1 Text 2Text 3Title 1 Bounding BoxOf Text 2<br>Render toPNG Remove redundant tags that have no semantic information ] Table 4 Table 4<br>1. Example .pdf .png Stage-three: Attribute Assignment and Relation Retrieval       Meta Data<br>Here is an example. .eps .jpg Explicitly-referred .png .tex .tex<br>As shown in Table 1. Standardize figures to PNG format<br>1 2 PNG of full paper Source-code of  Source-code of<br>full paper component unit<br>3 4 Title 1 Text 2 Text 3 Table 4<br>hyperref hyperref Subordinate Table bbox<br>Remove color attribute of ‘ hyperref ’ Non-title adjacent component unitAttribute of  Relationship between component unit Bounding box of component unit<br>**----- End of picture text -----**<br>


Figure 2: **Schematic of the designed DocParser pipeline for automated document annotation.** The process is divided into four distinct stages: 1) Data Preprocessing, 2) Unit Segmentation, 3) Attribute Assignment and Relation Retrieval, and 4) Color Rendering. DocParser can convert L[A] TEX source code of a complete document into annotations for component units with source code, attributes, relationships, and bounding box, as well as a rendered PNG of the entire document. 

Table 2: The definition of logical relationships between component units. 

|**Relation Name**|**Specifc Description**|**Example**|
|---|---|---|
|_Identical_|Two units share the same source code.|Cross-column text; Cross-page text.|
|_Title adjacent_|The two titles are adjacent.|(\section{introduction}, \section{method})|
|_Subordinate_|One unit is a subclass of another unit.|(\section{introduction}, paragraph within|
|||Introduction)|
|_Non-title adjacent_|The two text or equation units are adjacent.|(Paragraph 1, Paragraph 2)|
|_Explicitly-referred_|One unit refers to another unit via footnote,|(As shown in \ref{Fig: 5} ..., Figure 5)|
||reference, etc.||
|_Implicitly-referred_|The caption unit refers to the corresponding|(Table Caption 1, Table 1)|
||foat environment.||



`TexSoup`[¶] library to decompose the L[A] TEX source code into a structured list, delineating each individual component unit. This list is organized according to the reading order, ensuring a logical progression and facilitating the subsequent retrieval of relationships between the component units. 

**Stage 3: Attribute Assignment and Relation Retrieval.** We have defined **13** fine-grained layout attributes (more details in Table A.1 of Appendix C) for the component units decomposed in Stage 2, encompassing elements such as Algorithms, Captions, Equations, etc. For each unit, we match an appropriate attribute from the predefined set using keyword queries and regularization techniques to ensure a tailored and precise categorization. In the analysis of component unit relationships, units are categorized into two classes: **1) fixed-form units** , including Text, Title, Abstract, etc., which are characterized by sequential reading and hierarchical relationships readily discernible from the list obtained in Stage 2, and **2) floating-form units** , including Table, Figure, etc., which establish directional references to fixed-form units through commands like `\ref` and `\label` . The comprehensive set of **6** entity relationships is detailed in Table 2. 

**Stage 4: Color Rendering.** The bounding box of a component unit is an additional label we aim to extract. After the segmentation phase in Stage 2, we render the target unit in black and all other units in white, to create two distinct PDFs. By performing a subtraction operation between these documents, we can obtain the detection box containing only the current unit, as illustrated in the top-right corner of Fig. 2. For component units that traverse across hurdles or pages, we standardize the bounding box labels based on their unified source code information. This method effectively 

> ¶TextSoup package: `https://github.com/alvinwan/TexSoup` . 

4 

mitigates the issue where bounding boxes may be inadvertently divided, ensuring seamless and unified labeling for such units. 

We automate the annotation process by sequentially applying DocParser’s four stages and leveraging the complete L[A] TEX source code. This yields not only the document’s PDF but also the individual source code, bounding box, specific attributes for each component unit, and the relationships between units. Together, these elements constitute our DocGenome dataset. 

## **3.2 DocGenome Benchmark Analyses** 

Utilizing the DocParser automated annotation tool, we have annotated a corpus comprising 500K academic articles from the arXiv repository. Our analysis explores the diversity of the DocGenome benchmark, focusing on discipline distribution, content distribution, and quality grading. 

**Discipline Distribution.** The DocGenome consists of 8 primary disciplines, which collectively encompass 153 secondary disciplines[||] , reflecting a diverse and extensive coverage of academic research areas. The distribution across these disciplines is detailed in Fig. A.2 of Appendix D. 

**Year Distribution.** DocGenome archives articles from arXiv, ranging from 2007 to 2022, with a median publication year of 2016. A significant portion, approximately 32.88%, of these articles have been published since 2020. The distribution of these publications over time is depicted in Fig. 3a. 

**Content Distribution.** We have examined two key aspects: the distribution of page counts and the labeling of component units. On the dimension of page counts, the dataset’s documents have an average page count of 13, with the longest document reaching 50 pages. The distribution of page counts is graphically represented in Fig. A.1 of Appendix C. Moving to the labeling perspective, we have annotated a substantial collection of 500K documents, totaling 74.5M component units and 68.5M relationship labels. In Fig. 1, we present a detailed visualization of the distribution of both the attribute tags of the component units and the relationship labels. 

**Quality Grading.** We establish two metrics to grade the data quality of the auto-labeled data that are generated using our DocParser. The first metric, designated as Eq. 1, measures the overlap among auto-annotated bounding boxes within each paper, thereby evaluating the intra-consistency of annotations: 

**==> picture [284 x 28] intentionally omitted <==**

where _J_ ( _Bi, Bj_ ) = _A_ ( _Bi_ )+ _AO_ (( _BBji_ ) _,B−jO_ )( _Bi,Bj_ )[is the] _[ IoU]_[between bounding boxes] _[ B][i]_[and] _[ B][j]_[.] _[N]_[is] the total number of annotated bounding boxes in each paper. _O_ ( _Bi, Bj_ ) represents the overlap area between bounding boxes _Bi_ and _Bj_ . _A_ ( _·_ ) refers to the area of the bounding box. 

Eq. 2 shows the second metric that quantifies the overlap between these annotated bounding boxes and the reference bounding boxes (predicted by DocXChain [53]), providing an assessment of the annotations’ alignment with established benchmarks, as formulated in Eq. 2: 

**==> picture [255 x 26] intentionally omitted <==**

where _Gi_ is the _i_ -th reference bounding box generated by DocXChain [53], _Bi_ refers to the bounding box that is closest to _Gi_ within our annotated ones. 

A lower _IoU_ intra with a higher _IoU_ align indicates a higher quality of auto-annotated bounding boxes. Specifically, we split the collected paper into three tiers based on the annotation results. For the _Tier_ -1 set, we select the papers with _IoU_ intra _<_ 0 _._ 05% and _IoU_ align _>_ 60%, while those with 0 _._ 05% _≤ IoU_ intra _<_ 1% and _IoU_ align _>_ 35% are packed in the _Tier_ -2 set, and the remaining papers are categorized as the _Tier_ -3 set. The distribution of three-tier data sets is shown in Fig. 3b, indicating that 28.56% of the data was allocated to _Tier_ -1, 61.30% to _Tier_ -2, and the other 10.14% to _Tier_ -3. 

> || According to the arXiv Category Taxonomy: `https://arxiv.org/category_taxonomy` . 

5 

**==> picture [394 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
70000 Documents CountsAccumulated Documents Counts 500000 1.0 Tier 1 (28.56%)Tier 2 (61.30%)<br>Tier 3 (10.14%)<br>60000 400000 0.8<br>50000<br>300000 0.6<br>40000<br> Accumulated Documants CountsMedian (50%) of<br>30000 200000 0.4<br>20000<br>100000 0.2<br>10000<br>0<br>0 2007 2008 2009 2010 2011 2012 2013Publication Year2014 2015 2016 2017 2018 2019 2020 2021 2022 0.0 0 0.005 0.001 0.005IoU_intra0.01 0.1 1<br>(a) (b)<br>IoU_align<br>Documents Counts<br>Accumulated Documents Counts<br>**----- End of picture text -----**<br>


Figure 3: **Visualization of data distribution in DocGenome** . (a) Document publication counts over the years. (b) Distribution of three _Tiers_ determined by _IoU_ intra and _IoU_ align. 

## **4 DocGenome-test: A Multi-task, Multi-modal, Comprehensive Evaluation Set for Document Understanding** 

## **4.1 Principles of Constructing Evaluation Set** 

We use two principles to split the auto-annotated data into a high-quality evaluation set ( **termed as DocGenome-test** ) with precise annotation and a large-scale multi-modal training set ( **termed as DocGenome-train** ). First, the evaluation set should share the same discipline distribution as the collected data. Hence, the test data are uniformly sampled across each discipline. Second, the annotation of test data should be as precise as possible. Therefore, the test data are only sampled from the _Tier_ -1 set. Based on these two principles, we finally sampled 1,004 papers (covering 9K pages) as the test set from the overall 500K auto-annotated papers (containing 6.8M pages). As a result, the DocGenome-test covers 1,004 scientific documents with 1K document classification examples, 2K visual grounding examples, 3K QA pairs, 110K layout bounding boxes, 3K Table-L[A] TEX pairs, and 5K Equation-L[A] TEX pairs. 

## **4.2 QA Pair Generation and Quality Assurance** 

In the DocGenome-test, we further design multiple Question-Answering (QA) pairs for each paper to comprehensively evaluate the document understanding capabilities of different models. For each paper sampler, two single-page QA pairs and two multi-page QA pairs are generated using GPT4V [33]. Specifically, we instruct GPT-4V to randomly select two representative pages, extract useful information from the two pages respectively, and then generate corresponding single-page QA pairs. Additionally, we utilize GPT-4V to search for content-related paragraphs from different pages to construct the cross-page QA pairs, testing the model’s ability to understand and integrate information across multiple pages. The QA pairs involve various commonly raised questions whose answers can be precisely inferred from the given paper. 

After generating QA pairs for all paper samples in the DocGenome-test, we invited professional faculty members from various fields to conduct the quality assurance checks. Each QA pair is reviewed by three reviewers for cross-verification. The first step involves the initial review by Kimi[††] , a well-known paper understanding model, to assess the initial correctness and identify the target location of QA information on the assigned page. Next, based on the provided location of QA information, two professional faculty members are assigned to manually and independently check each QA pair for accuracy, relevance, and clarity. At this stage, the quality evaluation involves the correctness, relevance, and rationality of the designed questions and the accuracy of the provided answer. Finally, the two manually-evaluated results, along with the automatically-evaluated result are cross-verified with the original text to ensure accuracy and consistency. Please refer to Appendix E for more details. 

> ††Kimi online API: `https://kimi.moonshot.cn` . 

6 

Table 3: Comparison of state-of-the-art multi-modal large language models on the proposed DocGenome-test, including document classification, visual grounding, open-ended single-page, and multi-page QA tasks. Please refer to Sec. 4.4 for the employed evaluation metrics. 

|Model<br>#Params|Classifcation<br>Acc_↑_|Classifcation<br>Acc_↑_|Visual Grounding<br>Document QA<br>Title<br>Abstract<br>Single-Page<br>Multi-Page<br>Edit Distance_↓_<br>Edit Distance_↓_<br>GPT-acc_↑_<br>GPT-acc_↑_|Visual Grounding<br>Document QA<br>Title<br>Abstract<br>Single-Page<br>Multi-Page<br>Edit Distance_↓_<br>Edit Distance_↓_<br>GPT-acc_↑_<br>GPT-acc_↑_|Visual Grounding<br>Document QA<br>Title<br>Abstract<br>Single-Page<br>Multi-Page<br>Edit Distance_↓_<br>Edit Distance_↓_<br>GPT-acc_↑_<br>GPT-acc_↑_|Visual Grounding<br>Document QA<br>Title<br>Abstract<br>Single-Page<br>Multi-Page<br>Edit Distance_↓_<br>Edit Distance_↓_<br>GPT-acc_↑_<br>GPT-acc_↑_|
|---|---|---|---|---|---|---|
|**_Multi-modal Large Language Models_**|||||||
|QWen-VL [5]<br>9.6B<br>CogAgent [15]<br>17.3B<br>DocOwl-1.5 [16]<br>8.1B<br>Text-Monkey [26]<br>10B<br>InternVL 1.5 [8]<br>26B<br>InternVL 2<br>26B<br>GPT-4V<br>N/A<br>GPT-4o<br>N/A||0.8237<br>0.5857<br>0.3307<br>0.7331<br>0.7590<br>0.8855<br>**0.9821**<br>0.9761||0.0775<br>0.8054<br>0.0166<br>0.5306<br>0.0509<br>0.6555<br>0.0371<br>0.4551<br>0.0222<br>0.3601<br>0.0176<br>0.2320<br>0.0096<br>**0.0431**<br>**0.0095**<br>0.0654||0.1156<br>0.0627<br>0.1772<br>-<br>0.3084<br>-<br>0.1142<br>-<br>0.4529<br>0.3577<br>0.5019<br>0.4125<br>0.6101<br>0.6501<br>**0.7183**<br>**0.6762**|



## **4.3 Evaluation Tasks** 

To comprehensively evaluate the models’ understanding capability of scientific documents, we design **7** tasks _w.r.t_ each paper document for the DocGenome-test, including document classification, visual grounding, open-ended single-page, and multi-page QA tasks, document layout detection, Equation-to-L[A] TEX transformation, and Table-to-L[A] TEX transformation. 

Specifically, document classification involves recognizing the field to which a paper belongs. Visual grounding involves identifying the content according to the provided visual components and textual prompts. Document layout detection refers to the localization and recognition of each layout block in given papers. Document transformation encompasses two format conversions, _i.e._ , Table-to-L[A] TEX and Equation-to-L[A] TEX transformation. All tasks take the paper images as visual input for inference. The visual examples for each task are illustrated in Fig. A.8 in Appendix H. 

## **4.4 Evaluation Metrics** 

**Document Classification:** Top-1 Accuracy (%) is used as the metric for document classification tasks, where higher values indicate better performance. 

**Visual Grounding:** Edit Distance is used to evaluate the accuracy of visual grounding, with lower values indicating better performance. 

**Document Layout Detection:** mAP@0.5:0.95 is evaluated as the metric for document layout detection, where higher values indicate better performance. 

**Document Transformation:** We utilize Edit Distance, Jaccard Similarity, Cosine Similarity, and BLEU as metrics to comprehensively evaluate the document transformation task. 

**Open-ended QA:** GPT-acc (%) is designed for tasks with open-ended answers, where outputs are evaluated against the ground truth using GPT-4. Please refer to Appendix F for more details. 

## **5 Experiments** 

## **5.1 Compared Baselines and Implementation** 

**Compared Baselines.** We select various models as baselines for different tasks to provide comprehensive comparisons. Specifically, various multi-modal language models, _e.g._ , QWen-VL [5], CogAgent [15], DocOwl-1.5 [16], Text-Monkey [26], IntenVL 1.5 [8], and GPT-4V [33] are tested on document classification, visual grounding, open-ended single-page QA and multi-page QA tasks. For the Document Layout Detection task, we compare DocXChain [53] and YOLOv8 [18]. Additionally, we employ Mathpix, a representative commercial software for mathematical formula transformation, as the compared method for the Document Transformation task, including Equation-to-L[A] TEX and Table-to-L[A] TEX transformations. 

**Implementation Details.** We utilize a combination of document images and instruction prompts as the input. Note that all tasks use a single-page document image as the input, except for the multi-page QA task, which contains at least two consecutive pages of the document. Besides, the multi-page QA task can only be evaluated on the models that support multi-image inputs. For the layout detection task, which uses the single-page document image as input, we use YOLOv8 [18] as the training baseline, trained for 30 epochs with the AdamW optimizer [28], with a learning rate of 0.01. For 

7 

Table 4: Experiments on scaling up the data using the DocGenome-train, with the resulting models evaluated on document layout detection task. We fine-tune YOLOv8 [18] model using the DocGenome-train with different amounts of training data. 

|Model|Training Data Amount|mAP@0.5:0.95_↑_|Title<br>Text<br>Figure<br>Caption<br>Equation<br>Table<br>Footnote|
|---|---|---|---|
|**_Layout detection task on DocGenome-test_**||||
|DocXChain [53]|N/A|53.20|49.21<br>79.22<br>43.85<br>48.18<br>49.36<br>72.79<br>29.79|
|YOLOv8 [18]<br>YOLOv8 [18]<br>YOLOv8 [18]|7K<br>70K<br>700K|77.47<br>89.42<br>**91.37**|71.79<br>92.48<br>76.29<br>86.56<br>80.65<br>85.81<br>48.43<br>83.46<br>95.56<br>86.36<br>94.92<br>90.13<br>92.77<br>82.72<br>**86.05**<br>**95.96**<br>**88.46**<br>**95.71**<br>**93.06**<br>**93.77**<br>**86.52**|



Table 5: Experiments on scaling up the data using the DocGenome-train, with the resulting models evaluated on equation and table transformation tasks. EqVLM-B and TableVLM-B mean that we train a visual encoder and a text decoder using the DocGenome-train for the equation and table transformation task, respectively. 

|Model<br>Training Data Amount|Edit Distance_↓_<br>Jaccard Similarity_↑_<br>Cosine Similarity_↑_<br>BLEU_↑_|
|---|---|
|**_Equation-to-LaTeX task on DocGenome-test_**<br>||
|Mathpix‡<br>N/A<br>EqVLM-B<br>10K<br>EqVLM-B<br>100K<br>**EqVLM-B**<br>1M|0.4738<br>0.7226<br>0.6045<br>0.4472<br>0.3781<br>0.8157<br>0.7840<br>0.5165<br>0.2795<br>0.8505<br>0.8317<br>0.5862<br>**0.2111**<br>**0.8736**<br>**0.8621**<br>**0.6352**|
|**_Table-to-LaTeX task on DocGenome-test_**||
|Mathpix§<br>N/A<br>TableVLM-B<br>5K<br>TableVLM-B<br>10K<br>TableVLM-B<br>100K<br>**TableVLM-B**<br>500K|0.4436<br>0.7730<br>0.5826<br>0.3528<br>0.4821<br>0.8158<br>0.7804<br>0.4596<br>0.4738<br>0.8635<br>0.8187<br>0.4973<br>0.3091<br>0.8903<br>0.8571<br>0.5340<br>**0.2223**<br>**0.8997**<br>**0.8800**<br>**0.5552**|



Equation-to-L[A] TEX and Table-to-L[A] TEX tasks, we first use the layout annotations to crop out different modalities, _e.g._ , Table, Equation, _etc_ ., from the original images. We then employ the same model structure as Pix2Struct-B (0.2B parameters) [20] to perform the fine-tuning on DocGenome-train, resulting in EqVLM-B and TableVLM-B. The fine-tuning process lasts for 30 epochs on 64 NVIDIA A100 80G GPUs, with an initial learning rate of 0 _._ 00005 and a weight decay of 0 _._ 01. 

## **5.2 Performance on DocGenome-test** 

We evaluate the performance of several state-of-the-art multi-modal large language models on the proposed DocGenome-test, covering document classification, visual grounding, and both single-page and multi-page QA tasks. As shown in Table 3, among the tested models, GPT-4V [33] achieves the highest classification accuracy with 98.0% Top-1 Acc, while QWen-VL [5] and InternVL 1.5 [8] also show competitive results with 82.4% and 75.9% accuracy, respectively. For the visual grounding task, GPT4V showcases the best performance in the Title OCR Grounding task with the lowest Edit Distance of 0 _._ 0104, while InternVL 1.5 outperforms other models in the Abstract OCR Grounding task with the lowest Edit Distance of 0 _._ 3601. In the single-page QA task, GPT-4V attains the highest GPT-acc score of 61.0%, indicating its superior ability to handle document-based QA tasks. For the multi-page QA task, GPT-4V again leads with a GPT-acc score of 65.0%, further demonstrating its robustness in handling multi-page document queries. 

## **5.3 Effectiveness of DocGenome-train** 

To validate the effectiveness of the proposed DocGenome-train, we further conduct experiments on scaling up the training data using the DocGenome-train dataset, evaluating the performance improvements of different tasks, _e.g.,_ layout detection and document transformation tasks. 

Specifically, for the layout detection task, we present the evaluation performance of YOLOv8 [18] under three different training scales in Table 4. It shows that the model’s layout detection capacity continually and significantly improves by increasing the training data volume. Regarding the per-attribute performance improvement, the most significant benefit is observed for “Footnote” attribute, which increases from 48.43% to 86.52% mAP after scaling up the training data from 7K to 700K. Compared with DocXChain [53] that only supports the annotation of seven attributes, our trained YOLOv8 consistently outperforms it in seven attributes, validating the effectiveness of the DocGenome-train. 

8 

Table 6: Comparisons with state-of-the-art tools on Out-Of-Distribution (OOD) data, where Mathpix is a closed-source commercial software that requires a subscription, while ours is an open-source and free tool. 

|Model<br>mAP@0.5:0.95_↑_|Title<br>Text<br>Figure<br>Caption<br>Equation<br>Table<br>Footnote|Title<br>Text<br>Figure<br>Caption<br>Equation<br>Table<br>Footnote|
|---|---|---|
|**_Layout detection task on Human-annotated data_**|||
|DocXChain [53]<br>37.99<br>YOLOv8 [18]<br>**50.15**|32.53<br>59.00<br>**67.17**<br>38.71<br>12.98<br>38.99<br>16.54<br>**42.59**<br>**64.87**<br>56.65<br>**64.51**<br>**47.14**<br>**47.08**<br>**28.21**||
||||
|Model||Edit Distance_↓_<br>Jaccard Similarity_↑_<br>Cosine Similarity_↑_<br>BLEU_↑_|
|**_Equation-to-LaTex task on Sci-Hub data_**<br>Mathpix‡.<br>**0.4873**<br>**0.7437**<br>**0.7295**<br>**0.1137**<br>EqVLM-B<br>0.6627<br>0.6303<br>0.5726<br>0.0602|||



As illustrated in Table 5, for the document transformation task, we conduct similar experiments on Equation-to-L[A] TEX task and Table-to-L[A] TEX task, respectively. In these two tasks, we further explore different scaling up settings, with the observation that both tasks benefits the most from scaling up training data from 10K to 100K. Additionally, considering that Edit Distance is more reliable and rigorous to evaluate the similarity, we can observe that the Table-to-L[A] TEX task has the potential to improve more than the Equation-to-L[A] TEX task by continuous scaling up. This is because the performance improvement between 100K and 500K training data for TableVLM-B largely exceeds the improvement between 100K and 1M training data for EqVLM-B as shown in Table 5. 

## **5.4 Further Discussions** 

**Generalization on Out-Of-Distribution (OOD) Data.** We discuss the generalization ability of models trained on our DocGenome-train to OOD data. Specifically, we conduct experiments on human-annotated data for the layout detection task and Scihub data for the Equation-to-L[A] TEX task. As shown in Table 6, for the layout detection task, YOLOv8 [18] trained using DocGenome-train presents better generalization ability than DocXChain on human-annotated data. Regarding the Equation-to-L[A] TEX task, although the performance of EqVLM-B declines on OOD data (Scihub data), it still maintains relatively strong results with an Edit Distance of 0.6627. Considering that Mathpix is a closed-source tool with potential exposure to various data distributions in its commercial usage, it is natural that our trained model performs relatively worse than Mathpix in the OOD data. 

**Potential Applications of DocDenome.** 1) Conducting document transformation task for more modality types: DocGenome includes various types of data within scientific documents, such as Charts, Equations, Tables, Algorithms, Lists, Codes, and Footnotes, _etc_ . For this paper, we study the document transformation using only two types of modalities: Table-to-L[A] TEX and Equation-to-L[A] TEX. Similarly, we can also train a model (image-encoder followed by a text-decoder) that can address the Algorithm-to-L[A] TEX or List-to-L[A] TEX transformation task, _etc_ using DocGenome. 

2) Performing document-level tasks with entity relations: DocGenome contains the logical relationships between component units, we can input different component units to examine the model’s understanding of long-range contextual relationships. 

3) Conducting document OCR task on any page at any location: the layout annotations of DocGenome are very comprehensive, covering almost all locations in the document, and DocGenome has the ground truth text of the entire document. Therefore, we can use the layout information and text information to perform OCR tasks on any page at any location, not just the title and abstract regions, which further examines both the OCR capability and the visual grounding capability of the model. 

## **6 Conclusion** 

In this paper, we introduced DocGenome, a large-scale, structured, multi-task, and multi-modal dataset for scientific documents. We constructed DocGenome using DocParser, our developed autolabeling pipeline, to extract structured attributes and relationships between units. DocGenome’s comprehensive task coverage, logicality, diversity, and correctness make it a valuable resource for training models related to scientific documents and evaluating the capabilities of such large models. 

9 

## **Acknowledgement** 

The research was supported by the National Key R&D Program of China (Grant No. 2022ZD0160104), the Science and Technology Commission of Shanghai Municipality (Grant No. 22DZ1100102), and Shanghai Rising Star Program (Grant No. 23QD1401000). 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 35:23716–23736, 2022. 

- [2] Anthropic. The claude 3 model family: Opus, sonnet, haiku. `https://www.anthropic.com,` , 2024. 

- [3] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R Manmatha. Docformerv2: Local features for document understanding. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 709–718, 2024. 

- [4] Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey Ovchinnikov, Gyu Rie Lee, Jue Wang, Qian Cong, Lisa N Kinch, R Dustin Schaeffer, et al. Accurate prediction of protein structures and interactions using a three-track neural network. _Science_ , 373(6557):871–876, 2021. 

- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- [6] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. _IEEE transactions on pattern analysis and machine intelligence_ , 35(8):1798–1828, 2013. 

- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 

- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 

- [9] Cheng Da, Chuwei Luo, Qi Zheng, and Cong Yao. Vision grid transformer for document layout analysis. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19462–19472, 2023. 

- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. _ArXiv_ , abs/2010.11929, 2020. 

- [11] Richard Evans, Michael O’Neill, Alexander Pritzel, Natasha Antropova, Andrew Senior, Tim Green, Augustin Žídek, Russ Bates, Sam Blackwell, Jason Yim, Olaf Ronneberger, Sebastian Bodenstein, Michal Zielinski, Alex Bridgland, Anna Potapenko, Andrew Cowie, Kathryn Tunyasuvunakool, Rishub Jain, Ellen Clancy, Pushmeet Kohli, John Jumper, and Demis Hassabis. Protein complex prediction with alphafold-multimer. _bioRxiv_ , 2021. doi: 10.1101/2021.10.04.463034. URL `https://www.biorxiv. org/content/early/2021/10/04/2021.10.04.463034` . 

- [12] Jiabang He, Lei Wang, Yi Hu, Ning Liu, Hui Liu, Xing Xu, and Heng Tao Shen. Icl-d3ie: In-context learning with diverse demonstrations updating for document information extraction. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19485–19494, 2023. 

- [13] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 16000–16009, 2022. 

- [14] Xinyi He, Mengyu Zhou, Xinrun Xu, Xiaojun Ma, Rui Ding, Lun Du, Yan Gao, Ran Jia, Xu Chen, Shi Han, et al. Text2analysis: A benchmark of table question answering with advanced data analysis and unclear queries. _arXiv preprint arXiv:2312.13671_ , 2023. 

- [15] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. _arXiv preprint arXiv:2312.08914_ , 2023. 

10 

- [16] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_ , 2024. 

- [17] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4083–4091, 2022. 

- [18] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLO, January 2023. URL `https://github. com/ultralytics/ultralytics` . 

- [19] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A A Kohl, Andrew J Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with AlphaFold. _Nature_ , 596(7873):583–589, 2021. doi: 10.1038/s41586-021-03819-2. 

- [20] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _International Conference on Machine Learning_ , pages 18893–18912. PMLR, 2023. 

- [21] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. _arXiv preprint arXiv:2403.00231_ , 2024. 

- [22] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. _arXiv preprint arXiv:2006.01038_ , 2020. 

- [23] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint arXiv:2311.06607_ , 2023. 

- [24] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. _arXiv preprint arXiv:2401.15947_ , 2024. 

- [25] Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for fine-grained multi-page document understanding. _arXiv preprint arXiv:2405.14295_ , 2024. 

- [26] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. _arXiv preprint arXiv:2403.04473_ , 2024. 

- [27] Zhaoyang Liu, Zeqiang Lai, Zhangwei Gao, Erfei Cui, Zhiheng Li, Xizhou Zhu, Lewei Lu, Qifeng Chen, Yu Qiao, Jifeng Dai, et al. Controlllm: Augment language models with tools by searching on graphs. _arXiv preprint arXiv:2310.17796_ , 2023. 

- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- [29] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. _arXiv preprint arXiv:2402.14800_ , 2024. 

- [30] Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, Zhi Yu, and Cong Yao. Layoutllm: Layout instruction tuning with large language models for document understanding. _arXiv preprint arXiv:2404.05225_ , 2024. 

- [31] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, et al. Kosmos-2.5: A multimodal literate model. _arXiv preprint arXiv:2309.11419_ , 2023. 

- [32] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 

- [33] OpenAI. Gpt-4v(ision) system card. `https://openai.com/contributions/gpt-4v` , 2023. 

11 

- [34] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. Doclaynet: a large human-annotated dataset for document-layout segmentation. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 3743–3751, 2022. 

- [35] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- [36] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. _arXiv preprint arXiv:2307.05222_ , 2023. 

- [37] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19254–19264, 2023. 

- [38] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- [39] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. _arXiv preprint arXiv:2401.10208_ , 2024. 

- [40] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19528–19540, 2023. 

- [41] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Neural Information Processing Systems_ , 2017. URL `https://api.semanticscholar.org/CorpusID:13756489` . 

- [42] Bin Wang, Zhuangcheng Gu, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition. _arXiv preprint arXiv:2404.15254_ , 2024. 

- [43] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He. Cdm: A reliable metric for fair and accurate formula recognition evaluation. _arXiv preprint arXiv:2409.03643_ , 2024. 

- [44] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. Docllm: A layout-aware generative language model for multimodal document understanding. _arXiv preprint arXiv:2401.00908_ , 2023. 

- [45] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023. 

- [46] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. _arXiv preprint arXiv:2402.19474_ , 2024. 

- [47] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. _arXiv preprint arXiv:2403.15377_ , 2024. 

- [48] Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee, and Sandeep Tata. Vrdu: A benchmark for visuallyrich document understanding. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 5184–5193, 2023. 

- [49] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. _arXiv preprint arXiv:2406.18521_ , 2024. 

- [50] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. _arXiv preprint arXiv:2309.05519_ , 2023. 

12 

- [51] Renqiu Xia, Bo Zhang, Haoyang Peng, Ning Liao, Peng Ye, Botian Shi, Junchi Yan, and Yu Qiao. Structchart: Perception, structuring, reasoning for visual chart understanding. _arXiv preprint arXiv:2309.11268_ , 2023. 

- [52] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. _arXiv preprint arXiv:2402.12185_ , 2024. 

- [53] Cong Yao. Docxchain: A powerful open-source toolchain for document parsing and beyond. _arXiv preprint arXiv:2310.12430_ , 2023. 

- [54] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. _arXiv preprint arXiv:2307.02499_ , 2023. 

- [55] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. _arXiv preprint arXiv:2307.03601_ , 2023. 

- [56] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In _2019 International conference on document analysis and recognition (ICDAR)_ , pages 1015–1022. IEEE, 2019. 

- [57] Zhanpeng Zhou, Zijun Chen, Yilan Chen, Bo Zhang, and Junchi Yan. Cross-task linearity emerges in the pretraining-finetuning paradigm. _arXiv preprint arXiv:2402.03660_ , 2024. 

- [58] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

13 

## **A Overview of Appendix** 

We provide more information on our benchmark and further experiment details from the following aspects: 

- Sec. B: Limitations and Dataset Accessibility. 

   - Sec. B.1: Limitations. 

   - Sec. B.2: Dataset Accessibility. 

- Sec. C: Annotation Explanations. 

- Sec. D: More Statistical Distributions of DocGenome. 

- Sec. E: Details of Quality Assurance. 

- Sec. F: Prompt Design for GPT-acc. 

- Sec. G: Annotation Examples in DocGenome. 

- Sec. H: Task Examples in DocGenome-test. 

## **B Limitations and Dataset Accessibility** 

## **B.1 Limitations** 

The purpose of our DocGenome is to build a comprehensive scientific document dataset, promoting the development of intelligent document processing and effective evaluation of MLLMs in document understanding tasks. Although our DocGenome provides annotations for 6 categories of entity relationships, exploring the impact of these entity relationship annotations on large models’ understanding of scientific documents is highly meaningful. For future works, we will explore the role of the entity relationships in understanding scientific documents. 

## **B.2 Dataset Accessibility** 

**Dataset Documentation:** We have documented our dataset and its intended uses, as required. The website of our dataset is available at the following link: `https://github.com/ UniModal4Reasoning/DocGenome` , which includes metadata, format details, and visualizations. Besides, the download link for the dataset is: `https://drive.google.com/drive/folders/ 1OIhnuQdIjuSSDc_QL2nP4NwugVDgtItD?usp=sharing` . 

**Dataset Statistics and Analyses:** We have conducted extensive data statistics and analyses, along with thorough quality checks including DocGenome-train and DocGenome-test datasets, which are presented in Sec. 3.2 and Sec. 4.2. 

**Long-term Preservation:** To ensure the long-term preservation of the DocGenome dataset, we have uploaded it to Google Drive[§] . This ensures continuous accessibility to the dataset for an extended duration. Furthermore, we will routinely back up the data and monitor its availability to maintain continued accessibility. 

**Terms of Use and License:** We have chosen the CC BY 4.0 license for our dataset, as required. This information is included in our paper submission and will also be clearly stated on our dataset website. 

**A Persistent Dereferenceable Identifier:** We have obtained a DOI for our dataset, referred to as 10.5281/zenodo.11488587. This persistent dereferenceable identifier ensures long-term accessibility and citability of the dataset. 

**Discussion of Personally Identifiable Information.** All the scientific documents in our DocGenome are sourced from the arXiv open-access community, where papers are released under the CC license. Besides, the arXiv community ensures that papers uploaded by authors adhere to legal and ethical guidelines, including the protection of personal information and the avoidance of offensive material. Thus, we can confirm that our DocGenome does not contain personally identifiable information or offensive content. 

> §The download link for the dataset is available at: `https://drive.google.com/drive/folders/ 1OIhnuQdIjuSSDc_QL2nP4NwugVDgtItD?usp=sharing` . 

1 

**==> picture [397 x 239] intentionally omitted <==**

**----- Start of picture text -----**<br>
500000<br>35000<br>80%<br>400000<br>30000<br>25000<br>300000<br>50%<br>20000 Documents Counts<br>Accumulated Documents Counts<br>200000<br>15000<br>20%<br>10000<br>100000<br>5000<br>0<br>0<br>0 10 20 30 40 50<br>Pages of One Document<br>Documents Counts<br>Accumulated Documents Counts<br>**----- End of picture text -----**<br>


Figure A.1: Page distribution of DocGenome. 20% of documents are five pages or fewer, 50% are ten pages or fewer, and 80% are nineteen pages or fewer. 

Table A.1: Category descriptions of the layout annotation performed by our DocParser. Note that we do not use the “others” category and the “reference” category, and their indices are 6 and 11, respectively. 

|**Index**|**Category**|**Notes**|
|---|---|---|
|0|Algorithm||
|1|Caption|Titles of Images, Tables, and Algorithms|
|2|Equation||
|3|Figure||
|4|Footnote||
|5|List||
|7|Table||
|8|Text||
|9|Text-EQ|Text block with inline equations|
|10|Title|Section titles|
|12|PaperTitle||
|13|Code||
|14|Abstract||



## **C Annotation Explanations** 

We provide the annotation details of DocGenome in Table A.1, where the index number in the annotation corresponds to the category index in the attribute list. 

## **D More Statistical Distributions of DocGenome** 

In addition to the statistical distribution described in Sec. 3, we provide more statistical distributions in this section. As shown in Fig. A.2, the sample counts of all secondary disciplines are summarized and marked with different colors, from which it can be observed that the inter-discipline and intradiscipline distributions are both diverse, with Physics, Computer Science, and Mathematics papers occupying the major components of DocGenome. 

> ‡The version of the online API we used for evaluation: `https://mathpix.com/equation-to-latex` . 

§ Online API we used for evaluation: `https://mathpix.com/table-to-latex` . 

2 

We also present the page distribution of DocGenome in Fig. A.1, which indicates the diversity of paper length in DocGenome. Specifically, 50% papers in DocGenome have nearly or fewer than 10 pages, with 80% papers having fewer than 19 pages. 

## **E Details of Quality Assurance for QA Data** 

**The QA Generation Details.** We provide a general prompt template for QA pair generation in Fig. A.3. The discipline-specific guidance is imposed to generate the corresponding ground-truth labels to achieve diversity and relevance. 

**The Quality Checking Details.** During independent verification by professional faculty members, each judgment was assigned with a confidence value ranging from 0 to 3. The confidence criterion is designed as follows: 

**Confidence 3** : The reviewer is confident that the QA pair is accurate and relevant to the provided paper. 

**Confidence 2** : The reviewer thinks the QA pair is mostly accurate and relevant to the provided paper but is unsure whether it is absolutely correct. 

**Confidence 1** : The reviewer has no idea about the correctness or relevance of the QA pair to the provided paper. 

**Confidence 0** : The reviewer is confident that the QA pair is wrong or irrelevant to the provided paper. 

During the cross-verification, the confidence values of the two professional faculty reviewers were compared with the automatically-annotated correctness. The QA pairs with inconsistent results were re-analyzed by the two reviewers and updated to a precise version with consistent confidence. 

## **F Prompt Design for GPT-acc** 

We adopt GPT-acc as the evaluation metric for the QA tasks. The complete prompts are concluded in Fig. A.4. 

## **G Examples in Document-level Annotation from DocGenome** 

We present one example in DocGenome in Figs. A.5, A.6, and A.7 to visualize the annotations of each page in a whole document [41]. The blocks marked with different colors refer to different attributes of component units and the arrows with different colors denote different relations between units. 

## **H Examples of Tasks in DocGenome-test** 

We provide visual demonstrations in Fig. A.8 for all 7 tasks in DocGenome-test, including document classification, visual grounding, open-ended single-page and multi-page QA tasks, document layout detection, Equation-to-L[A] TEX transformation, and Table-to-L[A] TEX transformation. 

3 

**==> picture [397 x 595] intentionally omitted <==**

**----- Start of picture text -----**<br>
Computer Science<br>Economics<br>Electrical Engineering and Systems Science<br>Mathematics<br>stat.MEstat.MLstat.OT PhysicsQuantitative Biology<br>stat.COstat.AP Quantitative Finance<br>q-fin.TR Statistics<br>q-fin.ST<br>q-fin.RM<br>q-fin.PR<br>q-fin.PM<br>q-fin.MF<br>q-fin.GN<br>q-fin.EC<br>q-fin.CP<br>q-bio.TO<br>q-bio.SC<br>q-bio.QM<br>q-bio.PE<br>q-bio.OT<br>q-bio.NC<br>q-bio.MN<br>q-bio.GN<br>q-bio.CB<br>q-bio.BM<br>quant-ph<br>physics.space-ph<br>physics.soc-ph<br>physics.pop-ph<br>physics.plasm-ph<br>physics.optics<br>physics.med-ph<br>physics.ins-det<br>physics.hist-ph<br>physics.geo-ph<br>physics.gen-ph<br>physics.flu-dyn<br>physics.ed-ph<br>physics.data-an<br>physics.comp-ph<br>physics.class-ph<br>physics.chem-ph<br>physics.bio-ph<br>physics.atom-ph<br>physics.atm-clus<br>physics.app-ph<br>physics.ao-ph<br>physics.acc-phnucl-th<br>nucl-ex<br>nlin.SI<br>nlin.PS<br>nlin.CG<br>nlin.CD<br>nlin.AO<br>math-ph<br>hep-th<br>hep-ph<br>hep-lat<br>hep-ex<br>cond-mat.supr-concond-mat.str-elgr-qc<br>cond-mat.stat-mech<br>cond-mat.soft<br>cond-mat.quant-gascond-mat.other<br>cond-mat.mtrl-sci<br>cond-mat.mes-hall<br>cond-mat.dis-nn<br>astro-ph.SR<br>astro-ph.IM<br>astro-ph.HE<br>astro-ph.GA<br>astro-ph.EP<br>astro-ph.CO<br>math.STastro-ph<br>math.SP<br>math.SG<br>math.RT<br>math.RA<br>math.QAmath.PR<br>math.OC<br>math.OA<br>math.NT<br>math.NA<br>math.MG<br>math.LO<br>math.KT<br>math.HO<br>math.GT<br>math.GR<br>math.GN<br>math.GM<br>math.FA<br>math.DS<br>math.DG<br>math.CV<br>math.CT<br>math.CO<br>math.CA<br>math.AT<br>math.AP<br>math.AG<br>math.AC<br>eess.SY<br>eess.SP<br>eess.IV<br>eess.AS<br>econ.TH<br>econ.GN<br>econ.EM<br>cs.SY<br>cs.SI<br>cs.SE<br>cs.SD<br>cs.SC<br>cs.RO<br>cs.PL<br>cs.PF<br>cs.OS<br>cs.OH<br>cs.NI<br>cs.NE<br>cs.NA<br>cs.MS<br>cs.MM<br>cs.MA<br>cs.LO<br>cs.LG<br>cs.IT<br>cs.IR<br>cs.HC<br>cs.GT<br>cs.GR<br>cs.GL<br>cs.FL<br>cs.ET<br>cs.DS<br>cs.DM<br>cs.DL<br>cs.DC<br>cs.DB<br>cs.CY<br>cs.CV<br>cs.CR<br>cs.CL<br>cs.CG<br>cs.CE<br>cs.CC<br>cs.AR<br>cs.AI<br>10 [1] 10 [2] 10 [3] 10 [4]<br>Count<br>Secondary Discipline<br>**----- End of picture text -----**<br>


Figure A.2: Distribution of secondary disciplines in our DocGenome. The count on the x-axis represents the number of documents, and documents from the same primary discipline are marked with the same color. 

4 

**==> picture [324 x 219] intentionally omitted <==**

**----- Start of picture text -----**<br>
QA Generation Template<br>Assume you are an expert in the analysis of arxiv papers. Based on the input images of the paper, design a pair of<br>questions that are slightly difficult, are frequently asked in related categories, require understanding of different<br>pages to give an answer, can be answered from the original paper.<br>Each answer should not contain any hints, explanations, or notes, etc.<br>Make sure your answers are accurate. After you generate the questions and answers, perform one or two self-<br>checks to make sure your answers are correct.<br>Design questions as clearly as possible, give answers as succinctly as possible, and avoid summarizing narrative<br>questions and answers.<br>The questions should be in the form of a question-answer pair.<br>Make sure the answer to the question is taken directly from the original text, not from your summary and make sure<br>answers are as short and direct as possible.<br>Here are some simple examples ：<br>1. Q: What are the two experimental measurements from HERA that are combined and used to determine the proton<br>distribution functions HERAPDF as mentioned in section 3 HERAPDF?<br>    A: H1 and ZEUS<br>2. Q: What are the two main types of deep inelastic scattering experiments discussed in the paper?<br>    A: Inclusive and semi-inclusive<br>3. Q: Does the Mercator model allow for the adjustment of node degrees to match the expected degree sequence in<br>a network as part of the embedding process?<br>    A: Yes<br>4. Q: According to Figure 2, what is the name of the region where the solar wind flow is deflected around a small<br>magnetic obstacle or \"bubble\"?<br>    A: Narrow barrier region<br>5. Q: What was the cross-validation relative absolute error percentage of the Kstar model used for predicting fatal<br>police shooting rates on the state level as mentioned in section 6.1?<br>    A: 28.53%<br>Please follow this format and give two pairs of answers to the questions.<br>**----- End of picture text -----**<br>


Figure A.3: Template prompts using GPT-4V [33] for document QA pair generation. 

**==> picture [323 x 300] intentionally omitted <==**

**----- Start of picture text -----**<br>
GPT-acc for DocVQA<br>Examples:<br>{<br> "query": "<question> What was the incremental increase in revenue from 2020 to 2021 ？ <groundtruth answer> 5<br>million $ <answer> 20\n</s>",<br> "answer": "False"<br>},<br>{<br>"query": "<question> What percentage of government spending was allocated to infrastructure in 2020?<br><groundtruth answer> 10% <answer> 14-4=10\n</s>",<br>"answer": "True"<br>},<br>{<br>"query": "<question> What is the total production of Wind Energy in the four months from January to April 2021?<br><groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four months from January<br>to April 2021 is 2450 MW.",<br>"answer": "False"<br>},<br>{<br>"query": "<question> What is the value of baseline distance L for the DUNE analysis mentioned in Table I?<br><groundtruth answer> 1300km <answer> The value of baseline distance L for the DUNE analysis mentioned in<br>Table I is 1300km.",<br>"answer": "True"<br>},<br>{<br>"query": "<question> According to the caption of Figure 5, what is the fixed value of M_N1 used to predict the<br>relic density as a function of m_η? <groundtruth answer> 200 GeV <answer> The fixed value of M_N1 used to<br>predict the relic density as a function of m_η is 200 GeV.",<br>"answer": "True"<br>Instruction:<br>Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of predictions.<br>The output should be only "True" or "False"<br>Input:<br> f```<br>   <question> {question} <groundtruth answer> {answer_gt} <answer> {answer_pred}<br> ```<br>**----- End of picture text -----**<br>


Figure A.4: Detailed prompts in GPT-acc metric for document QA tasks. 

5 

**==> picture [397 x 636] intentionally omitted <==**

**----- Start of picture text -----**<br>
Visualization of Annotations in DocGenome<br>Page 1 of 10 Page 2 of 10<br>SSS‘attention Is All You Need = finsarchitecturessequences.FEGASTTCTONReamavAT snsconandmOdeTsAligningPOBIETAS(38) alTypicallythe2475)SUCHpositionsTactoraSpateTaBUAGE compRatiopto  oundeps MHOUETNEZ|in emputationalongofAeen theMHACHEsymboltime, languagetheypositionsWaASTATOWgeneratemsoF (35 endaThe Mapusequence2}. nodesandNUWTETOUSof outpuydoer hidden}<br>jstates /iy, as a function of the prevjais hidden gate h,_, and the input for position t. This inherently<br>Jcquentiallequence lengths,nature precludesas memoryonstraintspargRization limitwithfnpa t chingaining examples,across examples.whichRecentbecomesworkertih a ls a tchievlong e dr<br>ignticamt improvements jyrComputational effiqency through factorization tricks (21) and conditional<br>Ashish . . . . |computation (32), whilg/lso improving mode performancein ease ofthe latter. The fundamental<br>Google VaswaniBrain NoamGoogleShazebr’Braj GoogleNiki Parmar’Research JakobGoogle UszkoreitResearch constraintof sequeng a f computation,showever remains.<br>avaswanigoogle.comLiendgoogle.comGoogleLionResearchJones" noam@google__aidancs.tronto.eduUniversityAidan N. qfGomez"fromTorontonikipOgoogle.com| _—LukaszkaiserOgoogle.conLukaszGoogleusz@google.com Kaiser* Brain ition[fettentiontinelareet input usedffAngmodelsworkentirely ortputmechap&msjconjunction weinfarious proposeon sequencesan attentiontasks,withhavethe allowinga recurrentbecomeTransformer,mechanismanIn integralmodeling netwoHkall butato njodelpartfrawfewofdependencies casesarchitecturglobalof compelling (27),dep e sequencendencieshowever, sucheschewingwithout regardbetween recurre attentionmodeling to theiri n ceputand mechanisms distanceandandtransduc-|insteoutput. in<br>i11ia. Ilia Holosukhin® ! he{translationTransformerquality allowsafter beingfor significantlytrained for moreas littleparallelizationas twelve hoursandoncan eightreach P100a new GPUs.state ofthe art in<br>polofukhindgnail.com ‘BleBeckground<br>PPconvctalionalia nan eal Sequence exlwcictsTan slucloWestsAbstracttiederesmodels Greased cocator ad'son COMGION teoon ROTEATTWie taal y phelock,the{16},numeri=BYteNetebmputing‘of ofreducingoperations (18) hidden and ConvS2s requiredrepresentatignsYequential (9),computation to reve.allinofwhich parallelsignalsalso use convolutional from forall inputforms twothe  arbitraryfoundation and outputneuralinputof the networks position.or Extended outputasInpositions basic these Neural building models, GPUgrows<br>rerfonaiagISSA, models also cormoctthe encoder tad docoaer though 2 atten 7 in the diftance between positions, linearly for GonvS2S and logarithmically for ByteNet. This makes<br>inesscibsiyspoley/onWatwpone wearer marhatoek an smukac a oyiek:rcis/toracetegth n eeiwa,carvench te Tuagpiofoaeonan itreducedfo more fificulta constantto learnnumber dependenciesof operations, betwenbe distanthe t  positions costa of reduced(12). In effective the Transformerresolutionthis duc is<br>beless [superior] Econchnentsieain quality while totais being mre parallelizabletansiniva tas ange€@uiring uneotiee: signijtCantlveal el todescrib. averaging attention-weighted positions, anfeffect we counteract with Multi-Head Attention as<br>to-Gerensembles,jourtrainingtagcat timodel m eanfortoestablishestrain.translationby over3.5 daysOur2 BLEU,a modeltask,eightnye single-ryed€lffimpe6%vingachightstheGPUs,WML26T4a small over28.4state-of-the-artfractionBLEWsftheEnglish-to-Frenchthe-Existingof theBLEU,bestWMT resulyéincluding|sfaininggfanslation201¥English: costsof 41.8 of [the]] aftertask loflused{textualBeiE-aiterfion,=  a singhsuecfssfullyenfailmentsequencesometimes inandain variety learning ordercalledtoof computetask-independqht tasks includinginira-altentionfisa readingrejfesentationsentencean=  attentioncomprehension,representationsof mechanism the sequence. relatingabstractive(4)[27)[28)[22)Self-attention different summarization,positionshas been=<br>ether tata$egAGeyaPobghae  taperEe ectenctullyWe thonso oat agliGos  ameteceeToate oonitansparting teak lllll alignedBad 1oex{T [reurrence] memory [and]  networks [ have][ been]  ae [shown]  based [ to]  on [ perform] afecurentaienton well [on][ simple-language]  mechanism [question]  Tnstead [ answering] of sequence [ and]<br>lange and jufitgatfaining data. language‘the be# thodetingof our tasksknowledge,(34), however, the [ransformer is the first transduction model relying<br>entirely on|self-attention to compute represent{tions of its input and output without using sequence-|<br>‘Wietnefad laligned‘el-attentifaR N s andor [convolution,] discuss its advantagesIn the followingover modelssections,suchweaswill(T7)(18)describeand the(9) Transformer, motivate<br>Redurrent neural networks, long Sttorttemm nygfaory (13) and gated recurrent (7) neural networks tle aA<br>jinthehasattentiondetail.tensor2tensor particular, been=EaqualeffortNikicruciallyandthetocontribution, evaluatedesigned, have beenLionparameter-freeinvolvedalsothisimplemented,Listing experimentedidea,firmly establishedin every aspectAshish,ondepositiontunedrandom,withwith ofrepresentation and Ia, thisnovelevaluatedasJakob designedwork.modelStateproposed countlessNoamofandvariants,theand becameproposed implemented art replacing modelapproachesthe scaled dot-product variantsotherRNNsthe fistperson withinin sequence our Transformerinvolvedslf-attention original attention, modelingin codebasenearly and m odelsul-hea statedevery an andai d BieModiRaFeelotlsequenceits,continuoys ERStis (yf packersexkoder....NSrepresentationsjm) of magescarat  symbolsknSaEeedopa2 an one=eeheeaie(21,024). elemenijatSWof symbolaGiven time.ST repestanttices AtRS2,  eachtheWE decoder step theey thenWises modelSee generates ee)is auto-regressiveto poeX anSeence output<br>leatTegan+‘Work Work inftvace performedperformed [ar] nd vinetzaons.whilewhile [ene,] atat [Ppbatin]  Google GoogleLaan Bran Research.One oe ee Aldeaaay apod remy continapocrinewas responsibleJog  eetdays Seilgsiog fr oureset inital vrlouser codebase, pcs fendSeer and Bileconnectedrespectively.ith.Fhe Transforiper consumpEncoderlandlayprs forthefollows revieusyDecoder both this the Stacks encoderoverall generated and sympols dedoder,architecipre using stacked as shown adtonal in the input sel-atiention left when and night generating and halves point-wise, the of Figure] next full<br>$1st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.<br>@ = identical @ = non-title adjac @ title adjacent @ = implicitly-referred @ = explicitly-referred @ = subordinate<br>igure ProbapiltiesOut Sealed Dot-Product Attention Mult-Head Attention<br>(Sonmax”) || 3)<br>— a {!<br>Foard Beale ay we<br>Foowardrood ssMatierin Head Ne lAYy/<br>Nx =)Ma ead —Matransaie a,Head ipodattention layers(ety runningSealed inDor parallelPadua Atgpty Raton consis of Sever<br>Positional € >, Positional<br>neoang QO-Pus enbekigQO Encosing FRintofaprtigice, we compyy?a the$1attentionpe Va3 7 aT friesTw simultaneously,anpacked WIENStogether| OT<br>Inputs Outputs lthe matrixa matrixof Q.ouypfitsTheeysas: and val ef into matrices KC and V. We compute<br>Figure1: (ietBransformer -\owtedeh model architecture, Faterention| (Q, ° KV) ==NJsoftmsot aOXer yy a)<br>[RowecRIQe:lsub-layers.ithelayers,|sub-layerslattentionlaroundlsub-layerILaverNorm(zitself.ise twofullyTo eachproducesub-layers,overfacilitateinconnected—TheinThe of eachthe the + fisttheoutputs output Sublayer(2)),decoderdecoder encoderthesesub-layers, isfollowedfeed-forward a of theof multi-headresidual dimensionstack layer,also by encoderfollowedtoComposed layerwhere theconnections,preventnetwork,self-attention decoderdost Sublafer()stack.normalfzation byOTpositions a=laferWe;nsertsnormalization.P12.flackSimilarhil employ mechanism,sub-layersOF ifromtoa{i}. third theV theaTC= encoder,attending functionThatresidualinsub-layer, andtheis,Wdentcal model,We connection wethe implemented bythe alsoto second employsubsequentoutputTayers.which modifyas wellC“C*é‘CSC(S isa residualperformsofIn{Li} as the each simple,the,additi around the mul embeddingsub-layercy sub-layerT position o  each ofn offTheis [dc,CTT]ilmuchmatrixlp...WHIEGor singlePhatexoelyot producttheGphost faster dotmultiplication hidden©)Additivesmall attention, sual productscommonlyandatiention values more ginal]layer. attention growDot-product without space-efficientcode.ofWhile theusedlargedj computesTothescaling atention cowntornctthigettfct,in attention magnitude,twotwo are mechani for thein practicl simil larged compyfbilityFunctions is idenyalFshjhgFeatvalfssay/ fit weformaditivetheofbreticalfunction dacantour softmax acasimilarly, algorithm,be(2. attention complexity, implemented the usingWe dat prococisfunction suspect additivea except forthe feed-forward[2],intodot-product using highly and that attention regions by fordor-product he large scaling network whereoutperforms]attention optimized values factormali} withit has} isof}<br>‘Badamasking,RadpredictionshWet? \_—_——_ScaledcombinedontionersforWhereTunetioncorresponding—= ey, valspositionEradutwiththe iweight canfactIPBEdeSeibe Atenion dependbythatandassigne ofpthe onlyoutputa Mapping onembeddings thevessaeetsknowna queryoutputsW,are anaesygfx6aoffsetgas atbybypositions computedaone compatibilitypositivale 1p aspaisth  weightedFunctionA a OUT}of suthea llinearloutputsubspaces——E—EOeEvrOqueries,Jepicted=yeGi  foundE@ofprojectionsvalues. keysbain performingitat aon Figurebeneficial andThesedifferent valuesto dy, aretopositions.allows wedylinearly concatenated thenandie d,Withperformpre:dsdimeiisieagg joya and the OOsingle fttentionPpHries, aetentionofcerespectively. againWithkeysoTafoaton deosa-dimensionalfunction projected,head,andOn averagingvalues eachin parallel,resulting/ owoftimestheseinhibits yieldingKeys,withinprojected= the valuesTeathis.different, d,-dimensional finalpesandversionsvalues, queries)learnedofas| |<br>\queries andour keysparticular of dimensionattentiondy, and"Scaied-Dot-by@uct valuof dim e nsionAttention”s d,. We (Figure). compute theThedotinput products consistsof theof] FootTotitstratevariables with meanwhy the0 anddot varianceproducts1. Thenget large,their assmedot product, that theq-componentsk= So! ofquki, @ andhas& meanare independent0 and variara n don}cedi<br>© identical © @ non-title adjac @ title adjacent + @ implicitly-referred © @ explicitly-referred © @ subordinate<br>Page 3 of 10 Page 4 of 10<br>Figure A.5: Annotations of a complete document in DocGenome, taking ‘ Attention is All Your<br>Need ’ [41] as an example.<br>6<br>**----- End of picture text -----**<br>


**==> picture [397 x 639] intentionally omitted <==**

**----- Start of picture text -----**<br>
Visualization of Annotations in DocGenome<br>Page 5 of 10 Page 6 of 10<br>Rauefiead(Q,K,V) = Goncat(head),..., head)W: ‘TabRA‘sizefor differentof convolutionsMaximumlayer types.pathand lengths, rn theis thesize per-layersequence of the  complexityneigifborhoodlength, d isand thein minimum restrictedrepresentationnumberself-attention.dimension,of sequentialk is operations the kernel<br>where head,| == Aptention(QW?,Atention(Qw?, KWKW. VW!VW!) Self-Attentionlfer- Type ‘Complexityraiin fpri Layer OperationsSequentialOU) _MaximumOm)Path Length<br>aK - ae supe Recurrent O(n-B) O(n) O(n)<br>— - Self-Attention (restricted) O(r-n|a) ou) O(njr)<br>‘AQSieeeisiisfetikDworksimi= lasdy  wo= demst/h thatApplicationsppfweof employ sinatten= G4. DueAttentionhead.ha— attentionto§ parallelinthe ourwithahModelreduced fullFulaedimensionatifntioniseeedimensionallayers,of eachor heads.head, theFortotal each computationalof these we usd BeMtGRAllearnedss theembeddings.andof thefixedcacoder(9)so that and thedecodertwo  Sacks.canvabeZZ)Thdsyggffed.positionalThere encodings are many choiceshave theofsamepositional dimension encodings,Gar<br>fpolnandpositionpica the2a“cncoderfdccoderin fhemerfory decoderr-decoderkeys attSgiion”toandNalues atfugd overTayers,comemechanisms all the from positions in the queriestheinoutput of sequence-to-sequencecome Trom input the the sequence. encoder. previouThi Thismodels s all wsmimicsdec o der suchevery the|Tayer]as| eS == e jnwhan(poe/aS10000 ui<br>‘Theand encoddf contains self-attention Yyers. In a self-attentionlayer all of the keys, values} (pos /100007/4—")<br>encoder.all‘encoder.‘Similarly, pose)EdBhsfif-attentionqueriesfcomeinposition the decoderfrom in thelayers the sameup inencoderto place andthe decoder including caNg attendthis case, the allow tothatalleach poston.positionspositionoutput We of theininncedlthe previoustheprevious decoderio poeveat tolayerlayer leftwardattend in of the| the| to correspondsFeiaiEhoseative thisGos pbsitions.is thetp sinusoid.becausesitionsince forandThe anywe | Iswavelengthshypothesizedfixed offset the  forfadimeasioh.k ftPE,..+1awouldThatgeometric is, allow cachcan theprogression dimensionbe representedmodel to from of easily the as2x apositional lineartolearn to 10000 func ion at encoding t 2x.end We|byof|<br>‘ofinformatioinside the ofsoftypaxsopled flowwhichdot-productin the decoderattentioncorrespond totopreservebyillegalmaskingconnections.the auto-regressiveout (settingSee Figure]toproperty.—7c) al valuesWe implementin the inputthis ‘WisePE, onsexperimentedproduced nearlywithidentical using learnedresultsposifional (sep TablefS}rowembeddings(E)).(0)Weinstead. chose theand found sinusoidalthat the version twa<br>‘3TeddditionPositon-wiseRiecd.to attey Forward“eaakiof Networks the layers in our encoder and decoder contains a fully PecatseWSRing trainingmay allowAiieation the model to extrapolate fo seqrnce lengths Jonge than the ones encountered<br>Bepation—max(0, ri7 + HW 2 + be %B jlayerposalF1)~-#q) inayersa typicaltO commonlyanothersequencesequenceusedtransduction [Sipping] for of equal codencodgr variable-lengthTength or(=;,...,2,).decoder. Motivatingwithscocnce,,, ourof symbol€useRY,of self-attentionsuch rpresatonsas a hidden|we |<br>‘Bdelfromdys he dimensiofality=layerEs2048.c to—fayer.. ofAnother input andwayTRUEoutputof TEdescribyefis  NTOdng this= 512,is asandpositwo the© BE pd yAerneldimensionality size | lability|traverseRisa‘dependenciesbbe parallelized,Wandsto learntieis asuchaspa keymeasureddependencieschallengeBBetweenby  in manytheisminimum long-rangethe lengi sequfnceofdependenciesthetransduction pathsof sequentialforwardtdtasks.operationsandOne backwardork.  key factor affectingLearningrequired signalslong-rangahave the to,<br>SGRTEDYjokens ionmodel.andandTo oe pierfoftmax sharedistput tokenssequentfunctionths seteto vectorsGapweightjoe6 matrixPodlfensiontheHRI,dings.betweenWecoder Weoutput TEtheWe TERTalsototwymbeddingpMicted uythecaps usualnext-tokelayers lear andWo n  CORNETed linearprobabilithe pre-softmax|TRE t ies.ransfor TapIn| land[theMifferentSURG maximum outputinlayercs thesequences,TEN types.path network. lengthvaltheThehice betweeneasicr shorteritisLSany two theseto learncoma lfg-rangeingutpats  andbetweenal outputdependenciesanypositionscombination ([3}.in networksofHenceewpositions wecomposed alsoin the compare of input thear<br>‘38° Poslema omputational complexity, self-attention layes are faster than recurrent layers when the sequence<br>cocokenssin in the sequence.Aaigtnaiaaconmpenletin-estTo this end.teuwe add “positionalstain encodings”ceepeegaibaichaardugto [the] input embeddingsreeds atpohthd Bolseryentencelong1 represis seqsmaller e ntationsthan‘self-attcusedthe rby e pres stat -of- he-artcould e nta bieee te ionntrestr cted i modelson to inconsideringmachined. whichtranslations,only isa mostneighborhood oftensuchaa the as of caseword-piecesize with|r in<br>© identical @ nonttitieadjac @ titleadjacent_ © = implicitiy-referred © explicitly-referred  @ subordinate<br>ffextigatBathPER  tenthTAT convolutionalSELIETCEto Ofn/r)CENTETETlayerWe AFOOTplantwil TNEimexizate“widTEP ieF <napproachSUT does notPONTE uberconnect inTS faureall pairsWIT workof inputrE andWa outpat] abREEngin Modalo-TheGemTransformer a n nEnghachieveso-Frenchbetter BLEUnfs scoresBLEUthantt previous a fraction state-of-the-art‘TrainingoftheCost using con(FLOPS)models on the<br>Ipositions. Doing so requires a stack of O(n/k) tonvolutional layers inthe case of contiguous kernels, NDE EN-FR EN-DE EN-FR<br>or O(logs(n)) in the case of dilated convolutfons {8}. increasing the length of the longest paths ‘ByieNet O87 7S<br>jrecurrentconsiderably,between anyeeeolutionlayers,istwo equaltopositionsbyO(ktoa factor the combination-n-d-+n-d?),in theof k. network. SeparableofEvena seff-attentionCofvolutional ithfonvolutions  k = n,layerlayershowever,6}, and aare generallyhowever, point-wisethe complexitydecreasemorefeed-forwardtheexpensiveof acomplexity separable| than|layer, Deep-AttGNMTConvS25MoE 62)+RL(9)+ PosUnkG3)  (39) Fs0351646 4046405639.92392 23-109.6.10!"-2.0-10! 14-10”1.5.101.2.10-1010%<br>from h eads  ourbenefit,models self-attentionand presenteRNS,couland d  yieldiscuss moreexam ple rpretable modelss in the appendix . WeNot inspectonly doatte i n tiondivd i stibutdualattent Conv$2SGNMTDeep-Att+ RL EnsembleEnsemble+ PosUnk Ensemble9) 68} (39) fo36F630 40441.2941.16 7.710!18-10 80-10_1.2-1.1. 102"10%<br>nd clearly lear to perform different tasks, many appear to exhibit behavior related tothe syntactig TeameSiriana (pana oa 7338 ‘33-10<br>‘BiieQkainingsemantic structure of the sentences ‘Transformer (big) bs 418 23-10"<br>(Eis sect NG escriber Ths tanng regime Tor ou moe] BrResults<br>‘Biles fatirained pairf.ofTrainidyjData|theSentences —~ Sandiand BatchingWMT wer?eacoded2014 usingEnglish-German byte-pair encodingdataset consisting(3), which hasof abouta shared 45 source-|million) ‘aDRUROWHITje Matpine—2014TranslationEnal pi task, the big transformer model (Transformer (big)<br>batch014cabularygetet Englishfvocabubirycontaiged(i).French dataset Sentenceaofsetaboutof sentencepairs37000consisting werepairs batchedtokéti»-For ofcontainingEnglish-French,36M togethersentencesapproximately by approximate sequenceand wesplitused25000tokens the significantlysourceinto length.a tokens32000Eachlargerwordt WMTpies in Tablef2]IBLEU,istepassesin of hilloutperformsexjablishingboompreviouslyinea newthe bestofpublished state-of-the-arRLEU Table]p  Trainingmodels andfknsembles, Wokpried score3.5models daysofata28.4.(includingon fraction§ P00 The configuration of GPUs.ensembles)the training Even byof thisourcostmorebaseofthanmodel modelany2.0 isof]<br>‘RiebogeitramerHardyortokenparametershasepesour modelsMOdEISdescribed for OTtotal ofthrdwahgutIS WII1 0 , 0 the6-steps 8paper,NVIDINoF each12 hours.trainingPTO GPUSFora bigFPstep¢6ok  models,(deseribedCaraboutBNE0.4 seconds.THOUS TIEon thdWa dropoutCoRAREW>Fo‘compfitiveperforfringBa rfteHITfoee makdhte-of-the-attPropBOTTallmodels ofthe= 0.1,English-to-Frenchwe previously wed insteadmodel. a tucksTheofpublished0.3.Transfolfnerwalwanslatif ingletask,(big)models,tuandihyour bigmodel modelat lesstrainedweushp than achieves for1/4 English-to-Frenchdelathea BLEUPahatraining scorecost of ofused| thewi 1 .0]<br>fie; of abe, pte was 1.0 seconds. The bg gels were trained for 300,000 step ifn at IO-munute ntcrvale: For the Ug mandlarwe sveenped the lat 20 checkpoints Wi<br>3 ed bean search with a beam size of 4 and IEngth penalty« = 0.6 (38). These hyperparamete<br>soaph ——Pome [wae] fc anuies aahofperleatin DOatelywitha ACNaRformula:artshy = 0.98 and <= 10", We varied the learning jarchitectpresinferencd=ere chofenPspmimarizesto afterfrominput lengthexperimentationthe 0 literature.+ 50, butcomparesWe terminaton the estimatedev ofr e  theefrlylfpment numberSStranslationwhen set. ofWe floatingpossiblequality set the(38)andpointmaximum training costsoperationsoutput used lengtho otherto traindurinmod a<br>Eawation/05 - my -p_num- warmup_steps~**) 3) jmodel pifcisionb multiplying floating-pointthe training time,capacity  theof nfmbereach GPUP})of GPUs used, and an estimate of the s utained<br>——<br>bet‘tilea DgarizatichrsJs =thereafterTopAreasing4000. proportionallythe Teaming to the inversetatofincarly Torsquarethe Histrootwarmup_stcps of the step number,WainingWe usedSteps] [faifable[Sjrowsin differentcheckpointevalua jelopmentaveraging.ways,heset, importance newstest2013,measuring We presentvFdifrentthe Wecltheseused comonens restbef{1}ferformance insear Table oftheas desonTransformer, English-to-German c ribedh in thewe previous variedtranslation section,ourbase onbutmodelthe<br>heceping (A), we vary the number of aifestion heads and the attention Key and value dimensions<br>hurts[RegidaDI[Kaba|positional[Paropb-layerSmoothing= 0.1.inputDropoutencostingsand normalized.DuringWein  applyboth thewaining,WOpaqgInencoderadditiOMmaygwe [IM empfoyedandto Theapply outputPecoder labeldropoutstacks. smoothingoT cach toFor thethesabsums baseofayer,valueof model,Petorethe q, embeddings Ws=we0.1 useae(6).a andrateTheThi theof| huuggestsfunstionigger.9 BLEUERmodels thanthatrowworsnounofdot s determiningaree than(B), beter. andproduct thewe compattion ob bemaycompatibility s t beervesetdropout beneficial] t ing,hat onais veryredudingif helpful Wequalifyntfodescribedthefurthereasy attendropsin  observevoidingand offinthat wiSectionKeyover-fitting. in t ha sizeitomorerows . o  Whilenmany (C) andd,sophisticated hurts heads.Insingle-headrow(D)model(E) that wecompatibilityquality. attentionasreplace expectedThisouris<br>perplexity. as the model learns to be more unsure, but improves accuracy and BLEU score.<br>© identical @ nontitie adjac —@ title adjacent @ —implicitly-referred @ —explicitly-referred + @ subordinate<br>Page 7 of 10 Page 8 of 10<br>Figure A.6: Annotations of a complete document in DocGenome, taking ‘ Attention is All Your<br>Need ’ [41] as an example.<br>7<br>**----- End of picture text -----**<br>


**==> picture [397 x 318] intentionally omitted <==**

**----- Start of picture text -----**<br>
Visualization of Annotations in DocGenome<br>Page 9 of 10 Page 10 of 10<br>(TabRI@) Variations on the Transformer architecture, Unlivalue s aret W e nicald to those ofthebase san<br>per-wordperpleitiespetplexitics.are per-wordpiece, according to ofr byte-pair encoding, and should not be compared te SieqEAOrmedsection }, learning onlyarates small andnumber.beamofsizexpPrnfnts L] Sectionto select22 thedevelopmentdropout, Bothset,attentionall otherandparamete resid}<br>“ly ain & a a a PL BLEU params rsaied-mained tonesunchanged fromvotes:the nap Engl mangh» base300,translation We used abeammodel,  saeDuringofinference, anda = 0wd<br>bem |[6512208 5 Se  SHON2612aa.2216 3I ON stepsOOK [492258| (dev)491a:Soleee(dev)258254- _x10°6s [forbrisingly‘RecurrentGut—Sdonirast resisbothWSJwell,Neuraltoin RNNonlyyieldingTHbLENetwo andseghPT thebey— semi-yApgatencea— mofels| SCIkreviouslying of[STOYTaS-apecINGtask-speciicthereported Transformer modelstuningTanta  outperforms with our modelOuFWGK] the exceptiperfthePETOERE Berkeley’ o rmsn—of sur] SO the<br>16 $16 381 38 Parser (29) even why ‘on the WSI training set of 40K sentences<br>:z 32 GiiSo l se 2 5437 60.OS36 ‘WieGonclusigte SoXecciia7 7.the Fist sequence Wansduction entirely0<br>es a3 i co. Of i Arypmmonly used in encoder-decoder architectures with<br>1028 128 12 466 260 168 i pafited J<br>hd 40961024 0002 0002D 475262495467$12377246pyle255254253 0 53 pay[phelish-to-Frenchonfecurrentad!fare inslationoutperformsexciiedof abouttasks,convolutionaltranslationeven thethe futureall Transformer tasks,layers.ofpreviously attention-bafed weOnreport!can achiefebofhbdtrainedensemblesmodelsWMTa new significantlyandstate2014 oftheEnglish-to-Germanplantoat.faster applyIn  themthanthe former architectures toand other tasks. taskWMT ourbased2014 best We<br>[vigREET(E) | 60gpositional5096 embedding 6instead of sinusoids 0K |Sig492LZ257, 7s)Z| iplanitouch investigate to ascode extend images,welocal,the audioused Transformer restrictedtoandtrain video. attention toand problemsMakingevaluate mechgnismsgenefption invplvingofr model less toinput efficiently s equentia andis avai able output l  ishandle ano modalities othera t her re largehttp s  inputsearch: //github. com goals than and text outputsofous and|<br>jof [WS)] Tee eA We ETSE paAI Ho SONY Jensorfow/tensor2tensor :<br>ieee Kane cr QOL ETP werent heen | a pas eaknowledgementsymments, correctionsandWe are gratcTulinspiration. to Nall Kalchbrenner and Stephan Gouws Tor their Waitt<br>Petrov et al. (2006) (29) ) WS! only, di 904<br>Zhu et al. 2013) G0) WSJ only 904 References<br>[restorerDyerZhu et al. (2016)Taverns)(8) [weWS! ptyoat discriminative [313917 1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey=  E7 Hinton. Layer normalization. arXiv preprint<br>etal. 2013) EO Gpmiamon z ‘arXiv: 1607.06450, 2016.<br>poMensameekCuongDyerestoreRieeci etal.etal.eee20918)(OTS)(SlayesOO  elALIE _semteuperteed1 sensesearngenerativesmullt-tas V4 7S {3}[4][2] JianpenglearningDennymachineDemitry Britz, Bahdanau, translation Cheng,to align and Anna Li Goldie, Kyunghyun Dong, translate.architectures, and Minh-Thang MirellaCoRR,Cho,CoRR, and abs/1408.0473, Lapata. Luong, Yoshua abs/1703,03906,Long and Bengio. Quoc shor-term2014,NeuralV,2017.Le, memory-networksMassivemachine exploration oftranslationfor by machine jointly neural<br>reading. arXiv preprint arXiv;1601,06733, 2016.<br>raresouls ted srobase pe 7 ia a ee ae [5] and YoshuaKyunghyun Cho,Bengio, BartLearning phrase van Merrienboer, representationsCaglar Gulcehre,using Fethimn encoder-decoder Bougares, Holgerfor Schwenk,statistical<br>BO" English wast machine translation, CoRR, abs/1406,1078, 2014.<br>[Ree‘constituency{constraintsFrodslsvaluate heve ifandparsing.aotthe bossis TranstsignificantlyThis task presentcanlonger thanpecif tfe inpyf Furthermore,the‘Wwe performedoutput RNis subjectexperimentssequence-to-sequenesto strongonstructuralEnglist [6] preprintFrancoisarXiv:1610.02337,Chollet. Xception:2016.Deep learning with depthwise separable convolutions. arXiv<br>toolPennWBbieMhed Treebanka 4-layer(25),transformeroboutable  40Kto asia tainingwithsat-y.oaercf.scmtdees. | 024Weon  aloisicsubamapactn::the Wall trainedStreetit  Journalin a sem-supervisedsctng|(WS)E portion of the 17]{8} JunyoungofChris [gated] Dyer, [recurrent]  Chung,Adhiguna Caglar [neural] Kuncoro, Galgehre, [ networks] Miguel Kyunghyunon sequenceBallesteros, Cho, modeling. and Yoshuaand NoahCoRR, Bengio. abs/1412.3555,A. Smith,EmpiricalRecurrent 2014, evaluationneural<br>Jing the larger high-confidenceand BerkleyParser corpora from with approximately 17M sentences] network grammars. In Proc. of NAACL, 2016<br>© identical @ non-title adjac_—@ title adjacent ~—@ —implicitly-referred — @ —explicitly-referred_@ subordinate<br>Figure A.7: Annotations of a complete document in DocGenome, taking ‘ Attention is All Your<br>Need ’ [41] as an example.<br>**----- End of picture text -----**<br>


8 

|**7 Tasks in DocGenome-test**|**7 Tasks in DocGenome-test**|**7 Tasks in DocGenome-test**|**7 Tasks in DocGenome-test**|
|---|---|---|---|
||||**1. Document Classification**|
||||Q: Which discipline does this article belong to? Select the answer|
||||from the given options (quant-ph, physics.hist-ph, cs.CL,math.PR).|
||||A: quant-ph|
|||||
|||||
||||**2. Visual Grounding**|
||||Q: Please print the full content of the abstract section of this|
||||article.|
||||A: We consider indirect detection of meta-stable dark matter|
||||particles decaying into astable neutral particle and a pair of|
||||standard model fermions, Due to the softer energy……|
||||**3. Layout Detection**|
||||Title: [232, 448,1416,672]|
||||Abstract: [230,1430, 1469,1877]|
||||**4. Single-page QA**|
||||Q: What is the best result achieved by the HeunNet model for|
||||ECG heartbeat classification?|
||||A: 98.80%|
|babaibaPale<br>mi2ieia<br>{4|babaibaPale<br>mi2ieia<br>C4<br>{4||**5. Multi-page QA**<br>Q: According to Figure 5, what are the shaded yellow regions<br>indicative of in the power spectra P_cb for models M000n1 and|
||||M000n2?|
||||A: They show power spectra within 2% of the corresponding Time-|
||||RG curves.|
||||**6. Equation to LaTeX**|
|MIlj-.||(15)|\\begin{equation}\n\\begin{aligned}\n& \\|{\\bf A} - {\\bf<br>Q}_1{\\bf M}{\\bf Q}_2^T\\|_F^2 = \\|{\\bf A} - {\\bf Q}_1{\\<br>bf Q}_1^T{\\bf A}{\\bf Q}_2{\\bf Q}_2^T \\\\ & + {\\bf<br>Q}_1{\\bf Q}_1^T{\\bf A}{\\bf Q}_2{\\bf Q}_2^T \n- {\\bf|
||||Q}_1{\\bf M}{\\bf Q}_2^T\\|_F^2\\\\\n……\\end{equation}|
|Inception||FID||
||||**7. Table to LaTeX**|
|||11.27<br>7.67|\\begin{tabular}{| l | c c c c|}\n\\hline\nModel & L1 & MS-SSIM &|
|||8.49<br>15.30<br>15.66<br>14.62|Inception & FID \\\\\n\\hline\n\\multicolumn{5}{|c|}{Internal<br>benchmark}\\\\\n\\hline\nNon-exemplar & 0.018 & 5.05E-2 &<br>3.96 & 11.27\\\\\nReference & 0.014 & 3.97E-2 & 3.82 &<br>7.67\\\\\nCode & 0.015 & 4.15E-2 & 3.94……\\end{tabular}|



Figure A.8: Visualization examples of 7 tasks in DocGenome-test. 

9 

