## **MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations** 

**Yubo Ma**[1] **, Yuhang Zang**[2][∗] **, Liangyu Chen**[1] **, Meiqi Chen**[3] **, Yizhu Jiao**[4] **Xinze Li**[1] , **Xinyuan Lu**[5] , **Ziyu Liu**[6] , **Yan Ma**[7] , **Xiaoyi Dong**[2] , **Pan Zhang**[2] **Liangming Pan**[8] , **Yu-Gang Jiang**[9] , **Jiaqi Wang**[2] , **Yixin Cao**[9][∗] , **Aixin Sun**[1] 1 S-Lab, Nanyang Technological University, 2 Shanghai AI Laboratory, 3 Peking University 

4 University of Illinois Urbana-Champaign, 5 National University of Singapore, 6 Wuhan University 

7 Singapore Management University, 8 University of Arizona, 9 Fudan University 

## **Abstract** 

Understanding documents with rich layouts and multi-modal components is a long-standing and practical task. Recent Large Vision-Language Models (LVLMs) have made remarkable strides in various tasks, particularly in single-page document understanding (DU). However, their abilities on long-context DU remain an open problem. This work presents **MMLONGBENCH-DOC** , a long-context, multimodal benchmark comprising 1,082 expert-annotated questions. Distinct from previous datasets, it is constructed upon 135 lengthy PDF-formatted documents with an average of 47.5 pages and 21,214 textual tokens. Towards comprehensive evaluation, answers to these questions rely on pieces of evidence from (1) different sources (text, image, chart, table, and layout structure) and (2) various locations ( _i.e.,_ page number). Moreover, 33.7% of the questions are _cross-page questions_ requiring evidence across multiple pages. 20.6% of the questions are designed to be _unanswerable_ for detecting potential hallucinations. Experiments on 14 LVLMs demonstrate that long-context DU greatly challenges current models. Notably, the best-performing model, GPT-4o, achieves an F1 score of only 44.9%, while the second-best, GPT-4V, scores 30.5%. Furthermore, 12 LVLMs (all except GPT-4o and GPT-4V) even present worse performance than their LLM counterparts which are fed with lossy-parsed OCR documents. These results validate the necessity of future research toward more capable long-context LVLMs. 

## **1 Introduction** 

Documents are one of the fundamental forms of information preservation and exchange. In each year, tens of millions of documents are created, read, saved, and dispatched [1]. Beyond unstructured puretext, documents feature both complicated layout structures and information across distinct modalities such as text, table, chart, image, _etc._ Accordingly, the automatic understanding of documents (Document Understanding; DU) stands as a long-standing task in urgent and practical needs. 

Recently, a number of LVLMs, both closed-source ones (GPT-4o [2], Gemini-1.5 [3], Claude-3 [4], _etc._ ) and open-source ones (InternLM-XC2-4KHD [5], InternVL-Chat [6], Otter [7], LLaVANeXT [8], CogVLM [9], mPLUG-DocOwl 1.5 [10], TextMonkey [11], _etc._ ) have been developed and presented the great potential to handle documents. Most of them have achieved promising performance on single-page DU datasets like DocVQA [12], ChartQA [13], InfoVQA [14], TATDQA [15], _etc._ However, considerable amounts of documents in the real world are long-context 

*Corresponding Authors. 

Project Page: `https://mayubo2333.github.io/MMLongBench-Doc` 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 

**==> picture [306 x 145] intentionally omitted <==**

**----- Start of picture text -----**<br>
Single-Page Question 20970.9<br>= e Question:  I want to see a doctor in the campus hospital. After  7000 PWC MMLongBench-Doc(Ours)<br>_ =e registering at the registration area, what is the next step? Answer: internal medicine, surgical medicine, dental medicine) Evidence Page: Evidence Source:  Go to the medical department you registered at (i.e. Page 22Image 1831.5577.0 MP-DocVQA I DUDETAT-QAInfoVQA SlideVQA<br>DocVQA<br>— : p=a Cross-Page QuestionQuestion:  I’m at location “J” shown in the  151.5 1 ChartQA5.7 12 Average Pages20 47.5<br>Prec a campus map. Tell me the nearest coffee shop. (b) Dataset Statistics<br>— ze a ed Answer:  Ten Years After Café. 50.<br>E Evidence Pages:  Page 18, Page 30 =<br>. —, Evidence Sources:  Image, Table 9%8<br>Unanswerable Question ai 10:<br>Question:  According to this document, what is the main color of Tsinghua<br>Campus Bicycle? Give me the color name only.  Answer:  Not Answerable ° Open-src(Best) Claude-3(Opus) (1.5-Pro)Gemini— GPT-4V GPT-4o<br>(a) Dataset Example (c) Performance Overview<br>Average Text Tokens<br>**----- End of picture text -----**<br>


Figure 1: MMLONGBENCH-DOC evaluates understanding abilities of LVLMs on lengthy documents that span tens of pages and incorporate multi-modal elements. Experiments (bottom-right) indicate that most LVLMs struggle, even falling behind LLMs that are fed with only OCR-parsed documents. 

documents with tens or even hundreds of pages. The understanding of these lengthy documents brings new challenges for LVLMs from at least two aspects: (1) **Localization** : identify and retrieve information from massive, heterogeneous information (similar to the _needle in a haystack_ task); (2) **Cross-page comprehension** : collect and reason over multi-source information across different pages. These two kinds of abilities are beyond the evaluation scopes of the aforementioned singlepage DU datasets. Some recent DU datasets [16; 17; 18] feature multiple-page DU, but almost all their documents are either as short of only several pages or of low information density, making the localization-related questions over-simple. Additionally, few (if any) questions in these datasets necessitate cross-page comprehension. See more detailed related work in Section 2. In summary, there lacks a unified and high-quality benchmark on lengthy documents, leaving the evaluation of long-context DU largely unexplored. 

In this paper, we present MMLONGBENCH-DOC, a benchmark designed to evaluate the **M** ulti- **M** odality **Long** -context **Doc** ument understanding abilities of LVLMs. Towards a comprehensive benchmark, it incorporates lengthy documents from both four existing datasets [13; 17; 18; 19] and other various papers, brochures, _etc._ Consequently, our benchmark includes 135 PDF-formatted documents spanning across 7 diverse domains, with each document averaging 47.5 pages and 21,214.1 textual tokens. Regarding the questions, we employ ten expert-level annotators to (1) edit questions associated with documents from existing datasets to meet our benchmark’s standard and (2) create new questions for all collected documents to expand the scale of the benchmark. Then a threeround, semi-automatic reviewing process ensures the benchmark’s annotation quality. As a result, MMLONGBENCH-DOC comprises 1,082 human-annotated questions, with 184 sourced from four existing datasets and 898 newly annotated. Being a multi-modal benchmark, the answer to each question requires evidence from one or more of these five in-document sources: _text_ , _layout_ , _chart_ , _table_ , and _image_ . Questions are categorized into three types based on the number of evidence pages[1] , with examples illustrated in Figure 1(a): (1) 494 _single-page_ questions (with one evidence page) mainly to evaluate localization abilities, (2) 365 _cross-page_ questions (with multiple evidence pages) to assess cross-page comprehension, and (3) 223 _unanswerable_ questions (no evidence for answering it, _i.e.,_ no evidence pages) to reduce shortcuts and measure LVLMs’ potential hallucinations. Metainformation including evidence pages, sources, and answer formats, is preserved for fine-grained evaluation and analysis. Detailed descriptions of the annotation pipeline and statistics can be found in Section 3. 

We conduct extensive experiments on MMLONGBENCH-DOC to evaluate the long-context DU abilities of 14 LVLMs, including 4 proprietary and 10 open-source ones. Given a document, we screenshot each page and feed all of these PNG-formatted images to LVLMs in an end-to-end approach. For comparison, we also convert the documents to textual format by optical character recognition (OCR) and evaluate another 6 proprietary and 4 open-source 10 LLMs (6 proprietary and 

> 1Given a document _D_ and a question _q_ upon _D_ , We call page _P_ (in document _D_ ) an _evidence page_ of _q_ if the answer of _q_ necessitates one or more pieces of evidence in page _P_ . 

2 

Table 1: Comparison between our benchmark and previous DU datasets. **Unans.** : unanswerable question. **TXT/L/C/TAB/I** : pure text/generalized layout/chart/table/image. **Doc. Rel.** : document relevance. Whether document information is indispensable for the answer. **Avg. Position** : the average page index on which the answer evidence is located. *:Statistics from [20]. 

|**Benchmarks**|**Document**<br># Pages<br># Tokens|**Question type**<br>**Answer Evidence**<br>Cross-page (%)<br>Unans. (%)<br>Doc. Rel.<br>Source<br>Avg. Position|
|---|---|---|
|DocVQA [12]<br>ChartQA [13]2<br>InfoVQA [14]2<br>TAT-DQA [15]<br>VisualWebBench [21]2<br>PWC [22]<br>MP-DocVQA [16]<br>DUDE [17]<br>SlideVQA [18]|1.0<br>151.5<br>1.0<br>236.9<br>1.2<br>288.0<br>1.1<br>577.0<br>1.0<br>452.4<br>~12*<br>~7000*<br>8.3<br>2026.6<br>5.7<br>1831.5<br>20.0<br>2030.5|✗<br>✗<br>✔<br>✗<br>TXT/L/C/TAB/I<br>-<br>✗<br>✗<br>✓<br>C<br>-<br>✗<br>✗<br>✔<br>✗<br>L/C/TAB/I<br>-<br>✗<br>✗<br>✔<br>✗<br>TXT/TAB<br>-<br>✗<br>✗<br>✓<br>LAY/I<br>-<br>✗<br>✗<br>✔<br>✗<br>TAB<br>-<br>✗<br>✗<br>✔<br>✗<br>TXT/L/C/TAB/I<br>6.0<br>✓(2.1%)<br>✓(12.7%)<br>✔<br>✗<br>TXT/L/C/TAB/I<br>2.5<br>✓(13.9%)<br>✗<br>✔<br>✗<br>TXT/L/C/TAB/I<br>9.1|
|MMLONGBENCH-DOC|47.5<br>21214.1|✓(33.0%)<br>✓(22.5%)<br>✓<br>TXT/L/C/TAB/I<br>23.6|



4 open-source ones). The results in Figure 1(c) highlight the challenges that current LVLMs face with long-context DU. The best-performing LVLM, GPT-4o, achieves an overall F1 score of only 44.9%, while the second-best LVLM, GPT-4V, scores 30.5%. Moreover, all the remaining LVLMs tested with multi-modal documents performed worse than single-modal LLMs handling lossy, OCR-parsed texts. Specifically, the Gemini-1.5-Pro and Claude-3-Opus present 4.2% and 6.4% absolute decrease when the inputs change from document screenshots to OCR-parsed texts. Regarding open-source models, the best-performing LVLM lags behind the best-performing LLM by 11.7%. These results reveal that long-context DU is a far-from-resolved task for current LVLMs. 

## **2 Related Work** 

**Benchmarks for Document Understanding.** A great amount of datasets have emerged to evaluate the DU capabilities of LVLMs. Many datasets focus exclusively on either a single component ( _e.g.,_ table, chart) [13; 15; 21; 22] or a single page [12; 14] from the full documents. Some recent DU datasets [16; 17; 18; 23; 19] attempt to assess multi-page documents, but still exhibit shortcomings in terms of document length (page number), information density (token number) and the construction approaches. Specifically, MP-DocVQA [16] is an extension of DocVQA [12] and inherently absent of both crosspage and unanswerable questions. Annotating from scratch, DUDE [17] includes a small percentage of cross-page questions (2.1%) and unanswerable questions (12.7%). However, due to the relatively short context length (5.3 pages on average) and the use of crowd-sourced annotations, questions in DUDE tend to be less challenging and somewhat less rigorous. SlideVQA features 20-page documents and cross-page questions (12.9%). Nevertheless, the documents in SlideVQA are in slide-deck format and of relatively low information density. Moreover, these cross-page questions are HotpotQAstyle [24] created by instantiating entity graphs and co-referencing in-graph entities across multiple pages. The entity graph from a closed document tends to be sparse and has significant shortcuts (see examples in Appendix A.4). These shortcuts sometimes lead to false cross-page questions that actually do not require answer evidence across different pages. The recent FinanceBench [19] features both extremely long-context documents and practical, scalable cross-page questions. However, its documents are exclusively financial reports. Additionally, the reference answers are in open-ended formats, making the expert-level manual evaluation indispensable. The above reasons limit the broader applicability of FinanceBench. To our best knowledge, MMLONGBENCH-DOC is the first comprehensive, qualified, and easy-to-use benchmark on the long-context DU task. More detailed descriptions and comparisons are presented in Table 1. 

**Models for Document Understanding.** There are two main branches of models for automatic DU tasks. The first approach employs two-stream, OCR-dependent architectures to separately encode textual information (parsed via OCR) and visual information (images and/or layout structures) [25; 26; 27]. In contrast, the second approach develops OCR-free models that understand documents 

> 2We view website screenshots and posters as generalized documents and define _equivalent page number_ ( `EPN` ) to measure their context lengths: `EPN(D) = ceil` ( `[Pixel(D)]` _P_ ). Here `Pixel(D)` is the pixel number of generalized document `D` , and `P` is the average pixel numbers of each page (converting from .pdf to .png format with resolution 240) in MMLONGBENCH-DOC. 

3 

in an end-to-end manner [28; 29]. With the rapid advancement of LVLMs, the latter approach has dominated the current DU solutions. As mentioned above, a range of LVLMs demonstrate promising performance on single-page DU datasets. However, as shown in Section 4, even the most advanced LVLMs fall significantly short of achieving satisfactory performance on our benchmark. It reveals that understanding lengthy documents still poses great challenges to current LVLMs. 

**Long-context LVLMs and LLMs.** Lengthy documents necessitate the use of LVLMs or LLMs with extended context sizes. Several benchmarks [30; 31; 32; 33] and solutions [34; 35; 36; 37] have been proposed to evaluate and develop long-context LLMs. However, there exists limited related work for long-context LVLMs, leaving this area largely unexplored. Until very recently, contemporary studies [38; 39; 40] assess and/or improve LVLMs’ multi-image understanding capabilities. Evaluations on both MMLONGBENCH-DOC and these works indicate that current LVLMs are still not fully equipped to handle long-context DU and many other practical tasks that require extensive contextual comprehension. 

## **3 MMLONGBENCH-DOC** 

We design a three-stage annotation pipeline for the construction of our benchmark. The three stages will be introduced in Section 3.1, Section 3.2, and Section 3.3, respectively. We also provide key statistics of our benchmark in Section 3.4. 

## **3.1 Document Collection** 

As a long-context DU benchmark, the documents shall be of diverse topics and lengthy enough. To this end, we crawl a great amount of documents from various sources. Then we select the lengthy ones from these documents. Specifically, we encompass a diverse array of documents from two approaches. (1) **Existing documents** from four previous datasets: DUDE [17], SlideVQA [18], ChartQA [13], and FinanceBench [19]. (2) **Newly-collected documents** from Arxiv[3] , ManualsLib[4] and Google Search[5] . Then we (1) filter out the documents with fewer than 15 pages or license restrictions and (2) down-sample documents from DUDE, SlideVQA, and FinanceBench for a more balanced distribution. Detailed descriptions of our selection and processing procedure can be found in Appendix A.1 and Appendix A.2. 

In summary, we collect a total of 135 documents. Among them, 76 documents are from existing datasets and incorporate previously annotated questions (represented as triangles). The remaining 59 documents are newly collected and incorporate no existing questions. We manually categorize them into 7 types: _Research Report_ , _Financial Report_ , _Academic Paper_ , _Brochure_ , _Guideline_ , _Administration & Industry File_ , _Tutorial / Workshop_ . We showcase some instances of these documents in Appendix A.3. 

## **3.2 Question and Answer Collection** 

To serve as a high-quality and comprehensive benchmark, the question annotation of our benchmark adheres to the following standards: (1) All questions shall be neither over-easy nor over-difficult. (2) Questions are not repetitively derived from the same page or the same pattern. (3) The distribution of evidence numbers, evidence sources, and evidence locations for the questions shall be balanced. (4) No questions shall be answered correctly without accessing the relevant documents. 

Ten authors serve as expert-level annotators for the question-and-answer collection. All of them are doctors or Ph.D. students proficient in English reading and writing. Before formal annotation, they undergo a training session and pre-annotate three documents for practice. We iteratively review their annotation results and provide personalized feedback until their annotations meet the standards mentioned above. Regarding the formal annotation, we divide 135 documents into 54 batches (each having 2-4 documents) and dispatch these batches to annotators. We then ask the annotators to submit their results in units of batches and set reasonable time intervals for each batch’s submission. We 

> 3 `https://arxiv.org` 

> 4 `https://www.manualslib.com` 

> 5 `https://www.google.com.sg` 

4 

timely evaluate their annotations after each submission and remind the annotators if their questions in this turn diverge from the standards. It avoids the annotators rushing all assignments in a short time and benefits the annotation quality. We recommend the annotators take 60-90 minutes on each document. Specifically, the annotators shall rapidly read through the whole document in the first 15-30 minutes. For the remaining time, they shall dive deep into specific components to modify existing annotations and/or add new annotations as detailed below. 

**Modify Existing Questions.** Documents collected from existing datasets had been annotated with some questions and answers from previous work. However, their crowd-sourcing annotations inevitably make some questions, answers, and other meta information unqualified. Therefore, we edit their annotations before including them as a component of our benchmark. 

Specifically, we classify six potential problems in original annotations: _Wrong Answers or Evidence Pages_ , _Repetitive Question_ , _Ambiguous Question_ , _Decontextualization-required Question_ , _Low Document-relevant Question_ and _Potential Shortcut_ . See detailed explanations and examples about these problems in Appendix A.4. Given an existing document, the annotators are tasked to evaluate each existing question’s quality according to whether they have one or more above problems and assign a label from { `Retain` , `Revise` , `Remove` } for each question. Then the annotators would revise the `Revise` questions to meet our quality criteria and remove the `Remove` questions. Among all 425 original questions from 76 existing documents, 32.2% of them are revised and 46.1% are removed. We finally collect 211 questions in this procedure. The corresponding GUI is shown in Appendix A.7. 

**Add New Questions.** We newly annotate questions on both existing and newly collected documents to expand the questions in our benchmark. Specifically, we ask annotators to add about 3 questions on existing documents, and 6 questions on newly-collected documents. Given most existing questions (even after editing) are single-page ones and sourced from texts, we put more focus on (1) cross-page and unanswerable questions and (2) questions sourced from tables, charts, and images for newly added questions to balance the distribution. We detail the quantitative requirements in Appendix A.5. Associated with questions, annotators also provide reference answers and meta-information ( _i.e.,_ evidence sources, answer format, evidence locations) for all samples. We finalized a collection of 965 samples in this procedure. The corresponding GUI is shown in Appendix A.7. 

## **3.3 Quality Control** 

Combining the merits of humans and LVLMs, we adopt a three-round, semi-automatic quality control procedure to improve the annotation quality of our benchmark. We detail each round in the following components and leave the discussion of potential bias in Appendix A.6. 

**Document-relevant Detection.** Our benchmark is designed to evaluate LVLMs’ long-context document understanding abilities. All questions are expected to be unanswerable without access to corresponding documents. To remove low document-relevant questions ( _i.e.,_ questions not relying on documents), we feed each annotated question **WITHOUT** documents to GPT-4o. A question will be identified as _low document-relevant_ question if GPT-4o correctly predicts under this case. Ultimately, 94 samples are identified as low document-relevant questions and removed in this round. 

**Self-reflection.** We draw inspirations from MMBench [41] and leverage LVLMs to reduce the wrongly-annotated samples. Specifically, we feed the remaining questions from the last round **WITH** their documents to GPT-4o. Samples whose model predictions are inconsistent with the reference answers are sent back to corresponding annotators. The annotators are asked to check each question and identify whether the inconsistency is caused by _problematic annotation_ or not. As a result, 13.8% of the samples are identified as problematic annotations. The annotators revise them accordingly. 

**Cross-checking.** In parallel, annotators cross-check the annotated samples from other annotators and determine the inconsistency reasons the same as described above. We calculate Cohen’s kappa value of their identifications as 0.42 (17.5% inconsistent samples), showing a moderate agreement. Regarding the 17.5% inconsistent samples, two primary authors serve as meta-annotators and make final decisions on them (and if necessary, revise accordingly). 

## **3.4 Dataset Overview and Analysis** 

The main statistics of MMLONGBENCH-DOC are presented in Table 2. Overall, our benchmark consists of 1,082 questions. These questions are constructed upon 135 lengthy documents across 7 

5 

|**Statistic**|**Number**|
|---|---|
|**Documents**<br>- Type<br>- Average/Medium pages<br>- Average/Medium length|135<br>7<br>47.5 / 28<br>21,214.1 / 12,179|
|**Total questions**|1,082|
|- Single-page question<br>- Cross-page questions<br>- Unanswerable questions|494 (45.7%)<br>365 (33.7%)<br>223 (20.6%)|
|- Derived questions<br>- Newly-annotated questions|184 (17.0%)<br>898 (83.0%)|
|(Evidence source)||
|- Pure-text<br>- Layout|305 (35.5%)<br>119 (13.9%)|
|- Table|218 (25.4%)|
|- Chart|178 (20.7%)|
|- Image<br>304 (35.4%)<br>(Answer Format)<br>~~SSS~~||
|- String|250 (29.1%)|
|- Integer|299 (34.8%)|
|- Float|159 (18.5%)|
|- List|151 (17.6%)|
|Avg./Max. question length|16.4 / 60|
|Avg./Max. answer length|2.8 / 54|



Table 2: Dataset Statistics 

**==> picture [154 x 87] intentionally omitted <==**

**----- Start of picture text -----**<br>
Brochure Financial Report<br>(11.1%) (8.1%)<br>Guidebook<br>Tutorial / Workshop<br>(16.3%)<br>(12.6%)<br>Administration &<br>Industry File<br>Academic Paper (7.4%)<br>(19.3%)<br>Research Report<br>(25.2%)<br>**----- End of picture text -----**<br>


Figure 2: Detailed distribution of documents. **Top** : Document type. **Middle** : Page Number. **Bottom** : Token Number. 

Figure 3: Detailed distribution of questions & answers. **Left** : Absolute position of answer evidences (the page index). **Middle** : Relative position (the page index/document page number). **Right** : Evidence page number of each question. (0: unanswerable question; >2: cross-page question). 

document types, with an average of 47.5 pages and 21,214.1 tokens. Please see detailed distributions of these documents in Figure 2. Regarding the questions, there are 494 single-page questions (1 evidence page), 365 cross-page questions (2+ evidence pages), and 223 unanswerable questions (no evidence page). These three types of questions evaluate the LVLMs’s long-context DU capabilities from complementary aspects: the localization ability, the cross-page comprehension ability, and the hallucination severity, respectively. For single-page and cross-page questions, their answer evidence is scattered among different context sources ( _i.e.,_ text, layout, table, chart, image) and evenly distributed across different locations of the documents (see Table 2, Figure 3 Left and Middle). Also notably, 28.6% of cross-page questions have more than two evidence pages, which further enhances the challenge of our benchmark. 

## **4 Evaluation** 

## **4.1 Evaluation Protocol** 

We follow MATHVISTA [56] to conduct a three-step evaluation protocol: _response generation_ , _answer extraction_ , and _score calculation_ . We adopt such a protocol out of three considerations: (1) Current LVLMs are instructed to generate long responses, rather than short-form answers, in conventional settings. (2) The evaluation of long responses, however, remains an open and challenging problem. (3) We focus on the document understanding (not instruction following) abilities of LVLMs. 

6 

Table 3: **Evaluation of various models on MMLONGBENCH-DOC.** We report the generalized accuracy of five types of evidence sources including pure text (TXT), layout (LAY), chart (CHA), table (TAB), and image (IMG). We also present the generalized accuracy of questions categorized by the number of evidence pages: single-page (SIN), cross-page (MUL), and unanswerable (UNA) questions. The **best** and **second-best** performance in each section are highlighted. 

|**Model**<br>**#Param Context**<br>**Window**|**Evidence Source**<br>TXT<br>LAY<br>CHA<br>TAB<br>FIG|**Evidence Page**<br>SIN<br>MUL UNA|**ACC**<br>**F1**|
|---|---|---|---|
|_OCR (Tesseract [42]) + Large Language Models (LLMs)_||||
|_Open-source Models_<br>ChatGLM-128k [37]<br>6B<br>128k<br>23.4<br>12.7<br>9.7<br>10.2<br>12.2<br>18.8<br>11.5<br>18.1<br>16.3<br>14.9<br>Mistral-Instruct-v0.2 [43]<br>7B<br>32k<br>19.9<br>13.4<br>10.2<br>10.1<br>11.0<br>16.9<br>11.3<br>24.1<br>16.4<br>13.8<br>Mixtral-Instruct-v0.1 [44]<br>8x7B<br>32k<br>24.2<br>14.8<br>12.5<br>15.0<br>13.7<br>21.3<br>14.1<br>13.1<br>17.0<br>16.9<br>Mixtral-Instruct-v0.1 [44]<br>8x22B<br>64k<br>34.2<br>21.3<br>19.5<br>21.3<br>19.2<br>27.7<br>21.9<br>32.4<br>26.9<br>24.7<br>_Proprietary Models_||||
|QWen-Plus [45]<br>-<br>32k<br>DeepSeek-V2 [46]<br>-<br>32k<br>Claude-3 Opus [4]<br>-<br>32k<br>Gemini-1.5-Pro [3]<br>-<br>32k<br>GPT-4-turbo [47]<br>-<br>128k<br>GPT-4o [2]<br>-<br>128k|17.4<br>15.6<br>7.4<br>7.9<br>8.8<br>27.8<br>19.6<br>8.8<br>17.0<br>9.4<br>30.8<br>30.1<br>16.4<br>24.4<br>16.3|14.2<br>10.6<br>42.2<br>20.2<br>15.4<br>48.1<br>32.0<br>18.6<br>30.9|18.9<br>13.4<br>24.9<br>19.6<br>26.9<br>24.5<br>31.2<br>24.8<br>27.6<br>25.9<br>30.1<br>30.5|
||29.3<br>15.9<br>12.5<br>17.7<br>11.5<br>|21.2<br>16.4<br>**73.4**||
||36.5<br>21.0<br>20.7<br>24.3<br>17.3|28.7<br>23.8<br>31.2<br>35.4<br>29.3<br>18.6||
||41.1<br>23.4<br>28.5<br>38.1<br>22.4|||
|_Large Visual Language Models (LVLMs)_||||
|_Open-source, 7-14B Models_<br>DeepSeek-VL-Chat [48]<br>7.3B<br>4k<br>7.2<br>6.5<br>1.6<br>5.2<br>7.6<br>5.2<br>7.0<br>12.8<br>7.4<br>5.4<br>Idefcs2 [49]<br>8B<br>8k<br>9.0<br>10.6<br>4.8<br>4.1<br>8.7<br>7.7<br>7.2<br>5.0<br>7.0<br>6.8<br>MiniCPM-Llama3-V2.5 [50; 51]<br>8B<br>2k<br>11.9<br>10.8<br>5.1<br>5.9<br>12.2<br>9.5<br>9.5<br>4.5<br>8.5<br>8.6<br>InternLM-XC2-4KHD [5]<br>8B<br>16k<br>9.9<br>14.3<br>7.7<br>6.3<br>13.0<br>12.6<br>7.6<br>9.6<br>10.3<br>9.8<br>mPLUG-DocOwl 1.5 [52]<br>8.1B<br>4k<br>8.2<br>8.4<br>2.0<br>3.4<br>9.9<br>7.4<br>6.4<br>6.2<br>6.9<br>6.3<br>Qwen-VL-Chat [53]<br>9.6B<br>6k<br>5.5<br>9.0<br>5.4<br>2.2<br>6.9<br>5.2<br>7.1<br>6.2<br>6.1<br>5.4<br>Monkey-Chat [54]<br>9.8B<br>2k<br>6.8<br>7.2<br>3.6<br>6.7<br>9.4<br>6.6<br>6.2<br>6.2<br>6.2<br>5.6<br>_Open-source, >14B Models_<br>CogVLM2-LLaMA3-Chat [9]<br>19B<br>8k<br>3.7<br>2.7<br>6.0<br>3.2<br>6.9<br>3.9<br>5.3<br>3.7<br>4.4<br>4.0<br>InternVL-Chat-v1.5 [6]<br>26B<br>4k<br>14.0<br>16.2<br>7.1<br>10.1<br>16.6<br>14.9<br>12.2<br>17.5<br>14.6<br>13.0<br>EMU2-Chat [55]<br>37B<br>2k<br>6.1<br>9.7<br>2.6<br>3.8<br>7.7<br>5.7<br>6.1<br>16.5<br>8.3<br>5.5<br>_Proprietary Models_||||
|Claude-3 Opus [4]<br>-<br>200k<br>Gemini-1.5-Pro [3]<br>-<br>128k<br>GPT-4V(ision) [47]<br>-<br>128k<br>GPT-4o [2]<br>-<br>128k|24.9<br>24.7<br>14.8<br>13.0<br>17.1|25.6<br>13.8<br>7.6|17.4<br>18.1<br>28.2<br>20.6|
||21.0<br>17.6<br>6.9<br>14.5<br>15.2|21.1<br>11.1<br>69.2||
||34.4<br>28.3<br>28.2<br>32.4<br>26.8|36.4<br>27.0<br>31.2|32.4<br>31.2|
||**46.3**<br>**46.0**<br>**45.3**<br>**50.0**<br>**44.1**|**54.5**<br>**41.5**<br>20.2|**42.8**<br>**44.9**|
|_Human Baseline_||||
|Human Experts<br>-<br>-|-<br>-<br>-<br>-<br>-|-<br>-<br>-|65.8<br>66.0|



Specifically, we impose no limitations on _response generation_ stage to encourage LVLMs to answer the questions in a freestyle. Then we propose a unified LLM-based _answer extractor_ (GPT-4o under our setting) to convert their long responses to short-form answers. Finally, we use a rule-based _score calculator_ to evaluate the converted short answers. We report both generalized accuracy and generalized F1 score to balance the answerable (positive) and unanswerable (negative) questions. The used prompt, the high correlation between our automatic _answer extractor_ and human evaluation, and the detailed rules of our _score calculation_ are described in Appendix B. 

## **4.2 Experimental Setup** 

We evaluate 14 LVLMs on MMLONGBENCH-DOC, including 4 proprietary LVLMs and 10 opensource LVLMs. To purely evaluate LVLMs’ long-context DU abilities, we screenshot each page of the PDF-formatted document with 144 DPI and feed all these PNG-formatted images to LVLMs in an end-to-end approach. Notably, all evaluated open-source LVLMs do not support multi-image inputs or present significant performance drops when fed with excessive images ( _e.g.,_ more than 10 or 20 images). Therefore, we employ a concatenation strategy that combines all screenshot pages into 1 or 5 images and feeds these concatenated images to open-source LVLMs. Regarding proprietary LVLMs, we adopt the same concatenation strategy and reduce the image number to 20 for Claude-3-Opus to fit its maximum image threshold. For GPT-4o, GPT-4V, and Gemini-1.5-Pro, we directly send all original screenshots to them ( _i.e.,_ the image number equals the page number). 

For comparison, we also use the Tesseract [42] OCR model to recognize and extract texts from the documents and feed the parsed documents to 10 LLMs, including 6 proprietary and 4 open-source 

7 

ones. Texts exceeding their context lengths are truncated. Notably, as a key component of the classical solution for the DU task, the OCR model can handle most flattened texts and some structured tables in the document. However, it cannot perceive the information from the charts or images. Thus the TXT-formatted, OCR-parsed documents are lossy documents in which the information is not fully preserved. More detailed hyperparameters are introduced in Appendix B.5. Additionally, we also conduct manual evaluation on a subset of our datasets (238 questions from 29 documents) to indicate the difficulty of this task for humans. 

## **4.3 Main Results** 

We compare the performance of different LVLMs and LLMs in Table 3, reporting their generalized accuracy and F1 scores (shown in the last two columns). Regarding LVLMs, we draw several conclusions as below: (1) The performance demonstrates that long-context DU is still a challenging and unsolved task for current LVLMs. The best-performing LVLM, GPT-4o, merely achieves a 44.9% F1 score. The second best-performing LVLM, GPT-4V, lags behind by over 10% percent and presents a 31.4% F1 score. All other LVLMs only achieve about 20% or even lower F1 scores. (2) Though far from satisfactory, GPT-4o performs much better than all other models (including GPT-4V). Thus we speculate that the multi-modal pre-training paradigm significantly benefits LVLMs’ cross-modality understanding capabilities. (3) Proprietary LVLMs perform better than open-source LVLMs by a large margin. We attribute it to the difference of acceptable image numbers: open-source LVLMs only support single-image or several-image inputs, while proprietary LVLMs can be fed with at least 20 images or even more. Given that lengthy documents have tens of even hundreds of pages, it is impractical for open-source LVLMs to accurately perceive the information in the documents from the excessively concatenated images. (4) The performances of different models are highly correlated with their acceptable image numbers and maximum image resolutions. Notably, open-source LVLMs that support high-resolution images ( _i.e.,_ InternLM-XC2-4KHD and InternVL-Chat-v1.5) exhibit superior performance compared to those with lower resolution limits. 

Surprisingly, LVLMs even demonstrate overall worse performance than LLMs, even LLMs are fed with lossy OCR-parsed documents. Specifically, Gemini-1.5-Pro and Claude-3 Opus have 4.2% and 6.4% absolute F1-score degradations on vision versions. And the best-performing LLM (Mixtral) also surpasses the best-performing LVLM (InternVL-v1.5) by 11.7%. The above results clearly reveal that most current LVLMs are still not proficient in cross-modality, long-context document understandings. It is promising that GPT-4o and GPT-4-turbo achieve better performance when seeing multi-modality PDF documents than parsed text by 14.4% and 5.3% F1-score, respectively. Their performances validate the feasibility, benefit, and necessity of understanding documents in an end-to-end, cross-modality approach. We speculate that the scarce related pre-training corpus ( _i.e.,_ extremely multi-image or lengthy documents) hinders the long-context DU capabilities of other LVLMs. We will leave related explorations for future work. 

Regarding the human evaluation, we observe 66.0% F1-score from our annotators and a significant performance gap (exceeding 20% in absolute) between the current LVLMs and humans. This gap highlights the challenges of document understanding for LVLMs and the necessity of our benchmark. 

## **4.4 Fine-grained Results.** 

**Document Type.** As illustrated in Figure 4, LVLMs and LLMs exhibit distinct performance patterns across various document types. Our findings include: (1) All evaluated models demonstrate decent performance on industrial documents, which tend to have more standardized formats and less non-textual information. (2) The GPT series and Mixtral ( _i.e.,_ the SoTA open-source LLM) show relatively balanced performance across different document types. In contrast, other models perform significantly worse in specialized domains such as academic papers and financial reports. (3) When equipped with OCR, LLM-based models like GPT-4 and Mixtral achieve comparable or even superior performance on industrial documents, academic papers, and brochures. Conversely, end-to-end LVLMs outperform OCR+LLMs in areas such as tutorials, research reports, and guidelines. We speculate that comprehending these latter document types requires more extensive multi-modal information, from which LVLMs significantly benefit. 

**Evidence Source.** We categorize questions based on their evidence sources and present fine-grained results in Figure 4 and Table 3. Our observations reveal that only GPT-4o exhibits relatively balanced 

8 

**==> picture [299 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
GPT-4o |_| GPT-4V |_| GPT-4 |_| Gemini-1.5-Pro |_| Claude-3-Opus<br>SoTA Open-source LVLM | | SoTA Open-source LLM<br>Tutorial Overall F1<br>46.5 53.0 53.3 44.9<br>Financial Industry<br>46.3 44.1<br>Text Image<br>Brochure g ]} Report<br>Layout Table<br>42.7 43.9<br>SS 46.0 50.0<br>44.7 35.9 45.3<br>Guidebook Academic Chart<br>**----- End of picture text -----**<br>


Figure 4: Fine-grained results on various document types and evidence sources. 

performance across the different sources. Other LVLMs, however, show inferior performance on questions related to charts and/or images compared to those related to text and/or layout. Additionally, LLMs generally demonstrate better or comparable performance to LVLMs on text- and table-related questions but show worse performance on questions involving other elements. This highlights the limitations of OCR (and other PDF parsers) when dealing with charts and images, as well as the gap in OCR capabilities between LVLMs and pure-text LLMs. 

**Evidence Position.** We also examine how the evidence locations ( _i.e.,_ the page indexes where the answer evidence is found) affect model performance. The results shown in Figure 5 reinforce that MMLONGBENCHDOC poses significant challenges for current models, at least partially due to the extended length of the documents. Almost all models (except InternVL-v1.5) exhibit their best performance on questions derived from the initial pages, while their performance declines progressively as the page index increases. Interestingly, two proprietary models, Gemini-Pro-1.5 and Claude-3-Opus, experience particularly sharp declines in performance. 

**Number of Evidence Page.** We observe a consistent trend that all models achieve higher scores on singlepage questions than cross-page questions. It reveals that gathering and reasoning over all necessary information 

**==> picture [176 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
300<br>50<br>250<br>40<br>200<br>30 150<br>100<br>20<br>50<br>10 0<br>< 10 10-20 20-50 > 50<br>Evidence Page #index (Locations)<br> (%)<br>Accuracy<br># Sample<br>**----- End of picture text -----**<br>


Figure 5: Relationships between evidence positions and model performances. 

across different pages is not trivial for current LVLMs and LLMs. More interestingly, evaluated LVLMs behave differently on unanswerable questions. GPT-4o and Claude-3 Opus adopt more aggressive strategies and usually tend to provide some answers. It makes their answers more likely helpful, but also increases the risk of hallucination and unfaithfulness (see their scores on unanswerable questions are much lower than answerable questions). On the contrary, Gemini-1.5-Pro, DeepSeek-VL-Chat, and EMU2-Chat are much more cautious and tend to refuse to answer questions about which they are uncertain. It makes their answers safer but less helpful (with large amounts of responses like _I don’t know_ ). 

## **5 Analysis & Discussion** 

## **5.1 Oracle Setting** 

We conduct additional experiments to explore to what extent the challenges of MMLONGBENCHDOC are caused by the long-context lengths of documents. Specifically, we feed 820 answerable questions along with their oracle evidence pages (instead of the whole documents) to three representative LVLMs and show results in Figure 6. On one hand, it indicates that long-context length is a 

9 

significantly challenging factor for document understanding. Compared with the oracle-page setting, lengthy documents lead to more than 20% absolute performance degradation on Gemini-1.5-Pro and InternLM-XC2-4KHD. Regarding the single-page questions, the performance difference even achieves up to 30%. On the other hand, the overall performance achieves only about 40% and 30% for Gemini-1.5-Pro and InternLM-XC2-4KHD even under oracle-page setting. And the improvement for GPT-4o is much less (about 10%). It demonstrates that the development of long-context LVLMs can largely facilitate, though still can not fully solve, the long-context DU task. 

**==> picture [120 x 6] intentionally omitted <==**

**----- Start of picture text -----**<br>
Whole-document Oracle-pages<br>**----- End of picture text -----**<br>


Figure 6: Performance comparisons between normal setting (feeding models with the whole documents) and oracle setting (feeding models only with the evidence pages) among three LVLMs. 

## **5.2 Error Analysis** 

**==> picture [131 x 106] intentionally omitted <==**

**----- Start of picture text -----**<br>
Irrelevant Answer<br>(11%)<br>Perceptual Error<br>Incomplete  (28%)<br>Evidence (10%)<br>Reasoning Error<br>(5%)<br>Hallucinated<br>Extractor Error<br>Evidence (33%)<br>(10%)<br>Knowledge Lacking<br>(3%)<br>**----- End of picture text -----**<br>


Figure 7: Error distribution 

We further conduct error analysis to understand the bottleneck of current LVLMs in a qualitative approach. Specifically, we randomly select 72 error predictions from GPT-4o’s responses and manually check their error reasons. These errors are categorized into seven types: _Perceptual Error_ , _Irrelevant Answer_ , _Incomplete Evidence_ , _Hallucinated Evidence_ , _Extractor Error_ , _Reasoning Error_ and _Knowledge Lacking_ . The distribution of these errors is illustrated in Figure 7. It indicates that most errors come from the model’s hallucination ( _i.e.,_ wrong explanations and answers to unanswerable questions) and perceptual errors (mainly in visual contexts). Additionally, GPT-4o sometimes misunderstands the intent of questions and provides irrelevant responses. The errors caused by collecting incomplete evidence (for cross-page questions) are also unignorable. The descriptions and examples of these error types are detailed in Appendix C.1. 

## **6 Conclusion** 

In this work, we present MMLONGBENCH-DOC to evaluate the long-context DU capabilities of LVLMs. Extensive experiments on 14 LVLMs (and 10 LLMs for comparison) reveal that the understanding of lengthy documents poses great challenges to current LVLMs. Even though the performance of GPT-4o proves the benefit of end-to-end, multi-modality perception for DU tasks, most LVLMs struggle on long visual contexts ( _i.e.,_ extremely multiple images) and show inferior performance compared to OCR+LLM pipelines. We hope that the construction of our benchmark could push forward the development of more powerful LVLMs on lengthy document understanding. 

## **Acknowledgements** 

This study is supported under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). This work is also supported by Shanghai Artificial Intelligence Laboratory, the National Key R&D Program of China (2022ZD0160201). 

10 

## **References** 

- [1] Lutz Bornmann and Rüdiger Mutz. Growth rates of modern science: A bibliometric analysis based on the number of publications and cited references. _Journal of the Association for Information Science and Technology_ , 66, 2014. 

- [2] Open AI. Hello gpt-4o, 2024. 

- [3] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 

- [4] Anthropic. Introducing the next generation of claude, 2024. 

- [5] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-Xcomposer2-4KHD: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _ArXiv preprint_ , abs/2404.06512, 2024. 

- [6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _ArXiv preprint_ , abs/2404.16821, 2024. 

- [7] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning, 2023. 

- [8] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. LLaVA-NeXT: Stronger llms supercharge multimodal capabilities in the wild, 2024. 

- [9] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. CogVLM: Visual expert for pretrained language models, 2023. 

- [10] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding, 2024. 

- [11] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document, 2024. 

- [12] Minesh Mathew, Dimosthenis Karatzas, R. Manmatha, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. _2021 IEEE Winter Conference on Applications of Computer Vision (WACV)_ , pages 2199–2208, 2020. 

- [13] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. 

- [14] Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. _2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pages 2582–2591, 2021. 

- [15] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. Towards complex document understanding by discrete reasoning. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4857–4866, 2022. 

- [16] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa, 2023. 

- [17] Jordy Van Landeghem, Rubèn Pérez Tito, Łukasz Borchmann, Michal Pietruszka, Pawel J’oziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Ackaert, Ernest Valveny, Matthew B. Blaschko, Sien Moens, and Tomasz Stanislawek. Document understanding dataset and evaluation (DUDE). In _ICCV_ , 2023. 

- [18] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. SlideVQA: A dataset for document visual question answering on multiple images. In _AAAI_ , 2023. 

- [19] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. FinanceBench: A new benchmark for financial question answering, 2023. 

11 

- [20] Łukasz Borchmann, Michal Pietruszka, Tomasz Stanislawek, Dawid Jurkiewicz, Michał Turski, Karolina Szyndler, and Filip Gralinski. Due: End-to-end document understanding benchmark. In _NeurIPS Datasets and Benchmarks_ , 2021. 

- [21] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. VisualWebBench: How far have multimodal llms evolved in web page understanding and grounding?, 2024. 

- [22] Marcin Kardas, Piotr Czapla, Pontus Stenetorp, Sebastian Ruder, Sebastian Riedel, Ross Taylor, and Robert Stojnic. AxCell: Automatic extraction of results from machine learning papers. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 8580–8594, Online, 2020. Association for Computational Linguistics. 

- [23] Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon, Ryan A. Rossi, and Franck Dernoncourt. Pdftriage: Question answering over long, structured documents, 2023. 

- [24] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. 

- [25] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In Rajesh Gupta, Yan Liu, Jiliang Tang, and B. Aditya Prakash, editors, _KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020_ , pages 1192–1200. ACM, 2020. 

- [26] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 2579–2591, Online, 2021. Association for Computational Linguistics. 

- [27] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , MM ’22, page 4083–4091, New York, NY, USA, 2022. Association for Computing Machinery. 

- [28] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision (ECCV)_ , 2022. 

- [29] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: screenshot parsing as pretraining for visual language understanding. In _Proceedings of the 40th International Conference on Machine Learning_ , ICML’23. JMLR.org, 2023. 

- [30] Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, and Omer Levy. SCROLLS: Standardized CompaRison over long language sequences. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 12007–12021, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 

- [31] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models, 2023. 

- [32] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. _ArXiv preprint_ , abs/2308.14508, 2023. 

- [33] Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. ∞bench: Extending long context evaluation beyond 100k tokens, 2024. 

- [34] Szymon Tworkowski, Konrad Staniszewski, Mikolaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Milo’s. Focused transformer: Contrastive training for context scaling. _ArXiv preprint_ , abs/2307.03170, 2023. 

- [35] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. _ArXiv preprint_ , abs/2309.12307, 2023. 

12 

- [36] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. _ArXiv preprint_ , abs/2309.00071, 2023. 

- [37] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. _ArXiv preprint_ , abs/2401.18058, 2024. 

- [38] Dingjie Song, Shunian Chen, Guiming Hardy Chen, Fei Yu, Xiang Wan, and Benyou Wang. Milebench: Benchmarking mllms in long context. _ArXiv preprint_ , abs/2404.18532, 2024. 

- [39] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning, 2024. 

- [40] Yujie Lu, Xiujun Li, Tsu-Jui Fu, Miguel Eckstein, and William Yang Wang. From text to pixel: Advancing long-context understanding in mllms, 2024. 

- [41] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMbench: Is your multi-modal model an all-around player? _ArXiv preprint_ , abs/2307.06281, 2023. 

- [42] Ray Smith. An overview of the tesseract ocr engine. In _ICDAR_ , 2007. 

- [43] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. 

- [44] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. 

- [45] Qwen Team. Introducing qwen1.5, 2024. 

- [46] DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model, 2024. 

- [47] OpenAI. GPT-4 technical report, 2024. 

- [48] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. DeepSeek-VL: towards real-world vision-language understanding. _ArXiv preprint_ , abs/2403.05525, 2024. 

- [49] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building visionlanguage models?, 2024. 

- [50] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. RLAIF-V: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. _ArXiv preprint_ , abs/2405.17220, 2024. 

- [51] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-resolution images. _ArXiv preprint_ , abs/2403.11703, 2024. 

- [52] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mPLUG-DocOwl 1.5: Unified structure learning for ocr-free document understanding. _ArXiv preprint_ , abs/2403.12895, 2024. 

- [53] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. _ArXiv preprint_ , abs/2308.12966, 2023. 

- [54] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _ArXiv preprint_ , abs/2311.06607, 2023. 

- [55] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. _ArXiv preprint_ , abs/2312.13286, 2023. 

13 

- [56] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _International Conference on Learning Representations (ICLR)_ , 2024. 

- [57] Tomasz Stanislawek, Filip Grali’nski, Anna Wr’oblewska, Dawid Lipi’nski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and P. Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In _IEEE International Conference on Document Analysis and Recognition_ , 2021. 

- [58] S. Svetlichnaya. DeepForm: Understand structured documents at scale., 2020. 

- [59] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. _2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)_ , 2:1–6, 2019. 

- [60] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. _2019 International Conference on Document Analysis and Recognition (ICDAR)_ , pages 1516–1520, 2019. 

- [61] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 5376–5384, 2017. 

- [62] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In _The IEEE Winter Conference on Applications of Computer Vision (WACV)_ , March 2020. 

- [63] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. _ArXiv_ , abs/2101.11272, 2021. 

- [64] Xingyu Chen, Zihan Zhao, Lu Chen, JiaBao Ji, Danyang Zhang, Ao Luo, Yuxuan Xiong, and Kai Yu. WebSRC: A dataset for web-based structural reading comprehension. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 4173–4185, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. 

14 

## **A Benchmark Construction Details** 

## **A.1 Existing Document Collection** 

Although previous datasets contain a relatively small proportion of lengthy documents, their absolute quantity should not be disregarded. Therefore, we compile lengthy documents from various datasets to include them as part of the documents in this benchmark. Specifically, we review and consider 21 previous document understanding (DU) datasets, and ultimately select 4 of them for further document selection. The selection reasons are shown in Table 4. All of these four datasets are licensed under the Creative Commons license (CC-BY) or other open-source licenses. Regarding the 4 selected datasets: DUDE [17], SlideVQA [18], ChartQA [13] and FinanceBench [19], we collect a total of 76 documents and detail our collection procedures as below. 

Table 4: Comparison of selected and considered datasets for our benchmark. 

|**Dataset**|**Selected**|**Comment**|
|---|---|---|
|DUDE [17]<br>SlideVQA [18]<br>ChartQA [13]<br>FinanceBench [19]|✓<br>✓<br>✓<br>✓|-<br>-<br>-<br>-|
|DocVQA [12]<br>MP-DocVQA [16]<br>Kleister Charity [57]<br>Kleister NDA [57]<br>DeepForm [58]<br>FUNSD [59]<br>SROIE [60]<br>Infograohics VQA [14]<br>TAT-QA [15]<br>PWC [22]<br>PaperQA [56]<br>TextbookQA [61]<br>PlotQA [62]<br>VisualMRC [63]<br>WebSRC [64]<br>VisualWebBench [21]<br>PDFTriage [23]|✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗|Repetitive with some documents/questions in DUDE; Single-page documents only<br>Repetitive with some documents/questions in DUDE; Single-page questions only<br>Repetitive with some documents/questions in DUDE; Over-simple<br>Repetitive with some documents/questions in DUDE; Over-simple<br>Repetitive with some documents/questions in DUDE; Over-simple<br>Repetitive with some documents/questions in DUDE; Over-simple<br>Repetitive with some documents/questions in DUDE; Over-simple<br>Infographs are not long-context documents<br>Repetitive with some documents/questions in FinanceBench<br>Repetitive with our self-annotated questions from academic papers<br>Repetitive with our self-annotated questions from academic papers<br>Low document-relevance; Over-simple<br>Repetitive with our self-annotated questions from academic papers and research reports<br>Human performance reached; Website screenshots are not long-context documents<br>Human performance reached; Website screenshots are not long-context documents<br>Human performance reached; Website screenshots are not long-context documents<br>Not publicly available|



**DUDE:** We first filter all documents over 15 pages in the validation set of the original dataset, resulting in 87 documents. From these, we randomly sample 23 to include as a component of our benchmark documents. 

**SlideVQA** : We download slide decks in the test set by following the instructions in the original repository[6] . Pursuing lengthy documents, we slightly modified the code to remove the 20-page truncation procedure. Then we randomly select 27 slide decks for our benchmark documents. 

**FinanceBench** : We randomly sample 5 financial reports from the test set. 

**ChartQA** : Different from the above three datasets, ChartQA only contains chart screenshots cropped from documents. We take the following steps to recover these original documents: (1) We use the Tesseract OCR model [42] to recognize the text within the charts. (2) We use these texts as keywords to search for related documents on Google Search. (3) We manually identify these documents and remove all those that are less than 15 pages. From the ChartQA test set, we finalize a collection of 53 research reports from the Pew Research Center. We randomly sample 18 of these documents to include as a component of our benchmark documents. 

> 6https://github.com/nttmdlab-nlp/SlideVQA 

15 

## **A.2 Newly-annotated Document Collection** 

Most documents collected from previous datasets are _Industrial Files_ , _Tutorial & Workshop_ , _Finance Report_ and _Research Report_ . To diversify our benchmark, we additionally collect 59 documents including _Academic Paper_ , _Brochure_ , and _Guideline_ . We detail the collection procedures as below. 

**Academic Paper** We collect 24 academic papers from Arxiv. All selected papers are over 15 pages (including references and appendix). To ensure annotation quality, each paper is either written or thoroughly read by at least one of the annotators. 

**Guideline and Brochure** We collect 21 guidelines and 14 brochures from either ManualsLib or Google Search, covering diverse topics such as school, company, institution, products, service _etc._ . Each document is manually reviewed by one corresponding annotator and other primary authors to ensure its availability for academic use[7] . 

## **A.3 Document Examples** 

As stated in Section 2.1, the documents in MMLONGBENCH-DOC can be categorized into seven types. We show the examples of each type as below. 

Figure 8: Document example about **Administration & Industrial File** 

Figure 9: Document example about **Tutorial & Workshop** (only show first 50 pages) 

> 7Should any authors request the removal of their documents, we will promptly comply. 

16 

Figure 10: Document example about **Research Report** 

Figure 11: Document example about **Financial Report** (only show first 50 pages) 

17 

Figure 12: Document example about **Academic Paper** 

Figure 13: Document example about **Guidebook** 

Figure 14: Document example about **Brochure** 

18 

## **A.4 Existing Question Editing** 

Documents collected from existing datasets had been annotated with some questions and answers. However, their crowd-sourcing annotations inevitably make some questions, answers, and other meta information unqualified. So we conduct a systematic and manual pipeline to edit their annotations. Specifically, we classify six potential problems in original annotations. The definitions and examples of these problems are shown below. 

**1. Wrong Answer or Evidence Pages:** The reference answers and/or evidence pages in original datasets are wrongly annotated. 

**==> picture [189 x 226] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original Question:<br>Why is the service not safe?<br>Original Answer:<br>Medicines were not managed safely.<br>Corrected Answer:<br>1. Medicines were not managed safely<br>2. There were insufficient staff to care for people’s<br>needs during the evening and at night.<br>3. Some risks to people’s health and wellbeing were<br>not assessed and action was not taken to reduce the<br>risk.<br>4. Safeguarding incidents were not investigated or<br>reported appropriately<br>Error type:<br>Wrong Answer or Evidence<br>Comment:<br>The original answer only mentions single<br>point of the safety problem and is incomplete:<br>in fact, there are a total of four points.<br>**----- End of picture text -----**<br>


Figure 15: Example of the original annotation with _Wrong Answer or Evidence Pages_ . 

19 

**2. Repetitive Question:** Too many questions with the same types ( _e.g.,_ key information extraction) occur in a single document (or even on the same page or point). 

**==> picture [180 x 219] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original Question 1:<br>What is the designation of Diane Hanson?<br>Original Answer 1:<br>Superior court of Delaware<br>Original Question 2:<br>Who is the superior court of Delaware?<br>Original Answer 2:<br>Diane Hanson<br>Error type:<br>Repetitive Question<br>Comment:<br>The above two questions are created<br>repeatedly upon a single point of the<br>document. We call them repetitive questions<br>and drop any one of them.<br>**----- End of picture text -----**<br>


Figure 16: Example of the original annotation with _Repetitive Question_ . 

20 

**3. Ambiguous Question:** The question is ambiguous at the document level ( _e.g.,_ the absence of entity, period, exact section or page, _etc._ ), or too broad to exactly answer. 

**==> picture [177 x 229] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original Question:<br>What is the telephone no?<br>Original Answer:<br>01983 873655<br>Error type:<br>Ambiguous Question<br>Comment:<br>The are two main entities occurred in this<br>report: (1) The Limes Residential Home,<br>and (2) Care Quality Commission. There is<br>neither explicit statement nor the implicit<br>coreference about any entity in this<br>question, making it ambiguous.<br>Revised Question:<br>What is the telephone no for The<br>Limes Residential Home?<br>Revised Answer:<br>01983 873655<br>**----- End of picture text -----**<br>


Figure 17: Example of the original annotation with _Ambiguous Question_ . 

21 

**4. Potential Shortcut:** The resolution of the question does not rely on two entities (across different pages) but only one of them, _i.e.,_ there exists a shortcut for this question. 

## **Original Question:** 

Why did the company which Mamoon Hamid is affiliated with invest $1M in the seed for greenhouse? **Original Answer:** 

Strong conviction in team, market and early customer validation. 

## **Error Type:** Potential Shortcut **Comment:** 

The coreference of Adventure Capital circled in white in the left slide, _i.e., the company which Mammon Hamid is affiliated with_ , makes no sense for answering this question. It is because that the words circled in blue in the right slide, _i.e., invest $1M in the seed_ , is a potentially a strong shortcut for answering this question. Though seemingly relying on the information across two pages, it is still likely a single-page question. 

## **Revised Question:** 

Why did greenhouse invest $1M in the seed for greenhouse? **Revised Answer:** Strong conviction in team, market and early customer validation. 

Figure 18: Example of the original annotation with _Potential Shortcut_ . 

22 

**5. Low Document-relevant Question:** The resolution of the question does not rely on the information from the document. It can be solved by the parametric knowledge in the LVLMs. 

**==> picture [165 x 216] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original Question:<br>Kailali is in which region of Nepal?<br>Original Answer:<br>Far-western region<br>Error type:<br>Low document-relevant Question<br>Comment:<br>This question is originally intended to<br>evaluate the model’s understanding ability<br>on this map. However, the development of<br>LVLMs makes this question not relying on<br>the information of this document anymore<br>(can be answered by model’s parametric<br>knowledge).<br>Revised Question:<br>What is the color of Kailali in the map of<br>Page 12?<br>Revised Answer:<br>Yellow<br>**----- End of picture text -----**<br>


Figure 19: Example of the original annotation with _Low Document-relevant Question_ . 

23 

**6. Decontextulization-required Question:** The understanding of the question is conditioned on a single page or even a single component of the document. 

**Original Question:** What is the percentage value of the first gray bar from the top? **Original Answer:** 29 **Error type:** Decontextulization-required Question **Comment:** The questions in ChartQA are dependent on a single chart, see the description like _the first gray bar from the top_ . We shall rephrase (decontextulize) the question to make it a document-level one. **Revised Question:** What is the percentage value of west Germany respondents viewing Germany’s relationship with the United States as important as its relationship with Russia? **Revised Answer:** 29 

Figure 20: Example of the original annotation with _Decontextulization-required Question_ . 

When dealing with questions categorized under any of these six problem types, annotators are instructed to either revise or remove them. Typically, repetitive questions and those with potential shortcuts are removed. In contrast, wrongly-annotated or decontextualization-required questions are generally revised. For ambiguous and low document-relevant questions, the course of action depends more on the annotators’ discretion. 

24 

## **A.5 New Question Annotation** 

We annotate new questions on both existing and newly-collected documents. To ensure a diverse range of questions, we impose limitations on the question distributions categorized by their types ( _i.e.,_ single-page, cross-page or unanswerable) and evidence sources ( _i.e.,_ table, chart, image). To balance existing questions which are mostly single-page and text-based, we place greater emphasis on cross-page, unanswerable, table-related, chart-related, and image-related questions. The detailed standards are as follows: 

|**Document Type**|**Evidence Page**<br>**Evidence Source**<br>**All**<br>Cross-page<br>Unanswerable<br>Table<br>Chart<br>Image|**Evidence Page**<br>**Evidence Source**<br>**All**<br>Cross-page<br>Unanswerable<br>Table<br>Chart<br>Image|**Evidence Page**<br>**Evidence Source**<br>**All**<br>Cross-page<br>Unanswerable<br>Table<br>Chart<br>Image|
|---|---|---|---|
|||||
|Industrial File<br>Workshop & Tutorial<br>Research Report<br>Financial Report<br>Academic Paper<br>Guidebook<br>Brochure|≥2<br>-<br>≥2<br>≥1<br>≥3<br>≥1<br>≥5<br>≥2<br>≥3<br>≥1<br>≥3<br>≥1<br>≥2<br>≥1|-<br>——≥3——<br>≥2<br>≥2<br>-<br>≥7<br>-<br>-<br>≥2<br>—-≥3—-<br>-<br>-<br>≥4<br>-<br>-<br>≥3|≥3<br>≥6<br>≥5<br>≥10<br>≥6<br>≥7<br>≥7|



Table 5: The **minimum** requirements for the number and distribution of questions, categorized by the evidence page numbers and evidence sources. We have set varying requirements for different document types based on their specific characteristics. 

## **A.6 Potential Bias for LVLM-based Quality Checking** 

As described in Section 3.3, we employ GPT-4o to remove document-agnostic ( _i.e.,_ can be correctly answer without documents) samples and review potential wrongly-labeled samples. A reasonable speculation raises that our final benchmark can be biased toward GPT-4o’s answers, especially when GPT-4o outperforms others by a large margin. We discuss this potential bias as follows. 

We check the effect of GPT-4o’s involvement in the quality control step-by-step. Specifically, we compare the performance of samples remained after each step across GPT-4o and two other competitive models (GPT-4V and Gemini-1.5-Pro). We show their results in the table below. 

||**GPT-4o**<br>**GPT-4V**<br>**Gemini-1.5-Pro**|
|---|---|
|No quality control<br>+ document-relevance detection<br>+ document-relevance detection + self-refection / cross-checking|43.1%<br>35.2%<br>23.3%<br>41.2%<br>31.0%<br>20.5%<br>42.7%<br>31.4%<br>20.9%|



Table 6: Step-wise performance comparison with and without LVLM-based quality checking 

The results illustrate that the potential bias in step 1 (document-relevance detection) actually reduce, rather than increase, the performance gap between GPT4o and other models. It is because that we filter out all samples correctly answered by GPT4o without the access to documents. Under this case, the more significant performance drop of GPT-4V and Gemini-1.5-Pro can only be attributed to their limited document understanding and over-reliance on their internal knowledge. Regarding the step 2 and 3 (self-reflection and cross-checking), we provide inconsistent answers between human annotations and GPT4o’s predictions to annotators and ask them to check and revise accordingly. The potential bias of this step does lead to a slight performance bias (1.1% absolute difference at maximum). We believe that such bias is NOT the main cause of GPT4o’s significantly best performance. Without the involvement of GPT-4o in the quality control process, GPT-4o still significantly outperforms GPT-4V by 7.9% (43.1% - 35.2%) and Gemini-1.5-Pro by 19.8% (43.1% - 23.3%). Accordingly, all primary conclusions in our paper still hold. 

25 

## **A.7 GUI Screenshots** 

We present the screenshots for editing existing questions and annotating new questions (along with their reference answers and meta-data) in Figure 21 and Figure 22 respectively. 

Figure 21: GUI screenshot for editing existing questions (along with reference answers and meta-data) 

Figure 22: GUI screenshot for annotating new questions (along with reference answers and meta-data) 

## **A.8 Annotation Cost** 

This benchmark is annotated by the authors of this paper. Therefore, the data collection does not need compensation. And we count the time cost of our benchmark as below. 

**Pre-annotation** (about 45h): the development of annotation interface (10h), the writing of annotation guideline (5h), training session (10h), preliminary annotation and personalized feedback (20h). 

**Annotation** (about 150h): It takes about 60-90 minutes for the annotation of each document. And all of the 130 documents take about 150 hours. 

**Post-annotation** (about 45h): quality checking (30h), data processing and release preparation (15h). 

In summary, our benchmark annotation approximately takes a total of 45+150+45=240 hours (1.36 man months). 

26 

## **B Experimental Details** 

## **B.1 Prompt for Response Generation** 

Listing 1: Prompt used for response generation. The `[Document]` is in PNG format (page screenshots) for LVLMs, and TXT format for LLMs. 

```
[Document]
```

```
Readtheabovedocumentsandanswerthisquestion:
[question]
```

```
Pleasemakeyouranswerasconciseaspossible.
```

## **B.2 Prompt for Answer Extraction** 

Listing 2: Prompt used for answer extraction. 

- `Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.` 

- `Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".` 

- `Please make your response as concise as possible. Also note that your response should be formatted as below:` 

```
‘‘‘
Extractedanswer:[answer]
Answerformat:[answerformat]
‘‘‘
```

```
Pleasereadthefollowingexample,thenextracttheanswerfromthemodelresponse
andtypeitattheendoftheprompt.
```

```
---
```

```
Question:Listtheprimaryquestionsaskedabouttheservicesinthisreport.
Analysis:TheprimaryquestionsaskedabouttheservicesinthereportforTheLimes
ResidentialHomeare:
```

`1. Is the service safe?` 

`2. Is the service effective?` 

`3. Is the service caring?` 

`4. Is the service responsive?` 

```
5.Istheservicewell-led?
Extractedanswer:[’Istheservifesafe?’,’Istheserviceeffective’,’Istheserve
caring?’,’Istheserviceresponsive?’,’Istheservicewell-led?’]
Answerformat:List
```

```
---
```

- `Question: How many regulations of the HSCA 2008 are breached in all according to this report?` 

- `Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:` 

`1. Regulation 13: Safeguarding service users from abuse and improper treatment` 

`2. Regulation 12: Safe care and treatment` 

`3. Regulation 18: Staffing` 

`4. Regulation 11: Need for consent` 

`5. Regulation 10: Dignity and respect` 

`6. Regulation 9: Person-centred care` 

`7. Regulation 17: Good governance` 

`8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents` 

`9. Regulation 18: Failure to maintain an accurate and up-to-date care plan` 

27 

`10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively` 

```
Thesebreachesinvolveissuesconcerningstaffing,safeguarding,medicines
management,dignityandrespect,consent,careplanning,governance,and
failuretonotifytheCQCofincidents.
```

```
Extractedanswer:10
```

```
Answerformat:Integer
```

```
---
```

```
Question:AccordingtothesurveythatisthepercentageofChinesewhoarepaying
moreoraboutthesameattentiontopoliticsafterTrump’selection?
```

```
Analysis:ThesurveyprovideddoesnotspecifythepercentageofChineseindividuals
specificallywhoarepayingmoreoraboutthesameattentiontopoliticsafter
Trump’selection.ThereportfocusesprimarilyonAmericandemographicsand
doesnotincludespecificdetailsabouttheChinesepopulationinrelationto
thisquestion.Ifyouneedinformationaboutadifferentdemographicora
summaryofthefindingsfromtheAmericandemographic,Icancertainlyhelp
withthat!
```

```
Extractedanswer:Notanswerable
Answerformat:None
```

```
---
```

```
Question:Howmanyquotationsfrommalerespondentover50yearsoldareincludedin
thisreport?
```

```
Analysis:Theimageyou’veprovidedappearstobeascreenshotofadocumentwith
multiplecharts.However,thetextistoosmallandblurrytoreadaccurately.
Ifyoucanprovideaclearerimageormorecontext,Imightbeabletohelpyou
withyourquestion.
```

```
Extractedanswer:Failtoanswer
Answerformat:None
```

```
---
```

## **B.3 Rules for Score Calculation** 

We evaluate the model’s responses by scoring the extracted answers against the reference answers. The scorer is rule-based and employs different strategies according to the format of the reference answer. We detail its rules as below: 

**String:** We firstly use a series of regular expressions to determine whether the answers require exact matching ( _e.g.,_ telephone numbers, email addresses, website addresses, file names, times, dates, _etc._ ). If an exact match is needed, we perform a straightforward string comparison and score the answer either 0 or 1. Otherwise, we follow previous work [17] and calculate the ANLS (Average Normalized Levenshtein Similarity) with a pre-defined threshold ( _τ_ = 0 _._ 5). 

**Integer:** We perform an exact match comparison and score the answer either 0 or 1. 

**Float:** We view the prediction and reference answers as equal if they fall within a 1% relative tolerance. 

**List:** We adopt a relatively strict rule for scoring answers in list format: predictions that do not have the same number of elements as the reference receive a score of 0. For the remaining predictions, as Eq. 1 indicates, we score each element in order and use the minimum element-wise score as the score for the entire list. The element-wise scoring strategies is determined by the formats of elements ( _i.e.,_ string, integer or float). 

`pred_list` _,_ `ref_list` = `sorted(pred_list)` _,_ `sorted(ref_list) Score(pred_list, ref_list)` = `min` ( 

**==> picture [312 x 11] intentionally omitted <==**

(1) 

) 

28 

Evaluation detailed in the Appendix B.4 shows that while this scorer is not perfect, it aligns well with human judgment. We will continue refining these rules to cover more corner cases and enhance their accuracy. 

## **B.4 Human Evaluation on the Automatic Evaluation Pipeline** 

We conduct human evaluations to assess the performance of our automatic evaluation pipeline, which includes the answer extractor and the score calculator. Specifically, we randomly select 100 questions and review their responses from two representative LVLMs: GPT-4o and Gemini1.5-Pro. We manually evaluate the correctness of each response and compare the results between human evaluation and automatic evaluation. The performance, as shown in Table 7, indicates a high correlation between human judgment and our automatic pipeline. 

|**Model**|**Inconsistent Evaluation**<br>Ans. Extractor<br>Scorer<br>Overall|
|---|---|
|||
|GPT-4o<br>Gemini-1.5-Pro|4<br>2<br>6<br>2<br>2<br>4|



Table 7: We manually check 100 responses from GPT-4o and Gemni-1.5-Pro, and compare the evaluation results between humans and our automatic pipeline. 

## **B.5 Model Hyperparameters** 

The hyperparameters of used LVLMs and LLMs in Section 3.3 are detailed in Table 8. The temperature is set as 0 _._ 0, and the max_new_tokens is set as 1024 for all the models. The ‘concatenated_images’ parameter determines the maximum number of images that can be combined into a single input for LVLMs. By concatenating multiple images, we can meet the minimum context window requirements. The ‘max_pages’ parameter specifies the maximum number of images that can be directly input into the LVLMS without concatenation. 

|**Model**|**Hyperparameters**|
|---|---|
|_LLM_||
|ChatGLM-128k<br>Mistral-Instruct-v0.2-7B<br>Mixtral-Instruct-v0.1-8x7B<br>Mixtral-Instruct-v0.1-8x22B<br>QWen-Plus<br>DeepSeek-V2|max_input_words=60000<br>max_input_words=20000<br>max_input_words=20000<br>max_input_words=40000<br>max_input_words=16000<br>max_input_words=20000|
|_LVLM_||
|DeepSeek-VL-Chat<br>Qwen-VL-Chat<br>Idefcs2<br>MiniCPM-Llama3-V2.5<br>InternLM-XC2-4KHD<br>Monkey-Chat<br>CogVLM2-Llama3-Chat<br>InternVL-Chat-v1.5<br>EMU2-Chat|concatenated_images=5<br>concatenated_images=5<br>concatenated_images=5<br>concatenated_images=2<br>concatenated_images=2<br>concatenated_images=1<br>concatenated_images=1<br>concatenated_images=5<br>concatenated_images=5|
|_LLM & LVLM_||
|Claude-3 Opus<br>Gemini-1.5-Pro<br>GPT-4-turbo<br>GPT-4o|version=`claude-3-opus-20240229`, concatenated_images=20<br>max_pages=120, version=`gemini-1.5-pro-latest`<br>max_pages=120, version=`gpt-4-turbo-2024-04-09`<br>max_pages=120, version=`gpt-4o-2024-05-13`|



Table 8: Model Hyperparameters 

29 

## **C Qualitative Study** 

## **C.1 Error Analysis** 

We delve into the analysis of error by GPT-4o to further understand its bottlenecks and potentials on long-context document understanding. We manually check 72 incorrect responses and categorized their error reasons into 7 types. Except for the _Extraction Error_ caused by our automatic evaluation pipeline (see Appendix B.4), we detail and showcase another six reasons as below: 

**Perceptual Error:** GPT-4o sometimes struggles to extract or understand visual information from document screenshots. For instance, it misinterprets the axes and colored circles in the charts shown in Figure 23. Additionally, it inaccurately counts the number of green bars in Figure 24. They demonstrate that even the cutting-edge LVLMs still fall short in fundamental perceptual capabilities. 

**Incomplete Evidence:** Though GPT-4o has achieved significantly better _global searching abilities_ compared to other models when dealing with lengthy, multi-modal documents, it sometimes still omits certain information. For example, GPT-4o misses one chapter author from Columbia University in the full list (Figure 25). Additionally, it overlooks an app that appears across two pages (Figure 26). 

**Hallucinated Evidence:** As stated in Section 3.4, GPT-4o adopts more aggressive strategies and tends to provide more false-positive answers. It sometimes even fabricates non-existent evidence in documents to support its incorrect responses. For example, it references a non-existent page in Figure 27, and fabricates the content of a page in Figure 28. The above examples clearly reveal the importance of further research on LVLMs’ hallucination and safety. 

**Knowledge Lacking:** Resolving certain questions requires both information from the documents and the parametric knowledge within LVLMs. We have observed error cases stemming from the absence of specific knowledge. For example, GPT-4o overlooks details about the _fixed asset turnover ratio_ and uses the single-point value instead of the average value to calculate this metric (Figure 29). Additionally, it misidentifies buildings at Tsinghua University in Figure 30. 

**Reasoning Error:** Though not a primary cause, flawed reasoning based on correctly collected evidence and information from documents can sometimes lead to wrong answers. For example, GPT-4o correctly gathers all data but calculates a relative percentage instead of an absolute percentage in Figure 31. Additionally, as shown in Figure 32, it correctly lists all quizzes but inaccurately counts them in the final step. 

**Irrelevant Answer:** GPT-4o sometimes misunderstands the intent of questions and provides irrelevant responses. For instance, in Figure 33, GPT-4o answers about button operations when the question asks about button functions. Similarly, in Figure 34, where the question asks for the MOST discrimination type, GPT-4o summarizes all types instead. 

30 

## **Perceptual Error: Case 1** 

> Question: Among people who age 30-49, what is the difference of percentage value between Americans and Germen having positive view on their bilateral relationship? 

> Response: The difference in the percentage of people aged 30-49 who have a positive view of the U.S.-Germany relationship between Americans and Germans is 9% points. According to data on page 5, 40% of Americans aged 30-49 view the relationship positively, while only 31% of Germans in the same age group share this view. 

> Error Type: Perceptual Error 

> Analysis: The model wrongly identifies the percentages of people aged 30-49 who have a positive view of the U.S.-Germany relationship from the chart. It should be 35% and 72%, rather than 31% and 40%. 

Figure 23: Error example about _Perceptual Error_ 

31 

## **Perceptual Error: Case 2** 

**==> picture [207 x 85] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question:  How many green bars appear in Figure 1?<br>Response:  In Figure 1, there are four green bars.<br>Error Type:  Perceptual Error<br>Analysis:  The model wrongly counts 4 green bars<br>in Figure 1. Obviously, there are 6 green bars.<br>**----- End of picture text -----**<br>


Figure 24: Error example about _Perceptual Error_ 

32 

## **Incomplete Evidence: Case 1** 

> Question: How many chapter authors of this report come from Columbia University? 

> Response: The report lists the chapter authors on page xii. From Columbia University, the chapter authors are: 1. Radley Horton, 2. Patrick Kinney. Therefore, there are two chapter authors from Columbia University. 

> Error Type: Incomplete Evidence 

> Analysis: The model only finds out 2 of 3 chapter authors from Columbia University, leaving out Garcia-Pando shown in the top-left of Page 15. 

Figure 25: Error example about _Incomplete Evidence_ 

33 

## **Incomplete Evidence: Case 2** 

> Question: What are the overlapped apps between page 21 and 62? 

> Response: The overlapped apps between page 21 and page 62 are: - LinkedIn - SlideShare 

> Error Type: Incomplete Evidence 

> Analysis: The model correct identifies the two shared apps (LinkedIn and SlideShare) between two pages. But it leaves out another one app: Podcast. 

Figure 26: Error example about _Incomplete Evidence_ 

34 

## **Hallucinated Evidence: Case 1** 

**==> picture [308 x 112] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question:  How many sampled questions illustrated in this slide?<br>Response: There are five sample questions illustrated in this slide deck. They are<br>shown on the following slides:<br>1. Slide 10: "The following search string will produce which results according to the<br>Venn diagram?"<br>2. Slide 11: "A role you were previously recruiting for just re-opened. Which Talent<br>Pipeline feature should you utilize to source talent already in your pipeline?"<br>3. Slide 12: "Based on the report below, which recruiter should you go to for advice on<br>InMail best practices?"<br>4. Slide 13: "You recruit for Retail Bankers in high volume. Which efficiency tool(s)<br>will help you constantly uncover new leads?"<br>**----- End of picture text -----**<br>


**==> picture [301 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
5. Slide 14: "What happens to my certification with the rollout of Next-Gen Recruiter?"<br>**----- End of picture text -----**<br>


> Error Type: Hallucinated Evidence 

> Analysis: The model totally fabricates the contents of Slide 14 (Page 14). This slide is not about sampled question. 

Figure 27: Error example about _Hallucinated Evidence_ 

35 

## **Hallucinated Evidence: Case 2** 

> Question: Is there blue color handwritten words in page 30? 

> Response: Yes, there are blue handwritten words on page 30. 

> Error Type: Hallucinated Evidence 

> Analysis: This document has only 20 pages. There is no “so-called” page 30 in the response. 

Figure 28: Error example about _Hallucinated Evidence_ 

36 

## **Knowledge Lacking: Case 1** 

> Question: What is the FY2019 fixed asset turnover ratio for Activision Blizzard? 

> Response: The fixed asset turnover ratio is calculated by dividing net sales by net fixed assets. The formula is: Fixed Asset Turnover Ratio = Net Sales / Net Fixed Assets From the provided document, we can find the necessary values for the fiscal year 2019: - Net Sales (Net Revenues) for 2019: $6,489 million 

- Net Fixed Assets (Property and Equipment, net) at the end of 2019: $253 million Using these values, the fixed asset turnover ratio is: 6,489 / 253 ≈ 25.65 

Therefore, the fixed asset turnover ratio for Activision Blizzard for the fiscal year 2019 is approximately 25.65. 

## Error Type: Knowledge Lacking 

> Analysis: The model uses a wrong formula to calculate the financial metric. To calculate the Fixed Asset Turnover Ratio, the denominator should be the average value of net fixed assets in the past two years, instead of the value of a single year. 

Figure 29: Error example about _Knowledge Lacking_ 

37 

## **Knowledge Lacking: Case 2** 

> Question: How many figures in this document show the old gate of Tsinghua ("Er Xiao Men" in Chinese)? Response: The document contains two figures showing the old gate of Tsinghua ("Er Xiao Men" in Chinese). The first one is on page 2, and the second one is on page 27 

> Error Type: Knowledge Lacking 

> Analysis: Though it has similar appearance, the building shown in Page 27 (i.e. the right figure) is not “Er Xiao Men” but “Qing Hua Xue Tang”. 

Figure 30: Error example about _Knowledge Lacking_ 

38 

## **Reasoning Error: Case 1** 

> Question: If I drop the green module shown in Figure 6, what is the absolute percentage drop on 100-shot TACREV setting? 

Response: The absolute percentage drop on the 100-shot TACREV setting when the green module is dropped can be found in Table 4. For the 100-shot TACREV setting, the F1 score before reranking is 72.0, and after reranking, it is 74.3. The absolute percentage drop is calculated as follows: 

Absolute Percentage Drop = ((F1 score before reranking - F1 score after reranking) / F1 score after reranking) × 100 

Absolute Percentage Drop = ((74.3 - 72.0) / 74.3) × 100 Absolute Percentage Drop = (2.3 / 74.3) × 100 Absolute Percentage Drop ≈ 3.1% 

So, the absolute percentage drop on the 100-shot TACREV setting when the green module is dropped is approximately 3.1%. 

## Error Type: Reasoning Error 

> Analysis: The formula listed in model response is the relative percentage drop, instead of the absolute percentage drop. 

Figure 31: Error example about _Reasoning Error_ 

39 

## **Reasoning Error: Case 2** 

**==> picture [270 x 304] intentionally omitted <==**

**----- Start of picture text -----**<br>
on this exercise, develop a description of an effective manager. Question:  How many quizzes are there<br>out what they want to do with their time (planning), determine the<br>yourin that time?timeDoframeyou(organizing),think ahead, andor do makeyou tendsure theyto stay on in units 4, 5, and 6 combined?<br>leadershipyour time, inandan identifyinspirationalat leastrole threein a companyareas whereandyoumanagementcan improvein Response:  There are three quizzes in units 4,<br>thefitother? seemshes goodDo to you putmanagersthinkleadershipa companyand andbad managementleaders?can succeedAve managersinifait hierarchy.has badand 5, and 6 combined:<br>be comfortable using all kinds of technology. Do an inventory of - Unit 4: Quiz 2: Twenty Core Concepts on<br>in aasAftercomputerMicrosoftlistingclass yourExcel areasonbyorbuilding off of weakness,campus.a spreadsheetYoumakemaya wanttoplantrack toto practiceyourincrease Unit 3 & 4<br>slides for your next class project, and Microsoft Outlook by<br>Technology) - Unit 5: No quiz listed<br>- Unit 6: Quiz #3: Comprehension of Twenty<br>companies using? Core Concepts from Unit 5 & 6<br>structures?<br>be withinalteredtheirto make organizations? an organization more successful? So, the total number of quizzes in units 4, 5,<br> performancediffer?organize? of the company? and 6 is 3.<br>ITaboutReplacement hisand topiccite your and sources.then wrte responses to the folowing<br>companiestrainor providetheir justifiedreplacements? trainingin foroutsourcing displacedtechnology jobs workers? Shouldto organizationsIndia? Do Error Type:  Reasoning Error<br>“company organizational charts,” and find at least three<br> corporations,graphic organizerIs it organizednot-for-profits, provided.by function,or governmentproduct/service,agencies.process, Analysis:  the quizzes in each unit, but count their The response analyze correctly<br>Relations number wrongly at the last step.<br>and how are human resource needs determined?<br>Figure 32: Error example about  Reasoning Error<br>40<br>**----- End of picture text -----**<br>


## **Irrelevant Answer: Case 1** 

> Question: How many distinct button functions are there during a workout? Response: During a workout, the HONOR Watch GS Pro has the following distinct button functions: 

1. Press the Up button: Lock, mute, or end workout. 

2. Press the Down button: Switch screen. 

3. Press and hold the Up button: Finish the workout. 

4. Press and hold the Down button: Wake up the voice assistant. 

5. Swipe up or down on the screen: Switch screen. 

6. Swipe left or right on the screen: Switch screen. So, there are 6 distinct button functions during a workout. 

> Error Type: Irrelevant Answer 

> Analysis: The question asks the button functions. The response, however, answers the button operations and outputs an irrelevant answer. 

Figure 33: Error example about _Irrelevant Answer_ 

41 

## **Irrelevant Answer: Case 2** 

> Question: What types of discrimination are most commonly reported by women in STEM jobs compared to men in the survey of Pew Research Center, January 2018. 

> Response: In the Pew Research Center survey titled "Women and Men in STEM Often at Odds Over Workplace Equity," published in January 2018, several types of discrimination were reported by women in STEM jobs compared to men, and by women in workplaces with a higher proportion of women. Here are the key findings: 

1. **Gender Discrimination**: 

- **Overall Experience**: 50% of women in STEM jobs reported experiencing gender discrimination at work, compared to 19% of men in STEM jobs. 

- **Specific Types of Discrimination**: Women in STEM jobs reported various forms of gender discrimination, including earning less than a man doing the same job (29%), being treated as if they were not competent (29%), experiencing repeated small slights at work (20%), and receiving less support from senior leaders (18%). 

...... 

5. **Advanced Degrees**: 

- **Higher Discrimination**: 62% of women in STEM jobs with a postgraduate degree reported experiencing gender discrimination, compared to 41% of women with some college or less education. 

- **Impact on Fair Treatment**: Women with advanced degrees were less likely to believe that women are treated fairly in opportunities for promotion and advancement. 

These findings highlight the significant gender disparities and challenges that women in STEM fields face, particularly in male-dominated environments and specific job roles like computer-related positions. 

> Error Type: Irrelevant Answer 

> Analysis: The question asks about the MOST discrimination types, while response summarizes various types of discriminations towards women in STEM jobs. 

Figure 34: Error example about _Irrelevant Answer_ 

42 

## **C.2 Case Study** 

**==> picture [375 x 225] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question:  What is Interest Coverage Ratio for Activsion Blizzard In F2019?<br>Reference Answer:  17.85<br>thon GPT-4oAnswer: ✅ Interest Coverage Ratio (ICR) for Activision Blizzard in Fiscal Year 2019 can be  _somennnrgnmemnemnnnrnvnvnvnenrennnnenren Gemini-1.5-ProAnswer: The provided document is the 2019 10-K report for Activision Blizzard, but it does  ❌<br>calculated using the formula: not contain the specific term "Interest Coverage Ratio". To calculate this ratio, you need both<br>Interest Coverage Ratio = Earnings Before Interest and Taxes (EBIT) / Interest Expense the company's earnings before interest and taxes (EBIT) and its interest expense.<br>From the provided financial statements for the year ended December 31, 2019: You can find the following information within the document to calculate the Interest Coverage<br>- Operating Income (EBIT) = $1,607 million Ratio:<br>- Interest Expense = $90 million (from Interest and other expense (income), net) * **Operating income:** Page 47 (Consolidated Statements of Operations Data)<br>Thus, the Interest Coverage Ratio is:ICR = 1607/90 ~17.86  * **Interest expense (income), net:** Page 47 (Consolidated Statements of Operations Data)Please note:<br>This means that Activision Blizzard earned about 17.86 times its interest expenses before taxes  * You will need to use the provided figures to calculate the Interest Coverage Ratio using the<br>and interest charges in 2019. formula: **Interest Coverage Ratio = EBIT / Interest expense**.<br>GPT-4VAnswer: ✅ Interest Coverage Ratio (ICR) is calculated by dividing a company's earnings before  * The document may refer to the components of the ratio using slightly different terminology.Once you have located the relevant figures, you can calculate the Interest Coverage Ratio for Activision Blizzard in F2019.<br>interest and taxes (EBIT) by its interest expense.<br>From the provided document, we can find the necessary figures for Activision Blizzard for the<br>fiscal year 2019: InternVL-v1.5 x ❌<br>1. **Operating Income (EBIT)**: $1,607 million Operations) (from the Consolidated Statements of  Answer: The Interest Coverage Ratio for Activision Blizzard in F2019 is 1.9.<br>2. **Interest Expense**: $90 million  (from the Interest and Other Expense (Income), Net<br>section) OCR + GPT-4 x ❌<br>Using these figures, the Interest Coverage Ratio is calculated as follows: Answer: The Interest Coverage Ratio for Activision Blizzard in Fiscal Year 2019 is not<br>Interest Coverage Ratio = EBIT/Interest Expense = 1,607/90 ~ 17.86  Therefore, the Interest  explicitly stated in the provided text. To calculate the Interest Coverage Ratio, you would<br>Coverage Ratio for Activision Blizzard in fiscal year 2019 is approximately 17.86. typically use the formula:<br>Interest Coverage Ratio = EBIT / Interest Expense<br>OCR + Mixtral-8x22B x ❌ Where:<br>Answer: The Interest Coverage Ratio for Activision Blizzard in F2019 is not explicitly  - EBIT (Earnings Before Interest and Taxes) is typically operating income or earnings before<br>stated in the provided document. However, it can be calculated using the information  the deduction of interest and tax expenses.<br>provided in the document. The Interest Coverage Ratio is calculated by dividing Earnings  - Interest Expense is the cost incurred by an entity for borrowed funds.<br>Before Interest and Taxes (EBIT) by Interest Expense. According to the document,  From the provided text, specific figures for EBIT and Interest Expense for Activision Blizzard<br>Activision Blizzard had an EBIT of $1,633 million and an Interest Expense of $28 million in  in Fiscal Year 2019 are needed to calculate the Interest Coverage Ratio, and these figures are<br>F2019. Therefore, the  Interest Coverage Ratio for Activision Blizzard in F2019 is  not provided in the text excerpts. Therefore, it is not possible to calculate the Interest Coverage<br>approximately 58.32. Ratio without additional financial data from the company's financial statements or reports.<br>**----- End of picture text -----**<br>


Figure 35: Case Study. Evidence source: table. The evidence pages are zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red. 

43 

> Question: What devices other than phone are introduced for setting device+? 

> Reference Answer: 1.Vision  2.Bluetooth devices 

## GPT-4o ✅ 

> Answer: Other than phones, the devices introduced for setting Device+ in the Huawei Nova Y70 User Guide include: 

1. **Vision**: Ensure that it is connected to the same LAN and logged in to the same HUAWEI ID as your phone. 

2. **Bluetooth devices**: Some Bluetooth devices (such as Bluetooth headsets) can be linked with your phone via Device+ after establishing a Bluetooth connection. 

- These devices can be linked to your phone to enable seamless collaboration and task transfer 

- GPT-4V ❌ 

> Answer: The images you provided are pages from the user guide for the Huawei Nova Y70 smartphone. This guide includes detailed instructions on various features and settings of the device, such as basic gestures, camera usage, app management, and settings adjustments. If you have specific questions about any of the features or need further details on how to use the device, feel free to ask! 

## Gemini-1.5-Pro ❌ 

> Answer: The document you provided is a user guide for the Huawei nova Y70 phone and it doesn't contain information on "device+". Therefore, I can't answer your question. 

InternVL-v1.5 ❌ 

> Answer: The devices introduced for setting device+ are Bluetooth devices, such as Bluetooth headphones, and other devices that support Bluetooth. 

## OCR + GPT-4 ✅ 

> Answer: The devices other than phones that are introduced for setting up Device+ include: 

1. **Vision**: This refers to devices like smart TVs or displays that can be linked with your phone for a seamless multi-device experience. 

2. **Bluetooth devices**: Such as Bluetooth headsets, which can be connected to your phone via Device+ for audio streaming or calls. 

These devices can be controlled and interacted with through your phone, allowing for tasks like transferring MeeTime calls, audio, or video content to your Vision device, or managing audio playback through connected Bluetooth devices. 

OCR + Mixtral-8x22B ✅ 

> Answer: For setting Device+, ensure that devices to be connected support Device+. Before you get started, enable Bluetooth and Wi-Fi and log in to your HUAWEI ID on your phone. For other devices, perform the following settings: 

- + Vision: Ensure that it is connected to the same LAN and logged in to the same HUAWEI ID as your phone. 

- + Bluetooth device: Some Bluetooth devices (such as Bluetooth headsets) can be linked with your phone via Device+ after establishing a Bluetooth connection. 

- @ Device+ does not support collaboration between phones. 

- If Device+ is hidden, access Control Panel and go to > Show Device+. 

Figure 36: Case Study. Evidence source: text. The evidence pages are zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red. 

44 

> Question: At NTU, how many types of Field Sports can students participate in? 

> Reference Answer: 4 

## GPT-4o ✅ 

> Answer: At Nanyang Technological University (NTU), students can participate in four types of field sports: 1.Cricket 2.Football 3.Rugby 4.Touch Football. 

GPT-4V ✅ 

> Answer: At Nanyang Technological University (NTU), students can participate in several types of field sports. According to the guidebook, the field sports available include: 1. Cricket 2. Football 3. Rugby 4. Touch Football These sports provide a variety of options for students interested in field sports, catering to different interests and skills. 

## Gemini-1.5-Pro ❌ 

> Answer: The provided pamphlet does not list the number of Field Sports students can participate in. Therefore, I cannot answer your question. 

## InternVL-v1.5 ❌ 

> Answer: Based on the image, students at NTU can participate in 5 types of Field Sports. These are listed as Football, Netball, Rugby, Volleyball, and Ultimate Frisbee. OCR + Mixtral-8x22B ❌ 

> Answer: Students at NTU can participate in 10 types of Field Sports: Badminton, Cricket, Football, Rugby, Touch Football, Ultimate Frisbee, Golf, Mind Games, International Chess, and Snooker and Pool. 

## OCR + GPT-4 ❌ 

> Answer: At NTU, students can participate in six types of Field Sports. These are Cricket, Football, Rugby, Touch Football, Mind Games, and Other Sports. 

Figure 37: Case Study. Evidence source: layout. The evidence page is zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red. 

45 

> Question: How many children are being read to in the photograph of Page 3? 

> Reference Answer: 2 

GPT-4o ✅ 

> Answer: In the photograph on Page 3, there are two children being read to. GPT-4V ✅ 

> Answer: In the photograph on Page 3, there are two children being read to by an adult. Gemini-1.5-Pro ✅ 

> Answer: There are **two** children being read to in the photograph on page 3. 

> InternVL-v1.5 x ❌ 

> Answer: In the photograph of Page 3, there are three children being read to by the woman. 

OCR + GPT-4 ❌ 

> Answer: The text provided does not contain any information about a photograph or the number of children being read to in a photograph on Page 3. Therefore, I'm unable to answer the question based on the given text. If you have access to the photograph or additional details, please share them, and I would be happy to help with your query. 

> OCR + Mixtral-8x22B x ❌ 

> Answer: There are three children being read to in the photograph on Page 3. 

Figure 38: Case Study. Evidence source: image. The evidence page is zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red. 

46 

## **D Limitations** 

MMLONGBENCH-DOC is the first comprehensive benchmark designed to evaluate the long-context document understanding capabilities of LVLMs. While our benchmark addresses significant gaps in the previous datasets, we acknowledge several limitations. 

One primary limitation is the scale of the benchmark. Currently, our benchmark includes a test set comprising 135 documents and 1,082 questions. It is much smaller compared to previous datasets. The complexity and difficulty of annotations limit the scale of our benchmark. As a long-context benchmark, our documents average about 50 pages and 20,000 tokens. And most questions require either complicated reasoning or cross-page comprehension. It takes more than one hour for an expert-level annotator to read through a single document, and then edit existing instances and create new instances on this document. Given the purpose of MMLONGBENCH-DOC as an evaluation benchmark, we prioritize annotation quality over quantity. Moreover, the results presented in Sections 3.3 and 3.4 confirm that the scale of our benchmark is sufficient for fine-grained evaluations across different document types, evidence sources, evidence pages, _etc._ . Additionally, we plan to expand our benchmark by adding more documents and questions in future iterations. 

We roughly categorize these questions into three types, _i.e.,_ single-page, cross-page, and unanswerable questions, based on whether evidence can be found in the documents and the number of evidence pages. However, unlike MMBench [41] or MathVista [56], we provide no further taxonomy to classify some ( _e.g.,_ 7 or 20) fine-grained, evaluated reasoning or perception capabilities out of two main reasons: (1) Prior ( _i.e.,_ pre-annotation) taxonomy limits the diversity of the questions. Therefore we provide no predefined classifications in our guideline and encourage the expert-level annotators to freely write questions without constraints. (2) The intrinsic complexity of document understanding presents significant challenges for establishing a posterior ( _i.e.,_ post-annotation) taxonomy. 

While there exist limitations in our benchmark, MMLONGBENCH-DOC surely represents a significant step forward in this field. We would iteratively maintain and refine this benchmark and hope it could push forward the development of long-context document understanding. 

## **E Social Impacts** 

The development and use of MMLONGBENCH-DOC may have potential societal implications. For instance, biased or inaccurate outputs from benchmarked models could perpetuate harmful stereotypes or reinforce existing social inequalities. Additionally, the ability to process and analyze long documents could potentially be used to surveil or monitor individuals’ personal information. Developers and users of MMLONGBENCH-DOC benchmark must be aware of these potential consequences and take steps to ensure responsible development and deployment of AI models. 

## **F Author Statement** 

The authors state that all of the previous datasets that we collected are licensed under the Creative Commons license (CC-BY) or other open-source licenses. Using this dataset should abide by the policy of OpenAI. Regarding the newly collected documents, we manually check them to ensure their availability for academic use. Should any authors request the removal of their documents, we will promptly comply. 

47 

## **Checklist** 

1. For all authors... 

   - (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] 

   - (b) Did you describe the limitations of your work? [Yes] See Appendix D. 

   - (c) Did you discuss any potential negative societal impacts of your work? [Yes] See supplemental material E. 

   - (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes] 

2. If you are including theoretical results... 

   - (a) Did you state the full set of assumptions of all theoretical results? [N/A] We didn’t involve theory in this benchmark. 

   - (b) Did you include complete proofs of all theoretical results? [N/A] 

3. If you ran experiments (e.g. for benchmarks)... 

   - (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] `https://mayubo2333.github.io/MMLongBench-Doc` 

   - (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [N/A] 

   - (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [N/A] 

   - (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] See Section 4. 

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets... 

   - (a) If your work uses existing assets, did you cite the creators? [Yes] 

   - (b) Did you mention the license of the assets? [Yes] See Appendix F. 

   - (c) Did you include any new assets either in the supplemental material or as a URL? [Yes] `https://mayubo2333.github.io/MMLongBench-Doc` 

   - (d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [Yes] See Appendix A.1 and A.2 

   - (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A] 

5. If you used crowdsourcing or conducted research with human subjects... 

   - (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A] 

   - (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A] 

   - (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A] 

48 

