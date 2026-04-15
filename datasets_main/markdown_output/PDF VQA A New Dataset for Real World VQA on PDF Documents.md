# **PDF-VQA: A New Dataset for Real-World VQA on PDF Documents** 

Yihao Ding[1] _[⋆]_ and Siwen Luo[1] _[∗]_ , Hyunsuk Chung[3] , Soyeon Caren Han[1] _[,]_[2] _[⋆⋆]_ 

> 1 Unversity of Sydney, Sydney NSW 2006 

_{_ `yihao.ding,siwen.luo,caren.han` _}_ `@sydney.edu.au` 

> 2 The University of Western Australia, Perth WA 6009 `caren.han@uwa.edu.au` 

> 3 FortifyEdge, Sydney, Australia, NSW `davidchungproject@gmail.com` 

**Abstract.** Document-based Visual Question Answering examines the document understanding of document images in conditions of natural language questions. We proposed a new document-based VQA dataset, PDF-VQA, to comprehensively examine the document understanding from various aspects, including document element recognition, document layout structural understanding as well as contextual understanding and key information extraction. Our PDF-VQA dataset extends the current scale of document understanding that limits on the single document page to the new scale that asks questions over the full document of multiple pages. We also propose a new graph-based VQA model that explicitly integrates the spatial and hierarchically structural relationships between different document elements to boost the document structural understanding. The performances are compared with several baselines over different question types and tasks[4] . 

**Keywords:** Document Understanding · Document Information Extraction · Visual Question Answering 

## **1 Introduction** 

With the rise of digital documents, document understanding received much attention from leading industrial companies, such as IBM [34] and Microsoft [31,30]. Visual Question Answering (VQA) on visually-rich documents (i.e. scanned document images or PDF file pages) aims to examine the comprehensive document understandings in conditions of the given questions [13]. A comprehensive understanding of a document includes understanding document contents [6,7], the document layout structures [24] and the recognition of document elements [18,25]. 

The existing document VQA mainly examines the understanding of the document in terms of contextual understanding [20,28] and key information extraction [10,23]. Their questions are designed to ask about certain contents 

> _⋆_ Co-First Authors 

> _⋆⋆_ Corresponding Author 

> 4 The full dataset will be released after paper acceptance. The partial dataset is provided in the supplementary material. 

2 Y. Ding et al. 

on a document page. For example, the question “What is the income value of consulting fees in 1979?” expects the specific value from the document contents. Such questions examine the model’s ability to understand questions and document textual contents simultaneously. 

Apart from the contents, the other important aspect of a document is its structured layout which forms the content hierarchically. Including such structural layout understandings in the document, the VQA task is also critical to improve the model’s capabilities in understanding the documents from a high level. Because in real-world document understandings, apart from querying about certain contents, it is common to query a document from a higher level. For example, a common question would be “What is the figure on this page about?” and answering such a question requires the model to recognize the figure element and understand that the figure caption, which is structurally associated with the figure, should be extracted and returned as the best answer. 

Additionally, the existing document VQA limits the scale of document understanding to a single independent document page [20,28]. But most document files of human’s daily work are multi-page documents with successively logical connections between pages. It is a more natural demand to holistically understand the full document file and capture the connections of textual contents and their structural relationships across multiple pages rather than the independent understanding of each page. Thus, it is significant to expand the current scale of page-level document understanding to the full document-level. 

In this work, we propose a new document VQA dataset, PDF-VQA, that contains questions to comprehensively examine document understandings from the aspects of 1)document element recognition 2) and their structural relationship understanding 3) from both page-level and full document-level. Specifically, we set up three tasks for our dataset with questions that target different aspects of document understanding. The first task mainly aims at the document elements recognition and their relative positional relationship understandings on the pagelevel, the second task focuses on the structural understanding and information extraction on the page level, and the third task targets the hierarchical understanding of document contents on the full document level. Moreover, we adopted the automatic question-answer generation process to save human annotation time and enrich the dataset with diverse question patterns. We have also explicitly annotated the relative hierarchical and positional relationships between document elements. As shown in Table 1, our PDF-VQA provides the hierarchically logical relational graph and spatial relational graph, indicating the different relationship types between document elements. This graph information can be used in model construction to learn the document element relationships. We also propose a graph-based model to give insights into how those graphs can be used to gain a deeper understanding of document element relationships from different aspects. 

Our contributions are summarized as 1) We propose a new document-based VQA dataset to examine the document understanding of comprehensive aspects, including the document element recognition and the structural layout understanding; 2) We are the first to boost the scale of document VQA questions from 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

3 

**Table 1.** Summary of conventional document-based VQA. Answer type abbreviations are MCQ: Multiple Choice; Ex: Extractive; Num: Numerical answer; Y/N: yes/no; Ab: Abstractive. Datasets with a tick mark in Text Info. the column provides the textual information/OCR tokens on the image/document page ROI. LR graph: logical relational graph; SR graph: spatial relational graph. 

|**Dataset**|**Source**|**Q. Coverage**|**Answer Type**|**Img. #**|**Q. #**|**Text Info.**|**Relation Info.**|
|---|---|---|---|---|---|---|---|
|TQA [15]<br>DVQA [13]<br>FigureQA [14]<br>PlotQA [21]<br>LEAFQA [3]<br>DocVQA [20]<br>VisualMRC [28]<br>InfographicVQA [19]|Science Diagrams<br>Bar charts<br>Charts<br>Charts<br>Charts<br>Single Doc Page<br>Webpage Screenshot<br>Infographic|diagram contents<br>chart contents<br>chart contents<br>chart contents<br>chart contents<br>doc contents<br>page contents<br>graph contents|MCQ<br>Ex, Num, Y/N<br>Y/N<br>Ex, Num, Y/N<br>Ex, Num, Y/N<br>Ex<br>Ab<br>Ex, Num|1K<br>300K<br>180K<br>224K<br>250K<br>12K<br>10K<br>5.4K|26K<br>3.4M<br>2.4M<br>29M<br>2M<br>50K<br>30K<br>30K|✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓|✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗|
|**PDF-VQA TaskA**<br>**PDF-VQA TaskB**<br>**PDF-VQA TaskC**|Single Doc Page<br>Single Doc Page<br>Entire Doc|doc elements<br>doc structure<br>doc contents|Ex, Num, Y/N<br>Ex<br>Ex|12k<br>12K<br>1147|81K<br>54K<br>5.7K|✓<br>✓<br>✓|LR graph<br>SR graph|



the page-level to the full document level; 3) We provide the explicit annotations of spatial and hierarchically logical relation graphs of document elements for the easier usage of relationship features for future works; 4) We propose a strong baseline for PDF-VQA by adopting the graph-based components. 

## **2 Related Work** 

Since the VQA task was introduced [1], the image source of the VQA task could be divided into three types: realistic/synthetic photos, scientific charts, and document pages. _**VQA with realistic or synthetic photos**_ is widely known as the conventional VQA [1,8,12,11]. These realistic photos contain diverse object types and the questions of the conventional VQA query about the recognition of objects and their attributes and the positional relationship of the objects. The later proposed scene text VQA problem [22,26,2,29] involves realistic photos with scene texts, such as the picture of a restaurant with its brand name. The questions of scene text VQA query about recognising the scene texts associated with objects in the photos. _**VQA with scientific charts**_ [13,14,3,21] contain the scientificstyle plots, such as bar charts. The questions usually query trend recognition, value comparison, and the identification of chart properties. _**VQA with document pages**_ involves images of various document types. For example, the screenshots of web pages that contain short paragraphs and diagrams [28], info-graphics [19], and single document pages of scanned letters/reports/forms/invoices [20]. These questions usually query the textual contents of a document page, and most answers are text spans extracted from the document pages. 

VQA tasks on document pages are related to Machine Reading Comprehension (MRC) tasks in terms of questions about the textual contents and answered by extractive text spans. Some research works [20,28] also consider it as an MRC task, so it can be solved by applying language models on the texts extracted from the document pages. However, input usage is the main difference between MRC 

Y. Ding et al. 

4 

**Table 2.** Data Statistics of Task A, B, and C. The numbers in _Image_ row for Task A/B refer to the number of document pages but the entire document number for Task C. 

|**Task**||**Type**|**Train**|**Valid**|**Test**|**Total**|
|---|---|---|---|---|---|---|
|Task|A|Image<br>Question|8,593<br>59,688|1,280<br>7,247|2,464<br>14,150|12,337<br>81,085|
|Task|B|Image<br>Question|8,593<br>37,428|1,280<br>5,660|2,464<br>10,784|12,337<br>53,872|
|Task|C|Document<br>Question|800<br>3,951|115<br>581|232<br>1,121|1,147<br>5,653|



and VQA. Whereas MRC is based on pure texts of paragraphs and questions, document-based VQA focuses on the processing of image inputs and questions. Our PDF-VQA is based on the document pages of published scientific articles, which requires the simultaneous processing of PDF images and questions. We compare VQA datasets of different attributes in Table 1. While the questions of previous datasets mainly ask about the specific contents of document pages or the certain values of scientific charts/diagrams, our PDF-VQA dataset questions also query the document layout structures and examine the positional and hierarchical relationships understandings among the recognized document elements. 

## **3 PDF-VQA Dataset** 

Our PDF-VQA dataset contains three subsets for three different tasks to mainly examine the different aspects of document understanding: Task A) Page-level Document Element Recognition, B) Page-level Document Layout Structure Understanding, and C) Full Document-level Understanding. Detailed dataset statistics are in Table 2. 

**Task A** aims to examine the document element recognition and their relative spatial relationship understanding on the document page level. Questions are designed into two types to verify the existence of the document elements and count the element numbers. Both question types examine relative spatial relationships and understandings between different document elements. For example, “Is there any table _below_ the ’Results’ section?” in Figure 1 and ”How many tables are on this page?”. Answers are yes/no and numbers from a fixed answer space. 

**Task B** focuses on understanding the document layout structures spatially and logically based on the recognized document elements on the document page level and extracting the relevant texts as answers to the questions. There are two main question types: structural understanding and object recognition. The structural understanding questions relate to examining spatial structures from both relative positions or human reading order. For example, “What is the _bottom_ section about?” requires understanding the document layout structures from the relative bottom position and “What is the _last_ section about?” requires identifying the last section based on the human reading order of a document. The object recognition questions explicitly contain a specific document element in 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

5 

**Fig. 1.** PDF-VQA sample questions and document pages for Task A, B, and C. 

the questions and require to recognition of the queried element first, such as the question “What is the bottom table about?” in Figure 1. Answering these two types of questions require a logical understanding of the hierarchical relationships of document elements. For instance, based on the textual contents, the section title would be a logically high-level summarization of its following section and is regarded as the answer to “What is the last section about?”. Similarly, a table caption is logically associated with a table; table caption contents would best describe a table. 

**Task C** questions have a sequence of answers extracted from multi-pages of the full document. It enhances the document understanding from the page to the full document level. Answering a question in Task C requires reviewing the full document contents and identifying the contents hierarchically related to the queried item in the question. For example, the question “Which section does describe Table 2?” in Figure 1 requires the identification of all the sections of the full document that have described the queried table. The answers to such questions are the texts of the corresponding section titles extracted as the high-level summarization of the identified sections. Identifying the items at the higher-level hierarchy of the queried item is defined as the parent relation understanding the question in PDF-VQA. Oppositely, Task C also contains the questions of identifying the items at the lower-level hierarchy of the queried item, and such questions are defined as the child relation understanding. For example, a question, “What does the ‘Methods’ section about?” requires extracting all the subsection titles as the answer. 

The detailed question type distribution of each task is shown in Table 3. 

6 Y. Ding et al. 

## **3.1 Data Source** 

Our PDF-VQA dataset collected the PDF version of visually-rich documents from the PubMed Central (PMC) Open Access Subset[5] . Each document file has a corresponding XML file that provides the structured representations of textual contents and graphical components of the article[6] . We applied the pretrained Mask-RCNN [34] over the collected document pages to get the bounding boxes and categories for each document element. The categories initially consisted of five common PDF document element types: _title_ , _text_ , _list_ , _figure_ , and _table_ . We then labelled the _text_ elements that are positionally closest to the tables and figures into two additional categories _table caption_ and _figure caption_ respectively. 

## **3.2 Relational Graphs** 

Visually rich documents of scientific articles consist of fixed layout structures and hierarchically logical relationships among the sections, subsections and other elements such as tables/figures and table/figure captions. Understanding such layout structures and relationships is essential to boost the understanding of this type of document. The graph has been used as an effective method to represent the relationships between objects in many tasks [4,32,33,18]. Inspired by this, for each document, we annotated the hierarchically _logical relational_ graph (LR graph) and _spatial relational_ graph (SR graph) to explicitly represent the logical and spatial relationships between document elements respectively. Those two graphs can be directly used by any deep-learning mechanisms to enhance the feature representation. In Section 6, we propose a graph-based model to enlighten how such relational information can solve the PDF-VQA questions. The SR graph indicates the relative spatial relationships between document elements based on their absolute geometric positions with their bounding box coordinates. For each document element of a single document page, we identify its relative spatial relationships with all the other document elements among eight spatial types: _top, bottom, left, right, top-left, top-right, bottom-left_ and _bottom-right_ . The LR graph indicates the potential affiliation between document elements by identifying the parent object and their children’s objects based on the hierarchical structures of document layouts. We follow [18] to annotate the parent-child relations between the document elements in a single document page to generate the LR graph. The graph of the full document of multiple pages are augmented by the graphs of its document pages. 

## **3.3 Question Generation** 

Visually rich documents of scientific articles have consistent spatial and logical structures. The associated XML files of these documents provide detailed logical 

- 5 `https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/` 

- 6 It follows the XML schema module provided by the Journal Archiving and Interchange Tag Suite created by the National Library of Medicine (NLM) `https://dtd.nlm. nih.gov/` 

7 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

**Table 3.** Ratio and exact number of various question types of Task A, B and C. 

|**Tasks**|**Question Type**|**Percentage**|**Total**|
|---|---|---|---|
|Task A|Counting<br>Existence|17.74<br>82.26|14,387<br>66,698|
|Task B|Structural Understanding<br>Object Recognition|88.58<br>11.42|47,722<br>6,150|
|Task C|Parent Relationship Understanding<br>Child Relationship Understanding|79.71<br>20.29|4,506<br>1,147|



structures between semantic entities. Based on this structural information and the pre-defined question template, we applied an automatic question-generation process to generate large-scale question-answer pairs efficiently. For example, the question _“How many tables are above the ‘Discussion’?”_ is generated from the question template _“How many 〈E1〉 are 〈R〉 the ‘〈E2〉’?”_ by filling the masked terms _〈E1〉_ , _〈R〉_ and _〈E2〉_ with document element label (“table”), positional relationship (“above”) and title name extracted from document contents (“Discussion”) respectively. We prepare each question template with various language patterns to diversify the questions. For instance, the above template can also be written as _“What is the number of 〈E1〉 are 〈R〉 the ‘〈E2〉’?”_ . We have 36, 15, and 15 question patterns for Task A, B, and C, respectively. We limit the parameter values of the document element label to only _title, list, table, figure_ as asking for the number/existence/position of _text_ elements would be less valuable. The parameter values include four document element labels, eight positional relationships ( _top, bottom, left, right, top-left, top-right, bottom-left_ and _bottom-right_ ), ordinal form ( _first, last_ ) and the texts from document contents (e.g. section title, references, etc.). We also replace some parameter values with their synonyms, such as _“on the top of”_ for _“above”_ . 

To automatically generate the ground truth answers to our questions, we first represent each document page (for Task A and B)/the full document (for Task C) with all the document elements and the associated relations from the two relational graphs as in Section 3.2. We then apply the functional program, which is uniquely associated with each question template and contains a sequence of functions representing a reasoning step, over such document(page) representations to reach the answer. For example, the functional program for question _“How many tables are above of the ‘Discussion’?”_ consists of a sequence of functions _filter-unique → query-position → filter-category → count_ to filter out the document elements that satisfy the asked positional relationships and count the numbers of them as the ground-truth answer. 

Moreover, we conduct the question balancing from answer-based and questionbased aspects to avoid question-conditional biases and balance the answer distributions. Firstly, we conduct an answer-based balancing by down-sampling questions based on the answer distribution. We identify the QA pairs with large ratios, divide identified questions into groups based on the patterns, and reduce QA pairs with large ratios until the answer distributions are balanced. After that, 

8 Y. Ding et al. 

we further conducted the question-based balancing to avoid duplicated question types. To achieve this, we smooth over the distributions of parameter values filled in the question templates by removing the questions with large proportions of certain parameter values until the balanced distribution of parameter value combinations. Since the parameter values of Task C question templates are almost unique, as all of them are the texts from document contents, we did not conduct the balancing over Task C. After the balancing, Task A questions are down-sampled from 444,967 to 81,085, and Task B questions are down-sampled from 246,740 to 53,872. 

## **4 Dataset Analysis and Evaluation** 

## **4.1 Dataset Analysis** 

The average number of questions per document page/document in Task A, B, and C are 6.57, 4.37, and 4.93. The average question length for Task A, B and C are 25, 10 and 15, respectively[7] . A sunburst plot showing each task’s top 4 question words is shown in Figure 2. We can see that Task A question priors are more diverse to complement the simplicity of document element and position recognition questions and to prevent the model from memorizing question patterns. For Task B and C, question priors distribute over “What”, “When”, “Can you”, “Which”. And we also specifically design questions in a declarative sentence with “Name out the section...” in Task C. 13.43%, 0.24% and 29.38% of the questions in Task A, B, and C are unique questions. This unique question ratio seems low compared to other document-based VQA datasets. This is because, rather than only aiming at the textual understanding of certain page contents, our PDF dataset targets more the spatial and hierarchically structural understandings of document layouts. Our questions are generally formed to ask about the document structures from a higher level and thus contain less unique texts that are associated with the specific contents of each document page. Answers for Task A questions are from the fixed answer space that contains eight possible answers: _“yes”, “no”, “0”, “1”, “2”, “3”, “4”_ and _“5”_ . Answers for Task B and C are texts retrieved from the document page/entire document. We also analyzed the top 15 frequent question patterns in Task A, B and C as shown in Figure 3 to show the common questions of each question type in each task. We used a placeholder “X” to replace the different figures, table numbers or section titles that would exist in the questions to present the common question patterns in this analysis. 

## **4.2 Human Evaluation** 

To evaluate the quality of automatically generated question-answer pairs, we invited ten raters, including deep-learning researchers and crowd-sourcing workers. Firstly, to determine the relevance between the question and the corresponding page/document, we define the _Relevance_ criteria. Correspondingly, we define 

> 7 We provide the question length distribution analysis in Appendix C.1 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

9 

**Fig. 2.** The top 4 words of questions in Task A, B and C. 

**Fig. 3.** Top 15 Frequency Questions of Task A, B and C. 

_Correctness_ to determine whether the auto-generated answer is correct to the question. In addition, we ask raters to judge whether our QA pairs are meaningful and possibly appear in the real world by using _Meaningfulness_ criteria[8] . After we collect the raters’ feedback, we calculate the positive rate of each perspective and apply Fleiss Kappa to measure the agreements between multiple raters, as can be seen in Table 4. All three tasks achieve decent positive rates with substantial or almost perfect agreements. For Task A, _Relevance_ and _Correctness_ can reach positive rates with nearly perfect agreements. Few raters gave negative responses regarding the _Meaningfulness_ of questions about the existence of tables or figures, while those questions are crucial to understanding the document layout for any upcoming table/figure contents understanding questions. In Task B, all three perspectives achieve high positive rates with substantial agreements. 

- 8 More details and human evaluation survey examples can be found in Appendix B. 

10 Y. Ding et al. 

**Table 4.** Positive rates ( **Pos(%)** ) and Fleiss Kappa Agreement ( **Kappa** ) of human evaluation. 

|**Task A**<br>**Task B**<br>**Task C**|**Task A**<br>**Task B**<br>**Task C**|**Task A**<br>**Task B**<br>**Task C**|**Task A**<br>**Task B**<br>**Task C**|
|---|---|---|---|
|**Perspective**|**Pos(%) Kappa**|**Pos(%) Kappa**|**Pos(%) Kappa**|
|Relevance<br>Correctness<br>Meaningfulness|98.46<br>94.02<br>99.49<br>98.12<br>96.94<br>88.97|91.67<br>77.07<br>89.44<br>72.56<br>93.61<br>77.67|100<br>100<br>94.55<br>80.93<br>99.27<br>97.34|



The disagreements about Task B mainly come from the questions with no specific answer (N/A), some raters thought those questions were incorrect and meaningless, but these questions are crucial to understanding the commonly appearing real-world cases. Because it is possible that a page does not contain the queried elements in the question, and no specific answer is a reasonable answer for such cases. Finally, for Task C, both positive rates and agreement across three perspectives are notable. In addition, except for three perspectives, raters agree most of the questions in Task C need cross-page understanding (the positive rate is 82.91%). 

## **5 Baseline Models** 

We experimented with several baselines on our PDF-VQA dataset to provide a preliminary view of different models’ performances. We choose the visionand-language models that have proved good performances on VQA tasks and a language model as listed in Table 5. We followed the original settings of each baseline but only made modifications on the output layers to suit different PDF-VQA tasks[9] . 

## **6 Proposed Model: LoSpa** 

In this paper, we introduce a strong baseline, Logical and Spatial Graph-based model ( _**LoSpa**_ ), which utilizes logical and spatial relational information based on logical (LR) and spatial (SR) graphs introduced in Section 3.2. 

**Input Representation** : we treat questions as sequential plain text inputs and encode them by BERT. For document elements of given document page _I_ such as _Title_ , _Text_ , _Figure_ , we use pre-trained ResNet-101 backbones to extract visual representations _Xv ∈_ R _[N][×][d][f]_ and use [CLS] token from BERT as the semantic representation _Xs ∈_ R _[N][×][d][s]_ for the texts of each document element. 

**Relational Information Learning** : We construct two graphs: logical graph _Gl_ = ( _Vl, El_ ) and spatial graph _Gs_ = ( _Vs, Es_ ) for each document page. For the logical graph _Gl_ , based on [18], we define the semantic feature as node 

> 9 The detailed baseline model setup can be found in Appendix C, and the code for the baseline model and our proposed model will be released in GitHub after paper acceptance. 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

11 

**Fig. 4.** Logical and Spatial Graph-based Model Architecture for three tasks. Task A, B and C use the same relational information to enhance the object representation but different model architectures in the decoding stage. 

representation _Vl_ and the existence of parent-child relation between document elements (extracted from the logical relational graph annotation in our dataset) as binary edge values _El {_ 0 _,_ 1 _}_ . Similarly, for spatial graph _Gs_ , we follow [18] to use the visual features of document elements as node representation _Vs_ and the distance with two nearest document elements to weight edge value _Es_ . 

For each document page _I_ , we take _Xs ∈_ R _[N][×][d][s]_ and _Xv ∈_ R _[N][×][d][f]_ as the initial node feature matrix for _Gl_ and _Gs_ respectively. These initial node features are fed into a two-layer Graph Convolution Network (GCN) and trained by predicting each node category. After the GCN training, we extract the first layer hidden states as the updated node representations _Xs[′][∈]_[R] _[N][×][d]_[and] _[X] v[′][∈]_[R] _[N][×][d]_ that has augmented the relational information between document elements for _Gs_ and _Gf_ respectively, where _d_ = 768. For each aspect feature, we conduct separated linear transformations to the initial feature matrices ( _Xv_ / _Xs_ ) and the updated feature matrices ( _Xs[′]_[/] _[X] v[′]_[).][Inspired][by][[][18][],][we][apply][the][element-] wise max-pooling over them. The pooled features _Xs[′′]_[and] _[X] v[′′]_[are][the][final] semantic and visual representations of nodes enhanced by logical and spatial relations, respectively. Finally, we concatenate semantic and visual features of each document element, yielding relational information enriched multi-modal object representations _O_ 1 _, O_ 2 _, ..., ON_ . 

**QA prediction** : We sum up the object features _O_ 1 _, O_ 2 _, ..., ON_ with positional embedding to integrate the information of document elements orders, which are inputs into multiple transformer encoder layers together with the results of the sequence of question word features _q_ 1 _, q_ 2 _, ..., qT_ . We pass the encoder outputs into the transformer decoders and apply a pointer network upon the decoder output to predict the answers. We apply a one-step decoding process each time using the word embedding _wi_ of one answer from the fixed answer space as the decoder input. Let the _zi[dec]_ be the decoder output for the decoder input _wi_ ; we then conduct the score _yt,i_ between _zi[dec]_ and the answer word embedding _wi_ following _yt,i_ = ( _wi_ ) _[T] zt[dec]_ + _b[dec] i_ , where _i_ = 1 _, ..., C_ , and _C_ are the total answer numbers 

12 Y. Ding et al. 

of the fixed answer space for each task. We apply a softmax function over all the scores _y_ 1 _, ..., yC_ and choose the answer word with the highest probability as the final answer for the current image-question pair. We treat Task B and C as the same classification problem as Task A, where the answers are fixed to 25 document element index numbers for Task B and 400 document element index numbers for Task C. The index numbers for document elements start from 0 and increase following the human-reading order (i.e. top to bottom, left to right) over a single document page (for Task B) and across multiple document pages (for Task C). OCR tokens are extracted from the document element with the corresponding predicted index number for the final retrieved answers for Task B and C questions. We use the Sigmoid function for Task C questions with multiple answers and select all the document elements whose probability has passed 0.5. 

## **7 Experiments** 

## **7.1 Performance Comparison** 

We compare the performances of baseline models and our proposed relational information-enhanced model over three tasks of our PDF-VQA dataset in Table 5. All the models process the questions in the same way as the sequence of question words encoded by pretrained BERT but differ in other features’ processing. The three large vision-and-language pretrained models (VLPMs): VisualBERT, ViLT and LXMERT, achieved better performances than other baselines with inputting only question and visual features. The better performance of VisualBERT than ViLT indicates that object-level visual features are more effective than image patch representations on the PDF-VQA images with segmented document elements. Among these three models, LXMERT, which used the same object-level visual features and the additional bounding box features, achieved the best results over Task A and B, indicating the effectiveness of bounding box information in the cases of PDF-VQA task. However, its performance on Task C is lower than VisualBERT. This might be because Task C inputs the sequence of objects (document elements) from multiple pages. The bounding box coordinates are independent on each page and therefore cause noise during training. Surprisingly, LayoutLM2, pretrained on document understanding datasets, achieved much lower accuracy than the three VLPMs. This might be because LayoutLM2 used token-level visual and bounding box features, which are ineffective for the whole document element identification. Compared to LayoutLM2 used the token-level contextual features, M4C, as a non-pretrained model, inputting object-level bounding box, visual and contextual features achieved higher performances. Such results further indicate that the object-level features are more effective for our PDF-VQA tasks. The object-level contextual features of each document element are represented as the [CLS] hidden states from the pretrained BERT model inputting the OCR token sequence extracted from each document element. 

Our proposed _LoSpa_ achieves the highest performance compared to all baselines, demonstrating the effectiveness of our adopted GCN-encoded relational features. Overall, all models’ performances are the highest on Task A among all 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

13 

**Table 5.** Performance Comparison over Task A, B, and C. Acronym of feature aspects: Q: Question features; B: Bounding box coordinates; V: Visual appearance features; C: Contextual features; R: Relational Information. 

|**Feature Aspects**<br>**Task A**<br>**Task B**<br>**Task C**|**Feature Aspects**<br>**Task A**<br>**Task B**<br>**Task C**|**Feature Aspects**<br>**Task A**<br>**Task B**<br>**Task C**|**Feature Aspects**<br>**Task A**<br>**Task B**<br>**Task C**|**Feature Aspects**<br>**Task A**<br>**Task B**<br>**Task C**|
|---|---|---|---|---|
|**Model**|**Q.**<br>**B.**<br>**V.**<br>**C.**<br>**R.**|**Val.**<br>**Test**|**Val.**<br>**Test**|**Val.**<br>**Test**|
|VisualBERT [17]<br>ViLT [16]<br>LXMERT [27]<br>BERT [5]<br>LayoutLM2 [30]<br>M4C [9]|✓<br>✗<br>✓<br>✗<br>✗<br>✓<br>✗<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✗|92.72<br>92.34<br>90.82<br>91.31<br>94.34<br>94.41<br>82.35<br>81.87<br>83.27<br>83.49<br>87.89<br>87.98|82.00<br>79.43<br>54.36<br>53.45<br>86.61<br>86.36<br>22.41<br>23.64<br>22.70<br>23.73<br>56.80<br>55.29|21.55<br>18.52<br>10.21<br>9.87<br>16.37<br>14.41<br>-<br>-<br>-<br>-<br>12.14<br>13.77|
|**Our LoSpa**|✓<br>✓<br>✓<br>✓<br>✓|**94.98**<br>**94.55**|**91.10**<br>**90.64**|**30.21**<br>**28.99**|



**Table 6.** Validating the effectiveness of proposed logical-relation (LR) and spatialrelation (SR) based graphs. 

|**Confgurations**|**Task A**<br>**Task B**<br>**Task C**|**Task A**<br>**Task B**<br>**Task C**|**Task A**<br>**Task B**<br>**Task C**|
|---|---|---|---|
||**Val.**<br>**Test**|**Val.**<br>**Test**|**Val.**<br>**Test**|
|**None**<br>**Logical Relation (LR)**<br>**Spatial Relation (SR)**|94.17<br>94.12<br>94.59<br>93.72<br>94.58<br>94.27|90.02<br>89.59<br>90.97<br>**90.67**<br>90.39<br>90.02|27.13<br>27.71<br>29.22<br>27.91<br>28.11<br>27.90|
|**LR&SR**|**94.98**<br>**94.55**|**91.10**<br>90.64|**30.21**<br>**28.99**|



tasks due to the relatively simple questions associated with object recognition and counting. The performances of all the models naturally dropped on Task B when the ability of contextual and structural understanding are simultaneously required. Performances on Task C are the lowest for all models. It indicates the difficulty of document-level questions and produces massive room for improvement for future research on this task. 

## **7.2 Relational Information Validation** 

To further demonstrate the influences of relational information on document VQA tasks, we perform the ablation studies on each task, as shown in Table 6. For all three tasks, adding both aspects of relational information can effectively improve the performance of our _LoSpa_ model. Firstly, Spatial relation (SR) enhanced models can make the models of all three tasks more robust. Regarding logical relation (LR), it can lead to more apparent improvements on Task B since Task B involves more questions that require understanding document structure more comprehensively. Moreover, since the graph representation of two relation features is trained on the training set, most of the test set performance is lower than the validation set during the QA prediction stage. 

## **7.3 Breakdown Results** 

We conduct the breakdown performance comparison over different question types of each task as shown in Table 7. Generally, all models’ performances on Exis- 

14 Y. Ding et al. 

**Table 7.** Task A, B and C performance on different question types. Same as the overall performance shown previously, the metric of Task A/B is F1 and Task C is Accuracy. 

|**Model**|**Task A**|**Task A**|**Task B**|**Task B**|**Task C**|**Task C**|
|---|---|---|---|---|---|---|
||**Existence**|**Counting**|**Struct-UD**|**Obj-Reg**|**Parent**|**Child**|
||**Val.**<br>**Test**|**Val.**<br>**Test**|**Val.**<br>**Test**|**Val.**<br>**Test**|**Val.**<br>**Test**|**Val**<br>**Test**|
|VisualBERT [17]<br>ViLT [16]<br>LXMERT [27]<br>BERT [5]<br>LayoutLM2 [30]<br>M4C [9]|94.11 91.62<br>92.34 93.40<br>96.02 94.59<br>86.25 86.04<br>87.22 85.78<br>90.78 89.15|92.52 92.45<br>90.62 91.01<br>94.10 94.38<br>81.80 81.31<br>82.70 83.19<br>87.51 87.87|83.24 80.86<br>53.41 51.97<br>86.65 86.86<br>30.42 30.55<br>33.18 31.80<br>60.74 60.29|71.49 70.30<br>59.54 61.66<br>86.46 83.15<br>21.37 22.33<br>21.55 22.63<br>21.29 20.39|21.55 19.91<br>11.04 10.21<br>26.66 23.57<br>-<br>-<br>-<br>-<br>13.63 14.34|19.64 18.52<br>8.75<br>8.79<br>8.56<br>9.51<br>-<br>-<br>-<br>-<br>12.21<br>9.89|
|**Our LoSpa**|**97.40 95.73**|**94.39 94.63**|**91.61 91.14**|**86.66 87.29**|**33.14 29.87**|**29.11 28.74**|



tence/Structural Understanding/Parent Relation Understanding questions are slightly better than Counting/Object Recognition/Child Relation Understanding questions in tasks A, B and C, respectively, due to their larger question numbers when training. Overall, all models’ performances are stable on different question types of each task and follow the same performance trend as on all questions in Table 5. However, M4C’s performance on Object Recognition is much lower than its performance on the Structural Understanding questions. This indicates that M4C is more powerful in recognising the contexts and identifying the semantic structures between document elements. However, it does not have enough capacity to identify the elements and related semantic elements simultaneously. Also, the LXMERT’s performances on Parent Relation Understanding questions are much better than those on Child Relation Understanding questions. This is because answers to parent questions are normally located on the same page as the queried elements. In contrast, answers to child questions are normally distributed over several pages, which is impacted by the independent bounding box coordinates of each page. The stable performances of M4C over the two question types of task C also indicate that using contextual features would eliminate such issues. Our _LoSpa_ , incorporating relational information between document elements, achieves stable performances over both question types in Task C. 

## **8 Conclusion** 

We proposed a new document-based VQA dataset to comprehensively examine the document understanding in conditions of natural language questions. In addition to contextual understanding and information retrieval, our dataset questions also specifically emphasize the importance of document structural layout understanding in terms of comprehensive document understanding. This is also the first dataset that introduces document-level questions to boost the document understanding to the full document level rather than being limited to one single page. We enriched our dataset by providing a Logical Relational graph and a Spatial Relational graph to annotate the different relationship types between document elements explicitly. We proved that such graph information integration enables outperforming all the baselines. We hope our PDF-VQA 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

15 

dataset will be a useful resource for the next generation of document-based VQA models with an entire multi-page document-level understanding and a deeper semantic understanding of vision and language. 

## **Ethical Consideration** 

This study was reviewed and approved by the ethics review committee of the authors’ institution and conducted in accordance with the principles of the Declaration. Written informed consent was obtained from each participant. 

## **References** 

1. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.: Vqa: Visual question answering. In: Proceedings of the IEEE international conference on computer vision. pp. 2425–2433 (2015) 

2. Biten, A.F., Tito, R., Mafla, A., Gomez, L., Rusinol, M., Valveny, E., Jawahar, C., Karatzas, D.: Scene text visual question answering. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4291–4301 (2019) 

3. Chaudhry, R., Shekhar, S., Gupta, U., Maneriker, P., Bansal, P., Joshi, A.: Leaf-qa: Locate, encode & attend for figure question answering. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 3512–3521 (2020) 

4. Davis, B., Morse, B., Price, B., Tensmeyer, C., Wiginton, C.: Visual fudge: Form understanding via dynamic graph editing. In: International Conference on Document Analysis and Recognition. pp. 416–431. Springer (2021) 

5. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 4171–4186 (2019) 

6. Ding, Y., Huang, Z., Wang, R., Zhang, Y., Chen, X., Ma, Y., Chung, H., Han, S.C.: V-doc: Visual questions answers with documents. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21492–21498 (2022) 

7. Ding, Y., Long, S., Huang, J., Ren, K., Luo, X., Chung, H., Han, S.C.: Form-nlu: Dataset for the form language understanding. arXiv preprint arXiv:2304.01577 (2023) 

8. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D.: Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 6904–6913 (2017) 

9. Hu, R., Singh, A., Darrell, T., Rohrbach, M.: Iterative answer prediction with pointer-augmented multimodal transformers for textvqa. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9992– 10002 (2020) 

10. Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., Jawahar, C.: Icdar2019 competition on scanned receipt ocr and information extraction. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1516–1520. IEEE (2019) 

16 Y. Ding et al. 

11. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6700–6709 (2019) 

12. Johnson, J., Hariharan, B., Van Der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., Girshick, R.: Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2901–2910 (2017) 

13. Kafle, K., Price, B., Cohen, S., Kanan, C.: Dvqa: Understanding data visualizations via question answering. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5648–5656 (2018) 

14. Kahou, S.E., Michalski, V., Atkinson, A., K´ad´ar, A.,[´] Trischler, A., Bengio, Y.: Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300 (2017) 

15. Kembhavi, A., Seo, M., Schwenk, D., Choi, J., Farhadi, A., Hajishirzi, H.: Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In: Proceedings of the IEEE Conference on Computer Vision and Pattern recognition. pp. 4999–5007 (2017) 

16. Kim, W., Son, B., Kim, I.: Vilt: Vision-and-language transformer without convolution or region supervision. In: International Conference on Machine Learning. pp. 5583–5594. PMLR (2021) 

17. Li, L.H., Yatskar, M., Yin, D., Hsieh, C.J., Chang, K.W.: Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557 (2019) 

18. Luo, S., Ding, Y., Long, S., Poon, J., Han, S.C.: Doc-gcn: Heterogeneous graph convolutional networks for document layout analysis. In: Proceedings of the 29th International Conference on Computational Linguistics. pp. 2906–2916 (2022) 

19. Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., Jawahar, C.: Infographicvqa. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1697–1706 (2022) 

20. Mathew, M., Karatzas, D., Jawahar, C.: Docvqa: A dataset for vqa on document images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 2200–2209 (2021) 

21. Methani, N., Ganguly, P., Khapra, M.M., Kumar, P.: Plotqa: Reasoning over scientific plots. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1527–1536 (2020) 

22. Mishra, A., Shekhar, S., Singh, A.K., Chakraborty, A.: Ocr-vqa: Visual question answering by reading text in images. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 947–952. IEEE (2019) 

23. Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., Lee, H.: Cord: a consolidated receipt dataset for post-ocr parsing. In: Workshop on Document Intelligence at NeurIPS 2019 (2019) 

24. Rausch, J., Martinez, O., Bissig, F., Zhang, C., Feuerriegel, S.: Docparser: Hierarchical document structure parsing from renderings. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 4328–4338 (2021) 

25. Shen, Z., Zhang, R., Dell, M., Lee, B.C.G., Carlson, J., Li, W.: Layoutparser: A unified toolkit for deep learning based document image analysis. In: Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part I 16. pp. 131–146. Springer (2021) 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 17 

26. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8317–8326 (2019) 

27. Tan, H., Bansal, M.: Lxmert: Learning cross-modality encoder representations from transformers. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). pp. 5100–5111 (2019) 

28. Tanaka, R., Nishida, K., Yoshida, S.: Visualmrc: Machine reading comprehension on document images. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 13878–13888 (2021) 

29. Wang, X., Liu, Y., Shen, C., Ng, C.C., Luo, C., Jin, L., Chan, C.S., Hengel, A.v.d., Wang, L.: On the general value of evidence, and bilingual scene-text visual question answering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10126–10135 (2020) 

30. Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C., Che, W., et al.: Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). pp. 2579–2591 (2021) 

31. Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of text and layout for document image understanding. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. pp. 1192–1200 (2020) 

32. Zhang, P., Li, C., Qiao, L., Cheng, Z., Pu, S., Niu, Y., Wu, F.: Vsr: a unified framework for document layout analysis combining vision, semantics and relations. In: International Conference on Document Analysis and Recognition. pp. 115–130. Springer (2021) 

33. Zhang, Z., Ma, J., Du, J., Wang, L., Zhang, J.: Multimodal pre-training based on graph attention network for document understanding. arXiv preprint arXiv:2203.13530 (2022) 

34. Zhong, X., Tang, J., Yepes, A.J.: Publaynet: largest dataset ever for document layout analysis. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1015–1022. IEEE (2019) 

## **A Question Templates with Examples** 

All question templates for three tasks are listed in Table 8, 9, 10 with the corresponding real question examples. Task A contains 36 question patterns, including 22 Existence type question patterns and 14 Counting type question patterns. For Task B, the Structural Understanding type contains 10 question patterns, and Object Recognition contains 5. Regarding Task C, there are 5 patterns provided for Child Relation Understanding and 10 patterns designed for Parent Relation Understanding, respectively. 

## **B Human Evaluation Details** 

We randomly selected 30, 30 and 40 question-answer pairs from Task A, Task B and Task C, respectively and put them with the related document page 

18 Y. Ding et al. 

|**Question Pattern**|**Question Example**|
|---|---|
|**Existence Type Question Patterns**||
|Is there any [E] on the [pos] of this page?|Is there any table on the top of this page?|
|Can you fnd any [E] on the [pos] of this page?|Can you fnd any fgure on the right of this page?|
|On the [pos] of this page, is there a [E]?|On the left of this page, is there a table?|
|Is it correct that there is no [E] at the [pos]?|Is it correct that there is no fgure at the bottom?|
|When you check the [pos] of this page, can you fnd any [E]?|When you check the right of this page, can you fnd any table?|
|Are there any [E1] are [R] the [E2]?|Are there any fgures upper the ’Competition analysis’?|
|Can you fnd any [E1] [R] the [E2]?|Can you fnd any table above the ’Balanced networks’?|
|Is there a [E1] found [R] the [E2]?|Is there a table found under the ’Competition analysis’?|
|Is it correct that there is no [E1] [R] the [E2]?|Is it correct that there is no table upper the ’Discussion’?|
|Confrm if there are any [E1] [R] the [E2]?|Confrm if there are any fgures upper the ’Result and Discussion’?|
|When you check the page, is there any [E1] [R] the [E2]?|When you check the page, is there any table below the ’Results’?|
|Is there any [E]?|Is there any table?|
|Are there any [E] on this page?|Are there any fgures in this page?|
|Is there a [E] in this page?|Is there a table on this page?|
|Can you fnd a [E] on this page?|Can you fnd a fgure on this page?|
|When you check this page, can you fnd any [E]?|When you check this page, can you fnd any table?|
|Is there a [E] on this page?|Is there a ’Results’ on this page?|
|Can you fnd a [E] on this page?|Can you fnd a ’Discussion’ on this page?|
|Does this page include a [E]?|Does this page include a ’Conclusion’?|
|Can [E] be found on this page?|Can ’Abstract’ be found on this page?|
|When you check this page, can you fnd [E]?|When you check this page, can you fnd ’Introduction’?|
|Confrm if there is [E] on this page.|Confrm if there is an ’Abstract’ on this page.|
|**Counting Type Question Patterns**||
|How many [E1] are [R] the [E2]?|How many tables are left for the ’Result and Discussion’?|
|What is the number of [E1] [R] the [E2]?|What is the number of tables below the ’Background & Summary’?|
|How many [E1] can you fnd on the [R] of [E2]?|How many fgures are upper the ’Discussion’?|
|Count the number of [E1] on the [R] of [E2].|Count the number of fgures below ’Material and methods’.|
|When you check this page, how many [E1] can you fnd on the [R] of|When you check this page, how many tables can you fnd on the top|
|[E2]?|of ’Background’?|
|Can you fnd [num] [E](s) on the page?|Can you fnd 2 table(s) in the page?|
|Does this page include [num] [E](s)|Does this page include 2 fgures?|
|Confrm if there are [num] [E](s) on this page.|Confrm if there are 1 table(s) in this page.|
|Are there [num] [E](s) on this page?|Are there 3 fgure(s) in this page?|
|Is there only [num] [E](s) on this page?|Is there only 2 table(s) in this page?|
|How many [E]s on this page?|How many tables in this page?|
|When you check this page, how many [E]s are on this page?|When you check this page, how many tables are on this page?|
|What is the number of [E]s on this page?|What is the number of fgures on this page?|
|How many [E]s can be found on this page?|How many fgures can be found on this page?|



**Table 8.** Task A question pattern templates with corresponding example questions. 

|**Question Pattern**|**Question Example**|
|---|---|
|**Structural Understanding**||
|What is the [turn] section in this page?|What is the last section in this page?|
|Can you describe the [turn] section of this page?|Can you describe the frst section of this page?|
|What does the [turn] section include in this page?|What does the last section include in this page?|
|What is the main contents of the [turn] section in this page?|What is the main contents of the frst section in this page?|
|When you check the [turn] section of this page, what information can|When you check the last section of this page, what information can|
|you get?|you get?|
|What is the [pos] section about?|What is the top section about?|
|What is the [pos] of the page about?|What is the left of the page about?|
|What is the topic of [pos] section?|What is the topic of bottom section?|
|Can you describe the main topic of the [pos] section?|Can you describe the main topic of the right section?|
|When you check the [pos] of this page, what information can you get?|When you check the bottom of this page, what information can you|
||get?|
|**Object Recognition**||
|What is the [E] on the [pos] of the page?|What is the table on the top of the page?|
|What is the [pos] [E] about?|What is the bottom table about?|
|Can you describe the [E] on the [pos] of the page?|Can you describe the fgure on the bottom of the page?|
|What information does the [pos] [E] contain?|What information does the left fgure contain?|
|When you check the [pos] [E], what information can you get?|When you check the top table, what information can you get?|



**Table 9.** Task B question pattern templates with corresponding example questions. 

images or file links in the google forms (An example of Task C can refer to Figure 5). For each task, raters need to check each generated question-answer pair together with the attached document page or file to determine whether 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 

19 

|**Question Pattern**|**Question Example**|
|---|---|
|**Child Relation **|**Understanding**|
|What does the [E] include?|What does the Introduction include?|
|What is the [E] about?|What is the Competing interests about?|
|What subsections are in the [E]?|What subsections are in the 2. Clinical Presentation?|
|What subsections can be found in the [E]?|What subsections can be found in the Materials and methods?|
|When you check the [E], which subsections are included?|When you check the Methods, which subsections are included?|
|**Parent Relation **|**Understanding**|
|Which section does describe the [E] ?|Which section does describe the Table 3?|
|Which section does include the description of the [E]?|Which section does include the description of the Table 2?|
|Name out the section that include the [E].|Name out the section that include the Table 2.|
|Where can you fnd the [E]?|Where can you fnd the Table 2?|
|When you search for the description of [E], which sections do you need|When you search for the description of Figure 1, which sections do|
|to check?|you need to check?|
|Which section does include the [E]?|Which section does include the ’Corwin HL et al,2009’?|
|Which section does cite the [E]?|Which section does cite the ’Wang C et al,2017’?|
|Where is the [E] cited in the document?|Where is the ’Horner KC et al,2005’ cited in the document?|
|Where can [E] be found in the document?|Where can ’Guan KL et al,1991’ be found in the document?|
|When you search for the citation of [E], which sections can you fnd|When you search for the citation of ’Zhang Z et al,2013’, which sections|
|it?|can you fnd it?|



**Table 10.** Task C question pattern templates with corresponding example questions. 

the question-answer pairs meet the requirements of three aspects, _Relevance_ , _Correctness_ , _Meaningfulness_ . For example, for a given question in Figure 5, ”Name out the section that describes Figure 1”, raters need to first go through the entire document to check whether the document has Figure 1 and then check which sections provide the description of that figure to compare with the provided answer. Finally, raters are required to determine whether this question will be asked in the real world. We show the detailed definition of each aspect to ensure raters can understand the evaluation metrics of each criterion at the beginning of the questionnaire of each task, as Figure 5 shows. 

## **C Additional Dataset Analysis** 

## **C.1 Distribution of Question Length** 

We show the distribution of question length of each task in Figure 6. The average question length for Task A, B and C are 25, 10 and 15, respectively. 

## **D Baseline Details** 

## **D.1 Baseline Descriptions** 

**– M4C** [9] applies the multimodal transformer, which takes into the question embedding, OCR token embedding and image object features as inputs, and iteratively decodes the answers over the combined answer space of OCR tokens and the fixed answer list. 

**– VisualBERT** [17] is a pretrained vision-and-language model that passes the sequence of text and object region embeddings to a transformer to get the integrated vision-and-language representations. 

20 Y. Ding et al. 

**Fig. 5.** A human evaluation sample with evaluation criteria of Task C. Task A and B have a similar style as Task C. 

- **LXMERT** [27] applies three transformer encoders to encode the text embeddings, object region embeddings and the cross-modality learning between texts and image features. 

- **ViLT** [16] operates linear projection over image patches to get a sequence of image patch representations and input to the transformer encoder together with the text embeddings to get a pretrained vision-and-language model. 

- **BERT** [5] is a pretrained language model that applies the structure of a multi-layer bidirectional transformer encoder. We used only the textual features from document pages as the inputs. 

- **LayoutLMv2** [30] is a pre-trained model to operate on the position and textual features of document elements and generate the integrated representations that can be used for downstream document-related tasks. 

## **D.2 Baseline Setup** 

- **M4C** applied multiple transformer layers, learning question embeddings, image object features and OCR token features in the common embedding space and iteratively decoding answer tokens from a fixed answer space or the OCR tokens in the image. The OCR tokens are encoded in rich representations, including the textual embedding of each token, appearance features of the token region on the image, Pyramidal Histogram of Characters (PHOC) features and the location features. We evaluated all three tasks with M4C but slightly modified the inputs and output layer to suit documentbased VQA. Firstly, since the number of OCR tokens is much larger in PDF 

PDF-VQA: A New Dataset for Real-World VQA on PDF Documents 21 

**Fig. 6.** Question length distribution of Task A, B and C 

documents than that in real-life scenes, instead of inputting the features of all OCR tokens in the page, we used the BERT [CLS] token features to represent the sequence of textural contents in each document element region and took them together with the question embedding and the visual features of each document element region as the input sequence to the multi-layer transformer. Secondly, in the decoding part, Task B and C, we used the _d_ -dimensional representations for the index numbers of the corresponding document element region in the page and generated the scores through the dynamic pointer network to predict the index number of document element region over the list of document element region index numbers. For applying M4C to Task A, we set fixed answer space as the decode inputs and put the pointer network on top to get a final prediction. 

- **BERT** , **LayoutLM2** are used only for Task A and B because the inputs of both models are question and context token level information with the 512 maximum limitations. For multi-page documents, the number of tokens is normally much higher than 512 tokens, which means those two models can only catch the first-page context information. In this case, we did not select those two models for conducting Task C tests. For both Task A and B, we directly extract 768-dimension [CLS] token embedding and feed it into classifiers for predicting the corresponding answer or object sequential index. 

- **VisualBERT** , **LXMERT** can process visual features of document layout elements extracted from pretrained ResNet101-Res5. After we feed those raw object-level visual features and question tokens into those vision-language pretrained models, we extract the enhanced visual representation of document layout elements and feed them into a pointer network to get final scores for predicting corresponding answers for all three tasks. 

- **ViLT** is directly applied for conducting Task A and B by using the provided feature extractor and pre-trained 

- **ViltForQuestionAnswering** model to predict the corresponding answer based on input questions and image patch features. For addressing task C, we concatenate all document pages into an image pixel matrix and feed into the feature extractor to extract image patch features for feeding forward pass. The outputs pass through a Sigmoid layer instead of the Softmax function 

22 Y. Ding et al. 

adopted by other tasks for backward propagation in the training stage and answer prediction in the inference stage. 

## **E Implementation Detail** 

Dimension for the visual features of each document element region _df_ is 2048. The activation function used in GCN is Tanh. The GCN is trained with AdamW optimizer and 0.0001 learning rate for 10 epochs. Each question token is encoded into a 768-dimension fine-tuned on the BERT-base model. Our model utilized a 6 layers transformer encoder and a 4 layers transformer decoder with 12 heads and 768-dimension hidden size. The maximum numbers for input question tokens and objects (document layout elements) are 50 and 25, respectively, for Task A and B and 50 and 400 for Task C. For a fair comparison, epoch times are selected as 5, 10, and 20 for all Task A, B and C models, respectively. All the experiments are conducted on 51 GB Tesla V100-SXM2 with CUDA 11.2. 

