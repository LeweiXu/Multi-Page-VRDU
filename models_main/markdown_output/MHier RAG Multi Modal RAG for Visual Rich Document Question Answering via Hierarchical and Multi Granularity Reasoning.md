# **MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-Granularity Reasoning** 

## **Ziyu Gong**[1] **, Chengcheng Mai**[2] **, Yihua Huang**[1] 

1State Key Laboratory for Novel Software Technology, School of Computer Science, Nanjing University 2School of Computer Science and Electronic Informatics, Nanjing Normal University ziyugong@smail.nju.edu.cn, yhuang@nju.edu.cn, maicc@njnu.edu.cn 

## **Abstract** 

The multi-modal long-context document question-answering task aims to locate and integrate multi-modal evidences (such as texts, tables, charts, images, and layouts) distributed across multiple pages, for question understanding and answer generation. The existing methods can be categorized into Large Vision-Language Model (LVLM)-based and RetrievalAugmented Generation (RAG)-based methods. However, the former were susceptible to hallucinations, while the latter struggled for inter-modal disconnection and cross-page fragmentation. To address these challenges, a novel multi-modal RAG model, named MHier-RAG, was proposed, leveraging both textual and visual information across long-range pages to facilitate accurate question answering for visual-rich documents. A hierarchical indexing method with the integration of flattened in-page chunks and topological cross-page chunks was designed to jointly establish in-page multi-modal associations and long-distance cross-page dependencies. By means of joint similarity evaluation and large language model (LLM)-based re-ranking, a multi-granularity semantic retrieval method, including the page-level parent page retrieval and document-level summary retrieval, was proposed to foster multi-modal evidence connection and long-distance evidence integration and reasoning. Experimental results performed on public datasets, MMLongBench-Doc and LongDocURL, demonstrated the superiority of our MHier-RAG method in understanding and answering modality-rich and multi-page documents. 

**==> picture [240 x 271] intentionally omitted <==**

**----- Start of picture text -----**<br>
61%2008. say they expect ther Original document, Page 4  fami’ nancial situationto improve, up from 56% who sid this in Questions: see economic upward mobility for their children?According to the report, how do 5% of the Latinos<br>[tino adutsaso see upward mobility in thei childrens | <<<a Relevant Evidences ：<br>Betteroff iaancially than thip themschesare now.” SSRNoe tne ON 1)  Pure-text  ( Plain-text ) [from  Page 4 ]<br>TheseLatinos,surveycellular [findings] of telephones. 1500nationemerge a llydults represetativetwasconduftedfrom fidefre 2015 [from]  on bilingual bothNational lndine telephone Surveyand of Lesswell off _fanssarenou Latino adults also see upward mobility in their children’s futures. 2) Latinos see economic upward mobility for their children.  Generalized-text  ( Layout ) [from  Page 4 ]<br>2015,fioprereleaseptinicin and hasa margin mini ofsomerbe eseofatinpas oFOct. Sol minus 25ein toNov. 3.3 30, pomatne5 3) 5%  Chart  [from  Page 4 ] —) Lesswell off =1<br>group, numbering Multi-modality  Peart one [L<br>ernbyS%between Connection 2000 and 2014. With this fast poe<br>demdpaphingganch hsecame agundagimnpast nthe come<br>nation's economy. Between2009 and 2033, Latinos “oc pn aan 16% Rotter off<br>Latinos driving most ofthat job growth. Andthe group's “avis wewneyaven<br>purchasingUniversityGrowth,community the powerof purchasingGeorgia'sis ontheSelig power rise,Center oftheAccording for US. EconomicLatinoto the row‘ram,nsenoncomenaaSn bate conan Tne ea 72%<br>2000 and higherwas $1. thantrillion that of nblacks 2014 (1. againtilion) of 5%and singeAsians—§ ($770 billion).<br>(a) Multi-modal connection<br>Original document, Page 54 Original document, Page 55 Questions: I'm a Macbook Air user in<br>Onscreen Help 50dtelephonecamps ERO on te te pts ad Mexico.which numberAccordingshall toI callthis forguidebook,seeking<br>YoucanVeilehctngten find anes tour questions sels nd = fes may app 1-800-275-2273 = AppleCare service and support?<br>é ROOTED STE fi ts | wean<br>AppleCarecoverageSystemAppleniversion_wnwapplecomi/suppotproducts‘warfourpacceptUsted.MacBook waranycoveage(>i Profilerhowbyand much ServiceAboutpurcsucAisoft has coreshis Macingyou and Supportremorsean Pfthetnplewilt romAppleCare Apple uthoted90lacnok theG3ysor Long-distance   melsStealvstO a Reasoning n  heubardiceAuemoetechnicalNN websiteProtection and aan ToSenisupport P lan.oder openthen cick adressFaroan Sst ou information,and Moreh, for Appewuboioed— cantshovsyounhatoneyour eterrofl,year countryvitof chooseyurhay  |||9 |[|||||—_ EI Ney =Tens‘ted_apDT,CURES zea goAa) COMpTETE Ts alaeject15)snonoaess onthe ODT change 66ra  web:andl and atorwowpawagonetcapperwappeconiunpant applcamiehoppetaewepone tes ny Relevant Evidences1) AppleCare Service and SupportCall the support center nearest you 2) Telephone supportTelephone numbers 3)  TablePure-textPure-text [from (Plain-text)(Plain-text) Page ：  55 ] [from[from PagePage 54 55 ]]<br>Foinstalinged andasisanc opening aplication,Ape epee’and bas troubleshootingcn lp yo with Country Phone ...<br>BT TERN (he rst 90 days are complimertay). HaveCllthe the purchase Supportdatecent and United States 1-800-275-2273 1-800-275-2273 ...<br>Lou cok kisi mbes ext when ou ca<br>(b) Long-distance reasoning<br>Figure 1: Two challenges for multi-modal long-context doc-<br>ument question-answering.<br>**----- End of picture text -----**<br>


## **Introduction** 

Document question-answering (Doc-QA) aims to answer questions based on document content. With the rise of Large Vision-Language Models (LVLMs), Doc-QA research (Ding et al. 2022; Mishra et al. 2019; Ma et al. 2025; Tanaka et al. 2023; Luo et al. 2024) has transitioned from simple text-based approaches to more sophisticated multimodal methods. In multi-modal long-context Doc-QA (Suri et al. 2025; Cho et al. 2024; Han et al. 2025), commonly seen in scientific papers, business reports, and instructional manuals, etc., relevant evidences needed to answer a question are often scattered across multiple pages and modalities, including texts, tables, charts, layouts and images, which poses challenges for complex multi-modal comprehension and long-distance reasoning over scattered content. 

Existing methods for multi-modal long-context Doc-QA can be categorized into two types: LVLM-based methods 

(Lu et al. 2024; Dong et al. 2024; Hu et al. 2024) and RAGbased methods (Xing et al. 2025; Xia et al. 2024, 2025). LVLM-based methods directly processed the entire document by using large multi-modal models, jointly encoding both textual and visual features. However, these models tended to suffer context length limitations, insufficient longrange reasoning capabilities, and fact hallucinations. RAGbased methods focused on retrieving relevant evidences from the document before answer generation, which improved scalability but still failed to capture complex multimodal dependencies and overlook cross-page evidence. 

Based on the above research status of existing methods for the multi-modal Doc-QA, we listed two key challenges and outlined the corresponding solutions, as follows: 

**Challenge 1: Absence of multi-modality connections.** Multi-modal long-context document question-answering requires the integration of multi-modal information to synthe- 

size accurate answers, rather than relying solely on either textual or visual cues. As shown in Figure 1a, for the question “how do 5% of the Latinos see economic upward mobility for their children?”, several textual evidences are relatively easy to be retrieved, such as plain text and layoutbased image caption, since the question shares common keywords and semantic meaning with these textual content. However, the actual answer “Less well off” resides within the visual image, which struggles to yield a high retrieval score due to few direct textual overlaps and semantic cues belonging to the pie chart. Therefore, it is essential to establish connections between the corresponding visual elements and surrounding textual information. 

**Solution 1: Build multi-modal connections by combining in-page indexing structure with page-level parent page retrieval method.** To address the challenge of multi-modality disconnection, a flattened in-page indexing strategy combined with the page-level parent-page retrieval was presented. Since semantically relevant textual and visual elements tend to co-occur in the same page, the parent page contains multi-modal retrieved contents related to the answer, which helps to bridge different modalities. This method transformed textual evidence into entry points for accessing associated visual information, enabling the model to aggregate multi-model evidence more effectively. 

**Challenge 2: Lack of cross-page evidence linking and reasoning ability.** Another difficulty in multi-modal longcontext Doc-QA is the isolation of evidence dispersed across different pages, which requires models to reason and aggregate long-distance cross-page evidences. As shown in Figure 1b, for the question “I’m a Macbook Air user in Mexico. According to this guidebook, which number shall I call for seeking AppleCare service and support?”, the relevant evidences span multiple modalities, including plain text and tables, and are distributed across different pages. Specifically, the actual phone number “1-800-275-2273” appears in a table on page 55, while the explanatory instruction, indicating that users should “call the support center number nearest you” is located on page 54. This demands the model to associate cross-page evidences and perform multi-step reasoning across multiple document sections, which is also not available in most existing models. 

**Solution 2: Achieve long-distance reasoning by combining cross-page indexing structure with documentlevel summary retrieval method.** To solve this problem, a topological cross-page indexing strategy combined with document-level summary-based retrieval was proposed. Semantically related content from different pages are grouped together through clustering and summarized by large language models, thereby promoting the retrieval scope to span across multiple pages. This method helped to aggregate evidence scattered across different pages, promoting longdistance reasoning. 

Considering the above challenges and solution ideas, a novel retrieval-augmented generation method, named MHier-RAG, was proposed for the multi-modal longcontext document question-answering task. The major contributions can be summarized as follows: 

• In proposed MHier-RAG, a hierarchical index structure 

with flattened in-page and topological cross-page chunks was proposed to establish correlations between diverse modalities and multiple pages. 

- A multi-granularity retrieval method with page-level parent page retrieval and document-level summary retrieval was also proposed to facilitate multi-modal evidence connection and long-distance evidence integration and reasoning. 

- Extensive experiments conducted on two public datasets, MMLongBench-Doc and LongDocURL, verified the superiority and effectiveness of our method for the multimodal long-context Doc-QA task. 

## **Our MHier-RAG Methodology** 

## **Task Description** 

The objective of multi-modal long-context Doc-QA is to answer questions based on document content that contains both textual and visual content (e.g., texts, tables, charts, layouts and images), where relevant evidences may be dispersed across multiple pages and modalities. 

Figure 2 presents an overview of our MHier-RAG method. Given a query _Q_ and its corresponding document _D_ = _{d_ 1 _, d_ 2 _, ..., dN }_ , where _N_ is denoted as the page number and _di_ represents the _i_ -th page.The multi-modal longcontext Doc-QA can be formalized as follows: 

**==> picture [226 x 30] intentionally omitted <==**

where _Yans_ is the answer response generated by _LLM θ_ , _Rq[context]_ represents multi-modal retrieved evidences, _HierIndex_ ( _·_ ) is the hierarchical index structure for constructing retrievable corpus of _D_ , _MMRetriever_ ( _·_ ) is the proposed multi-granularity retriever that searches for evidences related to _Q_ from document corpus, and _PCoT_ is a Chain-of-Thought (CoT) prompting strategy, which is specifically designed to guide the model in generating step-by-step reasoning paths during answer generation. 

## **Hierarchical Index Construction Method with Multi-Modal Semantic Encoding** 

For each page _di_ = _{Ti, Vi}_ in document _D_ , the textual content, _Ti_ , consisted of three key components: (1) pure-text that preserved in original form; (2) tables that serialized into arranged sequences to retain its row-column-cell structure; (3) layout-based text. The composition of _Ti_ was denoted as _Ti_ = _Ti[T ext] ∪ Ti[T able] ∪ Ti[Layout]_ . 

The visual context, _Vi_ , was transformed into descriptive textual information via LVLMs, denoted as _Ti[Des]_ , converting visual semantics into text-based semantic encoding and retrieval space. Meanwhile, the raw images and charts have also been retained as the original and complete visual features, denoted as _Vi[raw]_ , to alleviate information loss. _Vi_ can be represented as _Vi_ = _Ti[Des] ∪ Vi[Raw]_ . 

**==> picture [463 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
Global Visual Corpus  𝑽 [𝑪𝒐𝒓𝒑𝒖𝒔]<br>QUESTION InformationVisual … RetrievalImage LVLM Generation<br>𝑽 [𝑹𝒂𝒘] Image, Chart …  Modality   Connection Visual Context 𝑽𝒇𝒊𝒏𝒂𝒍𝒒<br>Image Information Textual Corpus with In-Page Index   𝑰𝒊𝒏<br>𝑽 𝑻 [𝑫𝒆𝒔] Parent Page<br>LVLM 𝑪𝟏 𝑪𝟐 𝑪𝟑 𝑪𝟒 … 𝑪𝑵 Retrieval 𝑷𝟏 𝑷𝟑 … 𝑷𝑵<br>Text Information<br>Textual Corpus with Cross-Page Index   𝑰𝒄𝒓𝒐𝒔𝒔<br>𝑻 [𝑻𝒆𝒙𝒕] QUESTION<br>Table Information InformationTextual Cluster summary 𝒄𝒍𝒔𝟏 1 Cluster summary 𝒄𝒍𝒔𝑲 LLM-Based K Summary LLM-Based Re-Ranking<br>𝑻 [𝑻𝒂𝒃𝒍𝒆]<br>𝑪𝟏 𝑪𝟑 𝑪𝟒 … 𝑪𝟐 𝑪𝑵 𝑷𝒇𝒊𝒏𝒂𝒍𝒒<br>Layout Information Cross-Page 𝑷𝟑 𝑷𝑵 … 𝑷𝟏<br>Clustering<br>𝑻 [𝑳𝒂𝒚𝒐𝒖𝒕] 𝑪𝟏 𝑪𝟐 𝑪𝟑 𝑪𝟒 … 𝑪𝑵 SummaryRetrieval 𝑺𝒖𝒎𝒎𝒂𝒓𝒚𝒇𝒊𝒏𝒂𝒍𝒒<br>— eee CO Summary Context<br>Long-distance reasoning<br>Hierarchical Index Construction → Multi-Granularity Evidence Retrieval  Answer Generation<br>Response Answer<br>LLM  +  Prompts  (CoT+SO)<br>**----- End of picture text -----**<br>


Figure 2: Overview of MHier-RAG with hierarchical index and multi-granularity retrieval for multi-modal Doc-QA. ( _Pn_ is the parent page of _cn_ , _clsK_ is the clustered block, ‘CoT’ denotes chain-of-thought and ‘SO’ denotes a structured output format.) 

Therefore, textual and visual content of documents can be separately defined as follows: 

**==> picture [213 x 12] intentionally omitted <==**

**==> picture [214 x 11] intentionally omitted <==**

Based on the multi-modal content obtained above, two complementary hierarchical index structures were constructed at different levels, with the following formula: 

**==> picture [207 x 12] intentionally omitted <==**

where _**Iin**_ represents the flattened in-page index for establishing the association of different modality information within one page, and _**Icross**_ denotes the topological crosspage index for establishing the interaction of long-distance cross-page information. 

**A. The Flattened In-Page Index.** The flattened in-page indexing converted textual information (i.e., _Ti[corpus]_ for each document page _di_ ) into a page-level list of smaller and uniformly-sized textual chunks, enabling direct access to pieces of evidence and facilitating finer-grained evidence extraction for retrieval. These textual chunks in _i_ -th page of document _D_ can be denoted as _ci_ = _{ci,_ 1 _, ci,_ 2 _, ..., ci,Ki}_ , where _Ki_ is the chunk number of _i_ -th page. Language models (LMs) were utilized for encoding the text attributes, thereby learning representations that capture their semantic meaning, denoted as _Zi[c]_[=] _[LM]_[(] _[c][i]_[)][. Thus, the flattened in-] page index of document _D_ was defined as: 

**==> picture [153 x 30] intentionally omitted <==**

where _N_ is the page number of document _D_ and _I_ ( _·_ ) represents the index of the encoded text chunks. 

**B. The Topological Cross-Page Index.** The topological cross-page indexing was conducted at the document-level scope, partitioning the entire textual content (i.e., _T[corpus]_ ) into clustered blocks with similar semantic meanings for encoding and summarizing the similar semantics. These textual blocks (denoted as _B_ = _{b_ 1 _, b_ 2 _, ..., bK}_ , where _K_ is 

the number of textual blocks in document _D_ ) were also encoded by language models, represented as _Zi[b]_[=] _[ LM]_[(] _[b][i]_[)][.] 

Through iterative processing, all textual blocks were organized into a topological tree, where leaf nodes retained original attributes, intermediate nodes aggregated semantically related cross-page blocks with the gaussian mixture model for clustering the text blocks, and the root node summarized the topic-level semantics of the document with the large language model, thereby creating a multi-scale representation that captures both local multi-modal details and global cross-page document structure. 

For instance, at each layer _l_ , leaf embeddings _Zi[b,l][−]_[1] were clustered, and summaries were generated as follows: 

**==> picture [194 x 16] intentionally omitted <==**

**==> picture [192 x 30] intentionally omitted <==**

where _Kl_ is the number of clusters, and then summaries were re-embed to form higher-layer nodes, denoted as _Zi[b,l]_ = _LM_ ( _SK_[(] _[l]_[)] _l_[)][.] 

Therefore, the topological cross-page index of document _D_ was formulated as: 

**==> picture [196 x 30] intentionally omitted <==**

where _I_ ( _·_ ) represents the index of the encoded textual nodes. 

## **Multi-granularity Retrieval Method** 

To alleviate the disconnection problem between multi-modal information and the difficulty of long-distance reasoning, a multi granularity content retriever with page-level parentpage retrieval by in-page index and document-level summary retrieval by cross-page index was presented, by searching for evidence related to the question in the corpus with hierarchical indexing. Our multi-granularity content retriever 

can be defined as follows: 

_MMRetriever_ ( _HierIndex_ ( _D_ ) _, Q_ ) = (10) _MMRetriever_ ( _Iin, Q_ ) _∪ MMRetriever_ ( _Icross, Q_ ) 

**A. Page-level Parent Page Retrieval Method with LLM-based Re-ranking for Modality Connection.** Since semantically related textual and visual evidence tend to be distributed on the same page, the parent-page retrieval method was proposed to augment the integration of multimodal information. Firstly, we retrieved the textual content chunks most relevant to the question, and then associated chunks with their corresponding parent page. These parent pages contained more semantically similar visual and structural information, such as tables, layouts, charts, and images. 

Specifically, given the query _Q_ , we calculated the similarity between the embedded query and textual chunks with the flattened index _Iin_ , and then selected the Top _K_ most relevant chunks based on semantic similarity as follows: 

_Cq_ = _{c[c] i,j[}]_[ =] _[ argTopK] Zi,j[c][∈][I][in][Sim]_[(] _[LM]_[(] _[Q]_[)] _[, Z] i,j[c]_[)] (11) where _c[c] i,j_[is][the] _[j]_[-th][chunk][on][page] _[i]_[of][document] _D_ and can be navigated to its source page _Pi_ . The set of retrieved parent pages was obtained as _Pq_ = _{ParentPage_ ( _c[c] i,j[|][c][c] i,j[∈][C][q]_[)] _[}]_[.] 

To further improve parent page retrieval, a fine-grained LLM-based re-ranking method was proposed to select pages that were more relevant to the question, where the large language model (LLM) was guided to rate the relevance between those retrieved pages _Pq_ and the problem _Q_ on a scale of 0 to 1. The final reordered pages based on the scores allocated by the LLM can be defined as: 

_Pq[final]_ = _argTopKPi∈Pq LLM_ ~~_S_~~ _core_ ( _Q, Pi_ ) (12) Meanwhile, we searched for images belonging to the parent page set _Pq[final]_ in visual corpus _V[Corpus]_ (denoted as _Vq_ ), and then used the Large Vision-Language Model (LVLM) to provide highly-relevant evidence related to the problem for the images (denoted as _Vq[final]_ ), which can be formulated as follows: 

**==> picture [192 x 14] intentionally omitted <==**

**==> picture [192 x 14] intentionally omitted <==**

Ultimately, through parent page retrieval and LLM-based re-ranking, the page-level multi-modal content retrieved by in-page indexing was defined as: 

_MMRetriever_ ( _Iin, Q_ ) = _Pq[final]_ + _Vq[final]_ (15) 

**B. Document-level Summary Retrieval Method for Long-Distance Reasoning.** To associate and reason the long-distance evidence fragments across multiple pages, we utilized the topological indexing to achieve document-level retrieval of summaries across multiple pages, serving as a supplement to the page-level retrieval method. 

Based on the given question _Q_ , we calculated the semantic similarity between all nodes in the topological structure and the question, and then selected the Top _K_ relevant ones. Therefore, the document-level multi-modal content retrieved by cross-page indexing was denoted as follows: 

**==> picture [222 x 33] intentionally omitted <==**

## **Answer Generation** 

For each question _Q_ , the retrieved evidences from corpus, such as _Pq[final]_ , _Vq[final]_ and _Summaryq[final]_ , were integrated into the context window of the large language model (LLM), and the final answer was generated as follows: 

**==> picture [224 x 30] intentionally omitted <==**

To enhance response quality, a chain-of-thought (CoT) prompting method with a structured output (SO) format was also proposed for answer reasoning, which enabled direct extraction of final answers without the need for lengthy parsing. Details of prompts can be found in Appendix. 

## **Experiments** 

## **Dataset** 

We evaluated models on two public datasets for multi-modal Doc-QA. **MMLongBench-Doc** (Ma et al. 2024b) comprises 135 long PDF documents, each containing an average of 47.5 pages and 21,214 tokens. It consists of 1,082 expertannotated questions. **LongDocURL** (Deng et al. 2025) is constructed upon 396 long PDF documents, with an average length of 85.6 pages and 43,622.6 tokens. It collects 2,325 high-quality question-answer pairs. Their answers rely on evidences from multi-modalities and multi-pages. 

## **Implementation Details** 

Docling (Livathinos et al. 2025) was used for pdf parsing. The off-the-shelf LLMs/LVLMs were utilized for answer generation. All experiments were conducted on a single NVIDIA A100 GPU. More implementation details are in Appendix. 

## **Metrics** 

For MMLongBench-Doc and LongDocURL, we followed their official evaluation setups. We reported the accuracy of distinct evidence modality types and evidence pages. The generalized accuracy and F1 score were also recorded. 

## **Main Results** 

We compared our MHier-RAG with existing SOTA LVLM/LLM-based and RAG-based methods on the MMLongBench-Doc and LongDocURL datasets. 

**MMLongBench-Doc.** Table 1 listed the performance of models on the MMLongBench-Doc dataset. We observed that: (1) Both LVLM-based methods and LLMbased methods with OCR-parsed documents exhibited poor performance and struggled with multi-modal comprehension and long-distance reasoning for long-context document. Our MHier-RAG surpassed the best-performing LVLM, i.e., GPT-4V, by 19.9% and 14.8% in generalized accuracy and F1 score, respectively. (2) Compared with the RAG-based SOTA M3DocRAG, when the page number was four, MHier-RAG achieved superiority on all metrics, with improvements of 27.2% and 18.8% on generalized accuracy and F1 score. When the page number was set to ten, MHier-RAG achieved better performance with an accuracy 

|**Model**|**Evidence Source**<br>TXT<br>LAY<br>CHA<br>TAB<br>FIG|**Evidence Page**<br>**Acc.**<br>**F1**<br>SIN<br>MUL<br>UNA|
|---|---|---|
|**_OCR(Tesseract(Smith 2007))_**+**_Large Language Models(LLMs)_**|||
|QWen-Plus (Qwen 2024)<br>17.4<br>15.6<br>7.4<br>7.9<br>8.8<br>DeepSeek-V2 (Liu et al. 2024)<br>27.8<br>19.6<br>8.8<br>17.0<br>9.4<br>Claude-3 Opus (Anthropic 2024)<br>30.8<br>30.1<br>16.4<br>24.4<br>16.3<br>Gemini-1.5-Pro (Gemini et al. 2024)<br>29.3<br>15.9<br>12.5<br>17.7<br>11.5<br>GPT-4o (OpenAI 2024)<br>41.1<br>23.4<br>28.5<br>38.1<br>22.4||14.2<br>10.6<br>42.2<br>18.9<br>13.4<br>20.2<br>15.4<br>48.1<br>24.9<br>19.6<br>32.0<br>18.6<br>30.9<br>26.9<br>24.5<br>21.2<br>16.4<br>73.4<br>31.2<br>24.8<br>35.4<br>29.3<br>18.6<br>30.1<br>30.5|
|**_Large Visual Language Models(LVLMs)_**|||
|DeepSeek-VL (Lu et al. 2024)<br>7.2<br>6.5<br>1.6<br>5.2<br>7.6<br>InternLM-XC2-4KHD (Dong et al. 2024)<br>9.9<br>14.3<br>7.7<br>6.3<br>13.0<br>mPLUG-DocOwl 1.5 (Hu et al. 2024)<br>8.2<br>8.4<br>2.0<br>3.4<br>9.9<br>Qwen-VL (Bai et al. 2023)<br>5.5<br>9.0<br>5.4<br>2.2<br>6.9<br>GPT-4V (Achiam et al. 2023)<br>34.4<br>28.3<br>28.2<br>32.4<br>26.8||5.2<br>7.0<br>12.8<br>7.4<br>5.4<br>12.6<br>7.6<br>9.6<br>10.3<br>9.8<br>7.4<br>6.4<br>6.2<br>6.9<br>6.3<br>5.2<br>7.1<br>6.2<br>6.1<br>5.4<br>36.4<br>27.0<br>31.2<br>32.4<br>31.2|
|**_RAG methods_**|||
|ColBERTv2 (Santhanam et al. 2022) + LLaMA-3.1-8B<br>23.7<br>17.7<br>14.9<br>24.0<br>11.9<br>M3DocRAG (Cho et al. 2024) (page=4)<br>30.0<br>23.5<br>18.9<br>20.1<br>20.8<br>**MHier-RAG (OURS)**(page=4)<br>41.6<br>30.2<br>40.9<br>48.9<br>25.1<br>**MHier-RAG (OURS)**(page=10)<br>**45.9**<br>**34.4**<br>**44.9**<br>**51.1**<br>**37.5**||25.7<br>12.2<br>38.1<br>23.5<br>19.7<br>32.4<br>14.8<br>5.8<br>21.0<br>22.6<br>48.5<br>31.7<br>74.9<br>48.2<br>41.4<br>**53.5**<br>**36.8**<br>**76.2**<br>**52.3**<br>**46.0**|



Table 1: Experimental results on the MMLongBench-Doc dataset. The highest performance was bolded and the second best performance (except for ours) was underlined. ‘page=n’ represents the number of pages retrieved from the parent page. ‘SIN’, ‘MUL’ and ‘UNA’ separately denote singe-page, cross-page and unanswerable questions. 

|**Model**|**Acc.**|
|---|---|
|**_OCR(Tesseract)_**+**_Large Language Models(LLMs)_**||
|LLaVA-OneVision (Li et al. 2025)|23.3|
|Qwen-VL (Bai et al. 2023)|25.0|
|Gemini-1.5-Pro (Gemini et al. 2024)|32.0|
|GPT-4o (OpenAI 2024)|34.7|
|O1-preview (OpenAI 2024)|35.8|
|**_Large Visual Language Models(LVLMs)_**||
|InternLM-XC2.5 (Dong et al. 2024)|2.4|
|mPLUG-DocOwl2 (Hu et al. 2024)|5.3|
|Pixtral (Agrawal et al. 2024)|5.6|
|Llama-3.2 (Meta 2024)|9.2|
|LLaVA-OneVision (Li et al. 2025)|22.0|
|Qwen-VL (Bai et al. 2023)|30.6|
|**_RAG methods_**||
|ColBERTv2+LLaMA-3.1-8B|49.1|
|M3DocRAG (Jain et al. 2025)(page=10)|52.2|
|**MHier-RAG(OURS)**(page=4)|52.4|
|**MHier-RAG(OURS)**(page=10)|**57.2**|



Table 2: Experimental results on the LongDocURL dataset. The highest performance was bolded and the second best performance (except for ours) was underlined. 

of 52.3% and a F1 score of 46.0%. To be specific, the accuracy of visual charts (CHA) and figures (FIG) separately increased to 44.9% and 37.5%. The accuracy of multipage (MUL) and unanswerable-questions (UNA) attained 36.8% and 76.2%, respectively. (3) These experimental results proved the advantages of our model in multiple modality understanding and long-distance reasoning, and the capability of alleviating hallucinations caused by LLMs. 

**LongDocURL.** Table 2 showed the performance of models on the LongDocURL dataset. Similar phenomena were found: (1) Our MHier-RAG achieved better performance, compared to the top-performing LVLM (i.e., Qwen2-VL) and LLM (i.e., o1-preview). (2) MHier-RAG method surpassed the current SOTA M3DocRAG by a margin of 5% in terms of generalized accuracy, when the page number was set to ten. (3) These results further validated the effectiveness of our approach for the connection of multi-modality and the link of cross-page evidence. 

## **Ablation Experiments** 

**MMLongBench-Doc.** Table 3 presented ablation results on the MMLongBench-Doc dataset. After removing all visual information, the performance of the variant model MHier-RAG _v_ 1 decreased to 46.9%, which highlighted the importance of simultaneously integrating textual and visual information for the RAG-based method in Doc-QA. MHier-RAG _v_ 2 restricted the number of parent page from ten to one, resulting in a performance decline to 46.6%. The drop suggested that expanding the scope of parent page retrieval can enhance model performance. The removal of summary retrieval in MHier-RAG _v_ 3 degraded the generalized accuracy to 43.3%, indicating that the document-level summary retrieval is essential for aggregating evidences across multiple pages. After discarding the page-level parent page retrieval, MHier-RAG _v_ 4 showed the lowest accuracy of 37.5%, thereby confirming the importance of parent page retrieval in integrating multi-modal content. 

It is worth noting that, the accuracy of unanswerable questions (UNA) in four variant models increased, suggesting that overly broad retrieval may introduce distractors in unanswerable cases. 

|**Variants**|**Parent Page**<br>**Retrieval**|**Summary**<br>**Retrieval**|**Visual**<br>**Info**|**TXT**<br>**LAY**<br>**CHA**<br>**TAB**<br>**FIG**<br>**SIN**<br>**MUL**<br>**UNA**<br>**Acc.**|
|---|---|---|---|---|
||||||
|MHier-RAG<br>MHier-RAG_v_1<br>MHier-RAG_v_2<br>MHier-RAG_v_3<br>MHier-RAG_v_4|✓(page=10)<br>✓(page=10)<br>✓(page=1)<br>✓(page=1)<br>_×_|✓<br>✓<br>✓<br>_×_<br>✓|✓<br>_×_<br>✓<br>✓<br>✓|**45.9**<br>**34.4**<br>**44.9**<br>**51.1**<br>**37.5**<br>**53.5**<br>**36.8**<br>76.2<br>**52.3**<br>41.0<br>22.8<br>27.3<br>50.9<br>21.9<br>42.0<br>29.4<br>88.2<br>46.9<br>40.6<br>27.6<br>31.9<br>39.7<br>28.5<br>51.2<br>18.9<br>83.9<br>46.6<br>34.7<br>25.1<br>28.6<br>34.3<br>25.7<br>47.2<br>13.9<br>84.8<br>43.3<br>26.4<br>15.6<br>22.8<br>23.8<br>21.9<br>32.2<br>14.0<br>**90.1**<br>37.5|



Table 3: Ablation experiments for parent page retrieval with flattened in-page index, summary retrieval with topological crosspage index and visual information on the MMLongBench-Doc dataset. 

|**Variants**|**Parent Page**<br>**Retrieval**|**Summary**<br>**Retrieval**|**Visual**<br>**Info**|**Acc.**|
|---|---|---|---|---|
||||||
|MHier-RAG<br>MHier-RAG_v_1<br>MHier-RAG_v_2<br>MHier-RAG_v_3<br>MHier-RAG_v_4|✓(page=10)<br>✓(page=10)<br>✓(page=1)<br>✓(page=1)<br>_×_|✓<br>✓<br>✓<br>_×_<br>✓|✓<br>_×_<br>✓<br>✓<br>✓|**55.7**<br>53.1<br>50.4<br>47.4<br>31.2|



|**Cornerstone LLMs**|**MMLongBench-Doc**<br>**Acc.**|**LongDocURL**<br>**Acc.**|
|---|---|---|
||||
|Qwen-turbo<br>GPT-4o<br>DeepSeek-chat<br>ERNIE-turbo|**52.3**<br>46.7<br>51.8<br>45.9|55.7<br>**57.2**<br>57.0<br>48.6|



Table 5: Extension on different LLMs for answer generation. 

Table 4: Ablation experiments for parent page retrieval with flattened in-page index, summary retrieval with topological cross-page index and visual inforamtion on the LongDocURL dataset. 

**==> picture [237 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
Impact of page number (summary number = 10) Impact of summary number (page number = 1)<br>52.5 Accuracy F1-Score 52.3 48 Accuracy F1-Score 46.6<br>50.0 48.2 49.6 4644 43.4 44.8 44.9<br>47.5 46.6 42<br>45.0 46.0 40<br>42.5 42.9 38 37.9<br>40.0 41.4 36 36.4 36.8<br>37.5 37.9 34 33.9<br>32<br>1 4 10 14 1 4 10 20<br>Page Number Summary Number<br>(a) Page Number. (b) Summary Number.<br>Metric(%) Metric(%)<br>**----- End of picture text -----**<br>


Figure 3: The trend of our MHier-RAG model performance changing with the page number and summary number on the MMLongBench-Doc dataset. 

**LongDocURL.** Table 4 reported similar ablation results on the LongDocURL dataset. The accuracy of MHier-RAG _v_ 1 declined with the loss of visual information, proving the indispensability of multi-modal information. When the scope of parent page was restricted to one, as in MHier-RAG _v_ 2 , the generalized accuracy dropped to 50.4%, highlighting that retrieving a broader set of parent pages is important for contextual grounding oriented at question answering. The removal of summary retrieval in MHier-RAG _v_ 3 reduced the accuracy to 47.4%, emphasizing the critical role of topological crosspage summary chunks in capturing connections between dispersed evidence across pages. The worst-performing variant, MHier-RAG _v_ 4 , which removed parent pages, yielded an average accuracy of 31.2%, suggesting the role of parent page retrieval in enabling rich multi-modal comprehension. 

## **Parameter Analysis on Content Size** 

Figure 3 showed the parameter analysis of retrieved page number and summary number for answer generation. MHier-RAG reached peak accuracy and F1 score when the parent page count was set to 10, but began to decline when the number of pages exceeded 10, which suggested that blindly increasing the page count may negatively impact performance. Meanwhile, as the number of summaries increased, the performance of MHier-RAG initially improved and subsequently declined, reaching its peak when the summary count was 10, which further supported the notion that an excessive amount of content may introduce irrelevant or distracting information, thereby negatively impacting the quality of the responses. Therefore, we set the number of pages and summaries to 10 to ensure that the retrieved content contains more evidence and avoids noisy information. 

## **Extension on LLMs for Answer Generation** 

As shown in Table 5, four different LLMs were selected as the cornerstone models to evaluate the performance impact on the document question-answering task. We found that Qwen-turbo achieved the highest accuracy at 52.3% on the MMLongBench-Doc dataset, while GPT-4o attained the best performance with an accuracy of 57.2% on the LongDocURL dataset. These results demonstrated that our proposed Retrieval-Augmented Generation (RAG)-based method, MHier-RAG, achieved excellent response on different large language models, which again validated the universality and transferability of our method. 

## **Case Study** 

Figure 4 demonstrated two cases from the MMLongBenchDoc and LongDocURL datasets. The first question asked the comparison of fan numbers, with the answer required integrating textual statistics of Vietnam from page 28 with visual data of Appota on page 31. VLVM-based methods, 

**==> picture [491 x 218] intentionally omitted <==**

**----- Start of picture text -----**<br>
ssonttnn tigen Ll P110 P111 a<br>$$Trendsin|engagement Vietnam igre‘With aboutOe,ae 700,000 saebuteeemreactive expenseusers inateantherSS|Vietnam, svertangpiatons.adhoe yon fnatageemesting snore 1 $$‘Appota'sFacebook|[|ee =rsass21,134,462 fanpage__—______HHH_____.“Android -7Store' Vigt Nan"@:aa became the biggestEEE)% 2,164,246VN tech communityon inches—>=aad[sires. atin| amenaES)ceree.vueieaoa| eohel [1pmlrnaaBa . aia|yfeyiotetatetdooHate|! ert)«|| ! 1SeI11 aSSSCsfo=Sa——>]2we SSSetGREEere—_eaeees -SSMs eyoeee-_ee 1<br>ORa: e Bese = | = a4 ~— CT see iH—a' Nae nceires Gent Ft Crea Boab | — 4<br>Question: Evidence Pages:  Are there more active Instagram users in Vietnam, or are there more fans for Appota's fan page?[P28, P31] P28 P31 Question: branch circuit breakers are listed in Table? How many different ratings of 1492-MCA  saa _————= ———eeeeeastane- se[asa<br>Ground Truth:  Appota Evidence Pages:  [P110, P111]<br>GPT-4o √ DeepSeek-chat × Ground Truth:  11<br>Response: There are more fans for Appota's fanpage  on Response:  There is no direct data  comparing the number GPT-4o × DeepSeek-chat √<br>Facebook than there are active Instagram users in Vietnam. of active Instagram users in Vietnam to the fan count of Response:  The table for 1492-MCA branch circuit breakers Response:  1492-MCA Branch Circuit Breakers – Current<br>There are about 700,000 active Instagram users in Vietnam. Appota's Facebook page.  (Wrong!) lists nine  different ratings: 10, 15, 20, 25, 30, 35, 40, 45, 50, Ratings: 10 A , 15 A , 20 A , 25 A , 30A, 35A, 40A, 45A, 50A,<br>Appota's1,134,462 fans. Qwen-VL-PlusResponse: 1,134,4622,164,246 mentioned(Wrong!) fanpagetotalper but The  (Correct!) × week,fans not report"Android quantified and while indicatesanStoreaverage Instagramin Việt Nam" terms thatuserthe ofadvertising engagement active hasfanpagea total users. hasofof is ERNIE-TurboResponse: userspage,Instagram700,000Appota's1,134,462 fans.in Appota's VietnamactivefanUsersComparing  (Correct!) page, √ Instagram fan into Vietnam:the page "Androidthenumberusers.Appota's had numberThere significantly Storeof fansofwereVietactiveforFanapproximatelyAppota'sNam," more PageInstagramFans: fans. hadfan and 60 Amperes Qwen-VL-PlusResponse: CircuitTheseathas a corresponding width per pole and catalog number forboth 120VAC and 240VAC configurations.40 ° CratingsBreakers,(104In ° correspondF),the . ×  (Wrong!) rangingprovidedthere aretofromthetable 10 15Acontinuousdifferentforto1492-MCA60A.  (Wrong!) ratingsampereEachBranchlisted.ratingrating 55A,the table. ERNIE-TurboResponse: differentcircuit to current quantities". directly 60A.breakers  (Correct!) ratedThereThe provide × currentdocumentinare  (Wrong!) table a11specific valuesform.different does Therefore,of figurenot currentthe explicitly for1492-MCAitratings listed"differentis impossiblelist branchallratedthein<br>MHier-RAGResponse: statesStorefollowers." Final Answer: ViệtInstagramNam'1. √ The Appotahashasquestion'about'1,134,462  (Correct!) compares700,000totalactivefans'.activeusers'4.Instagram1.1 inmillionVietnam.usersfansin3.>VietnamPage700,00033versusexplicitlyusers, Appota'sconfirmingmentionsfanAppota'spageAppota'sfollowers.fanfanpagepage2.has'AndroidPagemore28 MHier-RAGReasoning: 2. The relevant table is found on page 111 under the section '1492-MCA Branch Circuit Breakers'. 3. The table lists continuous ampere ratings from 10A to 60A, with increments of 5A. 4. Each rating (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60) is explicitly listed as a separate row in the table. 5. Counting these entries confirms there are 11 distinct ratings. Final Answer:   1. The question asks for the number of different ratings of 1492-MCA branch circuit breakers listed in the table.  √ 11  (Correct!)<br>(a) Test Case in MMLongBench-Doc. (b) Test Case in LongDocURL.<br>**----- End of picture text -----**<br>


Figure 4: Case Study on dataset MMLongBench-Doc and LongDocURL to compare the answer response of our MHier-RAG and LVLM-based methods (such as GPT-4o, DeepSeek-chat, Qwen-VL-Plus and ERNIE-Turbo). 

such as Deepseek-chat and Qwen-VL-Plus, failed to answer the question. However, MHier-RAG correctly retrieved fan count from separate pages and concluded that Appota had more followers. The second question asked the number of 1492-MCA branch circuit breakers’ rating, with the answer found in structured tables across page 110 and 111. GPT-4o and Qwen-VL-Plus gave incorrect answers of 9 and 10, respectively, which failed to identify all the entries. However, MHier-RAG correctly located the relevant table and identified the rating values. In summary, these cases again demonstrated that our method is adaptable for the comprehension of multi-modal and multi-page evidences that retrieved from document content. 

## **Related Work** 

Document question-answering has progressed from processing textual documents (Joshi et al. 2017; Kwiatkowski et al. 2019; Zhu et al. 2022) to tackle lengthy documents involved with multi-modal elements and complicated structures across multiple pages (Ma et al. 2024b; Deng et al. 2025; Hui, Lu, and Zhang 2024), which demands capability of modality comprehension and long-distance reasoning. 

**LVLM-based Methods for Multi-modal Doc-QA Task.** Large Vision-Language Models (Bai et al. 2023; Hu et al. 2024; Dong et al. 2024) were regarded as an effective solution to handle multi-modal document question-answering, since they combined the deep linguistic capabilities of large language models with advanced visual processing for document images. However, Ma et al. (Ma et al. 2024b) and Deng et al. (Deng et al. 2025) indicated that LVLMs still faced challenges in integrating evidences from different modalities and pages, and were prone to hallucinations. 

**RAG-based Methods for Multi-modal Doc-QA Task.** Traditional Retrieval-Augmented Generation (RAG)-based 

models (Lewis et al. 2020; Khattab and Zaharia 2020) exhibited a uni-modality bias, predominantly relying on textual information while inadequately incorporating visual evidence from documents. To fully exploit visual elements within documents, Colpali (Faysse et al. 2025), DSE (Ma et al. 2024a) and VisRAG (Yu et al. 2025) directly encoded the images of document pages for retrieval. MDocAgent (Han et al. 2025) separately used text-based and imagebased agents to handle textual and visual information, thereby obtaining critical information within their respective modalities in the retrieval phrase and generating refined answers. However, these existing RAG-based methods often overlooked the mutual connections between different modalities of information and remained inadequate in addressing cross-page integration and reasoning challenges. 

## **Conclusion** 

A retrieval-augmented generation method (MHier-RAG) was presented for multi-modal long-context document question-answering. A hierarchical index structure with flattened in-page and topological cross-page index was constructed to establish multi-modal connection and longdistance linkage. A multi-granularity retrieval with pagelevel parent page retrieval and document-level summary retrieval was proposed for searching required evidences, which were scattered across multi-modalities and multipages. Experiments conducted on two public datasets demonstrated the superiority of MHier-RAG in multi-modal long-context Doc-QA. 

## **Acknowledgments** 

This work was supported by the National Natural Science Foundation of China (Nos.61572250 and 62476135). 

Jiangsu Province Science & Tech Research Program(BE2021729), Open project of State Key Laboratory for Novel Software Technology, Nanjing University (KFKT2024B53), Jiangsu Province Frontier Technology Research and Development Program (BF2024005), Nanjing Science and Technology Research Project (202304016) and Collaborative Innovation Center of Novel Software Technology and Industrialization, Jiangsu, China. 

## **References** 

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ . 

Agrawal, P.; Antoniak, S.; Hanna, E. B.; Bout, B.; Chaplot, D. S.; Chudnovsky, J.; Costa, D.; Monicault, B. D.; Garg, S.; Gervet, T.; Ghosh, S.; H´eliou, A.; Jacob, P.; Jiang, A. Q.; Khandelwal, K.; Lacroix, T.; Lample, G.; de Las Casas, D.; Lavril, T.; Scao, T. L.; Lo, A.; Marshall, W.; Martin, L.; Mensch, A.; Muddireddy, P.; Nemychnikova, V.; Pellat, M.; von Platen, P.; Raghuraman, N.; Rozi`ere, B.; Sablayrolles, A.; Saulnier, L.; Sauvestre, R.; Shang, W.; Soletskyi, R.; Stewart, L.; Stock, P.; Studnia, J.; Subramanian, S.; Vaze, S.; Wang, T.; and Yang, S. 2024. Pixtral 12B. _CoRR_ , abs/2410.07073. 

Anthropic. 2024. Introducing the next generation of claude. Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 1(2): 3. 

Cho, J.; Mahata, D.; Irsoy, O.; He, Y.; and Bansal, M. 2024. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ . 

Deng, C.; Yuan, J.; Bu, P.; Wang, P.; Li, Z.; Xu, J.; Li, X.; Gao, Y.; Song, J.; Zheng, B.; and Liu, C. 2025. LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics_ , volume 1, 1135– 1159. 

Ding, Y.; Huang, Z.; Wang, R.; Zhang, Y.; Chen, X.; Ma, Y.; Chung, H.; and Han, S. C. 2022. V-Doc : Visual questions answers with Documents. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 21460–21466. 

Dong, X.; Zhang, P.; Zang, Y.; Cao, Y.; Wang, B.; Ouyang, L.; Zhang, S.; Duan, H.; Zhang, W.; Li, Y.; et al. 2024. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _Advances in Neural Information Processing Systems_ , 37: 42566–42592. 

Faysse, M.; Sibille, H.; Wu, T.; Omrani, B.; Viaud, G.; Hudelot, C.; and Colombo, P. 2025. ColPali: Efficient Document Retrieval with Vision Language Models. In _The Thirteenth International Conference on Learning Representations_ . 

Gemini; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.; Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.; et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ . 

Han, S.; Xia, P.; Zhang, R.; Sun, T.; Li, Y.; Zhu, H.; and Yao, H. 2025. MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding. _CoRR_ , abs/2503.13964. 

Hu, A.; Xu, H.; Ye, J.; Yan, M.; Zhang, L.; Zhang, B.; Li, C.; Zhang, J.; Jin, Q.; Huang, F.; et al. 2024. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_ . 

Hui, Y.; Lu, Y.; and Zhang, H. 2024. UDA: A Benchmark Suite for Retrieval Augmented Generation in Real-World Document Analysis. In _Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems_ . 

Jain, C.; Wu, Y.; Zeng, Y.; Liu, J.; hengyu Dai, S.; Shao, Z.; Wu, Q.; and Wang, H. 2025. SimpleDoc: Multi-Modal Document Understanding with Dual-Cue Page Retrieval and Iterative Refinement. _CoRR_ , abs/2506.14035. 

Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics_ , volume 1, 1601–1611. Association for Computational Linguistics. 

Khattab, O.; and Zaharia, M. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In _Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval_ , 39–48. 

Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A. P.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin, J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang, M.; Dai, A. M.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019. Natural Questions: a Benchmark for Question Answering Research. _Trans. Assoc. Comput. Linguistics_ , 7: 452–466. 

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; K¨uttler, H.; Lewis, M.; Yih, W.; Rockt¨aschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In _Advances in Neural Information Processing Systems 33_ . 

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; and Li, C. 2025. LLaVA-OneVision: Easy Visual Task Transfer. _Trans. Mach. Learn. Res._ , 2025. 

Liu, A.; Feng, B.; Wang, B.; Wang, B.; Liu, B.; Zhao, C.; Dengr, C.; Ruan, C.; Dai, D.; Guo, D.; et al. 2024. Deepseekv2: A strong, economical, and efficient mixture-of-experts language model. _arXiv preprint arXiv:2405.04434_ . 

Livathinos, N.; Auer, C.; Lysak, M.; Nassar, A.; Dolfi, M.; Vagenas, P.; Ramis, C. B.; Omenetti, M.; Dinkla, K.; Kim, Y.; Gupta, S.; de Lima, R. T.; Weber, V.; Morin, L.; Meijer, I.; Kuropiatnyk, V.; and Staar, P. W. J. 2025. Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion. _CoRR_ , abs/2501.17887. 

Lu, H.; Liu, W.; Zhang, B.; Wang, B.; Dong, K.; Liu, B.; Sun, J.; Ren, T.; Li, Z.; Yang, H.; et al. 2024. Deepseekvl: towards real-world vision-language understanding. _arXiv preprint arXiv:2403.05525_ . 

Luo, C.; Shen, Y.; Zhu, Z.; Zheng, Q.; Yu, Z.; and Yao, C. 2024. LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 15630–15640. 

Ma, X.; Lin, S.; Li, M.; Chen, W.; and Lin, J. 2024a. Unifying Multimodal Retrieval via Document Screenshot Embedding. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , 6492–6505. 

Ma, X.; Zhuang, S.; Koopman, B.; Zuccon, G.; Chen, W.; and Lin, J. 2025. VISA: Retrieval Augmented Generation with Visual Source Attribution. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics_ , 30154–30169. 

Xia, P.; Zhu, K.; Li, H.; Zhu, H.; Li, Y.; Li, G.; Zhang, L.; and Yao, H. 2024. RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , 1081–1093. 

Xing, S.; Wang, Y.; Li, P.; Bai, R.; Wang, Y.; Qian, C.; Yao, H.; and Tu, Z. 2025. Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization. _CoRR_ , abs/2502.13146. 

Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y.; Liu, Z.; Wang, S.; Han, X.; Liu, Z.; and Sun, M. 2025. VisRAG: Vision-based Retrieval-augmented Generation on Multimodality Documents. In _The Thirteenth International Conference on Learning Representations_ . 

Zhu, F.; Lei, W.; Feng, F.; Wang, C.; Zhang, H.; and Chua, T. 2022. Towards Complex Document Understanding By Discrete Reasoning. In _MM ’22: The 30th ACM International Conference on Multimedia_ , 4857–4866. 

Ma, Y.; Zang, Y.; Chen, L.; Chen, M.; Jiao, Y.; Li, X.; Lu, X.; Liu, Z.; Ma, Y.; Dong, X.; et al. 2024b. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems_ , 37: 95963–96010. 

Mishra, A.; Shekhar, S.; Singh, A. K.; and Chakraborty, A. 2019. OCR-VQA: Visual Question Answering by Reading Text in Images. In _2019 International Conference on Document Analysis and Recognition_ , 947–952. 

## OpenAI. 2024. Hello gpt-4o. 

## Qwen. 2024. Introducing qwen1.5. 

Santhanam, K.; Khattab, O.; Saad-Falcon, J.; Potts, C.; and Zaharia, M. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , 3715–3734. 

Smith, R. 2007. An Overview of the Tesseract OCR Engine. In _9th International Conference on Document Analysis and Recognition_ , 629–633. 

Suri, M.; Mathur, P.; Dernoncourt, F.; Goswami, K.; Rossi, R. A.; and Manocha, D. 2025. VisDoM: MultiDocument QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation. In _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics_ , 6088– 6109. 

Tanaka, R.; Nishida, K.; Nishida, K.; Hasegawa, T.; Saito, I.; and Saito, K. 2023. SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images. In _ThirtySeventh AAAI Conference on Artificial Intelligence_ , 13636– 13645. 

Xia, P.; Zhu, K.; Li, H.; Wang, T.; Shi, W.; Wang, S.; Zhang, L.; Zou, J.; and Yao, H. 2025. MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models. In _The Thirteenth International Conference on Learning Representations_ . 

## **Appendix** 

## **Research Motivation** 

Multi-modal long-context document question-answering (Doc-QA) involves answering queries by analyzing and integrating evidences across texts, tables, charts, images and layouts within multiple pages, requiring the capability of multi-modal connection and long-distance reasoning. However, there are currently few multi-modal long-context DocQA methods equipped with these abilities, which are worth further research. 

Figure 5 illustrates the capabilities required for multimodal long-context Doc-QA and the advantage of our proposed MHier-RAG method. For the question “What percentage of respondents of the sector in which 15% are doing promotions to customers over Wi-Fi use wifi at stores?”, evidences required for the answer are scattered in the visual charts on page 11 and page 14. To correctly answer this question, multi-modal Doc-QA methods need to establish multi-modal connections. Due to the common keywords and similar semantics, the question is relatively easy to retrieve the textual titles “% RESPONDENTS USING WIFI AT STORES” in page 11 and “Are you doing promotions to customers over Wi-Fi?” in page 14. Therefore, it is crucial to associate these textual captions with their surrounding related visual bar charts. Multi-modal Doc-QA methods also need to achieve long-distance reasoning to synthesize information on these non-consecutive pages. To be specific, Doc-QA methods first require to identify the “Hospitality” industry (the only category with a 15% promotion from the visual chart on page 14), and then retrieve its corresponding 100% Wi-Fi usage rate from the data on page 11. 

Our MHier-RAG first confirmed from page 14 that hospitality is the only industry that conducts a 15% promotion through Wi-Fi, then found the proportion of respondents in the hotel industry who use Wi-Fi in stores from page 11, and finally concluded the correct answer 100 through reasonable multi-step reasoning. This indicates that our MHier-RAG has the ability of modality correlation and long-distance reasoning, which is crucial for integrating multi-modal crosspage evidences to answer questions. 

## **Implementation Details** 

In the settings of our MHier-RAG, we used docling to achieve pdf parsing and used off-the-shelf Large VisionLanguage Models (such as Qwen-vl-plus) to describe visual information in documents. For the flattened in-page index, the chunk size was set to 300 tokens (about 15 sentences) and encoded by text-embedding-v4 provided by Qwen. To avoid information loss due to text split, a text overlap with 50 tokens was added. For the topological cross-page index, the chunk size was set to 100 (about 5 sentences) and encoded by multi-qa-mpnet-base-cos-v1 provided by Sentence-Transformer. The gaussian mixture model was used for chunk clustering, and Gpt-3-turbo was adopted for summarizing the clustered chunks. For multi-granularity evidence retrieval, we selected 10 highest-scored parent pages and summaries as the input context for each question. We used Large Vision-Language Models (such as Qwen-turbo) 

**==> picture [223 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
Long-distance<br>es — a — Reasoning — |<<br>s | o—_— S|.<br>oa a 7 =. =<br>Who Has Wifi Installed 1Differentiated Useof [In-Store][ Wi-Fi]<br>conesneftice sc, a Multi-modality  en<br>Connection<br>1<br>rr a<br>P11 P14<br>Question:  What percentage of respondents of the sector in which 15% are doing<br>promotions to customers over Wi-Fi use wifi at stores?<br>Evidence Pages:  [P11, P14] Ground Truth:  100<br>**----- End of picture text -----**<br>


**Reasoning:** 1. The question asks for the percentage of respondents in the sector where 15% are doing promotions over Wi-Fi who use Wi-Fi at stores. 2. On **page 14** , the chart **'Are you doing promotions to customers over Wi-Fi?** ' shows that the 'Hospitality' sector has 15% 'Yes' responses. 3. The sector in question is Hospitality, as it is the only category with 15% promotions over Wi-Fi. 4. On **page 11** , the chart **'% RESPONDENTS USING WIFI AT STORES'** indicates that the 'Hospitality' sector has a score of 100%. 5. This 100% represents the percentage of respondents in the Hospitality sector who use Wi-Fi at stores. **Final Answer:** 100 **（** Our MHier-RAG Method **）** 

Figure 5: Necessities of multi-modal connection and longdistance reasoning for multi-modal long-context Doc-QA methods. 

to conduct retrieval re-ranking. We directly used off-theshelf LLMs for answer generation. All experiments were conducted on a single NVIDIA A100 GPU. 

## **Prompt Setting** 

For answer generation in multi-modal long-context document question-answering, a multi-step reasoning method was proposed for encompassing evidence curation and chain-of-thought reasoning for the retrieved relevant sources. 

**Prompt for Answer Generation** The prompt for answer generation can be divided into several parts: 

- Template for Context and Questions 

- General Guidelines 

- Response Formats 

- Question-Answer Examples 

## **A. Template for Context and Questions** 

Here is the context: _{_ context _}_ Here is the question: _{_ question _}_ 

## **B. General Guidelines** 

You are a RAG (Retrieval-Augmented Generation) answering system. Your task is to answer the given question based only on information from the pdf report, which is uploaded in the format of relevant evidences extracted using RAG. 

Before giving a final answer, carefully think out loud and step by step. Pay special attention to the wording of the question. 

- Keep in mind that the content containing the answer may be worded differently than the question. - The question was autogenerated from a template, so it may be meaningless or not applicable to the given report. 

## **C. Response Formats** 

The response format consists of four parts: (1) Step By Step Analysis, (2) Reasoning Summary, (3) Relevant Pages, and (4) Final Answer. 

## Final Answer [List] 

A list of values extracted from the context. Each value should be: 

- For strings: exactly as it appears in the context - For numbers: converted to appropriate type (int or float) 

- For nested lists: maintain the original structure - Return ‘Not answerable’ if information is not available in the context 

## Final Answer [Integer] 

- An integer value is expected as the answer. - Pay attention to units (thousands, millions, etc.) and adjust accordingly 

- Round to nearest integer if necessary 

- Return ‘Not answerable’ if: 

- The value is not an integer 

- Information is not available 

- Currency mismatch occurs 

## Step By Step Analysis 

Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. 

## Final Answer [String] 

- A string value is expected as the answer. - Extract exactly as it appears in the context 

- Do not modify or summarize the text 

## Reasoning Summary 

- Return ‘Not answerable’ if information is not available in the context 

Concise summary of the step-by-step reasoning process. Around 50 words. 

## Final Answer [Float] 

## Relevant Pages 

List of page numbers containing information directly used to answer the question. Include only: - Pages with direct answers or explicit statements. 

- Pages with key information that strongly supports the answer. 

- Do not include pages with only tangentially related information or weak connections to the answer. At least one page should be included in the list. 

A floating-point number is expected as the answer. - Maintain original decimal precision from the context 

- Pay attention to units (thousands, millions, etc.) 

- and adjust accordingly 

- Return ‘Not answerable’ if: 

- The value is not a number 

- Information is not available 

- Currency mismatch occurs 

## **D. Question-Answer Examples** 

## Final Answer 

Note: different prompts for different data types. 

The expected answer data for questions includes various types, such as List, Integer, String and Float. Different prompt variants have been proposed to achieve higherquality responses. 

Depending on different data types, several questionanswer examples are also provided to improve response quality, which contain step by step analysis, reasoning summary, relevant pages and final answer. It helps the large language model to respond required structured output format for different answer types. 

Examples of answer type as a list, a float value, an integer value and a string are shown as follows: 

## Examples [List] 

## **Question** 

What are the quarterly revenue figures for Apple Inc. in 2022? 

## Examples [String] 

## **Question** 

What’s the percentage of people who are democrats and voted in the last election compared to the entire population in 2024? 

## **Answer** 

**step** ~~**b**~~ **y** ~~**s**~~ **tep** ~~**a**~~ **nalysis:** 1. The question asks for quarterly revenue figures for Apple Inc. in 2022, which implies we need to find four distinct values corresponding to each quarter. 2. Examining the context, we find a table titled ‘Quarterly Financial Results’ on page 45 that lists revenue figures for each quarter of 2022. 3. The table shows: Q1(123.9B), Q2(97.3B), Q3(82.96B), Q4(90.15B). 4. We verify these are indeed revenue figures by checking the column header and accompanying notes. 5. The values are extracted exactly as presented, converted to float type for consistency. 

**reasoning** ~~**s**~~ **ummary:** The ‘Quarterly Financial Results’ table on page 45 provides the exact quarterly revenue figures for 2022, which are extracted and converted to float values. 

**relevant** ~~**p**~~ **ages:** [45] 

## **Answer** 

**step** ~~**b**~~ **y** ~~**s**~~ **tep** ~~**a**~~ **nalysis:** 1. Question requires two precise data points: democrat voters and total population 2. Searched for ‘2024 election’ references - none found 3. Checked all demographic sections - no voting breakdown by party 4. Verified document metadata - report finalized Q3 2023 (pre-election) 5. Attempted alternative queries - no matching tables/charts 6. Conclusion: Data unavailable in this report 

**reasoning** ~~**s**~~ **ummary:** Document contains no 2024 election data (pre-dates election) and lacks democrat-specific voting percentages, making question unanswerable. 

**relevant** ~~**p**~~ **ages:** [] 

**final** ~~**a**~~ **nswer:** Not Answerable 

**final** ~~**a**~~ **nswer:** [123.9, 97.3, 82.96, 90.15] 

## Examples [Float] 

## **Question** 

What was the gross profit margin percentage for NVIDIA Corporation in Q3 2022? 

## Examples [Integer] 

## **Question** 

How many employees did Tesla Inc have at the end of 2022? 

## **Answer** 

**step** ~~**b**~~ **y** ~~**s**~~ **tep** ~~**a**~~ **nalysis:** 1. The question asks for the number of Tesla Inc. employees at the end of 2022, which should be a whole number. 2. On page 56 of the annual report, we find the statement: ‘As of December 31, 2022, we employed approximately 127,855 full-time employees worldwide.’ 3. The number 127,855 is explicitly stated as the employee count. 4. We verify this is a global total by checking the context which mentions ‘worldwide’. 5. No unit conversion is needed as this is already a direct count. 

**reasoning** ~~**s**~~ **ummary:** Page 56 explicitly states Tesla employed 127,855 full-time employees worldwide as of December 31, 2022. 

**relevant** ~~**p**~~ **ages:** [56] 

**final** ~~**a**~~ **nswer:** 127855 

## **Answer** 

**step** ~~**b**~~ **y** ~~**s**~~ **tep** ~~**a**~~ **nalysis:** 1. The question asks for NVIDIA’s gross profit margin percentage in Q3 2022, which should be a decimal number. 2. On page 32 of the quarterly report, we find the statement: ‘Gross margin for the quarter was 53.6%, down from 56.1% in the prior quarter.’ 3. The value 53.6% is explicitly stated as the gross margin for the quarter. 4. We verify this is for Q3 2022 by checking the report header and date. 5. The percentage is converted to its decimal equivalent (53.6). 

**reasoning** ~~**s**~~ **ummary:** Page 32 states NVIDIA’s Q3 2022 gross margin was 53.6%, which is converted to the decimal value 53.6. 

**relevant** ~~**p**~~ **ages:** [32] 

**final** ~~**a**~~ **nswer:** 53.6 

**Prompt for LLM-Based Re-Ranking** The prompt for LLM-based re-ranking in the page-level parent page retrieval can be devided into two parts: (1) Template for Pages and Questions, (2) General Guidelines. 

## **A. Template for Pages and Questions** 

Here is the query: _{_ query _}_ Here is the retrieved text block: _{_ retrieved ~~p~~ age _}_ 

## **B. General Guidelines** 

You are a RAG (Retrieval-Augmented Generation) retrievals ranker. You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided. 

Instructions: 

## **1. Reasoning:** 

Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided. 

## **2. Relevance Score (0 to 1, in increments of 0.1):** 

0 = Completely Irrelevant: The block has no connection or relation to the query. 

0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query. 0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection. 

0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail. 

0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive. 0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance. 

0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity. 

0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information. 

0.8 = Very Relevant: Strongly relates to the query and provides significant information. 

0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information. 1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information. 

## **3. Additional Guidance:** 

- Objectivity: Evaluate block based only on their content relative to the query. 

- Clarity: Be clear and concise in your justifications. - No assumptions: Do not infer information beyond what’s explicitly stated in the block. 

