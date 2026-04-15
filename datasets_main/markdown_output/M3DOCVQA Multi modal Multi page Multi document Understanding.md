This ICCV Workshop paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **M3DOCVQA: Multi-modal Multi-page Multi-document Understanding** 

Jaemin Cho[1][*] Debanjan Mahata[2] Ozan[˙] Irsoy[2] Yujie He[2] Mohit Bansal[1] 1UNC Chapel Hill 2Bloomberg _}_ @cs.unc.edu _{_ dmahata,oirsoy,yhe247 _}_ @bloomberg.net 

_{_ jmincho,mbansal _}_ @cs.unc.edu 

## **Abstract** 

_Document Visual Question Answering (DocVQA) offers a promising approach to extracting insights from large document corpora. However, existing benchmarks focus on evaluating multi-modal understanding within a single document. This gap hinders the development of methods integrating scattered information across pages and documents. To address this, we introduce_ _**M3DocVQA** , the first benchmark designed for multi-modal, multi-page, and multi-document understanding. M3DocVQA comprises over 3,000 PDF documents with more than 40,000 pages, offering a challenging environment where evidence is distributed across diverse sources and modalities. Alongside the dataset, we introduce M3DocRAG, a baseline method based on multi-modal retrieval-augmented generation. M3DocRAG flexibly handles both single and multiple document settings while preserving critical visual information, establishing a useful starting point for future work in open-domain multi-modal document understanding. Our experiments across three benchmarks (M3DocVQA, MMLongBench-Doc, and MP-DocVQA) show that existing methods struggle with open-domain question answering over extensive, multi-modal documents. Although M3DocRAG has shown promising performance, there is large room for future improvement. We provide comprehensive ablation studies of different indexing, multi-modal language models, and multi-modal retrieval models, along with qualitative examples to guide future research._ 

## **1. Introduction** 

Document visual question answering (DocVQA) [16, 32, 42, 44, 58] is a multi-modal task that answers textual questions by interpreting information contained within document images. The capability of accurately and efficiently answering questions across numerous, lengthy documents with intricate layouts would greatly benefit many domains such as finance, healthcare, and law, where document AI 

> * _Work done during an internship at Bloomberg as a recipient of the Bloomberg Data Science Ph.D. Fellowship._ 

assistants can streamline the daily processing of large volumes of documents, improving productivity and enabling faster, more informed decision-making. However, existing DocVQA benchmarks focus on evaluating question answering (QA) capabilities within a single document, so their questions assume that a QA model already knows the context of that specific document. For example, they have questions given a single-page CV and _“In which year did the author publish their first journal article?”_ as shown in Fig. 1 (left). This gap hinders the development of methods integrating scattered information across pages and documents. 

To address this limitation, we introduce **M3DOCVQA** ( **M** ulti-modal **M** ulti-page **M** ulti- **Doc** ument **V** isual **Q** uestion **A** nswering), an open-domain dataset that significantly raises the challenge of DocVQA to answering questions from a large document corpus (Sec. 2). As exemplified in Fig. 1 (right), M3DOCVQA supports scenarios like reviewing a corpus of thousands of multi-page CVs and answering a question like _“Which candidate has published in ICCV on document understanding?”_ By extending the MultimodalQA dataset’s [55] closed-domain context to an open-domain setting, M3DOCVQA introduces 2,441 questions spanning 3,368 PDF documents, which collectively contain over 41,005 pages of diverse multi-modal content, including text, images, and tables. This dataset presents real-world challenges by requiring models to navigate complex reasoning paths across pages and within various types of document elements, better reflecting the intricacies of document understanding. 

As a useful starting point for M3DOCVQA, we introduce **M3DOCRAG** , a baseline method based on multi-modal retrieval-augmented generation (Sec. 3). M3DOCRAG retrieves relevant document pages using a multi-modal retrieval model, such as ColPali [19], and generates answers to questions from the retrieved pages using a multi-modal language model (MLM), such as Qwen2VL [60]. M3DOCRAG operates in three stages: In (1) document embedding (Sec. 3.1), we convert all document pages into RGB images and extract visual embeddings ( _e.g_ ., via ColPali) from the page images. In (2) page retrieval (Sec. 3.2), we retrieve the top-K pages of high similarity 

6237 

**==> picture [419 x 99] intentionally omitted <==**

**----- Start of picture text -----**<br>
Existing DocVQA datasets: Closed-domain M3DocVQA (Ours): Open-domain<br>Context: Single PDF Context: 3K PDFs<br>Model Answer Model Answer<br>— on [|].<br>Context-specific question (e.g., given a CV) Open-domain question (e.g., given thousands of CVs)<br>“In which year did the author publish their first journal article?” “Which candidate has published in ICCV on document understanding?”<br>Se)<br>**----- End of picture text -----**<br>


Figure 1. Comparison of existing DocVQA datasets (left; _e.g_ ., DocVQA [44]) and our **M3DOCVQA** dataset (right). In contrast to previous DocVQA datasets that have questions that are specific to a single provided PDF, M3DOCVQA has information-seeking questions that benchmark open-domain question answering capabilities across more than 3,000 PDF documents ( _i.e_ ., 40,000+ pages). 

with text queries ( _e.g_ ., MaxSim operator for ColPali). For the open-domain setting, we create approximate page indices, such as inverted file index (IVF) [53, 68], for faster search. In (3) question answering (Sec. 3.3), we conduct visual question answering with MLM to obtain the final answer. M3DOCRAG can flexibly handle DocVQA in both closed domain ( _i.e_ ., a single document) and open-domain ( _i.e_ ., a large corpus of documents) settings. 

We benchmark state-of-the-art methods and M3DOCRAG baseline in three datasets: M3DOCVQA, MMLongBench-Doc [42], and MP-DocVQA [58], which cover both open-domain (Sec. 5.1) and closed-domain (Sec. 5.2) DocVQA settings. We find existing methods struggle with open-domain document understanding in M3DOCVQA, and M3DOCRAG achieves the text-only RAG baseline, but there remains large room for future improvement. We also provide a comprehensive analysis (Sec. 5.3) about different indexing, MLMs, and retrieval components and qualitative examples where M3DOCRAG can successfully handle various scenarios, such as when the relevant information exists across multiple pages and when answer evidence only exists in images. 

## **2. M3DOCVQA: A New Benchmark for Multi-modal, Multi-page, Multi-document Understanding** 

We present **M3DOCVQA** ( **M** ulti-modal **M** ulti-page **M** ulti- **Doc** ument **V** isual **Q** uestion **A** nswering), a new opendomain DocVQA benchmark designed to evaluate the ability to answer questions using multi-modal information from a large corpus of documents. 

As illustrated in Fig. 1 and Table 1, existing DocVQA datasets [16, 32, 42, 44, 58] primarily focus on evaluating question answering within the context of a single document ( _i.e_ ., closed-domain). These datasets are not well-suited for benchmarking open-domain visual question answering, where relevant information, often in multiple modalities such as text, images, and tables, must be retrieved 

Table 1. Comparison of recent DocVQA datasets with the proposed M3DOCVQA dataset in terms of context size. 

|Datasets|Multi-page|Multi-document|Avg. # pages per question|
|---|---|---|---|
|DocVQA [44]|✗|✗|1|
|MP-DocVQA [58]|✓|✗|8.3|
|DUDE [32]|✓|✗|5.7|
|MMVQA [16]|✓|✗|9.6|
|MMLongBench-Doc [42]|✓|✗|47.5|
|**M3DOCVQA**(ours)|✓|✓|41,005|



Table 2. M3DOCVQA statistics. 

|**# Documents**|3,368|
|---|---|
|**# Pages**|41,005|
|- Avg. # pages per document|12.2|
|**# Questions**|2,441|
|(Answer modalities)||
|- Text|1,048 (35.2%)|
|- Table|860 (42.9%)|
|- Image|533 (21.8%)|
|(Question hops)||
|- Single-hop|1,461 (59.9%)|
|- Multi-hop|980 (40.1%)|
|Avg. # characters for question|100.8|
|Avg. # characters for answer|7.1|



from multiple documents. This limitation stems from their questions being designed around specific content on certain pages within a single document. In real-world scenarios, users often seek answers that span across multiple documents and modalities, making open-domain settings critical. However, the questions in the existing DocVQA datasets are not applicable in such an open-domain setting. For example, a question from MP-DocVQA, such as _“What was the gross profit in the year 2009?”_ assumes that the model already has access to specific information within the document. M3DOCVQA challenges models in an open-domain DocVQA setting, where they must navigate a large ‘haystack’ of multi-modal documents and re- 

6238 

**==> picture [485 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
MultimodalQA (Talmor et al., 2021) Our PDFs in M3DocVQA<br>“Question”: “…”,“Question”: “…”,“Answer”: “…”“Supporting Contexts”: [  {https-13_La_Liga  },... ://en.wikipedia.org/wiki/2012“text”: “…”,“title”: “2012-13 La Liga”,“url”:  ... 1.2.3. Obtain URLs of supporting contextsRender ina web browserCreate PDFs ArticleFrom Wikipedia, the free encyclopediaThe 82nd since its establishment. The campaign began on 18 August 2012, and ended on 1 June2013.and amassing 100 points, equalling in previous years, BBVA model to be used throughout the season for all matches.TeamsA total of 20 teams contested the league, including 17 sides from the three promoted from the Segunda División, and the victorious team of the play-offs.Villarreal CFSegunda DivisiónSporting de Gijón returned to Segunda División after a four-year tenure in La Liga, whileRacing de Santander ended ten consecutive seasons in La Liga, the longest period in itshistory.The three teams that were relegated were replaced by three sides: División championafter a five-year absence. The third promoted team was decided in the where  Stadia and locations Athletic BilbaoAtlético MadridBarcelonaBetisCelta VigoDeportivo La CoruñaEspanyolGetafeGranadaLevanteMálagaMallorca 2012–13  2012–13 La Li [[2]] Deportivo de La CoruñaReal Valladolid Team TalkBarcelona[,  editSporting de Gijón La Liga  ]  the previous season: Villarreal were relegated after twelve years in Nike. The second-placing team  won the league for a 22nd time, after leading the league the entire season returned to La Liga after two seasons in Segunda División. season (known as the  provided the official ball for all matches, with a new Nike Maxim Liga Location of stadium BilbaoMadridBarcelonaSevilleVigoA CoruñaBarcelonaGetafeGranadaValenciaMálagaPalma2011–12 Segunda División[ edit made an immediate return to the top level as  and g ]a Racing de SantanderReal Madrid Liga BBVA Celta de Vigo's points record from the San MamésVicente CalderónCamp NouBenito VillamarínBalaídosRiazorCornellà-El PratColiseum Alfonso PérezNuevo Los CármenesCiutat de ValènciaLa RosaledaIberostar Stadium. This included the two top teams from the for sponsorship reasons) was the Stadium  were relegated to  was also promoted to La Liga2011–12 Segunda División [[3][4]] 2011–12 seasonpromotion play-offsprevious season2012–13 Capacity Segunda39,75054,85199,35452,74531,80034,60040,50017,70022,52425,53428,96323,142La Liga and. As, SeasonDatesChampionsRelegatedChampionsLeagueEuropa LeagueMatchesplayedGoals scoredTop goalscorerBestgoalkeeperBiggest homewinBiggest awaywinHighestscoringLongestwinning runLongestunbeaten runLongestwinless runLongest losingrunHighestattendance Read 2012–1318 August 2012 – 1 June2013Barcelona22nd titleMallorcaDeportivo La CoruñaZaragozaBarcelonaReal MadridAtlético MadridReal SociedadValenciaReal BetisSevilla3801,091 (2.87 per match)Lionel Messi(46 goals)Thibaut Courtois(0.78 goals/match)Atlético MadridDeportivo La Coruña(9 December 2012)Rayo VallecanoBarcelona(27 October 2012)Mallorca(28 October 2012)Valencia(20 January 2013)Deportivo La CoruñaBarcelona12 matchesBarcelona19 matchesBarcelona15 matchesZaragoza6 matchesDeportivo La CoruñaMallorca96,589BarcelonaEdit La Liga Create account [[1]] View history 0–5  0–5  [[1]][[1]]  (20 October 2012) [[1]][[1]]  2–2  38 languages Real MadridReal Madrid 6–0Real Madrid 0–5Log in 4–5Tools Personnel and sponsorship OsasunaRayo VallecanoReal MadridReal SociedadSevillaValenciaValladolidZaragozaAthletic BilbaoAtlético MadridBarcelonaBetisCelta de VigoDeportivo La CoruñaEspanyolGetafeGranadaLevanteMálagaMallorcaOsasunaRayo VallecanoReal MadridReal SociedadSevillaValenciaValladolidZaragoza1.  ^Team Huawei is the sponsor for select matches.PamplonaMadridMadridSan SebastiánSevilleValenciaValladolidZaragozaTito VilanovaMarcelo BielsaDiego SimeonePepe MelPaco HerreraFernando VázquezJavier AguirreLuis García PlazaLucas AlcarazJuan Ignacio MartínezManuel PellegriniGregorio ManzanoJosé Luis MendilibarPaco JémezJosé MourinhoPhilippe MontanierUnai EmeryErnesto ValverdeMiroslav ĐukićManolo Jiménez Head Coach [ editEl SadarCampo de VallecasSantiago BernabéuAnoetaRamón SánchezPizjuánMestallaJosé ZorrillaLa Romareda ] Andrés PalopCarlos GurpeguiGabiCarles PuyolJuanmaBorja OubiñaManuel PabloCristian ÁlvarezJaime GavilánManuel LucenaSergio BallesterosJesús GámezJosé NunesPatxi PuñalPitiIker CasillasXabi PrietoDavid AlbeldaJavier BarajaJavier Paredes Captain 19,55315,48985,45432,07645,50055,00026,51234,596 manufacturer UmbroNikeNikeMacronLi-NingLottoPumaJomaLuanviKelmeNikeMacronAstoreErreàAdidasNikeUmbroJomaKappaMercuryLocation of teams in Celta VigoDeportivo LaCoruña Kit Atlético MadridSevillaBetisPetronorAzerbaijanQatar FoundationCirsa and CitroënEstrella GaliciaCancúnConfremar and Caja GranadaComunitat ValencianaUNESCORiviera MayaLacturale and NevirAE — Adquisiciones Empresariales and NevirBWINCanal+InterwettenJinKO SolarEl Norte de CastillaProniñoValladolid Averageattendance2012–13 La Liga Málaga2011–12Real MadridGranadaGetafeAthletic Bilbao [6][4]  and Canal+ and  and  [5] RayoVallecanoAndalucía, HuaweiKutxaEstrella GaliciaOsasuna Shirt sponsor IG MarketsReal SociedadZaragoza29,430, UNICEF [2][4][1][2] Valencia [6][4]  and  [[1]] LevanteEspanyolKyocera [4][2][2][3]  and  [4] Mallorca2013–14Barcelona [2] TV3 [6] [2] Managerial changes League tableBarcelonaValenciaRayoVallecanoGranadaEspanyolValenciaValenciaDeportivo LaCoruñaSevillaGranadaMallorcaDeportivo LaCoruñaCelta de Vigo Pos 1234562. 3. 4. 5. 6.  Team^^^^^ BarcelonaReal MadridAtlético MadridReal SociedadValenciaMálagaOn the back of shirt.Barcelona makes a donation to UNICEF in order to display the charity's logo on the back of the club's kit.On the shorts.Málaga makes a donation to UNESCO in order to display the charity's logo on the club's kit.On the left sleeve. Team(C) SandovalPochettinoPellegrino(caretaker)AnquelaCaparrósPaciência[ edit Outgoing Abel Resino manager Pep GuardiolaUnai EmeryJosé RamónMauricioMauricioVoroJosé Luis OltraMíchelJuan AntonioJoaquínDomingosPaco Herrera ] [ edit ] Pld 383838383838 End of contractEnd of contractEnd of contractEnd of contractMutual consentSackedEnd of tenure ascaretakerSackedSackedSackedSackedMutual consentSacked322623181916 WManner ofdeparture 12 D 47789 1311 L 2588 103115 GF 6570675330 June 201230 June 201230 June 201230 June 201226 November20121 December20125 December201230 December201214 January201330 January20134 February201311 February201318 February2013 vacancyDate ofGA 404231495450 [[13]][[15]][[16]][[18]][[20]][[22]][[24]][[26]][[28]] +75+61+34+21+13 GD +3 [[5]][[7]][[9]][[11]] 100Pts8576666557 PellegrinoAnquela(caretaker)ValverdePaciênciaManzanoVázquez Replaced by Qualification for the Abel ResinoTito VilanovaMauricioPaco JémezJuan AntonioJavier AguirreVoroErnestoDomingosUnai EmeryLucas AlcarazGregorioFernandoQualification for the Qualification for the  Qualification or relegation 13 June 20124 June 201214 June 201218 June 201228 November20121 December20123 December201231 December201214 January201330 January20135 February 201311 February201318 February2013 appointment Champions League play-off roundChampions League group stageEuropa League group stage [[14]][[15]][[17]][[19]][[21]][[23]][[27]][[28]] Date of [b] [[8]][[6]][[10]][[12]][[25]] Pre-SeasonPre-SeasonPre-SeasonPre-Season20th12th12th20th12th17th19th20th18th Position intable [[a]] …<br>**----- End of picture text -----**<br>


Figure 2. Illustration of PDF collections in M3DOCVQA. We first collect the URLs of all supporting contexts (Wikipedia documents) of individual questions of MultimodalQA [55]. Then, we create PDF versions from their URLs by rendering them in a web browser. 

trieve relevant information to generate the final answer. The dataset consists of 2,441 questions spread across 3,368 PDF documents, totaling 41,005 pages. Each question is supported by evidence found in one or more documents, spanning multiple modalities such as text, images, and tables, capturing the complexity and diversity typical of real-world documents. In Table 2, we provide detailed statistics of M3DOCVQA. Additionally, we provide the training split, consisting of 24,162 Wikipedia PDFs. Although the documents in the training split were not utilized in our experiments, they offer future researchers the opportunity to explore even larger-scale retrieval tasks or use the documents for training models, further expanding the potential applications of M3DOCVQA. 

To create M3DOCVQA, we extend the question-answer pairs from a short-context VQA dataset to a more complex setting that includes 1) PDF documents and 2) open-domain contexts. Specifically, we use the question-answer pairs from the development split[1] of MultimodalQA [55], where models answer multi-hop questions based on short multimodal contexts ( _e.g_ ., short text passages, 1-2 images, a table) sourced from Wikipedia. We retrieved the URLs of all Wikipedia documents used as context in any of the MultimodalQA development split questions. Then we generated PDF versions of the Wikipedia pages by rendering them in a Chromium web browser [57], using the Playwright Python package [46]. These PDFs retain all vector graphics and metadata, ensuring zoom-in functionality and maintaining operational hyperlinks. In addition, no objects are split between different pages in the resulting PDFs. 

While both M3DOCVQA and MultimodalQA [55] share the goal of evaluating question answering given multimodal context, M3DOCVQA introduces a more demanding scenario by requiring models to retrieve relevant information from a large set of documents, as opposed to being provided with a short context. In MultimodalQA, 

> 1The test split of MultimodalQA [55] is unavailable, and previous works have used the development split for comparison. 

models are given short, curated context ( _e.g_ ., a paragraph from a Wikipedia document) that directly contains the information needed to answer the questions, simplifying the task to reasoning within the provided material. In contrast, M3DOCVQA presents an open-domain setting, where models must retrieve information from a diverse collection of 3,368 PDF documents before attempting to answer any question. This not only requires handling largescale document retrieval but also dealing with multi-modal content–text, images, and tables–distributed across multiple documents. This key distinction highlights M3DOCVQA’s ability to simulate real-world challenges, where the relevant data is often spread across multiple sources. Consequently, M3DOCVQA serves as a robust benchmark for retrieval-augmented generation tasks in document understanding, pushing the boundaries of models to deal with large-scale, multi-modal, and multi-document settings. 

## **3. M3DOCRAG: A New Baseline for Opendomain Document Understanding** 

As a useful starting point for M3DOCVQA, we propose **M3DOCRAG** , a baseline method based on multi-modal retrieval-augmented generation. As illustrated in Fig. 3, M3DOCRAG operates in three stages: (1) encoding document images into visual embeddings (Sec. 3.1), (2) retrieving relevant document pages (Sec. 3.2), and (3) generating answers to questions based on the retrieved pages (Sec. 3.3). Below, we explain the problem definition and the details of each stage. 

**Problem definition.** We define a corpus of documents as _C_ = _{D_ 1 _, D_ 2 _, . . . , DM }_ , where _M_ is the total number of documents, and each document _Di_ consists of a set of pages, _Pi_ , represented as RGB images. From the documents in _C_ , we construct a global set of page images _P_ = � _Mi_ =1 _[P][i]_[=] _[{][p]_[1] _[, p]_[2] _[, . . . , p][N][}]_[,][where][each] _[p][j]_[represents][an] individual page image, and _N_ is the total number of page 

6239 

**==> picture [389 x 176] intentionally omitted <==**

**----- Start of picture text -----**<br>
1) Document Embedding Visual embeddings of all pages  𝑃 [𝑁, 𝑛 [!] , 𝑑]<br>Corpus  𝐶<br>[𝑛 [!] , 𝑑]<br>Page embeddings of 1 [st]  doc ... ...<br>Convert toImages Visual Encoder(ColPali) Page embeddings of i [th]  doc ...... [𝑛[𝑛 [!][!] , 𝑑], 𝑑]<br>𝑀  documents ... [𝑛... [!] , 𝑑]<br>Page embeddings of M [th]  doc<br>(with  𝑁  total pages) [𝑛 [!] , 𝑑]<br>2) Page Retrieval 3) Question Answering<br>Visual embeddings  (in open-domain setting)  Text Query  𝑞<br>of all pages Faster search with Text Encoder<br>approximate indexing<br>(ColPali)<br>MaxSim Multimodal LM Answer 𝑎<br>... SRS Top- 𝐾  Pages  (𝑃!")<br>**----- End of picture text -----**<br>


Figure 3. Our M3DOCRAG framework (Sec. 3) consists of three stages: (1) document embedding (Sec. 3.1), (2) page retrieval (Sec. 3.2), and (3) question answering (Sec. 3.3). In **(1) document embedding** , we extract visual embedding (with ColPali) to represent each page from all PDF documents. In **(2) page retrieval** , we retrieve the top-K pages of high relevance (MaxSim scores) with text queries. In an open-domain setting, we create approximate page indices for faster search. In **(3) question answering** , we conduct visual question answering with multi-modal LM ( _e.g_ . Qwen2-VL) to obtain the final answer. 

images across all documents in _C_ ( _i.e_ ., _N_ =[�] _[M] i_ =1 _[|][P][i][|]_[).] The objective of M3DOCRAG is to accurately answer a given question _q_ using the multi-modal information available in the corpus of documents _C_ . First, we identify _PK[q]_[,] the top _K_ ( _≪ N_ ) pages that are most relevant to answering the query _q_ from the global page set _P_ . Then, we obtain the final answer with a question answering model that takes retrieved page images _PK[q]_[and query] _[ q]_[ as inputs. The problem] of question answering can be categorized into two settings with different document context sizes: 

_**Closed-domain question answering**_ – The query _q_ should be answerable from a given single document _Di_ . The retrieval model outputs the top _K_ relevant page images _PK[q]_[, from the page images] _[ P][i]_[of the document] _[ D][i]_[.] 

_**Open-domain question answering**_ – The query _q_ may require information from single or multiple documents within the entire document corpus _C_ . The retrieval model outputs the top _K_ relevant page images _PK[q]_[from the entire] set of page images _P_ . 

## **3.1. Document Embedding** 

In M3DOCRAG, both textual query _q_ and page images _P_ are projected into a shared multi-modal embedding space using ColPali [19]. ColPali is a multi-modal retrieval model based on a late interaction mechanism, which encodes the text and image inputs into unified vector representations and retrieves the top _K_ most relevant images. ColPali adopts both training objective and similarity scoring from ColBERT [30, 51], which utilizes a shared architecture to encode either textual or visual inputs. In our framework, each page _p ⊆ Pi_ of a document _Di_ is treated as a single image 

with fixed dimensions (width _×_ height). 

From an image of a page, we extract a dense visual embedding _E[p] ∈_ R _[n][v][×][d]_ , where _n[v]_ represents the number of visual tokens per page (which remains constant across all pages), and _d_ denotes the embedding dimension ( _e.g_ ., 128). For a textual query _q_ , we similarly obtain an embedding _E[q] ∈_ R _[n][q][×][d]_ , where _n[q]_ is the number of text tokens. 

For efficiency, we treat each page of a document independently. This allows us to flatten all pages in the document corpus _C_ into a single page-level embedding tensor: _E_[C] _∈_ R _[N][×][n][v][×][d]_ , where _N_ represents the total number of pages in the entire document corpus, _n[v]_ is the number of visual tokens per page, and _d_ is the embedding dimension. M3DOCRAG can flexibly adapt to different retrieval settings, such as a single-page document ( _N_ = 1), a single document with multiple pages ( _e.g_ . _N_ = 100), and a large corpus of multi-page documents ( _e.g_ . _N >_ 1 _,_ 000). 

## **3.2. Page Retrieval** 

The relevance between the query _q_ and the page _p_ is computed using the MaxSim score _s_ ( _q, p_ ): 

**==> picture [120 x 31] intentionally omitted <==**

where _·_ denotes the dot product, and _Ei,· ∈_ R _[d]_ denotes the _i_ -th row (vector) of the embedding matrix _E ∈_ R _[n][×][d]_ . We then identify _PK[q]_[,][the][top] _[K]_[(] _[≪][N]_[)][pages][that][are][most] relevant to answering the query _q_ ; _i.e_ . we search _K_ pages scoring highest _s_ ( _q, p_ ). That is, 

**==> picture [185 x 12] intentionally omitted <==**

6240 

**Approximate indexing for open-domain page retrieval.** Searching pages over in a large document corpus can be time-consuming and computationally expensive. When a faster search is desired, we create page indices offline by applying approximate nearest neighborhood search, based on Faiss [18, 27]. We use exact search for closed-domain page retrieval and employ inverted file index (IVF) [53, 68] (IVFFlat in Faiss) for an open-domain setting, which could reduce page retrieval latency from 20s/query to less than 2s/query when searching across 40K pages. See Sec. 5.3 for a detailed comparison of speed-accuracy tradeoffs across different indexing methods. 

## **3.3. Question Answering** 

We run visual question answering by giving the text query _q_ and retrieved page images _PK[q]_[to a multi-modal language] model to obtain the final answer. For this, we employ multimodal language models ( _e.g_ . Qwen2-VL [60]) that consist of a visual encoder Enc[Vis] and a language model LM. The visual encoder takes _K_ retrieved page images _PK[q]_[as inputs] and outputs visual embeddings (different from ColPali encoder’s outputs). The language model takes the visual embeddings and text embeddings of query _q_ as inputs and outputs the final answer _a_ in an autoregressive manner: 

**==> picture [104 x 13] intentionally omitted <==**

## **4. Experiment Setup** 

**Datasets.** We benchmark M3DOCRAG on three PDF document understanding datasets that represent different scenarios: (1) M3DOCVQA (Open-domain DocVQA); (2) MMLongBench-Doc [42] (Closed-domain DocVQA); (3) MP-DocVQA [58] (Closed-domain DocVQA). In M3DOCVQA, M3DOCRAG processes over 3,000 PDFs, totaling more than 40,000 pages. For MP-DocVQA, models handle a single PDF with up to 20 pages for each question. For MMLongBench-Doc, models handle a single PDF with up to 120 pages for each question. 

**Evaluation Metrics.** For M3DOCVQA, we follow the evaluation setup of MultimodalQA [55]. For MMLongBench-Doc [42] and MP-DocVQA [58], we follow their official evaluation setups. For M3DOCVQA, we evaluate answer accuracy with exact match (EM) and F1. For MMLongBench-Doc, we extract short answers with GPT4o [47] from the model outputs and report answer accuracy with generalized accuracy (based on a rule-based evaluation script covering different answer types) and F1 score. For MP-DocVQA, we report answer accuracy with ANLS [8] and page retrieval with accuracy (same as recall@1, as there is a single page annotation for each question) by submitting the generation results to the test server.[2] 

> 2https://rrc.cvc.uab.es/?ch=17&com=tasks 

**Models.** We mainly experiment with the ColPali v1 [19][3] retrieval model and various recent open source multi-modal LMs with _<_ 10B parameters, including Idefics 2 [34], Idefics 3 [33], InternVL 2 [12], and Qwen2-VL [60]. We also experiment with a text-based RAG pipeline by combining recent widely used text retrieval and language models: ColBERT v2 [51] and Llama 3.1 [38]. For reproducible evaluation, we use deterministic greedy decoding for answer generation. We compare these multi-modal and textbased RAG pipelines with recent top entries with comparable parameters ( _<_ 10B) reported on the leaderboards. 

**Other implementation details.** We use PyTorch [48, 49], Transformers [61], and FlashAttention-2 [14] libraries for running models. We use Tesseract [54] for OCR in text RAG baselines, following Ma et al. [42]. We use Faiss [18, 27] for document indexing. We use the pdf2image [6] library to convert each PDF page into an RGB image with a resolution of DPI=144. While all PDF pages in M3DOCVQA have the same size – 8.5 (width) _×_ 11 (height) in inches ( _i.e_ . US letter size) and 1224 (width) _×_ 1584 (height) in pixels, in MP-DocVQA and MMLongBench-Doc datasets, pages have slightly different sizes. To handle this, we resize page images to the most common image size within the dataset – 1700 (width) _×_ 2200 (height) for MP-DocVQA, and to the most common image size within each PDF document for MMLongBenchDoc. All experiments are conducted with a single H100 80GB GPU. We provide up to 4 pages as visual inputs to our multi-modal LMs, the maximum number of images we could fit in the single GPU. 

## **5. Results and Key Findings** 

In the following, we describe experiment results of M3DOCRAG and baselines in both open-domain (Sec. 5.1) and closed-domain settings (Sec. 5.2). Next, we provide ablation studies (Sec. 5.3) about different page indexing strategies, multi-modal LMs, and retrieval models. Lastly, we show a qualitative example (Sec. 5.4) where M3DOCRAG can tackle M3DOCVQA questions whose answer source exists in the visual modality. Please also see the appendix for additional qualitative examples. 

## **5.1. Open-domain DocVQA** 

**Multi-modal RAG outperforms text RAG, especially on non-text evidence sources.** Table 3 shows the evaluation results on M3DOCVQA. As a model needs to find relevant documents from 3,000+ PDFs for each question, we focus solely on RAG pipelines. We observe that our M3DOCRAG (ColPali + Qwen2-VL 7B) outperforms text RAG (ColBERT v2 + Llama 3.1 8B), across all different 

> 3https://huggingface.co/vidore/colpali 

6241 

Table 3. Open-domain DocVQA evaluation results on M3DOCVQA. The scores are based on F1, unless otherwise noted. 

|**Method**<br>**# Pages**|**Evidence Modalities**<br>Image<br>Table<br>Text|**Question Hops**<br>Single-hop<br>Multi-hop|**Overall**<br>EM<br>F1|
|---|---|---|---|
|_Text RAG (w/ ColBERT v2)_<br>Llama 3.1 8B<br>1<br>Llama 3.1 8B<br>2<br>Llama 3.1 8B<br>4|8.3<br>15.7<br>29.6<br>7.7<br>16.8<br>31.7<br>7.8<br>21.0<br>34.1|25.3<br>12.3<br>27.4<br>12.1<br>29.4<br>15.2|15.4<br>20.0<br>15.8<br>21.2<br>17.8<br>23.7|
|M3DOCRAG_(w/ ColPali)_||||
|Qwen2-VL 7B (Ours)<br>1|25.1<br>27.8<br>39.6|37.2<br>25.0|27.9<br>32.3|
|Qwen2-VL 7B (Ours)<br>2|**26.8**<br>**30.4**<br>**42.1**|41.0<br>25.2|29.9<br>34.6|
|Qwen2-VL 7B (Ours)<br>4|24.7<br>**30.4**<br>41.2|**43.2**<br>**26.6**|**31.4**<br>**36.5**|



Table 4. Closed-domain DocVQA evaluation results on MMLongBench-Doc. We report the generalized accuracy (ACC) across five evidence source modalities: text (TXT), layout (LAY), chart (CHA), table (TAB), and image (IMG), and three evidence locations: singlepage (SIN), cross-page (MUL), and unanswerable (UNA). The scores from non-RAG methods are from Ma et al. [42]. 

|**Method**<br>**# Pages**||**Evidence Modalities**||**Evidence Locations**||**Overall**|
|---|---|---|---|---|---|---|
|||TXT<br>LAY<br>CHA<br>TAB<br>IMG||SIN<br>MUL<br>UNA||ACC<br>F1|
|||_Text Pipeline_|||||
|_LMs_<br>ChatGLM-128k [5]<br>up to 120<br>Mistral-Instruct-v0.2 [26]<br>up to 120<br>_Text RAG_||23.4<br>12.7<br>9.7<br>10.2<br>12.2<br>19.9<br>13.4<br>10.2<br>10.1<br>11.0||18.8<br>11.5<br>18.1<br>16.9<br>11.3<br>24.1||16.3<br>14.9<br>16.4<br>13.8|
|ColBERT v2 + Llama 3.1<br>1<br>ColBERT v2 + Llama 3.1<br>4||20.1<br>14.8<br>12.7<br>17.4<br>7.4<br>23.7<br>17.7<br>14.9<br>**24.0**<br>11.9||21.8<br>7.8<br>**41.3**<br>25.7<br>12.2<br>38.1||21.0<br>16.1<br>**23.5**<br>19.7|
|||_Multi-modal Pipeline_|||||
|_Multi-modal LMs_<br>DeepSeek-VL-Chat [39]<br>up to 120<br>Idefcs2 [34]<br>up to 120<br>MiniCPM-Llama3-V2.5 [62,66]<br>up to 120<br>InternLM-XC2-4KHD [17]<br>up to 120<br>mPLUG-DocOwl 1.5 [23]<br>up to 120<br>Qwen-VL-Chat [4]<br>up to 120<br>Monkey-Chat [37]<br>up to 120<br>M3DOCRAG||7.2<br>6.5<br>1.6<br>5.2<br>7.6<br>9.0<br>10.6<br>4.8<br>4.1<br>8.7<br>11.9<br>10.8<br>5.1<br>5.9<br>12.2<br>9.9<br>14.3<br>7.7<br>6.3<br>13.0<br>8.2<br>8.4<br>2.0<br>3.4<br>9.9<br>5.5<br>9.0<br>5.4<br>2.2<br>6.9<br>6.8<br>7.2<br>3.6<br>6.7<br>9.4||5.2<br>7.0<br>**12.8**<br>7.7<br>7.2<br>5.0<br>9.5<br>9.5<br>4.5<br>12.6<br>7.6<br>9.6<br>7.4<br>6.4<br>6.2<br>5.2<br>7.1<br>6.2<br>6.6<br>6.2<br>6.2||7.4<br>5.4<br>7.0<br>6.8<br>8.5<br>8.6<br>10.3<br>9.8<br>6.9<br>6.3<br>6.1<br>5.4<br>6.2<br>5.6|
|ColPali + Idefcs2 (Ours)<br>1||10.9<br>11.1<br>6.0<br>7.7<br>15.7||15.4<br>7.2<br>8.1||11.2<br>11.0|
|ColPali + Qwen2-VL 7B (Ours)<br>1||25.7<br>21.0<br>18.5<br>16.4<br>19.7||30.4<br>10.6<br>5.8||18.8<br>20.1|
|ColPali + Qwen2-VL 7B (Ours)<br>4||**30.0**<br>**23.5**<br>**18.9**<br>20.1<br>**20.8**||**32.4**<br>**14.8**<br>5.8||21.0<br>**22.6**|



evidence modalities / question hops / # pages. The performance gap is especially big when the evidence involves images, underscoring that M3DOCRAG addresses the information loss over non-textual content by text-only pipelines. We also notice that providing more retrieved pages as context generally increases the performance of both text RAG and M3DOCRAG (using the top 4 pages yields higher performance than using only the top 1 or 2 pages). 

## **5.2. Closed-domain DocVQA** 

**Multi-modal RAG boosts long document understanding of MLMs.** In MMLongBench-Doc, the models must handle a long PDF document (up to 120 pages) for each question. Since many multi-modal LMs have limited context length, Ma et al. [42] employed a concatenation strategy 

that combines all screenshot pages into either 1 or 5 images and inputs these concatenated images to multi-modal LMs. Table 4 shows that M3DOCRAG with Idefics2 surpass Idefics2 without RAG, as well as all previous multimodal entries. In addition, M3DOCRAG with Qwen2-VL achieves the best scores in overall F1 and most evidence modality/page settings. This demonstrates the effectiveness of multi-modal retrieval over handling many pages by concatenating low-resolution images. As observed in M3DOCVQA experiments, we also notice that providing more retrieved pages as context generally increases the performance of both text RAG and M3DOCRAG (using the top 4 pages yields higher performance than using only the top 1 page). 

6242 

Table 5. Closed-domain DocVQA evaluation results on MPDocVQA. The RAG methods retrieve a single page to the downstream QA models. 

|**Method**|**Answer Accuracy**<br>ANLS|**Page Retrieval**<br>R@1|
|---|---|---|
|_Multi-modal LMs_|||
|Arctic-TILT 0.8B [10]|0.8122|50.79|
|GRAM [9]|0.8032|19.98|
|GRAM C-Former [9]|0.7812|19.98|
|ScreenAI 5B [3]|0.7711|77.88|
|_Text RAG_|||
|ColBERT v2 + Llama 3.1 8B|0.5603|75.33|
|M3DOCRAG|||
|ColPali + Qwen2-VL 7B (Ours)|**0.8444**|**81.05**|



**M3DOCRAG achieves the state-of-the-art performance in MP-DocVQA.** In MP-DocVQA, the models must handle a PDF document of up to 20 pages for each question. Table 5 presents the top-performing entries in the MP-DocVQA test split leaderboard, comparing text-based and multi-modal RAG pipelines. While the text RAG (ColBERT v2 + Llama 3.1) falls short compared to existing approaches, all multi-modal RAG pipelines outperform their text-based counterpart. Notably, the M3DOCRAG delivers the state-of-the-art results on MP-DocVQA. 

## **5.3. Additional Analysis of** M3DOCRAG 

**Different page indexing: speed and accuracy.** In Table 6, we analyze the speed and accuracy of M3DOCRAG pipeline with different document embedding indexing methods. While the naive indexing with exact search (FlatIP) is slow (21s per query), we find that using approximate indexing such as inverted file [53, 68] (IVFFlat) and product quantization [28] (IVFPQ) can retain most of the accuracy, while making the search significantly faster ( _<_ 2s per query). We use FlatIP+IVFFlat indexing by default, and users can choose appropriate indexing methods depending on their requirements. 

**Different QA models.** In Table 7, we compare four different QA models in the M3DOCRAG framework: Idefics2 8B [34], Idefics3 8B [33], InternVL2 8B [12], InternVL2.5 [13] and Qwen2-VL 7B [60]. The Qwen2-VL 7B model outperforms other MLMs in all three benchmarks. Thus, we use the model as the default MLM component for M3DOCRAG. 

**Different retrieval models.** In Table 8, we compare different text-only (ColBERTv2 [51]) and multi-modal (CLIP [50], DSE [41], VisRAG [65], and ColPali) retrieval models on M3DOCVQA. ColBERTv2 and ColPali use late-interaction [30] score calculation, while DSE and CLIP use dot-product scores. We find that ColPali achieves 

Table 6. Speed-accuracy tradeoff with different indexing strategies on M3DOCVQA. Backbones: ColPali + Qwen2-VL 7B. 

|**# Pages**<br>**Indexing**|**Latency (s) (**_↓_**)**<br>Retrieval<br>VQA|**Accuracy (**_↑_**)**<br>EM<br>F1|
|---|---|---|
|1<br>FlatIP|21.0<br>1.1|28.9<br>33.7|
|1<br>FlatIP + IVFFlat|1.8<br>1.1|27.9<br>32.3|
|1<br>FlatIP + IVFPQ|0.2<br>1.1|25.9<br>30.3|
|2<br>FlatIP + IVFFlat|1.8<br>2.4|29.9<br>34.6|
|2<br>FlatIP + IVFPQ|0.2<br>2.4|29.0<br>33.5|
|4<br>FlatIP + IVFFlat|1.8<br>4.8|31.4<br>36.5|
|4<br>FlatIP + IVFPQ|0.2<br>4.8|29.9<br>34.7|



Table 7. Comparison of different QA models within RAG pipelines, evaluated on M3DOCVQA. 

|**QA models**|**M3DOCVQA**<br>F1_↑_|
|---|---|
|M3DOCRAG_w/ ColPali_<br>Idefcs2 8B<br>Idefcs3 8B<br>InternVL 2 8B<br>InternVL 2.5 8B|27.8<br>31.8<br>30.9<br>32.1|
|Qwen2 VL 7B|**32.3**|
|Qwen2.5 VL 7B|29.1|
|_Text RAG w/ ColBERTv2_<br>Llama 3.1<br>InternVL 2.5 (text only)<br>Qwen2 VL 7B<br>Qwen2.5 VL 7B|18.8<br>17.5<br>19.5<br>20.9|



Table 8. Comparison of different retrieval models on M3DOCVQA. QA model=InternVL 2.5. Batch size=1. Precision=bfloat16. Measured on a single A6000 48GB GPU. 

|Retrievers|Embedding Time<br>(s/page)_↓_|Embed/Index Storage<br>(KB/page)_↓_|Accuracy<br>F1_↑_|
|---|---|---|---|
|_Text RAG w/ OCR_<br>ColBERTv2|2.21|60.6/118.7|17.5|
|M3DOCRAG<br>CLIP (ViT-L/14)<br>DSE (Qwen2-2B)<br>VisRAG<br>ColPali|**0.04**<br>0.15<br>0.37<br>0.08|**1.7/3.1**<br>3.2/6.2<br>4.7/9.2<br>227.8/530.4|23.8<br>26.6<br>24.7<br>**32.1**|



the best performance in M3DOCVQA, even outperforming DSE trained on Wikipedia document screenshots, showing the effectiveness of late-interaction approaches for multimodal document retrieval. Thus, we use ColPali as the default retrieval model for M3DOCRAG. Users should note that late-interaction approaches (ColBERTv2 and ColPali) requires more storage requirements than dot-product approaches (CLIP, DSE, and VisRAG), as they store multiple vectors instead of a single vector per document. 

6243 

**==> picture [208 x 22] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question: “SIE Bend Studio's 2019 game cover has man leaning on what?”<br>ColPali + Qwen2-VL 7B: “motorcycle”<br>**----- End of picture text -----**<br>


**==> picture [223 x 160] intentionally omitted <==**

**----- Start of picture text -----**<br>
Top 2 pages retrieved by ColPali<br>Create account Log in Create account Log in<br>Bend Studio 18 languages Days Gone 21 languages<br>ArticleFrom Wikipedia, the free encyclopedia _ Talk Read _ Edit View history Tools ArticleFrom Wikipedia, the free encyclopedia _ Talk Read ~ Edit View history Tools<br>(Redirected from  Bend StudioInc. Oregondeveloping  Gone PlayStation Studios History ) is an American . Since 2000, Bend Studio is a . Founded in 1992, the studio is best known for Bubsy 3D [ edit (formerly SIE Bend Studio ] . video game developer, the  Blank, Berlyn & Co., Inc. Syphon Filter ) first-party developer series, and  based in  and  Days Bend, Eidetic,  for FormerlyCompany typeBend Studio Blank, Berlyn & Co., Inc.(1992–1995)Eidetic, Inc. (1995–2000)Subsidiary heNDSTUDIC Days Gone by Entertainmentin April 2019. A  Days Gone the start of a pandemic that turned a portion of humanity intovicious zombie-like creatures. Former Deacon St. John discovers his wife Sarah, having beenassumed dead, may still be alive and goes on a quest to findBend Studio is set in  is a 2019 . The game was released for the  oO  and published by Windowspost-apocalypticaction-adventure port was released in May 2021.Sony InteractiveOregonoutlawvideo game-turned-drifter two years afterPlayStation 4 developed DAYSGON Days Gone :<br>Marc BlankBlank, Berlyn & Co. in 1992.and the product development director for Berlyn, an author of at Infocom before moving to approached by a California company after an employee hadused remembered that the company also developed games. Thatcompany was looking to release a "sound-oriented gamemachine for cars", for which Blank suggested a series ofsports gamesproject never went into production and Blank repurposed theidea for an resembling a TV broadcast. In 1992, he pitched the idea toBerlyn, wondering whether Accolade would be interested in such a title.A few months after the 1993 release of  Kind under the Blank, Berlyn & Co. name. Blank became the company. Mystery Capers released in November 1993 by StarCore, Newton., when Berlyn was on hiatus at Accolade, they began developing gamesCornerstone [[4][5]][[2]] American football and  The company's first games were the  Two further such games,  that would sound like radio broadcasts. The and Michael Berlyn, a software package by Infocom, andadventure games Dell Crossword Puzzles [[2][3]] Accolade video game with an ambiance founded Bend Studio as Blank had been a founder., had previously worked Dell Crossword Puzzles and Other Word [[2]] Bubsy in Claws Encounters of the Furred  Blank wasInfocomApple for the 's publishing label for thepuzzle video games, whileApple Newtonpresident of the new IndustryFoundedFoundersHeadquartersKey peopleProductsNumber ofemployeesParentWebsite . Both were Columbo's [[2]] co Video games1992; 32 years agoMarc BlankMichael BerlynBend, OregonChristopher Reese (director Bubsy 3DSyphon FilterDays Gone 150+PlayStation Studios(2000–present)bendstudio.com [[1]] Seen=  (2022) — ) , US studio her. The game is played from a which the player can explore an Players can use firearms, weaponshostile humans and cannibalistic creatures known asFreakers. A major game mechanic is Deacon's motorcycle,which is used as the player character's main mode oftransportation. Days Gone original property since development project for home consoles after spendingdecades working on spinoff games for handheld consoles. Thegame's development took approximately six years; BendStudio expanded nearly three-fold to support it. Major sourcesof inspiration for  Dead 2016delayed several times.Upon release, who criticized the game's mission design and technical issuesbut praised the graphics, artificial intelligence, and ; its release was originally planned for 2018 but was and , and can use  Sons of Anarchy  was Bend Studio's first open-world project, its first Days GoneDays GoneSyphon Filter stealth received mixed reviews from critics,melee weapons. The game was unveiled at  were  to defend themselves againstthird-person perspectiveopen world World War Z  (1999), and its first, and  environment., improvised The Walking Sam WitwerE3 in 's performance as Deacon, while the story Developer(s)Publisher(s)Director(s)Producer(s)Designer(s)Programmer(s)Artist(s)Writer(s)Composer(s)EnginePlatform(s)ReleaseGenre(s)Mode(s) Bend StudioSony InteractiveEntertainmentJohn GarvinJeff RossDarren YagerRon AllenJohn HoffmanDonald YatomiJohn GarvinNathan WhiteheadUnreal Engine 4PlayStation 4Windows PlayStation 4 April 26, 2019 Windows May 18, 2021Action-adventureSingle-player<br>**----- End of picture text -----**<br>


Figure 4. Qualitative example of M3DOCRAG on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The answer is only stored visually within the game logo, where a man is leaning on a motorcycle. Best viewed by zooming in for details. See additional examples in appendix. 

**Document as pixels vs. text.** We compare pixel-based and text-based representations of documents using the same MLM, InternVL 2.5, which can optionally take image input. The blue rows of Table 7 show that the pixel-based representation outperforms the text-based representation of documents. Table 8 also shows that text embedding (based on OCR) from PDFs is much slower than visual embedding due to additional costs incurred by the OCR model. 

## **5.4. Qualitative Examples** 

we provide qualitative examples of M3DOCRAG (ColPali + Qwen2-VL 7B)’s question answering results on several M3DOCVQA examples. In Fig. 4, the answer information is only visually stored within the game logo (‘man is leaning on a motorcycle’), and M3DOCRAG could find the information. Please see the appendix for additional qualitative examples where M3DOCRAG can tackle M3DOCVQA questions whose answer source exists in various modalities. 

## **6. Related Work** 

**Document visual question answering.** Mathew et al. [44] proposed document visual question answering (DocVQA) task, where a model extracts information from documents by treating them as images, as in generic visual question answering [1]. Most research on DocVQA focuses on handling a single-page document [23, 24, 31, 35, 43, 44, 56, 59, 64], and it has been now a common practice to include the single-page DocVQA [44] as a part of the image understanding eval- 

uation suite among recent MLMs [7, 12, 21, 33, 47, 60]. Several recent works propose DocVQA benchmarks with multi-page documents [15, 16, 32, 42, 58]. However, all previous DocVQA benchmarks have focused on handling questions in the context of a specific document, such as “What was the gross profit in the year 2009?”. While this is probably due to the limited context length of the backbone multi-modal LMs, this does not reflect real-world scenarios, where users often ask questions that require information across different pages/documents. We address the limitation by proposing M3DOCVQA that benchmarks open-domain document understanding capabilities from over 3,000 documents. 

**Retrieval-augmented generation.** Retrieval-augmented generation (RAG) [36] has emerged as a hybrid approach combining retrieval systems with generative models to improve the quality and relevance of generated content [20]. RAG has been widely studied for open-domain question answering [2, 22, 25, 29, 40, 67], where the community has well-established practices for text-based pipelines. A line of work in VQA studies RAG on visual questions that require world knowledge [11, 45, 52, 63], but their retrieval context is usually generic images and/or short text snippets and does not cover DocVQA settings. To the best of our knowledge, no prior work has explored RAG for multi-modal document understanding only with multi-modal models (instead of using OCR methods). Our M3DOCRAG tackles opendomain question answering over documents with complex multi-modal contexts. 

## **7. Conclusion** 

We introduce M3DOCVQA, the first benchmark that evaluates open-domain multi-modal document understanding capabilities. In contrast to previous DocVQA datasets that evaluate question answering within the context of single document, M3DOCVQA offers a challenging question answering task where the answers exist among 3,000+ PDF documents, totaling more than 40,000 pages, containing various modalities such as images, text, and tables. We also introduce M3DOCRAG, a multi-modal RAG baseline that flexibly accommodates various document contexts, question hops and evidence modalities. We benchmark state-of-the-art methods and M3DOCRAG baseline in three datasets: M3DOCVQA, MP-DocVQA, and MMLongBench-Doc. Existing methods struggle with open-domain document understanding in M3DOCVQA, and M3DOCRAG achieves the text-only RAG baseline, but there remains large room for future improvement. We hope our work encourages future advancements in multi-modal frameworks for document understanding, paving the way for more robust, scalable, and practical solutions in realworld applications. 

6244 

## **References** 

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In _ICCV_ , 2015. 8 

- [2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. 8 

- [3] Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor C˘arbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. ScreenAI: A Vision-Language Model for UI and Infographics Understanding, 2024. 7 

- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. _arXiv preprint_ , abs/2308.12966, 2023. 6 

- [5] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. _arXiv preprint_ , abs/2401.18058, 2024. 6 

- [6] Edouard Belval. pdf2image, 2017. 5 

- [7] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Boˇsnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3B VLM for transfer, 2024. 8 

- [8] Ali Furkan Biten, Andres Mafla, Lluis Gomez, Valveny C V Jawahar, and Dimosthenis Karatzas. Scene Text Visual Question Answering. In _ICCV_ , 2019. 5 

- [9] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. Gram: Global reasoning for multi-page vqa. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15598–15607, 2024. 7 

- [10] Łukasz Borchmann, Michał Pietruszka, Wojciech Ja´skowski, Dawid Jurkiewicz, Piotr Halama, Paweł J´oziak, Łukasz Garncarek, Paweł Liskowski, Karolina Szyndler, Andrzej Gretkowski, Julita Ołtusek, Gabriela Nowakowska, Artur Zawłocki, Łukasz Duhr, Paweł Dyda, and Michał Turski. Arctic-TILT. Business Document Understanding at SubBillion Scale, 2024. 7 

- [11] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. _arXiv preprint arXiv:2210.02928_ , 2022. 8 

- [12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng 

   - Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 5, 7, 8 

- [13] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. 7 

- [14] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International Conference on Learning Representations (ICLR)_ , 2024. 5 

- [15] Yihao Ding, Siwen Luo, Hyunsuk Chung, and Soyeon Caren Han. Pdfvqa: A new dataset for real-world vqa on pdf documents. In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , pages 585–601. Springer, 2023. 8 

- [16] Yihao Ding, Kaixuan Ren, Jiabin Huang, Siwen Luo, and Soyeon Caren Han. Mmvqa: A comprehensive dataset for investigating multipage multimodal information retrieval in pdf-based visual question answering. In _the 33rd International Joint Conference on Artificial Intelligence_ , 2024. 1, 2, 8 

- [17] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-Xcomposer24KHD: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv preprint_ , abs/2404.06512, 2024. 6 

- [18] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria Lomeli, Lucas Hosseini, and Herv´e J´egou. The faiss library, 2024. 5 

- [19] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. ColPali: Efficient Document Retrieval with Vision Language Models, 2024. 1, 4, 5 

- [20] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023. 8 

- [21] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 8 

- [22] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-Augmented Language Model Pre-Training. In _ICML_ , 2020. 8 

- [23] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding, 2024. 6, 8 

- [24] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified 

6245 

   - text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , page 4083–4091, New York, NY, USA, 2022. Association for Computing Machinery. 8 

- [25] Gautier Izacard and Edouard Grave. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. In _EACL_ , 2021. 8 

- [26] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 6 

- [27] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with gpus. _IEEE Transactions on Big Data_ , 7(3):535–547, 2021. 5 

- [28] Herve J´egou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 33(1):117– 128, 2011. 7 

- [29] Vladimir Karpukhin, O Barlas, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering. In _EMNLP_ , pages 6769–6781, 2020. 8 

- [30] Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. _SIGIR 2020 - Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 39–48, 2020. 4, 7 

- [31] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision (ECCV)_ , 2022. 8 

- [32] Jordy Van Landeghem, Rafał Powalski, Rub`en Tito, Dawid Jurkiewicz, Matthew Blaschko, Łukasz Borchmann, Micka¨el Coustaty, Sien Moens, Michał Pietruszka, Bertrand Ackaert, Tomasz Stanisławek, Paweł J´oziak, and Ernest Valveny. Document Understanding Dataset and Evaluation (DUDE). In _ICCV_ , 2023. 1, 2, 8 

- [33] Hugo Laurenc¸on, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions, 2024. 5, 7, 8 

- [34] Hugo Laurenc¸on, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 5, 6, 7 

- [35] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: screenshot parsing as pretraining for visual language understanding. In _Proceedings of the 40th International Conference on Machine Learning_ . JMLR.org, 2023. 8 

- [36] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen Tau Yih, Tim Rockt¨aschel, Sebas- 

tian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In _NeurIPS_ , 2020. 8 

- [37] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint_ , abs/2311.06607, 2023. 6 

- [38] Llama Team. The llama 3 herd of models, 2024. 5 

- [39] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. DeepSeek-VL: towards real-world vision-language understanding. _arXiv preprint_ , abs/2403.05525, 2024. 6 

- [40] Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. Sail: Search-augmented instruction learning, 2023. 8 

- [41] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying multimodal retrieval via document screenshot embedding. In _EMNLP_ , 2024. 7 

- [42] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. MMLongBench-Doc: Benchmarking long-context document understanding with visualizations. In _The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2024. 1, 2, 5, 6, 8 

- [43] Minesh Mathew, Viraj Bagal, Rub`en P´erez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. _2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pages 2582–2591, 2021. 8 

- [44] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 1, 2, 8 

- [45] Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha, Andr´e Araujo, and Vittorio Ferrari. Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 3090–3101, 2023. 8 

- [46] Microsoft. Playwright for python, 2021. 3 

- [47] OpenAI. Hello gpt-4o, 2024. 5, 8 

- [48] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chana, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In _NIPS Workshop_ , 2017. 5 

- [49] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. _Advances in Neural Information Processing Systems_ , 32 (NeurIPS), 2019. 5 

6246 

- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 7 

- [51] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. _NAACL 2022 - 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Proceedings of the Conference_ , pages 3715–3734, 2022. 4, 5, 7 

- [52] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge, 2022. 8 

- [53] Sivic and Zisserman. Video google: a text retrieval approach to object matching in videos. In _Proceedings Ninth IEEE International Conference on Computer Vision_ , pages 1470– 1477 vol.2, 2003. 2, 5, 7 

- [54] Ray Smith. An overview of the tesseract ocr engine. In _ICDAR_ , 2007. 5 

- [55] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question answering over text, tables and images. _arXiv preprint arXiv:2104.06039_ , 2021. 1, 3, 5 

- [56] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing, 2023. 8 

      - high-resolution images. _arXiv preprint_ , abs/2403.11703, 2024. 6 

   - [63] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-Augmented Multimodal Language Modeling. In _ICML_ , 2023. 8 

   - [64] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Mingshi Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Feiyan Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In _Conference on Empirical Methods in Natural Language Processing_ , 2023. 8 

   - [65] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. Visrag: Vision-based retrievalaugmented generation on multi-modality documents, 2025. 7 

   - [66] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. RLAIF-V: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. _arXiv preprint_ , abs/2405.17220, 2024. 6 

   - [67] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua. Retrieving and reading: A comprehensive survey on open-domain question answering. _arXiv preprint arXiv:2101.00774_ , 2021. 8 

   - [68] Justin Zobel and Alistair Moffat. Inverted files for text search engines. _ACM Comput. Surv._ , 38(2):6–es, 2006. 2, 5, 7 

- [57] The Chromium Project Authors. The chromium projects, 2024. 3 

- [58] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834, 2023. 1, 2, 5, 8 

- [59] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. DocLLM: A layout-aware generative language model for multimodal document understanding, 2023. 8 

- [60] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 1, 5, 7, 8 

- [61] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, and Jamie Brew. HuggingFace’s Transformers: State-of-the-art Natural Language Processing. In _EMNLP_ , 2020. 5 

- [62] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: An LMM perceiving any aspect ratio and 

6247 

