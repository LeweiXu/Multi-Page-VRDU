# **M3DOCRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding** 

Jaemin Cho[1][*] Debanjan Mahata[2] Ozan[˙] Irsoy[2] Yujie He[2] Mohit Bansal[1] 

1UNC Chapel Hill 2Bloomberg 

_{_ jmincho,mbansal _}_ @cs.unc.edu 

_{_ dmahata,oirsoy,yhe247 _}_ @bloomberg.net 

## **Abstract** 

## **1. Introduction and Background** 

_Document visual question answering (DocVQA) pipelines that answer questions from documents have broad applications. Existing methods focus on handling single-page documents with multi-modal language models (MLMs), or rely on text-based retrieval-augmented generation (RAG) that uses text extraction tools such as optical character recognition (OCR). However, there are difficulties in applying these methods in real-world scenarios: (a) questions often require information across different pages or documents, where MLMs cannot handle many long documents; (b) documents often have important information in visual elements such as figures, but text extraction tools ignore them. We introduce_ M3DOCRAG _, a novel multi-modal RAG framework that flexibly accommodates various document contexts (closed-domain and open-domain), question hops (singlehop and multi-hop), and evidence modalities (text, chart, figure,_ etc _.)._ M3DOCRAG _finds relevant documents and answers questions using a multi-modal retriever and an MLM, so that it can efficiently handle single or many documents while preserving visual information. Since previous DocVQA datasets ask questions in the context of a specific document, we also present_ M3DOCVQA _, a new benchmark for evaluating open-domain DocVQA over 3,000+ PDF documents with 40,000+ pages. In three benchmarks (_ M3DOCVQA _/MMLongBench-Doc/MP-DocVQA), empirical results show that_ M3DOCRAG _with ColPali and Qwen2-VL 7B achieves superior performance than many strong baselines, including state-of-the-art performance in MP-DocVQA. We provide comprehensive analyses of different indexing, MLMs, and retrieval models. Lastly, we qualitatively show that_ M3DOCRAG _can successfully handle various scenarios, such as when relevant information exists across multiple pages and when answer evidence only exists in images._ 

> * _Work done during an internship at Bloomberg as a recipient of the Bloomberg Data Science Ph.D. Fellowship._ 

Document visual question answering (DocVQA) [14, 31, 40, 42, 57] is a multi-modal task that answers textual questions by interpreting information contained within document images. Existing methods on DocVQA either focus on visual question answering (VQA) on a single-page document (Fig. 1 (a)) or extract text from documents ( _e.g_ ., via optical character recognition (OCR) [43, 53] or PDF text extraction [18, 49]) and use retrieval-augmented generation (RAG) [35], where a retrieval model finds relevant paragraphs and a language model answers questions given the paragraphs (Fig. 1 (b)). However, there are difficulties in applying these methods in real-world document understanding scenarios: (a) questions often require information across different pages or documents, where existing VQA methods cannot handle many long documents; (b) some documents feature complex visual formats such as tables, charts, and mixed layouts, but text extraction methods such as OCR ignore these nuances, leading to incomplete or inaccurate document interpretations. Accurately and efficiently answering questions across numerous, lengthy documents with intricate layouts would greatly benefit many domains such as finance, healthcare, and law, where document AI assistants can streamline the daily processing of large volumes of documents, improving productivity and enabling faster, more informed decision-making. 

To overcome these limitations of existing DocVQA approaches, we introduce **M3DOCRAG** ( **M** ulti-modal **M** ultipage **M** ulti- **Doc** ument **R** etrieval- **A** ugmented **G** eneration; Sec. 2), a novel multi-modal RAG framework that flexibly accommodates various document contexts (closed-domain and open-domain), question hops (single-hop and multihop), and evidence modalities (text, chart, figure, _etc_ .). As illustrated in Fig. 1 (c), the M3DOCRAG framework retrieves relevant document pages using a multi-modal retrieval model, such as ColPali [17], and generates answers to questions from the retrieved pages using a multimodal language model (MLM), such as Qwen2-VL [59]. M3DOCRAG operates in three stages: In (1) document 

1 

**==> picture [500 x 246] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Single-page DocVQA - Can’t handle many/long documents 😥<br>Text Query<br>Multi-modal LM Answer<br>5<br>)<br>Single-page Document<br>(b) Text-based RAG<br>- Ignore visual information 😥<br>Text Query<br>Text Extraction  Text<br>— (e.g., OCR) … Retrieval LM Answer<br>CG es 2 eo<br>Many Long Documents Extracted Text Few Relevant Paragraphs<br>(c) M3DocRAG (Ours) - Can handle many/long documents- Preserve visual information 🤗<br>Text Query<br>Multi-modal Multi-modal<br>Answer<br>Retrieval LM<br>Pd} — o o<br>Many Long Documents Few Relevant Pages<br>**----- End of picture text -----**<br>


Figure 1. Comparison of multi-modal document understanding pipelines. Previous works focus on **(a) Single-page DocVQA** that cannot handle many long documents or **(b) Text-based RAG** that ignores visual information. Our **(c) M3DOCRAG** framework retrieves relevant documents and answers questions using multi-modal retrieval and MLM components, so that it can efficiently handle many long documents while preserving visual information. 

**==> picture [370 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
Existing DocVQA datasets: Closed-domain M3DocVQA (Ours): Open-domain<br>Context: Single PDF Context: 3K PDFs<br>Model Answer Model Answer<br>= o Bho.<br>Context-specific question Open-domain question<br>“What was the gross profit in the year 2009?” “Which B.Piazza title came earlier: the movie S. Stallone’s son<br>starred in or the movie with half of a lady’s face on the poster?<br>— Co<br>**----- End of picture text -----**<br>


Figure 2. Comparison of existing DocVQA datasets (left; _e.g_ ., DocVQA [42]) and our **M3DOCVQA** dataset (right). In contrast to previous DocVQA datasets that have questions that are specific to a single provided PDF ( _e.g_ ., “What was the gross profit in the year 2009?”), M3DOCVQA has information-seeking questions that benchmark open-domain question answering capabilities across more than 3,000 PDF documents ( _i.e_ ., 40,000+ pages). 

embedding (Sec. 2.1), we convert all document pages into RGB images and extract visual embeddings ( _e.g_ ., via ColPali) from the page images. In (2) page retrieval (Sec. 2.2), we retrieve the top-K pages of high similarity with text queries ( _e.g_ ., MaxSim operator for ColPali). For the opendomain setting, we create approximate page indices, such as inverted file index (IVF) [52, 66], for faster search. In (3) question answering (Sec. 2.3), we conduct visual question answering with MLM to obtain the final answer. Please also see Fig. 3 for the detailed illustration of the framework. M3DOCRAG can flexibly handle DocVQA in both closed domain ( _i.e_ ., a single document) and open-domain ( _i.e_ ., a large corpus of documents) settings. 

While M3DOCRAG framework supports DocVQA in an open-domain setting, the existing DocVQA datasets are not adequate for this setting, since their questions are in the context of a specific document, such as “What was the gross profit in the year 2009?” [14, 40, 42, 57], as illustrated in Fig. 2 (left). Hence, we also introduce **M3DOCVQA** ( **M** ulti-modal **M** ulti-page **M** ulti- **Doc** ument **V** isual **Q** uestion **A** nswering), an open-domain dataset that significantly raises the challenge of DocVQA to answering questions from a large document corpus (Sec. 3). By extending the MultimodalQA dataset’s [54] closed-domain context to an open-domain setting, M3DOCVQA introduces 2,441 multi-hop questions spanning 3,368 PDF doc- 

2 

**==> picture [423 x 192] intentionally omitted <==**

**----- Start of picture text -----**<br>
1) Document Embedding Visual embeddings of all pages  𝑃 [𝑁, 𝑛 [!] , 𝑑]<br>Corpus  𝐶<br>[𝑛 [!] , 𝑑]<br>Page embeddings of 1 [st]  doc ... ...<br>Convert toImages Visual Encoder(ColPali) Page embeddings of i [th]  doc ... [𝑛[𝑛 [!][!] , 𝑑], 𝑑]<br>...<br>𝑀  documents ... [𝑛... [!] , 𝑑]<br>Page embeddings of M [th]  doc<br>(with  𝑁  total pages) [𝑛 [!] , 𝑑]<br>2) Page Retrieval 3) Question Answering<br>Visual embeddings  (in open-domain setting)  Text Query  𝑞<br>of all pages Faster search with Text Encoder<br>approximate indexing<br>(ColPali)<br>MaxSim Multimodal LM Answer 𝑎<br>=  SO S<br>... Top- 𝐾  Pages  (𝑃!")<br>**----- End of picture text -----**<br>


Figure 3. Our M3DOCRAG framework (Sec. 2) consists of three stages: (1) document embedding (Sec. 2.1), (2) page retrieval (Sec. 2.2), and (3) question answering (Sec. 2.3). In **(1) document embedding** , we extract visual embedding (with ColPali) to represent each page from all PDF documents. In **(2) page retrieval** , we retrieve the top-K pages of high relevance (MaxSim scores) with text queries. In an open-domain setting, we create approximate page indices for faster search. In **(3) question answering** , we conduct visual question answering with multi-modal LM ( _e.g_ . Qwen2-VL) to obtain the final answer. 

uments, which collectively contain over 41,005 pages of diverse multi-modal content, including text, images, and tables. This dataset presents real-world challenges by requiring models to navigate complex reasoning paths across pages and within various types of document elements, better reflecting the intricacies of document understanding. 

To demonstrate the effectiveness of M3DOCRAG, we compare M3DOCRAG with state-of-the-art baselines in three benchmarks: M3DOCVQA, MMLongBenchDoc [40], and MP-DocVQA [57], which cover both opendomain (Sec. 5.1) and closed-domain (Sec. 5.2) DocVQA settings. Experiment results show that M3DOCRAG with ColPali and Qwen2-VL 8B achieves superior performance than many strong baselines, including the state-of-the-art performance in MP-DocVQA. We also provide a comprehensive analysis (Sec. 5.3) about different indexing, MLMs, and retrieval components. Finally, we show qualitative examples (Sec. 5.4) where M3DOCRAG can successfully handle various scenarios, such as when the relevant information exists across multiple pages and when answer evidence only exists in images. Overall, M3DOCRAG is an effective, efficient, and flexible framework for answering questions from multi-modal documents in various settings. 

## **2. M3DOCRAG: A Unified Framework for Multi-modal, Multi-page, Multi-document Understanding** 

We propose **M3DOCRAG** , a novel multi-modal RAG framework that flexibly accommodates various document 

contexts (closed-domain and open-domain), question hops (single-hop and multi-hop), and evidence modalities (text, chart, figure, _etc_ .). As illustrated in Fig. 3, M3DOCRAG operates in three stages: (1) encoding document images into visual embeddings (Sec. 2.1), (2) retrieving relevant document pages (Sec. 2.2), and (3) generating answers to questions based on the retrieved pages (Sec. 2.3). Below, we explain the problem definition and the details of each stage. 

**Problem definition.** We define a corpus of documents as _C_ = _{D_ 1 _, D_ 2 _, . . . , DM }_ , where _M_ is the total number of documents, and each document _Di_ consists of a set of pages, _Pi_ , represented as RGB images. From the documents in _C_ , we construct a global set of page images _P_ = _Mi_ =1 _[P][i]_[=] _[{][p]_[1] _[, p]_[2] _[, . . . , p][N][}]_[,][where][each] _[p][j]_[represents][an] individual page image, and _N_ is the total number of page _M_ images across all documents in _C_ ( _i.e_ ., _N_ = _i_ =1 _[|][P][i][|]_[).] The objective of M3DOCRAG is to accurately answer a given question _q_ using the multi-modal information available in the corpus of documents _C_ . First, we identify _PK[q]_[,] the top _K_ ( _≪ N_ ) pages that are most relevant to answering the query _q_ from the global page set _P_ . Then, we obtain the final answer with a question answering model that takes retrieved page images _PK[q]_[and query] _[ q]_[ as inputs. The problem] of question answering can be categorized into two settings with different document context sizes: 

_**Closed-domain question answering**_ – The query _q_ should be answerable from a given single document _Di_ . The retrieval model outputs the top _K_ relevant page images 

3 

_PK[q]_[, from the page images] _[ P][i]_[of the document] _[ D][i]_[.] 

_**Open-domain question answering**_ – The query _q_ may require information from single or multiple documents within the entire document corpus _C_ . The retrieval model outputs the top _K_ relevant page images _PK[q]_[from the entire] set of page images _P_ . 

## **2.1. Document Embedding** 

In M3DOCRAG, both textual query _q_ and page images _P_ are projected into a shared multi-modal embedding space using ColPali [17]. ColPali is a multi-modal retrieval model based on a late interaction mechanism, which encodes the text and image inputs into unified vector representations and retrieves the top _K_ most relevant images. ColPali adopts both training objective and similarity scoring from ColBERT [29, 50], which utilizes a shared architecture to encode either textual or visual inputs. In our framework, each page _p ⊆ Pi_ of a document _Di_ is treated as a single image with fixed dimensions (width _×_ height). 

From an image of a page, we extract a dense visual embedding _E[p] ∈_ R _[n][v][×][d]_ , where _n[v]_ represents the number of visual tokens per page (which remains constant across all pages), and _d_ denotes the embedding dimension ( _e.g_ ., 128). For a textual query _q_ , we similarly obtain an embedding _E[q] ∈_ R _[n][q][×][d]_ , where _n[q]_ is the number of text tokens. 

For efficiency, we treat each page of a document independently. This allows us to flatten all pages in the document corpus _C_ into a single page-level embedding tensor: _E_[C] _∈_ R _[N][×][n][v][×][d]_ , where _N_ represents the total number of pages in the entire document corpus, _n[v]_ is the number of visual tokens per page, and _d_ is the embedding dimension. M3DOCRAG can flexibly adapt to different retrieval settings, such as a single-page document ( _N_ = 1), a single document with multiple pages ( _e.g_ . _N_ = 100), and a large corpus of multi-page documents ( _e.g_ . _N >_ 1 _,_ 000). 

## **2.2. Page Retrieval** 

The relevance between the query _q_ and the page _p_ is computed using the MaxSim score _s_ ( _q, p_ ): 

**==> picture [119 x 31] intentionally omitted <==**

where _·_ denotes the dot product, and _Ei,· ∈_ R _[d]_ denotes the _i_ -th row (vector) of the embedding matrix _E ∈_ R _[n][×][d]_ . We then identify _PK[q]_[,][the][top] _[K]_[(] _[≪][N]_[)][pages][that][are][most] relevant to answering the query _q_ ; _i.e_ . we search _K_ pages scoring highest _s_ ( _q, p_ ). That is, 

**==> picture [191 x 12] intentionally omitted <==**

**Approximate indexing for open-domain page retrieval.** Searching pages over in a large document corpus can be 

time-consuming and computationally expensive. When a faster search is desired, we create page indices offline by applying approximate nearest neighborhood search, based on Faiss [16, 26]. We use exact search for closed-domain page retrieval and employ inverted file index (IVF) [52, 66] (IVFFlat in Faiss) for an open-domain setting, which could reduce page retrieval latency from 20s/query to less than 2s/query when searching across 40K pages. See Sec. 5.3 for a detailed comparison of speed-accuracy tradeoffs across different indexing methods. 

## **2.3. Question Answering** 

We run visual question answering by giving the text query _q_ and retrieved page images _PK[q]_[to a multi-modal language] model to obtain the final answer. For this, we employ multimodal language models ( _e.g_ . Qwen2-VL [59]) that consist of a visual encoder Enc[Vis] and a language model LM. The visual encoder takes _K_ -retrieved page images _PK[q]_[as inputs] and outputs visual embeddings (different from ColPali encoder’s outputs). The language model takes the visual embeddings and text embeddings of query _q_ as inputs and outputs the final answer _a_ in the autoregressive manner: 

**==> picture [104 x 13] intentionally omitted <==**

## **3. M3DOCVQA: A New Benchmark for Opendomain Document Understanding** 

We present **M3DOCVQA** ( **M** ulti-modal **M** ulti-page **M** ulti- **Doc** ument **V** isual **Q** uestion **A** nswering), a new opendomain DocVQA benchmark designed to evaluate the ability to answer questions using multi-modal information from a large corpus of documents. 

As illustrated in Fig. 2, existing DocVQA datasets [31, 40, 42, 57] primarily focus on evaluating question answering within the context of a single document ( _i.e_ ., closeddomain). These datasets are not well-suited for benchmarking open-domain visual question answering, where relevant information, often in multiple modalities such as text, images, and tables, must be retrieved from multiple documents. This limitation stems from their questions being designed around specific content on certain pages within a single document. In real-world scenarios, users often seek answers that span across multiple documents and modalities, making open-domain settings critical. However, the questions in the existing DocVQA datasets are not applicable in such an open-domain setting. For example, a question from MP-DocVQA, such as _“What was the gross profit in the year 2009?”_ assumes that the model already has access to specific information within the document. 

M3DOCVQA challenges models in an open-domain DocVQA setting, where they must navigate a large ‘haystack’ of multi-modal documents and retrieve relevant 

4 

**==> picture [485 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
MultimodalQA (Talmor et al., 2021) Our PDFs in M3DocVQA<br>“Question”: “…”,“Question”: “…”,“Answer”: “…”“Supporting Contexts”: [  {https-13_La_Liga  },... ://en.wikipedia.org/wiki/2012“text”: “…”,“title”: “2012-13 La Liga”,“url”:  ... 1.2.3. Obtain URLs of supporting contextsRender ina web browserCreate PDFs ArticleFrom Wikipedia, the free encyclopediaThe 82nd since its establishment. The campaign began on 18 August 2012, and ended on 1 June2013.and amassing 100 points, equalling in previous years, BBVA model to be used throughout the season for all matches.TeamsA total of 20 teams contested the league, including 17 sides from the three promoted from the Segunda División, and the victorious team of the play-offs.Villarreal CFSegunda DivisiónSporting de Gijón returned to Segunda División after a four-year tenure in La Liga, whileRacing de Santander ended ten consecutive seasons in La Liga, the longest period in itshistory.The three teams that were relegated were replaced by three sides: División championafter a five-year absence. The third promoted team was decided in the where  Stadia and locations Athletic BilbaoAtlético MadridBarcelonaBetisCelta VigoDeportivo La CoruñaEspanyolGetafeGranadaLevanteMálagaMallorca 2012–13  2012–13 La Li [[2]] Deportivo de La CoruñaReal Valladolid Team TalkBarcelona[,  editSporting de Gijón La Liga  ]  the previous season: Villarreal were relegated after twelve years in Nike. The second-placing team  won the league for a 22nd time, after leading the league the entire season returned to La Liga after two seasons in Segunda División. season (known as the  provided the official ball for all matches, with a new Nike Maxim Liga Location of stadium BilbaoMadridBarcelonaSevilleVigoA CoruñaBarcelonaGetafeGranadaValenciaMálagaPalma2011–12 Segunda División[ edit made an immediate return to the top level as  and g ]a Racing de SantanderReal Madrid Liga BBVA Celta de Vigo's points record from the San MamésVicente CalderónCamp NouBenito VillamarínBalaídosRiazorCornellà-El PratColiseum Alfonso PérezNuevo Los CármenesCiutat de ValènciaLa RosaledaIberostar Stadium. This included the two top teams from the for sponsorship reasons) was the Stadium  were relegated to  was also promoted to La Liga2011–12 Segunda División [[3][4]] 2011–12 seasonpromotion play-offsprevious season2012–13 Capacity Segunda39,75054,85199,35452,74531,80034,60040,50017,70022,52425,53428,96323,142La Liga and. As, SeasonDatesChampionsRelegatedChampionsLeagueEuropa LeagueMatchesplayedGoals scoredTop goalscorerBestgoalkeeperBiggest homewinBiggest awaywinHighestscoringLongestwinning runLongestunbeaten runLongestwinless runLongest losingrunHighestattendance Read 2012–1318 August 2012 – 1 June2013Barcelona22nd titleMallorcaDeportivo La CoruñaZaragozaBarcelonaReal MadridAtlético MadridReal SociedadValenciaReal BetisSevilla3801,091 (2.87 per match)Lionel Messi(46 goals)Thibaut Courtois(0.78 goals/match)Atlético MadridDeportivo La Coruña(9 December 2012)Rayo VallecanoBarcelona(27 October 2012)Mallorca(28 October 2012)Valencia(20 January 2013)Deportivo La CoruñaBarcelona12 matchesBarcelona19 matchesBarcelona15 matchesZaragoza6 matchesDeportivo La CoruñaMallorca96,589BarcelonaEdit La Liga Create account [[1]] View history 0–5  0–5  [[1]][[1]]  (20 October 2012) [[1]][[1]]  2–2  38 languages Real MadridReal Madrid 6–0Real Madrid 0–5Log in 4–5Tools Personnel and sponsorship OsasunaRayo VallecanoReal MadridReal SociedadSevillaValenciaValladolidZaragozaAthletic BilbaoAtlético MadridBarcelonaBetisCelta de VigoDeportivo La CoruñaEspanyolGetafeGranadaLevanteMálagaMallorcaOsasunaRayo VallecanoReal MadridReal SociedadSevillaValenciaValladolidZaragoza1.  ^Team Huawei is the sponsor for select matches.PamplonaMadridMadridSan SebastiánSevilleValenciaValladolidZaragozaTito VilanovaMarcelo BielsaDiego SimeonePepe MelPaco HerreraFernando VázquezJavier AguirreLuis García PlazaLucas AlcarazJuan Ignacio MartínezManuel PellegriniGregorio ManzanoJosé Luis MendilibarPaco JémezJosé MourinhoPhilippe MontanierUnai EmeryErnesto ValverdeMiroslav ĐukićManolo Jiménez Head Coach [ editEl SadarCampo de VallecasSantiago BernabéuAnoetaRamón SánchezPizjuánMestallaJosé ZorrillaLa Romareda ] Andrés PalopCarlos GurpeguiGabiCarles PuyolJuanmaBorja OubiñaManuel PabloCristian ÁlvarezJaime GavilánManuel LucenaSergio BallesterosJesús GámezJosé NunesPatxi PuñalPitiIker CasillasXabi PrietoDavid AlbeldaJavier BarajaJavier Paredes Captain 19,55315,48985,45432,07645,50055,00026,51234,596 manufacturer UmbroNikeNikeMacronLi-NingLottoPumaJomaLuanviKelmeNikeMacronAstoreErreàAdidasNikeUmbroJomaKappaMercuryLocation of teams in Celta VigoDeportivo LaCoruña Kit Atlético MadridSevillaBetisPetronorAzerbaijanQatar FoundationCirsa and CitroënEstrella GaliciaCancúnConfremar and Caja GranadaComunitat ValencianaUNESCORiviera MayaLacturale and NevirAE — Adquisiciones Empresariales and NevirBWINCanal+InterwettenJinKO SolarEl Norte de CastillaProniñoValladolid Averageattendance2012–13 La Liga Málaga2011–12Real MadridGranadaGetafeAthletic Bilbao [6][4]  and Canal+ and  and  [5] RayoVallecanoAndalucía, HuaweiKutxaEstrella GaliciaOsasuna Shirt sponsor IG MarketsReal SociedadZaragoza29,430, UNICEF [2][4][1][2] Valencia [6][4]  and  [[1]] LevanteEspanyolKyocera [4][2][2][3]  and  [4] Mallorca2013–14Barcelona [2] TV3 [6] [2] Managerial changes League tableBarcelonaValenciaRayoVallecanoGranadaEspanyolValenciaValenciaDeportivo LaCoruñaSevillaGranadaMallorcaDeportivo LaCoruñaCelta de Vigo Pos 1234562. 3. 4. 5. 6.  Team^^^^^ BarcelonaReal MadridAtlético MadridReal SociedadValenciaMálagaOn the back of shirt.Barcelona makes a donation to UNICEF in order to display the charity's logo on the back of the club's kit.On the shorts.Málaga makes a donation to UNESCO in order to display the charity's logo on the club's kit.On the left sleeve. Team(C) SandovalPochettinoPellegrino(caretaker)AnquelaCaparrósPaciência[ edit Outgoing Abel Resino manager Pep GuardiolaUnai EmeryJosé RamónMauricioMauricioVoroJosé Luis OltraMíchelJuan AntonioJoaquínDomingosPaco Herrera ] [ edit ] Pld 383838383838 End of contractEnd of contractEnd of contractEnd of contractMutual consentSackedEnd of tenure ascaretakerSackedSackedSackedSackedMutual consentSacked322623181916 WManner ofdeparture 12 D 47789 1311 L 2588 103115 GF 6570675330 June 201230 June 201230 June 201230 June 201226 November20121 December20125 December201230 December201214 January201330 January20134 February201311 February201318 February2013 vacancyDate ofGA 404231495450 [[13]][[15]][[16]][[18]][[20]][[22]][[24]][[26]][[28]] +75+61+34+21+13 GD +3 [[5]][[7]][[9]][[11]] 100Pts8576666557 PellegrinoAnquela(caretaker)ValverdePaciênciaManzanoVázquez Replaced by Qualification for the Abel ResinoTito VilanovaMauricioPaco JémezJuan AntonioJavier AguirreVoroErnestoDomingosUnai EmeryLucas AlcarazGregorioFernandoQualification for the Qualification for the  Qualification or relegation 13 June 20124 June 201214 June 201218 June 201228 November20121 December20123 December201231 December201214 January201330 January20135 February 201311 February201318 February2013 appointment Champions League play-off roundChampions League group stageEuropa League group stage [[14]][[15]][[17]][[19]][[21]][[23]][[27]][[28]] Date of [b] [[8]][[6]][[10]][[12]][[25]] Pre-SeasonPre-SeasonPre-SeasonPre-Season20th12th12th20th12th17th19th20th18th Position intable [[a]] …<br>**----- End of picture text -----**<br>


Figure 4. Illustration of PDF collections in M3DOCVQA. We first collect the URLs of all supporting contexts (Wikipedia documents) of individual questions of MultimodalQA [54]. Then, we create PDF versions from their URLs by rendering them in a web browser. 

information to generate the final answer. The dataset consists of 2,441 multi-hop questions spread across 3,368 PDF documents, totaling 41,005 pages. Each question is supported by evidence found in one or more documents, spanning multiple modalities such as text, images, and tables, capturing the complexity and diversity typical of real-world documents. Additionally, we provide the training split, consisting of 24,162 Wikipedia PDFs. Although the documents in the training split were not utilized in our experiments, they offer future researchers the opportunity to explore even larger-scale retrieval tasks or use the documents for training models, further expanding the potential applications of M3DOCVQA. 

To create M3DOCVQA, we extend the question-answer pairs from a short-context VQA dataset to a more complex setting that includes 1) PDF documents and 2) open-domain contexts. Specifically, we use the question-answer pairs from the development split[1] of MultimodalQA [54], where models answer multi-hop questions based on short multimodal contexts ( _e.g_ ., short text passages, 1-2 images, a table) sourced from Wikipedia. We retrieved the URLs of all Wikipedia documents used as context in any of the MultimodalQA development split questions. Then we generated PDF versions of the Wikipedia pages by rendering them in a Chromium web browser [56], using the Playwright Python package [45]. These PDFs retain all vector graphics and metadata, ensuring zoom-in functionality and maintaining operational hyperlinks. In addition, no objects are split between different pages in the resulting PDFs. 

While both M3DOCVQA and MultimodalQA [54] share the goal of evaluating question answering given multimodal context, M3DOCVQA introduces a more demanding scenario by requiring models to retrieve relevant information from a large set of documents, as opposed to being provided with a short context. In MultimodalQA, models are given short, curated context ( _e.g_ ., a paragraph 

> 1The test split of MultimodalQA [54] is unavailable, and previous works have used the development split for comparison. 

from a Wikipedia document) that directly contains the information needed to answer the questions, simplifying the task to reasoning within the provided material. In contrast, M3DOCVQA presents an open-domain setting, where models must retrieve information from a diverse collection of 3,368 PDF documents before attempting to answer any question. This not only requires handling largescale document retrieval but also dealing with multi-modal content–text, images, and tables–distributed across multiple documents. This key distinction highlights M3DOCVQA’s ability to simulate real-world challenges, where the relevant data is often spread across multiple sources. Consequently, M3DOCVQA serves as a robust benchmark for retrieval-augmented generation tasks in document understanding, pushing the boundaries of models to deal with large-scale, multi-modal, and multi-document settings. 

## **4. Experiment Setup** 

**Datasets.** We benchmark M3DOCRAG on three PDF document understanding datasets that represent different scenarios: (1) M3DOCVQA (Open-domain DocVQA); (2) MMLongBench-Doc [40] (Closed-domain DocVQA); (3) MP-DocVQA [57] (Closed-domain DocVQA). In M3DOCVQA, M3DOCRAG processes over 3,000 PDFs, totaling more than 40,000 pages. For MP-DocVQA, models handle a single PDF with up to 20 pages for each question. For MMLongBench-Doc, models handle a single PDF with up to 120 pages for each question. 

**Evaluation Metrics.** For M3DOCVQA, we follow the evaluation setup of MultimodalQA [54]. For MMLongBench-Doc [40] and MP-DocVQA [57], we follow their official evaluation setups. For M3DOCVQA, we evaluate answer accuracy with exact match (EM) and F1. For MMLongBench-Doc, we extract short answers with GPT4o [46] from the model outputs and report answer accuracy with generalized accuracy (based on a rule-based 

5 

Table 1. Open-domain DocVQA evaluation results on M3DOCVQA. The scores are based on F1, unless otherwise noted. Index: FlatIP + IVFFlat. 

|**Method**<br>**# Pages**|**Evidence Modalities**<br>Image<br>Table<br>Text|**Question Hops**<br>Single-hop<br>Multi-hop|**Overall**<br>EM<br>F1|
|---|---|---|---|
|_Text RAG (w/ ColBERT v2)_<br>Llama 3.1 8B<br>1<br>Llama 3.1 8B<br>2<br>Llama 3.1 8B<br>4|8.3<br>15.7<br>29.6<br>7.7<br>16.8<br>31.7<br>7.8<br>21.0<br>34.1|25.3<br>12.3<br>27.4<br>12.1<br>29.4<br>15.2|15.4<br>20.0<br>15.8<br>21.2<br>17.8<br>23.7|
|M3DOCRAG_(w/ ColPali)_||||
|Qwen2-VL 7B (Ours)<br>1|25.1<br>27.8<br>39.6|37.2<br>25.0|27.9<br>32.3|
|Qwen2-VL 7B (Ours)<br>2|**26.8**<br>**30.4**<br>**42.1**|41.0<br>25.2|29.9<br>34.6|
|Qwen2-VL 7B (Ours)<br>4|24.7<br>**30.4**<br>41.2|**43.2**<br>**26.6**|**31.4**<br>**36.5**|



evaluation script covering different answer types) and F1 score. For MP-DocVQA, we report answer accuracy with ANLS [8] and page retrieval with accuracy (same as recall@1, as there is a single page annotation for each question) by submitting the generation results to the test server.[2] 

**Models.** We mainly experiment with the ColPali v1 [17][3] retrieval model and various recent open source multi-modal LMs with _<_ 10B parameters, including Idefics 2 [33], Idefics 3 [32], InternVL 2 [12], and Qwen2-VL [59]. We also experiment with a text-based RAG pipeline by combining recent widely used text retrieval and language models: ColBERT v2 [50] and Llama 3.1 [37]. We also compare ColPali v1 with ColQwen v0.1 [17],[4] another recent multi-modal retrieval model that was trained with same objective/dataset as ColPali but initialized with Qwen2-VL 2B [59] backbone. For reproducible evaluation, we use deterministic greedy decoding for answer generation. We compare these multi-modal and text-based RAG pipelines with recent top entries with comparable parameters ( _<_ 10B) reported on the leaderboards. 

**Other implementation details.** We use PyTorch [47, 48], Transformers [60], and FlashAttention-2 [13] libraries for running models. We use Tesseract [53] for OCR in text RAG baselines, following Ma et al. [40]. We use Faiss [16, 26] for document indexing. We use the pdf2image [6] library to convert each PDF page into an RGB image with a resolution of DPI=144. While all PDF pages in M3DOCVQA have the same size – 8.5 (width) _×_ 11 (height) in inches ( _i.e_ . US letter size) and 1224 (width) _×_ 1584 (height) in pixels, in MP-DocVQA and MMLongBench-Doc datasets, pages have slightly different sizes. To handle this, we resize page images to the most common image size within the dataset – 1700 (width) _×_ 

> 2https://rrc.cvc.uab.es/?ch=17&com=tasks 

> 3https://huggingface.co/vidore/colpali 

> 4https://huggingface.co/vidore/colqwen2-v0.1 

2200 (height) for MP-DocVQA, and to the most common image size within each PDF document for MMLongBenchDoc. All experiments are conducted with a single H100 80GB GPU. We provide up to 4 pages as visual inputs to our multi-modal LMs, the maximum number of images we could fit in the single GPU. 

## **5. Results and Key Findings** 

In the following, we describe experiment results of M3DOCRAG and baselines in both open-domain (Sec. 5.1) and closed-domain settings (Sec. 5.2). Next, we provide ablation studies (Sec. 5.3) about different page indexing strategies and different multi-modal LMs and retrieval models. Lastly, we show qualitative examples (Sec. 5.4) where M3DOCRAG can tackle M3DOCVQA questions whose answer source exists in various modalities. 

## **5.1. Open-domain DocVQA** 

## **Multi-modal RAG outperforms text RAG, especially on** 

**non-text evidence sources.** Table 1 shows the evaluation results on M3DOCVQA. As a model needs to find relevant documents from 3,000+ PDFs for each question, we focus solely on RAG pipelines. We observe that our M3DOCRAG (ColPali + Qwen2-VL 7B) significantly outperforms text RAG (ColBERT v2 + Llama 3.1 8B), across all different evidence modalities / question hops / # pages. The performance gap is especially big when the evidence involves images, underscoring that M3DOCRAG addresses the information loss over non-textual content by text-only pipelines. We also notice that providing more retrieved pages as context generally increases the performance of both text RAG and M3DOCRAG (using the top 4 pages gives higher performance than the top 1 and 2 pages). 

## **5.2. Closed-domain DocVQA** 

**Multi-modal RAG boosts long document understanding of MLMs.** In MMLongBench-Doc, the models must handle a long PDF document (up to 120 pages) for each ques- 

6 

Table 2. Closed-domain DocVQA evaluation results on MMLongBench-Doc. We report the generalized accuracy (ACC) across five evidence source modalities: text (TXT), layout (LAY), chart (CHA), table (TAB), and image (IMG), and three evidence locations: singlepage (SIN), cross-page (MUL), and unanswerable (UNA). The scores from non-RAG methods are from Ma et al. [40]. 

|**Method**<br>**# Pages**|**Evidence Modalities**|**Evidence Locations**|**Overall**|
|---|---|---|---|
||TXT<br>LAY<br>CHA<br>TAB<br>IMG|SIN<br>MUL<br>UNA|ACC<br>F1|
||_Text Pipeline_|||
|_LMs_<br>ChatGLM-128k [5]<br>up to 120<br>Mistral-Instruct-v0.2 [25]<br>up to 120<br>_Text RAG_|23.4<br>12.7<br>9.7<br>10.2<br>12.2<br>19.9<br>13.4<br>10.2<br>10.1<br>11.0|18.8<br>11.5<br>18.1<br>16.9<br>11.3<br>24.1|16.3<br>14.9<br>16.4<br>13.8|
|ColBERT v2 + Llama 3.1<br>1<br>ColBERT v2 + Llama 3.1<br>4|20.1<br>14.8<br>12.7<br>17.4<br>7.4<br>23.7<br>17.7<br>14.9<br>**24.0**<br>11.9|21.8<br>7.8<br>**41.3**<br>25.7<br>12.2<br>38.1|21.0<br>16.1<br>**23.5**<br>19.7|
||_Multi-modal Pipeline_|||
|_Multi-modal LMs_<br>DeepSeek-VL-Chat [38]<br>up to 120<br>Idefcs2 [33]<br>up to 120<br>MiniCPM-Llama3-V2.5 [61,64]<br>up to 120<br>InternLM-XC2-4KHD [15]<br>up to 120<br>mPLUG-DocOwl 1.5 [22]<br>up to 120<br>Qwen-VL-Chat [4]<br>up to 120<br>Monkey-Chat [36]<br>up to 120<br>M3DOCRAG|7.2<br>6.5<br>1.6<br>5.2<br>7.6<br>9.0<br>10.6<br>4.8<br>4.1<br>8.7<br>11.9<br>10.8<br>5.1<br>5.9<br>12.2<br>9.9<br>14.3<br>7.7<br>6.3<br>13.0<br>8.2<br>8.4<br>2.0<br>3.4<br>9.9<br>5.5<br>9.0<br>5.4<br>2.2<br>6.9<br>6.8<br>7.2<br>3.6<br>6.7<br>9.4|5.2<br>7.0<br>**12.8**<br>7.7<br>7.2<br>5.0<br>9.5<br>9.5<br>4.5<br>12.6<br>7.6<br>9.6<br>7.4<br>6.4<br>6.2<br>5.2<br>7.1<br>6.2<br>6.6<br>6.2<br>6.2|7.4<br>5.4<br>7.0<br>6.8<br>8.5<br>8.6<br>10.3<br>9.8<br>6.9<br>6.3<br>6.1<br>5.4<br>6.2<br>5.6|
|ColPali + Idefcs2 (Ours)<br>1|10.9<br>11.1<br>6.0<br>7.7<br>15.7|15.4<br>7.2<br>8.1|11.2<br>11.0|
|ColPali + Qwen2-VL 7B (Ours)<br>1|25.7<br>21.0<br>18.5<br>16.4<br>19.7|30.4<br>10.6<br>5.8|18.8<br>20.1|
|ColPali + Qwen2-VL 7B (Ours)<br>4|**30.0**<br>**23.5**<br>**18.9**<br>20.1<br>**20.8**|**32.4**<br>**14.8**<br>5.8|21.0<br>**22.6**|



tion. Since many multi-modal LMs have limited context length, Ma et al. [40] employed a concatenation strategy that combines all screenshot pages into either 1 or 5 images and inputs these concatenated images to multi-modal LMs. Table 2 shows that ColPali + Idefics2 surpass Idefics2 without RAG, as well as all previous multi-modal entries. In addition, ColPali + Qwen2-VL 7B achieves the best scores in overall F1 and most evidence modality/page settings. This demonstrates the effectiveness of multi-modal retrieval over handling many pages by concatenating low-resolution images. As observed in M3DOCVQA experiments, we also notice that providing more retrieved pages as context generally increases the performance of both text RAG and M3DOCRAG (using the top 4 pages gives higher performance than the top 1 page). 

**M3DOCRAG achieves the state-of-the-art performance in MP-DocVQA.** In MP-DocVQA, the models must handle a PDF document of up to 20 pages for each question. Table 3 presents the top-performing entries in the MP-DocVQA test split leaderboard, comparing text-based and multi-modal RAG pipelines. While the text RAG (ColBERT v2 + Llama 3.1) falls short compared to existing approaches, all multi-modal RAG pipelines outperform their text-based counterpart. Notably, the M3DOCRAG pipeline (ColPali + Qwen2-VL 7B) delivers the state-of-the-art results on MP-DocVQA. It is interesting that while the existing entries were fine-tuned specifically for MP-DocVQA, the components used in M3DOCRAG (ColPali or Qwen2- 

Table 3. Closed-domain DocVQA evaluation results on MPDocVQA. The RAG methods retrieve a single page to the downstream QA models. 

|**Method**|**Answer Accuracy**<br>ANLS|**Page Retrieval**<br>R@1|
|---|---|---|
|_Multimodal LMs_|||
|Arctic-TILT 0.8B [10]|0.8122|50.79|
|GRAM [9]|0.8032|19.98|
|GRAM C-Former [9]|0.7812|19.98|
|ScreenAI 5B [3]|0.7711|77.88|
|_Text RAG_|||
|ColBERT v2 + Llama 3.1 8B|0.5603|75.33|
|M3DOCRAG|||
|ColPali + Qwen2-VL 7B (Ours)|**0.8444**|**81.05**|



VL 7B) were not tailored to this dataset – although Qwen2VL 7B might have been trained on DocVQA [42], which shares some images with MP-DocVQA. 

## **5.3. Additional analysis** 

**Different page indexing: speed and accuracy.** In Table 4, we analyze the speed and accuracy of ColPali+Qwen2-VL 7B pipeline with different document embedding indexing methods. While the naive indexing with exact search (FlatIP) is slow (21s per query), we find that using approximate indexing such as inverted file [52, 66] (IVFFlat) and product quantization [27] (IVFPQ) can retain most of the accuracy, while making the search significantly faster ( _<_ 2s per query). We use 

7 

Table 4. Speed-accuracy tradeoff with different indexing strategies on M3DOCVQA. Method: ColPali + Qwen2-VL 7B. 

|**# Pages**<br>**Indexing**|**Latency (s) (**_↓_**)**<br>Retrieval<br>VQA|**Accuracy (**_↑_**)**<br>EM<br>F1|
|---|---|---|
|1<br>FlatIP|21.0<br>1.1|28.9<br>33.7|
|1<br>FlatIP + IVFFlat|1.8<br>1.1|27.9<br>32.3|
|1<br>FlatIP + IVFPQ|0.2<br>1.1|25.9<br>30.3|
|2<br>FlatIP + IVFFlat|1.8<br>2.4|29.9<br>34.6|
|2<br>FlatIP + IVFPQ|0.2<br>2.4|29.0<br>33.5|
|4<br>FlatIP + IVFFlat|1.8<br>4.8|31.4<br>36.5|
|4<br>FlatIP + IVFPQ|0.2<br>4.8|29.9<br>34.7|



FlatIP+IVFFlat indexing by default, and users can choose appropriate indexing methods depending on their deployment requirements. 

Table 5. Comparison of different multimodal LMs within M3DOCRAG, evaluated across different document understanding benchmarks. For retrieval, we use the top-1 page from ColPali for all datasets. We use FlatIP+IVFFlat indexing for M3DOCVQA. 

|**Multimodal LMs**|**M3DOCVQA**<br>F1 (_↑_)|**MMLongBench-Doc**<br>Acc (_↑_)|**MP-DocVQA**<br>ANLS (_↑_)|
|---|---|---|---|
|Idefcs2 8B<br>Idefcs3 8B<br>InternVL2 8B|27.8<br>31.8<br>30.9|10.8<br>16.4<br>17.3|0.56<br>0.77<br>0.81|
|Qwen2-VL 7B|**32.3**|**18.8**|**0.84**|



**Different multi-modal LMs.** In Table 5, we compare four different multi-modal LMs in the M3DOCRAG framework: Idefics2 8B [33], Idefics3 8B [32], InternVL2 8B [12], and Qwen2-VL 7B [59]. The Qwen2-VL 7B model outperforms other MLMs in all three benchmarks. Thus, we use the model as our default MLM component. 

Table 6. Comparison of different multi-modal retrieval models within M3DOCRAG framework, evaluated across different document understanding benchmarks. We provide Qwen2-VL 7B with top-4 pages for MMLongBench-Doc/M3DOCVQA and top1 page for MP-DocVQA from the retrieval models. We use FlatIP+IVFFlat indexing for M3DOCVQA. 

|**Ret. Models**|**M3DOCVQA**<br>F1 (_↑_)|**MMLongBench-Doc**<br>Acc (_↑_)|**MP-DocVQA**<br>ANLS (_↑_)|
|---|---|---|---|
|ColPali v1|**36.5**|21.0|0.84|
|ColQwen v0.1|32.1|**21.5**|**0.86**|



**Different multi-modal retrieval models.** In Table 6, we compare two different multi-modal retrival models in M3DOCRAG framework: ColPali v1 and ColQwen v0.1 

(see Sec. 4 for details). Both models are trained with the same training objectives but are initialized with different MLM architectures: PaliGemma 2B [7] and Qwen2VL 2B [59], respectively. We find that ColPali achieves significantly better performance in M3DOCVQA, while ColQwen achieves slightly better performance in MPDocVQA and MMLongBench-Doc. Thus, we use ColPali as our default retrieval model. 

## **5.4. Qualitative Examples** 

In Fig. 5, Fig. 6, and Fig. 7, we provide qualitative examples of M3DOCRAG (ColPali + Qwen2-VL 7B)’s question answering results on several M3DOCVQA examples. In Fig. 5, the answer information is only visually stored within the game logo (‘man is leaning on a motorcycle’), and M3DOCRAG could find the information. In Fig. 6, the question requires multi-hop reasoning across different pages/documents, and M3DOCRAG could combine information from multiple retrieved pages. In Fig. 7, although ColPali did not retrieve the page that contains information about a team whose logo features a bat, Qwen-2 VL leverages its own knowledge ‘Valencia CF has a logo featuring a bat’, and could provide the final answer. Overall, the qualitative examples showcase that M3DOCRAG can successfully tackle different questions whose answer sources exist in various modalities. 

## **6. Related Work** 

**Document visual question answering.** Mathew et al. [42] proposed document visual question answering (DocVQA) task, where a model extracts information from documents by treating them as images, like in generic visual question answering [1]. Most research on DocVQA focuses on handling a single-page document [22, 23, 30, 34, 41, 42, 55, 58, 63], and it has been now a common practice to include the single-page DocVQA [42] as a part of the image understanding evaluation suite among recent MLMs [7, 12, 20, 32, 46, 59]. Several recent works study applying MLMs for DocVQA on multi-page documents [31, 40, 57]. However, all previous works on DocVQA have focused on handling questions in the context of a specific document, such as “What was the gross profit in the year 2009?” [14, 40, 42, 57]. While this is probably due to the limited context length of the backbone multi-modal LMs, this does not reflect real-world scenarios, where users often ask questions that require information across different pages/documents. We address the limitation and propose M3DOCRAG framework and M3DOCVQA dataset for effective, efficient, and flexible document understanding under various document contexts (closed-domain and open-domain), question hops (singlehop and multi-hop), and evidence modalities (text, chart, figure, _etc_ .). 

8 

**==> picture [303 x 259] intentionally omitted <==**

**----- Start of picture text -----**<br>
Question: “SIE Bend Studio's 2019 game cover has man leaning on what?”<br>ColPali + Qwen2-VL 7B: “motorcycle”<br>Top 2 pages retrieved by ColPali<br>Create account Log in Create account Log in<br>Bend Studio 18 languages Days Gone 21 languages<br>Article Talk Read Edit View history Tools Article Talk Read Edit View history Tools<br>From Wikipedia, the free encyclopedia From Wikipedia, the free encyclopedia<br>- (Redirected from  Bend StudioInc. ) is an American  (formerly SIE Bend Studiovideo game developer Blank, Berlyn & Co., Inc. )  based in  and Bend, Eidetic, - Bend Studio gg Days Gone by EntertainmentBend Studio is a 2019 . The game was released for the  and published by action-adventureSony Interactivevideo gamePlayStation 4 developed _ Days Gone<br>Oregondeveloping  Gone PlayStation Studios. Since 2000, Bend Studio is a . Founded in 1992, the studio is best known for Bubsy 3D . , the  Syphon Filter first-party developer series, and  Days  for Formerly Blank, Berlyn & Co., Inc.(1992–1995) 73ENDSTUDIC in April 2019. A  Days Gone the start of a pandemic that turned a portion of humanity intovicious zombie-like creatures. Former  is set in Windowspost-apocalyptic port was released in May 2021.Oregonoutlaw-turned-drifter two years after Fare<br>Eidetic, Inc. (1995–2000) Deacon St. John discovers his wife Sarah, having been<br>History [ edit ] Company type Subsidiary assumed dead, may still be alive and goes on a quest to find<br>Marc BlankBlank, Berlyn & Co. in 1992.and the product development director for Berlyn, an author of at Infocom before moving to approached by a California company after an employee hadused remembered that the company also developed games. Thatcompany was looking to release a "sound-oriented gamemachine for cars", for which Blank suggested a series ofCornerstone and Michael Berlyn, a software package by Infocom, andadventure games [[2][3]] Accolade founded Bend Studio as Blank had been a founder., had previously worked [[2]]  Blank wasInfocom, while IndustryFoundedFoundersHeadquartersKey peopleProductsNumber ofemployees | Video games1992; 32 years agoMarc BlankMichael BerlynBend, OregonChristopher Reese (director Bubsy 3DSyphon FilterDays Gone 150+ [[1]]  (2022)) , US studio her. The game is played from a which the player can explore an Players can use firearms, weaponshostile humans and cannibalistic creatures known asFreakers. A major game mechanic is Deacon's motorcycle,which is used as the player character's main mode oftransportation. Days Gone original property since , and can use  was Bend Studio's first open-world project, its first Syphon Filter stealthmelee weapons to defend themselves againstthird-person perspectiveopen world (1999), and its first, and  environment.improvised in Developer(s)Publisher(s)Director(s)Producer(s)Designer(s)Programmer(s)Artist(s)Writer(s) Bend StudioSony InteractiveEntertainmentJohn GarvinJeff RossDarren YagerRon AllenJohn HoffmanDonald YatomiJohn Garvin<br>sports gamesproject never went into production and Blank repurposed theidea for an resembling a TV broadcast. In 1992, he pitched the idea toBerlyn, wondering whether Accolade would be interested in such a title.A few months after the 1993 release of  Kind under the Blank, Berlyn & Co. name. Blank became the company. Mystery Capers , when Berlyn was on hiatus at Accolade, they began developing games [[2]] American football The company's first games were the  that would sound like radio broadcasts. The and  Dell Crossword Puzzles  video game with an ambiance Bubsy in Claws Encounters of the Furred  for the puzzle video gamesApple Newtonpresident of the new ParentWebsite . Both were Columbo's [[2]] PlayStation Studios(2000–present)bendstudio.com aa development project for home consoles after spendingdecades working on spinoff games for handheld consoles. Thegame's development took approximately six years; BendStudio expanded nearly three-fold to support it. Major sourcesof inspiration for  Dead 2016delayed several times.Upon release, who criticized the game's mission design and technical issues; its release was originally planned for 2018 but was and  Sons of AnarchyDays GoneDays Gone  received mixed reviews from critics,. The game was unveiled at  were  World War Z ,  The Walking E3 Composer(s)EnginePlatform(s)ReleaseGenre(s)Mode(s) Nathan WhiteheadUnreal Engine 4PlayStation 4Windows PlayStation 4 April 26, 2019 Windows May 18, 2021Action-adventureSingle-player<br>released in November 1993 by StarCore, Newton. [[4][5]]  Two further such games,  Dell Crossword Puzzles and Other Word Apple's publishing label for the — but praised the graphics, artificial intelligence, and Sam Witwer's performance as Deacon, while the story<br>**----- End of picture text -----**<br>


Figure 5. Qualitative example of ColPali + Qwen2-VL 7B on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The answer information is only stored visually within the game logo, where a man is leaning on a motorcycle. 

Question: “What distance was the AP Warrior fast race at the Del Mar Racetrack?” 

ColPali + Qwen2-VL 7B: “Seven Furlongs” 

**==> picture [307 x 212] intentionally omitted <==**

**----- Start of picture text -----**<br>
Top 2 pages retrieved by ColPali<br>Create account Log in 1st4th San Felipe StakesEl Camino Real Derby One and One-Sixteenth MilesOne and One-Sixteenth Miles Santa Anita ParkBay Meadows FastFast<br>A.P. Warrior Add languages 4th Hollywood Futurity One and One-Sixteenth Miles Hollywood ParkRacetrack Fast<br>Article Talk Read Edit View history Tools 1st Allowance One Mile Oak Tree at Park Santa Anita Fast<br>From Wikipedia, the free encyclopedia A.P. Warrior thoroughbredcontender in 2006 and Grade II winner on both dirt and turf. (foaled February 24, 2003 in race horse who was a Kentucky DerbyKentucky) is a [[1]] SireGrandsireDam A.P. Warrior A.P. IndySeattle SlewWarrior Queen 2nd6th1st Norfolk StakesDel Mar FuturityMaiden Special Weight eS One and One-Sixteenth MilesSeven FurlongsSeven Furlongs | Oak Tree at ParkDel Mar RacetrackHollywood ParkRacetrack Santa Anita FastFastFast<br>Racing career [ edit ] Damsire Quiet American<br>A son of raced by Initially trained by Eoin Harty, horse for his last seven starts. He won four times in twelvestarts. A.P. IndyStanley E. Fulton and Warrior Queen, he was owned and, owner of John ShirreffsSunland Park Racetrack conditioned the . SexFoaledCountryColourBreeder Stallion2003United StatesDark BayJim Fleming References 1. 2. 3.  ^^^ A.P. Warrior sold, retired - NTRAA. P. Warrior : Interactive Stallion Directorya P Warrior Horse Pedigree[ edit ] Archived  November 25, 2006, at the Wayback Machine<br>In 2007, A.P. Warrior was retired and soldin Versailles, Kentucky where he stands for $15,000. [[2]]  to Stonewall Farm [[3]]  In 2011 OwnerTrainerRecord Stanley E. FultonJohn Shirreffs13: 4-2-3 CategoriesRacehorses trained in the United States:  2003 racehorse births Racehorses bred in KentuckyThoroughbred family 4-j<br>he moved to Stonewall Farms Ocala Location in Ocala, FL. He Earnings $548,595<br>died there in early 2016 of a heart attack. Major wins<br>San Felipe Stakes (2003) This page was last edited on 7 October 2023, at 07:43 (UTC).<br>Races [ edit ] La Jolla Handicap (2003) Text is available under the Terms of Use and Privacy PolicyCreative Commons Attribution-ShareAlike License 4.0. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc.; additional terms may apply. By using this site, you agree to the, a non-profit organization.<br>Finish Race Distance Track Condition Privacy policy About Wikipedia Disclaimers Contact Wikipedia Code of Conduct Developers Statistics Cookie statement<br>3rd Oak Tree Derby One and One-Eighth Miles(Turf) Oak Tree at Park Santa Anita Firm Mobile view<br>1st La Jolla Handicap One and One-Sixteenth Miles(Turf) Del Mar Racetrack Firm<br>3rd Swaps Breeders' CupStakes One and One-Eighth Miles Hollywood ParkRacetrack Fast<br>2nd Affirmed Handicap One and One-Sixteenth Miles Hollywood ParkRacetrack Fast<br>18th Kentucky Derby One and One-Quarter Miles Churchill Downs Fast<br>3rd Santa Anita Derby One and One-Eighth Miles Santa Anita Park Fast<br>**----- End of picture text -----**<br>


Figure 6. Qualitative example of ColPali + Qwen2-VL 7B on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The question requires multi-page/document reasoning. 

9 

Question: “What date was a player transferred in to Lorca FC in the 2017-18 season from the club with a logo featuring a bat?” ColPali + Qwen2-VL 7B: “11 July 2017” 

||**Top 1 page retrieved by ColPali**||
|---|---|---|
|**No.**<br>**Pos.**<br>**Nation**<br>**Player**<br>1<br>GK<br> ESP<br>Jaume Valens<br>2<br>DF<br> ESP<br>Juan Pedro Pina<br>_(Captain)_<br>3<br>DF<br> ESP<br>Carlos Pomares<br>**No.**<br>**Pos.**<br>**Nation**<br>**Player**<br>15<br>DF<br> SWE<br>Markus Holgersson<br>16<br>MF<br> ESP<br>Sito _(on loan fromValencia)_<br>17<br>FW<br> ESP<br>Manel Martínez _(on loan_<br>_fromGirona)_<br>=<br>=<br>~~.~~<br>~~==~~<br>~~=~~<br>=|||
|4<br>DF|ESP<br>Fran Cruz<br>18<br>DF<br> ESP<br>Molo||
|5<br>MF|ESP<br>Eugeni _(on loan from_<br>_Valencia)_<br>19<br>FW<br> URU<br>Miguel Merentiel _(on loan_<br>_fromPeñarol)_<br>~~==~~||
|6<br>MF|ESP<br>Haritz Albisua<br>20<br>DF<br> ESP<br>Carlos Peña<br>~~=~~<br>=||
||||
|7<br>MF<br> ESP<br>Carlos Martínez<br>8<br>MF<br> ESP<br>Tropi _(on loan from_<br>_Valencia)_<br>9<br>FW<br> ESP<br>Chumbi<br>10<br>MF<br> ESP<br>Alberto Noguera<br>21<br>FW<br> ESP<br>Dani Ojeda<br>22<br>MF<br> ESP<br>Adán Gurdiel<br>23<br>MF<br> ESP<br>Abel Gómez<br>24<br>MF<br> ESP<br>Javi Muñoz _(on loan from_<br>_Real Madrid)_<br>~~==~~<br>~~=~~<br>.<br>=<br>=<br>~~.~~=<br>=<br>.<br>—|||
|11<br>FW<br>12<br>MF|ESP<br>Manuel Onwu<br> ESP<br>Nando _(on loan from_<br>_Alavés)_<br>25<br>GK<br> URU<br>Franco Torgnascioli<br>26<br>DF<br> ESP<br>José Carlos _(on loan from_<br>_Betis)_<br>=<br>=<br>~~==~~||
|13<br>GK<br>14<br>MF|ESP<br>Francisco Dorronsoro<br> ESP<br>Cristian Bustos<br>28<br>FW<br> NGA<br>Manu Apeh<br>—<br>DF<br> ESP<br>Antonio López<br>~~=~~<br>=||
|**Transfers**<br>[edit]<br>List of Spanish football transfers summer 2017#Lorca FC|||
|**In**<br>[edit]|||
|**Date**|**Player**<br>**From**<br>**Type**<br>**Fee**<br>**Ref**||
|30 June<br>2017<br>30 June<br>2017<br>30 June<br>2017<br>6 July 2017<br>11 July<br>2017|Samu Martínez<br> Hospitalet<br>Loan<br>return<br>Free<br> Haritz Albisua<br> Lleida Esportiu<br>Loan<br>return<br>Free<br> Mikel<br>Fernández<br> Lleida Esportiu<br>Loan<br>return<br>Free<br> Jaume Valens<br> Mallorca B<br>Transfer<br>Free<br>[1]<br> Tropi<br> Valencia<br>Loan<br>Free<br>[2]<br>~~==~~<br>~~=~~<br>=<br>~~==~~<br>~~=~~<br>=||



Figure 7. Qualitative example of ColPali + Qwen2-VL 7B on M3DOCVQA. Image regions relevant to the question/answer are highlighted with orange boxes. The VQA component could combine both the retrieved knowledge (Tropi was transferred on 11 July 2017) and its own knowledge (Valencia CF has a logo with a bat) to provide the final answer. 

**Retrieval-augmented generation.** Retrieval-augmented generation (RAG) [35] has emerged as a hybrid approach combining retrieval systems with generative models to improve the quality and relevance of generated content [19]. RAG has been widely studied for open-domain question answering [2, 21, 24, 28, 39, 65], where the community has well-established practices for text-based pipelines. A line of work in VQA studies RAG on visual questions that require world knowledge [11, 44, 51, 62], but their retrieval context is usually generic images and/or short text snippets and does not cover DocVQA settings. To the best of our knowledge, no prior work has explored RAG setting for multi-modal document understanding only with multi-modal models (instead of using OCR methods). Our framework tackles opendomain question answering over documents with complex 

multi-modal contexts, including textual, tabular, and visual information across different pages and documents. 

## **7. Conclusion** 

We introduce M3DOCRAG, a novel multi-modal RAG framework that flexibly accommodates various document contexts (closed-domain and open-domain), question hops (single-hop and multi-hop), and evidence modalities (text, chart, figure, _etc_ .). In M3DOCRAG, a multi-modal retrieval model identifies relevant pages from single or multiple documents, which are then processed by a multimodal language model, where all documents are represented as pixels. Next, we introduce M3DOCVQA, the first benchmark that evaluates open-domain multi-modal document understanding capabilities. M3DOCVQA consists of 2,000+ questions and 3,000+ PDF documents, and the questions need to be answered with various modalities such as images, text, and tables. Our experiments in three datasets (M3DOCVQA, MP-DocVQA, and MMLongBench-Doc) demonstrate significant advantages of M3DOCRAG over existing methods, including the state-of-the-art performance in MP-DocVQA. We also provide analysis comparing different indexing strategies, multi-modal LMs, and multi-modal retrieval models. Finally, we show qualitative examples where M3DOCRAG can successfully tackle different questions whose answer sources exist in various modalities. We hope that our work encourages future advancements in multi-modal frameworks for document understanding, paving the way for more robust, scalable, and practical solutions in real-world applications. 

## **Ethical Considerations** 

**Limitations.** Since our multimodal retrieval models and multimodal LMs were trained with English-heavy datasets, they might not understand prompts or documents written in non-English. While our M3DOCRAG framework can benefit many document understanding applications, the model components could present false or biased information. Thus, the framework should be used with human supervision in real-world applications. Note that M3DOCRAG is designed with flexibility so that users can update or replace components as more accurate solutions for each element of the framework become available in the future. 

**Data collection.** We do not involve human subjects during data collection. We do not claim ownership/rights of the Wikipedia documents, and we attribute the source Wikipedia document URLs to all pages. 

## **References** 

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi 

10 

Parikh. VQA: Visual question answering. In _ICCV_ , 2015. 8 

- [2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. 10 

- [3] Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor C˘arbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. ScreenAI: A Vision-Language Model for UI and Infographics Understanding, 2024. 7 

- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. _arXiv preprint_ , abs/2308.12966, 2023. 7 

- [5] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. _arXiv preprint_ , abs/2401.18058, 2024. 7 

- [6] Edouard Belval. pdf2image, 2017. 6 

- [7] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Boˇsnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3B VLM for transfer, 2024. 8 

- [8] Ali Furkan Biten, Andres Mafla, Lluis Gomez, Valveny C V Jawahar, and Dimosthenis Karatzas. Scene Text Visual Question Answering. In _ICCV_ , 2019. 6 

- [9] Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. Gram: Global reasoning for multi-page vqa. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15598–15607, 2024. 7 

- [10] Łukasz Borchmann, Michał Pietruszka, Wojciech Ja´skowski, Dawid Jurkiewicz, Piotr Halama, Paweł J´oziak, Łukasz Garncarek, Paweł Liskowski, Karolina Szyndler, Andrzej Gretkowski, Julita Ołtusek, Gabriela Nowakowska, Artur Zawłocki, Łukasz Duhr, Paweł Dyda, and Michał Turski. Arctic-TILT. Business Document Understanding at SubBillion Scale, 2024. 7 

- [11] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W Cohen. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. _arXiv preprint arXiv:2210.02928_ , 2022. 10 

- [12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 6, 8 

- [13] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International Conference on Learning Representations (ICLR)_ , 2024. 6 

- [14] Yihao Ding, Siwen Luo, Hyunsuk Chung, and Soyeon Caren Han. Pdfvqa: A new dataset for real-world vqa on pdf documents. In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , pages 585–601. Springer, 2023. 1, 2, 8 

- [15] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-Xcomposer24KHD: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv preprint_ , abs/2404.06512, 2024. 7 

- [16] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria Lomeli, Lucas Hosseini, and Herv´e J´egou. The faiss library, 2024. 4, 6 

- [17] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. ColPali: Efficient Document Retrieval with Vision Language Models, 2024. 1, 4, 6 

- [18] Mathieu Fenniak and PyPDF2 Contributors. The PyPDF2 library, version 2, 2022. 1 

- [19] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023. 10 

- [20] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 8 

- [21] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-Augmented Language Model Pre-Training. In _ICML_ , 2020. 10 

- [22] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding, 2024. 7, 8 

- [23] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , page 4083–4091, New York, NY, USA, 2022. Association for Computing Machinery. 8 

- [24] Gautier Izacard and Edouard Grave. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. In _EACL_ , 2021. 10 

- [25] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 7 

- [26] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with gpus. _IEEE Transactions on Big Data_ , 7(3):535–547, 2021. 4, 6 

- [27] Herve J´egou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. _IEEE Transactions_ 

11 

_on Pattern Analysis and Machine Intelligence_ , 33(1):117– 128, 2011. 7 

- [28] Vladimir Karpukhin, O Barlas, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering. In _EMNLP_ , pages 6769–6781, 2020. 10 

- [29] Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. _SIGIR 2020 - Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 39–48, 2020. 4 

- [30] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision (ECCV)_ , 2022. 8 

- [31] Jordy Van Landeghem, Rafał Powalski, Rub`en Tito, Dawid Jurkiewicz, Matthew Blaschko, Łukasz Borchmann, Micka¨el Coustaty, Sien Moens, Michał Pietruszka, Bertrand Ackaert, Tomasz Stanisławek, Paweł J´oziak, and Ernest Valveny. Document Understanding Dataset and Evaluation (DUDE). In _ICCV_ , 2023. 1, 4, 8 

- [32] Hugo Laurenc¸on, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions, 2024. 6, 8 

- [33] Hugo Laurenc¸on, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 6, 7, 8 

- [34] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: screenshot parsing as pretraining for visual language understanding. In _Proceedings of the 40th International Conference on Machine Learning_ . JMLR.org, 2023. 8 

- [35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen Tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In _NeurIPS_ , 2020. 1, 10 

- [36] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint_ , abs/2311.06607, 2023. 7 

- [37] Llama Team. The llama 3 herd of models, 2024. 6 

- [38] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. DeepSeek-VL: towards real-world vision-language understanding. _arXiv preprint_ , abs/2403.05525, 2024. 7 

- [39] Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. Sail: Search-augmented instruction learning, 2023. 10 

- [40] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, 

   - et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _arXiv preprint arXiv:2407.01523_ , 2024. 1, 2, 3, 4, 5, 6, 7, 8 

- [41] Minesh Mathew, Viraj Bagal, Rub`en P´erez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. _2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pages 2582–2591, 2021. 8 

- [42] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 1, 2, 4, 7, 8 

- [43] Jamshed Memon, Maira Sami, Rizwan Ahmed Khan, and Mueen Uddin. Handwritten optical character recognition (ocr): A comprehensive systematic literature review (slr). _IEEE Access_ , 8:142642–142668, 2020. 1 

- [44] Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha, Andr´e Araujo, and Vittorio Ferrari. Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 3090–3101, 2023. 10 

- [45] Microsoft. Playwright for python, 2021. 5 

- [46] OpenAI. Hello gpt-4o, 2024. 5, 8 

- [47] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chana, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In _NIPS Workshop_ , 2017. 6 

- [48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. _Advances in Neural Information Processing Systems_ , 32 (NeurIPS), 2019. 6 

- [49] pdfminer. pdfminer.six, 2019. 1 

- [50] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. _NAACL 2022 - 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Proceedings of the Conference_ , pages 3715–3734, 2022. 4, 6 

- [51] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge, 2022. 10 

- [52] Sivic and Zisserman. Video google: a text retrieval approach to object matching in videos. In _Proceedings Ninth IEEE International Conference on Computer Vision_ , pages 1470– 1477 vol.2, 2003. 2, 4, 7 

- [53] Ray Smith. An overview of the tesseract ocr engine. In _ICDAR_ , 2007. 1, 6 

- [54] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question 

12 

answering over text, tables and images. _arXiv preprint arXiv:2104.06039_ , 2021. 2, 5 

- [55] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing, 2023. 8 

- [56] The Chromium Project Authors. The chromium projects, 2024. 5 

- [57] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834, 2023. 1, 2, 3, 4, 5, 8 

- [58] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. DocLLM: A layout-aware generative language model for multimodal document understanding, 2023. 8 

- [59] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 1, 4, 6, 8 

- [60] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, and Jamie Brew. HuggingFace’s Transformers: State-of-the-art Natural Language Processing. In _EMNLP_ , 2020. 6 

- [61] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: An LMM perceiving any aspect ratio and high-resolution images. _arXiv preprint_ , abs/2403.11703, 2024. 7 

- [62] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-Augmented Multimodal Language Modeling. In _ICML_ , 2023. 10 

- [63] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Mingshi Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Feiyan Huang. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In _Conference on Empirical Methods in Natural Language Processing_ , 2023. 8 

- [64] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. RLAIF-V: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. _arXiv preprint_ , abs/2405.17220, 2024. 7 

- [65] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua. Retrieving and reading: A comprehensive survey on open-domain question answering. _arXiv preprint arXiv:2101.00774_ , 2021. 10 

- [66] Justin Zobel and Alistair Moffat. Inverted files for text search engines. _ACM Comput. Surv._ , 38(2):6–es, 2006. 2, 4, 7 

13 

