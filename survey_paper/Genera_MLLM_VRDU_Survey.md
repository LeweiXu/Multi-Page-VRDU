**026** 

**027** Visually-Rich Document Understanding (VRDU) **028** lies at the intersection of vision and language, aim- **029** ing to extract and understand information from **030** documents with multiple data modalities and com- **031** plex layouts (Park et al., 2019; Ding et al., 2023). **032** With the rapid digitization of physical documents **033** and the widespread use of structured and semi- **034** structured digital documents, the development of **035** robust, generalizable VRDU frameworks has at- **036** tracted significant attention for automating infor- **037** mation extraction, improving accessibility, and en- **038** hancing decision-making across diverse domains **039** such as finance, healthcare, and education. **040** Early VRDU frameworks relied on manu- **041** crafted rules and heuristics 

## **A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends** 

## **Anonymous ACL submission** 

## **Abstract** 

**001** Visually Rich Document Understanding **002** (VRDU) has become a pivotal area of re- **003** search, driven by the need to automatically **004** interpret documents that contain intricate **005** visual, textual, and structural elements. Re- **006** cently, Multimodal Large Language Models **007** (MLLMs) have demonstrated significant **008** promise in this domain, including both **009** OCR-based and OCR-free approaches for **010** information extraction from document im- **011** ages. This survey reviews recent advances in **012** MLLM-based VRDU, highlighting emerging **013** trends and promising research directions with **014** a focus on two key aspects: (1) techniques for **015** representing and integrating textual, visual, **016** and layout features; (2) training paradigms, **017** including pretraining, instruction tuning, and **018** training strategies. Moreover, we address **019** challenges such as data scarcity, handling **020** multi-page and multilingual documents, **021** and integrating emerging trends such as **022** Retrieval-Augmented Generation and agentic **023** frameworks. Our analysis offers a roadmap for **024** advancing MLLM-based VRDU toward more **025** scalable, reliable, and adaptable systems. 

## **1 Introduction** 

Early VRDU frameworks relied on manually crafted rules and domain-specific heuristics 

(Watanabe et al., 1995; Seki et al., 2007), which **042** experienced a sudden performance drop on unseen **043** documents across domains or with diverse lay- **044** outs. Conventional deep learning approaches em- **045** ployed CNNs (Katti et al., 2018; Yang et al., 2017) **046** and RNNs (Denk and Reisswig, 2019) to lever- **047** age visual or textual features, facilitating more in- **048** formative representations. However, these meth- **049** ods typically do not effectively integrate the di- **050** verse modalities in documents, limiting their ca- **051** pacity to capture the rich semantic structure inher- **052** ent in visually rich documents. With the success of **053** pretraining techniques in language modelling, nu- **054** merous VRDU models (Huang et al., 2022; Hong **055** et al., 2022; Lyu et al., 2024) have been pretrained **056** on large-scale scanned or PDF document datasets, **057** enabling more effective fusion of visual, textual, **058** and layout features for robust multimodal repre- **059** sentation. However, their effectiveness is con- **060** strained by the scope and diversity of their pre- **061** training data, often necessitating substantial fine- **062** tuning to achieve cross-domain generalizability. **063** Recently, MLLMs (OpenAI, 2024; Liu et al., **064** 2024b), trained on massive visual and linguistic **065** datasets, have demonstrated powerful represen- **066** tational capabilities and extensive world knowl- **067** edge, enabling a deeper understanding of text- **068** dense images with diverse visual appearances and **069** complex spatial layouts. By combining the supe- **070** rior text understanding of LLMs (Touvron et al., **071** 2023) with visual encoders (Dosovitskiy et al., **072** 2020) that capture image content and layout in- **073** formation, MLLM-based VRDU frameworks have **074** demonstrated strong performance across diverse **075** document question-answering and information- **076** extraction tasks, and generalizability across do- **077** mains without task-specific fine-tuning. **078** This paper provides a comprehensive survey **079** of recent developments in MLLM-based VRDU **080** frameworks. Previous surveys have either fo- **081** cused on a broad analysis of the diverse capabil- **082** 

1 

|**Model**|**Venue**|**Tasks**|**Mod.**|**LLM Backbone**|**Vision Encoder**|**PT**|**IT**|**FT**|**Pages**|**Prompt In.**|
|---|---|---|---|---|---|---|---|---|---|---|
|**OCR-Dependent**|||||||||||
|ICL-D3IE (2023)|ICCV|KIE|T, L|GPT-3|–|✗|✗|✗|SP|ICL+Layout|
|DocLLM (2024a)|ACL|KIE, QA, DC|T, L|Custom|–|✓|✓|✗|SP|T+B+Q|
|LAPDoc (2024)|ICDAR|KIE, QA|T, L|Multiple|–|✗|✗|✗|SP|Rule|
|LMDX (2024)|ACL|KIE|T, L|Gemini-pro|–|✗|✗|✗|SP|ICL+Layout|
|ProcTag (2025)|AAAI|QA|T, V, L|GPT-3.5|–|✗|✗|✓|SP|Rule+CoT|
|DocKD (2024)|EMNLP|KIE, QA, DC|T, L|Custom|–|✗|✗|✓|SP|Gen by VL|
|DoCo (2024)|CVPR|KIE, QA, DC|T, L|Multiple|LayoutLMv3|✓|✗|✓|SP|I+Q|
|InstructDoc (2024)|AAAI|KIE, QA|T, V, L|FlanT5|LayoutLMv3|✗|✓|✓|MP|I+Q|
|LayoutLLM (2024)|CVPR|KIE, QA|T, V, L|Vicuna-7B-v1.5|OpenCLIP+CLIP|✗|✓|✓|SP|I+Q+CoT|
|LLaVA-Read (2024c)|preprint|KIE, QA|T, V, L|Vicuna-1.5 13B|Multiple|✓|✓|✗|SP|I+Q|
|LayTextLLM (2024)|ACL|QA, KIE|T, L|Llama2-7B-base|–|✓|✗|✓|SP|T+B|
|DocLayLLM (2024)|CVPR|QA, KIE|T, V, L|Llama2-7B-chat|Pix2Struct-Large|✗|✓|✓|SP|I+Q+B|
|LayTokenLLM (2025b)|CVPR|QA|T, L|Multiple|–|✓|✗|✗|MP|I+Q+L|
|GPE (2025a)|ICLR|KIE, QA|T, L|Multiple|–|✗|✗|✓|SP|T+B+Q|
|MDocAgent (2025)|preprint|QA|T, V|Multiple|IXC2-VL-4KHD|✗|✗|✗|MP|I+Q|
|PDF-WuKong (2025)|preprint|QA|T, V|BGE-M3|LayoutLMv3|✗|✗|✓|MP|I+Q|
|DocAssistant (2025)|EMNLP|QA|T, V|InternVL2-Chat-2B|InternVL2 ViT|✗|✗|✓|SP|I+Q|
|AlignVLM (2025)|Neurips|QA|T, V|LLaMA-3.2 (1B, 3B)|SigLIP-400M|✓|✓|✓|SP|I+Q|
|DocThinker (2025)|ICCV|QA, KIE|T, V|Qwen2.5-VL (3B, 7B)|Qwen2.5-VL ViT|✗|✗|✓*|SP|I+Q|
|**OCR-Free**|||||||||||
|KOSMOS-2.5 (2023)|preprint|QA, KIE|V|Custom|mPLUG-Owl VE|✗|✓|✓|SP|I+Q|
|mPLUG-DocOwl (2023a)|preprint|QA|V|mPLUG-Owl|mPLUG-Owl VE|✗|✓|✗|SP|I+Q|
|UReader (2023b)|EMNLP|QA|V|mPLUG-Owl|mPLUG-Owl VE|✗|✓|✗|SP|I+Q|
|TGDoc (2023)|preprint|KIE, QA|V|Vicuna-7B|CLIP-ViT-L/14|✗|✓|✓|SP|I+Q+B|
|UniDoc (2023)|preprint|KIE, QA|V|Vicuna-7B|CLIP-ViT-L/14|✗|✓|✓|SP|I+Q+B|
|DocPedia (2024)|SCIS|KIE, QA|V|Vicuna-7B|Swin Trans.|✓|✗|✓|SP|I+Q|
|HRVDA (2024a)|CVPR|KIE, QA|V|LLama2-7B|Swin Trans.|✓|✓|✗|SP|I+Q|
|Vary (2024)|ECCV|QA, DocRead|V|Multiple|CLIP, ViTDet|✓|✗|✓|SP|I+Q|
|mPLUG-DocOwl1.5 (2024)|EMNLP|KIE, QA|V|mPLUG-Owl2|mPLUG-Owl2 VE|✗|✓|✓|SP|I+Q|
|HVFA (2024)|Neurips|QA, Cap.|V|Multi (BLIP-2, etc.)|ViT/L-14|✗|✓|✗|SP|I+Q|
|Texthawk (2024a)|preprint|QA|V|InternLM-XC|ViT|✗|✓|✓|SP|I+Q|
|Texthawk2 (2024b)|preprint|OCR, Grd, QA|V|Qwen2-7B-Instr|SigLIP-SO400M|✗|✓|✓|MP|I+Q+Task|
|TextMonkey (2024c)|preprint|KIE, QA|V|Qwen-VL|Vit-BigG|✗|✓|✗|SP|I+Q|
|Llavar (2024d)|preprint|QA|V|Vicuna-13B|CLIP-ViT-L/14|✗|✓|✓|SP|I+Q|
|TokenCorrCompressor (2024b)|preprint|QA, Cap.|V|LLaMA-2|CLIP-ViT/L14|✗|✗|✓|SP|I+Q|
|DocKylin (2024a)|AAAI|QA|V|Llama2-7B-chat|Donut-Swin|✗|✓|✓|SP|I+Q|
|Marten (2025b)|CVPR|QA|V|InterLM2|InternViT-300M|✗|✓|✓|SP|I+Q|
|PP-DocBee (2025)|preprint|QA|V|Qwen2-VL-2B|ViT|✗|✗|✓|SP|I+Q|
|mPLUG-DocOwl2 (2025)|ACL|KIE, QA|V|mPLUG-Owl2|ViT|✓|✗|✓|MP|I+Q|
|TokenFD (2025)|ICCV|QA, KIE|V|InternLM (2B, 8B)|ViT|✓|✓|✓|SP|I + Q|



Table 1: Comparison of existing MLLM-based VRDU frameworks. Mod.: Input modality; KIE: Key Information Extraction; QA: Question Answering; DC: Document Classification; T: Text; L: Layout; V: Vision; MP: MultiPage; SP: Single Page; I: Image; Q: Question; B: Bounding Box; CoT: Chain of Thought; Cap.: Captioning; Grd.: Grounding; Task: Task Information; VL: Vision-Language. 

**083** ities of MLLMs (Caffagni et al., 2024) or exam- **084** ined techniques applied to specific document un- **085** derstanding tasks, such as document layout anal- **086** ysis (Binmakhashen and Mahmoud, 2019), ques- **087** tion answering (Barboule et al., 2025), and rela- **088** tion extraction (Delaunay et al., 2023). A recent **089** study provides (Ding et al., 2025b) an overview **090** of deep learning-based frameworks for VRDU but **091** lacks a systematic perspective on MLLM-based **092** approaches. In contrast, this paper provides an **093** analysis of the MLLM-based VRDU frameworks **094** from the aspects of **Framework Architecture** that **095** covers both OCR- and OCR-free models (Sec 2), **096 Multimodal Representation** (Sec 3), **Training 097 Strategies** (Sec 4), and **Inference Prompt Set098 ting** (Sec 5). We also include a detailed discussion **099** of the challenges of VRDU and provide a critical **100** analysis of the trend and future directions (Sec 6). **101** Notably, this survey is limited to methods that **102** leverage MLLMs for document-level understand- 

ing, excluding multi-document applications, nonLLM-based methods, and MLLMs without VRDspecific adaptations. 

**103 104 105** 

## **2 Framework Architecture** 

**106** 

**General MLLM for VRDU.** Many closed- **107** (Team et al., 2024) and open-source (Chen et al., **108** 2024) general-domain MLLMs have been widely **109** adopted for VRDU tasks and have demonstrated **110** promising performance[1] . However, the text- **111** dense, visually rich, and layout-sensitive nature of **112** VRDs exposes fundamental limitations of general- **113** domain MLLMs when applied to VRDU, in- **114** cluding weak layout inductive bias, sensitivity to **115** OCR noise, and hallucination on these knowledge- **116** intensive tasks. Moreover, the wide range of **117** downstream VRDU applications necessitates spe- **118** cialized techniques that adapt existing LLM back- **119** 

1Refer to Appendix D see performance analysis. 

2 

**==> picture [214 x 82] intentionally omitted <==**

**----- Start of picture text -----**<br>
OCR-Dependent Frameworks OCR-Free Frameworks<br>Extracted Content<br>Text: Fart corporations acts.) Single or Multipage input<br>Document —_ Input booooeo Processors (OCR, PDF parsers) Parsers _ … BoundingBox ： eS Adaptor [88.0, 1.0, 169.0, 21.0], [197.0, 5.0,325.0, 21.0], [321.0, 2.0, 611.0, 26.0],[712.0, 0.0, 796.0, 26.0],…] …… | Query power percentage? ee Tokenizer: What was : [=]oooo0o0 ProcessorsVision [=] …… oooooCC Compressor oooo Adaptor … TokenizerReduce thenumber ofvisual tokens Query …<br>Encoders LLM Encoders LLM<br>Vision/Multimodal gogoa ooo [| oooo ooo<br>Boo000008Multimodal … GeneratedResponse Answer: 17.5% oo0000 Visual … GeneratedResponse Answer: 17.5%<br>**----- End of picture text -----**<br>


Figure 1: General OCR-dependent and OCR-free framework architectures. 

**120** bones (as shown in Figure 1) through VRDU- **121** specific multimodal representations, training ob- **122** jectives, and inference paradigms. In addition, as **123** VRDU tasks are often knowledge-intensive and **124** safety-critical, locally tuning open-source general- **125** domain LLMs on private document collections is **126** essential for practical deployment in sensitive do- **127** mains such as finance and industrial applications. 

**128 OCR-Dependent Frameworks.** As shown in **129** Figure 1, OCR-dependent frameworks leverage **130** off-the-shelf tools to extract textual and layout in- **131** formation from scanned or PDF documents. This **132** extracted data, in combination with the document **133** image, is typically fed into multimodal encoders **134** to generate joint representations. Some mod- **135** els (Wang et al., 2024a; He et al., 2023) input **136** the extracted text directly into LLMs, while oth- **137** ers (Luo et al., 2024; Zhu et al., 2025a) incor- **138** porate visual (Dosovitskiy et al., 2020) or mul- **139** timodal encoders (Huang et al., 2022) to project **140** those cues into language space via various adap- **141** tors or projects. These systems rely on external **142** tools to capture structural information without ex- **143** tensive pretraining (e.g., text recognition). How- **144** ever, reliance on OCR or parsing tools can intro- **145** duce cumulative errors, especially in handwritten **146** or low-quality scanned documents, hindering the **147** development of fully end-to-end models. Addi- **148** tionally, using low-resolution inputs may reduce **149** the expressiveness of document representations, **150** limiting the overall performance. 

**151 OCR-Free Frameworks.** OCR-free approaches **152** have been introduced for end-to-end VRD under- **153** standing tasks. These frameworks bypass text ex- **154** traction by directly processing document images. **155** Visual features are extracted via one or more vi- **156** sion encoders, fused with the user query, and de- **157** coded by an LLM to generate responses. Repre- **158** sentative models include Donut (Kim et al., 2022), **159** mPLUG-DocOwl (Ye et al., 2023a), and URe- 

ader (Ye et al., 2023b). Accurate comprehension **160** of fine-grained text in these OCR-free settings re- **161** quires high-resolution images, which, in turn, lead **162** to lengthy visual sequences requiring visual com- **163** pression modules (Liu et al., 2024a; Hu et al., **164** 2025). Moreover, effective text recognition in **165** these models often relies on large-scale pretrain- **166** ing or instruction-tuning to integrate textual and **167** layout features via tasks such as text spotting (Liu **168** et al., 2024c) and image captioning (Feng et al., **169** 2024). This paradigm, however, demands substan- **170** tial dataset construction and considerable compu- **171** tational resources, posing practical challenges. **172** 

## **3 Multimodal Representation** 

**173 3.1 Text Modality 174** OCR-dependent methods rely on external tools to **175** extract text for encoding, while OCR-free models **176** use document images directly, treating text as a **177** learning target. **178 Text Encoding via LLM.** Given the frequent **179** text recognition challenges faced by MLLMs, **180** stemming from low-resolution inputs or un- **181** dertrained vision encoders, off-the-shelf OCR- **182** extracted text is commonly embedded directly **183** into LLM prompts to enhance document compre- **184** hension (Wang et al., 2024a; Kim et al., 2024) **185** (see Figure 2). However, the extracted content **186** is often unordered; to address this, frameworks **187** such as ICL-D3IE (He et al., 2023) and LLaVA- **188** Read (Zhang et al., 2024c) employ the XY-cut al- **189** gorithm to reorder the text sequence. Addition- **190** ally, to handle long documents, some methods seg- **191** ment the text into chunks, though this may intro- **192** duce semantic discontinuities (Xie et al., 2025). In **193** sum, directly adding extracted text to prompts im- **194** proves context and reduces reliance on additional **195** encoders; however, performance remains limited **196** by OCR and LLM errors and weak multimodal in- **197** tegration. **198** 

**Text Encoding via Auxiliary Encoder.** To en- **199** hance multimodal integration, many frameworks **200** introduce auxiliary encoders to enhance text em- **201** beddings. Several methods (Luo et al., 2024; Zhu **202** et al., 2025a) enhance text representation and mul- **203** timodal fusion by feeding extracted text, image **204** patches, and bounding boxes into pretrained Lay- **205** outLMv3 (Huang et al., 2022). Notably, Zhu et **206** al. (2025a) propose a ROI Aggregation module **207** that aggregates fine-grained tokens (e.g., words) **208** 

3 

**219** 

**240** 

**241 242 243 244 245** 

**==> picture [443 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
Text Modality Visual Modality Layout Modality Multimodal Fusion<br>i) Text Encoding via LLM i) Low Resolution Image Encoding i) Positional Encoding i) Neural-based Fusion<br>—— Extracted ContentText: [‘Form’, ‘604’, ‘Corporations’, ‘Act’, …] Prompt LLM Image Patches = = … ! There was a change 2D Positional Encoder The previous notice 6/2/2003 Image Patches oOo I … P Layout … Go OCR Text  t … Bounding Box oO b …<br>ii) Text Encoding via Auxiliary Encoder  <Text (t), Bounding Box (b)>:  GO t Auxiliary Encoder (e.g. LayoutLM3) b Extracted Content oO t [<‘Form’, [88.0, 1.0, 169.0, 21.0]>,          <‘604’, [197.0, 5.0, 325.0, 21.0]>]  b oo t b ii) High Resolution Image Encoding ———Ooa IL Vision Encoder IL Sub-Images IL … ii) Layout as Prompt “— Given the following document“““ DESCtax invoice5119 QTY(RM)1 O P1 $30PRICE …… OO P2Prompt P3LLM … Text Recognition TaskInput: Document ImageOutput: Text + Coordinates ii) Target-oriented Fusion Fusion EncoderLLM Text: Coordinates:  [197.0, 5.0, 325.0, 21.0]6/2/2023<br>Cross-Attention or Self-Attention ”””<br>LLM Cropping  & Compression iii) Integration during Training iii) Prompt-based Fusion<br>iii) Text as Training Objectives … Pre-Training Q: What is the  Instruction Tuning  Q: Generate all the text and layout  Generated QA Prompt<br>Text Reading notice dated?the previous When was  Text Grounding Predict the bbox of the <OCR> 17. [197.0, 5.0, 325.0, 21.0] IH Vision Encoder IH IH … Mask Alignment hidden text?A: 6/2/2023 G4G1G7 G2G5G8 G3G6G9 Q: Where is the text “6/2/2023”A: Grid 5 Fusion Encoder in the documentA: < ‘Form’, [88.0, 1.0, 169.0, 21.0]><br>**----- End of picture text -----**<br>


Figure 2: Multimodal feature representation and fusion mechanisms. 

**209** into object-level features (e.g., paragraphs), facil- **210** itating downstream object-level contrastive learn- **211** ing. Instruct-Doc (Tanaka et al., 2024) introduces **212** an enhanced Q-Former (Li et al., 2023), termed **213** _Document Former_ , serving as a bridging module **214** that integrates visual, textual, and layout informa- **215** tion from document images into the LLM input **216** space via cross- and self-attention. In sum, exter- **217** nal encoders improve representations but require **218** additional pretraining and fine-tuning to align with **219** LLMs’ latent spaces. 

**220 Text as Training Objectives.** Some frameworks **221** rely exclusively on document images as input **222** to predict answers. Models such as mPLUG- **223** DocOwl (Ye et al., 2023a) and LLaVA-R (Zhang **224** et al., 2024d), built upon mPLUG-Owl (Ye et al., **225** 2023c), demonstrate strong OCR capabilities and **226** are further instruction-tuned on diverse VRDU **227** benchmarks. Other approaches incorporate text **228** recognition, detection, and spotting tasks (Wang **229** et al., 2023; Feng et al., 2023) to integrate text in- **230** formation. To better understand the hierarchical **231** structure of documents, Hu et al. (2024, 2025) pro- **232** pose a multi-grained text localization task span- **233** ning the word-to-block level. While these meth- **234** ods deliver robust results using only visual inputs, **235** they place heavy demands on pretraining and fine- **236** tuning. Additionally, high-resolution images are **237** often necessary to accommodate extremely long **238** visual sequences and to preserve fine-grained fea- **239** tures (Liu et al., 2024a; Yu et al., 2024a). 

## **3.2 Visual Modality** 

To integrate visual information, OCR-dependent frameworks use extracted text and coarse visual cues, thereby enabling the use of **lower-resolution** images. In contrast, OCR-free frameworks require direct text recognition, demanding fine-grained 

perception and **high-resolution** inputs. See the Appendix A.4 for input resolution details. 

**246 247** 

**Low Resolution Image Encoding.** Some **248** frameworks directly feed image patches into **249** pretrained vision encoders to obtain patch em- **250** beddings (Xie et al., 2025; Tanaka et al., 2024). **251** Others (Han et al., 2025; Luo et al., 2024; Liao **252** et al., 2024) employ pretrained VRDU models, **253** i.e., LayoutLMv3 (Huang et al., 2022), to extract **254** multimodal-enhanced visual embeddings. Due **255** to the limitations of low-resolution inputs in **256** capturing fine-grained details, recent works have **257** adopted dual-encoder architectures that process **258** both low- and medium-resolution images (Ye **259** et al., 2023b; Zhang et al., 2024c), followed by **260** visual feature compression techniques to man- **261** age the increased feature volume. While using **262** low-resolution images offers a straightforward **263** pathway to multimodal understanding, achieving **264** effective alignment often requires additional **265** pretraining and instruction tuning. Moreover, **266** the absence of fine-grained visual detail often **267** necessitates additional OCR tools to extract text **268** for accurate VRD interpretation. **269** 

**High Resolution Image Encoding.** To cap- **270** ture fine-grained level information for end-to-end **271** training and inference, many frameworks support **272** high-resolution image input. For ViT-style (Doso- **273** vitskiy et al., 2020) pretrained vision encoders, Hu **274** et al. (2024) splits high-resolution images into pre- **275** defined sub-images. To handle images of various **276** shapes, UReader (Ye et al., 2023b) introduces a **277** _Shape-Adaptive Cropping Module_ that adaptively **278** divides images into fixed-size sub-images using **279** grids of various shapes. However, the image crop- **280** ping may disrupt semantic continuity across sub- **281** images. To address this, Liu et al. (2024c) in- **282** 

4 

**283** troduced a _Shifted Window Attention_ to enhance **284** cross-sub-images connection via self-attention. In **285** short, high-resolution images support fine-grained **286** information extraction, but efficiently processing **287** the resulting large number of visual tokens re- **288** mains challenging, requiring a balance between **289** resource usage and the number of visual tokens. 

**305** 

**306 307 308 309 310** 

**290 Visual Feature Compression.** Yu et al. **291** (2024a,b) utilize Q-Former (Li et al., 2023), while **292** Liu et al. (2024c) adopts the _Resampler_ from **293** Qwen-VL (Wang et al., 2024b) to reduce the **294** number of visual tokens. Considering the layout- **295** aware nature of VRDs, Hu et al. (2024) introduces **296** a convolutional module that preserves layout by **297** compressing horizontal features and reducing the **298** number of tokens. It further enhances this with **299** layout-aware cross-attention to handle multi-page **300** input. Liu et al. (2024a) use a _Content Detector_ **301** to filter non-informative tokens by segmenting **302** text-rich regions, while Zhang et al. (2024a) **303** propose eliminating low-information areas and **304** clustering and aggregating the remaining features. 

## **3.3 Layout Modality** 

Unlike natural scene images, VRDs feature dense text and complex layout structures. Methods for encoding layout information can be categorized into positional encoding-based, prompt-based, and task-oriented approaches. 

**311 Positional Encoding.** OCR-dependent models **312** use OCR tools to extract textual and layout infor- **313** mation, combining text embeddings with 2D po- **314** sitional encodings (Xu et al., 2020) to incorpo- **315** rate layout into LLMs (Han et al., 2025; Tanaka **316** et al., 2024). However, these approaches re- **317** quire extra training for feature alignment. In con- **318** trast, Zhu et al. (2025a) assigns unique positional **319** embeddings to attention heads based on multi- **320** dimensional layout features without altering the **321** model architecture or requiring further pretrain- **322** ing. Wang et al. (2024a) treats layout as a sep- **323** arate modality and introduces disentangled spa- **324** tial attention for cross-modal interactions with- **325** out visual encoders. Zhu et al. (2025b) addresses **326** long-context inference limits by encoding layout **327** as a single token sharing the position with its text. **328** However, these methods implicitly integrate lay- **329** out information and rely heavily on large-scale **330** pretraining, resulting in high computational costs **331** and reduced effectiveness for tasks that demand **332** explicit layout understanding. 

**Layout as Prompt.** To integrate explicit layout **333** information, some frameworks include layout de- **334** tails in prompts alongside the user query and doc- **335** ument content. He et al. (2023) introduces an **336** in-context learning based approach to incorporate **337** layout-aware demonstrations into bounding box **338** representations. Lamott et al. (2024) and Perot **339** et al. (2024) encode layout into text sequence **340** through rule-based verbalization or quantized co- **341** ordinate tokens. These methods enable layout- **342** awareness without training. However, these meth- **343** ods increase input length, rely on LLMs to inter- **344** pret layout as text, and overlook visual cues essen- **345** tial for encoding relative positional information. **346** 

**Integrating During Training.** OCR-free frame- **347** works incorporate text by formulating recognition **348** and detection tasks that also aid in understanding **349** layout (Wang et al., 2023; Feng et al., 2023). To **350** further enhance this, some models (Wang et al., **351** 2025b; Zhang et al., 2024c) leverage layout-aware **352** pretraining tasks (Section 4.1) and layout-specific **353** instruction-tuning tasks, such as visual ground- **354** ing (Liu et al., 2024a,c) and table reconstruc- **355** tion (Liao et al., 2024). However, these meth- **356** ods typically require large-scale datasets for pre- **357** training or instruction tuning, leading to substan- **358** tial computational costs and data bottlenecks. **359** 

## **3.4 Multimodal Fusion** 

**360** We categorize multimodal fusion methods into **361** four types: direct, neural-based, task-oriented, and **362** prompt-based. Direct fusion relies on simple fea- **363** ture summation or concatenation with alignment **364** training, while this survey primarily focuses on the **365** latter three approaches in MLLM-based VRDU **366** frameworks. **367** 

**Neural-based Fusion.** The simplest multimodal **368** feature encoding uses external document encoders **369** such as LayoutLMv3 (Xu et al., 2021), which fuse **370** multimodal features via self- or cross-attention **371** and leverage pretraining knowledge. Wang et al. **372** (2024a) stands out by employing a layout-aware **373** transformer with disentangled attention over text **374** and spatial layouts, enabling effective document **375** understanding without requiring image encoders. **376** In OCR-free frameworks, visual encoders extract **377** visual cues, with adaptors like LoRA (Yu et al., **378** 2024b) or linear projectors (Zhang et al., 2024d; **379** Wang et al., 2023) mapping features into the lan- **380** guage space. Masry et al. (2025) propose a method **381** 

5 

**419** 

**423** 

**424 425 426 427 428 429** 

**382** that maps visual features to a weighted textual em- **383** bedding to reduce misalignment issues observed **384** in previous approaches. These neural-based fu- **385** sion methods benefit from dedicated encoders or **386** modified architectures, but often require extensive **387** pretraining or SFT and face challenges in scal- **388** ability, computational overhead, and adaptability **389** to diverse document layouts, especially in noisy **390** OCR scenarios. 

**391 Target-oriented Fusion.** Target-oriented strate- **392** gies establish multimodal connections through su- **393** pervised objectives that span the input-to-output **394** space (Hu et al., 2024) and are widely applied to **395** text and layout features in OCR-free frameworks. **396** For instance, in text recognition tasks, models **397** are trained to map visual features directly to text **398** and spatial coordinates, thereby aligning fusion **399** with task-specific goals. While these approaches **400** improve end-to-end multimodal integration, they **401** also increase demands on data preparation, anno- **402** tation quality, and training complexity in practice. 

**403 Prompt-based Fusion.** Prompts for multimodal **404** tasks may include text, images, and bounding **405** box coordinates. While many frameworks adopt **406** Layout-as-Prompt strategies to encode layout in- **407** formation, others use Chain-of-Thought (CoT) **408** reasoning to further enhance multimodal learning. **409** For example, Luo et al. (2024) utilizes a _Layout-_ **410** _CoT_ approach that divides reasoning into question **411** analysis, region localization, and answer genera- **412** tion, explicitly modeling spatial layout. Liao et al. **413** (2024) leverages CoT pretraining and CoT anneal- **414** ing to support layout-aware reasoning for VRDU. **415** However, these methods often depend on prede- **416** fined reasoning strategies, intermediate-step eval- **417** uations, and well-trained prior frameworks, limit- **418** ing their generalizability to unseen domains. 

## **4 Training Paradigms** 

**420** To facilitate multimodal understanding, instruc- **421** tion following, and domain adaptation, various **422** training tasks and strategies have been developed. 

## **4.1 Pretraining** 

To enhance mono- and multi-modal document understanding, VRDU frameworks adopt various self-supervised pretraining tasks, such as masked information modeling and cross-modality alignment (Ding et al., 2025b). OCR-dependent frameworks typically utilize pretrained VRDU mod- 

els or vision encoders to obtain enriched mul- **430** timodal representations. Some models propose **431** additional self-supervised learning tasks (e.g., Li **432** et al. (2024) applies object-level contrastive learn- **433** ing between visual and multimodal features). **434** Wang et al. (2024a) introduces a transformer **435** architecture with disentangled spatial-text atten- **436** tion to perform block-wise text infilling to en- **437** hance text-layout correlation modeling. OCR- **438** free frameworks (Zhang et al., 2024c; Hu et al., **439** 2024) focus on pretraining tasks like text recog- **440** nition, detection, and captioning to integrate text **441** and layout information. Hu et al. (2025) fur- **442** ther targets multi-page layout coherence. Feng **443** et al. (2024) aligns frequency features with LLMs **444** through text-centric pretraining. Although these **445** self-supervised tasks are effective in fusing mul- **446** timodal features and learning general knowledge, **447** they remain computationally intensive and often **448** lack instruction-based tuning, limiting their capac- **449** ity to follow real-world user instructions. **450 4.2 Instruction Tuning 451** To benefit task orientation in LLM-based frame- **452** works, many VRD approaches, following In- **453** structGPT (Ouyang et al., 2022), are trained on **454** instruction-response pairs to better align model **455** outputs with user prompts. Pretraining tasks such **456** as text reading, recognition, and image caption- **457** ing are reformulated as instruction-based formats, **458** with images paired with task descriptions. Be- **459** yond improving multimodal fusion, goal-oriented **460** tasks, including VRD question answering (Ding **461** et al., 2024b), key information extraction (Ding **462** et al., 2023), and VRD classification (Harley et al., **463** 2015), are conducted on large-scale datasets. For **464** better generalizability, some frameworks synthet- **465** ically generate large instruction-tuning datasets **466** (See Appendix B for more details). To fur- **467** ther improve localization and information extrac- **468** tion, Wang et al. (2023) and Feng et al. (2023) **469** propose predicting answers alongside bounding **470** boxes, thereby enhancing the framework’s reli- **471** ability. Instruction tuning not only strengthens **472** user query understanding but also boosts multi- **473** modal fusion. Instruction tuning on large-scale **474** datasets substantially enhances zero-shot perfor- **475** mance. However, the requirement for extensive **476** training data leads to substantial resource con- **477** sumption. Furthermore, synthetic datasets, often **478** generated with off-the-shelf OCR tools and LLMs, **479** may yield low-quality QA pairs, particularly in **480** 

6 

**481 482** 

**483** 

**524 525 526 527 528 529 530** 

low-resource domains such as scanned documents, thereby impacting zero-shot performance. 

## **4.3 Training Strategies** 

**484** MLLM-based document understanding frame- **485** works typically consist of multiple sub-modules to **486** encode multimodal information and are trained in **487** a stepwise manner. Few frameworks leverage in- **488** context learning (He et al., 2023) or multimodal **489** prompts (Perot et al., 2024) to develop training- **490** free architectures. The majority, however, involve **491** pretraining to capture general-domain knowledge, **492** followed by instruction tuning to improve inter- **493** pretation of user prompts. Furthermore, some **494** frameworks are subsequently **Supervised Fine495 Tuned** on benchmark datasets (Wang et al., 2024a; **496** Zhu et al., 2025a) or a synthetic set (Kim et al., **497** 2024) to enhance domain-specific adaptation. To **498** integrate multimodal information, these frame- **499** works mainly employ an LLM with various multi- **500** modal encoders (Han et al., 2025; Xie et al., 2025), **501** sometimes incorporating adaptors (Hu et al., 2024; **502** Lu et al., 2024) or linear projectors (Park et al., **503** 2024) for fusion or alignment. Depending on the **504** training stage, sub-modules may be either train- **505** able or frozen, balancing the acquisition of new **506** knowledge with the preservation of valuable infor- **507** mation from the original backbone. 

**508 LLM Backbone.** As most LLMs are exten- **509** sively pretrained on large-scale datasets and cap- **510** ture broad knowledge, many frameworks freeze **511** the LLM, using it solely to generate human- **512** understandable outputs. In frameworks involv- **513** ing pretraining or instruction tuning (Zhang et al., **514** 2024a; Liu et al., 2024a), freezing the LLM back- **515** bone helps preserve its knowledge and reduce **516** training costs. However, some approaches en- **517** able LLMs to be trained during continued pretrain- **518** ing (Zhu et al., 2025b) or instruction tuning (Liao **519** et al., 2024) to better capture VRD domain knowl- **520** edge and enhance multimodal alignment. In su- **521** pervised fine-tuning stages, the LLM backbone is **522** typically made trainable to adapt to the target do- **523** main (Zhang et al., 2024d). 

**Vision/Multimodal Encoders.** They are employed to encode multimodal features, which are subsequently aligned with LLM text representations by projectors or adaptors. Similar to LLM backbones, vision (Dosovitskiy et al., 2020), and multimodal encoders (Huang et al., 2022) are often kept frozen during pretraining to preserve 

learned knowledge (Yu et al., 2024b; Zhang et al., **531** 2024d). Feng et al. (2024) use a Swin Trans- **532** former to encode frequency-domain images, pre- **533** trained from scratch. To enhance multimodal fea- **534** ture learning, Li et al. (2024) make the ViT en- **535** coder trainable while freezing LayoutLMv3, en- **536** abling knowledge distillation via contrastive learn- **537** ing. During instruction tuning, vision encoders **538** are typically unfrozen to improve alignment and **539** task-specific adaptation (Zhang et al., 2024a; Liu **540** et al., 2024a). Conversely, in dual-encoder frame- **541** works, vision encoders with inputs at diverse res- **542** olutions are often frozen to enhance the represen- **543** tation of hierarchical inputs. In supervised fine- **544** tuning, there is no standard practice for encoder **545** trainability. **546 Projectors and Adaptors.** They play a crucial **547** role in feature alignment and lightweight tun- **548** ing. Projectors are typically employed to align vi- **549** sual or layout features with the LLM input space **550** (Park et al., 2024) and encode layout informa- **551** tion (Tanaka et al., 2024). These modules are **552** mainly trainable throughout the entire training **553** process. Adaptors, on the other hand, are designed **554** for efficient, task-specific tuning, often leveraging **555** LoRA-style updates (Ye et al., 2023a; Hu et al., **556** 2024) or cross-attention mechanisms (Liu et al., **557** 2024c; Yu et al., 2024a) to integrate multi-aspect **558** inputs with minimal parameter changes. Plug- **559** and-play components, such as visual abstractors **560** (Ye et al., 2023a) or compressors (Hu et al., 2025), **561** have also been introduced to reduce the dimen- **562** sionality of visual features. These adaptors are **563** usually trained during instruction tuning or during **564** supervised fine-tuning. **565 4.4 Training Datasets Overview 566** Diverse datasets are required to meet specific **567** training objectives. Pretraining typically leverages **568** large-scale cross-domain VRD collections to re- **569** duce domain gaps and enhance multimodal fusion, **570** sometimes requiring more domain-specific data **571** (e.g., medical, slides) to improve domain aware- **572** ness. For instruction tuning, synthetic datasets **573** are often used to strengthen instruction-following **574** and reasoning abilities, particularly in OCR-free **575** frameworks that generate instruction-aligned OCR **576** or layout understanding tasks. Additionally, SFT **577** is commonly applied on original or post-processed **578** benchmark datasets (e.g., converting key-value **579** pairs into QA format) to further boost perfor- **580** 

7 

**581** 

**582** 

**610** 

**625 626 627 628** 

mance. For more dataset details, see Appendix C. 

## **5 Inference Prompt Setting** 

**583** MLLM-based frameworks adopt diverse prompt **584** formats depending on their architecture. For **585** OCR-free frameworks in Table 1, the prompt typ- **586** ically includes a document image, occasionally **587** multiple pages (Hu et al., 2025; Wang et al., **588** 2025b), alongside a textual user query. Some **589** frameworks not only predict answers to user **590** queries but also localize bounding boxes, often **591** requiring an additional prompt for localization **592** (Wang et al., 2023; Feng et al., 2023). OCR- **593** dependent frameworks first preprocess input us- **594** ing off-the-shelf tools to extract textual and lay- **595** out information. Vision-free models (He et al., **596** 2023; Wang et al., 2024a) process only the ex- **597** tracted content alongside the query. In contrast, **598** vision-dependent models also incorporate the doc- **599** ument image into the vision (Xie et al., 2025) **600** or into multimodal encoders (Liao et al., 2024), **601** aligning visual and textual features for the final **602** prediction. Furthermore, some frameworks inte- **603** grate layout information into prompts via bound- **604** ing boxes (Zhu et al., 2025a) or markdown-style **605** formatting. The inference strategies are closely **606** tied to the model architecture and reflect a grow- **607** ing trend toward unified, multimodal understand- **608** ing and layout-aware reasoning to improve docu- **609** ment comprehension accuracy and versatility. 

## **6 Challenges and Future Direction** 

**611 Synthetic Data.** Acquiring high-quality, manu- **612** ally curated datasets for new document collections **613** is often quite costly. Leveraging synthetically gen- **614** erated datasets offers a cost-effective alternative **615** for adapting to the target domain (Ding et al., **616** 2025a). For large-scale instruction-tuning, many **617** frameworks generate instruction-response pairs **618** using benchmarks, templates, or LLMs. However, **619** these synthetic datasets often lack validation, re- **620** sulting in low-quality or inaccurate pairs. Since **621** synthetic data may not fully capture real user in- **622** put, future research should prioritize human-in- **623** the-loop and reinforcement learning approaches to **624** improve authenticity and task relevance. 

**Long Document Understanding.** In practice, VRDs frequently span multiple pages; however, most existing frameworks are tailored for singlepage inputs. Multi-page approaches typically rely 

on retrievers to identify relevant pages, which are **629** then processed by MLLM-based VRDU systems. **630** These methods often fall short of capturing se- **631** mantic and logical dependencies among document **632** entities, resulting in incomplete contextual under- **633** standing. Furthermore, handling long input se- **634** quences remains challenging, as existing multi- **635** page benchmarks focus mainly on extractive tasks **636** and rarely support complex multi-hop or multi- **637** modal reasoning. **638 Multilingual VRDU.** Most existing models and **639** benchmarks remain heavily English-centric, lim- **640** iting their generalization to documents with di- **641** verse languages and layouts. This bias is further **642** amplified by large-scale pretraining corpora that **643** predominantly reflect English document struc- **644** tures, leading to performance degradation in low- **645** resource settings. Although few multilingual **646** datasets have been proposed (Xu et al., 2022; **647** Chen et al., 2025), future research should explore **648** more multilingual and culturally diverse bench- **649** marks, language-agnostic representation learning, **650** and hybrid approaches to mitigate linguistic bias **651** to handle real-world document diversity. **652 Effective RAG Framework.** While RAG has **653** become a common paradigm (Jain et al., 2025; **654** Faysse et al., 2025), existing approaches often **655** exhibit brittle retrieval due to layout ambiguity **656** and misaligned multimodal embeddings, leading **657** to unreliable evidence selection. Moreover, most **658** RAG pipelines decouple retrieval from reasoning **659** and remain largely text-centric, limiting their abil- **660** ity to capture spatial and visual semantics in com- **661** plex documents. Future work should explore mul- **662** timodal RAG frameworks that support iterative **663** reasoning and dynamic evidence refinement, and **664** enable more robust and interpretable VRDU. **665 Agentic LLM in VRDU.** Recent works (Han **666** et al., 2025; Sun et al., 2025) incorporate exter- **667** nal tools (e.g., PDF parsers or retrievers) to gener- **668** ate intermediate outputs, enhancing both the accu- **669** racy and interpretability of practical VRDU appli- **670** cations. However, future research should explore **671** a wider variety of agent types and architectural in- **672** novations to enable automatic handling of diverse **673** formats, cross-domain scenarios, and fine-grained **674** elements such as charts and tables. Additionally, **675** challenges in agentic AI, such as multi-agent co- **676** ordination and knowledge conflicts, remain signif- **677** icant barriers to broader adoption for VRDU. **678** 

8 

**679** 

**695** 

**700** Galal M Binmakhashen and Sabri A Mahmoud. 2019. **701** Document layout analysis: a comprehensive survey. **702** _ACM Computing Surveys (CSUR)_ , 52(6):1–36. 

## **Limitations** 

**680** While this survey offers a comprehensive **681** overview of MLLM-based VRDU research, our **682** analysis is necessarily qualitative. It does not **683** provide exhaustive head-to-head comparisons, as **684** the field’s rapid evolution and breadth prioritize **685** trend summarization over detailed benchmarking. **686** Although academic advances are thoroughly **687** reviewed, discussion of real-world deployments **688** and industrial challenges remains limited, in **689** part because many practical applications are **690** proprietary and unpublished. In future work, we **691** aim to provide more quantitative meta-analyses, **692** incorporate insights from industrial adoption, and **693** continuously update the survey to capture the **694** latest developments as the field progresses. 

## **References** 

- **696** Camille Barboule, Benjamin Piwowarski, and Yoan **697** Chabot. 2025. Survey on question answering over **698** visually rich documents: Methods, challenges, and **699** trends. _arXiv preprint arXiv:2501.02235_ . 

- **703** Davide Caffagni, Federico Cocchi, Luca Barsellotti, **704** Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi, **705** Marcella Cornia, and Rita Cucchiara. 2024. The **706** revolution of multimodal large language models: A **707** survey. In _Findings of the Association for Computa-_ **708** _tional Linguistics: ACL 2024_ , pages 13590–13618. 

- **709** Ketong Chen, Yuhao Chen, and Yang Xue. 2025. Mo- **710** saicdoc: A large-scale bilingual benchmark for vi- **711** sually rich document understanding. _arXiv preprint_ **712** _arXiv:2511.09919_ . 

- **713** Wenhu Chen, Han Zhu, Wenhao Wang, Kai-Wei **714** Chang, William Yang Zhang, and William Wang. **715** 2020. Tabfact: A large-scale dataset for table-based **716** fact verification. In _International Conference on_ **717** _Learning Representations (ICLR)_ . 

- **718** Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo **719** Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, **720** Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scal- **721** ing up vision foundation models and aligning for **722** generic visual-linguistic tasks. In _Proceedings of the_ **723** _IEEE/CVF Conference on Computer Vision and Pat-_ **724** _tern Recognition_ , pages 24185–24198. 

- **725** Julien Delaunay, Hanh Thi Hong Tran, Carlos- **726** Emiliano Gonz´alez-Gallardo, Georgeta Bordea, **727** Nicolas Sidere, and Antoine Doucet. 2023. A com- **728** prehensive survey of document-level relation extrac- **729** tion (2016-2023). _arXiv preprint arXiv:2309.16396_ . 

- Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong- **730** Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun **731** Song, Bo Zheng, and Cheng-Lin Liu. 2025. Long- **732** DocURL: a comprehensive multimodal long docu- **733** ment benchmark integrating understanding, reason- **734** ing, and locating. In _Proceedings of the 63rd An-_ **735** _nual Meeting of the Association for Computational_ **736** _Linguistics (Volume 1: Long Papers)_ , pages 1135– **737** 1159, Vienna, Austria. Association for Computa- **738** tional Linguistics. **739** 

- Timo I Denk and Christian Reisswig. 2019. Bertgrid: **740** Contextualized embedding for 2d document repre- **741** sentation and understanding. In _Workshop on Docu-_ **742** _ment Intelligence at NeurIPS 2019_ . **743** 

- Yihao Ding, Soyeon Caren Han, Yanbei Jiang, Yan **744** Li, Zechuan Li, and Yifan Peng. 2025a. Syn- **745** doc: A hybrid discriminative-generative frame- **746** work for enhancing synthetic domain-adaptive doc- **747** ument key information extraction. _arXiv preprint_ **748** _arXiv:2509.23273_ . **749** 

- Yihao Ding, Soyeon Caren Han, Jean Lee, and Eduard **750** Hovy. 2025b. Deep learning based visually rich **751** document content understanding: A survey. _arXiv_ **752** _preprint arXiv:2408.01287_ . **753** 

- Yihao Ding, Siqu Long, Jiabin Huang, Kaixuan Ren, **754** Xingxiang Luo, Hyunsuk Chung, and Soyeon Caren **755** Han. 2023. Form-nlu: Dataset for the form natu- **756** ral language understanding. In _Proceedings of the_ **757** _46th International ACM SIGIR Conference on Re-_ **758** _search and Development in Information Retrieval_ , **759** pages 2807–2816. ACM. **760** 

- Yihao Ding, Kaixuan Ren, Jiabin Huang, Siwen Luo, **761** and Soyeon Caren Han. 2024a. Mmvqa: A com- **762** prehensive dataset for investigating multipage mul- **763** timodal information retrieval in pdf-based visual **764** question answering. In _Proceedings of the Thirty-_ **765** _Third International Joint Conference on Artificial_ **766** _Intelligence, IJCAI_ , pages 3–9. ijcai.org. **767** 

- Yihao Ding, Kaixuan Ren, Jiabin Huang, Siwen **768** Luo, and Soyeon Caren Han. 2024b. Mvqa: A **769** dataset for multimodal information retrieval in pdf- **770** based visual question answering. _arXiv preprint_ **771** _arXiv:2404.12720_ . **772** 

- Alexey Dosovitskiy, Lucas Beyer, Alexander **773** Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, **774** Thomas Unterthiner, Mostafa Dehghani, Matthias **775** Minderer, G Heigold, S Gelly, et al. 2020. An im- **776** age is worth 16x16 words: Transformers for image **777** recognition at scale. In _International Conference on_ **778** _Learning Representations_ . **779** 

- Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Om- **780** rani, Gautier Viaud, C´eline Hudelot, and Pierre **781** Colombo. 2025. Colpali: Efficient document re- **782** trieval with vision language models. In _ICLR_ . **783** 

- Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang **784** Zhou, Houqiang Li, and Can Huang. 2024. Doc- **785** pedia: Unleashing the power of large multimodal **786** 

9 

**790** Hao Feng, Zijian Wang, Jingqun Tang, Jinghui Lu, **791** Wengang Zhou, Houqiang Li, and Can Huang. 2023. **792** Unidoc: A universal large multimodal model for si- **793** multaneous text detection, recognition, spotting and **794** understanding. _arXiv preprint arXiv:2308.11592_ . 

**795** Tongkun Guan, Zining Wang, Pei Fu, Zhengtao Guo, **796** Wei Shen, Kai Zhou, Tiezhu Yue, Chen Duan, Hao **797** Sun, Qianyi Jiang, et al. 2025. A token-level text im- **798** age foundation model for document understanding. **799** _arXiv preprint arXiv:2503.02304_ . 

**806** Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, **807** Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: **808** A multi-modal multi-agent framework for document **809** understanding. 

**787** model in the frequency domain for versatile doc- **788** ument understanding. _Science China Information_ **789** _Sciences_ , 67(12):1–14. 

- **800** Pranay Gupta, Minesh Mathew, C.V. Jawahar, and **801** Marcus Liwicki. 2022. Infovqa: Visual question **802** answering on infographics with a multi-modal en- **803** tity graph. In _Proceedings of the IEEE/CVF Win-_ **804** _ter Conference on Applications of Computer Vision_ **805** _(WACV)_ . 

- **810** Adam W Harley, Alex Ufkes, and Konstantinos G Der- **811** panis. 2015. Evaluation of deep convolutional nets **812** for document image classification and retrieval. In **813** _2015 13th International Conference on Document_ **814** _Analysis and Recognition (ICDAR)_ , pages 991–995. **815** IEEE. 

- **816** Jiabang He, Lei Wang, Yi Hu, Ning Liu, Hui Liu, Xing **817** Xu, and Heng Tao Shen. 2023. Icl-d3ie: In-context **818** learning with diverse demonstrations updating for **819** document information extraction. In _Proceedings_ **820** _of the IEEE/CVF International Conference on Com-_ **821** _puter Vision_ , pages 19485–19494. IEEE. 

- **822** Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok **823** Hwang, Daehyun Nam, and Sungrae Park. 2022. **824** Bros: A pre-trained language model focusing on text **825** and layout for better key information extraction from **826** documents. In _Proceedings of the AAAI Conference_ **827** _on Artificial Intelligence_ , pages 10767–10775. 

- **828** Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang **829** Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, **830** and Jingren Zhou. 2024. mplug-docowl 1.5: Uni- **831** fied structure learning for ocr-free document under- **832** standing. In _Findings of the Association for Com-_ **833** _putational Linguistics: EMNLP 2024_ , pages 3096– **834** 3120. 

- **835** Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming **836** Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren **837** Zhou. 2025. mplug-docowl2: High-resolution com- **838** pressing for ocr-free multi-page document under- **839** standing. In _Proceedings of the 63rd Annual Meet-_ **840** _ing of the Association for Computational Linguistics_ **841** _(Volume 1: Long Papers)_ , pages 5817–5834. 

- Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and **842** Furu Wei. 2022. Layoutlmv3: Pre-training for doc- **843** ument ai with unified text and image masking. In **844** _Proceedings of the 30th ACM International Confer-_ **845** _ence on Multimedia_ , pages 4083–4091. ACM. **846** 

- Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Di- **847** mosthenis Karatzas, Shijian Lu, and CV Jawahar. **848** 2019. Icdar2019 competition on scanned receipt ocr **849** and information extraction. In _2019 International_ **850** _Conference on Document Analysis and Recognition_ **851** _(ICDAR)_ , pages 1516–1520. IEEE. **852** 

- Chelsi Jain, Yiran Wu, Yifan Zeng, Jiale Liu, Shengyu **853** Dai, Zhenwen Shao, Qingyun Wu, and Huazheng **854** Wang. 2025. SimpleDoc: Multi-Modal document **855** understanding with Dual-Cue page retrieval and it- **856** erative refinement. In _Proceedings of the 2025 Con-_ **857** _ference on Empirical Methods in Natural Language_ **858** _Processing_ , pages 28398–28415, Suzhou, China. **859** Association for Computational Linguistics. **860** 

- Guillaume Jaume, Hazim Kemal Ekenel, and Jean- **861** Philippe Thiran. 2019. Funsd: A dataset for form **862** understanding in noisy scanned documents. In _2019_ **863** _International Conference on Document Analysis and_ **864** _Recognition Workshops (ICDARW)_ , volume 2, pages **865** 1–6. IEEE. **866** 

- Anoop R Katti, Christian Reisswig, Cordula Guder, Se- **867** bastian Brarda, Steffen Bickel, Johannes H¨ohne, and **868** Jean Baptiste Faddoul. 2018. Chargrid: Towards un- **869** derstanding 2d documents. In _Proceedings of the_ **870** _2018 Conference on Empirical Methods in Natural_ **871** _Language Processing_ , pages 4459–4469. Associa- **872** tion for Computational Linguistics. **873** 

- Geewook Kim, Teakgyu Hong, Moonbin Yim, **874** JeongYeon Nam, Jinyoung Park, Jinyeong Yim, **875** Wonseok Hwang, Sangdoo Yun, Dongyoon Han, **876** and Seunghyun Park. 2022. Ocr-free document **877** understanding transformer. In _Computer Vision–_ **878** _ECCV 2022: 17th European Conference, Tel Aviv,_ **879** _Israel, October 23–27, 2022, Proceedings, Part_ **880** _XXVIII_ , pages 498–517. Springer. **881** 

- Sungnyun Kim, Haofu Liao, Srikar Appalaraju, Peng **882** Tang, Zhuowen Tu, Ravi Kumar Satzoda, R Man- **883** matha, Vijay Mahadevan, and Stefano Soatto. 2024. **884** Dockd: Knowledge distillation from llms for open- **885** world document understanding models. In _Proceed-_ **886** _ings of the 2024 Conference on Empirical Methods_ **887** _in Natural Language Processing, EMNLP 2024, Mi-_ **888** _ami, FL, USA, November 12-16, 2024_ , pages 3167– **889** 3193. Association for Computational Linguistics. **890** 

- Marcel Lamott, Yves-Noel Weweler, Adrian Ulges, **891** Faisal Shafait, Dirk Krechel, and Darko Obradovic. **892** 2024. Lapdoc: Layout-aware prompting for docu- **893** ments. In _International Conference on Document_ **894** _Analysis and Recognition_ , pages 142–159. Springer. **895** 

- David D Lewis, Gady Agam, Shlomo Argamon, Ophir **896** Frieder, David Grossman, and James Heard. 2006. **897** Building a test collection for complex document in- **898** formation processing. In _Proceedings of the 29th_ **899** 

10 

Zhang, Kun Yao, Errui Ding, et al. 2024. Struc- **954** textv3: An efficient vision-language model for text- **955** rich image perception, comprehension, and beyond. **956** _arXiv preprint arXiv:2405.21013_ . **957** 

**900** _annual international ACM SIGIR conference on Re-_ **901** _search and development in information retrieval_ , **902** pages 665–666. 

- **903** Junnan Li, Dongxu Li, Silvio Savarese, and Steven **904** Hoi. 2023. Blip-2: Bootstrapping language-image **905** pre-training with frozen image encoders and large **906** language models. In _International Conference on_ **907** _Machine Learning (ICML)_ . 

- **908** Xin Li, Yunfei Wu, Xinghua Jiang, Zhihao Guo, Ming- **909** ming Gong, Haoyu Cao, Yinsong Liu, Deqiang **910** Jiang, and Xing Sun. 2024. Enhancing visual doc- **911** ument understanding with contrastive learning in **912** large visual-language models. In _Proceedings of the_ **913** _IEEE/CVF Conference on Computer Vision and Pat-_ **914** _tern Recognition_ , pages 15546–15555. 

- **915** Wenhui Liao, Jiapeng Wang, Hongliang Li, Chengyu **916** Wang, Jun Huang, and Lianwen Jin. 2024. Do- **917** clayllm: An efficient and effective multi-modal **918** extension of large language models for text- **919** rich document understanding. _arXiv preprint_ **920** _arXiv:2408.15045_ . 

- **921** Chaohu Liu, Kun Yin, Haoyu Cao, Xinghua Jiang, Xin **922** Li, Yinsong Liu, Deqiang Jiang, Xing Sun, and Linli **923** Xu. 2024a. Hrvda: High-resolution visual docu- **924** ment assistant. In _Proceedings of the IEEE/CVF_ **925** _conference on computer vision and pattern recog-_ **926** _nition_ , pages 15534–15545. 

- **927** Haotian Liu, Chunyuan Li, Qingyang Wu, and **928** Yong Jae Lee. 2024b. Visual instruction tuning. _Ad-_ **929** _vances in neural information processing systems_ , 36. **930** Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin **931** Ma, Shuo Zhang, and Xiang Bai. 2024c. Textmon- **932** key: An ocr-free large multimodal model for under- **933** standing document. 

- **934** Jinghui Lu, Haiyang Yu, Yanjie Wang, Yongjie Ye, **935** Jingqun Tang, Ziwei Yang, Binghong Wu, Qi Liu, **936** Hao Feng, Han Wang, et al. 2024. A bounding box **937** is worth one token: Interleaving layout and text in a **938** large language model for document understanding. **939** _arXiv preprint arXiv:2407.01976_ . 

- **940** Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, **941** Zhi Yu, and Cong Yao. 2024. Layoutllm: Layout **942** instruction tuning with large language models for **943** document understanding. In _IEEE/CVF Conference_ **944** _on Computer Vision and Pattern Recognition, CVPR_ **945** _2024, Seattle, WA, USA, June 16-22, 2024_ , pages **946** 15630–15640. IEEE. 

- **947** Tengchao Lv, Yupan Huang, Jingye Chen, Yuzhong **948** Zhao, Yilin Jia, Lei Cui, Shuming Ma, Yaoyao **949** Chang, Shaohan Huang, Wenhui Wang, et al. 2023. **950** Kosmos-2.5: A multimodal literate model. _arXiv_ **951** _preprint arXiv:2309.11419_ . 

- **952** Pengyuan Lyu, Yulin Li, Hao Zhou, Weihong Ma, **953** Xingyu Wan, Qunyi Xie, Liang Wu, Chengquan 

- Ahmed Masry, Juan A Rodriguez, Tianyu Zhang, **958** Suyuchen Wang, Chao Wang, Aarash Feizi, Ak- **959** shay Kalkunte Suresh, Abhay Puri, Xiangru Jian, **960** Pierre-Andre Noel, et al. 2025. Alignvlm: Bridg- **961** ing vision and language latent spaces for multimodal **962** document understanding. In _The Thirty-ninth An-_ **963** _nual Conference on Neural Information Processing_ **964** _Systems_ . **965** 

- Minesh Mathew, Dimosthenis Karatzas, and CV Jawa- **966** har. 2021. Docvqa: A dataset for vqa on docu- **967** ment images. In _Proceedings of the IEEE/CVF win-_ **968** _ter conference on applications of computer vision_ , **969** pages 2200–2209. IEEE. **970** 

- Oshri Naparstek, Roi Pony, Inbar Shapira, Foad Abo **971** Dahood, Ophir Azulai, Yevgeny Yaroker, Nadav Ru- **972** binstein, Maksym Lysak, Peter Staar, Ahmed Nas- **973** sar, Nikolaos Livathinos, Christoph Auer, Elad Am- **974** rani, Idan Friedman, Orit Prince, Yevgeny Bur- **975** shtein, Adi Raz Goldfarb, and Udi Barzelay. 2024. **976** Kvp10k : A comprehensive dataset for key-value **977** pair extraction in business documents. **978** 

- Feng Ni, Kui Huang, Yao Lu, Wenyu Lv, Guanzhong **979** Wang, Zeyu Chen, and Yi Liu. 2025. Pp- **980** docbee: Improving multimodal document under- **981** standing through a bag of tricks. **982** 

- Eri Onami, Shuhei Kurita, Taiki Miyanishi, and Taro **983** Watanabe. 2024. Jdocqa: Japanese document ques- **984** tion answering dataset for generative language mod- **985** els. **986** 

- OpenAI. 2024. Hello gpt-4o. https://openai. **987** com/index/hello-gpt-4o/. **988** 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, **989** Carroll Wainwright, Pamela Mishkin, Chong Zhang, **990** Sandhini Agarwal, Katarina Slama, Alex Ray, et al. **991** 2022. Training language models to follow instruc- **992** tions with human feedback. _Advances in neural in-_ **993** _formation processing systems_ , 35:27730–27744. **994** 

- Jaeyoo Park, Jin Young Choi, Jeonghyung Park, and **995** Bohyung Han. 2024. Hierarchical visual feature ag- **996** gregation for ocr-free document understanding. _Ad-_ **997** _vances in Neural Information Processing Systems_ , **998** 37:105972–105996. **999** 

- Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, **1000** Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. **1001** 2019. Cord: a consolidated receipt dataset for post- **1002** ocr parsing. In _Workshop on Document Intelligence_ **1003** _at NeurIPS 2019_ . **1004** 

- Panupong Pasupat and Percy Liang. 2015. Composi- **1005** tional semantic parsing on semi-structured tables. In **1006** _Proceedings of the 53rd Annual Meeting of the As-_ **1007** _sociation for Computational Linguistics (ACL)_ . **1008** 

11 

**1016** Minenobu Seki, Masakazu Fujio, Takeshi Nagasaki, **1017** Hiroshi Shinjo, and Katsumi Marukawa. 2007. In- **1018** formation management system using structure anal- **1019** ysis of paper/electronic documents and its applica- **1020** tions. In _Ninth International Conference on Docu-_ **1021** _ment Analysis and Recognition (ICDAR 2007)_ , vol- **1022** ume 2, pages 689–693. IEEE. 

**1023** Yufan Shen, Chuwei Luo, Zhaoqing Zhu, Yang Chen, **1024** Qi Zheng, Zhi Yu, Jiajun Bu, and Cong Yao. 2025. **1025** Proctag: Process tagging for assessing the efficacy **1026** of document instruction data. 

**1033** ˇStˇep´an ˇSimsa, Milan ˇSulc, Michal Uˇriˇc´aˇr, Yash Patel, **1034** Ahmed Hamdi, Matˇej Koci´an, Maty´aˇs Skalick`y, Jiˇr´ı **1035** Matas, Antoine Doucet, Micka¨el Coustaty, and Di- **1036** mosthenis Karatzas. 2023. DocILE benchmark for **1037** document information localization and extraction. 

**1056** Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu **1057** Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri **1058** Faiz Bin Mahmood, Hao Feng, Zhen Zhao, Yangfan **1059** He, Kuan Lu, Yanjie Wang, Yuliang Liu, Hao Liu, **1060** Xiang Bai, and Can Huang. 2025. Mtvqa: Bench- **1061** marking multilingual text-centric visual question an- **1062** swering. 

**1063** Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Bur- **1064** nell, Libin Bai, and et al. 2024. Gemini 1.5: Unlock- **1065** ing multimodal understanding across millions of to- **1066** kens of context. 

- **1009** Vincent Perot, Kai Kang, Florian Luisier, Guolong Su, **1010** Xiaoyu Sun, Ramya Sree Boppana, Zilong Wang, **1011** Zifeng Wang, Jiaqi Mu, Hao Zhang, et al. 2024. **1012** Lmdx: Language model-based document informa- **1013** tion extraction and localization. In _Findings of_ **1014** _the Association for Computational Linguistics ACL_ **1015** _2024_ , pages 15140–15168. 

   - Rub`en Tito, Dimosthenis Karatzas, and Ernest Val- **1067** veny. 2023. Hierarchical multimodal transform- **1068** ers for multipage docvqa. _Pattern Recognition_ , **1069** 144:109834. **1070** 

   - Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier **1071** Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, **1072** Baptiste Rozi`ere, Naman Goyal, Eric Hambro, **1073** Faisal Azhar, et al. 2023. Llama: Open and effi- **1074** cient foundation language models. _arXiv preprint_ **1075** _arXiv:2302.13971_ . **1076** 

   - Jordy Van Landeghem, Rub`en Tito, Łukasz Borch- **1077** mann, Michał Pietruszka, Pawel Joziak, Rafal **1078** Powalski, Dawid Jurkiewicz, Micka¨el Coustaty, **1079** Bertrand Anckaert, Ernest Valveny, et al. 2023. **1080** Document understanding dataset and evaluation **1081** (dude). In _Proceedings of the IEEE/CVF Inter-_ **1082** _national Conference on Computer Vision_ , pages **1083** 19528–19540. **1084** 

- **1027** Maxim Sidorov, Amanpreet Singh, Yu Li, Jianfeng **1028** Liao, Ming Liao, Yaxing Wang, Lichao Wang, **1029** Shouling Gong, Chen Change Loy, and Xiang Bai. **1030** 2020. Textcaps: A dataset for image captioning with **1031** reading. In _Proceedings of the European Confer-_ **1032** _ence on Computer Vision (ECCV)_ . 

- **1038** Amanpreet Singh, Vedanuj Natarajan, Yu Jiang, Xinlei **1039** Chen, Meet Shah, Marcus Rohrbach, Dhruv Batra, **1040** Devi Parikh, and Aniruddha Krishnamurthy. 2019. **1041** Textvqa: Visual question answering with reading. In **1042** _Proceedings of the IEEE/CVF Conference on Com-_ **1043** _puter Vision and Pattern Recognition (CVPR)_ . 

- **1044** Li Sun, Liu He, Shuyue Jia, Yangfan He, and Chenyu **1045** You. 2025. Docagent: An agentic framework for **1046** multi-modal long-context document understanding. **1047** In _Proceedings of the 2025 Conference on Empiri-_ **1048** _cal Methods in Natural Language Processing_ , pages **1049** 17712–17727. 

- **1050** Ryota Tanaka, Taichi Iki, Kyosuke Nishida, Kuniko **1051** Saito, and Jun Suzuki. 2024. Instructdoc: A dataset **1052** for zero-shot generalization of visual document un- **1053** derstanding with instructions. In _Proceedings of_ **1054** _the AAAI conference on artificial intelligence_ , pages **1055** 19071–19079. AAAI Press. 

- Dongsheng Wang, Natraj Raman, Mathieu Sibue, **1085** Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong **1086** Pei, Armineh Nourbakhsh, and Xiaomo Liu. 2024a. **1087** Docllm: A layout-aware generative language model **1088** for multimodal document understanding. In _Pro-_ **1089** _ceedings of the 62nd Annual Meeting of the Associa-_ **1090** _tion for Computational Linguistics (Volume 1: Long_ **1091** _Papers)_ , pages 8529–8548. Association for Compu- **1092** tational Linguistics. **1093** 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi- **1094** hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, **1095** Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, **1096** Mengfei Du, Xuancheng Ren, Rui Men, Dayi- **1097** heng Liu, Chang Zhou, Jingren Zhou, and Junyang **1098** Lin. 2024b. Qwen2-vl: Enhancing vision-language **1099** model’s perception of the world at any resolution. **1100** _arXiv preprint arXiv:2409.12191_ . **1101** 

- Yonghui Wang, Wengang Zhou, Hao Feng, Keyi **1102** Zhou, and Houqiang Li. 2023. Towards im- **1103** proving document understanding: An exploration **1104** on text-grounding via mllms. _arXiv preprint_ **1105** _arXiv:2311.13194_ . **1106** 

- Zhaowei Wang, Wenhao Yu, Xiyu Ren, Jipeng Zhang, **1107** Yu Zhao, Rohit Saxena, Liang Cheng, Ginny Wong, **1108** Simon See, Pasquale Minervini, Yangqiu Song, and **1109** Mark Steedman. 2025a. Mmlongbench: Bench- **1110** marking long-context vision-language models effec- **1111** tively and thoroughly. **1112** 

- Zining Wang, Tongkun Guan, Pei Fu, Chen Duan, **1113** Qianyi Jiang, Zhentao Guo, Shan Guo, Junfeng Luo, **1114** Wei Shen, and Xiaokang Yang. 2025b. Marten: Vi- **1115** sual question answering with mask generation for **1116** multi-modal document understanding. **1117** 

- Toyohide Watanabe, Qin Luo, and Noboru Sugie. 1995. **1118** Layout recognition of multi-kinds of table-form doc- **1119** uments. _IEEE Transactions on Pattern Analysis and_ **1120** _Machine Intelligence_ , 17(4):432–445. **1121** 

12 

**1128** Kun Xiang, Heng Li, Terry Jingchen Zhang, Yinya **1129** Huang, Zirong Liu, Peixin Qu, Jixi He, Jiaqi Chen, **1130** Yu-Jie Yuan, Jianhua Han, Hang Xu, Hanhui Li, **1131** Mrinmaya Sachan, and Xiaodan Liang. 2025. Seep- **1132** hys: Does seeing help thinking? – benchmarking **1133** vision-based physics reasoning. 

**1134** Xudong Xie, Hao Yan, Liang Yin, Yang Liu, Jing Ding, **1135** Minghui Liao, Yuliang Liu, Wei Chen, and Xiang **1136** Bai. 2025. Pdf-wukong: A large multimodal model **1137** for efficient long pdf reading with end-to-end sparse **1138** sampling. 

**1168** Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jian- **1169** qiang Wan, Humen Zhong, Xuejing Liu, Mingkun **1170** Yang, Peng Wang, Shuai Bai, LianWen Jin, and Jun- **1171** yang Lin. 2024. Cc-ocr: A comprehensive and chal- **1172** lenging ocr benchmark for evaluating large multi- **1173** modal models in literacy. 

- **1122** Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, **1123** Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, **1124** and Xiangyu Zhang. 2024. Vary: Scaling up the **1125** vision vocabulary for large vision-language model. **1126** In _European Conference on Computer Vision_ , pages **1127** 408–424. Springer. 

- **1139** Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu **1140** Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha **1141** Zhang, Wanxiang Che, et al. 2021. Layoutlmv2: **1142** Multi-modal pre-training for visually-rich document **1143** understanding. In _Proceedings of the 59th Annual_ **1144** _Meeting of the Association for Computational Lin-_ **1145** _guistics and the 11th International Joint Conference_ **1146** _on Natural Language Processing (Volume 1: Long_ **1147** _Papers)_ , pages 2579–2591. Association for Compu- **1148** tational Linguistics. 

- **1149** Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, **1150** Furu Wei, and Ming Zhou. 2020. Layoutlm: Pre- **1151** training of text and layout for document image **1152** understanding. In _Proceedings of the 26th ACM_ **1153** _SIGKDD International Conference on Knowledge_ **1154** _Discovery & Data Mining_ , pages 1192–1200. ACM. 

- **1155** Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yi- **1156** juan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. **1157** 2022. Xfund: A benchmark dataset for multilingual **1158** visually rich form understanding. In _Findings of_ **1159** _the association for computational linguistics: ACL_ **1160** _2022_ , pages 3214–3224. 

- **1161** Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, **1162** Daniel Kifer, and C Lee Giles. 2017. Learning **1163** to extract semantic structure from documents using **1164** multimodal fully convolutional neural networks. In **1165** _Proceedings of the IEEE Conference on Computer_ **1166** _Vision and Pattern Recognition_ , pages 5315–5324. **1167** IEEE Computer Society. 

- **1174** Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming **1175** Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chen- **1176** liang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei **1177** Huang. 2023a. mplug-docowl: Modularized mul- **1178** timodal large language model for document under- **1179** standing. 

- Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, **1180** Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, **1181** Qi Qian, Ji Zhang, et al. 2023b. Ureader: Univer- **1182** sal ocr-free visually-situated language understand- **1183** ing with multimodal large language model. In _Find-_ **1184** _ings of the Association for Computational Linguis-_ **1185** _tics: EMNLP 2023_ , pages 2841–2858. **1186** 

- Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, **1187** Ming Yan, Yiyang Zhou, Junyang Wang, An- **1188** wen Hu, Pengcheng Shi, Yaya Shi, et al. 2023c. **1189** mplug-owl: Modularization empowers large lan- **1190** guage models with multimodality. _arXiv preprint_ **1191** _arXiv:2304.14178_ . **1192** 

- Wenwen Yu, Zhibo Yang, Yuliang Liu, and Xiang **1193** Bai. 2025. Docthinker: Explainable multimodal **1194** large language models with rule-based reinforce- **1195** ment learning for document understanding. In _Pro-_ **1196** _ceedings of the IEEE/CVF International Conference_ **1197** _on Computer Vision_ , pages 837–847. **1198** 

- Ya-Qi Yu, Minghui Liao, Jihao Wu, Yongxin Liao, **1199** Xiaoyu Zheng, and Wei Zeng. 2024a. Texthawk: **1200** Exploring efficient fine-grained perception of multi- **1201** modal large language models. **1202** 

- Ya-Qi Yu, Minghui Liao, Jiwen Zhang, and Jihao Wu. **1203** 2024b. Texthawk2: A large vision-language model **1204** excels in bilingual ocr and grounding with 16x fewer **1205** tokens. **1206** 

- Jiaxin Zhang, Wentao Yang, Songxuan Lai, Zecheng **1207** Xie, and Lianwen Jin. 2024a. Dockylin: A large **1208** multimodal model for visual document understand- **1209** ing with efficient visual slimming. _arXiv preprint_ **1210** _arXiv:2406.19101_ . **1211** 

- Jinxu Zhang, Qiyuan Fan, and Yu Zhang. 2025. Do- **1212** cassistant: Integrating key-region reading and step- **1213** wise reasoning for robust document visual question **1214** answering. In _Findings of the Association for Com-_ **1215** _putational Linguistics: EMNLP 2025_ , pages 3496– **1216** 3511. **1217** 

- Renshan Zhang, Yibo Lyu, Rui Shao, Gongwei Chen, **1218** Weili Guan, and Liqiang Nie. 2024b. Token-level **1219** correlation-guided compression for efficient multi- **1220** modal document understanding. _arXiv_ . **1221** 

- Ruiyi Zhang, Yufan Zhou, Jian Chen, Jiuxiang Gu, **1222** Changyou Chen, and Tong Sun. 2024c. Llava-read: **1223** Enhancing reading ability of multimodal language **1224** models. _arXiv preprint arXiv:2407.19185_ . **1225** 

- Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, **1226** Nedim Lipka, Diyi Yang, and Tong Sun. 2024d. **1227** Llavar: Enhanced visual instruction tuning for text- **1228** rich image understanding. **1229** 

- Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. **1230** 2019. Publaynet: largest dataset ever for document **1231** layout analysis. In _2019 International Conference_ **1232** _on Document Analysis and Recognition (ICDAR)_ , **1233** pages 1015–1022. IEEE. **1234** 

13 

**1235** Yuke Zhu, Yue Zhang, Dongdong Liu, Chi Xie, Zi- **1236** hua Xiong, Bo Zheng, and Sheng Guo. 2025a. En- **1237** hancing document understanding with group posi- **1238** tion embedding: A novel approach to incorporate **1239** layout information. In _The Thirteenth International_ **1240** _Conference on Learning Representations_ . 

**1241 1242 1243 1244 1245** 

- Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao, Hangdi Xing, Qi Zheng, and Ji Zhang. 2025b. A simple yet effective layout token in large language models for document understanding. _arXiv preprint arXiv:2503.18434_ . 

14 

**1246** 

**1247 A.1 Open-source Frameworks 1248** Table 2 presents official open-source links for **1249** VRDU and MLLM frameworks, underscoring the **1250** vital role of open access in fostering transparency, **1251** reproducibility, and accelerated innovation within **1252** the research community. 

**1253** 

**1269** 

**1290** 

**1291 1292 1293** 

## **A More Framework Details** 

## **A.2 Model Training Paradigm Comparison** 

**1254** Table 3 provides a comprehensive compari- **1255** son of MLLM-based VRDU frameworks across **1256** three major training stages: Pretraining (PT), **1257** Instruction-tuning (IT), and Supervised Fine- **1258** tuning (SFT). OCR-dependent models generally **1259** rely on external text extraction and have lim- **1260** ited pretraining because they are trained on OCR- **1261** processed inputs. In contrast, OCR-free mod- **1262** els, which operate directly on document images, **1263** demonstrate richer instruction-tuning and fine- **1264** tuning strategies, often involving frozen or LoRA- **1265** based vision and language encoders. This high- **1266** lights the diverse training paradigms and modular **1267** designs adopted to balance efficiency, adaptability, **1268** and performance across frameworks. 

## **A.3 Document Parsing Tools** 

**1270** Table 5 provides a comparative overview of rep- **1271** resentative OCR engines, document parsing APIs, **1272** and vision–language models for document under- **1273** standing. The table highlights clear trade-offs **1274** across deployment modes, pricing models, and **1275** functional capabilities: traditional OCR engines **1276** are predominantly open-source and locally de- **1277** ployable but offer limited support for structured **1278** document parsing, while commercial document **1279** APIs and vision LLMs more frequently provide **1280** GPU acceleration and native document-structure **1281** extraction at the cost of cloud dependency and **1282** usage-based pricing. Recent vision–language **1283** models bridge OCR and higher-level reasoning by **1284** supporting multimodal inputs (image and PDF) **1285** and multilingual processing, yet vary substantially **1286** in openness and deployment flexibility. Over- **1287** all, the comparison illustrates the evolving land- **1288** scape from text-centric OCR toward multimodal, **1289** structure-aware document understanding systems. 

## **A.4 Model Component Details** 

Table 4 presents a comprehensive comparison of component configurations adopted by recent MLLM-based frameworks for VRDU, spanning 

both OCR-Dependent and OCR-Free paradigms. **1294** For each model, we summarize its LLM back- **1295** bone (e.g., Vicuna, Qwen, LLaMA, GPT), vision **1296** encoder (e.g., CLIP, ViT, Swin), input resolution **1297** (including dynamic scaling and cropping), and **1298** specialized adaptors or projectors (e.g., LoRA, **1299** MLP, QPN) used for multimodal fusion. OCR- **1300** Dependent models typically incorporate layout- **1301** aware encoders (e.g., LayoutLMv3, DocFormer) **1302** and rely on structured textual inputs. In contrast, **1303** OCR-Free models process raw document images **1304** directly, often requiring higher resolutions and ad- **1305** ditional modules such as resamplers, visual ab- **1306** stractors, or cropping strategies. The table also **1307** lists the maximum supported image resolution, in- **1308** dicating each model’s capacity for fine-grained vi- **1309** sual understanding. This comparison highlights **1310** the increasing diversity in MLLM architectures **1311** and the adoption of lightweight tuning techniques **1312** for scalable VRDU. **1313** 

## **B Dataset** 

**1314** 

**Pretraining Datasets.** The goal of pretraining **1315** is to enhance multimodal understanding and im- **1316** prove generalization across VRDU tasks. Sim- **1317** ilar to pretrained VRDU frameworks, MLLM- **1318** based approaches commonly perform continued **1319** pretraining on large-scale, cross-domain docu- **1320** ment collections such as IIT-CDIP (Lewis et al., **1321** 2006), which contains over 6 million scanned **1322** documents across diverse domains, though lack- **1323** ing explicit layout annotations—often supple- **1324** mented with OCR-derived bounding boxes. RVL- **1325** CDIP (Harley et al., 2015), a curated sub- **1326** set with 400,000 documents across 16 cate- **1327** gories, is widely used for document classifi- **1328** cation and low-resource pretraining. Beyond **1329** these general-purpose datasets, recent frame- **1330** works (Zhang et al., 2024d; Wang et al., 2023) **1331** have introduced self-collected datasets to target **1332** domain-specific or task-oriented scenarios, in- **1333** cluding slide decks (Feng et al., 2024), academic **1334** papers (Wang et al., 2024a), and other structured **1335** document types (Yu et al., 2024b), as summarized **1336** in Table 6. **1337** 

**Instruction-tuning Datasets.** Instruction- **1338** tuning aims to enhance a model’s understanding **1339** of user queries. Many frameworks (Zhang et al., **1340** 2024b; Park et al., 2024) perform instruction- **1341** tuning directly on benchmark document collec- **1342** tions to improve downstream task performance. **1343** 

15 

**1377 1378** 

**1372** 

**1373 1374 1375** 

**1376** 

Figure 3: MLLM-based VRDU framework training paradigms. [Yifan: This figure is not referred to in the text.] 

**1344** Others (Luo et al., 2024; Liu et al., 2024a) **1345** generate large-scale synthetic datasets using **1346** OCR tools to extract text and layout information **1347** from VRD-related benchmarks such as layout **1348** analysis (Zhong et al., 2019) and document classi- **1349** fication (Harley et al., 2015). Instruction-response **1350** pairs are then created based on predefined task **1351** definitions. Some frameworks also construct their **1352** own multi-domain datasets to improve generaliz- **1353** ability and prevent data leakage (Wei et al., 2024; **1354** Feng et al., 2023). Instruction-tuning is critical for **1355** domain adaptation and accurate instruction inter- **1356** pretation. As shown by Table 7, some frameworks **1357** increasingly generate synthetic instruction-tuning **1358** datasets tailored to their architectures, prioritizing **1359** alignment over generalizability achieved through **1360** benchmark-based tuning. 

**1361 Supervised Fine-tuning Datasets.** To improve **1362** performance on downstream tasks, some frame- **1363** works apply supervised fine-tuning on question **1364** answering datasets such as DocVQA (Mathew **1365** et al., 2021) and MPDocVQA (Tito et al., 2023). **1366** Additionally, several key information extraction **1367** benchmarks—such as FUNSD (Jaume et al., **1368** 2019), FormNLU (Ding et al., 2023), and CORD **1369** (Park et al., 2019), have been reformulated into **1370** QA-style formats to enable evaluation with gen- **1371** erative frameworks. 

## **C Benchmark Datasets** 

Based on differences in downstream tasks and the benchmark dataset’s domain, we list the widely used VRDU dataset and its key attributes in Table 8, including both VRD-related Key Informa- 

tion Extraction (KIE) and Visual Question Answering (VQA). 

**Key Information Extraction** Benchmarks for **1379** Key Information Extraction (KIE) are shift- **1380** ing from early schema-constrained tasks (e.g., **1381** SROIE (Huang et al., 2019), FUNSD (Jaume **1382** et al., 2019)) toward larger, multilingual, cross do- **1383** main, multi page and open-vocabulary challenges. **1384** While form-like structures (e.g., DocILE (Simsa[ˇ] **1385** et al., 2023), Form-NLU (Ding et al., 2023)) still **1386** dominate the landscape, modern resources such as **1387** KVP10k (Naparstek et al., 2024) and CC-OCR- **1388** KIE (Yang et al., 2024) focus on _open-category_ **1389** extraction without predefined schemas. Further- **1390** more, a clear trend of dataset consolidation and **1391** multilingual expansion has emerged. **1392** 

**Visual Question Answering** has undergone a **1393** comparable evolution, shifting from early single- **1394** page, text-centric retrieval to benchmarks that **1395** probe multiple dimensions of complexity. This **1396** progression is reflected in broader multilingual **1397** coverage (e.g., MTVQA (Tang et al., 2025), **1398** JDocQA (Onami et al., 2024)) and more diverse, **1399** multi-domain settings (e.g., DUDE (Van Lan- **1400** deghem et al., 2023)). Recent datasets in- **1401** creasingly emphasize long-context comprehen- **1402** sion over multi-page documents: benchmarks **1403** such as LongDocURL (Deng et al., 2025) and **1404** MMLongBench-Doc (Wang et al., 2025a) con- **1405** tain documents averaging dozens of pages and of- **1406** ten demand non-trivial cross-page evidence ag- **1407** gregation and reasoning. In parallel, reason- **1408** ing requirements have deepened toward domain- **1409** 

16 

traction, OCR-free approaches are quickly matur- **1459** ing and expanding the frontier of end-to-end doc- **1460** ument understanding. **1461** 

**1431** 

**1432** 

**1458** 

**1410** specific expertise, as illustrated by vision-essential **1411** physics problem solving in SEEPHYS (Xiang **1412** et al., 2025). Finally, dataset scale has ex- **1413** panded substantially—reaching millions of in- **1414** stances in collections such as MMVQA (Ding **1415** et al., 2024a)—thereby enabling rigorous stress- **1416** testing of the capacity and reasoning limits of **1417** modern multimodal models. 

**1418 Other Domain Datasets** Many frameworks are **1419** evaluated on other domain-specific datasets as **1420** well, including those for chart understanding and **1421** webpage analysis. For instance, InfoVQA (Gupta **1422** et al., 2022) focuses on visual question answer- **1423** ing for information-centric records. Benchmarks **1424** like WTQ (Pasupat and Liang, 2015) and TabFact **1425** (Chen et al., 2020) assess a model’s ability to rea- **1426** son over tabular data, and ChartQA evaluates chart **1427** comprehension skills. Additionally, TextVQA **1428** (Singh et al., 2019) and TextCaps (Sidorov et al., **1429** 2020) target text recognition and semantic reason- **1430** ing in natural images. 

## **D Quantitive Analysis** 

## **D.1 Performance on Single Page Benchmarks** 

## **D.2 Performance on Multi-Page Benchmarks** 

**Performance on Multi-Page Benchmarks 1462** We report the performance of existing multi-page **1463** frameworks on two multi-page VRDU bench- **1464** marks in Table 10. General-domain models can **1465** achieve reasonable performance; however, frame- **1466** works equipped with mechanisms explicitly de- **1467** signed for visually rich documents (VRDs) consis- **1468** tently yield substantial improvements. Currently, **1469** most high-performing multi-page methods rely on **1470** OCR-dependent pipelines and achieve strong re- **1471** sults by leveraging external OCR tools. While **1472** such designs reduce the burden of directly under- **1473** standing and compressing visual representations, **1474** they also inherit the limitations of OCR-based **1475** approaches, including error accumulation as ob- **1476** served in single-page scenarios. For multi-page **1477** tasks, this challenge is further amplified, high- **1478** lighting the need for more effective strategies to **1479** manage the large number of visual tokens and to **1480** improve text understanding in multi-page, text- **1481** dense document inputs. **1482** 

**1433** Table 9 highlights clear trends in the per- **1434** formance of general-domain LLMs/MLLM and **1435** OCR-dependent and OCR-free document un- **1436** derstanding frameworks across several popular **1437** benchmarks. Generally, OCR-dependent mod- **1438** els achieve consistently strong results on classic **1439** form and receipt datasets such as FUNSD, CORD, **1440** and SROIE—often exceeding 80% accuracy, with **1441** top models such as PDF-WuKong, GPE, and Do- **1442** cLayLLM achieving state-of-the-art performance. **1443** In contrast, OCR-free frameworks, while demon- **1444** strating rapid progress, still lag on these traditional **1445** datasets but show remarkable advances on more **1446** visually and semantically complex benchmarks **1447** such as DocVQA, ChartVQA, and InfoVQA. No- **1448** tably, the latest OCR-free models, including Tex- **1449** thawk2, Marten, and PP-DocBee, have begun to **1450** outperform or match OCR-dependent methods on **1451** DocVQA and chart-centric tasks, signalling a nar- **1452** rowing of the gap in real-world document rea- **1453** soning capabilities. However, coverage remains **1454** uneven, with many OCR-free models perform- **1455** ing poorly on specific datasets, indicating ongo- **1456** ing challenges with generalizability and bench- **1457** mark saturation. Overall, while OCR-dependent **1458** methods remain dominant for structured text ex- 

17 

|**Framework**||**Model Name**|**Offcial Open Source Link**|
|---|---|---|---|
|mPLUG-DocOwl|1.5|DocOwl 1.5|github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5|
|mPLUG-DocOwl|2|DocOwl 2|github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl2|
|UReader||UReader|github.com/X-PLUG/mPLUG-DocOwl/tree/main/UReader|
|KOSMOS-2.5||KOSMOS-2.5 / 2.5-CHAT|aka.ms/kosmos25|
|LLaVAR||LLaVAR|github.com/SALT-NLP/LLaVAR|
|Marten||Marten|github.com/PriNing/Marten|
|LEOPARD||LEOPARD|github.com/Jill0001/Leopard|



Table 2: Official open-source links for some VRDU/MLLM frameworks. 

18 

|**Vision Encoder**<br>**LLM Backbone**<br>**Adaptors**<br>~~a~~|**Vision Encoder**<br>**LLM Backbone**<br>**Adaptors**<br>~~a~~|
|---|---|
|**Model Name**<br>PT<br>IT<br>SFT<br>PT<br>IT<br>SFT<br>PT<br>IT<br>SFT<br>~~a~~||
|**OCR-Dependent**||
|ICL-D3IE (2023)<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–||
|DocLLM (2024a)<br>T<br>T<br>–<br>–<br>–<br>–<br>T<br>T<br>–<br>LAPDoc (2024)<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>~~@~~<br>®@<br>@<br>®@|~~@~~|
|LMDX (2024)<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–||
|ProcTag (2025)<br>–<br>–<br>T<br>–<br>–<br>T<br>–<br>–<br>T<br>@<br>@<br>@||
|DocKD (2024)<br>–<br>–<br>T<br>–<br>–<br>T<br>–<br>–<br>–<br>DoCo (2024)<br>F<br>–<br>F<br>T<br>–<br>F<br>T<br>–<br>T<br>InstructDoc (2024)<br>–<br>F<br>F<br>–<br>F<br>F<br>–<br>T<br>T<br>LayoutLLM (2024)<br>–<br>F<br>T<br>–<br>F<br>F<br>–<br>T<br>T<br>LLaVA-Read (2024c)<br>F<br>T<br>–<br>F<br>F<br>–<br>T<br>T<br>–<br>LayTextLLM (2024)<br>F<br>–<br>T<br>–<br>–<br>–<br>T<br>–<br>T<br>LayTokenLLM (2025b)<br>F<br>–<br>F<br>–<br>–<br>–<br>T<br>–<br>T<br>GPE (2025a)<br>–<br>–<br>T<br>–<br>–<br>–<br>–<br>–<br>–<br>@<br>@<br>@<br>@ |e<br>@ |e<br>.<br>@ Q<br>@<br>OQ<br>@<br>®@<br>**@** ®@<br>Qe @<br>@ ®@<br>@<br>@<br>@<br>@@<br>9)<br>@<br>@<br>.<br>@<br>9)<br>@<br>@<br>@||
|MDocAgent (2025)<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–||
|PDF-WuKong (2025)<br>–<br>–<br>T<br>–<br>–<br>T<br>–<br>–<br>–<br>DocLayLLM (2024)<br>F<br>F<br>–<br>T<br>T<br>–<br>T<br>T<br>–<br>DocAssistant (2025)<br>-<br>F<br>-<br>-<br>F<br>-<br>-<br>T<br>-<br>AlignVLM (2025)<br>T<br>T<br>F<br>T<br>T<br>T<br>T<br>T<br>T<br>DocThinker (2025)<br>-<br>-<br>T<br>-<br>-<br>T<br>-<br>-<br>T<br>@<br>@<br>@® @<br>® @<br>@<br>®@<br>@<br>@<br>@<br>@©9egeqeeg@e@8@®@<br>@<br>@<br>@||
|**OCR-Free**||
|KOSMOS-2.5 (2023)<br>–<br>T<br>T<br>–<br>T<br>F<br>–<br>T<br>T<br>mPLUG-DocOwl (2023a)<br>–<br>F<br>–<br>–<br>F<br>–<br>–<br>T<br>–<br>UReader (2023b)<br>–<br>F<br>–<br>–<br>F<br>–<br>–<br>T<br>–<br>TGDoc (2023)<br>–<br>F<br>T<br>–<br>F<br>F<br>–<br>T<br>T<br>UniDoc (2023)<br>–<br>F<br>T<br>–<br>F<br>F<br>–<br>T<br>T<br>DocPedia (2024)<br>F<br>–<br>T<br>T<br>–<br>T<br>T<br>–<br>T<br>HRVDA (2024a)<br>F<br>F<br>–<br>T<br>F<br>–<br>T<br>T<br>–<br>Vary (2024)<br>T<br>–<br>T<br>T<br>–<br>F<br>T<br>–<br>T<br>mPLUG-DocOwl 1.5 (2024)<br>–<br>F<br>T<br>–<br>T<br>F<br>–<br>T<br>T<br>HVFA (2024)<br>–<br>F<br>–<br>–<br>F<br>–<br>–<br>T<br>–<br>mPLUG-DocOwl2 (2025)<br>–<br>F<br>T<br>–<br>T<br>F<br>–<br>T<br>T<br>Texthawk (2024a)<br>–<br>F<br>T<br>–<br>F<br>F<br>–<br>T<br>T<br>Texthawk2 (2024b)<br>–<br>F<br>T<br>–<br>F<br>T<br>–<br>T<br>T<br>TextMonkey (2024c)<br>–<br>T<br>–<br>–<br>T<br>–<br>T<br>T<br>–<br>Llavar (2024d)<br>–<br>F<br>T<br>–<br>F<br>F<br>–<br>T<br>T<br>TokenCorrCompressor (2024b)<br>–<br>–<br>F<br>–<br>–<br>F<br>–<br>–<br>T<br>DocKylin (2024a)<br>–<br>F<br>T<br>–<br>T<br>T<br>–<br>T<br>T<br>Marten (2025b)<br>–<br>F<br>T<br>–<br>T<br>T<br>–<br>T<br>T<br>PP-DocBee (2025)<br>–<br>–<br>T<br>–<br>–<br>F<br>–<br>–<br>–<br>TokenFD (2025)<br>T<br>F<br>T<br>T<br>F<br>T<br>T<br>T<br>T<br>@<br>@<br>@®<br>Oo<br>@<br>®@<br>~~@~~<br>~~@~~<br>~~®~~<br>~~@~~<br>9)<br>.<br>Q ®@<br>QO<br>© @<br>©<br>Q@<br>e@® ®@<br>@<br>\an<br>@ @<br>®<br>Q@ O<br>@ 0<br>@ ®@<br>@<br>@(®@<br>@ |e<br>.<br>@ @<br>@® O<br>@ ®@<br>@<br>9)<br>.<br>Q ®@<br>@® Oo<br>@ ®@<br>@ @<br>@<br>Q@<br>e@ ®@<br>@ @<br>@<br>®@<br>@ ®@<br>@<br>@<br>@ ®@<br>©<br>@<br>©<br>@<br>e@ ®@<br>@<br>@<br>.<br>@<br>®@<br>@<br>®@<br>@® ®@<br>°$ ° $s<br>@ ®@<br>@9oe\000'0<br>~~@@~~||



Table 3: Comparison of MLLM-based VRDU frameworks. PT - Pretraining, IT - Instruction-tuning, SFT - Supervised Fine-tuning. 

19 

**1483** 

**1484** 

Table 4: Comparison of MLLM-based VRDU frameworks: Backbone and Adapter configurations. “–” denotes the component is not applicable or not disclosed. 

|**Model Name**|**LLM Backbone**|**Vision Backbone**|**Resolution**|**Adaptors and Projectors**|
|---|---|---|---|---|
|**OCR-Dependent**|||||
|ICL-D3IE (2023)|GPT-3, ChatGPT|–|–|–|
|DocLLM (2024a)|Falcon-1B/LLaMA2-7B|–|–|Disentangled Spatial Attention|
|LAPDoc (2024)|ChatGPT, Solar|–|–|–|
|LMDX (2024)|PaLM 2-S, Gemini Pro|–|–|–|
|ProcTag (2025)|Qwen-7B/Qwen-VL-7B|qwen2vl vision encoder|Dynamic (224_×_224 to|qwen2vl projector|
||||448_×_448)||
|DocKD (2024)|DocFormerv2 language|DocFormerv2 vision encoder|Derived from CNN backbone|–|
||decoder||||
|DoCo (2024)|Qwen-VL-Chat/mPLUG-Owl|ViT-bigG|224_×_224|Position-Aware|
|||||Vision-Language Adapter,|
|||||Visual Abstractor|
|InstructDr (2024)|Flan-T5|CLIP|224_×_224|Document-former|
|LayoutLLM (2024)|Vicuna-7B-v1.5,|LayoutLMv3|224_×_224|MLP|
||LLaMA2-7B-chat||||
|LLaVA-Read (2024c)|Vicuna-1.5 13B|CLIP-ViT-L/14-336 +|336_×_336|MLP|
|||ConvNext-L/32-320|||
|LayTextLLM (2024)|Llama2-7B-base|–|320_×_320|Spatial Layout Projector +|
|||||Layout Partial LoRA|
|LayTokenLLM (2025b)|Qwen1.5-7B, LLaMA3-8B|–|–|Layout Tokenizer + LORA|
|GPE (2025a)|LLaMA2-7B, Qwen2-7B,|–|–|–|
||ChatGLM-6B||||
|MDocAgent (2025)|LLaMA-3.1-8B (Text),|ColPali|448_×_448|–|
||Qwen2-VL-7B (Others)||||
|PDF-WuKong (2025)|IXC2-VL-4KHD|IXC2-VL-4KHD|Dynamic (336_×_336 to|–|
||||3840_×_1600)||
|DocLayLLM (2024)|LLaMA2-7B, LLaMA3-8B|LayoutLMv3 ve|224_×_224|Layout Embedder + Projector +|
|||||LORA|
|DocAssistant (2025)|InternVL2-Chat-2B|InternVL2-Chat-2B Vision|448_×_448|Mixture-of-Modality|
|||Encoder||Adaptation + Projector +|
|||||LORA|
|AlignVLM (2025)|Llama 3.1-1B_\_3B_\_8B|SigLip-400M|Dynamic (14×14 patches)|ALIGN Module|
|DocThinker (2025)|Qwen2.5-VL-3B_\_7B|Qwen2.5-VL-3B_\_7B Vision|336_×_336, 1536_×_1536|-|
|||Encoder|||
|**OCR-Free**|||||
|KOSMOS-2.5 (2023)|Transformer decoder|Pix2Struct-Large ViT-based|1024_×_1024|Resampler|
|mPLUG-DocOwl (2023a)|mPLUG-Owl|ViT|224_×_224|Visual Abstractor + Lora|
|UReader (2023b)|mPLUG-Owl|CLIP-like ViT|224_×_224 (_×_20 crops)|Visual Abstractor + Lora|
|TGDoc (2023)|Vicuna-7B|CLIP-ViT-L/14|224_×_224 and 336_×_336|MLP|
|UniDoc (2023)|Vicuna|CLIP-ViT-L/14|224_×_224 and 336_×_336|MLP|
|DocPedia (2024)|Vicuna-7B|Swin Transformer|2560_×_2560|MLP|
|HRVDA (2024a)|LLaMA-2-7B|Swin Transformer|1536_×_1536|Content Detector + MLP|
|||||Projector + LoRA|
|Vary (2024)|OPT125M + Qwen-7B,|CLIP + SAM|1024_×_1024|MLP|
||Vicuna-7B||||
|mPLUG-DocOwl 1.5 (2024)|mPLUG-Owl2|ViT/L-14|448_×_448 (_×_9 crops)|H-Reducer|
|HVFA (2024)|BLIP-2-OPT-2.7B,|ViT|224_×_224_×_crops|HVFA + Lora + Resampler|
||mPLUG-Owl-7B||||
|mPLUG-DocOwl2 (2025)|mPLUG-Owl2|ViT|504_×_504 (_×_12 crops)|H-Reducer|
|Texthawk (2024a)|InternLM-XComposer 7B|SigLIP-SO (ViT)|224_×_224_×_crops|Resampler + LoRA + QPN +|
|||||MLCA|
|Texthawk2 (2024b)|Qwen2-7B-Instruct|SigLIP-SO (ViT)|224_×_224_×_crops (up to 72|Resampler + QPN + MLCA +|
||||crops)|Detection Head + LoRA|
|TextMonkey (2024c)|Qwen-VL-Chat, mPLUG-Owl|ViT-BigG|448_×_448_×_crops|Image + Token Resampler|
|Llavar (2024d)|Vicuna-13B|CLIP-ViT-L/14|224_×_224 and 336_×_336|MLP|
|TokenCorrCompressor (2024b)|LLaMA2-7B|CLIP-ViT-L/14|224_×_224 and 336_×_336|Token Correlation Compressor|
|||||+ LORA|
|DocKylin (2024a)|Qwen-7B-Chat|Swin (Donut-Swin, 0.07B)|1728_×_1728|MLP + APS + DTS|
|Marten (2025b)|InternLM2-7B|InternViT-300M|448_×_448 (_×_6 crops)|MLP + Mask Generator|
|||||Module|
|||||Continued on next page|



20 

**1485** 

Table 4: Comparison of MLLM-based VRDU frameworks: Backbone and Adapter configurations. “–” denotes the component is not applicable or not disclosed. (Continued) 

|**Model Name**|**LLM Backbone**|**Vision Backbone**|**Resolution**|**Adaptors and Projectors**|
|---|---|---|---|---|
|PP-DocBee (2025)|Qwen2-VL-2B|ViT|1680_×_1204|–|
|TokenFD (2025)|token embedding layer|ViT|448_×_448 (_×_6 crops)|Token abstractor|



21 

|**Tool Name**|**Provider**|**Tool Type**|**Deployment**|**Pricing**|**Input Modalities**|**Languages**|**Openness**|**GPU**|**Doc Parsing**|
|---|---|---|---|---|---|---|---|---|---|
|pdfminer.six|Y. Shinyama et al.|OCR Engine|Local|Free|PDF|Multi|Open-source|No|No|
|Mistral OCR|Mistral AI|Document API|Cloud|Paid (Usage-based)|Image, PDF|Multi|Closed|Supported|Yes|
|LightOnOCR|LightOnAI|Vision LLM|Cloud|Paid (Usage-based)|Image, PDF|Multi|Closed|Supported|No|
|Google Cloud Vision|Google|Document API|Cloud|Paid (Usage-based)|Image, PDF|Multi|Closed|Supported|Yes|
|Kraken|Inria et al.|OCR Engine|Local|Free|Image, PDF|Multi|Open-source|Supported|No|
|Qwen3-VL|Aliyun|Vision LLM|Hybrid|Free*|Image, PDF|Pretrained/Dependent|Closed|Supported|No|
|olmOCR|AI2|OCR Engine|Hybrid|Free|Image, PDF|Multi|Open-source|Supported|Yes|
|AttentionOCR|Guo & Deng|OCR Engine|Local|Free|Image|Multi|Open-source|Supported|No|
|Calamari|Univ. W¨urzburg|OCR Engine|Local|Free|Image|Multi|Open-source|Supported|No|
|EasyOCR|JaidedAI|OCR Engine|Local|Free|Image|Multi|Open-source|Supported|No|
|OpenAI Vision|OpenAI|Vision LLM|Cloud|Paid (Usage-based)|Image, PDF|Multi|Closed|Supported|Yes|
|Tesseract|S. Weil|OCR Engine|Local|Free|Image|Multi|Open-source|No|No|
|Adobe PDF Extract|Adobe|Document API|Cloud|Paid (Usage-based)|PDF|Multi|Closed|Supported|Yes|
|PaddleOCR|PaddlePaddle|OCR Engine|Cloud|Free|Image, PDF|Multi|Open-source|Supported|Yes|
|docTR|Mindee|OCR Engine|Local|Free|Image, PDF|Pretrained/Dependent|Open-source|Supported|No|
|DeepSeek-OCR|DeepSeek AI|Vision LLM|Hybrid|Paid (Usage-based)|Image, PDF|Multi|Open-source|Supported|No|
|HunyuanOCR|Tencent|Vision LLM|Local|Free|Image, PDF|Multi|Open-source|Supported|No|
|Ocular|Berkeley NLP|OCR Engine|Local|Free|Image, PDF|Multi|Open-source|Supported|No|
|MinerU|OpenDataLab|Document API|Local|Free|PDF|Multi|Open-source|Supported|Yes|
|SuryaOCR|Datalab|OCR Engine|Local|Free|Image, PDF, Word, PPT|Multi|Open-source|Supported|No|
|Seed-VL|ByteDance Seed|Vision LLM|Cloud|Paid (Usage-based)|Image, PDF|Multi|Open-source|Supported|Yes|



Table 5: Comparison of OCR engines, document parsing APIs, and vision-language models for document understanding. 

22 

|||||**Public**|
|---|---|---|---|---|
|**Study**|**Dataset**|**Source**|**Size**|**Available**|
|Vary|Document Data Engine|ArXiv, CC-MAIN, E-books|2M|✗|
||Chart Data Engine|matplotlib, pyecharts, NLP corpora|1.5M|✗|
||Detection Data Engine|Objects365, OpenImages|_∼_3M|✓|
|LLaVAR|LAION|LAION images fltered for text-rich content, OCR applied|0.4M|✓|
|DoCo|DoCo-Processed|CC3M (LLaVA) + LAION, processed with PaddleOCR|1.0M|✗|
|Texthawk2|100M pretraining|Diverse, mainly public datasets|100M|✗|
|Docpedia|PDF Images|arXiv (public scientifc preprints)|325K|✓|
||PPT Images|Common Crawl (web-crawled PPTs)|600K|Partly|



Table 6: Summary of pretraining datasets created and used in recent MLLM-based VRDU frameworks. 

23 

|**Framework**|**Category**<br>**Source / Description**<br>**Size (K)**<br>**Open Source**|
|---|---|
|Leopard|Multi-image<br>(text-rich)<br>69K public multi-page docs/slides; Adapted<br>single-page to multi-image (DocVQA, ArxivQA);<br>Raw slides + GPT-4o QAs; Multi-chart/table (open,<br>synth.); Webpage snapshots (Mind2Web, OmniACT,<br>WebScreenshots, etc.)<br>739<br>Partially|
||Single-image<br>Text-rich single images from public datasets; Natural<br>images (e.g., ShareGPT4V, etc.)<br>186<br>Partially|
|LLaVAR|Noisy Instruction-<br>Following<br>Text-rich images from LAION, selected via classifer +<br>CLIP clustering, instructions via OCR-based prompts<br>422,000<br>Yes|
||High-Quality<br>Instruction-<br>Following<br>Subset of LAION text-rich images (4 clusters),<br>multi-turn QAs generated by prompting text-only<br>GPT-4 with OCR+caption info<br>16,000<br>Yes|



Table 7: Summary of instruction-tuning datasets for Leopard and LLaVAR. 

24 

|**Dataset**|**Venue**|**Year**|**Domain**|**Docs**|**Images**|**Keys / Qs**|**Multi page**|**Language**|**Metrics**|**Format**|
|---|---|---|---|---|---|---|---|---|---|---|
|**Key Information Extraction**|||||||||||
|FUNSD|ICDAR-w|2019|Multi-source|–|199|4|✗|English|F1|P, H|
|SROIE|ICDAR-c|2019|Scanned Receipts|–|973|4|✗|English|F1*|P|
|CORD|NeurIPS-w|2019|Scanned Receipts|–|1,000|54|✗|English|F1|P|
|Payment-Invoice|ACL|2020|Invoice Form|–|14,832|7|✗|English|F1|D|
|Payment-Receipts|ACL|2020|Scanned Receipts|–|478|2|✗|English|F1|P|
|Kleister-NDA|ICDAR|2021|Private Agreements|540|3,229|4|✓|English|F1|D|
|Kleister-Charity|ICDAR|2021|AFR|2,778|61,643|8|✓|English|F1|D, P|
|EPHOIE|AAAI|2021|Exam Paper|–|1,494|10|✗|Chinese|F1|P, H|
|XFUND|ACL|2022|Synthetic Forms|–|1,393|4|✗|Multilingual|F1|D, P, H|
|Form-NLU|SIGIR|2023|Financial Form|–|857|12|✗|English|F1|D, P, H|
|VRDU-Regist. Form|KDD|2023|Registration Form|–|1,915|6|✗|English|F1|D|
|VRDU-Ad-buy Form|KDD|2023|Political Invoice Form|–|641|9+1(5)|✗|English|F1|D, P|
|DocILE|ICDAR|2023|Invoice Form|6,680|106,680|55|✓|English|AP, CLEval|D, P|
|KVP10k|ICDAR|2024|Cross-domain|–|10,707|118,868|✗|English|F1, IOU|D, H|
|CC-OCR-KIE|ICCV|2025|Cross-domain|–|2,008|34(-)|✗|Multilingual|F1|D, P, H|
|**Visual Question Answering**|||||||||||
|DocVQA|WACV|2021|Industrial Reports|–|12,767|50,000|✗|English|ANLS|D, P, H|
|VisualMRC|AAAI|2021|Website|–|10,197|30,562|✗|English|BLEU, etc|D|
|TAT-DQA|MM|2022|Financial Reports|2,758|3,067|16,558|✓|English|EM, F1|D|
|RDVQA|MM|2022|Data Analysis Report|8,362|8,514|41,378|✗|English|ANLS, ACC|D|
|CS-DVQA|MM|2022|Industry Documents|–|600|1,000|✗|English|ANLS|D, P, H|
|InfographicVQA|WACV|2022|Infographics|–|5,400|3,000|✗|English|ANLS, F1|D|
|PDFVQA-Task A|ECML-PKDD|2023|Academic Paper|–|12,337|81,085|✗|English|F1|D|
|PDFVQA-Task B|ECML-PKDD|2023|Academic Paper|–|12,337|53,872|✗|English|F1|D|
|PDFVQA-Task C|ECML-PKDD|2023|Academic Paper|1,147|12,337|5,653|✓|English|EM|D|
|MPDocVQA|PR|2023|Industrial Reports|6,000|48,000|46,000|✓|English|ANLS|D, P, H|
|DUDE|ICCV|2023|Cross-domain|5,019|28,709|41,541|✓|English|ANLS|D|
|SlideVQA|AAAI|2023|Slide, decks|–|5,200|14,500|✓|English|EM, F1|D|
|MMLONGBENCH-DOC|NIPS|2024|Cross-domain|135|6,413|1,082|✓|English|ACC, F1|D|
|MMVQA|IJCAI|2024|Academic Paper|3,146|30,239|262,928|✓|English|EM, PM, MR|D|
|JDocQA|LREC-COLING|2024|Cross-Domain|5,504|268,000|11,600|✓|Japanese|F1|D|
|BoundingDocs|IJDAR|2025|Cross-domain, Mixed|48,151|237,437|249,016|✗|Multilingual|ANLS|D, P, H|
|LongDocURL|ACL|2025|Cross-domain|396|33,000|2,325|✓|English|F1|D|
|MMDocIR|EMNLP|2025|Cross-domain|6,878|224,223|73,843|✓|Multilingual|F1|D|
|MTVQA|EMNLP|2025|Cross-domain|-|8,794|28,607|✗|Multilingual|ANLS|D, P, H|
|SEEPHYS|NIPS|2025|Physics|-|2,245|2,000|✗|English|Accuracy|D|



Table 8: Benchmark datasets for Key Information Extraction and Visual Question Answering in visually rich documents. P - Scanned **P** rinted, H - Scanned **H** andwritten, D - **D** igtial Born 

25 

|**Model Name**|**FUNSD**|**CORD**|**SROIE**|**DocVQA**|**ChartVQA**|**InfoVQA**|
|---|---|---|---|---|---|---|
|**General Domain LLM**|||||||
|Qwen1.5-7B-Chat|52.5|29.7|–|64.3|–|–|
|Llama3-8B-Instruct|57.5|40.0|–|74.2|–|–|
|**General Domain MLLM**|||||||
|QwenVL-7B|47.1|30.0|–|65.1|–|–|
|InterVL2-8B|75.8|79.9|–|91.7|–|–|
|Claude-3.5 Sonnet|–|–|–|88.5|51.8|59.1|
|GeminiPro-1.5|–|–|–|91.2|34.7|73.9|
|GPT4o 20240806|–|–|–|92.8|85.7|66.4|
|**OCR-Dependent**|||||||
|DocLLM (2024a)|51.8|67.4|91.9|69.5|–|–|
|LAPDoc (2024)|–|–|–|79.8|–|54.9|
|DoCo (2024)|–|–|–|64.8|68.9|34.9|
|InstructDr (2024)|38.1|62.7|–|22.3|–|37.6|
|LayoutLLM (2024)|78.7|62.2|71.0|74.3|–|–|
|LLaVA-Read (2024c)|36.9|–|58.3|71.0|74.6|36.4|
|LayTextLLM (2024)|64.0|96.5|95.8|77.2|–|–|
|LayTokenLLM(2025b)|71.0|75.4|–|85.1|–|–|
|GPE (2025a)|82.6|86.9|97.8|78.1|–|–|
|PDF-WuKong (2025)|85.1|–|–|76.9|80.0|61.3|
|DocLayLLM (2024)|80.7|79.4|84.4|72.8|–|–|
|AlignVLM (2025)|–|–|–|81.2|75.0|53.8|
|DocAssistant (2025)|–|–|–|89.8|81.4|66.7|
|DocThinker (2025)|-|–|81.4|80.2|–|69.7|
|**OCR-Free**|||||||
|KOSMOS-2.5 (2023)|–|–|–|81.1|62.3|41.3|
|mPLUG-DocOwl (2025)|–|–|–|62.2|57.4|38.2|
|UReader (2023b)|–|–|–|65.4|59.3|42.2|
|TGDoc (2023)|1.7|–|3.0|9.0|11.7|12.8|
|UniDoc (2023)|1.2|–|1.4|6.5|10.5|13.8|
|DocPedia (2024)|40.1|–|57.7|49.3|47.8|15.5|
|HRVDA (2024a)|–|89.3|89.3|91.0|72.1|43.5|
|Vary-base (2024)|–|–|–|76.3|66.1|–|
|mPLUG-DocOwl 1.5 (2024)|–|–|–|81.6|70.5|50.4|
|HVFA (2024)|–|–|–|72.7|63.3|45.9|
|mPLUG-DocOwl2 (2025)|–|–|–|80.7|70.0|46.4|
|Texthawk (2024a)|–|–|–|76.4|66.6|50.6|
|Texthawk2 (2024b)|–|–|–|89.6|81.4|67.8|
|TextMonkey (2024c)|65.5|67.5|47.0|73.0|66.9|28.6|
|Llavar-7B (2024d)|1.7|13.6|2.4|11.6|–|–|
|TokenCorrCompressor (2024b)|–|–|–|78.3|68.9|50.2|
|DocKylin (2024a)|25.5|–|49.5|77.3|66.8|46.6|
|Marten (2025b)|44.4|–|80.4|92.0|81.7|75.2|
|PP-DocBee (2025)|–|–|–|90.6|74.6|66.2|
|TokenFD (2025)|42.2|–|81.9|94.2|86.6|76.5|



Table 9: Performance comparison between OCR-dependent and OCR-free document understanding frameworks across benchmark datasets. 

26 

|**Model**|**Type**|**Venue**|**Year**|**MPDocVQA**|**DUDE**|
|---|---|---|---|---|---|
|Longformer|General VLPM|Preprint|2020|55.1|20.3|
|BigBird|General VLPM|NeurIPS|2020|58.5|26.3|
|GPT-4v|General MLLM|–|2023|–|53.9|
|Idefcs3-8B|General MLLM|Preprint|2024|67.2|38.7|
|LLaVA-next-interleave-7B|General MLLM|Preprint|2024|44.9|28.0|
|Hi-VT5|OCR-dependent VLPM|PR|2023|61.8|35.7|
|GRAM|OCR-dependent VLPM|CVPR|2024|**83.0**|53.4|
|InstructDoc|OCR-Dependent VLPM|AAAI|2024|–|46.8|
|mPLUG-DocOwl2|OCR-free VLPM|Preprint|2024|69.4|46.7|
|PDF-WuKong|OCR-Dependent VLPM|Preprint|2024|76.9|56.1|
|LayTokenLLM|OCR-Dependent VLPM|CVPR|2025|74.3|52.0|
|DocThinker (2025)|OCR-Dependent VLPM|ICCV|2025|–|**56.8**|



Table 10: Performance comparison of state-of-the-art models on MPDocVQA and DUDE benchmarks. Best scores are highlighted in red. 

27 

