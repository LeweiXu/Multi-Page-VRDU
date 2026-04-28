# **GRAM: Global Reasoning for Multi-Page VQA** 

Tsachi Blau[*] Technion, Israel 

Sharon Fogel AWS AI Labs 

Roi Ronen[*] Alona Golts[†] Roy Ganz[*] Technion, Israel AWS AI Labs Technion, Israel 

Elad Ben Avraham Aviad Aberdam AWS AI Labs AWS AI Labs 

Shahar Tsiper Ron Litman[‡] AWS AI Labs AWS AI Labs 

## **Abstract** 

_The increasing use of transformer-based large language models brings forward the challenge of processing long sequences. In document visual question answering (DocVQA), leading methods focus on the single-page setting, while documents can span hundreds of pages. We present GRAM, a method that seamlessly extends pretrained single-page models to the multi-page setting, without requiring computationally-heavy pretraining. To do so, we leverage a single-page encoder for local page-level understanding, and enhance it with document-level designated layers and learnable tokens, facilitating the flow of information across pages for global reasoning. To enforce our model to utilize the newly introduced document tokens, we propose a tailored bias adaptation method. For additional computational savings during decoding, we introduce an optional compression stage using our compressiontransformer(C-Former ),reducing the encoded sequence length, thereby allowing a tradeoff between quality and latency. Extensive experiments showcase GRAM’s stateof-the-art performance on the benchmarks for multi-page DocVQA, demonstrating the effectiveness of our approach._ 

## **1. Introduction** 

Document understanding, particularly in the context of DocVQA, has gained substantial research interest [5, 6, 16, 25, 36, 37] and offers a wide array of practical applications, focusing on data extraction and analysis of single page documents. However, Multi-Page DocVQA (MPDocVQA) poses a more realistic challenge, considering that the majority of documents, including contracts, manuals 

> *Work conducted during an internship at Amazon. 

> †Corresponding author: alongolt@amazon.com 

> ‡Corresponding author: litmanr@amazon.com 

**==> picture [190 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
Decoder<br>C-Former Compression Transformer<br>t I t<br>Multi Page Encoder<br>Doc Sub-Layer<br>Page Sub-Layer<br>anne °seen Doc Attention ——<br>Global-Local Encoder Block<br>\ 6 cenit<br>Doc Sub-Layer  Page<br>Attention<br>Page Sub-Layer<br>How many<br>+ diagrams<br>are there?<br>**----- End of picture text -----**<br>


Figure 1. **An Overview of GRAM.** We suggest an interleaved encoder architecture combining page- with document-attention layers, allowing information to propagate between different pages. An optional compression transformer (C-former) is introduced to allow a trade-off between quality and latency. 

and scientific papers, often extend well beyond a single page. Despite the practical relevance of MPDocVQA, it has received limited attention, primarily due to the absence of suitable datasets. Two recently introduced datasets, MPDocVQA [33] and DUDE [18], have opened up new avenues for MP-DocVQA research. 

Recent DocVQA approaches rely on transformers [35], at the heart of their architecture. While transformers are a powerful tool, they face challenges when dealing with long input sequences [4, 7, 10–12, 27, 38]. This difficulty stems from the self-attention mechanism, which scales quadratically in terms of computation and memory, with respect to the input sequence length. Addressing this limitation has attracted significant research efforts, primarily in the field 

of natural language processing (NLP). Proposed NLP-based solutions can be divided into two main directions: The former aims to modify the attention mechanism to cut computational costs [4, 7, 38]. The latter involves altering the positional embedding mechanism to improve performance on longer sequences, with minimal fine-tuning [11, 12, 27]. 

A possible option of tackling MPDocVQA is to extend NLP-based approaches to handle multi-modal document data, including visual representations, along with OCR text and corresponding 2D locations and relative page position. However, this requires extensive pre-training, with relatively scarce multi-page document data, and thus is also sub-optimal in terms of performance. Instead, we opt for leveraging powerful single-page DocVQA models, especially pretrained on millions of single-page documents, and finetuning them to the multi-page scenario. For this purpose, we combine concepts of local (page) and global (document) tokens, which promote an exchange of information within and across pages, while keeping computational cost in check. We choose pages as atomic units in our proposed scheme, as page structure often represent a semantic unit in DocVQA. 

We present GRAM ( **G** lobal **R** e **A** soning for **M** ulti-page VQA), a novel approach for endowing multi-page processing capabilities to existing single-page DocVQA models. Alongside page tokens that encapsulate both textual and visual contents of each page, we introduce doc(ument) learnable tokens, which aim is dispersing global information across all pages. These two sets of tokens interact within our newly-devised two-stage encoder blocks. The initial stage utilizes an existing single-page layer and enhances it by including both page and doc tokens as input, allowing them to freely interact. In the second stage, we prioritize computational efficiency by restricting self-attention solely to the global doc tokens. This global reasoning layer captures collective information from multiple pages, enabling the system to respond to cross-page inquiries, as illustrated in Fig. 1. Considering that doc tokens did not appear in pretraining, to boost their significance during finetune, we employ a designated bias adaptation mechanism which strikes a balance between local and global learnable tokens. 

While our method inherently deals with long sequences, we circumvent a quadratic reliance on sequence length by segmenting the document into pages — its semantically logical parts. We restrict interaction solely among doc learnable tokens, across all pages, thereby mitigating the computational burden of depending quadratically on the page count. Apart from encoding, the auto-regressive decoding stage poses a computational burden in long sequences. To this end, we introduce a compression stage that precedes the decoder, implemented with a compression transformer, termed **CFormer** . The CFormer receives the concatenated output of all pages and compresses it to a 

much shorter sequence, distilling the most pertinent information in the document. Our key contributions are: 

- We propose GRAM, an approach to endow single-page DocVQA methods with multi-page capabilities, without pretraining, allowing the model to process multi-page documents, while preserving single-page performance. 

- We introduce document learnable tokens and bias adaptation that enable an effective communication and collaboration between individual pages to support reasoning over multiple page documents. 

- Our C-Former module suggests a trade-off between accuracy and compute, distilling information from multi-page sequences into more compact representations. 

- We obtain SOTA results over the MPDocVQA and DUDE datasets, and provide extensive ablations to each component in our method. 

## **2. Related Work** 

**Long Sequence Approaches** are an active field of research in NLP, aiming to improve the design of chat-systems [24] and image instruction tasks [19, 21]. In these applications, the ability to manage and process long sequences is vital, as conversations cannot be cut short, or limited to just a few interactions. Common approaches to tackle long sequences include sparse attention mechanisms [4, 7, 38] and methods to improve results on long sequences during inference [11, 27, 28]. ‘Sliding window’ approaches of limiting the range of neighbors each token can attend to, lead to a significant reduction in computation and memory consumption. Prominent works of this kind include LongFormer [7], where each token attends to a set of its nearest neighbors, along with additional global tokens. The work of Big-Bird [38] adds additional non-neighboring tokens at random, whereas Colt5 [4] uses the same sliding window approach, but performs heavier computations for important tokens and shallow operations for filler words or punctuation. Although Tito _et. al._ [33] have demonstrated that the above approaches do not perform as well on the task of MPDocVQA, we do incorporate the ideas of combining both local and global tokens throughout encoding to expand the attention onto additional pages. 

**DocVQA** has attracted increasing attention [5, 6, 9, 16, 25, 26, 31, 36, 37] with the introduction of the DocVQA dataset by Mathew et al. [22]. Most methods in DocVQA leverage OCR [1–3, 20, 23] to input both text and layout information (bounding box coordinates and possibly font type) into the model, where some further explore different techniques to combine the two types of data streams, or alternatively, clever schemes of pretraining. DocVQA methods can be roughly divided to two categories: extractive and abstractive. Extractive methods [16, 25, 36, 37] rely on the fact that the explicit answer resides in the written text, thus only output a corresponding text span within the input 

**==> picture [366 x 272] intentionally omitted <==**

**----- Start of picture text -----**<br>
Decoder<br>ee Global-Local Encoder Layer<br>C-Former<br>Global-Local Encoder Layer<br>Multi Page Encoder<br>Global-Local Encoder Layer<br>E [j]<br>doc<br>Doc Sub-Layer<br>Global-Local Encoder Layer<br>Global-Local Encoder Layer<br>E [j] E [j] E [j]<br>page page page<br>Page Sub-Layer Page Sub-Layer Page Sub-Layer<br>How many<br>diagrams<br>are there? Xij Gij<br>page  doc<br>tokens tokens<br>(a) (b)<br>**----- End of picture text -----**<br>


Figure 2. **GRAM Architecture. (a)** Depicts a high-level architecture overview. For each page, the visual, textual and question tokens are concatenated together with learnable doc tokens (darker color shade). The processed information is fed into the multi-page encoder. The encoder output can be fed directly into the decoder to create the final prediction. Optionally, a compression model, C-Former , can be used between the encoder and the decoder to compress the encoder output into a predetermined length, thus reducing overall latency for long documents. **(b)** Shows a global-local encoder layer, containing two sub-layers. The first sub-layer uses self-attention that operates on each page separately, while the second applies a self-attention step on the doc tokens to fuse information between the different pages. The corresponding tokens are then routed back to their respective page and go into the next global-local encoder layer. 

sequence. Abstractive methods [5, 6, 14, 26, 31], on the other hand, have the capacity to generate free-form answers which do not necessarily appear in the text, thus providing flexibility in real-world applications. Notably, existing research in DocVQA does not scale in a straightforward way to deal with the more realistic multi-page scenario. 

**MPDocVQA** has recently gained momentum with the launch of two new multi-page datasets: MP-DocVQA [33] and DUDE [18], offering two separate recipes to tackle longer documents. The first approach of Tito _et. al._ , referred to as HiVT5 [33], suggests compressing the encoding of each page separately, and feeding the decoder with the concatenation of the compressed outputs from each single-page encoder. While this approach is advantageous in terms of computation, we later show the compression may severely hinder the results. In addition, there is no communication between the single-page encoders until the final stage of decoding, whereas in our method, we allow for exchange of page and document information throughout all stages of encoding. Another prominent approach, proposed by Landeghem _et. al._ [18], which relatively preserves qual- 

ity, involves concatenating all the pages into one long sequence and feeding it to a standard encoder-decoder structure. This, however, poses a heavy computational burden as transformers’ self-attention component scales quadratically with input sequence length. 

## **3. GRAM** 

## **3.1. Base Architecture** 

The underlying idea in our approach is using existing encoder-decoder single-page models for document understanding and extending them to multi-page scenarios, without additional pretraining. In this work, we provide such a recipe over the notable DocFormerv2 [6]. For the sake of completeness, DocFormerv2 is a T5-based [28] encoderdecoder transformer model which operates over both visual and textual features to support document understanding. Each page is represented by textual features **T** _∈ R[N][t][×][d]_ which encapsulate OCR tokens and their corresponding 2D bounding box positions [36], along with visual **V** _∈ R[N][v][×][d]_ and question **Q** _∈ R[N][q][×][d]_ embeddings. Where _Nt_ , _Nv_ and 

_Nq_ are the lengths of the OCR, visual features and the question. The output result **Y** is obtained by passing the concatenated inputs through the encoder-decoder model, 

**==> picture [179 x 11] intentionally omitted <==**

where **E** and **D** , are the encoder and decoder, respectively. 

Our method uses these basic building blocks in designing a multi-page solution. To this end, we introduce a bi-level global-local encoder, as illustrated in Fig. 2. At the local page-level of each block, we utilize the layers of the existing single-page encoder **E** to process each page separately, together with learnable doc tokens. Next, we introduce a slim global layer in each block that facilitates communication between doc tokens across all pages. This bi-level localized processing ensures the model can understand the content of each page effectively, while also combining information across pages in the document. After _M_ such blocks, we feed the encoded features from all pages into the existing decoder **D** to produce the overall output. 

## **3.2. Global-Local Reasoning** 

To operate on multiple pages we break down the document to _K_ pages, and the single-page encoder to _M_ encoder layers, **E** _[j] , j_ = 0 _, ..., M −_ 1. We then construct _M_ blocks, with two sub-layers each. The first page sub-layer originates from the existing pretrained encoder layer, referred to as **E** _[j] page_[, and operates in parallel, with shared weights, for] all pages in the document. This layer receives both page and doc tokens. The second, newly introduced, document layer **E** _[j] doc_[collects][only][the][doc][tokens][from][all][pages][and] promotes sharing information across all of the document. 

Formally, we augment the input of the standard singlepage encoder with page-specific indexing ( **T** _i,_ **V** _i,_ **Q** ) and incorporate page-positional embedding **P** _i_ to both text and visual features, where _i_ = 0 _, ..., K −_ 1: 

**==> picture [190 x 12] intentionally omitted <==**

Next, we formulate our bi-level global-local block. The input to the first page-level sub-layer in each block is the concatenation of the textual, visual and question features, denoted **X** _[j] i_[= concat(˜] **[T]** _[i][,]_ **[V]**[˜] _[i][,]_ **[ Q]** _[i]_[)][, along with page-] specific doc tokens **G** _[j] i[∈][R][N][g][×][d]_[,] 

**==> picture [200 x 15] intentionally omitted <==**

Here, the features undergo self-attention, normalization and feed-forward layers. The layer output **X** _[j] i_[+1] is passed on as input to the next bi-level block, whereas only the doc tokens **G**[˜] _[j] i_[+1] , enter the second doc sub-layer, which again includes self-attention, normalization and feed-forward 

**==> picture [210 x 14] intentionally omitted <==**

**==> picture [210 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
Page SA Doc SA<br>Page 1<br>Page 2<br>Page k<br>(a) (b) (c)<br>Long Seq Encoder  Multi-Page Encoder<br>Enc Layer<br>Global-Local Layer Page Sub-Layer  Doc Sub-Layer Page Sub-Layer  Doc Sub-Layer Page Sub-Layer  Doc Sub-Layer<br>**----- End of picture text -----**<br>


Figure 3. **Global-Local Attention** : In long sequence approaches (a), attention is applied jointly to the entire sequence of concatenated local and global tokens. Our method, separates the computation into two steps — page-level (b) and document-level (c)— leveraging the natural division of documents into pages. 

In this stage, the doc tokens can interact and pass information from page to page, after which being passed on to the next block, as depicted in Eq. (3). This design allows information to flow between pages while keeping computational costs in check. When concluding the traversal over _M_ such layers, the outputs across all pages are concatenated and fed to the decoder **D** , 

**==> picture [202 x 13] intentionally omitted <==**

To visualize the difference between the attention masks in our method, we compare it with previous long sequence approaches [4, 7, 38] in Fig. 3. These prior methods optimize computation by using attention masking on nearby tokens and allowing limited global connections. However, naively applying such methods to multi-page documents will treat it as a single stream, which does not consider the division into pages. Our global-local blocks, with a twostage attention-masking mechanism, better suit multi-page documents. In addition, our two-level design benefits from existing, extensively pretrained single page models. 

## **3.3. Bias Adaptation** 

An already-pretrained model, introduced with a new stream of data, might disregard it altogether [13, 14, 34, 39]. To overcome this, we force the system to account for the newly-introduced doc tokens by modifying the encoder’s bias method. Originally, the bias method intervenes in the attention mechanism, diminishing the relationships between distant tokens. However, in our specific case, the distance between doc and page tokens does not represent their actual relevance. To enforce the encoder to pay closer attention to the doc tokens, we assign them a positive constant bias value. Particularly, we replace the values in the bias matrix, corresponding with the doc tokens, with fixed ones. Instead of a single bias value, we utilize a different value 

for each attention head, as performed in ALiBi [27], enabling more fine-grained control of the global features in each head. Specifically, the constant doc bias value is set to _c·_ 2[1] _[a]_[, where] _[ c]_[ is a constant and] _[ a]_[ is the attention head index.] This yields a decaying bias value across different attention heads, resulting in hierarchical importance of the document information, where the first heads are more oriented towards doc tokens and the last towards page tokens. 

## **3.4. Compression Transformer** 

Our global-local solution to MP-DocVQA resolves the problematic quadratic dependency on the number of pages _K_ during encoding. However, the auto-regressive decoding complexity scaling linearly with _K_ also poses a practical challenge during inference time, as we later discuss in Sec. 3.5. To alleviate this burden, we place an optional transformer-based model, named C-Former ( **C** ompression Trans **Former** ), between the encoder outputs and the decoder, as depicted in Fig. 2. The C-Former has the ability to revise the information across all pages and distill only the important details, required to correctly answer the question. 

Specifically, the C-Former is a light-weight transformerbased decoder [28], denoted as **D** _C_ , featuring crossattention, layer norm and feed-forward layers in each block. The input to C-Former includes _Nc_ learnable tokens **C** _∈ R[N][c][×][d]_ , concatenated with the input question **C**[˜] = concat( **C** _,_ **Q** ). In addition, we feed it with the outputs of the global-local interlaced encoder, concatenated to one long sequence, referred as **O** , where **O** = concat( _{_ **X** _[M] i[,]_ **[ G]** _[M] i[}][K] i_ =0 _[−]_[1][)][.][The output of C-Former is thus] 

## **O** _C_ = **D** _C_ ( _Q_ = **C**[˜] _, K_ = **O** _, V_ = **O** ) _,_ 

where we pass forward only the first set of _Nc_ output embeddings and ignore the rest, setting the output sequence dimension to _Nc_ . C-Former offers flexibility in controlling the tradeoff between ANLS quality and computational efficiency by controlling the output sequence length _Nc_ . 

## **3.5. Computation Analysis** 

Next, we turn to provide a thorough computational complexity analysis. We consider a document that comprises of _K_ pages, each with _N_ tokens, and the maximum answer length is _L_ . For simplicity, we assume that all encoders and decoders have one layer. The na¨ıve way to support multipage documents is using an existing single-page encoderdecoder model, fusing all of the textual page inputs together, and feeding them as one long sequence. We refer to this approach as ‘concat’. The self-attention complexity of such a configuration scales quadratically with the sequence length, _O_ (( _N · K_ )[2] ). Conversely in our method, we operate on the document pages with two alternating encoding stages in each layer. The first stage performs a self-attention over both the page and doc tokens. Hence, the complexity of 

such sub-layer is _O_ (( _N_ + _Ng_ )[2] _· K_ ), where _Ng_ is the number of doc tokens. The second stage features a self-attention operation over the doc tokens, across all pages in the document. The complexity of this operation is _O_ (( _Ng · K_ )[2] ). Overall, the total complexity for one global-local encoder block is _O_ (( _N_ + _Ng_ )[2] _· K_ + ( _Ng · K_ )[2] ). Since _Ng_ is a constant, and the number of pages is usually less than the number of words in each page ( _K < N_ ), we obtain a complexity of _O_ ( _N_[2] _· K_ ), which is not quadratic in _K_ . 

Prior to decoding, the outputs of all per-page encoders are concatenated, thus the output sequence length is ( _N_ + _Ng_ ) _· K_ . Since the decoder is auto-regressive, its complexity depends quadratically on the maximum output length, _L_ , namely, _O_ (( _N_ + _Ng_ ) _· K · L_[2] ) = _O_ ( _N · K · L_[2] ). Since this operation of decoding is performed iteratively during inference, the combined sequence length ( _N_ + _Ng_ ) _· K_ becomes computationally heavy. To alleviate this concern, we propose an optional C-Former model, which performs compression prior to decoding. The overall complexity in this decoding scheme includes passing through the C-Former and then through the decoder, leading to _O_ (( _N_ + _Ng_ ) _· K · Nc_ ) + _Nc · L_[2] ) which is equivalent to _O_ ( _N · K_ + _L_[2] ), since _Nc_ is a constant, denoting the number of compression tokens in C-Former. 

## **4. Experiments** 

## **4.1. Experimental Settings** 

**Datasets and Metrics** The MPDocVQA dataset [33] features 46K questions, spanning over 48K images, and includes layout elements as figures, tables, lists and diagrams, with printed, handwritten and typewritten text. MPDocVQA contains mostly extractive questions, for which answers are present in the given text. DUDE is smaller in size (23.7K questions over 3K documents), but offers complex questions that require a reader to rationalize beyond the written text content. We report our results using the ANLS metric, introduced in [8], computing a generalized accuracy. Results for DUDE can be broken apart to several types of questions, categorized to four groups: ‘extractive’ – for which the answer is found directly in the text; ‘abstractive’ – requiring a free-form answer that does not necessarily appear in the document; ‘list of answers’ – requiring a list of answers, as opposed to a single one, and ‘unanswerable’ – where the result cannot be determined using the text. 

**Implementation Details** Our underlying architecture is based on Docformerv2 [6]. Recall, our interlaced encoder features _M_ blocks (12 in ‘base’ and 24 in ‘large’), where each block contains a page sub-layer which originates from an extension of Docformerv2’s encoder layer. Every structure contains self-attention, normalization and feed-forward 

|**Method**<br>**Params**|**MPDocVQA**<br>**DUDE**<br>**ANLS**<br>**ANLS**<br>**ANLS per Question Type**<br>Extractive<br>Abstractive<br>List of answers<br>Unanswerable|**MPDocVQA**<br>**DUDE**<br>**ANLS**<br>**ANLS**<br>**ANLS per Question Type**<br>Extractive<br>Abstractive<br>List of answers<br>Unanswerable|
|---|---|---|
|Longformer [7]<br>148_M_<br>BigBird [38]<br>131_M_<br>LayoutLMv3 [38]<br>125_M_<br>Hi-VT5_†_<br>_beamsearch_ [15]<br>316_M_<br>Hi-VT5[33]<br>316_M_<br>Hi-VT5*<br>257_M_<br>DocFormerv2_concat_ [6]<br>257_M_|55_._06<br>58_._54<br>55_._13<br>_−_<br>62_._01<br>60_._78<br>69_._67|27_._14<br>43_._58<br>8_._55<br>10_._62<br>10_._78<br>26_._27<br>40_._26<br>7_._11<br>8_._46<br>12_._75<br>20_._31<br>32_._60<br>8_._10<br>7_._82<br>8_._82<br>35_._74<br>28_._31<br>32_._98<br>10_._60<br>62_._90<br>23_._06<br>17_._60<br>33_._94<br>6_._83<br>61_._67<br>23_._86<br>7_._21<br>16_._56<br>3_._53<br>72_._77<br>44_._21<br>41_._66<br>41_._86<br>15_._13<br>**65**_._**19**|
|GRAM_C−F ormer_<br>286_M_|70_._80|40_._07<br>40_._43<br>39_._61<br>11_._42<br>52_._55|
|GRAM<br>281_M_|**73**_._**68**|**46**_._**15**<br>**46**_._**07**<br>**44**_._**82**<br>**15**_._**27**<br>62_._18|
|T5-2D [18]<br>770_M_<br>DocGptVQA [30]<br>_>_3_._5_B_<br>DocBlipVQA [29]<br>_>_3_._5_B_<br>Hi-VT5* [33]<br>784_M_<br>DocFormerv2_concat_ [6]<br>784_M_|_−_<br>_−_<br>_−_<br>71_._35<br>76_._40|46_._06<br>**55**_._**65**<br>**50**_._**81**<br>5_._43<br>**68**_._**62**<br>50_._02<br>51_._86<br>48_._32<br>28_._22<br>62_._04<br>47_._62<br>50_._69<br>46_._31<br>**30**_._**73**<br>55_._22<br>28_._89<br>18_._21<br>26_._17<br>6_._84<br>58_._99<br>48_._44<br>50_._82<br>48_._06<br>17_._67<br>59_._04|
|GRAM_C−F ormer_<br>864_M_|77_._60|45_._47<br>47_._63<br>44_._91<br>14_._34<br>56_._99|
|GRAM<br>859_M_<br>|**80**_._**32**|**51**_._**15**<br>53_._67<br>50_._35<br>18_._40<br>63_._23|
|Hi-VT5*_†_ [33]<br>784_M_<br>DocFormerv2_†_<br>_concat_ [6]<br>784_M_|73_._51<br>76_._77|49_._18<br>49_._29<br>48_._35<br>13_._30<br>**65**_._**95**<br>50_._79<br>52_._70<br>49_._61<br>17_._33<br>65_._14|
|GRAM_†_<br>_C−F ormer_<br>864_M_<br>|78_._12|50_._97<br>55_._15<br>50_._46<br>17_._26<br>61_._04|
|GRAM_†_<br>859_M_|**79**_._**67**|**53**_._**36**<br>**56**_._**83**<br>**52**_._**32**<br>**19**_._**96**<br>65_._43|



Table 1. **Quantitative Results** . We present ANLS results for the MPDocVQA [33] and DUDE [18] test sets. The methods are grouped according to the model type and size, starting from encoder-only models (top), T5-base models (middle) and T5-large models (bottom). _[†]_ denotes training with both MPDocVQA and DUDE. 

|**Method**|**Training Data**<br>**ANLS**<br>**DocVQA MPDocVQA**<br>**DocVQA MPDocVQA**|**Training Data**<br>**ANLS**<br>**DocVQA MPDocVQA**<br>**DocVQA MPDocVQA**|
|---|---|---|
|**DocFormerv2**_concat_|✓<br>✗<br>✗<br>✓<br>✓<br>✓|86_._60<br>72_._73<br>85_._28<br>76_._40<br>86_._47<br>75_._37|
|**GRAM**|✓<br>✗<br>✗<br>✓<br>✓<br>✓|86_._70<br>73_._12<br>85_._29<br>80_._32<br>86_._32<br>78_._66|



Table 2. **DocVQA vs. MPDocVQA Performance.** Test results over both datasets using the large model variants. A checkmark denotes whether a dataset was included in training or not. 

from C-Former is _Nc_ = 256. Finally the decoder is initialized with pretrained weights from DocformerV2. 

The model is trained with the Hugging Face Trainer [17] for 200 _k_ steps, starting with a warm-up of 1 _k_ steps, with linear learning rate decay. We use learning rates of 3 _e[−]_[5] and 1 _e[−]_[4] for the already pretrained encoder and decoder weights, versus the newly initialized doc sub-layer weights. Training is performed on a cluster of 8 _× A_ 100 GPUs, each with 40 _GB_ of RAM. During training, each page encoder receives 800 tokens, dealing with up to 4 pages. During testing, we increase the maximum length of tokens to 8 _,_ 000. 

layers. We extend the page layer from Docformerv2 to feature also the doc learnable embeddings. The second doc sub-layer is similar in structure to the first sub-layer, only it is initialized from scratch, with the following specification: _dff_ = 1024, _dkv_ = 64, _nheads_ = 4, _d_ = 256. We implement 32 doc learnable tokens for each page, uniformly initialized to random values. For bias adaptation, the initial bias value is set to _c_ = 20, with variations between encoder heads, as described in Sec. 3.3. We incorporate an additional optional compression stage using C-Former – a randomly-initialized T5 [28] tiny decoder, with an encoder mask instead of a causal one. The output sequence extracted 

**Baselines** We report the results of previous work on both MP-DocVQA and DUDE datasets (if those exist), including the NLP-based Longformer [7] and BigBird [38], which were adapted to MPDocVQA by [33]; LayoutLMv3 [16], originally designed for DocVQA; and Hi-VT5 [33] and T52D [32], specifically suggested for MP-DocVQA Task. We also add for reference the results of methods published in the leader-boards of MPDocVQA and DUDE, which do not have corresponding papers, including DocGptVQA [30], DocBlipVQA [30], and Hi-VT5 _beamsearch_ [15] ([15] was trained on both MP-DocVQA and DUDE). In our approach, we present two variations: GRAM and GRAM _C−F ormer_ . While GRAM utilizes the full length of the encoder output, 

Figure 4. Qualitative comparison between our approach and Hi-VT5 [33] indicate that the integration of our global-local encoder enhances reasoning capabilities, especially when the inquiries require multi-page context. 

GRAM _C−F ormer_ allows the user to control the trade-off between performance and latency. 

To ensure a fair comparison, since we use the pretrained model of DocFormerv2 [6], we implement two additional baselines, referred to as Hi-VT5* and DocFormerv2 _concat_ . The first follows a similar structure as Hi-VT5 [33], with the encoder originating from DocFormerv2, however without the page answer prediction, as it does not exist in DUDE. The second recreates the approach of [18], where only the textual tokens of all pages are concatenated to one long sequence, then passing through the DocFormerv2 model. The second approach poses a computational burden, thus we use only 600 tokens during training per page, with up to 4 pages, and during test only 400 tokens. 

## **4.2. Results** 

We present the performance of our method over the MPDocVQA [33] and DUDE [18] datasets in Tab. 1. The methods are divided into three groups: the top contains encoderonly, and methods that rely on the T5-base model (up to 316 _M_ parameters); the middle section, approaches that use the T5-large model (over 770 _M_ parameters), and finally the bottom, T5-large models, trained on both datasets. 

As can be seen, in the first group, the encoder-only NLP methods, LongFormer [7], BigBird [38] and LayoutLMv3 [16] can only handle relatively well ‘extractive’ style tasks as in MPDocVQA dataset [33], but often struggle with ‘abstractive’ questions that are more abundant in DUDE [18]. As to T5-‘base’ models, versus our best competitor Docformerv2 _concat_ , we obtain an improvement of (+4% _,_ +1 _._ 9%) on MP-DocVQA and DUDE datasets. As to methods that combine an additional compression before decoding (Hi-VT5, Hi-VT5*), our C-Former achieves an increase in (+8 _._ 8% _,_ +16 _._ 2%) over the best candidates on the MP-DocVQA and DUDE datasets. 

As for the group of ‘large’ models, we include the results of T5-2D [18] DocGptVQA [30] and 

DocBlipVQA [29]. Note that our model surpasses DocFormer _concat_ , the primary baseline, achieving improvements of (+3 _._ 9% _,_ +2 _._ 7%) on MP-DocVQA and DUDE, respectively. We also outperform DocGptVQA [30], a method that appears in the leaderboard of DUDE, by +1 _._ 1%, thereby obtaining SOTA results for GRAM ‘large’. 

The final category showcases large encoder-decoder models, fine-tuned on both MP-DocVQA and DUDE training sets, showcasing the benefits of augmented training data. GRAM consistently demonstrates performance gains over the baseline, illustrating its robustness across different datasets and training scenarios. Next, we present in Tab. 2 the effect of training on DocVQA vs. MPDocVQA . Our method achieves performance on-par on the single page task, while enhancing performance on the multi-page scenario by +3 _._ 3%, compared to the baseline. 

In Fig. 4, We show qualitative results on the DUDE dataset of GRAM versus Hi-VT5* [33]. Our method demonstrates proficiency in addressing questions that involve attention over multiple pages ( _‘how many diagrams are there’_ ), an increased visual analysis capability ( _‘Which month shows the hurricane?’_ ), and heightened abstractive ability ( _‘What is the EPS code for Little Rock?’_ ). 

## **5. Ablation Study** 

We perform an ablation study on our approach, evaluating the influence of each constituent component using DUDE’s validation set [18]. This validation set enables the grouping of documents by their respective page counts: 1, 2–4, 5–10, 11–end, encompassing 1747 _,_ 2259 _,_ 1062 _,_ 1241 samples in each category, respectively. Our investigation delves into the impact of the number of doc tokens and the bias adaptation methods. Moreover, we employ the C-Former for sequence compression, adjusting the compression ratio and examining the balance between performance and latency (see supplementary for more details). 

|**#Doc**<br>**Tokens**<br>**Bias**<br>**Type**<br>**Compression**<br>**Dimension**|**ANLS by Number of Pages**<br>**DUDE validation dataset**<br>All<br>1<br>2-4<br>5-10<br>11-end|
|---|---|
|✗<br>✗<br>✗|46_._16<br>47_._18<br>48_._66<br>43_._34<br>42_._57|
|16<br>✗|46_._39<br>48_._35<br>49_._06<br>43_._16<br>41_._56|
|32<br>Decaying<br>✗|**47**_._**88**<br>**49**_._**29**<br>**49**_._**90**<br>**45**_._**90**<br>**43**_._**94**|
|64<br>✗|46_._70<br>47_._98<br>49_._22<br>44_._00<br>42_._60|
|32<br>✗<br>✗<br>Constant<br>✗|47_._52 **49**_._**85 49**_._**93** 44_._90<br>42_._10<br>46_._14<br>47_._41<br>48_._13<br>44_._44<br>42_._19|
|Decaying<br>✗|**47**_._**88**<br>49_._29<br>49_._90<br>**45**_._**90**<br>**43**_._**94**|



Table 3. **GRAM Ablation Study** . Results on DUDE validation set ablating over (a) the dimension of doc tokens, (b) the attention bias employed and (c) the C-former input dimension. 

**GRAM Components** We focus our initial exploration on the impact of the number of doc tokens _Ng_ . As can be seen in Tab. 3, while _Ng_ = 16 leads to performance onpar with not using doc tokens at all, for the optimal value of _Ng_ = 32, we obtain an increase of +1 _._ 7% in ANLS. Shifting our focus to bias adaptation methods, Tab. 3 shows that using constant bias has a negative effect on the results, suggesting this method is not flexible enough in maintaining a balance between the page and doc tokens. However, our decaying bias-adaptation approach does improve results overall, versus no-bias (+0 _._ 36%), especially for longer documents (+1% improvement for 5-10 pages and +1 _._ 84% for 11 pages and more). This is to be expected, since incorporating new doc tokens and increasing their importance can potentially affect single-page performance. Finally, in Tab. 4, we reinforce our choice of pages as semantic logical units for MPDocVQA. We first ablate our method with and without page embedding. Next, we compare our pagebased division with varying fixed-length division of tokens for encoder. Results in Tab. 4 clearly demonstrate an advantage towards page-level encoding in MPDocVQA. This aligns with our initial assumption that structured documents are often designed with page-division in mind. 

**Performance-Latency Trade-off** We assess the impact of C-Former on performance, considering compression output lengths of 8 _,_ 32 _,_ 256 _,_ 1024 _,_ 4096. Note, performance gradually improves with an increase in the compression output length. However, longer output lengths correspond to heightened model latency. Note that using C-Former for shorter documents can be redundant, as there is little to no compression compared to the input sequence length and results decrease. In Fig. 5, we scrutinize the tradeoff between computational efficiency and compression rate 

|**Page**<br>**Embedding**<br>**Segment**<br>**Length**|**ANLS by Number of Pages**<br>**DUDE validation dataset**<br>All<br>1<br>2-4<br>5-10<br>11-end|
|---|---|
|✓<br>✗|**47**_._**88**<br>**49**_._**29**<br>**49**_._**90**<br>**45**_._**90**<br>**43**_._**94**|
|✗<br>✗|46_._12<br>48_._74<br>48_._11<br>43_._59<br>40_._99|
|✓<br>256<br>✓<br>512<br>✓<br>1024|45_._22<br>46_._38<br>46_._69<br>44_._13<br>41_._83<br>45_._09<br>45_._90<br>47_._32<br>42_._65<br>41_._98<br>44_._39<br>44_._98<br>46_._63<br>41_._69<br>41_._78|



Table 4. **The Significance of Pages as Semantic Units** . Results on DUDE validation set ablating over (a) utilization of pageembedding, (b) segment length for fixed-size encoding inputs. 

**==> picture [211 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
OOM<br>**----- End of picture text -----**<br>


Figure 5. **Latency comparison** . We compare the dependency between overall latency and the number of pages in input document for GRAM, GRAM _C−F ormer_ , DocFormerv2 _concat_ and Hi-VT5. 

by comparing to DocFormerv2 _concat_ [6] and Hi-VT5* [33]. We discover that DocFormerv2 _concat_ reaches a memory limit at approximately 20 pages, due to its quadratic memory increase with sequence length. At this juncture, GRAM _C−F ormer_ surpasses DocFormerv2 _concat_ by performing 3 _._ 5 seconds faster. Notably, GRAM _C−F ormer_ can gracefully handle documents surpassing 300 pages, effectively bridging the gap between performance and latency. 

## **6. Conclusions** 

Our method, termed GRAM, extends existing single-page document models to efficiently handle multi-page documents without necessitating computationally-intensive pretraining. Leveraging the single-page encoder for local pagelevel comprehension, we introduce document learnable tokens and designated layers, enabling seamless information exchange across pages. Additionally, our proposed bias adaptation method enforces effective utilization of our newly introduced document tokens. The incorporation of a C-Former model reduces sequence length, balancing quality with latency in the decoding step. Extensive experiments demonstrate GRAM’s state-of-the-art performance across multi-page DocVQA benchmarks. 

## **References** 

- [1] Aviad Aberdam, Ron Litman, Shahar Tsiper, Oron Anschel, Ron Slossberg, Shai Mazor, R Manmatha, and Pietro Perona. Sequence-to-sequence contrastive learning for text recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15302–15312, 2021. 2 

- [2] Aviad Aberdam, Roy Ganz, Shai Mazor, and Ron Litman. Multimodal semi-supervised learning for text recognition. _arXiv preprint arXiv:2205.03873_ , 2022. 

- [3] Aviad Aberdam, David Bensa¨ıd, Alona Golts, Roy Ganz, Oren Nuriel, Royee Tichauer, Shai Mazor, and Ron Litman. Clipter: Looking at the bigger picture in scene text recognition. _arXiv preprint arXiv:2301.07464_ , 2023. 2 

- [4] Joshua Ainslie, Tao Lei, Michiel de Jong, Santiago Onta˜n´on, Siddhartha Brahma, Yury Zemlyanskiy, David Uthus, Mandy Guo, James Lee-Thorp, Yi Tay, et al. Colt5: Faster long-range transformers with conditional computation. _arXiv preprint arXiv:2303.09752_ , 2023. 1, 2, 4 

- [5] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha. Docformer: End-to-end transformer for document understanding. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 993–1003, 2021. 1, 2, 3 

- [6] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R Manmatha. Docformerv2: Local features for document understanding. _arXiv preprint arXiv:2306.01733_ , 2023. 1, 2, 3, 5, 6, 7, 8 

- [7] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_ , 2020. 1, 2, 4, 6, 7 

- [8] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 4291–4301, 2019. 5 

- [9] Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, and R Manmatha. Latr: Layout-aware transformer for scene-text vqa. pages 16548–16558, 2022. 2 

- [10] Aydar Bulatov, Yuri Kuratov, and Mikhail S Burtsev. Scaling transformer to 1m tokens and beyond with rmt. _arXiv preprint arXiv:2304.11062_ , 2023. 1 

- [11] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. _arXiv preprint arXiv:2306.15595_ , 2023. 2 

- [12] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. _arXiv preprint arXiv:1901.02860_ , 2019. 1, 2 

- [13] Robert M French. Catastrophic forgetting in connectionist networks. _Trends in cognitive sciences_ , 3(4):128–135, 1999. 4 

- [14] Roy Ganz, Oren Nuriel, Aviad Aberdam, Yair Kittenplon, Shai Mazor, and Ron Litman. Towards models that can see and read. _arXiv preprint arXiv:2301.07389_ , 2023. 3, 4 

- [15] JiangLong He, Mamatha N, Shiv Vignesh, and Deepak Kumar. Hivt5beam. Hi-VT5 model pretrained with private custom document collection using span masking objective. Pretrained model is then trained with DUDE dataset and MultiPage DocVQA dataset, 2023. 6 

- [16] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4083–4091, 2022. 1, 2, 6, 7 

- [17] HuggingFace. Huggingface trainer. https://huggingface.co/docs/transformers/training. 6 

- [18] Jordy Landeghem, Rub´en Tito, Łukasz Borchmann, Michał Pietruszka, Paweł J´oziak, Rafał Powalski, Dawid Jurkiewicz, Micka¨el Coustaty, Bertrand Ackaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). _arXiv preprint arXiv:2305.08455_ , 2023. 1, 3, 6, 7 

- [19] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 2 

- [20] Ron Litman, Oron Anschel, Shahar Tsiper, Roee Litman, Shai Mazor, and R Manmatha. Scatter: selective context attentional scene text recognizer. In _proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 11962–11972, 2020. 2 

- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 2 

- [22] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 2 

- [23] Oren Nuriel, Sharon Fogel, and Ron Litman. Textadain: Paying attention to shortcut learning in text recognizers. In _European Conference on Computer Vision_ , pages 427–445. Springer, 2022. 2 

- [24] R OpenAI. Gpt-4 technical report. _arXiv_ , pages 2303– 08774, 2023. 2 

- [25] Qiming Peng, Yinxu Pan, Wenjin Wang, Bin Luo, Zhenyu Zhang, Zhengjie Huang, Teng Hu, Weichong Yin, Yongfeng Chen, Yin Zhang, et al. Ernie-layout: Layout knowledge enhanced pre-training for visually-rich document understanding. _arXiv preprint arXiv:2210.06155_ , 2022. 1, 2 

- [26] Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Pałka. Going full-tilt boogie on document understanding with text-image-layout transformer. In _Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16_ , pages 732–747. Springer, 2021. 2, 3 

- [27] Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. _arXiv preprint arXiv:2108.12409_ , 2021. 1, 2, 5 

- [28] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and 

- Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ , 21(1):5485–5551, 2020. 2, 3, 5, 6 

Lit: Zero-shot transfer with locked-image text tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18123–18133, 2022. 4 

- [29] RenZhou, QiaolingDeng, XinfengChang, LuyanWang, XiaochenHu, HuiLi, and YaqiangWu. Docblipvqa. We integrated the prediction outputs from the UDOP model and Blip2 to enhance our results,and we optimized the image encoder and included page number features to address the challenge of multi-page documents. GPT to generate python-like modular programs., 2023. 6, 7 

- [30] RenZhou, QiaolingDeng, XinfengChang, LuyanWang, XiaochenHu, HuiLi, and YaqiangWu. Docgptvqa. We integrated the prediction outputs from the UDOP model and Blip2 to enhance our results,and we optimized the image encoder and included page number features to address the challenge of multi-page documents. GPT to generate python-like modular programs., 2023. 6, 7 

- [31] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19254–19264, 2023. 2, 3 

- [32] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Document collection visual question answering. In _Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16_ , pages 778–792. Springer, 2021. 6 

- [33] Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multi-page docvqa. _arXiv preprint arXiv:2212.05935_ , 2022. 1, 2, 3, 5, 6, 7, 8 

- [34] Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. _Advances in Neural Information Processing Systems_ , 34:200–212, 2021. 4 

- [35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 1 

- [36] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , pages 1192–1200, 2020. 1, 2, 3 

- [37] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. _arXiv preprint arXiv:2012.14740_ , 2020. 1, 2 

- [38] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. _Advances in neural information processing systems_ , 33:17283–17297, 2020. 1, 2, 4, 6, 7 

- [39] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. 

## **GRAM: Global Reasoning for Multi-Page VQA** 

## Supplementary Material 

|**Group**|**Parameter Name**<br>**Parameter Value**|
|---|---|
|fne-tune|batch size<br>8<br>training steps<br>200K<br>warmup steps<br>1000<br>fp16<br>True<br>training number of pages<br>4<br>evaluation number of pages<br>unlimited<br>number of image tokens<br>128|
|DocFormer_concat_[6]|encoder learning rate<br>3e-5<br>decoder learning rate<br>3e-5<br>training text tokens per page<br>600<br>inference text tokensperpage<br>400|
|HiVT5* [33]|encoder learning rate<br>3e-5<br>decoder learning rate<br>3e-5<br>training text tokens per page<br>800<br>inference text tokens per page<br>8000<br>number of compression tokensperpage<br>10|
|GRAM|encoder learning rate<br>3e-5<br>decoder learning rate<br>3e-5<br>global encoder learning rate<br>1e-4<br>training text tokens per page<br>800<br>inference text tokens per page<br>8000<br>number of global tokens<br>32<br>bias adaptation constant ‘_c_’<br>20|
|GRAM_C−F ormer_|encoder learning rate<br>3e-5<br>decoder learning rate<br>3e-5<br>global encoder learning rate<br>1e-4<br>C-Former learning rate<br>1e-4<br>training text tokens per page<br>800<br>inference text tokens per page<br>8000<br>number of global tokens<br>32<br>bias adaptation constant ‘_c_’<br>20<br>compression length<br>256|



Table 5. **Hyper-Parameters** . 

## **A. Parameters** 

We present in Tab. 5 all of the relevant hyperparameters. 

parable memory footprint to the GRAM model. Nevertheless, there is potential for improvement, as HiVT5* exhibits lower memory consumption. Despite this, we achieve inference times similar to HiVT5* [33], accompanied by a noteworthy enhancement in ANLS. 

**==> picture [214 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
50<br>DocFormerv2concat<br>Hi-VT5*<br>40 GRAM C Former<br>GRAM<br>30<br>20<br>10<br>0<br>0 50 100 150 200 250 300<br>Number of Pages<br>Latency comparison . We compare the dependency be-<br>tween overall latency and the number of pages in input document.<br>35 DocFormerv2concat<br>Hi-VT5*<br>30<br>GRAMC Former<br>25 GRAM<br>20<br>15<br>10<br>5<br>0<br>0 50 100 150 200 250 300<br>Number of Pages<br>Latency [sec]<br>Memory [GB]<br>**----- End of picture text -----**<br>


Figure 6. **Latency comparison** . We compare the dependency between overall latency and the number of pages in input document. 

Figure 7. **Memory consumption comparison** . We compare the dependency between overall memory consumption and the number of pages in input document. 

## **B. Inference Resources Consumption** 

We compare three key properties of MP-DocVQA baselines and our method: inference time, memory consumption, and maximal document length. The latency and memory consumption are illustrated in Fig. 6 and Fig. 7, respectively, both as functions of the number of pages in the document. We compare the following baselines: DocFormerv2 _concat_ [6], Hi-VT5 * [33], and our GRAM and GRAM _C−F ormer_ , utilizing the same computational resources employed in all experiments— 8 _× A_ 100 GPUs with 40 _GB_ of memory. 

The memory consumption of DocFormerv2 _concat_ [6] reaches its maximum capacity for documents with only 20 pages, while our method efficiently processes documents, spanning hundreds of pages. Moreover, the presented figures demonstrate that GRAM _C−F ormer_ maintains a com- 

## **C. Qualitative Results** 

Finally, we present a few qualitative results on the DUDE dataset in Fig. 8, showcasing the advantages of our approach over Hi-VT5* [33]. In the first three examples, we demonstrate cases where GRAM is correct and HiVT5* is wrong. The last two examples present cases where both our method and HiVT5* are incorrect. 

## **D. Comparison with DocFormerV2concat** 

We provide additional qualitative examples with DocFormerV2concat. Examples demonstrate the effectiveness of GRAM in tackling questions that involve multiple pages in the document. 

|**Method**|**ANLS by Number of Pages**<br>**DUDE validation dataset**<br>All<br>1<br>2-4<br>5-10<br>11-end|
|---|---|
|GRAM|**47**_._**88**<br>**49**_._**29**<br>**49**_._**90**<br>**45**_._**90**<br>**43**_._**94**|
|DocFormerv2concat<br>DocFormerV2Longformer<br>DocFormerV2AliBi|44_._32<br>46_._08<br>47_._05<br>42_._81<br>38_._17<br>45_._88<br>47_._01<br>47_._75<br>43_._22<br>43_._13<br>34_._73<br>36_._55<br>37_._00<br>30_._99<br>31_._25|



Table 6. **Comparison to NLP methods** . Results on DUDE validation comparing GRAM with LongFormer [7] and AliBi [27]. 

## **E. Comparison with NLP-based Approaches** 

We present additional experiments, comparing GRAM with two NLP-based approaches: the sparse attention-based LongFormer [7], and the bias-based AliBi [27]. Both approaches are implemented on top of DocFormerv2 for fair comparison. Results in Tab. 6 shows an advantage in our local-global approach of utilizing existing powerful models for single-page and extending them to support the multipage scenario. 

## **How many chapters are in the books?** correct answer: “” HiVT5: “4” GRAM: “” 

**==> picture [423 x 61] intentionally omitted <==**

**----- Start of picture text -----**<br>
CRIMINAL LAW Date:   _____________________DPS – Law Enforcement Academy PRESENTED BY:ONLINE: PROPERTY CRIMES 1 Santa Fe, New Mexico GOALS Upon completion of this course, students will be able to: SOURCESOBJECTIVESESTIMATED TIMEPREPARED BY Know the difference between larceny and possession of stolen property.State the different ways to charge receiving stolen property.Know the alternative ways to commit shoplifting.State the legal presumption when one conceals an item inside a store.Know the difference between larceny and shoplifting.Articulate the difference between criminal damage to property and graffiti.Understand how to establish value of stolen and recovered property.Students will gain an in-depth understanding of certain property crimes includingLarceny, Receiving Stolen Property, Shoplifting, Criminal Damage to Property, Graffiti.Students will develop an understanding about charging possession of stolen property. New Mexico Criminal and Traffic Manual.New Mexico Statutes AnnotatedState and federal case law.Legal InstructorDepartment of Public SafetyLaw Enforcement AcademySanta Fe, New MexicoIncluded in a ten hour block on Criminal Law.2 INTRODUCTIONLARCENYRECEIVING STOLEN PROPERTY come in contact with and have questions about. Larcenyover $500 is a fourth degree felony (18 months), over $2500 a third degree felony (three years), and over $20,000 a second degree felony (nine years).Larceny of livestockfelony, regardless of value.Larceny of a firearmfelony when its value is less than $2500.We are going to discuss certain property crimes that police officers are most likely to In 2006 the amount to become a felony for most property crimes increased from $250 to $500.The penalty for larceny will generally depend upon the value of the item stolen. Larceny Whoever commits larceny when the property . . . is livestock is guilty of a third degree Whoever commits larceny when the property . . .is a firearm is guilty of a fourth degree The elements of larceny are:The elements of Receiving Stolen Property are:NMSA 1978, Section 30-16-1stealing anything of valuewhich belongs to anotherdefendant intended to permanently deprive the owner of the property at the time he or she took it.intentionallyPenalties for LarcenyNMSA 30-16-113 (1974).SITUATION #3Larceny and “Disposing” of stolen property?Answer:“disposing” of an item is different from larceny of an item. Since the items were disposed at two locations, there would be two counts of Receiving Stolen Property (dispose). charge Larceny and “Retaining” of stolen property?property but not both. Officers charge larceny if it can be proven the person stole the item. However, if a person is in possession of stolen property and it can’t be proven how they got the item, the correct charge would be Receiving Stolen Property (retain).SITUATION #1STOLEN PROPERTY and it can be charged in one of three ways:SITUATION #2Answer:Police execute a search warrant at offender’s residence. Numerous stolen items were An offender steals and is apprehended by police in possession of stolen property. Can we An offender steals items and disposes of them at two different locations. Can we charge The offender can be charged with larceny Since larceny is a continuing offense, we can charge larceny or “retaining” of stolen A point of confusion:Each one of these three ways or charges has distinctive characteristics.In New Mexico the statute for possession of stolen property is called RECEIVING receiving, retaining or disposing of stolen propertyknowing that it is stolen or believing it to be stolenunless the property is received, retained or disposed of with intent to restore it to the owner.the property may be “received,” orthe property may be “retained,” orthe property may be “disposed.”(yes)4 and  “disposing” of stolen property. The act of (no) State v. SmithState v. Mitchell(1983). recovered belonging to five victims. Each item had a misdemeanor value but together they had a felony value. Do we have one count of “retaining” or five?Answer:be charged with one count of “retaining” stolen property. fourth degree felony when its value is less than $2,500. NMSA 1978, Section30-16-11 (I).FACTS:and also some stolen firearms. The property was taken from the same victim at the same place and time, and it was acquired and possessed by Defendant at the same time. Do we have one count of receiving stolen property or two?the merchant of all or some part of the value of the merchandise.shoplifting. Suppose someone conceals an item on his or her person. A loss prevention officer (LPO) stops the person. The person says they inadvertently, accidentally put the item inside theircoat pocket.  What should the loss prevention officer do?Receiving stolen property of a firearmAnswer:Stolen Property (Firearm). This is not double jeopardy since the legislature clearly intended that possession of a stolen firearm would be an additional or separate crime.  SHOPLIFTING Although the offender “retains” stolen items from different victims, the offender will only Whoever commits receiving stolen property when the property is a firearm is guilty of aDefendant received some generic stolen property (DVD’s, camera equipment, gym bags) In each of these four ways to commit shoplifting, the offender has the intent to deprive NMSA 1978, Section 30-16-22 mention a presumption that might be helpful:Court of Appeals affirmed convictions for Receiving Stolen Property and Receiving You may have noticed the word “willfully” in describing the four ways to do a There are four different ways to commit shoplifting:NMSA 1978, Section 30-16-20willfully taking possession of any merchandise.willfully concealing any merchandise.willfully altering any label, price tag or marking upon any merchandise.willfully transferring merchandise from one container to another.(two) 5 State v. WatkinsSanchez v. State(2008).(1982). NMSA 1978, Section 30-16-22.  Presumptions created.Note:Note:parent using a child to conceal items.shoplifting occurs outside your presence. What options do you have in this situation?NMSA 1978, Section 30-16-23FACTS:$449, and with tax the price is over $500.Answer:the terms “market value” and “retail value” are identical. Answer:retail or actual market price of an item includes the tax.  Answer:or her property. A person goes to a store and steals an item. The wholesale price is $399, the retail price is A.B. Any person who willfully conceals merchandise on his or her person or on the person of another or among his or her belongings or the belongings of another . . . on or outside the premises of the store shall be . . . presumed to have concealed the merchandise with the intention of converting it without paying for it.An example of a person concealing merchandise on the person of another would be a “store” means a place where merchandise is sold or offered to the public for sale at retail;“merchandise” means chattels (items) of any type or description regardless of the value offered for sale in or about a store. NMSA 1978, Section 30-16-19.an officer to make an arrest (with probable cause) although it occurred outside the officer’s presence. The officer still has a choice, however, as to whether to arrest a person or issue a citation. Market value is used when value is an issue. Supreme Court of New Mexico has held that Does market value include taxes?Tax is not to be considered when determining value of an item, unless the advertised How do we determine value if a private citizen is a victim of property theft?It is a general rule that an owner of property is competent to testify as to the value of his Normally, an officer must witness a misdemeanor to make an arrest. This section allows To arrest for most misdemeanors, the offense must occur in your presence. A  State v. RomeroLARCENY V. SHOPLIFTING(1975). 6 Tunnel v. StateTunnell v. State (1983).(1983). CRIMINAL DAMAGE TO PROPERTY charge if a person stole cash from a cash register?  Answer:Another difference between larceny and shopliftinganother with graffiti or other inscribed material inscribed with ink, paint, spray paint, crayon, charcoal or the use of any subject without the consent or reasonable ground to believe there is consent of the owner of the property. NMSA 1978, Section 30-15.1.1.mandatory restitution and community service.encounter. Knowing the elements of these crimes will assist our ability to more accurately charge them. the dollar amount to become a felony is $1,000. UNAUTHORIZED GRAFFITI ON PERSONAL OR REAL PROPERTY someone’s front lawn or a tire on a front porch. For a misdemeanor larceny, an officer cannot make an arrest unless the offense occurred in his or her presence.Shoplifting involves taking merchandise from a store. It allows an officer to make a misdemeanor arrest even though the offense did not occur in the officer’s presence. NMSA 1978, Section 30-16-23.Larceny generally refers to a theft occurring other than a store, i.e., a bicycle on It would be larceny since cash is not merchandise.It’s shoplifting if a person takes merchandise (an item offered for sale) in a store. WhatGraffiti consists of intentionally and maliciously defacing any real or person property of For this charge the dollar amount to become a felony is $1,000. This statute provides for For most crimes the dollar amount for a felony is $500. For criminal damage to property The crimes discussed are some of the property crimes officers are most likely to The elements are:intentionallydamaging any real or personal property of anotherwithout the consent of another. CONCLUSION 7 NMSA 1978, Section 30-15-1<br>**----- End of picture text -----**<br>


**==> picture [335 x 35] intentionally omitted <==**

**----- Start of picture text -----**<br>
How many types of complaints were listed in the document?     correct answer: “9”<br>HiVT5: “10”<br>GRAM: “9”<br>**----- End of picture text -----**<br>


**==> picture [251 x 37] intentionally omitted <==**

**----- Start of picture text -----**<br>
how many pages are there in this text?     correct answer: “9”<br>HiVT5: “1”<br>GRAM: “9”<br>**----- End of picture text -----**<br>


**==> picture [234 x 36] intentionally omitted <==**

**----- Start of picture text -----**<br>
Which pages show graphs?     correct answer: “['3', '6', '5']”<br>HiVT5: “1”<br>GRAM: “2”2”” J. on Network Security , Vol. 02, No.<br>**----- End of picture text -----**<br>


**==> picture [416 x 78] intentionally omitted <==**

**----- Start of picture text -----**<br>
GRAM: “2”2”” J. on Network Security , Vol. 02, No.  > |<br>© 2011 ACEEEDOI: 01.IJNS.02.0.254rW (t)W*(t)Ar (t)rr Abstract ACRONYMSNHPPSRGMMVFMLETEFLOCMSENOTATIONSm (t)ë (t)n (t)mm software quality. Before software delivered in to market it isthoroughly checked and errors are removed. Every softwareindustry wants to develop software that should be error free.Software reliability growth models are helping the softwareindustries to develop software which is error free and reliable.In this paper an analysis is done based on incorporating thelogistic-exponential testing-effort in to NHPP Softwarereliability growth model and also observed its release policy.Experiments are performed on the real datasets. Parametersare calculated and observed that our model is best fitted forthe datasets. Keywords- Effort, Non-homogeneous Poisson Process (NHPP), SoftwareCost. 12            : Constant fault detection rate in the Delayed S-drExponential Testing-Effort Function and Analysis of (t) (t)Software Reliability Growth Model with Logistic- -  Software reliability is one of the important factors of : W (t)-W (0): Non Homogeneous Poisson Process: Software Reliability Growth Model: Mean Value Function: Maximum Likelihood Estimation: Failure detection rate function: Cumulative number of faults isolated up to t.: Expected number of initial faults: Constant fault isolated rate in the Delayed S-  shaped model with logistic-Exponential TEF  shaped    model with logistic-Exponential TEF: Constant fault detection rate function.: Testing Effort Function  in time (0,t]: Cumulative number of faults detected upto t: Failure intensity for m(t): Fault content function: Cumulative testing effort consumption at: Lines of Code: Mean Square fitting Error: Expected mean number of faults detected Software Reliability, Software Testing, Testing 1Dept. of Computer Science, Sri Mittapalli Institute of Technology for women, Guntur, A.P, India2Dept. of Computer Science, Sri Mittapalli College of Engineering, Guntur, A.P, IndiaSoftware Release PolicyShaik.Mohammad RafiE-mail:shaheda.akthar@yahoo.comE-mail:mdrafi.527@gmail.comtimet.ACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201138 [ 1]  The following are some of them  A.   Exponential Testing effort function is [20]  B.  Rayleigh Testing effort curve: commutation devices and electronics equipments every placewe find software. The goal of every software industries isdevelop software which is error and fault free. Every industryis adopting a new testing technique to capture the errorsduring the testing phase. But even though some of the faultswere undetected. These faults create the problems in future.Reliability is defined as the working condition of the softwareover certain time period of time in a given environmentalconditions. Large numbers of papers are presented in thiscontext. Testing effort is defined as effort needed to detectand correct the errors during the testing. Testing-effort canbe calculated as person/ month, CPU hours and number oftest cases and so on. Generally the software testing consumesa testing-effort during the testing phase [20 21].SRGMproposed by several papers incorporated traditional effortcurves like Exponential, Rayleigh, and Weibull. The TEFwhich gives the effort required in testing and CPU time thesoftware for better error tracking. Many papers are publishedbased on TEF in NHPP models [4, 5, 8, 11, 120, 12, 20, 21].All of them describe the tracking phenomenon with testexpenditure.curve and incorporated in the SRGM. The result shows thatthe SRGM with logistic-exponentialliterature. w(t) is defined as the current testing effort andW(t) describes the cumulative testing effort. The followingequation shows the relation between the w(t) and W(t), Shaheda Akthar The cumulative testing effort consumed in the time (0,t]Software becomes crucial in daily life. Computers,This paper we used logistic-exponential testing-effortSeveral software testing-effort functions are defined inII. SOFTWARE TESTING EFFORT FUNCTIONSI.  INTRODUCTION [2]                         (1)       (2) © 2011 ACEEEDOI: 01.IJNS.02.02.254is [12,20]gradually with decelerating rate.  C.   Logistic-exponential testing-effort: of the hazard rate function, can be used in a variety ofproblems for modeling software failure data.period (0,t] can be expressed as [27]  A.   Software reliability growth model with logistic- reliability growth modeling [1, 8, 11, 20, 21, 22]The Rayleigh curve increases to the peak and descends exponential TEF The cumulative testing effort consumed in the time (0,t]It has a great flexibility in accommodating all the formsThe following assumptions are made for software(i)(ii)  The software system is subjected to failure at(iii) The mean time number of faults detected in the time(iv) The proportionality is constant over the time.(v)  Consumption curve of testing effort is modeled by(vi)  Each time a failure occurs, the fault that caused it(vii) — III. SThe logistic-exponential cumulative TEF over timeHomogeneous Poisson process (NHPP)random time caused by faults remaining in thesystem.interval (t, t+Ät) by the current test effort isproportional for the mean number of remainingfaults in the system.a logistic-exponential TEF.is immediately removed and no new faults areintroduced.testing-effort based on followingThe fault removal process follows  the Non-We can describe the mathematical expression of aOFTWARE RELIABILITY GROWTH                        (3) MODELS ACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201139  A. The goodness of fit technique measure of the difference between actual and predictedvalues. The MSE defined as  B.   Yamada Delayed S-shaped model with logistic-exponential testing-effort function Yamada [24]  and it is different from NHPP by consideringthat software testing is not only for error detection but errorisolation. And the cumulative errors detected follow the S-shaped curve. This behavior is indeed initial phase testersare familiar with type of errors and residual faults becomemore difficult to uncover [1, 6, 15, 16]. From the above stepsdescribed section 3.1, we will get a relationship betweenm(t) and w(t). For extended Yamada S-shaped softwarereliability model.The extended S-shaped model [24] ismodeled byA smaller MSE indicate a smaller fitting error and betterperformance.Here we used MSE [5, 11, 17, 23 ]which gives realThe delayed ‘S’ shaped model originally proposed byIV. EVALUATION CRITERIA © 2011 ACEEEDOI: 01.IJNS.02.02.254 B. Coefficient of multiple determinations (R2) mean accounted for the fitted model and tells us how well acurve fits the data. It is frequently employed to comparemodel and access which model provies the best fit to thedata. The best model is that which proves higher Rcloser to 1. C. The predictive Validity Criterion from present & past failure behavior is called predictivevalidity. This approach, which was proposed by [26], can berepresented by computing RE for a data set A.   DS1: 1984 [15].the system is PL/1 data base application software,consisting of approximately 1,317,000lines of code .Duringnineteen weeks of experiments, 47.65 CPU hours wereconsumed and about 328 software errors are removed.mating the model parameter from actual failure data. Herewe used the LSE (non-linear least square estimation) andMLE to estimate the parameters. Calculations are given inappendix AWhich measures the percentage of total variation aboutThe capability of the model to predict failure behavior.The first set of actual data is from the study by OhbaV. MODEL PERFORMANCE ANALYSISFitting the model to the actual data means by esti- [2] . that isACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201140 LSE. The unknown parameters of Logistic-exponential TEFare á=72(CPU hours), ë=0.04847, and k=1.387.Correspondingly the estimated parameters of Rayleigh TEFN=49.32 and b=0.00684/week. Fig.1 plots the comparisonbetween observed failure data and the data estimated byLogistic-exponential TEF and Rayleigh TEF. The PE, Bias,Variation, MRE and RMS-PE for Logistic-exponential andRayleigh are listed in Table I. From the TABLE I we can seethat Logistic-exponential has lower PE, Bias, Variation, MREand RMS-PE than Rayleigh TEF. We can say that ourproposed model fits better than the other one. In the TABLEII we have listed estimated values of SRGM with differenttesting-efforts. We have also given the values of SSE, Rand MSE. We observed that our proposed model has smallestMSE and SSE value when compared with other models. The95% confidence limits for the all models are given in theTable III.  B.   DS2: a subset of products for four separate software releases atTandem Computer Company. Wood Reported that thespecific products & releases are not identified and the testdata has been suitably transformed in order to avoidConfidentiality issue. Here we use release 1 for illustrations.Over the course of 20 weeks, 10000 CPU that SRGM withlogistic-exponential TEF have less MSE than other models.Fig 1. Observed/estimated logistic-exponential and Rayleigh TEF forAll parameters of other distribution are estimated throughThe dataset used here presented by wood [2] fromDS1. [2] © 2011 ACEEEDOI: 01.IJNS.02.02.254 [=see=[=ee— iee ACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201141 [= [ayanf Nf\ K ----erese>. © 2011 ACEEEDOI: 01.IJNS.02.02.254aim is to determine the optimal software release time thatminimizes the total software cost to achieve the desiredsoftware reliability. Therefore, the optimal software releasepolicy for the proposed software reliability can be formulatedtesting time needed to reach the reliability objective RCexpenditure and T A.   Software Release-Time Based on Reliability Criteria reliability of a software system. Here in this first we discussthe optimal time based on reliability criterion. If we knowsoftware has reached its maximum reliability for a particulartime. By that we can decide right time for the software to bedelivered out. Goel and Okumoto [1] first dealed with thesoftware release problem considering the software cost-benefit. The conditional reliability function after the lastfailure occurs at time t is obtained by B.   Optimal release time based on cost-reliabilitycriterion cost-reliability criterion. Using the total software costevaluated by cost criterion, the cost of testing-effortexpenditures during software testing/development phase andthe cost of fixing errors before and after release are: [9, 13,25]> C2 is the cost of correcting an error during the operation, CFrom reliability criteria, we can obtain the requiredGenerally software release problem associated with theThis section deals with the release policy based on theWhere CVI.   OPTIMAL SOFTWARE RELEASE POLICY1, C3 is the cost of testing per unit testing effort1   the cost of correcting an error during testing,LC is the software life-cycle length. 0. Our i 2ACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201142 total software cost under desired software reliability and thenthe optimal software release time is obtained. That is canminimize the C(T) subjected to R(t+Ät/t)e” Rsatisfying Eq.(31)  Tt)=Rsetting it to zero, we obtain=max{T>0, Ät>0, 0 < Rwe can easily get the required testing time needed to reachthe reliability objective R<1 [9,25]Tas Minimize C(T) subjected to R(t+Ät/t)e” R =e [*] =optimal software release time or total testing timeDifferentiate the equation (30) with respect to T and0 0, T1}.Where T0 <1.1 =finite and unique T satisfying R(t+Ät/00 . here our goal is to minimize the=finite and unique solution T r 0 0for C where 0< R2 > C : 1, C30 © 2011 ACEEEDOI: 01.IJNS.02.02.254By combining the above analysis and combining the costand reliability requirements we have the following theorem.estimated time Testimated time TTTLogistic-exponential TEF á=72(CPU hours),  ë=0.04847 /<1.  Let T*be the optimal software release timeweek, k=1.387, a=578.8 and r=0.01903 when Ät=0.1 R=0.85 and we let Cis T*=39.5 weeks. Fig 10 shows the change in software costduring the time span. Now total cost of the software at optimaltime 8354.Logistic-exponential TEF á=12600(CPU hours), ë=0.06352/week, k=1.391, a=135.6 and r=0.0001432 when Ät=0.1 R=0.85 and we let Cis  T*=18.1 weeks. Fig 11 shows the change in software costduring the time span. Now total cost of the software at optimaltime 20,100.00 =39.5 weeks. Now optimal Release Time max (37.1, 39.5)=8.05 weeks. Now optimal Release Time max (8.05, 18.1)Theorem 1:From the dataset one estimated values of SRGM withFrom the dataset two estimated values of SRGM with11=18.1 weeks and release time from Eq 31=37.1 weeks and release time from eq 30Assume C2<C1<0, C3<0, Ät>0, and 0<R11=2, C2 =50, C=1, C2 =200, C3 =150 and T3 =2 and T os LCLC =100 the =100 the a 000ACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201143 Logistic-exponential testing effort function that is completelydifferent from the logistic type Curve. We Observed that mostof software failure is time dependent. By incorporatingtesting-effort into SRGM we can make realistic assumptionsabout the software failure. The experimental results indicatethat our proposed model fits fairly well.[1][2][3][4][5]  C.-Y. Huang, S.-Y. Kuo, J.Y. Chen, Analysis of a software[6] Goel, A.L., “Software reliability models: Assumptions,[7]  Huang, C.Y. and Kuo, S.Y. (2002), “Analysis of incorporating[8]  Huang, C.Y., Kuo, S.Y. and Chen, I.Y. (1997), “Analysis of ‘ a In this paper, we proposed a SRGM incorporating theBokhari, M.U. and Ahmad, N. (2006), “Analysis of a softwareA.L. Goel and K. Okumoto, A time dependent error detectionrate model for a large scale software system,  Japan Computer Conference,  (1978).(1996) 69–77.growth modeling for exponentiated Weibull functions withactual software failures data”, in Proceedings of 3rdInternational Conference on Innovative Applications ofInformation Technology for Developing World (AACC’2005),Nepal.reliability growth models: the case of log-logistic test-effortfunction”, in Proceedings of the 17th International Conferenceon Modelling and Simulation (MS’2006), Montreal, Canada,pp. 540-5.reliability growth model with logistic testing effort functionproceeding of Eighth International Symposium on SoftwareReliability Engineering, 1997, pp. 378–388.limitations, and applicability”,  Engineering logistic testing-effort function into software reliabilitymodeling”, IEEE Transactions on Reliability, Vol. 51 No. 3,pp. 261-70.software reliability growth model with logistic testing-effortfunction”, in Proceeding of 8th International Symposium onSoftware Reliability Engineering  (ISSRE’1997),Albuquerque, New Mexico, pp. 378-88.A.Wood, Predicting software reliability, IEEE computers 11 Bokhari, M.U. and Ahmad, N. (2005), “Software reliability SE-11 (1985) 1411-1423.CONCLUSIONREFERENCES IEEE Transactions on Software pp. 3540, San Francisco, CA Proc. 3rd USA- } |. © 2011 ACEEEDOI: 01.IJNS.02.02.254[9] Huang, C.Y., Kuo, S.Y. and Lyu, M.R. (1999), “Optimal[10] Huang, C.Y., Kuo, S.Y. and Lyu, M.R. (2000), “Effort-index[11]Huang, Lyu and Kuo “An Assesment of testing effort dependent[12]Huang and S. Y. Kuo, “Analysis and assessment of[13] K. Pillai and V. S. Sukumaran Nair, “A model for software[14] K. Srinivasan and D. Fisher, “Machine learning approaches[15] M. Ohba, Software reliability analysis models, IBM J. Res.[16] M.R. Lyu, Handbook of Software Reliability Engineering,[17] Pham, H. (2000), Software Reliability, Springer-Verlag, New[18]  Quadri, S.M.K., Ahmad, N., Peer, M.A. and Kumar, M.software release policy based on cost, reliability and testingefficiency”, in Proceedings of the 23rd IEEE AnnualInternationalbased software reliability growth models and performanceassessment”, in Proceedings of the 24th IEEE AnnualInternational Computer Software and Applications Conference(COMPSAC’2000), pp. 454-9.software reliability Growth model”. IEEE transactions onReliability Vol 56, No: 2, June 2007incorporating logistic testing effort function into softwarereliability modeling,” pp. 261–270, Sept. 2002.development effort and cost estimation,”  Engineering to estimating software development effort,”  Software Engineering Dev. 28 (1984) 428–443.Mcgraw Hill, 1996.York, NY.(2006), “Nonhomogeneous Poisson process softwarereliability growth model with generalized exponential testing, vol. 23, no. 8, August 1997. |i , vol. 21, no. 2, pp. 126–136, 1995. IEEE Trans. ReliabilityIEEE Trans. Software , vol. 51, no. 3, IEEE Trans. ACEEE  Int. J. on Network Security , Vol. 02, No. 02, Apr 201144 [19] Rameshwar D. Gupta and Debasis Kundu “generalized[20] S. Yamada, H. Ohtera and R. Narihisa, “Software Reliability[21]   S. Yamada, H. Ohtera, Software reliability growth model for[22] S.Yamada, S.Osaki, “Software reliability growth modeling:[23] Xie, M. (1991), Software Reliability Modeling, World[24] Yamada, S., Ohba, M., Osaki, S., 1983. S-shaped reliability[25] Yamada, S. and Osaki, S. (1985b), “Cost-reliability optimal[26]J.D. Musa, A. Iannino, and K. Okumoto, Software[27]  Y. Lan, and L. Leemis, (Aug. 2007) “The Logistic-Exponentialeffort function”, RAU Journal of Research, Vol. 16 Nos 1-2,pp. 159-63.exponential distribution: different method of estimations” j.statist. comput. simul., 2000, vol. 00, pp. 1 – 22 14 november2000.Growth Models with Testing-Effort,” IEEE Trans. Reliability,Vol. R-35, pp. 19-23 (1986).testing effort control, Eur. J. Oper. Res. 46 (1990) 343–349.models and applications”, vol.l I, no.12, p.1431-1437, December 1985.Scientific Publication, Singapore.growth modeling for software error detection. IEEE Trans.Reliab. 12, 475–484.release policies for software systems”, IEEE Transactions onReliability, Vol. R-34 No. 5, pp. 422-4.Reliability:Measurement, Prediction, Application, McGraw-Hill NewYork, 1987.Survival Distribution,” volume 55, number 3, pp. 252-264. Naval Research Logistics (NRL)IEEE Trans. Software Engineering,<br>**----- End of picture text -----**<br>


## **Can real property be improved within a mortgage loan?** correct answer: “The within mortage does HiVT5: “”                                                                                                                                          not cover real property GRAM: “”                                                                                                                                           improved” 

Figure 8. Qualitative comparison between our approach and Hi-VT5 [33] indicates that the integration of our global-local encoder enhances reasoning capabilities, especially when inquiries require multi-page context. 

**==> picture [20 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
 concat<br>**----- End of picture text -----**<br>


**==> picture [20 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
 concat<br>**----- End of picture text -----**<br>


**==> picture [20 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
 concat<br>**----- End of picture text -----**<br>


**==> picture [48 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
DocFormer  concat<br>**----- End of picture text -----**<br>


Figure 9. Comparisons between DocFormerV2concat and GRAM. 

