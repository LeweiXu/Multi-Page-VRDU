# **DocLayLLM: An Efficient Multi-modal Extension of Large Language Models for Text-rich Document Understanding** 

Wenhui Liao[1] _[†]_ , Jiapeng Wang[1,2] _[†]_ , Hongliang Li[1] , Chengyu Wang[2] _[∗]_ , Jun Huang[2] , Lianwen Jin[1,3] _[∗]_ 

1South China University of Technology, Guangzhou, China 

2Alibaba Cloud Computing, Hangzhou, China 

3SCUT-Zhuhai Institute of Modern Industrial Innovation, Zhuhai, China 

_{_ eelwh, eejpwang, eehongliangli _}_ @mail.scut.edu.cn, eelwjin@scut.edu.cn 

_{_ chengyu.wcy, huangjun.hj _}_ @alibaba-inc.com 

## **Abstract** 

_Text-rich document understanding (TDU) requires comprehensive analysis of documents containing substantial textual content and complex layouts. While Multimodal Large Language Models (MLLMs) have achieved fast progress in this domain, existing approaches either demand significant computational resources or struggle with effective multi-modal integration. In this paper, we introduce_ _**DocLayLLM** , an efficient multi-modal extension of LLMs specifically designed for TDU. By lightly integrating visual patch tokens and 2D positional tokens into LLMs’ input and encoding the document content using the LLMs themselves, we fully take advantage of the document comprehension capability of LLMs and enhance their perception of OCR information. We have also deeply considered the role of chain-of-thought (CoT) and innovatively proposed the techniques of CoT Pre-training and CoT Annealing. Our DocLayLLM can achieve remarkable performances with lightweight training settings, showcasing its efficiency and effectiveness. Experimental results demonstrate that our DocLayLLM outperforms existing OCR-dependent methods and OCR-free competitors. Code and model are available at https://github.com/whlscut/DocLayLLM._ 

## **1. Introduction** 

Text-rich document understanding (TDU) focuses on analyzing documents with extensive text and complex layouts, requiring a combined understanding of text, layout, and images. Early research [1, 2, 12, 15, 20, 28, 33– 37, 44, 52, 55, 65, 68, 69, 72, 73, 77] often used BERTlike [8] Transformer-based document encoders (DE), designing various pre-training tasks for large-scale unlabeled 

> †Equal contribution. 

> *Co-corresponding authors. 

document corpora and fine-tuning their models on specific downstream tasks like key information extraction (KIE) and visual question answering (VQA) [39, 40]. However, these pre-training tasks often diverge significantly from downstream tasks, requiring the model to acquire substantial task-specific knowledge during fine-tuning. This may result in limited generalization capability across different tasks and datasets. 

With the rapid development of large language models (LLMs), their impressive versatility and generalization capabilities have captured widespread interest [74]. This has led many researchers [17, 38, 61, 76, 79] to overcome the challenges of TDU with LLMs. Initially, efforts focused on employing multi-modal large language models (MLLMs) for OCR-free TDU, where document images are fed into MLLMs, directly generating responses to comprehension tasks. These MLLMs, which can process images and text inputs, are among the most recent advancements in this domain. To date, several general MLLMs have been developed to manage diverse scenarios, and researchers often build their TDU models based on these general models. However, these TDU models typically require highresolution image inputs to recognize dense text in documents and a large volume of training data to effectively develop this capability, both of which bring significant computational demands. Furthermore, within the extensive collections of image and instruction data used to train these models, the training data for downstream evaluation tasks was often included, which might potentially impact the assessment of their generalization to previously unseen datasets. 

To overcome the constraints of OCR-free methods, scholars have pursued diverse methodologies for integrating OCR information into LLMs. Initial strategies [14, 70] involved inputting coordinates of OCR bounding boxes as numeric text. However, this textual representation of numbers often resulted in excessively long inputs that could 

1 

decrease inference speeds and potentially degrade model performance. In response, many researchers [11, 24, 45] have adopted an auxiliary document encoder like LayoutLMv3 [20] to process textual, visual, and layout features from documents and integrated into LLMs. Although these methods avoid high-resolution image inputs, the textual content of documents must also be encoded alongside the layout and image features within the DE due to its structural limitations. However, the document comprehension capabilities of these DEs may not be on par with those of LLMs themselves, which might have led to significant finetuning efforts for aligning the feature spaces of DEs and LLMs to improve performances during the whole training phase, demanding substantial computational resources. 

To address the limitations of OCR-dependent TDU methods, we introduce **DocLayLLM** , an efficient multimodal extension of large language models for text-rich document understanding. Our method diverges from existing approaches that utilize an extra DE by integrating the document’s image patch tokens and 2D positional tokens with textual content into a natural language expression, and then encoding them through a unified LLM. This innovative integration not only taps into the LLM’s inherent text comprehension abilities but also allows it to perceive document layouts. Besides, by incorporating patch tokens, the model supports image-related pre-training tasks and further gains a rudimentary understanding of document structures. 

To advance model performance, we have fully integrated the concept of the chain-of-thought (CoT) [3] into each stage of our training process. Firstly, we developed a novel _CoT Pre-training_ strategy. CoT is engineered to preserve thematic coherence and logical progression during model reasoning, further enhancing the inference ability of LLM, which has proven effectiveness in various subsequent studies [14, 45, 71]. Recently, the use of Long CoT reasoning to enhance the inference capability of models has attracted significant attention [7, 49, 63], and DeepSeek-R1 [7] further emphasized the importance of utilizing CoT data for a cold start before the reinforcement learning stage. Our proposed CoT Pre-training offers a method for rapidly generating large-scale CoT data from non-QA data, thereby improving the model’s reasoning ability. By employing CoT in the data generation process and subsequently pre-training the model with the generated QA-formatted data, we have aligned the added visual and layout features with LLM’s inherent textual features while improving the model’s comprehension ability. Furthermore, we introduced a _CoT Annealing_ technique. While previous research [31, 42] has validated the effectiveness of annealing strategies that gradually increase the proportion of high-quality data towards the end of training, the impact of CoT on data quality had not been explored. To address this, we first propose an annealing strategy that considers data quality from a CoT per- 

spective, further improving model performance. 

Thanks to the above design, utilizing only around 3M pre-training data and 300K supervised fine-tuning (SFT) data with the efficient tuning technology of LoRA [18], our DocLayLLM finishes the whole training process in 36 A100 days, therefore resulting an efficient extension of LLM. Even with such low resources, our DocLayLLM surpasses existing OCR-dependent TDU methods. Furthermore, it also excels in specific tasks without fine-tuning on downstream tasks and outperforms state-of-the-art (SOTA) OCRfree methods when trained with training sets of downstream tasks as other methods, further showcasing its effectiveness. Our contributions are: 

- We propose an efficient multi-modal extension of LLMs, which has augmented LLM in understanding text-rich documents and greatly reduced training resources in need. 

- We introduce _CoT Pre-training_ , a novel approach for generating CoT data that helps enhance the comprehension capabilities of models. 

- We revisit data annealing from a novel perspective and introduce the _CoT Annealing_ strategy, which further enhances the model’s efficiency in data utilization and boosts its ultimate performance. 

- Our method outperforms existing OCR-dependent TDU solutions and surpasses current OCR-free methods under comparable supervised fine-tuning conditions. 

## **2. Related Works** 

**OCR-free MLLMs for Text-Rich Documents Understanding.** To date, a number of studies have attempted to directly answer the instruction for TDU tasks from document images, thereby obviating the need for extra OCR processing and resulting in an end-to-end paradigm. Such models are referred to as OCR-free methods. Early examples like Donut [23], UDOP [62], and OmniParser [66] have explored end-to-end document understanding solutions without relying on LLMs or MLLMs. However, with the rapid development of MLLMs, many new studies have begun to focus on enhancing the capabilities of general MLLMs in document understanding. Some of them attempt to boost the document understanding ability of general MLLMs by tuning them with synthesizing TDU data generated by proprietary LLMs, including LLaVAR [79] and TextSquare [61], while mPLUG-DocOwl 1.5 [17] enhances the document understanding capability by integrating existing document comprehension data and converting it into vision-related QA-formatted data. Moreover, researchers have also explored image encoding methods to read the text more clearly. For instance, UReader [76] and Monkey [38] empower MLLMs to handle high-resolution images by splitting them into sub-images, encoding each separately, and then concatenating them. Some general MLLMs, such as LLaVA-Next [29], LLaVA- 

2 

**Answer** : “Response Code Request Form” 

**==> picture [433 x 145] intentionally omitted <==**

**----- Start of picture text -----**<br>
Large Language Model (LLM)<br>TE VE & VP TE TE LE & LP TE LE & LP … TE<br>OO —EcSSS) CO OG [OG] [TG)]<br>and the document text<br>H document patches:Given the  = == ne “i]t contents and locations in the form of "text, [left,  H Text1 Box1 Text2 Box2 … Question<br>top, right, bottom]":<br>Legend Optical Character<br>Recognition (OCR) Result<br>TE Text Embedder<br>Text1: Response Code Request Form<br>Visual Embedder   Box1: [x0, y0, x1, y1]<br>VE & VP &<br>Visual Projector<br>LE & LP Layout Embedder&Layout Projector Text Box22: RJR Brand: [x0, y0, x1, y1] Question : “What is the<br>Fixed Prompt Expression Text Box33: Winston: [x0, y0, x1, y1 …… ] title of the document?”<br>**----- End of picture text -----**<br>


Figure 1. The overall architecture of DocLayLLM. 

OneVision [27], and Idefics3 [26], also adopt this approach to address the challenge of TDU. Although these above methods have achieved notable success in TDU tasks, they typically require high-resolution image inputs to identify text within images, significantly increasing computational demands. Besides, to equip general MLLMs with the ability to recognize the rich text within a document, a substantial amount of training data is required, including training sets of the benchmarks tested, which resulted in a high demand for training resources and might affect the evaluation of their zero-shot inference ability. 

**OCR-dependent MLLMs for Text-Rich Documents Understanding.** To address the challenges of OCR-free MLLMs, some researchers have attempted to integrate OCR information from text-rich documents into LLMs to avoid high-resolution image inputs. ICL-D3IE [14] feeds coordinates of OCR boxes into the LLM using a textual expression of _[left, top, right, bottom]_ , while LATIN-Prompt [70] simulates the document layout by inserting spaces and newline characters between OCR-recognized text segments, thus providing the LLM with a document input containing a rudimentary layout. Besides, LMDX [53] also input the layout in a way similar to ICL-D3IE. However, these methods of inputting OCR information in textual form lead to overly lengthy inputs, thereby slowing inference and impacting model performance. Subsequent studies have explored encoding OCR information using additional encoders. Cream [24] utilizes BLIP-2 [30] to encode document OCR positions, images, text, and additional object detection information, with the output of BLIP-2 fed into the LLM. InstructDr [60] introduces an extra feature extractor to encode document OCR information, compressing it with learnable tokens before inputting it into the LLM. Additionally, Fujitake [11] and Luo et al. [45] adopt Lay- 

outLMv3 [20] to encode the OCR information of documents before inputting them into the LLM. Although these methods avoid the issue of overly lengthy text inputs, they must use the extra DE to encode all the OCR textual, positional information from the document due to the structural restriction of DE. However, the additional DE might not be comparable with LLMs themselves in terms of document comprehension capability. This might be the reason why they usually need extensive training of both the DE and the LLM to align the features encoded by the DE with those of the LLM, which consumes significant computational resources. DocLLM [67] proposes an extension where the hidden states derived from OCR bounding boxes interact with those of OCR text features encoded by LLMs within each attention block via cross-attention [5, 6]. However, they use an extra projection layer at each attention block to transform and inject layout information, which might excessively interfere with the original attention computation of LLM, thereby potentially affecting LLM’s generalization capability. Moreover, it does not adequately consider the role of image features in TDU. Overall, designing an efficient method to provide OCR information to LLMs remains a research question worthy of deep exploration. 

## **3. DocLayLLM** 

DocLayLLM is an LLM-based methodology that encodes OCRed textual, visual, and positional information directly within an LLM, eliminating the need for additional DE. The training regime of DocLayLLM includes both pretraining and supervised fine-tuning phases, which are both instruction-tuning-based processes. To further enhance our model, we propose the integration of CoT into our pretraining tasks, denoted as _CoT Pre-training_ . Additionally, we introduce _CoT Annealing_ , which involves the applica- 

3 

**==> picture [454 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
title<br>ns en I Response_A 13 25 28 I context. We were encouraged to continue our conversations and share our ideas online via #TechWomenaAfriea,<br>author 291 Dacaie, Dacon 34141, [era] [of][Kare,] end 1 nse_B i] Pm 24 i} | ook forwardto the third Summit. As a young socialinnovator, this workshop has helped me in my exploration of<br>Test Se Sa of Rome || Response_c 24 20 4 |_| etenaandtotne<br>text hie CCD avian connlines of rion Invbalble-aion aodela are nomena Ia naricnlar te i] 1 raner fs unm forsuch an edvstve workshop. ™<br>1 i}<br>Question: What’s the layout type of element at [250,63,286,77] ? 1 Question: What’s the element at row 3, column 2? 1 Question: What’s the distance between “Concluding” and “AIR”?<br>withoutCoT Step the top left Step [291,63,329,77],repectively. Step of “QCD.” 31 :: 2 Based:TheThe ofTheyelementonthe document. Layout-Type-Related CoT nearestarethiswithboth ofatinformation[250,63,286,77]theboxesthe layout typetext, weare“between”can[188,63,245,77]inferis “QCD”.titlethe.andlayoutIt is“axion”neartypeand i}i}i}i}i}i}i}i}i}i}1 withoutCoT Step “Age_group_2”,“Age_Group_1” Step vertically[“Age_Group_1”,[323,72,353,97], Step 321 :::TheTheAllwithelementelementsthe Table-Structure-Aware CoT attheir“Age_group_3”].[320,127,351,152],[195,16,348,45].“13”,tableinheader.rowin theheaders“31”,3 isSosamethe 3rdAndthe“24”]are:[320,180,353,205]].columntheelementselement[“atheaderare”, [[195,16,348,45],inof column“Age_Group_1”,roughlyofcolumncolumnaligned22.2areis i}i}i}i}i}i}i}i}i}i}i} withoutCoT Step for Step Euclideanto the“Concluding”816–858 Step [816,259,858,289]. Step “AIR”, 3412  top:::: TheTheThe“Concluding”forDistanceleft cornerplacingandcentersminimumvertical“AIR”;259–289“AIR” Geometry-Related CoT fromareofdistanceprojectionistheat“AIR”thelocated(101,33)forthehorizontalbottom“AIR,” bottom right.fromatisfor35–167[35,17,167,49]with noright“Concluding”“Concluding”projectioncornerforoverlapof “Concluding.”.“Concluding”toofandis.and“AIR”“Concluding”17–49(827,274)“AIR”is andtheforat<br>Final answer: Title. with CoT i} Final answer: 31. with CoT i} Final answer: 682. with CoT<br>(a) Layout-Type-Related CoT. (b) Table-Structure-Aware CoT. (c) Geometry-Related CoT.<br>**----- End of picture text -----**<br>


Figure 2. Examples of CoT Pre-training across different tasks. Red text highlights key steps in the chain of thought for each task. 

tion of annealing to the data utilized in the supervised finetuning phase, specifically from the perspective of CoT. 

## **3.1. Model Architecture** 

The overall architecture of our DocLayLLM is illustrated in Figure 1. To process a document image, we first employ an OCR engine. This engine extracts the OCR results, which encompass the textual content, denoted as _Text_ 1: _N_ , and the associated bounding box, represented by _Box_ 1: _N_ . Here, _N_ indicates the total number of text segments detected by the OCR, and each _Boxi_ = [ _x_ 0 _, y_ 0 _, x_ 1 _, y_ 1] defines the bounding box for _Texti_ . The bounding box coordinates correspond to the _left, top, right,_ and _bottom_ boundaries of the text segment. After the OCR process, the document image is segmented into patches, denoted as _Patch_ 1: _M_ , where _M_ represents the total number of patches. Next, the OCR results and the document patches are integrated with the fixed part of the prompt and the specific _Question_ related to the document. This integration forms the final prompt input to the model, presented in a natural language expression _T_ : _Given the document patches: {Patch_ 1: _M } and the document text contents and locations in the form of “text,_ [ _left, top, right, bottom_ ] _”_ : _{_ ( _Texti, Boxi_ )1: _N } {Question}_ 

Subsequently, the textual components of the prompt, including the fixed expression, _Text_ 1: _N_ , and _Question_ , are processed using the LLM’s native text embedder. This generates the embeddings for the text content as follows: 

**==> picture [181 x 11] intentionally omitted <==**

where _TE_ is the native text embedder of LLM, and _TextContent_ consists of the fixed expression, _Text_ 1: _N_ , and _Question_ . 

As for _Patch_ 1: _M_ and _Box_ 1: _N_ , they are first embedded with an extra embedder, then mapped to the LLM’s feature space via a linear projector, yielding _EmbV_ and _EmbL_ as: 

**==> picture [185 x 26] intentionally omitted <==**

where _V E_ and _V P_ represent the visual embedder and visual projector, _LE_ and _LP_ denote the layout embedder and layout projector, respectively. It is worth noting that despite the introduction of _V E_ , _V P_ , _LE_ , and _LP_ , these are all simple embedding layers or linear layers, thereby minimizing additional computational burden. 

Finally, LLM will comprehend the embeddings provided and utilize them to generate the _Answer_ to the _Question_ : 

**==> picture [229 x 11] intentionally omitted <==**

Here _π_ represents organizing _EmbT_ , _EmbV_ , _EmbL_ into a feature sequence according to _T_ . 

With our design, we have preserved the original structure of the LLM, ensuring its generalization capability. At the same time, we have enabled the LLM to easily accept and process visual and layout features, making it better suited for understanding text-rich documents with various layouts. This results in a simple yet effective multi-modal extension. 

## **3.2. CoT Pre-training** 

During the pre-training phase, we incorporate a wide range of tasks (as detailed in the appendix) to boost the LLM’s understanding of the visual and layout features inspired by Luo et al. [45]. Since the datasets related to these tasks are not designed explicitly for instruction-tuning, their annotations are not in a QA format. Therefore, we manually create QA-formatted instructions based on their annotations for training. However, simply converting task descriptions into _Question_ and annotations into _Answer_ is highly inefficient regarding data utilization. To address this, we incorporate CoT during the conversion to instruction-tuning data for specific tasks, asking LLM to solve our problem stepby-step thus enhancing the model’s comprehension abilities, as the examples illustrated in Fig 2. Our designed CoT primarily falls into the following three categories: 

## **3.2.1. Layout-Type-Related CoT** 

We conduct a layout-type-related CoT for the layout analysis task. To determine the layout type of a given area, it is 

4 

beneficial first to locate its region within the document and then infer based on the nearby layout element. When giving an area bounded by a bounding box, our layout-type-related CoT can be formulated as following steps: 

1. Identify the text within the area and determine its approximate region within the document (e.g., top left, top right part of the document, etc). 

2. Locate the nearest bounding boxes and assess their layout type. 

3. Infer the layout type of the given area based on the above information. 

Through this CoT process, we enable the LLM to fully utilize the visual, layout, and textual features of the document, thereby improving their ability to understand the document’s overall structure. 

## **3.2.2. Table-Structure-Aware CoT** 

Table-structure-aware CoT is primarily employed in the table analysis task. We have integrated the XY-Cut algorithm [48]—a widely-used technique for analyzing table structures. For instance, to locate an element at the _i_ -th row and _j_ -th column, the CoT process is outlined as follows: 

1. Identify all the table headers and their respective locations. The header of the _j_ -th column corresponds to the _j_ -th element in the headers list. 

2. Output all elements in the _j_ -th column when told that the elements of the same column are roughly vertically aligned with their corresponding headers. 

3. Determine the final answer based on the _i_ -th element of the contents in the _j_ -th column. 

Applying the above CoT, we break down a complex question into simple steps using the basic XY-Cut thought. This approach improves the LLM’s inference ability through this breakdown and enhances data utilization by effectively leveraging the overall structure of the table. 

## **3.2.3. Geometry-Related CoT** 

This is used in geometric analysis to identify geometric relationships between two text segments. It assists LLM to inference with geometric relations. Specifically, when giving the text content of two lines: 

1. Recover the bounding box of the given text segments. 

2. Analyze their vertical and horizontal projections to determine if they overlap through interval analysis. 

3. Calculate the center point of each segment and determine their relative location by comparing their center points. 

4. Calculate the minimum Euclidean distance based on the previous analysis. If they overlap, the distance is 0; if they overlap in vertical/horizontal projections, the minimum distance is the smallest interval in the horizontal/vertical projections; otherwise, it is the minimum Euclidean distance between their nearest corners. 

This design advances the LLM’s arithmetic reasoning capabilities and helps it more accurately grasp the spatial re- 

lationships between different boxes, and also contributes to layout-type-aware CoT to find the nearest bounding boxes. 

## **3.3. CoT Annealing** 

Data annealing involves gradually increasing the proportion of high-quality data towards the end of supervised finetuning to improve data utilization efficiency and augment the model’s reasoning abilities. This technique has been proven to be effective and is widely adopted in models like MiniCPM [19], DataComp-LM [31], and Llama-3.1 [42]. 

During the SFT stage, we employ data with a layoutaware CoT [45] that helps focus on potential regions of questions and answers within documents, thereby better leveraging the layout features learned during pre-training. It also allows the model to strengthen its reasoning capacity through step-by-step thinking. However, in the latter stages of SFT, our goal for LLM is to output precise and direct answers to questions. At this point, using CoT data may introduce unnecessary noise with a long response, and the overly uniform layout-aware CoT data might partially compromise the LLM’s generalization abilities. 

To balance the strengths and weaknesses of such data, we propose CoT annealing, which reconsiders this data annealing issue from a CoT perspective. Specifically, we generate direct-answer data without CoT from the CoT data, treating it as high-quality data. During the SFT process, we gradually increase the proportion of this direct-answer data, thereby enhancing the LLM’s ability to respond with a clear and direct answer in the end. Initially, we only utilize the data with CoT, gradually adjusting the ratio of CoT to nonCoT data as the SFT process progresses. Ultimately, we conclude the training using only the data without CoT. 

## **4. Experiments** 

## **4.1. Datasets** 

During the pre-training stage, we conduct our pre-training tasks on several datasets, including RVL-CDIP [13], PubLayNet [80], PubTabNet [81], DocLayNet [54], DocBank [32], DocILE [56], and the Document Dense Description data from LayoutLLM [45]. This results in a total of 3.1 million pairs of instruction tuning data, which is significantly less than the volume used in previous OCRdependent TDU methods. For the supervised fine-tuning stage, we apply CoT annealing on the layout-aware document QA data utilized in LayoutLLM [45]. 

We evaluate our DocLayLLM on widely used textrich document benchmarks, including FUNSD [22], CORD [50], SROIE [21], POIE [25], DocVQA [46], InfoVQA [47], VisualMRC [59], DeepForm [58], KLC [57], WTQ [51], and TabFact [4]. For a fair comparison with other approaches, we use the default OCR data from the of- 

5 

|**Method**|**Processed Data**<br>**at Pre-training**|**Document-oriented VQA**<br>DocVQA<br>VisualMRC<br>**Avg.**|**KIE**<br>FUNSD_†_<br>CORD_†_<br>SROIE_†_<br>**Avg.**|
|---|---|---|---|
|**Text**<br>Llama2-7B-Chat [64]<br>Llama3-8B-Instruct [42]|-<br>-|20.50<br>9.90<br>15.20<br>51.79<br>47.77<br>49.78|15.10<br>20.00<br>35.6<br>23.57<br>68.56<br>52.31<br>61.24<br>60.70|
|**Text + Box**<br>Llama2-7B-Chat_coor_ [14]<br>Llama3-8B-Instruct_coor_ [14]<br>LayTextLLM_zero_ [43]|-<br>-<br>10.0M (323%)|12.30<br>12.20<br>12.25<br>49.13<br>41.75<br>45.44<br>65.50<br>37.40<br>51.45|11.90<br>6.40<br>39.40<br>19.23<br>74.00<br>62.20<br>63.15<br>66.45<br>72.00<br>45.50<br>82.00<br>66.50|
|**Text + Box + Patch**<br>LayoutLLM_Llama_2[45]|5.7M (184%)|74.25<br>55.73<br>64.99|78.65<br>62.21<br>70.97<br>70.61|
|_DocLayLLMLlama_2**(Ours)**|**3.1M (100%)**|72.83<br>55.04<br>63.94|78.67<br>70.81<br>83.27<br>77.58|
|_DocLayLLMLlama_3**(Ours)**|**3.1M (100%)**|**78.36**<br>**58.55**<br>**68.46**|**84.11**<br>**71.34**<br>**84.36**<br>**79.94**|



Table 1. Comparison with OCR-dependent methods using metrics from LayoutLLM [45]. **Processed Data at Pre-training** represents the total data volume processed at the pre-training stage, equivalent to dataset size multiplied by epoch count, with values in parentheses representing size ratios compared to DocLayLLM. **Avg.** represents the average performance across different benchmarks, and _[†]_ indicates the cleaned test set used in LayoutLLM. The suffix _layout_ denotes using method from He et al. [14], which provides LLM with layout features via OCR box coordinates in textual form. **Bold** indicates the best performance, and underline indicates the second-best one. 

|**Method**|**Processed Data**<br>**at Pre-training**|**Document-oriented VQA**<br>DocVQA<br>VisualMRC<br>**Avg.**|**Document-oriented VQA**<br>DocVQA<br>VisualMRC<br>**Avg.**|**Document-oriented VQA**<br>DocVQA<br>VisualMRC<br>**Avg.**|**KIE**<br>FUNSD<br>CORD<br>SROIE<br>**Avg.**|
|---|---|---|---|---|---|
|**Text**<br>Llama2-7B-Chat [64]<br>Llama3-8B-Instruct [42]|-<br>-|20.50<br>51.79|6.30<br>229.74|13.40<br>140.77|23.40<br>51.80<br>58.60<br>44.60<br>66.67<br>74.71<br>82.51<br>74.63|
|**Text + Box**<br>Llama2-7B-Chat_coor_ [14]<br>Llama3-8B-Instruct_coor_ [14]<br>LMDX [53]<br>DocLLM [67]<br>LayTextLLM_zero_[43]|-<br>-<br>-<br>16.8M (542%)<br>10.0M (323%)|12.30<br>49.13<br>-<br>69.50_∗_<br>65.50|28.00<br>211.69<br>-<br>264.10_∗_<br>200.20|20.15<br>130.41<br>-<br>166.80_∗_<br>132.85|14.40<br>38.10<br>50.60<br>34.37<br>74.71<br>75.33<br>75.93<br>75.32<br>-<br>66.95<br>-<br>-<br>51.80_∗_<br>67.60_∗_<br>**91.90**_∗_<br>70.43_∗_<br>47.20<br>77.20<br>83.80<br>69.40|
|**Text + Box + Patch**||||||
|_DocLayLLMLlama_2**(Ours)**|**3.1M (100%)**|72.83|310.60|191.72|80.74<br>79.37<br>84.37<br>81.49|
|_DocLayLLMLlama_3**(Ours)**|**3.1M (100%)**|**78.35**|**357.04**|**217.70**|**86.47**<br>**83.66**<br>86.08<br>**85.40**|



Table 2. Comparison with OCR-dependent methods using metrics from DocLLM [67]. _[∗]_ indicates that the model uses the training set of the benchmark during training. 

ficial datasets otherwise commercial OCR engines[1] if not provided, and follow their evaluation pipeline and metrics. 

## **4.2. Implement Details** 

Our method is based on Llama3-8B-Instruct [42], with a Llama2-7B-Chat [64] version for better comparison with other methods. They are denoted as DocLayLLM _Llama_ 3 and DocLayLLM _Llama_ 2 respectively. We initialize _V E_ and _LE_ with weights from LayoutLMv3 [20] to leverage document knowledge acquired during its pre-training, while _V P_ and _LP_ are randomly initialized. Besides, the image resolution is fixed at 224*224, resulting in 196 visual patch tokens, to capture an overview of document layout while bringing less computation burden. During pre-training and fine-tuning, we use LoRA [18] with a rank of 64 on LLM and keep _V E_ , _V P_ , _LE_ , and _LP_ trainable, leading to few tuning parameters. We train our model on 8 NVIDIA Tesla A100-80G GPUs, taking about 30 A100 days for pre- 

> 1https://www.textin.com/ 

training and 6 A100 days for supervised fine-tuning. Additional details can be found in the appendix. 

## **4.3. Comparisons with SOTA OCR-dependent methods** 

We have compared our DocLayLLM with existing SOTA OCR-dependent methods. In order to exhibit the zero-shot inferring ability precisely, we chose their performances under the zero-shot scenario unless they are not provided. 

The experimental results are shown in Table 1 and Table 2. Our method outperforms the baseline models, regardless of their type or whether the document’s layout information is provided to them. It validates the rationale behind our design of multi-modal extension. 

Additionally, our method significantly outperforms previous SOTA methods despite fewer training resources. As shown in Table 1, it surpasses the performance of LayoutLLM [45] and LayTextLLM [43] though requiring fewer computational resources by using less data and tuning few 

6 

|**Method**|**Document-oriented VQA**<br>DocVQA<br>InfoVQA<br>**Avg.**|**Document-oriented VQA**<br>DocVQA<br>InfoVQA<br>**Avg.**|**Document-oriented VQA**<br>DocVQA<br>InfoVQA<br>**Avg.**|**Document-oriented VQA**<br>DocVQA<br>InfoVQA<br>**Avg.**|**KIE**<br>FUNSD<br>SROIE<br>POIE<br>**Avg.**|
|---|---|---|---|---|---|
|LLaVAR [79]<br>UniDoc [10]<br>DocPedia [9]<br>Monkey [38]<br>TextMonkey [41]<br>DocOwl 1.5 [17]<br>TextSquare [61]<br>DocKylin [78]||12.30<br>7.70<br>47.10_∗_<br>50.10_∗_<br>66.70_∗_<br>-<br>-<br>65.10_∗_|16.50<br>14.70<br>15.20_∗_<br>25.80_∗_<br>28.60_∗_<br>-<br>-<br>34.80_∗_|14.40<br>11.20<br>31.15_∗_<br>37.95_∗_<br>47.65_∗_<br>-<br>-<br>49.95_∗_|0.50<br>5.20<br>5.90<br>3.87<br>1.00<br>2.90<br>5.10<br>3.00<br>29.90<br>21.40<br>39.90<br>30.40<br>24.10<br>41.90<br>19.90<br>28.63<br>42.90<br>46.20<br>32.00<br>40.37<br>-<br>48.30<br>51.80<br>-<br>-<br>53.20<br>71.80<br>-<br>58.70<br>25.50<br>49.50<br>44.57|
|_DocLayLLMLlama_3**(Ours)**||**77.79**|**42.02**|**59.91**|**80.28**<br>**76.59**<br>**75.13**<br>**77.33**|



Table 3. Comparison with OCR-free methods using the precision metric from TextMonkey [41]. _[∗]_ indicates that the model uses the training set of the benchmark during training. 

|**Method**|**Document-oriented VQA**<br>DocVQA<br>InfoVQA<br>VisualMRC|**Document-oriented VQA**<br>DocVQA<br>InfoVQA<br>VisualMRC|**Avg.**|**KIE**<br>DeepForm<br>KLC<br>**Avg.**|**Table Understanding**<br>WTQ<br>TabFact<br>**Avg.**|
|---|---|---|---|---|---|
|DocPedia [9]<br>DocOwl [75]<br>UReader [76]<br>Monkey [38]<br>TextMonkey [41]<br>DocKylin [78]<br>CogAgent [16]<br>DocOwl 1.5 [17]<br>TextSquare [61]|47.10<br>15.20<br>62.20<br>38.20<br>65.40<br>42.20<br>66.50<br>36.10<br>73.00<br>-<br>77.30<br>46.60<br>81.60<br>44.50<br>82.20<br>50.70<br>84.30<br>51.50|-<br>188.80<br>221.70<br>-<br>-<br>319.90<br>-<br>246.40<br>-|-<br>96.40<br>109.77<br>-<br>-<br>147.93<br>-<br>126.43<br>-|-<br>-<br>-<br>42.60<br>30.30<br>36.45<br>49.50<br>32.80<br>41.15<br>40.60<br>32.80<br>36.70<br>59.70<br>37.80<br>48.75<br>64.20<br>-<br>-<br>-<br>-<br>-<br>68.80<br>38.70<br>53.75<br>-<br>-<br>-|-<br>-<br>-<br>26.90<br>60.20<br>43.55<br>29.40<br>67.60<br>48.50<br>25.30<br>-<br>-<br>31.90<br>-<br>-<br>32.40<br>-<br>-<br>-<br>-<br>-<br>40.60<br>80.20<br>60.40<br>49.70<br>**84.20**<br>66.95|
|_DocLayLLM ‡_<br>_Llama_3 **(Ours)**|**86.52**<br>**58.36**|**327.91**|**157.60**|**77.07**<br>**40.73**<br>**58.90**|**58.62**<br>83.36<br>**70.99**|



Table 4. Comparison with OCR-free methods using the evaluation framework and metrics from DocOwl 1.5 [17]. All methods, including our _DocLayLLMLlama[‡]_ 3[variant, are exposed to the train set corresponding to the evaluation set during training.] 

parameters at training. Even with Llama2 as the backbone, our approach matches the performance of previous SOTA methods in the Document-oriented VQA task and significantly outperforms them in the KIE tasks, further demonstrating the efficiency and effectiveness of our approach. 

Besides, our method exhibits robust zero-shot generalization capability without fine-tuning with specific downstream tasks. From the comparison with DocLLM [67] in Table 2, it is clear that our method, even without fine-tuning, significantly outperforms DocLLM across most datasets and achieves a considerable overall performance improvement. This validates our design, which has preserved the strong generalization ability of the original LLM structure. 

In summary, DocLayLLM serves as an efficient extension of LLMs. It demonstrates strong zero-shot generalization and effective document understanding by encoding documents within a unified model. Even with limited training resources, DocLayLLM achieves new SOTA performance compared to OCR-dependent competitors. 

## **4.4. Comparisons with SOTA OCR-free methods** 

We have also compared our method with SOTA OCR-free TDU models. Table 3 presents the precision of our method and some advent OCR-free TDU methods across various benchmarks. Even though in the circumstance that OCR- 

free models use some of the training sets of these benchmarks during training, while our method infers in a purely zero-shot manner, our DocLayLLM still outperforms these models, particularly in KIE task, which heavily relies on layout information. This clearly demonstrates the superiority of our design to provide LLM with OCR information. 

We additionally test DocLayLLM on a broader set of benchmarks from DocOwl 1.5 [17]. To further assess the performance of DocLayLLM on seen datasets, we incorporate the training sets of these benchmarks used in DocOwl 1.5 during the SFT process as other OCR-free TDU methods do during training. To maintain the efficiency advantage of our method, we ensure that the SFT is conducted with the same number of steps as the standard DocLayLLM. The results in Table 4 show that with the inclusion of downstream task data during SFT, our method outperforms the SOTA OCR-free TDU models in most datasets. Moreover, our overall performance in the Document-oriented VQA, KIE, and Table Understanding sub-tasks surpasses the OCR-free TDU methods using limited training resources, showcasing the tremendous potential of OCR-dependent methods and further reinforcing the reliability of our method. 

Overall, our DocLayLLM consistently shows strong zero-shot inference ability, outperforming many OCR-free TDU methods without prior exposure to downstream data. 

7 

**==> picture [482 x 171] intentionally omitted <==**

**----- Start of picture text -----**<br>
rgTo) 15, JALAN PERHAS 10/7, PERMASf) 08/26 Ground Truth : CONTENTO Ground Truth :  2.6g a Experiment DocVQA Document-oriented VQA VisualMRC Avg. DeepForm KIE KLC Avg.<br>aa50 nasit sno ] em ! a DocLayLLM Llama 3 77.43 316.27 196.85 35.41 26.25 30.83<br>;1 User fC) What is the name of company in this receipt? ; ! User Cc) for total fat of per What is the value serving? 1 w/o CoT Pre-trainingw/o CoT Annealing 76.0077.40 295.49310.17 185.75193.79 24.5434.23 25.4926.21 25.0230.22<br>I × Not mentioned in the document. LayTextLLM × 7.9g LayTextLLM w/o pt VE & LE 77.42 308.49 192.96 31.31 25.76 28.54<br>1 × CL} YAN NIACK DocOwl 1.5 I × Lo} 7.9g DocOwl 1.5 1 (a) a Impact of different settings: w/o CoT Pre-training and w/o CoT<br>toe SROIE [🗸] LCL} CONTENTO DocLayLLM(Ours) !! POIE 🗸 L___} 2.6g DocLayLLM(Ours) 11 Annealing tuning with indicateonly CoTpre-trainingdata, whilewithout w/o pt VE CoT and and PE superviseduses randomlyfine-<br>i|Seee Ground Truth efforts were successful. Retail accounts that previously stocked Style Full Price accepted the introduction of the low price.:  Overall pre- sell  ) Lene Ground Truth Koplow, Mr. Arthur F. Golden, Mr. Martin Barrington, Mr. F. Anthony Burke, Mr. Ronald Milstein, Mr. Charles A. Blixt, Mr. Stephen R. Patton. : Mr. Meyer G.  I initialized visual and layout embedders instead of pre-trained ones.<br>;1! User Cd × What is the content in the "PRE-SELL" field? Not specified in the document. LayTextLLM ;!| User Cc) × "Recipient(s):" field?What is the content in the Company: Phone No.: Fax No.: LayTextLLM 1I a T ✓ Modality ✗ L V ✗ | DocVQA73.57 Document-oriented VQA VisualMRC280.27 176.93 Avg. | DeepForm15.76 KIE 24.09KLC 19.93 Avg.<br>!1!br FUNSD 🗸 × CL} Overall pre-sell efforts were accepted the introduction of accounts that previously stocked Style Full Price Overall pre- sell efforts were successful. Retail the low price.successful.  DocOwl DocLayLLM(Ours) 1.5 ! l I FUNSD 🗸 × LC}CL} Mr. Martin Barrington, Mr. Stephen R. Patton.Mr. F. Anthony Burke, Mr. Arthur F. Golden, Mr. Meyer G. Koplow, Mr. Charles A. Blixt, Mr. Ronald Milstein, Company: DocOwl DocLayLLM(Ours) 1.5 I 1 I| (b) ——————————— ✓✓✓ Impacts(II)(II)(I) ✓✗✗of different75.7277.32 77.43 modality306.51310.75 316.27 configurations.191.12194.03 196.85 14.1333.75 35.41T , L , 24.2325.97 26.25 and V 19.1829.86 30.83 repre-<br>**----- End of picture text -----**<br>


|**Modality**<br>**Document-oriented VQA**<br>**KIE**<br>**T**<br>**L**<br>**V**<br>DocVQA73.57<br>VisualMRC280.27<br>**Avg.**<br>DeepForm15.76<br>KLC<br>**Avg.**<br>✓<br>✗<br>✗<br>73.57<br>280.27<br>176.93<br>15.76<br>24.09KLC<br>19.93<br>✓✓✓<br>(I)<br>✗✗of<br>75.7277.32<br>306.51310.75<br>191.12194.03<br>14.1333.75<br>24.2325.97<br>19.1829.86<br>✓✓<br>(II)(II)(I)<br>✗of<br>77.32<br>310.75<br>194.03<br>33.75<br>25.97<br>29.86<br>✓<br>(II)(I)<br>✓✗✗of<br>**77.43**<br>**316.27**<br>**196.85**<br>**35.41T**<br>**26.25**<br>**30.83**<br>(b) Impacts(II)(II)(I) of different75.7277.32 modality306.51310.75 configurations.191.12194.03 **T**, **L**, and **V** repre-<br>1I<br>|<br>|<br>I<br>~~a~~<br>I<br>1!br<br>~~l~~<br>~~1~~<br>!br<br>I|<br>~~———————————~~<br>||
|---|
|sent textual, layout, and visual modalities, respectively. **(I)** indicates|
|other<br>inputting OCR coordinates as text following He et al. [14], while**(II)**|
|Zoom in for better view.<br>embeds coordinates with_LE_ in our approach.|



Figure 3. Qualitative comparisons of DocLayLLM with other OCR-dependent and OCR-free methods. Zoom in for better view. 

Table 5. Ablation study on various designs of our DocLayLLM. 

Additionally, when incorporating downstream datasets during SFT like other OCR-free approaches, DocLayLLM surpasses them, highlighting its potential in solving TDU tasks by integrating OCR information into the LLM. 

## **4.5. Qualitative Results** 

Figure 3 shows some qualitative examples to visualize the performance on KIE tasks of our DocLayLLM compared with LayTextLLM and DocOwl 1.5. The example from SROIE clearly shows that DocLayLLM has improved document comprehension and answer accuracy. And the example from POIE demonstrates our advantage in handling table-formatted documents to give an accurate response. Furthermore, as seen in the FUNSD examples, DocLayLLM excels at extracting answers spanning multiple text lines, outperforming both OCR-dependent and OCRfree methods. For more visualization of qualitative examples, please refer to the appendix. 

## **4.6. Ablation Study** 

We have also conducted additional experiments to explore the effectiveness of our proposed method. To conserve computational resources, we perform pre-training at 25% of the standard scale (including the baseline) but full-scale SFT. We focus on the impact of two main factors: (a) the role of CoT pre-training and CoT annealing, and (b) the influence of different modality configurations. 

Experimental results in Table 5a indicate that omitting CoT Pre-training leads to an obvious performance drop in both the Document-oriented VQA and KIE tasks, while not using CoT Annealing primarily affects VQA performance. Besides, using randomly initialized _V E_ and _LE_ partially compromises the proposed extension, resulting in performance degradation in both the VQA and KIE tasks. 

Additionally, performances in Table 5b show the efficacy 

of our DocLayLLM to incorporate different modalities for TDU tasks. It is obvious that using only text from the document is insufficient for TDU, while adding layout information in textual form provides some gain on VQA tasks but leads to over-length input (see appendix for analysis), greatly slowing the inference. By our design to integrate layout features via an _LE_ , we achieve notable improvements on these TDU tasks with shorter inputs. And adding visual features further boosts the performance of our method. 

Overall, the combined impact of these designs ensures the efficiency and effectiveness of our method. 

## **5. Conclusion** 

We propose DocLayLLM, an efficient multi-modal extension of large language models for text-rich document understanding. By streamlined integration of document patch tokens and OCR positional tokens into the LLM inputs, it achieves a simple yet efficient multi-modal extension of LLM to enable it to have the capability to perceive and process textual, visual, and layout information from the OCR results of the document. Additionally, we introduce CoT Pre-training and CoT Annealing techniques centered around CoT, offering a new way to generate CoT data and further enhancing the model’s understanding of textrich documents. Despite requiring fewer training resources, our method surpasses existing OCR-dependent TDU models, demonstrating strong zero-shot generalization capability. Furthermore, it also exceeds the performance of SOTA OCR-free TDU methods under comparable training conditions, showcasing remarkable performance across various document understanding tasks. Our work significantly advances efficient MLLM-based document understanding while maintaining high-performance standards. 

8 

## **Acknowledgement** 

This research is supported in part by National Natural Science Foundation of China (Grant No.: 62476093, 62441604). 

## **References** 

- [1] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. DocFormer: End-to-End Transformer for Document Understanding. In _Proc. ICCV_ , pages 993–1003, 2021. 1 

- [2] Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R. Manmatha. DocFormerv2: Local Features for Document Understanding. In _Proc. AAAI_ , pages 709–718, 2024. 1 

- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language Models are Few-shot Learners. _Proc. NIPS_ , 33:1877–1901, 2020. 2 

- [4] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. TabFact: A Large-scale Dataset for Table-based Fact Verification. In _Proc. ICLR_ , 2020. 5 

- [5] Gang Dai, Yifan Zhang, Qingfeng Wang, Qing Du, Zhuliang Yu, Zhuoman Liu, and Shuangping Huang. Disentangling Writer and Character Styles for Handwriting Generation. In _Proc. CVPR_ , pages 5977–5986, 2023. 3 

- [6] Gang Dai, Yifan Zhang, Quhui Ke, Qiangya Guo, and Shuangping Huang. One-dm: One-shot diffusion mimicker for handwritten text generation. In _Proc. ECCV_ , pages 410– 427, 2025. 3 

- [7] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _arXiv: Comp. Res. Repository_ , abs/2501.12948, 2025. 2 

- [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _Proc. NAACL_ , pages 4171–4186, 2019. 1 

- [9] Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, and Can Huang. DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding. _arXiv: Comp. Res. Repository_ , abs/2311.11810, 2023. 7 

- [10] Hao Feng, Zijian Wang, Jingqun Tang, Jinghui Lu, Wengang Zhou, Houqiang Li, and Can Huang. UniDoc: A Universal Large Multimodal Model for Simultaneous Text Detection, Recognition, Spotting and Understanding. _arXiv: Comp. Res. Repository_ , abs/2308.11592, 2023. 7 

- [11] Masato Fujitake. LayoutLLM: Large Language Model Instruction Tuning for Visually Rich Document Understanding. In _Proc. LREC/COLING_ , pages 10219–10224, 2024. 2, 3 

- [12] Łukasz Garncarek, Rafał Powalski, Tomasz Stanisławek, Bartosz Topolski, Piotr Halama, Michał Turski, and Filip Grali´nski. LAMBERT: Layout-Aware Language Modeling for Information Extraction. In _Proc. ICDAR_ , pages 532–547, 2021. 1 

- [13] Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis. Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval. In _Proc. ICDAR_ , pages 991– 995, 2015. 5 

- [14] Jiabang He, Lei Wang, Yi Hu, Ning Liu, Hui Liu, Xing Xu, and Heng Tao Shen. ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction. In _Proc. ICCV_ , pages 19428–19437, 2023. 1, 2, 3, 6, 8, 14 

- [15] Teakgyu Hong, DongHyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae Park. BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents. In _Proc. AAAI_ , pages 10767–10775, 2022. 1 

- [16] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and Jie Tang. CogAgent: A Visual Language Model for GUI Agents. In _Proc. CVPR_ , pages 14281– 14290, 2024. 7 

- [17] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mPLUG-DocOwl 1.5: Unified Structure Learning for OCRfree Document Understanding. In _Findings EMNLP_ , pages 3096–3120, 2024. 1, 2, 7, 13, 14 

- [18] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In _Proc. ICLR_ , 2022. 2, 6 

- [19] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zhen Leng Thai, Kai Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. _arXiv: Comp. Res. Repository_ , abs/2404.06395, 2024. 5 

- [20] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In _Proc. ACM MM_ , pages 4083–4091, 2022. 1, 2, 3, 6 

- [21] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. In _Proc. ICDAR_ , pages 1516–1520, 2019. 5 

- [22] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. In _Workshop on ICDAR_ , pages 1–6, 2019. 5 

- [23] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. OCR-Free 

9 

Document Understanding Transformer. _Proc. ECCV_ , 13688: 498–517, 2022. 2 

- [24] Geewook Kim, Hodong Lee, Daehee Kim, Haeji Jung, Sanghee Park, Yoonsik Kim, Sangdoo Yun, Taeho Kil, Bado Lee, and Seunghyun Park. Visually-Situated Natural Language Understanding with Contrastive Reading Model and Frozen Large Language Models. In _Proc. EMNLP_ , pages 11989–12010, 2023. 2, 3 

- [25] Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, and Xiang Bai. Visual Information Extraction in the Wild: Practical Dataset and End-to-End Solution. In _Proc. ICDAR_ , pages 36–53, 2023. 5 

- [26] Hugo Laurenc¸on, Andr´es Marafioti, Victor Sanh, and Leo Tronchon. Building and better understanding visionlanguage models: insights and future directions. In _Workshop on NIPS_ , 2024. 3 

- [27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy Visual Task Transfer. _Transactions on Machine Learning Research_ , 2025. 3 

- [28] Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si. StructuralLM: Structural Pre-training for Form Understanding. In _Proc. ACL_ , pages 6309–6318, 2021. 1 

- [29] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. LLaVaNeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models. _arXiv: Comp. Res. Repository_ , abs/2407.07895, 2023. 2 

- [30] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. _Proc. ICML_ , 202:19730–19742, 2023. 3 

- [31] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, ChengYu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah M. Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Raghavi Chandu, Thao Nguyen, Igor Vasiljevic, Sham M. Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. DataComp-LM: In search of the next generation of training sets for language models. _arXiv: Comp. Res. Repository_ , abs/2406.11794, 2024. 2, 5 

- [32] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. DocBank: A Benchmark Dataset for Document Layout Analysis. In _Pro. COLING_ , pages 949–960, 2020. 5 

- [33] Peizhao Li, Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Varun Manjunatha, and Hongfu Liu. SelfDoc: Self-Supervised Document Representation Learning. In _Proc. CVPR_ , pages 5652–5660, 2021. 1 

- [34] Qiwei Li, Zuchao Li, Xiantao Cai, Bo Du, and Hai Zhao. Enhancing Pisually-Pich Pocument Pnderstanding via Payout Ptructure Podeling. In _Proc. ACM MM_ , page 4513–4523, 2023. 

- [35] Qiwei Li, Zuchao Li, Ping Wang, Haojun Ai, and Hai Zhao. Hypergraph based Understanding for Document Semantic Entity Recognition. In _Proc. ACL_ , pages 2950–2960, 2024. 

- [36] Xin Li, Yan Zheng, Yiqing Hu, Haoyu Cao, Yunfei Wu, Deqiang Jiang, Yinsong Liu, and Bo Ren. Relational Representation Learning in Visually-Rich Documents. In _Proc. ACM MM_ , page 4614–4624, 2022. 

- [37] Yulin Li, Yuxi Qian, Yuechen Yu, Xiameng Qin, Chengquan Zhang, Yan Liu, Kun Yao, Junyu Han, Jingtuo Liu, and Errui Ding. StrucTexT: Structured Text Understanding with MultiModal Transformers. In _Proc. ACM MM_ , page 1912–1920, 2021. 1 

- [38] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models. In _Proc. CVPR_ , pages 26763–26773, 2024. 1, 2, 7 

- [39] Zening Lin, Jiapeng Wang, and Lianwen Jin. Visual Information Extraction Deep Learning Method: A Critical Review. _Journal of Image and Graphics_ , 28(08):2276–2297, 2023. 1 

- [40] Chenglin Liu, Lianwen Jin, Xiang Bai, Xiaohui Li, and Fei Yin. Frontiers of intelligent document analysis and recognition: review and prospects. _Journal of Image and Graphics_ , 28(08):2223–2252, 2023. 1 

- [41] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. TextMonkey: An OCRFree Large Multimodal Model for Understanding Document. _arXiv: Comp. Res. Repository_ , abs/2403.04473, 2024. 7 

- [42] Meta AI Llama Team. The Llama 3 Herd of Models. _arXiv: Comp. Res. Repository_ , abs/2407.21783, 2024. 2, 5, 6, 13, 14 

- [43] Jinghui Lu, Haiyang Yu, Yanjie Wang, Yongjie Ye, Jingqun Tang, Ziwei Yang, Binghong Wu, Qi Liu, Hao Feng, Han Wang, Hao Liu, and Can Huang. A Bounding Box is Worth One Token: Interleaving Layout and Text in a Large Language Model for Document Understanding. _arXiv: Comp. Res. Repository_ , abs/2407.01976, 2024. 6 

- [44] Chuwei Luo, Changxu Cheng, Qi Zheng, and Cong Yao. GeoLayoutLM: Geometric Pre-Training for Visual Information Extraction. In _Proc. CVPR_ , pages 7092–7101, 2023. 1 

- [45] Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, Zhi Yu, and Cong Yao. LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding. In _Proc. CVPR_ , pages 15630–15640, 2024. 2, 3, 4, 5, 6 

- [46] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A Dataset for VQA on Document Images. In _Proc. WACV_ , pages 2199–2208, 2021. 5, 14 

10 

- [47] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. InfographicVQA. In _Proc. WACV_ , pages 1697–1706, 2022. 5, 14 

- [48] G. Nagy, S. Seth, and M. Viswanathan. A Prototype Document Image Analysis System for Technical Journals. _Computer_ , 25(7):10–22, 1992. 5 

- [49] OpenAI. Hello GPT-4o, 2024. 2 

- [50] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. In _Workshop on NIPS_ , 2019. 5 

- [51] Panupong Pasupat and Percy Liang. Compositional Semantic Parsing on Semi-Structured Tables. In _Proc. ACL_ , pages 1470–1480, 2015. 5, 14 

- [52] Qiming Peng, Yinxu Pan, Wenjin Wang, Bin Luo, Zhenyu Zhang, Zhengjie Huang, Yuhui Cao, Weichong Yin, Yongfeng Chen, Yin Zhang, Shikun Feng, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. ERNIE-Layout: Layout Knowledge Enhanced Pre-training for Visually-rich Document Understanding. In _Findings of EMNLP_ , pages 3744– 3756, 2022. 1 

- [53] Vincent Perot, Kai Kang, Florian Luisier, Guolong Su, Xiaoyu Sun, Ramya Sree Boppana, Zilong Wang, Zifeng Wang, Jiaqi Mu, Hao Zhang, Chen-Yu Lee, and Nan Hua. LMDX: Language Model-based Document Information Extraction and Localization. In _Findings of ACL_ , pages 15140– 15168, 2024. 3, 6 

- [54] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter W. J. Staar. DocLayNet: A Large HumanAnnotated Dataset for Document-Layout Segmentation. In _Proc. ACM SIGKDD Int. Conf. Knowledge discovery & data mining_ , pages 3743–3751, 2022. 5 

- [55] Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Pałka. Going Full-TILT Boogie on Document Understanding with Text-Image-Layout Transformer. In _Proc. ICDAR_ , pages 732–747, 2021. 1 

- [56] Step´an Simsa, Milan Sulc, Michal Uric´ar, Yash Patel, Ahmed Hamdi, Matej Koci´an, Maty´as Skalick´y, Jir´ı Matas, Antoine Doucet, Micka¨el Coustaty, and Dimosthenis Karatzas. DocILE Benchmark for Document Information Localization and Extraction. In _Proc. ICDAR_ , pages 147– 166, 2023. 5 

- [57] Tomasz Stanislawek, Filip Gralinski, Anna Wr´oblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemyslaw Biecek. Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts. In _Proc. ICDAR_ , pages 564–579, 2021. 5 

- [58] Stacey Svetlichnaya. DeepForm: Understand Structured Documents at Scale. https://wandb.ai/stacey/ deepform_v1, 2020. Accessed: 2024-08-14. 5, 14 

- [59] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. VisualMRC: Machine Reading Comprehension on Document Images. In _Proc. AAAI_ , pages 13878–13888, 2021. 5, 14 

- [60] Ryota Tanaka, Taichi Iki, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. InstructDoc: A Dataset for Zero-shot Generalization of Visual Document Understanding with Instructions. In _Proc. AAAI_ , pages 19071–19079, 2024. 3 

- [61] Jingqun Tang, Chunhui Lin, Zhen Zhao, Shu Wei, Binghong Wu, Qi Liu, Hao Feng, Yang Li, Siqi Wang, Lei Liao, Wei Shi, Yuliang Liu, Hao Liu, Yuan Xie, Xiang Bai, and Can Huang. TextSquare: Scaling up Text-Centric Visual Instruction Tuning. _arXiv: Comp. Res. Repository_ , abs/2404.12803, 2024. 1, 2, 7 

- [62] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying Vision, Text, and Layout for Universal Document Processing. In _Proc. CVPR_ , pages 19254–19264, 2023. 2 

- [63] Qwen Team. QwQ: Reflect Deeply on the Boundaries of the Unknown, 2024. 2 

- [64] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models. _arXiv: Comp. Res. Repository_ , abs/2307.09288, 2023. 6, 13 

- [65] Yi Tu, Ya Guo, Huan Chen, and Jinyang Tang. LayoutMask: Enhance Text-Layout Interaction in Multi-modal Pretraining for Document Understanding. In _Proc. ACL_ , pages 15200–15212, 2023. 1 

- [66] Jianqiang Wan, Sibo Song, Wenwen Yu, Yuliang Liu, Wenqing Cheng, Fei Huang, Xiang Bai, Cong Yao, and Zhibo Yang. OmniParser: A Unified Framework for Text Spotting Key Information Extraction and Table Recognition. In _Proc. CVPR_ , pages 15641–15653, 2024. 2 

- [67] Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding. In _Proc. ACL_ , pages 8529–8548, 2024. 3, 6, 7 

- [68] Jiapeng Wang, Lianwen Jin, and Kai Ding. LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding. In _Proc. ACL_ , pages 7747–7757, 2022. 1 

- [69] Jiapeng Wang, Zening Lin, Dayi Huang, Longfei Xiong, and Lianwen Jin. LiLTv2: Language-substitutable layout-image transformer for visual information extraction. _ACM Transactions on Multimedia Computing, Communications and Applications_ , 21(3):1–27, 2025. 1 

11 

- [70] Wenjin Wang, Yunhao Li, Yixin Ou, and Yin Zhang. Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering. _arXiv: Comp. Res. Repository_ , abs/2306.00526, 2023. 1, 3 

- [71] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. _Proc. NIPS_ , 35:24824–24837, 2022. 2 

- [72] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. In _Proc. ACM SIGKDD Int. Conf. Knowledge discovery & data mining_ , page 1192–1200, 2020. 1 

- [73] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. In _Proc. ACL_ , pages 2579–2591, 2021. 1 

- [74] Hao Yan, Yuliang Liu, Lianwen Jin, and Xiang Bai. The development, application, and future of LLM similar to ChatGPT. _Journal of Image and Graphics_ , 28(9):2749–2762, 2023. 1 

- [75] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mPLUGDocOwl: Modularized Multimodal Large Language Model for Document Understanding. _arXiv: Comp. Res. Repository_ , abs/2307.02499, 2023. 7 

- [76] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Lin, and Fei Huang. UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model. In _Findings of EMNLP_ , pages 2841–2858, 2023. 1, 2, 7 

- [77] Chong Zhang, Ya Guo, Yi Tu, Huan Chen, Jinyang Tang, Huijia Zhu, Qi Zhang, and Tao Gui. Reading Order Matters: Information Extraction from Visually-rich Documents by Token Path Prediction. In _Proc. EMNLP_ , pages 13716– 13730, 2023. 1 

- [78] Jiaxin Zhang, Wentao Yang, Songxuan Lai, Zecheng Xie, and Lianwen Jin. DocKylin: A Large Multimodal Model for Visual Document Understanding with Efficient Visual Slimming. _arXiv: Comp. Res. Repository_ , abs/2406.19101, 2024. 7 

- [79] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding. _arXiv: Comp. Res. Repository_ , abs/2306.17107, 2023. 1, 2, 7 

- [80] Xu Zhong, Jianbin Tang, and Antonio Jimeno-Yepes. PubLayNet: Largest Dataset Ever for Document Layout Analysis. In _Proc. ICDAR_ , pages 1015–1022, 2019. 5 

- [81] Xu Zhong, Elaheh ShafieiBavani, and Antonio JimenoYepes. Image-Based Table Recognition: Data, Model, and Evaluation. _Proc. ECCV_ , 12366:564–580, 2020. 5 

12 

## **DocLayLLM: An Efficient Multi-modal Extension of Large Language Models for Text-rich Document Understanding** 

## Supplementary Material 

## **A. Pre-traing tasks** 

We adopt pre-training tasks as shown in Table A. These tasks facilitate the alignment of layout and visual features with the LLM’s feature space while enhancing the LLM’s understanding of document content. 

|**Task**|**Description**|
|---|---|
|Document Description|Provide a brief overview of the document.|
|Text and Box Reconstruction|Recover the coordinates of bounding boxes of|
||all the OCR text.|
|Layout Analysis|Determine the layout type (e.g., Title, Author,|
||Paragraph, etc) of a giving area or locate|
||specifc layout elements.|
|Table Analysis|Decode the structure of tables and identify the|
||positions of elements within.|
|Mask Language Model|Restore masked words in the provided OCR|
||text.|
|Mask Position Model|Reconstruct the box for text elements missing|
||the coordinates of the bounding box.|
|Geometric Analysis|Calculate distances or directions between two|
||specifed text elements.|



Table A. The training tasks during the pre-training stage. 

## **B. More Implementation Details** 

We implement our DocLayLLM using Llama2-7BChat [64] and Llama3-8B-Instruct [42]. The hyperparameters for both pre-training and supervised fine-tuning are detailed in Table B. As shown in the table, our DocLayLLM demonstrates efficiency, requiring fewer training resources while maintaining high performance. This underscores the method’s capability to deliver robust results without the need for extensive computational power, making it a resource-efficient solution for text-rich document understanding tasks. 

## **C. More Qualitative Examples** 

We also provide additional qualitative examples of our DocLayLLM. As shown in the comparison between DocLayLLM and the SOTA OCR-free method DocOwl 1.5 [17] in Figure Aa, our DocLayLLM demonstrates superior document understanding capabilities, delivering accurate answers in examples from InfoVQA and VisualMRC. Furthermore, in the DeepForm example, we observe that our design to integrate OCR information helps reduce the occurrence of hallucinated outputs compared to OCR-free methods. Moreover, in the examples from DocVQA and WTQ, DocLayLLM reliably exhibits robust table comprehension abilities. These results collectively highlight the 

|**Parameters**|**Pre-Training**|**Supervised Fine-Tuning**|
|---|---|---|
|**LoRA Rank**|64|64|
|**Batch Size**|512|64|
|**Max Length**|2560|2560|
|**Precision**|bf16|bf16|
|**Trainable**|170M/Llama2;|170M/Llama2;|
|**Parameters**|178M/Llama3|178M/Llama3|
|**Fixed**|6.7B/Llama2;|6.7B/Llama2;|
|**Parameters**|8.0B/Llama3|8.0B/Llama3|
|**Learning Rate**|1e-4|2e-5|
|**Weight Decay**|0.01|0.01|
|**Scheduler**|cosine|cosine|
|**Adam Betas**|[0.9, 0.999]|[0.9, 0.999]|
|**Adam Epsilon**|1e-8|1e-8|
|**Epoch**|1|3|



Table B. Hyper-parameters of DocLayLLM. 

effectiveness of our design in incorporating OCR information. 

Furthermore, we also present the results of whether Table-Structure-Aware CoT is used during the pre-training stage. As shown in Figure Ab, models incorporating CoT demonstrate a more comprehensive understanding of table structures, leading to more accurate and thorough answers in table-related downstream document understanding tasks. This validates the effectiveness of our proposed CoT pretraining approach. 

Additionally, we visualized the outputs with and without the use of CoT Annealing. As shown in the visualization of VisualMRC in Figure Ac, DocLayLLM employing CoT Annealing tends to provide more straightforward and accurate answers. This is particularly evident in yes-or-no questions, where the model without CoT Annealing often fails to directly respond with a clear ”yes” or ”no” but repeats the sentence in the document where the answer is located. In contrast, the model using CoT Annealing typically provides a direct answer first, followed by an explanation. These observations indicate that CoT Annealing enables the model to answer questions more directly, thereby enhancing its performance. 

13 

**==> picture [461 x 210] intentionally omitted <==**

**----- Start of picture text -----**<br>
Ground Truth :  To build a strong<br>coalition of digital freedom<br>advocates, with a specific focus on  Ground Truth :<br>Ground Truth :  24% international intellectual property  POL/MICHAELBLOOMBERG/PRE<br>initiatives that impact national  SIDENT/US/D<br>policies and laws and restrict users'<br>rights and impede access to<br>knowledge.<br>What percentage of  What is the goal of  What is the value<br>User men use LinkedIn? User this page?  User for the advertiser?<br>× 80% DocOwl 1.5 × questions you would be asked in The goal of this page is to help you understand the kind of your presentation. DocOwl 1.5 × POL/MICHAELBLOOMBERG/PRESIDENT/US/DEM DocOwl 1.5<br>Lo} Lo} LC}<br>InfoVQA🗸 24% DocLayLLM(Ours) VisualMRC🗸 digital freedom advocates.The goal of this page is to build a strong coalition of  DocLayLLM(Ours) DeepForm🗸 POL/MICHAELBLOOMBERG/PRESIDENT/US/D DocLayLLM(Ours)<br>|} [} |}<br>Ground Truth :  Liver Lobule Ground Truth :  10 Ground Truth : Acadia<br>fi<br>Which lobule in the  faculties can be listed How many nos. of  What is the first<br>User rats is studied?  User in part B in first column? User name listed?<br>Co) Co) fo)<br>× Peripotal lobule DocOwl 1.5 × 9 DocOwl 1.5 × Olympic DocOwl 1.5<br>LC} Lo} LS}<br>🗸 Liver Lobule DocLayLLM 🗸 10 DocLayLLM 🗸 Acadia DocLayLLM<br>DocVQA (Ours) DocVQA (Ours) WTQ (Ours)<br>|} LS |}<br>**----- End of picture text -----**<br>


- (a) Qualitative comparisons with DocOwl 1.5 [17] across various benchmarks. The document-oriented VQA tasks include InfoVQA [47], VisualMRC [59], and DocVQA [46]; the KIE task includes DeepForm [58]; and the Table Understanding task includes WTQ [51]. 

**==> picture [461 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
Ground Truth columns. :  15 rows and 3  Ground Truth Tunisia. :  Morocco, Turkey,  Culina, Elrich, Griffiths, Skoko, Zdrilic. Ground Truth :  Cahill, Colosimo,<br>How many rows and  Which countries did not  Players who scored<br>columns are present  win any bronze medals?  at most 1 total goal?<br>User in the table? User Directly answer. User Directly answer.<br>× 14 rows and 3 columns. w/o CoTPre-training ? Morocco, Turkey. w/o CoTPre-training ? Griffiths, Skoko, Zdrilic.Colosimo, Culina, Elrich,  w/o CoTPre-training<br>Lo} Step 1 :All the headers are:  LL} LCL}<br>Pre-trainingTable AnalysisTask 🗸 Final Answer: each row are: [“”, “No. of …”, …].[“”, “Number”, “Range”].Step 2: Column 1 of  15 rows and 3 columns. w/ CoTPre-training WTQ 🗸 Morocco, Turkey, Tunisia. w/ CoTPre-training WTQ 🗸 Cahill, Colosimo, Culina, Elrich, Griffiths, Skoko, Zdrilic. w/ CoTPre-training<br>|} [| |}<br>**----- End of picture text -----**<br>


- (b) Qualitative comparisons between the use and absence of CoT Pre-training. **w/o Pre-training** indicates the absence of CoT at the pre-training stage, while **w/ CoT Pre-training** denotes its application. **“?”** represents that the answer is ambiguous. 

**==> picture [450 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
Ground Truth :  No, Murmansk<br>Ground Truth :  Cuba harbour does not freeze in the  Ground Truth :  Yes, it was.<br>winter.<br>User country needs a visa?Which caribbean User harbour freeze in Does Murmansk the winter? User conquered by Was Sylhet Muslims?<br>× citizens of certain countries.Grenada requires a visa for  w/o CoTAnnealing ? White Sea, its harbor does Unlike Arkhangelsk on the not freeze in winter. w/o CoTAnnealing ? Muslims, it was ruled by local Before the conquest by the chieftains. w/o CoTAnnealing<br>Ll} Ll} L<br>VisualMRC🗸 Cuba. w/ CoTAnnealing VisualMRC🗸 No, Murmansk harbour does not freeze in the winter. w/ CoTAnnealing VisualMRC🗸 Yes, Sylhet was conquered by Muslims. w/ CoTAnnealing<br>|} | _<br>**----- End of picture text -----**<br>


- (c) Qualitative comparisons between the use and absence of CoT Annealing. **w/o CoT Annealing** indicates the absence of CoT Annealing, while **w/ CoT Annealing** denotes its application. **“?”** represents that the answer is ambiguous. 

Figure A. Further qualitative comparisons of DocLayLLM against the SOTA OCR-free method and under various settings. 

## **D. Input Length Analysis** 

In our ablation study, we evaluated the performance of different methods for incorporating OCR information. This section further examines the input length of OCR information under various approaches. The analysis was conducted using Llama3 [42] as the base model, with its tokenizer applied for tokenization. Table C presents a comparison of 

the average input length of OCR information across several benchmarks under two configurations: **(I)** encoding OCR bounding box coordinates as plain text, following the approach of ICL-D3IE [14], and **(II)** encoding OCR bounding box coordinates using a layout embedder _LE_ . 

The results clearly show that encoding with _LE_ significantly reduces the input length, thereby enhancing effi- 

14 

ciency during both training and inference. These findings underscore the efficiency of our proposed DocLayLLM. 

|**Input**|**Document-oriented VQA**|**Document-oriented VQA**|**KIE**||
|---|---|---|---|---|
|**Method**|DocVQA|VisualMRC|DeepForm|KLC|
|**(I)**|1571.80|6269.17|4952.87|457.58|
|**(II)**|**455.17**|**2095.35**|**1198.97**|**125.40**|



||||||ANLS_↑_|
|---|---|---|---|
|Official OCR|||86.52|
|Commercial Engine|||**87.52**|



Table D. Performances of DocLayLLM on DocVQA benchmark with different OCR results. 

Table C. The average input length of OCR information across various benchmarks, comparing different ways to input OCR bounding box coordinates. 

## **E. OCR Result Impacts** 

Since DocLayLLM requires OCR result input, we explored the impact of OCR quality on the performance of DocLayLLM. In the results presented in the main text, we used the official OCR results when evaluating on the DocVQA benchmark. To assess the model’s applicability in realworld scenarios, we employed a commercial OCR engine[2] to process DocVQA and used the recognized text for further testing DocLayLLM’s performance. The results in Table D suggest that the reported results in the main text have not fully reflected the potential of DocLayLLM. The model could achieve even better performance with realworld OCR results. 

Furthermore, as illustrated in Figure B, we observed that when OCR errors occur, DocLayLLM has the capability to correct these errors and produce the final correct answer. This further substantiates the robustness of DocLayLLM in real-world scenarios. 

**==> picture [122 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
OCR results:<br>[“ITC Limaited”,<br>“Notes to the Financial Statements”,<br>…]<br>Ground Truth :<br>ITC Limited<br>What is the name of<br>the company<br>User according to the logo?<br>🗸 ITC Limited<br>DocLayLLM<br>**----- End of picture text -----**<br>


Figure B. Illustration of DocLayLLM’s OCR error correction capability. 

> 2https://www.textin.com/ 

15 

