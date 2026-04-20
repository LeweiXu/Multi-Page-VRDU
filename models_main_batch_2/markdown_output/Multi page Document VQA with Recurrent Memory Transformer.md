## **Multi-page Document VQA with Recurrent Memory Transformer** 

Qi Dong[1] © , Lei Kang[1(][B][)] © , and Dimosthenis Karatzas[1,2] 

> 1 Computer Vision Center, Barcelona, Spain {qdong,lkang,dimos}@cvc.uab.cat 

> 2 Universitat Autònoma de Barcelona, Barcelona, Spain http://www.cvc.uab.es 

**Abstract.** Multi-page document Visual Question Answering (VQA) poses realistic challenges in the realm of document understanding due to its complexity and volume of information distributed across multiple pages. Current state-of-the-art methods often struggle to process lengthy documents, because they either exceed the model’s input token limits when treated as single-page document VQA problems, or compress pages into vectors that may omit crucial information. To our knowledge, our proposed method is the first to integrate recurrent memory mechanisms with the transformer architecture specialized for multi-page document VQA. Extensive experiments demonstrate that our proposed method achieves state-of-the-art performance while maintaining a manageable model size. 

**Keywords:** Document Visual Question Answering _·_ Multi-page Document VQA _·_ Recurrent Memory Transformer 

## **1 Introduction** 

Document Visual Question Answering (DocVQA) [15] has arisen as a new paradigm for document image analysis and recognition, harnessing the power of Large Language Models for document understanding. On one hand, DocVQA provides a unifying framework for multi-task learning, as many document analysis tasks can in principle be cast into natural language prompts and responses [7,14]. At the same time, documents are large-scale, high-resolution images, and typically comprise multiple pages that need to be processed and reasoned upon jointly. This tests the input limits of LLM architectures, which need to resort to different techniques to cope with this amount of information. 

Multi-page DocVQA was introduced by Tito _et al._ [19], and was quickly followed up by new methods [5,12]. A direct approach is to first employ single-page DocVQA to extract possible answers in each page, before selecting the response with the highest confidence [12]. Alternatively, Tito _et al._ [19] employ an hierarchical approach, where special tokens extracted from each page a fused at a 

> _⃝_ c The Author(s), under exclusive license to Springer Nature Switzerland AG 2024 G. Sfikas and G. Retsinas (Eds.): DAS 2024, LNCS 14994, pp. 57–70, 2024. https://doi.org/10.1007/978-3-031-70442-0_4 

58 Q. Dong et al. 

later stage to produce the question. In both cases, single page DocVQA is applied independently on each page, before fusing the information extracted. 

Most promising methods to date [5] employ iterative local (page) and global (document) attention layers, where page-level information extraction and document-level reasoning are interleaved, to efficiently extract information taking the context from other pages into account. Although they scale up well, and are shown to perform substantially better with higher number of parameters, they still perform comparably to previous architectures at same-size models. 

In this work, we take a different approach to the problem, and use a Recurrent Memory Transformer for multi-page document VQA. In our proposed approach, pages are processed in sequence, while information is carried from one page to the next, and enriched with new content. This approach allows context from previous pages to be taken into account and updated iteratively, while keeping the number of parameters low. 

We show that the proposed model yields better performance compared to models of the same size, while keeping the parameter count low, making it a viable choice for Document Analysis Systems. 

## **2 Previous Work** 

Most existing language models, such as BERT [9] and LayoutLMv3 [11], indeed face limitations when processing multi-page documents at one time. This limitation primarily stems from the model architecture and input constraints. BERT, for example, is designed to handle input sequences with a maximum length of 512 tokens, which is a significant limitation for processing longer texts found in multi-page documents. The retrieval-based approach [12] utilizes a singlepage document VQA model for inferring answers from multi-page documents by treating each page as an independent document. It conducts separate inference on each page to determine the correlation score between the question and each page, then selects the page with the highest score as the evidence page. Subsequently, this transforms into a typical single-page document VQA task. The same retrieval-based method is the Unified Retrieval and Question Answering (URA) [22] model and PDFTriage-augmented models [18]. ScreenAI [3] transforms the multi-page document VQA task into a series of single-page document VQA tasks. First, the document and its accompanying question are divided into individual page-question pairs, with each page evaluated separately to identify the most relevant one. Second, the answers from each single-page visual question answering (VQA) task are assessed, and the final answer is selected based on the highest confidence score across all pages. These methods circumvent the multipage document input limitation but does not allow for cross-page reasoning, operating under the assumption that the answer resides entirely on a single page. The problem of input limitations can be mitigated by using large-scale models to handle exceptionally long inputs, as demonstrated by LATIN-Prompt [20]. The authors proposed LATIN-Prompt, which includes layout-aware document content and task-aware instructions. They used spaces and line breaks to restore 

Multi-page Document VQA with Recurrent Memory Transformer 

59 

layout information between text segments acquired by OCR tools. Task-aware instructions ensure generated answers meet formatting requirements. Additionally, LATIN-Tuning was introduced to improve the performance of smaller models like Alpaca by enhancing their ability to understand layout information. 

Hi-VT5 [19] introduces a hierarchical transformer structure that employs special [PAGE] tokens to direct the encoder to summarize each document page based on the posed question. Each page is concatenated with the question and processed individually, that encodes the summary of relevant information into a page feature corresponding to the [PAGE] token input. These page features are then concatenated and input into the decoder to produce the answer. This approach effectively addresses the limitations of language models, but each page is processed independently, descontextualised from the other pages, until the decoder step. 

GRAM [5] addresses the above limitations by integrating local (page-level) and global (document-level) reasoning throughout the encoding process. This approach introduces learnable document-level tokens and a bias adaptation mechanism to enhance the cross-page information flow, which significantly improve the performance on multi-page document VQA tasks. However, GRAM employs DocFormer [1] as its backbone model, which is proprietary and does not have publicly available weights. DocFormer represents a notable development in this field by introducing a novel multi-modal self-attention mechanism that integrates text, vision, and spatial features within a single encoder-only transformer architecture. Unlike previous models, DocFormer does not rely on pretrained object detection networks for visual feature extraction, opting instead to use ResNet50 [10] for visual embeddings. This approach not only reduces memory usage but also facilitates better feature correlation across modalities. GRAM yields 29 _._ 5% better ANLS performance than Hi-VT5, at a cost of 1 _._ 7 time more parameters with 859M compared to the 316M of Hi-VT5. The small version of GRAM, which is comparable with the size of Hi-VT5 actually yields 18 _._ 8% better ANLS performance than Hi-VT5. In addition, the backbone is not publicly available. 

Transformer-XL [8] introduces a segment-level recurrent mechanism and a new positional encoding scheme that can capture dependencies beyond a fixed length while maintaining temporal consistency. Building on this foundation, Bulatov _et al._ proposed the Recurrent Memory Transformer (RMT) [6], which extends Transformer-XL by adding read and write tokens for memory operations. RMT caches the hidden states from the previous segment for each transformer layer _n_ . The input to the _n_ -th layer includes the last _m_ states from the cache and the output of the previous transformer layer for the current segment. The authors later expanded the Recurrent Memory Transformer to both encoderonly and decoder-only models. For encoder-only models like BERT, similar to Transformer-XL, the cached memory layer implements the memory function. For decoder-only models, such as GPT-2 [16], the authors designed a special wrapper that modifies the model’s input and output, using special tokens and gradient back-propagation to implement the recurrent memory. 

60 Q. Dong et al. 

Drawing inspiration from the memory tokens used in RMT, we introduce a recurrent memory transformer within an encoder-decoder framework. Unlike RMT, which uses a specialized wrapper to manage memory, we employ specialized memory tokens to efficiently implement recurrent memory effectively. As far as we know, we are the first to introduce recurrent method for multi-page document VQA tasks. The comparison of the main categories of the state of the art is shown in Table 1. 

**Table 1.** State of the art for multi-page document VQA tasks. 

|Category|Method|Year|
|---|---|---|
|Single-page adapted method|Retrieval-based [12]<br>ScreenAI [3]<br>LATIN-Prompt [20]|2024<br>2024<br>2023|
|Hierarchical method|Hi-VT5 [19]|2023|
|Local-global method|GRAM [5]|2024|
|Recurrent method|**Our proposed**|2024|



## **3 Proposed Method** 

In our model, the structure design is based on the T5 [17] model, but adapted to handle the visual question answering task of multipage document. T5 is a general text-to-text framework originally designed to handle various NLP tasks by unifying all text-based input and output tasks into a text-to-text format. The T5 excels in natural language processing, especially in understanding and generation tasks. 

The encoder’s task is to extract the most relevant information from the current page to the question. The purpose of this step is to extract and refine the high-level feature representation of the document and turn it into a “memory”. This memory is a highly abstract representation of the information relevant to the question on the current page. We adopt the concept of ‘memory’ cells from Recurrent Neural Networks (RNN) to selectively retain or forget information over time. Additionally, we integrate a T5 model at each timestep for effective high-resolution document page processing. Figure 1 shows an overview of the model. The memory not only retains the key information of the current page, but it is also passed to the encoder of the next page. When the next page of the document is processed, these memories are fed into the encoder along with the content of the new page. This recurrent mechanism allows the model to carry contextual information from the previous page as it processes each page. This allows the model to better understand the coherence and contextual relevance of the entire document. Finally, memory cells from all pages are concatenated and fed into the decoder to generate the correct answer. At each page (timestep), the encoder processes the previous memory cell, textual question tokens, and visual patches from the current page. 

Multi-page Document VQA with Recurrent Memory Transformer 

61 

**Text and Image Representation.** We summarize the OCR _ok_ and the spatial embedding _sk_ to get the total text representation _tk_ = _ok_ + _sk_ . At the same time, we use DiT [13] to extract the features of the document image and represent it as a set of patch embeddings _vk_ . 

**Recurrent Memory Transformer.** Inspired by Hi-VT5 and RMT, we introduce m learnable Memory cells to store the current summary information. Then we use the Memory tokens of the current page and the page information of the next page together as the encoder input. This recurrent mechanism allows the model to carry contextual information from the previous page when processing each page. It’s better understanding the coherence and contextual relevance of the entire document. 

**Fig. 1.** Architecture of the RM-T5 model. First, we initialize the memory tokens, and then input the memory tokens and the current page into the encoder. After that we can get the memory containing the summary of the current page required to answer the question. Next, we take the memory tokens of the current page and the content of the next page as the input of the next page encoder. Finally, in this way we get the memory tokens containing all the context information. And we connect them together as the overall representation of the document and input them into the decoder. At the same time, we use the answer page prediction module to predict the page where the answer is located, providing an explainability metric for the answers generated by the model. 

Assume there are _N_ documents in the dataset, with each document comprising a variable number of pages, _M_ . The textual question for each document is denoted by _tk ∈ T_ , where _T_ = _{t_ 1 _, t_ 2 _, ..., tN }_ represents the set of questions for all documents. For _k_ -th document, the _i_ -th visual page image is represented by 

62 Q. Dong et al. 

_vk[i][∈][V]_[ ,][where] _[V]_[=] _[{][v]_ 1[1] _[, v]_ 1[2] _[, ..., v] k[i][, ..., v] N[M][}]_[encompasses][all][visual][page][images] in the collection and _i ∈{_ 1 _,_ 2 _, ..., M }_ represents the page number within each document. Thus, the previous memory cell _mi−_ 1, the textual question _tk_ , and the current page image _vk[i]_[are][utilized][as][the][input][of][the][encoder] _[E]_[,][so][that] the encoder output _Ek[i]_[at] _[i]_[-th][page][of] _[k]_[-th][document][can][be][obtained.][Then][we] update the memory state: 

**==> picture [221 x 13] intentionally omitted <==**

**==> picture [204 x 13] intentionally omitted <==**

where _L_ is a hyper-parameter used to select a subset from the encoded features _Ek[i]_[.] 

Finally, all memory tags for all pages are concatenated to create an overall representation of the document _M[′]_ . 

In a multipage document, information may be spread across multiple pages. Key information or answer clues may be scattered across different pages. By accumulating key memories for each page, the model can track the entire document content, ensuring that information is not lost when switching pages. In this way, the decoder can consider the global information of the document when generating the final answer, thereby improving the accuracy and relevance of the answer. 

**Pretraining.** We introduce a curriculum learning method to pretrain the model. At the beginning of training, a single page of documents is used as input for finetuning, and one page of input is added after the model converges. And so on until the document length specified by us is reached. 

## **4 Results** 

## **4.1 Dataset and Metrics** 

The MP-DocVQA dataset is the first multipage document VQA dataset. The dataset contains 5,928 documents, with a total of 60,884 pages of document images, and 46176 questions extracted. Documents in the dataset have a maximum of 20 pages and a maximum of 42,313 recognized OCR words. The annotations in the dataset are OCR annotations extracted by Amazon Textract2 for all document images, as shown in Fig. 2. 

To assess the performance of our proposed model, we employ the standard evaluation metrics: accuracy (ACC) and Average Normalized Levenshtein Similarity (ANLS) from the DocVQA benchmark. For evaluating page predictions, we utilize the metric of page accuracy (Page ACC). 

Multi-page Document VQA with Recurrent Memory Transformer 

63 

**Fig. 2.** Original image and OCR annotation of document image. 

## **4.2 Implementation Details** 

In our experiments, Hugging Face T5 base model pretrained weights were used for initialization. The learning rate is set to 2e-4, warmed up for 1,000 steps, and then linearly decayed. We use curriculum learning methods for training. Let the RM-T5 model start training from one page and increase the number of pages after convergence. Due to limited hardware resources for training, it is impossible for us to train with a complete 20-page document, so we increase it to 3 pages at most. Then the parameters of the encoder are fixed, and the decoder and page selector are fine-tuned with the full 20-page documentation. We trained for 10 epochs using a single NVIDIA A40 GPU with a batch size of 4 and a maximum input of 512 tokens per page. 

We fixed the number of memory tokens to 100 in our experiments. From Table 3 we can see the impact of memory tokens of different lengths on performance. The model performs best with 100 memory tokens, but performance decreases when this number is exceeded. 

Table 2 shows the performance of our method on the MP-DocVQA dataset. We compare the performance of three encoder-only models Longformer, Big Bird, and LayoutLMv3 in maximum confidence mode. Input the multipage document into the model as a single page, get the logits of the model for each page as the confidence score, sort the answers generated by each page, and select the answer with the highest score as the final output answer. However, treating a multipage document as a single page for reasoning only considers the content of the current page and lacks the contextual relevance of the page. The best 

64 Q. Dong et al. 

**Table 2.** Quantitative results. We show the results of RM-T5 and other methods on the MP-DocVQA dataset. All the results are in %. Parameters are model parameters, the evaluation index ACC is the answer prediction accuracy, ANLS is Average Normalized Levenshtein Similarity, and Page ACC is the answer page prediction accuracy. 

|Model|Parameters|ACC|ANLS|Page ACC|
|---|---|---|---|---|
|Longformer [4]<br>BigBird [21]<br>LayoutLMv3 [11]<br>T5 [17]<br>Hi-VT5 [19]<br>Retrieval-based [12]<br>DocFormerv2 [2]<br>GRAM [5]|148M<br>131M<br>125M<br>223M<br>316M<br>273M<br>257M<br>281M|45.87<br>49.57<br>42.7<br>41.8<br>48.28<br>/<br>/<br>/|55.06<br>58.54<br>55.13<br>50.47<br>62.01<br>61.99<br>69.97<br>73.68|70.37<br>72.27<br>74.02<br>/<br>79.23<br>81.55<br>/<br>19.98|
|Ours|312M|54.96|64.01|88.32|



performing model is Big Bird ANLS which is 58.54% and our model increases it by 5.47%. The backbone of our approach is T5, so we added the results of T5’s multipage VQA. We compare T5’s performance when connecting the context of all pages. Because it uses softmax in its vocabulary, it cannot use maximum confidence to rank answers. Compared to T5, our model increased by 13.54%. 

The Hi-VT5 model is the first dedicated VQA model for multipage documents. Its performance is greatly improved compared with the proposed NLP model. Special [PAGE] tokens realize page compression, but Hi-VT5 utilized a special [PAGE] token to guide the extraction of page-specific information separately across all the pages in a document. Our method employs a memory recurrent mechanism to consistently transmit the content of each page sequentially, ultimately linking all memories. This process ensures that the compressed information encompasses both individual page content and the global context of a document. RM-T5 increased by 2.00% compared to Hi-VT5. The performance of our page recognition has also improved significantly by 9.09%. 

GRAM is currently the best-performing multipage DocVQA model, achieving an ANLS of 73 _._ 68%. Its backbone is DocFormerv2, and the experimental result of multipage DocVQA is 69 _._ 67%. The most specifically designed model for document understanding, DocFormerv2 shows good performance. We attribute the primary performance disparity between the RM-T5 model and GRAM to differences in their underlying architectures. Since DocFormerv2 is proprietary, lacking publicly available pretraining weights and model structure, we are unable to use it as our backbone. 

The picture shows our success and failure cases. Figure 3 shows that the RMT5 prediction page is accurate and the predicted answers are accurate. The model in Fig. 4 predicts answers accurately and predicts page errors. For multi-page documents, the answer may be available on more than one page. While predicting page errors, the model can also get the correct answer from information from 

Multi-page Document VQA with Recurrent Memory Transformer 

65 

other pages. Figure 5 is a failure case, predicting page errors and predicting answer page errors. 

**Fig. 3.** The predicted page is accurate and the predicted answer is accurate. QuestionId: 49172 Question: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)? Answer: lee a. waller Pre answer: lee a. waller ✓ Answer page idx: 2 Pre page idx: 2 ✓ 

Figure 6 is a qualitative comparison between RM-T5 and Hi-VT5 showing that our recurrent memory transformer enhances reasoning capabilities, especially when queries require multiple pages of context. 

**Table 3.** Experimental results on the impact of different memory token numbers on model performance. 

|Mem. Tokens|ANLS|Page ACC|
|---|---|---|
|20|54.02|87.70|
|40|62.90|87.58|
|100|64.01|88.32|
|200|54.39|88.05|



Table 3 shows the impact of memory tokens of different lengths on performance. In order to find the appropriate length of memory tokens, we trained models with different lengths of memory tokens. As can be seen from Table 2, as the memory token length increases, the performance of the model increases significantly. But when the length is greater than 100, the performance drops instead. Therefore, we fixed the number of memory tokens at 100. 

66 Q. Dong et al. 

**Fig. 4.** Predict answers accurately, predict page errors. QuestionId: 57368 Question: How many nomination committee meetings has Y. C. Deveshwar attended? Answer: 2 Pre answer: 2 ✓ Answer page idx: 0 Pre page idx: 2 ✕ 

**Fig. 5.** Predict answers errors, predict page errors. QuestionId: 16447 Question: What percentage of non-smokers feel the need to restore romance and mystery to modern life? Answer: 57 Pre answer: 61 ✕ Answer page idx: 0 Pre page idx: 2 ✕ 

**Fig. 6.** Qualitative comparison between Hi-VT5 and RM-T5. Correct and incorrect predictions are in blue and red, respectively. (Color figure online) 

Multi-page Document VQA with Recurrent Memory Transformer 

67 

Because our training resources and time are limited, we train our model on shortened documents. We gradually train our model using different number of pages. As shown in the Table 4, when the RM-T5 model is trained on a single-page document, the ANLS results of evaluating the single-page document indicate that our model can seamlessly perform the single-page DocVQA task. Gradually increasing the document length for training, ANLS evaluates documents with a maximum of 20 pages. But the training length is not as long as possible, so we decided to use only 3 pages for training, and finally fine-tuned on the 20-page complete document. 

**Table 4.** Curriculum learning training pages. We use documents of different lengths to train the model, and Page ANLS represents the results of evaluating documents of limited length under limited training length. ANLS is the result of evaluating the complete document length. 

|Num Page|Page ANLS|ANLS|
|---|---|---|
|1|61.24|48.68|
|2|62.45|56.84|
|3|64.46|62.79|
|10|61.75|61.54|



## **5 Ablation** 

We conduct an ablation study of our method, using the MP-DocVQA dataset to evaluate the influence of each component. The results are recorded in Table 5 to facilitate analysis and comparison. 

**Pretraining for Curriculum Learning.** Curriculum learning is a strategy that mimics human learning patterns. The curriculum learning is to let the model start with processing simple single page documents and gradually increase the number of pages. This progressive learning approach helps the model gradually adapt to more complex inputs while learning how to effectively maintain and utilize memory across multipage documents. In our experiments, if this pretraining step is removed, that is, fine-tuning is performed directly on the 3-page document, from the second row in the Table 4 we can see that the ANLS metric of the model drops significantly. This suggests that curriculum learning plays a critical role in models understanding and remembering document content, especially when processing complex queries involving long documents. 

68 Q. Dong et al. 

**The Memory Effect.** In our model design, memory is the mechanism that retains key information extracted from each page. If the model only uses the memory of the last page as input to the decoder, its performance will drop significantly, as shown in row 3 of the Table 5. The problem with this design is that a single memory may not be enough to contain the key information for the entire document, especially if the answer is not on the last page of the document. In the process of document processing, the information on each page is extremely important, and some details may be lost or forgotten in the process of memory passing down. Therefore, relying on the memory of a single page to answer a question limits the model’s ability to take advantage of all the information provided by the document. 

**Page Prediction Module.** We also explore the influence of page prediction module on model performance. As shown in the last row of Table 5, although from the overall results, the performance improvement of the page prediction model is not as significant as other components. The results still show that multi task training is beneficial to the model. The page prediction task requires the model to be able to identify the pages most relevant to the question, which not only improves the accuracy of the question answer, but also helps the model better understand the document structure and content layout. Multi task training allows the model to learn to handle question and answer tasks while also enhancing the overall understanding of the document. These tasks are complementary. 

**Table 5.** Ablation experiments. Effect of different components of the RM-T5 model on performance. It includes curriculum learning pretraining (Pretrain), different memory cell (Last Memory) and page prediction module (Page Prediction). 

|Model|ACC|ANLS|Page ACC|
|---|---|---|---|
|RM-T5|54.96|64.01|88.32|
|_−_Pretrain|46.79|58.57|85.70|
|_−_Last Memory|36.84|49.41|85.04|
|_−_Page Prediction|53.93|63.04|00.00|



## **6 Conclusions** 

In this paper, we present a novel recurrent memory transformer for multipage document VQA, combining sequential processing and memory retention to enhance cross-page reasoning. This pioneering method, which amalgamates recurrent memory mechanisms with transformer architecture, establishes a new benchmark for multi-page document VQA tasks. Our work is poised to inspire further innovations within the document understanding community. 

Multi-page Document VQA with Recurrent Memory Transformer 

69 

**Acknowledgments.** Chinese Scholarship Council (CSC) No.202208410099, European Lighthouse on Safe and Secure AI (ELSA) from the European Union’s Horizon Europe programme under grant agreement No 101070617, Beatriu de Pinós del Departament de Recerca i Universitats de la Generalitat de Catalunya (2022 BP 00256). 

## **References** 

1. Appalaraju, S., Jasani, B., Kota, B.U., Xie, Y., Manmatha, R.: Docformer: end-toend transformer for document understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 993–1003 (2021) 

2. Appalaraju, S., Tang, P., Dong, Q., Sankaran, N., Zhou, Y., Manmatha, R.: Docformerv2: local features for document understanding. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 709–718 (2024) 

3. Baechler, G., et al.: ScreenAI: a vision-language model for UI and infographics understanding. arXiv preprint arXiv:2402.04615 (2024) 

4. Beltagy, I., Peters, M.E., Cohan, A.: Longformer: the long-document transformer (2020) 

5. Blau, T., et al.: Gram: global reasoning for multi-page VQA. arXiv preprint arXiv:2401.03411 (2024) 

6. Bulatov, A., Kuratov, Y., Burtsev, M.: Recurrent memory transformer. In: Advances in Neural Information Processing Systems, vol. 35, pp. 11079–11091 (2022) 

7. Cheng, H., et al.: M6Doc: a large-scale multi-format, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15138–15147 (2023) 

8. Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q.V., Salakhutdinov, R.: Transformer-XL: attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860 (2019) 

9. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018) 

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778 (2016) 

11. Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F.: LayoutLMv3: pre-training for document AI with unified text and image masking. In: Proceedings of the 30th ACM International Conference on Multimedia, pp. 4083–4091 (2022) 

12. Kang, L., Tito, R., Valveny, E., Karatzas, D.: Multi-page document visual question answering using self-attention scoring mechanism. arXiv preprint arXiv:2404.19024 (2024) 

13. Li, J., Xu, Y., Lv, T., Cui, L., Zhang, C., Wei, F.: DIT: self-supervised pre-training for document image transformer. In: Proceedings of the 30th ACM International Conference on Multimedia, pp. 3530–3539 (2022) 

14. Luo, C., Shen, Y., Zhu, Z., Zheng, Q., Yu, Z., Yao, C.: LayoutLLM: layout instruction tuning with large language models for document understanding. arXiv preprint arXiv:2404.05225 (2024) 

15. Mathew, M., Karatzas, D., Jawahar, C.: DocVQA: a dataset for VQA on document images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2200–2209 (2021) 

70 Q. Dong et al. 

16. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI Blog **1** (8), 9 (2019) 

17. Raffel, C., et al.: Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. **21** (140), 1–67 (2020) 

18. Saad-Falcon, J., et al.: PDFTriage: question answering over long, structured documents (2023) 

19. Tito, R., Karatzas, D., Valveny, E.: Hierarchical multimodal transformers for multipage DocVQA. Pattern Recogn. **144** , 109834 (2023) 

20. Wang, W., Li, Y., Ou, Y., Zhang, Y.: Layout and task aware instruction prompt for zero-shot document image question answering. arXiv preprint arXiv:2306.00526 (2023) 

21. Zaheer, M., et al.: Big bird: transformers for longer sequences. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems, vol. 33, pp. 17283–17297. Curran Associates, Inc. (2020) 

22. Zhang, L., Hu, A., Zhang, J., Hu, S., Jin, Q.: MPMQA: multimodal question answering on product manuals (2023) 

