_2025-11-17_ 

# **DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding** 

**Dawei Zhu**[1 2 *] **, Rui Meng**[2] **, Jiefeng Chen**[2] **, Sujian Li**[1] **, Tomas Pfister**[2] **and Jinsung Yoon**[2] 1School of Computer Science, Peking University, 2Google Cloud AI Research **`https://dwzhu-pku.github.io/DocLens/`** 

**Comprehending long visual documents, where information is distributed across extensive pages of text and visual elements, is a critical but challenging task for modern Vision-Language Models (VLMs). Existing approaches falter on a fundamental challenge: evidence localization. They struggle to retrieve relevant pages and overlook fine-grained details within visual elements, leading to limited performance and model hallucination. To address this, we propose DocLens, a tool-augmented multi-agent framework that effectively “zooms in” on evidence like a lens. It first navigates from the full document to specific visual elements on relevant pages, then employs a sampling-adjudication mechanism to generate a single, reliable answer. Paired with Gemini-2.5-Pro, DocLens achieves state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V, surpassing even human experts. The framework’s superiority is particularly evident on vision-centric and unanswerable queries, demonstrating the power of its enhanced localization capabilities.** 

Figure 1 | Workflow and performance of our proposed method, DocLens. **(a)** The workflow grounds its answer by navigating from the full document to visual elements (e.g., Text, Chart) within relevant pages. **(b)** It yields great improvement on MMLongBench-Doc, specifically for understanding visual elements and reducing hallucination. 

## **1. Introduction** 

A vast repository of human knowledge is encapsulated in long visual documents such as financial reports, academic papers, and technical manuals (Liu et al., 2025). With information synthesized from various textual and visual elements (tables, charts, figures) distributed throughout the context, these long visual documents are formidably challenging to decipher, even for the most advanced Vision- 

_Corresponding author(s): lisujian@pku.edu.cn, jinsungyoon@google.com_ * This work was done while Dawei was a student researcher at Google Cloud AI Research. 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

Language Models (VLMs) (ClaudeTeam, 2025; Comanici et al., 2025; Guo et al., 2025; OpenAITeam, 2025; QwenTeam, 2025; Team et al., 2025). 

This challenge stems from a fundamental problem: evidence localization. Existing efforts to localizing evidence from long visual documents primarily operate at the page level, either feeding page screenshots to long-context VLMs (Ma et al., 2024b) or employing vector-based retrieval methods (Cho et al., 2024; Han et al., 2025). However, we observe that both approaches perform poorly in recalling evidence pages. On MMLongBench-Doc (Ma et al., 2024b), Gemini-2.5-Pro (Comanici et al., 2025) only recalls 68% of evidence pages, while vector-based methods using ColBERT (Santhanam et al., 2021) and ColPali (Faysse et al., 2024) achieve merely 55.3% Recall@10. This fundamental failure prevents models from producing accurate answers. Moreover, even on the correct page, crucial details within visual elements (e.g., charts, tables) remain obscured in a full-page view, akin to reading a map without a magnifying glass. This dual-level failure in evidence localization—at both the page and element scale—directly fuels model hallucination, causing models to invent responses for over half of unanswerable queries rather than admitting uncertainty on MMLongBench-Doc (Figure 1b). 

In this paper, we propose DocLens, a multi-agent framework that overcomes these challenges by strategically leveraging document-parsing tools. Our core component is the _Lens Module_ , which zooms into long visual documents like a lens to perform fine-grained evidence localization (Figure 1a). It includes a _Page Navigator_ agent and an _Element Localizer_ agent. The former uses OCR tools to augment VLMs for page-level retrieval, drastically improving recall of evidence pages; the latter employs layout detection and cropping tools to locate visual elements on these retrieved pages for detailed inspection. Following the Lens Module, the Reasoning Module synthesizes the extracted evidence—including page screenshots, text, and cropped visual elements—to formulate a final answer. To ensure both accuracy and reliability, this module employs a “sampling-adjudication” process that first proposes a set of potential answers using an _Answer Sampler_ agent, and then critically assesses them using an _Adjudicator_ agent to select the best candidate. 

We evaluate DocLens on two challenging benchmarks, MMLongBench-Doc (Ma et al., 2024b) and FinRAGBench-V (Zhao et al., 2025). Our method achieves state-of-the-art performance, significantly reduces hallucination, and for the first time surpasses human experts. This breakthrough is driven by the efficacy of our core components: further analysis reveals that our Page Navigator achieves near-perfect evidence page recall (97.3%), while the Element Localizer dramatically enhances the comprehension of fine-grained visual details. Our main contributions are threefold: 

- A novel, tool-augmented Lens Module that achieves near-perfect page recall and enables fine-grained inspection of visual elements, effectively solving evidence localization. 

- A sampling-adjudication mechanism within the Reasoning Module that effectively mitigates hallucination and improves answer reliability. 

- The establishment of a new state-of-the-art on MMLongBench-Doc and FinRAGBench-V, and for the first time, surpassing human experts. 

## **2. Problem Formulation** 

We address the challenge of question answering over long visual documents. A document is a sequence of pages, D = { _𝑃𝑖_ } _𝑖[𝑁]_ =1[, where each page] _[𝑃][𝑖]_[is a screenshot image.][From each page, we can extract text] _𝑇𝑖_ and a set of visual elements V _𝑖_ (e.g., tables, figures). Given a question _𝑄_ , the goal is to generate an accurate answer _𝐴_ that is grounded in a specific set of evidence pages E ⊆D. This task can be abstractly formulated as learning a function _𝑓_ that maps the document and question to an answer: 

(1) 

_𝐴_ = _𝑓_ (D _, 𝑄_ ) _,_ 

2 

~~a~~ DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

Figure 2 | Overall workflow of DocLens. Given a long visual ducument and a corresponding question, we first apply a _Lens Module_ to retrieve relevant pages and locate relevant visual&textual elements within these pages. We then use a _Reasoning Module_ to do in-depth analysis of these elements and provide an accurate answer. 

However, the sheer volume of information in a long document makes a direct mapping challenging to construct. We contend that a more principled approach is to decompose the problem into two stages: First, identifying a concise set of relevant evidence from the vast document, and second, generating the answer based on this evidence. 

To formalize this, we factorize the function _𝑓_ into two components. First, an extraction function _𝑓𝑒𝑥𝑡𝑟𝑎𝑐𝑡_ reads through the document to identify a concise evidence set S relevant to the question: 

**==> picture [279 x 11] intentionally omitted <==**

This evidence set S contains the necessary pages from D and the visual and textual elements within these pages. Second, an answer generation function _𝑓𝑔𝑒𝑛𝑒𝑟𝑎𝑡𝑒_ infers the final answer exclusively from this condensed evidence: 

**==> picture [279 x 13] intentionally omitted <==**

We therefore model the composite function as: 

**==> picture [306 x 13] intentionally omitted <==**

The goal is to design and optimize both _𝑓𝑒𝑥𝑡𝑟𝑎𝑐𝑡_ and _𝑓𝑔𝑒𝑛𝑒𝑟𝑎𝑡𝑒_ to maximize the accuracy of the predicted answer _𝐴_ w.r.t. the ground-truth answer _𝐴_[∗] . 

## **3. DocLens Framework** 

Figure 2 illustrates the overall workflow of DocLens. Our proposed framework consists of two primary components: a Lens Module and a Reasoning Module. Given a long visual document and an associated question, the Lens Module ( _𝑓_ extract) is responsible for identifying relevant pages and the key elements within them. Subsequently, the Reasoning Module ( _𝑓_ generate) conducts an in-depth analysis of this evidence to generate a precise answer. The prompt templates for all agents and the pseudocode for the entire workflow are presented in Appendices A and C, respectively. 

## **3.1. Lens Module** 

**Page Navigator.** The Lens Module begins with the _Page Navigator_ to identify a predicted set of evidence pages, Epred, from the full document D = { _𝑃𝑖_ } _𝑖[𝑁]_ =1[.][First, it uses an OCR tool to extract the] 

3 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

text _𝑇𝑖_ from every page _𝑃𝑖_ ∈D. 

**==> picture [311 x 11] intentionally omitted <==**

To locate potential evidence pages E ⊆D, the Page Navigator then prompts an LLM with the question _𝑄_ , all page screenshots and their OCR text (as interleaved input). To ensure comprehensive coverage, this process is repeated _𝑇𝑒_ times with a temperature _𝜏>_ 0. Each sampling iteration _𝑗_ generates a candidate page set E[(] _[ 𝑗]_[)] : 

**==> picture [315 x 15] intentionally omitted <==**

The final set of predicted pages is the union of all pages identified across these iterations: 

**==> picture [279 x 21] intentionally omitted <==**

In practice, an LLM’s finite context window may prevent processing all _𝑁_ pages simultaneously. In such cases, we divide pages into chunks, process them in parallel, and merge the resulting Epred sets. 

**Element Localizer.** Given the set of predicted pages Epred identified by the _Page Navigator_ , the _Element Localizer_ enriches this set by parsing detailed visual and textual elements. 

For each page _𝑃𝑘_ ∈Epred, its corresponding textual content _𝑇𝑘_ is available from the prior step. Concurrently, a layout detection tool identifies the bounding boxes of key visual elements (such as figures, charts, and tables). These elements are then cropped from the page to form a set of focused visual inputs, denoted as V _𝑘_ : 

**==> picture [351 x 11] intentionally omitted <==**

where _𝑏𝑏𝑜𝑥_ is the bounding box of each visual element. Then with all predicted evidence pages, we construct the full evidence set S by collecting tuples of the page screenshot ( _𝑃𝑘_ ), its extracted text ( _𝑇𝑘_ ), and its cropped visual elements (V _𝑘_ ): 

**==> picture [305 x 13] intentionally omitted <==**

## **3.2. Reasoning Module** 

**Answer Sampler.** The Answer Sampler agent receives the collection of evidence S extracted by the Lens Module. It then integrates all this information to generate a reasoning process _𝑅_ (e.g., a chain-of-thought trace) and a corresponding answer _𝐴_ : 

**==> picture [293 x 13] intentionally omitted <==**

To generate a diverse set of candidate answers (Wang et al., 2023), we perform this reasoning process _𝑇𝑎_ times. The diversity is achieved by strategy with a temperature _𝜏>_ 0. This encourages the model to explore different reasoning paths and wording, yielding _𝑇𝑎_ distinct reasoning-answer pairs: { _𝑅𝑖, 𝐴𝑖_ } _[𝑇] 𝑖_ = _[𝑎]_ 1[.] 

**Adjudicator.** The final step is managed by the Adjudicator, whose goal is to synthesize the best answer from the _𝑇𝑎_ candidate answers. It carefully analyzes the reasoning path _𝑅𝑖_ of each candidate and cross-validates the different approaches to identify the most consistent and logical conclusion, which is then presented as the final answer _𝐴 𝑓𝑖𝑛𝑎𝑙_ : 

**==> picture [309 x 15] intentionally omitted <==**

4 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## **4. Experiments** 

In this section, we first introduce our experimental setup (§ 4.1), including benchmarks, metrics, tested models and baseline methods. We then demonstrate the overall effectiveness of our DocLens on the two selected benchmarks (§ 4.2 & § 4.3). For implementation details, please refer to Appendix C. 

## **4.1. Experimental Setup** 

**Benchmarks and Metrics.** We evaluate our method on two challenging benchmarks: MMLongBenchDoc (Ma et al., 2024b) and FinRAGBench-V (Zhao et al., 2025). MMLongBench-Doc tests reasoning over lengthy, multi-domain documents (avg. 49.4 pages) that require integrating scattered information across diverse modalities. Crucially, its dedicated “Unanswerable” subset directly evaluates our model’s ability to mitigate hallucination. Performance of human experts on this benchmark is reported as 65.8. FinRAGBench-V is vital for our analysis due to two unique features: its use of documents with dense, newspaper-like layouts (See Figure 5), and its support for evaluating visual citation (pinpointing block-level evidence), which provides a direct assessment of our fine-grained localization strategy. We adhere to the original evaluation protocols: rule-based scoring for MMLongBench-Doc and an LLM-as-a-judge approach for FinRAGBench-V. Further statistics and evaluation details are provided in Appendices B,C.3, and C.4. 

**Models and Baselines.** We evaluate our proposed agentic framework on three cutting-edge proprietary models: Gemini-2.5-Pro (Comanici et al., 2025), Gemini-2.5-Flash (Comanici et al., 2025), and Claude-4-Sonnet (ClaudeTeam, 2025). We benchmark its performance against three categories of baselines. The first is the vanilla setting, which uses only page screenshots. The second augments screenshots with OCR text appended to each page, an approach we found particularly effective during our pilot study. The third category comprises existing agentic frameworks: MACT (Yu et al., 2025), M3DocRAG (Cho et al., 2024), MDocAgent (Han et al., 2025), and SimpleDoc (Jain et al., 2025). For MACT, M3DocRAG, and MDocAgent, we report the best scores from their original papers. For SimpleDoc, the most recent and best-performing training-free framework, we reproduce results across all proprietary models using our metrics to ensure fair comparison. 

## **4.2. Main Results on MMLongBench-Doc** 

Table 1 presents our main experimental results. On MMLongBench-Doc, our approach yields substantial performance improvements across all three backbone models. These gains are particularly pronounced for comparatively weaker models, such as Claude-4-Sonnet and Gemini-2.5-Flash, compared to the more powerful Gemini-2.5-Pro. Notably, our DocLens framework enables both Claude-4-Sonnet and Gemini-2.5-Flash to achieve near-human performance. Furthermore, Gemini2.5-Pro augmented with our method surpasses the human baseline by ∼ 2%. These results strongly demonstrate the effectiveness of DocLens. 

Additionally, our method achieves significant improvements on the Unanswerable (UNA) subset, with absolute gains of +8.2%, +13.0%, and +13.8% for Claude-4-Sonnet, Gemini-2.5-Flash, and Gemini-2.5-Pro, respectively. This indicates that our agentic framework effectively mitigates model hallucination, a critical capability for real-world applications. 

Finally, we observe that augmenting models with OCR text substantially improves performance across all four backbones compared to the vanilla setting. We attribute this improvement to OCR’s effectiveness in facilitating (implicit) evidence page retrieval, as further analyzed in Section 5.2. 

5 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

|**Model**|||**MMLongBench-Doc**|**MMLongBench-Doc**|**MMLongBench-Doc**|||**FinRAGBench-V**|**FinRAGBench-V**|**FinRAGBench-V**|**FinRAGBench-V**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||**TXT**|**LAY**|**CHA**|**TAB**|**FIG**|**UNA**|**ALL**|**TXT**|**TAB**|**CHA**||**ALL**|
||||_Vanilla VLMs_|_Vanilla VLMs_|||||||||
|GPT-4o†|46.3|46.0|45.3|50.0|44.1|20.2|42.8|-|-|-||37.2|
|Claude-4-Sonnet|50.4|49.4|50.5|57.3|43.9|59.0|53.4|36.6|20.2|51.9||33.8|
|Gemini-2.5-Flash|44.0|53.2|46.0|43.9|48.2|56.7|49.6|49.0|41.6|41.0||43.0|
|Gemini-2.5-Pro|52.1|62.1|55.5|55.3|54.0|59.9|58.1|62.2|55.3|50.4||54.9|
|o4-mini†|-|-|-|-|-|-|-|-|-|-||62.4|
|||_VLMs Augmented with OCR_|_VLMs Augmented with OCR_||||||||||
|Claude-4-Sonnet|52.7|51.6|50.0|58.1|45.3|65.9|56.0|58.7|21.6|54.3||41.0|
|Gemini-2.5-Flash<br>Gemini-2.5-Pro|55.9<br>54.9<br>52.7<br>63.4<br>50.3<br>60.8<br>59.7<br>**65.3**<br>60.8<br>68.3<br>55.7<br>58.4<br>_VLM-based Agentic Frameworks_<br>~~ee ~~||||||58.5<br>63.3<br> ~~ee~~|67.6<br>70.0|64.4<br>70.0|46.1<br>56.2||58.3<br>64.9|
|MACT (w/ MiMo-VL-7B)†<br>M3DocRAG (w/ Qwen2-VL-7B)†<br>MDocAgent (w/ GPT-4o)†<br>**SimpleDoc**|-<br>30.0<br>-|-<br>23.5<br>-|-<br>18.9<br>-|-<br>20.1<br>-|-<br>20.8<br>-|-<br>5.8<br>-|47.4<br>21.0<br>42.0<br>pf|-<br>-<br>-|-<br>-<br>-|-<br>-<br>-||-<br>-<br>-|
|w/ Claude-4-Sonnet<br>w/ Gemini-2.5-Flash<br>w/ Gemini-2.5-Pro<br>**DocLens (Ours)**|52.1<br>45.5<br>48.4|53.3<br>57.4<br>54.8|58.3<br>49.0<br>55.7|62.4<br>51.6<br>56.1|46.9<br>45.2<br>52.5|66.5<br>66.5<br>59.7|58.6<br>59.6<br>53.3<br>**70.2**<br>56.6<br>67.5<br>f~~e~~||68.9<br>56.2<br>64.0|54.9<br>53.6<br>60.9||61.7<br>58.3<br>63.6|
|w/ Claude-4-Sonnet|59.9|58.2|54.4|63.9|55.3|**74.0**|63.3|**70.2**|66.0|60.3||64.8|
|w/ Gemini-2.5-Flash<br>w/ Gemini-2.5-Pro|59.5<br>61.5<br>54.8<br>66.9<br>**63.7**<br>64.6<br>**64.3**<br>**69.7**<br>~~ee~~||||59.0<br>**60.2**|73.8<br>72.2|64.7<br>69.9<br>**67.6**<br>∗<br>68.9<br>~~e~~e||71.3<br>**74.2**|64.5<br>**67.1**||68.5<br>**70.4**|



Table 1 | Main Results on the MMLongBench-Doc and FinRAGBench-V benchmarks. We report the accuracy of five types of evidence sources including pure text (TXT), layout (LAY), chart (CHA), table (TAB), and figure (FIG), and on unanswerable (UNA) samples. **Bold** indicates the best score per column; underlined indicates the best per column within each block.† denotes results reported in the original paper, hence some results are unavailable.[∗] Denotes results surpassing human experts (On MMLongBench-Doc, performance of human experts is 65.8). 

## **4.3. Main Results on FinRAGBench-V** 

On the FinRAGBench-V benchmark, our framework’s superiority is even more pronounced. Compared to the strongest baseline, DocLens achieves substantial gains when paired with Claude-4-Sonnet (+3.1%), Gemini-2.5-Flash (+10.2%), and Gemini-2.5-Pro (+5.5%). We hypothesize these larger gains stem from FinRAGBench-V’s higher proportion of documents with dense, complex visual layouts (e.g., newspapers). 

A closer analysis confirms this hypothesis, revealing that the performance boost is primarily driven by our method’s superior handling of visual evidence. On chart-based questions, for instance, DocLens elevates the performance of Gemini-2.5-Pro and Gemini-2.5-Flash by absolute margins of +10.9% and +23.5% over the strong OCR-augmented baseline. This trend continues for table-based questions, with corresponding gains of +4.2% and +6.9%. Collectively, these results demonstrate that as visual complexity increases, the advantage of our fine-grained element localization becomes increasingly critical—a capability we analyze in further detail in Section 5.3. 

6 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

|**Methods**<br>**MMLong**<br>**FinRAG**<br>ANS UNA<br>ALL<br>~~a~~|**Methods**<br>**MMLong**<br>**FinRAG**<br>ANS UNA<br>ALL<br>~~a~~|
|---|---|
|DocLens (Gemini-2.5-Pro)<br>w/o Lens Module<br>w/o Reasoning Module|66.4 72.2<br>**67.6**<br>**70.4**<br>63.3 64.4 63.5<br>65.1<br>**66.6**<br>68.2 67.0<br>69.9<br>~~===~~|
|DocLens (Gemini-2.5-Flash)|62.4<br>**73.8**<br>64.7<br>68.5<br>~~|~~|
|w/o Lens Module|58.4 68.4 60.4<br>60.9|
|w/o Reasoning Module|62.0 71.1 63.8<br>67.1|



|**Setting**|**#Pages **|**Recall **|**Prec **|**Final Acc**|
|---|---|---|---|---|
|Evidence Pages (Oracle)|1.5|100.0|100.0|69.1|
|**Baseline Retrievers**|||||
|MDocAgent’s Retriever|13.6|71.1|7.0|49.6|
|SimpleDoc’s Retriever|4.9|89.0|34.7|64.0|
|DocLens’s Page Navigator|||||
|w/ Gemini-2.5-Pro|3.5|**97.3**|55.1|**67.6**|
|w/ Gemini-2.5-Flash|3.1|95.2|**62.0**|67.1|
|w/ Gemini-2.5-Flash-Lite|3.2|90.2|60.0|64.4|



Table 2 | Ablation study of key modules in our proposed method. _MMLong_ , _FinRAG_ , _ANS_ , _UNS_ is short for _MMLongBench-Doc_ , _FinRAGBench-V_ , _Answerable_ , _Unanswerable_ , respectively. 

Table 3 | Final Accuracy On MMLongBench-Doc with varying retrieval backbones for the Page Navigator. _#Pages_ denotes average number of retrieved pages. _Prec_ is short for _Precision_ . 

## **5. Analysis** 

This section presents a comprehensive analysis of our framework. By default, all experiments are conducted with Gemini-2.5-Pro. We begin with an ablation study (§ 5.1), which confirms that the Lens Module significantly boosts performance and the Reasoning Module can further reduce hallucination. We then delve deeper into the Lens Module (§ 5.2 and § 5.3), demonstrating through quantitative analysis and case studies how its Page Navigator improves page recall and its Element Localizer enhances visual comprehension by pinpointing specific elements. Finally, we demonstrate the framework’s efficiency via a hybrid-backbone variant that outperforms baseline with much lower cost (§ 5.4). We also discussed test-time scaling effect of our method in the Appendix D.1. 

## **5.1. Ablation on Core Modules** 

Table 2 ablates the efficacy of our Lens Module and Reasoning Module. The ablation settings are as follows: to ablate the Lens Module, we provide the raw screenshot and OCR text directly to the Reasoning Module. To ablate the Reasoning Module, we take the output from the Lens Module (relevant pages and elements) and send it directly to the backbone VLM for answer generation. The results underscore the critical role of the Lens Module. Its removal leads to a substantial performance drop of 4.1% on MMLongBench-Doc and 5.3% on FinRAGBench-V for Gemini-2.5-Pro, and similar degradation for Gemini-2.5-Flash (7.6% and 1.4%, respectively). The Reasoning Module, meanwhile, can further reduce model hallucination. Its absence leads to a noticeable drop in performance on the unanswerable (UNA) queries. 

## **5.2. Analysis on Page Navigator** 

**Page Navigator achieves nearly perfect recall of evidence pages.** First, we demonstrate the effectiveness of our Page Navigator in terms of retrieving evidence pages. We assess this aspect in two ways: by calculating the recall of retrieved pages against the annotated evidence pages, and by measuring the final accuracy after processing these pages with Element Localizer, Answer Sampler, and Adjudicator. As presented in Table 3, on MMLongBench-Doc, our Page Navigator backboned with Gemini-2.5-Pro achieve a near-perfect recall of 97.3%, and its final accuracy is only 1.5% behind using oracle pages. 

7 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Page Navigator outperforms other retrievers.** We then conduct a comparative analysis by substituting our Page Navigator with retrievers from leading prior work. The first and most prevalent category is vector-based retrievers (Cho et al., 2024; Dong et al., 2025a; Han et al., 2025), exemplified by MDocAgent (Han et al., 2025), which retrieve the top-K most similar pages based on vector representations of the query and individual pages. We use ColBERT (Santhanam et al., 2021) for textual retrieval and ColPali (Faysse et al., 2024) for visual retrieval, combining their top-10 results. More recently, SimpleDoc (Jain et al., 2025) introduces a refinement through a two-stage pipeline: first retrieving pages via ColQwen2.5, then using an LLM to select evidence pages based on generated summaries. We substitute the original LLM with Gemini-2.5-Pro to maintain consistency with our framework’s backbone. 

As shown in Table 3, MDocAgent’s vector-based retriever retrieves 13.6 pages on average but achieves low recall on evidence pages, resulting in the poorest final accuracy. SimpleDoc significantly improves both recall and precision, scoring 64.4% for final accuracy. However, it still underperforms our Page Navigator even when using the weakest backbone. These results validate the effectiveness of our Page Navigator for evidence page retrieval. 

**OCR significantly enhances page retrieval.** We now examine each design choice of our Page Navigator. Table 4 presents ablation results on MMLongBench-Doc and FinRAGBench-V. We observe that both sampling and OCR augmentation improve evidence page recall, which successfully translates into higher final accuracy. Most notably, OCR provides substantial improvements: for Gemini-2.5-Pro, it increases recall by 10.0% and 40.5% on MMLongBench-Doc and FinRAGBench-V, respectively; for Gemini-2.5Flash, the gains are 15.9% and 50.1%. This indicates that VLMs are more adept at retrieving relevant information in the textual domain than in the visual domain. 

|**Methods**|**MMLong**|**MMLong**|**FinRAG**|**FinRAG**|
|---|---|---|---|---|
|**Gemini-2.5-Pro**<br>Page Navigator<br>w/o Sampling<br>w/o OCR<br>**Gemini-2.5-Flash**<br>Page Navigator<br>w/o Sampling|Recall<br>**97.3**<br>95.6<br>87.3<br>95.2<br>88.0|Final Acc<br>**67.6**<br>66.5<br>58.1<br>64.7<br>58.1|Recall<br>**94.3**<br>89.9<br>53.8<br>90.4<br>78.3|Final Acc<br>**70.4**<br>69.0<br>50.0<br>68.5<br>60.2|
|w/o OCR|79.3|49.6|40.3|45.6|



Table 4 | Ablation of Page Navigator in terms of retrieving evidence pages (recall) and impact on final performance (accuracy). _MMLong_ and _FinRAG_ is short for _MMLongBench-Doc_ and _FinRAGBench-V_ . 

## **5.3. Analysis on Element Localizer** 

**Element Localizer enhances block-level evidence identification.** FinRAGBench-V provides 202 cases with human-annotated bounding boxes for relevant blocks on evidence pages. We leverage this subset to examine whether the element localizer improves block-level evidence identification. Specifically, we provide evidence pages to Gemini-2.5-Pro and compare bounding box predictions with and without the element localizer. As shown in Figure 3, the Element Localizer substantially improves block-level performance, increasing precision by 4.9%, recall by 9.3%, and F1 score by 6.7%. (See Appendix C.4 for detailed calculations ) This also enhances the reliability and traceability of the final output (Ma et al., 2024a). 

**Effectiveness on visual-centric queries.** We further analyze the Element Localizer’s effectiveness across different evidence types. The MMLongBench-Doc benchmark comprises five evidence categories: Pure-text (Plain-text), Generalized-text (Layout), Table, Chart, and Figure. We partition samples into three distinct sets based on their evidence sources: _Text-Only_ , containing evidence exclusively from the first two categories; _Visual-Only_ , containing evidence solely from the latter three 

8 

**==> picture [472 x 196] intentionally omitted <==**

**----- Start of picture text -----**<br>
DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding<br>60 80 MMLongBench-Doc FinRAGBench-V<br>w/o Element Localizer<br>w/o EL 74 w/o EL<br>55 w/ Element Localizer 53.3 75 w/ EL w/ EL<br>50 70 72<br>46.0<br>45 44.0 65 70<br>40.4<br>40 39.3 60 68<br>35.5<br>35 55<br>66<br>30 “slat 50<br>Precision Recall F1 Text-Only Visual-Only      Text&Visual Text-Only Visual-Only<br>Wie wa<br>Figure 3 | Element Localizer enhances block-level Figure 4 | Effectiveness of Element Localizer ( EL )<br>evidence identification. on different evidence sources.<br>Score<br>**----- End of picture text -----**<br>


categories; and _Text&Visual_ , containing evidence from both domains. We compare Final Scores with and without the Element Localizer across these three data splits, with results presented in Figure 4. The Element Localizer demonstrates substantial benefits when evidence involves visual elements, while providing negligible improvement on Text-Only tasks. This pattern holds consistently on the FinRAGBench-V benchmark, reinforcing this finding. 

**Case Study.** Figure 5 presents two cases that highlight the effectiveness of the Element Localizer. The first case requires identifying a trend from a small bar chart embedded within a dense newspaper page. The second demands a more intricate task: locating a specific line plot in a research paper, extracting precise numerical values from it, and then presenting them in descending order. By first identifying and then cropping these visual elements for detailed inspection, our Localizer effectively addresses such complex visual challenges. 

## **5.4. Hybrid Backbones for Cost Efficiency** 

Our framework’s separation of high-cost retrieval from low-cost reasoning creates a natural opportunity for efficiency gains. This cost imbalance arises because the Page Navigator must process the entire document (avg. 49.4 pages on MMLongBench-Doc), while the Reasoner only analyzes the retrieved pages (avg. 3.5). To facilitate cost efficiency, we explore hybrid backbones by substituting the Page Navigator’s Gemini-2.5-Pro backbone with cheaper alternatives: Gemini-2.5-Flash and Gemini2.5-Flash-Lite[1] . As shown in Table 3, using Gemini-2.5-Flash (67.1%) and even the lightweight Gemini-2.5-Flash-Lite (64.4%) for retrieval both outperform the vanilla Gemini-2.5-Pro baseline (63.3%). This confirms the potential of our framework to balance cost and performance. 

## **6. Related Work** 

**Visual Document Understanding.** Visual document understanding aims to extract information from documents containing both textual and visual elements, including text, tables, charts, and figures. Early efforts primarily focus on understanding short, single-page visual documents, establishing foundational benchmarks such as DocVQA (Mathew et al., 2021), ChartQA (Masry et al., 2022), and SlideVQA (Tanaka et al., 2023). With recent advances in VLMs, models have achieved strong performance on these benchmarks, prompting the research community to shift toward two more 1We refer to `https://ai.google.dev/gemini-api/docs/pricing` . Gemini-2.5-Pro / Flash / Flash-Lite costs $1.25 / $0.3 / $0.1 per million input tokens, respectively. 

9 

> DocLens Q : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

Figure 5 | Cases demonstrating the effectiveness of Element Localizer. 

challenging directions. The first involves multi-document scenarios (Cho et al., 2024; Dong et al., 2025a; Zhao et al., 2025), which focus on retrieving relevant documents from a corpus and performing retrieval-augmented generation, as exemplified by ViDoRAG (Wang et al., 2025a), M3DocRAG (Cho et al., 2024), and VRAG-RL (Wang et al., 2025b). The second setting involves single, long-document comprehension (Deng et al., 2024; Dong et al., 2025b; Ma et al., 2024b; Zou et al., 2024), challenging models to process cohesive but extensive visual documents (Han et al., 2025; Jain et al., 2025; Yu et al., 2025). Our work situates within this latter context. 

**Evidence Localization in Visual Document Understanding.** A central challenge in both multidocument and single long-document scenarios is the localization of evidence. Conventional approaches (Tanaka et al., 2025; Yu et al., 2024) primarily rely on vector-based models (Chen et al., 2024; Faysse et al., 2024; Günther et al., 2023; Santhanam et al., 2021; Wang et al., 2022; Zhu et al., 2024) to retrieve top-K pages by combining textual and visual features. However, embedding models often struggle to capture complex reasoning relationships (Hongjin et al., 2025), resulting in suboptimal page recall. A recent improvement, SimpleDoc (Jain et al., 2025), employs an iterative pipeline that retrieves pages via vector-based models, uses an LLM to select evidence pages from generated summaries, and refines the query based on missing information, repeating until sufficient information is gathered. In contrast, our work demonstrates that directly employing long-context VLMs with OCR augmentation can achieve significant improvements in both recall and precision for page-level retrieval. Another key differentiator of our method is its localization granularity. While prior work typically operates only at the page level, our framework leverages document parsing tools to pinpoint specific visual elements such as tables, figures, and charts. This fine-grained localization enables substantial improvements in comprehending visually complex elements. 

**Agentic Frameworks for Long Context Modeling** Fueled by recent LLM advancements, agentbased systems are increasingly prominent for their ability to handle complex reasoning tasks via multi-role collaboration and tool usage (Song et al., 2023; Tran et al., 2025; Wu et al., 2024; Yao et al., 2022). In the realm of long-context modeling, agentic frameworks (Chen et al., 2023a; Edge et al., 2024; Li et al., 2025; Ouyang et al., 2025; Yang et al., 2025b; Zhang et al., 2025; Zhao et al., 2024; Zhou et al., 2023) offer a flexible alternative to extending context windows and processing target tasks 

10 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

via end-to-end training (Chen et al., 2023b; Peng et al., 2023; Zhu et al., 2023). These frameworks generally fall into two categories: retrieval-augmented generation and memory-augmented generation. The first category (Edge et al., 2024; Han et al., 2025; Li et al., 2025; Yang et al., 2025b; Zhao et al., 2024) focuses on first retrieving relevant information followed by generating an answer based on these pieces. The second category (Chen et al., 2023a; Ouyang et al., 2025; Zhang et al., 2025; Zhou et al., 2023), in contrast, first compresses long context segments into summaries or more abstract ‘memory’—often using divide-and-conquer or on-the-fly approaches—and then directs the model to answer questions based on this compressed memory. Our method extends the first category of agentic frameworks by leveraging existing document parsing tools to perform more fine-grained analysis and localization of relevant elements within the document. 

## **7. Conclusion** 

In this paper, we introduced DocLens, a tool-augmented multi-agent framework that addresses critical challenges in long visual document understanding: evidence localization. Through its Lens Module for precise evidence retrieval and a Reasoning Module for robust analysis, our framework significantly improves the performance of various VLMs on the MMLongBench-Doc and FinRAGBenchV benchmarks. Notably, DocLens with Gemini-2.5-Pro not only achieves SOTA results but also surpasses human expert performance, demonstrating its effectiveness. 

## **Limitations** 

While this paper makes substantial progress in evidence localization for long visual document understanding and achieves state-of-the-art performance on two challenging benchmarks, several limitations remain. 

First, regarding visual element comprehension, although our Lens module delivers notable improvements for understanding Charts, Tables, and Figures, many challenging cases (e.g., See Appendix E) persist that cannot be adequately addressed through simple “zooming-in” strategies. Effectively handling these cases requires either designing dedicated agentic frameworks tailored to specific visual element types or advancing the fundamental perception capabilities of backbone LLMs. 

Second, our current approach does not distinguish between document domains. In realistic scenarios, documents from specialized domains such as legal, medical, or financial fields often require domain-specific expert knowledge for accurate interpretation. Automatically constructing expert-level agents tailored to different document domains represents a promising direction for future work. 

## **Acknowledgement** 

We thank all members of Google Cloud AI Research for their valuable support during the project. 

## **References** 

- H. Chen, R. Pasunuru, J. Weston, and A. Celikyilmaz. Walking down the memory maze: Beyond context limit through interactive reading. _arXiv preprint arXiv:2310.05029_ , 2023a. 

- J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. Bge m3-embedding: Multi-lingual, multifunctionality, multi-granularity text embeddings through self-knowledge distillation, 2024. 

11 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

- S. Chen, S. Wong, L. Chen, and Y. Tian. Extending context window of large language models via positional interpolation. _arXiv preprint arXiv:2306.15595_ , 2023b. 

- J. Cho, D. Mahata, O. Irsoy, Y. He, and M. Bansal. M3docrag: Multi-modal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ , 2024. 

- ClaudeTeam. Claude sonnet 4: Hybrid reasoning model with superior intelligence for high-volume use cases, and 200k context window. `https://www.anthropic.com/claude/sonnet` , 2025. 

- G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_ , 2025. 

- C. Deng, J. Yuan, P. Bu, P. Wang, Z.-Z. Li, J. Xu, X.-H. Li, Y. Gao, J. Song, B. Zheng, et al. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. _arXiv preprint arXiv:2412.18424_ , 2024. 

- K. Dong, Y. Chang, X. D. Goh, D. Li, R. Tang, and Y. Liu. Mmdocir: Benchmarking multi-modal retrieval for long documents. _arXiv preprint arXiv:2501.08828_ , 2025a. 

- K. Dong, Y. Chang, S. Huang, Y. Wang, R. Tang, and Y. Liu. Benchmarking retrieval-augmented multimomal generation for document question answering. _arXiv preprint arXiv:2505.16470_ , 2025b. 

- D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson. From local to global: A graph rag approach to query-focused summarization. _arXiv preprint arXiv:2404.16130_ , 2024. 

- M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, and P. Colombo. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ , 2024. 

- M. Günther, J. Ong, I. Mohr, A. Abdessalem, T. Abel, M. K. Akram, S. Guzman, G. Mastrapas, S. Sturua, B. Wang, et al. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. _arXiv preprint arXiv:2310.19923_ , 2023. 

- D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang, J. Jiang, J. Wang, J. Chen, J. Huang, K. Lei, L. Yuan, L. Luo, P. Liu, Q. Ye, R. Qian, S. Yan, S. Zhao, S. Peng, S. Li, S. Yuan, S. Wu, T. Cheng, W. Liu, W. Wang, X. Zeng, X. Liu, X. Qin, X. Ding, X. Xiao, X. Zhang, X. Zhang, X. Xiong, Y. Peng, Y. Chen, Y. Li, Y. Hu, Y. Lin, Y. Hu, Y. Zhang, Y. Wu, Y. Li, Y. Liu, Y. Ling, Y. Qin, Z. Wang, Z. He, A. Zhang, B. Yi, B. Liao, C. Huang, C. Zhang, C. Deng, C. Deng, C. Lin, C. Yuan, C. Li, C. Gou, C. Lou, C. Wei, C. Liu, C. Li, D. Zhu, D. Zhong, F. Li, F. Zhang, G. Wu, G. Li, G. Xiao, H. Lin, H. Yang, H. Wang, H. Ji, H. Hao, H. Shen, H. Li, J. Li, J. Wu, J. Zhu, J. Jiao, J. Feng, J. Chen, J. Duan, J. Liu, J. Zeng, J. Tang, J. Sun, J. Chen, J. Long, J. Feng, J. Zhan, J. Fang, J. Lu, K. Hua, K. Liu, K. Shen, K. Zhang, K. Shen, K. Wang, K. Pan, K. Zhang, K. Li, L. Li, L. Li, L. Shi, L. Han, L. Xiang, L. Chen, L. Chen, L. Li, L. Yan, L. Chi, L. Liu, M. Du, M. Wang, N. Pan, P. Chen, P. Chen, P. Wu, Q. Yuan, Q. Shuai, Q. Tao, R. Zheng, R. Zhang, R. Zhang, R. Wang, R. Yang, R. Zhao, S. Xu, S. Liang, S. Yan, S. Zhong, S. Cao, S. Wu, S. Liu, S. Chang, S. Cai, T. Ao, T. Yang, T. Zhang, W. Zhong, W. Jia, W. Weng, W. Yu, W. Huang, W. Zhu, W. Yang, W. Wang, X. Long, X. Yin, X. Li, X. Zhu, X. Jia, X. Zhang, X. Liu, X. Zhang, X. Yang, X. Luo, X. Chen, X. Zhong, X. Xiao, X. Li, Y. Wu, Y. Wen, Y. Du, Y. Zhang, Y. Ye, Y. Wu, Y. Liu, Y. Yue, Y. Zhou, Y. Yuan, Y. Xu, Y. Yang, Y. Zhang, Y. Fang, Y. Li, Y. Ren, Y. Xiong, Z. Hong, Z. Wang, Z. Sun, Z. Wang, Z. Cai, Z. Zha, Z. An, Z. Zhao, Z. Xu, Z. Chen, Z. Wu, Z. Zheng, Z. Wang, Z. Huang, Z. Zhu, and Z. Song. Seed1.5-vl technical report, 2025. URL `https://arxiv.org/abs/2505.07062` . 

12 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

- S. Han, P. Xia, R. Zhang, T. Sun, Y. Li, H. Zhu, and H. Yao. Mdocagent: A multi-modal multi-agent framework for document understanding. _arXiv preprint arXiv:2503.13964_ , 2025. 

- S. Hongjin, H. Yen, M. Xia, W. Shi, N. Muennighoff, H.-y. Wang, L. Haisu, Q. Shi, Z. S. Siegel, M. Tang, et al. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. In _The Thirteenth International Conference on Learning Representations_ , 2025. 

- C. Jain, Y. Wu, Y. Zeng, J. Liu, Z. Shao, Q. Wu, H. Wang, et al. Simpledoc: Multi-modal document understanding with dual-cue page retrieval and iterative refinement. _arXiv preprint arXiv:2506.14035_ , 2025. 

- X. Li, G. Dong, J. Jin, Y. Zhang, Y. Zhou, Y. Zhu, P. Zhang, and Z. Dou. Search-o1: Agentic searchenhanced large reasoning models. _arXiv preprint arXiv:2501.05366_ , 2025. 

- J. Liu, D. Zhu, Z. Bai, Y. He, H. Liao, H. Que, Z. Wang, C. Zhang, G. Zhang, J. Zhang, et al. A comprehensive survey on long context language modeling. _arXiv preprint arXiv:2503.17407_ , 2025. 

- X. Ma, S. Zhuang, B. Koopman, G. Zuccon, W. Chen, and J. Lin. Visa: Retrieval augmented generation with visual source attribution. _arXiv preprint arXiv:2412.14457_ , 2024a. 

- Y. Ma, Y. Zang, L. Chen, M. Chen, Y. Jiao, X. Li, X. Lu, Z. Liu, Y. Ma, X. Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems_ , 37:95963–96010, 2024b. 

- A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 

- M. Mathew, D. Karatzas, and C. Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 

- J. Niu, Z. Liu, Z. Gu, B. Wang, L. Ouyang, Z. Zhao, T. Chu, T. He, F. Wu, Q. Zhang, Z. Jin, G. Liang, R. Zhang, W. Zhang, Y. Qu, Z. Ren, Y. Sun, Y. Zheng, D. Ma, Z. Tang, B. Niu, Z. Miao, H. Dong, S. Qian, J. Zhang, J. Chen, F. Wang, X. Zhao, L. Wei, W. Li, S. Wang, R. Xu, Y. Cao, L. Chen, Q. Wu, H. Gu, L. Lu, K. Wang, D. Lin, G. Shen, X. Zhou, L. Zhang, Y. Zang, X. Dong, J. Wang, B. Zhang, L. Bai, P. Chu, W. Li, J. Wu, L. Wu, Z. Li, G. Wang, Z. Tu, C. Xu, K. Chen, Y. Qiao, B. Zhou, D. Lin, W. Zhang, and C. He. Mineru2.5: A decoupled vision-language model for efficient high-resolution document parsing, 2025. URL `https://arxiv.org/abs/2509.22186` . 

OpenAITeam. Introducing gpt-5. `https://openai.com/index/introducing-gpt-5/` , 2025. 

- S. Ouyang, J. Yan, I. Hsu, Y. Chen, K. Jiang, Z. Wang, R. Han, L. T. Le, S. Daruki, X. Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. _arXiv preprint arXiv:2509.25140_ , 2025. 

- B. Peng, J. Quesnelle, H. Fan, and E. Shippole. Yarn: Efficient context window extension of large language models. _arXiv preprint arXiv:2309.00071_ , 2023. 

- QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action. `https: //qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research. latest-advancements-list` , 2025. 

- K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. _arXiv preprint arXiv:2112.01488_ , 2021. 

13 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

- Y. Song, W. Xiong, D. Zhu, W. Wu, H. Qian, M. Song, H. Huang, C. Li, K. Wang, R. Yao, et al. Restgpt: Connecting large language models with real-world restful apis. _arXiv preprint arXiv:2306.06624_ , 2023. 

- R. Tanaka, K. Nishida, K. Nishida, T. Hasegawa, I. Saito, and K. Saito. Slidevqa: A dataset for document visual question answering on multiple images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 13636–13645, 2023. 

- R. Tanaka, T. Iki, T. Hasegawa, K. Nishida, K. Saito, and J. Suzuki. Vdocrag: Retrieval-augmented generation over visually-rich documents. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 24827–24837, 2025. 

- C. Team, Z. Yue, Z. Lin, Y. Song, W. Wang, S. Ren, S. Gu, S. Li, P. Li, L. Zhao, L. Li, K. Bao, H. Tian, H. Zhang, G. Wang, D. Zhu, Cici, C. He, B. Ye, B. Shen, Z. Zhang, Z. Jiang, Z. Zheng, Z. Song, Z. Luo, Y. Yu, Y. Wang, Y. Tian, Y. Tu, Y. Yan, Y. Huang, X. Wang, X. Xu, X. Song, X. Zhang, X. Yong, X. Zhang, X. Deng, W. Yang, W. Ma, W. Lv, W. Zhuang, W. Liu, S. Deng, S. Liu, S. Chen, S. Yu, S. Liu, S. Wang, R. Ma, Q. Wang, P. Wang, N. Chen, M. Zhu, K. Zhou, K. Zhou, K. Fang, J. Shi, J. Dong, J. Xiao, J. Xu, H. Liu, H. Xu, H. Qu, H. Zhao, H. Lv, G. Wang, D. Zhang, D. Zhang, D. Zhang, C. Ma, C. Liu, C. Cai, and B. Xia. Mimo-vl technical report, 2025. URL `https://arxiv.org/abs/2506.03569` . 

- K.-T. Tran, D. Dao, M.-D. Nguyen, Q.-V. Pham, B. O’Sullivan, and H. D. Nguyen. Multi-agent collaboration mechanisms: A survey of llms. _arXiv preprint arXiv:2501.06322_ , 2025. 

- L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei. Text embeddings by weakly-supervised contrastive pre-training. _arXiv preprint arXiv:2212.03533_ , 2022. 

- Q. Wang, R. Ding, Z. Chen, W. Wu, S. Wang, P. Xie, and F. Zhao. Vidorag: Visual document retrievalaugmented generation via dynamic iterative reasoning agents. _arXiv preprint arXiv:2502.18017_ , 2025a. 

- Q. Wang, R. Ding, Y. Zeng, Z. Chen, L. Chen, S. Wang, P. Xie, F. Huang, and F. Zhao. Vrag-rl: Empower vision-perception-based rag for visually rich information understanding via iterative reasoning with reinforcement learning. _arXiv preprint arXiv:2505.22019_ , 2025b. 

- X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Selfconsistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In _First Conference on Language Modeling_ , 2024. 

- A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report, 2025a. URL `https://arxiv.org/abs/2505.09388` . 

- Z. Yang, T. Peng, C. Gao, C. Wang, H. Huang, and Y. Deng. A deep dive into retrieval-augmented generation for code completion: Experience on wechat. _arXiv preprint arXiv:2507.18515_ , 2025b. 

- S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. In _The eleventh international conference on learning representations_ , 2022. 

14 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

- S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y. Yan, Z. Liu, S. Wang, X. Han, Z. Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. _arXiv preprint arXiv:2410.10594_ , 2024. 

- X. Yu, Z. Chen, Y. Zhang, S. Lu, R. Shen, J. Zhang, X. Hu, Y. Fu, and S. Yan. Visual document understanding and question answering: A multi-agent collaboration framework with test-time scaling. _arXiv preprint arXiv:2508.03404_ , 2025. 

- Q. Zhang, C. Hu, S. Upasani, B. Ma, F. Hong, V. Kamanuru, J. Rainton, C. Wu, M. Ji, H. Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. _arXiv preprint arXiv:2510.04618_ , 2025. 

- Q. Zhao, R. Wang, Y. Cen, D. Zha, S. Tan, Y. Dong, and J. Tang. Longrag: A dual-perspective retrievalaugmented generation paradigm for long-context question answering. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pages 22600–22632, 2024. 

- S. Zhao, Z. Jin, S. Li, and J. Gao. Finragbench-v: A benchmark for multimodal rag with visual citation in the financial domain. _arXiv preprint arXiv:2505.17471_ , 2025. 

- W. Zhou, Y. E. Jiang, P. Cui, T. Wang, Z. Xiao, Y. Hou, R. Cotterell, and M. Sachan. Recurrentgpt: Interactive generation of (arbitrarily) long text. _arXiv preprint arXiv:2305.13304_ , 2023. 

- D. Zhu, N. Yang, L. Wang, Y. Song, W. Wu, F. Wei, and S. Li. Pose: Efficient context window extension of llms via positional skip-wise training. _arXiv preprint arXiv:2309.10400_ , 2023. 

- D. Zhu, L. Wang, N. Yang, Y. Song, W. Wu, F. Wei, and S. Li. Longembed: Extending embedding models for long context retrieval. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pages 802–816, 2024. 

- A. Zou, W. Yu, H. Zhang, K. Ma, D. Cai, Z. Zhang, H. Zhao, and D. Yu. Docbench: A benchmark for evaluating llm-based document reading systems, 2024. URL `https://arxiv.org/abs/2407. 10701` . 

15 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## **A. Prompt Templates for our Agentic Framework.** 

In this section, we provide the system prompts for our Page Navigator, Answer Sampler, Adjudicator. 

## **Prompt used for the Page Navigator** 

## `## ROLE` 

- `You are an expert AI assistant specializing in multimodal long document understanding. Given a multimodal long document and a user question, your task is to systematically locate the indices of all pages that might contain information useful for answering the user’s question, and then provide an answer to the question.` 

- `## Follow these instructions carefully:` 

- `Core Objective: Your primary goal is to identify all pages relevant to the question. The pages you identify will be passed to a specialized agent for detailed examination, making recall your most important optimization goal. If a page might be useful, you should include it; it is better to be over-inclusive and let the subsequent agent perform the detailed check.` 

- `Provide References: While fulfilling the Core Objective, provide the corresponding reference pages. If the user’s question explicitly refers to a specific page, slide, figure, or section (e.g., "in slide 5", "on page 10"), then index of the corresponding page MUST be included in the located_pages list. However, it is crucial to understand that when a document has printed page numbers, a user’s reference to "Page X" typically means the page with that printed number, not its sequential index in the file. For instance, if a PDF has a cover page, a user referring to "Page 2" means the page with the printed number ’2’, but its actual index might be 3. You must resolve the user’s referenced page number into its correct page index. Crucially, the values you return in the located_pages list must always be these page indices (starting from 1). This rule is non-negotiable and overrides any other consideration about the page’s content or sufficiency.` 

- `Rules of numerical answers:` 

   - `If the user asks for an absolute number (e.g., with questions like` 

   - `"How many...?"), you must first attempt to locate the number directly. If it cannot be found, you must find the pages containing the relevant percentage and total count (or other necessary data) to calculate the absolute number. If the calculated absolute number for discrete entities (e.g., people, companies, objects) is a decimal, you must round it to the nearest whole number.` 

   - `If the user asks for a percentage (or proportion), you must first` 

   - `attempt to locate the percentage directly. If it cannot be found, you must find the pages containing the absolute numbers of the subgroup and the total count (or other necessary data) to calculate the percentage. - If the user’s question is ambiguous and does not explicitly specify a` 

   - `number or percentage (e.g., "What’s the gap between...?"), you must default to providing the absolute value. If you can only find relative values (percentages) in the chart, you must make every effort to find a total number within the provided context to calculate the absolute value. Only return the relative value as a last resort if a total number cannot be found, and explain that you cannot find total number in this case.` 

16 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## `## Output Format:` 

- `Your entire response MUST be a single, valid JSON object and nothing else. Do not wrap it in markdown code blocks or add any other text. The JSON object must contain exactly three fields: analysis (string), located_pages (string), and prediction (string).` 

- `analysis field: Briefly explain your thought process. Describe how you located the answer within the document, which pages, tables, or figures you referenced, and how you connected the information to the question.` 

- `located_pages field: This must be a string representation of a list of integers. Page indices start at 1. If relevant pages are found, it should look like this: "[3, 10, 12]". If no pages contain relevant evidence, it MUST be an empty list: "[]". Always return the index of the target page (starting from 1), not the page number printed on the page.` 

- `prediction field: This must be a string containing the direct answer to the user’s question.` 

17 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## **Prompt used for the Answer Sampler** 

- `## ROLE` 

- `You are an expert AI assistant specializing in multimodal long document understanding. Your task is to carefully analyze the provided page images (which may contain text, figures, tables, and other content) and provide a precise answer to the user’s question. Treat the provided pages as a curated and sufficient set of information. A preceding agent has already identified them as the key relevant pages from the full document, so you do not need to second-guess the relevance of the provided content. For example, if the question is about an appendix, but the provided pages aren’t explicitly labeled as such, you should assume they are the correct appendix pages. If the question refers to a page range and you are only given images, assume those images constitute the content of those pages. If the question asks for a specific item (e.g., the "5th FAQ") and you are shown only one, treat that as the target item. Your task is to carefully review these pages and provide an accurate answer.` 

## `## Follow these instructions carefully:` 

- `Core Objective: Your primary goal is to accurately and concisely answer the user’s question based on the content of the provided document pages.` 

- `- Rules of numerical answers:` 

   - `If the user asks for an absolute number (e.g., with questions like` 

   - `"How many...?"), you must first attempt to locate the number directly. If it cannot be found, find the relevant percentage and total count (or other necessary data) to calculate the absolute number. If the calculated absolute number for discrete entities (e.g., people,` 

   - `companies, objects) is a decimal, you must round it to the nearest whole number.` 

   - `If the user asks for a percentage (or proportion), you must first` 

   - `attempt to locate the percentage directly. If it cannot be found, find the absolute numbers of the subgroup and the total count (or other necessary data) to calculate the percentage.` 

   - `If the user’s question is ambiguous and does not explicitly specify a` 

   - `number or percentage (e.g., "What’s the gap between...?"), you must default to providing the absolute value. If you can only find relative values (percentages) in the chart, you must make every effort to find a total number within the provided context to calculate the absolute value. Only return the relative value as a last resort if a total number cannot be found, and explain that you cannot find total number in this case.` 

- `Zoom-in Feature: When a page image contains figures or tables and requires closer inspection, we may provide zoomed-in images of these elements, appended after the main page image (Noted as "---- Zoomed-in Figures and Charts of this page ----"), to help you examine them closely. We will also extract text from the page image into Markdown format. Note: For questions related to page layout, you must refer to the original page image itself, not the zoomed-in images or the Markdown text, as they may lose layout information. For instance, if asked for the first figure on the page, you should consult the full page image to determine its order, not the sequence of the provided zoomed-in images.` 

- `- Page Numbering: Page numbers in the user’s question typically refer to the number printed on the page image, not the page’s index in the document file. For example, if a PDF’s first page is the cover and the` 

18 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

- `third page is the first page of content (labeled "Page 1"), a user’s question about "page 1" refers to that third page. Similarly, when asked to provide a page number, you should return the printed page number from the image. Only return the page index if no number is printed on the page.` 

- `- Rule of faithfulness: Be faithful. If the provided pages do not contain sufficient information to answer the user’s question, you should answer ‘Not answerable‘. For example, if the user asks for a man in green shirts, but there are only man in red shirts in the provided pages, you should answer ‘Not answerable‘; if the user asks for the boy playing badminton, but there are only boys playing football in the provided pages, you should answer ‘Not answerable‘; if the user asks for a certain year’s data but the provided pages only contain data for other years, you should answer ‘Not answerable‘; if the user asks for the color of a certain object but the provided pages do not contain that object, you should answer ‘Not answerable‘.` 

## `## Output Format:` 

- `Your entire response MUST be a single, valid JSON object and nothing else. Do not wrap it in markdown code blocks or add any other text. The JSON object must contain exactly two fields: analysis (string), and prediction (string).` 

- `- analysis field: Briefly explain your thought process. Describe how you located the answer within the document, which pages, tables, or figures you referenced, and how you connected the information to the question.` 

- `- prediction field: This must be a string containing the direct answer to the user’s question.` 

19 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## **Prompt used for the Answer Adjudicator** 

- `## Role:` 

- `You are an expert AI assistant specializing in multimodal long document understanding. Your primary role is to serve as an aggregator of different answers (and corresponding analyses) provided by multiple AI agents for a given question about a complex long document containing various information formats such as text, images, and charts.` 

- `## Follow these instructions carefully:` 

- `Core Objective: Your ultimate goal is to accurately and concisely answer the user’s question based on the content of the provided document pages. You will be presented will several answers and analyses from different agents, and you must determine which answer is the most appropriate by evaluating the reasoning behind each one.` 

- `Serving as a judge, not a executor. Despite we are tackling document understanding, the target document will only be presented to the previous agents, but not you. So your primary objective is not to solve the problem from scratch yourself, but to examine the existing analyses, and find the correct answer.` 

- `Avoid Frequency Bias: You must ignore the frequency with which an answer appears. An answer being repeated by multiple agents does not make it correct. Your judgment must be based solely on factual evidence from the document, not on consensus.` 

- `Be careful about faithfulness: Sometimes the question might be unanswerable given the provided document pages. In this case, "Not answerable" should be the desired answer. However, not all agents will be aware of this. Some of them might provide an hallucinated answer, or first twist the question to make it answerable. An example is the user asks for a specific year, but the provided pages only contain data for other years. In this case, some agents might answer with the closest year. Despite they are trying to be helpful, this is not faithful to the document. Another example is if a user asks for the meaning of a specific fruit on a given page, but that page only contains information about a different fruit. Trying to be helpful, the agent might say that the requested fruit is not on the page, and then proceed to explain the meaning of the other fruit that is present. In such cases, the desired answer must still be "Not answerable". It is your duty to indentify such cases, and choose "Not answerable" as the final answer.` 

- `Rule of Common Sense: Sometimes, an agent can be overly pedantic or literal about certain concepts. For example, when asked if a "line plot" exists on a page, an agent might get bogged down in the technical definition and misidentify upward or downward arrows as a line plot. This clearly defies common sense. In reality, the user is an ordinary person. You must interpret their intent in the most common-sense way and select the agent’s answer that best aligns with a general, conventional understanding.` 

`## Input Format` 

- `You will first be provided with the question, and then a list of Agent responses in the following format:` 

`**Question:**` 

`[The exact question that was asked will be stated here]` 

20 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

`**List of Agent Analyses and Answers:** Agent 1 Analysis: [The reasoning process provided by Agent 1] Answer: [The final answer provided by Agent 1] Agent 2 Analysis: [The reasoning process provided by Agent 2] Answer: [The final answer provided by Agent 2] Agent 3 [...]` 

`## Output Format:` 

`Your entire response MUST be a single, valid JSON object and nothing else. Do not wrap it in markdown code blocks or add any other text. The JSON object must contain exactly two fields: analysis (string), and prediction (string).` 

`- analysis field: Insert your detailed meta-analysis here. You must explicitly reference and critique the analysis of the different agents. - prediction field: Insert the exact text of the correct agent answer here, with no prefix` 

21 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## **B. Dataset Statistics** 

Statistics of MMLongBench-Doc and FinRAGBench-V is presented in Table 5. MMLongBench-Doc comprises documents from 7 various domains, including Research report/Introduction, Tutorial/Workshop, Academic Paper, Guidebook, Brochure, Administration/Industry file, Financial Report. FinRAGBench-V, on the other hand, focuses sole on financial reports. 

|**Documents**<br>- Average/Medium pages<br>- Average/Medium words<br>~~=~~|135<br>301<br>47.5 / 28<br>76.1 / 57.0<br>8,393 / 5,743<br>36,026 / 16,329<br>~~=~~|
|---|---|
|**Total question**<br>- Single-page question<br>- Cross-page question<br>- Unanswerable question<br>~~=~~|1,082<br>1,394<br>494 (45.7%)<br>1,218 (87.4%)<br>365 (33.7%)<br>178 (12.6%)<br>223 (20.6%)<br>-<br>~~=~~|
|**Evidence source**<br>- Pure-text<br>- Layout<br>- Table<br>- Chart<br>- Image<br>~~=~~|305 (35.5%)<br>302 (21.7%)<br>119 (13.9%)<br>-<br>218 (25.4%)<br>573 (41.1%)<br>178 (20.7%)<br>519 (37.2%)<br>304 (35.4%)<br>-<br>~~=~~|
|Avg. / Max. question words<br>Avg. / Max. answer words<br>~~=~~|16.2 / 54<br>35.8 / 108<br>2.1 / 66<br>23.4 / 174<br>~~=~~|



## **C. Implementation Details** 

## **C.1. Implementing DocLens** 

**Document parsing tools.** We employ MinerU (Niu et al., 2025), a recently proposed document parsing tool, to perform OCR, layout detection, and cropping. 

**Sampling.** Most mainstream API interfaces, including vLLM[2] , OpenAI[3] , and Google GenAI SDK[4] , support generating n independent candidate responses for a single input. The cost structure is: input token cost × 1 + total output token cost across all candidates. Compared to invoking the API nn n times sequentially, this approach is substantially more efficient in both time and cost. This efficiency is particularly pronounced in long-context scenarios, where output length is negligible relative to input length. Consequently, for long visual document understanding tasks, multiple sampling incurs minimal additional token overhead. The sole exception is Anthropic’s API, which does not support this functionality; therefore, for experiments involving Claude, we invoke the API N times sequentially. In our implementation, both the retriever sampling count _𝑇𝑒_ and the answer sampler sampling count _𝑇𝑎_ are set to 8, with a temperature _𝜏_ = 0 _._ 7. 

> 2 `https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html` 

> 3 `https://platform.openai.com/docs/api-reference/chat/create#chat-create-n` 

> 4 `https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig. candidate_count` 

22 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Page Navigator.** The Page Navigator takes as input screenshots of all pages along with their OCR text. In practice, an LLM’s finite context window may prevent processing all _𝑁𝑝_ pages simultaneously for extremely long documents. To address this, we employ a chunking strategy that divides pages into chunks of _𝐾_ pages each, processes them in parallel with the Page Navigator, and merges the identified pages. In our experiments, this strategy is not required for Gemini-2.5-Pro and Gemini-2.5-Flash, both of which support a 1M token context window. However, Claude-4-Sonnet frequently encounters context limitations on FinRAGBench-V. For this model, we set _𝐾_ = 50, processing 50 pages per iteration. This substantially improves evidence page recall from 72% to 87% on FinRAGBench-V. 

**Miscellaneous.** Some APIs impose restrictions on image resolution. When an API raises an error regarding excessive image size, we reduce the resolution by half and retry the request. In the VLMs Augmented with OCR setting, if context limit is exceed, we discard the text and roll back to the image only setting. 

**Pseudo Code.** Algorithm 1 presents overall workflow of our DocLens framework. 

## **C.2. Reproducing SimpleDoc** 

SimpleDoc (Jain et al., 2025) is the most recent training-free agentic framework for long visual document understanding. By optimizing the page retriever, it significantly improves upon previous methods including MDocAgent (Han et al., 2025) and M3DocRAG (Cho et al., 2024). Its page retriever includes a vector-based retrieval phase backed by ColQwen2.5, and a summary-augmented reranking phase backed by Qwen3-30B-A3B (Yang et al., 2025a). The selected pages, along with the question, are then fed into Qwen2.5-VL-32B-Instruct for answer generation. This process is iterated multiple times to include all potentially relevant pages and deliver the most reliable answer. Finally, it uses GPT-4o to evaluate the consistency between the generated answer and ground truth. 

To reproduce their results for fair comparison, we adopt the following settings: 1) For the vectorbased page retrieval phase, we return the top-30 pages, as suggested by the paper. 2) We replaced all generative backbones (Qwen3-30B-A3B, Qwen2.5-VL-32B-Instruct) with Gemini-2.5-Flash, Gemini2.5-Pro, and Claude-4-Sonnet; 3) We adopted the evaluation protocol proposed in the original paper. Notably, for MMLongBench-Doc, the original evaluation process first uses an LLM to extract answers, then applies rule-based metrics to calculate accuracy. Based on our observations, this approach is overly stringent compared to directly using an LLM to assess semantic consistency, resulting in somewhat lower scores. For reference, when using LLM-based evaluation, our DocLens with Gemini2.5-Pro achieves a score of 75, whereas under the original metric it obtains only 67.6. Nevertheless, in this paper, we report the original metric to facilitate comparison with results from the original paper (including human experts) and other previous work. 

23 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Algorithm 1** Overall Workflow of `DocLens` 

**Require:** A visual document D = { _𝑃𝑖_ } _𝑖[𝑁]_ =1[; A question] _[𝑄]_[in natural language.] **Ensure:** The final answer _𝐴 𝑓𝑖𝑛𝑎𝑙_ . 

1: _⊲_ **Lens Module** : Extract relevant evidence from the document 2: E _𝑝𝑟𝑒𝑑_ ←∅ _⊲⊲_ Initialize the set of predicted evidence pages. 3: **for** _𝑖_ = 1 to _𝑁_ **do** 4: _𝑇𝑖_ ← OCR( _𝑃𝑖_ ) _⊲⊲_ Extract text from each page. 

5: **end for** 

6: _⊲_ **Page Navigator** : Identify relevant pages 7: **for** _𝑗_ = 1 to _𝑇𝑒_ **do** _⊲⊲_ Sample multiple times for comprehensive coverage. 8: E[(] _[ 𝑗]_[)] ← LLM _PageNav_ ( _𝑄,_ {( _𝑃𝑖, 𝑇𝑖_ )} _𝑖[𝑁]_ =1[)] 9: E _𝑝𝑟𝑒𝑑_ ←E _𝑝𝑟𝑒𝑑_ ∪E[(] _[ 𝑗]_[)] 10: **end for** 

11: _⊲_ **Element Localizer** : Extract visual and textual elements from predicted pages 12: S ←∅ _⊲⊲_ Initialize the full evidence set. 

13: **for all** page _𝑃𝑘_ ∈E _𝑝𝑟𝑒𝑑_ **do** 14: _𝑇𝑘_ ← text extracted for _𝑃𝑘_ in line 4 15: _𝑏𝑏𝑜𝑥𝑒𝑠_ ← LayoutDetect( _𝑃𝑘_ ) _⊲⊲_ Identify bounding boxes of visual elements. 16: V _𝑘_ ←∅ 17: **for all** _𝑏𝑏𝑜𝑥_ ∈ _𝑏𝑏𝑜𝑥𝑒𝑠_ **do** 18: V _𝑘_ ←V _𝑘_ ∪{Crop( _𝑃𝑘, 𝑏𝑏𝑜𝑥_ )} _⊲⊲_ Crop visual elements from the page. 19: **end for** 20: S ←S ∪{( _𝑃𝑘, 𝑇𝑘,_ V _𝑘_ )} _⊲⊲_ Aggregate page screenshot, text, and visuals. 21: **end for** 

22: _⊲_ **Reasoning Module** : Generate the final answer from the evidence 23: _⊲_ **Answer Sampler** : Generate multiple candidate answers 24: { _𝑅𝑖, 𝐴𝑖_ } _[𝑇] 𝑖_ = _[𝑎]_ 1[←∅] _⊲⊲_ Initialize a set for reasoning-answer pairs. 25: **for** _𝑖_ = 1 to _𝑇𝑎_ **do** _⊲⊲_ Generate diverse reasoning paths. 26: _𝑅𝑖, 𝐴𝑖_ ← LLM _Sampler_ ( _𝑄,_ S) 27: **end for** 28: _⊲_ **Adjudicator** : Synthesize the final answer 29: _𝐴 𝑓𝑖𝑛𝑎𝑙_ ← LLM _Adjud_ ({( _𝑅𝑖, 𝐴𝑖_ )} _[𝑇] 𝑖_ = _[𝑎]_ 1[)] _⊲⊲_ Select the most consistent and logical conclusion. 

30: **return** _𝐴 𝑓𝑖𝑛𝑎𝑙_ 

## **C.3. Evaluation on MMLongBench-Doc** 

1. For MMLongBench-Doc, we manually fixed some annotations about evidence pages. 

## **Prompt used for Answer Extraction on MMLongBench-Doc** 

`Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.` 

- `Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis and the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".` 

24 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

`- Please make your response as concise as possible. Also note that your response should be formatted as below: ‘‘‘ Extracted answer: [answer] Answer format: [answer format] ‘‘‘` 

`Please read the following example, then extract the answer from the model response and type it at the end of the prompt.` 

`--Question: List the primary questions asked about the services in this report. Analysis: The primary questions asked about the services in the report for The Limes Residential Home are:\n\n1. Is the service safe?\n2. Is the service effective?\n3. Is the service caring?\n4. Is the service responsive?\n5. Is the service well-led? Extracted answer: [’Is the servife safe?’, ’Is the service effective’, ’Is the serve caring?’, ’Is the service responsive?’, ’Is the service well-led?’] Answer format: List` 

`---` 

`Question: How many regulations of the HSCA 2008 are breached in all according to this report?` 

`Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\n\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\n2. Regulation 12: Safe care and treatment\n3. Regulation 18: Staffing\n4. Regulation 11: Need for consent\n5. Regulation 10: Dignity and respect\n6. Regulation 9: Person-centred care\n7. Regulation 17: Good governance\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\n\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.` 

`Extracted answer: 10` 

`Answer format: Integer` 

`---` 

`Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump’s election?` 

`Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump’s election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that! Extracted answer: Not answerable` 

25 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

`Answer format: String --Question: How many quotations from male respondent over 50 years old are included in this report? Analysis: The image you’ve provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question. Extracted answer: Fail to answer Answer format: String ---` 

26 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

## **C.4. Evaluation on FinRAGBench-V** 

## **Prompt used for Answer Evaluation on FinRAGBench-V** 

- `### ROLE` 

- `You are an expert evaluator. Your task is to determine if a model’s generated answer is correct by comparing it to a ground truth value.` 

- `### TASK` 

- `You will be given a question, the prediction which includes reasoning steps and a final answer, and a ground_truth which is the correct answer. You must determine if the final conclusion of the prediction matches the ground_truth.` 

- `### INSTRUCTIONS 1. **Understand the Goal:** Read the question to understand what information needs to be found.` 

- `2. **Extract the Final Answer:** Carefully analyze the prediction. Ignore the reasoning steps and identify only the final, conclusive answer provided by the model. The answer is often at the end of the text and might be bolded.` 

- `3. **Compare with Ground Truth:** Compare the extracted final answer with the ground_truth. Be flexible with formatting-for example, a model answer of "45 percent" should be considered a match for a ground truth of "45".` 

`4. **Generate Analysis:** Write a brief analysis of your finding.` 

`### INPUTS` 

`You will receive the data like this: Question: [The user’s question] Ground Truth: [The expected answer] Prediction: [The model’s actual answer]` 

`## OUTPUT FORMAT:` 

`Your response MUST be a JSON object with two keys: 1. score: A float, either 1.0 for a correct prediction or 0.0 for an incorrect one.` 

`2. reasoning: A brief, one-sentence explanation for your decision.` 

**Calculation of page-level recall, precision, and F1.** To evaluate page retrieval performance, we calculate the page-level recall, precision, and F1 scores as follows. Let Ppred denote the set of predicted pages and Pgt denote the set of ground truth pages. We define true positives (TP) as |Ppred ∩Pgt|, false positives (FP) as |Ppred \ Pgt|, and false negatives (FN) as |Pgt \ Ppred|. The metrics are then calculated as: 

**==> picture [155 x 77] intentionally omitted <==**

27 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Calculation of element-level recall, precision, and F1.** For element-level evaluation, we assess the quality of predicted bounding boxes against ground truth annotations. Let Bgt = { _𝑏_[gt] 1 _[, 𝑏]_ 2[gt] _[, . . . , 𝑏]_[gt] _𝑛_[}] denote the set of ground truth bounding boxes and Bpred = { _𝑏_[pred] 1 _, 𝑏_[pred] 2 _, . . . , 𝑏_[pred] _𝑚_ } denote the set of predicted bounding boxes, where each box is represented as [ _𝑥_ 1 _, 𝑦_ 1 _, 𝑥_ 2 _, 𝑦_ 2]. 

We employ Intersection over Union (IoU) as the matching criterion. For two boxes _𝑏𝑖_ and _𝑏 𝑗_ , IoU is computed as: 

**==> picture [125 x 28] intentionally omitted <==**

A predicted box is considered a true positive if it achieves an IoU ≥ 0 _._ 5 with a ground truth box, and each ground truth box is matched to at most one predicted box. Based on the matching results, we calculate: 

**==> picture [154 x 76] intentionally omitted <==**

where TP (true positives) is the number of successfully matched boxes, FP (false positives) is the number of unmatched predicted boxes, and FN (false negatives) is the number of unmatched ground truth boxes. 

## **D. Further Discussion** 

## **D.1. Test-time Scaling of Page Navigator and Answer Sampler** 

**==> picture [460 x 177] intentionally omitted <==**

**----- Start of picture text -----**<br>
Retriever's Recall vs. Sample Num Adjudicated vs Best of N<br>95 80<br>Adjudicated<br>Best of N<br>90<br>75<br>85<br>70<br>80<br>/ a PO<br>yon<br>75 65<br>1 2 3 4 5 6 7 8 1 2 4 8<br>#Sample Num #Sample Num<br>Final Acc (%)<br>Retriever's Recall (%)<br>**----- End of picture text -----**<br>


Figure 6 | Test-time scaling of Page Navigator and Answer Sampler 

As illustrated in Figure 6, we observe distinct scaling behaviors between the Page Navigator and Answer Sampler components during test-time inference. For the Page Navigator (retrieval evidence pages), increasing the number of samples yields substantial improvements in retriever’s recall performance. The recall metric rises from approximately 78% with a single sample to over 90% when utilizing 8 samples. However, the marginal gains diminish as the sample count increases. Notably, 

28 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding ~~ee~~ 

beyond 8 samples, the incremental improvement becomes negligible, falling below 1 percentage point. This suggests that the Page Navigator benefits significantly from test-time scaling but exhibits diminishing returns at higher sample counts. 

In contrast, the Answer Sampler demonstrates markedly different scaling characteristics. When comparing the adjudicated results against the best-of-N selection, we observe that performance improvements plateau rapidly. While increasing from 1 to 2 samples produces a notable gain (approximately 5 percentage points), further scaling beyond this point yields minimal additional benefits. The adjudicated performance remains relatively flat at around 68-69% across varying sample numbers, while even the best-of-N approach shows limited improvement beyond 2 samples. This behavior indicates that the Answer Sampler module does not require sophisticated test-time scaling strategies, and a modest sample size is sufficient to achieve near-optimal performance. These findings suggest that computational resources for test-time scaling should be allocated primarily to the Page Navigator component, where increased sampling continues to provide meaningful improvements, rather than the Answer Sampler, where additional samples beyond 2 offer diminishing returns. 

## **E. Hard Cases** 

In this sections, we present some vision-centric cases that even our DocLens fail to solve. Effectively handling these cases might require either designing dedicated agentic frameworks tailored to specific visual element types or advancing the fundamental perception capabilities of backbone LLMs, which we leave for future work. 

**Hard Case 1:** From 2009 to 2013, for the adviser’s organic growth rate, how many years are higher than 2011? ( **Ground Truth:** 1; **DocLens:** 4) 

29 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding ~~a~~ 

**Hard Case 2:** which age group experienced the greatest change in the percentage holding an "unfavorable" opinion of China between 2005 and 2010? ( **Ground Truth:** 50 and older; **DocLens:** 30-49) 

30 

DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding ~~a~~ 

**Hard Case 3:** How many "WC"s are shown in the last page’s map? ( **Ground Truth:** 19; **DocLens:** 9) 

31 

~~a~~ DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Hard Case 4:** In figure 4, which nodes are retrieved by RAPTOR for both questions? ( **Ground Truth:** [16, 19, 25]; **DocLens:** [1, 11, 16, 17, 20]) 

32 

