# **Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling** 

Xinlei Yu[1] Chengming Xu[2] Zhangquan Chen[3] Yudong Zhang[3] Shilin Lu[5] Cheng Yang[7] Jiangning Zhang[6] Shuicheng Yan[1] Xiaobin Hu[1][*] 

1National University of Singapore 2Tencent Youtu Lab 3Tsinghua University 

4University of Science and Technology of China 

5Nanyang Technological University 6Zhejiang University 7DeepWisdom 

## **Abstract** 

_The dominant paradigm of monolithic scaling in VisionLanguage Models (VLMs) is failing for understanding and reasoning in documents, yielding diminishing returns as it struggles with the inherent need of this domain for document-based procedural reasoning, cognitive complexity, and factual accuracy. To this end, we introduce_ _**MACT** , a_ _**M** ulti-_ _**A** gent_ _**C** ollaboration framework with agent-wise adaptive_ _**T** est-time scaling that pioneers a paradigm shift to procedural scaling, adapting dynamically to the functional entities of visual documents understanding and reasoning. MACT decomposes the visual document processing flow into four specialized agents, i.e., planning, execution, judgment, and answer, to resolve cognitive overload and introduce a critical self-correction loop for factual grounding. This collaborative architecture is amplified by an agentwise adaptive test-time scaling strategy that intelligently allocates computational resources based on the complexity and redundancy of each functionality. Evaluated on multiple visual document understanding benchmarks, MACT achieves superior performance with a smaller parameter scale, adapting effectively to various document scenarios without compromising its general or mathematical reasoning capabilities. The three variants of MACT consistently attain top-three average performance rankings, with average performance enhancements of 9.9–11.5% over the base models. The source code will be released publicly._ 

## **1. Introduction** 

For humans, acquiring and understanding visual information from documents is an indispensable part of daily life. The corresponding tasks of visual documents, _e.g._ , Visual Question Answering (VQA), serves as a quantifiable benchmark for a model’s ability to comprehend complex layouts, dense text, and structured data. To this end, the field has 

> *Corresponding authors. 

seen a rapid evolution of benchmarks, moving from simple documents to multi-page reports, complex charts, and webpages. The questions have similarly evolved from simple lookups to requiring semantic understanding and multi-hop reasoning, pushing the capability boundaries of VLMs. 

The dominant paradigm for advancing VLM capabilities, both for documents and other general vision data, has been **monolithic scaling** , _i.e._ , increasing parameter size, leveraging more high-quality training data. The approach has led to remarkable success, with phenomenal models like GPT-4o [5], Gemini-2.0-Pro [3] and Claude-3.7-Sonnet [1], demonstrating powerful general-purpose abilities. However, for document-based VQA, this strategy of uniform scaling is showing diminishing returns. As shown in Fig. 1, only limited performance gain is attained along parameter scaling for open-sourced models [31, 37] with exponential growth in computational overhead, suggesting that simply increasing parameter count is a brute-force solution that fails to address the core challenges of the domain. 

The core problem is that **documents are not monolithic entities that are suitable for uniform scaling** . Unlike natural images, they possess a unique set of characteristics that make naive scaling less effective: **(1) Suboptimal procedural reasoning.** Document analysis is not a single action of perception but rather an inherently procedural, multistep workflow [8]: decomposing the question, forming a strategy, locating relevant information, and synthesizing a final answer. Monolithic models, with their single forwardpass architecture, attempt to solve this entire procedure at once, leading to less robust reasoning paths. **(2) Cognitive overload.** The procedural steps of document analysis demand a diverse set of specialized skills, such as global layout parsing, fine-grained text extraction, logical inference, and numerical calculation. A monolithic model is forced to encode all these disparate functions into a single set of weights, leading to cognitive overload and task interference. This leads to a compromise where the model achieves broad generalization at the expense of deep expertise in any single task. **(3) Vulnerability to factual errors.** The seman- 

1 

**==> picture [494 x 149] intentionally omitted <==**

**----- Start of picture text -----**<br>
Math-Vision Math-Vision Math-Vision<br>MathVista MathVerse MathVista MathVerse MathVista MathVerse<br>80 ScienceQA 80 ScienceQA 80 ScienceQA<br>TableBench TableBench TableBench<br>60 60 60<br>40 RealWorldQA 40 RealWorldQA 40 RealWorldQA<br>TableVQA-Bench TableVQA-Bench TableVQA-Bench<br>20 20 20<br>Qwen2.5-VL-7B-Instruct InternVL3-8B-Instruct<br>CharXiv Qwen2.5-VL-32B-InstructQwen2.5-VL-72B-InstructMACT-Qwen2.5-VL-Series-24B DocVQA CharXiv MiMo-VL-7B-SFTMACT-MiMo-VL-Series-28B DocVQA CharXiv InternVL3-14B-InstructInternVL3-38B-InstructInternVL3-78B-InstructMACT-InternVL3-Series-28B DocVQA<br>DUDE DUDE DUDE<br>ChartQA ChartQA ChartQA<br>SlideVQA SlideVQA SlideVQA<br>InfographicVQA InfographicVQA InfographicVQA<br>VisualMRC MMLongBench-Doc VisualMRC MMLongBench-Doc VisualMRC MMLongBench-Doc<br>(a) Comparisons on Qwen2.5-VL Series [31] (b) Comparisons on MiMo-VL Series [29] (c) Comparisons on InternVL-3 Series [37]<br>**----- End of picture text -----**<br>


Figure 1. Comparisons among the three variants of MACT, the base models of these variants, and larger-scale models within the same model family, indicating the superiority of our framework over monolithic scaling paradigms. 

tic meaning in documents is extremely sensitive to minor procedural mistakes [4], even paragraph truncation errors or misaligning a table row can invalidate the entire answer. Monolithic, feed-forward models lack an internal verification or self-correction loop. An initial extraction error, however small, inevitably cascades through the reasoning process without being challenged, identified, and corrected. 

These challenges reveal the need for a new paradigm: a shift from **monolithic scaling** to **procedural scaling** , where the problem is decomposed and specialize test-time scalings are applied to each entity. To achieve this, we propose **MACT** , a **M** ulti- **A** gent **C** ollaboration framework with agent-wise adaptive **T** est-time scaling, enabling a more effective scaling framework. Specifically, it deconstructs the monolithic model into four specialized agents that explicitly mirror the required cognitive functions. Within this framework, we employ four relatively collaborative agents: planning, execution, judgment, and answer agents. This design inherently enforces procedural reasoning, dedicating a specialist to each cognitive step from planning to synthesis. This decomposition is what resolves the cognitive overload problem, allowing each agent to achieve mastery over its specific function. The key to unlocking this specialized performance is our agent-wise adaptive test-time scaling approach. Rather than uniform scaling, it is intelligent, ondemand allocation of computational resources, allowing the execution agent to apply maximum scrutiny to a single data point while the planning agent operates efficiently on the whole document. Finally, the framework directly confronts the high cost of errors. The judgment agent acts as a builtin verification, providing the self-correction capabilities that monolithic models lack. 

We train three variants of MACT based on different groups of base models using a two-stage pipeline, and evaluate them on 15 benchmarks, comprising 10 documentbased and 5 non-document-based ones. Our MACT showcases superior document understanding and reasoning with- 

out compromising general capabilities. Compared to models of comparable scale and larger-scale ones, the three variants of MACT demonstrate an average increase of 3.2% to 5.6%, securing the top three positions and achieving the best performance in 13 out of 15 benchmarks. The results demonstrate that monolithic scaling is not the optimal approach for visual document-based tasks. In contrast, the adaptive test-time scaling in multi-agent framework emerges as a more advantageous alternative, as it achieves superior scaling performance. 

## **2. Methodology** 

This section provides a detailed elaboration of our proposed MACT, divided into three subsections. We first introduce the multi-agent framework, which decomposes the document-tailored workflow into four functionalities: planning, execution, judgment, and answer, and defines the collaboration mechanisms among them. Then, we detail the agent-wise adaptive test-time scaling and mixed reward modeling tailored to the multi-agent collaboration framework. More details are presented in Appendix 7. 

## **2.1. Multi-Agent Collaboration Framework** 

Document understanding and reasoning constitutes not merely a unilateral perceptual activity, but an expected multistage process [8]: decomposing the question, formulating execution strategies, locating information pertinent to the document, conducting validation with necessary corrections, and synthesizing a final response. As shown in the upper part of Fig. 2, MACT consists of four collaborative agents: sequentially planning agent ( _Aplan_ ), execution agent ( _Aexe_ ), judgment agent ( _Ajudg_ ), and answer agent ( _Aans_ ). Each agent is role-specialized, which accomplishes its functionality and then passes to the subsequent agent or outputs the final answers. 

**Planning Agent.** Given a question _Q_ and corresponding visual document inputs _D_ , _Aplan_ mainly focuses on the 

2 

**==> picture [434 x 236] intentionally omitted <==**

**----- Start of picture text -----**<br>
Q       Planning       Agent      Execution       Agent        Agent   Judgment       AgentAnswer<br>Function: Generate high- Function:plans with the selected tools  Excute step-by-step  Function:plan or execution process and  Judge execution  Function:answers by previous incorrect  Generate final<br>8 level execution plan. and obtain execution process. redirect to the previous agents. segments and correct process.<br>Question<br>VLM VLM LLM<br>Execution  Execution CorrectProcess  LLM<br>Plan  Process<br>a, == =a cum © Output<br>L-2-D-2-D39 3-5-5 Answers<br>Generated<br>Visual Inputs Relevant Plans  Tool Library Mistakes<br>“ew? | [8 GE2 oe<br>Multi-Agent Collaboration Framework<br>Agent-Wise Adaptive Test-Time Scaling Mixed Reward<br>eV Agent-Specific<br>Reward<br>OV UOsls\-a | QB oi 22<br>Global Outcome<br>Con Reward<br>gor Seles ga? 72 ,<br>eo oe eee 2? Og 00 Agent Outputs<br>Rejected Agent<br>Outputs<br>e->¢¢g672%¢ > & Q2 OPE o Budget Forcing Outputs<br>**----- End of picture text -----**<br>


Figure 2. The overview of MACT. The upper part demonstrates our procedural framework, with four tailored and collaborative agents to conduct the process of document analysis. When the judgment agent detects mistakes, it redirects to previous agents for corrections. The lower part illustrates the agent-wise adaptive test-time scaling and mixed reward modeling for the multi-agent framework. 

analysis and decomposition of the original question, making high-level executive plans on the document-based tasks. Before generating the plans for the original task, we generate _Np_ similar and relevant instances, and their plans simultaneously, which is adapted from analogical prompting principles [33]. For each generated example and corresponding plan, _Aplan_ generates a distinct execution plan respectively, which could be formulated as: 

**==> picture [181 x 26] intentionally omitted <==**

where generated relevant plans _Prel_ = { _r_ 1 _, r_ 2 _, . . . , rNp_ } , and execution plans _P_ = { _p_ 1 _, p_ 2 _, . . . , pNp_ } , which provide different pathways to accomplishment. Besides, _M_ is the mistake from _Ajudg_ for correction, and it is initialized to empty. For each relevant plan and execution plan _p_ , it composed of steps _{s_ 1 _, s_ 2 _, . . . , sn}_ . It should be noted that we only yield high-level execution plans with limited details in this agent. To be more precise, the high-level plan describes the expected targets and requirements of each step, but does not output specific execution processes directly. It ensures that _Aplan_ can better project the overall plan instead of execution details and would not interfere with the execution of _Aexe_ . Besides, it avoids premature commitment to specific tools, enabling _Aexe_ to dynamically select optimal resources for diverse tasks. 

**Execution Agent.** _Aexe_ is designed to execute plans step by step and output the execution process using the se- 

lected tools from the tool library _T_ = _{t_ 1 _, t_ 2 _, . . . , tn}_ (see in Appendix 7.2). Specifically, it breaks down the execution plan and each step will be regarded as an execution unit, with information for each unit populated into a template and comprised of: (1) a specific definition; (2) expected target and output; (3) existing inputs or results from the previous step. _Aexe_ then executes the units sequentially, deriving the intended outputs from each. The execution of single step could be presented as: 

**==> picture [176 x 11] intentionally omitted <==**

Once all steps are completed, it concatenates the full output and the execution processes of all steps in sequence, passing _E_ = _{e_ 1 _, e_ 2 _, . . . , en}_ to subsequent agents. 

**Judgment Agent.** Document-based visual tasks exhibit high sensitivity to process-related errors [4], especially in monolithic models that omit verification and correction procedures. Trivial procedural discrepancies inevitably distort the overall reasoning trajectory, leading to a snowball effect that results in wholly erroneous outputs. Thus, some multi-agent systems incorporate mechanisms that either (a) internally correct errors within the same agent [19, 32] or (b) deploy an additional agent to handle both judgment and correction [6, 9]. However, the (a) internal correction mechanism struggles to identify most mistakes and may fall into cognitive blind spots, as generation and correction rely on the same model. In contrast, the latter approach (b) requires the agent to possess strong capabilities for both judgment 

3 

and correction, thereby necessitating models with larger parameters and more complex reward modeling designs. Furthermore, using a separate agent to regenerate mistaken components may result in incompatibilities or even conflicts with existing parts, given the autonomy of the two agents. In RL, the optimization objective for both (a) and (b), as illustrated in Fig. 3, is to pass verification. This can lead to strategic production of vague statements or the omission of details, resulting in corrections that appear correct superficially but are actually misleading. And utilizing another agent to regenerate the mistaken part might be incompatible or even causing conflicts with the existing parts because of the independence of two autonomous agents. 

To address the limitations of both approaches, we design a simple but effective judgment strategy for multi-agent systems, which introduces an additional judgment agent to separate judgment from correction, thereby fostering specialization of labor. More specifically, _Ajudg_ is primarily responsible for assessing the correctness of steps in previously generated execution plans and processes, without engaging in direct correction. The judgment process is summarized as: 

**==> picture [165 x 11] intentionally omitted <==**

where _J_ = _{flagplan, flagexe, M}_ , _flagplan_ = _flagexe ∈{true, false}_ , and _M_ is the mistake description. If mistakes are detected in any step, _Ajudg_ identifies the specific problematic step, provides a brief description of the mistake, and routes the issue to the appropriate agent, either _Aplan_ or _Aexe_ , for correction. 

For each plan-process pair that is already correct or has been corrected, _Ajudg_ forwards them to _Aans_ . This strategy, by decoupling judgment from correction, introduces a neutral judge with reduced subjective bias, whose focus remains solely on judgment rather than correction. Additionally, the reward design for the judgment agent can be intuitively simplified, without evaluating correction outcomes. To prevent infinite correction loops, the maximum number of corrections _Nc_ is set to 3. 

a **Answer Agent.** The primary function of _Aans_ is to output the final answer through both the incorrect processes and the correct process. Counterintuitively, _Aans_ incorporates both the correct execution process and incorrect segments from prior processes. This facilitates direct focus on modifications within the corrected process, thereby preventing the omission of error-prone details. Additionally, the mistake-correction pair presents a natural and complete logical closed-loop structure, which benefits answer generation based on the execution process. Thus, the final output answer is: 

**==> picture [165 x 10] intentionally omitted <==**

**Collaboration.** Throughout the full implementation of the multi-agent collaboration framework, visual inputs and 

**==> picture [164 x 188] intentionally omitted <==**

**----- Start of picture text -----**<br>
Generation Judgment Correction Result<br>2-B— 2—B<br>G-B—-G@—B<br>(a) Internal Correction<br>SoA | ~B<br>ga" Pa<br>(b) Agent for Judgment and Correction<br>(c) Independent  Judgment Agent (Ours)<br>**----- End of picture text -----**<br>


Figure 3. Comparisons of (a) internal correction, (b) an extra agent for both judgment and correction, and (c) our strategy utilizing an independent judgment agent. 

questions are first fed into _Aplan_ , whose generated plan is then executed by _Aexe_ . _Ajudg_ judges the correctness the resulting execution plan-process pairs and outputs mistake flags. For correct pairs, _Aans_ outputs the final answers, while the information of mistakes will be passed to the previous agent _Aplan_ or _Aexe_ for correction, and the procedure repeats. For clarity in illustrating the workflow, we set the number of generated relevant plans to 1 and simplify the test-time scaling designs in Algorithm 1, and these details are elaborated in the subsequent section. 

## **2.2. Agent-Wise Adaptive Test-Time Scaling** 

For visual document understanding and reasoning, the key challenge is not visual documents themselves, but the procedural reasoning that links global question decomposition, plan formation, information extraction, and mistake judgment across multiple steps. Monolithic scaling enlarges model capacity uniformly, yet fails to accommodate this structured reasoning process. We therefore propose an agent-wise adaptive scaling strategy during inference, dynamically allocating additional computing resources to different functional entities and enabling on-demand allocation during the test time. Existing test-time scaling methods fall into four main categories: parallel scaling, sequential scaling, hybrid scaling, and internal scaling. However, these strategies are originally designed for single models or agents, overlooking the different division of labor among agents and achieving suboptimal performance when applied to multi-agent systems. Accordingly, we propose an agentwise adaptive scaling strategy tailored to multi-agent architectures, which significantly enhances both agent-specific capabilities and collaborative performance. 

For the first three agents with various functions, we cus- 

4 

**Algorithm 1** Multi-Agent Collaboration Procedure 

|**Algorithm 1gorithm 1orithm 1**Multi-Agent Collaboration Proceduregent Collaboration Procedureent Collaboration Procedure|**Algorithm 1gorithm 1orithm 1**Multi-Agent Collaboration Proceduregent Collaboration Procedureent Collaboration Procedure|**Algorithm 1gorithm 1orithm 1**Multi-Agent Collaboration Proceduregent Collaboration Procedureent Collaboration Procedure|**Algorithm 1gorithm 1orithm 1**Multi-Agent Collaboration Proceduregent Collaboration Procedureent Collaboration Procedure|**Algorithm 1gorithm 1orithm 1**Multi-Agent Collaboration Proceduregent Collaboration Procedureent Collaboration Procedure||
|---|---|---|---|---|---|
|**Require:** question _Q_, visual document inputs _D_|||||=|
||_{v_1_, v_2_, . . . , vn}_, planning agent _Aplan_,|||execution||
||agent _Aexe_, the tool library of|the former||agent _T_|=|
||_{t_1_, t_2_, . . . , tn}_, judgment agent||_Ajudg_, answer agent|||
||_Aans_, maximum number of corrections_Nc_|||||
|**Ensure:** answer output to the question_O_||||||
|1:|Initialize the four agents: _Aplan_,_A_||_Aexe_,_Ajudg_,_Aans_|||
|2:|Initialize_Nc_ = 3_, t_= 0|||||
|3:|Initialize prompts formatter_PF_|||||
|4:|Initialize_flagplan_ =_flagexe_ =_false,_||_false, M_=|=_empty_||
|5:|**while**true**do**|||||
|6:|_p_1 _←PF_(_Q_)|||||
|7:|_Prel ←Aplan_(_D, p_1)||▷Relevant plans|||
|8:|_p_2 _←PF_(_Q, Prel, M_)|||||
|9:|_P ←Aplan_(_D, p_2)||▷Execution plan|||
|10:|**for**each step_si_ **in**_P_ **do**|||||
|11:|_p_3 _←PF_(_Q, si, T , M_)|||||
|12:|_ei ←Aexe_(_D, p_3_, T_)|▷Step-wise execution||||
|13:|**end for**|||||
|14:|_E ←{e_1_, e_2_, . . . , en}_||▷Execution process|||
|15:|_p_4 _←PF_(_Q, P, E_)|||||
|16:|_J ←Ajudg_(_p_4)||▷Judgment|||
|17:|_{flagplan, flagexe, M} ←J_||▷Mistakes|||
|18:|**if**not_flagplan_**and**not_flagexe_**or**_t ≥Nc_**then**|||||
|19:|**break**|||||
|20:|**else if**_flagplan_**then**|||||
|21:|_t ←t_+ 1|||||
|22:|**goto**line 8<br>▷Mistakes in execution plan|||||
|23:|**else if**_flagexe_**then**|||||
|24:|_t ←t_+ 1|||||
|25:|**goto**line 10<br>▷Mistakes in execution process|||||
|26:|**end if**|||||
|27:|**end while**|||||
|28:|_p_5 _←PF_(_Q, E, M_)|||||
|29:|_O ←Aans_(_p_5)|||▷Answer||



node for subsequent steps, with all others rejected. (3) Judgment agent: To judge the correctness of the execution plans and processes, _Ajudg_ requires logical analysis and reasoning to comprehensively and accurately detect mistakes to avoid the error snowballing in document contexts. We therefore adopt the budget forcing scaling method [15] for this agent, which enforces a minimum number of thinking tokens. When token usage falls below the budget, the agent is encouraged to generate additional thinking tokens, thereby promoting accurate judgments. For the answer agent, whose core function lies in synthesizing prior information and generating the final response, test-time scaling confers merely marginal improvements. 

## **2.3. Mixed Reward Modeling** 

According to the latest research [17, 35], adding RL strategy, _e.g._ , reward modeling, to optimize test-time scaling is significantly superior to fine-tuning or training-free methodologies. To optimize our proposed procedural scaling by elevating task-agent-centric and collaborative performance, we design a mixed reward strategy that synthesizes agentspecific rewards and global outcome-driven reward signals. Given that the four agents in the framework have distinct functions and reward signal preferences, we first incorporate agent-tailored rewards: 

**==> picture [201 x 27] intentionally omitted <==**

where are _Rprm_ and _Rorm_ are multi-modal process reward and outcome reward models, respectively. The former yields step-wise process rewards, providing instant feedback for each step and hierarchical rewards for _Aplan_ and _Aexe_ , and the latter generates single reward signal for each output of _Ajudg_ and _Aans_ . Furthermore, we apply a global reward based on the final selected path of the four agents: 

**==> picture [199 x 11] intentionally omitted <==**

tomize different test-time scaling strategies to their unique characteristics: (1) oo Planning agent: In our design of _Aplan_ , it naturally provides relevant sample plans as references for formulating multiple execution plans for the original task. We therefore prompt the generation of _Np_ relevant plans independently, yielding _Np_ parallel paths per question. This establishes a basis for scaling subsequent agents within document processing workflows, increasing the probability that at least one reasoning path will produce an accurate answer aligned with document semantics. (2) @ Execution agent: Given that the execution process is divided by step in _Aexe_ , we treat each step as an evaluation node. For each node, the agent produces _Ne_ candidate executions, which are scored using a pretrained reward model. The top-scoring candidate is selected as the base 

which reinforces incentives for correct paths and moderately mitigates agent self-interest. Since individual agents tend to optimize their own local rewards, the integration of a global objective effectively alleviates such self-serving tendencies. Please refer to Appendix 7.3 for details. 

## **3. Experiments** 

More details and descriptions about the training pipelines, datasets, evaluations and prompts for each agent are provided in Appendix 8.1, 8.2, 8.3, and 8.4, respectively. 

## **3.1. Training Pipeline** 

Since the functions and optimization goals of each agent are independent and diverse, we design a two-stage SFT 

5 

Table 1. Comparisons of our MACT and other counterparts on 15 benchmarks. The table cells with green, blue, and gray colors denote the best, the second best, and the third best values. Results with - indicate exceeding the maximum context token limit or not being equipped with a specific ability. _△,_ □ _,_ ♢ correspond to the base models of three variants of MACT for comparisons. 

|||**Document**<br>**Text**<br>**Webpage**<br>**Chart**<br>**Table**<br>DVQA DUDE SVQA MMLong<br>VisMRC InfVQA<br>ChartQA CharXiv<br>TabVQA TabBen|**Document**<br>**Text**<br>**Webpage**<br>**Chart**<br>**Table**<br>DVQA DUDE SVQA MMLong<br>VisMRC InfVQA<br>ChartQA CharXiv<br>TabVQA TabBen|**Document**<br>**Text**<br>**Webpage**<br>**Chart**<br>**Table**<br>DVQA DUDE SVQA MMLong<br>VisMRC InfVQA<br>ChartQA CharXiv<br>TabVQA TabBen|**Document**<br>**Text**<br>**Webpage**<br>**Chart**<br>**Table**<br>DVQA DUDE SVQA MMLong<br>VisMRC InfVQA<br>ChartQA CharXiv<br>TabVQA TabBen|**Non-Document**<br>**General**<br>**Mathematical**<br>SciQA RealQA<br>MVista MVision MVerse|**Non-Document**<br>**General**<br>**Mathematical**<br>SciQA RealQA<br>MVista MVision MVerse|**Avg.**|
|---|---|---|---|---|---|---|---|---|
|**Generalist**|**Closed-Source**<br>GPT-4o-latest<br>93.1<br>52.7<br>81.0<br>40.3<br>86.4<br>79.2<br>86.5<br>78.7<br>64.2<br>51.9<br>**83.3**<br>76.2<br>62.4<br>30.7<br>39.4<br>67.2<br>Claude-3.7-Sonnet<br>94.0<br>58.1<br>**83.6**<br>33.9<br>82.5<br>75.3<br>**92.2**<br>83.5<br>70.3<br>54.7<br>80.4<br>71.9<br>69.7<br>41.2<br>45.0<br>69.1<br>Gemini-2.0-Pro<br>91.8<br>54.3<br>78.9<br>32.2<br>**91.4**<br>81.6<br>88.8<br>83.1<br>71.2<br>**59.9**<br>**80.9**<br>70.5<br>74.8<br>**54.2**<br>**56.6**<br>71.3||||||||
||GPT-4o-latest|93.1<br>52.7<br>81.0<br>40.3|86.4<br>79.2|86.5<br>78.7|64.2<br>51.9||||
||Claude-3.7-Sonnet|94.0<br>58.1<br>**83.6**<br>33.9|82.5<br>75.3|**92.2**<br>83.5|70.3<br>54.7||||
||Gemini-2.0-Pro|91.8<br>54.3<br>78.9<br>32.2|**91.4**<br>81.6|88.8<br>83.1|71.2<br>**59.9**|**80.9**<br>70.5|74.8<br>**54.2**<br>**56.6**||
||**Size**_<_**20B**<br>DeepSeek-VL2-4.5B<br>78.8<br>-<br>-<br>-<br>70.4<br>65.3<br>74.0<br>46.9<br>30.8<br>-<br>66.6<br>60.7<br>29.5<br>6.9<br>7.4<br>-<br>LLaVA-OneVision-7B-si<br>86.6<br>48.4<br>52.9<br>5.3<br>76.1<br>74.8<br>82.3<br>54.3<br>47.9<br>33.3<br>72.8<br>62.0<br>57.9<br>17.9<br>19.1<br>52.8<br>Qwen2.5-VL-7B-Instruct_△_<br>94.6<br>62.3<br>76.2<br>22.4<br>86.8<br>82.8<br>87.5<br>68.2<br>60.4<br>49.2<br>74.0<br>68.4<br>67.8<br>25.5<br>40.8<br>64.5<br>MiMo-VL-7B-SFT□<br>92.9<br>62.3<br>74.2<br>27.5<br>87.3<br>82.7<br>86.1<br>72.0<br>57.5<br>52.5<br>75.1<br>67.9<br>70.7<br>**46.9**<br>53.3<br>67.3<br>InternVL3-8B-Instruct♢<br>91.1<br>60.1<br>72.9<br>28.6<br>84.4<br>78.5<br>85.4<br>67.1<br>59.7<br>52.4<br>76.4<br>66.7<br>65.5<br>28.8<br>38.9<br>63.8<br>Llama-3.2-11B-Vision-Instruct<br>87.3<br>45.0<br>66.5<br>8.8<br>79.6<br>73.8<br>79.6<br>63.6<br>50.4<br>44.6<br>76.9<br>60.7<br>56.5<br>16.3<br>32.3<br>56.1<br>LlaVa-1.6-vicuna-13B<br>79.0<br>42.9<br>65.6<br>7.9<br>77.9<br>74.7<br>76.6<br>52.5<br>24.9<br>39.2<br>70.7<br>58.1<br>50.2<br>11.6<br>27.5<br>50.6<br>InternVL3-14B-Instruct<br>90.7<br>61.0<br>74.5<br>28.9<br>87.1<br>83.1<br>86.2<br>75.1<br>58.2<br>51.4<br>77.1<br>66.4<br>66.4<br>36.6<br>43.5<br>65.7<br>Ovis2-16B<br>92.4<br>59.6<br>73.8<br>29.0<br>88.5<br>78.1<br>87.9<br>70.7<br>59.5<br>46.8<br>72.5<br>67.9<br>69.6<br>35.6<br>45.0<br>65.1||||||||
||MiMo-VL-7B-SFT□<br>|92.9<br>62.3<br>74.2<br>27.5|87.3<br>82.7|86.1<br>72.0|57.5<br>52.5|75.1<br>67.9|||
||InternVL3-8B-Instruct♢<br>Llama-3.2-11B-Vision-Instruct<br>LlaVa-1.6-vicuna-13B<br>InternVL3-14B-Instruct<br>Ovis2-16B|91.1<br>60.1<br>72.9<br>28.6<br>87.3<br>45.0<br>66.5<br>8.8<br>79.0<br>42.9<br>65.6<br>7.9<br>90.7<br>61.0<br>74.5<br>28.9<br>92.4<br>59.6<br>73.8<br>29.0|84.4<br>78.5<br>79.6<br>73.8<br>77.9<br>74.7<br>87.1<br>83.1<br>88.5<br>78.1|85.4<br>67.1<br>79.6<br>63.6<br>76.6<br>52.5<br>86.2<br>75.1<br>87.9<br>70.7|59.7<br>52.4<br>50.4<br>44.6<br>24.9<br>39.2<br>58.2<br>51.4<br>59.5<br>46.8|76.4<br>66.7<br>76.9<br>60.7<br>70.7<br>58.1<br>77.1<br>66.4<br>72.5<br>67.9|||
||**30B**_≤_**Size**_<_**40B**<br>Qwen2.5-VL-32B-Instruct<br>94.9<br>66.4<br>75.0<br>22.9<br>86.4<br>83.8<br>87.7<br>75.9<br>62.7<br>51.0<br>74.8<br>71.7<br>65.8<br>32.6<br>46.4<br>66.5<br>LlaVa-1.6-34B<br>87.8<br>50.1<br>69.5<br>9.9<br>80.1<br>78.4<br>85.3<br>60.4<br>32.9<br>44.4<br>76.2<br>67.0<br>56.7<br>15.8<br>17.0<br>55.4<br>Ovis2-34B<br>93.6<br>63.1<br>72.4<br>31.0<br>85.6<br>79.1<br>88.9<br>71.9<br>61.6<br>47.1<br>72.0<br>70.7<br>74.0<br>31.4<br>49.5<br>66.1<br>InternVL3-38B-Instruct<br>95.0<br>61.7<br>76.3<br>30.2<br>89.9<br>84.6<br>88.0<br>79.8<br>63.1<br>54.1<br>77.5<br>72.8<br>75.2<br>34.0<br>47.6<br>68.7||||||||
||**70B**_≤_**Size**_<_**100B**<br>LLaVA-OneVision-72B-si<br>91.7<br>62.7<br>63.9<br>10.6<br>81.4<br>75.2<br>86.1<br>63.8<br>52.8<br>41.5<br>75.1<br>70.5<br>72.5<br>25.2<br>27.0<br>60.0<br>Qwen2.5-VL-72B-Instruct<br>**95.7**<br>67.0<br>82.4<br>24.9<br>90.3<br>87.5<br>89.5<br>80.6<br>68.4<br>58.1<br>74.4<br>74.8<br>77.6<br>39.8<br>46.2<br>70.5<br>InternVL3-78B-Instruct<br>95.3<br>68.9<br>80.7<br>32.4<br>90.7<br>86.2<br>89.6<br>77.9<br>65.5<br>57.9<br>78.0<br>**77.9**<br>79.3<br>44.3<br>48.7<br>71.6<br>Llama-3.2-90B-Vision-Instruct<br>91.4<br>58.6<br>76.7<br>19.5<br>86.0<br>82.3<br>86.3<br>74.4<br>60.1<br>49.4<br>78.5<br>68.8<br>58.7<br>23.4<br>35.0<br>63.3||||||||
||Qwen2.5-VL-72B-Instruct||||||||
||InternVL3-78B-Instruct|95.3<br>68.9<br>80.7<br>32.4|90.7<br>86.2|89.6<br>77.9|65.5<br>57.9|78.0<br>**77.9**|||
||Llama-3.2-90B-Vision-Instruct|91.4<br>58.6<br>76.7<br>19.5|86.0<br>82.3|86.3<br>74.4|60.1<br>49.4|78.5<br>68.8|||
|**Specialist**|MMCA-7B<br>UReader-7B<br>mPLUG-DocOwl2-8B<br>TextMonkey-10B<br>M3DocRAG-10B<br>CogAgent-17B<br>MDocAgent-39B|68.9<br>-<br>-<br>-<br>72.3<br>-<br>-<br>-<br>87.8<br>46.5<br>-<br>-<br>77.2<br>36.8<br>-<br>-<br>79.2<br>39.5<br>57.9<br>29.2<br>80.9<br>-<br>-<br>-<br>86.4<br>58.4<br>68.6<br>36.7|59.0<br>54.9<br>70.7<br>66.6<br>76.1<br>72.2<br>74.5<br>68.1<br>71.8<br>67.3<br>70.3<br>65.3<br>82.8<br>74.4|64.4<br>47.5<br>67.0<br>52.0<br>80.8<br>52.4<br>72.5<br>52.0<br>73.7<br>56.5<br>69.9<br>54.3<br>82.8<br>70.2|-<br>-<br>-<br>-<br>25.7<br>19.8<br>21.7<br>16.9<br>22.5<br>16.5<br>21.7<br>-<br>58.8<br>48.7|56.6<br>52.6<br>60.4<br>55.7<br>65.7<br>60.0<br>65.9<br>63.4<br>65.4<br>58.9<br>58.8<br>52.6<br>69.8<br>64.8|-<br>-<br>-<br>-<br>-<br>-<br>41.3<br>9.2<br>14.4<br>33.7<br>5.4<br>11.7<br>41.8<br>14.8<br>20.8<br>-<br>-<br>-<br>46.6<br>20.1<br>29.3|-<br>-<br>-<br>-<br>47.7<br>-<br>59.9|
|**Ours**|**MACT**<br>|**96.6**<br>**72.5**<br>**85.3**<br>**43.7**|**92.0**<br>**89.4**|**91.9**<br>**85.2**|**74.0**<br>57.2|78.5<br>**77.7**|**81.2**<br>41.8<br>**54.7**|**74.8**|
||**-Qwen2.5-VL-Series-24B**_△_|+2.0<br>+10.2<br>+9.1<br>+21.3|+5.2<br>+6.6|+4.4<br>+17.0|+13.6<br>+8.0|+4.5<br>+9.3|+13.4<br>+16.3<br>+13.9|+10.3|
||**MACT**<br>|94.4<br>**70.8**<br>**83.9**<br>**47.4**|**93.8**<br>**88.6**|**91.4**<br>**87.2**|**71.6**<br>**62.7**|79.2<br>76.1|**85.4**<br>**60.1**<br>**65.3**|**77.2**|
||**-MiMo-VL-Series-28B**□|+1.5<br>+8.5<br>+9.7<br>+19.9|+6.5<br>+5.9|+5.3<br>+15.2|+14.1<br>+10.2|+4.1<br>+8.2|+14.7<br>+13.2<br>+12.0|+9.9|
||**MACT**<br>|**96.1**<br>**69.8**<br>81.3<br>**46.5**|91.0<br>**87.7**|90.6<br>**85.0**|**75.3**<br>**61.0**|**81.3**<br>**80.1**|**83.8**<br>45.8<br>54.0|**75.3**|
||**-InternVL3-Series-28B**♢|+5.0<br>+9.7<br>+8.4<br>+17.9|+6.6<br>+9.2|+5.2<br>+17.9|+15.6<br>+8.6|+4.9<br>+13.4|+18.3<br>+17.0<br>+15.1|+11.5|



and RL pipeline for MACT. To understand visual inputs and better accomplish their labors, we choose VLMs for the _Aplan_ and _Aexe_ , while LLMs for the _Ajudg_ and _Aans_ . For all the four agents, we initiate with pretrained models, and we select three groups of small-parameter base models for different variants of MACT: **(1) Qwen2.5-VL series based** [2, 31]: Qwen2.5-VL-7B-Instruct and Qwen2.57B/3B-Instruct; **(2) MiMo-VL series based** [29, 30]: MiMo-VL-7B-SFT and MiMo-7B-SFT; **(3) InternVL3 series based** [37]: InternVL3-9B/8B/2B-Instruct. In the first SFT stage, we initially train a tuned 11B/7B/7B VLM on the selected document-based or non-document-based datasets, mixing data with or without CoT. It aims to enhance their visual understanding and reasoning abilities, providing more robust models for future multi-agent collaboration. Next, we fine-tune an 8B/7B/7B LLM on judgment labels generated via GPT-4o [5] and rule-based verifications. Finally, we fine-tune another 3B/3B/7B LLM on the outputs from preceding agents and ground-truths to enhance its summary ability. The fine-tuned VLM are 

applied as _Aplan_ and _Aexe_ respectively in the subsequent stage, while the two LLMs function as the _Ajudg_ and _Aans_ . In the second RL stage, we generate reward signals based on pretrained reward models and optimize our model via GRPO [18]. Process reward model VisualPRM [24] is utilized for the step-by-step reward signals of _Aplan_ and _Aexe_ , while Skywork-VL-Reward [25] is used for the rewards generation of _Ajudg_ and _Aans_ . 

## **3.2. Benchmarks** 

To comprehensively and objectively evaluate our model’s document-based capabilities, we selected 15 datasets encompassing four real-world document categories and two non-document general categories. The four document types are comprised of: **(1) Text-based:** DocVQA [13], DUDE [22], SlideVQA [21], MMLongBench-Doc [11]; **(2) Webpage-based:** VisualMRC [20], InfographicVQA [14]; **(3) Chart-based:** ChartQA [12], CharXiv [26]; **(4) Tablebased:** TableVQA-Bench [7], and TableBench [27]. To validate our document-centric paradigm without sacrific- 

6 

ing capabilities in general domains, two non-document types are involved: **(1) General:** ScienceQA [16], RealWorldQA [28]; **(2) Mathematical:** MathVista [10], MathVision [23], and MathVerse [36]. We adhere to the original training and testing splits and use the testing splits as evaluation benchmarks. Each instance in these datasets consists of a visual input sequence and a question, spanning difficulty levels from easy to hard and encompassing various question types for comprehensive evaluation. 

## **3.3. Evaluations** 

For other reproducible models, we utilize LMMs-Eval [34] for fair comparisons on both natively supported and our registered benchmarks. For most benchmarks, we employ GPT-4o [5] as a judge model to evaluate the correctness of the generated answers based on LMMs-Eval. For the remaining benchmarks, we utilize their original evaluation metrics, _e.g._ , ANLS and F1, as detailed in Tab. 8. 

## **4. Results and Discussions** 

## **4.1. Main Results** 

We evaluate MACT across 15 benchmarks: 10 documentbased benchmarks to assess its document understanding performance; 2 general benchmarks and 3 mathematical benchmarks are designed to verify that it retains full capabilities in non-document-centric and reasoning task settings. For comparison, we select state-of-the-art (SOTA) methods, encompassing both generalist and specialist models, and categorize them by parameter size. 

**Benchmark Results.** As shown in Tab. 1, the MACTMiMo-VL-Series-28B variant delivers the best average performance, followed by MACT-InternVL3-Series-28B and MACT-Qwen2.5-VL-Series-24B, which rank second and third, respectively. Despite having fewer than 30B parameters, MACT models outperform all comparative methods with under 100B parameters, as well as closed-source models. Notably, MACT-MiMo-VL-Series-28B achieves significant improvements of 5.6% and 5.9% over the topperforming open-source and closed-source models in terms of the average scores. Additionally, the three variants top 13 of the 15 benchmarks, with MACT-MiMo-VL-Series28B leading on seven. Notably, in MMLongBench-Doc, which features the longest visual context, and across the three mathematical reasoning benchmarks, MACT-MiMoVL-Series-28B outperforms the second-highest scorer by 7.1%, 10.6%, 5.9%, and 8.7%, respectively. These results highlight that document-based tasks, far from being monolithic constructs, are better suited to procedural scaling than to monolithic models with larger parameters, supporting flexible adaptation to functional entities and computational resource allocation. 

**Comparisons with Base Models.** As shown in Fig. 1, 

Table 2. Results of ablations on multi-agent collaboration, agentwise adaptive test-time scaling, and mixed reward modeling. 

||**MMLong TabBen MVision Avg.**|
|---|---|
|Monolithic<br>w/o Multi-Agent Collaboration<br>w/o Agent-Wise Adaptive Scaling<br>w/o Mixed Reward Modeling|32.5<br>50.8<br>32.4<br>66.2<br>24.7<br>44.4<br>26.0<br>58.6<br>34.9<br>50.8<br>34.5<br>71.1<br>38.3<br>54.2<br>36.7<br>71.4|
|**MACT**|**43.7**<br>**57.2**<br>**41.8**<br>**74.8**|



Table 3. Results of different combinations of agents. 

|_Aplan_+_Aexe Ajudg Aans_|**MMLong TabBen MVision Avg.**|
|---|---|
|�<br>�<br>�<br>�<br>�|36.3<br>48.5<br>34.9<br>68.4<br>42.2<br>56.7<br>40.2<br>73.9<br>36.6<br>48.6<br>35.4<br>68.8|
|�<br>�<br>�|**43.7**<br>**57.2**<br>**41.8**<br>**74.8**|



radar charts compare three variants of our proposed MACT framework with their base models and larger models from the same series. It is observed that MACT models significantly outperform their base models by 10.3%, 9.9%, and 11.5% on average across the 15 benchmarks. Besides, compared to their corresponding larger monolithic models, _i.e._ , Qwen2.5-VL-72B-Instruct and InternVL3-78BInstruct, our MACT-Qwen2.5-VL-Series-24B and MACTInternVL3-Series-28B achieve average performance gains of at least 6.6% and 3.7%, respectively, with more pronounced improvements observed on long-context and reasoning tasks. These results demonstrate that our method, equipped with procedural scaling paradigm, is superior to monolithic paradigm in the document-based tasks and also more general domains. 

## **4.2. Additional Quantitative Analysis** 

To further demonstrate the effectiveness of our proposed procedural scaling paradigm, we perform ablation experiments and analyses of scaling components. Experiments here are conducted using MACT-Qwen2.5-VL-Series-24B, and the training pipelines and training data are consistent with the main experiment. We select three lowestperforming benchmarks, distinguished by long visual contexts and reasoning-intensive demands, and the average performance across all benchmarks, aiming to quantify their impacts on both specialized capabilities. 

**Ablation Studies.** The ablation settings are as follows: (1) Monolithic: Use a monolithic model to directly execute tasks and output answers, with particular scaling and reward modeling retained. (2) w/o multi-agent collaboration: Use a single agent, which incorporates all prompts from our four proposed agents as a unified workflow, with our proposed scaling and reward strategies retained. (3) w/o agentwise adaptive test-time scaling: Retain the multi-agent col- 

7 

Table 4. Results of different setting of test-time scaling strategies. 

|**MMLong TabBen MVision Avg.**|**MMLong TabBen MVision Avg.**|
|---|---|
|No Scaling<br>Parallel Scaling<br>Sequential Scaling<br>Hybrid Scaling<br>Internal Scaling|34.9<br>50.8<br>34.5<br>71.1<br>39.1<br>55.2<br>37.5<br>72.0<br>41.3<br>54.7<br>40.0<br>72.4<br>39.8<br>55.5<br>39.4<br>73.0<br>41.8<br>55.9<br>41.6<br>72.3|
|**Agent-Wise Adaptive**|**43.7**<br>**57.2**<br>**41.8**<br>**74.8**|



Table 5. Results of different setting of reward modeling. 

||**MMLong TabBen MVision Avg.**|
|---|---|
|No Reward<br>Agent-Specifc Reward<br>Global Reward|38.3<br>54.2<br>36.7<br>71.4<br>42.8<br>56.3<br>40.5<br>72.7<br>34.1<br>51.6<br>35.1<br>70.2|
|**Mixed Reward**|**43.7**<br>**57.2**<br>**41.8**<br>**74.8**|



laborative framework and apply the mixed reward strategy directly, without test-time scaling. (4) w/o mixed reward modeling: Retain the multi-agent collaborative framework and agent-wise adaptive test-time scaling, without reward signals. As reported in Tab. 2, our document-centric procedural scaling paradigm, outperforming the monolithic system by 8.6%. Notably, the integration of all functionalities into a single model results in suboptimal performance, even outperforming the base model negatively, which indicates the inherent limitation of monolithic paradigms in document understanding and reasoning. Both the agentwise adaptive test-time scaling strategy and mixed reward modeling contribute to an average accuracy increase of at least 3%. Specifically, the former yields more remarkable gains on reasoning tasks in documents than average ones, with particularly striking improvements that underscore how adaptive scaling reduces cognitive overload. 

**Analysis of Multi-Agent Collaboration.** To assess the utility of each agent and the effectiveness their procedural collaboration, we perform additional experiments on different agent combinations, which are demonstrated in Tab. 3. The combination of _Aplan_ and _Aexe_ , as a basic procedural multi-agent system, results in a 3.9% average increase over the base model, while adding _Ajudg_ further boosts performance by 5.5%. _Aans_ additionally improves the scores by 0.9% on the average. Additionally, as in Fig. 4a, we compare our judgment agent strategy with the other two counterparts, _i.e._ , internal correction and agent for both judgment and correction. Our proposed strategy outperforms the others by at least 2.6% on average, while requiring an average of 0.3 fewer corrections. Moreover, our method achieves optimal correction results when the maximum number of corrections _Nc_ is set to 3, whereas the alternatives require _Nc_ to be set to 5, highlighting its greater accuracy and efficiency. It could also be observed that when setting a higher _Nc_ , the performance will not improve continuously, primar- 

**==> picture [237 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.31.6 74.872.2 75 74.5 74.8<br>70 2.3 69.5 68.8 74 73.9 74.1<br>73.4<br>60 Baseline Internal Correction 73 72.8 72.9 73.4 72.5 72.8<br>Agent for Judgment and Correction<br>Independent  Judgment Agent (Ours) 72 72.2 72.0 72.2<br>50 1.92.3 47.645.1 71 70.8 71.5 72.0 NNpe((NNpe = 16)= 16)<br>40 2.8 40.8 40.2 70.271.1 NNpe((NNpe = 1) = 1)<br>70<br>3 5 7 10 1 2 4 8 16<br>Avg/Max Number of Corrections Values of Np or Ne<br>(a) Analysis of Corrections (b) Impact of  Np and  Ne<br>Performance Performance<br>**----- End of picture text -----**<br>


Figure 4. (a) Line graph shows the impact of various maximum numbers of corrections, with solid and dashed lines denoting average values across all and three selected benchmarks, respectively. The bar charts show the average judgment numbers when the maximum is set to 3. (b) The line graphs represent the impacts of the number of generated plans _Np_ and candidate executions _Ne_ . 

ily because excessive corrections may confuse the agent and obscure correct answers. 

**Analysis of Agent-Wise Adaptive Test-Time Scaling and Mixed Reward modeling.** We perform experiments on our proposed agent-wise adaptive test-time scaling and mixed reward modeling. For the former, we benchmark our method with the other four effective test-time scaling strategies, as listed in Tab. 4. Here, we select budget forcing [15] as the internal scaling. Our agent-wise adaptive scaling strategy based on the computational resources and cognitive load of each agent, outperforms all four existing strategies and improves the average scores by at least 1.8%. As shown in Fig. 4b, higher values of _Np_ and _Ne_ increase the likelihood of finding the correct answers, which leads to more attempts at plans and execution steps. For the latter, we separately use the agent-wise reward and global reward for comparison, as reported in Tab. 5. Although the improvement from global reward is relatively limited, it avoids the selfishness of agents that only use agent-specific reward, yielding a further 1.3% increase in average scores. 

## **5. Conclusion** 

Our proposed MACT is an innovative multi-agent collaboration framework with adaptive test-time scaling strategy for visual document understanding and reasoning, shifting the monolithic scaling into procedural scaling. It comprises four collaborative agents that deconstructs the workflow of document analysis, achieving effective role division and collaboration. Notably, the judgment agent strategy outperforms existing self-correction mechanisms while requiring fewer corrections. Additionally, agent-wise adaptive testtime scaling and mixed reward modeling further reduce the cognitive overload adaptively. Extensive comparative experiments and supplementary analysis validate the superiority of our MACT. This procedural paradigm constitutes a meaningful exploration of multi-agent frameworks and adaptive test-time scaling strategies for document-based scenarios and tasks. 

8 

## **References** 

- [1] Anthropic. Claude 3.7 sonnet, 2024. 1 

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_ , 2025. 6 

- [3] Google DeepMind. Gemini 2.0 pro, 2025. 1 

- [4] Tanya Fitzgerald. Documents and documentary analysis. _Research methods in educational leadership and management_ , 3:296–308, 2012. 2, 3 

- [5] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. _arXiv preprint arXiv:2410.21276_ , 2024. 1, 6, 7 

- [6] Md. Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. MapCoder: Multi-agent code generation for competitive problem solving. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL) (Volume 1: Long Papers)_ , pages 4912–4944. Association for Computational Linguistics, 2024. 3 

- [7] Yoonsik Kim, Moonbin Yim, and Ka Yeon Song. Tablevqabench: A visual question answering benchmark on multiple table domains. _arXiv preprint arXiv:2404.19205_ , 2024. 6 

- [8] Benjamin Kutsyuruba. Document analysis. In _Varieties of qualitative research methods: Selected contextual perspectives_ , pages 139–146. Springer, 2023. 1, 2 

- [9] Bingxuan Li, Yiwei Wang, Jiuxiang Gu, Kai-Wei Chang, and Nanyun Peng. Metal: A multi-agent framework for chart generation with test-time scaling. _arXiv preprint arXiv:2502.17651_ , 2025. 3 

- [10] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _International Conference on Learning Representations (ICLR)_ , 2024. 7 

- [11] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems (NeurIPS)_ , 37:95963– 96010, 2025. 6 

- [12] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 2263–2279. Association for Computational Linguistics, 2022. 6 

- [13] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pages 2200–2209, 2021. 6 

- [14] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pages 1697–1706, 2022. 6 

- [15] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling. _arXiv preprint arXiv:2501.19393_ , 2025. 5, 8 

- [16] Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. Scienceqa: A novel resource for question answering on scholarly articles. _International Journal on Digital Libraries_ , 23(3):289–301, 2022. 7 

- [17] Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. Scaling test-time compute without verification or rl is suboptimal. _arXiv preprint arXiv:2502.12118_ , 2025. 5 

- [18] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_ , 2024. 6 

- [19] Simeng Sun, Yang Liu, Shuohang Wang, Dan Iter, Chenguang Zhu, and Mohit Iyyer. PEARL: Prompting large language models to plan and execute actions over long documents. In _Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL) (Volume 1: Long Papers)_ , pages 469–486. Association for Computational Linguistics, 2024. 3 

- [20] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 13878–13888, 2021. 6 

- [21] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 13636–13645, 2023. 6 

- [22] Jordy Van Landeghem, Rub`en Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Micka¨el Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 19528–19540, 2023. 6 

- [23] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. _Advances in Neural Information Processing Systems (NeurIPS)_ , 37:95095–95169, 2024. 7 

- [24] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. _arXiv preprint arXiv:2503.10291_ , 2025. 6 

- [25] Xiaokun Wang, Jiangbo Pei, Wei Shen, Yi Peng, Yunzhuo Hao, Weijie Qiu, Ai Jian, Tianyidan Xie, Xuchen Song, Yang Liu, et al. Skywork-vl reward: An effective reward model for multimodal understanding and reasoning. _arXiv preprint arXiv:2505.07263_ , 2025. 6 

- [26] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart 

9 

understanding in multimodal llms. _Advances in Neural Information Processing Systems (NeurIPS)_ , 37:113569–113697, 2024. 6 

- [27] Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Jiaheng Liu, Xeron Du, Di Liang, Daixin Shu, Xianfu Cheng, Tianzhen Sun, et al. Tablebench: A comprehensive and complex benchmark for table question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 25497–25506, 2025. 6 

   - [37] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. _arXiv preprint arXiv:2504.10479_ , 2025. 1, 2, 6 

- [28] xAI. Realworldqa: A benchmark for real-world spatial understanding, 2024. 7 

- [29] Bingquan Xia, Bowen Shen, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, Liang Zhao, et al. Mimo: Unlocking the reasoning potential of language model–from pretraining to posttraining. _arXiv preprint arXiv:2505.07608_ , 2025. 2, 6 

- [30] LLM-Core-Team Xiaomi. Mimo-vl technical report. _arXiv preprint arXiv:2506.03569_ , 2025. 6 

- [31] Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 1, 2, 6 

- [32] Zhiyu Yang, Zihan Zhou, Shuo Wang, Xin Cong, Xu Han, Yukun Yan, Zhenghao Liu, Zhixing Tan, Pengyuan Liu, Dong Yu, Zhiyuan Liu, Xiaodong Shi, and Maosong Sun. MatPlotAgent: Method and evaluation for LLM-based agentic scientific data visualization. In _Findings of the Association for Computational Linguistics: ACL 2024_ , pages 11789–11804. Association for Computational Linguistics, 2024. 3 

- [33] Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong Pasupat, Jure Leskovec, Percy Liang, Ed H Chi, and Denny Zhou. Large language models as analogical reasoners. In _International Conference on Learning Representations (ICLR)_ , 2024. 3 

- [34] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. _arXiv preprint arXiv:2407.12772_ , 2024. 7 

- [35] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, et al. A survey on test-time scaling in large language models: What, how, where, and how well? _arXiv preprint arXiv:2503.24235_ , 2025. 5 

- [36] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In _European Conference on Computer Vision (ECCV)_ , pages 169– 186. Springer, 2024. 7 

10 

