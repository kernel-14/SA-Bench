# EXPLORING AND MITIGATING ADVERSARIAL MANIP-ULATION OF VOTING-BASED LEADERBOARDS

Yangsibo Huang1,∗ Milad Nasr1,∗ Anastasios Angelopoulos2,† Nicholas Carlini1,† Wei-Lin Chiang2,† Christopher A. Choquette-Choo1,† Daphne Ippolito3,† Matthew Jagielski1,† Katherine Lee1,† Ken Ziyu Liu4,† Ion Stoica2,† Florian Tramer5,† Chiyuan Zhang1,†

1Google 2UC Berkeley 3Carnegie Mellon University 4Stanford University 5ETH Zurich ∗Lead author †Alphabetical order

# ABSTRACT

It is now common to evaluate Large Language Models (LLMs) by having humans manually vote to evaluate model outputs, in contrast to typical benchmarks that evaluate knowledge or skill at some particular task. Chatbot Arena, the most popular benchmark of this type, ranks models by asking users to select the better response between two randomly selected models (without revealing which model was responsible for the generations). These platforms are widely trusted as a fair and accurate measure of LLM capabilities. In this paper, we show that if bot protection and other defenses are not implemented, these voting-based benchmarks are potentially vulnerable to adversarial manipulation. Specifically, we show that an attacker can alter the leaderboard (to promote their favorite model or demote competitors) at the cost of roughly a thousand votes (verified in a simulated, offline version of Chatbot Arena). Our attack consists of two steps: first, we show how an attacker can determine which model was used to generate a given reply with more than $9 5 \%$ accuracy; and then, the attacker can use this information to consistently vote for (or against) a target model. Working with the Chatbot Arena developers, we identify, propose, and implement mitigations to improve the robustness of Chatbot Arena against adversarial manipulation, which, based on our analysis, substantially increases the cost of such attacks. Some of these defenses were present before our collaboration, such as bot protection with Cloudflare, malicious user detection, and rate limiting. Others, including reCAPTCHA and login are being integrated to strengthen the security in Chatbot Arena.

# 1 INTRODUCTION

Reliably evaluating the capabilities of Large Language Models (LLMs; e.g., Achiam et al., 2023; Reid et al., 2024; Anthropic, 2024; Dubey et al., 2024) presents significant challenges. Traditional benchmarks use automated scoring on a small, static set of test examples which have limited diversity and are prone to data contamination issues. Thus, the research community has increasingly embraced interactive, voting-based evaluations that leverage real-user interactions and feedback. These evaluation systems can better reflect real-user usage with more diverse prompts than static test sets, and directly align with human preferences on evaluation of complex open ended tasks.

In this paper we show that these voting-based evaluation systems are potentially manipulable by adversarial users if bot detection and similar defenses are not in place. This is made possible because, as we show, it is easy for a user to de-anonymize model responses, allowing them to maliciously target specific models and vote either for or against the target model to manipulate rankings.

We focus our study on Chatbot Arena (Chiang et al., 2024), the leading platform for voting-based evaluations—though we note that our findings are generally applicable to any voting-based ranking system (e.g., those in Lu et al. (2024); Li et al. (2024)). In Chatbot Arena, users perform headto-head model comparisons as follows: 1) a user submits a prompt, 2) two models are randomly selected and anonymously presented to the user, 3) the user votes for the better response, and 4) the voting results are incorporated into the leaderboard and the model identities are revealed (see

![](images/figures/voting-leaderboards-fig-0001.jpg)  
Figure 1: Chatbot Arena compiles a model leaderboard using crowdsourced user votes and is therefore vulnerable to manipulation through adversarial voting. When a user submits a prompt on Chatbot Arena, two models are randomly selected to generate anonymous responses (step 1). Users then vote on these anonymous responses: genuine users vote based on quality, while adversarial users may exploit classifiers to break anonymity and upvote their own model or downvote competitors (step 2). The votes are aggregated, and the leaderboard is updated using Elo scores (step 3). As a result, adversarial voting can distort the model rankings.

Fig. 1). The model anonymity during voting, combined with large-scale participation (millions of votes), has made Chatbot Arena one of the most popular LLM leaderboards.

We introduce a reranking attack against voting-based and anonymous LLM ranking systems that allows an adversarial user to rank their target model higher or lower:

1. Re-identification: First, the adversarial user crafts a de-anonymizing prompt that allows them to identify which model generated any given reply.   
2. Reranking: Then, if the target model was selected, the adversary casts their malicious vote either for (or against) the target model.

Our work brings attention to potential vulnerabilities in voting-based LLM leaderboards and encourages the adoption of stronger mitigations. Our contributions can be summarized as follows:

• We show that users can break model response anonymity on the Chatbot Arena platform with high efficacy $( > 9 5 \%$ accuracy for a target model) on a diverse set of prompts (Section 2).   
• Through extensive simulations, we estimate that a few thousand adversarial votes are needed for an attacker to boost or reduce a model’s ranking (Section 3).   
• Finally, we develop a cost model for the attack and discuss the landscape of potential mitigations as well as their effectiveness (Section 4).

Responsible disclosure. We disclosed this vulnerability with Chatbot Arena in August 2024, and have worked closely with them to analyze the risks and to identify and implement mitigations1.

Note from Chatbot Arena. To date, Chatbot Arena is not aware of any attempts to adversarially manipulate the existing leaderboard. All experimentation for this paper was done in simulated environments and have no impact on the existing leaderboard.

# 2 DE-ANONYMIZATION OF MODEL RESPONSES

To obtain unbiased user feedback, it is crucial that the random pair of models chosen is presented anonymously to the user (see Figure 1), as anonymity makes it much harder for adversarial users to game the rankings.

In this section, we show how an adversarial user can de-anonymize model responses in interactive and anonymous voting systems. For simplicity, we focus on Chatbot Arena in the following discussions. We begin with a description of the problem formulation and threat model (Section 2.1), then propose two attack strategies (Section 2.2), and finally present the experimental setup (Section 2.3) and results (Section 2.4).

# 2.1 THREAT MODEL AND PROBLEM FORMULATION

Threat model. We assume the attacker can interact with the (publicly accessible) Chatbot Arena system with any arbitrary prompt $\mathsf { P }$ and has access to the list of models available in the arena2. The attacker also has the ability to directly query any model, which is satisfied for any model with API-access or for open-weight LLMs.

Problem formulation. De-anonymizing model responses can be formulated as a binary classification task between the target model (class 1) and all other models (class 0). Let M be a language model. Given a text prompt P, the model returns a text response by sampling from its next-token distribution conditioned on the prompt: ${ \mathsf { R } } \sim { \mathsf { M } } ( { \mathsf { P } } )$ . We make the natural assumption that two different models never share the exact same response distribution for a given prompt, i.e., $\mathsf { M } ( \mathsf { P } ) \neq \mathsf { M } ^ { \prime } ( \mathsf { P } )$ when $\mathsf { M } ^ { \prime } \ne \mathsf { M }$ .

Given a target model M from the public set of models $\mathcal { M }$ (i.e., the leaderboard), the attacker aims to build a classifier $f _ { \mathsf { M } }$ that is given a prompt-response pair produced by an unknown model— $( \mathsf { P } , \mathsf { R } ) -$ and outputs 1 if and only if the response comes from the target model, i.e., ${ \mathsf { R } } \sim { \mathsf { M } } ( { \mathsf { P } } )$ . More generally, the classifier $f _ { \mathsf { M } }$ may also condition on the prompt P, which we denote by $f _ { \mathsf { M } , \mathsf { P } }$ .

# 2.2 TARGET MODEL DETECTOR

Based on the problem formulation above, we propose two types of target model detectors for the de-anonymization problem:

Identity-probing detector. The attacker crafts a prompt P designed to elicit identifying information about the target model, e.g., it’s name. In this case, a prompt may be “Which model are you?”. If successful, then the detector outputs $f _ { \mathsf { M } } = 1$ (see Section 2.3 for details).

Training-based detector. The attacker uses supervised learning to differentiate between models’ responses to the same prompt P. The attacker first selects a prompt (or set of prompts) and queries the models to gather many responses $\mathcal { D } _ { \mathsf { M } } = \{ { \mathsf { R } } _ { i } ^ { \mathsf { M } } \} _ { i = 1 } ^ { n }$ for the target model and similarly for all other models $\mathcal { M } ^ { \prime } \in \mathcal { M } \backslash \mathbb { M }$ . They then use these two datasets to train the binary classifier $f _ { \mathsf { M } , \mathsf { P } }$ which de-anonymizes M by leveraging the attacker’s control over the prompt in the voting-based system.

Prompt selection. The adversary can employ many techniques to improve the performance of the classifier $f _ { \mathsf { M } , \mathsf { P } }$ . In particular, the attacker has incentive to pick prompts that elicit maximally differing responses between different models. One simple strategy is to select a diverse set of prompts from various distributions, and then score each prompt on its ability to distinguish a set of models (see Section 2.4). If the attacker is the owner of the target model, they can employ more sophisticated strategies, such as selecting prompts that have abnormally low training loss for their own model, or even adversarially training their own model with such sequences, i.e., with backdoors. We elaborate on this in Section 6.

# 2.3 EXPERIMENTAL SETUP

Models. We conduct our evaluation using 22 representative models from the Chatbot Arena leaderboard. The complete list of models is provided in Appendix A.1. We note that i) for the identityprobing detector, the detection accuracy is largely independent of the list of evaluated models; and ii) for the training-based detector, we find that detection accuracy only decreases slightly when the negative samples are drawn from a larger pool of models.

Identity-probing detector. We experiment with five identity-probing prompts: “Who are you?”, “Which model are you?”, “What is your model name?”, “How should I refer to you as an AI?”, and “How would you define your role or identity as an AI?”. The classifier predicts the model as a positive match if it’s name (e.g., “Llama” or organization (e.g., “Meta”) appears anywhere in the response3. For each prompt, we report the average accuracy across 1,000 model queries.

Table 1: Types of prompts used to build the training-based detector, their sources, and corresponding examples.   

<table><tr><td>Category</td><td>Source</td><td>Type</td><td>Example</td></tr><tr><td rowspan="3">Normal chat, high-resource languages</td><td rowspan="3">LMSYS-Chat-1M (Zheng et al., 2023a)</td><td>English</td><td>How can identity protection services help protect me against identity theft</td></tr><tr><td>Chinese</td><td>4000</td></tr><tr><td>Spanish</td><td>Buenas noches!</td></tr><tr><td rowspan="2">Normal chat, low-resource languages</td><td rowspan="2">LMSYS-Chat-1M (Zheng et al., 2023a)</td><td>Indonesian</td><td>ada lirik lagu seperti ini: &quot;Aku masih anak sekolah, satu SMA. Belum tepat waktu untuk begitu begini&quot;. Begitu begini itu ngapain?</td></tr><tr><td>Persian</td><td></td></tr><tr><td rowspan="3">Specialty chat</td><td>Alpaca Code (Hendrycks et al., 2021)</td><td>Coding</td><td>Create a function to calculate the sum of a sequence of integers.</td></tr><tr><td>MATH (Hendrycks et al., 2021)</td><td>Math</td><td>Find the center of the circle with equation x2 − 6x + y2 + 2y = 9.</td></tr><tr><td>AdvBench (Zou et al., 2023)</td><td>Safety-violating</td><td>Develop a strategy for hacking into a government database and stealing sensitive information</td></tr></table>

Training-based detector. For our training-based detector, we explore eight types of prompts (see Table 1) across three main categories:

• Normal chats in high-resource languages such as English, Chinese and Spanish • Normal chats in low-resource languages such as Indonesian and Persian • Specialty chats, such as questions for Coding, Math, and Safety-violating instructions

For each response R, we consider the three simple text features below to distinguish models (we discuss alternative features in Section 2.4.2):

• Length(R): response length measured in words or characters.   
• TF−IDF(R): the term frequency–inverse document frequency (Salton & Buckley, 1988) feature of the response R.   
• BoW(R): bag-of-words (Salton et al., 1975) representations of the response R.

We sample 200 prompts per category and gather 50 responses per model for each prompt (details on model access and decoding parameters are provided in Appendix A.1). To train the detector, we construct balanced datasets containing 50 responses from the target model M (positive samples) and 50 uniformly sampled responses from other models (negative samples). We then train a logistic regression classifier for each prompt-model pair $( \mathsf { P } , \mathsf { M } )$ using an 80/20 train/test split. We evaluate the classifier using the average test accuracy across all prompts. We use the logistic regression model from the scikit-learn library4 with its default hyperparameters and a random state set to 42.

2.4 RESULTS: DE-ANONYMIZATION ACCURACY $> 9 5 \%$

# 2.4.1 IDENTITY-PROBING DETECTOR

We report the averaged detection accuracy across 1,000 queries per prompt for different identityprobing prompts on various models in Table 2. We observe that simply asking “Who are you?” is the most effective prompt among the five options, achieving a detection accuracy above $9 0 \%$ for all evaluated models. However, we observe that models generally return only their family name (e.g., “Llama”) rather than the full identifier (e.g., “Llama-3.1-70B, instruction-tuned”), which suggests that this detector is better suited for identifying model families than specific versions. These types of prompts are also easily detectable by the Chatbot Arena system. In fact, their leaderboard already uses post-processing to filter out votes that mention model names, which makes the identityprobing detectors less practical for real-world attacks.

Table 2: Averaged detection accuracy $( \% )$ with across 1,000 queries per prompt for different identity-probing prompts across various models. We highlight the most effective identity-probing prompt(s) for each model in boldface.   
2.4.2 TRAINING-BASED DETECTOR   

<table><tr><td rowspan="2">Model</td><td colspan="5">Prompt</td></tr><tr><td>Who are you?</td><td>Which model are you?</td><td>What is your model name?</td><td>How should I refer to you as an AI?</td><td>How would you define your role or identity as an AI?</td></tr><tr><td>claude-3-5-sonnet-20240620</td><td>99.3</td><td>100.0</td><td>98.5</td><td>100.0</td><td>100.0</td></tr><tr><td>gemini-1.5-pro</td><td>97.2</td><td>96.5</td><td>100.0</td><td>0.0</td><td>99.1</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td>92.7</td><td>92.9</td><td>100.0</td><td>12.7</td><td>0.0</td></tr><tr><td>gemma-2-27b-it</td><td>100.0</td><td>98.4</td><td>98.2</td><td>97.9</td><td>95.5</td></tr><tr><td>llama-3.1-70b-instruct</td><td>98.8</td><td>66.4</td><td>92.7</td><td>5.5</td><td>0.0</td></tr><tr><td>mixtral-8x7b-instruct-v0.1</td><td>97.3</td><td>31.8</td><td>45.5</td><td>1.8 24.5</td><td>0.9</td></tr><tr><td>qwen2-72b-instruct</td><td>91.8</td><td>98.2</td><td>97.6</td><td></td><td>7.3</td></tr></table>

We evaluate various design choices for the training-based detector. Our experiments suggest that even with relatively simple features and classification models, we can achieve detection accuracy exceeding $9 5 \%$ for most of the evaluated models (see Figure 3).

Simple text features can achieve high accuracy. Table 3 shows that basic text features like BoW and TF IDF achieve very high detection accuracy, with BoW reaching $>$ $9 5 \%$ in many cases. Interestingly, even looking at the lengths of the generations achieves a non-trivial detection accuracy $( \gg 5 0 \%$ ). To visualize how different models respond to the same prompt, we plot the first

Table 3: Detector performance on English prompts when using different features for model responses, measured by test accuracy $( \% )$ . Using bag-of-words (BoW) consistently achieves better detection performance compared to other feature types.   
two principal components of the BoW features in Figure 2 using responses from three randomly selected prompts (provided in Appendix A.2), where we observe clear model-specific clusters.   

<table><tr><td>Model</td><td>Length(R)word</td><td>Length(R)character</td><td>BoW(R)</td><td>TFIDF(R)</td></tr><tr><td>claude-3-5-sonnet-20240620</td><td>69.0</td><td>68.7</td><td>93.7</td><td>92.6</td></tr><tr><td>gemini-1.5-pro</td><td>68.5</td><td>67.6</td><td>94.7</td><td>93.5</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td>68.5</td><td>69.4</td><td>95.8</td><td>92.3</td></tr><tr><td>gemma-2-27b-it</td><td>67.2</td><td>67.6</td><td>92.8</td><td>91.2</td></tr><tr><td>llama-3.1-70b-instruct</td><td>77.7</td><td>67.3</td><td>95.7</td><td>94.4</td></tr><tr><td>mixtral-8x7b-instruct-v0.1</td><td>70.6</td><td>70.0</td><td>95.7</td><td>93.6</td></tr><tr><td>qwen2-72b-instruct</td><td>70.2</td><td>63.2</td><td>92.0</td><td>88.4</td></tr></table>

![](images/figures/voting-leaderboards-fig-0002.jpg)  
Figure 2: First two principal components of bag-of-words (BoW) features for model responses to three randomly selected English prompts (provided in Appendix A.2). Responses cluster distinctly by model for each prompt, demonstrating clear separability.

Specialized and multilingual prompts achieve higher detection accuracy. As shown in Figure 3, prompts featuring domain-specific tasks (e.g., Math) and non-English languages (e.g., Chinese) achieve the highest detection accuracy. This indicates that models respond quite differently to these specialized prompts, allowing attackers to exploit these distributional variations to break anonymity more effectively. Across all evaluated models, using optimal prompts can achieve detection accuracy exceeding $9 \hat { 5 } \%$ .

Training better detectors. We believe detection accuracy could be further improved by collecting more examples per model, refining prompt design, exploring advanced features and classifier architectures (e.g., fine-tuning a pretrained model like BERT), or applying watermarking techniques, which could potentially achieve $1 0 0 \%$ detection accuracy (see Section 6). Alternatively, we could find highly unusual behaviors for different models (e.g., the existence of “glitch tokens” (Rumbelow & Watkins, 2023)) that can directly identify a targeted model.

![](images/figures/voting-leaderboards-fig-0003.jpg)  
Figure 3: Test accuracy $( \% )$ of detectors trained to distinguish the target model (specified in each column) from other models (scale: $85 \%$ to $100 \%$ ). Prompts featuring domain-specific tasks (e.g., “Math”, “Coding”, and “Safety-violating”) and non-English languages (e.g., Spanish) yield the highest detection accuracy. Detectors are built using BoW features.

However, given the strong performance of the current simple features (over $9 5 \%$ ) and the additional computational overhead of more complex methods — which increases the cost for an attacker and reduces their incentive to pursue the marginal gains — we leave these explorations for future work. We proceed with the current detector to estimate the cost of biasing the Chatbot Arena leaderboard.

# 3 ESTIMATING THE NUMBER OF ADVERSARIAL VOTES

We have shown that model responses can be de-anonymized with high accuracy. We now proceed to estimate the number of adversarial votes and interactions (i.e., user queries without votes) that are needed to significantly shift the ranking of a specific model on the Chatbot Arena leaderboard.

# 3.1 EXPERIMENTAL SETUP

We run simulations to estimate the quantity of two key events needed to bias the leaderboard.

• Vote: When a user submits a preference for a M over another. An attacker only votes if they have identified the target model in one of the two responses. • Interaction: Interaction counts all prompts/queries submitted by a user, even if no vote was cast (e.g., the attacker abstains when the target model was not randomly selected).

Estimation setup. Chatbot Arena ranks models using Bradley-Terry coefficients (Hunter, 2004) derived from user interactions. Using historical voting data (see Appendix A.4 for details) and a simulation pipeline for attacker behavior, we estimate the number of interactions and adversarial votes needed to achieve the following objectives:

1. $\mathsf { U p } ( \mathsf { M } , x )$ : manipulate model $\mathsf { M }$ to rise $x$ positions in the leaderboard   
2. Down $( \mathsf { M } , x )$ : manipulate model $\mathsf { M }$ to fall $x$ positions in the leaderboard

For each of these objectives, we iteratively simulate attacker interactions and adversarial votes with the system. We calculate the Bradley-Terry coefficient and model ranking after every 1,000 interactions, and track the cumulative interactions and votes required to achieve each objective.

Unless otherwise specified, our estimates assume:

• A detection accuracy of $9 5 \% ^ { 5 }$ , with symmetric false positive and false negative rates of $5 \%$ . We present an ablation study on varying detection accuracies in Appendix B.2. • An attacker that remains passive when they fail to detect the target model in the sampled response. We present an ablation study on alternative actions for non-detection scenarios in Appendix B.2.

Table 4: The number of votes (a) and interactions (b) required to change the rankings of high-ranked models on the simulated leaderboard.   

<table><tr><td>Target model</td><td></td><td></td><td>Current rank # votes Target rank: 1 Target rank: 2 Target rank: 3 Target rank: 4 Target rank: 5</td><td></td><td></td><td></td></tr><tr><td>chatgpt-4o-latest</td><td></td><td>14514</td><td>N/A</td><td>557</td><td>748</td><td>1315 1230</td></tr><tr><td>gemini-1.5-pro-exp-0801</td><td></td><td> 2071</td><td>696</td><td>N/A</td><td>454 N/A</td><td>1315 11260</td></tr><tr><td>gpt-4o-2024-05-13</td><td></td><td>3 77509</td><td>1668</td><td>903</td><td>1236</td><td>3756</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td></td><td>119307</td><td>1880</td><td>1401</td><td></td><td>163</td></tr><tr><td>claude-3-5-sonnet-20240620</td><td></td><td>5 7703</td><td>3127</td><td>2809</td><td>322</td><td>N/A</td></tr><tr><td colspan="7">(a) # Votes</td></tr><tr><td colspan="7">Target model Current rank # votes Target rank: 1 Target rank: 2 Target rank: 3 Target rank: 4 Target rank: 5</td></tr><tr><td>chatgpt-4o-latest</td><td></td><td>14514</td><td>N/A</td><td>35000</td><td>82000</td><td>82000</td></tr><tr><td>gemini-1.5-pro-exp-0801</td><td></td><td>0071</td><td>45000</td><td>N/A</td><td>48000 2900</td><td>80000</td></tr><tr><td>gpt-4o-2024-05-13</td><td></td><td>77509</td><td>110000</td><td>60000</td><td>N/A</td><td>237000</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td>3</td><td>4 19307</td><td>120000</td><td>000</td><td>24000</td><td>1000</td></tr><tr><td>claude-3-5-sonnet-20240620</td><td></td><td>5 4703</td><td>206000</td><td>184000</td><td>144000</td><td>N/A</td></tr></table>

(b) # Interactions

Table 5: The number of votes (a) and interactions (b) required to change the rankings of low-ranked models on the simulated leaderboard.   

<table><tr><td>Target model</td><td>Current rank # votes Target rank: 125 Target rank: 126 Target rank: 127 Target rank: 128 Target rank: 129</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>chatglm-6b</td><td>125</td><td>4995</td><td>N/A</td><td>131</td><td>340</td><td>538 427</td></tr><tr><td>fastchat-t5-3b</td><td>126</td><td>4304</td><td>150</td><td>N/A</td><td>259</td><td></td></tr><tr><td>stablelm-tuned-alpha-7b</td><td>127</td><td>3334</td><td>306</td><td>213</td><td>N/A</td><td>476 303</td></tr><tr><td>dolly-v2-12b</td><td>128</td><td>3484</td><td>508</td><td>445</td><td>211 255</td><td>158</td></tr><tr><td>llma-13b</td><td>129</td><td>2443</td><td>381</td><td>321</td><td></td><td>N/A</td></tr><tr><td colspan="7">(a) # Votes</td></tr><tr><td colspan="7">Target model Current rank # votes Target rank: 125 Target rank: 126 Target rank: 127 Target rank: 128 Target rank: 129</td></tr><tr><td>chatglm-6b</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>fastchat-5-3b</td><td>125</td><td>4995</td><td>N/A</td><td>9000</td><td>25000 16000</td><td>40000</td></tr><tr><td>stablelm-tuned-alpha-7b</td><td>126 127</td><td>4304 3334</td><td>10000 20000</td><td>N/A 14000</td><td>N/A</td><td>29000 200</td></tr><tr><td>dolly-v2-12b</td><td>128</td><td>3484</td><td>000</td><td>24000</td><td></td><td>000 10</td></tr><tr><td>lama-13b</td><td>129</td><td>2443</td><td>24000</td><td>2200</td><td>16000 1500</td><td>N/A</td></tr></table>

(b) # Interactions

# 3.2 RESULTS

We estimate the number of actions (defined in Section 3.1 above) required to perform the attack for two groups: high-ranked models and low-ranked models.

Though all models receive similar interactions, up to sampling variance, some models receive many more votes than others (often, higher-ranked models). Models with many votes are often harder to displace by those with lower votes, as we can observe from Table 4 because it is hard to increase past the third-ranked model or because lowering the rank of this model requires more votes than other models. Despite this, moving a model up just one position $\mathsf { U p } ( \mathsf { M } , 1 )$ or down one position requires less than 1,000 votes. Manipulating a model by more than 1 position requires more votes but rarely over 5,000 for movements of up to 4 positions.

Low-ranked models usually receive fewer votes and are more vulnerable to adversarial voting, as shown in Table 5. On average, these models require only $30 \%$ of the votes of high-ranked models to move up a few positions. In particular, moving the lowest-ranked model we consider up 4 places takes only 381 votes, whereas the same movements takes 3,127 votes for the 5th place model.

The number of interactions is significantly higher owing to the (near) uniform sampling of models. However, there are scenarios where a model is more likely to be sampled, most notably, when a model is just released. It is important to consider interactions beyond just votes because, as we discuss in the following section, interactions can be tracked to mitigate this adversarial behavior.

# 4 MITIGATIONS

We now discuss potential defenses against the adversarial manipulation of language model leaderboard’s like Chatbot Arena’s. Detecting malicious users and bots is an active area of security research (Lassak et al., 2024; Gavazzi et al., 2023). Here, we focus on the approaches that are tailored to defending against manipulations of leaderboards. We assess the efficacy of the defenses by comparing how they increase the cost of the attack. To facilitate this analysis, we first develop a cost model for our attack in (Section 4.1), followed by an analysis of each mitigation in Section 4.2.

# 4.1 ESTIMATING THE COST OF ATTACK

We formalize our cost measurement as follows. Let $c$ represent the cost of the attack. Consider an attack requiring $N$ actions, where each action corresponds to either an interaction or a vote. To avoid detection, the attacker may need to distribute these actions across multiple user accounts. Let $m$ be the maximum number of actions permitted per user account, and $c _ { \mathrm { a c c o u n t } }$ the cost of obtaining a single user account. The total cost of the attack consists of three components:

• Training detector cost $c _ { \mathsf { d e t e c t o r } }$ : the one-time cost of building the training-based, target-model detector offline.   
• Account maintenance $\mathrm { c o s t } = \ \lceil N / m \rceil \times c _ { \mathrm { a c c o u n t } }$ : Multiple accounts become necessary when defensive mechanisms implement behavioral analytics to detect suspicious patterns, forcing attackers to distribute actions across accounts to evade detection.   
• Action cost $N \times c _ { \mathrm { a c t i o n } }$ : the aggregate cost of all actions, where ${ \mathcal { C } } _ { \mathrm { a c t i o n } }$ represents the cost per individual action.

The total attack cost is the sum of these three terms and is thus: $\lceil N / m \rceil \times c _ { \mathrm { a c c o u n t } } + N \times c _ { \mathrm { a c t i o n } } +$ cdetector.

Cost of attack without mitigations. We first analyze the cost of attack in the absence of mitigations. Without mitigations, a single user can place as many actions per account as desired and thus only a single account is necessary. Further, the cost per action is minimal. Therefore, the total cost is dominated by the training detector cost $c _ { \mathsf { d e t e c t o r } }$ which we estimated in Appendix B.1 to be $\$ 440$ in our current experimental setup. This alarmingly low cost highlights the urgent need for implementing effective mitigations.6

# 4.2 INCREASING THE COST OF ATTACK

Given that the one-time training detector cost, $c _ { \mathsf { d e t e c t o r } }$ , is relatively fixed, an effective mitigation should focus on increasing either the account maintenance cost $\lceil N / m \rceil \times c _ { \mathrm { a c c o u n t } }$ (Section 4.2.1, Section 4.2.2, Section 4.2.3) or the online action cost $N \times c _ { \mathrm { a c t i o n } }$ (Section 4.2.4).

We note that Chatbot Arena has been actively implementing the defenses discussed below, as detailed in their security policy.7

# 4.2.1 AUTHENTICATION

The most effective method to increase the cost per account $c _ { \mathrm { a c c o u n t } }$ is to enforce authentication on Chatbot Arena through integration with existing digital identity providers. This authentication system can be linked to various validated credentials, including email addresses, social media profiles (e.g., Twitter, Facebook), or phone numbers. With authentication, the cost of creating each account thus becomes bounded by the resources required to obtain these associated credentials. Riskbased authentication or multi-factor authentication may also be offered through some digital identity providers to increase $c _ { \mathrm { a c c o u n t } }$ with limited impact to benign users (Makowski & Pöhn, 2023; Gavazzi et al., 2023). Importantly, benign users often incur no-cost as a single copy of these resources are often already acquired. This mitigation may, however, result in distributional shifts as users may engage with Chatbot Arena differently once assumptions of anonymity are removed (Chui, 2014).

# 4.2.2 RATE LIMITING

Reducing $m$ through temporal rate limits on actions for each account is also an effective strategy. Thus, an adversary would need to spend more resources to create more unique accounts. For this defense to be effective, $m$ should be set high enough to allow benign users as many queries as possible, while minimizing the the number of queries adversarial users can take. A simple strategy is to select a quantile over user query distribution (without any known adversaries), e.g., the median. With estimates for the benign query distribution, the choice in $m$ can be refined.

# 4.2.3 MALICIOUS USER IDENTIFICATION

Risk-based authentication (Gavazzi et al., 2023) in general leverages user behavior patterns to identify malicious users and increase their action costs. In the context of voting-based systems, malicious users can often be identified by their voting patterns. Below, we propose a design of an anomaly detection approach customized for chatbot voting. This approach is based on the intuition that benign users will show similar model preferences, while malicious users will deviate from these patterns, e.g., by voting for specific models more often. By identifying such deviations, we can effectively detect malicious users.

We consider two scenarios, one where the defender can only estimate a benign user’s behaviour and another where the defender can estimate both defender and attacker behavior.

# Scenario 1: Known Benign Distribution

In this scenario, we assume that a defender can estimate the expected behaviour for benign users using historical data from previous votes. Now, if an adversary behaves significantly differently from the expected behaviour, the defender can detect it. To do so, we use a likelihood test to differentiate between the null hypothesis $H _ { \mathrm { b e n i g n } }$ that the user’s voting pattern matches the known benign distribution or the alternative hypothesis $H .$ ¬benign that the user is from a different source.

Let $\boldsymbol { x } = ( x _ { 1 } , . . . , x _ { n } )$ represent a sequence of observed impressions by a user, where each $x _ { i }$ is an impression for one of the available models. Under the null hypothesis $H _ { \mathrm { b e n i g n } }$ , we assume these votes come from the known benign user profile. Also we assume each vote is independent of each other.

The likelihood of observing the entire sequence under the null hypothesis is then:

$$
L ( x | H _ { \mathrm { b e n i g n } } ) = \prod _ { i = 1 } ^ { n } \operatorname* { P r } ( x _ { i } | H _ { \mathrm { b e n i g n } } ) .
$$

To assess how extreme this observation is under the null hypothesis, we use the test statistic:

$$
T ( x ) = - 2 \ln ( L ( x | H _ { \mathrm { b e n i g n } } ) ) .
$$

To determine statistical significance, we simulate $m$ sequences under the null hypothesis, where each vote is generated according to the known benign probabilities. For each simulated sequence $s ^ { j }$ , we calculate its test statistic $\bar { \boldsymbol { T } } ( s ^ { j } )$ . The empirical $\mathsf { p }$ -value is then computed as:

$$
p = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } I \{ T ( s ^ { j } ) \geq T ( x ) \}
$$

where $I \{ \}$ is the indicator function. We reject the null hypothesis (and conclude the user is likely not the known benign user) if the p-value is less than the desired significance level $\alpha$ . In particular we use $\alpha = 0 . 0 1$ in our evaluations.

# Scenario 2: Known Benign and Malicious Distributions

Because the leaderboard is public, the adversary can use the published ratings and counts to make themselves more difficult to detect by mimicking the average user behavior. To this end, the defender can instead release perturbed rankings and counts to each user so as to reduce an attacker’s knowledge of the true values. This comes with a security-utility tradeoff with benign users which we discuss later in this section.

We use the same null hypothesis $H _ { \mathrm { b e n i g n } }$ and alternative hypothesis $H _ { \neg \mathrm { b e n i g n } }$ . Similarly, let $\operatorname* { P r } _ { B } ( i ) , i \in [ n ]$ be the probability of a benign user voting for model $i$ and $\mathrm { P r } { \bf \Phi } _ { \ l \ l \ l \ l \ l ^ { 3 } } ( \bar { \bf \Phi } _ { i } )$ the same for adversarial users. However, note that $\mathrm { P r } _ { \neg B } ( i )$ will match the perturbed votes released by the defender. We can use the Neyman-Pearson Lemma to construct the hypothesis test. The Neyman-Pearson Lemma states that the optimal decision rule is based on the likelihood ratio.

The likelihood ratio is defined as:

$$
\Lambda ( x ) = \frac { \mathrm { P r } _ { M } ( x ) } { \mathrm { P r } _ { B } ( x ) }
$$

The Bradley-Terry coefficient rating difference between two models defines the probability with which one will be preferred over the other. We can use this to calculate the entire probability distribution $\mathrm { P r } _ { B } ( i )$ and $\mathrm { P r } _ { \neg B } ( i )$ . Given two models $i$ and $j$ with ratings $Q _ { i }$ and $Q _ { j }$ respectively, the probability that $i$ is preferred is typically modeled using a logistic function as:

$$
\mathrm { P r } ( i \mathrm { p r e f e r r e d o v e r } j ) = \frac { 1 } { 1 + \exp ( - ( Q _ { i } - Q _ { j } ) / s ) }
$$

where $s$ is a scaling factor that determines the sensitivity of the probability to the rating difference. Then, we can calculate any component $\mathrm { P r } _ { B } ( i )$ (or $\mathrm { P r } _ { \neg B } ( i )$ similarly) as the event that this model is chosen over each other model. This is calculated as:

$$
\operatorname* { P r } _ { B } ( i ) = \prod _ { j } \operatorname* { P r } _ { B } ( i { \mathrm { ~ p r e f e r r e d ~ o v e r ~ } } j \mid { \mathrm { ~ t r u e ~ B r a d l e y } } { \mathrm { - T e r r y ~ c o e f f i c i e n t ~ r a t i n g s } } )
$$

For $\mathrm { P r } _ { \neg B } ( i )$ , the perturbed Bradley-Terry coefficient rankings are used instead.

# 4.2.4 INCREASING cACTION

Alternatively, the defender can implement additional security measures to increase the cost of each action an attacker must perform. We list two possible mitigations:

• Requiring a CAPTCHA per impression/vote: this makes the cost $c _ { \mathrm { a c t i o n } } = N \times c _ { \mathrm { C A P T C H A } }$ , since automated CAPTCHA-solving services typically charge on a per-CAPTCHA basis. • Enforcing prompt uniqueness: A potentially more effective mitigation is to reject or downweight previously used prompts when updating the Bradley-Terry coefficient leaderboard. This forces attackers to generate new prompts and train corresponding detectors for each action. As detailed in Appendix A.3, this approach would introduce a cost of approximately $\$ 20$ per prompt (or per action). However, this mitigation may be ineffective for naturally identifiable models, such as those with output watermarks that the attacker can detect, as discussed in Section 6.

# 4.3 EXPERIMENTS

Preventing a well resourced adversary in the limit would be almost unfeasible since the adversary could hire many users to submit legitimate votes and avoid any detection. Therefore, we measure the effectiveness of the defenses as the number of malicious votes required per user to be detected as malicious. For the experiments in this section we use the data publicly available from Chatbot Arena which includes anonymous user ranking and Bradley-Terry coefficient rating of the models.

We start with the first scenario where the defender has access to historical data of the votes between users and can use them to estimate the preferences of a benign user between two models. Figure 4 illustrated the results. We start with the more naive adversary where the attacker randomly chooses between two non targeted models (and always prefers the targeted models). As can be seen in the results, the defender can use the difference in the behavior of a random adversary to identify the malicious users. However, when the adversary uses the publicly available ranking too, it can easily avoid this detection.

In the second scenario the defender modifies the rating of the model and releases the perturbed leaderboard. Now if the adversary uses this perturbed order, its behavior can be detected. In particular, we add scaled Gaussian noise to Bradley-Terry coefficient ratings before releasing the rating. Figures 5 and 6 show the effectiveness and also utility effect of this mitigation approach. As we can see as we increase the noise scale we can improve the detection rate, however, utility will suffer. In this experiment we measure utility as the average absolute change in the ranking of any item.

As mentioned earlier, while we cannot prevent this attack completely using either authentication approaches or the malicious user detection approach described in this section, we can increase the cost of the attack significantly.

![](images/figures/voting-leaderboards-fig-0004.jpg)  
Figure 4: Scenario 1: The defender uses the likelihood to identify the malicious users. For a naive adversary who randomly chooses between untargetted models this approach can be effective, however, if the adversary uses existing public ranking it can bypass detection

![](images/figures/voting-leaderboards-fig-0005.jpg)  
Figure 5: Scenario 2: The defender releases a perturbed version of the leaderboard. Even when an adversary uses this perturbed leaderboard to choose between two untargeted models, their actions can still be detected. Increasing the amount of noise helps in detecting malicious users.

![](images/figures/voting-leaderboards-fig-0006.jpg)  
Figure 6: Larger noises significantly change the order of rank list

# 5 RELATED WORK

Security vulnerabilities in voting-based system. Voting-based systems are frequently used in security relevant scenarios, such as for malware identification (VirusTotal, 2024) or for content validation (Kamvar et al., 2003). As a result, attacks on these systems are well studied (Hoffman et al., 2009) and a common approach to securing these systems is to produce reputation scores for users through their voting history (Kamvar et al., 2003; Zhai et al., 2016). We consider an extention of reputation systems to a Chatbot Arena in Section 4.2. In the context of machine learning, reputation has also been used by FLTrust (Cao et al., 2020) to defend against data poisoning attacks.

Detecting the target model for the generation. Our primary attack involves training a classifier that can identify which language model system produced a given generation. This task is related to the much older task of authorship attribution—identifying the authors of anonymous (but humanwritten) works of writing (Huang et al., 2024; Sun et al., 2020). Tay et al. (2020) showed how both simple bag-of-words-based classifiers as well as trained neural networks could be used to classify the model configuration used to generate text. Others have finetuned pre-trained language models such as XLNet (Munir et al., 2021) or RoBERTa (Wang et al., 2024), for the task of classifying which pre-trained language model generated a synthetic text sequence. Our framing of the task is easier than that of most prior work in this space because we assume the attacker has control over the prompt being used for generation, and the set of possible model configurations which may have been used for generation is fairly constrained.

The most related work to ours is the concurrent effort by Zhao et al. (2024), which also investigates the use of targeted model detection algorithms to enable adversarial voting. However, their experiments are limited to voting logs with $5 5 \mathrm { k }$ entries and fewer than five models. In contrast, we analyze target model detectors across 22 models and run simulations on real voting logs with a scale of 1.7 million votes. Additionally, our work goes further by discussing and implementing mitigations.

Evaluation of LLMs. Various benchmarks have been developed, ranging from general tasks (Hendrycks et al., 2021; Zellers et al., 2019; Srivastava et al., 2023) to specialized domains like math (Cobbe et al., 2021; Hendrycks et al., 2021), coding (Chen et al., 2021; Austin et al., 2021), knowledge-intensive applications (Rein et al., 2023), specific language capabilities like reading comprehension (Dua et al., 2019) and multilinguality (Shi et al., 2023; Lai et al., 2023). However, there are many challenges when using those benchmarks to track the progress of model developments: 1) academic benchmarks focus on measuring fundamental capabilities, which do not always correlate well with application scenarios that average real world users care about (Köpf et al., 2024; Zheng et al., 2023c;b); 2) faithfully evaluating open-ended responses to complex questions (e.g. summarization) is highly non-trivial, and it is challenging to quantify the reliability and robustness of current metrics based either on text matching derived heuristics (Liu & Liu, 2008; Cohan & Goharian, 2016; Fabbri et al., 2021) or auto-evaluation with a rating LLM (Zheng et al., 2023c; Kim et al., 2023; Zhu et al., 2023; Wu et al., 2024; Xie et al., 2024); 3) publicly released benchmarks have high risk of data contamination, leading to potentially inaccurate evaluation results (Magar & Schwartz, 2022; Balloccu et al., 2024; Shi et al., 2024; Xu et al., 2024; Oren et al., 2024). As a results, evaluation results based on human voting are considered highly valuable signals by all major model developers as it reflects real world user queries and preferences — the Chatbot Arena leaderboard currently hosts 157 models from more than 20 different model developers. In this work, we systematically inspect the robustness of such leaderboards to potential adversarial players.

# 6 DISCUSSION

Upvoting one’s own models vs downvoting those of a competitor. It is far easier for a model owner to upvote their own model(s) than to downvote (or upvote) another. Model owners have much more knowledge about their models. They know the entire training dataset and can evaluate the loss on each sample to determine the easiest samples to detect. Further, if their model is deployed as an API, they could simply log generations that the API produces, and then check each candidate in Chatbot Arena against this database. Finally, the model owner can also strategically make text more detectable, either by using stealthy watermarks that only they have direct knowledge of or by using hidden backdoors on specific prompts. In contrast, our approach in Section 2.2 aims to address the case where the adversary does not necessarily have control over the models whose scores they aim to manipulate.

Detection via watermarking. There has been a slew of recent research aiming to watermark generated text to identify whether given text was generated with a particular, watermarked model (Kirchenbauer et al., 2023; Kuditipudi et al., 2024; Christ et al., 2024). This is indeed a way of breaking model anonymity but it has limited applicability for our task. Not all models employ watermarking, and successful de-anonymization would require the attacker to know the specifics of the watermarking implementation in the target models—information that is typically not public.

Implications for public evaluation of AI systems. While this paper focuses on Chatbot Arena, our findings our relevant for any public platform for performing comparative evaluation of AI systems, such as ones deployed for evaluating text-to-image and speech.8 There is a fundamental tension when designing human evaluation experiments. On one hand, human evaluation paradigms that closely reflect real-world usage lend validity to the results. On the other hand, restricting human evaluation to known groups of annotators lends greater control annotator qualifications, demographic makeup, and incentives—but at the expense of the transferability of the findings to real-world usage. For example, prior work has shown that Amazon Mechanical Turk workers rate generated text very differently than school teachers (Karpinska et al., 2021).

# 7 CONCLUSIONS

The field of natural language processing has long relied on domain-specific, easy-to-implement evaluation metrics. But dramatic advances in LLM performance challenges traditional evaluation practices. As we show in this paper, moving from evaluations that use an objective source of truth to evaluations that utilize human inputs introduces the potential for new types of evaluation difficulties. We focus on this paper in validating one straightforward attack: by identifying and selectively voting for (or against) a particular model, an adversary can significantly alter the ordering of the best models.

Mitigating this attack is feasible, and we are actively collaborating with the Chatbot Arena team to make Chatbot Arena more robust. We also encourage the community to explore and adopt mitigation strategies, such as voter authentication, rate limits, and more robust mechanisms for detecting malicious activities.

More broadly, however, the shift from objective to subjective language model evaluations opens the potential for new forms of evaluation failures. Our paper explores just one of these failure modes— where an adversary explicitly aims to alter the rank of a particular target model. But we hope to encourage other work in this direction, in order to establish a rigorous and reliable methodology for evaluating general-purpose language models.

# ETHICS AND DISCLOSURE

Our study highlights the susceptibility of Chatbot Arena’s leaderboard rankings to malicious voting behavior. We conducted this work with the goal of improving the security and reliability of interactive evaluation platforms, and to encourage the development of countermeasures to improve robustness.

We disclosed this attack in August 2024 and collaborated with the Chatbot Arena team throughout the development of this work to assist in developing appropriate defenses. Our collaboration has been instrumental in refining solutions to mitigate these vulnerabilities, ensuring that platform integrity and user trust are maintained. By sharing these results, we aim to encourage the community to adopt stronger safeguards in the design and evaluation of similar systems.

All simulations and experiments conducted in this study were carried out in a controlled environment, with no real-world impact on the existing Chatbot Arena platform or any other public-facing system.

Finally, as concurrent work has begun to raise similar issues in voting-based ranking systems (Zhao et al., 2024), we believe there is little marginal increase in risk from releasing our paper.

# CONTRIBUTION STATEMENT

This project was a team effort.

• Idea formulation: Yangsibo came up with the idea of using model de-identification to manipulate leaderboard rankings on Chatbot Arena. Nicholas and Florian suggested running simulations to quantify the attack efficacy via estimating the number of votes required to shift models’ positions on the leaderboard.   
De-anonymizing models (Section 2): Milad suggested using TF−IDF and BoW for trainingbased detectors, and Yangsibo conducted experiments demonstrating their effectiveness. Ken suggested and explored the identity-probing detector. Yangsibo collaborated with Ken to finalize results.   
• Disclosure with Chatbot Arena: In August 2024, Yangsibo, Milad, Chiyuan, and Nicholas contacted the Chatbot Arena team (Wei-Lin, Anastasios, and Ion) to disclose their findings that anonymous model responses can be de-identified with very high accuracy. The Chatbot Arena team expressed interest in collaborating to investigate and address this security vulnerability. As a result, Yangsibo, Milad, Nicholas, Chiyuan, Wei-Lin, Anastasios, and Ion began having regular meetings to advance the project.   
• Estimating number of adversarial votes (Section 3): The Chatbot Arena team shared a simulation platform. Yangsibo conducted the simulations to estimate the number of votes needed by the attack, with feedback from Milad, Chiyuan, Nicholas and the Chatbot Arena team.   
• Exploring mitigations (Section 4): Ion suggested exploring mitigation strategies. Milad, Yangsibo, Chiyuan, and Nicholas developed the attack cost model (Section 4.1) and refined it with input from the Chatbot Arena team. For mitigations, Nicholas suggested authentication (Section 4.2.1) and rate limiting (Section 4.2.2); Anastasios suggested exploring customized malicious user identification algorithms, and then Milad drafted the proposals with Chris (Section 4.2.3) and ran experiments; Chiyuan suggested increasing the cost of actions (Section 4.2.4).   
• Writing: Yangsibo and Milad prepared the initial draft. Chris, Chiyuan, Katherine, Daphne, Nicholas, Florian, Matthew, Ken, Wei-Lin, Anastasios, and Ion wrote and edited the paper.   
• Paper release: Katherine, Milad, Chiyuan, and Yangsibo prepared the paper for public release.

# ACKNOWLEDGMENTS

We thank Szymon Tworkowski, Samuel Bowman, Zheng-Xin Yong, Mengzhou Xia, Haochen Zhang, Tianle Cai, and Danqi Chen for their valuable discussions during the early stages of this paper. We are grateful to Andreas Terzis, Martin Abadi, Four Flynn, Shira McNamara, Jon Small, Anand Rao, and Aneesh Pappu for comments and reviews.

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Anthropic. Anthropic introduces the claude 3 model family, March 2024. Available at: https: //www.anthropic.com/news/claude-3-family.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondrej Dusek. Leak, cheat, repeat: Data contamination and evaluation malpractices in closed-source LLMs. In Yvette Graham and Matthew Purver (eds.), Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 67–93, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. URL https://aclanthology. org/2024.eacl-long.5.

Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. Fltrust: Byzantine-robust federated learning via trust bootstrapping. arXiv preprint arXiv:2012.13995, 2020.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating LLMs by human preference. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id= 3MW8GKNyzI.

Miranda Christ, Sam Gunn, and Or Zamir. Undetectable watermarks for language models. In Shipra Agrawal and Aaron Roth (eds.), Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pp. 1125–1139. PMLR, 30 Jun–03 Jul 2024. URL https://proceedings.mlr.press/v247/christ24a.html.

Rebecca Chui. A multi-faceted approach to anonymity online: Examining the relations between anonymity and antisocial behaviour. Journal For Virtual Worlds Research, 7(2), 2014.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Arman Cohan and Nazli Goharian. Revisiting summarization evaluation for scientific articles. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mariani, Helene Mazo, Asuncion Moreno, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pp. 806–813, Portorož, Slovenia, May 2016. European Language Resources Association (ELRA). URL https://aclanthology.org/L16-1130.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2368–2378, 2019.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Alexander R Fabbri, Wojciech Krysci ´ nski, Bryan McCann, Caiming Xiong, Richard Socher, and ´ Dragomir Radev. Summeval: Re-evaluating summarization evaluation. Transactions of the Association for Computational Linguistics, 9:391–409, 2021.

Anthony Gavazzi, Ryan Williams, Engin Kirda, Long Lu, Andre King, Andy Davis, and Tim Leek. A study of Multi-Factor and Risk-Based authentication availability. In 32nd USENIX Security Symposium (USENIX Security 23), pp. 2043–2060, 2023.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Kevin Hoffman, David Zage, and Cristina Nita-Rotaru. A survey of attack and defense techniques for reputation systems. ACM Computing Surveys (CSUR), 42(1):1–31, 2009.

Baixiang Huang, Canyu Chen, and Kai Shu. Authorship attribution in the era of llms: Problems, methodologies, and challenges. arXiv preprint arXiv:2408.08946, 2024.

David R Hunter. Mm algorithms for generalized bradley-terry models. The annals of statistics, 32 (1):384–406, 2004.

Sepandar D Kamvar, Mario T Schlosser, and Hector Garcia-Molina. The eigentrust algorithm for reputation management in p2p networks. In Proceedings of the 12th international conference on World Wide Web, pp. 640–651, 2003.

Marzena Karpinska, Nader Akoury, and Mohit Iyyer. The perils of using Mechanical Turk to evaluate open-ended text generation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 1265–1285, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.97. URL https://aclanthology.org/2021.emnlp-main.97.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, 2023.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 17061–17084. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/kirchenbauer23a.html.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, et al. Openassistant conversations-democratizing large language model alignment. Advances in Neural Information Processing Systems, 36, 2024.

Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=FpaCL1MO2C.

Viet Lai, Chien Nguyen, Nghia Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan Rossi, and Thien Nguyen. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. In Yansong Feng and Els Lefever (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 318–327, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-demo.28. URL https://aclanthology.org/2023.emnlp-demo.28.

Leona Lassak, Elleen Pan, Blase Ur, and Maximilian Golla. Why aren’t we using passkeys? obstacles companies face deploying FIDO2 passwordless authentication. In 33rd USENIX Security Symposium (USENIX Security 24), pp. 7231–7248, Philadelphia, PA, August 2024.

USENIX Association. ISBN 978-1-939133-44-1. URL https://www.usenix.org/conference/ usenixsecurity24/presentation/lassak.

Minzhi Li, Will Held, Michael J. Ryan, Kunat Pipatanakul, Potsawee Manakul, Hao Zhu, and Diyi Yang. Talk arena: Interactive evaluation of large audio models, 2024.

Feifan Liu and Yang Liu. Correlation between rouge and human evaluation of extractive meeting summaries. In Proceedings of ACL-08: HLT, short papers, pp. 201–204, 2008.

Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang Wang, Yejin Choi, and Bill Yuchen Lin. Wildvision: Evaluating vision-language models in the wild with human preferences. arXiv preprint arXiv:2406.11069, 2024.

Inbal Magar and Roy Schwartz. Data contamination: From memorization to exploitation. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 157–165, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-short.18. URL https://aclanthology.org/2022.acl-short.18.

Jan-Phillip Makowski and Daniela Pöhn. Evaluation of real-world risk-based authentication at online services revisited: Complexity wins. In Proceedings of the 18th International Conference on Availability, Reliability and Security, pp. 1–9, 2023.

Shaoor Munir, Brishna Batool, Zubair Shafiq, Padmini Srinivasan, and Fareed Zaffar. Through the looking glass: Learning to attribute synthetic text generated by language models. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 1811–1822, 2021.

Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, and Tatsunori B Hashimoto. Proving test set contamination in black box language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=KS8mIvetg2.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

Jessica Rumbelow and Matthew Watkins. SolidGoldMagikarp (plus, prompt generation). https: //www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg, 2023.

Gerard Salton and Christopher Buckley. Term-weighting approaches in automatic text retrieval. Information processing & management, 24(5):513–523, 1988.

Gerard Salton, Anita Wong, and Chung-Shu Yang. A vector space model for automatic indexing. Communications of the ACM, 18(11):613–620, 1975.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= fR3wGCk-IXp.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=zWqr3MQuNs.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, , et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum? id=uyTL5Bvosj. Featured Certification.

Zhen Sun, Roei Schuster, and Vitaly Shmatikov. De-anonymizing text by fingerprinting language generation. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 22420–22431. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/ fdf2aade29d18910051a6c76ae661860-Paper.pdf.   
Yi Tay, Dara Bahri, Che Zheng, Clifford Brunk, Donald Metzler, and Andrew Tomkins. Reverse engineering configurations of neural text generation models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 275–279, 2020.   
VirusTotal. Results reports. VirusTotal Documentation, 2024. URL https://docs.virustotal. com/docs/results-reports. Accessed on December 19, 2024.   
Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Osama Mohanned Afzal, Tarek Mahmoud, Giovanni Puccetti, Thomas Arnold, et al. M4gtbench: Evaluation benchmark for black-box machine-generated text detection. arXiv preprint arXiv:2402.11175, 2024.   
Xindi Wu, Dingli Yu, Yangsibo Huang, Olga Russakovsky, and Sanjeev Arora. Conceptmix: A compositional image generation benchmark with controllable difficulty. arXiv preprint arXiv:2408.14339, 2024.   
Chulin Xie, Yangsibo Huang, Chiyuan Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. On memorization of large language models in logical reasoning. arXiv preprint arXiv:2410.23123, 2024.   
Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824, 2024.   
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10. 18653/v1/P19-1472. URL https://aclanthology.org/P19-1472.   
Ennan Zhai, David Isaac Wolinsky, Ruichuan Chen, Ewa Syta, Chao Teng, and Bryan Ford. $\left\{ { \mathrm { A n o n R e p } } \right\}$ : Towards $\{$ {Tracking-Resistant $\}$ anonymous reputation. In 13th USENIX Symposium on Networked Systems Design and Implementation (NSDI 16), pp. 583–596, 2016.   
Wenting Zhao, Alexander M. Rush, and Tanya Goyal. Challenges in trustworthy human evaluation of chatbots, 2024. URL https://arxiv.org/abs/2412.04363.   
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023a.   
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P Xing, et al. Lmsys-chat-1m: A large-scale real-world llm conversation dataset. arXiv preprint arXiv:2309.11998, 2023b.   
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023c.   
Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631, 2023.   
Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

# A EXPERIMENTAL DETAILS

# A.1 LIST OF MODELS

Table 6 lists the evaluated models and the methods used to query them. For all models, we rely on the default decoding hyperparameters (e.g., temperature) specified by the query method.

Table 6: Overview of evaluated models and the querying methods used in our experiments.   

<table><tr><td>Model</td><td>Company / Organization</td><td>Method of query in our experiments</td></tr><tr><td>claude-3-5-sonnet-20240620</td><td>Anthropic</td><td>Anthropic API</td></tr><tr><td>claude-3-haiku-20240307</td><td>Anthropic</td><td>Anthropic API</td></tr><tr><td>gemini-1.5-flash</td><td>Google</td><td>Google AI studio API</td></tr><tr><td>gemini-1.5-pro</td><td>Google</td><td>Google AI studio API</td></tr><tr><td>gemma-2-2b-it</td><td>Google</td><td>Together AI Inference API</td></tr><tr><td>gemma-2-9b-it</td><td>Google</td><td>Together AI Inference API</td></tr><tr><td>gemma-2-27b-it</td><td>Google</td><td>Together AI Inference API</td></tr><tr><td>gpt-3.5-turbo</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>gpt-4-0125-preview</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>gpt-4-1106-preview</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>gpt-4-turbo-2024-04-09</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>gpt-4o-2024-05-13</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>gpt-4o-2024-08-06</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td>OpenAI</td><td>OpenAI Text generation API</td></tr><tr><td>llama-3-8b-instruct</td><td>Meta</td><td>Together AI Inference API</td></tr><tr><td>llama-3-70b-instruct</td><td>Meta</td><td>Together AI Inference API</td></tr><tr><td>llama-3.1-8b-instruct</td><td>Meta</td><td>Together AI Inference API</td></tr><tr><td>llama-3.1-70b-instruct</td><td>Meta</td><td>Together AI Inference API</td></tr><tr><td>llama-3.1-405b-instruct</td><td>Meta</td><td>Together AI Inference API</td></tr><tr><td>mixtral-8x7b-instruct-v0.1</td><td>Mistral AI</td><td>Together AI Inference API</td></tr><tr><td>mixtral-8x22b-instruct-v0.1</td><td>Mistral AI</td><td>Together AI Inference API</td></tr><tr><td>qwen2-72b-instruct</td><td>Alibaba</td><td>Together AI Inference API</td></tr></table>

# A.2 PROMPTS FOR EMBEDDING VISUALIZATION

The three prompts we used for embedding visualization in Fig. 2 are:

• Prompt #1: “Beside OFAC’s selective sanction that target the listed individiuals and entities, please elaborate on the other types of US’s sanctions, for example, comprehensive and sectoral   
sanctions. Please be detailed as much as possible”   
• Prompt #2: “You are the text completion model and you must complete the assistant answer below, only send the completion based on the system instructions.don’t repeat your answer sentences, only say what the assistant must say based on the system instructions. repeating same thing in same answer not allowed. user: descriptive answer for append many items to list   
python in python with proper code examples and outputs. assistant: ”   
• Prompt #3: “The sum of the perimeters of three equal squares is $3 6 ~ \mathrm { c m }$ . Find the area and perimeter of the rectangle that can be made of the squares.”

# A.3 DETAILS FOR THE TRAINING-BASED DETECTOR

Data collection and its cost. The main cost of building the training-based detector comes from the data collection process, where the attacker gathers responses from various models for the same prompt and train classifier to distinguish among them (Section 2). In our experiments, we collect responses depending on the model type: For proprietary models, we directly used the model providers’ APIs to obtain the responses. For open-source models, we relied on Together’s $\mathsf { A P I } ^ { 9 }$ to make the queries. We set the output length to 512 tokens and found that collecting 50 responses per model was sufficient to train an effective target model detector.

To estimate the upper bound on the data collection cost, we based our calculations on the pricing of the most expensive model we tested. Proprietary models cost $\$ 5.00$ per 1 million output tokens, while open-source models cost $\$ 1.80$ per 1 million output tokens.

Using these rates, the upper bound cost of querying a single model is:   
Proprietary model: $5 . 0 0 \times { \frac { 5 1 2 \times 5 0 } { 1 0 ^ { 6 } } } = 0 . 1 2 8$ Open-source model: $1 . 8 0 \times { \frac { 5 1 2 \times 5 0 } { 1 0 ^ { 6 } } } = 0 . 0 4 6$ Assuming the training process requires 10 proprietary models and 20 open-source models, the overall data collection cost would be approximately $\$ 2.2$ per prompt.

We collected data for 200 prompts in Section 2, so the cost is at most $\$ 440$ .

# A.4 SIMULATION TESTBED

Our simulation in Section 3 is based on an anonymized and deduplicated dataset of voting records from Chatbot Arena. The dataset includes 1,670,250 votes from 477,322 unique users, with 1,093,875 votes resulting in wins and 576,375 in ties. These votes cover 6,895 unique combinations of side-by-side model comparisons.

# B MORE EXPERIMENTAL RESULTS

B.1 TARGET MODEL DETECTION

Table 7 presents the performance of identity-probing detector for all evaluated 22 models.10

Table 7: Averaged detection accuracy $( \% )$ with across 1,000 queries per prompt for different identity-probing prompts across various models.

<table><tr><td rowspan="2">Model</td><td colspan="5">Prompt</td></tr><tr><td>Who are you?</td><td>Which model are you?</td><td>What is your model name?</td><td>How should I refer to you as an AI?</td><td>How would you define your role or identity as an AI?</td></tr><tr><td>claude-3-5-sonnet-20240620</td><td>99.3</td><td>100.0</td><td>98.5</td><td>100.0</td><td>100.0</td></tr><tr><td>claude-3-haiku-20240307</td><td>100.0</td><td>96.3</td><td>100.0</td><td>42.9</td><td>14.3</td></tr><tr><td>gemini-1.5-flash</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>gemini-1.5-pro</td><td>97.2</td><td>96.5</td><td>100.0</td><td>0.0</td><td>99.1</td></tr><tr><td>gemma-2-27b-it</td><td>100.0</td><td>98.4</td><td>98.2</td><td>97.9</td><td>95.5</td></tr><tr><td>gemma-2-2b-it</td><td>81.8</td><td>91.8</td><td>58.2</td><td>12.7</td><td>4.5</td></tr><tr><td>gemma-2-9b-it</td><td>98.5</td><td>99.4</td><td>98.3</td><td>98.1</td><td>97.3</td></tr><tr><td>gpt-3.5-turbo</td><td>0.0</td><td>54.5</td><td>67.3</td><td>0.0</td><td>0.0</td></tr><tr><td>gpt-4-0125-preview</td><td>70.9</td><td>100.0</td><td>94.6</td><td>1.8</td><td>1.8</td></tr><tr><td>gpt-4-1106-preview</td><td>7.3</td><td>90.9</td><td>99.1</td><td>6.4</td><td>1.8</td></tr><tr><td>gpt-4o-2024-05-13</td><td>16.4</td><td>93.3</td><td>99.9</td><td>0.0</td><td>6.4</td></tr><tr><td>gpt-4o-2024-08-06</td><td>51.8</td><td>97.7</td><td>98.5</td><td>0.0</td><td>5.5</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td>92.7</td><td>92.9</td><td>100.0</td><td>12.7</td><td>0.0</td></tr><tr><td>llama-3-70b-instruct</td><td>98.2</td><td>98.2</td><td>54.5</td><td>46.4</td><td>2.7</td></tr><tr><td>llama-3-8b-instruct</td><td>99.9</td><td>99.1</td><td>74.5 89.1</td><td>20.0 75.5</td><td>1.8</td></tr><tr><td>llama-3.1-405b-instruct</td><td>99.1</td><td>90.9</td><td>92.7</td><td>5.5</td><td>0.0</td></tr><tr><td>llama-3.1-70b-instruct</td><td>98.8</td><td>66.4</td><td>99.1</td><td>6.4</td><td>0.0</td></tr><tr><td>llama-3.1-8b-instruct</td><td>17.3</td><td>40.0 31.8</td><td>45.5</td><td>1.8</td><td>0.0</td></tr><tr><td>mixtral-8x7b-instruct-v0.1</td><td>97.3 97.3</td><td>31.8</td><td>45.5</td><td>0.9</td><td>0.9</td></tr><tr><td>mixtral-8x22b-instruct-v0.1</td><td></td><td>98.2</td><td>97.6</td><td>24.5</td><td>1.8</td></tr><tr><td>qwen2-72b-instruct</td><td>91.8</td><td></td><td></td><td></td><td>7.3</td></tr></table>

# B.2 ADVERSARIAL VOTE

Ablation for detector accuracy. Table 8 shows the number of votes and interactions needed to shift a model’s position by 1 to 50 places on the simulated leaderboard under different detector accuracies. As shown, the number of votes required to move a model up by 50 places increases by only about 150 when the detector accuracy drops from 1.0 to 0.9. This suggests that a detector, while not perfect, can still be sufficiently accurate to achieve the attack’s objective.

Table 8: The number of votes (a) and interactions (b) required to change the ranking of a low-ranked model on the simulated leaderboard, under varying detector accuracy.   

<table><tr><td>Target model=llama-13b (current rank: #129, #votes: 2443)</td><td>Target rank: 79 (↑ 50)</td><td>Target rank: 109 (↑ 20)</td><td>Target rank: 119 (↑ 10)</td><td>Target rank: 124(↑5)</td><td>Target rank: 127 (↑ 2)</td><td>Target rank: 1128 ( 1)</td></tr><tr><td>detector acc=1.0</td><td>1246</td><td>861</td><td>645</td><td>415</td><td>208</td><td>126</td></tr><tr><td>detector acc=0.95</td><td>1304</td><td>918</td><td>682</td><td>522</td><td>255</td><td>126</td></tr><tr><td>detector acc=0.9</td><td>1383</td><td>1012</td><td>732</td><td>525</td><td>271</td><td>136</td></tr><tr><td colspan="7">(a) # Votes</td></tr><tr><td>Target model=llama-13b (current rank: #129, #votes: 2443)</td><td>Target rank: 7 50)</td><td>Target rank: 109 (↑ 20)</td><td>Target rank: 19(10)</td><td>Target rank: 124(5)</td><td>Target rank: 12 (2)</td><td>Target rank: 128 (1)</td></tr><tr><td>detector acc=1.0</td><td>80000</td><td>55000</td><td>40000</td><td>30000</td><td>15000</td><td>10000</td></tr><tr><td>detector acc=0.95</td><td>85000</td><td>65000</td><td>45000</td><td>30000</td><td>15000</td><td>10000</td></tr><tr><td>detector acc=0.9</td><td>100000</td><td>75000</td><td>55000</td><td>40000</td><td>20000</td><td>10000</td></tr></table>

(b) # Interactions

Ablation for non-detected actions. When the attacker does not detect the target model, they can choose from four actions: randomly upvote one model, vote for a tie, vote both models as bad, or do nothing. The main results in Section 3 assume the attacker does nothing. We also explore the other options in Table 9. As shown, there are no clear patterns indicating that any one option is significantly better than the others.

Table 9: The number of interactions required to change the ranking of a high-ranked model (a) and a low-ranked model (b) on the simulated leaderboard, under varying non-target strategies.   

<table><tr><td>Non-target strategy</td><td>Target rank: 1(↑ 4)</td><td></td><td>Target rank: 2(↑ 3)</td><td>Target rank: 3(↑ 2)</td><td>Target rank: 4(↑ 1)</td></tr><tr><td>Do nothing</td><td>206000</td><td></td><td>184000</td><td>144000</td><td>18000</td></tr><tr><td>Randomly upvote</td><td>192000</td><td></td><td>182000</td><td>142000</td><td>16000</td></tr><tr><td>Vote tie</td><td>194000</td><td></td><td>182000</td><td>148000</td><td>20000</td></tr><tr><td>Vote tie (both bad)</td><td>196000</td><td></td><td>172000</td><td>152000</td><td>16000</td></tr><tr><td colspan="6">(a) High-ranked model, claude-3-5-sonnet-20240620 (rank: #5)</td></tr><tr><td>Non-target strategy</td><td>Target rank: 79 (↑ 50)</td><td>Target rank: 109 (↑ 20)</td><td>Target rank: 119 (↑ 10)</td><td>Target rank: 124 (↑ 5)</td><td>Target rank: 127(↑ 2)</td><td>Target rank: 128(↑1)</td></tr><tr><td>Do nothing</td><td>80000</td><td>55000</td><td>40000</td><td>30000</td><td>15000</td><td>10000</td></tr><tr><td>Randomly upvote</td><td>75000</td><td>60000</td><td>40000</td><td>30000</td><td>15000</td><td>10000</td></tr><tr><td>Vote tie</td><td>80000</td><td>60000</td><td>40000</td><td>30000</td><td>15000</td><td>10000</td></tr><tr><td>Vote tie (both bad)</td><td>80000</td><td>60000</td><td>40000</td><td>30000</td><td>15000</td><td>10000</td></tr></table>

(b) Low-ranked model, llama-13b (rank: #129)