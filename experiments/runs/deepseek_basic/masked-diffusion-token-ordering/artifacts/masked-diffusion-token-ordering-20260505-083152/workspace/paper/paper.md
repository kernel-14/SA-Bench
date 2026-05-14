# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Jaeyeon Kim \* 1 Kulin Shah \* 2 Vasilis Kontonis 2 Sham Kakade 1 Sitan Chen 1

# Abstract

# 1. Introduction

In recent years, masked diffusion models (MDMs) have emerged as a promising alternative approach for generative modeling over discrete domains. Compared to autoregressive models (ARMs), MDMs trade off complexity at training time with flexibility at inference time. At training time, they must learn to solve an exponentially large number of infilling problems, but at inference time, they can decode tokens in essentially arbitrary order. In this work, we closely examine these two competing effects. On the training front, we theoretically and empirically demonstrate that MDMs indeed train on computationally intractable subproblems compared to their autoregressive counterparts. On the inference front, we show that a suitable strategy for adaptively choosing the token decoding order significantly enhances the capabilities of MDMs, allowing them to sidestep hard subproblems. On logic puzzles like Sudoku, we show that adaptive inference can boost solving accuracy in pretrained MDMs from $< 7 \%$ to $\approx 9 0 \%$ , even outperforming ARMs with $7 \times$ as many parameters and that were explicitly trained via teacher forcing to learn the right order of decoding. This shows that MDMs without knowledge of the correct token generation order during training and inference can outperform ARMs trained with knowledge of the correct token generation order. We also show the effectiveness of adaptive MDM inference on reasoning tasks such as coding and math on the 8B large language diffusion model (LLaDa 8B).

While diffusion models (Ho et al., 2020; Song et al., 2021) are now the dominant approach for generative modeling in continuous domains like image, video, and audio, efforts to extend this methodology to discrete domains like text and proteins (Austin et al., 2021; Lou et al., 2024; Hoogeboom et al., 2021b) remain nascent. Among numerous proposals, masked diffusion models (MDMs) (Lou et al., 2024; Sahoo et al., 2025; Shi et al., 2024) have emerged as a leading variant, distinguished by a simple and principled objective: to generate samples, learn to reverse a noise process which independently and randomly masks tokens.

In many applications, such as language modeling, masked diffusion models (MDMs) still underperform compared to autoregressive models (ARMs) (Nie et al., 2024; Zheng et al., 2024), which instead learn to reverse a noise process that unmasks tokens sequentially from left to right. However, recent studies suggest that MDMs may offer advantages in areas where ARMs fall short, including reasoning (Nie et al., 2024; Kitouni et al., 2025), planning (Ye et al., 2024), and infilling (Gong et al., 2024). This raises a key question: what are the strengths and limitations of MDMs compared to ARMs, and on what type of tasks can MDMs be scaled to challenge the dominance of ARMs in discrete generative modeling?

To understand these questions, we turn a microscope to two key competing factors when weighing the merits of MDMs over ARMs:

• Complexity at training time: MDMs face a more challenging training task by design. While ARMs predict the next token given an unmasked prefix, MDMs predict a token conditioned on a set of unmasked tokens in arbitrary positions. This inherently increases their training complexity.

• Flexibility at inference time: On the other hand, the sampling paths taken by an MDM are less rigid. Unlike the fixed left-to-right decoding of ARMs, MDMs decode tokens in random order at inference. Even more is possible: MDMs can be used to decode in any order (including left-to-right).

Therefore, we ask:

Are the benefits of inference flexibility for MDMs enough to outweigh the drawbacks of training complexity?

In this work, we provide dual perspectives on this question.

(1) Training for the worst. First, we provide theoretical and empirical evidence that the overhead imposed by training complexity quantifiably impacts MDMs’ performance.

Theoretically, we show examples of simple data distributions with a natural left-to-right order, where ARMs can provably generate samples efficiently. In contrast, there are noise levels at which a large fraction of the corresponding subproblems solved by MDMs for these distributions are provably computationally intractable. Empirically, we validate this claim on real-world text data, known to have left-to-right order and show that the imbalance in training complexity across subproblems persists even in real-world text data (Fig. 2, left).

(2) Planning for the best. While the above might appear to be bad news for MDMs, in the second part of this paper, we answer our guiding question in the affirmative by building upon the observation (Chang et al., 2022; Zheng et al., 2023) that MDMs which can perfectly solve all masking subproblems can be used to decode in any order.

In first part of the paper, we show that the imbalance in complexity across subproblems during the training of MDMs results in some of the subproblems being poorly trained and the vanilla MDM inference that unmasks tokens in random order results in evaluating the poorly trained marginals. Therefore, in place of vanilla MDM inference, we consider adaptive strategies that carefully select which token to unmask next. Our key insight is that the adaptive strategies makes it possible to sidestep the hard subproblems from training (Fig. 1). In particular, we find that even without modifying how MDMs are trained, the resulting models’ logits contain enough information to determine the right order in which to unmask. We show the effectiveness of the adaptive inference in solving logic puzzles, coding, math and infilling tasks. For example, on Sudoku puzzles, a simple adaptive strategy (Section 4.1) improves the accuracy of MDMs from $< 7 \%$ to almost $90 \%$ .

Advantage of MDMs over ARMs. We show that the main effectiveness of MDMs lies in tasks that do not have the same natural token generation order across all sequences (e.g., logic puzzles and reasoning tasks like coding and math). By carefully designing experiments on logic puzzles, we show that MDMs without the knowledge of the correct token generation order during training and inference can outperform ARMs trained with the knowledge of the correct token generation order. In particular, we show that

MDMs that decide the correct token generation order during inference via adaptive strategies can outperform ARMs that are trained to learn the right token generation order via supervised teacher forcing (Shah et al., 2024; Lehnert et al., 2024).

Organization. In Section 2, we provide preliminaries on MDMs and set notation. In Section 3, we examine MDM training and demonstrate the imbalance in computational intractability across subproblems. In Section 4, we consider adaptive inference in MDMs and investigate its impact on likelihood modeling across various tasks.

# 2. Masked Diffusion Models (MDM)

In this section, we explain the framework of Masked Diffusion Models (Shi et al., 2024; Sahoo et al., 2025) and highlight its interpretation as an order-agnostic learner. MDMs gradually add masking noise to the true discrete data and learn the marginal distribution of the induced reverse process. We formally define both the forward and reverse processes for MDMs below.

Let the distribution $p _ { \mathrm { d a t a } }$ on $\{ 1 , \ldots , m \} ^ { L }$ be the data distribution over sequences of length $L$ and with vocabulary $\{ 1 , \ldots , m \}$ . We use 0 to denote the “mask” token.

Forward process. For a given $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ and a noise level $t \in [ 0 , 1 ]$ , the forward process $x _ { t } \sim q _ { t | 0 } ( \cdot | x _ { 0 } )$ is a coordinate-independent masking process via $\dot { q } _ { t | 0 } ( x _ { t } | x _ { 0 } ) =$ $\begin{array} { r } { \prod _ { i = 0 } ^ { L - 1 } q _ { t | 0 } ( x _ { t } ^ { i } | x _ { 0 } ^ { i } ) } \end{array}$ , where

$$
q _ { t | 0 } ( x _ { t } ^ { i } \mid x _ { 0 } ^ { i } ) = \mathrm { C a t } \big ( \alpha _ { t } \mathbf { e } _ { x _ { 0 } ^ { i } } + ( 1 - \alpha _ { t } ) \mathbf { e } _ { 0 } \big ) .
$$

Here, $\alpha _ { t }$ is a predefined noise schedule satisfying $\alpha _ { 0 } \approx$ $1 , \alpha _ { 1 } \approx 0$ and $\mathbf { e } _ { x _ { 0 } ^ { i } } \in \mathbb { R } ^ { m + 1 }$ is a one-hot vector corresponding to the value of token $x _ { 0 } ^ { i }$ . $\operatorname { C a t } ( \pi )$ denotes the categorical distribution given by $\pi \in \Delta ^ { m }$ . In other words, for each $i$ -th coordinate, $\ v { x } _ { t } ^ { i }$ is masked to the mask token 0 with probability $1 - \alpha _ { t }$ and remains unchanged otherwise.

Reverse process. The reverse process of the above forward process is denoted by $q _ { s \mid t } \big ( x _ { s } | x _ { t } , x _ { 0 } \big )$ and is given by $\begin{array} { r } { q _ { s | t } ( x _ { s } | x _ { t } , x _ { 0 } ) = \prod _ { i = 0 } ^ { L - 1 } q _ { s | t } ( x _ { s } ^ { i } | x _ { t } , x _ { 0 } ) } \end{array}$ for any $s < t$ where

$$
\begin{array} { r } { q _ { s | t } ( x _ { s } ^ { i } | x _ { t } , x _ { 0 } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( \mathbf { e } _ { x _ { t } ^ { i } } ) } & { x _ { t } ^ { i } \neq 0 } \\ { \mathrm { C a t } \left( \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } \mathbf { e } _ { 0 } + \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } \mathbf { e } _ { x _ { 0 } ^ { i } } \right) } & { x _ { t } ^ { i } = 0 . } \end{array} \right. } \end{array}
$$

The reverse transition probability $q _ { s | t } \big ( x _ { s } ^ { i } | x _ { t } , x _ { 0 } \big )$ is approximated using $g _ { \theta } ( x _ { s } ^ { i } | x _ { t } ) \ \triangleq \ q _ { s | t } ( x _ { s } ^ { i } | x _ { t } , x _ { 0 } \  \ p _ { \theta } ( \cdot | x _ { t } , t ) )$ where $p _ { \theta } ( \cdot | x _ { t } , t )$ is a denoising network trained to predict the marginal distribution on $x _ { 0 } ^ { i }$ via an ELBO-based loss for all masked tokens at noise scale $t$ (i.e., for all $i$ such that

# MDM training

![](images/figures/masked-diffusion-token-ordering-fig-0001.jpg)  
MDM inferences (Vanilla vs. Adaptive)

![](images/figures/masked-diffusion-token-ordering-fig-0002.jpg)  
Figure 1. (Top) MDM training can be seen as learning multiple masked prediction problems, where some are harder to learn, leading to performance imbalance (Section 3). (Bottom) During inference, adaptive MDM can avoid difficult problem instances, improving performance (Section 4).

$x _ { t } ^ { i } = 0 .$ ). To be precise, $q _ { s | t } ( x _ { s } ^ { i } \mid x _ { t } , x _ { 0 }  p _ { \theta } ( \cdot | x _ { t } , t ) )$ indicates the conditional probability where $p _ { \theta } ( \cdot | x _ { t } , t )$ is placed in the position of $e _ { x _ { 0 } ^ { i } }$ within $q _ { s \mid t } \big ( x _ { s } ^ { i } \mid x _ { t } , x _ { 0 } \big )$ . The denoising network is trained to minimize the following loss derived from the score-entropy (Lou et al., 2024; Sahoo et al., 2025; Shi et al., 2024; Ou et al., 2024):

$$
\mathcal { L } _ { \theta } = \int _ { 0 } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \mathop { \mathbb { E } } _ { x _ { t } \sim p _ { \mathrm { d a t a } } } \sum _ { i : x _ { t } ^ { i } = 0 } - \log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { t } , t ) d t ,
$$

where $\begin{array} { r } { \alpha _ { t } ^ { \prime } \ = \ \frac { d \alpha _ { t } } { d t } } \end{array}$ and the summation is computed over masked tokens (i.e., all $i$ such that $x _ { t } ^ { i } = 0 ,$ ). In practice, a time-embedding-free architecture for the denoising network, i.e., $p _ { \theta } ( \cdot | x _ { t } , t ) = p _ { \theta } ( \cdot | x _ { t } )$ is generally used as $x _ { t }$ implicitly contains information about $t$ via the number of masked tokens.

The reverse sampling process starts from the fully masked sentence $x _ { 1 } = ( 0 , \ldots , 0 )$ . Suppose we have a partially \fully masked sequence $x _ { t }$ at a given noise level $t \in ( 0 , 1 ]$ Then, to obtain $x _ { s }$ for a predetermined noise level $s < t$ , we sample $x _ { s } ^ { i } \sim g _ { \theta } ( \cdot | x _ { t } )$ for all $i$ . This process is repeated recursively from $t = 1$ to $t = 0$ .

# 2.1. Reformulating the training and inference of MDMs

In this section, we first discuss training of MDMs and compare it with “left-to-right” order training of autoregressive models in Section 2.1.1. Then, we reformulate vanilla MDM inference in Section 2.1.2 to set the stage for the upcoming discussion.

# 2.1.1. ORDER-AGNOSTIC TRAINING OF MDMS

Recent works (Zheng et al., 2024; Ou et al., 2024) have observed that the learning problem of MDM is equivalent to a masked language model. Building upon their analysis, we reformulate the loss $\mathcal { L } _ { \theta }$ to show that $\mathcal { L } _ { \theta }$ is a linear combination of the loss for all possible infilling masks. We first define $x _ { 0 } [ M ]$ as a masked sequence, obtained from original sequence $x _ { 0 }$ where indices in the mask set $M$ (a subset of $[ L ] \triangleq \{ 1 , 2 , \dots , L \} )$ are replaced with mask token 0.

Proposition 2.1. Assume $\alpha _ { 0 } ~ = ~ 1$ , $\alpha _ { 1 } ~ = ~ 0$ and denoising network $p _ { \theta }$ is time-embedding free. Then $\mathcal { L } _ { \boldsymbol { \theta } } ~ \leq$ $- \mathbb { E } _ { x _ { 0 } \sim p _ { \mathrm { d a t a } } } [ \log p _ { \theta } ( x _ { 0 } ) ]$ and

$$
\mathcal { L } _ { \theta } = - \sum _ { \substack { M \subseteq [ L ] , i \in M } } \frac { 1 } { | M | } \frac { 1 } { \binom { L } { | M | } } \operatorname* { \mathbb { E } } _ { x _ { 0 } \sim p _ { \mathrm { d a t a } } } [ \log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] ) ] ,
$$

where $| M |$ is the size of the set $M$ and $p _ { \theta } ( x _ { i } \mid x _ { 0 } [ M ] )$ indicates the conditional probability of the $i$ -th coordinate from $p _ { \theta } ( x _ { t } )$ .

The proof of the above proposition is given in Appendix E. As the MDM loss is a linear combination of the loss for all possible infilling mask $M$ , the minimizer of the loss $\mathcal { L } _ { \theta }$ learns to solve every masking problem. In other words, the optimal predictor $p _ { \theta }$ is the posterior marginal of the $i \cdot$ -th token, conditioned on $x _ { 0 } [ M ]$ for all masks $M$ .

On the other hand, Autoregressive Models (ARMs) learn to predict $i ^ { \mathrm { { t h } } }$ token $x ^ { i }$ based on all preceding tokens, from $x ^ { 0 }$ to $x ^ { i - 1 }$ . This is equivalent to predicting $x ^ { i }$ by masking positions from $i$ to $L - 1$ . Therefore, the training objective for ARMs can be expressed as:

$$
\log p _ { \theta } ( x _ { 0 } ) = \sum _ { i = 0 } ^ { L - 1 } \log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ \{ i , \dots , L - 1 \} ] ) .
$$

Typically, ARMs are trained to predict tokens sequentially from left to right. We refer to this as left-to-right training. However, it’s also possible to train these models to predict tokens sequentially based on a fixed, known permutation of the sequence. We refer to this general approach as orderaware training.

To understand the comparison between the training objective of MDMs and ARMs, we want to highlight the equivalence between any-order autoregressive loss and MDM loss (Hoogeboom et al., 2021a; Ou et al., 2024). In particular, under conditions of Proposition 2.1, MDM loss is equal to

$$
\mathcal { L } _ { \theta } = - \underset { \pi \sim \mathrm { U n i f } ( \mathbb { S } _ { L } ) } { \mathbb { E } } \left[ \sum _ { i = 0 } ^ { L - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } \big [ \pi \{ i , \dots , L - 1 \} \big ] \right) \right] ,
$$

where $\mathrm { U n i f } ( \mathbb { S } _ { L } )$ is a uniform distribution over all the permutations of length $L$ (See Appendix E.1 for the proof). Observe that if the expectation is only with respect to the identity permutation, then the loss becomes an autoregressive loss. This shows that MDM loss solves exponentially more subproblems than ARM loss. In contrast to ARM loss, MDM does not prefer any particular (e.g., left-to-right) order during the training; therefore, we call its training order-agnostic training.

# 2.1.2. ORDER-AGNOSTIC INFERENCE OF MDMS

The MDM inference can be decomposed into two steps: (a) randomly selecting a set of positions to unmask and (b) assigning token values to each position via the denoising network $p _ { \theta }$ . More precisely, we can reformulate the reverse process $x _ { s } \sim g _ { \theta } ( \cdot | x _ { t } )$ as follows.

# Vanilla MDM inference

(a) Sample a set of masked tokens ${ \mathcal { S } } \subseteq \{ i \mid x _ { t } ^ { i } = 0 \}$ , $\begin{array} { r } { \mathbb { P } ( i \in \mathcal { S } ) = \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } } \end{array}$ . (b) For each $i \in S$ , sample $x _ { s } ^ { i } \sim p _ { \theta } ( x ^ { i } | x _ { t } )$ .

Therefore, the inference in MDM is implemented by randomly selecting $S$ and then filling each token value according to the posterior probability $p _ { \theta } ( x _ { s } ^ { i } | x _ { t } )$ .

On the other hand, ARMs are trained to predict tokens sequentially from left to right and therefore, generate tokens also in left-to-right order. In contrast, vanilla MDM inference generates the tokens in a random order.

# 3. MDMs train on hard problems

In this section, we provide theoretical and empirical evidence that when the data distribution has left-to-right order (or any fixed known order) then autoregressive training in left-to-right order (or in the known order) is more tractable than MDMs. In particular, for such distributions with fixed order, we show that ARMs can efficiently sample from the distributions but for MDMs, we theoretically and empirically demonstrate that a large portion of masking subproblems $p _ { \theta } ( x _ { 0 } ^ { i } \mid x _ { 0 } [ M ] )$ can be difficult to learn.

In Section 3.1, we show several examples of simple, nonpathological distributions for which: (1) the masking problems encountered during order-aware training (such as in ARMs) are computationally tractable, yet (2) many of the ones encountered during order-agnostic training (such as in MDMs) are computationally intractable. In Section 3.2, we empirically show that text data also exhibits this gap between the computational complexity of orderaware and order-agnostic training and therefore, MDMs train on subproblems of wide variety of complexity (depending on the order/masks). In Section 3.3, we empirically show that the variety in training complexity results in performance imbalance across subproblems: MDMs trained on data from such distributions exhibits small errors on easy subproblems but suffers from large errors on harder ones.

# 3.1. Benign distributions with hard masking problems

We now describe a simple model of data under which we explore the computational complexity of masking problems and show the contrast between masking problems encountered by MDMs and ARMs.

Definition 3.1. A latents-and-observations $( L \& O )$ distribution is a data distribution $p _ { \mathrm { d a t a } }$ over sequence of length $L$ with alphabet size $m$ (precisely, $p _ { \mathrm { d a t a } }$ is over $\{ 0 , \ldots , m \} ^ { L } )$ is specified by a permutation $\pi$ over indices $\{ 1 , 2 , \ldots , L \}$ , number of latent tokens $N$ , number of observation tokens $P$ such that $N + P = L$ , prior distribution $p _ { \mathrm { p r i o r } }$ of latent variables over $\{ 1 , \ldots , m \}$ and efficiently learnable observation functions $\mathcal { O } _ { 1 } , \dots , \mathcal { O } _ { P } : \{ 1 , \dots , m \} ^ { N } \to \Delta ( \{ 0 , \dots , m \} )$ , 1 • (Latent tokens) For $i = 1 , \ldots , N$ , sample $x ^ { \pi ( i ) }$ independently from the prior distribution $p _ { \mathrm { p r i o r } }$ of the latents.

• (Observation tokens) For $j = 1 , \ldots , P$ , sample $x ^ { \pi ( N + j ) }$ independently from $\mathcal { O } _ { j } ( x ^ { \pi ( 1 ) } , \dots , x ^ { \pi ( N ) } )$ .

L&O distributions contain two types of tokens: (1) latent tokens and (2) observation tokens. Intuitively, latent tokens are tokens in the sequence, indexed by $\pi ( 1 ) , \pi ( 2 ) , \ldots , \pi ( N )$ that serve as “seeds” that provide randomness in the sequence; the remaining tokens, called observation tokens (indexed by $\pi ( N + 1 ) , \pi ( N + 2 ) , \ldots , \pi ( N + P ) )$ , are determined as (possibly randomized) functions of the latent tokens via $\mathcal { O } _ { 1 } , \ldots , \mathcal { O } _ { P }$ . Observe that L&O distributions specified by a permutation $\pi$ have a natural generation order by permutation $\pi$ .

Order-aware training Order-aware training, i.e. by permuting the sequence so that $\pi$ becomes the identity permutation and then performing autoregressive training, is computationally tractable: predicting $x ^ { \bar { \pi } ( i ) }$ given $x ^ { \pi ( 1 ) } , \ldots , x ^ { \pi ( i - 1 ) }$ is trivial when $i \leq N$ as the tokens are independent, and computationally tractable when $i > N$ because $x ^ { \pi ( i ) }$ only depends on $x ^ { \pi ( 1 ) } , \ldots , x ^ { \pi ( N ) }$ and is efficiently learnable by assumption. In contrast, below we will show examples where if one performs order-agnostic training $\grave { a }$ la MDMs, one will run into hard masking problems with high probability.

![](images/figures/masked-diffusion-token-ordering-fig-0003.jpg)  
Figure 2. Left: MDMs train on hard problems (Section 3.2). $\mathbf { X }$ -axis and y-axis correspond to $\log ( \mathrm { F L O P s } )$ and $- \log p \theta ( x )$ , respectively. MDM (Blue) is worse than ARM (Orange) in likelihood modeling. Most masking problems (Other lines) that MDM is trained on are harder than those encountered by ARM, as indicated by small log-likelihoods. Right: Task error imbalance (Section 3.3). MDM’s performance varies across different tasks. For text data (top right), this is indicated by validation loss. For L&O-NAE-SAT (bottom right), MDM performs well on the masking problems for observation positions (light region) but struggles with latent positions (dark region).

Order-agnostic training We first note that if the observations $( \mathcal { O } _ { 1 } , \ldots , \mathcal { O } _ { P } )$ are given by a cryptographic hash function, then the masking problem of predicting $( x ^ { \pi ( 1 ) } , \ldots , x ^ { \pi ( L ) } )$ given $( x ^ { \pi ( N + \bar { 1 } ) } , \bar { , } \dotsc , x ^ { \pi ( N + \bar { P ) } } )$ is computationally intractable by design because it requires inverting the hash function. While this is a well-known folklore observation regarding the role of token ordering in language modeling, it is not entirely satisfying because this construction is worst-case in nature – in real-world data, one rarely trains on sequences given by cryptographic hash functions. Furthermore, it only establishes hardness for a specific masking pattern which need not be encountered in the course of running the reverse process.

We provide several simple instances of L&O distributions that address these issues: instead of leveraging delicate cryptographic constructions, they are average-case in nature and furthermore we can establish hardness for typical masking problems encountered along the reverse process.

In all these examples, the hardness results we establish hold even if the algorithm knows all of the parameters of $p _ { \mathrm { d a t a } }$ as well as the observation functions $\mathcal { O } _ { 1 } , \ldots , \mathcal { O } _ { P }$ . Due to space constraints, here we focus on the following example, deferring two others to Apps. B.1 and B.2.

Example 3.2 (Sparse predicate observations). Consider the following class of L&O distributions. Given arity $k \geq 2$ , fix a predicate function $g : \{ 1 , \ldots , m \} ^ { k }  \{ 0 , 1 \}$ . Consider the set of all ordered subsets of $\{ 1 , 2 , \ldots , N \}$ of size $k$ and set the total number of observation latents $P$ equal to the size of this set (hence $P = N ! / ( N - k ) ! =$ $N ( N - 1 ) \cdots ( N - k + 1 ) / $ . To sample a new sequence, we first sample latent tokens $x ^ { \pi ( 1 ) } , \ldots , x ^ { \pi ( N ) }$ from the prior distribution $p _ { p r i o r }$ and an observation latent corresponding to a $k$ -sized subset $S$ is given by $g ( \{ x ^ { \pi ( i ) } \} _ { i \in S } )$ . In other words, each observation latent corresponds to a $k$ -sized subset $S$ of $\{ 1 , 2 , \ldots , N \}$ and the corresponding observation function $\mathcal { O } _ { S } ( x ^ { \pi ( 1 ) } , \dots , x ^ { \pi ( N ) } )$ is given by $g ( \{ x ^ { \pi ( i ) } \} _ { i \in S } )$ .

Proposition 3.3. Let $x$ be a sample from an L&O distribution pdata with sparse predicate observations as defined in Example 3.2, with arity $k$ and predicate $g$ satisfying Assumption B.11, and let $\gamma$ be the probability that $g$ is satisfied by a random assignment from $\{ 1 , \ldots , m \} ^ { k }$ Let $D _ { \mathrm { K S } }$ and $D _ { \mathrm { c o n d } }$ be some constants associated with the predicate function $g$ (see Definition B.12). Suppose each token in $x$ is independently masked with probability $\alpha$ , and $M$ is the set of indices for the masked tokens. If $1 - \gamma ^ { - 1 } D _ { \mathrm { K S } } / k N ^ { k - 1 } \leq \alpha \leq 1 - \gamma ^ { - 1 } D _ { \mathrm { c o n d } } / k N ^ { k - 1 }$ , then under the 1RSB cavity prediction (see Conjecture B.13), with probability $\Omega _ { k } ( 1 )$ over the randomness of the masking, no polynomial-time algorithm can solve the resulting subproblem of predicting any of the masked tokens among $x ^ { \pi ( 1 ) } , \ldots , x ^ { \pi ( N ) }$ given $x [ M ]$ .

The complete proof of the proposition is given in Appendix B.4. We also provide a proof outline in Appendix B.3 for a comprehensive understanding.

# 3.2. Empirical evidence of hardness via likelihoods

In the previous section, we provided theoretical evidence that order-aware training is tractable when data has a natural order but the order-agnostic training is not. In this section, we provide empirical evidence to support this claim, using natural text data. Additionally, recent studies (Nie et al., 2024; Zheng et al., 2024) have shown that masked diffusion models (MDMs) underperform compared to autoregressive models (ARMs) on natural text data. In this section, we provide evidence that this performance gap is primarily due to the order-agnostic training of MDMs. Natural text inherently follows a left-to-right token order, and we show that as training deviates from this order, model performance progressively declines.

To understand the importance of the order during the training, we use the following setting: Given a permutation $\pi$ of indices $\{ 0 , 1 , \ldots , L - 1 \}$ , define a $\pi$ -learner to be a likelihood model $\log p _ { \theta } ( x _ { 0 } )$ given as follows:

$$
\log p _ { \theta } ( x _ { 0 } ) = \sum _ { i = 0 } ^ { L - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } [ \pi \{ i , \dots , L - 1 \} ] \right)
$$

In other words, the $\pi$ -learner predicts the token at position $\pi ( i )$ given the clean tokens $x _ { 0 } ^ { \bar { \pi } ( 0 ) } , \ldots , x _ { 0 } ^ { \pi ( i - 1 ) }$ and masked tokens x0 $x _ { 0 } ^ { \pi ( i ) } , \ldots , x _ { 0 } ^ { \pi ( L - 1 ) }$ xπ(L−1)0 . If π is the identity permutation, this reduces to the standard (left-to-right) autoregressive training. Note that the MDM loss encodes a $\pi$ -learner for every permutation $\pi$ because the MDM loss (1) is equivalent to the average loss of those $\pi$ -learners over $\pi$ sampled from $\mathrm { U n i f } ( \mathbb { S } _ { L } )$ :

$$
\mathcal { L } _ { \theta } = - \underset { \pi \sim \mathrm { U n i f } ( \mathbb { S } _ { L } ) } { \mathbb { E } } \left[ \sum _ { i = 0 } ^ { L - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } \big [ \pi \{ i , \dots , L - 1 \} \big ] \right) \right] ,
$$

where $\mathbb { S } _ { L }$ denotes the set of all permutations over $\{ 0 , 1 , \ldots , L - 1 \}$ . The proof of the above equivalence is given in Appendix E. Therefore, by measuring the ‘hardness’ of each $\pi$ -learner, we can probe differences in hardness between arbitrary masking problems and left-to-right masking problems.

Experimental setup. We use the Slimpajama dataset (Soboleva et al., 2023) to evaluate the performance of training in different orders. To train a $\pi$ -learner, we employ a transformer with causal attention and use permuted data $\pi ( \boldsymbol { x } _ { 0 } )$ as input. By varying $\pi$ while maintaining all other training configurations (e.g., model, optimization), we can use the resulting likelihood (computed using Equation (3)) as a metric to capture the hardness of subproblems solved by the $\pi$ -learner.

In our experiments, the sequence length $L$ is 2048, so repeating the scaling laws for each $\pi$ is infeasible. Instead, we sample $\pi \sim \mathrm { U n i f } ( \mathbb { S } _ { L } )$ and examine the scaling law of the $\pi$ -learner’s likelihood. We leverage the codebase from (Nie et al., 2024), where the baseline scaling laws of MDM and ARM were introduced. Moreover, given that RoPE has an inductive bias towards left-to-right ordering, we employ a learnable positional embedding layer for all experiments to correct this. Consequently, we also re-run the baseline results, where RoPE was employed. To investigate how the distance between $\pi$ and the identity permutation affects the scaling law, we consider two interpolating distributions over permutations between $\mathrm { U n i f } ( \mathbb { S } _ { L } )$ (i.e, MDM training) and the point mass at the identical permutation (i.e, ARM training). We sample three permutations from the interpolating distribution and $\mathrm { U n i f } ( \mathbb { S } _ { L } )$ and plot the scaling law for each of the permutation. Due to space constraints, we provide further experimental details in Appendix C.1.

Results. As shown in Fig. 2, the scaling law for a $\pi$ -learner with uniformly random $\pi$ is worse than that of an ARM. This elucidates the inherent hardness of masking problems $p _ { \theta } ( x _ { i } \mid x _ { 0 } [ M ] )$ beyond left-to-right prediction and also explains why MDM, which is trained simultaneously on all $\pi \in \mathbb { S } _ { L }$ , is worse than ARM in likelihood modeling. Additionally, as $\pi$ gets closer to the identity permutation, the scaling laws also get closer to ARM ( $\bar { \pi }$ -learner-closer and $\pi$ -learner-much-closer in Fig. 2). This also supports the common belief that ARM is a good fit for text data as it inherently follows a left-to-right ordering.

That said, it should also be noted that even though MDMs are trained on exponentially more masking problems than ARM $\left( \Theta ( L 2 ^ { L } ) \right)$ versus $L$ ), its performance is not significantly worse than $\pi$ -learners. We attribute this to the blessing of task diversity; multi-task training can benefit both the optimization dynamics (Kim et al., 2024) and validation performance (Tripuraneni et al., 2021; Maurer et al., 2016; Ruder, 2017) due to positive transfers across tasks.

# 3.3. Error is imbalanced across masking problems

In previous sections, we have demonstrated that the hardness of different masking problems $p _ { \theta } ( x ^ { i } \mid x _ { 0 } [ M ] )$ can vary significantly, potentially hindering the MDM’s learning. In this section, we provide empirical evidence that the MDM’s final performance exhibits a similar imbalance across subproblems. Details are provided in App. C.2.

L&O-NAE-SAT. Consider an L&O distribution with $\pi$ given by the identity permutation and where each observation ${ \mathcal { O } } _ { j }$ is deterministically given by $\mathrm { N A E } ( x _ { i _ { 1 } } , x _ { i _ { 2 } } , x _ { i _ { 3 } } ) \triangleq$ $1 - \mathbf { 1 } [ x _ { i _ { 1 } } = x _ { i _ { 2 } } = x _ { i _ { 3 } } ]$ for some randomly chosen (prefixed) triples $( i _ { 1 } , i _ { 2 } , i _ { 3 } ) \ \in \ [ N ]$ . For an MDM trained on this distribution, we measure the error it achieves on each task $\log p _ { \theta } ( x _ { 0 } | x _ { 0 } [ M ] )$ via $\mathbb { E } _ { x _ { 0 } } \bigg \| \log p _ { \theta } ( x _ { 0 } | x _ { 0 } [ M ] ) -$ $\log p _ { \mathrm { d a t a } } ( x _ { 0 } | x _ { 0 } [ M ] ) \Big \| ^ { 2 }$ , where $p _ { \mathrm { d a t a } } ( x _ { 0 } | x _ { 0 } [ M ] )$ denotes the Bayes-optimal predictor. Technically, we do not have access to this, so instead we train another MDM for a much larger number of iterations and use this as a proxy. Fig. 2 reveals that prediction tasks for latent positions (light region) exhibit larger errors compared to those for observation positions (dark region).

Text. Here we revisit the text experiment from Section 3.2. Since we do not have access to the Bayes-optimal predictor, we use the metric $\begin{array} { r } { \mathbb { E } _ { x _ { 0 } \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { i = 0 } ^ { \hat { L } - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } [ \pi \{ i , \dots , L - 1 \} ] \right) \right] } \end{array}$ . This captures the accumulation of error across subproblems $p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } [ \pi \{ i , \dots , L - 1 \} ] \right)$ , since $p _ { \theta } ( x _ { 0 } | x _ { 0 } [ M ] ) = p _ { \mathrm { d a t a } } ( x _ { 0 } | x _ { 0 } [ M ] )$ minimizes this metric. Fig. 2 shows a clear gap between different subproblems.

The theoretical and empirical evidence demonstrates that MDMs perform better in estimating $p _ { \theta } ( x _ { 0 } | x _ { 0 } [ M ] )$ for some subproblems $M$ than for others. We therefore want to avoid encountering hard subproblems $M$ at inference time. In the next section, we show that while vanilla MDM inference can run into such subproblems, simple modifications at the inference stage can effectively circumvent these issues, resulting in dramatic, training-free performance improvements.

# 4. MDMs can plan around hard problems

We previously argued that due to the complex nature of masking subproblems, MDM must perform poorly on certain ones $p _ { \theta } ( x ^ { i } | x _ { t } )$ . Therefore, during vanilla MDM inference, MDM inevitably encounters such difficult subproblems at Step (b). While this might suggest that we need to fundamentally revisit how MDMs are trained, in this section we show that, surprisingly, simple modifications at the inference stage—without any further training—can sidestep these issues and lead to significant performance improvements.

MDM offers multiple sampling paths. The vanilla MDM inference (Algorithm 1) aim to align the intermediate distributions with the forward process, as used in continuous diffusion. However, unlike continuous diffusion, the reverse process of MDM allows multiple valid sampling paths (different orders of unmasking the tokens) that match the starting distribution of the forward process of MDM.

We first show that when we have an ideal MDM that perfectly solves all masking problems, i.e., $p _ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] ) =$ $p _ { \mathrm { d a t a } } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] )$ , then using any sampling path (unmasking the tokens in any order) results in the same distribution. Consider the following sampler: For every step, $S$ is a set with one index selected agnostically (without following any distribution). For any clean sample $x _ { 0 }$ generated by this sampler, note that $p _ { \theta } ( x _ { 0 } ) \ =$ $\begin{array} { r } { \prod _ { i = 0 } ^ { L - 1 } p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } [ \pi \{ i , \dots , L - 1 \} ] \right) } \end{array}$ by chain rule, and this is equal to $\begin{array} { r } { \prod _ { i = 0 } ^ { L - 1 } p _ { \mathrm { d a t a } } \left( x _ { 0 } ^ { \pi ( i ) } \Big | x _ { 0 } \big [ \pi \{ i , \dots , L - 1 \} \big ] \right) = } \end{array}$ $p _ { \mathrm { d a t a } } ( x _ { 0 } )$ . Therefore, other choices of $S$ , not necessarily following Algorithm 1, still capture the true likelihood.

![](images/figures/masked-diffusion-token-ordering-fig-0004.jpg)  
Figure 3. Generative Perplexity. We compare the resulting generative perplexity (GenPPL) of adaptive vs. vanilla MDM inference. We employ a pretrained 170M MDM and LLaMA-7B (Touvron et al., 2023) as inference and evaluation, respectively. Adaptive MDM inference (Blue) leads to a substantial reduction in generative perplexity, while maintaining the entropy.

In practice, unlike this ideal case, MDM does not perform equally well on all subproblems, as shown in Section 3.3. Consequently, different sampling paths result in varying likelihood modeling abilities. Motivated by this observation, we consider adaptive inference for MDMs:

# Adaptive MDM inference

(a) Sample a set of masked tokens ${ \mathcal { S } } = { \mathcal { F } } \left( \theta , x _ { t } \right) \subseteq$ $\{ i \mid x _ { t } ^ { i } = 0 \}$ . (b) For each $i \in S$ , sample $x _ { s } ^ { i } \sim p _ { \theta } ( x ^ { i } | x _ { t } )$ .

Instead of selecting $S$ randomly, adaptive MDM inference leverages an oracle $\mathcal { F } ( \boldsymbol { \theta } , \boldsymbol { x } _ { t } )$ to select $S$ strategically to avoid hard masking problems. This naturally raises the question of how to design an effective oracle $\mathcal { F }$ .

In the following sections, we demonstrate that adaptive MDM inference with careful choices of $\mathcal { F }$ enhance MDM’s likelihood matching ability. In other words, a pretrained MDM, even if it performs poorly on certain hard subproblems, still contains sufficient information to avoid them when paired with an effective oracle $\mathcal { F }$ .

# 4.1. Effective design of ordering oracle

We introduce two different oracles, Top probability and Top probability margin. Intuitively, both strategies are based on the idea that $S$ should be selected based on how “certain” the model is about each position. We caution that these strategies should not be confused with notions like nucleus sampling in ARMs (Holtzman et al., 2019); the oracles we describe are for selecting the position of the next token to decode, rather than the value, and thus are only meaningful in the context of MDMs.

Table 1. L&O-NAE-SAT. Adaptive MDM inference achieves better likelihood matching than vanilla MDM inference. Note that naive guessing leads to $7 5 \%$ accuracy, indicating that vanilla inference performs similarly or worse than naive guessing.   

<table><tr><td>(N, P)</td><td>Vanilla inference</td><td>Adaptive inference</td></tr><tr><td>(25, 275)</td><td>78.06%</td><td>93.76%</td></tr><tr><td>(30, 270)</td><td>75.70%</td><td>93.54%</td></tr><tr><td>(40, 260)</td><td>74.60%</td><td>92.21%</td></tr><tr><td>(50, 250)</td><td>67.94%</td><td>90.01%</td></tr><tr><td>(100, 200)</td><td>62.84%</td><td>88.91%</td></tr></table>

Top probability (Zheng et al., 2023). Suppose we want to unmask $K$ positions at time step $t$ , i.e., select $| S | = K$ In the top probability, the uncertainty of a position is estimated by the maximum probability assigned to any value in the vocabulary. More precisely, the certainty at position $i$ is $\begin{array} { r } { \operatorname* { m a x } _ { j \in \{ 0 , \dots , m - 1 \} } p _ { \theta } ( x ^ { i } = j | x _ { t } ) } \end{array}$ and $\mathcal { F } ( \boldsymbol { \theta } , \boldsymbol { x } _ { t } ) =$ Top $K \left( \operatorname* { m a x } p _ { \theta } ( x ^ { i } | x _ { t } ) \right)$ .

Top probability strategy is a good proxy for many tasks and works well in practice (Zheng et al., 2023; Ye et al., 2024; Wang et al., 2024). However, this approach can often provide misleading estimates of uncertainty. Consider when an MDM is confused between two token values, thus assigning them almost equal but high probabilities. In this case, unmasking according to top probability may still choose to unmask this position, despite its uncertainty. To mitigate this issue, we propose the following alternative strategy.

Top probability margin. In this strategy, the uncertainty of a position is instead estimated using the absolute difference between the two most probable values at position $i$ . More precisely, if $j _ { 1 }$ and $j _ { 2 }$ are the two most probable values in vocabulary according to $p _ { \theta } ( x ^ { i } | x _ { t } )$ in position $i$ , the certainty in the position is given by $| p _ { \theta } ( x ^ { i } \ = \ j _ { 1 } | x _ { t } ) \ - \ p _ { \theta } ( x ^ { i } \ = \ j _ { 2 } | x _ { t } ) |$ and $\mathcal { F } ( \boldsymbol { \theta } , \boldsymbol { x } _ { t } ) \ =$ Top $K \left( | p _ { \theta } ( x ^ { i } = j _ { 1 } | x _ { t } ) - p _ { \theta } ( x ^ { i } = j _ { 2 } | x _ { t } ) | \right)$ . When multiple values have similar probabilities at a position, top probability margin strategy will provide a better estimate of the uncertainty of a position, and when there is a single best choice of value then top probability and top probability margin work similarly.

# 4.2. Adaptive MDM inference

In this section, we experimentally validate that adaptive MDM inference helps MDMs avoid hard subproblems, leading to better likelihood matching. We first show our results on L&O-NAE-SAT and text data, before turning to our primary application to logic puzzles.

Table 2. Comparison of accuracy for solving the Sudoku puzzle.   

<table><tr><td>Method</td><td># Param</td><td>Accuracy</td></tr><tr><td>ARM (w/o ordering) ARM (with ordering)</td><td>42M</td><td>9.73% 87.18%</td></tr><tr><td>MDM (vanilla)</td><td></td><td>6.88%</td></tr><tr><td>MDM (Top probability) MDM (Top prob. margin)</td><td>6M</td><td>18.51% 89.49%</td></tr></table>

L&O-NAE-SAT and text data. For the L&O-NAE-SAT distribution defined in Section 3.3, we evaluate the effectiveness of adaptive inference by measuring the accuracy in predicting the observation tokens. Table 1 in the appendix reveals a clear improvement over vanilla inference. For the text dataset, we evaluate using the standard metric of generative perplexity, by which likelihood is measured by a large language model. We also compute the entropy of the generated samples to ensure both inference strategies exhibit similar levels of diversity. As shown in Fig. 3, we observe a substantial decrease in generative perplexity using adaptive inference. We defer further experimental details to Appendix D.1.

Logic puzzles. We consider two different types of logic puzzles: Sudoku and Zebra (Einstein) puzzles. Intuitively, for Sudoku, some empty (masked) cells are significantly easier to predict than others and we want to choose the cells that are easier to predict during the inference. We evaluate the effectiveness of adaptive MDM inference over vanilla MDM inference in selecting such cells.2

To measure the performance of an inference method, we use the percentage of correctly solved puzzles. For both puzzles, we use train and test datasets from (Shah et al., 2024). For the Sudoku puzzle (Table 2) we observe that adaptive MDM inference, in particular, Top probability margin strategy, obtains substantially higher accuracy $( 8 9 . 4 9 \% )$ compared to vanilla MDM inference $( 6 . 8 8 \% )$ . Additionally, Top probability margin obtains higher accuracy $( 8 9 . 4 9 \% )$ than Top probability strategy $( 1 8 . 5 1 \% )$ ). As mentioned in Section 4.1, this is because Top probability margin strategy more reliably estimates uncertainty when multiple competing values are close in probability at a given position, as is often the case in Sudoku. For the Zebra puzzle, as shown in Table 3, we observe a consistent result: Top probability $( 9 8 . 5 \% )$ and

Top probability margin $( 9 8 . 3 \% )$ outperform vanilla MDM inference $( 7 6 . 9 \% )$ .

Table 3. Comparison of accuracy for solving the Zebra puzzle.   

<table><tr><td>Method</td><td># Param</td><td>Accuracy</td></tr><tr><td>ARM (w/o ordering) ARM (with ordering)</td><td>42M</td><td>80.31 % 91.17 %</td></tr><tr><td>MDM (vanilla) MDM (Top probability)</td><td>19M</td><td>76.9 % 98.5 %</td></tr><tr><td>MDM (Top prob. margin)</td><td></td><td>98.3 %</td></tr></table>

# 4.3. Eliciting sequence-dependent reasoning paths using adaptive MDM inference in logic puzzles

In this section, we study the effectiveness of adaptive MDM inference in finding the right reasoning/generation order for tasks where every sequence has a different “natural” order. To do so, we will compare the performance of adaptive MDM inference to that of ARM on Sudoku and Zebra puzzles. For these puzzles, the natural order of generation is not only different from left-to-right, but it is also sequencedependent. For such tasks, prior works have shown that ARMs struggle if the information about the order is not provided during the training (Shah et al., 2024; Lehnert et al., 2024). Therefore, to obtain a strong baseline, we not only consider an ARM trained without the order information but also consider an ARM trained with the order information for each sequence in the training data. Note that the latter is a much stronger baseline than the former as one can hope to teach the model to figure out the correct order by some form of supervised teacher forcing (as performed in Shah et al. (2024); Lehnert et al. (2024)), eliminating the issue of finding the right order in an unsupervised manner.

We compare ARMs and MDMs for Sudoku in Table 2 and Zebra puzzles in Table 3. We observe that for both, Top probability margin-based adaptive MDM inference not only outperforms the ARM trained without ordering information, but it even outperforms the ARM trained with ordering information! This shows that the unsupervised way of finding the correct order and solving such logic puzzles using adaptive MDM inference outperforms the supervised way of finding the correct order and solving such puzzles using an ARM, and is significantly less computationally intensive.

# 4.4. Adaptive MDM inference on natural language tasks

To examine the effect of different inference strategies on text benchmarks, we adapted LLaDA, the 8B MDM model from (Nie et al., 2025). We compare three inference strategies: vanilla, top probability, and top probability margin. The results are presented in Table 4.

We see that both adaptive MDM inference strategies, top probability and top probability margin, consistently outperform vanilla MDM inference. Notably, top probability margin demonstrates a clear advantage over top probability in challenging tasks like HumanEval-Multiline (infill), HumanEval-Split Line (infill), and Math. This is because Top probability margin provides a more reliable estimate of uncertainty when multiple tokens have similar probabilities, a frequent occurrence in these difficult tasks. These results further underscore the potential for developing new, sophisticated adaptive inference strategies for various tasks. We provide experimental details in Appendix D.3.

# 4.5. Easy to hard generalization

In the previous section we showed that when the training and inference sequences come from the same distribution, order-agnostic training of MDMs combined with adaptive inference can perform very well on logic puzzles. To evaluate if the model has learned the correct way of solving the puzzles and test the robustness of adaptive inference, we also test the MDMs on harder puzzles than the ones from training, for Sudoku.

We keep the training dataset the same as proposed in Shah et al. (2024). Shah et al. (2024) created this dataset from Radcliffe (2020) by selecting the puzzles that can be solved using 7 fixed strategies and do not require backtrackingbased search. We use the remaining puzzles in Radcliffe (2020) as our hard dataset. Hence, these puzzles all use a strategy not seen during training and/or backtracking to obtain the correct solution.

We measure the accuracy of MDMs and ARMs on the hard test set and present the results in Table 5. We see that the Top probability margin-based adaptive MDM inference strategy $( 4 9 . 8 8 \% )$ again significantly outperforms ARMs trained with order information $( 3 2 . 5 7 \% )$ . In particular, although the accuracy drops for both methods due to the more challenging test set, MDMs with adaptive inference appear to be more robust to this distribution shift than ARMs. We believe this is due to the fact that MDMs try to solve a significantly higher number of infilling problems than ARMs $( \exp ( L )$ compared to $L$ ) and therefore are able to extract knowledge about the problem more efficiently than ARMs.

# 5. Conclusion

In this work, we examined the impact of token generation order on training and inference in MDMs. We provided theoretical and experimental evidence that MDMs train on hard masking problems. We also demonstrated that adaptive inference strategies can be used to sidestep these hard problems. For logic puzzles, we find that this leads to dramatic improvements in performance not just over vanilla MDMs, but even over ARMs trained with teacher forcing to learn the right order of decoding. An important direction for future work is to go beyond the relatively simple adaptive strategies to find a better generation order like top probability and top probability margin considered here.

Table 4. Performance of different inference strategies for LLaDa 8B model on coding and math tasks.   

<table><tr><td>Method</td><td>HumanEval-Single</td><td>HumanEval-Multi</td><td>HumanEval-Split</td><td>Math</td><td>MMLU</td><td>ROCStories</td></tr><tr><td>Vanilla</td><td>31.8%</td><td>16.5%</td><td>14.2%</td><td>28.5%</td><td>33.2%</td><td>21.23%</td></tr><tr><td>Top probability</td><td>32.9%</td><td>20.8%</td><td>18.4%</td><td>31.3%</td><td>36.5%</td><td>21.10%</td></tr><tr><td>Top prob. margin</td><td>33.5%</td><td>25.4%</td><td>22.3%</td><td>34.3%</td><td>35.4%</td><td>21.41%</td></tr></table>

Table 5. Comparison of accuracy for solving the hard Sudokus.   

<table><tr><td>Method</td><td>#Param</td><td>Accuracy</td></tr><tr><td>ARM (with ordering)</td><td>42M</td><td>32.57 %</td></tr><tr><td>MDM (random)</td><td></td><td>3.62 %</td></tr><tr><td>MDM (Top probability)</td><td>6M</td><td>9.44 %</td></tr><tr><td>MDM (Top prob. margin)</td><td></td><td>49.88 %</td></tr></table>

Acknowledgements. JK thanks Kiwhan Song for discussions about MDM training. KS and VK are supported by the NSF AI Institute for Foundations of Machine Learning (IFML). KS and VK thank the computing support on the Vista GPU Cluster through the Center for Generative AI (CGAI) and the Texas Advanced Computing Center (TACC) at UT Austin. KS thanks Nishanth Dikkala for the initial discussions about the project. SK acknowledges: this work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence and support from the Office of Naval Research under award N00014-22-1-2377. SC is supported by the Harvard Dean’s Competitive Fund for Promising Scholarship and thanks Brice Huang and Sidhanth Mohanty for enlightening discussions about computational-statistical tradeoffs for planted CSPs.

# Impact statement

This paper advances the understanding of discrete diffusion models, contributing to the broader field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Alaoui, A. E. and Gamarnik, D. Hardness of sampling solutions from the symmetric binary perceptron. arXiv preprint arXiv:2407.16627, 2024.

Alekhnovich, M. More on average case vs approximation complexity. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings., pp. 298–307. IEEE, 2003.

Aubin, B., Perkins, W., and Zdeborova, L. Storage capacity ´ in symmetric binary perceptrons. Journal of Physics A: Mathematical and Theoretical, 52(29):294003, 2019.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. Structured denoising diffusion models in discrete state-spaces. NeruIPS, 2021.

Bavarian, M., Jun, H., Tezak, N., Schulman, J., McLeavey, C., Tworek, J., and Chen, M. Efficient training of language models to fill in the middle, 2022. URL https://arxiv.org/abs/2207.14255.

Bormashenko, O. A coupling argument for the random transposition walk. arXiv preprint arXiv: 1109.3915, 2011.

Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T. Maskgit: Masked generative image transformer. CVPR, 2022.

Chen, H. and Ying, L. Convergence analysis of discrete diffusion model: Exact implementation through uniformization. arXiv preprint arXiv: 2402.08095, 2024.

Chen, X., Chi, R. A., Wang, X., and Zhou, D. Premise order matters in reasoning with large language models. arXiv preprint arXiv:2402.08939, 2024.

Decelle, A., Krzakala, F., Moore, C., and Zdeborova, L.´ Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications. Phys. Rev. E, 84:066106, Dec 2011.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for

Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019.

Gamarnik, D. The overlap gap property: A topological barrier to optimizing over random structures. Proceedings of the National Academy of Sciences, 118(41):e2108492118, 2021.

Golovneva, O., Allen-Zhu, Z., Weston, J., and Sukhbaatar, S. Reverse training to nurse the reversal curse. arXiv preprint arXiv:2403.13799, 2024.

Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M., An, C., Zhao, P., Bi, W., Han, J., et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.

Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., Berg, R. v. d., and Salimans, T. Autoregressive diffusion models. arXiv preprint arXiv:2110.02037, 2021a.

Hoogeboom, E., Nielsen, D., Jaini, P., Forre, P., and Welling, ´ M. Argmax flows and multinomial diffusion: Learning categorical distributions. NeurIPS, 2021b.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Kim, J., Kwon, S., Choi, J. Y., Park, J., Cho, J., Lee, J. D., and Ryu, E. K. Task diversity shortens the icl plateau. arXiv preprint arXiv:2410.05448, 2024.

Kitouni, O., Nolte, N. S., Williams, A., Rabbat, M., Bouchacourt, D., and Ibrahim, M. The factorization curse: Which tokens you predict underlie the reversal curse and more. Advances in Neural Information Processing Systems, 37: 112329–112355, 2025.

Krzakala, F. and Zdeborova, L. Hiding quiet solutions in ´ random constraint satisfaction problems. Physical review letters, 102(23):238701, 2009.

Lehnert, L., Sukhbaatar, S., Su, D., Zheng, Q., McVay, P., Rabbat, M., and Tian, Y. Beyond a\*: Better planning with transformers via search dynamics bootstrapping. 2024.

Liao, Y., Jiang, X., and Liu, Q. Probabilistically masked language model capable of autoregressive generation in arbitrary word order. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 263–274. Association for Computational Linguistics, 2020.

Liu, A., Broadrick, O., Niepert, M., and Broeck, G. V. d. Discrete copula diffusion. arXiv preprint arXiv:2410.01949, 2024a.

Liu, S., Mohanty, S., and Raghavendra, P. On statistical inference when fixed points of belief propagation are unstable . In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pp. 395–405. IEEE Computer Society, 2022.

Liu, S., Nam, J., Campbell, A., Stark, H., Xu, Y., Jaakkola, ¨ T., and Gomez-Bombarelli, R. Think while you generate: ´ Discrete diffusion with planned denoising. arXiv preprint arXiv:2410.06264, 2024b.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Lou, A., Meng, C., and Ermon, S. Discrete diffusion modeling by estimating the ratios of the data distribution. ICML, 2024.

Maurer, A., Pontil, M., and Romera-Paredes, B. The benefit of multitask representation learning. JMLR, 17(81):1–32, 2016.

Montanari, A. Estimating random variables from random sparse observations. European Transactions on Telecommunications, 19(4):385–403, 2008.

Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M., and Li, C. Scaling up masked diffusion models on text. arXiv preprint arXiv:2410.18514, 2024.

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., and Li, C. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

Papadopoulos, V., Wenger, J., and Hongler, C. Arrows of time for large language models. arXiv preprint arXiv:2401.17505, 2024.

Peng, F. Z., Bezemek, Z., Patel, S., Yao, S., Rector-Brooks, J., Tong, A., and Chatterjee, P. Path planning for masked diffusion model sampling. arXiv preprint arXiv:2502.03540, 2025.

Radcliffe, D. G. 3 million sudoku puzzles with ratings, 2020. URL https://www.kaggle.com/ dsv/1495975.

Rector-Brooks, J., Hasan, M., Peng, Z., Quinn, Z., Liu, C., Mittal, S., Dziri, N., Bronstein, M., Bengio, Y., Chatterjee, P., et al. Steering masked discrete diffusion models via discrete denoising posterior prediction. arXiv preprint arXiv:2410.08134, 2024.

Ruder, S. An overview of multi-task learning in deep neural networks. arXiv 1706.05098, 2017.

Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J., Rush, A., and Kuleshov, V. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136– 130184, 2025.

Schiff, Y., Sahoo, S. S., Phung, H., Wang, G., Boshar, S., Dalla-torre, H., de Almeida, B. P., Rush, A., Pierrot, T., and Kuleshov, V. Simple guidance mechanisms for discrete diffusion models. arXiv preprint arXiv:2412.10193, 2024.

Shah, K., Dikkala, N., Wang, X., and Panigrahy, R. Causal language modeling can elicit search and reasoning capabilities on logic puzzles. arXiv preprint arXiv:2409.10502, 2024.

Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K. Simplified and generalized masked diffusion for discrete data. NeurIPS, 2024.

Shih, A., Sadigh, D., and Ermon, S. Training and inference on any-order autoregressive models the right way. NeurIPS, 2022.

Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hestness, J., and Dey, N. Slimpajama: A 627b token cleaned and deduplicated version of redpajama, June 2023.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen,

M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv: 2307.09288, 2023.

Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. ICML, 2015.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. ICLR, 2021.

Tripuraneni, N., Jin, C., and Jordan, M. I. Provable metalearning of linear representations. ICML, 2021.

Varma, H., Nagaraj, D., and Shanmugam, K. Glauber generative model: Discrete diffusion models via binary classification. arXiv preprint arXiv: 2405.17035, 2024.

Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., and Gu, Q. Diffusion language models are versatile protein learners. ICML, 2024.

Xu, M., Geffner, T., Kreis, K., Nie, W., Xu, Y., Leskovec, J., Ermon, S., and Vahdat, A. Energy-based diffusion language models for text generation. arxiv preprint arXiv: 2410.21357, 2024.

Ye, J., Gao, J., Gong, S., Zheng, L., Jiang, X., Li, Z., and Kong, L. Beyond autoregression: Discrete diffusion for complex reasoning and planning. arXiv preprint arXiv: 2410.14157, 2024.

Zhang, P., Zeng, G., Wang, T., and Lu, W. Tinyllama: An open-source small language model. arXiv preprint arXiv: 2401.02385, 2024.

Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., and Zhang, Q. Masked diffusion models are secretly timeagnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.

Zheng, L., Yuan, J., Yu, L., and Kong, L. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.

# A. Related works

Discrete diffusion models. (Continuous) diffusion models were originally built on continuous-space Markov chains with Gaussian transition kernels (Sohl-Dickstein et al., 2015; Ho et al., 2020). This was later extended to continuous time through the theory of stochastic differential equations (Song et al., 2021). In a similar vein, discrete diffusion models have emerged from discrete-space Markov chains (Hoogeboom et al., 2021b). Specifically, (Austin et al., 2021) introduced D3PM with various types of transition matrices. Later, Lou et al. (2024) proposed SEDD, incorporating a theoretically and practically robust score-entropy objective. Additionally, Varma et al. (2024); Liu et al. (2024b) introduced novel modeling strategies that classify tokens in a noisy sequence as either signal (coming from clean data) or noise (arising from the forward process). In particular, Liu et al. (2024b) uses this to give a planner that adaptively determines which tokens to denoise. While this is similar in spirit to our general discussion about devising adaptive inference strategies, we emphasize that their approach is specific to discrete diffusions for which the forward process scrambles the token values, rather than masking them.

Masked diffusion models. Meanwhile, the absorbing transition kernel has gained popularity as a common choice due to its better performance than other kernels. Building on this, Sahoo et al. (2025); Shi et al. (2024) aligned its framework with continuous diffusion, resulting in a simple and principled training recipe, referring to it as Masked Diffusion Model. Subsequent studies have explored various aspects of MDM. Gong et al. (2024) efficiently trained MDM via adaptation from autoregressive models, scaling MDM up to 7B parameters. Zheng et al. (2024) interpreted MDMs as order-agnostic learners and proposed a first-hitting sampler based on this insight. Ye et al. (2024); Gong et al. (2024) demonstrated that MDM outperforms autoregressive models in reasoning and planning tasks, emphasizing its impact on downstream applications. Nie et al. (2024) examined the scaling laws of MDM, while Xu et al. (2024); Liu et al. (2024a) identified limitations in capturing coordinate dependencies when the number of sampling steps is small and proposed additional modeling strategies to address this issue. Schiff et al. (2024) studied conditional generation using MDM and Rector-Brooks et al. (2024) tackled the challenge of controlling generated data distributions through steering methodologies. Chen & Ying (2024) provided a theoretical analysis showing that sampling error is small given accurate score function estimation.

Any-order reasoning. Even though language tasks generally have a natural order of “left-to-right” token generation, in many tasks like planning, reasoning, and combinatorial optimization, the natural order of token generation can be quite different from “left-to-right”. Even though prominent autoregressive-based language models achieve impressive performance on various tasks, many works (Golovneva et al., 2024; Chen et al., 2024; Kitouni et al., 2025) have shown that this performance is tied to the training order of the tasks and therefore can cause brittleness from it. For example, Chen et al. (2024) showed that simply permuting the premise order on math tasks causes a performance drop of $30 \%$ . The reason behind such brittleness regarding the ordering is the inherent “left-to-right” nature of the autoregressive models. Several works (Liao et al., 2020) have tried to address this issue in the autoregressive framework. In particular, (Papadopoulos et al., 2024) highlighted the significance of left-to-right ordering in natural language by comparing its likelihood to that of the reverse (right-to-left) ordering.

Recently, discrete diffusion models have emerged as a promising approach for discrete data apart from autoregressive models. Additionally, the order-agnostic training of discrete diffusion models opens up the multiple sampling paths during the inference but it also faces some challenges during the training therefore, they seem a promising approach to elicit any order reasoning. Zheng et al. (2023) proposed different ways of implementing an adaptive inference strategy for MDM but a concrete understanding of why such an adaptive inference strategy is needed is still lacking. In this work, we explore various aspects of vanilla MDM training and how adaptive MDM inference can mitigate the issues raised by vanilla MDM training and elicit any order reasoning.

We also want to mention the concurrent work by Peng et al. (2025) that proposes an alternative adaptive inference strategy by selecting $\mathcal { F } ( \boldsymbol { \theta } , \boldsymbol { x } _ { t } )$ based on the BERT model or the denoiser itself. In particular, Peng et al. (2025) uses the BERT model or the denoiser to obtain the uncertainty of a token and then uses Top- $K$ to decide the positions to unmask it. In contrast to their work, we disentangle the impact of token ordering on MDM training vs. MDM inference and provide a more complete understanding of the motivations for and benefits of adaptive inference. Additionally, our results indicate drawbacks to using Top- $K$ strategy as opposed to Top- $K$ margin in deciding which tokens to unmask when there are multiple values with high probabilities.

Beyond autoregressive models. Efforts to learn the natural language using non-autoregressive modeling began with BERT (Devlin et al., 2019). Non-causal approaches can take advantage of the understanding the text data representation.

(Chang et al., 2022) adopted a similar approach for learning image representations. Building on these intuitions, (Shih et al., 2022; Hoogeboom et al., 2021a) proposed any-order modeling, which allows a model to generate in any desired order. Shih et al. (2022) made the same observation that any-order models by default have to solve exponentially more masking problems than autoregressive models. However, whereas our work shows that learning in the face of this challenging task diversity can benefit the model at inference time, their work sought to alleviate complexity at training time by reducing the number of masking problems that need to be solved.

# B. Technical details from Section 3

Notations. Throughout this section, we use $x ^ { i }$ to denote the $i$ -th coordinate of the vector $x$ and $z ( j )$ to denote the $j$ -th example. The $i$ -th coordinate of the vector $z ( j )$ is denoted by $z ( j ) ^ { i }$ .

# B.1. Additional example: sparse parity observations

Example B.1 (Noisy sparse parity observations). Let $m = 2$ , $k \in \mathbb { N } ,$ , and $N ^ { 2 } \log N \ll P \leq N ^ { 0 . 4 9 k }$ . $F i x$ noise rate $\eta > 0$ as well as strings $z ( 1 ) , \ldots , z ( P )$ sampled independently and uniformly at random from the set of $k$ -sparse strings in $\{ 0 , 1 \} ^ { N }$ . For each $j \in [ P ]$ , define ${ \mathcal { O } } _ { j } ( x )$ to be the distribution which places mass $1 - \eta$ on 1 (resp. 2) and mass $\eta$ on 2 (resp. 1) $i f$ $\Sigma _ { i } x ^ { i } z ( j ) ^ { i }$ is odd (resp. even). Note that for $k = O ( 1 )$ , each of these observations is efficiently learnable by brute-force.

Below we show that for a certain range of masking fractions, a constant fraction of the masking problems for the corresponding L&O distributions are computationally hard under the Sparse Learning Parity with Noise assumption (Alekhnovich, 2003). Formally we have:

Proposition B.2. Let $0 < \alpha < 1$ be an arbitrary absolute constant, and let $\eta = 1 / \mathrm { p o l y } ( N )$ be sufficiently large. Let x be a sample from a L&O distribution $p _ { \mathrm { d a t a } }$ with noisy parity observations as defined in Example B.1. Suppose each token is independently masked with probability $\alpha _ { i }$ , and $M$ is the set of indices for the masked tokens. If $1 - 1 / N \le \alpha \le 1 - 1 / 2 N _ { }$ , then under the Sparse Learning Parity with Noise (SLPN) assumption (see Definition B.3), with constant probability over $M$ , no polynomial-time algorithm can solve the resulting masking problem of predicting any of the masked tokens among $x ^ { \pi ( 1 ) } , \bar { . } . . , \bar { x } ^ { \pi ( N ) }$ given $x [ M ]$ .

We note that it is important for us to take the observations to be sparse parities and to leverage the Sparse Learning Parity with Noise assumption. If instead we used dense parities and invoked the standard Learning Parity with Noise (LPN) assumption, we would still get the hardness of masking problems, but the observations themselves would be hard to learn, assuming LPN. This result is based on the following standard hardness assumption:

Definition B.3 (Sparse Learning Parity with Noise). Given input dimension $N$ , noise parameter $0 < \eta < 1 / 2$ , and sample size $P$ , an instance of the Sparse Learning Parity with Noise (SLPN) problem is generated as follows:

• Nature samples a random bitstring $x$ from $\{ 0 , 1 \} ^ { N }$   
• We observe $P$ examples of the form $( x ( i ) , y ( i ) )$ where $x ( i )$ is sampled independently and uniformly at random from $k$ -sparse bitstrings in $\{ 0 , 1 \} ^ { N }$ , and $y$ is given by $\epsilon _ { i } + \langle x ( i ) , x \rangle$ (mod 2), where $\epsilon _ { i }$ is 1 with probability $\eta$ and 0 otherwise.

Given the examples $\{ ( x ( i ) , y ( i ) ) \} _ { i = 1 } ^ { P }$ , the goal is to recover $x$

The SLPN assumption is that for any $P = N ^ { ( 1 - \rho ) k / 2 }$ for constant $0 < \rho < 1$ , and any sufficiently large inverse polynomial noise rate $\eta$ , no $\mathrm { p o l y } ( N )$ -time algorithm can recover $x$ with high probability.

Proof of Proposition B.2. With probability at least $1 - ( 1 - 1 / N ) ^ { N } \ge \Omega ( 1 )$ , all of the variable tokens $x ^ { \pi ( i ) }$ for $i \leq N$ are masked. Independently, the number of unmasked tokens among the observation tokens ${ \mathcal { O } } _ { j }$ is distributed as $\mathrm { B i n } ( P , 1 - \alpha )$ , so by a Chernoff bound, with probability at least $1 - e ^ { - \Omega ( P / N ^ { 2 } ) } = 1 - 1 / \mathrm { p o l y } ( N )$ we have that at least $P / 4 N = \Omega ( N \log N )$ observation tokens are unmasked. The masking problem in this case amounts to an instance of SLPN with input dimension $N$ and sample size in $[ \Omega ( N \log N ) , O ( N ^ { 0 . 4 9 k } ) ]$ . Because of the lower bound on the sample size, prediction of $\mathbf { x } ^ { M }$ is information-theoretically possible. Because of the upper bound on the sample size, the SLPN assumption makes it computationally hard. As a result, estimating the posterior mean on any entry of $\mathbf { x } ^ { M }$ given the unmasked tokens is computationally hard as claimed. □

# B.2. Additional example: random slab observations

Example B.4 (Random slab observations). Let $m = 2$ and $P = \gamma N ^ { 2 }$ for constant $\gamma > 0$ . Fix slab width $\beta$ and vectors $z ( 1 ) , \ldots , z ( P )$ sampled independently from √ $\mathcal { N } ( 0 , I )$ . For each $j \in [ P ]$ , define the corresponding observation ${ \mathcal { O } } _ { j } ( x )$ to be deterministically 1 $\begin{array} { r } { \cdot i f { \left| \left. z ( j ) , 2 x - \mathbf { 1 } \right. \right| } \leq \beta \sqrt { N } } \end{array}$ , and deterministically 0 otherwise.

In (Alaoui $\&$ Gamarnik, 2024), it was shown that stable algorithms (Definition B.7), which encompass many powerful methods for statistical inference like low-degree polynomial estimators, MCMC, and algorithmic stochastic localization (Gamarnik, 2021), are unable to sample from the posterior distribution over a random bitstring conditioned on it√ satisfying $| \langle z ( j ) , x \rangle | \le \beta \sqrt { N }$ for any $\Theta ( N )$ number of constraints $z ( 1 ) , \ldots , z ( P ^ { \prime } )$ , provided $P ^ { \prime }$ is not too large that the support of the posterior is empty. This ensemble is the well-studied symmetric perceptron (Aubin et al., 2019). The following is a direct reinterpretation of the result of (Alaoui & Gamarnik, 2024):

Proposition B.5. Let $p _ { \mathrm { d a t a } }$ be a L&O distribution with random slab observations as defined in Example B.4, with parameter $\gamma > 0$ and slab width $\beta > 0$ . There exists a constant $c _ { \beta } > 0$ such that for any absolute constant $0 ~ < ~ c ~ < ~ c _ { \beta }$ , $i f$ $1 - c _ { \beta } N / 2 P \le \alpha \le 1 - c N / P$ and $\gamma > c _ { \beta }$ , the following holds. Let $p _ { \mathrm { d a t a } } ^ { \prime }$ denote the distribution given by independently√ masking every coordinate in $p _ { \mathrm { d a t a } }$ with probability $\alpha$ . Then any $( 1 - \tilde { \Omega } ( 1 / \sqrt { N } ) )$ -stable algorithm, even one not based on masked diffusion, which takes as input a sample $x ^ { \prime }$ from $p _ { \mathrm { d a t a } } ^ { \prime }$ and, with probability $1 - o ( 1 )$ outputs a Wassersteinapproximate3 sample from $p _ { \mathrm { d a t a } }$ conditioned on the unmasked tokens in $x ^ { \prime }$ , must run in super-polynomial time.

The upshot of this is that any stable, polynomial-time masked diffusion sampler will, with non-negligible probability, encounter a computationally hard masking problem at some point during the reverse process.

For the proof, we first formally define the (planted) symmetric Ising perceptron model:

Definition B.6. Let $\alpha , \beta > 0$ . The planted symmetric Ising perceptron model is defined as follows:

• Nature samples $\sigma$ uniformly at random from $\{ \pm 1 \} ^ { N }$   
• For each √ $j = 1 , \ldots , P = \lfloor \alpha N \rfloor$ , we sample $z ( j )$ independently from $\mathcal { N } ( 0 , I _ { N } )$ conditioned on satisfying $| \langle z ( j ) , \sigma \rangle | \leq$ $\beta \sqrt { N }$ .

The goal is to sample from the posterior on $\sigma$ conditioned on these observations $\{ z ( i ) \} _ { i = 1 } ^ { P }$

Next, we formalize the notion of stable algorithms.

Definition B.7. Given a matrix $Z \sim \mathcal { N } ( 0 , 1 ) ^ { \otimes P \times N }$ , define $Z _ { t } = t Z + \sqrt { 1 - t ^ { 2 } } Z ^ { \prime }$ for independent $Z ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) ^ { \otimes P \times N }$ . A randomized algorithm $\mathcal { A }$ which takes as input $Z \in \mathbb { R } ^ { P \times N }$ and outputs an element of $\{ \pm 1 \} ^ { N }$ is said to be $t _ { N }$ -stable if $\begin{array} { r } { \operatorname* { l i m } _ { N  \infty } W _ { 2 } ( \mathrm { l a w } ( \boldsymbol { A } ( Z ) ) , \mathrm { l a w } ( \boldsymbol { A } ( Z _ { t } ) ) ) = 0 } \end{array}$ .

As discussed at depth in (Gamarnik, 2021), many algorithms like low-degree polynomial estimators and Langevin dynamics are stable.

Theorem B.8 (Theorem 2.1 in (Alaoui & Gamarnik, $2 0 2 4 ) ^ { 4 }$ ). For any constant $\beta > 0$ , there exists $c _ { \beta } > 0$ such that the following holds for all constants $0 < \alpha < c _ { \beta }$ . For $t _ { N } \le 1 - \Omega ( \log ^ { 2 } ( n ) / n ^ { 2 } )$ , any $t _ { N }$ -stable randomized algorithm $\mathcal { A }$ which takes as input $Z = ( z ( 1 ) , \ldots , z ( P ) )$ and outputs an element of $\{ \pm 1 \} ^ { N }$ will fail to sample from the posterior on √ $\sigma$ conditioned on $Z$ in the symmetric Ising perceptron model to Wasserstein error $o ( \sqrt { N } )$ .

Proof of Proposition B.5. By a union bound, with probability at least $1 - ( 1 - \alpha ) N \ge 1 - c _ { \beta } N ^ { 2 } / P \ge 1 - c _ { \beta } / \gamma$ over a draw $x ^ { \prime } \sim p _ { \mathrm { d a t a } } ^ { \prime }$ , all of the $\bar { x ^ { \pi ( i ) } }$ tokens are masked. The number of unmasked tokens in $x ^ { \prime }$ among the observations ${ \mathcal { O } } _ { j }$ is distributed as $\mathrm { B i n } ( P , 1 - \alpha )$ . By a Chernoff bound, this is in $\left[ 3 c N / 4 , 3 c _ { \beta } N / 4 \right]$ with at least constant probability. The claim then follows immediately from Theorem B.8 above.

# B.3. Proof outline of Proposition 3.3

To understand the proof idea, we consider the case where all the latent tokens are masked and some of the observation tokens are unmasked. In this case, the prediction task reduces to learning to recover the latent tokens that are consistent with the observations. Intuitively, each observation provides some constraints and the task is to recover an assignment that satisfies the constraints. This is reminiscent of Constraint Satisfaction Problems (CSPs). Indeed, to show the hardness result, we use the rich theory developed for planted CSPs at the intersection of statistical physics and average-case complexity.

![](images/figures/masked-diffusion-token-ordering-fig-0005.jpg)  
Figure 4. Overlap achieved by belief propagation initialized at ground truth versus random for planted CSP with $k = 3$ , $m = 3$ , and $g = \mathrm { N A E }$ , for $N = 1 0 0 0 0$ and varying choices of average degree $D$ . $D _ { \mathrm { K S } } / K$ can be shown analytically to be 64, consistent with the phase transition depicted. Plot suggests $D _ { \mathrm { c o n d } } / K \approx 5 0$ . By Prop. 3.3 this implies a range of masking fractions at which $\Omega ( 1 )$ fraction of masking problems are computationally hard.

In a planted CSP, there is an unknown randomly sampled vector $y$ of length $N$ and, one is given randomly chosen Boolean constraints which $y$ is promised to satisfy, and the goal is to recover $y$ as best as possible (see Definition B.9). Prior works have shown the hardness of efficiently learning to solve the planted CSP problem (Krzakala & Zdeborova´, 2009; Alaoui & Gamarnik, 2024). We show the hardness of masking problems in L&O distributions based on these results. Consider the ground truth latent tokens as the random vector $y$ and each observation as a constraint. In this case, the problem of learning to recover the latent tokens from the observation tokens reduces to recovery for the planted CSP.

There are precise predictions for the values of vocabulary size $m$ and the number of observations for which the informationtheoretically best possible overlap and the best overlap achievable by any computationally efficient algorithm are different. We show that these predictions directly translate to predictions about when masking problems become computationally intractable:

As a simple example, let us consider sparse predicate observations with $k = 2$ and $g ( x ^ { \prime } , x ^ { \prime \prime } ) = \mathbf { 1 } [ x ^ { \prime } \neq x ^ { \prime \prime } ]$ . These can be formally related to the well-studied problem of planted $m$ -coloring. In the planted $m$ -coloring, a random graph of average degree $D$ is sampled consistent with an unknown vertex coloring and the goal is to estimate the coloring as well as possible (Krzakala & Zdeborova´, 2009), as measured by the overlap of the output of the algorithm to the ground-truth coloring (see Definition B.9). As a corollary of our main result, we show that when all the latent tokens $x ^ { \pi ( 1 ) } , \ldots , x ^ { \pi ( N ) }$ are masked and a few unmasked observation tokens provide the information of the form $g ( x ^ { \pi ( i ) } , x ^ { \pi ( j ) } ) = \mathbf { 1 } [ x ^ { \pi ( i ) } \neq x ^ { \pi ( j ) } ]$ for $i , j \le N$ , then solving the masking problem can be reduced to solving planted coloring.

For planted $m$ -coloring, when $m = 5$ the thresholds in Proposition 3.3 are given by $D _ { \mathrm { K S } } / 2 = 1 6$ and $D _ { \mathrm { c o n d } } / 2 \approx$ 13.23 (Krzakala & Zdeborova´, 2009) (the factor of 2 here is simply because the observations correspond to ordered subsets of size 2). For general predicates and arities, there is an established recipe for numerically computing $D _ { \mathrm { K S } }$ and $D _ { \mathrm { c o n d } }$ based on the behavior of the belief propagation algorithm (see the discussion in Appendix B.4). As an example, in Fig. 4, we execute this recipe for $m = 3$ , $k = 3$ , and $g$ given by the Not-All-Equal predicate $\mathrm { N A E } ( x ^ { \prime } , x ^ { \prime \prime } , x ^ { \prime \prime } ) = 1 - 1 [ x ^ { \prime } = x ^ { \prime \prime } = x ^ { \prime \prime \prime } ]$ to obtain thresholds that can be plugged into Proposition 3.3.

Additional examples of the hardness. The above setup can also be generalized to capture Bayesian constraint satisfaction problems (Montanari, 2008; Liu et al., 2022), one notable example of which is the stochastic block model (Decelle et al., 2011). There are analogous predictions for the onset of hardness of inference, which can likewise be translated to hardness of masking problems for seemingly benign L&O distributions. In Appendix B.1 and B.2, we give two more examples of L&O distributions for which order-aware training is tractable yet order-agnostic training of the MDM is computationally

hard.

First, we consider L&O distributions whose observations are sparse, noisy parities in the latents and deduce hardness for order-agnostic training from the Sparse Learning Parity with Noise assumption (Alekhnovich, 2003). We then consider L&O distributions whose observations are generalized linear models in the latents, and deduce hardness for a large class of efficient algorithms from existing results on Lipschitz hardness (Alaoui & Gamarnik, 2024) for the symmetric binary perceptron (Aubin et al., 2019).

# B.4. Proof of Proposition 3.3: sparse predicate observations

Here we formally define the relevant notions needed to formalize our claim about hardness in Proposition 3.3.

Definition B.9 (Planted CSPs). Given arity $k \in \mathbb N$ , vocabulary/alphabet size $m \in \mathbb { N }$ , predicate $g : \{ 1 , . . . , m \} ^ { k }  \{ 0 , 1 \}$ latent dimension $N$ , and clause density $P / N$ , the corresponding planted constraint satisfaction problem is defined as follows: Nature samples an unknown assignment $\sigma$ uniformly at random from $\{ 1 , \ldots , m \} ^ { N }$ , and then for each ordered $k$ -tuple $S$ of distinct elements from $[ N ]$ , we observe the clause $S$ independently with probability $\phi / N ^ { k - 1 }$ if $g ( \sigma \vert _ { S } ) = 1$ .

To measure the quality of an algorithm for recovering $\sigma$ given the observations, define the overlap between an estimate $\hat { \sigma }$ and the ground truth $\sigma$ by $d ( \sigma , \hat { \sigma } ) \triangleq \operatorname* { m i n } _ { \pi \in \mathbb { S } _ { N } } \sum _ { i } { \bf 1 } [ \sigma _ { i } = \pi ( \hat { \sigma } _ { i } ) ]$ where $\mathbb { S } _ { N }$ denotes the set of all permutations of $\{ 0 , 1 , \ldots , N - 1 \}$ . Define the average degree to be $k P / N$ , i.e. the expected number of variables that share at least one clause with a given variable.

We begin by defining the central algorithm driving statistical physics predictions about hardness for random constraint satisfaction problems: belief propagation (BP).

Definition B.10 (BP update rules). Belief propagation is an algorithm that iteratively updates a set of messages $\{ \mathbf { M S } _ { c } ^ { i \to S } [ t ] , \mathbf { M S } _ { c } ^ { S \to i } [ t ] \}$ , where $i , S$ range over all pairs of variable indices $i \in [ N ]$ and observations $S \ni i$ . At time $t + 1$ , the messages are computed via

$$
\begin{array} { r l } & { \mathbf { M S } _ { c } ^ { i  S } [ t + 1 ] \propto \displaystyle \prod _ { T : i \in T \neq S } \mathbf { M S } _ { c } ^ { T  i } [ t ] } \\ & { \mathbf { M S } _ { c } ^ { S  i } [ t + 1 ] \propto \displaystyle \sum _ { \overline { { \sigma } } \in \{ 1 , \ldots , m \} ^ { S \setminus i } } g ( \overline { { \sigma } } \cup _ { i } c ) \prod _ { j : i \neq j \in S } \mathbf { M S } _ { \overline { { \sigma } } _ { j } } ^ { j  S } [ t ] , } \end{array}
$$

where $\overline { { \sigma } } \cup _ { i } c \in \{ 1 , \ldots , m \} ^ { S }$ assigns $c$ to entry $i$ and $\overline { { \sigma } }$ to the remaining entries.

A set of messages can be used to estimate the marginals of the posterior on $\sigma$ conditioned on the observations as follows. The marginal on the $i$ -th variable has probability mass function over $\{ 1 , \ldots , m \}$ proportional to $\{ \textstyle \prod _ { T : i \in T } \mathbf { M S } _ { c } ^ { T \to i } \}$ . Given a set of marginals, a natural way to extract an estimate for $\sigma$ is to round to the color in $\{ 1 , \ldots , m \}$ at which the probability mass function is largest.

Throughout we will make the following assumption that ensures that the trivial messages $ { \mathbf { M } }  { \mathbf { S } } _ { c } ^ { i \to S } = 1 / m$ and $\mathbf { M S } _ { c } ^ { S  i } = \mathbf { \Phi }$ $1 / m$ are a fixed point, sometimes called the paramagnetic fixed point, for the iteration above:

Assumption B.11. The quantity ${ \textstyle \sum _ { { \overline { { \sigma } } } \in \{ 1 , \dots , m \} ^ { [ k ] } \setminus i } g ( { \overline { { \sigma } } } \cup _ { i } c ) }$ is constant across all $c \in \{ 1 , \ldots , m \}$ and $i \in [ k ]$

Definition B.12. Given $k , m , g$ , the Kesten-Stigum threshold $D _ { \mathrm { K S } }$ is defined to be the largest average degree for which BP is locally stable around the paramagnetic fixed point, that is, starting from a small perturbation of the paramagnetic fixed point, it converges to the paramagnetic fixed point. More formally, $D _ { \mathrm { K S } }$ is the largest average degree at which the Jacobian of the BP operator $\{ { \bf M S } ^ { i \stackrel { . . . } {  } S } [ t ] \} \stackrel { \bf { \sigma } } { \mapsto } \{ { \bf M S } ^ { i  S } [ t + 1 ] \}$ has spectral radius less than 1.

The condensation threshold $D _ { \mathrm { c o n d } }$ is defined to be the largest average degree at which the planted CSP ensemble and the following simple null model become mutually contiguous and thus statistically indistinguishable as $N  \infty$ . The null model is defined as follows: there is no single unknown assignment, but instead for every ordered subset $S$ of $k$ variables, Nature independently samples an unknown local assignment $\sigma _ { S } \in \{ 1 , \dots , m \} ^ { S }$ , and the observation is included with probability $\phi / N ^ { k - 1 }$ if $g ( \sigma _ { S } ) = 1$ .

For $D _ { \mathrm { c o n d } } < k P / N < D _ { \mathrm { K S } }$ , there exists some other fixed point of the BP operator whose marginals, once rounded to an assignment, achieves strictly higher overlap than does BP with messages initialized randomly. The prediction is that in this regime, no efficient algorithm can achieve optimal recovery (Krzakala & Zdeborova´, 2009).

Conjecture B.13 (1RSB cavity prediction). Suppose $k , m , g$ satisfy Assumption B.11, and let $D _ { \mathrm { K S } }$ and $D _ { \mathrm { c o n d } }$ denote the associated Kesten-Stigum and condensation thresholds for the average degree. Then for all $P$ for which $D _ { \mathrm { c o n d } } < k P / N <$ $D _ { \mathrm { K S } }$ , the best overlap achieved by a computationally efficient algorithm for recovering $\sigma$ is strictly less than the best overlap achievable.

Proof of Proposition 3.3. At masking fraction $\alpha$ satisfying the bounds in the Proposition, with probability at least $\alpha ^ { N } \geq$ $( 1 - \gamma ^ { - 1 } D _ { \mathrm { K S } } / N ^ { k - 1 } ) ^ { N } \geq \Omega ( 1 )$ we have that all tokens corresponding to latents $x _ { \pi ( i ) }$ get masked. Independently of this, the number of unmasked tokens among the observation tokens ${ \mathcal { O } } _ { S }$ is distributed as $\mathrm { B i n } ( \dot { N } ( N - 1 ) \cdots ( N - k + 1 ) , 1 - \alpha )$ , so by standard binomial tail bounds, with constant probability (depending on the gap between $D _ { \mathrm { c o n d } }$ and $D _ { \mathrm { K S } } ,$ ) this lies between $\gamma ^ { - 1 } D _ { \mathrm { c o n d } } N / k$ and $\gamma ^ { - 1 } D _ { \mathrm { K S } } N / k$ . Furthermore, of these unmasked tokens in expectation $\gamma$ fraction of them correspond to observations for which the associated predicate evaluates to 1. Conditioned on the above events, the masking problem thus reduces exactly to inference for a planted constraint satisfaction problem at average degree $D _ { \mathrm { c o n d } } < D < D _ { \mathrm { K S } }$ , from which the Proposition follows.

# C. Experimental details in Section 3

# C.1. Experimental details in Section 3.2

$\pi$ -learner configurations. We consider two distributions of $\pi$ that interpolate between Unif $\left( \mathbb { S } _ { L } \right)$ where $\mathbb { S } _ { L }$ denote the uniform distribution over all permutations of indices $\{ 0 , 1 , \ldots , L - 1 \}$ and the point mass at the identical distribution: (Closer) and (Much-closer). To construct those distributions, we start from the identity permutation and perform a certain number of random swapping operations. Since $L \log ( L )$ number of swaps results in a distribution that is very close to Unif $\left( \mathbb { S } _ { L } \right)$ (Bormashenko, 2011), we use $L / 1 0$ and $\sqrt { L }$ swaps to construct the (Closer) and (Much-closer) distributions, respectively. For consistency, we repeat this sampling process three times.

Model and training configurations. As explained in Section 3.2, to evaluate the scaling law of the $\pi$ -learner, we can simply adapt the autoregressive training setup (a transformer with causal attention) by modifying the input to $\pi ( \boldsymbol { x } _ { 0 } )$ and using a learnable positional embedding layer instead of RoPE. We borrow the training configurations from (Nie et al., 2024), which are also consistent with the TinyLlama (Zhang et al., 2024) configurations. In particular, we use AdamW optimizer (Loshchilov & Hutter, 2017), setting $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 5$ , and a weight decay of 0.1 and $L = 2 0 4 8$ . A cosine learning rate schedule is applied, with a maximum learning rate of $4 \times 1 0 ^ { - 4 }$ and a minimum learning rate of $4 \times 1 0 ^ { - 5 }$ . We also note that unless otherwise specified, we maintain the same training configuration throughout the paper.

Examining scaling laws. We conduct IsoFLOP analysis (Hoffmann et al., 2022). For a given number of FLOPs $C$ , by varying the number of non-embedding parameters of transformers, we set the iteration numbers so that the total number of tokens observed by the model during training equals $C / 6 N$ , following prior studies (Hoffmann et al., 2022; Kaplan et al., 2020). We then select the smallest validation loss and set it as a data point.

# C.2. Experimental details in Section 3.3

# C.2.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION

We consider the L&O-NAE-SAT distribution with $( N , P ) = ( 2 0 , 2 8 0 )$ . For each example sequence from L&O-NAE-SAT, we pad the last 212 tokens with an additional token value of 2. We employ a 19M MDM with RoPE and a maximum sequence length of 512. Then, this MDM is trained for $2 \times 1 0 ^ { 3 }$ iterations. To attain a proxy MDM for the Bayes optimal predictor, we further train it for $5 \times 1 0 ^ { 4 }$ iterations.

To measure the error across different tasks, we consider the following setup. For each $\ell \in [ 1 , N - 1 ]$ , we randomly mask $\ell$ tokens in the latent positions and $\ell \times ( P / N )$ tokens in the observed positions. Across all masked prediction positions, $\ell ( 1 + P / N )$ , we measure the error for each position. For certainty, we repeat this process 1000 times. The result in Figure 2 corresponds to the case when $\ell = 1 1$ , and we observe the same tendency for other values of $\ell$ .

# C.2.2. EXPERIMENT ON TEXT DATA

We take a 170M MDM pretrained with text data for a baseline model. To measure the performance imbalance between likelihood modeling tasks

$$
\mathbb { E } _ { \boldsymbol { x } _ { 0 } \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { i = 0 } ^ { L - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( i ) } \Big | \boldsymbol { x } _ { 0 } [ \pi \{ i , \dots , L - 1 \} ] \right) \right] .
$$

As done in the experiments in Section 3.2, we sample $\pi \mathbf { S }$ from three different distributions: $\mathrm { U n i f } ( \mathbb { S } _ { L } )$ , (Closer), the point mass of identical distribution. For each case, we calculate the expectation over 1024 samples of $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ .

# D. Experimental details in Section 4

# D.1. Experimental details in Section 4.2

D.1.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION

We consider five instances of L&O-NAE-SAT: $( N , P ) = ( 2 5 , 2 7 5 ) , ( 3 0 , 2 7 0 ) , ( 4 0 , 2 6 0 ) , ( 5 0 , 2 5 0 )$ , (100, 200). For each distribution, we train a 19M MDM and measure the accuracy difference between vanilla inference and adaptive inference using top probability margin.

# D.1.2. EXPERIMENT ON TEXT DATA

Top probability margin sampler with temperature. To modify our inference for text data modeling, which does not have a determined answer, we found that adding a certain level of temperature to the oracle is useful. This is because the top probability margin or the top probability often leads to greedy sampling, which harms the diversity (entropy) of the generated samples. Therefore, we consider a variant of the oracle as follows, incorporating a Gaussian noise term $\epsilon$ .

$$
\mathcal { F } ( \theta , x _ { t } ) = \mathrm { T o p } K \left( \vert p _ { \theta } ( x ^ { i } = j _ { 1 } \vert x _ { t } ) - p _ { \theta } ( x ^ { i } = j _ { 2 } \vert x _ { t } ) \vert + \epsilon \right) .
$$

Note that this approach has also been employed for unconditional sampling (Wang et al., 2024; Zheng et al., 2023).

Generative perplexity and entropy. We employ a 1.1B MDM pretrained on text data as a baseline. For each sampling step, we unconditionally generate samples using both vanilla and adaptive inference. Next, we calculate the likelihood using LLama2-7B as a baseline large language model. Moreover, we denote the entropy of a generated sample $x$ as $\sum p _ { i } \log p _ { i }$ , where $p _ { i } = \# \{ x ^ { i } = i \} / L$ .

Choice of number of tokens to unmask. We set the number of tokens to unmask $K$ so that the number of unmasked tokens matches that of vanilla MDM inference in expectation. For an inference transition from step $t$ to $s$ , vanilla MDM expects (# mask tokens in the current $\begin{array} { r } { x _ { t } ) \times \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } } \end{array}$ unmasked. Accordingly, we choose $K = ( \#$ mask tokens in the current $x _ { t } ) \times$ $\frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }$ . This choice keeps the number of revealed tokens balanced throughout inference. Alternatively, one can sample $K$ stochastically from Binom( $\#$ mask tokens in the current $x _ { t }$ , $\frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } }$ ). We found that both the deterministic and stochastic choices of $K$ result in comparable generative perplexity.

This choice of $K$ can be potentially helpful when the network is time-conditioned, since this keeps $\#$ mask tokens in the current $x _ { t } ) \approx ( 1 - \alpha _ { t } ) \times L$ where $L$ is the max sequence length–matching the marginal that the model saw during training.

# D.2. Experimental details on Sudoku and Zebra puzzles

Dataset. For both Sudoku and Zebra puzzles, we use the dataset provided in Shah et al. (2024) to train our model. To evaluate our model on the same difficulty tasks, we use the test dataset proposed in Shah et al. (2024). This dataset is created by filtering the puzzles from (Radcliffe, 2020) that can be solved using a fixed list of 7 strategies. To create a hard dataset to evaluate easy-to-hard generalization, we use the remaining puzzles from (Radcliffe, 2020) as they either require a new strategy unseen during the training and/or require backtracking. The hard dataset contains around 1M Sudoku puzzles.

Model, training, and inference. For the training and inference, we use the codebase of (Ye et al., 2024) with keeping most of the hyperparameters default given in the codebase. For the Sudoku dataset, we use 6M GPT-2 model, and for the Zebra dataset, we use 19M model. We set the learning rate to 0.001 with a batch size of 128 to train the model for 300 epochs. For the inference, we use 50 reverse sampling steps using the appropriate strategy. Additionally, we add Gumbel noise with a coefficient of 0.5 to the MDM inference oracle $\mathcal { F }$ .

# D.3. Experimental details on LLaDA-8B

Our evaluation covers two task categories: (i) infilling(HumanEval-Infill and ROCStories) and (ii) instruction–answering (Math). For instruction–answering tasks, we employ a semi-autoregressive sampling strategy, whereas for infilling tasks we retain the non-autoregressive approach. For infilling tasks, the output length is predetermined—matching the size of the masked span—whereas instruction–answering tasks require an explicit length specification. For the latter, we follow the sampling configuration of (Nie et al., 2025).

For HumanEval-Infill, we adopt the problem set introduced by (Bavarian et al., 2022). Each instance is grouped by the span of the masked code—the region the model must infill—into three categories: single-line, multi-line, and split. The task difficulty rises as the length of the masked span increases.

# E. Omitted proofs

Proof of Proposition 2.1. We build on Proposition 3.1 from (Zheng et al., 2024) to obtain the result of Proposition 2.1. We first re-state the result from (Zheng et al., 2024) for the case when the denoising network $p _ { \theta }$ does not depend on the noise-scale $t$ explicitly. Let $x ( n )$ be a sequence with $n$ tokens being masked from $x _ { 0 }$ , and $x ^ { i } ( n )$ denotes the $i ^ { \mathrm { { t h } } }$ token value of the sequence $x ( n )$ . Let $\tilde { q } ( x ( n ) | x _ { 0 } )$ be the probability distribution corresponding to randomly and uniformly masking $n$ tokens of $x _ { 0 }$ .

Proposition E.1 (Proposition 3.1 of (Zheng et al., 2024)). For clean data $x _ { 0 }$ , let ${ \tilde { q } } ( x ( n ) \mid x _ { 0 } )$ be the discrete forward process that randomly and uniformly masks n tokens of $x _ { 0 }$ . Suppose the noise schedules $\alpha _ { t }$ satisfies $\alpha _ { 0 } = 0$ and $\alpha _ { 1 } = 1$ . Then, the MDM training loss (1) can be reformulated as

$$
\mathcal { L } _ { \theta } = - \sum _ { n = 1 } ^ { L } \underset { x ( n ) \sim \tilde { q } ( \cdot | x _ { 0 } ) } { \mathbb { E } } \left[ \frac { 1 } { n } \sum _ { \ell : x ^ { \ell } ( n ) = 0 } \log p _ { \theta } ( x _ { 0 } ^ { \ell } \mid x ( n ) ) \right] .
$$

To obtain an alternative formulation of (6), we expand the expectation $x ( n ) \sim \tilde { q } ( { \cdot } \mid x _ { 0 } )$ . Since there are total $L$ positions of $x _ { 0 }$ , we have the probability assigned for each $x ( n )$ equals $1 / { \binom { L } { n } }$ . Therefore, expanding the above equation with the expectation $x ( n )$ and treating $x ( n )$ as $x [ M ]$ for some set $M$ of size $n$ , we obtain the result.

$$
\mathcal { L } _ { \theta } = - \sum _ { M \in [ L ] , i \in M } \frac { 1 } { \binom { L } { | M | } } \cdot \frac { 1 } { | M | } \log p _ { \theta } ( x _ { 0 } ^ { \ell } \mid x [ M ] ) .
$$

# E.1. Equivalence between the MDM loss and any-order autoregressive loss

In this section, we will demonstrate the equivalence for MDM loss and any-order autoregressive loss. In particular, for all $x _ { 0 }$ , we show

$$
\underset { \mathrm { \times U n i f } ( \mathbb { S } _ { L } ) } { \mathbb { E } } \left[ \sum _ { j = 0 } ^ { L - 1 } \log p \_ { \theta } \left( x _ { 0 } ^ { \pi ( j ) } \Big \vert x _ { 0 } [ \pi \{ j \} , \dots , \pi \{ L - 1 \} ] \right) \right] = - \sum _ { M \subseteq [ L ] , i \in M } \frac { 1 } { \binom { L } { | M | } } \frac { 1 } { | M | } \log p \_ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] ) .
$$

We now consider $\{ \pi ( j ) , \dots , \pi ( L - 1 ) \} = M \subseteq [ L ]$ and $\pi ( j ) = i$ and count the number of $\pi \in \mathbb { S } _ { L }$ that induces a specific term $\log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] )$ . To induce the term, for a given $M \in [ L ]$ and $i \in M$ , $\pi$ must satisfy

$$
\pi ( j ) = i , \quad \{ \pi ( j ) , \ldots , \pi ( L - 1 ) \} = { \cal M } .
$$

The number of $\pi$ that satisfies above is $( L - | M | ) ! \times ( | M | - 1 ) !$ . Using this and the number of total permutations is $L !$ , we obtain the result.

$$
\begin{array} { l } { { \displaystyle \operatorname* { \Pi } _ { \pi \sim \operatorname { U n i f } ( \mathcal { S } , \pi ) } \left[ \sum _ { j = 0 } ^ { L - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( j ) } \Big | x _ { 0 } [ \pi \{ j \} , \dots , \pi \{ L - 1 \} ] \right) \right] } } \\ { { \displaystyle = \frac { 1 } { L ! } \sum _ { \pi \in \operatorname { U n i f } ( \mathcal { S } , L ) } \sum _ { j = 0 } ^ { L - 1 } \log p _ { \theta } \left( x _ { 0 } ^ { \pi ( j ) } \Big | x _ { 0 } [ \pi \{ j \} , \dots , \pi \{ L - 1 \} ] \right) } } \\ { { \displaystyle = \frac { 1 } { L ! } \sum _ { M \in [ L ] , i \in M } \left[ \log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] ) \times ( L - 1 - | M | ) ! \times ( | M | - 1 ) ! \right] } } \\ { { = \displaystyle \sum _ { M \in [ L ] , i \in M } \frac { 1 } { \left( \lfloor M \rfloor \right) } \frac { 1 } { | M | } \log p _ { \theta } ( x _ { 0 } ^ { i } | x _ { 0 } [ M ] ) . } } \end{array}
$$