# Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control

Carles Domingo-Enrich1, Michal Drozdzal1, Brian Karrer1, Ricky T. Q. Chen1

1FAIR, Meta

Dynamical generative models that produce samples through an iterative process, such as Flow Matching and denoising diffusion models, have seen widespread use, but there have not been many theoreticallysound methods for improving these models with reward fine-tuning. In this work, we cast reward fine-tuning as stochastic optimal control (SOC). Critically, we prove that a very specific memoryless noise schedule must be enforced during fine-tuning, in order to account for the dependency between the noise variable and the generated samples. We also propose a new algorithm named Adjoint Matching which outperforms existing SOC algorithms, by casting SOC problems as a regression problem. We find that our approach significantly improves over existing methods for reward fine-tuning, achieving better consistency, realism, and generalization to unseen human preference reward models, while retaining sample diversity.

Correspondence: Carles Domingo-Enrich at cd2754@nyu.edu

![](images/figures/adjoint-matching-fig-0001.jpg)  
Figure 1 We introduce Adjoint Matching, a theoretically-driven yet simple algorithm for reward fine-tuning that works for a large family of dynamical generative models, including for the first time, Flow Matching models. Text prompts: “Beautiful colorful sunset midst of building in Bangkok Thailand ”, “Beautiful grandma and granddaughter are mixing salad and smiling while cooking in kitchen”, “The beautiful young woman in sunglasses is standing at the background of field and hill. She is smiling and looking over shoulder ”, “Chess, intellectual games, figure horse, chess board ”.

# 1 Introduction

Flow Matching (Lipman et al., 2023; Albergo and Vanden-Eijnden, 2023; Liu et al., 2023) and denoising diffusion (Song and Ermon, 2019; Ho et al., 2020; Song et al., 2021b; Kingma et al., 2021) models are being used for many generative modeling applications, including text-to-image (Rombach et al., 2022; Esser et al., 2024), text-to-video (Singer et al., 2022), and text-to-audio (Le et al., 2024; Vyas et al., 2023). In most cases, the base generative model does not achieve the desired sample quality. To improve the generated samples, it is common to resort to techniques such as classifier-free guidance (Ho and Salimans, 2022; Zheng et al., 2023) to get better text-to-sample alignment, or to fine-tune using human preference reward models to improve sample quality and realism (Wallace et al., 2023a; Clark et al., 2024).

In the adjacent field of large language models, the behavior of the model is aligned to human preferences through fine-tuning with reinforcement learning from human feedback (RLHF). Either explicitly or implicitly, RLHF methods (Ziegler et al., 2020; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022) assume a reward model $r ( x )$ that captures human preferences, with the goal of modifying the base generative model such that it generates the following tilted distribution:

$$
p ^ { * } ( x ) \propto p ^ { \mathrm { b a s e } } ( x ) \exp ( r ( x ) ) ,
$$

where $p _ { \mathrm { b a s e } }$ is the base generative model’s sample distribution.

Inspired by this, fine-tuning methods have been developed to improve denoising diffusion models based on human preference data; either using a reward-based approach (Fan and Lee, 2023; Black et al., 2024; Fan et al., 2023; Xu et al., 2023; Clark et al., 2024; Uehara et al., 2024a,b), or direct preference optimization (Wallace et al., 2023a). However, unlike the fine-tuning methods designed for large language models, most of the existing methods to a large degree ignore $p ^ { \mathrm { b a s e } }$ and focus solely on the reward model. Reward models can range from standard evaluation metrics such as ClipScore (Hessel et al., 2021; Kirstain et al., 2023) to specialized models that have been trained on human preferences (Schuhmann and Beaumont, 2022; Xu et al., 2023; Wu et al., 2023c). As these are parameterized by neural networks, they fall pray to adversarial examples which lead to the generation of undesirable artifacts (Goodfellow et al., 2014; Mordvintsev et al., 2015). This has led some works to consider adding regularization during fine-tuning (Fan et al., 2024; Uehara et al., 2024b) to incentivize staying close to the base model distribution; however, there does not yet exist a simple approach which actually provably generates from the tilted distribution (1).

The main contributions of our paper are as follows:

(i) We present a stochastic optimal control (SOC) formulation for reward fine-tuning of dynamical generative models. Importantly, we prove that the naïve approach considered by prior works lead to a value function bias problem that biases the fine-tuned model away from the tilted distribution (1). This problem has also been observed by Uehara et al. (2024b) but they propose a more complicated solution which involves training a separate generative model for the optimal noise distribution.   
(ii) Instead, we propose a very simple solution: the memoryless noise schedule. This is a unique noise schedule that completely removes the dependency between noise variables and the generated samples, resulting in provable convergence to the tilted distribution. This allows us to fine-tune dynamical generative models in full generality, including being the first to fine-tune noiseless Flow Matching models.   
(iii) We also propose a new method for solving SOC problems, called Adjoint Matching, which combines the scalability of gradient-based methods and the simplicity of a least-squares regression objective. This is orthogonal to the reward fine-tuning application and can be applied to general SOC problems.   
(iv) We perform extensive comparisons to baseline approaches, and analyze them from multiple perspectives such as realism, consistency, and diversity. We find that our proposed method provides generalization to unseen human preference reward models, better text-to-sample consistency, and retains good diversity.

In the following, sections are broken down as follows: Section 2 summarizes the algorithms used for sampling from pre-trained Flow Matching and diffusion models, while Section 3 provides a common notation that we will use throughout. Sections 4 and 5 form the core of our contributions. Section 4 details the value function bias problem and our proposed solution via the memoryless noise schedule. Section 5 details the new Adjoint Matching algorithm for solving SOC problems.

# 2 Preliminaries on dynamical generative models

We are interested in fine-tuning base generative models $p ^ { \mathrm { b a s e } } ( X _ { 1 } )$ where samples are generated through the simulation of a stochastic process. That is, these models transform noise variables into a sample through an iterative process. In particular, we discuss the specific constructions and sampling processes of Flow Matching (Lipman et al., 2023; Liu et al., 2023; Liu, 2022; Albergo and Vanden-Eijnden, 2023) and Denoising Diffusion Models (Ho et al., 2020; Song et al., 2021b,a). The goal of this section is to provide background information on these methods, which we will later unify into a single consistent notation in Section 3.

Given random variables from an initial distribution $X _ { 0 } \sim p _ { 0 } = \mathcal { N } ( 0 , I )$ , and $X _ { 1 }$ which are distributed according to some data distribution, we define the reference flow $\bar { \pmb X } = ( \bar { X } _ { t } ) _ { t \in [ 0 , 1 ] }$ where

$$
\bar { X } _ { t } = \beta _ { t } \bar { X } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 } ,
$$

where $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] } , ( \beta _ { t } ) _ { t \in [ 0 , 1 ] }$ are functions such that $\alpha _ { 0 } = \beta _ { 1 } = 0$ and $\alpha _ { 1 } = \beta _ { 0 } = 1$ . Diffusion models and Flow Matching construct generative Markov processes $X _ { t }$ with initial distribution $X _ { 0 } \sim \mathcal { N } ( 0 , I )$ that result in flows $\pmb { X } = ( X _ { t } ) _ { t \in [ 0 , 1 ] }$ with the same time marginals as the reference flow $\bar { X }$ , i.e., the random variables $X _ { t }$ and $X _ { t }$ have identical distribution for all times $t \in [ 0 , 1 ]$ . This implies $X _ { 1 }$ has the same distribution as the data distribution, so simulating the Markov process from random noise $X _ { 0 }$ is a way to generate artificial samples1.

# 2.1 Flow Matching

In its simplest form, the generative Markov process of a Flow Matching model is an ordinary differential equation (ODE) of the form:

$$
\begin{array} { r } { \mathrm { d } X _ { t } = v ( X _ { t } , t ) \mathrm { d } t , \qquad X _ { 0 } \sim \mathcal { N } ( 0 , I ) . } \end{array}
$$

where $v ( X _ { t } , t )$ is a parametric velocity that is optimized to match the derivative of the reference flow, i.e., $\begin{array} { r } { v ( X _ { t } , t ) = \operatorname * { a r g m i n } _ { \hat { v } } \mathbb { E } \big \| \hat { v } ( \bar { X } _ { t } , t ) - \frac { \mathrm { d } } { \mathrm { d } t } \bar { X } _ { t } \big \| ^ { 2 } } \end{array}$ (see e.g. Lipman et al. (2023) for details on pre-training Flow Matching models). It can then be proven that the solution of the generative process (3) has the same time marginals as the reference flow (Lipman et al., 2023; Liu, 2022; Albergo and Vanden-Eijnden, 2023), and a commonly used choice is $\alpha _ { t } = t$ and $\beta _ { t } = 1 - t$ . One can also consider a family of stochastic differential equations (SDEs) with an arbitrary state-independent diffusion coefficient2:

$$
\begin{array} { r } { \mathrm { d } X _ { t } = \bigg ( v ( X _ { t } , t ) + \frac { \sigma ( t ) ^ { 2 } } { 2 \beta _ { t } ( \frac { \partial t } { \partial _ { t } } \beta _ { t } - \tilde { \beta } _ { t } ) } \left( v ( X _ { t } , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } \right) \bigg ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim \mathcal { N } ( 0 , I ) , } \end{array}
$$

where $( B _ { t } ) _ { t \geq 0 }$ is a Brownian motion. The generative processes in (3) and (4) have the same time marginals. This can be seen by writing down the Fokker-Planck equations for (3) and (4), and observing that they are the same up to a cancellation of terms (Maoutsa et al., 2020). The diffusion coefficient $\sigma ( t )$ in (4) is compensated by the second term in the drift which scales proportionally as $\sigma ( t ) ^ { 2 }$ .

# 2.2 Denoising Diffusion Models

We next discuss diffusion models, in particular the sampling scheme proposed by Denoising Diffusion Implicit Model (DDIM; Song et al. (2021a)) which we will later relate to Denoising Diffusion Probabilistic Models (DDPM; Ho et al. (2020)) as a particular case of the former. For sampling from a diffusion model, the DDIM update rule $^ 3$ (Song et al. (2021a), Eq. 12), typically stated in discrete time with $k \in \{ 0 , \ldots , K \}$ , is:

$$
\begin{array} { r l } { \sqrt { \bar { \alpha } _ { k + 1 } } \big ( \frac { X _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon ( X _ { k } , k ) } { \sqrt { \bar { \alpha } _ { k } } } \big ) + \sqrt { 1 - \bar { \alpha } _ { k + 1 } - \sigma _ { k } ^ { 2 } } \epsilon ( X _ { k } , k ) + \sigma _ { k } \varepsilon _ { k } , } & { { } \quad \varepsilon _ { k } \sim \mathcal { N } ( 0 , I ) , \ X _ { 0 } \sim \mathcal { N } ( 0 , I ) . } \end{array}
$$

where $\alpha _ { k }$ is an increasing sequence such that $\bar { \alpha } _ { 0 } = 0$ , $\bar { \alpha } _ { K } = 1$ , and the sequence $\sigma _ { k }$ is arbitrary. That is, one samples an initial Gaussian random variable $x _ { 0 }$ , and applies the stochastic update (5) iteratively $K$ times in order to obtain an artificial sample $X _ { K }$ . Updates can be interpreted as progressively denoising the iterate: $x _ { 0 }$ is completely noisy and $x _ { K }$ is fully denoised. The noise predictor model $\epsilon ( x _ { k } , k )$ is trained to predict the noise of $x _ { k }$ (see e.g. Ho et al. (2020) for details on pre-training denoising diffusion models).

# 3 Flow Matching and diffusion models from a common perspective

We formulate Flow Matching and diffusion models in a unified framework, which we will later use throughout the paper. Firstly, to simplify notation, we will be using continuous-time formulations. This will also directly enable fine-tuning methods inspired by the continuous-time paradigm, which we find tends to perform better than discrete-time counterparts in our empirical validations. Secondly, by consolidating notation, we will be able to discuss fine-tuning of dynamical generative models that follow the same time marginals as the reference flow (2), pre-trained with either the Denoising Diffusion or Flow Matching framework, in full generality.

To convert DDIM to a continuous-time stochastic process, we can show that the DDIM update rule (5), up to a first-order approximation, is equivalent to the Euler-Maruyama discretization of the following SDE:

$$
\begin{array} { r } { \mathrm { d } X _ { t } = \big ( \frac { \dot { \alpha } _ { t } } { 2 \bar { \alpha } _ { t } } X _ { t } - \big ( \frac { \dot { \alpha } _ { t } } { 2 \bar { \alpha } _ { t } } + \frac { \sigma ( t ) ^ { 2 } } { 2 } \big ) \frac { \epsilon ^ { \mathrm { b a s e } } ( X _ { t } , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } \big ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim \mathcal { N } ( 0 , I ) . } \end{array}
$$

See Appendix B.1 for the full derivation. To go from (5) to (6), we assumed a uniform discretization of time, i.e. $\textstyle t = { \frac { k } { K } }$ . This results in identifying the discrete-time process $( X _ { k } ) _ { k \in \{ 0 , \ldots , K \} }$ with a continuous-time process $( X _ { t } ) _ { t \in [ 0 , 1 ] }$ , where $\alpha _ { k } : = \alpha _ { t }$ , $\begin{array} { r } { \sigma _ { k } : = \frac { 1 } { \sqrt { K } } \sigma ( t ) } \end{array}$ , and $\epsilon ( X _ { k } , k )$ with $\epsilon ^ { \mathrm { b a s e } } ( X _ { k } , t )$ . In relation to the reference flow (2),√ the generative process in (6) has the same time marginals when $\alpha _ { t } = \sqrt { \bar { \alpha } _ { t } }$ and $\beta _ { t } = \sqrt { 1 - \bar { \alpha } _ { t } }$ (Ho et al., 2020).

Furthermore, when viewed up to first order approximations, the DDPM sampling scheme (Ho et al. (2020); Algorithm 2) can be seen as special instance of the DDIM sampling scheme when $\sigma ( t ) = \sqrt { \dot { \bar { \alpha } } _ { t } / \bar { \alpha } _ { t } }$ . This results in the following generative process:

$$
\begin{array} { r } { \mathrm { d } X _ { t } = \big ( \frac { \dot { \bar { \alpha } } _ { t } } { 2 \bar { \alpha } _ { t } } X _ { t } - \frac { \dot { \bar { \alpha } } _ { t } } { \bar { \alpha } _ { t } } \frac { \epsilon ^ { \mathrm { b a s e } } ( X _ { t } , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } \big ) \mathrm { d } t + \sqrt { \frac { \dot { \bar { \alpha } } _ { t } } { \bar { \alpha } _ { t } } } \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim \mathcal { N } ( 0 , I ) , } \end{array}
$$

We can further consolidate notation by converting all quantities to the score function ${ \mathfrak { s } } ( x , t )$ —defined as the gradient of the log density of the random variable $X _ { t }$ —which is possible when $X _ { 0 }$ is Normal-distributed and under the affine reference flow (2). In particular, the velocity $v ^ { \mathrm { b a s e } }$ from Flow Matching can be expressed in terms of the score function (see Appendix B.4):

$$
\begin{array} { r } { v ^ { \mathrm { b a s e } } ( x , t ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x + \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) \mathfrak { s } ( x , t ) . } \end{array}
$$

And the noise predictor $\epsilon ^ { \mathrm { b a s e } }$ also admits an expression in terms of the score function (see Appendix B.3):

$$
\begin{array} { r } { \mathfrak { s } ( x , t ) = - \frac { \epsilon ^ { \mathrm { b a s e } } ( x , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } . } \end{array}
$$

Plugging these two equations into (4) and (6), respectively, and rewriting them in terms of only the $\alpha _ { t }$ and $\beta _ { t }$ in (2), we can unify both the Flow Matching and continuous-time DDIM generative processes as:

$$
\begin{array} { r l } & { \mathrm { d } X _ { t } = b ( X _ { t } , t ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim \mathcal { N } ( 0 , I ) , } \\ & { \mathrm { w h e r e ~ } b ( x , t ) = \kappa _ { t } x + \big ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \big ) \mathfrak { s } ( x , t ) , \quad \kappa _ { t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } , \quad \eta _ { t } = \beta _ { t } \big ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \big ) } \end{array}
$$

where $( \alpha _ { t } , \beta _ { t } )$ are coefficients of the reference flow (2). We have hence expressed the generative process of a base model, whether it is a Flow Matching or a diffusion model, as an SDE of the form (10)-(11), unified by the choice of reference flow. This expression has been written before for DDIM, e.g. Bartosh et al. (2024a,b).

# 4 Fine-tuning as “memoryless” stochastic optimal control

We now discuss the crux of the problem: how to produce a fine-tuned generative model that produces samples $X _ { 1 }$ which follow the tilted distribution involving a reward model (1). An obvious direction is to construct a fine-tuning objective involving both the base generative model and the reward model, where the optimal solution results in a fine-tuned generative model for the tilted distribution. However, as we will explain, this turns out to be non-trivial, because a naïve formulation will introduce bias into the solution.

In Section 4.1, we discuss the problem formulation of stochastic optimal control, a general framework for optimizing SDEs, and its relation to the maximum entropy reinforcement learning framework commonly used for RLHF fine-tuning. Next, in Section 4.2, we discuss the initial value function bias problem which plagues existing approaches and so far has seen no simple solution. Finally, in Section 4.3, we propose a novel simple solution that circumvents the bias problem, by enforcing a particular diffusion coefficient, the memoryless noise schedule, to be used during fine-tuning. This results in an extremely simple fine-tuning objective that provably converges to a model which generates the tilted distribution (1) without any statistical bias.

# 4.1 Preliminaries on the stochastic optimal control problem formulation

Stochastic optimal control (SOC; Bellman (1957); Fleming and Rishel (2012); Sethi (2018)) considers general optimization problems over stochastic differential equations, but we only need to consider a common instantiation, the quadratic cost control-affine problem formulation:

$$
\begin{array} { r l r } & { \underset { u \in \mathcal { U } } { \operatorname* { m i n } } \mathbb { E } \big [ \int _ { 0 } ^ { 1 } \big ( \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u } , t ) \big ) \mathrm { d } t + g ( X _ { 1 } ^ { u } ) \big ] , } \\ & { \mathrm { s . t . ~ d } X _ { t } ^ { u } = \big ( b ( X _ { t } ^ { u } , t ) + \sigma ( t ) u ( X _ { t } ^ { u } , t ) \big ) \mathrm { ~ d } t + \sigma ( t ) \mathrm { d } B _ { t } , } & { \quad X _ { 0 } ^ { u } \sim p _ { 0 } } \end{array}
$$

where in (13), $X _ { t } ^ { u } \in \mathbb { R } ^ { d }$ is the state of the stochastic process, $u : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$ is commonly referred to as the control vector field, $b : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$ is a base drift, and $\sigma : [ 0 , 1 ] \to \mathbb { R } ^ { d \times d }$ is the diffusion coefficient. These jointly define the controlled process $X ^ { u } \sim p ^ { u }$ that we are interested in optimizing; often both $b$ and $\sigma$ are fixed and we only optimize over the control $u$ .

As part of the objective functional (12), we have an affine control cost $\textstyle \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 }$ , a running state cost $f : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R }$ and a terminal state cost $g : \mathbb { R } ^ { d }  \mathbb { R }$ .

The stochastic optimal control (SOC) objective (12) can be decomposed recursively from the final time value. It is common to define the cost functional which is the expected future cost starting from state $x$ at time $t$ :

$$
\begin{array} { r } { J ( u ; x , t ) : = \mathbb { E } _ { X \sim p ^ { u } } \left[ \int _ { t } ^ { 1 } \left( \frac { 1 } { 2 } \| u ( X _ { s } , s ) \| ^ { 2 } + f ( X _ { s } , s ) \right) \mathrm { d } s + g ( X _ { 1 } ) \ \middle | \ X _ { t } = x \right] . } \end{array}
$$

From here, the value function is the optimal value of the cost functional4 :

$$
\begin{array} { r } { V ( x , t ) : = \operatorname* { m i n } _ { u \in \mathcal { U } } J ( u ; x , t ) = J ( u ^ { * } ; x , t ) , } \end{array}
$$

where $u ^ { * }$ is the optimal control, i.e., minimizer of (12). Furthermore, a classical result is that the value function can be expressed in terms of the uncontrolled base process $p ^ { \mathrm { b a s e } }$ (Kappen (2005), see Domingo-Enrich et al. 2023, Eq. 8, App. B for a self-contained proof):

$$
\begin{array} { r } { V ( x , t ) = - \log \mathbb { E } _ { X \sim p ^ { \mathrm { b a s e } } } \left[ \exp ( - \int _ { t } ^ { 1 } f ( X _ { s } , s ) \mathrm { d } s - g ( X _ { 1 } ) ) \middle \vert X _ { t } = x \right] . } \end{array}
$$

A useful expression for the optimal control (which we will make use of in deriving the Adjoint Matching objective in Section 5) is that it is related to the gradient of the value function:

$$
u ^ { * } ( x , t ) = - \sigma ( t ) ^ { \top } \nabla _ { x } V ( x , t ) = - \sigma ( t ) ^ { \top } \nabla _ { x } J ( u ^ { * } , x , t ) .
$$

Relation to MaxEnt RL. Stochastic optimal control with the control-affine formulation (12) is the continuoustime equivalence of maximum entropy reinforcement learning (MaxEnt RL; Todorov (2006); Ziebart et al. (2008)) with a KL regularization instead of only an entropy regularization. In particular, by the Girsanov theorem (Theorem 2), the affine control cost is equivalent to a Kullback–Leibler (KL) divergence between the base process $p ^ { \mathrm { b a s e } }$ , when $u = 0$ , and the controlled process $p ^ { u }$ , when conditioned on the same initial state $X _ { 0 }$ (see Appendix C.4):

$$
D _ { \mathrm { K L } } \big ( p ^ { u } ( { \pmb X } | X _ { 0 } ) \big | \big | p ^ { b a s e } ( { \pmb X } | X _ { 0 } ) \big ) = \mathbb { E } _ { { \pmb X } ^ { u } \sim p ^ { u } } \left[ \int _ { 0 } ^ { 1 } { \frac { 1 } { 2 } } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t \right] ,
$$

resulting in the KL-regularized RL interpretation of (12):

$$
\operatorname* { m a x } _ { u \in \mathcal { U } } \mathbb { E } _ { X _ { 0 } \sim p _ { 0 } } \left[ \mathbb { E } _ { X \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } { \big [ } \int _ { 0 } ^ { 1 } - f ( X _ { t } ^ { u } , t ) \mathrm { d } t - g ( X _ { 1 } ^ { u } ) { \big ] } - D _ { \mathrm { K L } } ( p ^ { u } ( X \vert X _ { 0 } ) \parallel p ^ { b a s e } ( X \vert X _ { 0 } ) ) \right] ,
$$

where the negative state costs correspond to intermediate and terminal rewards in the RL interpretation. The KL divergence incentivizes the optimal solution to stay close to the distribution of the base process.

# 4.2 The initial value function bias problem

We next discuss why naïvely adding a KL regularization does not lead to the tilted distribution (1). From (19), we can also show that the optimal distribution conditioned on $X _ { 0 }$ is5

$$
\begin{array} { r } { p ^ { * } ( X | X _ { 0 } ) \propto p ^ { \mathrm { b a s e } } ( X | X _ { 0 } ) \exp \big ( - \int _ { 0 } ^ { 1 } f ( X _ { t } , t ) \mathrm { d } t - g ( X _ { 1 } ) \big ) . } \end{array}
$$

This is analogous to the exponentiated reward distribution in MaxEnt RL (Rawlik et al., 2013), but since we generalize the entropy regularization to a KL regularization, $p ^ { \mathrm { b a s e } }$ acts as a prior distribution.

In order to relate this to the tilted distribution (1) that we want to achieve for fine-tuning, first notice that the normalization constant of the right-hand side (RHS) of (20) is exactly the value function at $t = 0$ :

$$
\begin{array} { r } { \mathbb { E } _ { X \sim p ^ { \mathrm { b a s e } } ( X | X _ { 0 } ) } \left[ \exp \big ( - \int _ { 0 } ^ { 1 } f ( X _ { t } , t ) \mathrm { d } t - g ( X _ { 1 } ) \big ) \right] = \exp \left( - V ( X _ { 0 } , 0 ) \right) , } \end{array}
$$

where the equality is due to (16). Dividing the RHS of (20) by (21) and multiplying by $p _ { 0 } ( X _ { 0 } )$ , we obtain the normalized distribution over the full path $\pmb { X }$ ,

$$
\begin{array} { r } { p ^ { * } ( X ) = p ^ { \mathrm { b a s e } } ( X ) \exp \big ( - \int _ { 0 } ^ { 1 } f ( X _ { t } , t ) \mathrm { d } t - g ( X _ { 1 } ) + V ( X _ { 0 } , 0 ) \big ) . } \end{array}
$$

Setting $f = 0$ and $g = - r$ , we arrive at an expression for the optimal distribution

$$
p ^ { * } ( X _ { 0 } , X _ { 1 } ) = p ^ { \mathrm { b a s e } } ( X _ { 0 } , X _ { 1 } ) \exp { \left( r ( X _ { 1 } ) + V ( X _ { 0 } , 0 ) \right) } .
$$

This unfortunately does not lead to the tilted distribution (1) because we have a bias in the optimal distribution that is due to the value function of the initial distribution $V ( X _ { 0 } , 0 )$ . That is to say, naïvely adding a KL regularization (18) to the fine-tuning objective in the sense of (19) leads to a biased distribution (22) after fine-tuning and is not equivalent to the tilted distribution (1). For instance, when the sampling procedure is noiseless, i.e., $\sigma ( t ) = 0$ , fine-tuning naïvely will not have any effect because $X _ { 0 }$ completely determines $X _ { 1 }$ .

This is unlike the situation for large language models (Ouyang et al., 2022; Rafailov et al., 2023), where there is no dynamical process that samples $X _ { 1 }$ iteratively and hence no dependence on the initial noise variable $X _ { 0 }$ . Although this KL regularization is a common objective for RLHF of large language models, it has seen seldom use in fine-tuning diffusion models, likely due to this issue of the initial value function bias.

In the context of diffusion models, KL regularization (19) has been explored in prior works (Fan et al., 2024), but its behavior was not well-understood and they did not relate the fine-tuned model to the tilted distribution (1). Another direction that has been proposed is to learn the initial distribution $p _ { 0 }$ to cancel out the bias (Uehara et al., 2024b; Tang, 2024) but this simply shifts the work into tilting the initial distribution and requires an auxiliary model for parameterizing the optimal initial distribution. In contrast, we show in the next section that it is possible to remove the value function bias by simply choosing a very particular noise schedule during the fine-tuning procedure.

# 4.3 The memoryless noise schedule for fine-tuning dynamical generative models

In this section, we propose a very simple method of turning (23) into the tilted distribution (1) through the use of a particular memoryless noise schedule. Throughout, we provide an intuitive explanation of why this noise schedule is sufficient for fine-tuning while discussing the full theoretical result where we show that the memoryless noise schedule is actually not only sufficient but also necessary.

Intuitively, the main reason we cannot arrive at the tilted distribution from (23) is due to the $p ^ { \mathrm { b a s e } } ( X _ { 0 } , X _ { 1 } )$ distribution not factoring into $X _ { 0 }$ and $X _ { 1 }$ . Hence, we define a memoryless generative process as follows:

Definition 1 (Memoryless generative process). A generative process of the form (10)-(11) is memoryless if $X _ { 0 }$ and $X _ { 1 }$ are independent, i.e., $p ^ { b a s e } ( X _ { 0 } , X _ { 1 } ) = p ^ { b a s e } ( X _ { 0 } ) p ^ { b a s e } ( X _ { 1 } )$ .

Table 1 Diffusion coefficient $\sigma ( t )$ and the factors $\kappa _ { t }$ , $\eta _ { t }$ for the Flow Matching, Memoryless Flow Matching, DDIM,√ and DDPM generative processes. When the diffusion coefficient is $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ , the generative process is memoryless, $i$ .e., samples $X _ { 1 }$ will be independent of the initial noise $X _ { 0 }$ .   

<table><tr><td></td><td>Kt</td><td>nt</td><td>Diffusion coefficient σ(t) Memoryless Xt</td><td></td></tr><tr><td>Flow Matching (3)</td><td>αt αt</td><td>βt(αt βt − βt)</td><td>General (commonly 0)</td><td>No</td></tr><tr><td>Memoryless Flow Matching (4)</td><td>αt αt</td><td>βt(αt βt − βt)</td><td>√2t</td><td>Yes</td></tr><tr><td>DDIM (6)</td><td>$rt }$ at$</td><td>$fra{ }$ 2t$</td><td>General (commonly 0)</td><td>No</td></tr><tr><td>DDPM (7)</td><td>$\r }$ a}</td><td>$ar}$ 2 α}</td><td>√2nt</td><td>Yes</td></tr></table>

When the base generative process is memoryless, this implies:

$$
\begin{array} { r } { p ^ { * } ( X _ { 1 } ) = \int p ^ { \mathrm { b a s e } } ( X _ { 0 } ) p ^ { \mathrm { b a s e } } ( X _ { 1 } ) \exp ( r ( X _ { 1 } ) + V ( X _ { 0 } , 0 ) ) \mathrm { d } X _ { 0 } \propto p ^ { \mathrm { b a s e } } ( X _ { 1 } ) \exp ( r ( X _ { 1 } ) ) . } \end{array}
$$

That is, solving the SOC problem (12)-(13) with a memoryless base model will result in a fine-tuned model that generates samples $p ^ { * } ( X _ { 1 } )$ according to the tilted distribution (1). This memoryless property is not satisfied generally by the family of generative processes captured by (12)-(13). For instance, the Flow Matching and DDIM generative processes with zero diffusion coefficient (i.e., $\sigma ( t ) = 0$ ) are definitely not memoryless due to $X _ { 0 }$ and $X _ { 1 }$ being theoretically invertible. Below, we provide the sufficient and neccessary condition for the noise schedule in order to have a memoryless generative process.

Proposition 1 (Memoryless noise schedules). Within the family of generative processes (10)-(11), a generative process is memoryless if and only if the noise schedule is chosen as:

$$
\begin{array} { r } { ( t ) ^ { 2 } = 2 \eta _ { t } + \chi ( t ) , \ w h e r e \ \chi : [ 0 , 1 ] \to { \mathbb R } \ i s \ s . t . \ \forall t \in ( 0 , 1 ] , \quad \operatorname* { l i m } _ { t ^ { \prime } \to 0 ^ { + } } \alpha _ { t ^ { \prime } } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( s ) } { 2 \beta _ { s } ^ { 2 } } { \mathrm { d } } s \big ) = 0 } \end{array}
$$

where $\eta _ { t }$ is the coefficient defined in (11) (see also Table 1). In particular, we refer to $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ as the memoryless noise schedule.

Due to the endpoint constraints of $( \alpha _ { t } , \beta _ { t } )$ for the reference flow (2), the memoryless noise schedule $\sigma ( t )$ is infinite at $t = 0$ and approaches zero at $t = 1$ . This provides a way for the generative process to mix when close to noise $X _ { 0 }$ while stay steadying when close to the sample $X _ { 1 }$ . Hence, the sample will have no information about $X _ { 0 }$ due to the enormous amount of mixing with a large diffusion coefficient. Furthermore, while we have intuitively justified the memoryless noise schedule through its independence property, our theoretical result is actually even stronger: all generative models of the form (10)-(11) must be fine-tuned using the memoryless noise schedule. We formalize this in the following theorem, which we prove in Appendix D.2:

Theorem 1 (Fine-tuning recipe for general noise schedule sampling). Within the family of generative processes (10)-(11), in order to allow the use of arbitrary noise schedules and still generate samples according to the tilted distribution (1), the fine-tuning problem (12)-(13) with $f = 0$ and $g = - r$ must be done with the memoryless noise schedule $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ .

Theorem $1$ states that we need to use the memoryless noise schedule for fine-tuning with the SOC objective— or equivalently, the KL regularized reward objective (19). This is the only noise schedule that retains the relationship between the velocity and score function, allowing the conversion to arbitrary noise schedules (e.g., $\sigma ( t ) = 0$ ) after fine-tuning. It is worth noting that when using the memoryless noise schedule for DDIM, this recovers what we derived as the continuous-time limit of the DDPM generative process (7). However, the DDPM sampler (Ho et al., 2020) is not commonly used while the DDIM sampler (Song et al., 2021a) and Flow Matching models typically generate samples using $\sigma ( t ) = 0$ , so an explicit conversion to the memoryless noise schedule is necessary for fine-tuning. To the best of our knowledge, we are not aware of any existing works that have proposed a time-varying diffusion coefficient with theoretical guarantees. Table 1 summarizes the memoryless schedule for diffusion and Flow Matching models, which we refer to as Memoryless Flow Matching. In Figure 2, we visualize fine-tuning a 1D model, where we see that constant $\sigma ( t )$ leads to biased distributions whereas the memoryless noise schedule perfectly converges to the tilted distribution (1).

![](images/figures/adjoint-matching-fig-0002.jpg)  
Figure 2 Visualization of Theorem 1 showing that fine-tuning must be done with the memoryless noise schedule to ensure convergence to the tilted distribution (1). (a) Shows the base Flow Matching model. (b, c) Fine-tuning using a constant $\sigma ( t )$ leads to biased distributions. (d) Fine-tuning using the memoryless noise schedule leads to the correct tilted distribution. Note that sample generation can use any noise schedule after fine-tuning, including $\sigma ( t ) = 0$ .

For convenience, we plug the memoryless noise schedule into the controlled process for fine-tuning (13), and express them in terms of each respective framework. Let $\epsilon ^ { \mathrm { b a s e } }$ , $v ^ { \mathrm { b a s e } }$ denote the pre-trained vector fields and $\epsilon ^ { \mathrm { f i n e t u n e } }$ , $v ^ { \mathrm { f i n e t u n e } }$ the fine-tuned vector fields. Then we have the following expressions for the full drift $b ( \boldsymbol { x } , t ) + \sigma ( t ) u ( \boldsymbol { x } , t )$ and control $\boldsymbol { u } ( \boldsymbol { x } , t )$ when $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ :

${ D D I M } / { \ D D P M }$

$$
\begin{array} { r } { x , t ) + \sigma ( t ) u ( x , t ) = \frac { \dot { \hat { \alpha } } _ { t } } { 2 \hat { \alpha } _ { t } } x - \frac { \dot { \hat { \alpha } } _ { t } } { \hat { \alpha } _ { t } } \frac { \epsilon ^ { \mathrm { f i n e t u m e } } ( x , t ) } { \sqrt { 1 - \hat { \alpha } _ { t } } } , \qquad u ( x , t ) = - \sqrt { \frac { \dot { \alpha } _ { t } } { \hat { \alpha } _ { t } ( 1 - \hat { \alpha } _ { t } ) } } \big ( \epsilon ^ { \mathrm { f i n e t u m e } } ( x , t ) - \epsilon ^ { \mathrm { b a s e } } ( x , t ) \big ) } \end{array}
$$

Memoryless Flow Matching:

$$
\begin{array} { r l r l } & { b ( x , t ) + \sigma ( t ) u ( x , t ) = 2 v ^ { \mathrm { f i n e t u m e } } ( x , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x , } & & { u ( x , t ) = \sqrt { \frac { 2 } { \beta _ { t } ( \frac { \alpha _ { t } } { \alpha _ { t } } \beta _ { t } - \bar { \beta } _ { t } ) } } \big ( v ^ { \mathrm { f i n e t u m e } } ( x , t ) - v ^ { \mathrm { b a s e } } ( x , t ) \big ) } \end{array}
$$

Thus, to solve the SOC problem (12)-(13) in practice, we parameterize the control $u$ in terms of $\epsilon ^ { \mathrm { f i n e t u n e } }$ or $v ^ { \mathrm { f i n e t u n e } }$ and optimize these vector fields instead. After plugging in (26)-(27), the SOC problem (12)-(13) can then be solved using any SOC algorithm in order to perform fine-tuning, and we proposed an especially effective algorithm next in Section 5. After fine-tuning, $\epsilon ^ { \mathrm { f i n e t u n e } }$ and $v ^ { \mathrm { f i n e t u n e } }$ can simply be plugged back into their respective generative processes (3)-(7) to sample from the tilted distribution (1) using any choice of diffusion coefficient.

# 5 Adjoint Matching for control-affine stochastic optimal control

We discuss existing methods and also propose a new method for optimizing control-affine SOC problems. The new Adjoint Matching method is a combination of the time-tested continuous adjoint method (Pontryagin, 1962) with recent developments on constructing least-squares objectives for solving SOC problems (Domingo-Enrich et al., 2023). In this section, we briefly discuss preliminaries on existing methods, their pros and cons, then detail the Adjoint Matching algorithm and its surprising connections to the prior methods. For numerical optimization, we now assume that the control $u$ is a parametric model with parameters $\theta$ .

# 5.1 Existing methods for stochastic optimal control

# 5.1.1 The adjoint method

The most basic method of optimizing the simulation of an SDE is to directly differentiate through the simulation using gradients from the SOC objective function (Han and E, 2016). The adjoint method simply uses the objective:

$$
\begin{array} { r } { \mathcal { L } ( u ; \boldsymbol { X } ) : = \int _ { 0 } ^ { 1 } \left( \frac { 1 } { 2 } \| u ( \boldsymbol { X } _ { t } , t ) \| ^ { 2 } + f ( \boldsymbol { X } _ { t } , t ) \right) \mathrm { d } t + g ( \boldsymbol { X } _ { 1 } ) , \qquad \boldsymbol { X } \sim p ^ { u } . } \end{array}
$$

This is a stochastic estimate of the control objective in (12), and the goal is to take compute the gradient of $\mathcal { L } ( u ; X )$ with respect to the parameters $\theta$ of the control $u$ . Due to the continuous-time nature of SDEs, there are two main approaches to implementing this numerically. Firstly, the Discrete Adjoint method uses a “discretize-then-differentiate” approach, where the numerical solver for simulating the SDE is simply stored in memory then differentiated through, and it has been studied extensively (e.g., Bierkens and Kappen (2014); Gómez et al. (2014); Hartmann and Schütte (2012); Kappen et al. (2012); Rawlik et al. (2013); Haber and Ruthotto (2017)). This approach, however, uses an extremely large amount of memory as the full computational graph of the numerical solver must be stored in memory and implementations often must rely on gradient checkpointing (Chen et al., 2016) to reduce memory usage.

Secondly, the Continuous Adjoint method exploits the continuous-time nature of SDEs and uses an analytical expression for the gradient of the control objective with respect to the intermediate states $X _ { t }$ , expressed as an adjoint ODE, and then applies a numerical method to simulate this gradient itself, hence it is referred to as a “differentiate-then-discretize” approach (Pontryagin, 1962; Chen et al., 2018; Li et al., 2020). We first define the adjoint state as:

$$
\begin{array} { r } { a ( t ; \mathbf { \nabla } _ { } \boldsymbol { X } , u ) : = \nabla _ { \boldsymbol { X } _ { t } } \big ( \int _ { t } ^ { 1 } \big ( \frac { 1 } { 2 } \| u ( \boldsymbol { X } _ { t ^ { \prime } } , t ^ { \prime } ) \| ^ { 2 } + f ( \boldsymbol { X } _ { t ^ { \prime } } , t ^ { \prime } ) \big ) \mathrm { d } t ^ { \prime } + g ( \boldsymbol { X } _ { 1 } ) \big ) , } \\ { \mathrm { w h e r e ~ } \mathbf { \nabla } _ { \boldsymbol { X } } \mathrm { ~ s o l v e s ~ d } \boldsymbol { X } _ { t } = \big ( b ( \boldsymbol { X } _ { t } , t ) + \sigma ( t ) u ( \boldsymbol { X } _ { t } , t ) \big ) \mathrm { ~ d } t + \sigma ( t ) \mathrm { d } B _ { t } . } \end{array}
$$

This implies that $\operatorname { \mathbb { E } } _ { X \sim p ^ { u } } \left[ a ( t ; X , u ) \mid X _ { t } = x \right] = \nabla _ { x } J ( u ; x , t )$ , where $J$ denotes the cost functional defined in (14). It can then be shown that this adjoint state satisfies 6:

$$
\begin{array} { r l } & { { \frac { \mathrm { d } } { \mathrm { d } t } a \mathrm { ( } t ; X , u \mathrm { ) } } = - [ a ( t ; X , u ) ^ { \top } ( \nabla _ { X _ { t } } ( b ( X _ { t } , t ) + \sigma ( t ) u ( X _ { t } , t ) ) ) + \nabla _ { X _ { t } } ( f ( X _ { t } , t ) + { \frac { 1 } { 2 } } \| u ( X _ { t } , t ) \| ^ { 2 } )  } \\ & { \quad  a ( 1 ; X , u ) = \nabla g ( X _ { 1 } ) . } \end{array}
$$

The adjoint state is solved backwards in time, starting from the terminal condition (31). Computation of (30) can be efficiently done as a vector-Jacobian product on automatic differentiation software (Paszke et al., 2019). Once the adjoint state has been solved for $t \in [ 0 , 1 ]$ , then the gradient of $\mathcal { L } ( u ; X )$ with respect to the parameters $\theta$ can be obtained by integrating over the entire time interval:

$$
\begin{array} { r } { \frac { \mathrm { d } \mathcal { L } } { \mathrm { d } \theta } = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \frac { \partial } { \partial \theta } \| u ( X _ { t } , t ) \| ^ { 2 } \mathrm { d } t + \int _ { 0 } ^ { 1 } \frac { \partial u ( X _ { t } , t ) } { \partial \theta } ^ { \top } \sigma ( t ) ^ { \top } a ( t ; \mathbf { X } , u ) \mathrm { d } t , } \end{array}
$$

where the first term is the partial derivative of $\mathcal { L }$ w.r.t. $\theta$ and the second term is the partial derivative through the sample trajectory $\pmb { X }$ . See Proposition 6 in Appendix E.1 for a statement and proof of this result. The discrete and continuous adjoint methods converge to the same gradient as the step size of the numerical solvers go to zero. Both are scalable to high dimensions and have seen their fair share of usage in optimizing neural ODE/SDEs (Chen et al., 2018, 2021; Li et al., 2020). As the adjoint methods are essentially gradient-based optimization algorithms applied on a highly non-convex problem, many have also reported they can be unstable empirically (Mohamed et al., 2020; Suh et al., 2022; Domingo-Enrich et al., 2023).

# 5.1.2 Importance-weighted matching objectives for regressing onto the optimal control

An alternative is to consider regressing onto the optimal control $u ^ { * }$ , which is the approach of the cross-entropy method (Rubinstein and Kroese, 2013; Zhang et al., 2014) and stochastic optimal control matching (SOCM; Domingo-Enrich et al. (2023)). These methods make use of path integral theory (Kappen, 2005) to express

the optimal control through importance sampling, resulting in an importance-weighted least-squares objective function

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S O C M } } ( u ; \boldsymbol { X } ) : = \int _ { 0 } ^ { 1 } \| u ( X _ { t } , t ) - \hat { u } ^ { * } ( X _ { t } , t ) \| ^ { 2 } \mathrm { d } t \times \omega ( u , \boldsymbol { X } ) , \qquad \boldsymbol { X } \sim p ^ { u } , } \end{array}
$$

where $\omega$ is an importance weighting that approximates sampling from the optimal distribution $p ^ { * }$ , and $\hat { u } ^ { * }$ is a stochastic estimator of the optimal control relying on having sampled from the optimal process. We defer to Domingo-Enrich et al. (2023) for the exact details. The functional landscape of this objective is convex, which is argued to help yield stable training. However, the need for importance sampling renders this impractical for high dimensional applications: the variance of the importance weighting $\omega$ grows exponentially with dimension of the stochastic process, leading to catastrophic failure. This unfortunately means that such importance-weighted matching objectives are impractical for fine-tuning dynamical generative models; however, a least-squares objective is greatly coveted as it can lead to stable training and simple interpretations.

# 5.2 Adjoint Matching

We make two important observations which lead to our proposed method: $( i )$ it is possible to construct a matching objective without any importance weighting, and $( i i )$ there are unnecessary terms in the adjoint differential equation (30) that can lead to higher variance at convergence.

Firstly, we notice that we can simply match the gradient of the cost functional under the current control. That is, while SOCM carefully constructs an importance-weighted estimator of the optimal control $u ^ { * } =$ $- \sigma ( t ) ^ { 1 } \nabla J ( u ^ { * } ; x , t )$ (17), we claim that we can actually just regress onto the target vector field $- \sigma ( t ) ^ { 1 } \nabla J ( u ; x , t )$ where $u$ is the current control, and furthermore, this results in a gradient equal in expectation to the continuous adjoint method. We formalize this in the following proposition, proven in Appendix E.2:

Proposition 2. Let us define, for now, the basic Adjoint Matching objective as:

$$
\begin{array} { r } { \mathrm { a s i c - A d j - M a t h } ( u ; X ) : = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left\| u ( X _ { t } , t ) + \sigma ( t ) ^ { \mathsf { T } } a ( t ; X , \bar { u } ) \right\| ^ { 2 } \mathrm { d } t , \qquad X \sim p ^ { \bar { u } } , \quad \bar { u } = s t o p g r a d ( u ; X ) . } \end{array}
$$

where $\bar { u } = s t o p g r a d ( u )$ means that the gradients of $\bar { u }$ with respect to the parameters $\theta$ of the control u are artificially set to zero. The gradient of $\mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ( u ; X )$ with respect to $\theta$ is equal to the gradient $\frac { \mathrm { d } { \mathcal { L } } } { \mathrm { d } \theta }$ in equation (32). Importantly, the only critical point of $\mathbb { E } \left[ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } \right]$ is the optimal control $u ^ { * }$ .

Critical points of $\mathcal { L }$ are controls $u$ such that $\begin{array} { r } { \frac { \delta } { \delta u } \mathcal { L } ( u ) = 0 } \end{array}$ , where $\begin{array} { r } { \frac { \delta } { \delta u } \mathcal { L } } \end{array}$ denotes the first variation of the functional $\mathcal { L }$ . In other words, Proposition 2 states that the only control that satisfies the first-order optimality condition for the basic Adjoint Matching objective is the optimal control, which provides theoretical grounding for gradient-based optimization algorithms.

An intuitive way to understand the basic Adjoint Matching objective is that it is a consistency loss. The Adjoint Matching objective is based off of the observation that the optimal control $\boldsymbol { u } ^ { * } ( x , t )$ is the unique fixed-point of the relation $\boldsymbol { u } ( \boldsymbol { x } , t ) = - \sigma ( t ) ^ { \mathsf { I } } \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } ; \boldsymbol { x } , t )$ (see Lemma 6 in Appendix E.2) and so we are directly optimizing for a control that fits this relation, while using the adjoint state as a stochastic estimator of $\nabla _ { x } J ( u ; x , t )$ (29).

The basic Adjoint Matching objective in Proposition 2 does not yet yield a novel algorithm for stochastic optimal control, because it produces the same gradient as the continuous adjoint method. This can be seen by taking the gradient w.r.t. $\theta$ after expanding the square in (34) and removing terms that do not depend on $\theta$ to arrive exactly at the continuous adjoint method (32). However, it provides the means of deriving a simpler leaner objective function.

The “Lean” Adjoint. The minimizer of a least-squares objective is the conditional expectation of the regression target, so for the Adjoint Matching objective, at the optimum we have that

$$
u ^ { * } ( x , t ) = \mathbb { E } _ { X \sim p ^ { * } } \left[ - \sigma ( t ) ^ { \top } a ( t ; X , u ^ { * } ) | X _ { t } = x \right] .
$$

Multiplying both sides by the Jacobian $\nabla _ { x } u ^ { * } ( x , t )$ and re-arranging, we get the relation

$$
\mathbb { E } _ { X \sim p ^ { * } } \left[ u ^ { * } ( x , t ) ^ { \mathsf { T } } \nabla _ { x } u ^ { * } ( x , t ) + a ( t ; X , u ^ { * } ) ^ { \mathsf { T } } \sigma ( t ) \nabla _ { x } u ^ { * } ( x , t ) \mid X _ { t } = x \right] = 0 .
$$

Input: Pre-trained FM velocity field $v ^ { \mathrm { b a s e } }$ , step size $h$ , number of fine-tuning iterations $N$ . Initialize fine-tuned vector fields: $v ^ { \mathrm { f i n e t u n e } } = v ^ { \mathrm { b a s e } }$ with parameters $\theta$ . for $n \in \{ 0 , \ldots , N - 1 \}$ do

Sample $m$ trajectories $\pmb { X } = ( X _ { t } ) _ { t \in \{ 0 , \ldots , 1 \} }$ with memoryless noise schedule $\begin{array} { r } { \sigma ( t ) = \sqrt { 2 \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) } } \end{array}$ , e.g.:

$$
\begin{array} { r } { X _ { t + h } = X _ { t } + h \left( 2 v _ { \theta } ^ { \mathrm { f n e t u n e } } ( X _ { t } , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } \right) + \sqrt { h } \sigma ( t ) \varepsilon _ { t } , \qquad \varepsilon _ { t } \sim \mathcal { N } ( 0 , I ) , \qquad X _ { 0 } \sim \mathcal { N } ( 0 , I ) . } \end{array}
$$

For each trajectory, solve the lean adjoint $O D E$ (38)-(39) backwards in time from $t = 1$ to $_ 0$ , e.g.:

$$
\begin{array} { r l r } { \tilde { a } _ { t - h } = \tilde { a } _ { t } + h \tilde { a } _ { t } ^ { \mathsf { T } } \nabla _ { X _ { t } } \left( 2 v ^ { \mathrm { b a s e } } ( X _ { t } , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } \right) , } & { } & { \tilde { a } _ { 1 } = - \nabla _ { X _ { 1 } } r ( X _ { 1 } ) . } \end{array}
$$

Note that $X _ { t }$ and $\ddot { a } _ { t }$ should be computed without gradients, i.e., $X _ { t } = \tt s t o p g r a d ( X _ { t } )$ , $\tilde { \boldsymbol { a } } _ { t } = \mathsf { s t o p g r a d } ( \tilde { \boldsymbol { a } } _ { t } )$

For each trajectory, compute the Adjoint Matching objective (37):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { A d j - M a t c h } } ( \theta ) = \sum _ { t \in \{ 0 , \dots , 1 - h \} } \big \| \frac { 2 } { \sigma ( t ) } \big ( v _ { \theta } ^ { \mathrm { f u n e t u n e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) \big ) + \sigma ( t ) \tilde { a } _ { t } \big \| ^ { 2 } . } \end{array}
$$

Compute the gradient $\nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } )$ and update $\theta$ using favorite gradient descent algorithm.

Output: Fine-tuned vector field $v$ finetune

Notice that the terms inside the expectation in (36) show up as part of the adjoint differential equation (30), which we have now shown to have expectation zero at the optimal solution. Therefore, we motivate the definition of a lean adjoint state $\tilde { a }$ with the terms in (36) removed. Plugging this lean adjoint back into the least-squares objective, we obtain our final proposed Adjoint Matching objective:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { A d j - M a t c h } } ( u ; \mathbf { X } ) : = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left\| u ( X _ { t } , t ) + \sigma ( t ) ^ { \top } \tilde { a } ( t ; \mathbf { X } ) \right\| ^ { 2 } \mathrm { d } t , } & { \quad \mathbf { X } \sim p ^ { \bar { u } } , \quad \bar { u } = \mathbf { s t o p g r a d } ( u ) , } \\ { \mathrm { w h e r e } } & { ~ \frac { \mathrm { d } } { \mathrm { d } t } \tilde { a } ( t ; \mathbf { X } ) = - ( \tilde { a } ( t ; \mathbf { X } ) ^ { \top } \nabla _ { x } b ( X _ { t } , t ) + \nabla _ { x } f ( X _ { t } , t ) ) , } \\ & { ~ \tilde { a } ( 1 ; \mathbf { X } ) = \nabla _ { x } g ( X _ { 1 } ) . } \end{array}
$$

Equations (38)-(39) define the lean adjoint state, and (37) is the complete Adjoint Matching objective. The unique critical point of $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ is the optimal control, which we prove relying on Proposition 2 and equation (36) (see Proposition 7 in Appendix E.3).

Compared to the importance sampling methods (Section 5.1.2), Adjoint Matching is a simple least-squares regression objective and has no importance weighting. This allows it to avoid the pitfalls of high variance importance weights and makes it as scalable as the adjoint methods while retaining the interpretation of matching a target vector field.

Compared to the adjoint method (Section 5.1.1), Adjoint Matching produces a different gradient in expectation than the continuous adjoint. This is because the lean adjoint state is not related to the gradient of the cost functional anymore, i.e., (29) is not true, except at the optimum when $u = u ^ { * }$ . Even at the optimal solution, since Adjoint Matching removes terms that have expectation zero, it can potentially exhibit better convergence and lower variance than the continuous adjoint method. Additionally, computation of the lean adjoint state (38) also exhibits a smaller computational cost due to the removal of the extra terms (no longer need the Jacobian of the control $\nabla _ { x } u$ ). We provide a rigorous derivation of Adjoint Matching and the above claims in Appendix E.3.

Adjoint Matching can be applied to reward fine-tuning of dynamical generative models through the memoryless SOC formulation discussed in Section 4. We provide pseudo-code for this in Algorithm 1 for Flow Matching models and in Algorithm 2 in Appendix E.4 for denoising diffusion models.

# 6 Related work

Fine-tuning from human feedback. There are two main overarching approaches to RLHF: the reward-based approach (Ziegler et al., 2020; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022) and direct preference optimization (DPO; Rafailov et al. (2023)). The reward-based approach (Ziegler et al., 2020; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022) consists in learning the reward model $r ( x )$ from human preference data, and then solving a maximum entropy RL problem with rewards produced by $r ( x )$ . DPO merges the two previous steps into one: there is no need to learn $r ( x )$ as human preference data is directly used to fine-tune the model. However, DPO is typically only applied with a filtered dataset, and does not work explicitly with a reward model. Furthermore, for flow and diffusion models specifically, it is possible to differentiate the reward function, so there is a larger emphasis on reward-based approaches.

Fine-tuning for diffusion models. Among existing reward-based diffusion fine-tuning methods, Fan and Lee (2023) interpret the denoising process as a multi-step decision-making task and use policy gradient algorithms to fine-tune diffusion samplers. Black et al. (2024) makes use of proximal policy gradients for fine-tuning but this does not make use of the differentiability of the reward model. Fan et al. (2023) also consider KL-regularized rewards (19) but do not make the critical connection to the tilted distribution (1) that we flesh out in Section 4.2. The fine-tuning algorithms of $\mathrm { X }$ u et al. (2023); Clark et al. (2024) directly take gradients of the reward model and use heuristics to try to stay close to the original base generative model, but their behavior is not well understood and unrelated to the tilted distribution: Xu et al. (2023) takes gradients of the reward applied on the denoised sample at different points in time, and Clark et al. (2024) backpropagates the reward function through all or part of the diffusion trajectory. Finally, Uehara et al. (2024b) also fine-tune diffusion models with the goal of sampling from the tilted distribution (1), but their approach is much more involved than ours as it requires learning a value function, and solving two stochastic optimal control problems. Additional reward fine-tuning works include Bruna and Han (2024), that provide theoretical guarantees to sample from the tilted distribution when the reward is a quadratic function, and Zhang et al. (2024), that propose a reward fine-tuning algorithm for the GFlowNet architecture.

Inference-time optimization methods. Some have proposed methods that do not update the base model but instead modify the generation process directly. One approach is to add a guidance term to the velocity (Chung et al., 2022; Song et al., 2023; Pokle et al., 2023); however, this is a heuristic and it is not well-understood what particular distribution is being generated. Another approach is to directly optimize the initial noise distribution (Li, 2021; Wallace et al., 2023b; Ben-Hamu et al., 2024); this is taking an opposite approach to the inital value bias problem than us by moving all of the work into optimizing the initial distribution. A more computationally intensive approach is to perform online estimation of the optimal control, for the purpose of heuristically solving an optimal control problem within the sampling process (Huang et al., 2024; Rout et al., 2024); these approaches aim to solve a separate control problem for each generated sample, instead of performing amortization (Amos et al., 2023) to learn a fine-tuned generative model.

Optimal control in generative modeling. Methods from optimal control have been used to train dynamical generative models parameterized by ODEs (Chen et al., 2018), SDEs (Li et al., 2020), and jump processes (Chen et al., 2021), enabled through the adjoint method. They can be used to train arbitrary generative processes, but for simplified constructions these have fallen in favor of simulation-free matching objectives such as denoising score matching (Vincent, 2011) and Flow Matching (Lipman et al., 2023). The optimal control formalism also has significance in sampling from un-normalized distributions (Zhang and Chen, 2022; Berner et al., 2023; Vargas et al., 2023, 2022; Richter and Berner, 2024; Tzen and Raginsky, 2019). The inclusion of a state cost has been used to solve transport problems where intermediate path distributions are of importance (Liu et al., 2024; Pooladian et al., 2024). These collective advances naturally lead to the consideration of the optimal control formalism for reward fine-tuning.

Conditional sampling in inverse problems. Denker et al. (2024) and Wu et al. (2023a) independently consider a pre-trained diffusion model $p ( x )$ , and an observation $y$ on the generated sample $x$ , as well as the analytic likelihood $p ( y | x )$ . Their aim is to sample from the posterior $p ( x ) p ( y | x )$ , and their applications include inpainting, class-conditional generation, super-resolution, phase retrieval, non-linear deblurring, computed tomography, and protein design. Their setting reduces to a particular case of our reward fine-tuning framework by setting $r ( x ) = \log p ( y | x )$ . Denker et al. (2024) formulate an SOC problem, and they solve it via the log-variance loss (Richter et al. (2020); Nüsken and Richter (2021)), and the moment loss (Nüsken and Richter, 2021)7, which they refer to as the trajectory balance loss (Malkin et al., 2023). Wu et al. (2023a) propose Twisted Diffusion Sampler, an algorithm based on Sequential Monte Carlo that uses increased inference-time compute to reduce bias. A third work that also tackles the conditional sampling problem is Du et al. (2024), which use a Lagrangian formulation that they solve approximately using Gaussian paths.

Table 2 Evaluation metrics of different fine-tuning methods for text-to-image generation. The second and third columns show the noise schedules $\sigma ( t )$ used for fine-tuning and for sampling: $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ corresponds to Memoryless Flow Matching, and $\sigma ( t ) = 0$ to the Flow Matching ODE (3). We report standard errors estimated over 3 runs of the fine-tuning algorithm on random sets of 40000 training prompts, each evaluated over a random set of 1000 test prompts.   

<table><tr><td>Fine-tuning Method</td><td>Fine-tuning σ(t)</td><td>Sampling σ(t)</td><td>ClipScore ↑</td><td>PickScore ↑</td><td>HPS v2↑</td><td>DreamSim Diversity↑</td></tr><tr><td>None (Base model)</td><td>N/A</td><td>√2ηt 0</td><td>24.15±0.26 28.32±0.22</td><td>17.25±0.06 18.15±0.07</td><td>16.19±0.17 17.89±0.16</td><td>53.60±1.37 56.53±1.52</td></tr><tr><td>DRaFT-1 sesg DRaFT-40</td><td>√2t 0 √2t 0</td><td>√2ηt 0 √2t 0</td><td>30.18±0.24 30.95±0.28 26.94±0.28 30.07±0.39</td><td>19.38±0.08 19.37±0.06 18.34±0.19 19.45±0.08</td><td>24.61±0.17 24.37±0.17 19.98±1.02 24.06±0.24</td><td>25.54±0.99 27.39±1.14 41.98±2.14 36.53±1.69</td></tr><tr><td>DPO ReFL</td><td>√2t 0 √2t</td><td>√2t 0 √2t</td><td>24.11±0.22 27.77±0.18 28.59±0.31</td><td>17.24±0.06 17.92±0.07 18.68±0.10</td><td>16.15±0.14 17.30±0.20 22.24±0.46</td><td>53.27±1.36 54.11±1.50 32.71±2.76</td></tr><tr><td>Cont. Adjoint λ = 12500 Nae occc</td><td>0 √2t</td><td>0 √2nt 0</td><td>30.06±0.63 26.99±0.43 29.49±0.32</td><td>19.07±0.21 18.33±0.16 18.98±0.16</td><td>23.06±0.41 20.83±0.63 21.34±0.53</td><td>32.69±1.28 46.59±1.40 48.41±1.44</td></tr><tr><td>Disc. Adjoint λ = 12500</td><td>√2t</td><td>√2ηt 0</td><td>28.04±0.57 29.28±0.17</td><td>18.44±0.21 18.82±0.14</td><td>20.04±0.39 19.73±0.17</td><td>54.90±2.03 53.36±2.48</td></tr><tr><td>Adj.-Matching λ = 1000</td><td>√2t</td><td>√2t 0</td><td>30.36±0.22 31.41±0.22</td><td>19.29±0.08 19.57±0.09</td><td>24.12±0.17 23.29±0.18</td><td>40.89±1.50 43.10±1.76</td></tr><tr><td>Adj.-Matching λ = 2500</td><td>√2t</td><td>√2t 0</td><td>30.59±0.40 31.64±0.21</td><td>19.49±0.10 19.71±0.09</td><td>24.85±0.23 24.12±0.27</td><td>37.07±1.47 39.88±1.59</td></tr><tr><td>Adj.-Matching λ = 12500</td><td>√2t</td><td>√2t 0</td><td>30.62±0.30 31.65±0.19</td><td>19.50±0.09 19.76±0.08</td><td>24.95±0.28 24.49±0.27</td><td>34.50±1.33 37.24±1.57</td></tr></table>

# 7 Experiments

We experimentally validate our proposed method on reward fine-tuning a Flow Matching base model (Lipman et al., 2023). In particular, we use the usual setup of pre-training an autoencoder for $5 1 2 \times 5 1 2$ resolution images, then training a text-conditional Flow Matching model on the latent variables with a U-net architecture (Long et al., 2015), similar to the setup in Rombach et al. (2022). We pre-trained our base model using a dataset of licensed text and image pairs. Then for fine-tuning, we consider the reward function:

$$
r ( x ) : = \lambda \times \mathtt { R e w a r d M o d e l } ( x )
$$

corresponding to a scaled version of the reward model, which we take to be ImageReward (Xu et al., 2023).   
Different values of $\lambda$ provide different tradeoffs between the KL regularization and the reward model (19).

![](images/figures/adjoint-matching-fig-0003.jpg)  
Figure 3 Our proposed Adjoint Matching using the memoryless SOC formulation introduces a much more principled way of trading off how close to stay to the base model while optimizing the reward model. In contrast, baseline methods such as DRaFT-1 only optimize the reward model and must rely on early stopping to perform this trade off, resulting in a much more sensitive hyperparameter. Samples are produced using $\sigma ( t ) = 0$ with the same noise sample. Text prompts: “Handsome Smiling man in blue jacket portrait” and “Quinoa and Feta Stuffed Baby Bell Peppers”.

![](images/figures/adjoint-matching-fig-0004.jpg)

Text prompt: “Man sitting on sofa at home in front of fireplace and using laptop computer, rear view ”

![](images/figures/adjoint-matching-fig-0005.jpg)  
Text prompt: “3D World Food Day Morocco”   
Figure 4 Generated samples from varying classifier-free guidance weight $w$ , from an Adjoint Matching fine-tuned model. Higher guidance increases text-to-image consistency but loses diversity and has use cases for generating highly structured images such as 3D renderings. Corresponding samples from the base model can be found in Figure 7.

For evaluation and benchmarking purposes, we report metrics that separately quantify text-to-image consistency, human preference, and sample diversity, capturing the tradeoff between each aspect of generative models (Astolfi et al., 2024). For consistency, we make use of the standard ClipScore (Hessel et al., 2021) and PickScore (Kirstain et al., 2023); for generalization to unseen human preferences, we use the HPSv2 model (Wu et al., 2023b); and for diversity, we compute averages of pairwise distances of the DreamSim features (Fu et al., 2023). More details are provided in Appendix G.4.

As our baselines, we consider the DPO (Wallace et al., 2023a), ReFL (Xu et al., 2023), and DRaFT-K algorithms (Clark et al., 2024). DPO does not use gradients from the reward function, while ReFL and DRaFT make use of heuristic gradient stopping approaches to stay close to the base generative model. Out of these baseline methods, we find that DRaFT-1 performs the best, so we perform additional ablation experiments comparing to this method. Within the same SOC formulation as our method, we also consider the discrete and continuous adjoint methods. We provide full experimental details in Appendix G; an important implementation detail is that we slightly offset $\sigma ( t )$ in order to avoid division by zero.

![](images/figures/adjoint-matching-fig-0006.jpg)  
Figure 5 Tradeoffs between different aspects of generative models: text-to-image consistency (ClipScore), sample diversity for each prompt (DreamSim Diversity), and generalization to unseen human preferences (HPS v2). Different points are obtained from varying values of $\lambda$ for Adjoint Matching and varying number of fine-tuning iterations for the DRaFT-1 baseline. Overall, we find our proposed method Adjoint Matching has the best Pareto fronts.

Evaluation results. In Table 2 we report the evaluation metrics for the baselines as well as our proposed Adjoint Matching approach. We compare each method at roughly the same wall clock time (see the times and number of iterations in Table 4, and comments in Appendix G.5). We find that across all metrics, our proposed memoryless SOC formulation outperforms existing baseline methods. The choice of SOC algorithms also obviously favors Adjoint Matching over continuous and discrete adjoint methods, which result in poorer consistency and human preference metrics.

Ablation: base model vs. reward tradeoff. We note that the scaling in front of the reward model $\lambda$ determines how strongly the we should prefer the reward model over the base model. As such, we see a natural tradeoff curve: higher $\lambda$ results in better consistency and human preference, but lower diversity in the generated samples. Overall, we find that Adjoint Matching performs stably across all values of $\lambda$ . Our method of regularizing the fine-tuning procedure through memoryless SOC works much better than baseline methods which often must employ early stopping. We show the qualitative effect of varying $\lambda$ in Figure 3, while for the DRaFT-1 baseline we show the effect of varying the number of fine-tuning iterations.

Ablation: classifier-free guidance. We note that it is possible to apply classifier-free guidance (CFG; Ho and Salimans (2022); Zheng et al. (2023)) after fine-tuning. We use the formula $( 1 + w ) v ( x , t | y ) - w v ( x , t )$ where $w$ is the guidance weight, $v ( x , t | y )$ is a fine-tuned text-to-image model while $\boldsymbol { v } ( \boldsymbol { x } , t )$ is an unconditional image model. This is not principled as only the conditional model is fine-tuned, but generally it is unclear what distribution guided models sample from anyhow. In Figure 5 we show the evaluation metrics with classifier-free guidance applied. Comparing three different guidance weight values, we see a higher weight does improve text-to-image consistency, and to some extent, human preference, but this comes at the cost of being worse in terms of diversity. We show qualitative differences in Figure 4.

# 8 Conclusion

We investigate the problem of fine-tuning dynamical generative models such as Flow Matching and propose the use of a stochastic optimal control (SOC) formulation with a memoryless noise schedule. This ensures we converge to the same tilted distribution that the large language modeling literature uses for learning from human feedback. In particular, the memoryless noise schedule corresponds to DDPM sampling for diffusion models and a new Memoryless Flow Matching generative process for flow models. In conjunction, we propose a novel training algorithm for solving stochastic optimal control problems, by casting SOC as a regression problem, which we call the Adjoint Matching objective. Empirically, we find that our memoryless SOC formulation works better than multiple existing works on fine-tuning diffusion models, and our Adjoint Matching algorithm outperforms related gradient-based methods. In summary, we are the first to provide a theoretically-driven algorithm for fine-tuning Flow Matching models, and we find that our approach significantly outperforms baseline methods across multiple axes of evaluation—text-to-image consistency, generalization to unseen human preference, and sample diversity—on large-scale text-to-image generation.

# References

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023. Cited on page 36.

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023. Cited on pages 2, 3, and 36.

Brandon Amos et al. Tutorial on amortized optimization. Foundations and Trends® in Machine Learning, 16(5): 592–732, 2023. Cited on page 12.

Pietro Astolfi, Marlene Careil, Melissa Hall, Oscar Mañas, Matthew Muckley, Jakob Verbeek, Adriana Romero Soriano, and Michal Drozdzal. Consistency-diversity-realism pareto fronts of conditional image generative models. arXiv preprint arXiv:2406.10429, 2024. Cited on page 14.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. Cited on pages 2 and 12.

Grigory Bartosh, Dmitry Vetrov, and Christian A. Naesseth. Neural diffusion models. arXiv preprint arXiv:2310.08337, 2024a. Cited on page 4.

Grigory Bartosh, Dmitry Vetrov, and Christian A. Naesseth. Neural flow diffusion models: Learnable forward process for improved diffusion modelling. arXiv preprint arXiv:2404.12940, 2024b. Cited on page 4.

Richard Bellman. Dynamic programming. Princeton Landmarks in Mathematics. Princeton University Press, Princeton, NJ, 2010., 1957. Cited on page 5.

Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. arXiv preprint arXiv:2402.14017, 2024. Cited on page 12.

Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. arXiv preprint arXiv:2211.01364, 2023. Cited on page 12.

Joris Bierkens and Hilbert J Kappen. Explicit solution of relative entropy weighted control. Systems & Control Letters, 72:36–43, 2014. Cited on page 9.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024. Cited on pages 2, 12, and 37.

Joan Bruna and Jiequn Han. Posterior sampling with denoising oracles via tilted transport. arXiv preprint arXiv:2407.00745, 2024. Cited on page 12.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. Cited on pages 9 and 12.

Ricky T. Q. Chen, Brandon Amos, and Maximilian Nickel. Learning neural event functions for ordinary differential equations. In International Conference on Learning Representations, 2021. Cited on pages 9 and 12.

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. Cited on page 9.

Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022. Cited on page 12.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. In The Twelfth International Conference on Learning Representations, 2024. Cited on pages 2, 12, and 14.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. In Advances in Neural Information Processing Systems, volume 34, pages 17695–17709. Curran Associates, Inc., 2021. Cited on page 34.

Alexander Denker, Francisco Vargas, Shreyas Padhy, Kieran Didi, Simon Mathis, Vincent Dutordoir, Riccardo Barbano, Emile Mathieu, Urszula Julia Komorowska, and Pietro Lio. Deft: Efficient finetuning of conditional diffusion models by learning the generalised $h$ -transform. arXiv preprint arXiv:2406.01781, 2024. Cited on pages 12 and 13.

Carles Domingo-Enrich. A taxonomy of loss functions for stochastic optimal control. arXiv preprint arXiv:2410.00345, 2024. Cited on page 13.

Carles Domingo-Enrich, Jiequn Han, Brandon Amos, Joan Bruna, and Ricky T. Q. Chen. Stochastic optimal control matching. arXiv preprint arXiv:2312.02027, 2023. Cited on pages 5, 8, 9, 10, 46, and 47.

Yuanqi Du, Michael Plainer, Rob Brekelmans, Chenru Duan, Frank Noé, Carla P. Gomes, Alan Apsuru-Guzik, and Kirill Neklyudov. Doob’s lagrangian: A sample-efficient variational approach to transition path sampling. arXiv preprint arXiv:2410.07974, 2024. Cited on page 13.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. Cited on page 2.

Ying Fan and Kangwook Lee. Optimizing ddpm sampling with shortcut fine-tuning. In International Conference on Machine Learning, 2023. Cited on pages 2 and 12.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023. Cited on pages 2 and 12.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. Cited on pages 2 and 6.

W.H. Fleming and R.W. Rishel. Deterministic and Stochastic Optimal Control. Stochastic Modelling and Applied Probability. Springer New York, 2012. Cited on page 5.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023. Cited on pages 14 and 54.

Vicenç Gómez, Hilbert J Kappen, Jan Peters, and Gerhard Neumann. Policy search for path integral control. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 482–497. Springer, 2014. Cited on page 9.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014. Cited on page 2.

Eldad Haber and Lars Ruthotto. Stable architectures for deep neural networks. Inverse problems, 34(1):014004, 2017. Cited on page 9.

Jiequn Han and Weinan E. Deep learning approximation for stochastic control problems. arXiv preprint arXiv:1611.07422, 2016. Cited on page 9.

Carsten Hartmann and Christof Schütte. Efficient rare event simulation by optimal nonequilibrium forcing. Journal of Statistical Mechanics: Theory and Experiment, 2012(11):P11004, 2012. Cited on page 9.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. Cited on pages 2 and 14.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. Cited on pages 2, 15, and 25.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33. Curran Associates, Inc., 2020. Cited on pages 2, 3, 4, and 7.

Yujia Huang, Adishree Ghatare, Yuanzhe Liu, Ziniu Hu, Qinsheng Zhang, Chandramouli S Sastry, Siddharth Gururani, Sageev Oore, and Yisong Yue. Symbolic music generation with non-differentiable rule guided diffusion. arXiv preprint arXiv:2402.14285, 2024. Cited on page 12.

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. Cited on page 54.

H J Kappen. Path integrals and symmetry breaking for optimal control theory. Journal of Statistical Mechanics: Theory and Experiment, 2005(11), nov 2005. Cited on pages 5 and 9.

Hilbert J Kappen, Vicenç Gómez, and Manfred Opper. Optimal control as a graphical model inference problem. Machine learning, 87(2):159–182, 2012. Cited on page 9.

Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. On density estimation with diffusion models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. Cited on page 2.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. Cited on pages 2, 14, and 54.

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36, 2024. Cited on page 2.

Dongzhuo Li. Differentiable gaussianization layers for inverse problems regularized by deep generative models. arXiv preprint arXiv:2112.03860, 2021. Cited on page 12.

Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics, pages 3870–3882. PMLR, 2020. Cited on pages 9 and 12.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. Cited on pages 2, 3, 12, 13, and 36.

Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and Ricky T. Q. Chen. Generalized schrödinger bridge matching. In The Twelfth International Conference on Learning Representations, 2024. Cited on page 12.

Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022. Cited on page 3.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. Cited on pages 2 and 3.

Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3431–3440, 2015. Cited on page 13.

Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in gflownets. arXiv preprint arXiv:2201.13259, 2023. Cited on page 13.

Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–planck equations through gradient–log–density estimation. Entropy, 22(8):802, 2020. Cited on page 3.

Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte carlo gradient estimation in machine learning. Journal of Machine Learning Research, 21(132):1–62, 2020. Cited on page 9.

Alexander Mordvintsev, Christopher Olah, and Mike Tyka. Inceptionism: Going deeper into neural networks. Google research blog, 20(14):5, 2015. Cited on page 2.

Nikolas Nüsken and Lorenz Richter. Solving high-dimensional Hamilton–Jacobi–Bellman pdes using neural networks: perspectives from the theory of controlled diffusions and measures on path space. Partial differential equations and applications, 2:1–48, 2021. Cited on page 13.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022. Cited on pages 2, 6, and 12.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. Cited on page 9.

Ashwini Pokle, Matthew J Muckley, Ricky T. Q. Chen, and Brian Karrer. Training-free linear image inversion via flows. arXiv preprint arXiv:2310.04432, 2023. Cited on page 12.

L.S. Pontryagin. The Mathematical Theory of Optimal Processes. Interscience Publishers, 1962. Cited on pages 8 and 9.

Aram-Alexandre Pooladian, Carles Domingo-Enrich, Ricky T. Q. Chen, and Brandon Amos. Neural optimal transport with lagrangian costs. arXiv preprint arXiv:2406.00288, 2024. Cited on page 12.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. Cited on pages 6 and 12.

Konrad Rawlik, Marc Toussaint, and Sethu Vijayakumar. On stochastic optimal control and reinforcement learning by approximate inference. In Twenty-Third International Joint Conference on Artificial Intelligence, 2013. Cited on pages 6 and 9.

Lorenz Richter and Julius Berner. Improved sampling via learned diffusions. In The Twelfth International Conference on Learning Representations, 2024. Cited on page 12.

Lorenz Richter, Ayman Boustati, Nikolas Nüsken, Francisco Ruiz, and Omer Deniz Akyildiz. VarGrad: A low-variance gradient estimator for variational inference. Advances in Neural Information Processing Systems, 33, 2020. Cited on page 13.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. Cited on pages 2 and 13.

Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. Cited on page 12.

Reuven Y Rubinstein and Dirk P Kroese. The cross-entropy method: a unified approach to combinatorial optimization, Monte-Carlo simulation and machine learning. Springer Science & Business Media, 2013. Cited on page 9.

Christoph Schuhmann and Romain Beaumont. Laion-aesthetics, 2022. Cited on page 2.

S.P. Sethi. Optimal Control Theory: Applications to Management Science and Economics. Springer International Publishing, 2018. Cited on page 5.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022. Cited on page 2.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. Cited on pages 3, 7, and 31.

Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023. Cited on page 12.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. arXiv preprint arXiv:1907.05600, 2019. Cited on page 2.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR 2021), 2021b. Cited on pages 2 and 3.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008–3021. Curran Associates, Inc., 2020. Cited on pages 2 and 12.

Hyung Ju Suh, Max Simchowitz, Kaiqing Zhang, and Russ Tedrake. Do differentiable simulators give better policy gradients? In International Conference on Machine Learning, pages 20668–20696. PMLR, 2022. Cited on page 9.

Wenpin Tang. Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond. arXiv preprint arXiv:2403.06279, 2024. Cited on page 6.

Emanuel Todorov. Linearly-solvable markov decision problems. Advances in neural information processing systems, 19, 2006. Cited on page 5.

Belinda Tzen and Maxim Raginsky. Theoretical guarantees for sampling and inference in generative models with latent diffusions. arXiv:1903.01608, 2019. Cited on page 12.

Masatoshi Uehara, Yulai Zhao, Tommaso Biancalani, and Sergey Levine. Understanding reinforcement learning-based fine-tuning of diffusion models: A tutorial and review. arXiv preprint arXiv:2407.13734, 2024a. Cited on page 2.

Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M Tseng, Tommaso Biancalani, and Sergey Levine. Fine-tuning of continuous-time diffusion models as entropyregularized control. arXiv preprint arXiv:2402.15194, 2024b. Cited on pages 2, 6, 12, and 37.

Francisco Vargas, Andrius Ovsianas, David Lopes Fernandes, Mark Girolami, Neil D Lawrence, and Nikolas Nüsken. Bayesian learning via neural schrödinger-föllmer flows. In Fourth Symposium on Advances in Approximate Bayesian Inference, 2022. Cited on page 12.

Francisco Vargas, Will Sussman Grathwohl, and Arnaud Doucet. Denoising diffusion samplers. In The Eleventh International Conference on Learning Representations, 2023. Cited on page 12.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7): 1661–1674, 2011. Cited on page 12.

Apoorv Vyas, Bowen Shi, Matthew Le, Andros Tjandra, Yi-Chiao Wu, Baishan Guo, Jiemin Zhang, Xinyue Zhang, Robert Adkins, William Ngan, et al. Audiobox: Unified audio generation with natural language prompts. arXiv preprint arXiv:2312.15821, 2023. Cited on page 2.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023a. Cited on pages 2, 14, 22, and 52.

Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. arXiv preprint arXiv:2303.13703, 2023b. Cited on page 12.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. In Advances in Neural Information Processing Systems, volume 36, pages 31372–31403. Curran Associates, Inc., 2023a. Cited on pages 12 and 13.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023b. Cited on pages 14 and 54.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023c. Cited on page 2.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. Cited on pages 2, 12, 13, 14, 22, and 51.

Dinghuai Zhang, Yizhe Zhang, Jiatao Gu, Ruixiang Zhang, Josh Susskind, Navdeep Jaitly, and Shuangfei Zhai. Improving gflownets for text-to-image diffusion alignment. arXiv preprint arXiv:2406.00633, 2024. Cited on page 12.

Qinsheng Zhang and Yongxin Chen. Path integral sampler: A stochastic control approach for sampling. In International Conference on Learning Representations, 2022. Cited on page 12.   
Wei Zhang, Han Wang, Carsten Hartmann, Marcus Weber, and Christof Schütte. Applications of the cross-entropy method to importance sampling and optimal control of diffusions. SIAM Journal on Scientific Computing, 36(6): A2654–A2672, 2014. Cited on page 9.   
Qinqing Zheng, Matt Le, Neta Shaul, Yaron Lipman, Aditya Grover, and Ricky T. Q. Chen. Guided flows for generative modeling and decision making. arXiv preprint arXiv:2311.13443, 2023. Cited on pages 2 and 15.   
Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pages 1433–1438. Chicago, IL, USA, 2008. Cited on pages 5 and 37.   
Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2020. Cited on pages 2 and 12.

# Appendix

# Contents

# A Additional Figures & Tables 23

B Results on DDIM and Flow Matching 31   
B.1 The continuous-time limit of DDIM . 31   
B.2 Forward and backward stochastic differential equations 3 1   
B.2.1 Proof of Lemma 1 3 3   
B.2.2 Proof of Lemma 2 3 3   
B.2.3 Proof of Proposition 4 3 4   
B.3 The relationship between the noise predictor $\epsilon$ and the score function 3 6   
B.4 The relationship between the vector field $v$ and the score function . 36

# C Stochastic optimal control as maximum entropy RL in continuous space and time 37

C.1 Maximum entropy RL 37   
C.2 From maximum entropy RL to stochastic optimal control 3 8   
C.3 Proof of Proposition 5: from MaxEnt RL to SOC 39   
C.4 Proof of equation (18): the control cost is a KL regularizer 4 1   
Proofs of Section 4.3: memoryless noise schedule and fine-tuning recipe 42   
D.1 Proof of Proposition 1: the memoryless noise schedule 4 2   
D.2 Proof of Theorem 1: fine-tuning recipe for general noise schedules 4 3

# E Loss function derivations

E.1 Derivation of the Continuous Adjoint method 4 6   
E.2 Proof of Proposition 2: Theoretical guarantees of the basic Adjoint Matching loss 48   
E.3 Theoretical guarantees of the Adjoint Matching loss 4 9   
E.4 Pseudo-code of Adjoint Matching for DDIM fine-tuning 50

# F Adapting diffusion fine-tuning baselines to flow matching 51

F.1 Adapting ReFL (Xu et al., 2023) to flow matching 51   
F.2 Adapting Diffusion-DPO (Wallace et al., 2023a) to flow matching 5 2

# G Experimental details

G.1 Noise schedule details 53   
G.2 Selection of gradient evaluation timesteps 54   
G.3 Loss function clipping: the LCT hyperparameter 54   
G.4 Computation of evaluation metrics 5 4   
G.5 Remarks on computational costs 55   
G.6 Remarks on number of sampling timesteps 5 5

# A Additional Figures & Tables

![](images/figures/adjoint-matching-fig-0007.jpg)  
Figure 6 Average values of ImageReward (reward function), control cost $\begin{array} { r l } {  { \big ( \int _ { 0 } ^ { t } \frac 1 2 \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t \big ) } \quad } & { { } } \end{array}$ , and ClipScore vs. wall-clock time for Adjoint Matching and our baselines. Lines show averages over three fine-tuning runs, evaluating on separate test datasets of size 200. Confidence intervals show standard errors of estimates.

![](images/figures/adjoint-matching-fig-0008.jpg)  
Text prompt: “Man sitting on sofa at home in front of fireplace and using laptop computer, rear view ”

![](images/figures/adjoint-matching-fig-0009.jpg)  
Text prompt: “3D World Food Day Morocco”

Figure 7 Generated samples from varying classifier-free guidance weights, from the pre-trained Flow Matching model.   
Corresponding samples from the fine-tuned model can be found in Figure 4.

Table 3 Metrics for various fine-tuning methods for text-to-image generation. The second and third columns show the√ noise schedules $\sigma ( t )$ used for fine-tuning and for inference: $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ corresponds to Memoryless Flow Matching, and $\sigma ( t ) = 0$ to the Flow Matching ODE (3). Confidence intervals show standard errors of estimates; computed over 3 runs of the fine-tuning algorithm on separate fine-tuning prompt datasets of size 40000 each. Test prompt sets are of size 1000, and also different for each run.   

<table><tr><td>Fine-tuning loss</td><td>Fine-tuning σ(t)</td><td>Sampling σ(t)</td><td>ImageReward ↑</td><td>ClipScore diversity ↑</td><td>PickScore diversity ↑</td><td>Total time (s)/ # iterations</td></tr><tr><td>None (CFG = 1.0)</td><td>N/A</td><td>√2t 0</td><td>−1.384±0.040 −0.920±0.042</td><td>28.07±1.40 30.29±1.53</td><td>1.63±0.08 1.82±0.09</td><td rowspan="2">N/A</td></tr><tr><td>DRaFT-1</td><td>√2t 0</td><td>√2t 0</td><td>1.357±0.039 1.251±0.040</td><td>16.86±0.98 16.76±1.06</td><td>1.21±0.07 1.27±0.07</td></tr><tr><td>DRaFT-40</td><td>√2t</td><td>√2t</td><td>−0.560±0.138 0.424±0.042</td><td>24.07±1.37</td><td>1.64±0.12</td><td>/4000 148k±4.2k</td></tr><tr><td>DPO</td><td>0 √2nt</td><td>0 √2nt</td><td>−1.386±0.033</td><td>20.99±1.54 27.80±1.40</td><td>1.67±0.08 1.62±0.08</td><td rowspan="2">/1500 118k±0.6k / 1000</td></tr><tr><td></td><td>0 √2ηt</td><td>0 √2nt</td><td>−0.957±0.040 0.687±0.085</td><td>29.81±1.43 19.49±1.76</td><td>1.68±0.10 1.22±0.08</td></tr><tr><td>ReFL Cont. Adjoint</td><td>0</td><td>0 √2ηt</td><td>0.709±0.080 −0.448±0.135</td><td>18.39±1.11 26.97±1.37</td><td>1.31±0.10 1.82±0.09</td><td>173k±10.9k /6000 153k±0.9k</td></tr><tr><td>λ = 12500 Disc. Adjoint</td><td>√2t</td><td>0</td><td>−0.249±0.116</td><td>26.25±1.30</td><td>1.90±0.10</td><td rowspan="2">/750 152k±1.5k /1000</td></tr><tr><td>λ = 12500</td><td>√2t</td><td>√2t 0</td><td>−0.557±0.113 −0.552±0.041</td><td>30.40±2.39 28.37±2.26</td><td>1.91±0.09 1.97±0.09</td></tr><tr><td>Adj.-Matching λ = 1000</td><td>√2t</td><td>√2t 0</td><td>0.550±0.043 0.454±0.055</td><td>23.00±1.27 22.76±1.40</td><td>1.65±0.08 1.73±0.09</td><td rowspan="5">156k±1.9k /1000</td></tr><tr><td rowspan="2">Adj.-Matching λ = 2500</td><td rowspan="2">√2t</td><td>√2t</td><td>0.755±0.040</td><td>21.33±1.71</td><td></td></tr><tr><td>0</td><td>0.671±0.047</td><td>21.42±1.54</td><td>1.55±0.08 1.64±0.08</td></tr><tr><td>Adj.-Matching</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>λ = 12500</td><td>√2ηt</td><td>√2t 0</td><td>0.882±0.058 0.778±0.050</td><td>20.49±1.48 20.34±1.49</td><td>1.50±0.09 1.57±0.09</td><td></td></tr></table>

<table><tr><td>Fine-tun. loss</td><td>Fine-tun. σ(t)</td><td>Generat. σ(t)</td><td>ImageReward↑</td><td>ClipScore ↑</td><td>PickScore ↑</td><td>HPS v2↑</td><td>DreamSim diversity ↑</td><td>Runtime/ #iter.</td></tr><tr><td rowspan="2">ReFL</td><td rowspan="2">√2t 0</td><td>√2t</td><td>0.459±0.096</td><td>28.46±0.25</td><td>18.77±0.09</td><td>22.54±0.17</td><td>37.51±3.50</td><td>43k±2.7k</td></tr><tr><td>0</td><td>0.330±0.114</td><td>29.63±0.61</td><td>19.08±0.18</td><td>22.46±0.77</td><td>39.51±1.30</td><td>/1500</td></tr><tr><td rowspan="2">DRaFT-1</td><td rowspan="2">√2nt 0</td><td>√snt</td><td>0.913±0.068</td><td>29.80±0.22</td><td>19.16±0.06</td><td>23.63±0.16</td><td>35.21±1.93</td><td>35k±1.5k</td></tr><tr><td>0</td><td>0.626±0.195</td><td>30.48±0.32</td><td>18.91±0.34</td><td>21.92±1.63</td><td>38.52±2.01</td><td>/1000</td></tr><tr><td rowspan="2">Draft-40</td><td rowspan="2">√2t 0</td><td>√2t</td><td>−1.427±0.267</td><td>23.39±1.72</td><td>17.24±0.45</td><td>15.72±1.80</td><td>41.98±2.14</td><td>49k±1.4k</td></tr><tr><td>0</td><td>−0.097±0.052</td><td>29.12±0.41</td><td>18.97±0.14</td><td>21.93±0.20</td><td>46.35±1.34</td><td>/500</td></tr><tr><td rowspan="2">Adj.-Match. λ = 1000</td><td rowspan="2">√2nt</td><td>√2ηt</td><td>0.107±0.046</td><td></td><td>19.05±0.07</td><td></td><td></td><td></td></tr><tr><td></td><td>0.051±0.044</td><td>29.37±0.25 30.58±0.17</td><td>19.31±0.07</td><td>22.79±0.20 21.93±0.23</td><td>46.38±1.36 48.12±1.56</td><td></td></tr><tr><td rowspan="2">Adj.-Match.</td><td rowspan="2">√2t</td><td>0 √2nt</td><td></td><td></td><td></td><td></td><td></td><td>39k±0.5k</td></tr><tr><td>0</td><td>0.199±0.068 0.106±0.067</td><td>29.27±0.21 30.43±0.24</td><td>19.07±0.10 19.32±0.11</td><td>22.98±0.30 22.16±0.33</td><td>45.03±1.61</td><td>/250</td></tr><tr><td rowspan="2">Adj.-Match.</td><td rowspan="2">√2t</td><td>√2t</td><td></td><td></td><td></td><td></td><td>47.61±1.49</td><td></td></tr><tr><td>0</td><td>0.299±0.095 0.224±0.051</td><td>29.61±0.37 30.70±0.23</td><td>19.26±0.14</td><td>23.67±0.27</td><td>43.36±1.93</td><td></td></tr><tr><td rowspan="2">λ = 12500 Cont. Adj.</td><td rowspan="2">√2t</td><td></td><td></td><td></td><td>19.52±0.11</td><td>22.93±0.21</td><td>44.62±1.79</td><td></td></tr><tr><td>√2t 0</td><td>−0.910±0.116</td><td>26.29±0.44</td><td>18.06±0.16</td><td>18.86±0.88</td><td>51.60±1.97</td><td>51k±0.3k</td></tr><tr><td rowspan="2">λ = 12500 Disc. Adj.</td><td rowspan="2">√2nt</td><td></td><td>−0.681±0.051</td><td>28.50±0.19</td><td>18.69±0.11</td><td>19.90±0.50</td><td>50.87±1.52</td><td>/250</td></tr><tr><td>√2nt 0</td><td>−0.978±0.123 −0.791±0.065</td><td>26.68±0.76 28.66±0.33</td><td>18.51±0.11 18.51±0.11</td><td>18.53±0.28 18.53±0.28</td><td>55.95±1.70 54.78±2.00</td><td>38k±0.4k /250</td></tr></table>

Table 4 Additional metrics for various fine-tuning methods for text-to-image generation, which complement the ones in Table 2 (both tables correspond to the same runs). The second and third columns show the noise schedules $\sigma ( t )$ used for fine-tuning and for inference: $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ corresponds to Memoryless Flow Matching, and $\sigma ( t ) = 0$ to the Flow Matching ODE (3).

Table 5 Evaluation metrics when using classifier-free guidance (CFG; Ho and Salimans (2022)).   

<table><tr><td>w</td><td>Fine-tuning loss</td><td>#iter. /λ</td><td>Fine-tun. σ(t)</td><td>Sampl. σ(t)</td><td>ImageReward ↑</td><td>ClipScore↑</td><td>PickScore ↑</td><td>HPS v2 ↑</td><td>DreamSim diversity↑</td></tr><tr><td>0.0</td><td>None</td><td>N/A</td><td>N/A</td><td>√2ηt 0</td><td>−1.384±0.040 −0.920±0.042</td><td>24.15±0.26 28.32±0.22</td><td>17.25±0.06 18.15±0.07</td><td>16.19±0.17 17.89±0.16</td><td>53.60±1.37 56.53±1.52</td></tr><tr><td></td><td></td><td>1000</td><td>√2t 0</td><td>√2t 0</td><td>0.913±0.068 0.626±0.195</td><td>29.80±0.22 30.48±0.32</td><td>19.16±0.06 18.91±0.34</td><td>23.63±0.16 21.92±1.63</td><td>35.21±1.93 38.52±2.01</td></tr><tr><td>0.0</td><td>DRaFT-1</td><td>2000</td><td>√s 0</td><td>√2t 0</td><td>1.204±0.046 1.052±0.088</td><td>29.90±0.43 30.65±0.24</td><td>19.29±0.12 19.27±0.11</td><td>24.40±0.27 23.81±0.44</td><td>28.51±1.68 32.11±2.37</td></tr><tr><td></td><td></td><td>3000</td><td>$√snt 0</td><td>√s 0</td><td>1.307±0.041 1.173±0.058</td><td>29.96±0.22 30.86±0.25</td><td>19.31±0.06 19.37±0.06</td><td>24.42±0.13 24.17±0.23</td><td>26.57±1.32 29.69±1.30</td></tr><tr><td></td><td></td><td>4000</td><td>√2t 0</td><td>√snt 0</td><td>1.357±0.039 1.251±0.040</td><td>30.18±0.24 30.95±0.28</td><td>19.38±0.08 19.37±0.06</td><td>24.61±0.17 24.37±0.17</td><td>25.54±0.99 27.39±1.14</td></tr><tr><td></td><td></td><td>1000</td><td>√st 0</td><td>√2t 0</td><td>0.550±0.043 0.454±0.055</td><td>30.36±0.22 31.41±0.22</td><td>19.29±0.08 19.57±0.09</td><td>24.12±0.17 23.29±0.18</td><td>40.89±1.50 43.10±1.76</td></tr><tr><td>0.0</td><td>Adj.-Match.</td><td>2500</td><td>√2ηt 0</td><td>√2nt 0</td><td>0.755±0.040 0.671±0.047</td><td>30.59±0.40 31.64±0.21</td><td>19.49±0.10 19.71±0.09</td><td>24.85±0.23 24.12±0.27</td><td>37.07±1.47 39.88±1.59</td></tr><tr><td></td><td></td><td>12500</td><td>$√st 0</td><td>√t 0</td><td>0.882±0.058 0.778±0.050</td><td>30.62±0.30 31.65±0.19</td><td>19.50±0.09 19.76±0.08</td><td>24.95±0.28 24.49±0.27</td><td>34.50±1.33 37.24±1.57</td></tr><tr><td>1.0</td><td>None</td><td>N/A</td><td>N/A</td><td>√s 0</td><td>−0.269±0.050 −0.123±0.041</td><td>30.41±0.22 31.83±0.17</td><td>18.74±0.07 19.28±0.07</td><td>20.47±0.18 20.95±0.16</td><td>43.82±1.24 42.59±1.23</td></tr><tr><td></td><td></td><td>1000</td><td>√2nt</td><td>√2ηt</td><td>1.123±0.051</td><td>32.06±0.19</td><td>19.69±0.06</td><td>24.56±0.17</td><td>28.25±1.55</td></tr><tr><td>1.0</td><td>DRaFT-1</td><td>2000</td><td>0 0</td><td>0 0</td><td>0.856±0.167 1.177±0.053</td><td>32.32±0.25 32.36±0.18</td><td>19.38±0.34 19.67±0.08</td><td>22.88±1.54 24.48±0.28</td><td>29.98±1.86 25.09±1.82</td></tr><tr><td></td><td></td><td>3000</td><td>0</td><td>0</td><td>1.255±0.038</td><td>32.36±0.19</td><td>19.70±0.06</td><td>24.64±0.17</td><td>23.24±1.19</td></tr><tr><td></td><td></td><td>4000</td><td>0</td><td>0</td><td>1.296±0.033</td><td>32.30±0.19</td><td>19.68±0.06</td><td>24.71±0.14</td><td>21.54±0.96</td></tr><tr><td></td><td></td><td>1000</td><td>0</td><td>0</td><td>0.782±0.044</td><td>33.05±0.22</td><td>20.20±0.09</td><td>24.81±0.18</td><td>32.67±1.26</td></tr><tr><td>1.0</td><td>Adj.-Match.</td><td>2500</td><td>√st</td><td>√st</td><td>1.027±0.038</td><td>32.85±0.21</td><td>20.08±0.08</td><td>25.88±0.20</td><td>29.83±1.00</td></tr><tr><td></td><td></td><td>12500</td><td>0 0</td><td>0 0</td><td>0.910±0.040 0.985±0.041</td><td>33.20±0.17 33.10±0.18</td><td>20.29±0.09 20.28±0.08</td><td>25.39±0.24 25.61±0.27</td><td>30.34±1.51 28.86±1.37</td></tr><tr><td>4.0</td><td>None</td><td>N/A</td><td>N/A</td><td>√st</td><td>0.277±0.043</td><td>32.68±0.18</td><td>19.50±0.07</td><td>22.29±0.16</td><td>35.12±0.92</td></tr><tr><td></td><td></td><td></td><td>√s2nt</td><td>0 √2t</td><td>0.209±0.046 1.062±0.045</td><td>32.83±0.17 32.29±0.16</td><td>19.79±0.07 19.48±0.06</td><td>22.30±0.17 23.67±0.13</td><td>32.05±1.05 25.03±1.32</td></tr><tr><td>4.0</td><td>DRaFT-1</td><td>1000</td><td>0</td><td>0</td><td>0.604±0.395</td><td>31.80±0.86</td><td>19.09±0.53</td><td>21.69±2.10</td><td>25.92±2.57</td></tr><tr><td></td><td></td><td>2000 3000</td><td>0 0</td><td>0</td><td>1.112±0.046 1.151±0.036</td><td>32.29±0.20 32.31±0.21</td><td>19.34±0.11 19.36±0.06</td><td>23.31±0.22 23.29±0.14</td><td>21.02±1.67 19.53±1.24</td></tr><tr><td></td><td></td><td>4000</td><td>0</td><td>0 0</td><td>1.172±0.040</td><td>32.20±0.22</td><td>19.30±0.07</td><td>23.20±0.15</td><td>18.45±1.06</td></tr><tr><td></td><td></td><td>1000</td><td>0</td><td>0</td><td>0.852±0.046</td><td>33.50±0.22</td><td>20.31±0.08</td><td>24.97±0.19</td><td>25.83±0.82</td></tr><tr><td>4.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Adj.-Match.</td><td>2500</td><td>√2ηt</td><td>√2t</td><td>1.052±0.039</td><td>33.51±0.19</td><td>20.15±0.07</td><td>25.56±0.18</td><td>26.21±0.73</td></tr><tr><td></td><td></td><td></td><td>0</td><td>0</td><td>0.942±0.042 1.007±0.052</td><td>33.61±0.19 33.48±0.20</td><td>20.35±0.08 20.29±0.08</td><td>25.34±0.21 25.50±0.29</td><td>24.30±0.86</td></tr></table>

Table 6 Metrics for alternative optimization hyperparameters (learning rate and Adam $\beta _ { 1 }$ ).   

<table><tr><td>LR/ Adam β1</td><td>Fine-tuning loss</td><td>Fine-tun. σ(t)</td><td>Generat. σ(t)</td><td>ImageReward↑</td><td>ClipScore ↑</td><td>PickScore ↑</td><td>HPS v2 ↑</td><td>DreamSim diversity↑</td></tr><tr><td>3 × 10−5</td><td>DRaFT-1</td><td>√2t</td><td>√2nt</td><td>1.467±0.029</td><td>30.28±0.56</td><td>19.37±0.09</td><td>24.70±0.15</td><td>21.20±0.93</td></tr><tr><td>/ 0.97</td><td>Adj.-Match. λ = 1200</td><td>√q2tt</td><td>√2t</td><td>1.130±0.034</td><td>31.01±0.27</td><td>19.60±0.08</td><td>25.01±0.25</td><td>26.73±0.88</td></tr><tr><td>2 × 10−5</td><td>Disc. Adj.</td><td>√st</td><td>√2nt</td><td>−1.186±0.553</td><td>21.95±4.29</td><td>16.94±0.95</td><td>12.34±4.40</td><td>28.33±10.26</td></tr><tr><td>/ 0.95</td><td>λ = 12500</td><td>0</td><td>0</td><td>−0.961±0.653</td><td>24.07±4.71</td><td>17.86±1.17</td><td>15.93±5.80</td><td>33.62±7.80</td></tr></table>

Table 7 Comparison with an alternative fine-tuning noise schedule $\sigma ( t ) = 1$ . We see that the initial value function bias (Section 4.2) results in the model not having a high reward function (ImageReward is the reward function used for fine-tuning). Its performance on other metrics are also lower than when fine-tuning with the memoryless noise schedule, except for diversity.   

<table><tr><td>Fine-tuning loss</td><td>Fine-tuning σ(t)</td><td>Generative σ(t)</td><td>ImageReward ↑</td><td>ClipScore ↑</td><td>PickScore ↑</td><td>HPS v2↑</td><td>DreamSim diversity ↑</td></tr><tr><td rowspan="2">Adj.-Matching λ = 12500</td><td rowspan="2">1</td><td>1</td><td>0.009±0.077</td><td>29.18±0.51</td><td>18.66±0.09</td><td>20.75±0.32</td><td>41.33±1.24</td></tr><tr><td>0</td><td>0.454±0.055</td><td>31.41±0.22</td><td>19.57±0.09</td><td>23.29±0.18</td><td>43.10±1.76</td></tr><tr><td rowspan="2">Adj.-Matching λ = 1200</td><td rowspan="2">√st</td><td>√2t</td><td>0.882±0.058</td><td>30.62±0.30</td><td>19.50±0.09</td><td>24.95±0.28</td><td>34.50±1.33</td></tr><tr><td>0</td><td>0.778±0.050</td><td>31.65±0.19</td><td>19.76±0.08</td><td>24.49±0.27</td><td>37.24±1.57</td></tr></table>

<table><tr><td>#sampl. timesteps</td><td>Fine-tuning loss</td><td>Fine-tun. σ(t)</td><td>Sampl. σ(t)</td><td>ImageReward↑</td><td>ClipScore↑</td><td>PickScore ↑</td><td>HPS v2↑</td><td>DreamSim diversity↑</td></tr><tr><td rowspan="6">10</td><td rowspan="2">None (Base)</td><td rowspan="2">N/A</td><td>√2t</td><td>−2.279±0.001</td><td>13.99±0.12</td><td>14.98±0.05</td><td>7.37±0.10</td><td>5.07±0.13</td></tr><tr><td>0</td><td>−1.386±0.040</td><td>26.26±0.24</td><td>17.64±0.07</td><td>14.92±0.17</td><td>51.26±1.38</td></tr><tr><td rowspan="2">DRaFT-1</td><td rowspan="2">√2nt</td><td>√2nt</td><td>1.033±0.051</td><td>25.98±0.25</td><td>18.28±0.07</td><td>22.08±0.18</td><td>14.47±0.67</td></tr><tr><td>0</td><td>1.236±0.038</td><td>31.54±0.27</td><td>19.53±0.07</td><td>24.47±0.19</td><td>24.78±0.88</td></tr><tr><td rowspan="2">Adj.-Match. λ = 12500</td><td rowspan="2">√2nt</td><td>√2t</td><td>−2.104±0.074</td><td>17.12±0.56</td><td>15.76±0.20</td><td>11.48±1.03</td><td>9.88±0.81</td></tr><tr><td>0</td><td>0.607±0.055</td><td>31.36±0.20</td><td>19.56±0.08</td><td>23.23±0.28</td><td>33.75±1.48</td></tr><tr><td rowspan="6">20</td><td rowspan="2">None (Base)</td><td rowspan="2">N/A</td><td>√2t</td><td>−2.275±0.002</td><td>14.58±0.13</td><td>15.07±0.05</td><td>7.47±0.10</td><td>11.27±0.33</td></tr><tr><td>0</td><td>−1.017±0.055</td><td>27.92±0.19</td><td>18.01±0.07</td><td>17.17±0.15</td><td>54.69±1.45</td></tr><tr><td rowspan="2">DRaFT-1</td><td rowspan="2">√2ηt</td><td>√2t</td><td>1.301±0.039</td><td>27.09±0.24</td><td>18.93±0.07</td><td>23.78±0.20</td><td>21.05±1.12</td></tr><tr><td>0</td><td>1.255±0.038</td><td>31.14±0.25</td><td>19.43±0.06</td><td>24.52±0.16</td><td>26.15±1.11</td></tr><tr><td rowspan="2">Adj.-Match.</td><td rowspan="2">√2nt</td><td></td><td>−0.032±0.072</td><td>25.07±0.27</td><td>18.01±0.07</td><td>20.75±0.23</td><td>29.06±2.34</td></tr><tr><td>√2t 0</td><td>0.768±0.048</td><td>31.70±0.17</td><td>19.73±0.08</td><td>24.30±0.26</td><td>35.90±1.52</td></tr><tr><td rowspan="6">40</td><td rowspan="2">None (Base)</td><td rowspan="2">N/A</td><td>√2t</td><td>−1.384±0.040</td><td>24.15±0.26</td><td>17.25±0.06</td><td>16.19±0.17</td><td>53.60±1.37</td></tr><tr><td>0</td><td>−0.920±0.042</td><td>28.32±0.22</td><td>18.15±0.07</td><td>17.89±0.16</td><td>56.53±1.52</td></tr><tr><td rowspan="2">DRaFT-1</td><td rowspan="2">√2nt</td><td>√2t</td><td>1.357±0.039</td><td>30.18±0.24</td><td>19.38±0.08</td><td>24.61±0.17</td><td>25.54±0.99</td></tr><tr><td>0</td><td>1.251±0.040</td><td>30.95±0.28</td><td>19.37±0.06</td><td>24.37±0.17</td><td>27.39±1.14</td></tr><tr><td rowspan="2">Adj.-Match.</td><td rowspan="2">√2ηt</td><td>√2t</td><td>0.882±0.058</td><td>30.62±0.30</td><td>19.50±0.09</td><td>24.95±0.28</td><td>34.50±1.33</td></tr><tr><td>0</td><td>0.778±0.050</td><td>31.65±0.19</td><td>19.76±0.08</td><td>24.49±0.27</td><td>37.24±1.57</td></tr><tr><td rowspan="6">100</td><td rowspan="2">None (Base)</td><td rowspan="2">N/A</td><td>√</td><td>−0.881±0.041</td><td>27.83±0.19</td><td>18.10±0.07</td><td>18.43±0.17</td><td>57.21±1.50</td></tr><tr><td>0</td><td>−0.881±0.036</td><td>28.65±0.18</td><td>18.22±0.06</td><td>18.20±0.17</td><td>57.73±1.68</td></tr><tr><td rowspan="2">DRaFT-1</td><td rowspan="2">√2nt</td><td>√2t</td><td>1.343±0.040</td><td>30.64±0.20</td><td>19.38±0.08</td><td>24.37±0.17</td><td>25.51±1.10</td></tr><tr><td>0</td><td>1.239±0.037</td><td>30.74±0.28</td><td>19.33±0.06</td><td>24.24±0.17</td><td>28.70±1.11</td></tr><tr><td rowspan="2">Adj.-Match. λ = 12500</td><td rowspan="2">√2t</td><td>√2t</td><td>0.892±0.044</td><td>31.23±0.23</td><td>19.65±0.08</td><td>24.92±0.23</td><td>35.13±1.40</td></tr><tr><td>0</td><td>0.779±0.048</td><td>31.64±0.17</td><td>19.76±0.08</td><td>24.57±0.25</td><td>38.26±1.65</td></tr><tr><td rowspan="6">200</td><td rowspan="2">None (Base)</td><td rowspan="2">N/A</td><td>√2t</td><td>−0.848±0.048</td><td>28.37±0.21</td><td>18.27±0.08</td><td>18.56±0.19</td><td>58.00±1.58</td></tr><tr><td>0</td><td>−0.871±0.036</td><td>28.50±0.18</td><td>18.23±0.06</td><td>18.25±0.14</td><td>57.84±1.60</td></tr><tr><td rowspan="2">DRaFT-1</td><td rowspan="2">√2t</td><td></td><td>1.331±0.044</td><td>30.69±0.23</td><td>19.36±0.07</td><td>24.21±0.17</td><td>26.41±1.18</td></tr><tr><td>√2nt 0</td><td>1.222±0.042</td><td>30.77±0.27</td><td>19.32±0.06</td><td>24.18±0.16</td><td>29.09±1.07</td></tr><tr><td rowspan="2">Adj.-Match.</td><td rowspan="2">0 √2nt</td><td>√st</td><td>0.869±0.062</td><td>31.33±0.21</td><td>19.68±0.09</td><td>24.81±0.30</td><td>35.90±1.55</td></tr><tr><td>0</td><td>0.766±0.050</td><td>31.61±0.16</td><td>19.75±0.08</td><td>24.52±0.24</td><td>38.60±1.38</td></tr></table>

Table 8 Performance metrics for different number of sampling steps. Only the number of sampling steps is ablated; the fine-tuned models used in all cases are the ones fine-tuned using 40 steps.

![](images/figures/adjoint-matching-fig-0010.jpg)  
Adjoint Matching (Ours)   
Figure 8 Generated samples with classifier-free guidance $w = 1$ ) and $\sigma ( t ) = 0$ across ten selected prompts. Each row corresponds to a different prompt and each image corresponds to a different random seed consistent across models.

![](images/figures/adjoint-matching-fig-0011.jpg)  
Adjoint Matching (Ours)   
Figure 9 Generated samples with classifier-free guidance ( $w = 1$ ) and $\sigma ( t ) = 0$ across ten selected prompts with people. Each row corresponds to a different prompt and each image corresponds to a different random seed consistent across models.

![](images/figures/adjoint-matching-fig-0012.jpg)  
Figure 10 Generated samples without guidance ( $w = 0$ ) and $\sigma ( t ) = 0$ across seven selected prompts. Each row corresponds to a different finetuning algorithm. Prompts: “ Seaside view poster with palm trees vector image”, “Cayucos Beach Inn”, “Happy Summer Life- Aloha Flowers and Melon - Pattern Metal Print”, “Castle Square, Warsaw Old Town”, “Funny girl blowing soap bubbles. High quality photo”, “Colombian man with sweatshirt over yellow wall listening to something by putting hand on the ear ”, “man in the hood black mask masquerade”.

![](images/figures/adjoint-matching-fig-0013.jpg)  
Figure 11 Generated samples without guidance ( $w = 0$ ) and $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ across seven selected prompts. Each row corresponds to a different finetuning algorithm. The prompts are the same as in Figure 10.

# B Results on DDIM and Flow Matching

# B.1 The continuous-time limit of DDIM

The DDIM inference update (Song et al., 2021a, Eq. 12) is

$$
\begin{array} { r } { x _ { k + 1 } = \sqrt { \bar { \alpha } _ { k + 1 } } \Big ( \frac { x _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon ( x _ { k } , k ) } { \sqrt { \bar { \alpha } _ { k } } } \Big ) + \sqrt { 1 - \bar { \alpha } _ { k + 1 } - \sigma _ { k } ^ { 2 } } \epsilon ( x _ { k } , k ) + \sigma _ { k } \epsilon _ { k } , \qquad x _ { K } \sim N ( 0 , I ) . } \end{array}
$$

If we let $\Delta \bar { \alpha } _ { k } = \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k }$ , we have that

$$
\begin{array} { r } { \sqrt { \frac { \bar { \alpha } _ { k + 1 } } { \bar { \alpha } _ { k } } } = \sqrt { \frac { \bar { \alpha } _ { k } + \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } } = \sqrt { 1 + \frac { \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } } = \sqrt { 1 + \frac { \Delta \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } } \approx 1 + \frac { \Delta \bar { \alpha } _ { k } } { 2 \bar { \alpha } _ { k } } , } \end{array}
$$

where we used the first-order Taylor approximation of $\sqrt { 1 + x }$ . And

$$
\begin{array} { r l } & { - \sqrt { \frac { \bar { \alpha } _ { k + 1 } } { \bar { \alpha } _ { k } } \big ( 1 - \bar { \alpha } _ { k } \big ) } + \sqrt { 1 - \bar { \alpha } _ { k + 1 } - \sigma _ { k } ^ { 2 } } = - \sqrt { \big ( 1 + \frac { \Delta \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } \big ) \big ( 1 - \bar { \alpha } _ { k } \big ) } + \sqrt { 1 - \bar { \alpha } _ { k + 1 } - \sigma _ { k } ^ { 2 } } } \\ & { = - \sqrt { 1 + \frac { \Delta \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } - \bar { \alpha } _ { k } - \Delta \bar { \alpha } _ { k } } + \sqrt { 1 - \bar { \alpha } _ { k + 1 } - \sigma _ { k } ^ { 2 } } = - \sqrt { 1 - \bar { \alpha } _ { k + 1 } + \frac { \Delta \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } } + \sqrt { 1 - \bar { \alpha } _ { k + 1 } - \sigma _ { k } ^ { 2 } } } \\ & { = \sqrt { 1 - \bar { \alpha } _ { k + 1 } } \big ( - \sqrt { 1 + \frac { \Delta \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } \big ( 1 - \bar { \alpha } _ { k + 1 } \big ) } } + \sqrt { 1 - \frac { \sigma _ { k } ^ { 2 } } { 1 - \bar { \alpha } _ { k + 1 } } } \big ) \approx \sqrt { 1 - \bar { \alpha } _ { k + 1 } } \big ( - \big ( 1 + \frac { \Delta \bar { \alpha } _ { k } } { 2 \bar { \alpha } _ { k } \big ( 1 - \bar { \alpha } _ { k + 1 } \big ) } \big ) + 1 - \frac { \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } \big ( 1 - \bar { \alpha } _ { k + 1 } \big ) } } \\ &  = - \big ( \frac { \Delta \bar { \alpha } _ { k } } { 2 \bar { \alpha } _ { k } } + \frac  \sigma _ { k } ^  \end{array}
$$

where we used the same first-order Taylor approximation. Thus, up to first-order approximations, (44) is equivalent to

$$
\begin{array} { r } { x _ { k - 1 } = \big ( 1 + \frac { \Delta \bar { \alpha } _ { k } } { 2 \bar { \alpha } _ { k } } \big ) x _ { k } - \big ( \frac { \Delta \bar { \alpha } _ { k } } { 2 \bar { \alpha } _ { k } } + \frac { \sigma _ { k } ^ { 2 } } { 2 } \big ) \frac { \epsilon ( x _ { k } , k ) } { \sqrt { 1 - \bar { \alpha } _ { k + 1 } } } + \sigma _ { k } \epsilon _ { k } , \qquad x _ { K } \sim N ( 0 , I ) . } \end{array}
$$

If we modify our notation slightly, we can rewrite this as

$$
\begin{array} { r } { X _ { ( k + 1 ) h } = \big ( 1 - \frac { h \bar { \alpha } _ { k h } } { 2 \bar { \alpha } _ { k h } } \big ) X _ { k h } + \big ( \frac { h \bar { \alpha } _ { k h } } { 2 \bar { \alpha } _ { k h } } - \frac { h \sigma ( k h ) ^ { 2 } } { 2 } \big ) \frac { \epsilon ( X _ { k h } , k h ) } { \sqrt { 1 - \bar { \alpha } _ { k h } } } + \sqrt { h } \sigma ( k h ) \epsilon _ { k } , \qquad X _ { 0 } \sim N ( 0 , I ) . } \end{array}
$$

To go from (47) to (48), we introduced a continuous time variable and a stepsize $h = 1 / K$ , and we regard the increment $h \bar { \alpha } _ { k }$ as approximately equal to $h$ times the derivative of $\alpha$ . We also identified $\sigma _ { k }$ with $\sqrt { h } \sigma ( k h )$ , where $\sigma ( k h )$ plays the role of a diffusion coefficient. Note that equation (48) can be reverse-engineered as the Euler-Maruyama discretization of the SDE

$$
\begin{array} { r } { \mathrm { d } X _ { t } = \big ( - \frac { \dot { \bar { \alpha } } _ { t } } { 2 \bar { \alpha } _ { t } } + \big ( \frac { \dot { \bar { \alpha } } _ { t } } { 2 \bar { \alpha } _ { t } } - \frac { \sigma ( t ) ^ { 2 } } { 2 } \big ) \frac { \epsilon ( X _ { t } , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } \big ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim N ( 0 , I ) . } \end{array}
$$

# B.2 Forward and backward stochastic differential equations

Let $\left( \kappa _ { t } \right) _ { t \in [ 0 , 1 ] }$ and $( \eta _ { t } ) _ { t \in [ 0 , 1 ] }$ such that

$$
\begin{array} { r } { t \in [ 0 , 1 ] , \quad \eta _ { t } \geq 0 , \qquad \int _ { 0 } ^ { 1 } \kappa _ { 1 - s } { \mathrm { d } } s = + \infty , \qquad 2 \int _ { 0 } ^ { 1 } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } { \mathrm { d } } s \big ) \mathrm { d } t ^ { \prime } = 1 } \end{array}
$$

As shown in Table 1, DDIM corresponds to $\begin{array} { r } { \kappa _ { t } = \frac { \dot { \bar { \alpha } } _ { t } } { 2 \bar { \alpha } _ { t } } } \end{array}$ = α¯˙ t2 ¯αt , ηt = 2 , and Flow Matching corresponds to $\begin{array} { r } { \kappa _ { t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } } \end{array}$ , $\begin{array} { r } { \eta _ { t } = \beta _ { t } \big ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \big ) } \end{array}$ .

Lemma 1 (DDIM and Flow Matching fulfill the conditions (50)). The choices of $( \kappa _ { t } ) _ { t \in [ 0 , 1 ] }$ and $( \eta _ { t } ) _ { t \in [ 0 , 1 ] }$ for DDIM and Flow Matching fulfill the conditions (50). For DDIM, we have that

$$
\begin{array} { c } { { \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s = - \frac 1 2 \log \bar { \alpha } _ { 1 - t } \implies \int _ { 0 } ^ { 1 } \kappa _ { 1 - s } \mathrm { d } s = + \infty , } } \\ { { 2 \int _ { 0 } ^ { t } \eta _ { t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = 1 - \bar { \alpha } _ { 1 - t } \implies 2 \int _ { 0 } ^ { 1 } \eta _ { t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = 1 . } } \end{array}
$$

For Flow Matching,

$$
\begin{array} { r l } & { \qquad \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s = - \log \alpha _ { 1 - t } \implies \int _ { 0 } ^ { 1 } \kappa _ { 1 - s } \mathrm { d } s = + \infty , } \\ & { 2 \int _ { 0 } ^ { t } \eta _ { t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = \beta _ { 1 - t } ^ { 2 } \implies 2 \int _ { 0 } ^ { 1 } \eta _ { t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = 1 . } \end{array}
$$

Forward and backward SDEs Consider the forward and backward SDEs

$$
\begin{array} { r l } & { \mathrm { d } \vec { X } _ { t } = - \kappa _ { 1 - t } \vec { X } _ { t } \mathrm { d } t + \sqrt { 2 \eta _ { 1 - t } } \mathrm { d } B _ { t } , \qquad \vec { X } _ { 0 } \sim p _ { \mathrm { d a t a } } , } \\ & { \mathrm { d } X _ { t } = \bigl ( \kappa _ { t } X _ { t } + 2 \eta _ { t } \mathfrak { s } ( X _ { t } , t ) \bigr ) \mathrm { d } t + \sqrt { 2 \eta _ { t } } \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim N ( 0 , I ) , } \end{array}
$$

where we let $\vec { p _ { t } }$ be the density of $\vec { X _ { t } }$ , and we define the score function as $\pmb { \mathfrak { s } } ( x , t ) : = \nabla \log \vec { p } _ { 1 - t } ( x )$ . Similarly, we let $p _ { t }$ be the density of $X _ { t }$ . $\vec { p _ { t } }$ and $p _ { t }$ solve the Fokker-Planck equations:

$$
\begin{array} { r l } & { \partial _ { t } \vec { p } _ { t } = \nabla \cdot \bigl ( \kappa _ { 1 - t } x \vec { p } _ { t } \bigr ) + \eta _ { 1 - t } \Delta \vec { p _ { t } } , \qquad \vec { p } _ { 0 } = p _ { \mathrm { d a t a } } , } \\ & { \partial _ { t } p _ { t } = \nabla \cdot \bigl ( \bigl ( - \kappa _ { t } x - 2 \eta _ { t } \nabla \log \vec { p } _ { 1 - t } ( X _ { t } ) \bigr ) p _ { t } \bigr ) + \eta _ { t } \Delta p _ { t } , \qquad p _ { 0 } = N ( 0 , I ) . } \end{array}
$$

Lemma 2 (Solution of the forward SDE). Let $( \kappa _ { t } ) _ { t \geq 0 }$ , $( \eta _ { t } ) _ { t \geq 0 }$ with $\eta _ { t } \geq 0$ , and $( \xi _ { t } ) _ { t \geq 0 }$ be arbitrary. The solution $\vec { X _ { t } }$ of the $S D E$

$$
\mathrm { d } \vec { X } _ { t } = \left( - \kappa _ { 1 - t } \vec { X } _ { t } + \xi _ { t } \right) \mathrm { d } t + \sqrt { 2 \eta _ { 1 - t } } \mathrm { d } B _ { t } , \qquad \vec { X } _ { 0 } \sim p _ { \mathrm { d a t a } }
$$

is

$$
\begin{array} { r } { \vec { X } _ { t } = \vec { X } _ { 0 } \exp \big ( - \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) + \int _ { 0 } ^ { t } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \xi _ { 1 - t ^ { \prime } } \mathrm { d } t ^ { \prime } + \int _ { 0 } ^ { t } \sqrt { 2 \eta _ { 1 - t ^ { \prime } } } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) } \end{array}
$$

which has the same distribution as the random variable

$$
\begin{array} { l } { \dot { \zeta } _ { t } = \vec { X } _ { 0 } \exp \Big ( - \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \Big ) + \int _ { 0 } ^ { t } \exp \Big ( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \Big ) \xi _ { 1 - t ^ { \prime } } \mathrm { d } t ^ { \prime } + \sqrt { 2 \int _ { 0 } ^ { t } \eta _ { 1 - t ^ { \prime } } \exp \Big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \Big ) } } \\ { \epsilon \sim N ( 0 , I ) . } \end{array}
$$

Applying Lemma 2 with $\xi _ { t } \equiv 0$ , we obtain that $\vec { p _ { 1 } }$ is also the distribution of

$$
\begin{array} { r } { \hat { X } _ { 1 } = \vec { X } _ { 0 } \exp \big ( - \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) + \sqrt { 2 \int _ { 0 } ^ { t } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } } \epsilon = \epsilon , } \end{array}
$$

where $\epsilon \sim N ( 0 , I )$ . The third equality in (61) holds by (50). Hence we obtain that $\vec { p _ { 1 } } = N ( 0 , I )$ . Note also that

$$
\begin{array} { r } { \partial _ { t } \vec { p } _ { 1 - t } = - \nabla \cdot \left( \kappa _ { t } x \vec { p } _ { 1 - t } \right) - \eta _ { t } \Delta \vec { p } _ { 1 - t } = - \nabla \cdot \left( \left( - \kappa _ { t } x - 2 \eta _ { t } \nabla \log \vec { p } _ { 1 - t } ( x ) \right) \vec { p } _ { 1 - t } \right) + \eta _ { t } \Delta \vec { p } _ { 1 - t } } \end{array}
$$

Thus, $\vec { p _ { 1 - t } }$ is a solution of the backward Fokker-Planck equation (57), which proves the following:

Proposition 3 (Equality of marginal distributions). For any time $t \in [ 0 , 1 ]$ , the densities of the solutions $\vec { X _ { t } }$ , $X _ { t }$ of the forward and backward SDEs are equal up to a time flip: $p _ { t } = \vec { p _ { 1 - t } }$ .

Forward and backward SDEs with arbitrary noise schedule Next, we look at the following pair of forwardbackward SDEs:

$$
\begin{array} { r l } & { \mathrm { d } \vec { X } _ { t } = \big ( - \kappa _ { 1 - t } \vec { X } _ { t } + \big ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \big ) \mathfrak { s } ( \vec { X } _ { t } , 1 - t ) \big ) \mathrm { d } t + \sigma ( 1 - t ) \mathrm { d } B _ { t } , \qquad \vec { X } _ { 0 } \sim p _ { \mathrm { d a t a } } , } \\ & { \mathrm { d } X _ { t } = \big ( \kappa _ { t } X _ { t } + \big ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \big ) \mathfrak { s } ( X _ { t } , t ) \big ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim N ( 0 , I ) , } \end{array}
$$

Here, the score function $\mathfrak { s }$ is the same vector field as in (64). Remark that equations (54)-(55) are a particular case of (63)-(64) for which $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ . The Fokker-Planck equations for (63)-(64) are:

$$
\begin{array} { r l } & { \partial _ { t } \vec { p } _ { t } = \nabla \cdot \bigl ( \bigl ( \kappa _ { 1 - t } x + \bigl ( - \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } + \eta _ { 1 - t } \bigr ) \mathfrak { s } ( X _ { t } , t ) \bigr ) \vec { p } _ { t } \bigr ) + \eta _ { 1 - t } \Delta \vec { p } _ { t } , \qquad \vec { p } _ { 0 } = p _ { \mathrm { d a t a } } , } \\ & { \partial _ { t } p _ { t } = \nabla \cdot \bigl ( \bigl ( - \kappa _ { t } x - \bigl ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \bigr ) \mathfrak { s } ( X _ { t } , t ) \bigr ) p _ { t } \bigr ) + \frac { \sigma ( t ) ^ { 2 } } { 2 } \Delta p _ { t } , \qquad p _ { 0 } = N ( 0 , I ) . } \end{array}
$$

It is straight-forward to see that for any $\sigma$ , the solutions $\vec { p _ { t } }$ and $p _ { t }$ of (65)-(66) are also solutions of (56)-(57). Hence, the marginals $\vec { X _ { t } }$ and $X _ { t }$ are equally distributed for all noise schedules $\sigma$ , and they are equal to each other up to a time flip.

Equality of distributions over trajectories The result in Proposition 3 can be made even stronger:

Proposition 4 (Equality of distributions over trajectories). Let $\vec { X }$ , $\pmb { X }$ be the solutions of the SDEs (63)-(64) with arbitrary noise schedule. For any sequence of times $( t _ { i } ) _ { 0 \leq i \leq I }$ , the joint distribution of $( \vec { X } _ { t _ { i } } ) _ { 0 \leq i \leq I }$ is equal to the joint distribution of $( X _ { 1 - t _ { i } } ) _ { 0 \leq i \leq I }$ , or equivalently, that the probability measures $\vec { \mathbb { P } }$ , $\mathbb { P }$ of the forward and backward processes $\vec { X }$ , $\pmb { X }$ are equal, up to a flip in the time direction.

This result states that sampling trajectories from the backward process is equivalent to sampling them from the forward process and then flipping their order.

# B.2.1 Proof of Lemma 1

As shown in Table 1, DDIM corresponds to $\begin{array} { r } { \kappa _ { t } = \frac { \bar { \bar { \alpha } } _ { t } } { 2 \bar { \alpha } _ { t } } } \end{array}$ α¯˙ t2 ¯αt , ηt = 2 ¯αt . Thus, $\eta _ { t } \geq 0$ because $\alpha _ { t }$ is increasing, and

$$
\begin{array} { r l } & { \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s = \int _ { 0 } ^ { t } \frac { \dot { \alpha } _ { 1 - s } } { 2 \tilde { \alpha } _ { 1 - s } } \mathrm { d } s = - \frac { 1 } { 2 } \int _ { 0 } ^ { t } \partial _ { s } \log \bar { \alpha } _ { 1 - s } \mathrm { d } s = - \frac { 1 } { 2 } ( \log \bar { \alpha } _ { 1 - t } - \log \bar { \alpha } _ { 1 } ) = - \frac { 1 } { 2 } \log \bar { \alpha } _ { 1 - t } , } \\ & { \implies \int _ { 0 } ^ { 1 } \kappa _ { 1 - s } \mathrm { d } s = - \frac { 1 } { 2 } \log \bar { \alpha } _ { 0 } = + \infty } \\ & { 2 \int _ { 0 } ^ { t } \eta _ { t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = \int _ { 0 } ^ { t } \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \tilde { \alpha } _ { 1 - t ^ { \prime } } } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \dot { \alpha } _ { 1 - s } } { \tilde { \alpha } _ { 1 - s } } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } } \\ &  = \int _ { 0 } ^ { t } \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \tilde { \alpha } _ { 1 - t ^ { \prime } } } \frac { \ddot { \alpha } _ { 1 - t } } { \tilde { \alpha } _ { 1 - t ^ { \prime } } } \mathrm { d } t ^ { \prime } = \bar { \alpha } _ { 1 - t } \int _ { 0 } ^ { t } \partial _ { t ^ { \prime } } \Big ( \frac { 1 } { \tilde { \alpha } _ { 1 - t ^ { \prime } } } \Big ) \mathrm { d } t ^ { \prime } = \bar { \alpha } _ { 1 - t } \Big ( \frac { 1 } { \tilde { \alpha } _ { 1 - t } } - \frac \end{array}
$$

where we used that $\bar { \alpha } _ { 1 } = 1$ and $\bar { \alpha } _ { 0 } = 0$ . And Flow Matching corresponds to $\begin{array} { r } { \kappa _ { t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } } \end{array}$ , $\begin{array} { r } { \eta _ { t } = \beta _ { t } \big ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \big ) } \end{array}$ . We have that $\eta _ { t } \geq 0$ because $\alpha _ { t }$ is increasing and $\beta _ { t }$ is decreasing, and

$$
\begin{array} { r l } & { \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s = \int _ { 0 } ^ { t } \frac { \dot { \alpha } _ { 1 - s } } { \alpha _ { 1 - s } } \mathrm { d } s = - \int _ { 0 } ^ { t } \partial _ { s } \log \alpha _ { 1 - s } \mathrm { d } s = - ( \log \alpha _ { 1 - t } - \log \alpha _ { 1 } ) = - \log \alpha _ { 1 - t } , } \\ & { \implies \int _ { 0 } ^ { 1 } \kappa _ { 1 - s } \mathrm { d } s = - \log \alpha _ { 0 } = + \infty , } \end{array}
$$

and

$$
\begin{array} { r l } & { 2 \int _ { 0 } ^ { t } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = 2 \int _ { 0 } ^ { t } \beta _ { 1 - t ^ { \prime } } \Big ( \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \beta _ { 1 - t ^ { \prime } } - \dot { \beta } _ { 1 - t ^ { \prime } } \Big ) \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \frac { \dot { \alpha } _ { 1 - s } } { \alpha _ { 1 - s } } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } } \\ & { = 2 \int _ { 0 } ^ { t } \beta _ { 1 - t ^ { \prime } } \Big ( \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \beta _ { 1 - t ^ { \prime } } - \dot { \beta } _ { 1 - t ^ { \prime } } \Big ) \Big ( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \Big ) ^ { 2 } \mathrm { d } t ^ { \prime } , } \end{array}
$$

To develop the right-hand side, note that by integration by parts,

$$
\begin{array} { r l } & { \int _ { 0 } ^ { t } \dot { \beta } _ { 1 - t ^ { \prime } } \beta _ { 1 - t ^ { \prime } } \left( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \right) ^ { 2 } \mathrm { d } t ^ { \prime } = - \int _ { 0 } ^ { t } \partial _ { t ^ { \prime } } \left( \frac { \beta _ { 1 - t ^ { \prime } } ^ { 2 } } { 2 } \right) \left( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \right) ^ { 2 } \mathrm { d } t ^ { \prime } } \\ & { = - \Big [ \frac { \beta _ { 1 - t ^ { \prime } } ^ { 2 } } { 2 } \Big ( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \Big ) ^ { 2 } \Big ] _ { 0 } ^ { 1 } + \int _ { 0 } ^ { t } \frac { \beta _ { 1 - t ^ { \prime } } ^ { 2 } } { 2 } \partial _ { t ^ { \prime } } \Big ( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \Big ) ^ { 2 } \mathrm { d } t ^ { \prime } = - \Big [ \frac { \beta _ { 1 - t ^ { \prime } } ^ { 2 } } { 2 } \Big ( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \Big ) ^ { 2 } \Big ] _ { 0 } ^ { t } + \int _ { 0 } ^ { t } \beta _ { 1 - t ^ { \prime } } ^ { 2 } \frac { \alpha _ { 1 - t } ^ { 2 } \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } ^ { 3 } } \mathrm { d } t ^ { \prime } . } \end{array}
$$

And if we plug this into the right-hand side of (70), we obtain

$$
\begin{array} { r l } & { 2 \int _ { 0 } ^ { t } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = \big [ \beta _ { 1 - t ^ { \prime } } ^ { 2 } \big ( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \big ) ^ { 2 } \big ] _ { 0 } ^ { t } = \beta _ { 1 - t } ^ { 2 } - \beta _ { 1 } ^ { 2 } \big ( \frac { \alpha _ { 1 - t } } { \alpha _ { 1 } } \big ) ^ { 2 } = \beta _ { 1 - t } ^ { 2 } , } \\ & { \implies 2 \int _ { 0 } ^ { 1 } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } = \beta _ { 1 } ^ { 2 } = 1 . } \end{array}
$$

where we used that $\beta _ { 1 } = 0$ , $\alpha _ { 1 } = 1$ .

# B.2.2 Proof of Lemma 2

We can solve this equation by variation of parameters. To simplify the notation, we replace $\kappa _ { 1 - s }$ , $\eta _ { 1 - s }$ and $\xi _ { 1 - s }$ by $\kappa _ { s }$ , $\eta _ { s }$ and $\xi _ { s }$ . Defining $\begin{array} { r } { f ( \vec { X } _ { t } , t ) = \vec { X } _ { t } \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) } \end{array}$ , we get that

$$
\begin{array} { r l } & { d f ( \vec { X } _ { t } , t ) = \kappa _ { 1 - t } \vec { X } _ { t } \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t + \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } \vec { X } _ { t } } \\ & { \qquad = \kappa _ { 1 - t } \vec { X } _ { t } \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t + \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \big ( ( - \kappa _ { 1 - t } \vec { X } _ { t } + \xi _ { 1 - t } ) \mathrm { d } t + \sqrt { 2 \eta _ { 1 - t } } \mathrm { d } E \big ) } \\ & { \qquad = \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \xi _ { 1 - t } \mathrm { d } t + \sqrt { 2 \eta _ { t } } \exp \big ( \int _ { 0 } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } B _ { t } . } \end{array}
$$

Integrating from $0$ to $t$ , we get that

$$
\begin{array} { r l } & { t \exp \left( \int _ { 0 } ^ { t } \kappa _ { 1 - s } { \mathrm { d } } s \right) = \vec { X } _ { 0 } + \int _ { 0 } ^ { t } \exp \left( \int _ { 0 } ^ { t ^ { \prime } } \kappa _ { 1 - s } { \mathrm { d } } s \right) \xi _ { 1 - t ^ { \prime } } { \mathrm { d } } t ^ { \prime } + \int _ { 0 } ^ { t } \sqrt { 2 \eta _ { 1 - t ^ { \prime } } } \exp \left( \int _ { 0 } ^ { t ^ { \prime } } \kappa _ { 1 - s } { \mathrm { d } } s \right) { \mathrm { d } } B _ { t ^ { \prime } } , } \\ & { \Longleftrightarrow \vec { X } _ { t } = \vec { X } _ { 0 } \exp \left( - \int _ { 0 } ^ { t } \kappa _ { 1 - s } { \mathrm { d } } s \right) + \int _ { 0 } ^ { t } \exp \left( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } { \mathrm { d } } s \right) \xi _ { 1 - t ^ { \prime } } { \mathrm { d } } t ^ { \prime } + \int _ { 0 } ^ { t } \sqrt { 2 \eta _ { 1 - t ^ { \prime } } } \exp \left( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } { \mathrm { d } } s \right) { \mathrm { d } } B _ { t ^ { \prime } } . } \end{array}
$$

Since

$$
\begin{array} { r } { \mathbb { E } \Big [ \Big ( \int _ { 0 } ^ { t } \sqrt { 2 \eta _ { 1 - t ^ { \prime } } } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } B _ { t ^ { \prime } } \Big ) ^ { 2 } \Big ] = 2 \int _ { 0 } ^ { t } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } , } \end{array}
$$

we obtain that $\begin{array} { r } { \int _ { 0 } ^ { t } \sqrt { 2 \eta _ { 1 - t ^ { \prime } } } \exp \left( - \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \right) \mathrm { d } B _ { t ^ { \prime } } } \end{array}$ has the same distribution as $\begin{array} { r } { \sqrt { 2 \int _ { 0 } ^ { t } \eta _ { 1 - t ^ { \prime } } \exp \big ( - 2 \int _ { t ^ { \prime } } ^ { t } \kappa _ { 1 - s } \mathrm { d } s \big ) \mathrm { d } t ^ { \prime } } \epsilon . } \end{array}$ where $\epsilon \sim N ( 0 , 1 )$ .

# B.2.3 Proof of Proposition 4

This is a result that has been used by previous works, e.g. (De Bortoli et al., 2021, Sec. 2.1), but their derivation lacks rigor as it uses some unexplained approximations. While natural, the result is not common knowledge in the area. We provide a derivation which is still in discrete time, and hence not completely formal, but that corrects the gaps in the proof of De Bortoli et al. (2021).

We introduce the short-hand

$$
\begin{array} { r l } & { \Vec { b } ( x , t ) = - \kappa _ { 1 - t } x + \big ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \big ) \mathfrak { s } ( x , 1 - t ) , } \\ & { b ( x , t ) = \kappa _ { t } X _ { t } + \big ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \big ) \mathfrak { s } ( X _ { t } , t ) , } \\ & { \Vec { \sigma } ( t ) = \sigma ( 1 - t ) . } \end{array}
$$

Remark that $b ( x , t ) = - \vec { b } ( x , 1 - t ) + \sigma ( t ) ^ { 2 } \mathfrak { s } ( X _ { t } , t ) .$ .

Suppose that we discretize the forward process $\vec { X }$ using $K + 1$ equispaced timesteps:

$$
x _ { k + 1 } = x _ { k } + h \vec { b } ( x _ { k } , k h ) + \sqrt { h } \vec { \sigma } ( k h ) \epsilon _ { k } , \qquad \mathrm { w i t h } \ \epsilon _ { k } \sim N ( 0 , 1 ) .
$$

It is important to remark that $x _ { k + 1 } - x _ { k } = O ( h ^ { 1 / 2 } )$ . Throughout the proof we will keep track of all terms up to linear order in $h$ , while neglecting terms of order ${ \cal O } ( h ^ { 3 / 2 } )$ and higher. The distribution of the discretized forward process is:

$$
\begin{array} { r }  | \mathop { = } \vec { p _ { 0 } } ( x _ { 0 } ) \prod _ { k = 0 } ^ { K - 1 } \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) , \qquad \mathrm { w h e r e } \qquad \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) \overset { \in \mathtt { e p } } { = } \frac { \exp \big ( - \frac { \| x _ { k + 1 } - x _ { k } - h \tilde { \ell } ( x _ { k } , k h ) \| ^ { 2 } } { 2 h \tilde { \sigma } ( k h ) ^ { 2 } } \big ) } { ( 2 \pi h \tilde { \sigma } ( k h ) ^ { 2 } ) ^ { d / 2 } } \sqrt { \frac { | x _ { k + 1 } - x _ { k + 1 } | h } { 2 h \tilde { \sigma } ( k h ) ^ { 2 } } } \end{array}
$$

Using telescoping products, we have that

$$
\begin{array} { r l } & { \vec { p } ( x _ { 0 : K } ) = \vec { p } _ { K } ( x _ { K } ) \prod _ { k = 0 } ^ { K - 1 } \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) \frac { \vec { p } _ { k } ( x _ { k } ) } { \vec { p } _ { k + 1 } ( x _ { k + 1 } ) } } \\ & { \qquad = \vec { p } _ { K } ( x _ { K } ) \prod _ { k = 0 } ^ { K - 1 } \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) \exp \big ( \log ( \vec { p } _ { k } ( x _ { k } ) ) - \log ( \vec { p } _ { k + 1 } ( x _ { k + 1 } ) ) \big ) } \end{array}
$$

We can use a discrete time version of Ito’s lemma:

$$
\begin{array} { r l } & { \log \vec { p } ( x _ { k + 1 } , ( k + 1 ) h ) \approx \log \vec { p } ( x _ { k } , k h ) + h \big ( \partial _ { t } \log \vec { p } ( x _ { k } , k h ) + \frac { \vec { \sigma } ( k h ) ^ { 2 } } { 2 } \Delta \log \vec { p } ( x _ { k } , k h ) \big ) } \\ & { \qquad + \left. \nabla \log \vec { p } ( x _ { k } , k h ) , x _ { k + 1 } - x _ { k } \right. + O ( h ^ { 3 / 2 } ) . } \end{array}
$$

Using equation (81) and a Taylor approximation, observe that

$$
\begin{array} { r l } & { \nabla \log p ( x _ { k } , k h ) , x _ { k + 1 } - x _ { k } \rangle } \\ & { = \langle \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) - \nabla ^ { 2 } \log p ( x _ { k + 1 } , ( k + 1 ) h ) ( x _ { k + 1 } - x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + O ( h ^ { 3 / 2 } ) } \\ & { = \langle \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) , x _ { k + 1 } - x _ { k } \rangle } \\ & { \qquad - \langle h \tilde { b } ( x _ { k } , k h ) + \sqrt { h } \tilde { \sigma } ( k h ) \epsilon _ { k } , \nabla ^ { 2 } \log p ( x _ { k + 1 } , ( k + 1 ) h ) \big ( h \tilde { b } ( x _ { k } , k h ) + \sqrt { h } \tilde { \sigma } ( k h ) \epsilon _ { k } \big ) \rangle + O ( h ^ { 3 / 2 } ) } \\ & { = \langle \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) , x _ { k + 1 } - x _ { k } \rangle - h \tilde { \sigma } ( k h ) ^ { 2 } \Delta \log p ( x _ { k + 1 } , ( k + 1 ) h ) + O ( h ^ { 3 / 2 } ) . } \end{array}
$$

And since $\vec { p }$ satisfies the Fokker-Planck equation

$$
\begin{array} { r } { \partial _ { t } \vec { p _ { t } } = \nabla \cdot \big ( ( - \vec { b } ( x , t ) + \frac { \vec { \sigma } ( t ) ^ { 2 } } { 2 } \nabla \log \vec { p _ { t } } ( x ) ) \vec { p _ { t } } \big ) , } \end{array}
$$

we have that

$$
\begin{array} { r l } & { \partial _ { t } \log \vec { p } _ { t } = \frac { \partial _ { t } \vec { p } _ { t } } { \vec { p } _ { t } } = \frac { \nabla \cdot \big ( ( - \vec { b } ( x , t ) + \frac { \vec { \sigma } ( t ) ^ { 2 } } { 2 } \nabla \log \vec { p } _ { t } ( x ) ) \vec { p } _ { t } \big ) } { \vec { p } _ { t } } } \\ & { \qquad = - \nabla \cdot \vec { b } ( x , t ) + \frac { \vec { \sigma } ( t ) ^ { 2 } } { 2 } \Delta \log \vec { p _ { t } } ( x ) + \langle - \vec { b } ( x , t ) + \frac { \vec { \sigma } ( t ) ^ { 2 } } { 2 } \nabla \log \vec { p } _ { t } ( x ) , \nabla \log \vec { p _ { t } } ( x ) \rangle . } \end{array}
$$

Hence,

$$
\begin{array} { r l } & { \partial _ { t } \log p ( x _ { k } , k h ) = \partial _ { t } \log p ( x _ { k + 1 } , ( k + 1 ) h ) + O ( h ^ { 1 / 2 } ) } \\ & { \quad = - \nabla \cdot \tilde { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \frac { \bar { \sigma } ( ( k + 1 ) h ) ^ { 2 } } { 2 } \Delta \log \tilde { p } ( x _ { k + 1 } , ( k + 1 ) h ) } \\ & { \qquad + \left. - \bar { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \frac { \bar { \sigma } ( ( k + 1 ) h ) ^ { 2 } } { 2 } \nabla \log \tilde { p } ( x _ { k + 1 } , ( k + 1 ) h ) , \nabla \log \tilde { p } ( x _ { k + 1 } , ( k + 1 ) h ) \right. + O ( \delta ^ { 2 } ) , } \end{array}
$$

If we plug (86) and (89) into (84), we obtain

$$
\begin{array} { r l } & { \log p ( x _ { k + 1 } , ( k + 1 ) h ) - \log p ( x _ { k } , k h ) } \\ & { = h \big ( - \nabla \cdot \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \langle - \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \frac { \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } } { 2 } \nabla \log \vec { p } ( x _ { k + 1 } , ( k + 1 ) h ) , \nabla \log \vec { p } ( x _ { k + 1 } , ( k + 1 ) h ) \big ) } \\ & { \qquad + \langle \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) , x _ { k + 1 } - x _ { k } \rangle + { \cal O } ( h ^ { 3 / 2 } ) } \\ & { = \frac { \langle 2 h \vec { \sigma } ( k h ) ^ { 2 } \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) , x _ { k + 1 } - x _ { k } - h \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) \rangle } { 2 h \vec { \sigma } ( k h ) ^ { 2 } } } \\ & { \qquad + h \big ( - \nabla \cdot \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \frac { \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } } { 2 } \| \nabla \log \vec { p } ( x _ { k + 1 } , ( k + 1 ) h ) \| ^ { 2 } \big ) + { \cal O } ( h ^ { 3 / 2 } ) . } \end{array}
$$

Applying a discrete time version of Ito’s lemma again, we have that

$$
\begin{array} { r l } & { \vec { b } ( x _ { k } , k h ) = \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) - h \big ( \partial _ { t } \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \frac { \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } } { 2 } \Delta \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) \big ) } \\ & { \qquad + \nabla \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) ^ { \top } ( x _ { k } - x _ { k + 1 } ) + O ( h ^ { 3 / 2 } ) } \\ & { \qquad = \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \nabla \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) ^ { \top } ( x _ { k } - x _ { k + 1 } ) + O ( h ) . } \end{array}
$$

where $\Delta \vec { b }$ denotes the component-wise Laplacian of $\vec { b }$ . Thus,

$$
\begin{array} { r l } & { \log \tilde { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) } \\ & { = - \frac { d } { 2 } \log \left( 2 \pi h \tilde { \sigma } ( k h ) ^ { 2 } \right) - \frac { \| x _ { k + 1 } - x _ { k } - h \tilde { \sigma } ( x _ { k } , k h ) \| ^ { 2 } } { 2 h \tilde { \sigma } ( k h ) ^ { 2 } } } \\ & { = - \frac { d } { 2 } \log \left( 2 \pi h \tilde { \sigma } ( k h ) ^ { 2 } \right) - \frac { \| x _ { k + 1 } - x _ { k } - h \tilde { \sigma } ( x _ { k + 1 } , ( k + 1 ) h ) + \nabla \tilde { b } ( x _ { k + 1 } , ( k + 1 ) h ) ^ { \top } ( x _ { k } - x _ { k + 1 } ) ) \| ^ { 2 } } { 2 h \tilde { \sigma } ( k h ) ^ { 2 } } + O ( h ^ { 3 / 2 } ) } \\ & { = - \frac { d } { 2 } \log \left( 2 \pi h \tilde { \sigma } ( k h ) ^ { 2 } \right) - \frac { \| x _ { k + 1 } - x _ { k } - h \tilde { b } ( x _ { k + 1 } , ( k + 1 ) h ) \| ^ { 2 } } { 2 h \tilde { \sigma } ( k h ) ^ { 2 } } + \frac { \langle x _ { k + 1 } - x _ { k } , \nabla \tilde { b } ( x _ { k + 1 } , ( k + 1 ) h ) ^ { \top } ( x _ { k } - x _ { k + 1 } ) \rangle } { \tilde { \sigma } ( k h ) ^ { 2 } } + O ( h ^ { 3 / 2 } ) } \\ &  = - \frac { d } { 2 } \log \left( 2 \pi h \tilde { \sigma } ( k h ) ^ { 2 } \right) - \frac { \| x _ { k + 1 } - x _ { k } - h \tilde { b } ( x _ { k + 1 } , ( k + 1 ) h ) \| ^ { 2 } } { h \tilde { \sigma } ( k h ) ^ { 2 } } - \frac { h \tilde { \sigma } ( k h ) ^ { 2 } \langle c _ { k } , \nabla \tilde { b } ( x _ { k + 1 } , ( k + 1 ) h \rangle ^ { \top } c _ { k } ) }  \end{array}
$$

Combining (90) and (92), we obtain that

$$
\begin{array} { r l } & { \log \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) - \bigl ( \log p ( x _ { k + 1 } , ( k + 1 ) h ) - \log p ( x _ { k } , k h ) \bigr ) } \\ & { = - \frac { d } { 2 } \log \bigl ( 2 \pi h \vec { \sigma } ( k h ) ^ { 2 } \bigr ) - \frac { \| x _ { k + 1 } - x _ { k } - h \vec { \sigma } ( x _ { k + 1 } , ( k + 1 ) h ) + h \vec { \sigma } ( k h ) ^ { 2 } \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) \| ^ { 2 } } { h \vec { \sigma } ( k h ) ^ { 2 } } + O ( h ^ { 3 / 2 } ) } \\ & { = - \frac { d } { 2 } \log \bigl ( 2 \pi h \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } \bigr ) - \frac { \| x _ { k + 1 } - x _ { k } - h \vec { \sigma } ( x _ { k + 1 } , ( k + 1 ) h ) + h \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) \| ^ { 2 } } { h \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } } + O ( \{ h ^ { 3 / 2 } \} ) , } \end{array}
$$

By Bayes rule, and taking the exponential of this equation, we obtain

$$
\begin{array} { r l } & { \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) : = \vec { p } _ { k + 1 | k } ( x _ { k + 1 } | x _ { k } ) \frac { \vec { p } _ { k } ( x _ { k } ) } { \vec { p } _ { k + 1 } ( x _ { k + 1 } ) } } \\ & { \quad \quad \quad = \frac { \exp \big ( - \frac { \| x _ { k } - x _ { k + 1 } + h \hat { b } ( x _ { k + 1 } , ( k + 1 ) h ) - h \sigma ( ( k + 1 ) h ) ^ { 2 } \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) \| ^ { 2 } } { 2 h \sigma ( ( k + 1 ) h ) ^ { 2 } } \big ) } { ( 2 \pi h \tilde { \sigma } ( ( k + 1 ) h ) ^ { 2 } ) ^ { d / 2 } } + O ( h ^ { 3 / 2 } ) . } \end{array}
$$

Up to the ${ \cal O } ( h ^ { 3 / 2 } )$ term, the right-hand side is the conditional Gaussian corresponding to the update

$$
x _ { k } = x _ { k + 1 } + h \big ( - \vec { b } ( x _ { k + 1 } , ( k + 1 ) h ) + \vec { \sigma } ( ( k + 1 ) h ) ^ { 2 } \nabla \log p ( x _ { k + 1 } , ( k + 1 ) h ) \big ) + \sqrt { h } \vec { \sigma } ( ( k + 1 ) h ) \epsilon
$$

If we define $y _ { k } = x _ { K - k }$ , and we use that $b ( x , t ) = - \vec { b } ( x , 1 - t ) + \vec { \sigma } ( t ) ^ { 2 } \nabla \log p ( x , 1 - t )$ , we can rewrite (95) as

$$
\begin{array} { r l } & { \kappa _ { - k } = y _ { K - k - 1 } + h \big ( - \vec { b } ( y _ { K - k - 1 } , ( K - k - 1 ) h ) + \vec { \sigma } ( ( K - k - 1 ) h ) ^ { 2 } \nabla \log p ( y _ { K - k - 1 } , ( K - k - 1 ) h ) } \\ & { \qquad + \sqrt { h } \vec { \sigma } ( ( K - k - 1 ) h ) \epsilon _ { k } = y _ { K - k - 1 } + h b ( y _ { K - k - 1 } , k h ) + \sqrt { h } \sigma ( k h ) \epsilon _ { K - k - 1 } , } \\ & { \Longrightarrow y _ { k + 1 } = y _ { k } + h b ( y _ { k } , k h ) + \sqrt { h } \sigma ( k h ) \epsilon _ { k } . } \end{array}
$$

And this is the Euler-Maruyama discretization of the backward process $\overleftarrow { X }$ . If we plug (94) into (83), we obtain that

$$
\begin{array} { r } { \vec { p } ( x _ { 0 : K } ) \approx \vec { p _ { K } } ( x _ { K } ) \prod _ { k = 0 } ^ { K - 1 } \vec { p _ { k + 1 | k } } ( x _ { k + 1 } | x _ { k } ) . } \end{array}
$$

which concludes the proof, as $\vec { p } _ { K } ( x _ { K } )$ is the initial distribution of the backward process, and $\vec { p _ { k + 1 | k } } ( x _ { k + 1 } | x _ { k } )$ are its transition kernels.

# B.3 The relationship between the noise predictor $\epsilon$ and the score function

Applying Lemma 2 with the choices of $( \kappa _ { t } ) _ { t \geq 0 }$ and $( \eta _ { t } ) _ { t \geq 0 }$ for DDIM, we obtain that $\vec { X _ { t } }$ has the same distribution as

$$
\hat { X } _ { t } = \sqrt { \bar { \alpha } _ { 1 - t } } \vec { X } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { 1 - t } } \epsilon , \qquad \epsilon \sim N ( 0 , 1 ) .
$$

Since $\vec { X _ { t } }$ and $\hat { X } _ { t }$ have the same distribution, predicting the noise of $\vec { X _ { t } }$ is equivalent to predicting the noise of $\hat { X } _ { t }$ . The noise predictor $\epsilon$ can be written as:

$$
\begin{array} { r } { ( x , t ) : = \mathbb { E } [ \epsilon | \hat { X } _ { 1 - t } = x ] = \mathbb { E } \big [ \epsilon | \sqrt { \overline { { \alpha _ { t } } } } \vec { X } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon = x \big ] = \mathbb { E } \big [ \frac { x - \sqrt { \alpha _ { t } } \vec { X } _ { 0 } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \big | \sqrt { \overline { { \alpha _ { t } } } } \vec { X } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon = x \big ] } \end{array}
$$

And the score function $\pmb { \mathfrak { s } } ( x , t ) : = \nabla \log \vec { p } _ { 1 - t } ( x )$ admits the expression

$$
\begin{array} { r } { \mathfrak { s } ( x , t ) : = \nabla \log \vec { p } _ { 1 - t } ( x ) = \frac { \nabla \vec { p } _ { 1 - t } ( x ) } { \vec { p } _ { 1 - t } ( x ) } = \frac { \nabla \mathbb { E } [ \vec { p } _ { 1 - t } \mathsf { 0 } ( x | \vec { X } _ { 0 } ) ] } { \vec { p } _ { 1 - t } ( x ) } = \frac { \mathbb { E } [ \nabla \log \vec { p } _ { 1 - t } \mathsf { 0 } ( x | \vec { X } _ { 0 } ) \vec { p } _ { 1 - t | \mathrm { 0 } } ( x | \vec { X } _ { 0 } ) ] } { \vec { p } _ { 1 - t } ( x ) } , } \end{array}
$$

where

$$
\begin{array} { r } { \vec { p } _ { 1 - t | 0 } ( x | \vec { X } _ { 0 } ) = \frac { \exp ( - \| x - \sqrt { \bar { \alpha _ { t } } } Y _ { 1 } \| ^ { 2 } / ( 2 ( 1 - \bar { \alpha } _ { t } ) ) ) } { ( 2 \pi ( 1 - \bar { \alpha } _ { t } ) ) ^ { d / 2 } } \implies \nabla \log \vec { p } _ { t | 1 } ( x | Y _ { 1 } ) = - \frac { x - \sqrt { \bar { \alpha } _ { t } } Y _ { 1 } } { 1 - \bar { \alpha } _ { t } } . } \end{array}
$$

Plugging this into the right-hand side of (100) and using Bayes’ rule, we get

$$
\begin{array} { r } { \mathfrak { s } ( x , t ) = \mathbb { E } \big [ - \frac { x - \sqrt { \bar { \alpha } _ { t } } \vec { X } _ { 0 } } { 1 - \bar { \alpha } _ { t } } \big | \sqrt { \bar { \alpha } _ { t } } \vec { X } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon = x \big ] . } \end{array}
$$

Comparing the right-hand sides of (99) and (102), we obtain that $\begin{array} { r } { \mathfrak { s } ( x , t ) = - \frac { \epsilon ( x , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } } \end{array}$ .

# B.4 The relationship between the vector field $v$ and the score function

By construction (Lipman et al., 2023; Albergo and Vanden-Eijnden, 2023; Albergo et al., 2023), we have tha

$$
\begin{array} { r l } & { v ( x , t ) = \mathbb { E } [ \dot { \alpha } _ { t } Y _ { 1 } + \dot { \beta } _ { t } Y _ { 0 } | x = \alpha _ { t } Y _ { 1 } + \beta _ { t } Y _ { 0 } ] } \\ & { \qquad = \mathbb { E } [ \frac { \dot { \alpha } _ { t } ( x - \beta _ { t } Y _ { 0 } ) } { \alpha _ { t } } + \dot { \beta } _ { t } Y _ { 0 } | x = \alpha _ { t } Y _ { 1 } + \beta _ { t } Y _ { 0 } ] } \\ & { \qquad = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x + ( \dot { \beta } _ { t } - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } ) \mathbb { E } [ Y _ { 0 } | x = \alpha _ { t } Y _ { 1 } + \beta _ { t } Y _ { 0 } ] , } \end{array}
$$

where we used that $Y _ { 1 } = ( x - \beta _ { t } Y _ { 0 } ) / \alpha _ { t }$ . Also, we can write the score as follows

$$
\begin{array} { r } { \mathfrak { s } ( x , t ) : = \nabla \log p _ { t } ( x ) = \frac { \nabla p _ { t } ( x ) } { p _ { t } ( x ) } = \frac { \nabla \mathbb { E } [ p _ { t \mid 1 } ( x \mid Y _ { 1 } ) ] } { p _ { t } ( x ) } = \frac { \mathbb { E } [ \nabla p _ { t \mid 1 } ( x \mid Y _ { 1 } ) ] } { p _ { t } ( x ) } = \frac { \mathbb { E } [ p _ { t \mid 1 } ( x \mid Y _ { 1 } ) \nabla \log p _ { t \mid 1 } ( x \mid Y _ { 1 } ) ] } { p _ { t } ( x ) } , } \end{array}
$$

where

$$
\begin{array} { r } { p _ { t | 1 } ( x | Y _ { 1 } ) = \frac { \exp ( - \| x - \alpha _ { t } Y _ { 1 } \| ^ { 2 } / ( 2 \beta _ { t } ^ { 2 } ) ) } { ( 2 \pi \beta _ { t } ^ { 2 } ) ^ { d / 2 } } \implies \nabla \log \vec { p } _ { t | 1 } ( x | Y _ { 1 } ) = - \frac { x - \alpha _ { t } Y _ { 1 } } { \beta _ { t } ^ { 2 } } } \end{array}
$$

Plugging this back into the right-hand side of (104), we obtain

$$
\begin{array} { r l } & { \mathfrak { s } ( x , t ) = - \frac { \mathbb { E } [ p _ { t | 1 } ( x | Y _ { 1 } ) \frac { x - \alpha _ { t } Y _ { 1 } } { \beta _ { t } ^ { 2 } } ] } { p _ { t } ( x ) } = - \frac { \int \tilde { p } _ { t | 1 } ( x | Y _ { 1 } ) p _ { 1 } ( Y _ { 1 } ) \frac { x - \alpha _ { t } Y _ { 1 } } { \beta _ { t } ^ { 2 } } d Y _ { 1 } } { \tilde { p } _ { t } ( x ) } } \\ & { \qquad = - \int p _ { 1 | t } \bigl ( Y _ { 1 } | x \bigr ) \frac { x - \alpha _ { t } Y _ { 1 } } { \beta _ { t } ^ { 2 } } d Y _ { 1 } = - \mathbb { E } [ \frac { x - \alpha _ { t } Y _ { 1 } } { \beta _ { t } ^ { 2 } } | x = \alpha _ { t } Y _ { 1 } + \beta _ { t } Y _ { 0 } ] = - \frac { \mathbb { E } [ Y _ { 0 } | x = \alpha _ { t } Y _ { 1 } + \beta _ { t } Y _ { 0 } ] } { \beta _ { t } } } \end{array}
$$

The last equality holds because $( x - \alpha _ { t } Y _ { 1 } ) / \beta _ { t } = Y _ { 0 }$ . Putting together (103) and (106), we obtain that

$$
\begin{array} { r } { v ( x , t ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x + \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) \mathfrak { s } ( x , t ) \iff \mathfrak { s } ( x , t ) = \frac { 1 } { \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) } \bigl ( v ( x , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x \bigr ) } \end{array}
$$

Thus, the ODE (3) can be rewritten like this:

$$
\begin{array} { r } { \frac { \mathrm { d } X _ { t } } { \mathrm { d } t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) \mathfrak { s } ( X _ { t } , t ) , \qquad X _ { 0 } \sim p _ { 0 } . } \end{array}
$$

To allow for an arbitrary diffusion coefficient, we need to add a correction term to the drift:

$$
\begin{array} { r } { \mathrm { d } X _ { t } = \big ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } + \big ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) \big ) \mathfrak { s } ( X _ { t } , t ) \big ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } \sim p _ { 0 } . } \end{array}
$$

This can be easily shown by writing down the Fokker-Planck equations for (108) and (109), and observing that they are the same up to a cancellation of terms. Finally, if we plug the right-hand side of (107) into (109), we obtain the SDE for Flow Matching with arbitrary noise schedule (equation (4)).

# C Stochastic optimal control as maximum entropy RL in continuous space and time

In this section, we bridge KL-regularized (or MaxEnt) reinforcement learning and stochastic optimal control. We show that when the action space is Euclidean and the transition probabilities are conditional Gaussians, taking the limit in which the stepsize goes to zero on the KL-regularized RL problem gives rise to the SOC problem. A consequence of this connection is that all algorithms for KL-regularized RL admit an analog for diffusion fine-tuning. This is not novel, but it may be useful for researchers that are familiar with RL fine-tuning formulations.

Appendix C.4 is providing a more direct, rigorous, continuous-time connection between SOC and MaxEnt RL, as it shows that the expected control cost is equal to the KL divergence between the distributions over trajectories, conditioned on the starting points (see equation (18)).

# C.1 Maximum entropy RL

Several diffusion fine-tuning methods (Black et al., 2024; Uehara et al., 2024b) are based on KL-regularized RL, also known as maximum entropy RL, which we review in the following. In the classical reinforcement learning (RL) setting, we have an agent that, starting from state $s _ { 0 } \sim p _ { 0 }$ , iteratively observes a state $s _ { k }$ , takes an action $a _ { k }$ according to a policy $\pi ( { a } _ { k } ; { s } _ { k } , k )$ which leads to a new state $s k { + 1 }$ according to a fixed transition probability $p ( s _ { k + 1 } | a _ { k } , s _ { k } )$ , and obtains rewards $r _ { k } ( s _ { k } , a _ { k } )$ . This can be summarized into a trajectory ${ \boldsymbol { \tau } } = ( ( s _ { k } , a _ { k } ) ) _ { k = 0 } ^ { K }$ . The goal is to optimize the policy $\pi$ in order to maximize the expected total reward, i.e. $\begin{array} { r } { \operatorname* { m a x } _ { \boldsymbol { \pi } } \mathbb { E } _ { \tau \sim \pi , p } [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) ] } \end{array}$ .

Maximum entropy RL (MaxEnt RL; Ziebart et al. (2008)) amounts to adding the entropy $H ( \pi )$ of the policy $\pi ( \cdot ; s _ { k } , k )$ to the reward for each step $k$ , in order to encourage exploration and improve robustness to changes in the environment: $\begin{array} { r } { \operatorname* { m a x } _ { \boldsymbol { \pi } } \mathbb { E } _ { \tau \sim \pi , p } [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) + \sum _ { k = 0 } ^ { K - 1 } H ( \boldsymbol { \pi } ( \cdot ; s _ { k } , k ) ) ] } \end{array}$ 8. As a generalization, one can regularize using the negative KL divergence between $\pi ( \cdot ; s _ { k } , k )$ and a base policy

$$
\begin{array} { r } { \operatorname* { m a x } _ { \boldsymbol { \pi } } \mathbb { E } _ { \tau \sim \pi , p } [ \sum _ { k = 0 } ^ { K } r _ { k } \big ( s _ { k } , { a } _ { k } \big ) - \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } ( \pi ( \cdot ; s _ { k } , k ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) ] , } \end{array}
$$

which prevents the learned policy to deviate too much from the base policy. Each policy $\pi$ induces a distribution $q ( \tau )$ over trajectories $\tau$ , and the MaxEnt RL problem (110) can be expressed solely in terms of such distributions (Lemma 3 in Appendix C.3):

$$
\begin{array} { r } { \operatorname* { m a x } _ { q } \mathbb { E } _ { \tau \sim q } [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) ] - \mathrm { K L } ( q | | q ^ { \mathrm { b a s e } } ) , } \end{array}
$$

where $q ^ { \mathrm { b a s e } }$ is the distribution induced by the base policy $\pi _ { \mathrm { b a s e } }$ , and the maximization is over all distributions $q$ such that their marginal for $s _ { 0 }$ is $p _ { 0 }$ . We can further recast this problem as (Lemma 4 in Appendix C.3):

$$
\begin{array} { r } { \operatorname* { m i n } _ { \boldsymbol { q } } \mathrm { K L } ( \boldsymbol { q } | | \boldsymbol { q } ^ { * } ) , \qquad \mathrm { w h e r e ~ } \boldsymbol { q } ^ { * } ( \tau ) : = \boldsymbol { q } ^ { \mathrm { b a s e } } ( \tau ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) - \mathcal { V } ( s _ { 0 } , 0 ) \big ) , } \end{array}
$$

where

$$
\begin{array} { r l } & { \mathcal { V } ( s _ { k } , k ) : = \log \left( \mathbb { E } _ { \tau \sim \pi _ { \mathrm { b a s e } } , p } [ \exp \left( \sum _ { k ^ { \prime } = k } ^ { K } r _ { k ^ { \prime } } ( s _ { k ^ { \prime } } , a _ { k ^ { \prime } } ) \right) | s _ { k } ] \right) } \\ & { \qquad = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \tau \sim \pi , p } \left[ \sum _ { k ^ { \prime } = k } ^ { K } r _ { k ^ { \prime } } ( s _ { k ^ { \prime } } , a _ { k ^ { \prime } } ) - \sum _ { k ^ { \prime } = k } ^ { K - 1 } \mathrm { K L } ( \pi ( \cdot ; s _ { k ^ { \prime } } , k ^ { \prime } ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k ^ { \prime } } , k ^ { \prime } ) ) | s _ { k } \right] } \end{array}
$$

is the value function. Problem (112) directly implies that the distribution induced by the optimal policy $\pi ^ { * }$ is the tilted distribution $q ^ { * }$ (which has initial marginal $p _ { 0 }$ ).

# C.2 From maximum entropy RL to stochastic optimal control

The following well-known result, which we prove in Appendix C.3, shows that in a natural sense, the continuous-time continuous-space version of MaxEnt RL is the SOC framework introduced in Section 4.1. In particular, when states and actions are vectors in $\mathbb { R } ^ { d }$ , policies are specified by a vector field $u$ (the control), and transition probabilities are conditional Gaussians, the MaxEnt RL problem becomes an SOC problem when the number of timesteps grows to infinity.

Proposition 5. Suppose that

(i) The state space and the action space are $\mathbb { R } ^ { d }$ ,   
(ii) Policies $\pi$ are specified as $\pi ( a _ { k } ; s _ { k } , k ) = \delta ( a _ { k } - u ( s _ { k } , k h ) )$ , where $u : \mathbb { R } ^ { d } \times [ 0 , T ]  \mathbb { R } ^ { d }$ is a vector field, and $\delta$ denotes the Dirac delta,   
(iii) Transition probabilities are conditional Gaussian densities: $p ( s _ { k + 1 } | a _ { k } , s _ { k } ) = N ( s _ { k } + h ( b ( s _ { k } , k h ) +$ $\sigma ( k h ) a _ { k } ) , h \sigma ( k h ) \sigma ( k h ) ^ { \top } )$ , where $h = T / K$ is the stepsize, and b and $\sigma$ are defined as in Section 4.1.

Then, in the limit in which the number of steps $K$ grows to infinity, the problem (110) is equivalent to the SOC problem (12)-(13), identifying

• the sequence of states $\left( \boldsymbol { s } _ { k } \right) _ { k = 0 } ^ { k }$ with the trajectory $X ^ { u } = ( X _ { t } ^ { u } ) _ { t \in [ 0 , 1 ] }$ ,   
• the running reward $\begin{array} { r } { \sum _ { k = 0 } ^ { K - 1 } r _ { k } ( s _ { k } , a _ { k } ) } \end{array}$ with the negative running cost $\begin{array} { r } { - \int _ { 0 } ^ { T } f ( X _ { t } ^ { u } , t ) \mathrm { d } t } \end{array}$ ,   
• the terminal reward $r _ { K } ( s _ { K } , a _ { K } )$ with the negative terminal cost $- g ( X _ { T } ^ { u } )$ ,   
• the KL regularization $\begin{array} { r } { \mathbb { E } _ { \tau \sim \pi , p } [ \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } ( \pi ( \cdot ; s _ { k } , k ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) ] } \end{array}$ with $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ times the expected $L ^ { 2 }$ norm of the control $\begin{array} { r } { \frac { 1 } { 2 } \mathbb { E } \big [ \int _ { 0 } ^ { T } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t \big ] } \end{array}$ ,   
• and the value function $\mathcal { V } ( s _ { k } , k )$ defined in (113) with the negative value function $- V ( x , t )$ defined in Section 4.1.

A first consequence of this result is that every loss function designed for generic MaxEnt RL problems has a corresponding loss function for SOC problems. The geometric structure of the latter allows for additional losses that do not have an analog in the classical MaxEnt RL setting; in particular, we can differentiate the state and terminal costs.

A second consequence of Proposition 5 is that the characterization (112) can be translated to the SOC setting. The analogs of the distributions $q ^ { * }$ , $q ^ { \mathrm { b a s e } }$ induced by the optimal policy $\pi ^ { * }$ and the base policy $\pi ^ { \mathrm { b a s e } }$ are the distributions $p ^ { * } , p ^ { \mathrm { b a s e } }$ induced by the optimal control $u ^ { * }$ and the null control. For an arbitrary trajectory $\pmb { X } = ( X _ { t } ) _ { t \in [ 0 , T ] }$ , the relation between $\mathbb { P } ^ { * }$ and $\mathbb { P } ^ { \mathrm { b a s e } }$ is given by

$$
\begin{array} { r } { \frac { \mathrm { d } \mathbb { P } ^ { * } } { \mathrm { d } \mathbb { P } ^ { \mathrm { b a s e } } } ( X ) = \exp ( - \int _ { 0 } ^ { T } f ( X _ { t } , t ) \mathrm { d } t - g ( X _ { T } ) + V ( X _ { 0 } , 0 ) ) } \end{array}
$$

where $V$ is the value function as defined in Section 4.1. Note that this matches the statement in (22).

# C.3 Proof of Proposition 5: from MaxEnt RL to SOC

Since the transition $p ( s _ { k + 1 } | a _ { k } , s _ { k } )$ is fixed, for each $\pi$ we can define

$\tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) = \pi ( a _ { k } ; s _ { k } , k ) p ( s _ { k + 1 } | a _ { k } , s _ { k } ) \mathrm { ~ a n d ~ } \tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) = \pi _ { \mathrm { b a s e } } ( a _ { k } ; s _ { k } , k ) p ( s _ { k + 1 } | a _ { k } , k ) .$ 1|ak, sk),

and reexpress (110) as (see Lemma 3)

$$
\begin{array} { r } { \operatorname* { m i n } _ { \tilde { \pi } } \mathbb { E } _ { \tau \sim \tilde { \pi } } [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) - \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } \big ( \tilde { \pi } ( \cdot , \cdot ; s _ { k } , k ) \big | \big | \tilde { \pi } _ { \mathrm { b a s e } } ( \cdot , \cdot ; s _ { k } , k ) \big ) ] . } \end{array}
$$

Using the hypothesis of the proposition, we can write

$$
\begin{array} { r l } & { \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) = \delta ( a _ { k } - u ( s _ { k } , k \eta ) ) N ( s _ { k } + \eta ( b ( s _ { k } , k \eta ) + \sigma ( k \eta ) a _ { k } ) , \eta \sigma ( k \eta ) \sigma ( k \eta ) ^ { \top } ) } \\ & { \qquad = \delta ( a _ { k } - u ( s _ { k } , k \eta ) ) \tilde { \pi } ( s _ { k + 1 } ; s _ { k } , k ) , } \end{array}
$$

where π˜(sk+1; s ${ \bf \varepsilon } _ { \ast } , k ) = N ( s _ { k } + \eta ( b ( s _ { k } , k \eta ) + \sigma ( k \eta ) u ( s _ { k } , k \eta ) ) , { \nu } $ ησ(kη)σ(kη)⊤) is the state transition kernel. We set the base policy as $\pi _ { \mathrm { b a s e } } ( a _ { k } ; s _ { k } , k ) = \delta ( a _ { k } )$ , and we obtain analogously that $\tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) =$ $\delta ( a _ { k } ) \tilde { \pi } _ { \mathrm { b a s e } } ( s _ { k + 1 } ; s _ { k } , k )$ with $\tilde { \pi } _ { \mathrm { b a s e } } ( s _ { k + 1 } ; s _ { k } , k ) = N ( s _ { k } + \eta b ( s _ { k } , k \eta ) , \eta \sigma ( k \eta ) \sigma ( k \eta ) ^ { \scriptscriptstyle 1 } )$ . Now, if we take $K$ large, the trajectory $( s _ { k } ) _ { k = 0 } ^ { K }$ generated by $\tilde { \pi }$ can be regarded as the Euler-Maruyama discretization of a solution $X ^ { u }$ of the controlled SDE (13), while the trajectory generated by $\tilde { \pi } _ { \mathrm { b a s e } }$ is the discretization of the uncontrolled process $X ^ { 0 }$ obtained by setting $u = 0$ . As a consequence

$$
\begin{array} { r l } & { \operatorname* { l i m } _ { K \to \infty } \mathbb { E } _ { \tau \sim \tilde { \pi } } [ \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } ( \tilde { \pi } ( \cdot , \cdot ; s _ { k } , k ) | | \tilde { \pi } _ { \mathrm { b a s e } } ( \cdot , \cdot ; s _ { k } , k ) ) ] } \\ & { = \operatorname* { l i m } _ { K \to \infty } \mathbb { E } _ { \tau \sim \tilde { \pi } } [ \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } ( \tilde { \pi } ( \cdot ; s _ { k } , k ) | | \tilde { \pi } _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) ] = \mathbb { E } _ { X ^ { u } \sim \mathbb { P } ^ { u } } [ \log \frac { \mathrm { d } \mathbb { P } ^ { u } } { \mathrm { d } \mathbb { P } ^ { 0 } } ( X ^ { u } ) ] , } \end{array}
$$

where $\mathbb { P } ^ { u }$ and $\mathbb { P } ^ { 0 }$ are the measures of the processes $X ^ { u }$ and $X ^ { 0 }$ , respectively. The Girsanov theorem (Theorem 2) implies that $\begin{array} { r l r } { \log \frac { \mathrm { d } \mathbb { P } ^ { u } } { \mathrm { d } \mathbb { P } ^ { 0 } } ( X ^ { u } ) } & { { } = } & { - \int _ { 0 } ^ { T } \langle u ( X _ { t } ^ { u } , t ) , \mathrm { d } B _ { t } \rangle - \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t } \end{array}$ , which implies that $\begin{array} { r } { \mathbb { E } _ { X ^ { u } \sim \mathbb { P } ^ { u } } [ \log \frac { \mathrm { d } \mathbb { P } ^ { u } } { \mathrm { d } \mathbb { P } ^ { 0 } } ( X ^ { u } ) ] = - \frac { 1 } { 2 } \mathbb { E } _ { X ^ { u } \sim \mathbb { P } ^ { u } } [ \int _ { 0 } ^ { T ^ { \prime } } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t ] } \end{array}$ . Setting the rewards $r _ { k } ( a _ { k } , s _ { k } ) = \eta f ( s _ { k } , k \eta )$ for $k \in$ $\{ 0 , \ldots , K - 1 \}$ and $r _ { K } ( a _ { K } , s _ { K } ) = \eta g ( s _ { k } )$ , where $f$ and $g$ are as in Section 4.1, yields the following limiting object:

$$
\begin{array} { r } { \operatorname* { l i m } _ { K \to \infty } \mathbb { E } _ { \tau \sim \tilde { \pi } } [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) ] = \mathbb { E } _ { X ^ { u } \sim \mathbb { P } ^ { u } } [ \int _ { 0 } ^ { T } f ( X _ { t } ^ { u } , t ) d t + g ( X _ { T } ^ { u } ) ] . } \end{array}
$$

Hence, the limit of the MaxEnt RL loss (116) is the SOC loss (12).

Lemma 3. Let $\tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k )$ and $\tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k )$ be as defined in (115). $\operatorname { K L } ( \tilde { \pi } ( \cdot , \cdot ; s _ { k } , k ) | | \tilde { \pi } _ { \mathrm { b a s e } } ( \cdot , \cdot ; s _ { k } , k ) ) ]$ and $\mathrm { K L } ( \pi ( \cdot ; s _ { k } , k ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) ]$ are equal. Moreover, if $q$ , $q ^ { \mathrm { b a s e } }$ denote the distributions over trajectories induced by $\pi$ , $\pi _ { \mathrm { b a s e } }$ , we have that

$$
\begin{array} { r } { \mathrm { K L } ( q | | q ^ { \mathrm { b a s e } } ) = \mathbb { E } [ \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } ( \pi ( \cdot ; s _ { k } , k ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) ] . } \end{array}
$$

Proof. We have that

$$
\begin{array} { r l } & { \mathrm { K L } ( \tilde { \pi } ( \cdot , \cdot ; s _ { k } , k ) | | \tilde { \pi } _ { \mathrm { b a s e } } ( \cdot , \cdot ; s _ { k } , k ) ) ] = \sum _ { a _ { k } , s _ { k } + 1 } \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) \log { \frac { \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } { \tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } } } \\ & { = \sum _ { a _ { k } , s _ { k + 1 } } \pi ( a _ { k } ; s _ { k } , k ) p ( s _ { k + 1 } | a _ { k } , s _ { k } ) \log { \frac { \pi ( a _ { k } ; s _ { k } , k ) p ( s _ { k + 1 } | a _ { k } , s _ { k } ) } { \pi _ { \mathrm { b a s e } } ( a _ { k } ; s _ { k } , k ) p ( s _ { k + 1 } | a _ { k } , s _ { k } ) } } } \\ & { = \sum _ { a _ { k } , s _ { k + 1 } } \pi ( a _ { k } ; s _ { k } , k ) p ( s _ { k + 1 } | a _ { k } , s _ { k } ) \log { \frac { \pi ( a _ { k } ; s _ { k } , k ) } { \pi _ { \mathrm { b a s e } } ( a _ { k } ; s _ { k } , k ) } } } \\ & { = \sum _ { a _ { k } } \pi ( a _ { k } ; s _ { k } , k ) \big ( \sum _ { s _ { k + 1 } } p ( s _ { k + 1 } | a _ { k } , s _ { k } ) \big ) \log { \frac { \pi ( a _ { k } ; s _ { k } , k ) } { \pi _ { \mathrm { b a s e } } ( a _ { k } ; s _ { k } , k ) } } } \\ &  = \sum _ { a _ { k } } \pi ( a _ { k } ; s _ { k } , k ) \log  \frac { \pi ( a _ { k } ; s _ { k } , k ) }  \pi _ { \mathrm { b a s e } } ( a _ { k } ; s _  \end{array}
$$

To prove (120), by construction we can write

$$
\begin{array} { r l r l } & { \boldsymbol { I } ( \tau ) = p _ { 0 } ( s _ { 0 } ) \prod _ { k = 0 } ^ { K - 1 } \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) , \qquad } & & { \boldsymbol { q } ^ { \mathrm { b a s e } } ( \tau ) = p _ { 0 } ( s _ { 0 } ) \prod _ { k = 0 } ^ { K - 1 } \tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) , } \end{array}
$$

which means that

$$
\begin{array} { r l } & { \mathrm { K L } ( q | | q ^ { \mathrm { b a s e } } ) = \mathbb { E } _ { \tau \sim q } [ \mathrm { l o g } \frac { q ( \tau ) } { q ^ { \mathrm { b a s e } } ( \tau ) } ] = \mathbb { E } _ { \tau \sim q } [ \sum _ { k = 0 } ^ { K - 1 } \log \frac { \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } { \tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } ] } \\ & { = \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } _ { \tau \sim q ^ { 0 } ; ( k + 1 ) } [ \mathrm { l o g } \frac { \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } { \tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } ] } \\ & { = \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } _ { \tau \sim q ^ { 0 ; k } } [ \sum _ { a _ { k } , s _ { k + 1 } } \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) \log \frac { \tilde { \pi } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } { \tilde { \pi } _ { \mathrm { b a s e } } ( a _ { k } , s _ { k + 1 } ; s _ { k } , k ) } ] } \\ & { = \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } _ { \tau \sim q ^ { 0 ; k } } [ \mathrm { K L } ( \tilde { \pi } ( \cdot ; s _ { k } , k ) | | \tilde { \pi } _ { \mathrm { b a s e } } ( \cdot , \cdot ; s _ { k } , k ) ) ] } \\ & { = \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } _ { \tau \sim q ^ { 0 ; k } } [ \mathrm { K L } ( \pi ( \cdot ; s _ { k } , k ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) ] } \\ &  = \mathbb { E } _  \end{array}
$$

Here, the notation $q ^ { 0 : k }$ denotes the trajectory $q$ up to the state $s _ { k }$

Lemma 4. The distribution-based MaxEnt RL formulation in (111) is equivalent to the the following problem:

$$
\begin{array} { r } { \operatorname* { m i n } _ { q } \mathrm { K L } ( q | | q ^ { * } ) , \qquad w h e r e \ q ^ { * } ( \tau ) : = \frac { q ^ { \mathrm { b a s e } } ( \tau ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) \big ) } { \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \big ) } , } \end{array}
$$

where the minimization is over $q$ with marginal $p _ { 0 }$ at step zero. The optimum of the problem is $q ^ { * }$ , which satisfies the marginal constraint. The following alternative characterization of $q ^ { * }$ holds:

$$
\begin{array} { r l } & { q ^ { * } ( \tau ) = q ^ { \mathrm { b a s e } } ( \tau ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) - \mathcal { V } ( s _ { 0 } , 0 ) \big ) , } \\ & { \mathcal { V } ( x , k ) = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \tau \sim \pi , p } \big [ \sum _ { k ^ { \prime } = k } ^ { K } r _ { k ^ { \prime } } ( s _ { k ^ { \prime } } , a _ { k ^ { \prime } } ) - \sum _ { k ^ { \prime } = k } ^ { K - 1 } \mathrm { K L } ( \pi ( \cdot ; s _ { k ^ { \prime } } , k ^ { \prime } ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k ^ { \prime } } , k ^ { \prime } ) ) | s _ { k } = x \big ] . } \end{array}
$$

Proof. Let us expand $\mathrm { K L } ( q | | q ^ { * } )$ :

$$
\begin{array} { r l } & { \mathrm { K L } ( q | | q ^ { * } ) = \mathbb { E } _ { \tau \sim q } \big [ \log \frac { q ( \tau ) } { q ^ { * } ( \tau ) } \big ] } \\ & { \qquad = \mathbb { E } _ { \tau \sim q } \big [ \log q ( \tau ) - \log q ^ { \mathrm { b a s e } } ( \tau ) - \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) } \\ & { \qquad + \log \big ( \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \big ) \big ) \big ] } \\ & { \qquad = \mathrm { K L } ( q | | q ^ { \mathrm { b a s e } } ) - \mathbb { E } _ { \tau \sim q } \big [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) \big ] } \\ & { \qquad + \mathbb { E } _ { s _ { 0 } \sim p _ { 0 } } \big [ \log \big ( \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \big ) \big ) \big ] , } \end{array}
$$

where the third equality holds because the marginal of $q$ at step zero is $p _ { 0 }$ by hypothesis. Since the third term in the right-hand side is independent of $q$ , this proves the equivalence between (111) and (124).

Next, we prove that the marginal of $q ^ { * }$ at step zero is $p _ { 0 }$ :

$$
\begin{array} { r }  \sum _ { \{ \tau \} s _ { 0 } = x \} q ^ { * } ( \tau ) : = \sum _ { \{ \tau | s _ { 0 } = x \} } \frac { q ^ { \mathrm { b a s e } } ( \tau ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) \big ) } { \frac { 1 } { p _ { 0 } ( x ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = x \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \big ) } = p _ { 0 } ( x ) . } \end{array}
$$

Now, for an arbitrary $s _ { 0 }$ , let $q _ { s _ { 0 } }$ , $q _ { s _ { 0 } } ^ { * }$ be the distributions $q$ , $q ^ { * }$ conditioned on the initial state being $s _ { 0 }$ . We can write an analog to equation (127) for $q _ { s _ { 0 } }$ , $q _ { s _ { 0 } } ^ { * }$ :

$$
\begin{array} { r l } & { \mathrm { K L } ( q _ { s _ { 0 } } | | q _ { s _ { 0 } } ^ { * } ) = \mathbb { E } _ { \tau \sim q _ { s _ { 0 } } } \left[ \log \frac { q _ { s _ { 0 } } ( \tau ) } { q _ { s _ { 0 } } ^ { * } ( \tau ) } \right] } \\ & { \phantom { = } = \mathbb { E } _ { \tau \sim q _ { s _ { 0 } } } \left[ \log q _ { s _ { 0 } } ( \tau ) - \log q _ { s _ { 0 } } ^ { \mathrm { b a s e } } ( \tau ) - \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) \right. } \\ & { \phantom { = } \left. \qquad + \log \left( \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q _ { s _ { 0 } } ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \left( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \right) \right) \right] } \\ & { \phantom { = } = \mathrm { K L } ( q _ { s _ { 0 } } | | q _ { s _ { 0 } } ^ { \mathrm { b a s e } } ) - \mathbb { E } _ { \tau \sim q _ { s _ { 0 } } } \left[ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) \right] } \\ & { \phantom { = } \quad + \log \left( \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \left( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \right) \right) , } \end{array}
$$

Hence,

$$
\begin{array} { r } { 0 = \operatorname* { m i n } _ { q _ { s _ { 0 } } } \mathrm { K L } ( q _ { s _ { 0 } } | | q _ { s _ { 0 } } ^ { * } ) = - \operatorname* { m a x } _ { q _ { s _ { 0 } } } \{ \mathbb { E } _ { \tau \sim q _ { s _ { 0 } } } \big [ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) \big ] - \mathrm { K L } ( q _ { s _ { 0 } } | | q _ { s _ { 0 } } ^ { \mathrm { b a s e } } ) \} } \\ { + \log \big ( \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \big ( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \big ) \big ) . } \end{array}
$$

And applying (120) from (120), we obtain that

$$
\begin{array} { r l } & { \log \left( \frac { 1 } { p _ { 0 } ( s _ { 0 } ) } \sum _ { \{ \tau ^ { \prime } | s _ { 0 } ^ { \prime } = s _ { 0 } \} } q ^ { \mathrm { b a s e } } ( \tau ^ { \prime } ) \exp \left( \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } ^ { \prime } , a _ { k } ^ { \prime } ) \right) \right) } \\ & { = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \tau \sim \pi , p } \left[ \sum _ { k = 0 } ^ { K } r _ { k } ( s _ { k } , a _ { k } ) - \sum _ { k = 0 } ^ { K - 1 } \mathrm { K L } ( \pi ( \cdot ; s _ { k } , k ) | | \pi _ { \mathrm { b a s e } } ( \cdot ; s _ { k } , k ) ) | s _ { 0 } \right] = \mathcal { V } ( s _ { 0 } , 0 ) , } \end{array}
$$

which concludes the proof.

# C.4 Proof of equation (18): the control cost is a KL regularizer

Theorem 2 (Girsanov theorem for SDEs). If the two SDEs

$$
\begin{array} { r l } & { \mathrm { d } X _ { t } = b _ { 1 } ( X _ { t } , t ) \mathrm { d } t + \sigma ( X _ { t } , t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } = x _ { \mathrm { i n i t } } } \\ & { d Y _ { t } = ( b _ { 1 } ( Y _ { t } , t ) + b _ { 2 } ( Y _ { t } , t ) ) \mathrm { d } t + \sigma ( Y _ { t } , t ) \mathrm { d } B _ { t } , \qquad Y _ { 0 } = x _ { \mathrm { i n i t } } } \end{array}
$$

admit unique strong solutions on $[ 0 , T ]$ , then for any bounded continuous functional $\Phi$ on $C ( [ 0 , T ] )$ , we have that

$$
\begin{array} { r l } & { \mathbb { E } [ \Phi ( { \pmb X } ) ] = \mathbb { E } \big [ \Phi ( { \pmb Y } ) \exp \big ( - \int _ { 0 } ^ { T } \sigma ( Y _ { t } , t ) ^ { - 1 } b _ { 2 } ( Y _ { t } , t ) \mathrm { d } B _ { t } - \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| \sigma ( Y _ { t } , t ) ^ { - 1 } b _ { 2 } ( Y _ { t } , t ) \| ^ { 2 } \mathrm { d } t \big ) \big ] } \\ & { \qquad = \mathbb { E } \big [ \Phi ( { \pmb Y } ) \exp \big ( - \int _ { 0 } ^ { T } \sigma ( Y _ { t } , t ) ^ { - 1 } b _ { 2 } ( Y _ { t } , t ) d \tilde { B } _ { t } + \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| \sigma ( Y _ { t } , t ) ^ { - 1 } b _ { 2 } ( Y _ { t } , t ) \| ^ { 2 } \mathrm { d } t \big ) \big ] , } \end{array}
$$

where $\begin{array} { r } { \tilde { B } _ { t } = B _ { t } + \int _ { 0 } ^ { t } \sigma ( Y _ { s } , s ) ^ { - 1 } b _ { 2 } ( Y _ { s } , s ) \mathrm { d } s } \end{array}$ . More generally, $b _ { 1 }$ and $b _ { 2 }$ can be random processes that are adapted to filtration of $\mathbfcal { B }$ .

Consider the SDEs

$$
\begin{array} { r l r l } & { \mathrm { d } X _ { t } = b ( X _ { t } , t ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , } & & { \quad \quad X _ { 0 } = x _ { 0 } , } \\ & { \mathrm { d } X _ { t } ^ { u } = \left( b ( X _ { t } ^ { u } , t ) + \sigma ( t ) u ( X _ { t } ^ { u } , t ) \right) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , } & & { \quad \quad X _ { 0 } ^ { u } = x _ { 0 } . } \end{array}
$$

If we let $\mathbb { P } | _ { x _ { 0 } }$ , $\mathbb { P } ^ { u } | _ { x _ { 0 } }$ be the probability measures of the solutions of (135) and (136), Theorem 2 implies that

$$
\begin{array} { r } { \log \frac { \mathrm { d } \mathbb { P } | _ { x _ { 0 } } } { \mathrm { d } \mathbb { P } ^ { u } | _ { x _ { 0 } } } ( \pmb { X } ^ { u } ) = - \int _ { 0 } ^ { 1 } u ( X _ { t } ^ { u } , t ) \mathrm { d } B _ { t } - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t . } \end{array}
$$

Hence,

$$
\begin{array} { r l } & { \mathbb { P } ^ { u } | _ { x _ { 0 } } \left\| \mathbb { P } | _ { x _ { 0 } } \right) = \mathbb { E } \big [ \log \frac { \mathrm { d } \mathbb { P } ^ { u } | _ { x _ { 0 } } } { \mathrm { d } \mathbb { P } | _ { x _ { 0 } } } ( \pmb { X } ^ { u } ) | \pmb { X } _ { 0 } ^ { u } = x _ { 0 } \big ] = - \mathbb { E } \big [ \log \frac { \mathrm { d } \mathbb { P } | _ { x _ { 0 } } } { \mathrm { d } \mathbb { P } ^ { u } | _ { x _ { 0 } } } ( \pmb { X } ^ { u } ) | \pmb { X } _ { 0 } ^ { u } = x _ { 0 } \big ] } \\ & { \qquad = \mathbb { E } \big [ \int _ { 0 } ^ { 1 } u ( X _ { t } ^ { u } , t ) \mathrm { d } B _ { t } + \frac 1 2 \int _ { 0 } ^ { 1 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t | X _ { 0 } ^ { u } = x _ { 0 } \big ] = \mathbb { E } \big [ \frac 1 2 \int _ { 0 } ^ { 1 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t | X _ { 0 } ^ { u } = x _ { 0 } \big ] . } \end{array}
$$

where we used that stochastic integrals are martingales.

# D Proofs of Section 4.3: memoryless noise schedule and fine-tuning recipe

# D.1 Proof of Proposition 1: the memoryless noise schedule

We consider the forward-backward SDEs (63)-(64) with arbitrary noise schedule. By Proposition 4, the trajectories $\vec { X }$ , $\pmb { X }$ of these two processes are equally distributed up to a time flip, which also means that their marginals satisfy $\vec { p _ { t } } = p _ { 1 - t }$ , for all $t \in [ 0 , 1 ]$ . First, we develop an explicit expression for the score function $s ( x , t ) = \nabla \log p _ { t } ( x )$ . By the properties of flow matching, we know that $p _ { t }$ is the distribution of the interpolation variable $\bar { X } _ { t } = \beta _ { t } \bar { X } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 }$ , where $\bar { X } _ { 0 } \sim N ( 0 , I ) , \bar { X } _ { 1 } \sim p ^ { \mathrm { d a t a } }$ are independent. Thus, X¯t−αtX¯1β ∼ N (0, I), which means that we can express the density pt as

$$
\begin{array} { r } { p _ { t } ( x ) = \int _ { \mathbb { R } ^ { d } } \frac { \exp \big ( - \frac { \| x - \alpha _ { t } y \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } \big ) } { ( 2 \pi \beta _ { t } ^ { 2 } ) ^ { d / 2 } } p ^ { \mathrm { d a t a } } ( y ) \mathrm { d } y . } \end{array}
$$

Thus,

$$
\begin{array} { r } { s ( x , t ) = \nabla \log p _ { t } ( x ) = - \frac { x } { \beta _ { t } ^ { 2 } } + \frac { \alpha _ { t } } { \beta _ { t } ^ { 2 } } \frac { \int _ { \mathbb { R } ^ { d } } y \exp \big ( - \frac { \| x - \alpha _ { t } y \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } \big ) p ^ { \mathrm { d a t a } } ( y ) \mathrm { d } y } { \int _ { \mathbb { R } ^ { d } } \exp \big ( - \frac { \| x - \alpha _ { t } y \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } \big ) p ^ { \mathrm { d a t a } } ( y ) \mathrm { d } y } : = - \frac { x - \alpha _ { t } \xi _ { t } ( x ) } { \beta _ { t } ^ { 2 } } , } \end{array}
$$

where we defined

$$
\begin{array} { r } { \xi _ { t } ( x ) = \frac { \int _ { \mathbb { R } ^ { d } } y \exp \big ( - \frac { \| x - \alpha _ { t } y \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } \big ) p ^ { \mathrm { d a t a } } ( y ) \mathrm { d } y } { \int _ { \mathbb { R } ^ { d } } \exp \big ( - \frac { \| x - \alpha _ { t } y \| ^ { 2 } } { 2 \beta _ { t } ^ { 2 } } \big ) p ^ { \mathrm { d a t a } } ( y ) \mathrm { d } y } . } \end{array}
$$

Hence, we can rewrite the forward SDE (63) as

$$
\begin{array} { r } { \mathrm { d } \vec { X } _ { t } = \left( - \kappa _ { 1 - t } \vec { X } _ { t } - \left( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \right) \frac { \vec { X } _ { t } - \alpha _ { 1 - t } \xi _ { 1 - t } ( \vec { X } _ { t } ) } { \beta _ { 1 - t } ^ { 2 } } \right) \mathrm { d } t + \sigma ( 1 - t ) \mathrm { d } B _ { t } , \qquad \vec { X } _ { 0 } \sim p _ { \mathrm { d a t a } } , } \end{array}
$$

Hence, if we substitute $\begin{array} { r } { \kappa _ { 1 - t }  \kappa _ { 1 - t } + \frac { \sigma ( 1 - t ) ^ { 2 } - 2 \eta _ { 1 - t } } { 2 \beta _ { 1 - t } ^ { 2 } } } \end{array}$ , $\begin{array} { r } { \xi _ { 1 - t }  \frac { \alpha _ { 1 - t } ( \sigma ( 1 - t ) ^ { 2 } - 2 \eta _ { 1 - t } ) } { 2 \beta _ { 1 - t } ^ { 2 } } \xi _ { 1 - t } ( \vec { X } _ { t } ) } \end{array}$ (where we ignore the dependency on $\vec { X _ { t } }$ ), $\sqrt { 2 \eta _ { 1 - t } }  \sigma ( 1 - t )$ , we can apply Lemma 2, which yields

$$
\begin{array} { r l } & { \vec { X } _ { t } = \vec { X } _ { 0 } \exp \big ( - \int _ { 0 } ^ { t } \big ( \kappa _ { 1 - s } + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \eta _ { 1 - s } } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s \big ) } \\ & { \qquad + \int _ { 0 } ^ { t } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \big ( \kappa _ { 1 - s } + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \eta _ { 1 - s } } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s \big ) \frac { \alpha _ { 1 - t ^ { \prime } } ( \sigma ( 1 - t ^ { \prime } ) ^ { 2 } - 2 \eta _ { 1 - t ^ { \prime } } ) } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } \xi _ { 1 - t ^ { \prime } } ( \vec { X } _ { t ^ { \prime } } ) \mathrm { d } t ^ { \prime } } \\ & { \qquad + \int _ { 0 } ^ { t } \sigma ( 1 - t ^ { \prime } ) \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \big ( \kappa _ { 1 - s } + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \eta _ { 1 - s } } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s \big ) \mathrm { d } B _ { t ^ { \prime } } . } \end{array}
$$

We simplify the recurring expression:

$$
\begin{array} { r } { \kappa _ { 1 - s } + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \eta _ { 1 - s } } { 2 \beta _ { 1 - s } ^ { 2 } } = \frac { \dot { \alpha } _ { 1 - s } } { \alpha _ { 1 - s } } + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \beta _ { 1 - s } \left( \frac { \dot { \alpha } _ { 1 - s } } { \alpha _ { 1 - s } } \beta _ { 1 - s } - \dot { \beta } _ { 1 - s } \right) } { 2 \beta _ { 1 - s } ^ { 2 } } = \frac { \sigma ( 1 - s ) ^ { 2 } } { 2 \beta _ { 1 - s } ^ { 2 } } + \frac { \dot { \beta } _ { 1 - s } } { \beta _ { 1 - s } } } \end{array}
$$

Thus,

$$
\begin{array} { r } { s + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \eta _ { 1 - s } } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s = \int _ { t ^ { \prime } } ^ { t } \big ( \frac { \sigma ( 1 - s ) ^ { 2 } } { 2 \beta _ { 1 - s } ^ { 2 } } - \partial _ { s } \log \beta _ { 1 - s } \big ) \mathrm { d } s = \int _ { t ^ { \prime } } ^ { t } \frac { \sigma ( 1 - s ) ^ { 2 } } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s - \big ( \log \beta _ { 1 - t } - \log \beta _ { 1 - t ^ { \prime } } \big ) , } \end{array}
$$

which means that

$$
\begin{array} { r l } & { \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \big ( \kappa _ { 1 - s } + \frac { \sigma ( 1 - s ) ^ { 2 } - 2 \eta _ { 1 - s } } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s \big ) = \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \sigma ( 1 - s ) ^ { 2 } } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \beta _ { 1 - t } } { \beta _ { 1 - t ^ { \prime } } } , } \\ & { \qquad \frac { \alpha _ { 1 - t ^ { \prime } } ( \sigma ( 1 - t ^ { \prime } ) ^ { 2 } - 2 \eta _ { 1 - t ^ { \prime } } ) } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } \xi _ { 1 - t ^ { \prime } } ( \vec { X } _ { t ^ { \prime } } ) = \alpha _ { 1 - t ^ { \prime } } \Big ( \frac { \sigma ( 1 - t ^ { \prime } ) ^ { 2 } } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } + \frac { \bar { \beta } _ { 1 - t ^ { \prime } } } { \beta _ { 1 - t ^ { \prime } } } - \frac { \hat { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \Big ) \xi _ { 1 - t ^ { \prime } } \big ( \vec { X } _ { t ^ { \prime } } \big ) . } \end{array}
$$

If we define $\chi ( 1 - s )$ such that $\begin{array} { r } { \sigma ^ { 2 } ( 1 - s ) = 2 \beta _ { 1 - s } \mathopen { } \mathclose \bgroup \left( \frac { \dot { \alpha } _ { 1 - s } } { \alpha _ { 1 - s } } \beta _ { 1 - s } - \dot { \beta } _ { 1 - s } \aftergroup \egroup \right) + \chi ( 1 - s ) } \end{array}$ , we obtain that

$$
\begin{array} { r l } & { \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \sigma ( 1 - s ) ^ { 2 } } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \beta _ { 1 - t } } { \beta _ { 1 - t ^ { \prime } } } = \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \big ( \frac { \dot { \alpha } _ { 1 - s } } { \alpha _ { 1 - s } } - \frac { \dot { \beta } _ { 1 - s } } { \beta _ { 1 - s } } + \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s \big ) \frac { \beta _ { 1 - t } } { \beta _ { 1 - t ^ { \prime } } } } \\ & { = \exp \big ( \int _ { t ^ { \prime } } ^ { t } \big ( \partial _ { s } \log \alpha _ { 1 - s } - \partial _ { s } \log \beta _ { 1 - s } - \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \big ) \mathrm { d } s \big ) \frac { \beta _ { 1 - t } } { \beta _ { 1 - t ^ { \prime } } } = \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } , } \\ &  \alpha _ { 1 - t ^ { \prime } } \big ( \frac { \sigma ( 1 - t ^ { \prime } ) ^ { 2 } } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } + \frac { \dot { \beta } _ { 1 - t ^ { \prime } } } { \beta _ { 1 - t ^ { \prime } } } - \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \big ) \xi _ { 1 - t ^ { \prime } } \big ( \vec { X } _ { t ^ { \prime } } \big ) = \frac { \alpha _ { 1 - t ^ { \prime } } \chi ( 1 - t ^ { \prime } ) }  2 \end{array}
$$

If we plug equations (148)-(149) into (146)-(147), and then those into (143), we obtain that

$$
\begin{array} { r } { \vec { X } _ { t } = \vec { X } _ { 0 } \exp \big ( - \int _ { 0 } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \alpha _ { 1 - t } } { \alpha _ { 1 } } + \alpha _ { 1 - t } \int _ { 0 } ^ { t } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \chi ( 1 - t ^ { \prime } ) } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } \xi _ { 1 - t ^ { \prime } } \big ( \vec { X } _ { t ^ { \prime } } \big ) \mathrm { d } t ^ { \prime } } \\ { + \int _ { 0 } ^ { t } \big ( 2 \beta _ { 1 - t ^ { \prime } } \big ( \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \beta _ { 1 - t ^ { \prime } } - \dot { \beta } _ { 1 - t ^ { \prime } } \big ) + \chi ( 1 - t ^ { \prime } ) \big ) \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \mathrm { d } B _ { t ^ { \prime } } . } \end{array}
$$

and if we take the limit $t  1 ^ { - }$ and use that $\alpha _ { 1 } = 1$ ,

$$
\begin{array} { r l } & { = \vec { X } _ { 0 } \big ( \operatorname* { l i m } _ { t \to 1 ^ { - } } \exp \big ( - \int _ { 0 } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \operatorname { d } s \big ) \alpha _ { 1 - t } \big ) + \operatorname* { l i m } _ { t \to 1 ^ { - } } \alpha _ { 1 - t } \int _ { 0 } ^ { t } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \operatorname { d } s \big ) \frac { \chi ( 1 - t ^ { \prime } ) } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } \xi _ { 1 - t ^ { \prime } } \big ( } \\ & { \qquad + \operatorname* { l i m } _ { t \to 1 ^ { - } } \int _ { 0 } ^ { t } \big ( 2 \beta _ { 1 - t ^ { \prime } } \big ( \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \beta _ { 1 - t ^ { \prime } } - \dot { \beta } _ { 1 - t ^ { \prime } } \big ) + \chi ( 1 - t ^ { \prime } ) \big ) \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \operatorname { d } s \big ) \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \operatorname { d } B _ { t ^ { \prime } } . } \end{array}
$$

The assumption on $\chi$ in (25) is equivalent, up to a rearrangement of the notation and a flip in the time variable, to the statement that for all $t ^ { \prime } \in [ 0 , 1 )$ ,

$$
\begin{array} { r } { \operatorname* { l i m } _ { t \to 1 ^ { - } } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \alpha _ { 1 - t } = 0 . } \end{array}
$$

Hence, under assumption (25), the factor accompanying $\vec { X _ { 0 } }$ in equation (151) is zero. Moreover, this assumption also implies that

$$
\begin{array} { r l } & { \operatorname* { l i m } _ { t \to 1 ^ { - } } \alpha _ { 1 - t } \int _ { 0 } ^ { t } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } { \mathrm { d } } s \big ) \frac { \chi ( 1 - t ^ { \prime } ) } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } \xi _ { 1 - t ^ { \prime } } ( \vec { X } _ { t ^ { \prime } } ) { \mathrm { d } } t ^ { \prime } } \\ & { = \int _ { 0 } ^ { 1 } \big ( \operatorname* { l i m } _ { t \to 1 ^ { - } } \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } { \mathrm { d } } s \big ) \alpha _ { 1 - t } \big ) \frac { \chi ( 1 - t ^ { \prime } ) } { 2 \beta _ { 1 - t ^ { \prime } } ^ { 2 } } \xi _ { 1 - t ^ { \prime } } ( \vec { X } _ { t ^ { \prime } } ) { \mathrm { d } } t ^ { \prime } = 0 . } \end{array}
$$

If we plug (152) and (153) into (151), we obtain that

$$
\begin{array} { r } { \vec { X } _ { 1 } = \operatorname* { l i m } _ { t  1 ^ { - } } \int _ { 0 } ^ { t } ( 2 \beta _ { 1 - t ^ { \prime } } ( \frac { \dot { \alpha } _ { 1 - t ^ { \prime } } } { \alpha _ { 1 - t ^ { \prime } } } \beta _ { 1 - t ^ { \prime } } - \dot { \beta } _ { 1 - t ^ { \prime } } ) + \chi ( 1 - t ^ { \prime } ) ) \exp \big ( - \int _ { t ^ { \prime } } ^ { t } \frac { \chi ( 1 - s ) } { 2 \beta _ { 1 - s } ^ { 2 } } \mathrm { d } s \big ) \frac { \alpha _ { 1 - t } } { \alpha _ { 1 - t ^ { \prime } } } \mathrm { d } B _ { t ^ { \prime } } , } \end{array}
$$

which shows that $\vec { X _ { 1 } }$ is independent of $\vec { X _ { 0 } }$ . Next, we leverage that $\vec { X }$ and $\pmb { X }$ have equal distributions over trajectories (Proposition 4). In particular, the joint distribution of $( \vec { X _ { 0 } } , \vec { X _ { 1 } } )$ is equal to the joint distribution of $( X _ { 1 } , X _ { 0 } )$ . We conclude that $X _ { 1 }$ and $X _ { 0 }$ are independent, which is the definition of the memorylessness property. Hence, the assumption (25) is sufficient for memorylessness to hold.

It remains to prove that the assumption (25) is necessary. Looking at equation (150) we deduce that generally, for any $t \in [ 0 , 1 )$ , $\vec { X _ { 0 } }$ and $\vec { X _ { t } }$ are not independent, because the first two terms in (150) are different from zero. Thus, if there existed a $t ^ { \prime } \in [ 0 , 1 )$ such that the limit (152) is different from zero, then $\vec { X _ { 1 } }$ would not be independent from $\vec { X } _ { t ^ { \prime } }$ , which means that in general it would not be independent of $\vec { X _ { 0 } }$ either.

# D.2 Proof of Theorem 1: fine-tuning recipe for general noise schedules

The proof of this result relies heavily on the properties of the Hamilton-Jacobi-Bellman equation:

Theorem 3 (Hamilton-Jacobi-Bellman equation). If we define the infinitesimal generator

$$
\begin{array} { r } { \mathcal L : = \frac { 1 } { 2 } \sum _ { i , j = 1 } ^ { d } ( { \sigma } { \sigma } ^ { \top } ) _ { i j } ( t ) \partial _ { x _ { i } } \partial _ { x _ { j } } + \sum _ { i = 1 } ^ { d } b _ { i } ( x , t ) \partial _ { x _ { i } } , } \end{array}
$$

the value function $V$ for the $S O C$ problem (12)-(13) solves the following Hamilton-Jacobi-Bellman (HJB) partial differential equation:

$$
\begin{array} { r l } & { \partial _ { t } V ( x , t ) = - \mathcal { L } V ( x , t ) + \frac { 1 } { 2 } \| ( \sigma ^ { \top } \nabla V ) ( x , t ) \| ^ { 2 } - f ( x , t ) , } \\ & { V ( x , T ) = g ( x ) . } \end{array}
$$

Consider forward SDEs like (63), starting from the distributions $p ^ { \mathrm { b a s e } }$ and $p ^ { * }$ , where $p ^ { * } ( x ) \propto p ^ { \mathrm { b a s e } } ( x ) \exp ( r ( x ) )$

$$
\begin{array} { r l r } & { \mathrm { d } \vec { X } _ { t } = \vec { b } ( \vec { X } _ { t } , t ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , } & { \vec { X } _ { 0 } \sim p ^ { \mathrm { b a s e } } , } \\ & { \mathrm { d } \vec { X } _ { t } ^ { * } = \vec { b } ^ { * } ( \vec { X } _ { t } ^ { * } , t ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , } & { \vec { X } _ { 0 } \sim p ^ { * } . } \end{array}
$$

where the drifts are defined as

$$
\begin{array} { r l } & { \Vec { b } ( x , t ) = - \kappa _ { 1 - t } x + \big ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \big ) \mathfrak { s } ( x , 1 - t ) = - \kappa _ { 1 - t } x + \big ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \big ) \nabla \log { \Vec { p _ { t } } } ( x ) , } \\ & { \Vec { b ^ { * } } ( x , t ) = - \kappa _ { 1 - t } x + \big ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \big ) \mathfrak { s ^ { * } } ( x , 1 - t ) = - \kappa _ { 1 - t } x + \big ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } - \eta _ { 1 - t } \big ) \nabla \log { \Vec { p _ { t } ^ { * } } ( x ) } , } \end{array}
$$

and $\vec { p _ { t } } , \vec { p _ { t } ^ { * } }$ are the densities of $X _ { t }$ , $\vec { X _ { t } }$ , respectively. $\vec { p _ { t } }$ , $\vec { p } _ { t } ^ { * }$ satisfy Fokker-Planck equations:

$$
\begin{array} { r } { \partial _ { t } \vec { p _ { t } } = \nabla \cdot ( \vec { b } ( x , t ) \vec { p _ { t } } ) + \nabla \cdot \bigl ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } \nabla \vec { p _ { t } } \bigr ) , \qquad \vec { p _ { 0 } } = p ^ { \mathrm { b a s e } } , } \\ { \partial _ { t } \vec { p _ { t } ^ { * } } = \nabla \cdot ( \vec { b ^ { * } } ( x , t ) \vec { p _ { t } ^ { * } } ) + \nabla \cdot \bigl ( \frac { \sigma ( 1 - t ) ^ { 2 } } { 2 } \nabla \vec { p _ { t } ^ { * } } \bigr ) , \qquad \vec { p _ { 0 } } = p ^ { * } . } \end{array}
$$

Plugging (159) into (160), we obtain

$$
\begin{array} { r l r } & { } & { \partial _ { t } \vec { p _ { t } } = \nabla \cdot \bigl ( \kappa _ { 1 - t } x \vec { p _ { t } } \bigr ) + \nabla \cdot \bigl ( \eta _ { 1 - t } \nabla \vec { p _ { t } } \bigr ) , \qquad \vec { p _ { 0 } } = p ^ { \mathrm { b a s e } } , } \\ & { } & { \partial _ { t } \vec { p _ { t } } = \nabla \cdot \bigl ( \kappa _ { 1 - t } x \vec { p _ { t } ^ { * } } \bigr ) + \nabla \cdot \bigl ( \eta _ { 1 - t } \nabla \vec { p _ { t } ^ { * } } \bigr ) , \qquad \vec { p _ { 0 } } = p ^ { * } . } \end{array}
$$

We apply the Hopf-Cole transformation to obtain PDEs for $- \log \vec { p _ { t } }$ (and $- \log { \vec { p _ { t } } }$ analogously):

$$
\begin{array} { r l } & { - \partial _ { t } ( - \log \vec { p _ { t } } ) = \frac { \partial _ { t } p _ { t } } { p _ { t } } = \frac { \nabla \cdot ( \kappa _ { 1 - t } x \vec { p } _ { t } ) + \nabla \cdot \big ( \eta _ { 1 - t } \nabla \vec { p } _ { t } \big ) } { p _ { t } } } \\ & { \qquad = \kappa _ { 1 - t } \nabla \cdot x + \kappa _ { 1 - t } \langle x , \nabla \log \vec { p _ { t } } \rangle + \eta _ { 1 - t } \frac { \nabla \cdot ( \nabla \log \vec { p _ { t } } \exp ( \log p _ { t } ) ) } { p _ { t } } } \\ & { \qquad = \kappa _ { 1 - t } d + \kappa _ { 1 - t } \langle x , \nabla \log \vec { p _ { t } } \rangle + \eta _ { 1 - t } \big ( \Delta \log \vec { p _ { t } } + \| \nabla \log \vec { p _ { t } } \| ^ { 2 } \big ) . } \end{array}
$$

Hence, if we define $\mathcal { V } ( x , t ) = - \log \vec { p _ { t } } ( x )$ , $\psi ^ { * } ( x , t ) = - \log \vec { p } _ { t } ^ { * } ( x )$ , then $\mathcal { V }$ and $\mathcal { V } ^ { * }$ satisfy the following Hamilton-Jacobi-Bellman equations:

$$
\begin{array} { r l } & { \quad - \partial _ { t } \mathcal { V } = \kappa _ { 1 - t } d - \kappa _ { 1 - t } \langle x , \nabla \mathcal { V } \rangle + \eta _ { 1 - t } \big ( - \Delta \mathcal { V } + \| \nabla \mathcal { V } \| ^ { 2 } \big ) , \qquad \mathcal { V } ( x , 0 ) = - \log p ^ { \mathrm { b a s e } } ( x ) , } \\ & { \quad - \partial _ { t } \mathcal { V } ^ { * } = \kappa _ { 1 - t } d - \kappa _ { 1 - t } \langle x , \nabla \mathcal { V } ^ { * } \rangle + \eta _ { 1 - t } \big ( - \Delta \mathcal { V } ^ { * } + \| \nabla \mathcal { V } ^ { * } \| ^ { 2 } \big ) , \qquad \mathcal { V } ^ { * } ( x , 0 ) = - \log p ^ { * } ( x ) . } \end{array}
$$

Now, define $\hat { \mathcal { V } } ( x , t ) = \mathcal { V } ^ { * } ( x , t ) - \mathcal { V } ( x , t )$ . Subtracting (164) from (163), we obtain

$$
\begin{array} { r l } & { \begin{array} { r l } { - \partial _ { t } \hat { \mathcal { V } } = - \kappa _ { 1 - t } \langle x , \nabla \hat { \mathcal { V } } \rangle + \eta _ { 1 - t } \big ( - \Delta \hat { \mathcal { V } } + \| \nabla \mathcal { V } ^ { * } \| ^ { 2 } - \| \nabla \mathcal { V } \| ^ { 2 } \big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = - \kappa _ { 1 - t } \langle x , \nabla \hat { \mathcal { V } } \rangle + \eta _ { 1 - t } \big ( - \Delta \hat { \mathcal { V } } + \| \nabla ( \hat { \mathcal { V } } + \mathcal { V } ) \| ^ { 2 } - \| \nabla \mathcal { V } \| ^ { 2 } \big ) } \end{array} } \\ & { \quad \quad \quad \quad \quad \quad \quad = - \kappa _ { 1 - t } \langle x , \nabla \hat { \mathcal { V } } \rangle + \eta _ { 1 - t } \big ( - \Delta \hat { \mathcal { V } } + \| \nabla \hat { \mathcal { V } } \| ^ { 2 } + 2 \langle \nabla \mathcal { V } , \nabla \hat { \mathcal { V } } \rangle \big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \big \langle - \kappa _ { 1 - t } x + 2 \eta _ { 1 - t } \nabla \mathcal { V } , \nabla \hat { \mathcal { V } } \rangle + \eta _ { 1 - t } \big ( - \Delta \hat { \mathcal { V } } + \| \nabla \hat { \mathcal { V } } \| ^ { 2 } \big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \big \langle - \kappa _ { 1 - t } x - 2 \eta _ { 1 - t } \mathfrak { s } ( x , 1 - t ) , \nabla \hat { \mathcal { V } } \big \rangle + \eta _ { 1 - t } \big ( - \Delta \hat { \mathcal { V } } + \| \nabla \hat { \mathcal { V } } \| ^ { 2 } \big ) , } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

Hence, $\hat { \mathcal { V } }$ also satisfies a Hamilton-Jacobi-Bellman equation. If we define $V$ such that $\hat { \mathcal { V } } ( x , t ) = V ( x , 1 - t )$ , we have that

$$
\begin{array} { r l r } { \mathbf { \sigma } } & { = \langle - \kappa _ { t } x - 2 \eta _ { t } \mathbf { s } ( x , t ) , \nabla V \rangle + \eta _ { t } \big ( - \Delta V + \| \nabla V \| ^ { 2 } \big ) , } & { } & { V ( x , 1 ) = r ( x ) - \log \big ( \int p ^ { \mathrm { b a s e } } ( y ) \exp ( r ( x ) ) \big ) \int _ { 0 } ^ { \infty } \mathbf { \eta } } \end{array}
$$

Using Theorem 3, we can reverse-engineer $V$ as the value function of the following SOC problem:

$$
\begin{array} { r l } & { \underset { u \in \mathcal { U } } { \operatorname* { m i n } } \mathbb { E } \big [ \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \mathrm { d } t - r ( x ) + \log \big ( \int p ^ { \mathrm { b a s e } } ( y ) \exp ( r ( y ) ) \mathrm { d } y \big ) \big ] , } \\ & { \mathrm { s . t . ~ d } X _ { t } ^ { u } = \big ( \kappa _ { t } x + 2 \eta _ { t } \mathfrak { s } ( x , t ) + \sqrt { 2 \eta _ { t } } u ( X _ { t } ^ { u } , t ) \big ) \mathrm { d } t + \sqrt { 2 \eta _ { t } } \mathrm { d } B _ { t } , \qquad X _ { 0 } ^ { u } \sim p _ { 0 } . } \end{array}
$$

Note that this SOC problem is equal to the problem (12)-(13) with the choices $f = 0$ , $g = - r$ , and $\sigma ( t ) = \sqrt { 2 \eta _ { t } }$ . By equation (17), the optimal control of the problem (167)-(168) is of the form:

$$
\begin{array} { r l } & { u ^ { * } ( x , t ) = - \sqrt { 2 \eta _ { t } } \nabla V ( x , t ) = - \sqrt { 2 \eta _ { t } } \nabla \hat { \psi } ( x , 1 - t ) = - \sqrt { 2 \eta _ { t } } \big ( \nabla \mathcal { V } ^ { * } ( x , 1 - t ) - \nabla \mathcal { V } ( x , 1 - t ) \big ) } \\ & { \qquad = - \sqrt { 2 \eta _ { t } } \big ( - \nabla \log \bar { p } _ { 1 - t } ^ { * } ( x ) + \nabla \log \bar { p } _ { 1 - t } ( x ) \big ) = \sqrt { 2 \eta _ { t } } \big ( \mathfrak { s } ^ { * } ( x , t ) - \mathfrak { s } ( x , t ) \big ) , } \\ & { \qquad \Longleftrightarrow \mathfrak { s } ^ { * } ( x , t ) = \mathfrak { s } ( x , t ) + u ^ { * } ( x , t ) / \sqrt { 2 \eta _ { t } } . } \end{array}
$$

As in (64), the backward SDEs corresponding to the forward SDEs (158) take the following form:

$$
\begin{array} { r } { \mathrm { d } X _ { t } ^ { * } = \bigl ( \kappa _ { t } X _ { t } ^ { * } + \bigl ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \bigr ) \mathfrak { s } ^ { * } ( X _ { t } ^ { * } , t ) \bigr ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } ^ { * } \sim N ( 0 , I ) . } \end{array}
$$

If we plug (170) into this equation, we obtain

$$
\begin{array} { r l } & { \mathrm { d } X _ { t } ^ { * } = \bigl ( \kappa _ { t } X _ { t } ^ { * } + \bigl ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \bigr ) \bigl ( \mathfrak { s } ( X _ { t } ^ { * } , t ) + \frac { u ^ { * } ( X _ { t } ^ { * } , t ) } { \sqrt { 2 \eta _ { t } } } \bigr ) \bigr ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } ^ { * } \sim N ( 0 , I ) , } \\ { \iff \mathrm { d } X _ { t } ^ { * } = \bigl ( b ( X _ { t } ^ { * } , t ) + \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } } { \sqrt { 2 \eta _ { t } } } u ^ { * } ( X _ { t } ^ { * } , t ) \bigr ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } ^ { * } \sim N ( 0 , I ) . } \end{array}
$$

where we used that $\begin{array} { r } { b ( x , t ) = \kappa _ { t } x + \big ( \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } \big ) \mathfrak { s } ( x , t ) } \end{array}$ by definition in equation (11).

The fine-tuned inference $S D E$ for DDIM Now, for DDIM, we have that $\begin{array} { r } { u ^ { * } ( x , t ) = - \sqrt { \frac { \dot { \alpha } _ { t } } { \alpha _ { t } ( 1 - \alpha _ { t } ) } } ( \epsilon ^ { * } ( x , t ) - } \end{array}$ $\epsilon ^ { \mathrm { b a s e } } ( x , t ) )$ by (26). Hence,

$$
\begin{array} { r l } & { \frac { 2 } { 2 \eta _ { t } } u ^ { * } ( x , t ) = - \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \frac { \lambda \epsilon _ { t } } { 2 \alpha _ { t } } } { \sqrt { \frac { \lambda \epsilon _ { t } } { \alpha _ { t } } } } \sqrt { \frac { \dot { \alpha } _ { t } } { \alpha _ { t } ( 1 - \alpha _ { t } ) } } ( \epsilon ^ { * } ( x , t ) - \epsilon ^ { \mathrm { b a s e } } ( x , t ) ) = - \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } } { \sqrt { 1 - \alpha _ { t } } } ( \epsilon ^ { * } ( x , t ) - \epsilon ^ { \mathrm { b a s e } } ( x , t ) ) } \\ & { \Rightarrow b ( x , t ) + \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } } { \sqrt { 2 \eta _ { t } } } u ^ { * } ( x , t ) = \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } X _ { t } - \left( \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } + \frac { \sigma ( t ) ^ { 2 } } { 2 } \right) \frac { \epsilon ^ { \mathrm { b a s e } } ( X _ { t } , t ) } { \sqrt { 1 - \alpha _ { t } } } - \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } } { \sqrt { 1 - \alpha _ { t } } } ( \epsilon ^ { * } ( x , t ) - \epsilon ^ { \mathrm { b a s e } } ( x , t ) } \\ & { \qquad = \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } X _ { t } - \left( \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } + \frac { \sigma ( t ) ^ { 2 } } { 2 } \right) \frac { \epsilon ^ { * } ( X _ { t } , t ) } { \sqrt { 1 - \alpha _ { t } } } . } \end{array}
$$

We obtain that the fine-tuned inference SDE for DDIM is

$$
\begin{array} { r } { \begin{array} { r } { \mathrm { d } X _ { t } ^ { * } = \big ( \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } X _ { t } ^ { * } - \big ( \frac { \dot { \alpha } _ { t } } { 2 \alpha _ { t } } + \frac { \sigma ( t ) ^ { 2 } } { 2 } \big ) \frac { \epsilon ^ { * } ( X _ { t } ^ { * } , t ) } { \sqrt { 1 - \alpha _ { t } } } \big ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } ^ { * } \sim N ( 0 , I ) , } \end{array} } \end{array}
$$

which is matches the SDE (6) with the choice $\epsilon = \epsilon ^ { * }$ .

The fine-tuned inference SDE for Flow Matching For Flow Matching, we have that $\begin{array} { r } { u ^ { * } ( x , t ) = \sqrt { \frac { 2 } { \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) } } ( v ^ { * } ( x , t ) - } \end{array}$ $v ^ { \mathrm { b a s e } } ( x , t ) )$ by (27). Hence,

$$
\begin{array} { r l } & { \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } } { \sqrt { 2 \eta _ { t } } } u ^ { * } ( x , t ) = \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \beta _ { t } ( \frac { \sin _ { t } \beta _ { t } - \beta _ { t } } { \sin \beta _ { t } } ) } { \sqrt { 2 \beta _ { t } } ( \frac { \sin _ { t } \beta _ { t } - \beta _ { t } } { \sin \beta _ { t } } ) } \sqrt { \frac { 2 } { \beta _ { t } ( \frac { \sin _ { t } \beta _ { t } - \beta _ { t } } { \cos \beta _ { t } } ) } } ( v ^ { * } ( x , t ) - v ^ { \mathrm { b a s e } } ( x , t ) ) } \\ & { \qquad = \big ( 1 + \frac { \sigma ( t ) ^ { 2 } } { 2 \beta _ { t } ( \frac { \sin _ { t } \beta _ { t } - \beta _ { t } } { \cos \beta _ { t } } ) } \big ) ( v ^ { * } ( x , t ) - v ^ { \mathrm { b a s e } } ( x , t ) ) . } \\ & { \implies b ( x , t ) + \frac { \frac { \sigma ( t ) ^ { 2 } } { 2 } + \eta _ { t } } { \sqrt { 2 \eta _ { t } } } u ^ { * } ( x , t ) = v ^ { \mathrm { b a s e } } ( x , t ) + \frac { \sigma ( t ) ^ { 2 } } { 2 \beta _ { t } ( \frac { \sin _ { t } \beta _ { t } } { \cos \beta _ { t } } ) } \big ( v ^ { \mathrm { b a s e } } ( x , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x \big ) } \\ & { \qquad + \big ( 1 + \frac { \sigma ( t ) ^ { 2 } } { 2 \beta _ { t } ( \frac { \cos _ { t } \beta _ { t } } { \cos \beta _ { t } } ) } \big ) ( v ^ { * } ( x , t ) - v ^ { \mathrm { b a s e } } ( x , t ) ) } \\ &  \qquad = v ^ { * } ( x , t ) + \frac { \sigma ( t ) ^ { 2 } }  2 \beta _ { t } ( \frac  \end{array}
$$

We obtain that the fine-tuned inference SDE for Flow Matching is

$$
\begin{array} { r } { \mathrm { d } X _ { t } ^ { * } = \big ( v ( X _ { t } ^ { * } , t ) + \frac { \sigma ( t ) ^ { 2 } } { 2 \beta _ { t } ( \frac { \alpha _ { t } } { \alpha _ { t } } \beta _ { t } - \bar { \beta } _ { t } ) } \big ( v ^ { * } ( X _ { t } ^ { * } , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } X _ { t } ^ { * } \big ) \big ) \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } , \qquad X _ { 0 } ^ { * } \sim N ( 0 , I ) , } \end{array}
$$

which matches equation (4) with the choice $v = v ^ { * }$ .

# E Loss function derivations

# E.1 Derivation of the Continuous Adjoint method

Proposition 6. The gradient $\textstyle { \frac { \mathrm { d } { \mathcal { L } } } { \mathrm { d } \theta } }$ of the adjoint loss $\mathcal { L } ( u ; X )$ defined in (28) with respect to the parameters $\theta$ of the control can be expressed as in (32).

Proof. First, note that we can write

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathbb { E } \big [ \int _ { 0 } ^ { T } \big ( \frac { 1 } { 2 } \| u _ { \theta } ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ] } \\ & { = \mathbb { E } \big [ \int _ { 0 } ^ { T } \nabla _ { \theta } u _ { \theta } ( X _ { t } ^ { u _ { \theta } } , t ) u _ { \theta } ( X _ { t } ^ { u _ { \theta } } , t ) \mathrm { d } t \big ] + \nabla _ { \theta } \mathbb { E } \big [ \int _ { 0 } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ] \big | _ { v = \mathrm { s t } } } \end{array}
$$

To develop the second term, we apply Lemma 5. Namely, by the Leibniz rule and equation (185), we have that

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathbb { E } \big [ \int _ { 0 } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ] \big | _ { v = \mathrm { s t o p g r a d } ( u _ { \theta } ) } } \\ & { \ = \mathbb { E } \big [ \nabla _ { \theta } \big ( \int _ { 0 } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ) \big | _ { v = \mathrm { s t o p g r a d } ( u _ { \theta } ) } \big ] } \\ & { \ = \mathbb { E } \big [ \int _ { 0 } ^ { T } ( \nabla _ { \theta } u _ { \theta } ) \big ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t \big ) ^ { \top } \sigma ( t ) ^ { \top } a _ { t } ( \omega ) \mathrm { d } t \big ] . } \end{array}
$$

Plugging the right-hand side of this equation into (180) concludes the proof.

Lemma 5. Let v be an arbitrary fixed vector field. The unique solution of the ODE

$$
\begin{array} { r l } & { \overline { { t } } ^ { a } ( t ; \mathbf { X } ^ { u } , u ) = - \left[ \left( \nabla _ { X _ { t } ^ { u } } \big ( b ( X _ { t } ^ { u } , t ) + \sigma ( t ) u ( X _ { t } ^ { u } , t ) \big ) \right) ^ { \top } a ( t ; \mathbf { X } ^ { u } , u ) + \nabla _ { X _ { t } ^ { u } } \left( f ( X _ { t } ^ { u } , t ) + \frac { 1 } { 2 } \| v ( X _ { t } ^ { u } , t ) \| ^ { 2 } \right) \right] } \\ & { } \\ & { a ( 1 ; \mathbf { X } ^ { u } , u ) = \nabla g ( X _ { 1 } ^ { u } ) , } \end{array}
$$

satisfies:

$$
\begin{array} { r l } & { a ( t ; \mathbf { } X ^ { u } , u ) : = \nabla _ { X _ { t } ^ { u } } \big ( \int _ { t } ^ { 1 } \big ( \frac { 1 } { 2 } \| u ( X _ { t ^ { \prime } } ^ { u } , t ^ { \prime } ) \| ^ { 2 } + f ( X _ { t ^ { \prime } } ^ { u } , t ^ { \prime } ) \big ) \mathrm { d } t ^ { \prime } + g ( X _ { 1 } ^ { u } ) \big ) , } \\ & { w h e r e \ X ^ { u } \ s o l v e s \mathrm { ~ d } X _ { t } ^ { u } = \big ( b ( X _ { t } ^ { u } , t ) + \sigma ( t ) u ( X _ { t } ^ { u } , t ) \big ) \ \mathrm { d } t + \sigma ( t ) \mathrm { d } B _ { t } . } \end{array}
$$

Moreover, when $u = u _ { \theta }$ is parameterized by $\theta$ we have that

$$
\begin{array} { r } { \nabla _ { \theta } \big ( \int _ { 0 } ^ { T } \big ( \frac 1 2 \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ) = \int _ { 0 } ^ { T } ( \nabla _ { \theta } u _ { \theta } ) ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \sigma ( t ) ^ { \top } a _ { t } ( \omega ) \mathrm { d } t . } \end{array}
$$

Proof. We use an approach based on Lagrange multipliers which mirrors and extends the derivation of the adjoint ODE (Domingo-Enrich et al., 2023, Lemma 8). For shortness, we use the notation $\tilde { b } _ { \theta } ( x , t ) : =$ $b ( x , t ) + \sigma ( t ) u _ { \theta } ( x , t )$ . Define a process $a : \Omega \times [ 0 , T ]   { \mathbb { R } ^ { d } }$ such that for any $\omega \in \Omega$ , $a ( \omega , \cdot )$ is differentiable. For a given $\omega \in \Omega$ , we can write

$$
\begin{array} { r l } & { \int _ { 0 } ^ { T } \left( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \right) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) } \\ & { = \int _ { 0 } ^ { T } \left( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \right) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) } \\ & { \qquad - \int _ { 0 } ^ { T } \langle a _ { t } ( \omega ) , ( d X _ { t } ^ { u _ { \theta } } ( \omega ) - \tilde { b } _ { \theta } ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \mathrm { d } t - \sigma ( t ) \mathrm { d } B _ { t } ) \rangle . } \end{array}
$$

By stochastic integration by parts (Domingo-Enrich et al., 2023, Lemma 9), we have that

$$
\begin{array} { r } { \int _ { 0 } ^ { T } \langle a _ { t } ( \omega ) , d X _ { t } ^ { u _ { \theta } } ( \omega ) \rangle = \langle a _ { T } ( \omega ) , X _ { T } ^ { u _ { \theta } } ( \omega ) \rangle - \langle a _ { 0 } ( \omega ) , X _ { 0 } ^ { u _ { \theta } } ( \omega ) \rangle - \int _ { 0 } ^ { T } \langle X _ { t } ^ { u _ { \theta } } ( \omega ) , \frac { d a _ { t } } { d t } ( \omega ) \rangle \mathrm { d } t . } \end{array}
$$

Hence, if $X _ { 0 } ^ { u _ { \theta } } = x _ { 0 }$ is the initial condition, we have that9

$$
\begin{array} { r l } & { \nabla _ { x v } \left( \int _ { 0 } ^ { T } \left( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \right) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \right) } \\ & { = \nabla _ { x } \left( \int _ { 0 } ^ { T } \left( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \right) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \right. } \\ & { \qquad - \left. u _ { T } ( \omega ) , X _ { T ^ { u _ { \theta } } } ^ { u _ { \theta } } ( \omega ) \right. + \left. a _ { 0 } ( \omega ) , X _ { 0 } ^ { u _ { \theta } } ( \omega ) \right. + \int _ { 0 } ^ { T } \left( \langle a _ { t } ( \omega ) , \bar { b } _ { \theta } ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \rangle + \left. \frac { d a _ { t } } { d t } ( \omega ) , X _ { t } ^ { u _ { \theta } } ( \omega ) \right. \mathrm { d } t \right. } \\ & { \qquad \left. + \int _ { 0 } ^ { T } \langle a _ { t } ( \omega ) , \sigma ( t ) \mathrm { d } B _ { t } \rangle \right) } \\ & { = \int _ { 0 } ^ { T } \nabla _ { x _ { a } } X _ { t } ^ { u _ { \theta } } ( \omega ) ^ { \top } \nabla _ { x } \left( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \right) \mathrm { d } t + \nabla _ { x _ { 0 } } X _ { T } ^ { u _ { \theta } } ( \omega ) ^ { \top } \nabla _ { x } g ( X _ { T } ^ { u _ { \theta } } ( \omega ) ) } \\ &  \qquad - \nabla _  x _   \end{array}
$$

In the last line we used that $\nabla _ { x _ { 0 } } X _ { 0 } ^ { u _ { \theta } } ( \omega ) = \nabla _ { x _ { 0 } } x _ { 0 } = \mathrm { I }$ . If choose $a$ such that

$$
\begin{array} { r l } & { { d a _ { t } } ( \omega ) = \big ( - \nabla _ { x } \tilde { b } _ { \theta } ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) ^ { \top } a _ { t } ( \omega ) - \nabla _ { x } \big ( \frac 1 2 \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \big ) \big ) \mathrm { d } t , } \\ & { { a _ { T } } ( \omega ) = \nabla _ { x } g ( X _ { T } ^ { u _ { \theta } } ( \omega ) ) , } \end{array}
$$

which is the ODE (182)-(183), then we obtain that

$$
\begin{array} { r } { \nabla _ { x _ { 0 } } \big ( \int _ { 0 } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ) = a _ { 0 } ( \omega ) } \end{array}
$$

Without loss of generality, this argument can be extended from $t = 0$ to an arbitrary $t \in [ 0 , 1 ]$ , which proves the first statement of the lemma.

To prove (185), we similarly write

$$
\begin{array} { r l } & { \tau _ { \theta } \Big ( \int _ { 0 } ^ { T } \big ( \frac 1 2 \vert \vert v ( X _ { t } ^ { u _ { \theta } } , t ) \vert \big \vert ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \Big ) } \\ & { = \nabla _ { \theta } \big ( \int _ { 0 } ^ { T } \big ( \frac 1 2 \vert v ( X _ { t } ^ { u _ { \theta } } , t ) \vert ^ { 2 } \big ) ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) } \\ & { \qquad - \langle a _ { T } ( \omega ) , X _ { t } ^ { u _ { \theta } } ( \omega ) \rangle + \langle a _ { 0 } ( \omega ) , X _ { 0 } ^ { u _ { \theta } } ( \omega ) \rangle + \int _ { 0 } ^ { T } \big ( \langle a _ { t } ( \omega ) , \tilde { b } _ { \theta } ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \rangle + \langle \frac { d a _ { t } } { d t } ( \omega ) , X _ { t } ^ { u _ { \theta } } ( \omega ) \rangle \big ) } \\ & { \qquad + \int _ { 0 } ^ { T } \langle a _ { t } ( \omega ) , \sigma ( t ) \mathrm { d } B _ { t } \rangle \Big ) } \\ & { = \int _ { 0 } ^ { T } \nabla _ { \theta } X _ { t } ^ { u _ { \theta } } ( \omega ) ^ { \top } \nabla _ { \nabla } \big ( \frac 1 2 \vert v ( X _ { t } ^ { u _ { \theta } } , t ) \vert \big \vert ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) \big ) \mathrm { d } t + \nabla _ { \theta } X _ { T } ^ { u _ { \theta } } ( \omega ) ^ { \top } \nabla _ { \mathbf { x } } g ( X _ { T } ^ { u _ { \theta } } ( \omega ) ) } \\ &  \qquad - \nabla _ { \theta } X _ { t } ^ { u _ { \theta } } ( \omega \end{array}
$$

In the last line we used that $\nabla _ { \theta } X _ { 0 } ^ { u _ { \theta } } ( \omega ) = \nabla _ { \theta } x = 0$ . When $a$ satisfies (189), we obtain that

$$
\begin{array} { r l } & { \nabla _ { \theta } \big ( \int _ { 0 } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { u _ { \theta } } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u _ { \theta } } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u _ { \theta } } ) \big ) } \\ & { \ = \int _ { 0 } ^ { T } ( \nabla _ { \theta } \tilde { b } _ { \theta } ) ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) a _ { t } ( \omega ) \mathrm { d } t = \int _ { 0 } ^ { T } ( \nabla _ { \theta } u _ { \theta } ) ( X _ { t } ^ { u _ { \theta } } ( \omega ) , t ) ^ { \top } \sigma ( t ) ^ { \top } a _ { t } ( \omega ) \mathrm { d } t . } \end{array}
$$

The last equality holds because $\tilde { b } _ { \theta } ( x , t ) : = b ( x , t ) + \sigma ( t ) u _ { \theta } ( x , t )$ .

# E.2 Proof of Proposition 2: Theoretical guarantees of the basic Adjoint Matching loss

Let $\bar { u } = \mathsf { s t o p g r a d } ( u _ { \theta } )$ . We can rewrite equation (32) as:

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } ( u _ { \theta } ; \mathbf { X } ^ { \bar { \boldsymbol { u } } } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \nabla _ { \theta } \| u _ { \theta } ( X _ { t } ^ { \bar { \boldsymbol { u } } } , t ) \| ^ { 2 } \mathrm { d } t + \int _ { 0 } ^ { 1 } \nabla _ { \theta } u ( X _ { t } ^ { \bar { \boldsymbol { u } } } , t ) ^ { \top } \sigma ( t ) ^ { \top } a ( t ; \mathbf { X } ^ { \bar { \boldsymbol { u } } } , \bar { \boldsymbol { u } } ) \mathrm { d } t } \\ & { \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \nabla _ { \theta } \| u _ { \theta } ( X _ { t } ^ { \bar { \boldsymbol { u } } } , t ) + \sigma ( t ) ^ { \top } a ( t ; \mathbf { X } ^ { \bar { \boldsymbol { u } } } , \bar { \boldsymbol { u } } ) \| ^ { 2 } \mathrm { d } t = \nabla _ { \theta } \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ( u _ { \theta } ; \mathbf { X } ^ { \bar { \boldsymbol { u } } } ) } \end{array}
$$

This proves the first statement of the proposition. To prove that the only critical point of the expected basic Adjoint Matching loss is the optimal control, we first compute the first variation of $\mathbb { E } [ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ]$ . Letting $v : \mathbb { R } ^ { d } \times [ 0 , T ] \to \mathbb { R } ^ { d }$ be arbitrary, we have that

$$
\begin{array} { r l } & { \frac { \mathrm d } { \mathrm { d } \epsilon } \mathbb E [ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ( u + \epsilon v ; X ^ { \bar { u } } ) ] = \frac { \mathrm d } { \mathrm { d } \epsilon } \mathbb E \big [ \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| ( u + \epsilon v ) ( X _ { t } ^ { \bar { u } } , t ) + \sigma ( t ) ^ { \top } a ( t , X ^ { \bar { u } } , \bar { u } ) \| ^ { 2 } \mathrm { d } t \big ] } \\ & { = \mathbb E \big [ \int _ { 0 } ^ { T } \langle v ( X _ { t } ^ { \bar { u } } , t ) , u ( X _ { t } ^ { \bar { u } } , t ) + \sigma ( t ) ^ { \top } a ( t , X ^ { \bar { u } } , \bar { u } ) \rangle \mathrm { d } t \big ] } \\ & { = \mathbb E \big [ \int _ { 0 } ^ { T } \langle v ( X _ { t } ^ { \bar { u } } , t ) , u ( X _ { t } ^ { \bar { u } } , t ) + \sigma ( t ) ^ { \top } \mathbb E \big [ a ( t , X ^ { \bar { u } } , \bar { u } ) | X _ { t } ^ { \bar { u } } \big ] \rangle \mathrm { d } t \big ] } \\ & { \implies \frac { \delta } { \delta u } \mathbb E [ \mathcal L _ { \mathrm { B a s i c - A d j - M a t c h } } ( u ) ( x , t ) = u ( x , t ) + \mathbb E \big [ a ( t , X ^ { \bar { u } } , \bar { u } ) | X _ { t } ^ { \bar { u } } = x \big ] } \end{array}
$$

Hence, critical points satisfy that

$$
\begin{array} { r l } & { ( x , t ) = - \sigma ( t ) ^ { \top } \mathbb { E } [ a ( t , X ^ { u } , u ) | X _ { t } ^ { u } = x ] = - \sigma ( t ) ^ { \top } \mathbb { E } \big [ \nabla _ { X _ { t } ^ { v } } \int _ { t } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { v } , t ) \| ^ { 2 } + f ( X _ { t } ^ { v } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { u } , t ) } \\ & { \qquad = - \sigma ( t ) ^ { \top } \nabla _ { x } \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \frac { 1 } { 2 } \| v ( X _ { t } ^ { v } , t ) \| ^ { 2 } + f ( X _ { t } ^ { v } , t ) \big ) \mathrm { d } t + g ( X _ { T } ^ { v } ) | X _ { 0 } ^ { v } = x \big ] = - \sigma ( t ) ^ { \top } \nabla J ( u ; x , t ) , } \end{array}
$$

In this equation, the second equality holds by equation (184) from Lemma 5, and the third equality holds by the Leibniz rule.

Lemma 6 shows that any control $u$ that satisfies (196) is equal to the optimal control, which concludes the proof.

Lemma 6. Suppose that for any $x \in \mathbb { R } ^ { d }$ , $t \in [ 0 , T ]$ , $\boldsymbol { u } ( \boldsymbol { x } , t ) = - \sigma ( t ) ^ { \top } \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } ; \boldsymbol { x } , t )$ . Then, $J ( u ; \cdot , \cdot )$ satisfies the Hamilton-Jacobi-Bellman equation (156). By the uniqueness of the solution to the HJB equation, we have that $J ( u ; x , t ) = V ( x , t )$ for any $x \in \mathbb { R } ^ { d }$ , $t \in [ 0 , T ]$ . Hence, $u ( x , t ) = - \sigma ( t ) ^ { \top } \nabla _ { x } V ( x , t )$ is the optimal control.

Proof. Since $\begin{array} { r } { J ( u ; x , t ) = \mathbb { E } \big [ \int _ { t } ^ { T ^ { \prime } } \big ( \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u } , t ) \big ) d s + g ( X _ { T } ^ { u } ) | X _ { t } ^ { u } = x \big ] } \end{array}$ , we have that

$$
\begin{array} { r } { J ( u ; x , t ) = \mathbb { E } \big [ J ( u ; X _ { t + \Delta t } ^ { u } , t + \Delta t ) | X _ { t } = x \big ] + \mathbb { E } \big [ \int _ { t } ^ { t + \Delta t } \big ( \frac 1 2 \| u ( X _ { s } ^ { u } , s ) \| ^ { 2 } + f ( X _ { s } ^ { u } , s ) \big ) d s | X _ { t } = x | \big ] } \end{array}
$$

which means that

$$
0 = \frac { \mathbb { E } [ J ( u ; X _ { t + \Delta t } ^ { u } , t + \Delta t ) | X _ { t } = x ] - J ( u ; x , t ) } { \Delta t } + \frac { \mathbb { E } \big [ \int _ { t } ^ { t + \Delta t } \big ( \frac 1 2 \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u } , t ) \big ) d s | X _ { t } = x _ { t } ^ { u } - \Delta t \big ( \frac 1 2 \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u } , t ) \big ) \big ] } { \Delta t } .
$$

Recall that the generator $\mathcal { T } ^ { u }$ of the controlled SDE (13) takes the form:

$$
\begin{array} { r l } & { \mathcal { T } ^ { u } f ( x , t ) : = \operatorname* { l i m } _ { \Delta t  0 } \frac { \mathbb { E } [ f ( X _ { t + \Delta t } ^ { u } , t ) | X _ { t } = x ] - f ( x , t ) } { \Delta t } } \\ & { \quad \quad \quad = \partial _ { t } f ( x , t ) + \langle \nabla f ( x , t ) , b ( x , t ) + \sigma ( t ) u ( x , t ) \rangle + \mathrm { T r } ( \frac { \sigma ( t ) \sigma ( t ) ^ { \top } } { 2 } \nabla ^ { 2 } f ( x , t ) ) } \end{array}
$$

Hence, if we take the limit $\Delta t \to 0$ on equation (198), we obtain that:

$$
\begin{array} { r l } & { = T ^ { u } J ( u ; x , t ) + \frac { 1 } { 2 } \| u ( x , t ) \| ^ { 2 } + f ( x , t ) } \\ & { = \partial _ { t } J ( u ; x , t ) + \langle \nabla J ( u ; x , t ) , b ( x , t ) + \sigma ( t ) u ( x , t ) \rangle + \mathrm { T r } \big ( \frac { \sigma ( t ) \sigma ( t ) ^ { \top } } { 2 } \nabla ^ { 2 } J ( u ; x , t ) \big ) + \frac { 1 } { 2 } \| u ( x , t ) \| ^ { 2 } + f _ { \varepsilon } ^ { 2 } . } \end{array}
$$

Now using that $\boldsymbol { u } ( \boldsymbol { x } , t ) = - \sigma ( t ) ^ { \mathrm { ~ l ~ } } \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } ; \boldsymbol { x } , t )$ , we have that

$$
\begin{array} { r l } & { \langle \nabla J ( u ; x , t ) , \sigma ( t ) u ( x , t ) \rangle + \frac { 1 } { 2 } \| u ( x , t ) \| ^ { 2 } = - \| \sigma ( t ) ^ { \top } \nabla _ { x } J ( u ; x , t ) \| ^ { 2 } + \frac { 1 } { 2 } \| \sigma ( t ) ^ { \top } \nabla _ { x } J ( u ; x , t ) \| ^ { 2 } } \\ & { \qquad = - \frac { 1 } { 2 } \| \sigma ( t ) ^ { \top } \nabla _ { x } J ( u ; x , t ) \| ^ { 2 } . } \end{array}
$$

Plugging this back into (200), we obtain that

$$
\begin{array} { r l } & { = \partial _ { t } J ( u ; x , t ) + \langle \nabla J ( u ; x , t ) , b ( x , t ) \rangle + \operatorname { T r } \big ( \frac { \sigma ( t ) \sigma ( t ) ^ { \top } } { 2 } \nabla ^ { 2 } J ( u ; x , t ) \big ) - \frac { 1 } { 2 } \| \sigma ( t ) ^ { \top } \nabla _ { x } J ( u ; x , t ) \| ^ { 2 } + f } \end{array}
$$

And since $J ( u ; x , T ) = g ( x )$ by construction, we conclude that $J ( u ; x , t )$ satisfies the HJB equation (156).

# E.3 Theoretical guarantees of the Adjoint Matching loss

Proposition 7 (Theoretical guarantee of the Adjoint Matching loss). The only critical point of the loss $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ is the optimal control $u ^ { * }$ .

Proof. Let $v$ be an arbitrary control. If $\tilde { a } ( t ; \mathbf { X } ^ { v } )$ is the solution of the Lean Adjoint ODE (38)-(39), it satisfies the integral equation

$$
\begin{array} { r } { \tilde { a } ( t ; \mathbf { X } ^ { v } ) = \int _ { t } ^ { T } \left( \nabla _ { x } b ( X _ { s } ^ { v } , s ) ^ { \top } \tilde { a } ( s ; \mathbf { X } ^ { v } ) + \nabla _ { x } f ( X _ { s } ^ { v } , s ) \right) \mathrm { d } s + \nabla g ( X _ { T } ^ { v } ) . } \end{array}
$$

Hence,

$$
\begin{array} { r l } & { \mathbb { E } \big [ \tilde { a } ( t ; \mathbf { X } ^ { v } ) \big | X _ { t } ^ { v } \big ] = \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \nabla _ { x } b ( X _ { s } ^ { v } , s ) ^ { \top } \tilde { a } ( s ; \mathbf { X } ^ { v } ) + \nabla _ { x } f ( X _ { s } ^ { v } , s ) \big ) \mathrm { d } s + \nabla g ( X _ { T } ^ { v } ) \big | X _ { t } ^ { v } \big ] } \\ & { \qquad = \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \nabla _ { x } b ( X _ { s } ^ { v } , s ) ^ { \top } \mathbb { E } \big [ \tilde { a } ( s ; \mathbf { X } ^ { v } ) \big | X _ { s } ^ { v } \big ] + \nabla _ { x } f ( X _ { s } ^ { v } , s ) \big ) \mathrm { d } s + \nabla g ( X _ { T } ^ { v } ) \big | X _ { t } ^ { v } \big ] , } \end{array}
$$

where we used the tower property of conditional expectation in the second equality.

Similarly, if $a \left( t ; \mathbf { X } ^ { v } , v \right)$ is the solution of the Adjoint ODE (30)-(31), it satisfies the integral equation

$$
\begin{array} { r } { t ; \mathbf { X } ^ { v } , v ) = \int _ { t } ^ { T } \left( \nabla _ { x } \left( b ( X _ { s } ^ { v } , s ) ^ { \top } a ( s ; \mathbf { X } ^ { v } , v ) + \sigma ( s ) v ( X _ { s } ^ { v } , s ) \right) + \nabla _ { x } \left( f ( X _ { s } ^ { v } , s ) + \frac { 1 } { 2 } \| v ( X _ { s } ^ { v } , s ) \| ^ { 2 } \right) \right) \mathrm { d } s } \end{array}
$$

and its expected value satisfies

$$
\begin{array} { r l } & { a ( t ; \mathbf { X } ^ { v } , v ) \big | X _ { t } ^ { v } \big | } \\ & { \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \nabla _ { x } \big ( b ( X _ { s } ^ { v } , s ) + \sigma ( s ) v ( X _ { s } ^ { v } , s ) \big ) ^ { \top } a ( s ; \mathbf { X } ^ { v } , v ) + \nabla _ { x } \big ( f ( X _ { s } ^ { v } , s ) + \frac { 1 } { 2 } \| v ( X _ { s } ^ { v } , s ) \| ^ { 2 } \big ) \big ) \mathrm { d } s + \nabla g ( X _ { s } ^ { v } , s ) } \\ & { \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \nabla _ { x } \big ( b ( X _ { s } ^ { v } , s ) + \sigma ( s ) v ( X _ { s } ^ { v } , s ) \big ) ^ { \top } \mathbb { E } \big [ a ( s ; \mathbf { X } ^ { v } , v ) \big | X _ { s } ^ { v } \big ] + \nabla _ { x } \big ( f ( X _ { s } ^ { v } , s ) + \frac { 1 } { 2 } \| v ( X _ { s } ^ { v } , s ) \| ^ { 2 } \big ) \big ) \mathrm { d } s + \nabla } \end{array}
$$

Let us rewrite $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ as follows:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ( u ) ] : = \mathbb { E } \left[ \int _ { 0 } ^ { T } \left\| u ( X _ { t } ^ { v } , t ) + \sigma ( t ) ^ { \top } \mathbb { E } \big [ \tilde { a } ( t , \mathbf { X } ^ { v } ) | X _ { t } ^ { v } \big ] \right\| ^ { 2 } \mathrm { d } t \right] | _ { v = \mathrm { s t o p g r a d } ( u ) } } \\ & { \qquad + \mathbb { E } \big [ \int _ { 0 } ^ { T } \left\| \sigma ( t ) ^ { \top } \big ( \mathbb { E } \big [ \tilde { a } ( t , \mathbf { X } ^ { v } ) | X _ { t } ^ { v } \big ] - \tilde { a } ( t , \mathbf { X } ^ { v } ) \big ) \right\| ^ { 2 } \mathrm { d } t \big ] | _ { v = \mathrm { s t o p g r a d } ( u ) } , } \end{array}
$$

Now, suppose that $\hat { u }$ is a critical point of $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ . By definition, this implies that the first variation of $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ is zero. Using (207), we can write this as follows:

$$
\begin{array} { r l } & { 0 = \frac { \delta } { \delta u } \mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ( \hat { u } ) ] ( x ) = 2 \big ( \hat { u } ( x , t ) + \sigma ( t ) ^ { \top } \mathbb { E } [ \tilde { a } ( t , \mathbf { X } ^ { \hat { u } } ) | X _ { t } ^ { \hat { u } } = x ] \big ) , } \\ & { \quad \implies \hat { u } ( x , t ) = - \sigma ( t ) ^ { \top } \mathbb { E } [ \tilde { a } ( t , \mathbf { X } ^ { \hat { u } } ) | X _ { t } ^ { \hat { u } } = x ] . } \end{array}
$$

Hence, we have

$$
\begin{array} { r l } & { \nabla _ { x } \widehat { u } ( X _ { t } ^ { \widehat { n } } , t ) ^ { \top } \sigma ( t ) ^ { \top } \mathbb { E } [ \widetilde { a } ( t , { \mathbf X } ^ { \widehat { n } } ) | X _ { t } ^ { \widehat { n } } ] + \nabla _ { x } \widehat { u } ( X _ { t } ^ { \widehat { n } } , t ) ^ { \top } \widehat { u } ( X _ { t } ^ { \widehat { n } } , t ) = 0 , } \\ & { \quad \implies \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \nabla _ { x } \big ( \sigma ( s ) \widehat { u } ( X _ { s } ^ { \widehat { n } } , s ) \big ) ^ { \top } \mathbb { E } \big [ \widetilde { a } ( s ; { \mathbf X } ^ { \widehat { n } } ) \big | X _ { s } ^ { \widehat { n } } \big ] + \nabla _ { x } \big ( \frac { 1 } { 2 } \| \widehat { u } ( X _ { s } ^ { \widehat { n } } , s ) \| ^ { 2 } \big ) \big ) \mathrm { d } s \big | X _ { t } ^ { \widehat { n } } \big ] = 0 . } \end{array}
$$

If we set $v = { \hat { u } }$ in equation (204), and add (211) to its right-hand side, we obtain that $\mathbb { E } [ \tilde { a } ( t , X ^ { \hat { u } } ) | X _ { t } ^ { \hat { u } } ]$ also solves the integral equation

$$
\begin{array} { r l } & { \lvert \tilde { a } ( t ; \mathbf { X } ^ { \hat { u } } ) \rvert X _ { t } ^ { \hat { u } } \rvert } \\ & { \mathbb { E } \big [ \int _ { t } ^ { T } \big ( \nabla _ { x } \big ( b ( X _ { s } ^ { \hat { u } } , s ) + \sigma ( s ) \hat { u } ( X _ { s } ^ { \hat { u } } , s ) \big ) ^ { \top } \mathbb { E } \big [ \tilde { a } ( s ; \mathbf { X } ^ { \hat { u } } ) \big \vert X _ { s } ^ { \hat { u } } \big ] + \nabla _ { x } \big ( f ( X _ { s } ^ { \hat { u } } , s ) + \frac { 1 } { 2 } \| \hat { u } ( X _ { s } ^ { \hat { u } } , s ) \| ^ { 2 } \big ) \big ) \mathrm { d } s + \nabla _ { \xi } } \end{array}
$$

Note that this integral equation is the same one as equation (206) when we set $\ v \ = \ \hat { u }$ in the latter. Proposition 8 states that the solution of the integral equation is unique, which means that $\mathbb { E } \big [ \tilde { a } ( t ; \mathbf { X } ^ { \hat { u } } ) \big | X _ { t } ^ { \hat { u } } \big ] =$ $\mathbb { E } \big [ a ( t ; \mathbf { X } ^ { \hat { u } } , \hat { u } ) \big | X _ { t } ^ { \hat { u } } \big ]$ for all $t \in [ 0 , T ]$ .

Since we can reexpress the basic Adjoint Matching loss as

$$
\begin{array} { r l } & { [ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ( u ) ] : = \mathbb { E } [ \int _ { 0 } ^ { T } \| u ( X _ { t } ^ { v } , t ) + \sigma ( t ) ^ { \top } \mathbb { E } \big [ a ( t ; \mathbf { X } ^ { v } , v ) | X _ { t } ^ { v } \big ] \| ^ { 2 } \mathrm { d } t \big ] | _ { v = \mathrm { s t o p g r a d } ( u ) } } \\ & { \qquad + \mathbb { E } [ \int _ { 0 } ^ { T } \| \sigma ( t ) ^ { \top } \big ( \mathbb { E } \big [ a ( t ; \mathbf { X } ^ { v } , v ) | X _ { t } ^ { v } \big ] - a ( t ; \mathbf { X } ^ { v } , v ) \big ) \| ^ { 2 } \mathrm { d } t ] | _ { v = \mathrm { s t o p g r a d } ( u ) } , } \end{array}
$$

we obtain that when $\hat { u }$ is a critical point of $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ ,

$$
\begin{array} { r l } & { \frac { \mathrm { d } } { \mathrm { d } u } \mathbb { E } \big [ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ( \hat { u } ) \big ] ( x ) = 2 \big ( \hat { u } ( x , t ) + \sigma ( t ) ^ { \top } \mathbb { E } [ a ( t ; \mathbf { X } ^ { \hat { u } } , \hat { u } ) | X _ { t } ^ { \hat { u } } = x ] \big ) } \\ & { \qquad = 2 \big ( \hat { u } ( x , t ) + \sigma ( t ) ^ { \top } \mathbb { E } [ \tilde { a } ( t ; \mathbf { X } ^ { \hat { u } } ) | X _ { t } ^ { \hat { u } } = x ] \big ) = 0 , } \end{array}
$$

where the second equality holds because $\mathbb { E } \big [ \tilde { a } ( t ; \mathbf { X } ^ { \hat { a } } ) \big | X _ { t } ^ { \hat { a } } \big ] \ = \ \mathbb { E } \big [ a ( t ; \mathbf { X } ^ { \hat { a } } , \hat { u } ) \big | X _ { t } ^ { \hat { a } } \big ]$ , and the third equality holds by equation (209). Thus, we deduce that the critical points of $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ are critical points of $\mathbb { E } [ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ]$ . By Proposition 2, $\mathbb { E } [ \mathcal { L } _ { \mathrm { B a s i c - A d j - M a t c h } } ]$ has a single critical point, which is the optimal control $u ^ { * }$ , which concludes the proof of the statement for $\mathbb { E } [ \mathcal { L } _ { \mathrm { A d j - M a t c h } } ]$ . □

Proposition 8. Let $v$ be an arbitrary control. Consider the integral equation:

$$
\begin{array} { r } { \displaybreaks _ { t } = \mathbb { E } \Big [ \int _ { t } ^ { T } \left( \nabla _ { x } \left( b ( X _ { s } ^ { v } , s ) + \sigma ( s ) v ( X _ { s } ^ { v } , s ) \right) ^ { \top } Y _ { s } + \nabla _ { x } \left( f ( X _ { s } ^ { v } , s ) + \frac { 1 } { 2 } \| v ( X _ { s } ^ { v } , s ) \| ^ { 2 } \right) \right) \mathrm { d } s + \nabla g ( X _ { T } ^ { v } ) \Big | X _ { t } ^ { v } \Big ] , } \end{array}
$$

where $t \in [ 0 , T ]$ . This equation has a unique solution, i.e. if $Y ^ { 1 }$ , $Y ^ { 2 }$ are two solutions then $Y _ { 1 } = Y _ { 2 }$

Proof. Let $Y ^ { 1 }$ , $Y ^ { 2 }$ be two solutions of the integral equation. We have that

$$
\begin{array} { r } { Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 } = \mathbb { E } \big [ \int _ { t } ^ { T } \big ( ( Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } ) ^ { \top } \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big ) \mathrm { d } s \big | X _ { t } ^ { * } \big ] . } \end{array}
$$

Thus,

$$
\begin{array} { r l } & { \| Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 } \| } \\ & { \leq \mathbb { E } \big [ \big \| \int _ { t } ^ { T } \big ( ( Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } ) ^ { \top } \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big ) \mathrm { d } s \big \| \big | X _ { t } ^ { * } \big ] \leq \mathbb { E } \big [ \int _ { t } ^ { T } \big \| \big ( ( Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } ) ^ { \top } \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big ) \big \| \mathrm { d } s \big | X _ { t } ^ { * } \big ] } \\ & { \leq \mathbb { E } \big [ \int _ { t } ^ { T } \big \| Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } \big \| \cdot \big \| \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big ) \big \| \mathrm { d } s \big | X _ { t } ^ { * } \big ] = \int _ { t } ^ { T } \mathbb { E } \big [ \big \| Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } \big \| \cdot \big \| \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big ) \big \| \big | X _ { t } ^ { * } \big ] \mathrm { d } s } \\ & { \leq \int _ { t } ^ { T } \big ( \mathbb { E } \big [ \big \| Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } \big \| ^ { 2 } \big | X _ { t } ^ { * } \big ] \big ) ^ { 1 / 2 } \cdot \big ( \mathbb { E } \big [ \big \| \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big \| ^ { 2 } \big | X _ { t } ^ { * } \big ] \big ) ^ { 1 / 2 } \mathrm { d } s } \end{array}
$$

And this implies that

$$
\begin{array} { r l } & { \operatorname* { s u p } _ { t ^ { \prime } \in [ 0 , t ] } \left( \mathbb { E } [ \| Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 } \| ^ { 2 } | X _ { t ^ { \prime } } ^ { * } ] \right) ^ { 1 / 2 } } \\ & { \leq \int _ { t } ^ { T } \left( \mathbb { E } \big [ \big \| Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } \big \| ^ { 2 } \big | X _ { t } ^ { * } \big ] \right) ^ { 1 / 2 } \cdot \big ( \mathbb { E } \big [ \big \| \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big \| ^ { 2 } \big | X _ { t } ^ { * } \big ] \big ) ^ { 1 / 2 } \mathrm { d } s } \\ & { \leq \int _ { t } ^ { T } \operatorname* { s u p } _ { t ^ { \prime } \in [ 0 , s ] } \big ( \mathbb { E } \big [ \big \| Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } \big \| ^ { 2 } \big | X _ { t ^ { \prime } } ^ { * } \big ] \big ) ^ { 1 / 2 } \cdot \operatorname* { s u p } _ { t ^ { \prime } \in [ 0 , s ] } \big ( \mathbb { E } \big [ \big \| \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big \| ^ { 2 } \big | X _ { t ^ { \prime } } ^ { * } \big ] \big ) ^ { 1 / 2 } \mathrm { d } s . } \end{array}
$$

Applying Grönwall’s inequality on the function $\begin{array} { r } { f ( t ) = \operatorname* { s u p } _ { t ^ { \prime } \in [ 0 , t ] } \left( \mathbb { E } [ \| Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 } \| ^ { 2 } | X _ { t ^ { \prime } } ^ { * } ] \right) ^ { 1 / 2 } } \end{array}$ , we obtain that $\begin{array} { r } { \operatorname* { s u p } _ { t ^ { \prime } \in [ 0 , t ] } \left( \mathbb { E } [ \| Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 } \| ^ { 2 } | X _ { t ^ { \prime } } ^ { * } ] \right) ^ { 1 / 2 } = 0 } \end{array}$ for all $t \in [ 0 , T ]$ , which means that $Y _ { t } ^ { 1 } = Y _ { t } ^ { 2 }$ almost surely. And since $\begin{array} { r } { \| Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 } \| \leq \int _ { t } ^ { T ^ { \prime } } \big ( \mathbb { E } \big [ \big \| Y _ { s } ^ { 1 } - Y _ { s } ^ { 2 } \big \| ^ { 2 } | X _ { t } ^ { * } \big ] \big ) ^ { 1 / 2 } } \end{array}$ · $\big ( \mathbb { E } \big [ \big \| \nabla _ { x } b ( X _ { s } ^ { * } , s ) \big \| ^ { 2 } | X _ { t } ^ { * } \big ] \big ) ^ { 1 / 2 } { \mathrm { d } } s = 0$ , we obtain that $Y ^ { 1 } = Y ^ { 2 }$ .

# E.4 Pseudo-code of Adjoint Matching for DDIM fine-tuning

Note that for each pair of equations (219)-(220), (221)-(222), (223)-(224), the first equation corresponds to the updates in the DDPM paper, while the second equation is an Euler-Maruyama / Euler discretization of the continuous-time object. To check that both discretizations are equal up to first order, remark that

$$
\begin{array} { r } { \sqrt { \frac { \bar { \alpha } _ { k + 1 } } { \bar { \alpha } _ { k } } } = \sqrt { 1 + \frac { \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } } \approx 1 + \frac { \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } } { 2 \bar { \alpha } _ { k } } + O ( ( \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } ) ^ { 2 } ) . } \end{array}
$$

Input: Pre-trained denoiser ϵbase, number of fine-tuning iterations $N$ . Initialize fine-tuned denoiser: $\epsilon ^ { \mathrm { f i n e t u n e } } = \epsilon ^ { \mathrm { b a s e } }$ with parameters $\theta$ . for $n \in \{ 0 , \ldots , N - 1 \}$ do

Sample $m$ trajectories $\pmb { X } = ( X _ { t } ) _ { t \in \{ 0 , \ldots , 1 \} }$ according to DDPM, e.g.:

$$
\begin{array} { r } { \mathrm { { V } } _ { k + 1 } = \sqrt { \frac { \bar { \alpha } _ { k + 1 } } { \bar { \alpha } _ { k } } \left( X _ { k } - \frac { 1 - \bar { \alpha } _ { k } / \bar { \alpha } _ { k + 1 } } { \sqrt { 1 - \bar { \alpha } _ { k } } } \epsilon ^ { \mathrm { f i n e t u m e } } ( X _ { k } , k ) \right) + \sqrt { \frac { 1 - \bar { \alpha } _ { k + 1 } } { 1 - \bar { \alpha } _ { k } } \left( 1 - \frac { \bar { \alpha } _ { k } } { \bar { \alpha } _ { k + 1 } } \right) } \varepsilon _ { k } } , \quad \varepsilon _ { k } \sim \mathcal { N } ( 0 , I ) , \ X _ { 0 } \sim \mathcal { N } ( 0 , I ) , } \end{array}
$$

$$
\begin{array} { r } { X _ { k + 1 } = X _ { k } + \frac { { \bar { \alpha } _ { k + 1 } } - { { \bar { \alpha } } _ { k } } } { 2 { { \bar { \alpha } } _ { k } } } X _ { k } - \frac { { { \bar { \alpha } } _ { k + 1 } } - { { \bar { \alpha } } _ { k } } } { { { \bar { \alpha } } _ { k } } \sqrt { 1 - { { \bar { \alpha } } _ { k } } } } \epsilon ^ { \mathrm { f i n e t u n e } } ( X _ { k } , k ) + \sqrt { \frac { { { \bar { \alpha } } _ { k + 1 } } - { { { \bar { \alpha } } _ { k } } } } { { { \bar { \alpha } } _ { k } } } } \varepsilon _ { k } . } \end{array}
$$

For each trajectory, solve the lean adjoint ODE (38)-(39) backwards in time from $k = K$ to $_ 0$ , e.g.:

$$
\begin{array} { r l } & { \quad \tilde { a } _ { k } = \tilde { a } _ { k + 1 } + \tilde { a } _ { k + 1 } ^ { \top } \nabla x _ { k } \left( \sqrt { \frac { \tilde { a } _ { k + 1 } } { \tilde { a } _ { k } } } \big ( X _ { k } - \frac { 1 - \tilde { a } _ { k } / \tilde { a } _ { k + 1 } } { \sqrt { 1 - \tilde { a } _ { k } } } \epsilon ^ { \mathrm { b a s e } } ( X _ { k } , k ) \big ) - X _ { k } \right) , \qquad \tilde { a } _ { K } = \nabla x _ { K } r ( X _ { K } ) , } \\ & { \mathrm { o r } \tilde { a } _ { k } = \tilde { a } _ { k + 1 } + \tilde { a } _ { k + 1 } ^ { \top } \nabla _ { X _ { t } } \left( \frac { \tilde { a } _ { k + 1 } - \tilde { a } _ { k } } { 2 \tilde { a } _ { k } } X _ { k } - \frac { \tilde { a } _ { k + 1 } - \tilde { a } _ { k } } { \tilde { a } _ { k } \sqrt { 1 - \tilde { a } _ { k } } } \epsilon ^ { \mathrm { b a s e } } ( X _ { k } , k ) \right) , \qquad \tilde { a } _ { K } = \nabla x _ { K } r ( X _ { K } ) . } \end{array}
$$

Note that $X _ { k }$ and $\tilde { a } _ { k }$ should be computed without gradients, i.e., $X _ { k } = \tt s t o p g r a d ( X _ { k } )$ , $\tilde { \boldsymbol { a } } _ { k } = \mathsf { s t o p g r a d } ( \tilde { \boldsymbol { a } } _ { k } )$ .

For each trajectory, compute the Adjoint Matching objective (37):

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { A d j - M a t c h } } ( \theta ) = \sum _ { k \in \{ 0 , \dots , K - 1 \} } \Big \| \sqrt { \frac { \bar { \alpha } _ { k + 1 } } { \bar { \alpha } _ { k } ( 1 - \bar { \alpha } _ { k + 1 } ) } \big ( 1 - \frac { \bar { \alpha } _ { k } } { \bar { \alpha } _ { k + 1 } } \big ) } ( \epsilon ^ { \mathrm { f u n e t u n e } } ( X _ { k } , k ) - \epsilon ^ { \mathrm { b a s e } } ( X _ { k } , k ) ) } \\ & { \qquad - \sqrt { \frac { 1 - \bar { \alpha } _ { k + 1 } } { 1 - \bar { \alpha } _ { k } } \big ( 1 - \frac { \bar { \alpha } _ { k } } { \bar { \alpha } _ { k + 1 } } \big ) } \tilde { u } _ { k } \Big \| ^ { 2 } , } \\ & { \gamma \mathcal { L } _ { \mathrm { A d j - M a t c h } } ( \theta ) = \sum _ { k \in \{ 0 , \dots , K - 1 \} } \Big \| \sqrt { \frac { \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } ( 1 - \bar { \alpha } _ { k } ) } } ( \epsilon ^ { \mathrm { f u n e t u n e } } ( X _ { k } , k ) - \epsilon ^ { \mathrm { b a s e } } ( X _ { k } , k ) ) - \sqrt { \frac { \bar { \alpha } _ { k + 1 } - \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } } \tilde { u } _ { k } \Big \| ^ { 2 } . } \end{array}
$$

Compute the gradient $\nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } )$ and update $\theta$ using favorite gradient descent algorithm. end

Output: Fine-tuned vector field $v$ finetune

# F Adapting diffusion fine-tuning baselines to flow matching

F.1 Adapting ReFL (Xu et al., 2023) to flow matching

Reward Feedback Learning (ReFL) is a diffusion fine-tuning algorithm introduced by Xu et al. (2023) which tries to increase the reward on denoised samples. Namely, if $\pmb { X } = ( X _ { t } ) _ { t \in [ 0 , 1 ] }$ is the solution of the DDPM SDE (7), we can denoise $X _ { t }$ as

$$
\begin{array} { r } { \hat { X } _ { 1 } ( X _ { t } ) = \frac { X _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ( X _ { t } , t ) } { \sqrt { \bar { \alpha } _ { t } } } . } \end{array}
$$

This equation follows from the stochastic interpolant equation (2) if we replace $X _ { 0 }$ with the noise predictor $\epsilon ( X _ { t } , t )$ . And then, the ReFL optimization update is based on the gradient:

$$
\begin{array} { r } { \nabla _ { \theta } r \big ( \hat { X } _ { 1 } ( X _ { t } ) \big ) = \nabla _ { \theta } r \Big ( \frac { X _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( X _ { t } , t ) } { \sqrt { \bar { \alpha } _ { t } } } \Big ) , } \end{array}
$$

where the trajectories have been detached.

To adapt ReFL to Flow Matching, we need to express the denoiser map in terms of the vector field $\boldsymbol { v }$ . We have that

$$
\begin{array} { r l } & { \boldsymbol { \nu } ( x , t ) = \mathbb { E } \big [ \dot { \beta } _ { t } \bar { X } _ { 0 } + \dot { \alpha } _ { t } \bar { X } _ { 1 } \big | \beta _ { t } \bar { X } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 } = x \big ] = \mathbb { E } \big [ \frac { \dot { \beta } _ { t } } { \beta _ { t } } \big ( \beta _ { t } \bar { X } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 } \big ) + \big ( \dot { \alpha } _ { t } - \frac { \dot { \beta } _ { t } } { \beta _ { t } } \alpha _ { t } \big ) \bar { X } _ { 1 } \big | \beta _ { t } \bar { X } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 } \big ] , } \\ & { \mathrm { ~ \ ~ \ } = \frac { \dot { \beta } _ { t } } { \beta _ { t } } x + \big ( \dot { \alpha } _ { t } - \frac { \dot { \beta } _ { t } } { \beta _ { t } } \alpha _ { t } \big ) \hat { X } _ { 1 } ( x , t ) . } \end{array}
$$

where we defined the denoiser map $\hat { X } _ { 1 } ( x , t ) : = \mathbb { E } \big [ \bar { X } _ { 1 } | \beta _ { t } \bar { X } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 } = x \big ]$ . Hence,

$$
\begin{array} { r } { \hat { X } _ { 1 } ( x , t ) = \frac { v ( x , t ) - \frac { \dot { \beta } _ { t } } { \beta _ { t } } x } { \dot { \alpha } _ { t } - \frac { \dot { \beta } _ { t } } { \beta _ { t } } \alpha _ { t } } . } \end{array}
$$

# F.2 Adapting Diffusion-DPO (Wallace et al., 2023a) to flow matching

The Diffusion-DPO loss assumes access to ranked pairs of generated samples $x _ { 1 } ^ { w } \succ x _ { 1 } ^ { l }$ , where $x ^ { w }$ and $x ^ { l }$ are the winning and losing samples. For DDPM, the loss implemented in practice reads (Wallace et al., 2023a, Eq. 46):

$$
\begin{array} { r l } & { L _ { \mathrm { D P O } } ( \theta ) = - \mathbb { E } _ { ( x _ { 1 } ^ { w } , x _ { 1 } ^ { l } ) \sim \mathcal { D } , k \sim U [ 0 , K ] , x _ { k h } ^ { w } \sim q ( x _ { k h } ^ { w } | x _ { 1 } ^ { w } ) , x _ { t } ^ { l } \sim q ( x _ { k h } ^ { l } | x _ { 1 } ^ { l } ) } \Big [ } \\ & { \qquad \log S \big ( - \frac { \tilde { \beta } } { 2 } \big ( \| \varepsilon ^ { w } - \epsilon _ { \theta } ( x _ { k h } ^ { w } , k h ) \| ^ { 2 } - \| \varepsilon ^ { w } - \epsilon _ { \mathrm { r e f } } ( x _ { k h } ^ { w } , k h ) \| ^ { 2 } } \\ & { \qquad - \left( \| \varepsilon ^ { l } - \epsilon _ { \theta } ( x _ { k h } ^ { l } , k h ) \| ^ { 2 } - \| \varepsilon ^ { l } - \epsilon _ { \mathrm { r e f } } ( x _ { k h } ^ { l } , k h ) \| ^ { 2 } \right) \big ) \big ] , } \end{array}
$$

where $\begin{array} { r } { S ( x ) = \frac { 1 } { 1 + e ^ { - x } } } \end{array}$ denotes the sigmoid function, and √ √ $q ( x _ { k h } ^ { * } | x _ { 1 } ^ { * } )$ is the conditional distribution of the forward process, i.e. Diffusion-DPO loss in (Wallace et al., 2023a, Sec. S4), we observe that the term is sampled as , $\epsilon \sim N ( 0 , I )$ . Following the derivation of the $\begin{array} { r } { - \frac { \bar { \beta } } { 2 } \| \varepsilon ^ { w } - \epsilon _ { \theta } ( x _ { \boldsymbol { k } h } ^ { w } , \boldsymbol { k } h ) \| ^ { 2 } } \end{array}$ arises from

$$
- \frac { \tilde { \beta } } { 2 \frac { 1 - \gamma _ { k h } } { \gamma _ { k h } } } \| \hat { x } _ { 1 } ( x _ { k h } ^ { w } ) - x _ { 1 } ^ { w } \| ^ { 2 } ,
$$

up to a constant term in $\theta$ . If we switch to the more general flow matching scheme, the analog of this term is

$$
\begin{array} { r } { - \frac { \tilde { \beta } } { 2 \frac { \beta _ { k h } ^ { 2 } } { \alpha _ { k h } ^ { 2 } } } \| \hat { x } _ { 1 } ( x _ { k h } ^ { w } ) - x _ { 1 } ^ { w } \| ^ { 2 } . } \end{array}
$$

Using the expression of the denoiser map in terms of the vector field $v$ in equation (229), we can rewrite (232) as:

$$
\begin{array} { r l } & { - \frac { \tilde { \beta } } { 2 \frac { \beta _ { k h } ^ { 2 } } { \alpha _ { k h } ^ { 2 } } } \Big \| \frac { v ( x _ { k h } ^ { w } , k h ) - \frac { \tilde { \beta } _ { k h } } { \beta _ { k h } } x _ { k h } ^ { w } } { \dot { \alpha } _ { k h } - \frac { \tilde { \beta } _ { k h } } { \beta _ { k h } } \alpha _ { k h } } - x _ { 1 } ^ { w } \Big \| ^ { 2 } = - \frac { \tilde { \beta } } { 2 } \Big \| \frac { v ( x _ { k h } ^ { w } , k h ) - \frac { \tilde { \beta } _ { k h } } { \beta _ { k h } } x _ { k h } ^ { w } } { \frac { \tilde { \alpha } _ { k h } } { \alpha _ { k h } } \beta _ { k h } - \tilde { \beta } _ { k h } } - \frac { \alpha _ { k h } } { \beta _ { k h } } x _ { 1 } ^ { w } \Big \| ^ { 2 } . } \end{array}
$$

Thus, the Diffusion-DPO loss for Flow Matching reads

$$
\begin{array} { r l } & { L _ { \mathrm { D P O } } ( \theta ) = - \mathbb { E } _ { ( x _ { 1 } ^ { w } , x _ { 1 } ^ { l } ) \sim \mathcal { D } , k \sim \mathcal { U } [ 0 , K ] , x _ { k } ^ { w } \sim \mathcal { G } ( x _ { k k } ^ { w } \mid x _ { 1 } ^ { w } ) , x _ { t } ^ { l } \sim \mathcal { G } ( x _ { k h } ^ { l } \mid x _ { 1 } ^ { l } ) } \Big [ } \\ & { \quad \quad \quad \quad \quad \log S \big ( - \frac { \hat { \beta } } { 2 } \big ( \big \| \frac { v _ { \theta } ( x _ { k h } ^ { w } , k h ) - \frac { \hat { \beta } _ { k h } } { \mathcal { \beta } _ { k h } } x _ { k h } ^ { w } } { \frac { \hat { \alpha } _ { k h } } { \hat { \alpha } _ { k h } } \hat { \beta } _ { k h } - \hat { \beta } _ { k h } } - \frac { \alpha _ { k h } } { \hat { \beta } _ { k h } } x _ { 1 } ^ { w } \big \| ^ { 2 } - \big \| \frac { v _ { \mathrm { r e f } } ( x _ { k h } ^ { w } , k h ) - \frac { \hat { \beta } _ { k h } } { \hat { \beta } _ { k h } } x _ { k h } ^ { w } } { \frac { \hat { \alpha } _ { k h } } { \hat { \alpha } _ { k h } } \hat { \beta } _ { k h } - \hat { \beta } _ { k h } } - \frac { \alpha _ { k h } } { \hat { \beta } _ { k h } } x _ { 1 } ^ { w } \big \| ^ { 2 } }  \\ &  \quad \quad \quad \quad \quad \quad \quad - \big ( \big \| \frac { v _ { \theta } ( x _ { k h } ^ { l } , k h ) - \frac { \hat { \beta } _ { k h } } { \hat { \beta } _ { k h } } x _ { k h } ^ { l } } { \frac { \hat { \alpha } _ { k h } } { \hat { \alpha } _ { k h } } \hat { \beta } _ { k h } - \hat { \beta } _ { k h } } - \frac { \alpha _ { k h } }  \hat  \end{array}
$$

(Wallace et al., 2023a, Sec. 5.1) claim that $\beta \in \left[ 2 0 0 0 , 5 0 0 0 \right]$ yields good performance on Stable Diffusion 1.5 and Stable Diffusion XL-1.0, which if we translate to our notation corresponds to $\tilde { \beta } \in [ 4 0 0 0 , 1 0 0 0 0 ]$ .

When we have access to the reward function $r$ , instead of a winning sample $x _ { 1 } ^ { w }$ and a losing sample $x _ { 1 } ^ { l }$ , we have a pair of samples $( x _ { 1 } ^ { a } , x _ { 1 } ^ { b } )$ with winning weights $\begin{array} { r } { S ( r ( x _ { 1 } ^ { a } ) - r ( x _ { 1 } ^ { b } ) ) = \frac { 1 } { 1 + \exp { \left( r ( x _ { 1 } ^ { b } ) - r ( x _ { 1 } ^ { a } ) \right) } } } \end{array}$ 11+exp  r(xb1)−r(xa1 ) , S(−(r(xa1 ) − r(xb1))) = $\frac { 1 } { 1 + \exp \left( - ( r ( x _ { 1 } ^ { b } ) - r ( x _ { 1 } ^ { a } ) ) \right) }$ Hence, the loss (234) becomes:

$$
\begin{array} { r } { L _ { \mathrm { D P O } } ( \theta ) = - \mathbb { E } _ { ( x _ { 1 } ^ { a } , x _ { 1 } ^ { b } ) \sim \mathcal { D } , k \sim U [ 0 , K ] , x _ { k h } ^ { a } \sim q ( x _ { k h } ^ { a } \mid x _ { 1 } ^ { a } ) , x _ { \mathrm { t } } ^ { b } \sim q ( x _ { k h } ^ { b } \mid x _ { 1 } ^ { b } ) } \Bigg [ \sum _ { s \in \{ \pm 1 \} } S \big ( s ( r ( x _ { 1 } ^ { a } ) - r ( x _ { 1 } ^ { b } ) ) \big ) \times } \\ { \log S \big ( - \frac { s \tilde { \beta } } { 2 } ( \big \| \frac { v _ { \theta } ( x _ { k h } ^ { a } , k h ) - \frac { \tilde { \beta } _ { k h } } { \beta _ { k h } } x _ { k h } ^ { a } } { \frac { \tilde { \alpha } _ { k h } } { \tilde { \alpha } _ { k h } } \beta _ { k h } - \tilde { \beta } _ { k h } } - \frac { \alpha _ { k h } } { \beta _ { k h } } x _ { 1 } ^ { a } \big \| ^ { 2 } - \big \| \frac { v _ { \mathrm { r e f } } ( x _ { k h } ^ { a } , k h ) - \frac { \tilde { \beta } _ { k h } } { \beta _ { k h } } x _ { k h } ^ { a } } { \frac { \tilde { \alpha } _ { k h } } { \alpha _ { k h } } \beta _ { k h } - \tilde { \beta } _ { k h } } - \frac { \alpha _ { k h } } { \beta _ { k h } } x _ { 1 } ^ { a } \big \| ^ { 2 } }  \Bigg .  \\  -  \big ( \big \| \frac { v _ { \theta } ( x _ { k h } ^ { b } , k h ) - \frac { \tilde { \beta } _ { k h } } { \beta _ { k h } } x _ { k h } ^ { b } } { \frac { \tilde { \alpha } _ { k h } } { \alpha _ { k h } } \beta _ { k h } - \tilde { \beta } _ { k h } } - \frac { \alpha _ { k h } } { \beta _ { k h } } x _ { 1 } ^  b \end{array}
$$

We want to emphasize that despite the similarities, even though the loss $\scriptstyle L _ { \mathrm { D P O } }$ that we use (equation (235)) is very similar to the one implemented by Wallace et al. (2023a), the preference data pairs that we use are very different from theirs. We sample the preference data from the current model, which results in imperfect samples, while they consider off-policy, high-quality, curated preference samples. The reason for this discrepancy is that the starting point of our work is a reward model, not a set of preference data, and we only benchmark against approaches that leverage reward models for an apples-to-apples comparison. Our experimental results on DPO (Table 2, Figure 6, Table 3) show that the resulting model performs like the base model, or a bit worse according to some metrics. Hence, we conclude that DPO is not a competitive alternative for on-policy fine-tune when the base model is not already good.

# G Experimental details

Unless otherwise specified, we used the same hyperparameters across all fine-tuning methods. Namely, we used:

• $K = 4 0$ timesteps.   
• Adam optimizer with learning rate $2 \times 1 0 ^ { - 5 }$ and parameters $\beta _ { 1 } = 0 . 9 5$ , $\beta _ { 2 } = 0 . 9 9 9$ , $\epsilon = 1 \times 1 0 ^ { - 8 }$ , weight decay $1 \times 1 0 ^ { - 2 }$ , gradient norm clipping value 1. For Discrete Adjoint, these hyperparameters resulted in fine-tuning instability (see Table 6); the results that we report in all other tables for Discrete Adjoint were obtained with learning rate $1 \times 1 0 ^ { - 5 }$ .   
• Bfloat16 precision.   
• Effective batch size 40; for each run we used two 80GB A100 GPUs with batch size 20 each.   
• A set of 40k fine-tuning prompts taken from a licensed dataset consisting of text and image pairs (note that we disregarded the images). Thus, each epoch lasts 1000 iterations; see the total amount of fine-tuning iterations for each algorithm in Table 3. For each of the three runs that we perform for each data point that we report, the set of 40k prompts is sampled independently among a total set of 100k prompts.

# G.1 Noise schedule details

Since we use $K = 4 0$ discretization steps, the timesteps are $t \in \{ 0 , 0 . 0 2 5 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 , . . . , 0 . 9 5 , 0 . 9 7 5 \}$ . To sample $X _ { t + h }$ from $X _ { t }$ we use equation (40). We use the choices $\alpha _ { t } = t$ , $\beta _ { t } = 1 - t$ , which means that $\begin{array} { r } { \sigma ( t ) = \sqrt { 2 \beta _ { t } ( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } ) } = \sqrt { 2 ( 1 - t ) ( \frac { 1 - t } { t } + 1 ) } = \sqrt { \frac { 2 ( 1 - t ) } { t } } } \end{array}$ .

Note that if we plug $t = 0$ into this expression, we obtain infinity, and if we plug $t \lessapprox 1$ , we obtain $\sigma ( t ) \approx 0$ For obvious reasons, the former issue requires a fix: we simply add a small offset to the denominator of $\sigma ( t )$ , replacing $\sqrt { 1 / t }$ by $\sqrt { 1 / ( t + h ) }$ (note that $h : = 1 / K = 0 . 0 2 5 ,$ ). But the latter issue is also not completely satisfactory from a practical standpoint, because looking at the adjoint matching loss (37), we observe that $u ( X _ { t } ^ { u } , t )$ is trained to approximate the conditional expectation of $\sigma ( t ) ^ { 1 } \tilde { a } ( t ; X ^ { \bar { u } } )$ . Thus, if we set $\sigma ( t )$ very close to zero for $t \lessapprox 1$ , we are forcing the control $u$ to be close to zero as well, or equivalently preventing $v$ finetune from deviating from $v ^ { \mathrm { b a s e } }$ . While this is the right thing to do from a theoretical perspective, we concluded experimentally that setting $\sigma ( t )$ just slightly larger results in substantially faster fine-tuning, thanks to the additional leeway provided to $v ^ { \mathrm { f i n e t u n e } }$ to deviate from $v ^ { \mathrm { b a s e } }$ . In particular, we added a small offset to the factor $1 - t$ in the numerator $1 - t$ of $\sigma ( t )$ : we replaced $1 - t$ by $1 - t + h$ . Thus, the expression that we used to compute the diffusion coefficient in our experiments is

$$
\begin{array} { r } { \sigma ( t ) = \sqrt { \frac { 2 ( 1 - t + h ) } { t + h } } . } \end{array}
$$

When solving the lean adjoint ODE (38)-(39) backwards in time via the Euler scheme (41), the timesteps we use are $t \in \{ 1 , 0 . 9 7 5 , 0 . 9 5 , 0 . 9 2 5 , 0 . 9 , \dots , 0 . 0 5 , 0 . 0 2 5 \}$ . We do not actually initialize the adjoint state as $\nabla _ { x } g ( X _ { 1 } )$ , but rather as $\nabla _ { x } g ( \hat { X } _ { 1 } )$ , where $X _ { 1 } : = X _ { 1 - h } + h v ^ { \mathrm { b a s e } } ( X _ { 1 - h } , 1 - h )$ . That is, $\ddot { X } _ { 1 }$ is obtained by performing a final noiseless update, instead of using noise $\sigma ( 1 - h ) = \sqrt { 4 h }$ given by equation (236). The reason for this is that the regular final iterate $X _ { 1 }$ contains some noise that was added in the final step, and that can distort the gradient $\nabla _ { x } g ( X _ { 1 } )$ . By setting $\tilde { a } ( 1 ; X ) = \nabla _ { x } g ( X _ { 1 } )$ , we get rid of this bias. Note that in the continuous time limit $h  0$ , ${ \dot { X } } _ { 1 } = X _ { 1 }$ , which means that this small trick is consistent.

# G.2 Selection of gradient evaluation timesteps

In Algorithm 1, equation (42), we state that the term $\begin{array} { r } { \left. \frac { 2 } { \sigma ( t ) } \bigl ( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) \bigr ) + \sigma ( t ) \tilde { a } _ { t } \right. ^ { 2 } } \end{array}$ must be computed for all $K$ steps in $\{ 0 , \ldots , 1 - h \}$ . However, the gradient signal provided by backpropagating through this expression for consecutive times sample a subset $\kappa$ of timesteps, and we only compute and backpropagate the terms and $t + h$ is quite similar. In the interest of computational efficiency, we $\begin{array} { r l } { \bigg \Vert \frac { 2 } { \sigma ( t ) } \left( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t } , t ) - \right. } \end{array}$ $v ^ { \mathrm { b a s e } } ( X _ { t } , t ) \big ) + \sigma ( t ) \tilde { a } _ { t } \big \| ^ { 2 }$ for those timesteps. We construct $\kappa$ by sampling ten timesteps uniformly without repetition among $\{ 0 , \ldots , 0 . 7 2 5 \}$ , and always sampling the last ten timesteps $\{ 0 . 7 5 , \ldots , 0 . 9 7 5 \}$ . This is because fine-tuning the last ten steps ( $2 5 \%$ of the total) well is critical for good empirical performance, while the initial steps are not as important.

# G.3 Loss function clipping: the LCT hyperparameter

Note that the magnitude of $\sigma ( t ) ^ { \mathsf { T } } a ( t ; X ^ { \bar { u } } , { \bar { u } } )$ is much larger for times $t \gtrapprox 0$ than for times $t \lessapprox 1$ . The reason is two-fold:

• As discussed in Appendix G.1, $\sigma ( t )$ is much larger for $t \gtrapprox 0$ than for $t \lessapprox 1$ . • The magnitude of the lean adjoint state $\tilde { a }$ grows roughly exponentially as $t$ goes backward in time. In fact, if we assumed that $\nabla _ { x } b ( X _ { t } , t )$ is constant in time, this statement would be exact.

Observe that when $\sigma ( t ) ^ { 1 } a ( t ; X ^ { \bar { u } } , \bar { u } )$ is large, the gradient $\begin{array} { r } { \nabla _ { \theta } \left\| \frac { 2 } { \sigma ( t ) } \big ( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) \big ) + \sigma ( t ) \tilde { a } _ { t } \right\| ^ { 2 } } \end{array}$ also has a high magnitude. Including such terms in our gradient computation decreases the signal to noise ratio of the gradient. Even more so, as discussed in Appendix G.2 for good practical performance it is critical to get a good gradient signal from the last $2 5 \%$ steps. Hence, including the high-magnitude terms for $t \lessapprox 0$ in our gradients can muffle these other important, low-magnitude terms.

To fix this issue, we clip the terms such that $\begin{array} { r } { \left. \frac { 2 } { \sigma ( t ) } \left( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) \right) + \sigma ( t ) \tilde { a } _ { t } \right. ^ { 2 } > \mathrm { L C T } } \end{array}$ , where LCT stands for the loss clipping threshold. That is, the adjoint matching loss that we use in our experiments is of the form:

$$
\begin{array} { r } { \hat { \mathcal { L } } _ { \mathrm { A d j - M a t c h } } ( \theta ) = \sum _ { t \in K } \operatorname* { m i n } \big \{ \mathrm { L C T } , \big \| \frac { 2 } { \sigma ( t ) } \big ( v _ { \theta } ^ { \mathrm { f i n e t u m e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) \big ) + \sigma ( t ) \tilde { a } _ { t } \big \| ^ { 2 } \big \} , } \end{array}
$$

where $\kappa$ is the random timestep subset described in Appendix G.2.

For adjoint matching, we set $\mathrm { L C T } = 1 . 6 \times \lambda ^ { 2 }$ . Remark that LCT needs to grow quadratically with $\lambda$ , because the magnitude of the lean adjoint $\tilde { a }$ grows quadratically with $\lambda$ . We set the constant 1.6 through experimentation; all or almost all of the terms for the last ten timesteps fall below LCT, but only a fraction of the terms ( $\approx 2 5 \%$ ) for the first ten steps fall below LCT. The constant for LCT is a relevant hyperparameter that needs to be tuned to obtain a similar behavior.

We also used loss function clipping on the continuous adjoint loss. For that loss we set $\mathrm { L C T } = 1 6 0 0 \times \lambda ^ { 2 }$ . The reason is that the magnitude of the regular adjoint states is significantly larger than the magnitude of the lean adjoint states (which is a big reason why adjoint matching outperforms the continuous adjoint).

# G.4 Computation of evaluation metrics

We used the open_clip library (Ilharco et al., 2021) to compute ClipScores. We computed ClipScore diversity as the variance of Clip embeddings of 40 generations for a given prompt, averaged across 25 prompts. Namely,

$$
\begin{array} { r } { \mathrm { C l i p S c o r e \_ D i v e r s i t y } = \frac { 1 } { 4 0 } \sum _ { k = 1 } ^ { 4 0 } \frac { 2 } { 2 5 \cdot 2 4 } \sum _ { 1 \leq i < j \leq 2 5 } \| \mathrm { C l i p } ( g _ { i } ^ { k } ) - \mathrm { C l i p } ( g _ { j } ^ { k } ) \| ^ { 2 } , } \end{array}
$$

where $g _ { i } ^ { k }$ denotes the $i$ -th generation for the $k$ -th prompt.

We used the transformers library to compute the PickScore processor and model (Kirstain et al., 2023).   
PickScore diversity is computed in analogy with ClipScore diversity.

We used the hps library to compute values of Human Preference Score v2 (Wu et al., 2023b).

To compute Dreamsim diversity we use the dreamsim library (Fu et al., 2023). Dreamsim diversity is computed in analogy with ClipScore diversity.

# G.5 Remarks on computational costs

Observe from the figures reported in Table 3 that the per iteration wall-clock time of Adjoint Matching (156 seconds) is very similar to that of the Discrete Adjoint loss (152 seconds). The reason is that both algorithms perform a similar amount of forward and backward passes on the flow matching model and the reward model. Namely, for each sample in the batch, both algorithms perform $K$ forward passes on the flow model to obtain the trajectories. In order to compute the gradient of the training loss, the Discrete Adjoint loss does $K$ additional forward passes to evaluate the base flow model, one forward and backward pass on the reward model, and $K$ backward passes on the current flow model, which typically use gradient checkpointing to avoid memory overflow. In the case of Adjoint Matching, solving the lean adjoint ODE requires one forward and backward pass on the reward model, and $K$ backward passes on the base flow model. Finally, computing the gradient of the loss takes $K / 2$ additional backward passes if we evaluate at only half of the timesteps as we do, although this computation is much quicker because it can be fully parallelized.

Meanwhile, computing the gradient of the Continuous Adjoint loss takes 204 seconds per iteration. With respect to Adjoint Matching, Continuous Adjoint performs additional backward passes to compute the gradients $\nabla _ { X _ { t } } \| u ( X _ { t } , t ) \| ^ { 2 }$ when solving the adjoint ODE. Finally, we observe that models that directly fine-tune the reward are quicker, but that comes with its own set of issues that we discuss throughout the paper.

# G.6 Remarks on number of sampling timesteps

In our experiments and all baselines, we used 40 timesteps in the fine-tuning procedure ( $h = 1 / 4 0$ in Algorithm 1). The experiments reported in all tables and figures except for Table 8 were performed at 40 inference timesteps. In Table 8 (Appendix A), we show experimental results at 10, 20, 40, 100, and 200 inference timesteps, for the base model and the models fine-tuned with adjoint matching and DRaFT-1. We make the following observations about the results:

• The metrics for Adjoint Matching at 100 and 200 timesteps are statistically equal to the ones for 40 timesteps, with slight increases in Dreamsim diversity. This suggests that fine-tuning at large numbers of timesteps is a good idea if we want to perform inference at a large number of timesteps, as otherwise the capabilities of the model are limited by the number of fine-tuning timesteps instead of the inference compute. Also, at 100 and 200 timesteps the difference in performance of Adjoint Matching relative to DRaFT-1 increases. • The metrics for Adjoint Matching at 10 and 20 timesteps are worse than at 40 timesteps, especially for 10. The difference in performance between Adjoint Matching and DRaFT-1 vanishes at 10 timesteps for all metrics except for diversity, for which Adjoint Matching is still clearly better.