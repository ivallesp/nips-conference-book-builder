## [Visualization for Machine Learning](https://neurips.cc/Conferences/2018/Schedule?showEvent=10986) 
__Tutorial | Mon Dec 3rd 08:30  -- 10:30 AM @ Room 220 E__ 
_Fernanda Viégas · Martin Wattenberg_ 
Visualization is a powerful way to understand and interpret machine learning--as well as a promising area for ML researchers to investigate. This tutorial will provide an introduction to the landscape of ML visualizations, organized by types of users and their goals. We'll discuss how each stage of the ML research and development pipeline lends itself to different visualization techniques: analyzing training data, understanding the internals of a model, and testing performance. In addition, we’ll explore how visualization can play an important role in ML education and outreach to non-technical stakeholders. 
The tutorial will also include a brief introduction to key techniques from the fields of graphic design and human-computer interaction that are relevant in designing data displays. These ideas are helpful whether refining existing visualizations, or inventing entirely new visual techniques.

_________________

## [Scalable Bayesian Inference](https://neurips.cc/Conferences/2018/Schedule?showEvent=10984) 
__Tutorial | Mon Dec 3rd 08:30  -- 10:30 AM @ Room 517 CD__ 
_David Dunson_ 
This tutorial will provide a practical overview of state-of-the-art approaches for analyzing massive data sets using Bayesian statistical methods.  The first focus area will be on algorithms for very large sample size data (large n), and the second focus area will be on approaches for very high-dimensional data (large p).  A particular emphasis will be on maintaining a valid characterization of uncertainty, ruling out many popular methods, such as (most) variational approximations and approaches for maximum a posteriori estimation.  I will briefly review classical large sample approximations to posterior distributions (e.g., Laplace’s method, Bayesian central limit theorem), and will then transition to discussing conceptually and practical simple approaches for scaling up commonly used Markov chain Monte Carlo (MCMC) algorithms.  The focus is on making posterior computation much faster to implement for huge datasets while maintaining accuracy guarantees.  Some useful classes of algorithms having increasing theoretical and practical support include embarrassingly parallel (EP) MCMC, approximate MCMC, stochastic approximation, hybrid optimization and sampling, and modularization.  Applications to computational advertising, genomics, neurosciences and other areas will provide a concrete motivation.  Code and notes will be made available, and research problems of ongoing interest highlighted.

_________________

## [Adversarial Robustness: Theory and Practice](https://neurips.cc/Conferences/2018/Schedule?showEvent=10978) 
__Tutorial | Mon Dec 3rd 08:30  -- 10:30 AM @ Room 220 CD__ 
_J. Zico Kolter · Aleksander Madry_ 
The recent push to adopt machine learning solutions in real-world settings gives rise to a major challenge: can we develop ML solutions that, instead of merely working “most of the time”, are truly reliable and robust? This tutorial will survey some of the key challenges in this context and then focus on the topic of adversarial robustness: the widespread vulnerability of state-of-the-art deep learning models to adversarial misclassification (aka adversarial examples). We will discuss the practical as well as theoretical aspects of this phenomenon, with an emphasis on recent verification-based approaches to establishing formal robustness guarantees. Our treatment will go beyond viewing adversarial robustness solely as a security question. In particular, we will touch on the role it plays as a regularizer and its relation to generalization.

_________________

## [Coffee Break](https://neurips.cc/Conferences/2018/Schedule?showEvent=12867) 
__Break | Mon Dec 3rd 10:30  -- 11:00 AM @__ 
__ 


_________________

## [Common Pitfalls for Studying the Human Side of Machine Learning](https://neurips.cc/Conferences/2018/Schedule?showEvent=10981) 
__Tutorial | Mon Dec 3rd 11:00 AM -- 01:00 PM @ Room 220 E__ 
_Deirdre Mulligan · Nitin Kohli · Joshua A. Kroll_ 
As machine learning becomes increasingly important in everyday life, researchers have examined its relationship to people and society to answer calls for more responsible uses of data-driven technologies. Much work has focused on fairness, accountability, and transparency as well as on explanation and interpretability. However, these terms have resisted definition by computer scientists: while many definitions of each have been put forward, several capturing natural intuitions, these definitions do not capture everything that is meant by associated concept, causing friction with other disciplines and the public. Worse, sometimes different properties conflict explicitly or cannot be satisfied simultaneously. Drawing on our research on the meanings of these terms and the concepts they refer to across different disciplines (e.g., computer science, statistics, public policy, law, social sciences, philosophy, humanities, and others), we present common misconceptions machine learning researchers and practitioners hold when thinking about these topics. For example, it is often axiomatic that producing machine learning explanations automatically makes the outputs of a model more understandable, but this is hardly if ever the case. Similarly, defining fairness as a statistical property of the distribution of model outputs ignores the many procedural requirements supporting fairness in policymaking and the operation of the law. We describe how to integrate the rich meanings of these concepts into machine learning research and practice, enabling attendees to engage with disparate communities of research and practice and to recognize when terms are being overloaded, thereby avoiding speaking to people from other disciplines at cross purposes.

_________________

## [Unsupervised Deep Learning](https://neurips.cc/Conferences/2018/Schedule?showEvent=10985) 
__Tutorial | Mon Dec 3rd 11:00 AM -- 01:00 PM @ Room 220 CD__ 
_Alex Graves · Marc'Aurelio Ranzato_ 
Unsupervised learning looks set to play an ever more important role for deep neural networks, both as a way of harnessing vast quantities of unlabelled data, and as a means of learning representations that can rapidly generalise to new tasks and situations. The central challenge is how to determine what the objective function should be, when by definition we do not have an explicit target in mind. One approach, which this tutorial will cover in detail, is simply to ‘predict everything’ in the data, typically with a probabilistic model, which can be seen through the lens of the Minimum Description Length principle as an effort to compress the data as compactly as possible. However, we will also survey a range of other techniques, including un-normalized energy-based models, self-supervised algorithms and purely generative models such as GANs. Time allowing, we will extend our discussion to the reinforcement learning setting, where the natural analogue of unsupervised learning is intrinsic motivation, and notions such as curiosity, empowerment and compression progress are invoked as drivers of learning.

_________________

## [Negative Dependence, Stable Polynomials, and All That](https://neurips.cc/Conferences/2018/Schedule?showEvent=10983) 
__Tutorial | Mon Dec 3rd 11:00 AM -- 01:00 PM @ Rooms 517 CD__ 
_Suvrit Sra · Stefanie Jegelka_ 
This tutorial provides an introduction to a rapidly evolving topic: the theory of negative dependence and its numerous ramifications in machine learning. Indeed, negatively dependent probability measures provide a powerful tool for modeling non-i.i.d. data, and thus can impact all aspects of learning, including supervised, unsupervised, interpretable, interactive, and large-scale setups. The most well-known examples of negatively dependent distributions are perhaps the Determinantal Point Processes (DPPs), which have already found numerous ML applications. But DPPs are just the tip of the iceberg; the class of negatively dependent measures is much broader, and given the vast web of mathematical connections it enjoys, its holds great promise as a tool for machine learning. This tutorial exposes the ML audience to this rich mathematical toolbox, while outlining key theoretical ideas and motivating fundamental applications. Tasks that profit from negative dependence include anomaly detection, information maximization, experimental design, validation of black-box systems, architecture learning, fast MCMC sampling, dataset summarization, interpretable learning.

_________________

## [Counterfactual Inference](https://neurips.cc/Conferences/2018/Schedule?showEvent=10982) 
__Tutorial | Mon Dec 3rd 02:30  -- 04:30 PM @ Room 517 CD__ 
_Susan Athey_ 
This tutorial will review the literature that brings together recent developments in machine learning with methods for counterfactual inference.  It will focus on problems where the goal is to estimate the magnitude of causal effects, as well as to quantify the researcher’s uncertainty about these magnitudes.  The tutorial will consider two strands of the literature.  The first strand attempts to estimate causal effects of a single intervention, like a drug or a price change.  The goal can be to estimate the average (counterfactual) effect of applying the treatment to everyone; or the conditional average treatment effect, which is the effect of applying the treatment to an individual conditional on covariates.  We will also consider the problem of estimating an optimal treatment assignment policy (mapping features to assignments) under constraints on the nature of the policy, such as budget constraints. We look at applications to assigning unemployed workers to re-employment services.  We finish by considering the case with multiple alternative treatments, as well as the link between this literature and the literature on contextual bandits.  The second strand of the literature attempts to infer individual’s preferences from their behavior (inverse reinforcement learning in machine learning parlance, or structural estimation in econometrics parlance), and then predict an individual’s behavior in new environments.  We look at applications to consumer choice behavior, and analyze counterfactuals around price changes.  We discuss how models such as these can be tuned when the goal is counterfactual estimation rather than predicting outcomes.

_________________

## [Statistical Learning Theory: a Hitchhiker's Guide](https://neurips.cc/Conferences/2018/Schedule?showEvent=10980) 
__Tutorial | Mon Dec 3rd 02:30  -- 04:30 PM @ Room 220 E__ 
_John Shawe-Taylor · Omar Rivasplata_ 
The tutorial will showcase what statistical learning theory aims to assess about and hence deliver for learning systems. We will highlight how algorithms can piggy back on its results to improve the performances of learning algorithms as well as to understand their limitations. The tutorial is aimed at those wishing to gain an understanding of the value and role of statistical learning theory in order to hitch a ride on its results.

_________________

## [Automatic Machine Learning](https://neurips.cc/Conferences/2018/Schedule?showEvent=10979) 
__Tutorial | Mon Dec 3rd 02:30  -- 04:30 PM @ Room 220 CD__ 
_Frank Hutter · Joaquin Vanschoren_ 
The success of machine learning crucially relies on human machine learning experts, who construct appropriate features and workflows, and select appropriate machine learning paradigms, algorithms, neural architectures, and their hyperparameters. Automatic machine learning (AutoML) is an emerging research area that targets the progressive automation of machine learning, which uses machine learning and optimization to develop off-the-shelf machine learning methods that can be used easily and without expert knowledge. It covers a broad range of subfields, including hyperparameter optimization, neural architecture search, meta-learning, and transfer learning. This tutorial will cover the methods underlying the current state of the art in this fast-paced field.

_________________

## [Coffee Break](https://neurips.cc/Conferences/2018/Schedule?showEvent=12869) 
__Break | Mon Dec 3rd 04:30  -- 05:00 PM @__ 
__ 


_________________

## [Opening Remarks](https://neurips.cc/Conferences/2018/Schedule?showEvent=12871) 
__Break | Mon Dec 3rd 05:00  -- 05:30 PM @__ 
__ 


_________________

## [The Necessity of Diversity and Inclusivity in Tech](https://neurips.cc/Conferences/2018/Schedule?showEvent=12870) 
__Invited Talk | Mon Dec 3rd 05:30  -- 06:20 PM @__ 
_Laura Gomez_ 


_________________

## [Opening Reception](https://neurips.cc/Conferences/2018/Schedule?showEvent=12872) 
__Break | Mon Dec 3rd 06:30  -- 08:30 PM @__ 
__ 
