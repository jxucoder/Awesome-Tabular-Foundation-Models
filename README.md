# Awesome Tabular Foundation Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources, papers, and code for **Tabular Foundation Models (TFMs)**, also known as **Large Tabular Models (LTMs)**.

## Introduction

**Tabular Foundation Models (TFMs)** are a class of machine learning models pre-trained on diverse tabular datasets to perform tasks on new, unseen tables—often in a single forward pass or with minimal fine-tuning.

Despite tabular data being the dominant modality in many fields—from electronic healthcare records to census data, from cybersecurity to credit scoring, and from finance to natural sciences—it has received surprisingly little attention in the foundation model era. As highlighted by [van Breugel & van der Schaar (2024)](https://arxiv.org/abs/2405.01147), tabular foundation models are heavily underrepresented in ML research compared to language and vision models.

## Why Tabular Foundation Models?

### The Case for LTMs

1. **Ubiquity of Tabular Data**: Tabular data is everywhere in the real world. Quantitative research across sciences relies on these datasets, which in turn progresses scientific knowledge and influences public policy. This is reflected in the prominence of tabular data in competitions like Kaggle and KDD Cup.

2. **Unsolved Challenges**: Despite two decades of ML research, tree-based models like XGBoost remain among top performers for supervised tabular learning. This presents unique opportunities for foundation model approaches.

3. **Potential Impact**: LTMs could revolutionize how science and ML use tabular data—not as single datasets analyzed in isolation, but contextualized with respect to related datasets.

### Why Tabular FMs Have Been Overlooked

- **Data scarcity**: Until recently, there was a lack of large tabular metadatasets
- **Difficulty**: Tabular ML baselines are strong; new methods may not consistently outperform them
- **Human perception**: Text and images are more "naturally" interpretable; tabular data is hard to visually inspect
- **Evaluation challenges**: Unlike images/text where human judgment can assess quality, tabular data evaluation is less intuitive

### Potential Applications

LTMs could serve as invaluable tools for:

- **Data preprocessing & cleaning**: Automated outlier detection, data validation
- **Dataset discovery**: Finding relevant datasets across domains or knowledge bases (e.g., Wikipedia tables)
- **Data augmentation**: Few-shot generation of additional columns, performing semantic joins
- **Synthetic data generation**: Privacy-preserving data sharing, bias reduction, domain simulation
- **Automated meta-analyses**: Cross-dataset analysis and insights
- **Embeddings**: Row or dataset embeddings for downstream prediction tasks

## Contents

- [Introduction](#introduction)
- [Why Tabular Foundation Models?](#why-tabular-foundation-models)
- [Software & Code](#software--code)
- [Papers](#papers)
  - [Foundations & Position Papers](#foundations--position-papers)
  - [Tabular Classification & Regression](#tabular-classification--regression)
  - [Synthetic Data & Generation](#synthetic-data--generation)
  - [Graph & Relational Data](#graph--relational-data)
  - [Causal Inference](#causal-inference)
  - [Physical Systems & ODEs](#physical-systems--odes)
  - [Optimization](#optimization)
  - [Architectures & Training](#architectures--training)
  - [Theory & Analysis](#theory--analysis)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Tutorials & Talks](#tutorials--talks)
- [Contributing](#contributing)

## Software & Code

- **[PFNs](https://github.com/automl/PFNs)**: The official general-purpose library for creating and training Prior-Data Fitted Networks.
- **[TabPFN](https://github.com/PriorLabs/TabPFN)**: The official repository for the TabPFN model.
- **[TabICL](https://github.com/soda-inria/tabicl)**: Official implementation of TabICL and TabICLv2, open-source tabular foundation models for classification and regression.
- **[TabSwift](https://github.com/LAMDA-Tabular/TabSwift)**: Official code for TabSwift, a lightweight row-wise attention TFM with adaptive early-exit.
- **[Nori](https://github.com/Synthefy/synthefy-nori)**: Synthefy's fully open-source (weights, inference, and training code) tabular foundation model for regression (`pip install synthefy-nori`).
- **[TabTune](https://arxiv.org/abs/2511.02802)**: A unified library for inference and fine-tuning across multiple tabular foundation models.
- **[TabArena](https://github.com/autogluon/tabarena)**: A living benchmarking system for tabular ML with a public leaderboard at [tabarena.ai](https://tabarena.ai).

## Papers

### Foundations & Position Papers

- **Why Tabular Foundation Models Should Be a Research Priority** (ICML 2024)
  *Boris van Breugel, Mihaela van der Schaar*
  [Paper](https://arxiv.org/abs/2405.01147)
  > A seminal position paper arguing for developing Large Tabular Models (LTMs). Discusses why tabular data is underrepresented in FM research, proposes desiderata for LTMs, and outlines potential applications: from few-shot tabular models to automating data science, and from synthetic data to empowering multidisciplinary scientific discovery.

- **Unlocking the Full Potential of Data Science Requires Tabular Foundation Models, Agents, and Humans** (NeurIPS 2025 Position Paper Track)
  *Tianji Cong, Julian Martin Eisenschlos, Daniel Gomm, Leo Grinsztajn, Andreas C Mueller, Anupam Sanghi, Jan-Micha Bodensohn, Vadim Borisov, et al.*
  [Paper](https://openreview.net/forum?id=aXMPvmBAm5)
  > Argues that the future of data science lies in collaborative systems that tightly integrate agents, tabular foundation models (TFMs), and human experts. Presents a research agenda for more accessible, robust, and human-centered data science.

- **Transformers Can Do Bayesian Inference** (ICLR 2022)
  *Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter*
  [Paper](https://openreview.net/forum?id=KSugKcbNf9) | [Code](https://github.com/automl/PFNs)
  > The seminal paper introducing Prior-Data Fitted Networks (PFNs). It demonstrates how Transformers can be trained to approximate posterior predictive distributions for diverse priors, including Gaussian Processes and simple neural networks.

- **Towards Foundation Models for Learning on Tabular Data** (arXiv 2023)
  *Hongyu Zhang, Xingyu Wen, Shuai Zheng, Wei Xu, Jiang Bian*
  [Paper](https://arxiv.org/abs/2310.07338)
  > Explores approaches toward building foundation models specifically designed for tabular data learning.

### Tabular Classification & Regression

- **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** (ICLR 2023)
  *Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter*
  [Paper](https://openreview.net/forum?id=cp5PboIqf7r) | [Code](https://github.com/automl/TabPFN)
  > Applies PFNs to tabular classification. TabPFN is a single model pre-trained on synthetic data that achieves state-of-the-art performance on small tabular datasets, often beating Gradient Boosted Decision Trees without tuning.

- **TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models** (arXiv 2025)
  *Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Benjamin Jäger, Dominik Safaric, Simone Alessi, Adrian Hayler, Mihir Manium, Rosen Yu, Felix Jablonski, Shi Bin Hoo, Anurag Garg, Jake Robertson, Magnus Bühler, Vladyslav Moroshan, Lennart Purucker, Clara Cornu, Lilly Charlotte Wehrhahn, Alessandro Bonetto, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, Frank Hutter*
  [Paper](https://arxiv.org/abs/2511.08667) | [Code](https://github.com/PriorLabs/TabPFN)
  > TabPFN-2.5 is built for datasets with up to 50,000 data points and 2,000 features, a 20x increase in data cells compared to TabPFNv2. Default TabPFN-2.5 has a 100% win rate against default XGBoost on small to medium-sized classification datasets and an 87% win rate on larger datasets up to 100K samples. Includes a distillation engine to convert TabPFN-2.5 into compact MLPs or tree ensembles for low-latency production deployment.

- **TabPFN-3 Technical Report** (Technical Report, 2026)
  *Prior Labs Team*
  [Paper](https://arxiv.org/abs/2605.13986)
  > The next generation of TabPFN, scaling state-of-the-art in-context learning to datasets with up to 1M training rows on a single GPU. Features a redesigned architecture with an attention-based many-class decoder, an improved preprocessing pipeline, inference-time optimizations, and an enhanced synthetic SCM prior. Brings substantial gains on time series, relational, and tabular-text data.

- **TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention** (ICML 2026, Spotlight)
  *Si-Yang Liu, Han-Jia Ye*
  [Paper](https://arxiv.org/abs/2606.07345) | [Code](https://github.com/LAMDA-Tabular/TabSwift)
  > Revisits the original TabPFN design, showing a lightweight row-wise attention-only backbone stays competitive with two enhancements: gated attention stabilization and learnable register tokens for global context. Supports both classification and regression, and adds an adaptive layer-wise early-exit mechanism that dynamically adjusts inference depth per sample for anytime, latency-sensitive serving.

- **Nori: A Tabular Foundation Model for Regression Trained on Synthetic Data** (Open-source release, Synthefy, 2026)
  *Synthefy (Po-han Li, Aditya Narayanan, Sai Shankar Narasimhan, et al.)*
  [Model](https://huggingface.co/Synthefy/Nori) | [Code](https://github.com/Synthefy/synthefy-nori) | [Blog](https://www.synthefy.com/blog/synthefy-tabular-release)
  > A compact (~6M parameter) FeaturesTransformer for tabular regression via in-context learning, trained entirely on synthetic data. Alternates feature attention (across columns) and sample attention (across rows), uses RBF feature embeddings with native missing-value handling, and predicts a full quantile distribution via pinball loss. Reported #1 on aggregate across 96 real-world datasets, beating tuned XGBoost and LightGBM on ~80% of them at ~1/10 the size of peers.

- **TabDPT: Scaling Tabular Foundation Models on Real Data** (NeurIPS 2025)
  *University of Toronto / Layer 6 AI*
  [Paper](https://www.cs.toronto.edu/~mvolkovs/NeurIPS2025_TabDPT.pdf)
  > A foundation model pre-trained on real-world tabular datasets using self-supervised learning (masking) and retrieval. Shows robust generalization to unseen tables without task-specific fine-tuning.

- **TabICL: A Tabular Foundation Model for In-Context Learning** (ICLR 2025)
  [Paper](https://openreview.net/forum?id=0VvD1PmNzM)
  > A model trained on millions of synthetic tables for direct in-context task learning. Emphasizes scalability and efficiency.

- **XTab: Cross-table Pretraining for Tabular Transformers** (ICML 2023)
  *Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, Mahsa Shoaran*
  [Paper](https://arxiv.org/abs/2305.06090)
  > Cross-table pretraining approach enabling knowledge transfer across diverse tabular datasets with different schemas.

- **On Finetuning Tabular Foundation Models** (arXiv 2025)
  *Ivan Rubachev, Akim Kotelnikov, Nikolay Kartashev*
  [Paper](https://arxiv.org/abs/2506.08982) | [Code](https://github.com/yandex-research/tabpfn-finetuning)
  > Systematically evaluates finetuning strategies for TabPFN (specifically TabPFNv2), finding that full finetuning is efficient and effective for adapting to new tasks.

- **TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks** (ICML 2024)
  *Benjamin Feuer, Kelly Zhang, Jennifer J. Sun, Asma Ghandeharioun, Frank Hutter, et al.*
  [Paper](https://arxiv.org/abs/2402.11137) | [Code](https://github.com/automl/TuneTables)
  > Proposes "context optimization" (prompt tuning) to scale PFNs to larger datasets and improve performance by optimizing the input context.

- **Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks** (NeurIPS 2023 Workshop)
  *Benjamin Feuer, Chinmay Hegde, Niv Cohen*
  [Paper](https://openreview.net/forum?id=b0OhN0ii36)
  > Investigates how to summarize large training datasets before feeding them to TabPFN. Studies sketching and feature selection methods, finding that feature dimensionality reduction is more impactful than sample sketching for scaling TabPFN to datasets exceeding its 100-feature/1000-sample limits.

- **Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data** (NeurIPS 2024)
  *Kai Helli, David Schnurr, Noah Hollmann, Samuel Müller, Frank Hutter*
  [Paper](https://neurips.cc/virtual/2024/poster/93581) | [OpenReview](https://openreview.net/forum?id=93581)
  > Addresses temporal distribution shifts in tabular data using a TabPFN-based approach. Uses structural causal models (SCMs) that gradually shift over time, with a secondary SCM to model parameter changes. Outperforms XGBoost, CatBoost, and standard TabPFN on 18 datasets while maintaining strong calibration.

- **TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model** (arXiv 2026)
  *Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan*
  [Paper](https://arxiv.org/abs/2602.11139) | [Code](https://github.com/soda-inria/tabicl)
  > A new state-of-the-art tabular foundation model for regression and classification, built on a novel synthetic data generation engine, scalable softmax attention, and the Muon optimizer. Without any tuning, TabICLv2 surpasses RealTabPFN-2.5 (hyperparameter-tuned, ensembled, and fine-tuned) on TabArena and TALENT benchmarks while being 10x faster.

- **Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data** (arXiv 2025)
  *Anurag Garg, Muhammad Ali, Noah Hollmann, Lennart Purucker, Samuel Müller, Frank Hutter*
  [Paper](https://arxiv.org/abs/2507.03971)
  > Shows that TabPFN performance can be significantly boosted by continued pre-training on a small, curated collection of large real-world datasets from OpenML and Kaggle, bridging the synthetic-to-real gap. Real-TabPFN outperforms default TabPFNv2 and every other baseline on 29 AutoML Benchmark datasets.

- **EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Network** (arXiv 2025)
  *Michael Arbel, David Salinas, Frank Hutter*
  [Paper](https://arxiv.org/abs/2502.06684)
  > Addresses the limitation that TabPFN-like models are constrained to a fixed number of target dimensions due to a lack of target equivariance. Introduces a fully target-equivariant architecture that eliminates the "equivariance gap," matching or surpassing existing methods on benchmarks with more classes than seen during pre-training.

- **TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations** (NeurIPS 2025)
  *Alan Arazi, Eilam Shapira, Roi Reichart*
  [Paper](https://arxiv.org/abs/2505.18125) | [Code](https://github.com/alanarazi7/TabSTAR)
  > Introduces target-aware tokens that integrate the target variable's identity as model input, combined with semantic encoding via a pretrained language model. Achieves state-of-the-art on classification benchmarks with text features through multi-task pretraining and LoRA finetuning.

- **TARTE: Table Foundation Models — On Knowledge Pre-training for Tabular Learning** (TMLR 2025)
  *Myung Jun Kim, Félix Lefebvre, Gaëtan Brison, Alexandre Perez-Lebel, Gaël Varoquaux*
  [Paper](https://arxiv.org/abs/2505.14415) | [Code](https://github.com/soda-inria/tarte-ai)
  > A pre-trained tabular model that captures associations between strings and numbers via knowledge pre-training. Enables simple downstream learners like Ridge regression to become strong baselines, and its transformer backbone can be reused and specialized for complex tables.

- **TabPFN-TS: From Tables to Time — Extending TabPFN-v2 to Time Series Forecasting** (arXiv 2025)
  *Shi Bin Hoo, Samuel Müller, David Salinas, Frank Hutter*
  [Paper](https://arxiv.org/abs/2501.02945)
  > Treats forecasting as a tabular regression problem by combining lightweight temporal featurization with TabPFN-v2. Requires no time-series-specific pretraining and, despite its compact 11M parameters, achieves state-of-the-art performance on covariate-informed forecasting.

- **TabClustPFN: A Prior-Fitted Network for Tabular Data Clustering** (arXiv 2026)
  *Tianqi Zhao, Guanyang Wang, Yan Shuo Tan, Qiong Zhang*
  [Paper](https://arxiv.org/abs/2601.21656)
  > A PFN for tabular data clustering that performs amortized Bayesian inference over both cluster assignments and cluster cardinality. Pretrained on synthetic datasets, it clusters unseen datasets in a single forward pass without retraining or hyperparameter tuning.

- **TabImpute: Universal Zero-Shot Imputation for Tabular Data** (arXiv 2025)
  *Jacob Feitelberg, Dwaipayan Saha, Kyuseong Choi, Zaid Ahmad, Anish Agarwal, Raaz Dwivedi*
  [Paper](https://arxiv.org/abs/2510.02625) | [Code](https://github.com/jacobf18/tabular)
  > Building on TabPFN, a pre-trained transformer delivering accurate and fast zero-shot imputations. Introduces entry-wise featurization for 100x speedup and MissBench, a comprehensive benchmark with 42 OpenML tables and 13 missingness patterns.

- **TabDPT-Turbo: Efficient In-Context Learning for Tabular Prediction** (ICML 2026 FMSD Workshop)
  *Rasa Hosseinzadeh, Alex Labach, Zexin Xue, Shuyi Han, Valentin Thomas, Anthony L. Caterini*
  [OpenReview](https://openreview.net/forum?id=Y00pwFyrHR)
  > An efficiency-focused variant of TabDPT that speeds up in-context learning for tabular prediction.

- **Localized TabICLv2: Scaling Tabular In-Context Learning through k-NN** (ICML 2026 FMSD Workshop)
  *Beimnet Bekele Guta*
  [OpenReview](https://openreview.net/forum?id=ddITrUyMTB)
  > Scales TabICLv2 to larger datasets by restricting the in-context examples to a k-nearest-neighbor retrieval around each query.

- **FlexTab: Towards a Flexible Encoder-Decoder Architecture for Tabular In-Context Learning** (ICML 2026 FMSD Workshop)
  *Marek Polewczyk, Maximilian Schambach, Marco Spinaci, Sam Thelin, Johannes Höhne*
  [OpenReview](https://openreview.net/forum?id=fOph6xxdyP)
  > Proposes a flexible encoder-decoder backbone for tabular in-context learning, aiming to handle heterogeneous schemas and tasks.

- **Memory Efficient Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Shuting Luo, Monika Mikhail Kanaan, Cameron Gordon, Anna Leontjeva, Simon Lucey*
  [OpenReview](https://openreview.net/forum?id=1Ov4RAWuW4)
  > Studies techniques to reduce the memory footprint of tabular foundation models at inference and/or training time.

- **TFM-Retouche: A Lightweight Input-Space Adapter for Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Duong Nguyen, Mohammed Jawhar, Nicolas Chesneau*
  [OpenReview](https://openreview.net/forum?id=P1bvn0jvGX)
  > Adapts frozen tabular foundation models to new tasks via a lightweight input-space transformation rather than fine-tuning model weights.

- **Pocket Foundation Models: Distilling TFMs into CPU-Ready GBDTs** (ICML 2026 FMSD Workshop)
  *Aditya Tanna, Nassim Bouarour, Mohamed Bouadi, Vinay Sankarapu, Pratinav Seth*
  [OpenReview](https://openreview.net/forum?id=n1TUx8fHpv)
  > Distills tabular foundation models into compact gradient-boosted decision trees for fast, CPU-friendly deployment.

- **Bounded Context Management for Tabular Foundation Models on Stream Learning** (ICML 2026 FMSD Workshop, Spotlight)
  *Jinmo Lee, Doyun Choi, Moongi Choi, Jaemin Yoo*
  [OpenReview](https://openreview.net/forum?id=L94GfndIir)
  > Manages a bounded in-context set so that tabular foundation models can operate under streaming data with limited memory.

- **Online Test-Time Adaptation in Tabular Data with Minimal High-Certainty Samples** (ICML 2026 FMSD Workshop)
  *Mingming Zhang, Zhiqing Xiao, Junbo Zhao*
  [OpenReview](https://openreview.net/forum?id=rmpLcZtJ4l)
  > Performs online test-time adaptation for tabular models using a small set of high-confidence samples.

- **Agentic Data Intelligence for General Tabular Modeling** (ICML 2026 FMSD Workshop)
  *Jun-Peng Jiang, An-Yang Ji, Jia-Yi Zhu, Han-Jia Ye*
  [OpenReview](https://openreview.net/forum?id=pj1XShgzSv)
  > Explores an agentic pipeline that automates general tabular modeling tasks.

- **Correcting Class Imbalance in Prior-Data Fitted Networks for Tabular Classification** (ICML 2026 FMSD Workshop)
  *Samuel McDowell, Nathan Stromberg, Lalitha Sankar*
  [OpenReview](https://openreview.net/forum?id=96HA4mxjkH)
  > Addresses degraded PFN performance under class imbalance in tabular classification.

- **SurvivalPFN: Amortizing Survival Prediction via In-Context Bayesian Inference** (ICML 2026 FMSD Workshop, Spotlight)
  *Shi-ang Qi, Vahid Balazadeh, Michael Cooper, Russell Greiner, Rahul G. Krishnan*
  [OpenReview](https://openreview.net/forum?id=PDik7bpFhE)
  > A PFN-style model that performs amortized in-context Bayesian inference for survival (time-to-event) prediction.

- **SurvPFN: Towards Foundation Models for Survival Predictions** (ICML 2026 FMSD Workshop)
  *Samuel Böhm, Lennart Purucker, Frank Hutter, Pascal Schlosser*
  [OpenReview](https://openreview.net/forum?id=kDEHp7ytr2)
  > Works toward prior-data fitted foundation models tailored to survival analysis tasks.

- **Staying Alive: Uncensored Survival Analysis with Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Mariana Vargas Vieyra*
  [OpenReview](https://openreview.net/forum?id=2EFykQheZD)
  > Investigates applying tabular foundation models to survival analysis, handling censoring in time-to-event data.

### Synthetic Data & Generation

- **TabuLa: Harnessing Language Models for Tabular Data Synthesis** (arXiv 2023)
  *Zilong Zhao, Robert Birke, Lydia Y. Chen*
  [Paper](https://arxiv.org/abs/2310.12746)
  > Leverages language models for generating synthetic tabular data.

- **CTSyn: A Foundation Model for Cross Tabular Data Generation** (arXiv 2024)
  *Xiaofeng Lin, Chenheng Xu, Matthew Yang, Guang Cheng*
  [Paper](https://arxiv.org/abs/2406.04619)
  > A diffusion-based generative foundation model for tabular data. Uses an autoencoder to consolidate diverse tables into a unified latent space and a conditional latent diffusion model for generation, conditioned on table schema. Outperforms existing synthesizers on standard benchmarks in both utility and diversity.

- **A Generative Foundation Model for Heterogeneous Tabular Data** (ICML 2026 FMSD Workshop)
  *Xiangjian Jiang, Mingxuan Liu, Nikola Simidjievski, Tassilo Klein, Mateja Jamnik*
  [OpenReview](https://openreview.net/forum?id=RcsaxrdpfE)
  > A generative foundation model designed to synthesize heterogeneous tabular data across mixed column types.

- **TableFactory: Generating Semantically Linked Tabular Data via Multi-Agent Behavioral Simulation** (ICML 2026 FMSD Workshop)
  *Mingxuan Liu, Xiangjian Jiang, Johannes Hoffart, Tassilo Klein*
  [OpenReview](https://openreview.net/forum?id=3bzbWeaL5j)
  > Generates semantically linked tabular data by simulating the behavior of multiple interacting agents.

- **Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework** (ICML 2026 FMSD Workshop)
  *Junfeng Nie, Alvin Jin, Xiaohui Chen*
  [OpenReview](https://openreview.net/forum?id=RiaXCBoWje)
  > A hybrid framework combining top-down and bottom-up strategies for hierarchical synthetic tabular data generation.

- **Implicit Reward Alignment For Training Causally-Coherent Tabular Data Generators** (ICML 2026 FMSD Workshop)
  *Matea Gjika, Giuseppe Iannone, Luca Sfragara, Pavithra Harsha, Georgia Perakis*
  [OpenReview](https://openreview.net/forum?id=Bei8F38H9r)
  > Uses implicit reward alignment to train tabular data generators that preserve causal coherence.

- **From Noisy Oracles to Useful Constraints: LLM-Guided Constraint Selection for Synthetic Tabular Data** (ICML 2026 FMSD Workshop)
  *Tejumade Afonja, Joscha Cüppers, Mario Fritz*
  [OpenReview](https://openreview.net/forum?id=1k9oK22A3R)
  > Leverages LLMs to select useful constraints from noisy oracle signals to improve synthetic tabular data generation.

### Graph & Relational Data

- **GraphPFN: A Prior-Data Fitted Graph Foundation Model** (NeurIPS 2025)
  [Paper](https://arxiv.org/abs/2509.21489)
  > Introduces GraphPFN, a PFN designed for node-level prediction tasks. It utilizes a novel prior distribution of synthetic attributed graphs and incorporates graph-aware structured causal models to generate node attributes and targets.

- **Introducing KumoRFM: A Foundation Model for In-Context Learning on Relational Data** (Whitepaper, 2025)
  *Matthias Fey, Vid Kocijan, Federico Lopez, Jure Leskovec*
  [Blog & Whitepaper](https://kumo.ai/company/news/kumo-relational-foundation-model/)
  > A Relational Foundation Model (RFM) extending in-context learning to multi-table relational graphs. It uses a Relational Graph Transformer to reason across arbitrary schemas and supports zero-shot prediction.

- **Foundation Models for Tabular Data within Systemic Context (FMSLT)** (arXiv 2025)
  [Paper](https://arxiv.org/abs/2505.19825)
  > Proposes modeling tabular data not in isolation but with explicit semantic and operational context (e.g., linked tables, foreign keys), aiming for a richer foundation than flat tabular learning.

- **G2T-FM: Turning Tabular Foundation Models into Graph Foundation Models** (arXiv 2025)
  *Dmitry Eremeev, Gleb Bazhenov, Oleg Platonov, Artem Babenko, Liudmila Prokhorenkova*
  [Paper](https://arxiv.org/abs/2508.20906) | [Code](https://github.com/yandex-research/G2T-FM)
  > Transforms graph tasks into tabular ones by augmenting features with neighborhood aggregations and structural features (degree, PageRank, Laplacian eigenvectors). Achieves strong results in a fully in-context regime, outperforming existing GFMs and performing on par with well-tuned GNNs.

- **TFM4GAD: Tabular Foundation Models are Strong Graph Anomaly Detectors** (WebConf 2026)
  *Yunhui Liu, et al.*
  [Paper](https://arxiv.org/abs/2601.17301) | [Code](https://github.com/Cloudy1225/TFM4GAD)
  > Adapts tabular foundation models for graph anomaly detection by flattening the graph into an augmented feature table with Laplacian embeddings, structural characteristics, and anomaly-sensitive neighborhood aggregations. The best variant achieves 89.92% AUROC, surpassing the strongest trained baseline.

- **Large-Scale Pretraining unlocks Few-Shot Prediction for Relational Data** (ICML 2026 FMSD Workshop)
  *Rishabh Ranjan, Vignesh Kothapalli, Harshvardhan Agarwal, Charilaos I. Kanatsoulis, Roshan Reddy Upendra, Tom Palczewski, Carlos Guestrin, Jure Leskovec*
  [OpenReview](https://openreview.net/forum?id=oQINTd9din)
  > Shows that large-scale pretraining enables few-shot prediction across multi-table relational databases.

- **Parameter-Free Encoders Remain Viable for RDB Foundation Models** (ICML 2026 FMSD Workshop)
  *Linjie Xu, David Wipf*
  [OpenReview](https://openreview.net/forum?id=wRWaegFYMx)
  > Argues that parameter-free encoders remain a competitive design choice for relational database (RDB) foundation models.

- **PluRel-to-RDB-PFN: Schema-Guided Synthetic Relational Pretraining** (ICML 2026 FMSD Workshop)
  *Mohammad Sadeq Abolhasani, Viswanath Ganapathy*
  [OpenReview](https://openreview.net/forum?id=RpNvhdvd2v)
  > A PFN for relational databases pretrained on schema-guided synthetic relational data.

- **Context Window Failures in Relational Foundation Models** (ICML 2026 FMSD Workshop)
  *Denis Oliveira Correa, Francisco Galuppo Azevedo*
  [OpenReview](https://openreview.net/forum?id=lkuOIfXLwJ)
  > Analyzes how relational foundation models degrade when relevant context exceeds their effective context window.

- **Beyond Average Leaderboards: When Explicit Graph Priors Help Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Franck Le, Keith Grueneberg, Erich M. Nahum, Vadim Sheinin*
  [OpenReview](https://openreview.net/forum?id=4gmLDG0aGC)
  > Examines the conditions under which adding explicit graph priors benefits tabular foundation models, beyond aggregate leaderboard scores.

- **Can LLMs Use Relational Transformer Embeddings?** (ICML 2026 FMSD Workshop)
  *Francisco Galuppo Azevedo, Clarissa Lima Loures*
  [OpenReview](https://openreview.net/forum?id=Z2n7WcIy6j)
  > Investigates whether large language models can effectively consume embeddings produced by relational transformers.

### Causal Inference

- **Foundation Models for Causal Inference via Prior-Data Fitted Networks (CausalFM)** (NeurIPS 2025)
  *Yuchen Ma, et al.*
  [Paper](https://arxiv.org/abs/2506.10914)
  > A framework for training PFN-based foundation models for various causal inference settings (back-door, front-door adjustments).

- **Do-PFN: In-Context Learning for Causal Effect Estimation** (NeurIPS 2025)
  *Jake Robertson, et al.*
  [Paper](https://arxiv.org/abs/2506.06039)
  > Applies PFNs to estimate causal effects without knowledge of the underlying causal graph.

- **Causal Foundation Models with Continuous Treatments** (ICML 2026 FMSD Workshop)
  *Christopher Stith, Medha Barath, Vahid Balazadeh, Jesse C. Cresswell, Rahul G. Krishnan*
  [OpenReview](https://openreview.net/forum?id=DzcWAYcR2n)
  > Extends PFN-based causal foundation models to settings with continuous (rather than binary) treatments.

- **Foundation Models for Partial Causal Identification** (ICML 2026 FMSD Workshop)
  *Alexis Bellot, Anish Dhir*
  [OpenReview](https://openreview.net/forum?id=jCbehzZBsk)
  > Studies foundation models for cases where causal effects are only partially identifiable.

- **Inducing Causal Order through Tabular In-Context Learning** (ICML 2026 FMSD Workshop)
  *Sascha Xu, Sarah Mameche, Jilles Vreeken*
  [OpenReview](https://openreview.net/forum?id=U4KiOBxY1X)
  > Uses tabular in-context learning to infer a causal ordering among variables.

- **CausalTab: Pretraining Across Causal Environments for Tabular Causal Discovery** (ICML 2026 FMSD Workshop)
  *Zi-Rong Li, Si-Yang Liu, Tian-Zuo Wang, Han-Jia Ye*
  [OpenReview](https://openreview.net/forum?id=og3UVhP7M1)
  > Pretrains across diverse causal environments to enable tabular causal discovery.

- **Causal Foundation Models Perform Better without Post-treatment Variables** (ICML 2026 FMSD Workshop)
  *Junha Ham, Deokgyu Kim, Doeun Kim, Serjin Kim, Sanghack Lee*
  [OpenReview](https://openreview.net/forum?id=ULoLF1aOo1)
  > Shows that excluding post-treatment variables improves the accuracy of causal foundation models.

- **A Causal Foundation Model for Structure and Outcome Prediction** (ICML 2026 FMSD Workshop)
  *Max Zhu, Martino Mansoldo, Ching-Hao Wang, Stefan Groha*
  [OpenReview](https://openreview.net/forum?id=GOf9c4lOCf)
  > A causal foundation model that jointly targets causal structure and outcome prediction.

- **Bayesian Tabular Few-shot Learning with Causal Information** (ICML 2026 FMSD Workshop)
  *Ole Ossen, Jake Robertson, Arik Reuter, Magnus Bühler, Lennart Purucker, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=2yvEiFhNCT)
  > Incorporates causal information into Bayesian few-shot learning for tabular tasks.

### Physical Systems & ODEs

- **Decoupled-Value Attention for Prior-Data Fitted Networks: GP Inference for Physical Equations** (NeurIPS 2025)
  *Kaustubh Sharma, et al.*
  [Paper](https://arxiv.org/abs/2509.20950)
  > Introduces Decoupled-Value Attention (DVA) to improve performance in physical systems and high-dimensional regression. Specifically targets inference for physical equations (ODEs/PDEs) using PFNs.

### Optimization

- **Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks** (NeurIPS 2023)
  *Steven Adriaensen, et al.*
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html)
  > LC-PFN is trained to extrapolate learning curves, enabling accurate posterior predictive distributions and efficient model selection / early stopping.

- **PFNs4BO: In-Context Learning for Bayesian Optimization** (ICML 2023)
  *Samuel Müller, et al.*
  [Paper](https://arxiv.org/abs/2305.17535)
  > Explores the application of PFNs as surrogate models for Bayesian Optimization.

- **Context-Aware Learning Curve Extrapolation with Prior-Data Fitted Networks** (ICML 2026 FMSD Workshop)
  *Cheng Yan, Steven Adriaensen, Tom Julian Viering*
  [OpenReview](https://openreview.net/forum?id=oN4FIXBVeS)
  > Extends PFN-based learning-curve extrapolation to incorporate additional context about the training run.

- **Can Tabular Foundation Models Predict Algorithm Runtime Distributions?** (ICML 2026 FMSD Workshop)
  *Hagverdi Ibrahimli, Steven Adriaensen*
  [OpenReview](https://openreview.net/forum?id=H6t3IZfnqt)
  > Investigates whether tabular foundation models can predict the runtime distributions of algorithms.

- **Covariance-Aware Transformers for Quadratic Programming and Decision Making** (ICML 2026 FMSD Workshop)
  *Kutay Tire, Yufan Zhang, Ege Onur Taga, Samet Oymak*
  [OpenReview](https://openreview.net/forum?id=XLFOyHaZmq)
  > Introduces covariance-aware transformers for quadratic programming and downstream decision-making problems.

- **Lookahead Automated Feature Engineering for Tabular Prediction via Kaggle-Guided Knowledge Transfer** (ICML 2026 FMSD Workshop)
  *Si-Yang Liu, Zong-Da Li, Chenming Xu, Han Li, Rui-Qiao Chen, Han-Jia Ye*
  [OpenReview](https://openreview.net/forum?id=FNtlPbGvwc)
  > Automates feature engineering for tabular prediction using lookahead search and knowledge transferred from Kaggle solutions.

- **Evolutionary Feature Engineering for Structured Data** (ICML 2026 FMSD Workshop)
  *Ege Onur Taga, Yilin Zhuang, Muhammed Emrullah Ildiz, Petros Mol, Abhimanyu Das, Karthik Duraisamy, Samet Oymak*
  [OpenReview](https://openreview.net/forum?id=EruNY8fps7)
  > Applies evolutionary search to automatically discover useful features for structured data.

### Architectures & Training

- **MotherNet: A Foundational Hypernetwork for Tabular Classification** (arXiv 2023)
  *Samuel Müller, Frank Hutter*
  [Paper](https://arxiv.org/abs/2312.08598) | [Code](https://github.com/automl/MotherNet)
  > A hypernetwork trained to generate the weights of a child network for a new tabular task in a single forward pass.

- **TabularFM: An Open Framework For Tabular Foundational Models** (arXiv 2024)
  [Paper](https://arxiv.org/abs/2406.09837)
  > A framework and dataset corpus designed to facilitate the training of various generative and tabular foundation models.

- **State-Space Models for Tabular Prior-Data Fitted Networks** (arXiv 2025)
  *Felix Koch, Marcel Wever, Fabian Raisch, Benjamin Tischler*
  [Paper](https://arxiv.org/abs/2510.14573)
  > Investigates using Hydra, a bidirectional linear-time structured state space model (SSM), as an alternative to Transformers in TabPFN. Proposes repeated context permutations (RCP) to reduce order-sensitivity. Achieves competitive predictive performance with reduced computational and memory complexity.

- **MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning** (CVPR 2026)
  *Wall Kim, Chaeyoung Song, Hanul Kim*
  [Paper](https://arxiv.org/abs/2602.20223) | [Code](https://github.com/too-z/MultiModalPFN)
  > Extends TabPFN to handle tabular and non-tabular modalities (images, text) in a unified manner. Uses modality projectors with multi-head gated MLP and cross-attention pooler to transform non-tabular embeddings into tabular-compatible tokens.

- **Robust Tabular Foundation Models (RTFM)** (arXiv 2025)
  *Matthew Peroni, Franck Le, Vadim Sheinin*
  [Paper](https://arxiv.org/abs/2512.03307)
  > A model-agnostic adversarial training framework that adapts the synthetic data generator to emphasize challenging datasets during training. Applied to TabPFN V2, RTFM improves benchmark performance by up to 6% in mean normalized AUC using fewer than 100K additional synthetic datasets.

- **Speedrunning Tabular Foundation Model Pretraining** (ICML 2026 FMSD Workshop)
  *Salih Bora Öztürk, Alexander Pfefferle, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=QT1ySCPeW3)
  > Investigates how to dramatically accelerate the pretraining of tabular foundation models.

- **Optimizing Pre-Training of Tabular Foundation Models by Shaping Geometry** (ICML 2026 FMSD Workshop)
  *Humzah Merchant, Sriniketh Vangaru, Randall Balestriero*
  [OpenReview](https://openreview.net/forum?id=IYnHchzvYB)
  > Improves tabular foundation model pretraining by shaping the geometry of the learned representation space.

- **RAD-TFM: Robust and Domain-Adapted Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Matthew Peroni, Franck Le, Vadim Sheinin*
  [OpenReview](https://openreview.net/forum?id=5BkHclEOW0)
  > Proposes robustness and domain-adaptation techniques for tabular foundation models.

- **Mutual Information-Guided Corruption for Improved Self-Supervised Representation Learning in Tabular Data** (ICML 2026 FMSD Workshop)
  *Michael Lawson, Emerald Sy, Kehui Zhang, Raymond H. Chan, Kannie W. Y. Chan, Rosa H. M. Chan*
  [OpenReview](https://openreview.net/forum?id=T8qbmiE0yZ)
  > Uses mutual information to guide the corruption process in self-supervised tabular representation learning.

- **Enhancing Tabular Learners with Context-Aware Semantic Embeddings** (ICML 2026 FMSD Workshop)
  *Günther Schindler, Maximilian Schambach, Johannes Höhne*
  [OpenReview](https://openreview.net/forum?id=QArxQg4U71)
  > Augments tabular learners with context-aware semantic embeddings of columns and values.

- **Towards Pretraining Text Encoders for TabPFN** (ICML 2026 FMSD Workshop)
  *Mustafa Tajjar, Alexander Pfefferle, Lennart Purucker, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=dA8IZj8R46)
  > Explores pretraining dedicated text encoders to handle textual columns within TabPFN.

- **HGR-TabE: Universal Tabular Embeddings via Maximal Correlation Alignment** (ICML 2026 FMSD Workshop)
  *Niharika S. D'Souza, Liane Vogel, Kavitha Srinivas, Sola Shirai, Oktie Hassanzadeh, Horst Samulowitz*
  [OpenReview](https://openreview.net/forum?id=FRj6pclhXE)
  > Learns universal tabular embeddings using maximal correlation (HGR) alignment.

### Theory & Analysis

- **Statistical Foundations of Prior-Data Fitted Networks** (ICML 2023)
  *Thomas Nagler*
  [Paper](https://arxiv.org/abs/2305.11097)
  > Provides a theoretical framework for PFNs, offering a frequentist interpretation and analyzing their convergence properties.

- **What exactly has TabPFN learned to do?** (ICLR 2024 Blogposts Track / arXiv 2025)
  *Calvin McCarter*
  [Paper](https://arxiv.org/abs/2502.08978) | [Code](https://github.com/calvinmccarter/tabpfn-eval)
  > An empirical analysis treating TabPFN as a black-box function approximator to understand its learned inductive biases. Explores behavior on simple 1D/2D settings and out-of-distribution tasks (gene expression, MNIST). Includes 2025 re-analysis on TabPFN-v2, showing it can approximately learn the parity function with impressive sample efficiency.

- **Towards Fair In-Context Learning with Tabular Foundation Models** (arXiv 2025)
  *Patrik Kenfack, Samira Ebrahimi Kahou, Ulrich Aïvodji*
  [Paper](https://arxiv.org/abs/2505.09503) | [Code](https://github.com/patrikken/Fair-TabICL)
  > The first investigation of fairness in tabular in-context learning, evaluating TabPFNv2, TabICL, and TabDPT. Finds that an uncertainty-based sample selection strategy consistently improves group fairness metrics (demographic parity, equalized odds) with minimal impact on accuracy.

- **Light-Weight Benchmarks Reveal the Hidden Hardware Cost of Zero-Shot Tabular Foundation Models** (arXiv 2025)
  *Ishaan Gangwani, Aayam Bansal*
  [Paper](https://arxiv.org/abs/2512.00888)
  > A reproducible benchmark pairing test accuracy with wall-clock latency, peak CPU RAM, and peak GPU VRAM. Shows that zero-shot TFMs incur up to 10,000x latency penalties vs. tree ensembles, suggesting their main value lies in rapid prototyping on small tables rather than production inference at scale.

- **Do Tabular Foundation Models Learn Rules or Memorize Exemplars?** (ICML 2026 FMSD Workshop)
  *Amir Rezaei Balef, Mykhailo Koshil, Behzad Nourani-Koliji, Katharina Eggensperger*
  [OpenReview](https://openreview.net/forum?id=9nCMtYGxQt)
  > Probes whether tabular foundation models generalize via rule learning or rely on memorizing training exemplars.

- **Probing Memorization of Tabular In-Context Learning** (ICML 2026 FMSD Workshop)
  *Francesco Capano, Jonas Böhler*
  [OpenReview](https://openreview.net/forum?id=7DZ3u0SD4b)
  > Investigates the extent to which tabular in-context learners memorize their context examples.

- **Tabular Foundation Models Are Effectively Shallow** (ICML 2026 FMSD Workshop)
  *Irene Cannistraci, Julia E. Vogt*
  [OpenReview](https://openreview.net/forum?id=kCnZUf1VYC)
  > Argues that tabular foundation models behave as effectively shallow function approximators.

- **Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function** (ICML 2026 FMSD Workshop)
  *Atharva Gupta, Dhruv Kumar, Murari Mandal, Saurabh Deshpande*
  [OpenReview](https://openreview.net/forum?id=LXSawSSeA9)
  > Uses causal localization to identify which attention heads carry out specific computations inside TabPFN.

- **Statistically Indistinguishable, Operationally Distinct: A Formal Barrier for Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Tassilo Klein, Johannes Hoffart*
  [OpenReview](https://openreview.net/forum?id=TUYc2XUdwz)
  > Establishes a formal barrier showing tabular foundation models can be statistically indistinguishable yet operationally distinct.

- **On the Uncertainty in Prior-Data Fitted Network Pretraining** (ICML 2026 FMSD Workshop)
  *Manuel Hülskamp, Julius Kobialka, Emanuel Sommer, David Rügamer*
  [OpenReview](https://openreview.net/forum?id=5Shv4Ar4N9)
  > Analyzes sources of uncertainty arising during the pretraining of prior-data fitted networks.

- **What You Pretrain On Matters: Synthetic Task Distributions Determine Tabular Foundation Model Quality** (ICML 2026 FMSD Workshop)
  *Mohamed Bouadi, Nassim Bouarour, Shivam Dubey, Varun Kulkarni, Aditya Tanna, Vinay Sankarapu*
  [OpenReview](https://openreview.net/forum?id=QfXHxB9VSS)
  > Shows that the choice of synthetic task distribution used in pretraining strongly determines tabular foundation model quality.

- **When Data Is Scarce: The Strength of the Prior in Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Florian D. Leeuwen, Sara van Erp*
  [OpenReview](https://openreview.net/forum?id=dznQA3JHfI)
  > Examines how strongly the learned prior drives tabular foundation model predictions in low-data regimes.

- **Towards Evaluating Data Priors for Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Zeynep Türkmen, Kürşat Kaya, Alexander Pfefferle, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=GUDjbVGFc1)
  > Works toward principled evaluation of the data priors used to train tabular foundation models.

- **Inspectable Tabular Foundation Models via In-Context Kernel Learning** (ICML 2026 FMSD Workshop)
  *Ratmir Miftachov, Bruno Charron, Simon Valentin*
  [OpenReview](https://openreview.net/forum?id=Q1P83jtxXY)
  > Makes tabular foundation models more inspectable by framing in-context learning as kernel learning.

- **Objective and data-driven Bayesian inference using TabPFN models** (ICML 2026 FMSD Workshop)
  *Elias Chaibub Neto*
  [OpenReview](https://openreview.net/forum?id=YRcXyqoemK)
  > Uses TabPFN models to perform objective, data-driven Bayesian inference.

- **Lost in Aggregation: How Benchmarks Overlook Irreplaceable Model Strengths** (ICML 2026 FMSD Workshop)
  *Andrej Tschalzev, Stefan Lüdtke, Heiner Stuckenschmidt, Christian Bartelt*
  [OpenReview](https://openreview.net/forum?id=5B1lb8jrgo)
  > Argues that aggregate benchmark scores can obscure model-specific strengths that matter in practice.

- **Revisiting Metafeatures to Explain Model Differences on Tabular Data** (ICML 2026 FMSD Workshop)
  *Markus Herre, Andrej Tschalzev, Sascha Marton, Christian Bartelt*
  [OpenReview](https://openreview.net/forum?id=FJSkVoD4k3)
  > Revisits dataset metafeatures to explain when and why models differ on tabular data.

- **Training Fair Tabular Foundation Models** (ICML 2026 FMSD Workshop, Spotlight)
  *Patrik Kenfack, Jesse C. Cresswell, Anthony L. Caterini, Samira Ebrahimi Kahou, Ulrich Aïvodji*
  [OpenReview](https://openreview.net/forum?id=ajIvCEbadL)
  > Proposes methods for training tabular foundation models that satisfy group fairness criteria.

- **FairOpt-PFN: Amortized Counterfactual Fairness with Optimal Fair Targets** (ICML 2026 FMSD Workshop)
  *Enes Hasani, Jake Robertson, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=M8o7jbKX9P)
  > A PFN approach to amortized counterfactual fairness that learns optimal fair targets.

- **Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice** (ICML 2026 FMSD Workshop)
  *Yingshuo Wang, Xian Sun, Yanhang Li, Zhichao Fan, Zexin Zhuang*
  [OpenReview](https://openreview.net/forum?id=Tda0qprAXQ)
  > Audits and corrects economic validity issues when tabular foundation models are used for discrete-choice modeling.

- **Dataset Inference for Data Provenance and Privacy Auditing in Tabular Foundation Models** (ICML 2026 FMSD Workshop, Spotlight)
  *Dariush Wahdany, Jesse C. Cresswell, Naiqing Guan, Atiyeh Ashari Ghomi, Franziska Boenisch, Adam Dziedzic*
  [OpenReview](https://openreview.net/forum?id=u2uOPq1u6I)
  > Uses dataset inference to audit data provenance and privacy in tabular foundation models.

- **TabPATE: Differentially Private Tabular In-Context Learning Without Public Data** (ICML 2026 FMSD Workshop)
  *Dariush Wahdany, Matthew Jagielski, Jesse C. Cresswell, Adam Dziedzic, Franziska Boenisch*
  [OpenReview](https://openreview.net/forum?id=Q5iGt2Z0Ql)
  > Brings differentially private in-context learning to tabular data without requiring public data.

## Benchmarks & Evaluation

Based on [van Breugel & van der Schaar (2024)](https://arxiv.org/abs/2405.01147), LTM benchmarks should evaluate models across multiple dimensions:

### Tasks

| Task | Description | Metrics |
|------|-------------|---------|
| **Supervised Learning** | Predictive performance using LTM embeddings or direct generation | Accuracy, AUC, RMSE |
| **Synthetic Data Generation** | Conditional and unconditional generation quality | Train-on-synthetic-test-on-real, fidelity, diversity, ε-identifiability |
| **Imputation** | Single imputation (E[X_unobs\|X_obs]) or multiple imputation (sampling from p(X_unobs\|X_obs)) | MAE, coverage, calibration |
| **LTMs for Science** | Dimensionality reduction, data cleaning, clustering, cross-dataset retrieval | Task-specific metrics |

### Experimental Settings

| Setting | Description | Key Considerations |
|---------|-------------|-------------------|
| **Few-shot** | Performance on new datasets with few samples | With/without fine-tuning; robustness |
| **Zero-shot** | No samples from target dataset | Requires true generalization |
| **In-distribution** | Hold-out test from training data | Quantify generalization gaps |

### Related Benchmarks

- **[TabArena](https://tabarena.ai)** - A continuously maintained "living" benchmark for tabular ML (NeurIPS 2025, [paper](https://arxiv.org/abs/2506.16791)). Curates 51 real-world datasets and evaluates tree-based models, neural networks, and tabular foundation models with a public Elo leaderboard. The de facto standard for ranking modern TFMs.
- **TabZilla** - Comprehensive tabular data benchmark
- **OpenML** - Large collection of tabular datasets
- **Kaggle Competitions** - Real-world tabular challenges

#### From the ICML 2026 FMSD Workshop

- **[MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://openreview.net/forum?id=r19rlhngOD)** - A benchmark for multimodal tabular learning that combines text and image features. *Alan Arazi, Eilam Shapira, Shoham Grunblat, Mor Ventura, Elad Hoffer, Gioia Blayer, David Holzmüller, Lennart Purucker, Gaël Varoquaux, Frank Hutter, Roi Reichart.*
- **[Are Tabular Foundation Model Rankings Reliable? A Generalizability Theory Analysis of RelBench and DBInfer](https://openreview.net/forum?id=7jbzkGYag6)** - Uses generalizability theory to assess the reliability of TFM rankings on RelBench and DBInfer. *Dinesh Katupputhur Ramprasath, Tom Palczewski, Joe Meyer, Roshan Reddy Upendra, Minghua Li.*
- **[Realistic Evaluation of TabPFN v2.5 in Open Environments](https://openreview.net/forum?id=qway3qFkUL)** - Evaluates TabPFN v2.5 under realistic open-environment conditions. *Zi-Jian Cheng, Ziyi Jia, Lan-Zhe Guo.*
- **[Ensembling Tabular Foundation Models: A Diversity Ceiling and a Calibration Trap](https://openreview.net/forum?id=FZaZoe67ne)** - Analyzes the limits of ensembling TFMs, highlighting a diversity ceiling and a calibration trap. *Aditya Tanna, Yash Jignesh Desai, Pratinav Seth, Mohamed Bouadi, Nassim Bouarour, Vinay Sankarapu.*
- **[Exploring Differences Between Tabular Enterprise Data and Public Benchmarks](https://openreview.net/forum?id=PXSBtjo3Gd)** - Contrasts the characteristics of enterprise tabular data with public benchmarks. *Myung Jun Kim, Maximilian Schambach, Frank Essenberger, Andre Sres, Johannes Höhne.*
- **[Benchmarking Attention for Tabular Foundation Models](https://openreview.net/forum?id=rwtcugrpDq)** - Benchmarks attention mechanisms used in tabular foundation models. *Maximilian Schambach, Clemens Biehl, Sam Thelin.*
- **[Beyond Accuracy: Toward Trustworthy Tabular Foundation Models in Industrial Applications](https://openreview.net/forum?id=r3RAi8Kqzl)** - Looks beyond accuracy toward trustworthiness of TFMs in industrial settings. *Johannes Keler, Matthias Woehrle, Jan Achterhold, Mark Schillinger, Maria Lyssenko, Luiz Ricardo Douat.*
- **[Benchmarking Tabular Foundation Models for Churn Prediction](https://openreview.net/forum?id=LtXucHLtiN)** - Benchmarks tabular foundation models on customer churn prediction. *Sobhan Seyedzadeh, Mostafa Karimi.*

## Tutorials & Talks

- **[Getting Started with PFNs](https://github.com/automl/PFNs/blob/main/Tutorial_1_Basics.ipynb)**: Official tutorial notebook from the PFNs repository.

## Key References from Position Paper

The following works are referenced in [van Breugel & van der Schaar (2024)](https://arxiv.org/abs/2405.01147) as relevant to LTM development:

- Tabular data surveys: [Borisov et al. (2022)](https://arxiv.org/abs/2110.01889), [Shwartz-Ziv & Armon (2022)](https://arxiv.org/abs/2106.03253)
- Benchmarking: [Grinsztajn et al. (2022)](https://arxiv.org/abs/2207.08815), [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959)
- Synthetic data: [van Breugel & van der Schaar (2023)](https://arxiv.org/abs/2302.04311)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to add new papers, resources, or fix links.

1. Fork the repository.
2. Create a new branch.
3. Add your resource.
4. Submit a PR.

---
*Last updated: June 2026*
