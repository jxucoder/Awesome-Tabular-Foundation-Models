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
  - [Time Series & Sequential Data](#time-series--sequential-data)
  - [Causal Inference](#causal-inference)
  - [Physical Systems & ODEs](#physical-systems--odes)
  - [Optimization](#optimization)
  - [Architectures & Training](#architectures--training)
  - [Theory & Analysis](#theory--analysis)
  - [Applications & Domain Studies](#applications--domain-studies)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Books & Learning Resources](#books--learning-resources)
- [Tutorials & Talks](#tutorials--talks)
- [Workshops & Venues](#workshops--venues)
- [Contributing](#contributing)

## Software & Code

- **[PFNs](https://github.com/automl/PFNs)**: The official general-purpose library for creating and training Prior-Data Fitted Networks.
- **[TabPFN](https://github.com/PriorLabs/TabPFN)**: The official repository for the TabPFN model.
- **[TabICL](https://github.com/soda-inria/tabicl)**: Official implementation of TabICL and TabICLv2, open-source tabular foundation models for classification and regression.
- **[TabSwift](https://github.com/LAMDA-Tabular/TabSwift)**: Official code for TabSwift, a lightweight row-wise attention TFM with adaptive early-exit.
- **[Nori](https://github.com/Synthefy/synthefy-nori)**: Synthefy's fully open-source (weights, inference, and training code) tabular foundation model for regression (`pip install synthefy-nori`).
- **[TabTune](https://arxiv.org/abs/2511.02802)**: A unified library for inference and fine-tuning across multiple tabular foundation models.
- **[TabArena](https://github.com/autogluon/tabarena)**: A living benchmarking system for tabular ML with a public leaderboard at [tabarena.ai](https://tabarena.ai).
- **[TabPFN-3 weights](https://huggingface.co/Prior-Labs/tabpfn_3)**: Prior Labs' TabPFN-3 checkpoints (research / internal-evaluation license), scaling in-context learning to 1M rows; see the [changelog](https://docs.priorlabs.ai/changelog/tabpfn-3).
- **[TabPFN-3.5](https://priorlabs.ai/technical-reports/tabpfn-3-5)**: Current Prior Labs flagship (`pip install -U tabpfn`); Fast / Plus / Thinking variants. Plus is also available on SAP AI Core. Docs: [docs.priorlabs.ai](https://docs.priorlabs.ai).
- **[Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular)**: NVIDIA's open TFM for classification and regression (Small 28M / Medium ~100M / Large 215M), pretrained only on synthetic data; `pip install structured-data-models` ([code](https://github.com/NVIDIA/structured-data-models)). OpenMDW-1.1 license, commercial use allowed.
- **[Mitra](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)**: Amazon's TFM trained on a curated mixture of synthetic priors, shipped inside [AutoGluon](https://github.com/autogluon/autogluon); Mitra-v2 ([report](https://arxiv.org/abs/2609.04540)) adds a 2D-attention backbone and a much larger synthetic task distribution.
- **[LimiX](https://github.com/limix-ldm-ai/LimiX)**: Open large structured-data models from StableAI / Tsinghua (LimiX-16M, LimiX-2M, and the 400M [LimiX-2](https://huggingface.co/stable-ai/LimiX-2)) covering classification, regression, imputation, and (in LimiX-2) causal-skeleton recovery.
- **[Orion-MSP](https://github.com/Lexsi-Labs/Orion-MSP)** / **[Orion-BiX](https://github.com/Lexsi-Labs/Orion-BiX)**: Lexsi Labs' open tabular ICL models with multi-scale sparse attention and bi-axial attention respectively.
- **[ConTextTab / SAP RPT-1](https://github.com/SAP-samples/sap-rpt-1-oss)**: SAP's semantics-aware tabular in-context learner trained on real-world tables with language-model embeddings for text cells.
- **[TabDPT inference](https://github.com/layer6ai-labs/TabDPT-inference)**: Layer 6's inference code for TabDPT and TabDPT-Turbo.
- **[TFM-Playground](https://github.com/automl/TFM-Playground)**: AutoML.org research playground for training and evaluating small tabular foundation models and their synthetic priors.
- **[modded-nanoTabPFN](https://github.com/borawhocodess/modded-nanotabpfn)**: Speedrun-style minimal TabPFN pretraining code from *Speedrunning Tabular Foundation Model Pretraining*.
- **[RDBLearn](https://github.com/HKUSHXLab/rdblearn)**: scikit-learn-style toolkit that auto-featurizes relational databases and runs any tabular ICL backend on the result.
- **[RDB-PFN](https://github.com/MuLabPKU/RDBPFN)**: First relational foundation model trained purely on synthetic multi-table databases.
- **[TabPrep](https://github.com/atschalz/tabprep)**: Lightweight feature-engineering pipeline that lifts tree, neural, and foundation models on TabArena.
- **[TACTICL](https://github.com/Hebog/tfm_compression)**: Task-aware layer pruning + adapters for compressing tabular ICL models.
- **[TabPFN IML](https://github.com/david-rundel/tabpfn_iml)**: Interpretability methods (feature effects, importance, Shapley) specialised for TabPFN.

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

- **Position: The Future of Bayesian Prediction Is Prior-Fitted** (ICML 2025 Position Track)
  *Samuel Müller, Arik Reuter, Noah Hollmann, David Rügamer, Frank Hutter*
  [Paper](https://arxiv.org/abs/2505.23947)
  > Argues that PFNs and other amortized-inference approaches are the future of Bayesian prediction: pre-training compute can be spent once on synthetic priors and reused across low-data problems. Surveys the growth of PFNs from small Bayesian tasks to tabular, time-series, and causal foundation models.

- **Data Language Models: A New Foundation Model Class for Tabular Data** (arXiv 2026)
  *Eda Erol, Giuliano Pezzoli, Ozer Cem Kelahmet*
  [Paper](https://arxiv.org/abs/2605.06290)
  > Position-style paper proposing Data Language Models (DLMs) that consume raw cell values natively, without serialization or preprocessing pipelines, as a tabular data layer for models, agents, and vertical applications.

- **The Current Generation of Tabular Foundation Models: A Critical Review** (Machine Learning and Knowledge Extraction, 2026)
  [Paper](https://www.mdpi.com/2504-4990/8/8/244)
  > Journal review of the 2024–2026 TFM release train (TabPFN, TabICL, Mitra, LimiX, Orion, ...), framing the shift from per-dataset training to amortized in-context inference and cataloguing open problems.

- **The State of Tabular Foundation Models (2026)** (Blog, Feb 2026)
  *Christoph Molnar (Mindful Modeler)*
  [Post](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models)
  > Practitioner-oriented overview of 16 TFMs released between 2021 and 2026, the emerging TFM start-up ecosystem, licensing concerns, and a recommendation of TabICLv2 as the fast, fully open default.

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

- **TabPFN-3.5: Technical Report** (Technical Report, Sep 2026)
  *Prior Labs Team (Benjamin Jäger, Nick Erickson, Léo Grinsztajn, Felix Birkel, Klemens Flöge, Oscar Key, et al.)*
  [Paper](https://arxiv.org/abs/2609.17895) | [Report](https://priorlabs.ai/technical-reports/tabpfn-3-5) | [Code](https://github.com/PriorLabs/TabPFN)
  > Flagship follow-up to TabPFN-3. Reports first place on TabArena and BeyondArena, with the largest gains on the data practitioners actually have: non-i.i.d. temporal and grouped splits, high-cardinality categoricals, wide tables, and tables mixed with text and images. Ships Fast (up to ~3–6× faster than the base model), Plus (advanced text/date handling; also on SAP AI Core), and Thinking (inference-time compute, up to 12× faster than TabPFN-3-Thinking) variants. Code is Apache 2.0; weights use the TABPFN-3.5 License.

- **TabFM: A Zero-Shot Foundation Model for Tabular Data** (Technical Report, Sep 2026)
  *Weihao Kong, Erez Louidor Ilan, Shuxin Nie, Taman Narayan, Rajat Sen, Yichen Zhou, Deqing Fu, Samet Oymak, Abhimanyu Das (Google)*
  [Paper](https://arxiv.org/abs/2609.37959)
  > Technical report for Google's TabFM: a 400M-parameter in-context learner trained entirely on synthetic tables from structural causal models, giving calibrated zero-shot predictions in a single forward pass. Across all 51 TabArena datasets (38 classification, 13 regression), zero-shot TabFM ranks first among default TFMs and beats tuned AutoML pipelines. Two extensions over the same frozen weights go further: TabFM+ (multi-view feature expansion, ensembling, post-hoc calibration) and TabFM-Auto (LLM-guided data processing and feature engineering).

- **TabFM-Auto: Self-Evolving Pipelines for Tabular Foundation Models** (arXiv, Sep 2026)
  *Deqing Fu, Huangyuan Su, Rajat Sen, Taman Narayan, Sujay Sanghavi, Abhimanyu Das, Weihao Kong (Google)*
  [Paper](https://arxiv.org/abs/2609.37989) | [Project](https://deqingfu.github.io/tabfm-auto/)
  > Pairs a frozen TabFM with an LLM agent that evolves the data pipeline around it (cleaning, feature engineering, context selection, post-processing) using column names, task descriptions, auxiliary files, and validation feedback. Five TabFM-Auto configurations take the top five spots on TabArena, with the best lifting TabFM from 1785 to 2013 Elo; discovered pipelines transfer to other frozen TFMs (+69 to +143 Elo) with no further search, and it ranks first among MLE agents on MLE-Bench's 8 tabular competitions. The project page has an explorer of the features the LLM agents found.

- **NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Foundation Models** (Blog, Sep 2026)
  *Jingang Qu, Valter Hudovernik, Martin Jurkovic, Cedric Lorenz, Akihiro Nitta, Dmitry Gordeev, Federico Lopez, Ramona Bendias, Gilberto Titericz Jr., Aleksandar S. Sokolovski, Isabel Hulseman, Jure Leskovec, Matthias Fey (NVIDIA / Kumo)*
  [Blog](https://huggingface.co/blog/nvidia/kumo-tabular) | [Model](https://huggingface.co/nvidia/Kumo-Tabular) | [Code](https://github.com/NVIDIA/structured-data-models)
  > Open single-table TFM from the KumoRFM team, pretrained only on synthetic data and released in three sizes (28M–215M). Column, row, and in-context attention with Fourier feature embeddings, induced self-attention over columns, rotary row attention, Test-GQA, and length-aware attention temperature. Reports first place on TabArena (Elo 1950), BeyondArena (Elo 1418), TALENT, and ScoringBench, and ~17× faster inference than LimiX-2 on a single GPU. Numerical and categorical features only; up to 10 classes natively (more via ECOC). OpenMDW-1.1 license.

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

- **TabICL: A Tabular Foundation Model for In-Context Learning on Large Data** (ICML 2025)
  *Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan*
  [Paper](https://arxiv.org/abs/2502.05564) | [Code](https://github.com/soda-inria/tabicl)
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

- **TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model** (ICML 2026)
  *Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan*
  [Paper](https://arxiv.org/abs/2602.11139) | [ICML](https://icml.cc/virtual/2026/poster/63874) | [Code](https://github.com/soda-inria/tabicl)
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

- **TabClustPFN: A Prior-Fitted Network for Tabular Data Clustering** (NeurIPS 2026)
  *Tianqi Zhao, Guanyang Wang, Yan Shuo Tan, Qiong Zhang*
  [Paper](https://arxiv.org/abs/2601.21656) | [OpenReview](https://openreview.net/forum?id=kgZzOmPAq0)
  > A PFN for tabular data clustering that performs amortized Bayesian inference over both cluster assignments and cluster cardinality. Pretrained on synthetic datasets, it clusters unseen datasets in a single forward pass without retraining or hyperparameter tuning.

- **TabImpute: Universal Zero-Shot Imputation for Tabular Data** (arXiv 2025)
  *Jacob Feitelberg, Dwaipayan Saha, Kyuseong Choi, Zaid Ahmad, Anish Agarwal, Raaz Dwivedi*
  [Paper](https://arxiv.org/abs/2510.02625) | [Code](https://github.com/jacobf18/tabular)
  > Building on TabPFN, a pre-trained transformer delivering accurate and fast zero-shot imputations. Introduces entry-wise featurization for 100x speedup and MissBench, a comprehensive benchmark with 42 OpenML tables and 13 missingness patterns.

- **TabDPT-Turbo: Efficient In-Context Learning for Tabular Prediction** (ICML 2026 FMSD Workshop)
  *Rasa Hosseinzadeh, Alex Labach, Zexin Xue, Shuyi Han, Valentin Thomas, Anthony L. Caterini*
  [Paper](https://arxiv.org/abs/2608.01400) | [OpenReview](https://openreview.net/forum?id=Y00pwFyrHR) | [Code](https://github.com/layer6ai-labs/TabDPT-inference)
  > An efficiency-focused variant of TabDPT that speeds up in-context learning for tabular prediction.

- **Localized TabICLv2: Scaling Tabular In-Context Learning through k-NN** (ICML 2026 FMSD Workshop)
  *Beimnet Bekele Guta*
  [Paper](https://arxiv.org/abs/2608.16429) | [OpenReview](https://openreview.net/forum?id=ddITrUyMTB)
  > Scales TabICLv2 to larger datasets by restricting the in-context examples to a k-nearest-neighbor retrieval around each query.

- **FlexTab: A Flexible Encoder-Decoder Architecture for In-Context Learning Across Diverse Tabular Tasks** (NeurIPS 2026; ICML 2026 FMSD Workshop)
  *Marek Polewczyk, Maximilian Schambach, Marco Spinaci, Sam Thelin, Johannes Höhne*
  [Paper](https://arxiv.org/abs/2606.30336) | [OpenReview](https://openreview.net/forum?id=LnVFK5y1uh) | [Workshop version](https://openreview.net/forum?id=fOph6xxdyP)
  > Proposes a flexible encoder-decoder backbone for tabular in-context learning, aiming to handle heterogeneous schemas and tasks.

- **Memory Efficient Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Shuting Luo, Monika Mikhail Kanaan, Cameron Gordon, Anna Leontjeva, Simon Lucey*
  [Paper](https://arxiv.org/abs/2607.27546) | [OpenReview](https://openreview.net/forum?id=1Ov4RAWuW4)
  > Studies techniques to reduce the memory footprint of tabular foundation models at inference and/or training time.

- **Attention Quantization for Tabular Foundation Models** (arXiv, Sep 2026)
  *Jonas M. Kübler, Benjamin Jäger, Klemens Flöge, Noah Hollmann, Frank Hutter*
  [Paper](https://arxiv.org/abs/2609.13031)
  > First FP8 attention study for TFMs. Quantizes queries, keys, and values (not weights or the KV cache) and aligns train/test quantization error so accuracy does not collapse; a Triton kernel reaches up to 1.7× speedup on TabPFN-v3 and TabICLv2 with no relevant TabArena/BeyondArena drop.

- **TFM-Retouche: A Lightweight Input-Space Adapter for Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Duong Nguyen, Mohammed Jawhar, Nicolas Chesneau*
  [Paper](https://arxiv.org/abs/2605.06047) | [OpenReview](https://openreview.net/forum?id=P1bvn0jvGX)
  > Adapts frozen tabular foundation models to new tasks via a lightweight input-space transformation rather than fine-tuning model weights.

- **Pocket Foundation Models: Distilling TFMs into CPU-Ready GBDTs** (ICML 2026 FMSD Workshop)
  *Aditya Tanna, Nassim Bouarour, Mohamed Bouadi, Vinay Sankarapu, Pratinav Seth*
  [OpenReview](https://openreview.net/forum?id=n1TUx8fHpv)
  > Distills tabular foundation models into compact gradient-boosted decision trees for fast, CPU-friendly deployment.

- **Bounded Context Management for Tabular Foundation Models on Stream Learning** (ICML 2026 FMSD Workshop, Spotlight)
  *Jinmo Lee, Doyun Choi, Moongi Choi, Jaemin Yoo*
  [Paper](https://arxiv.org/abs/2606.18677) | [OpenReview](https://openreview.net/forum?id=L94GfndIir) | [Code](https://github.com/morcellinus/CURE-ICML-FMSD)
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
  [Paper](https://arxiv.org/abs/2605.21742) | [OpenReview](https://openreview.net/forum?id=96HA4mxjkH)
  > Addresses degraded PFN performance under class imbalance in tabular classification.

- **SurvivalPFN: Amortizing Survival Prediction via In-Context Bayesian Inference** (NeurIPS 2026; ICML 2026 FMSD Workshop, Spotlight)
  *Shi-ang Qi, Vahid Balazadeh, Michael Cooper, Russell Greiner, Rahul G. Krishnan*
  [Paper](https://arxiv.org/abs/2605.15488) | [OpenReview](https://openreview.net/forum?id=bJ348fNOVg) | [Workshop version](https://openreview.net/forum?id=PDik7bpFhE) | [Code](https://github.com/rgklab/SurvivalPFN)
  > A PFN-style model that performs amortized in-context Bayesian inference for survival (time-to-event) prediction.

- **SurvPFN: Towards Foundation Models for Survival Predictions** (ICML 2026 FMSD Workshop)
  *Samuel Böhm, Lennart Purucker, Frank Hutter, Pascal Schlosser*
  [Paper](https://arxiv.org/abs/2606.04564) | [OpenReview](https://openreview.net/forum?id=kDEHp7ytr2)
  > Works toward prior-data fitted foundation models tailored to survival analysis tasks.

- **Staying Alive: Uncensored Survival Analysis with Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Mariana Vargas Vieyra*
  [Paper](https://arxiv.org/abs/2606.03689) | [OpenReview](https://openreview.net/forum?id=2EFykQheZD)
  > Investigates applying tabular foundation models to survival analysis, handling censoring in time-to-event data.

- **Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models** (NeurIPS 2025)
  *Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, et al.*
  [Paper](https://arxiv.org/abs/2510.21204) | [Blog](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)
  > Systematically studies which properties of synthetic priors let TFMs generalize, then trains Mitra on a curated mixture of priors (SCMs, tree-based generators, ...). Surpasses TabPFNv2 and TabICL on classification and regression and ships inside AutoGluon.

- **Mitra-v2 Technical Report** (Technical Report, 2026)
  *Yefan Tao, Xiyuan Zhang, Xinyi Liu, Boran Han, Danielle Maddix, et al. (Amazon)*
  [Paper](https://arxiv.org/abs/2609.04540)
  > Small 2D-Transformer backbone trained on a much larger and more diverse synthetic distribution than Mitra-v1, with longer contexts and wider feature spaces. Reports state-of-the-art results on the full TabArena and TALENT suites (300+ datasets), at the level of industry-scale TabFM and EXAONE Tabular.

- **LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence** (arXiv 2025)
  *Xingxuan Zhang, Gang Ren, Han Yu, Hao Yuan, et al., Peng Cui (StableAI / Tsinghua)*
  [Paper](https://arxiv.org/abs/2509.03505) | [Code](https://github.com/limix-ldm-ai/LimiX) | [Model](https://huggingface.co/stable-ai/LimiX-16M)
  > "Large structured-data models" that treat a table as a joint distribution over variables and missingness, pretrained with masked joint-distribution modeling so a single model handles classification, regression, imputation, and generation via query-based conditional prediction.

- **LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models** (ICML 2026)
  *Yuanrui Wang, Xingxuan Zhang, Han Yu, Mingchao Hao, Gang Ren, et al., Peng Cui*
  [Paper](https://arxiv.org/abs/2606.04485) | [ICML](https://icml.cc/virtual/2026/poster/63251) | [Code](https://github.com/limix-ldm-ai/LimiX)
  > Shows that affine scalar tokenization gives each feature a one-dimensional value channel that causes low effective rank. Proposes RaBEL (localized RBF tokenization) and a readout-aligned S→N→F block ordering; the resulting 2M-parameter model beats larger TabPFN-v2 and TabICL baselines at a fraction of the cost.

- **LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence** (Technical Report, Sep 2026)
  *Xingxuan Zhang, Gang Ren, Hao Yuan, Hao Zou, Hongze Tan, Hui Wang, et al., Peng Cui (StableAI / Tsinghua)*
  [Paper](https://arxiv.org/abs/2609.17488) | [Code](https://github.com/limix-ldm-ai/LimiX) | [Model](https://huggingface.co/stable-ai/LimiX-2)
  > 400M-parameter successor that switches from target-centric PFN prediction `p(y | x, D)` to Contextual Mechanism Networks learning the joint `p(x, y | D)` via Context-Conditional Masked Modeling on SCM-generated tables. One checkpoint does classification, regression, imputation, and causal-skeleton recovery. Reports #1 Elo on TabArena (1935), TALENT, and BCCO, above AutoGluon 1.6 and TabPFN-3. Non-commercial weights.

- **Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning** (arXiv 2025)
  *Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu (Lexsi Labs)*
  [Paper](https://arxiv.org/abs/2511.02818) | [Code](https://github.com/Lexsi-Labs/Orion-MSP)
  > Multi-scale feature processing, block-sparse attention that scales sub-quadratically in table width, and a Perceiver-style memory enabling bidirectional communication between components.

- **Orion-BiX: Bi-Axial Attention for Tabular In-Context Learning** (arXiv 2025)
  *Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu (Lexsi Labs)*
  [Paper](https://arxiv.org/abs/2512.00181) | [Code](https://github.com/Lexsi-Labs/Orion-BiX)
  > Alternates standard, grouped, hierarchical, and relational attention fused via multi-CLS summarization, with a label-aware ICL head that scales to large label spaces through hierarchical decision routing.

- **ConTextTab: A Semantics-Aware Tabular In-Context Learner** (NeurIPS 2025)
  *Marco Spinaci, Marek Polewczyk, Maximilian Schambach, Sam Thelin (SAP)*
  [Paper](https://arxiv.org/abs/2506.10707) | [Code](https://github.com/SAP-samples/sap-rpt-1-oss)
  > Combines a table-native ICL architecture with language-model embeddings for text cells and trains on large-scale real-world tables, bringing semantic understanding to tabular ICL while keeping long contexts.

- **TabH2O: A Unified Foundation Model for Tabular Prediction** (arXiv 2026)
  *Pascal Pfeiffer, Dmitry Gordeev, Mathias Müller, Laura Fink, et al., Sri Satish Ambati (H2O.ai)*
  [Paper](https://arxiv.org/abs/2605.18383)
  > Builds on TabICL with a dual-head model for classification and regression, single-stage pretraining stabilized by bounded scalable softmax and logit soft-capping, and noise-aware synthetic data with explicit irrelevant-feature dimensions.

- **Xiaomi-TabLDM: A Tabular Foundation Model Technical Report** (Technical Report, 2026)
  *Xiaomi-TabLDM Team*
  [Paper](https://arxiv.org/abs/2609.03880)
  > SCM-pretrained in-context learner for classification and regression that ranks first on OpenML-CTR23 regression and second on TALENT, TabArena, and BCCO regression while using substantially less compute than peers.

- **EXAONE Tabular 1.0: Technical Report** (Technical Report, 2026)
  *Moonjung Eo, Min-Kook Suh, Hye-Seung Cho, Jiwon Kim, et al. (LG AI Research)*
  [Paper](https://arxiv.org/abs/2608.25774)
  > Compact TFM family whose every layer interleaves feature-axis attention within each row with support-conditioned row-axis attention within each feature, mediated by summary tokens. Its 20.8M-parameter classifier ranks first overall on TabArena.

- **TabPFN-MT: A Natively Multitask In-Context Learner for Tabular Data** (arXiv 2026)
  *Cormac Cureton, Narges Armanfard*
  [Paper](https://arxiv.org/abs/2605.20234)
  > Trains on an expanded multi-target synthetic prior with a widened y-encoder and shared decoder so several targets are predicted jointly in one forward pass, sharing information across tasks; evaluated on 344 small-to-medium datasets.

- **TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts** (arXiv 2025)
  *Christopher Kolberg, Jules Kreuer, Jonas Huurdeman, Sofiane Ouaari, Katharina Eggensperger, Nico Pfeifer*
  [Paper](https://arxiv.org/abs/2510.06162)
  > Continued pre-training on a customized prior lets TabPFN handle >30,000 features (e.g. omics data) without feature reduction, matching the base model on standard data while improving noise robustness.

- **A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities** (arXiv 2025)
  *Han-Jia Ye, Si-Yang Liu, Wei-Lun Chao*
  [Paper](https://arxiv.org/abs/2502.17361)
  > Shows TabPFN v2 infers attribute relationships even from randomized attribute tokens, and that it can be turned into a feature extractor to handle high-dimensional, many-class, and large-scale tasks via divide-and-conquer.

- **When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach** (ICML 2026)
  *Xinpeng Lv, Yunxin Mao, Renzhe Xu, Chunyuan Zheng, et al., Haotian Wang*
  [Paper](https://arxiv.org/abs/2605.19662) | [ICML](https://icml.cc/virtual/2026/poster/62109)
  > When individuals strategically manipulate features after deployment, the non-strategic pretraining prior mismatches the post-manipulation distribution. Strategic PFN (SPN) constructs strategic in-context examples at inference time to realign the prior without retraining.

- **GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data** (ICML 2026)
  *Al Zadid Sultan Bin Habib, Md Younus Ahamed, Prashnna Kumar Gyawali, Gianfranco Doretto, Donald A. Adjeroh*
  [Paper](https://arxiv.org/abs/2606.05441) | [ICML](https://icml.cc/virtual/2026/poster/62523)
  > Graph-guided feature ordering (equivalent to weighted minimum linear arrangement) plus a subunit-compression unit that pools adjacent ordered features into meta-features, making TabPFN-style prediction practical in high-dimension, low-sample regimes.

- **SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning** (arXiv 2026)
  *Hao Wang, Siyu Zhang, Wei Ma*
  [Paper](https://arxiv.org/abs/2608.27882)
  > Asks whether attention is needed at every stage of tabular ICL: builds row/column representations with Mamba state-space mixing over stable latent slots and keeps attention only for query-conditioned retrieval from labeled context.

- **Tydra: An Efficient Hybrid Model for Tabular Data** (arXiv 2026)
  *Mieszko Komisarczyk, Saurabh Mathur, Maurice Kraus, Sriraam Natarajan, Kristian Kersting*
  [Paper](https://arxiv.org/abs/2608.21199)
  > Hybrid Transformer–SSM architecture that interleaves attention and Hydra layers, cutting inference time ~30% versus TabPFN on 30 OpenML datasets while retaining most of its accuracy.

- **TACTICL: Task-Aware Compression of Tabular ICL Models** (arXiv 2026)
  *Mykhailo Koshil, Matthias Feurer, Katharina Eggensperger*
  [Paper](https://arxiv.org/abs/2608.10837) | [Code](https://github.com/Hebog/tfm_compression)
  > Jointly prunes Transformer layers and replaces them with lightweight task-trained adapters, blending in-context with in-weight learning. Up to 85% of layers can be substituted without substantial loss on 47 datasets while keeping in-context robustness to shift.

- **GEAR: Generative Expansion and Real Anchoring for Two-Stage Distillation of Tabular Foundation Models** (arXiv 2026)
  *Qi Qin, Jiajie Zhu, Dali Chen, Yuzhao Zhang, et al., Yu Su*
  [Paper](https://arxiv.org/abs/2608.18849)
  > Distills TFMs into CPU-deployable MLPs or trees: synthetic covariates serve as teacher-query locations for soft targets, then the student is re-anchored on real labels and out-of-fold teacher predictions, with a risk certificate for the fidelity/volume trade-off.

- **Exploring Fine-Tuning for Tabular Foundation Models** (arXiv 2026)
  *Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Vinay Kumar Sankarapu*
  [Paper](https://arxiv.org/abs/2601.09654)
  > Compares zero-shot, meta-learning, supervised, and parameter-efficient fine-tuning across TALENT, OpenML-CC18, and TabZilla. Zero-shot is already strong; PEFT and meta-learning give moderate gains while full SFT often hurts accuracy or calibration.

- **Causal Data Augmentation for Robust Fine-Tuning of Tabular Foundation Models (CausalMixFT)** (arXiv 2026)
  *Magnus Bühler, Lennart Purucker, Frank Hutter*
  [Paper](https://arxiv.org/abs/2601.04110)
  > Fits SCMs on the target dataset to generate structurally consistent synthetic samples for fine-tuning under data scarcity; across 33 TabArena classification datasets and 2,300+ runs it beats CTGAN, TabEBM, and TableAugment augmentation.

- **KnowsTFM: Knowledge-Informed Fine-Tuning of Small Tabular Foundation Models** (arXiv 2026)
  *Boshko Koloski, Xiangjian Jiang, Senja Pollak, Blaž Škrlj, Mateja Jamnik, Nikola Simidjievski*
  [Paper](https://arxiv.org/abs/2606.30258)
  > Steers nanoscale TabPFN- and TabICL-style models toward niche, high-dimensional domains by injecting curated knowledge graphs and knowledge banks during fine-tuning.

- **SkillTFM: Gated Skill Evolution for Training-Free Adaptation of Tabular Foundation Models** (arXiv 2026)
  *Yi He, Zhengkang Guan, Anpeng Wu, Peng Cui, Fei Wu, Kun Kuang*
  [Paper](https://arxiv.org/abs/2608.06137)
  > Shifts TFM adaptation from parameter updates to the gated evolution of an agentic, verifiable skill bank, handling distribution shift and task-specific patterns without fine-tuning.

- **Context-Constrained Transfer Learning for Tabular Foundation Models via Data Distillation** (arXiv 2026)
  *Yijun Lin, Sai Li*
  [Paper](https://arxiv.org/abs/2607.04809)
  > TL-ANDI builds a compact source context via budget-constrained optimal transport that balances target coverage and posterior compatibility, then attaches locally distilled labels and a residual calibration step to avoid negative transfer.

- **Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment** (arXiv 2026)
  *Seunghan Lee*
  [Paper](https://arxiv.org/abs/2605.04363) | [Code](https://github.com/seunghan96/DistPFN)
  > DistPFN rescales TabPFN's class probabilities to down-weight the context's class prior, with a temperature-scaled variant that adapts to the prior/posterior gap. Training-free and architecture-agnostic.

- **Active In-Context Learning for Tabular Foundation Models** (arXiv 2026)
  *Wilailuck Treerath, Fabrizio Pittorino*
  [Paper](https://arxiv.org/abs/2603.27385)
  > Formalizes Tab-AICL, where the labeled context rather than model weights is iteratively optimized, with uncertainty, diversity, hybrid, and proxy-shortlisted acquisition rules built on TabPFN's calibrated predictions.

- **LUCoS: Latent Unsupervised Context Selection for Tabular Foundation Models** (arXiv 2026)
  *Oroel Ipas, Guillermo Gomez-Trenado, Rocío Romero-Zaliz, Isaac Triguero*
  [Paper](https://arxiv.org/abs/2605.27254)
  > Cold-start selection of which rows to label before any labels exist, performed geometrically in a TFM-induced latent space rather than the raw heterogeneous feature space.

- **ARASH: Adaptive Retrieval And Shot Selection for Tabular Prediction** (arXiv 2026)
  *Samirasadat Jamalidinan, Yue Xu, Kazem Cheshmi*
  [Paper](https://arxiv.org/abs/2608.17856)
  > Query-specific retrieval of the most relevant context rows via local neighborhood analysis, shrinking prompts and speeding up TFM inference.

- **CRUMB: Efficient Prior Fitted Network Inference via Distributionally Matched Context Batching** (NeurIPS 2026)
  *Jamie Heredge, Mattia J. Villani, Pranav Deshpande, Akshay Seshadri, Niraj Kumar*
  [Paper](https://arxiv.org/abs/2606.11473) | [OpenReview](https://openreview.net/forum?id=z3XMLStVPC)
  > Clusters test queries, greedily selects an MMD-matched training subset per cluster, and runs exact PFN inference on each reduced batch; architecture-agnostic and evaluated on TabArena with TabPFNv2, TabICL, and others.

- **Balanced Adaptive Prototype Selection for Scalable TabPFN Inference on Large-Scale Tabular Data** (arXiv 2026)
  *Mahboobe Jadid, Melika Rezaye Garkani, Ali Mousavi*
  [Paper](https://arxiv.org/abs/2608.12989)
  > Constructs compact contexts that preserve decision boundaries, density, class balance, and diversity; 512 prototypes retain strong accuracy and calibration on million-row HIGGS and SUSY on a CPU (~1,953x context compression).

- **Streaming Hierarchical Inference with Tabular Foundation Models** (arXiv 2026)
  *Vitor Crista, Afonso Lourenço, Diogo Martinho, Goreti Marreiros*
  [Paper](https://arxiv.org/abs/2609.07956)
  > HINT combines an edge-side approximate-nearest-neighbor memory over a sliding window with selective offloading of uncertain samples (plus retrieved context) to a cloud-hosted TFM, trading accuracy against communication cost.

- **Early Stopping Tabular In-Context Learning** (arXiv 2025)
  *Jaris Küken, Lennart Purucker, Frank Hutter*
  [Paper](https://arxiv.org/abs/2506.21387)
  > Dynamically decides after each encoder layer whether to stop in-context learning and decode with a pretrained layer-wise decoder, giving up to 1.3x (small) and 2.2x (larger) inference speedups with negligible accuracy loss.

- **Multi-Task Bayesian In-Context Learning** (arXiv 2026)
  *Qingyang Zhu, Eric Karl Oermann, Kyunghyun Cho*
  [Paper](https://arxiv.org/abs/2606.20538) | [Code](https://github.com/martianmartina/multi-task-bayesian-icl/)
  > Represents prior information explicitly as a prefix of related in-context datasets, enabling amortized hierarchical Bayesian prediction that adapts to new priors at test time rather than being locked to the training prior.

- **Prior-Aligned Data Cleaning for Tabular Foundation Models** (arXiv 2026)
  *Laure Berti-Equille*
  [Paper](https://arxiv.org/abs/2604.25154)
  > L2C2 frames data cleaning as prior alignment: a deep-RL policy sequences cleaning operators to minimize the distributional gap between dirty inputs and the TFM's synthetic prior, improving accuracy and calibration together.

- **TabPrep: Closing the Feature Engineering Gap in Tabular Benchmarks** (NeurIPS 2026)
  *Andrej Tschalzev, Nick Erickson, Yuyang Wang, Huzefa Rangwala, Stefan Lüdtke, Heiner Stuckenschmidt, Christian Bartelt*
  [Paper](https://arxiv.org/abs/2606.02384) | [OpenReview](https://openreview.net/forum?id=nMW7DYLGTo) | [Code](https://github.com/atschalz/tabprep)
  > Lightweight feature generators targeting three structural data patterns that most model classes are blind to; on TabArena they lift tree, neural, linear, and foundation models, often more than model-centric improvements.

- **SurvFM enables tabular foundation models for right-censored survival prediction** (arXiv 2026)
  *Yue Lyu, Steven H. Lin, Xuelin Huang, Ziyi Li*
  [Paper](https://arxiv.org/abs/2607.09577)
  > Converts censored follow-up into observation-level restricted-mean-survival-time targets so unmodified TFMs can do survival regression; leading RMST accuracy across 55 public datasets.

- **Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation** (arXiv 2026)
  *Minh-Khoi Pham, Luca Cotugno, Alina Sirbu, Tai Tan Mai, Martin Crane, Marija Bezbradica*
  [Paper](https://arxiv.org/abs/2606.12006)
  > Trains a survival-aware head on top of frozen TFM representations for censored time-to-event prediction in clinical settings.

- **Adaptation Interfaces for In-Context Tabular Foundation Models in Time-to-Event Prediction** (arXiv 2026)
  *Minh-Khoi Pham, Luca Cotugno, Dan Cernei, Alina Sirbu, et al., Marija Bezbradica*
  [Paper](https://arxiv.org/abs/2609.04901) | [Code](https://github.com/kaylode/survival-fm)
  > Compares temporal zero-shot reformulation, classification fine-tuning, and CoxPH/DeepHit survival heads on frozen TFM backbones across 74 single-risk and 4 competing-risk datasets; Cox interfaces are the most reliably strong as data scale.

- **uLEAD-TabPFN: Uncertainty-aware Dependency-based Anomaly Detection with TabPFN** (arXiv 2026)
  *Sha Lu, Jixue Liu, Stefan Peters, Thuc Duy Le, Craig Xie, Lin Liu, Jiuyong Li*
  [Paper](https://arxiv.org/abs/2604.20255)
  > Uses frozen PFNs to estimate conditional feature dependencies in a learned latent space and flags anomalies as dependency violations with uncertainty-aware scoring.

- **TabPFN-TSRA: Retrieval-Augmented TabPFN for Time-Series Forecasting** (ICML 2026 FMSD Workshop)
  *Zijian Tang, Ying Zhang, Zhenjun Liu, Sibo Cai, Shibin Yue, Meili Zhang*
  [OpenReview](https://openreview.net/forum?id=pGK4RbJTC9)
  > Augments TabPFN-style forecasting with retrieval of relevant historical windows as in-context examples.

- **HICL-TBI: Hyperbolic In-Context Learning for Traumatic Brain Injury Outcome Prediction** (ICML 2026 FMSD Workshop)
  *Thai Khanh Nguyen, Thanh-Hai Tran*
  [OpenReview](https://openreview.net/forum?id=XWF3Sa0xIO)
  > Applies tabular in-context learning with hyperbolic representations to clinical outcome prediction for traumatic brain injury.

- **Strengthening LLMs for Tabular Prediction with Structural Priors** (NeurIPS 2026)
  *Pengxiang Cai, Zihao Gao, Wanchen Lian, Guocong Li, Jintai Chen*
  [Paper](https://arxiv.org/abs/2510.17385) | [OpenReview](https://openreview.net/forum?id=pTunYUdPGz)
  > Brings column-permutation invariance into LLM post-training via Permutation Relative Policy Optimization (PRPO). On 139 OpenML datasets the resulting 8B model is competitive with specialized tabular baselines, dominates zero-shot settings, and matches 32-shot baselines, pointing toward LLM-based large tabular models.

- **When to Trust a PFN: Detecting Harmful Shift in Tabular Foundation Models** (NeurIPS 2026)
  *Viet Nguyen, Herman Bergström, Stephan Rabanser, Rahul G. Krishnan*
  [OpenReview](https://openreview.net/forum?id=ixQvsT5Hku)
  > Detects distribution shifts that actually harm a tabular PFN's predictions, helping practitioners decide when TFM outputs can be trusted.

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

- **Tabular Foundation Model for Generative Modelling (TabFORGE)** (NeurIPS 2026)
  *Xiangjian Jiang, Mingxuan Liu, Nikola Simidjievski, Tassilo Klein, Mateja Jamnik*
  [Paper](https://arxiv.org/abs/2605.09424) | [OpenReview](https://openreview.net/forum?id=s7BHCiG4IM)
  > Builds a generative TFM on pretrained tabular foundational representations aligned with the causal structural prior of heterogeneous tables, closing the gap to strong dataset-specific generators. Full version of the FMSD paper *A Generative Foundation Model for Heterogeneous Tabular Data*.

- **Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure** (arXiv 2026)
  *Davide Tugnoli, Andrea De Lorenzo, Marco Virgolin, Giovanni Cinà*
  [Paper](https://arxiv.org/abs/2603.10254)
  > Shows that TabPFN's autoregressive column-by-column generation produces spurious correlations when feature order conflicts with causal structure, and fixes it with DAG-aware (and partially directed) conditioning.

- **DataSynK: Causal-Symbolic EHR Synthesis for Tabular Foundation Models in Low-Resource Settings** (ICML 2026 FMSD Workshop)
  *Eduarda T. C. Chagas, Roberta Viola, Juarez Monteiro, Francisco Galuppo Azevedo, Saulo F. Saturnino, Adriano Veloso*
  [OpenReview](https://openreview.net/forum?id=3mYAxwt6q7)
  > Synthesizes electronic-health-record tables from causal and symbolic knowledge to support TFMs where real clinical data are scarce.

- **TabWorld: A World-Modeling Foundation Model for Tabular Generation** (NeurIPS 2026)
  *Xiaofeng Lin, Chunhe Wang, Tung Sum Thomas Kwok, Guang Cheng*
  [OpenReview](https://openreview.net/forum?id=j8Vn2ZHt0f)
  > A foundation model for tabular data generation built around a world-modeling view of how tables are produced.

- **Breaking the Quality–Privacy Tradeoff in Tabular Data Generation via In-Context Learning (DiffICL)** (NeurIPS 2026)
  *Xinyan Han, Yan Lu, Xiaoyu Lin, Yuanyuan Jiang, Yuanrui Wang, Xuanyue Li, Wenchao Zou, Xingxuan Zhang*
  [Paper](https://arxiv.org/abs/2605.04911) | [OpenReview](https://openreview.net/forum?id=PKGG71vTok)
  > Shows dataset-specific tabular generators trade privacy for quality in the small-data regime, and proposes DiffICL, which casts generation as in-context learning over structural priors pretrained on many datasets. Improves both fidelity and privacy on 14 real-world datasets and yields useful augmentation data.

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
  [Paper](https://arxiv.org/abs/2607.05476) | [OpenReview](https://openreview.net/forum?id=wRWaegFYMx)
  > Argues that parameter-free encoders remain a competitive design choice for relational database (RDB) foundation models.

- **PluRel-to-RDB-PFN: Schema-Guided Synthetic Relational Pretraining** (ICML 2026 FMSD Workshop)
  *Mohammad Sadeq Abolhasani, Viswanath Ganapathy*
  [Paper](https://arxiv.org/abs/2607.29129) | [OpenReview](https://openreview.net/forum?id=RpNvhdvd2v)
  > A PFN for relational databases pretrained on schema-guided synthetic relational data.

- **Context Window Failures in Relational Foundation Models** (ICML 2026 FMSD Workshop)
  *Denis Oliveira Correa, Francisco Galuppo Azevedo*
  [Paper](https://arxiv.org/abs/2609.00460) | [OpenReview](https://openreview.net/forum?id=lkuOIfXLwJ)
  > Analyzes how relational foundation models degrade when relevant context exceeds their effective context window.

- **Beyond Average Leaderboards: When Explicit Graph Priors Help Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Franck Le, Keith Grueneberg, Erich M. Nahum, Vadim Sheinin*
  [OpenReview](https://openreview.net/forum?id=4gmLDG0aGC)
  > Examines the conditions under which adding explicit graph priors benefits tabular foundation models, beyond aggregate leaderboard scores.

- **Can LLMs Use Relational Transformer Embeddings?** (ICML 2026 FMSD Workshop)
  *Francisco Galuppo Azevedo, Clarissa Lima Loures*
  [OpenReview](https://openreview.net/forum?id=Z2n7WcIy6j)
  > Investigates whether large language models can effectively consume embeddings produced by relational transformers.

- **Bringing Graphs to the Table: Zero-shot Node Classification via Tabular Foundation Models** (arXiv 2025)
  *Adrian Hayler, Xingyue Huang, İsmail İlkan Ceylan, Michael Bronstein, Ben Finkelshtein*
  [Paper](https://arxiv.org/abs/2509.07143)
  > TAG reformulates node classification as a table (feature, structure, and label columns per node) so a TFM performs zero-shot node classification via in-context learning.

- **TFMLinker: Universal Link Predictor by Graph In-Context Learning with Tabular Foundation Models** (ICML 2026 GFM Workshop)
  *Tianyin Liao, Chunyu Hu, Yicheng Sui, Xingxuan Zhang, Peng Cui, Jianxin Li, Ziwei Zhang*
  [Paper](https://arxiv.org/abs/2602.08592)
  > Adapts SCM-pretrained TFMs to universal link prediction across datasets and domains without relying on textual attributes.

- **Surprisingly Simple and Effective Multi-Domain Graph Foundation Model through Graph-to-Table Alignment** (arXiv 2026)
  *Chunyu Hu, Tianyin Liao, Ge Lan, Xingxuan Zhang, Jianxin Li, Peng Cui, Ziwei Zhang*
  [Paper](https://arxiv.org/abs/2607.11374)
  > GTAlign learns a graph-to-table alignment that injects graph structure into a TFM, yielding a multi-domain graph foundation model without GNN pretraining or LLM text dependence.

- **LoGIC: Budgeted Context Construction for Node-Level Graph In-Context Learning with Tabular Foundation Models** (arXiv 2026)
  *Mingqi Yang, Zidong Guo, Jihui Yang, Wenming Zuo*
  [Paper](https://arxiv.org/abs/2609.05955)
  > Studies which labeled and unlabeled nodes should form the prompt for G2T-FM / GraphPFN-style graph ICL, splitting a labeled-context budget from an unlabeled "halo" budget for adapter message passing.

- **Adapting Tabular Foundation Models for Graph Node-Level Tasks** (ICML 2026 GFM Workshop)
  *Dmitry Eremeev et al. (Yandex Research)*
  [Workshop](https://icml.cc/virtual/2026/workshop/54057)
  > Workshop follow-up to G2T-FM / GraphPFN on turning TFMs into node-level graph learners.

- **RelBench v2: A Large-Scale Benchmark and Repository for Relational Data** (arXiv 2026)
  *Justin Gu, Rishabh Ranjan, Charilaos Kanatsoulis, Haiming Tang, et al., Jure Leskovec*
  [Paper](https://arxiv.org/abs/2602.12606)
  > Expands RelBench to 11 databases (22M+ rows, 29 tables) and adds autocomplete tasks that require inferring missing attribute values inside relational tables, targeting relational foundation models.

- **PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models** (arXiv 2026)
  *Vignesh Kothapalli, Rishabh Ranjan, Valter Hudovernik, Vijay Prakash Dwivedi, Johannes Hoffart, Carlos Guestrin, Jure Leskovec*
  [Paper](https://arxiv.org/abs/2602.04029)
  > Synthesizes multi-table databases from scratch (schema graphs, primary–foreign-key bipartite graphs, causal feature mechanisms), unlocking scaling laws for relational foundation models where real databases are private.

- **Relational In-Context Learning via Synthetic Pre-training with Structural Prior (RDB-PFN)** (arXiv 2026)
  *Yanbo Wang, Jiaxuan You, Chuan Shi, Muhan Zhang*
  [Paper](https://arxiv.org/abs/2603.03805) | [Code](https://github.com/MuLabPKU/RDBPFN)
  > First relational foundation model trained purely on synthetic data: a Relational Prior Generator produces 2M+ single-table and multi-table tasks, and the model adapts to new databases via genuine in-context learning (19 real-world tasks).

- **RDBLearn: Simple In-Context Prediction Over Relational Databases** (arXiv 2026)
  *Yanlin Zhang, Linjie Xu, Quan Gan, David Wipf, Minjie Wang*
  [Paper](https://arxiv.org/abs/2602.18495) | [Code](https://github.com/HKUSHXLab/rdblearn)
  > Auto-featurizes each target row with relational aggregations over linked records and runs an off-the-shelf TFM on the materialized table; packaged as a scikit-learn-style toolkit with swappable ICL backends.

- **Advancing Open and Reproducible Relational Learning: RelArena-α, TabPFN-Rel and RPI** (arXiv 2026)
  *Adrian Hayler, Klemens Flöge, Alan Arazi, Rishabh Ranjan, Jure Leskovec, et al., Frank Hutter, Noah Hollmann (Prior Labs)*
  [Paper](https://arxiv.org/abs/2608.16319)
  > Prior Labs' α-release for relational learning: RelArena-α standardizes RelBench v1 evaluation in the spirit of TabArena, TabPFN-Rel is an open relational TFM baseline, and RPI is a relational preprocessing interface.

- **OpenRFM: Dissecting Relational In-Context Learning** (arXiv 2026)
  *Zhikai Chen, Junyu Yin, Jialiang Gu, Siheng Xiong, et al., Kai Guo*
  [Paper](https://arxiv.org/abs/2606.04320)
  > Analyzes the Relational Transformer: it performs relation-level ICL that fails under sparse label-cell coverage, and synthetic-only vs in-distribution pretraining drive the same architecture into lazy vs feature-learning regimes.

- **Curriculum Matters: Data-Efficient Relational PFN Pretraining with Synthetic Data** (arXiv 2026)
  *Mohammad Sadeq Abolhasani, Viswanath Ganapathy*
  [Paper](https://arxiv.org/abs/2607.29120)
  > Uses PluRel as the sole synthetic source for RDB-PFN-style pretraining and shows that a progressive single-table curriculum (widening schema complexity) is far more data-efficient than random ordering.

- **Task Scarcity and Label Leakage in Relational Transfer Learning** (arXiv 2026)
  *Francisco Galuppo Azevedo, Clarissa Lima Loures, Denis Oliveira Correa*
  [Paper](https://arxiv.org/abs/2603.29914)
  > Argues limited task diversity, not just limited data, constrains relational foundation models: task-specific shortcuts leak into representations, and a gradient-projection fix improves within-database transfer on RelBench.

- **A Fair Benchmarking of Deep Relational Database Learning Models** (arXiv 2026)
  *Kazi F. Akhter, Bharath Ajendla, Manar D. Samad*
  [Paper](https://arxiv.org/abs/2607.03659)
  > Re-implements recent deep RDB learners under one protocol across five databases; the Relational Transformer is strongest overall, and RDB-designed models even win on single-table tasks.

- **KGPFN: Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning** (arXiv 2026)
  *Yisen Gao, Jiaxin Bai, Haoyu Huang, Zhongwei Xie, et al., Yangqiu Song*
  [Paper](https://arxiv.org/abs/2605.14907) | [Code](https://github.com/HKUST-KnowComp/KGPFN)
  > A PFN-based knowledge-graph foundation model that combines transferable relation representations with inference-time in-context learning over local and global structured context.

- **Foundation Models for Sparse, Multi-Relational Risk Prediction in Global Supply Chains** (ICML 2026 FMSD Workshop)
  *Ruohong Li, George Pu, James Nordlund, Prasanth Meiyappan*
  [OpenReview](https://openreview.net/forum?id=W7NIrEh8bI)
  > Applies relational foundation models to sparse, multi-relational supply-chain risk prediction.

- **RelAgent: LLM Agents as Data Scientists for Relational Learning** (NeurIPS 2026; ICML 2026 GFM Workshop)
  *Xingyue Huang, Louis Tichelman, Jinwoo Kim, Krzysztof Olejniczak, İsmail İlkan Ceylan*
  [OpenReview](https://openreview.net/forum?id=YRrJHyYg9F) | [Workshop](https://icml.cc/virtual/2026/workshop/54057)
  > LLM agents that automate the data-science loop over relational databases.

### Time Series & Sequential Data

PFN-style and TFM-based approaches to forecasting, time-series classification, and event sequences. See also **TabPFN-TS** above.

- **TempoPFN: Synthetic Pre-training of Linear RNNs for Zero-shot Time Series Forecasting** (NeurIPS 2026)
  *Vladyslav Moroshan, Julien Siems, Arber Zela, Timur Carstensen, Frank Hutter*
  [Paper](https://arxiv.org/abs/2510.25502) | [OpenReview](https://openreview.net/forum?id=1GbECQvpNs)
  > A univariate forecasting foundation model built on linear RNNs (GatedDeltaProduct with state-weaving) and trained purely on synthetic data from SDEs, Gaussian processes, and audio synthesis. Top-tier zero-shot results on GIFT-Eval, fev-bench, and Chronos-ZS, beating all other synthetic-only models.

- **TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning** (NeurIPS 2026)
  *Etienne Le Naour, Tahar Nabil, Adrien Petralia*
  [Paper](https://arxiv.org/abs/2606.05878) | [OpenReview](https://openreview.net/forum?id=urL7kBfWB5)
  > A TabPFN-style probabilistic encoder–regressor that treats forecasting and imputation as timestamp-aligned regression, trained on synthetic data from a causal prior that naturally supports covariates. State of the art on imputation and competitive on forecasting, especially with partially observed look-back windows.

- **ForecastPFN: Synthetically-Trained Zero-Shot Forecasting** (NeurIPS 2023)
  *Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha Naidu, Colin White*
  [Paper](https://arxiv.org/abs/2311.01933)
  > The first zero-shot forecasting model trained purely on a synthetic time-series prior; a PFN that predicts new series with as few as 40 observations.

- **LaT-PFN: A Joint Embedding Predictive Architecture for In-context Time-series Forecasting** (arXiv 2024)
  *Stijn Verdenius, Andrea Zerio, Roy L. M. Wang*
  [Paper](https://arxiv.org/abs/2405.10093)
  > Performs PFN-style in-context forecasting in a JEPA latent space, using related series as context and a normalized abstract time axis for arbitrary granularity and horizon.

- **Mamba4Cast: Efficient Zero-Shot Time Series Forecasting with State Space Models** (arXiv 2024)
  *Sathya Kamesh Bhethanabhotla, Omar Swelam, Julien Siems, David Salinas, Frank Hutter*
  [Paper](https://arxiv.org/abs/2410.09385) | [Code](https://github.com/automl/Mamba4Cast)
  > Mamba-based zero-shot forecaster inspired by PFNs, trained only on synthetic data and generating whole horizons in one pass with much lower inference cost than Transformer TSFMs.

- **TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data** (AAAI 2025)
  *Ege Onur Taga, M. Emrullah Ildiz, Samet Oymak*
  [Paper](https://arxiv.org/abs/2502.16294)
  > Generates synthetic multivariate series via GP kernels and linear coregionalization and trains a patch-based architecture that exploits temporal and cross-channel dependencies.

- **Time-Aware Prior Fitted Networks for Zero-Shot Forecasting with Exogenous Variables (ApolloPFN)** (arXiv 2026)
  *Andres Potapczynski, Ravi Kiran Selvam, Tatiana Konstantinova, et al., Boris N. Oreshkin, Dmitry Efimov (Amazon)*
  [Paper](https://arxiv.org/abs/2603.15802)
  > A time-aware PFN that natively incorporates exogenous covariates (promotions, prices, weather, calendars), which most TSFMs ignore.

- **Zero-shot Multivariate Time Series Forecasting Using Tabular Prior Fitted Networks** (arXiv 2026)
  *Mayuka Jayawardhana, Nihal Sharma, Kazem Meidani, Bayan Bruss, Tom Goldstein, Doron Bergman*
  [Paper](https://arxiv.org/abs/2604.08400)
  > Recasts multivariate forecasting as tabular problems that keep inter-channel interactions, instead of splitting into independent univariate sub-problems.

- **TimEE: End-to-end Time Series Classification via In-Context Learning** (arXiv 2026)
  *Jaris Küken, Shi Bin Hoo, Martin Mráz, Frank Hutter, Lennart Purucker*
  [Paper](https://arxiv.org/abs/2607.07500) | [Code](http://github.com/automl/timee)
  > A 4.5M-parameter PFN for end-to-end time-series classification: given a labeled support set and a query series it outputs a class distribution in a single forward pass with no per-dataset training.

- **RocketPFN: Accurate Time Series Classification via In-Context Learning** (arXiv 2026)
  *Franco Martino O'Rourke, Ana Trisovic, Dimitris Bertsimas*
  [Paper](https://arxiv.org/abs/2606.21786)
  > Training-free pipeline combining Rocket random convolutional features with TabPFN v2.5; matches HC2 on 92 UCR datasets and beats MOMENT/Mantis features under the same classifier.

- **In-Context Time Series Classification with Random Convolutional Features (MASHT)** (arXiv 2026)
  *Joscha Cüppers, Jilles Vreeken*
  [Paper](https://arxiv.org/abs/2607.19234)
  > Pairs MultiRocket and Hydra features with an in-context TFM, bypassing task-specific training entirely.

- **TS2TabPFN: Time Series Classification and Extrinsic Regression through Feature Extraction and a Tabular Foundation Model** (arXiv 2026)
  *Gabriel da Costa Merlin, Diego Furtado Silva*
  [Paper](https://arxiv.org/abs/2608.04174)
  > Bridges explicit feature extraction and TabPFN 2.5 for time-series classification and extrinsic regression.

- **TSPFN: A Temporal Tabular Foundation Model for Physiological Time Series Classification** (arXiv 2026)
  *Jérémie Stym-Popper, Clément Rambour, Federica Granese, Nicolas Thome, Olivier Bernard*
  [Paper](https://arxiv.org/abs/2608.31013) | [Code](https://github.com/Jeremstym/TSPFN)
  > Redesigns TabPFN with structured temporal representations and positional embeddings, pretrained on 140k real physiological time series.

- **A Causal DAG Prior for Synthetic Time-Series Classification Datasets** (ICML 2026 FMSD Workshop)
  *Franco Martino O'Rourke, Ana Trisovic, Dimitris Bertsimas*
  [Paper](https://arxiv.org/abs/2606.21776) | [OpenReview](https://openreview.net/forum?id=A1Al2YF74W)
  > A synthetic prior that samples typed DAGs across tabular and time-series nodes to produce multivariate, multi-class TSC datasets; fine-tuning TabPFN v2.5 on it improves UCR/UEA results.

- **Synthetic Causal Priors for In-Context Time-Series Classification** (ICML 2026 FMSD Workshop)
  *Hao-Run Cai, Han-Jia Ye*
  [OpenReview](https://openreview.net/forum?id=nMRrgMF2S8)
  > Designs causal synthetic priors for PFN-style in-context time-series classification.

- **Beyond Task-Specific Classifiers: In-Context Inference for Time Series Classification Foundation Models** (ICML 2026 FMSD Workshop)
  *Juntao Fang, Shifeng Xie, Shengbin Nie, et al., Themis Palpanas, Ruichu Cai*
  [OpenReview](https://openreview.net/forum?id=HVvARHEA9M)
  > Replaces task-specific heads on time-series classification foundation models with in-context inference.

- **In-Context Learning Under Regime Change** (ICML 2026 FMSD Workshop)
  *Carson Dudley, Yutong Bi, Xiaofeng Liu, Samet Oymak*
  [Paper](https://arxiv.org/abs/2604.16988) | [OpenReview](https://openreview.net/forum?id=o3dwy5pDAJ)
  > Formalizes in-context change-point detection for non-stationary sequences and proves Transformers can solve it, with implications for ICL-based forecasting, tabular prediction, and control.

- **One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets (SRPFN)** (arXiv 2026)
  *Woosung Kang, Jiwon Jeong, Jonghyeok Shin, Jeongwhan Choi, Noseong Park*
  [Paper](https://arxiv.org/abs/2606.15752)
  > A PFN for next-item recommendation pretrained on 25.6M synthetic interaction sequences that predicts new domains in one forward pass without retraining.

- **PRAGMA: A Foundation Model for Banking Event Sequences** (ICML 2026 FMSD Workshop)
  *Maxim Ostroukhov, Ruslan Mikhailov, Vladimir Iashin, et al., Anton Repushko*
  [OpenReview](https://openreview.net/forum?id=9SLZGd9Iur)
  > Pretrained foundation model over heterogeneous banking event sequences.

- **SOHET: Sequence Of Heterogeneous Events Transformer with Self-Supervised Pre-Training** (ICML 2026 FMSD Workshop)
  *Kees Jan de Vries, Mustafa Radha, Mathijs de Jong*
  [OpenReview](https://openreview.net/forum?id=z2Jb4Swqjp)
  > Self-supervised Transformer for sequences of heterogeneous structured events.

**Selected time-series foundation model papers from the ICML 2026 FMSD Workshop** (the workshop unifies the tabular and time-series communities):

- [Foundations without Fundamentals: Zero-Shot Blind Spots in Time Series FMs](https://openreview.net/forum?id=iIRdd86Xkr) (Spotlight) - *Ghoroghchian, Zhang, Han, Labach, Stein (Layer 6)*
- [HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](https://openreview.net/forum?id=TdpqVBB8na) (Spotlight) - *Petersen et al.*
- [Latent Instructions as Context Surrogates: Enhancing Frozen Time Series Forecasters with Instance-Adaptive Prompts](https://openreview.net/forum?id=YCedi8zejy) (Spotlight) - *Xiao et al.*
- [Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting](https://openreview.net/forum?id=gsuqQs0G0K) - *Fu, Li, Papaioannou, Kim*
- [Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining](https://openreview.net/forum?id=ywumgXC5bh) - *Nagpal et al.*
- [Nonlinear RNNs as a Compute Shortcut for Time Series Foundation Models](https://openreview.net/forum?id=tFxw63oxzV) - *Zólyomi, Stap, Böck, Klambauer, Hochreiter*
- [Recall Residualisation: Decontaminating Foundation-Model Evaluation on Public Time-Series Benchmarks](https://openreview.net/forum?id=YCAl9rtW2n) - *Kotawala*
- [Language Pretraining Gives Structured Forecasters a Sequential Prior](https://openreview.net/forum?id=QpTkqxxSiw) - *Tai et al.*
- [Investigating simple target-covariate relationships for Chronos-2 and TabPFN-TS](https://arxiv.org/abs/2605.12200) (arXiv) - *Berthelier et al.* - finds TabPFN-TS captures simple covariate–target relationships better than Chronos-2 at short horizons.

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

- **CausalTab: Pretraining Across Causal Environments for Tabular Causal Discovery** (NeurIPS 2026; ICML 2026 FMSD Workshop)
  *Zi-Rong Li, Si-Yang Liu, Tian-Zuo Wang, Han-Jia Ye*
  [OpenReview](https://openreview.net/forum?id=eg7EMDLpBt) | [Workshop version](https://openreview.net/forum?id=og3UVhP7M1)
  > Pretrains across diverse causal environments to enable tabular causal discovery.

- **Causal Foundation Models Perform Better without Post-treatment Variables** (ICML 2026 FMSD Workshop)
  *Junha Ham, Deokgyu Kim, Doeun Kim, Serjin Kim, Sanghack Lee*
  [OpenReview](https://openreview.net/forum?id=ULoLF1aOo1)
  > Shows that excluding post-treatment variables improves the accuracy of causal foundation models.

- **A Causal Foundation Model for Structure and Outcome Prediction** (ICML 2026 FMSD Workshop)
  *Max Zhu, Martino Mansoldo, Ching-Hao Wang, Stefan Groha*
  [Paper](https://arxiv.org/abs/2606.26467) | [OpenReview](https://openreview.net/forum?id=GOf9c4lOCf)
  > A causal foundation model that jointly targets causal structure and outcome prediction.

- **Bayesian Tabular Few-shot Learning with Causal Information** (ICML 2026 FMSD Workshop)
  *Ole Ossen, Jake Robertson, Arik Reuter, Magnus Bühler, Lennart Purucker, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=2yvEiFhNCT)
  > Incorporates causal information into Bayesian few-shot learning for tabular tasks.

- **Frequentist Consistency of Prior-Data Fitted Networks for Causal Inference** (arXiv 2026)
  *Valentyn Melnychuk, Vahid Balazadeh, Stefan Feuerriegel, Rahul G. Krishnan*
  [Paper](https://arxiv.org/abs/2603.12037)
  > Shows PFN-based ATE estimators can suffer prior-induced confounding bias that data never overwrite, and proposes a one-step posterior correction that restores frequentist consistency.

- **Amortizing Causal Sensitivity Analysis via Prior Data-Fitted Networks** (arXiv 2026)
  *Emil Javurek, Dennis Frauen, Marie Brockschmidt, Jonas Schweisthal, Stefan Feuerriegel*
  [Paper](https://arxiv.org/abs/2605.10590)
  > In-context computation of causal-effect bounds under unobserved confounding for generalized treatment sensitivity models, using a Lagrangian construction to label synthetic training data.

- **DCD-PFN: A Decoupling-Aware Foundation Model for Causal Discovery** (arXiv 2026)
  *Zhengkang Guan, Yikang Chen, Yi He, Yunze Tong, et al., Kun Kuang*
  [Paper](https://arxiv.org/abs/2606.21212)
  > Amortizes local causal discovery: SCM-pretrained decoupling weights identify Markov boundaries, and parallel local discovery assembles global structure.

- **MapPFN: Learning Causal Perturbation Maps in Context** (arXiv 2026)
  *Marvin Sextro, Weronika Kłos, Gabriel Dernbach*
  [Paper](https://arxiv.org/abs/2601.21092)
  > A PFN pretrained on a synthetic biological prior with causal interventions that maps a sequence of single-cell perturbation experiments to post-perturbation distributions, adapting to new gene sets at inference.

- **PerturbPFN: Probing the Limits of Synthetic Priors in Drug Perturbation Modelling** (ICML 2026 FMSD Workshop)
  *Yuche Gao, José Miguel Hernández-Lobato, Siyuan Guo*
  [Paper](https://arxiv.org/abs/2607.23447) | [OpenReview](https://openreview.net/forum?id=xTxfxVoUux)
  > PFN-style amortized model that infers a latent system graph, sparse intervention targets, and strengths under a hierarchical synthetic structural prior, then propagates effects through an SCM decoder.

- **Interventional Time Series Priors for Causal Foundation Models (CausalTimePrior)** (arXiv 2026)
  *Dennis Thumm, Ying Chen*
  [Paper](https://arxiv.org/abs/2603.11090)
  > Generates synthetic temporal SCMs with paired observational and interventional series (hard, soft, time-varying interventions) so PFNs can do in-context causal effect estimation on time series.

- **Towards Continuous-time Causal Foundation Models** (ICML 2026 FMSD Workshop)
  *Dennis Thumm, Ruben Wiedemann, Ying Chen*
  [Paper](https://arxiv.org/abs/2605.28880) | [OpenReview](https://openreview.net/forum?id=iZO7RRZDCC)
  > Extends PFN-style causal foundation models to irregularly sampled, continuous-time dynamics.

- **Causal Foundation Models for Time Series based on Prior-Data fitted Networks** (ICML 2026 FMSD Workshop)
  *Dennis Thumm, Arik Reuter, Jake Robertson, Shi Bin Hoo, Adrian Weller, Frank Hutter, Ying Chen, Bernhard Schölkopf*
  [OpenReview](https://openreview.net/forum?id=CAaTQAfq7c)
  > PFN-based causal foundation models for temporal data.

- **Temporal Causal Prior-Data Fitted Networks for Panel Data with Learned Reliability Signals** (arXiv 2026)
  *Shravan Talupula, Saurabh Sharma*
  [Paper](https://arxiv.org/abs/2606.20889)
  > TCPFN adds a causal-judgment head (null-effect probability, confounding strength, identifiability, mediation) to zero-shot temporal CATE estimation, trained on a prior spanning six causal regimes.

- **When and Why LLM Causal Priors Help: Closed-Loop Prior Selection for Amortized Causal Inference** (arXiv 2026)
  *Haohao Zhou*
  [Paper](https://arxiv.org/abs/2609.06941)
  > Investigates injecting LLM-drawn causal graphs into the synthetic priors of Do-PFN / CausalPFN-style models and proposes closed-loop selection of when such priors help.

- **Prior-Data Fitted Networks for Causal Inference: a Simulation Study with Real-World Scenarios** (arXiv 2026)
  *Francisco Mourao, David Hajage, Daria Bystrova, et al., Benjamin Glemain*
  [Paper](https://arxiv.org/abs/2603.15928)
  > Evaluates TabPFN + g-computation / IPTW and CausalPFN for ATE estimation on simulated clinical scenarios; finds bootstrapped TabPFN too slow for routine use and g-computation with TabPFN biased.

- **Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN** (arXiv 2026)
  *Qinyi Liu, Mohammad Khalil, Naman Goel*
  [Paper](https://arxiv.org/abs/2601.17912)
  > TabPFN is accurate and robust to spurious correlations, but fairness gains from its SCM-based causal pretraining are moderate and inconsistent, especially under MNAR missingness.

- **Attention-Based Pretraining for Unsupervised Amortized Causal Discovery** (NeurIPS 2026)
  *Naiyu Yin, Tian Gao, Yue Yu*
  [OpenReview](https://openreview.net/forum?id=IWyVTvkGab)
  > Attention-based pretraining for amortized causal discovery in the unsupervised setting, inferring causal structure from new datasets without per-dataset optimization.

### Physical Systems & ODEs

- **Decoupled-Value Attention for Prior-Data Fitted Networks: GP Inference for Physical Equations** (NeurIPS 2025)
  *Kaustubh Sharma, et al.*
  [Paper](https://arxiv.org/abs/2509.20950)
  > Introduces Decoupled-Value Attention (DVA) to improve performance in physical systems and high-dimensional regression. Specifically targets inference for physical equations (ODEs/PDEs) using PFNs.

- **DynaPFN: Zero-Shot Dynamical System Forecasting with Tabular Prior-Fitted Networks** (NeurIPS 2026)
  *Chiara Roverato, Joseph Cotnareanu, Pablo Piantanida, Boris Oreshkin, Mark Coates*
  [OpenReview](https://openreview.net/forum?id=m581slK2rD)
  > Applies tabular prior-fitted networks to zero-shot forecasting of dynamical systems.

- **In-context learning to predict critical transitions in dynamical systems (TipPFN)** (NeurIPS 2026)
  *Yunus Sevinchan, Juan Nathaniel, Kai Ueltzhöffer, Carla Roesch, Tobias Weber, Vaios Laschos, Hang Fan, Gregor Ramien, Johannes Haux, Pierre Gentine, Benjamin Herdeanu*
  [Paper](https://arxiv.org/abs/2605.12308) | [OpenReview](https://openreview.net/forum?id=24hX13I66G)
  > TipPFN is a prior-data fitted network that infers how close a system is to a tipping point. Trained on a synthetic generator built from canonical bifurcations with randomized stochastic dynamics, it gives state-of-the-art early warning on unseen tipping regimes, sim-to-real settings, and real observations.

### Optimization

- **Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks** (NeurIPS 2023)
  *Steven Adriaensen, et al.*
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html)
  > LC-PFN is trained to extrapolate learning curves, enabling accurate posterior predictive distributions and efficient model selection / early stopping.

- **PFNs4BO: In-Context Learning for Bayesian Optimization** (ICML 2023)
  *Samuel Müller, et al.*
  [Paper](https://arxiv.org/abs/2305.17535)
  > Explores the application of PFNs as surrogate models for Bayesian Optimization.

- **Benchmarking Tabular Foundation Models as Surrogates in Expensive Evolutionary Optimization** (arXiv, Sep 2026)
  *Lu Han, Jin Wang, Yuchen Li, Haoran Gu, Shulei Liu, Ziyang Shi, Wenao Lu, Handing Wang*
  [Paper](https://arxiv.org/abs/2609.18130)
  > Puts TabPFN inside offline and online surrogate-assisted evolutionary algorithms across single-/multi-objective, constrained, combinatorial, mixed-variable, and engineering problems. Effectiveness is highly problem-dependent: TabPFN helps on some complex landscapes but does not universally replace RBF or Gaussian-process surrogates.

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

- **In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization (FT-PFN)** (ICML 2024)
  *Herilalaina Rakotoarison, Steven Adriaensen, Neeratyoy Mallik, Samir Garibov, Edward Bergman, Frank Hutter*
  [Paper](https://arxiv.org/abs/2404.16795)
  > A PFN surrogate for freeze-thaw BO that does Bayesian learning-curve extrapolation in a single forward pass, removing online surrogate retraining.

- **Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks** (NeurIPS 2025)
  *Dongwoo Lee, Dong Bok Lee, Steven Adriaensen, Juho Lee, Sung Ju Hwang, Frank Hutter, Seon Joo Kim, Hae Beom Lee*
  [Paper](https://arxiv.org/abs/2505.23032)
  > A PFN meta-trained on synthetic power-law-like functions to extrapolate neural scaling laws with calibrated uncertainty.

- **In-Context Decision Making for Optimizing Complex AutoML Pipelines (PS-PFN)** (arXiv 2025)
  *Amir Rezaei Balef, Katharina Eggensperger*
  [Paper](https://arxiv.org/abs/2508.13657) | [Code](https://github.com/amirbalef/CASHPlus)
  > Extends CASH to modern pipelines (fine-tuning, ensembling) via posterior sampling for the max k-armed bandit, with a PFN estimating the posterior of the maximal value in context.

- **α-PFN: Fast Entropy Search via In-Context Learning** (arXiv 2026)
  *Herilalaina Rakotoarison, Steven Adriaensen, Tom Viering, Carl Hvarfner, Samuel Müller, Frank Hutter, Eytan Bakshy*
  [Paper](https://arxiv.org/abs/2606.07134) | [Code](https://github.com/automl/AlphaPFN)
  > Two-stage amortization that learns entropy-search acquisition functions with PFNs, replacing slow Monte-Carlo information-gain approximations with a single forward pass.

- **In-Context Learning for Latent Space Bayesian Optimization** (arXiv 2026)
  *Tuan A. Vu, Harri Lähdesmäki, Julien Martinelli*
  [Paper](https://arxiv.org/abs/2606.09664)
  > Continues pretraining TFM surrogates on synthetic optimization tasks defined on a molecular VAE latent space to fix the mismatch between standard synthetic regression priors and latent-space BO.

- **PFN-TS: Thompson Sampling for Contextual Bandits via Prior-Data Fitted Networks** (arXiv 2026)
  *Yan Shuo Tan, Kenyon Ng, Ruizhe Deng, Sumetha Loganathan, Qiong Zhang, Bibhas Chakraborty*
  [Paper](https://arxiv.org/abs/2605.10137)
  > Converts PFN posterior predictives (TabPFN v2+, TabICL v2) into mean-reward samples via a subsampled predictive CLT, estimating posterior variance from O(log n) dataset prefixes.

- **Bootstrap-Conditioned Action Selection with Tabular Foundation Models (BC-ICL)** (ICML 2026 FMSD Workshop)
  *Devansh Gupta, Shiv Tavker, Dmitry Efimov, Suchitra Sathyanarayana, Gitanjali Bhutani, Boris N. Oreshkin (Amazon)*
  [Paper](https://arxiv.org/abs/2608.06559) | [OpenReview](https://openreview.net/forum?id=aBu8Xd6R6s)
  > Turns a frozen ICL model into a randomized bandit policy by conditioning on bootstrap resamples of the interaction history, with an arm-context architecture that shares strength across actions.

- **SymboLLM-FE: LLM-Accelerated Symbolic Regression for Automated Feature Engineering** (ICML 2026 FMSD Workshop)
  *Zi-Jian Cheng, Zi-Yi Jia, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo*
  [Paper](https://arxiv.org/abs/2608.28408) | [OpenReview](https://openreview.net/forum?id=Lb0pXRiYg8)
  > Uses LLMs to accelerate symbolic regression for interpretable automated feature engineering on tabular data.

- **Thompson Sampling using Prior-fitted Diffusion Transformers** (NeurIPS 2026)
  *Sihwa Park, Jingsen Zhu, Vinamr Jain, Sheng-Yen Chou, Alexander Terenin*
  [OpenReview](https://openreview.net/forum?id=hRNiI7uR2f)
  > Performs Thompson sampling for black-box optimization using prior-fitted diffusion transformers.

- **FoMEMO: Towards Foundation Models for Expensive Multi-objective Optimization** (NeurIPS 2026)
  *Yiming Yao, Fei Liu, Liang Zhao, Xi Lin, Yilu Liu, Qingfu Zhang*
  [Paper](https://arxiv.org/abs/2509.03244) | [OpenReview](https://openreview.net/forum?id=FBTD1aJKiR)
  > A foundation model pretrained on hundreds of millions of synthetic samples that conditions on any optimization trajectory and user preference, enabling fast in-context multi-objective optimization without rebuilding GP surrogates per problem.

- **In-Context Black-Box Optimization with Unreliable Feedback** (NeurIPS 2026)
  *Nicolas Samuel Blumer, Julien Martinelli, Samuel Kaski*
  [Paper](https://arxiv.org/abs/2605.06187) | [OpenReview](https://openreview.net/forum?id=9slt6ZbEiI)
  > Pretrains a feedback-aware transformer on a structured prior over how auxiliary signals (experts, simulators, predictors) relate to the true objective; at test time it estimates each source's reliability in context and uses it to speed up optimization.

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
  [Paper](https://arxiv.org/abs/2606.03681) | [OpenReview](https://openreview.net/forum?id=QT1ySCPeW3) | [Code](https://github.com/borawhocodess/modded-nanotabpfn)
  > Investigates how to dramatically accelerate the pretraining of tabular foundation models.

- **Optimizing Pre-Training of Tabular Foundation Models by Shaping Geometry** (ICML 2026 FMSD Workshop)
  *Humzah Merchant, Sriniketh Vangaru, Randall Balestriero*
  [OpenReview](https://openreview.net/forum?id=IYnHchzvYB)
  > Improves tabular foundation model pretraining by shaping the geometry of the learned representation space.

- **RAD-TFM: Robust and Domain-Adapted Tabular Foundation Models** (NeurIPS 2026; ICML 2026 FMSD Workshop)
  *Matthew Peroni, Franck Le, Vadim Sheinin*
  [OpenReview](https://openreview.net/forum?id=6dfLwFHIwj) | [Workshop version](https://openreview.net/forum?id=5BkHclEOW0)
  > Proposes robustness and domain-adaptation techniques for tabular foundation models.

- **Mutual Information-Guided Corruption for Improved Self-Supervised Representation Learning in Tabular Data** (ICML 2026 FMSD Workshop)
  *Michael Lawson, Emerald Sy, Kehui Zhang, Raymond H. Chan, Kannie W. Y. Chan, Rosa H. M. Chan*
  [OpenReview](https://openreview.net/forum?id=T8qbmiE0yZ)
  > Uses mutual information to guide the corruption process in self-supervised tabular representation learning.

- **Enhancing Tabular Learners with Context-Aware Semantic Embeddings** (NeurIPS 2026; ICML 2026 FMSD Workshop)
  *Günther Schindler, Maximilian Schambach, Johannes Höhne*
  [OpenReview](https://openreview.net/forum?id=pV2wVDrTqK) | [Workshop version](https://openreview.net/forum?id=QArxQg4U71)
  > Augments tabular learners with context-aware semantic embeddings of columns and values.

- **Towards Pretraining Text Encoders for TabPFN** (ICML 2026 FMSD Workshop)
  *Mustafa Tajjar, Alexander Pfefferle, Lennart Purucker, Frank Hutter*
  [Paper](https://arxiv.org/abs/2606.04876) | [OpenReview](https://openreview.net/forum?id=dA8IZj8R46)
  > Explores pretraining dedicated text encoders to handle textual columns within TabPFN.

- **HGR-TabE: Universal Tabular Embeddings via Maximal Correlation Alignment** (ICML 2026 FMSD Workshop)
  *Niharika S. D'Souza, Liane Vogel, Kavitha Srinivas, Sola Shirai, Oktie Hassanzadeh, Horst Samulowitz*
  [OpenReview](https://openreview.net/forum?id=FRj6pclhXE)
  > Learns universal tabular embeddings using maximal correlation (HGR) alignment.

- **Learned Sequence Representations over Raw Credit Events for Credit-Abuse Scoring** (ICML 2026 FMSD Workshop)
  *Tianming Zhou, Jiarui Xu, Nitesh Kumar, Alexander Statnikov*
  [OpenReview](https://openreview.net/forum?id=8Zxm19jTVM)
  > Learns representations directly from raw credit-event sequences to improve credit-abuse scoring, instead of relying on hand-engineered tabular features.

- **Latent Chain-of-Thought Improves Structured-Data Transformers** (ICML 2026 FMSD Workshop)
  *Carson Dudley, Samet Oymak*
  [Paper](https://arxiv.org/abs/2605.11262) | [OpenReview](https://openreview.net/forum?id=L0Mj9fY94c)
  > A recurrent scheme where a structured-data Transformer compresses query hidden states into feedback tokens and re-processes them for several rounds of latent computation; gains across 36 forecasting and tabular datasets versus deeper and looped baselines.

- **From Synthetic Priors to Model Behavior: Structural Coverage in Tabular Foundation Models** (arXiv 2026)
  *He Zhao, Ryan Thompson, Daniel M. Steinberg, Ashfaqur Rahman, Edwin V. Bonilla, Cheng Soon Ong*
  [Paper](https://arxiv.org/abs/2609.06912)
  > Reconstructs the synthetic generators of four TFMs and measures how broadly and densely each prior covers benchmark datasets in a space of structural descriptors, linking local prior support to downstream accuracy.

- **Understanding the Surprising Generalization Properties of Tabular Foundation Models** (NeurIPS 2026)
  *Nour Shaheen, Junwei Ma, Alex Labach, Frank Hutter, Valentin Thomas, Anthony L. Caterini*
  [Paper](https://arxiv.org/abs/2608.17957) | [OpenReview](https://openreview.net/forum?id=ikYK94iVvj)
  > Strong transfer can emerge from self-supervised pretraining on a single real table; usefulness is predicted by feature count rather than row count, motivating a task-centric view of corpus design.

- **Transformers Can Learn Posterior Predictive Distributions In-Context** (arXiv 2026)
  *Gyeonghun Kang, Changwoo J. Lee, Xiang Cheng*
  [Paper](https://arxiv.org/abs/2605.26713)
  > Constructive proof that Transformers can implement gradient descent toward GP posterior mean and variance followed by nonlinear binning, with error bounds in attention depth and bin resolution.

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
  [Paper](https://arxiv.org/abs/2606.31208) | [OpenReview](https://openreview.net/forum?id=7DZ3u0SD4b)
  > Investigates the extent to which tabular in-context learners memorize their context examples.

- **Tabular Foundation Models Are Effectively Shallow** (ICML 2026 FMSD Workshop)
  *Irene Cannistraci, Julia E. Vogt*
  [OpenReview](https://openreview.net/forum?id=kCnZUf1VYC)
  > Argues that tabular foundation models behave as effectively shallow function approximators.

- **Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function** (ICML 2026 FMSD Workshop)
  *Atharva Gupta, Dhruv Kumar, Murari Mandal, Saurabh Deshpande*
  [Paper](https://arxiv.org/abs/2606.12917) | [OpenReview](https://openreview.net/forum?id=LXSawSSeA9)
  > Uses causal localization to identify which attention heads carry out specific computations inside TabPFN.

- **Statistically Indistinguishable, Operationally Distinct: A Formal Barrier for Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Tassilo Klein, Johannes Hoffart*
  [Paper](https://arxiv.org/abs/2606.29091) | [OpenReview](https://openreview.net/forum?id=TUYc2XUdwz)
  > Establishes a formal barrier showing tabular foundation models can be statistically indistinguishable yet operationally distinct.

- **On the Uncertainty in Prior-Data Fitted Network Pretraining** (ICML 2026 FMSD Workshop)
  *Manuel Hülskamp, Julius Kobialka, Emanuel Sommer, David Rügamer*
  [OpenReview](https://openreview.net/forum?id=5Shv4Ar4N9)
  > Analyzes sources of uncertainty arising during the pretraining of prior-data fitted networks.

- **What You Pretrain On Matters: Synthetic Task Distributions Determine Tabular Foundation Model Quality** (ICML 2026 FMSD Workshop)
  *Mohamed Bouadi, Nassim Bouarour, Shivam Dubey, Varun Kulkarni, Aditya Tanna, Vinay Sankarapu*
  [Paper](https://arxiv.org/abs/2605.18971) | [OpenReview](https://openreview.net/forum?id=QfXHxB9VSS)
  > Shows that the choice of synthetic task distribution used in pretraining strongly determines tabular foundation model quality.

- **When Data Is Scarce: The Strength of the Prior in Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Florian D. Leeuwen, Sara van Erp*
  [OpenReview](https://openreview.net/forum?id=dznQA3JHfI)
  > Examines how strongly the learned prior drives tabular foundation model predictions in low-data regimes.

- **Towards Evaluating Data Priors for Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Zeynep Türkmen, Kürşat Kaya, Alexander Pfefferle, Frank Hutter*
  [Paper](https://arxiv.org/abs/2606.29241) | [OpenReview](https://openreview.net/forum?id=GUDjbVGFc1) | [Code](https://github.com/automl/TFM-Playground)
  > Works toward principled evaluation of the data priors used to train tabular foundation models.

- **Inspectable Tabular Foundation Models via In-Context Kernel Learning** (ICML 2026 FMSD Workshop)
  *Ratmir Miftachov, Bruno Charron, Simon Valentin*
  [Paper](https://arxiv.org/abs/2602.02162) | [OpenReview](https://openreview.net/forum?id=Q1P83jtxXY)
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
  [Paper](https://arxiv.org/abs/2605.28418) | [OpenReview](https://openreview.net/forum?id=FJSkVoD4k3)
  > Revisits dataset metafeatures to explain when and why models differ on tabular data.

- **Training Fair Tabular Foundation Models** (ICML 2026 FMSD Workshop, Spotlight)
  *Patrik Kenfack, Jesse C. Cresswell, Anthony L. Caterini, Samira Ebrahimi Kahou, Ulrich Aïvodji*
  [Paper](https://arxiv.org/abs/2608.14211) | [OpenReview](https://openreview.net/forum?id=ajIvCEbadL)
  > Proposes methods for training tabular foundation models that satisfy group fairness criteria.

- **FairOpt-PFN: Amortized Counterfactual Fairness with Optimal Fair Targets** (ICML 2026 FMSD Workshop)
  *Enes Hasani, Jake Robertson, Frank Hutter*
  [OpenReview](https://openreview.net/forum?id=M8o7jbKX9P)
  > A PFN approach to amortized counterfactual fairness that learns optimal fair targets.

- **Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice** (ICML 2026 FMSD Workshop)
  *Yingshuo Wang, Xian Sun, Yanhang Li, Zhichao Fan, Zexin Zhuang*
  [Paper](https://arxiv.org/abs/2605.26559) | [OpenReview](https://openreview.net/forum?id=Tda0qprAXQ)
  > Audits and corrects economic validity issues when tabular foundation models are used for discrete-choice modeling.

- **Dataset Inference for Data Provenance and Privacy Auditing in Tabular Foundation Models** (ICML 2026 FMSD Workshop, Spotlight)
  *Dariush Wahdany, Jesse C. Cresswell, Naiqing Guan, Atiyeh Ashari Ghomi, Franziska Boenisch, Adam Dziedzic*
  [OpenReview](https://openreview.net/forum?id=u2uOPq1u6I)
  > Uses dataset inference to audit data provenance and privacy in tabular foundation models.

- **TabPATE: Differentially Private Tabular In-Context Learning Without Public Data** (ICML 2026 FMSD Workshop)
  *Dariush Wahdany, Matthew Jagielski, Jesse C. Cresswell, Adam Dziedzic, Franziska Boenisch*
  [Paper](https://arxiv.org/abs/2606.31474) | [OpenReview](https://openreview.net/forum?id=Q5iGt2Z0Ql)
  > Brings differentially private in-context learning to tabular data without requiring public data.

- **Privacy Vulnerabilities of Attention Layers in Tabular Foundation Models and Protection of High-Risk Queries** (arXiv 2026)
  *Tânia Carvalho, Maxime Cordy*
  [Paper](https://arxiv.org/abs/2606.26021)
  > AMIA, a shadow-model-free membership inference attack exploiting attention concentration over in-context records, beats confidence-based attacks especially at low false-positive rates; proposes protection for high-risk queries.

- **Beyond IID: How General Are Tabular Foundation Models, Really? (BeyondArena)** (NeurIPS 2026)
  *Lennart Purucker, Andrej Tschalzev, Nick Erickson, Gioia Blayer, David Holzmüller, Alan Arazi, et al., Gaël Varoquaux, Frank Hutter*
  [Paper](https://arxiv.org/abs/2606.30410) | [OpenReview](https://openreview.net/forum?id=Mk0ohjszwO)
  > A unified holistic benchmark covering IID, temporal-shift, and other demanding task types, arguing that standard benchmarks over-represent tasks where TFMs already excel.

- **A Mechanistic Study of Tabular Foundation Models** (arXiv 2026)
  *Marin Biloš, James T. Wilson, Anderson Schneider, Yuriy Nevmyvaka*
  [Paper](https://arxiv.org/abs/2605.21288)
  > Different TFM families realize distinct similarity-based readouts (attention-weighted label votes vs class-conditional means), permutation invariances trace to specific positional parameters, and representation collapse is not a practical concern.

- **Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models** (arXiv 2026)
  *Amir Rezaei Balef, Mykhailo Koshil, Katharina Eggensperger*
  [Paper](https://arxiv.org/abs/2605.06510) | [Code](https://github.com/amirbalef/is_one_layer_enough)
  > Layerwise study of six tabular ICL models reveals depthwise redundancy and iterative refinement; a looped single-layer model with 20% of the parameters matches performance.

- **Towards Understanding Layer Contributions in Tabular In-Context Learning Models** (arXiv 2025)
  *Amir Rezaei Balef, Mykhailo Koshil, Katharina Eggensperger*
  [Paper](https://arxiv.org/abs/2511.15432)
  > "Layers as painters" analysis of TabPFN and TabICL: only subsets of layers share a representational language, suggesting structural redundancy exploitable for compression.

- **Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks** (arXiv 2026)
  *Kaustubh Sharma, Srijan Tiwari, Ojasva Nema, Parikshit Pareek*
  [Paper](https://arxiv.org/abs/2601.21731)
  > Probing, activation patching, and subspace interventions show PFNs (including TabPFN) linearly encode spectral/kernel information along a dominant axis that is causally used for prediction.

- **Entangled by Design: Spurious Intra-Variable Signal Routing in Tabular In-Context Learners** (arXiv 2026)
  *Athanasios Vlontzos, Giorgos Papanastasiou, Bernhard Kainz, Sotirios Tsaftaris*
  [Paper](https://arxiv.org/abs/2607.25532)
  > Proves that when a feature bundles a causal signal with a site-specific artefact, ridge ICL (and empirically TabPFN) unavoidably routes predictions through the spurious component regardless of context size.

- **Noise Immunity in In-Context Tabular Learning: An Empirical Robustness Analysis of TabPFN's Attention Mechanisms** (arXiv 2026)
  *James Hu, Mahdi Ghelichi*
  [Paper](https://arxiv.org/abs/2604.04868)
  > Controlled perturbations (irrelevant predictors, correlated feature groups, label noise) show TabPFN is highly robust to the data-quality issues common in industrial tables.

- **Topological Signatures of Context-Level Reliability in TabPFN** (arXiv 2026)
  *James Hu, Mahdi Ghelichi*
  [Paper](https://arxiv.org/abs/2607.17962)
  > Zigzag persistent homology over TabPFN's layer representations: H0 fragmentation correlates with dataset-level error on synthetic tasks with known topology (tori, knots, Swiss rolls).

- **Understanding Context Sampling in TabPFN on Small Tabular Datasets** (arXiv 2026)
  *Mohammed Abdullah*
  [Paper](https://arxiv.org/abs/2607.26628)
  > Larger contexts are both more accurate and more stable across random draws, and expensive selection (K-Means, farthest point) offers little over uniform sampling on 15 OpenML datasets.

- **Empirical Evaluation of Out-Of-Distribution Performance of Tabular Foundation Models** (arXiv 2026)
  *Malena Loza, David Chushig-Muzo, Eva Milara, Luis Bote-Curiel, Luis Estrada-Petrocelli, Felipe Grijalva*
  [Paper](https://arxiv.org/abs/2607.26000)
  > Nine TFMs (TabPFN v2/2.5/2.6/3, TabICL v1/v2, Mitra, LimiX, TabFM) evaluated on TableShift label, socioeconomic, and geographic shifts; all degrade OOD.

- **High Performance, Low Reliability: Uncertainty Benchmarking for Tabular Foundation Models** (arXiv 2026)
  *José Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan*
  [Paper](https://arxiv.org/abs/2605.28554) | [Code](https://github.com/jose-melo/high-performance-low-reliability)
  > On the 112 TALENT datasets, TFMs win on AUC but show worse conditional coverage under conformal prediction than GBDTs, exposing a performance–uncertainty trade-off.

- **On the Uncertainty Quantification Ability of Tabular Foundation Models** (arXiv 2026)
  *Tyler R. Johnson, Kian Ben-Jacob, Nima Negarandeh, Oriol Vendrell-Gallart, Ramin Bostanabad*
  [Paper](https://arxiv.org/abs/2606.01427) | [Code](https://github.com/kianswarehouse/GPvsPFN)
  > Systematic TabPFN v2.5 vs Gaussian process comparison for regression UQ across complexity, size, and dimensionality, highlighting the trade-off between explicit and learned priors.

- **Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors** (arXiv 2025)
  *Thomas Nagler, David Rügamer*
  [Paper](https://arxiv.org/abs/2505.11325)
  > A tuning-free martingale-posterior sampling procedure, with a convergence proof, that gives PFNs Bayesian uncertainty for predictive means, quantiles, and similar estimands.

- **Do Tabular Foundation Models Agree with Themselves?** (arXiv 2026)
  *Christian Klötergens, Vijaya Krishna Yalavarthi, Lars Schmidt-Thieme, Tom Hanika*
  [Paper](https://arxiv.org/abs/2608.06004)
  > Tests whether autoregressively composed TFM conditionals could come from any joint distribution via marginalization- and permutation-consistency requirements.

- **Interpretable Machine Learning for TabPFN** (arXiv 2024)
  *David Rundel, Julius Kobialka, Constantin von Crailsheim, Matthias Feurer, Thomas Nagler, David Rügamer*
  [Paper](https://arxiv.org/abs/2403.10923) | [Code](https://github.com/david-rundel/tabpfn_iml)
  > Adapts popular interpretability methods to TabPFN, exploiting in-context learning for efficient leave-one-covariate-out, Shapley, and feature-effect computations.

- **Why Large Language Models Fail at Tabular Prediction** (arXiv 2026)
  *Marta Garnelo, Wojciech M. Czarnecki*
  [Paper](https://arxiv.org/abs/2608.02412)
  > Evaluates five hypotheses (noise, CSV linearization, numeric tokenization, batch size, dimensionality) for why a frontier LLM fails at pure in-prompt tabular prediction, the founding premise of the TFM field.

- **Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models (NPE-PFN)** (NeurIPS 2025)
  *Julius Vetter, Manuel Gloeckler, Daniel Gedon, Jakob H. Macke*
  [Paper](https://arxiv.org/abs/2504.17660)
  > Repurposes TabPFN as a pre-trained autoregressive conditional density estimator for simulation-based inference, matching SBI baselines with far fewer simulations.

- **Pre-trained Tabular Foundation Models as Versatile Summary Networks for Neural Posterior Estimation** (arXiv 2026)
  *Elliot Pickens, Chiraag Gohel, Sidharth Satya*
  [Paper](https://arxiv.org/abs/2605.07765)
  > PFN-NPE uses a frozen TabPFN encoder as a training-free summary network for simulator outputs, paired with a normalizing-flow inference head.

- **Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit** (arXiv 2026)
  *Wassim Tenachi, Yashar Hezaveh, Laurence Perreault Levasseur, Pierre-Luc Bacon*
  [Paper](https://arxiv.org/abs/2609.02766)
  > On 316 physical equations TabPFN-3, TabICLv2, TabDPT, and Real-TabPFN-2.5 dominate baselines, yet their priors cannot represent noiseless mechanisms or physical units.

- **On the Rotation-Equivariance Geometry of Tabular Foundation Models** (ICML 2026 CTB Workshop)
  *Mert Ogul*
  [Workshop](https://sites.google.com/view/icml-ctb/technical-program/accepted-papers)
  > Studies the geometric (rotation-equivariance) properties of TFMs, presented at the ICML 2026 workshop on Combining Theory and Benchmarks.

### Applications & Domain Studies

TFMs applied outside classic tabular benchmarks, and domain-specific evaluations that report where they win or fail.

**Beyond tabular modalities**

- **Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images** (ICML 2026 FMSD Workshop, Spotlight)
  *Jan Philip Walter, Shashank Agnihotri, Margret Keuper*
  [Paper](https://arxiv.org/abs/2606.00872) | [OpenReview](https://openreview.net/forum?id=6uEcEWSEfs) | [Code](https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images)
  > Encodes each image with a frozen DINOv3 backbone, PCA-reduces to 500 dimensions, and lets TabPFN classify real vs fake in context, so adapting to a new generator only requires a new labeled context set.

- **Tabular foundation models for non-tabular tasks** (arXiv 2026)
  *Goran Nakerst, John Brennan, Wouter Beugeling, Masudul Haque*
  [Paper](https://arxiv.org/abs/2608.22594)
  > TabPFN v3 on MNIST, language identification, and Tiny ImageNet as missing-label prediction over rows, studied as a function of context size.

- **TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings** (arXiv 2026)
  *Jingxiang Zhang, Lujia Zhong, Zijie Zhu, Shuo Huang, Yuang Xu*
  [Paper](https://arxiv.org/abs/2607.11007) | [Code](https://github.com/Jingxiang-Zhang/tabpfn-multimodal-embeddings)
  > Across 22,820 episodes (14 datasets, 11 encoders, 3 modalities) TabPFN is the best-ranked zero-gradient head on frozen image/text/audio embeddings for both NLL and calibration error.

- **When Tabular Foundation Models Transfer Across Modalities: A Systematic Evaluation Across 95 Datasets, 7 Modalities, and Two Regimes** (arXiv 2026)
  *Julien Lafrance*
  [Paper](https://arxiv.org/abs/2606.02106)
  > A single ETF-preprocessing + TFM pipeline applied identically to vision, audio, speech, text, molecular, time-series, and tabular features, judged against the strongest lightweight tuned baseline.

- **Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?** (arXiv 2026)
  *Davy Guan, Lu Zhang, Asiri Wijesinghe, et al., Cheng Soon Ong, Daniel M. Steinberg*
  [Paper](https://arxiv.org/abs/2606.31126)
  > Despite a causal-graph prior unrelated to proteins or molecules, TabPFN and TabICL transfer surprisingly well to few-shot biomolecular property prediction on pretrained encoder features.

- **When Simpler ICL Outperforms Pretrained Tabular Foundation Models for RNA Editing** (ICML 2026 FMSD Workshop)
  *Ran Eisenberg, Efraim Rahamim, Erez Levanon, Ofir Lindenbaum*
  [OpenReview](https://openreview.net/forum?id=enXSrwGxRn)
  > A counter-example where simple in-context baselines beat pretrained TFMs on RNA-editing prediction.

- **Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference** (NeurIPS 2026)
  *Blazej Banaszewski, Andrew W. Fitzgibbon*
  [Paper](https://arxiv.org/abs/2608.18982) | [OpenReview](https://openreview.net/forum?id=ChiaszwjMu)
  > A molecular foundation model pretrained on 81M molecules that pairs learned representations with a PFN-style in-context head for data-limited bioassay prediction.

**Science & engineering**

- **Tabular foundation models for robust calibration of near-infrared chemical sensing data** (arXiv 2026)
  *Robin Reiter, Denis Cornet, Fabien Michel, Lauriane Rouan, Gregory Beurier*
  [Paper](https://arxiv.org/abs/2605.21544)
  > Benchmarks TabPFN on 66 NIR datasets (54 regression, 12 classification) against PLS, Ridge, CatBoost, and 1D CNNs, with and without preprocessing optimization.

- **RamanPFN: learning from Raman spectral structure with a tabular foundation model** (arXiv 2026)
  *Xingyu Pan, Huan Wang, Jinjia Guo, Zhenlin Zhao, Siming Dong, Jixi Lu*
  [Paper](https://arxiv.org/abs/2608.02157)
  > Spectral representation framework that lets TabPFN see related Raman bands jointly instead of feature-subsampled views of thousands of wavenumbers.

- **From field-scale to large-scale spectral libraries: Tabular foundation models in soil spectroscopy** (arXiv 2026)
  *Viacheslav Barkov, Jonas Schmidinger, Robin Gebbers, Martin Atzmueller*
  [Paper](https://arxiv.org/abs/2608.00608)
  > Compares TabPFN with CNN, Cubist, Random Forest, and PLSR across 85 vis-NIR/MIR soil-property regression tasks, including dimensionality-reduction strategies.

- **Tabular foundation models for the estimation of probabilistic quasar photometric redshifts in S-PLUS** (arXiv 2026)
  *Raquel R. Valença, Lilianne Nakazono, Rafael Izbicki, et al.*
  [Paper](https://arxiv.org/abs/2608.10280)
  > TabPFN 2.5, RealTabPFN 2.5, and TabICL vs eight task-specific density estimators for multi-modal quasar redshift posteriors under covariate shift; TabPFN 2.5 is best or tied across training-set sizes.

- **Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models** (arXiv 2026)
  *Raffael Theiler, Lev Telyatnikov, Leandro Von Krannichfeldt, Olga Fink*
  [Paper](https://arxiv.org/abs/2606.05481)
  > Converts fragmented, partially observed condition-monitoring signals into tabular rows so TFMs can diagnose system state and estimate remaining useful life in context.

- **Revisiting data-driven dynamic security assessment with a tabular foundation model** (arXiv 2026)
  *Olayiwola Arowolo, Maosheng Yang, Jochen Cremer*
  [Paper](https://arxiv.org/abs/2607.16031)
  > A single TFM assesses power-system stability across many contingencies in context, replacing one trained model per contingency.

- **Data-efficient flood depth prediction through domain-aware coreset selection and tabular foundation models** (arXiv 2026)
  *Lipai Huang, Adithi Srinath, Manas Singh, Junwei Ma, Ali Mostafavi*
  [Paper](https://arxiv.org/abs/2606.05265)
  > Storm- and watershed-stratified coresets condition a TFM at inference; 0.7% of the training pool reaches 98.5% of the supervised surrogate's R² and transfers to unseen watersheds.

- **Forecasting Food Inflation in Real Time with Tabular Foundation Models** (ICML 2026 FMSD Workshop)
  *Mason Linsky*
  [OpenReview](https://openreview.net/forum?id=kiupx7dsOf)
  > Real-time nowcasting of food inflation with TFMs.

**Healthcare**

- **Retrieval-aligned Tabular Foundation Models Enable Robust Clinical Risk Prediction in Electronic Health Records Under Real-world Constraints** (arXiv 2026)
  *Minh-Khoi Pham, Thang-Long Nguyen Ho, et al., Martin Crane, Marija Bezbradica*
  [Paper](https://arxiv.org/abs/2604.01841)
  > Multi-cohort EHR benchmark of classical, deep tabular, and tabular ICL models; PFN-based ICL is sample-efficient but degrades under naive retrieval, motivating AWARE, a task-aligned retrieval framework.

- **Target leakage, not model class, explains reported accuracy in survey-based cardiovascular screening** (arXiv, Sep 2026)
  *Raad Bin Tareaf, Murad Al-Rajab, Samia Loucif, Samer Ellaham, Cedric Schmitz*
  [Paper](https://arxiv.org/abs/2609.11838)
  > Leakage-tiered audit of 10 classifiers, including TabPFN and TabICL, on 442k BRFSS respondents. Removing two post-diagnostic features costs every model ~0.05 AUROC and collapses the field into a 0.0045-wide band; an explainable boosting machine is non-inferior and ~104× faster. Argues evaluation practice, not foundation-model capacity, is the binding constraint.

- **Multitask Multimodal Fusion with Tabular Foundation Models for Peak and Durability Prediction of Pertussis Booster Response** (ICML 2026 FMSD Workshop)
  *Divya Sitani*
  [Paper](https://arxiv.org/abs/2605.12852) | [OpenReview](https://openreview.net/forum?id=iIsQ9SiAxF)
  > Multi-task contrastive fusion of frozen TabPFN-v2 per-modality encoders to jointly predict vaccine response peak and durability from small, heterogeneous immunology data.

- **Are Tabular Foundation Models Robust to Realistic Query Distribution Shifts in Microbiome Data?** (arXiv 2026)
  *Giulia Perciballi, Ahmad Fall, Federica Granese, Edi Prifti, Jean-Daniel Zucker*
  [Paper](https://arxiv.org/abs/2606.24995) | [Code](https://github.com/UMMISCO/metagenomics-fm/)
  > Benchmark of biologically inspired support–query perturbations across six gut-microbiome datasets; protecting discriminative taxa is not enough to guarantee stability.

- **EveryQuery: Zero-Shot Clinical Prediction via Task-Conditioned Pretraining over Electronic Health Records** (ICML 2026 FMSD Workshop)
  *Payal Chandak, Gregory Kondas, Liat Antwarg Friedman, Isaac Kohane, Matthew McDermott*
  [Paper](https://arxiv.org/abs/2603.07900) | [OpenReview](https://openreview.net/forum?id=2Dammq2Yt0)
  > An EHR foundation model that answers structured clinical queries directly instead of sampling synthetic patient futures, making zero-shot prediction promptable and cheap.

**Economics, finance & marketing**

- **Is TabPFN the Silver Bullet for Insurance Pricing?** (arXiv 2026)
  *Bruno Deprez, Wouter Verbeke, Tim Verdonck*
  [Paper](https://arxiv.org/abs/2605.22892)
  > On two motor third-party-liability datasets TabPFN does not consistently beat GLMs or XGBoost, is slower, and is sensitive to data size, a cautionary result for actuarial use.

- **V4FinBench: Benchmarking Tabular Foundation Models, LLMs, and Standard Methods on Corporate Bankruptcy Prediction** (arXiv 2026)
  *Marcin Kostrzewa, Sebastian Tomczak, Roman Furman, et al., Maciej Zięba*
  [Paper](https://arxiv.org/abs/2605.10896)
  > One-million-record benchmark from the Visegrád economies with 131 features and six horizons, designed for evaluating TFMs and LLMs under severe class imbalance.

- **Tabular Foundation Models and the Unity of Economic Behaviour** (arXiv 2026)
  *Victor H. Aguiar*
  [Paper](https://arxiv.org/abs/2608.06842)
  > A frozen TFM recovers a decision maker's hidden choices in one domain from their choices in others, and a single random-utility model over its representation retains most of the gain.

- **Tabular Foundation Models for Discrete Choice Estimation** (arXiv 2026)
  *Liu Liu, Dan Zhang*
  [Paper](https://arxiv.org/abs/2607.13314)
  > Row-independent TFMs fit discrete choice poorly; encoding choice-set dependence and individual heterogeneity into rows closes the gap on a yogurt scanner panel.

- **Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees** (arXiv 2026)
  *Yingshuo Wang, Xian Sun, Yanhang Li, Zhichao Fan, Zexin Zhuang*
  [Paper](https://arxiv.org/abs/2606.26432)
  > A two-stage adapter embeds TFM choice probabilities inside a multinomial logit with sign-constrained coefficients, provably preserving economic logic (price monotonicity, availability). Companion to the FMSD paper on economic validity.

- **Tabular Foundation Models for Multi-View Information Cascade Popularity Prediction (TFM4POP)** (arXiv 2026)
  *Wenting Zhu, Chenghua Gong, Sanchuan Guo, Chaozhuo Li, Yueyue Zhang, Xi Zhang*
  [Paper](https://arxiv.org/abs/2608.25048)
  > First use of TFM priors to unify cascade, text, image, and tabular views for social-media popularity prediction.

- **Where Tabular Foundation Models Falter on Genetic Data: Datasets That Expose and Provide a Path to Address the Gap** (NeurIPS 2026 Evaluations & Datasets)
  *Anirban Das, Yan Cui*
  [OpenReview](https://openreview.net/forum?id=3qspmxVnC1)
  > Releases genetic datasets on which current TFMs underperform and outlines a path to close the gap.

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
- **[BeyondArena](https://arxiv.org/abs/2606.30410)** (NeurIPS 2026) - Unified holistic benchmark for TFMs beyond IID prediction (temporal shift and other demanding task types), from the TabArena / TabICL / TabPFN teams.
- **[TabPrep](https://arxiv.org/abs/2606.02384)** (NeurIPS 2026, [code](https://github.com/atschalz/tabprep)) - Adds systematic feature engineering to TabArena-style evaluation and shows it changes rankings for tree, neural, and foundation models.
- **[ScoringBench](https://arxiv.org/abs/2603.29928)** - 97 regression datasets evaluated with proper scoring rules (CRPS, interval score, energy score, ...) to exploit the full predictive distributions that TFMs produce; git-based leaderboard.
- **[TabBench-Bio](https://arxiv.org/abs/2609.07441)** - Living benchmark of 43 high-dimensional biomedical tables (thousands of features, tens–hundreds of samples), where RealTabPFN v2.5 leads at the 10k-feature / 100-sample reference cell.
- **[RelBench v2](https://arxiv.org/abs/2602.12606)** and **[RelArena-α](https://arxiv.org/abs/2608.16319)** - Benchmarks and standardized evaluation for relational foundation models.
- **[TabBioMed](https://openreview.net/forum?id=4WpEeaOeGW)** (NeurIPS 2026) - A large-scale benchmark for biomedical tabular learning.
- **[TRL-Bench](https://arxiv.org/abs/2606.09323)** (NeurIPS 2026, [OpenReview](https://openreview.net/forum?id=TkwRxfihFM)) - Standardizes representation-level evaluation of 20 tabular encoders across paradigms with row, column, and table embedding probes; finds encoder quality is capability-specific rather than captured by a single leaderboard.
- **[LLMTabBench](https://arxiv.org/abs/2605.24417)** - Evaluates LLMs on zero- to few-shot binary tabular classification, complementing TFM benchmarks.
- **[TALENT](https://github.com/LAMDA-Tabular/TALENT)** - Large tabular deep-learning toolbox and benchmark (300+ datasets) used by Mitra-v2, TabH2O, EXAONE Tabular, and others.
- **TabZilla** - Comprehensive tabular data benchmark ([paper](https://arxiv.org/abs/2305.02997))
- **OpenML** - Large collection of tabular datasets
- **Kaggle Competitions** - Real-world tabular challenges

#### From the ICML 2026 FMSD Workshop

- **[MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616)** (NeurIPS 2026 Spotlight, [OpenReview](https://openreview.net/forum?id=yOd4AxlEbb); workshop [OpenReview](https://openreview.net/forum?id=r19rlhngOD)) - A benchmark for multimodal tabular learning that combines text and image features. *Alan Arazi, Eilam Shapira, Shoham Grunblat, Mor Ventura, Elad Hoffer, Gioia Blayer, David Holzmüller, Lennart Purucker, Gaël Varoquaux, Frank Hutter, Roi Reichart.*
- **[Are Tabular Foundation Model Rankings Reliable? A Generalizability Theory Analysis of RelBench and DBInfer](https://openreview.net/forum?id=7jbzkGYag6)** - Uses generalizability theory to assess the reliability of TFM rankings on RelBench and DBInfer. *Dinesh Katupputhur Ramprasath, Tom Palczewski, Joe Meyer, Roshan Reddy Upendra, Minghua Li.*
- **[Realistic Evaluation of TabPFN v2.5 in Open Environments](https://openreview.net/forum?id=qway3qFkUL)** - Evaluates TabPFN v2.5 under realistic open-environment conditions. *Zi-Jian Cheng, Ziyi Jia, Lan-Zhe Guo.*
- **[Ensembling Tabular Foundation Models: A Diversity Ceiling and a Calibration Trap](https://openreview.net/forum?id=FZaZoe67ne)** - Analyzes the limits of ensembling TFMs, highlighting a diversity ceiling and a calibration trap. *Aditya Tanna, Yash Jignesh Desai, Pratinav Seth, Mohamed Bouadi, Nassim Bouarour, Vinay Sankarapu.*
- **[Exploring Differences Between Tabular Enterprise Data and Public Benchmarks](https://arxiv.org/abs/2606.30452)** ([OpenReview](https://openreview.net/forum?id=PXSBtjo3Gd)) - Contrasts the characteristics of enterprise tabular data with public benchmarks. *Myung Jun Kim, Maximilian Schambach, Frank Essenberger, Andre Sres, Johannes Höhne.*
- **[Benchmarking Attention for Tabular Foundation Models](https://openreview.net/forum?id=rwtcugrpDq)** (NeurIPS 2026, [OpenReview](https://openreview.net/forum?id=1MbSbrblLi)) - Benchmarks attention mechanisms used in tabular foundation models. *Maximilian Schambach, Clemens Biehl, Sam Thelin.*
- **[Beyond Accuracy: Toward Trustworthy Tabular Foundation Models in Industrial Applications](https://openreview.net/forum?id=r3RAi8Kqzl)** - Looks beyond accuracy toward trustworthiness of TFMs in industrial settings. *Johannes Keler, Matthias Woehrle, Jan Achterhold, Mark Schillinger, Maria Lyssenko, Luiz Ricardo Douat.*
- **[Benchmarking Tabular Foundation Models for Churn Prediction](https://openreview.net/forum?id=LtXucHLtiN)** - Benchmarks tabular foundation models on customer churn prediction. *Sobhan Seyedzadeh, Mostafa Karimi.*
- **[Are Tabular Foundation Model Rankings Reliable?](https://openreview.net/forum?id=7jbzkGYag6)** has a companion at the ICML 2026 Graph Foundation Models workshop: *Beyond Accuracy on RelBench: Item Response Theory Analysis of Relational Deep Learning Benchmarks* ([workshop](https://icml.cc/virtual/2026/workshop/54057)). *Ramprasath et al.*

## Books & Learning Resources

- **[Tabular Foundation Models](https://tabularfoundationmodels.com/)** (Christoph Molnar): Free online book on how TabPFN and TabICL predict without training, covering PFNs, in-context learning, pretraining, classification, regression, quantile regression, and time series forecasting, with hands-on Python and a look at whether to bet on TFMs.
- **[Table as Prompt: An Interactive Guide to Tabular Foundation Models](https://github.com/Affirm/tabular-foundation-models-tutorial)** (Affirm; NeurIPS 2026 Education Track): Interactive [website](https://affirm.github.io/tabular-foundation-models-tutorial/) walking through tabular in-context learning with TabICLv2 as the worked example, plus a [Colab primer notebook](https://colab.research.google.com/github/Affirm/tabular-foundation-models-tutorial/blob/main/materials/notebooks/01_tabicl_primer.ipynb), a model landscape, and curated learning and conference resources.

## Tutorials & Talks

- **[Getting Started with PFNs](https://github.com/automl/PFNs/blob/main/Tutorial_1_Basics.ipynb)**: Official tutorial notebook from the PFNs repository.
- **[TabPFN-3.5 model report](https://priorlabs.ai/technical-reports/tabpfn-3-5)** (Sep 2026): Prior Labs' flagship write-up. Live deep dive with Frank Hutter on Sep 28; [hackathon](https://x.com/prior_labs/status/2100151981444854150) through Oct 6 (DGX Spark / Jetson AGX Orin / RTX 4090).
- **[FMSD 2025 talks on SlidesLive](https://slideslive.com/icml-2025/1st-workshop-on-foundation-models-for-structured-data-fmsd)**: Recorded invited talks and spotlights from the 1st Workshop on Foundation Models for Structured Data (ICML 2025).
- **Invited talks at FMSD @ ICML 2026** ([workshop page](https://icml.cc/virtual/2026/workshop/54066)):
  - *Katharina Eggensperger* - "Scaling and Understanding Models for (Scientific) Tabular Data"
  - *David Holzmüller* - "TabICLv2: Advancing Open Tabular Foundation Models"
  - *Abhimanyu Das* - "Multimodal Time-Series Foundation Models"
- **Industry spotlights at FMSD @ ICML 2026**: *Noah Hollmann* (Prior Labs) - "Moving from Tensors to Systems in Tabular Foundation Models"; *Alex Labach* (Layer 6 / TD) - "TabDPT: Public Research and Enterprise Impact at TD"; *Sam Thelin* (SAP) - "The Idiosyncrasies of Enterprise Data"; *Xingxuan Zhang* (StableAI) - "Unleashing Structured-Data Modeling Capability for Generalist Intelligence" (LimiX); *Yury Gorishniy* (Yandex) - "Tabular Deep Learning: an Industry Researcher's Perspective"; *Kevin Scaman* (Fundamental) - "What LLMs Learn (and Don't) from Tables"; *Oleksandr Shchur* (AWS) - "Chronos-2: From Univariate to Universal Forecasting"; plus Nixtla and TimeCopilot on forecasting.
- **[Noah Hollmann - "Structured Foundation Models: From Tables to Graphs"](https://icml.cc/virtual/2026/workshop/54057)**: Invited talk at the ICML 2026 Graph Foundation Models workshop.

## Workshops & Venues

- **[Foundation Models for Structured Data (FMSD) @ ICML 2026](https://icml-structured-fm-workshop.github.io/)** - 2nd edition, July 11, 2026, Seoul. Unifies the tabular and time-series communities around data curation, scaling, evaluation/contamination, and deployment. 151 accepted papers ([accepted papers](https://icml-structured-fm-workshop.github.io/accepted-papers/), [OpenReview](https://openreview.net/group?id=ICML.cc/2026/Workshop/FMSD), [ICML page](https://icml.cc/virtual/2026/workshop/54066)). Organizers: Nick Erickson, Xiyuan Zhang, Mononito Goswami, Lennart Purucker, Boran Han, Maximilian Schambach, Arjun Ashok, Rajat Sen. Tabular papers from this workshop are listed throughout the sections above.
- **[FMSD @ ICML 2025](https://icml.cc/virtual/2025/workshop/39976)** - 1st edition (99 submissions, 500+ participants); [recordings](https://slideslive.com/icml-2025/1st-workshop-on-foundation-models-for-structured-data-fmsd).
- **[Structured Data for Health (SD4H) @ ICML 2026](https://structureddata4health.github.io/)** - Sibling workshop on tabular EHRs, biosignals, and disease networks, including foundation-model pretraining and scaling for health ([OpenReview](https://openreview.net/group?id=ICML.cc/2026/Workshop/SD4H)).
- **[Graph Foundation Models: A New Era for Graph Machine Learning @ ICML 2026](https://icml.cc/virtual/2026/workshop/54057)** - Explicitly solicits "LLMs/TFMs + Graphs" work; hosts TFMLinker, *Adapting Tabular Foundation Models for Graph Node-Level Tasks*, RelAgent, and *Large-Scale Pretraining unlocks Few-Shot Prediction for Relational Data*.
- **[Combining Theory and Benchmarks (CTB) @ ICML 2026](https://sites.google.com/view/icml-ctb/technical-program/accepted-papers)** - Foundation-model evaluation workshop featuring *On the Rotation-Equivariance Geometry of Tabular Foundation Models* and *Context Saturation in Zero-Shot Time-Series Foundation Models*.
- **NeurIPS 2026** (decisions Sep 24, 2026; [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Conference)) - Accepted TFM / large tabular model papers, all listed in the sections above:
  - *Tabular prediction & TFMs*: [FlexTab](https://openreview.net/forum?id=LnVFK5y1uh), [TabClustPFN](https://openreview.net/forum?id=kgZzOmPAq0), [SurvivalPFN](https://openreview.net/forum?id=bJ348fNOVg), [CRUMB](https://openreview.net/forum?id=z3XMLStVPC), [RAD-TFM](https://openreview.net/forum?id=6dfLwFHIwj), [Context-Aware Semantic Embeddings](https://openreview.net/forum?id=pV2wVDrTqK), [When to Trust a PFN](https://openreview.net/forum?id=ixQvsT5Hku), [PRPO: LLMs for Tabular Prediction with Structural Priors](https://openreview.net/forum?id=pTunYUdPGz)
  - *Generation*: [TabFORGE](https://openreview.net/forum?id=s7BHCiG4IM), [TabWorld](https://openreview.net/forum?id=j8Vn2ZHt0f), [DiffICL](https://openreview.net/forum?id=PKGG71vTok)
  - *Relational, causal, time series, dynamics*: [RelAgent](https://openreview.net/forum?id=YRrJHyYg9F), [CausalTab](https://openreview.net/forum?id=eg7EMDLpBt), [Amortized Causal Discovery](https://openreview.net/forum?id=IWyVTvkGab), [TempoPFN](https://openreview.net/forum?id=1GbECQvpNs), [TS-ICL](https://openreview.net/forum?id=urL7kBfWB5), [DynaPFN](https://openreview.net/forum?id=m581slK2rD), [TipPFN](https://openreview.net/forum?id=24hX13I66G)
  - *Optimization*: [Prior-fitted Diffusion Thompson Sampling](https://openreview.net/forum?id=hRNiI7uR2f), [FoMEMO](https://openreview.net/forum?id=FBTD1aJKiR), [In-Context BBO with Unreliable Feedback](https://openreview.net/forum?id=9slt6ZbEiI)
  - *Theory, benchmarks & applications*: [Surprising Generalization of TFMs](https://openreview.net/forum?id=ikYK94iVvj), [Beyond IID / BeyondArena](https://openreview.net/forum?id=Mk0ohjszwO), [MulTaBench](https://openreview.net/forum?id=yOd4AxlEbb) (Spotlight), [TabPrep](https://openreview.net/forum?id=nMW7DYLGTo), [Benchmarking Attention for TFMs](https://openreview.net/forum?id=1MbSbrblLi), [TFMs on Genetic Data](https://openreview.net/forum?id=3qspmxVnC1), [TabBioMed](https://openreview.net/forum?id=4WpEeaOeGW), [TRL-Bench](https://openreview.net/forum?id=TkwRxfihFM), [Monroe](https://openreview.net/forum?id=ChiaszwjMu)
- **ICML 2026 main conference** - Tabular FM papers include [TabICLv2](https://icml.cc/virtual/2026/poster/63874), [TabSwift](https://arxiv.org/abs/2606.07345) (Spotlight), [LimiX-2M](https://icml.cc/virtual/2026/poster/63251), [GOTabPFN](https://icml.cc/virtual/2026/poster/62523), and [Strategic Prior-data Fitted Networks](https://icml.cc/virtual/2026/poster/62109).

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
*Last updated: September 2026*
