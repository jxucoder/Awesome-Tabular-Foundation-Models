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
- **[TabPFN](https://github.com/automl/TabPFN)**: The official repository for the TabPFN model.
- **[TabICL](https://github.com/soda-inria/tabicl)**: Open-source, scikit-learn-compatible tabular foundation model for in-context learning (TabICLv2).
- **[TabTune](https://arxiv.org/abs/2511.02802)**: A unified library for inference and fine-tuning across multiple tabular foundation models.

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

### Synthetic Data & Generation

- **TabuLa: Harnessing Language Models for Tabular Data Synthesis** (arXiv 2023)
  *Zilong Zhao, Robert Birke, Lydia Y. Chen*
  [Paper](https://arxiv.org/abs/2310.12746)
  > Leverages language models for generating synthetic tabular data.

- **CTSyn: A Foundation Model for Cross Tabular Data Generation** (arXiv 2024)
  *Xiaofeng Lin, Chenheng Xu, Matthew Yang, Guang Cheng*
  [Paper](https://arxiv.org/abs/2406.04619)
  > A diffusion-based generative foundation model for tabular data. Uses an autoencoder to consolidate diverse tables into a unified latent space and a conditional latent diffusion model for generation, conditioned on table schema. Outperforms existing synthesizers on standard benchmarks in both utility and diversity.

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

### Causal Inference

- **Foundation Models for Causal Inference via Prior-Data Fitted Networks (CausalFM)** (NeurIPS 2025)
  *Yuchen Ma, et al.*
  [Paper](https://arxiv.org/abs/2506.10914)
  > A framework for training PFN-based foundation models for various causal inference settings (back-door, front-door adjustments).

- **Do-PFN: In-Context Learning for Causal Effect Estimation** (NeurIPS 2025)
  *Jake Robertson, et al.*
  [Paper](https://arxiv.org/abs/2506.06039)
  > Applies PFNs to estimate causal effects without knowledge of the underlying causal graph.

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

- **TabZilla** - Comprehensive tabular data benchmark
- **OpenML** - Large collection of tabular datasets
- **Kaggle Competitions** - Real-world tabular challenges

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
*Last updated: March 2026*
