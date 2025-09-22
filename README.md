# EffiChem — Efficient Adaptation of Chemical Language Models for Molecular Property Prediction  
[![Preprint](https://img.shields.io/badge/Preprint-ChemRxiv-blue?logo=chemrxiv&style=flat-square)](https://chemrxiv.org/engage/chemrxiv/article-details/68b7f02e728bf9025ef4fcfc)

This repository contains the implementation of **EffiChem**, described in:

**EffiChem: Efficient Adaptation of Chemical Language Models for Molecular Property Prediction**  
> Harsha Harod, Kavya Agrawal, Ines Simeone, Michele Ceccarelli, Sukrit Gupta, Raghvendra Mall.  
> Version 1, ChemRxiv (08 September 2025).

## Abstract (from the paper)
- Chemical language models (CLMs) such as ChemBERTa and Molformer enable compound property prediction, albeit using an over-parameterized model, and the computational demands of full finetuning such models limit their adoption in resource-constrained settings. Parameter-efficient finetuning techniques are critical to democratize AI-driven drug discovery and design.
- In this paper, we integrate Low-Rank Adapters (LoRA) with chemical language models, significantly reducing the number of trainable parameters required for finetuning. We observe improved performance of 3-5 % for AUC metric across classification tasks, for molecule toxicity, blood-brain barrier permeability, and flavor prediction tasks over state-of-the-art like Molformer-XL and ChemBerta.
- We attain optimal performance by comprehensive benchmarking of embeddings generated from zero-shot and finetuned CLMs combined with molecular physicochemical properties across three diverse datasets. Our approach achieves MCC scores ranging from **0.80 to 0.90** across three tasks, while reducing model parameters by **62 % to 96 %** in the top-performing models. This is achieved via lightweight adaptation of CLMs, which retains their performance efficiency while reducing the over-parameterization burden.
- This work provides a practical framework for deploying a scalable and sustainable paradigm for high-performance molecular AI tools in resource-constrained settings.
  
---

## Highlights

- Parameter-efficient finetuning using **LoRA** (Low-Rank Adapters) applied to large chemical language models (ChemBERTa, Molformer).  
- Benchmarking both zero-shot and finetuned embeddings, combined with **physicochemical descriptors** (RDKit).  
- Classification tasks: **toxicity (ClinTox)**, **blood-brain barrier permeability (BBBP)**, and **flavor prediction (Flavor / FART)**.  
- Significant improvements: **3-5 % in AUC**, **MCC scores 0.80–0.90**, **parameter reduction 62-96%** in top models. :contentReference[oaicite:3]{index=3}  

---

## Table of Contents

1. [How EffiChem works](#how-effichem-works)
2. [Installation](#installation)
3. [Running LoRA Finetuning](#running-lora-finetuning)
4. [Running Tree Models](#running-tree-models)
5. [Interpretability](#interpretability)   

---

## How EffiChem works
EffiChem combines LoRA-finetuned molecular transformers with tree-based models for molecular property prediction. The workflow is summarised as follows:
![EffiChem Workflow](assets/EffiChem%20Model%20Architecture.png)

### 1. Base Model & Dataset
- Start with a pre-trained transformer model (**MolFormer** or **ChemBERTa**).  
- Provide a task-specific dataset (e.g., **BBBP**, **ClinTox**, **Flavor**).

### 2. Custom LoRA Trainer
- Fine-tune only linear layers using **LoRA weights** while keeping the base model frozen.  
- Compute key metrics: **MCC, F1, Accuracy, AUC-ROC**.  
- Use a suitable loss function for binary or multiclass classification.  
- Optimize hyperparameters (**rank, alpha, dropout, learning rate**) using **Bayesian methods**.

### 3. Merged Model Weights
- Combine pre-trained base weights with updated LoRA weights to produce the **finetuned model**.

### 4. Feature Extraction
- Generate **molecular embeddings** from the finetuned model.  
- Combine embeddings with **RDKit descriptors** (physicochemical and structural features) to form an enhanced feature space.

### 5. Tree-Based Modeling
- Train **XGBoost**, **LightGBM**, and **CatBoost** models on the combined feature set.  
- Use **Optuna** for automatic hyperparameter tuning to maximise prediction performance.

### 6. Evaluation on Test Samples
- Input a new molecule (**SMILES string**).  
- Pass it through the **finetuned transformer + RDKit features pipeline** to generate input features for tree models.  
- Predict **binary or multiclass output** (e.g., property classification).  
- Store **results, probabilities, and performance metrics** for visualization and analysis.
- 
---
## Installation
We recommend using **micromamba** for environment management (faster and lighter than conda).

**1) Create new environment**
```bash
micromamba create -n effichem python=3.10 -y
micromamba activate effichem
````
**2) Core PyTorch + HuggingFace + PEFT stack**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets peft evaluate wandb
```
**3) Chemistry tools (RDKit)**
```bash
micromamba install -c conda-forge rdkit -y
```
**4) Tree-based models + hyperparameter tuning**
```bash
pip install optuna lightgbm catboost xgboost
```
**5) Data science & utils**
```bash
pip install pandas numpy scikit-learn joblib
```
**6) Visualization**
```bash
pip install matplotlib seaborn
```
**7) Optional (logging, interpretability)**
```bash
pip install captum transformers-interpret
```

---

## Running LoRA Finetuning
After installation, you can finetune EffiChem with **LoRA adapters** on different datasets. We provide Jupyter Notebooks for LoRA finetuning on all datasets. Each notebook is pre-configured with **Weights & Biases (W&B)** integration for logging and hyperparameter management, so you do **not** need to manually set training parameters (e.g., `lora_r`, `lora_alpha`, learning rate). W&B handles sweeps and experiment tracking automatically.

| Dataset | Description | Notebook Link |
|---------|-------------|---------------|
| **BBBP** | Blood–Brain Barrier Permeability | [finetune_bbbp.ipynb](https://github.com/kavyagl2/EffiChem/blob/main/lora_finetuned_models/bbbp_task/finetune_bbbp.ipynb) |
| **ClinTox** | Drug Toxicity | [finetune_clintox.ipynb](https://github.com/kavyagl2/EffiChem/blob/main/lora_finetuned_models/clintox_task/finetune_clintox.ipynb) |
| **Flavor / FART** | Molecular Flavor Classification | [finetune_flavor.ipynb](https://github.com/kavyagl2/EffiChem/blob/main/lora_finetuned_models/flavor_task/finetune_flavor.ipynb) |

---

## Running TREE Models
After completing LoRA finetuning, the next step involves running ensemble tree models (XGBoost, LightGBM, and CatBoost) on the generated embeddings combined with RDKit molecular descriptors.

### Pipeline Overview
The tree models pipeline consists of four integrated components: 
1. Embedding Generation - Extracting embeddings from LoRA finetuned models.
2. Embedding Processing - Load and convert LoRA-finetuned embeddings.
3. Feature Extraction - Calculate 17 RDKit molecular descriptors.
4. ML Modelling & Results - Downstream task evaluation using ensemble tree models (XGBoost, LightGBM, CatBoost) with hyperparameter optimization, training, and comprehensive evaluation with ROC/PR curves. 

### Steps to Run Tree Models

**Extract Embeddings**
- BBBP: [BBBP Embedding Extraction](https://github.com/kavyagl2/EffiChem/blob/main/Tree_models/bbbp_notebook/embedding_bbbp.ipynb)
- ClinTox: [Clintox Embedding Extraction](https://github.com/kavyagl2/EffiChem/blob/main/Tree_models/clintox_notebook/embedding_clintox.ipynb)
- Flavor: [Flavor Embedding Extraction](https://github.com/kavyagl2/EffiChem/blob/main/Tree_models/flavor_notebook/embedding_extraction.ipynb)

**Final Execution (main script)**
- Update the CSV file paths in main.py with your saved embedding files.
- Execute the script:
  ```bash
  python main.py
  ```
> **Note:** Each folder contains the complete pipeline (embedding_processing.py, feature_extraction.py, ml_modelling.py, main.py) configured for the specific dataset.

---

