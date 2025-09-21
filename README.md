# EffiChem — Efficient Adaptation of Chemical Language Models for Molecular Property Prediction  
[![Paper](https://img.shields.io/badge/Preprint-ChemRxiv-+-doi-10.26434%2Fchemrxiv-2025-2lljt-blue)](https://chemrxiv.org/engage/chemrxiv/article-details/68b7f02e728bf9025ef4fcfc)  

This repository contains the implementation of **EffiChem**, described in:

> **EffiChem: Efficient Adaptation of Chemical Language Models for Molecular Property Prediction**  
> Harsha Harod, Kavya Agrawal, Ines Simeone, Michele Ceccarelli, Sukrit Gupta, Raghvendra Mall.  
> Version 1, ChemRxiv (08 September 2025). :contentReference[oaicite:1]{index=1}  

---

## Abstract (from the paper)

> Chemical language models (CLMs) such as ChemBERTa and Molformer enable compound property prediction, albeit using an over-parameterized model, and the computational demands of full finetuning such models limit their adoption in resource-constrained settings. Parameter-efficient finetuning techniques are critical to democratize AI-driven drug discovery and design.  
> In this paper, we integrate Low-Rank Adapters (LoRA) with chemical language models, significantly reducing the number of trainable parameters required for finetuning. We observe improved performance of 3-5 % for AUC metric across classification tasks, for molecule toxicity, blood-brain barrier permeability, and flavor prediction tasks over state-of-the-art like Molformer-XL and MAMMAL.  
> We attain optimal performance by comprehensive benchmarking of embeddings generated from zero-shot and finetuned CLMs combined with molecular physicochemical properties across three diverse datasets. Our approach achieves MCC scores ranging from **0.80 to 0.90** across three tasks, while reducing model parameters by **62 % to 96 %** in the top-performing models. This is achieved via lightweight adaptation of CLMs, which retains their performance efficiency while reducing the over-parameterization burden.  
> This work provides a practical framework for deploying a scalable and sustainable paradigm for high-performance molecular AI tools in resource-constrained settings. :contentReference[oaicite:2]{index=2}  

---

## Highlights

- Parameter-efficient finetuning using **LoRA** (Low-Rank Adapters) applied to large chemical language models (ChemBERTa, Molformer).  
- Benchmarking both zero-shot and finetuned embeddings, combined with **physicochemical descriptors** (RDKit).  
- Classification tasks: **toxicity (ClinTox)**, **blood-brain barrier permeability (BBBP)**, and **flavor prediction (Flavor / FART)**.  
- Significant improvements: **3-5 % in AUC**, **MCC scores 0.80–0.90**, **parameter reduction 62-96%** in top models. :contentReference[oaicite:3]{index=3}  

---

## Table of Contents

1. [Installation](#installation)  
2. [Quickstart](#quickstart)  
3. [Reproduce Paper Results](#reproduce-paper-results)  
4. [Repository Structure](#repository-structure)  
5. [How EffiChem Works](#how-effichem-works)  
6. [Recommended Hyperparameters](#recommended-hyperparameters)  
7. [Interpretability](#interpretability)  
8. [Citation](#citation)  
9. [FAQ](#faq)  

---
