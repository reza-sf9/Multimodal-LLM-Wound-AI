# Multimodal LLM Wound AI

## Overview
This repository contains **sample code** for an ongoing project on **Multimodal Large Language Model (LLM) Wound AI**. The project explores advanced multimodal prompting techniques for medical wound assessment and decision-making.

**Note:** Due to privacy regulations, the dataset containing medical images is not shared in this repository. The provided code serves as a demonstration of the methodologies used in the project.

This repository supports **Llama 3.2** and **GPT API**, but other LLMs can be easily tested by adding your API key.

## Repository Structure

### 1. `CoT_ToT_FewShot_Multimodal/`
This folder contains implementations of **multimodal prompting techniques** that utilize both **image and text modalities** for wound assessment. The implemented techniques include:
- **Chain-of-Thought (CoT)**: Step-by-step reasoning for multimodal AI decision-making.
- **Tree-of-Thought (ToT)**: Hierarchical reasoning for structured medical evaluations.
- **Few-Shot Prompting**: Leveraging small labeled datasets to improve model generalization.

### 2. `agentic_workflow_reflection/`
This folder contains code implementing **reflection-based workflows** for multimodal prompting. It focuses on **agentic reasoning**, where models iteratively refine their responses based on past outputs. Key features include:
- Reflection-based adjustments in multimodal medical assessments.
- Agentic decision refinement for improved wound classification and recommendations.

## Requirements
- Python 3.x
- OpenAI GPT API / Llama 3.2
- Hugging Face Transformers
- NumPy, Pandas
- PyTorch
- Matplotlib (for visualization)




