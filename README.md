# RetCare: Towards Interpretable Clinical Decision Making through LLM-Driven Medical Knowledge Retrieval

RetCare is a workflow designed to enhance the interpretability and reliability of clinical decision-making models. By leveraging retrieval-augmented generation (RAG) pipelines and large language models (LLMs), RetCare integrates authoritative medical literature to validate machine learning outputs, providing detailed and interpretable reasoning to support clinical decisions.

## Features

- **Integration of Authoritative Medical Knowledge**: Uses PubMed to validate prediction results and feature importance.
- **Comprehensive Prompting Strategies**: Develops prompts to integrate model outputs with healthcare context.
- **Interpretable Reasoning Capacities**: Provides detailed explanations and refined predictions using LLMs.

## Environmental Setups

- Create an environment `colacare` and activate it.

```bash
conda create -n colacare python=3.9
conda activate colacare
```

- Install the required packages.

```bash
pip install -r requirements.txt
```