---
language:
- en
license: mit
pretty_name: PsTuts-RAG Q&A Dataset
size_categories:
- 10K<n<100K
tags:
- rag
- question-answering
- photoshop
- ragas
---

# 📊 PsTuts-RAG Q&A Dataset

This dataset contains question-answer pairs generated using [RAGAS](https://github.com/explodinggradients/ragas) from Photoshop tutorial video transcripts. It's designed for training and evaluating RAG (Retrieval-Augmented Generation) systems focused on Photoshop tutorials.

## 📝 Dataset Description

### Dataset Summary

The dataset contains 100 question-answer pairs related to Photoshop usage, generated from video transcripts using RAGAS's knowledge graph and testset generation capabilities. The questions are formulated from the perspective of different user personas (Beginner Photoshop User and Photoshop Trainer).

### Dataset Creation

The dataset was created through the following process:
1. Loading transcripts from Photoshop tutorial videos
2. Building a knowledge graph using RAGAS with the following transformations:
   - Headlines extraction
   - Headline splitting
   - Summary extraction
   - Embedding extraction
   - Theme extraction
   - NER extraction
   - Similarity calculations
3. Generating synthetic question-answer pairs using different query synthesizers:
   - SingleHopSpecificQuerySynthesizer (80%)
   - MultiHopAbstractQuerySynthesizer (10%)
   - MultiHopSpecificQuerySynthesizer (10%)

### Languages

The dataset is in English.

## 📊 Dataset Structure

### Data Instances

Each instance in the dataset contains:
- `user_input`: A question about Photoshop
- `reference`: The reference answer
- Additional metadata from RAGAS generation

Example:
```
{
  "user_input": "How can I use the Move tool to move many layers at once in Photoshop?",
  "reference": "If you have the Move tool selected in Photoshop, you can move multiple layers at once by selecting those layers in the Layers panel first, then dragging any of the selected layers with the Move tool."
}
```

### Data Fields

- `user_input`: String containing the question
- `reference`: String containing the reference answer
- Additional RAGAS metadata fields

### Data Splits

The dataset was generated from test and dev splits of the original transcripts.

## 🚀 Usage

This dataset can be used for:
- Fine-tuning RAG systems for Photoshop-related queries
- Evaluating RAG system performance on domain-specific (Photoshop) knowledge
- Benchmarking question-answering models in the design/creative software domain

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("mbudisic/pstuts_rag_qa")
```

## 📚 Additional Information

### Source Data

The source data consists of transcripts from Photoshop tutorial videos, processed and transformed into a knowledge graph using RAGAS.

### Personas Used in Generation

1. **Beginner Photoshop User**: Learning to complete simple tasks, use tools in Photoshop, and navigate the UI
2. **Photoshop Trainer**: Experienced trainer looking to develop step-by-step guides for Photoshop beginners

### Citation

If you use this dataset in your research, please cite:

```
@misc{pstuts_rag_qa,
  author = {Budisic, Marko},
  title = {PsTuts-RAG Q&A Dataset},
  year = {2023},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/mbudisic/pstuts_rag_qa}}
}
```

### Contributions

Thanks to RAGAS for providing the framework to generate this dataset. 