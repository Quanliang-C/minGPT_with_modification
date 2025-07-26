[中文版本](README_CN.md)

# minGPT_with_modification

## Table of Contents

- [Project Overview](#project-overview)
- [Core Technologies and Architecture](#core-technologies-and-architecture)
- [Base Model: minGPT](#base-model-mingpt)
- [Pre-training Task: Span Corruption](#pre-training-task-span-corruption)
- [Positional Encoding Exploration: From Learnable to RoPE](#positional-encoding-exploration-from-learnable-to-rope)
- [Experiments and Result Analysis](#experiments-and-result-analysis)
- [Experimental Setup](#experimental-setup)
- [Experiment 1: Fine-tuning without Pre-training](#experiment-1-fine-tuning-without-pre-training)
- [Experiment 2: Standard Pre-training and Fine-tuning (Vanilla Transformer)](#experiment-2-standard-pre-training-and-fine-tuning-vanilla-transformer)
- [Experiment 3: Pre-training and Fine-tuning with RoPE](#experiment-3-pre-training-and-fine-tuning-with-rope)
- [Summary of Results](#summary-of-results)
- [Conclusion](#conclusion)

## Project Overview

This project originates from the Stanford University course CS224N: NLP with Deep Learning and aims to deeply investigate the impact of pre-training on the performance of Transformer models in knowledge-intensive tasks. The project is built and extended upon Andrej Karpathy's minGPT library.

The core task is a simple form of "question answering": given the name of a notable person, the model must predict their birthplace. For example:

Question: "Where was [person] born?"
Answer: "[place]"

The challenge of this task is that the answer cannot be directly inferred from the input; it must rely on the "world knowledge" the model has acquired during its training. Through several key explorations, this project validates the power of pre-training and implements a more advanced positional encoding scheme:

- Implemented the Span Corruption pre-training task, inspired by the T5 paper, enabling the model to learn factual knowledge from large-scale unlabeled text (Wikipedia).
- Implemented Rotary Positional Embedding (RoPE) as an improvement over traditional learnable absolute positional embeddings and successfully applied it to the model.
- Clearly demonstrated the performance evolution from training from scratch, standard pre-training, to pre-training with RoPE through three comparative experiments.

## Core Technologies and Architecture

### Base Model: minGPT

The project uses a GPT-architecture Transformer model as its foundation, based on the minGPT implementation. minGPT is a clean and concise implementation of a Transformer, making it ideal for academic research and custom development. We modified this base to integrate our new pre-training task and positional encoding mechanism.

### Pre-training Task: Span Corruption

To enable the model to learn world knowledge, we introduced a pre-training step. Inspired by the T5 paper (Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer), we implemented Span Corruption as our pre-training objective.

The core idea is as follows:

- Randomly select a contiguous sequence of text (a span) from the input.
- Replace this span with a special [MASK] token.
- The model's task is to predict the original text of the masked span based on the surrounding context.

By performing this task on a massive Wikipedia corpus, the model is forced to learn and internalize linguistic structures, grammar, and, most importantly, factual knowledge (e.g., the association between a specific person and their birthplace). This is the key to the model's subsequent success in the question-answering task.

### Positional Encoding Exploration: From Learnable to RoPE

The Transformer architecture itself is permutation-invariant and requires positional encodings to process sequences. This project compares two different approaches.

1. Vanilla Transformer: Learnable Absolute Positional Embeddings

This is the default method in minGPT and a classic approach used in many early Transformer models like BERT and GPT-2. The model learns a unique vector for each absolute position (1st, 2nd, 3rd token, etc.), which is then added to the token embedding. A drawback of this method is its limited generalization; it may perform poorly on sequences longer than those seen during training because it has not learned embeddings for those new positions.

2. RoPE: Rotary Positional Embedding

As a core innovation of this project, we personally implemented RoPE (Roformer: Enhanced Transformer with Rotary Position Embedding) and replaced the original learnable positional embeddings with it.

The key advantage of RoPE is that it is a form of relative positional encoding:

- It doesn't "add" positional information to the token embeddings but rather "multiplies" it into the model's Query and Key vectors through a rotation operation.
- The dot product between any two token vectors is dependent only on their relative distance, not their absolute positions.

This allows the model to better understand the relative relationships between tokens and generalize more smoothly to unseen sequence lengths.

Our implementation follows the original RoPE paper, applying the rotational transformation to the Query and Key vectors within each head of the self-attention mechanism.

## Model Configuration and Key Hyper-parameters

| Parameter | Value | Note |
|-----------|-------|------|
| `n_layer` | **4** | Number of Transformer blocks |
| `n_head`  | **8** | Attention heads per block |
| `n_embd`  | **256** | Embedding / hidden size |
| `block_size` | **128 tokens** | Maximum context length |
| `vocab_size` | **256 chars** | Built dynamically from `wiki.txt` |

Other important training knobs (defined in `run.py` & `trainer.py`):

```python
# Pre-training (default)
max_epochs   = 650
batch_size   = 128
learning_rate= 6e-3

# Fine-tuning (with pre-train)
max_epochs   = 15
batch_size   = 256
learning_rate= 6e-4
```

`trainer.TrainerConfig` further controls weight-decay, gradient-clipping (`1.0`), cosine LR-decay with warm-up (`warmup_tokens = 512*20`) and automatic TensorBoard logging.

### Custom Datasets

| Dataset | Purpose | Length | Highlight |
|---------|---------|--------|-----------|
| `CharCorruptionDataset` | Span-corruption self-supervised pre-training | 128 | Implements T5-style masking: `prefix ⁇ suffix ⁇ masked_span + PAD` |
| `NameDataset` | Supervised fine-tuning on `birth_places_train.tsv` | 127 | Keeps the **question tokens masked in `y`** so loss only flows on the answer span |

Both datasets share the same vocabulary to guarantee embedding coherence between stages.

## Experiments and Result Analysis

### Experimental Setup

Pre-training Data: English Wikipedia (wiki.txt)
Fine-tuning/Evaluation Data: A dataset of person-birthplace pairs (birth_places_train.tsv, birth_dev.tsv)
Evaluation Metric: Accuracy on the development set.

### Experiment 1: Fine-tuning without Pre-training

In this baseline experiment, we fine-tuned a randomly initialized Transformer model directly, without any pre-training.

Objective: To verify if the model can learn the task solely from the fine-tuning data without prior knowledge.

Result: 1.8% Accuracy (9 correct out of 500).

Analysis: This result is close to random guessing. It clearly shows that the model cannot "create" knowledge out of thin air. The fine-tuning dataset alone is insufficient for the model to learn the real-world association between names and places, leading to extremely poor performance.

### Experiment 2: Standard Pre-training and Fine-tuning (Vanilla Transformer)

In this experiment, we first pre-trained the standard Transformer (with learnable positional embeddings) on Wikipedia using the Span Corruption task, and then fine-tuned it on the person-birthplace dataset.

Objective: To validate the effectiveness of pre-training for knowledge-intensive tasks.

Pre-training Params: batch_size=256, epochs=1200
Fine-tuning Params: batch_size=512, epochs=50

Result: 24.0% Accuracy (120 correct out of 500).

Analysis: A massive leap in performance. This strongly proves the value of pre-training. Through large-scale unsupervised learning on Wikipedia, the model successfully encoded a vast amount of factual knowledge into its parameters. During the fine-tuning stage, the model learned how to "retrieve" and utilize this stored knowledge to answer questions.

### Experiment 3: Pre-training and Fine-tuning with RoPE

This is our core experiment, where we replaced the learnable positional embeddings in the standard model with our implementation of RoPE and repeated the full pre-training and fine-tuning pipeline.

Objective: To evaluate the performance of our RoPE implementation against traditional positional embeddings.

Pre-training Params: batch_size=128, epochs=650
Fine-tuning Params: batch_size=256, epochs=15

Result: 24.4% Accuracy (122 correct out of 500).

Analysis: The model with RoPE not only worked but also achieved a higher accuracy than the vanilla Transformer. This indicates that our RoPE implementation is correct and effective. The relative positional encoding scheme showed a slight advantage in this task, demonstrating its potential as a more advanced positional encoding technique.

### Summary of Results

| Experiment | Positional Encoding | Pre-trained? | Dev Set Accuracy |
|------------|---------------------|--------------|------------------|
| 1. Finetune Only | Learnable | No | 1.8% |
| 2. Pretrain + Finetune | Learnable | Yes | 24.0% |
| 3. RoPE Pretrain + Finetune | Rotary (RoPE) | Yes | 24.4% |

## Conclusion

This project successfully reproduced and validated the critical role of pre-training in endowing Transformer models with world knowledge. Through a series of comparative experiments, we draw the following conclusions:

- Pre-training is Indispensable: For NLP tasks requiring external knowledge, a Transformer model without pre-training is almost incapable of performing the task.
- Span Corruption is an Effective Pre-training Method: This objective effectively encourages the model to learn and memorize factual information from text.
- RoPE is an Excellent Alternative: Our successful implementation of Rotary Positional Embedding (RoPE) performed slightly better than traditional learnable positional embeddings, validating the value of relative positional information for this type of task and the correctness of our engineering implementation.

In summary, this project was not only a practical application of knowledge from the CS224N course but also a successful exploration and implementation of cutting-edge Transformer technologies like RoPE and T5-style pre-training.

