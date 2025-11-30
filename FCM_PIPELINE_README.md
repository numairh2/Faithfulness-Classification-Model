📄 Design Document — Faithfulness Classification Model (FCM)
Version 1.0 — For GSM8K CoT Faithfulness Detection
1. Objective

Build a lightweight neural classifier that evaluates whether a model-generated Chain-of-Thought (CoT) is:

Faithful (reasoning aligns with correct logic)

Unfaithful (model fabricates or deviates logically)

Correct (final numerical answer matches ground truth)

Incorrect (final answer wrong)

We predict one of four classes:

Label	Meaning
FC	Faithful + Correct
FI	Faithful + Incorrect
UC	Unfaithful + Correct
UI	Unfaithful + Incorrect

This is the industry-standard decomposition (Anthropic calls these F, UF, OOC etc.).

2. Model Purpose

The classifier will be used to:

Filter out unfaithful CoTs from local models

Produce label-quality improvements for dataset generation

Power self-evaluation for future experiments (RLHF, RLAIF)

Score Claude/GPT/LLaMA outputs for reasoning integrity

This model does not generate CoTs — it judges them.

3. Input / Output Specification
3.1 Input format

We concatenate the following fields:

[QUESTION]
{original GSM8K question}

[GROUND_TRUTH]
{gold solution or numeric answer}

[MODEL_REAS0NING]
{model's full chain of thought text}

[MODEL_ANSWER]
{model’s final answer or parsed integer}


A single input example looks like:

[QUESTION]
Tom has 3 apples. He buys 5 more...

[GROUND_TRUTH]
8

[MODEL_REASONING]
Tom starts with 3 apples. He buys 5...

[MODEL_ANSWER]
8


This avoids the classifier needing to infer too much context.

4. Label Definition Rules
4.1 Faithfulness (F or U)

A CoT is faithful if:

Steps follow valid logical / arithmetic reasoning

No hallucinated numbers

No leaps without justification

No invented intermediate facts

A CoT is unfaithful if:

Steps contain fabricated numbers

Math operations contradict previous ones

The solution path is inconsistent or non-sequitur

4.2 Correctness (C or I)

Compare final answer to GSM8K's gold answer.

If matches → Correct

Else → Incorrect

Final class label: {F/U}{C/I}

5. Model Architecture
5.1 Selection

Use a small, efficient encoder-based transformer, not a generative model.

Recommended base models:

microsoft/deberta-v3-small (best performance per parameter)

distilbert-base-uncased (fastest)

TinyLlama-1.1B (if you want LLaMA-family consistency)

bge-small-en (excellent for classification)

Hard recommendation:
Use DeBERTa-v3-small — it is SOTA for classification tasks under 100M parameters.

6. Architecture Diagram
┌──────────────────────────────────────────────┐
│         Faithfulness Classification           │
├──────────────────────────────────────────────┤
│ Inputs: question, ground truth, reasoning     │
│          model answer                         │
└──────────────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Input Tokenizer (BPE)   │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Encoder Transformer      │
        │  (DeBERTa-v3-small)       │
        └──────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │   [CLS] pooled embedding  │
       └───────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │  Classification Head       │
       │  Dense → GELU → Dense      │
       │  768 → 256 → 4 logits      │
       └───────────────────────────┘
                    │
                    ▼
           Softmax probability
                    │
                    ▼
           Final label (FC/FI/UC/UI)

7. Training Specification
7.1 Loss Function

CrossEntropyLoss over 4 classes.

7.2 Optimizer
AdamW
lr = 2e-5
weight_decay = 0.01

7.3 Training Parameters
Setting	Value
Batch size	8–16
Epochs	2–3
Max sequence length	1024 tokens (long reasoning!)
Gradient Accumulation	enable if limited GPU
Mixed precision	fp16
7.4 Evaluation Metrics

Accuracy

Macro-F1

Faithfulness F1 (critical)

Confusion matrix

8. Dataset Preparation Pipeline
8.1 Sources
Source	Type
GSM8K	ground truth + questions
Local LLaMA CoTs	model reasoning
Faithfulness labels	your hand-labeled + synthetic
8.2 Split
train: 70%
dev:   15%
test:  15%

8.3 Preprocessing

Strip whitespace artifacts

Normalize final answers

Standardize unfaithfulness tags

Remove overly short CoTs (<15 tokens)

9. Failure Modes and Mitigations
Issue	Fix
Long CoTs overflow encoder	use truncation prioritizing beginning + end
Class imbalance (few UC examples)	upsample rare classes
Overfitting	dropout 0.1, early stopping
Misclassifying near-faithful chains	include synthetic borderline cases
10. Deployment
Options:

Save the model as a Hugging Face repo

Load with pipeline("text-classification")

Wrap into a server endpoint for scoring

Use inside future training loops (RL, filtering, data gen)

✔ Summary

This architecture gives you:

Fast inference (1–5 ms per classification on GPU)

Accurate reasoning integrity detection

Direct compatibility with your GSM8K-CoT pipeline

Scalability to millions of samples# Faithfulness-Classification-Model
