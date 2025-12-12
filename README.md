# Multi-Agent Symbolic Music Understanding

A multi-agent system for analyzing symbolic music notation (ABC format) and performing music understanding tasks including metadata QA and emotion recognition.

## Overview

This project implements a multi-agent architecture that uses Large Language Models (LLMs) to understand and analyze symbolic music scores in ABC notation format. The system can handle two main types of tasks:

1. **Music Theory & Metadata QA**: Questions about key signatures, time signatures, bars, chords, and other structural elements of music
2. **Emotion Recognition**: Classification of musical pieces into emotional categories (Q1-Q4) based on arousal-valence dimensions

## Architecture

The system consists of four main agents:

- **Agent A (Validattion and Controller)**: Validate input from user and routes user queries to appropriate specialized agents based on the question type
- **Agent B (syntax understanding)**: Analyzes ABC notation and answers music theory/metadata questions using a two-step approach (expert analysis + evaluator)
- **Agent C (Emotion recognition)**: Classifies emotions using an arousal-valence approach with multiple analysts and majority voting
- **Agent D (Aggregator)**: Combines results from multiple agents when both ABC and emotion questions are present



## Usage

### Interactive Mode

Run the multi-agent system interactively:

```bash
python src/multi_agent_system.py
```

Enter your questions about ABC notation or emotion classification. Type 'quit' or 'exit' to stop.

### Batch Processing

Process a CSV file with prompts:

```bash
python src/multi_agent_system.py data/your_dataset.csv
```

The CSV file should contain a `prompt` column. Results will be saved to a new CSV file with `_multi_agent_results.csv` suffix.

### Emotion Classification
Test the one-shot LLM classification on emotion:

```bash

Test the emotion classification system:
python src/emotion_baseline.py data/Emotion_Recognition_cleaned.csv
```

```bash
# Using 3-vote approach (default)
python src/multi_agent_test_emotion.py data/Emotion_Recognition_cleaned.csv

# Using single-call approach (faster, less accurate)
python src/multi_agent_test_emotion.py data/Emotion_Recognition_cleaned.csv --single

# Limit number of samples
python src/multi_agent_test_emotion.py data/Emotion_Recognition_cleaned.csv --20

# Filter by split (for rough4q format)
python src/multi_agent_test_emotion.py data/rough4q_full_raw.csv --20 test
```

### Metadata QA Models
```bash
Run baseline models for comparison:

# Metadata QA baseline
python src/metadata_QA_baseline.py

# Metadata QA with agent (two-step approach)
python src/metadata_QA_agent.py
```

### Controller Testing

Test the controller agent's controller accuracy:

```bash
python src/test_val_con.py
```

## Data Format

### Input CSV Format

The system expects CSV files with the following columns:

- **`prompt`**: The full prompt containing ABC score and question
- **`solution`** (optional): Ground truth label for evaluation

Example prompt format:

```
Input:
X:1
K:C
M:4/4
L:1/8
CDEF GABc|

Task:
What is the key signature of this score?

Options:
0. C   1. D
2. E   3. F

Answer:
```

### Data Preparation

Prepare data for test of controller:

```bash
python data/prepare_data.py
```

prepare data for emotion test(dataset too large for github):
```bash
python data/prepare_rough_dt.py
```

## Model Configuration

The system uses the following LLM models :

- **Default**: `google/gemma-3-27b-it`
- **Alternative**: `meta-llama/Meta-Llama-3.1-70B-Instruct`




## Citation for dataset


**ABC/Metadata QA and emotion recognition Dataset:**

ABC-Eval: A Comprehensive Benchmark for Evaluating Large Language Models 
on ABC Notation Understanding. https://anonymous.4open.science/r/ABC-Eval-B622


**Emotion Recognition Dataset:**
Zhou, M., Li, X., Yu, F., & Li, W. (2023).
*EMelodyGen: Emotion-Conditioned Melody Generation in ABC Notation with the Musical Feature Template.*
arXiv:2309.13259. https://arxiv.org/abs/2309.13259

Dataset: https://huggingface.co/datasets/monetjoe/EMelodyGen



