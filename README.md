# Multi-Agent Symbolic Music Understanding

A multi-agent system for analyzing symbolic music notation (ABC format) and performing music understanding tasks including metadata QA and emotion recognition.

## Overview

This project implements a multi-agent architecture that uses Large Language Models (LLMs) to understand and analyze symbolic music scores in ABC notation format. The system can handle two main types of tasks:

1. **Music Theory & Metadata QA**: Questions about key signatures, time signatures, bars, chords, and other structural elements of music
2. **Emotion Recognition**: Classification of musical pieces into emotional categories (Q1-Q4) based on arousal-valence dimensions

## Architecture

The system consists of four main agents:

- **Agent A (Controller)**: Routes user queries to appropriate specialized agents based on the question type
- **Agent B (ABC Expert)**: Analyzes ABC notation and answers music theory/metadata questions using a two-step approach (expert analysis + evaluator)
- **Agent C (Emotion Expert)**: Classifies emotions using an arousal-valence approach with multiple analysts and majority voting
- **Agent D (Aggregator)**: Combines results from multiple agents when both ABC and emotion questions are present

## Features
### Controller
### ABC Notation Analysis
- Analyzes key signatures, time signatures, note lengths, chords, bars, rhythm patterns, and melodic contours
- Answers questions about music structure and notation

### Emotion Recognition
- Classifies music into 4 emotional categories:
  - **Q1**: Happy (high valence, high arousal)
  - **Q2**: Angry (low valence, high arousal)
  - **Q3**: Sad (low valence, low arousal)
  - **Q4**: Relaxed (high valence, low arousal)
- Uses arousal-valence decomposition with multiple analysts and majority voting
- Provides detailed reasoning for classifications

### Input Validation
- Automatically extracts ABC scores from prompts
- Validates input completeness and question presence
- Handles multiple input formats (code blocks, "Input:" markers, etc.)


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

### Emotion Classification Only

Test the emotion classification system:

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

### Baseline Models

Run baseline models for comparison:

```bash
# Emotion recognition baseline (direct LLM)
python src/emotion_baseline.py data/Emotion_Recognition_cleaned.csv

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

Prepare raw data files:

```bash
python data/prepare_data.py
```

This script processes raw CSV files and creates cleaned versions:
- `Error_Detection_cleaned.csv`
- `Metadata_QA_cleaned.csv`
- `Emotion_Recognition_cleaned.csv`
- `Bar_Sequencing_cleaned.csv`

## Dataset

### ABC/Metadata QA Data

Part of the metadata QA and ABC notation analysis test data in this project comes from the **ABC-Eval dataset**.

**Dataset Information:**
- **Name**: ABC-Eval
- **Source**: [Anonymous Review Link](https://anonymous.4open.science/r/ABC-Eval-B622)

**Citation:**
```
ABC-Eval: A Comprehensive Benchmark for Evaluating Large Language Models 
on ABC Notation Understanding. https://anonymous.4open.science/r/ABC-Eval-B622
```

### Emotion Recognition Data

Part of the emotion-related test data in this project comes from the **Rough4Q subset of the EMelodyGen dataset** (Joe, 2024), which provides ABC-notation melodies annotated with valence–arousal–based emotion labels (Q1–Q4).

**Dataset Information:**
- **Name**: EMelodyGen (Rough4Q subset)
- **Source**: [HuggingFace Dataset](https://huggingface.co/datasets/monetjoe/EMelodyGen)
- **Format**: ABC notation melodies with emotion labels

**Citation:**
```
Joe, M. (2024). EMelodyGen: Emotion-Conditioned Melody Generation in ABC 
Notation with Musical Feature Templates. https://arxiv.org/abs/2405.16775
```

The dataset is publicly available and can be accessed via the HuggingFace link above. When using emotion recognition features or test data, please cite the original EMelodyGen paper.

## Model Configuration

The system uses the following LLM models :

- **Default**: `google/gemma-3-27b-it`
- **Alternative**: `meta-llama/Meta-Llama-3.1-70B-Instruct`


## Project Structure

```
.
├── src/
│   ├── multi_agent_system.py          # Main multi-agent system
│   ├── metadata_QA_agent.py           # test for ABC expert agent for metadata QA
│   ├── metadata_QA_baseline.py         # test for Baseline for metadata QA
│   ├── emotion_baseline.py             # test for Direct LLM baseline for emotion
│   ├── emotion_recognition_baseline_draft_version.py  # Alternative emotion baseline
│   ├── multi_agent_test_emotion.py     # test for Emotion classification system
│   ├── test_val_con.py                 # Controller testing script
│   └── inference_auth_token.py         # Authentication module
├── data/
│   ├── prepare_data.py                 # Data preparation script
│   ├── *.csv                           # Dataset files
│   └── *_cleaned.csv                   # Processed datasets
├── result/                             # Output results
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Key Components

### ABC Score Extraction

The system can extract ABC scores from various formats:
- Triple backticks: ` ```abc ... ``` `
- "Input:" markers
- "Score:" markers
- Pattern matching for ABC notation headers (X:, K:, M:, L:, R:)

### Emotion Classification Pipeline

1. **Arousal Classification**: Multiple analysts (default: 3) classify arousal as HIGH or LOW
2. **Valence Classification**: Multiple analysts classify valence as HIGH or LOW
3. **Majority Voting**: Final arousal/valence determined by majority vote
4. **Emotion Mapping**: Combined arousal-valence mapped to Q1-Q4 categories

### Controller Logic

The controller agent analyzes user questions and routes to:
- **ABC**: Music theory, notation, structure questions
- **EMOTION**: Emotion, mood, valence/arousal questions
- **BOTH**: When both types of questions are present
- **NONE**: When no specialized agent is needed

## Results

Results are saved as CSV files with additional columns:
- `agent_answer`: Full response from the agent
- `predicted_label`: Predicted emotion label (0-3)
- `predicted_arousal`: Predicted arousal level (HIGH/LOW)
- `predicted_valence`: Predicted valence level (HIGH/LOW)
- `ground_truth_arousal`: Ground truth arousal (for evaluation)
- `ground_truth_valence`: Ground truth valence (for evaluation)

## Evaluation Metrics

The system reports:
- **Overall Accuracy**: Label prediction accuracy
- **Arousal Accuracy**: Arousal classification accuracy
- **Valence Accuracy**: Valence classification accuracy

## Limitations

- Requires access to ALCF inference API
- ABC score extraction may fail on non-standard formats
- Emotion classification is based on symbolic features only (no audio)
- Model performance depends on the underlying LLM capabilities

## Contributing

When adding new features:
1. Follow the existing agent architecture pattern
2. Update input validation if needed
3. Add appropriate test scripts
4. Document new functionality in this README


## Citation


**ABC/Metadata QA and emotion recognition Dataset:**
```
ABC-Eval: A Comprehensive Benchmark for Evaluating Large Language Models 
on ABC Notation Understanding. https://anonymous.4open.science/r/ABC-Eval-B622
```

**Emotion Recognition Dataset:**
```
Joe, M. (2024). EMelodyGen: Emotion-Conditioned Melody Generation in ABC 
Notation with Musical Feature Templates. https://arxiv.org/abs/2405.16775
Dataset: https://huggingface.co/datasets/monetjoe/EMelodyGen
```


