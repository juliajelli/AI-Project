# ICD-10 Medical Diagnosis from Doctor-Patient Dialogues

Automated ICD-10 code prediction from doctor-patient consultation dialogues using two complementary approaches: a fine-tuned **LLM** (Ministral-3B with LoRA) that generates full SOAP notes with diagnosis codes, and a fine-tuned **embedding model** (Gemma-300M) that directly classifies dialogues into ICD-10 categories.

**Owners:** Julia Jellinek & Keno Hanken
**Institution:** Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam
**Course:** M. Grum -- Advanced AI-based Application Systems
**License:** AGPL-3.0
**Data Source:** [Ahmad0067/MedSynth on Huggingface](https://huggingface.co/datasets/Ahmad0067/MedSynth)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Docker Images](#docker-images)
4. [Data](#data)
5. [Pipeline Overview](#pipeline-overview)
6. [Step 1 -- Data Preparation](#step-1----data-preparation)
7. [Step 2 -- Training / Fine-Tuning](#step-2----training--fine-tuning)
8. [Step 3 -- Inference & Benchmarking](#step-3----inference--benchmarking)
9. [Docker Compose Usage](#docker-compose-usage)
10. [Infrastructure & Containers](#infrastructure--containers)
11. [File Reference](#file-reference)

---

## Project Overview

The project tackles the task of predicting ICD-10 diagnosis codes from synthetic doctor-patient consultation transcripts. Two model tracks run in parallel:

| Track | Base Model | Method | Input | Output |
|-------|-----------|--------|-------|--------|
| **LLM** | Ministral-3B (Mistral) | QLoRA (4-bit) fine-tuning | Dialogue transcript | SOAP note + ICD-10 code + diagnosis description |
| **Embedding** | google/embedding-gemma-300m | Classification head fine-tuning | Dialogue transcript | Top-k ICD-10 code predictions with probabilities |

The embedding model serves as a comparison baseline to the LLM approach, providing a non-generative classification method for the ICD-10 prediction task.

All training and inference is designed to run on an HPC cluster via **SLURM** with **NVIDIA A100 40 GB** GPUs, using **Apptainer** (Singularity) containers for reproducibility.

---

## Repository Structure

```
AI-Project/
├── README.md
│
├── images/                                  <- Docker images (AI-CPS structure)
│   ├── learningBase_icd10_diagnosis/        <- training & validation data
│   │   ├── Dockerfile
│   │   └── README.md
│   ├── activationBase_icd10_diagnosis/      <- activation/test data
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── activation_data.csv
│   ├── knowledgeBase_icd10_diagnosis/       <- trained model (LoRA adapters)
│   │   ├── Dockerfile
│   │   └── README.md
│   └── codeBase_icd10_diagnosis/            <- inference code
│       ├── Dockerfile
│       ├── README.md
│       ├── inference.py
│       ├── benchmark_icd10_bf16.py
│       └── classify.py
│
├── scenarios/                               <- Docker Compose application scenarios
│   ├── apply_icd10_diagnosis_llm/
│   │   └── docker-compose.yml
│   └── apply_icd10_diagnosis_embedding/
│       └── docker-compose.yml
│
├── code/                                    <- Data preparation scripts
│   ├── README.md
│   ├── dataprep/
│   │   ├── joint_data_collection.csv        <- raw source dataset (76 MB)
│   │   ├── dataprep_embedding.py
│   │   └── dataprep_llm.py
│   └── ipynb_notebooks/
│       ├── MedSynth_huggingface_final.csv
│       ├── cleaning_embedding.ipynb
│       └── cleaning_llm.ipynb
│
└── data/                                    <- Training/inference working directories
    ├── learningBase/                        <- fine-tuning environment
    │   ├── requirements.txt
    │   ├── embedding/
    │   │   ├── embedding_finetuning.py
    │   │   ├── Dockerfile
    │   │   ├── container_embedding_finetuning.def
    │   │   ├── run_embedding_finetuning.slurm
    │   │   ├── training_finetuning_embedding.json
    │   │   └── validation_finetuning_embedding.json
    │   └── llm/
    │       ├── README.md
    │       ├── llm_finetuning.py
    │       ├── Dockerfile
    │       ├── container_llm_finetuning.def
    │       ├── run_llm_finetuning.slurm
    │       ├── training_finetuning_llm.jsonl
    │       └── validation_finetuning_llm.jsonl
    └── activationBase/                      <- inference & benchmarking
        ├── README.md
        ├── Dockerfile
        ├── activation_data.csv
        ├── embedding/
        │   ├── readme.md
        │   ├── classify.py
        │   └── validation_finetuning_embedding.json
        └── llm/
            ├── README.md
            ├── Dockerfile
            ├── inference.py
            ├── benchmark_icd10_bf16.py
            ├── run_benchmark_bf16.slurm
            └── validation_finetuning_llm.jsonl
```

---

## Docker Images

The following Docker images follow the AI-CPS repository structure:

| Image | Purpose | Docker Pull Command |
|-------|---------|---------------------|
| `learningBase_icd10_diagnosis` | Training & validation data at `/tmp/learningBase/` | `docker pull <USERNAME>/learningbase_icd10_diagnosis` |
| `activationBase_icd10_diagnosis` | Activation data at `/tmp/activationBase/` | `docker pull <USERNAME>/activationbase_icd10_diagnosis` |
| `knowledgeBase_icd10_diagnosis` | Trained LoRA adapters at `/tmp/knowledgeBase/` | `docker pull <USERNAME>/knowledgebase_icd10_diagnosis` |
| `codeBase_icd10_diagnosis` | Inference scripts at `/tmp/codeBase/` | `docker pull <USERNAME>/codebase_icd10_diagnosis` |

> **Note:** Replace `<USERNAME>` with the Docker Hub username after publishing.

### Building Images Locally

```bash
cd AI-Project/images

# learningBase - requires data files to be copied first
cd learningBase_icd10_diagnosis
cp ../../data/learningBase/llm/training_finetuning_llm.jsonl .
cp ../../data/learningBase/llm/validation_finetuning_llm.jsonl .
cp ../../data/learningBase/embedding/training_finetuning_embedding.json .
cp ../../data/learningBase/embedding/validation_finetuning_embedding.json .
docker build -t learningbase_icd10_diagnosis .
cd ..

# activationBase
docker build -t activationbase_icd10_diagnosis activationBase_icd10_diagnosis/

# knowledgeBase - requires model files after training
# docker build -t knowledgebase_icd10_diagnosis knowledgeBase_icd10_diagnosis/

# codeBase
docker build -t codebase_icd10_diagnosis codeBase_icd10_diagnosis/
```

### Publishing to Docker Hub

```bash
docker login
docker tag learningbase_icd10_diagnosis <USERNAME>/learningbase_icd10_diagnosis
docker push <USERNAME>/learningbase_icd10_diagnosis
# Repeat for other images...
```

---

## Data

### Source Dataset

The raw data comes from the **MedSynth** Huggingface dataset. The master file `joint_data_collection.csv` contains **1,567,388 rows** with four columns:

| Column | Description |
|--------|-------------|
| `Note` | Structured SOAP clinical note (Subjective, Objective, Assessment, Plan) |
| `Dialogue` | Full doctor-patient conversation transcript |
| `ICD10` | ICD-10 diagnosis code (e.g. `N870`, `A047`) |
| `ICD10_desc` | Human-readable diagnosis name (e.g. `MILD CERVICAL DYSPLASIA`) |

### Data Cleaning

1. **UTF normalization** -- NFKC normalization, removal of zero-width characters
2. **Leading formatting removal** -- strips markdown prefixes before the first `**` in SOAP notes
3. **NA rows dropped**
4. **Underrepresented ICD codes removed** -- codes with fewer than 5 samples excluded
5. **Sorted** by `ICD10` then `Note`

### Train / Validation Split

| Split | Records |
|-------|---------|
| Training | 8,136 |
| Validation | 2,034 |

### Data Formats

- **Embedding data** (`.json`): JSON array with `Dialogue` and `ICD10` fields
- **LLM data** (`.jsonl`): Chat format with `system`, `user`, `assistant` messages

---

## Pipeline Overview

```
joint_data_collection.csv
        │
        ├── dataprep_embedding.py ──→ training_finetuning_embedding.json
        │                             validation_finetuning_embedding.json
        │
        └── dataprep_llm.py ────────→ training_finetuning_llm.jsonl
                                      validation_finetuning_llm.jsonl
        │                                       │
        ▼                                       ▼
  embedding_finetuning.py              llm_finetuning.py
  (Gemma-300M + classifier)            (Ministral-3B + QLoRA)
        │                                       │
        ▼                                       ▼
  classify.py                          inference.py / benchmark_icd10_bf16.py
  (inference & benchmark)              (inference & benchmark)
```

---

## Step 1 -- Data Preparation

```bash
cd AI-Project/code/dataprep
python dataprep_embedding.py   # Outputs: JSON files for embedding model
python dataprep_llm.py         # Outputs: JSONL files for LLM
```

---

## Step 2 -- Training / Fine-Tuning

### Embedding Model

```bash
cd data/learningBase/embedding
bash build_embedding_finetuning_image.sh
sbatch run_embedding_finetuning.slurm
```

### LLM Model

```bash
cd data/learningBase/llm
bash build_llm_finetuning_image.sh
sbatch run_llm_finetuning.slurm
```

---

## Step 3 -- Inference & Benchmarking

### Embedding Inference

```bash
python classify.py --model_dir ./finetuned_model --text "Doctor-patient dialogue..."
```

**Benchmark Results (2,034 samples):**
- Top-1 Accuracy: 23.55%
- Top-3 Accuracy: 38.84%
- Top-5 Accuracy: 45.87%

### LLM Inference

```bash
python inference.py
# Paste dialogue when prompted
```

---

## Docker Compose Usage

### Prerequisites

```bash
docker volume create ai_system
```

### Running with Docker Compose

```bash
cd AI-Project/scenarios/apply_icd10_diagnosis_llm

# Populate the shared volume with data
docker-compose up learningbase activationbase knowledgebase codebase

# Run inference
docker-compose run --rm inference
```

---

## Infrastructure & Containers

### Apptainer Definitions

| File | Base Image | Purpose |
|------|------------|---------|
| `container_llm_finetuning.def` | `nvcr.io/nvidia/pytorch:24.07-py3` | LLM fine-tuning |
| `container_embedding_finetuning.def` | `nvidia/cuda:12.1.1-devel-ubuntu22.04` | Embedding fine-tuning |

### SLURM Jobs

| File | GPUs | Time | Task |
|------|------|------|------|
| `run_embedding_finetuning.slurm` | 3x A100 40 GB | 4 h | Embedding fine-tuning |
| `run_llm_finetuning.slurm` | 3x A100 40 GB | 4 h | LLM QLoRA fine-tuning |
| `run_benchmark_bf16.slurm` | 3x A100 40 GB | 12 h | LLM benchmark |

---

## File Reference

### Key Scripts

| File | Purpose |
|------|---------|
| `code/dataprep/dataprep_embedding.py` | Prepare embedding training data |
| `code/dataprep/dataprep_llm.py` | Prepare LLM training data |
| `data/learningBase/embedding/embedding_finetuning.py` | Fine-tune Gemma-300M |
| `data/learningBase/llm/llm_finetuning.py` | Fine-tune Ministral-3B with QLoRA |
| `data/activationBase/embedding/classify.py` | Embedding inference & benchmark |
| `data/activationBase/llm/inference.py` | LLM inference |
| `data/activationBase/llm/benchmark_icd10_bf16.py` | LLM benchmark |

### Data Files

| File | Format | Description |
|------|--------|-------------|
| `joint_data_collection.csv` | CSV | Raw dataset (1.57M rows) |
| `training_finetuning_*.json/jsonl` | JSON/JSONL | Training sets (8,136 records) |
| `validation_finetuning_*.json/jsonl` | JSON/JSONL | Validation sets (2,034 records) |
| `activation_data.csv` | CSV | Single example for testing |
