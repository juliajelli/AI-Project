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
3. [Data](#data)
4. [Pipeline Overview](#pipeline-overview)
5. [Step 1 -- Data Preparation](#step-1----data-preparation)
6. [Step 2 -- Training / Fine-Tuning](#step-2----training--fine-tuning)
7. [Step 3 -- Inference & Benchmarking](#step-3----inference--benchmarking)
8. [Infrastructure & Containers](#infrastructure--containers)
9. [File Reference](#file-reference)

---

## Project Overview

The project tackles the task of predicting ICD-10 diagnosis codes from synthetic doctor-patient consultation transcripts. Two model tracks run in parallel:

| Track | Base Model | Method | Input | Output |
|-------|-----------|--------|-------|--------|
| **LLM** | Ministral-3B (Mistral) | QLoRA (4-bit) fine-tuning | Dialogue transcript | SOAP note + ICD-10 code + diagnosis description |
| **Embedding** | google/embedding-gemma-300m | Classification head fine-tuning | Dialogue transcript | Top-k ICD-10 code predictions with probabilities |

All training and inference is designed to run on an HPC cluster via **SLURM** with **NVIDIA A100 40 GB** GPUs, using **Apptainer** (Singularity) containers for reproducibility.

---

## Repository Structure

```
AI-Project/
├── README.md                          ← this file
├── code/
│   ├── dataprep/
│   │   ├── MedSynth_huggingface_final.csv   ← raw source dataset (76 MB, 1.57 M rows)
│   │   ├── dataprep_embedding.py            ← data cleaning & split for embedding model
│   │   └── dataprep_llm.py                  ← data cleaning & split for LLM
│   └── ipynb_notebooks/
│       ├── MedSynth_huggingface_final.csv   ← copy of raw dataset for notebook use
│       ├── cleaning_embedding.ipynb         ← interactive version of dataprep_embedding.py
│       └── cleaning_llm.ipynb               ← interactive version of dataprep_llm.py
│
└── data/
    ├── README2.md                           ← project metadata & licensing
    ├── activationBase/                      ← inference & benchmarking
    │   ├── activation_data.csv              ← single example case for quick testing
    │   ├── Dockerfile                       ← container packaging activation_data.csv
    │   ├── README.md
    │   ├── embedding/
    │   │   ├── readme.md
    │   │   ├── classify.py                  ← embedding inference & evaluation script
    │   │   ├── embedding_benchmark_result.json.json  ← benchmark output (2 034 samples)
    │   │   └── validation_finetuning_embedding.json  ← validation data (8.8 MB)
    │   └── llm/
    │       ├── inference.py                 ← interactive LLM inference script
    │       ├── benchmark_icd10_bf16.py      ← full LLM benchmark script
    │       ├── run_benchmark_bf16.slurm     ← SLURM job for LLM benchmark
    │       ├── run_training_bf16.slurm      ← SLURM job for bf16 LLM training
    │       └── validation_finetuning_llm.jsonl  ← validation data (17 MB, 2 034 records)
    │
    └── learningBase/                        ← training / fine-tuning
        ├── README3.md                       ← credits & attribution
        ├── requirements.txt                 ← conda environment spec (Python 3.13, PyTorch 2.8)
        ├── embedding/
        │   ├── embedding_finetuning.py      ← Gemma-300M fine-tuning script
        │   ├── build_embedding_finetuning_image.sh  ← build Apptainer image
        │   ├── container_embedding_finetuning.def   ← Apptainer definition
        │   ├── run_embedding_finetuning.slurm       ← SLURM job submission
        │   ├── training_finetuning_embedding.json   ← training data (36 MB, 8 136 records)
        │   └── validation_finetuning_embedding.json ← validation data (8.8 MB, 2 034 records)
        └── llm/
            ├── README.md                    ← step-by-step LLM fine-tuning guide
            ├── llm_finetuning.py            ← Ministral-3B QLoRA fine-tuning script
            ├── Dockerfile                   ← Docker alternative for LLM training
            ├── build_llm_finetuning_image.sh        ← build Apptainer image
            ├── container_llm_finetuning.def         ← Apptainer definition
            ├── run_llm_finetuning.slurm             ← SLURM job submission
            ├── training_finetuning_llm.jsonl        ← training data (67 MB, 8 136 records)
            └── validation_finetuning_llm.jsonl      ← validation data (17 MB, 2 034 records)
```

---

## Data

### Source Dataset

The raw data comes from the **MedSynth** Huggingface dataset. The master file `MedSynth_huggingface_final.csv` contains **10 240 rows** with four columns:

| Column | Description |
|--------|-------------|
| `Note` | Structured SOAP clinical note (Subjective, Objective, Assessment, Plan) |
| `Dialogue` | Full doctor-patient conversation transcript |
| `ICD10` | ICD-10 diagnosis code (e.g. `N870`, `A047`) |
| `ICD10_desc` | Human-readable diagnosis name (e.g. `MILD CERVICAL DYSPLASIA`) |

### Data Cleaning (applied by both prep scripts)

1. **UTF normalization** -- NFKC normalization, removal of zero-width characters and control characters
2. **Leading formatting removal** -- strips markdown prefixes before the first `**` in SOAP notes
3. **Row 10 236 dropped** -- single abnormative record removed manually
4. **NA rows dropped**
5. **Underrepresented ICD codes removed** -- codes with fewer than 5 samples are excluded (removes 3 codes / 12 rows)
6. **Sorted** by `ICD10` then `Note`

### Train / Validation Split

Splitting is done per ICD-10 group: for each code, the **first** record (index 0) goes to validation, records at indices **1--4** go to training. This yields:

| Split | Records |
|-------|---------|
| Training | 8 136 |
| Validation | 2 034 |

### Data Formats

- **Embedding data** (`.json`): JSON array of objects with `Dialogue` and `ICD10` fields. The `Note` and `ICD10_desc` columns are dropped since the embedding model only classifies from dialogue text.
- **LLM data** (`.jsonl`): One JSON object per line in chat format (`messages` array with `system`, `user`, `assistant` roles). The system prompt instructs SOAP note generation; the assistant response contains the SOAP note plus `**ICD-10 Code:** ...` and `**Diagnosis:** ...`.

### Activation Data

`activationBase/activation_data.csv` contains a single example case (one row) for quick smoke-testing of inference pipelines.

---

## Pipeline Overview

```
MedSynth_huggingface_final.csv
        │
        ├── dataprep_embedding.py ──→ training_finetuning_embedding.json
        │                             validation_finetuning_embedding.json
        │
        └── dataprep_llm.py ────────→ training_finetuning_llm.jsonl
                                      validation_finetuning_llm.jsonl
        │                                       │
        │                                       │
        ▼                                       ▼
  embedding_finetuning.py              llm_finetuning.py
  (Gemma-300M + classifier)            (Ministral-3B + QLoRA)
        │                                       │
        ▼                                       ▼
  classify.py                          inference.py
  (inference & benchmark)              (interactive inference)
                                       benchmark_icd10_bf16.py
                                       (automated benchmark)
```

---

## Step 1 -- Data Preparation

### Embedding Data Prep

**Script:** `code/dataprep/dataprep_embedding.py`
**Notebook:** `code/ipynb_notebooks/cleaning_embedding.ipynb`

Reads `MedSynth_huggingface_final.csv`, cleans the data, drops the `Note` and `ICD10_desc` columns (not needed for classification), splits into train/val, and exports as JSON.

```bash
cd AI-Project/code/dataprep
python dataprep_embedding.py
```

**Outputs:**
- `train/training_finetuning_embedding.json`
- `validation/validation_finetuning_embedding.json`

### LLM Data Prep

**Script:** `code/dataprep/dataprep_llm.py`
**Notebook:** `code/ipynb_notebooks/cleaning_llm.ipynb`

Same cleaning pipeline, but retains all four columns. Formats each record into a chat-style JSONL with a system prompt instructing SOAP note generation, the dialogue as user input, and the SOAP note + ICD-10 code as the assistant response.

```bash
cd AI-Project/code/dataprep
python dataprep_llm.py
```

**Outputs:**
- `train/training_finetuning_llm.jsonl`
- `validation/validation_finetuning_llm.jsonl`

### Interactive Notebooks

The Jupyter notebooks `cleaning_embedding.ipynb` and `cleaning_llm.ipynb` are interactive, step-by-step versions of the above scripts. They contain the same logic with intermediate outputs visible in each cell. Use these for data exploration and debugging.

---

## Step 2 -- Training / Fine-Tuning

### Embedding Model Fine-Tuning

**Script:** `data/learningBase/embedding/embedding_finetuning.py`
**Base model:** `google/embedding-gemma-300m` (must be downloaded locally beforehand)
**Method:** Full fine-tuning of the embedding encoder with a linear classification head on top. Mean-pooling over token embeddings, cross-entropy loss, AdamW optimizer with linear warmup schedule.

**Key hyperparameters (defaults):**
- Epochs: 20
- Batch size: 32
- Learning rate: 2e-5
- Max sequence length: 512
- Warmup ratio: 10%

**How to run on the HPC cluster:**

```bash
cd data/learningBase/embedding

# 1. Build the Apptainer container image
bash build_embedding_finetuning_image.sh

# 2. Submit the SLURM job
sbatch run_embedding_finetuning.slurm
```

The SLURM job requests 3x A100 40 GB GPUs, 128 GB RAM, 16 CPUs, with a 4-hour time limit. It runs the training script inside the Apptainer container with these arguments:

```bash
python embedding_finetuning.py \
    --model_path ./embedding_in \
    --train_data ./training_finetuning_embedding.json \
    --val_data ./validation_finetuning_embedding.json \
    --output_dir ./embedding_out \
    --epochs 20 --batch_size 32 --lr 2e-5 --max_length 512
```

**Outputs (written to `--output_dir`):**
- `best_model.pt` -- best checkpoint (by validation accuracy)
- `encoder/` -- saved encoder weights + tokenizer for inference
- `label_encoder.pkl` -- sklearn LabelEncoder mapping ICD-10 codes to integer indices
- `wandb/` -- offline Weights & Biases logs

**Multi-GPU:** Uses `nn.DataParallel` automatically when multiple GPUs are detected.

### LLM Fine-Tuning

**Script:** `data/learningBase/llm/llm_finetuning.py`
**Base model:** Ministral-3B (Mistral architecture, must be downloaded locally)
**Method:** QLoRA -- 4-bit NF4 quantization with double quantization, LoRA adapters (rank 64, alpha 32) on all attention and MLP projection layers. Trained with SFTTrainer from the `trl` library.

**Key hyperparameters:**
- Epochs: 3
- Per-device batch size: 2 (with gradient accumulation of 4 = effective batch size 24 across 3 GPUs)
- Learning rate: 2e-4 with cosine scheduler
- Max sequence length: 4 096
- LoRA rank: 64, alpha: 32, dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**How to run on the HPC cluster:**

```bash
cd data/learningBase/llm

# 1. Create conda environment
conda create --name aibasllm --file ../requirements.txt
conda activate aibasllm

# 2. Login to Huggingface and download the base model
hf auth login --token INSERT_TOKEN
./download_model.sh

# 3. Build the Apptainer container image
bash build_llm_finetuning_image.sh

# 4. Submit the SLURM job
sbatch run_llm_finetuning.slurm
```

The SLURM job requests 3x A100 40 GB GPUs, 128 GB RAM, 32 CPUs, with a 4-hour time limit. It uses `accelerate launch --multi_gpu --num_processes 3` inside the container.

**Outputs (written to `llm_model_out/`):**
- `final_model/` -- LoRA adapter weights + tokenizer
- Checkpoints at every 100 steps (keeps last 3)
- `wandb/` -- offline Weights & Biases logs

**Syncing experiment tracking:**
```bash
wandb login
wandb sync --sync-all wandb/
```

**Alternative: Docker**
A `Dockerfile` is also provided for environments without Apptainer:
```bash
docker build -t llm-finetuning .
docker run --gpus all llm-finetuning
```

---

## Step 3 -- Inference & Benchmarking

### Embedding Inference

**Script:** `data/activationBase/embedding/classify.py`

Classify a single dialogue or evaluate the model on the full validation set.

**Single dialogue inference:**
```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
  apptainer exec --nv --bind $PWD:$PWD $PWD/embedding_finetune.sif \
  python $PWD/classify.py \
    --model_dir ./finetuned_model \
    --text "INSERT DIALOGUE TEXT HERE"
```

Returns the top-k (default 5) most likely ICD-10 codes with probabilities.

**Benchmark on validation set:**
```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
  apptainer exec --nv --bind $PWD:$PWD $PWD/embedding_finetune.sif \
  python $PWD/classify.py \
    --model_dir ./finetuned_model \
    --input ./validation_finetuning_embedding.json \
    --evaluate \
    --output embedding_benchmark_result.json
```

**CLI arguments:**

| Argument | Description |
|----------|-------------|
| `--model_dir` | Path to `finetuned_model/` directory (contains `best_model.pt`, `encoder/`, `label_encoder.pkl`) |
| `--text` | Single dialogue string to classify |
| `--input` | Path to JSON file with `Dialogue` entries, or a plain text file (one dialogue per line) |
| `--evaluate` | Enable evaluation mode (requires JSON input with `ICD10` ground-truth labels) |
| `--output` | Path to write results JSON |
| `--top_k` | Number of top predictions to return (default: 5) |
| `--max_length` | Max token length (default: 512) |
| `--batch_size` | Inference batch size (default: 32) |

**Evaluation metrics reported:** Top-1/3/5 accuracy, macro/weighted precision, recall, F1, per-class classification report.

**Existing benchmark result** (`embedding_benchmark_result.json.json`):
- 2 034 samples evaluated
- Top-1 accuracy: 23.55%
- Top-3 accuracy: 38.84%
- Top-5 accuracy: 45.87%

### LLM Inference

**Script:** `data/activationBase/llm/inference.py`

Interactive inference with the fine-tuned LoRA model. Loads the base Ministral-3B model in 4-bit quantization and applies the LoRA adapter on top.

```bash
srun --partition=gpu --gres=gpu:a100_40gb:1 --mem=64G --time=1:00:00 --pty \
  apptainer exec --nv --bind $PWD:$PWD container.sif \
  python inference.py
```

Accepts input via:
- **stdin:** prompts for pasting a consultation transcript
- **file argument:** `python inference.py path/to/transcript.txt`

Outputs a generated SOAP note with ICD-10 code and diagnosis.

**Configuration (hardcoded paths -- adjust before use):**
- `BASE_DIR = /work/hanken/aibas_ft2`
- `MODEL_DIR` -- path to base Ministral-3B weights
- `ADAPTER_DIR` -- path to LoRA adapter (`output/final_model`)

### LLM Benchmark

**Script:** `data/activationBase/llm/benchmark_icd10_bf16.py`

Automated evaluation of the LLM on the full validation set. Runs inference on every dialogue, extracts the ICD-10 code from the generated SOAP note using regex (`**ICD-10 Code:** <CODE>`), and compares against ground truth.

```bash
sbatch data/activationBase/llm/run_benchmark_bf16.slurm
```

SLURM job: 3x A100 40 GB, 128 GB RAM, 32 CPUs, 12-hour time limit. Supports **multi-GPU inference** by distributing samples across GPUs via `torch.multiprocessing`.

**Outputs (written to `results_bf16/`):**
- `benchmark_icd10_results.csv` -- per-sample predictions (ground truth, predicted, exact/block/chapter match flags)
- `benchmark_icd10_report.txt` -- full classification report with confusion matrix
- `benchmark_icd10_kpis.json` -- aggregate metrics

**Metrics computed:**
- Exact code match accuracy
- Block-level accuracy (first 3 characters of ICD-10 code)
- Chapter-level accuracy (first character)
- Micro/macro/weighted precision, recall, F1
- Balanced accuracy, Cohen's Kappa, Matthews Correlation Coefficient, Hamming loss
- Per-class classification report
- Confusion matrix (top 30 codes)
- Error analysis (most common misclassifications)

---

## Infrastructure & Containers

### Apptainer Definitions

| File | Base Image | Purpose |
|------|-----------|---------|
| `learningBase/llm/container_llm_finetuning.def` | `nvcr.io/nvidia/pytorch:24.07-py3` | LLM fine-tuning (transformers 4.57.6, peft 0.14, trl 0.13, bitsandbytes 0.45) |
| `learningBase/embedding/container_embedding_finetuning.def` | `nvidia/cuda:12.1.1-devel-ubuntu22.04` | Embedding fine-tuning (torch 2.2.2, transformers 4.51, scikit-learn 1.5.1) |

Build with:
```bash
apptainer build container.sif container_*.def
```

### Dockerfiles

| File | Purpose |
|------|---------|
| `learningBase/llm/Dockerfile` | Docker alternative for LLM training (same deps as the .def) |
| `activationBase/Dockerfile` | Minimal busybox container packaging `activation_data.csv` for deployment |

### SLURM Jobs

| File | GPUs | Time | Task |
|------|------|------|------|
| `learningBase/embedding/run_embedding_finetuning.slurm` | 3x A100 40 GB | 4 h | Embedding fine-tuning |
| `learningBase/llm/run_llm_finetuning.slurm` | 3x A100 40 GB | 4 h | LLM QLoRA fine-tuning |
| `activationBase/llm/run_training_bf16.slurm` | 3x A100 40 GB | 6 h | LLM bf16 training variant |
| `activationBase/llm/run_benchmark_bf16.slurm` | 3x A100 40 GB | 12 h | LLM benchmark evaluation |

### Conda Environment

`data/learningBase/requirements.txt` is a full conda environment spec for Linux-64 (Python 3.13, PyTorch 2.8, Transformers 4.57.1). Recreate with:
```bash
conda create --name aibasllm --file data/learningBase/requirements.txt
```

---

## File Reference

### Python Scripts

| File | Purpose | How to use |
|------|---------|------------|
| `code/dataprep/dataprep_embedding.py` | Clean raw CSV and produce embedding train/val JSON splits | `python dataprep_embedding.py` (run from its directory, expects `MedSynth_huggingface_final.csv` alongside) |
| `code/dataprep/dataprep_llm.py` | Clean raw CSV and produce LLM train/val JSONL splits | `python dataprep_llm.py` (same directory requirement) |
| `data/learningBase/embedding/embedding_finetuning.py` | Fine-tune Gemma-300M embedding model with classification head for ICD-10 prediction | Called by SLURM job; see CLI args `--model_path`, `--train_data`, `--val_data`, `--output_dir`, `--epochs`, `--batch_size`, `--lr` |
| `data/learningBase/llm/llm_finetuning.py` | Fine-tune Ministral-3B with QLoRA for SOAP note generation | Called by SLURM job via `accelerate launch`; paths configured via environment variables |
| `data/activationBase/embedding/classify.py` | Inference and benchmark for the embedding model | `python classify.py --model_dir ./finetuned_model --text "..."` or `--input file.json --evaluate` |
| `data/activationBase/llm/inference.py` | Interactive LLM inference (stdin or file input) | `python inference.py [optional_file]` |
| `data/activationBase/llm/benchmark_icd10_bf16.py` | Full LLM benchmark with multi-GPU support | Called by SLURM job; reads `validation_finetuning_llm.jsonl`, writes results to `results_bf16/` |

### Jupyter Notebooks

| File | Purpose |
|------|---------|
| `code/ipynb_notebooks/cleaning_embedding.ipynb` | Interactive data exploration and cleaning for the embedding track; same logic as `dataprep_embedding.py` with visible intermediate outputs |
| `code/ipynb_notebooks/cleaning_llm.ipynb` | Interactive data exploration and cleaning for the LLM track; same logic as `dataprep_llm.py` with visible intermediate outputs |

### Data Files

| File | Format | Size | Description |
|------|--------|------|-------------|
| `code/dataprep/MedSynth_huggingface_final.csv` | CSV | 76 MB | Raw source dataset (10 240 rows: Note, Dialogue, ICD10, ICD10_desc) |
| `data/activationBase/activation_data.csv` | CSV | 8 KB | Single example case for smoke-testing |
| `data/learningBase/embedding/training_finetuning_embedding.json` | JSON | 36 MB | Embedding training set (8 136 records: Dialogue + ICD10) |
| `data/learningBase/embedding/validation_finetuning_embedding.json` | JSON | 8.8 MB | Embedding validation set (2 034 records) |
| `data/activationBase/embedding/validation_finetuning_embedding.json` | JSON | 8.8 MB | Copy of embedding validation set for benchmarking |
| `data/activationBase/embedding/embedding_benchmark_result.json.json` | JSON | 1.4 MB | Embedding benchmark results (accuracy, per-sample predictions) |
| `data/learningBase/llm/training_finetuning_llm.jsonl` | JSONL | 67 MB | LLM training set (8 136 records: system/user/assistant messages) |
| `data/learningBase/llm/validation_finetuning_llm.jsonl` | JSONL | 17 MB | LLM validation set (2 034 records) |
| `data/activationBase/llm/validation_finetuning_llm.jsonl` | JSONL | 17 MB | Copy of LLM validation set for benchmarking |
