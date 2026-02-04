# codeBase_icd10_diagnosis

## Ownership

**Authors:** Julia Jellinek and Keno Hanken

## Course Context

This Docker image was created as part of the course **"M. Grum: Advanced AI-based Application Systems"** offered by the Junior Chair for Business Information Science, esp. AI-based Application Systems, at the University of Potsdam.

## AI Model Characterization

This image contains the inference and benchmarking code for the ICD-10 diagnosis prediction system. The code supports two model tracks:

### LLM Inference (`/tmp/codeBase/llm/`)

- **inference.py** - Interactive inference script for the fine-tuned Ministral-3B model
  - Loads base model with 4-bit quantization
  - Applies LoRA adapters from knowledgeBase
  - Accepts dialogue input via stdin or file
  - Outputs generated SOAP note with ICD-10 diagnosis

- **benchmark_icd10_bf16.py** - Automated benchmark script
  - Evaluates model on full validation set
  - Computes exact match, block-level, and chapter-level accuracy
  - Generates classification report and confusion matrix
  - Supports multi-GPU inference

### Embedding Inference (`/tmp/codeBase/embedding/`)

- **classify.py** - Inference and evaluation script for the fine-tuned Gemma-300M embedding model
  - Single dialogue classification with top-k predictions
  - Batch evaluation on validation set
  - Computes Top-1/3/5 accuracy, precision, recall, F1

## Directory Structure

Code is stored in the image at `/data/` and copied to `/tmp/` at runtime via docker-compose:

```
/data/codeBase/  (image)  -->  /tmp/codeBase/  (runtime)
├── llm/
│   ├── inference.py           (interactive LLM inference)
│   └── benchmark_icd10_bf16.py (automated LLM benchmark)
├── embedding/
│   └── classify.py            (embedding inference & benchmark)
└── README.md
```

## Runtime Requirements

These scripts require a Python environment with the following dependencies:
- PyTorch 2.x with CUDA support
- transformers >= 4.57
- peft >= 0.14
- bitsandbytes >= 0.45
- accelerate >= 1.7
- scikit-learn >= 1.5

Recommended: Use the provided Apptainer/Docker runtime containers for execution.

## Usage Examples

### LLM Inference
```bash
python /tmp/codeBase/llm/inference.py \
    --model_dir /tmp/knowledgeBase/base_model \
    --adapter_dir /tmp/knowledgeBase/llm_lora_adapter
# Then paste dialogue text when prompted
```

### LLM Benchmark
```bash
python /tmp/codeBase/llm/benchmark_icd10_bf16.py \
    --model_dir /tmp/knowledgeBase/base_model \
    --adapter_dir /tmp/knowledgeBase/llm_lora_adapter \
    --validation_file /tmp/learningBase/validation/llm/validation_finetuning_llm.jsonl \
    --results_dir /tmp/results_bf16
```

### Embedding Classification
```bash
python /tmp/codeBase/embedding/classify.py \
    --model_dir /tmp/knowledgeBase/embedding_model \
    --text "Doctor-patient dialogue text here..."
```

### Embedding Benchmark
```bash
python /tmp/codeBase/embedding/classify.py \
    --model_dir /tmp/knowledgeBase/embedding_model \
    --input /tmp/learningBase/validation/embedding/validation_finetuning_embedding.json \
    --evaluate \
    --output results.json
```

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. By using this image, you agree to comply with the terms and conditions of this license.
