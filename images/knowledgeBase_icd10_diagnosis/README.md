# knowledgeBase_icd10_diagnosis

## Ownership

**Authors:** Julia Jellinek and Keno Hanken

## Course Context

This Docker image was created as part of the course **"M. Grum: Advanced AI-based Application Systems"** offered by the Junior Chair for Business Information Science, esp. AI-based Application Systems, at the University of Potsdam.

## AI Model Characterization

This image contains the trained AI models for ICD-10 diagnosis prediction from doctor-patient consultation dialogues. Two complementary approaches are provided:

### 1. LLM Model (Ministral-3B with QLoRA)

- **Base Model:** Ministral-3B (Mistral architecture)
- **Fine-tuning Method:** QLoRA (4-bit NF4 quantization with LoRA adapters)
- **Task:** Generate SOAP notes with ICD-10 diagnosis codes from dialogue transcripts
- **LoRA Configuration:**
  - Rank: 64
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**Important:** This image contains only the LoRA adapter weights (~100-500 MB). The base Ministral-3B model (~6 GB) must be downloaded separately from HuggingFace:
```bash
huggingface-cli download mistralai/Ministral-3B-Instruct-2410
```

### 2. Embedding Model (Gemma-300M)

- **Base Model:** google/embedding-gemma-300m
- **Fine-tuning Method:** Full fine-tuning with classification head
- **Task:** Direct ICD-10 code classification from dialogue transcripts
- **Output:** Top-k ICD-10 predictions with probability scores

## Directory Structure

Data is stored in the image at `/data/` and copied to `/tmp/` at runtime via docker-compose:

```
/data/knowledgeBase/  (image)  -->  /tmp/knowledgeBase/  (runtime)
├── llm_lora_adapter/
│   ├── adapter_model.safetensors    (LoRA weights)
│   ├── adapter_config.json          (LoRA configuration)
│   ├── tokenizer.json               (Tokenizer)
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── embedding_model/
│   ├── best_model.pt                (Fine-tuned model checkpoint)
│   ├── label_encoder.pkl            (ICD-10 code mapping)
│   └── encoder/                     (Encoder weights + tokenizer)
└── README.md
```

## Performance Metrics

### LLM Model
- Generates structured SOAP notes with ICD-10 codes
- Evaluated on exact code match, block-level (3-char), and chapter-level (1-char) accuracy

### Embedding Model (Benchmark on 2,034 validation samples)
- Top-1 Accuracy: 23.55%
- Top-3 Accuracy: 38.84%
- Top-5 Accuracy: 45.87%

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. By using this image, you agree to comply with the terms and conditions of this license.
