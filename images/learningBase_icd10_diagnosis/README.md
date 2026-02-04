# learningBase_icd10_diagnosis

## Ownership

**Authors:** Julia Jellinek and Keno Hanken

## Course Context

This Docker image was created as part of the course **"M. Grum: Advanced AI-based Application Systems"** offered by the Junior Chair for Business Information Science, esp. AI-based Application Systems, at the University of Potsdam.

## Data Origin

The training and validation data contained in this image originates from the **MedSynth** dataset, scraped from [https://huggingface.co/datasets/Ahmad0067/MedSynth](https://huggingface.co/datasets/Ahmad0067/MedSynth).

The data consists of synthetic doctor-patient consultation dialogues with associated ICD-10 diagnosis codes, prepared for training two complementary models:
- **LLM track:** JSONL format with chat-style messages for SOAP note generation
- **Embedding track:** JSON format with dialogue-to-ICD10 classification pairs

### Data Split
- **Training set:** 8,136 records (indices 1-4 per ICD-10 code)
- **Validation set:** 2,034 records (index 0 per ICD-10 code)

## Directory Structure

Data is stored in the image at `/data/` and copied to `/tmp/` at runtime via docker-compose:

```
/data/learningBase/  (image)  -->  /tmp/learningBase/  (runtime)
├── train/
│   ├── llm/
│   │   └── training_finetuning_llm.jsonl      (67 MB, 8,136 records)
│   └── embedding/
│       └── training_finetuning_embedding.json (36 MB, 8,136 records)
├── validation/
│   ├── llm/
│   │   └── validation_finetuning_llm.jsonl    (17 MB, 2,034 records)
│   └── embedding/
│       └── validation_finetuning_embedding.json (8.8 MB, 2,034 records)
└── README.md
```

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. By using this image, you agree to comply with the terms and conditions of this license.
