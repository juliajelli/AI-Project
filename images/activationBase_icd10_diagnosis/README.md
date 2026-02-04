# activationBase_icd10_diagnosis

## Ownership

**Authors:** Julia Jellinek and Keno Hanken

## Course Context

This Docker image was created as part of the course **"M. Grum: Advanced AI-based Application Systems"** offered by the Junior Chair for Business Information Science, esp. AI-based Application Systems, at the University of Potsdam.

## Data Origin

The activation data contained in this image originates from the **MedSynth** dataset, scraped from [https://huggingface.co/datasets/Ahmad0067/MedSynth](https://huggingface.co/datasets/Ahmad0067/MedSynth).

## Contents

This image contains a single example case (`activation_data.csv`) for smoke-testing the ICD-10 diagnosis inference pipelines. The example case includes:

| Column | Description |
|--------|-------------|
| `Note` | SOAP clinical note (Subjective, Objective, Assessment, Plan) |
| `Dialogue` | Doctor-patient consultation transcript |
| `ICD10` | Ground-truth ICD-10 diagnosis code (N870) |
| `ICD10_desc` | Diagnosis description (MILD CERVICAL DYSPLASIA) |

## Directory Structure

Data is stored in the image at `/data/` and copied to `/tmp/` at runtime via docker-compose:

```
/data/activationBase/  (image)  -->  /tmp/activationBase/  (runtime)
├── activation_data.csv   (single example case)
└── README.md
```

## Usage

This activation data is used to test the trained models:
- **LLM inference:** Input the dialogue, receive generated SOAP note with ICD-10 prediction
- **Embedding inference:** Input the dialogue, receive top-k ICD-10 code predictions with probabilities

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. By using this image, you agree to comply with the terms and conditions of this license.
