# Dataset credits

Picture material is coming from the following repositories:
Original data is coming from Huggingface repository [Ahmad0067](https://huggingface.co/datasets/Ahmad0067/MedSynth) under `unknown` license.

# Code ownership by

**Owner:** Julia Jellinek and Keno Hanken 
**Course:** M. Grum – Advanced AI-based Application Systems  
**Institution:** Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam  
**Data Origin:** scraped from https://huggingface.co/datasets/Ahmad0067/MedSynth
**License:** AGPL-3.0

# Description

## Code

The `code/` directory contains all data preparation and preprocessing logic for transforming the raw MedSynth dataset into training-ready formats for both model tracks.

### Structure
code/
├── dataprep/
│   ├── MedSynth_huggingface_final.csv
│   ├── dataprep_embedding.py
│   └── dataprep_llm.py
└── ipynb_notebooks/
├── MedSynth_huggingface_final.csv
├── cleaning_embedding.ipynb
└── cleaning_llm.ipynb

### dataprep/

Production-ready scripts that clean the raw CSV and produce the final train/validation splits.

- **MedSynth_huggingface_final.csv** -- Raw source dataset (76 MB, 10 240 rows). Contains four columns: `Note` (SOAP clinical note), `Dialogue` (doctor-patient transcript), `ICD10` (diagnosis code), `ICD10_desc` (diagnosis description).
- **dataprep_embedding.py** -- Cleans the CSV (UTF normalization, formatting fixes, NA removal, drops underrepresented ICD codes with < 5 samples), removes the `Note` and `ICD10_desc` columns, splits by ICD-10 group (index 0 → validation, indices 1-4 → training), and exports as JSON. Run with `python dataprep_embedding.py` from within the directory.
- **dataprep_llm.py** -- Same cleaning pipeline but retains all four columns. Formats each record into chat-style JSONL with a system prompt instructing SOAP note generation, the dialogue as user input, and the SOAP note + ICD-10 code as assistant response. Run with `python dataprep_llm.py` from within the directory.

### ipynb_notebooks/

Interactive Jupyter notebook versions of the dataprep scripts. They contain the same logic with cell-by-cell outputs visible for data exploration and debugging.

- **MedSynth_huggingface_final.csv** -- Copy of the raw dataset for notebook use.
- **cleaning_embedding.ipynb** -- Interactive version of `dataprep_embedding.py`.
- **cleaning_llm.ipynb** -- Interactive version of `dataprep_llm.py`.