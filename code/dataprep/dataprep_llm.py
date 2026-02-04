import json
import os
import sys
import re
import unicodedata
import pandas as pd
from pathlib import Path
from typing import List, Dict
import argparse

df = pd.read_csv('joint_data_collection.csv')


# Harmonizing UTF characters
def clean_string(s):
    if not isinstance(s, str):
        return s

    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


df[["Note", "Dialogue"]] = df[["Note", "Dialogue"]].map(clean_string)

# Erasing leading styling characters
s = df["Note"].astype("string")
df["Note"] = s.apply(lambda x: x[x.find("**"):] if isinstance(x, str) and "**" in x else x)

# Drop NA (and one abnormative) values
df.isna().sum()
df = df.drop([10236])
df = df.dropna()
df = df.sort_values(["ICD10", "Note"])
df.reset_index(inplace=True)
df = df.drop(["index"], axis=1)

# Deleting rows with underrepresented ICD codes
df = df[df['ICD10'].map(df['ICD10'].value_counts()) >= 5]

# Train test split
g = df.groupby('ICD10')
train_df = g.nth([1,2,3,4]).reset_index()
train_df = train_df.drop(["index"], axis=1)
val_df  = g.nth(0).reset_index()
val_df = val_df.drop(["index"], axis=1)


# Preparing data for LLM finetuning
def format_training_example(dialogue: str, note: str, icd10: str, icd10_desc: str):
    system_prompt = """You are a medical documentation assistant. Your task is to convert patient-doctor consultation dialogues into structured SOAP notes (Subjective, Objective, Assessment, Plan) with appropriate ICD-10 diagnosis codes.

Generate a comprehensive SOAP note that includes:
1. Subjective: Chief complaint, history of present illness, review of systems
2. Objective: Vital signs, physical examination findings
3. Assessment: Diagnosis with ICD-10 code and description, differential diagnoses
4. Plan: Management, referrals, further testing, patient education

Stick to the following rules with absolute authority:
- Do not include anything into the SOAP note that is not present in the presented dialogue.
- Do not assume anything. Be deterministic and only take what is named in the text.
- If you cannot fill out something in the SOAP notes, write only [UNKNOWN] to the corresponding dimension or subdimension.
"""

    user_prompt = f"""Convert the following patient-doctor consultation dialogue into a structured SOAP note:

{dialogue}"""

    # Include ICD10 in the assistant response
    assistant_response = f"""{note}

**ICD-10 Code:** {icd10}
**Diagnosis:** {icd10_desc}"""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def process_json_file(input_dataframe: object, output_file_name: str):
    data_records = []
    for idx in range(len(input_dataframe)):
        dialogue = input_dataframe['Dialogue'][idx]
        note = input_dataframe['Note'][idx]
        icd10 = input_dataframe['ICD10'][idx]
        icd10_desc = input_dataframe['ICD10_desc'][idx]

        formatted_example = format_training_example(dialogue, note, icd10, icd10_desc)
        data_records.append(formatted_example)

    # Save to JSONL format
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for record in data_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    pass


process_json_file(train_df, "train/training_data_llm.jsonl")
process_json_file(val_df, "validation/validation_data_llm.jsonl")