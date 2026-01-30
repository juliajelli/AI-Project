import json
import os
import sys
import re
import unicodedata
import pandas as pd

df = pd.read_csv('MedSynth_huggingface_final.csv')

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

# Deleting unneccessary columns
df = df.drop(["Note", "ICD10_desc"], axis=1)

# Train test split
g = df.groupby('ICD10')
train_df = g.nth([1,2,3,4]).reset_index()
train_df = train_df.drop(["index"], axis=1)
val_df  = g.nth(0).reset_index()
val_df = val_df.drop(["index"], axis=1)

# Export as JSON
train_df.to_json("train/training_finetuning_embedding.json", orient="records")
val_df.to_json("validation/validation_finetuning_embedding.json", orient="records")