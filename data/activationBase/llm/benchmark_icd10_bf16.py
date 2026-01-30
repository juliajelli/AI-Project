"""
Benchmark the bf16 fine-tuned model on ICD-10 code prediction using the validation dataset.

Loads every dialogue from the validation JSONL, runs inference, extracts the
predicted ICD-10 code, and compares it against the ground-truth code.

Outputs:
  - Per-sample results CSV  (results_bf16/benchmark_icd10_results.csv)
  - Full classification report, confusion matrix, and aggregate metrics to
    stdout and a text file   (results_bf16/benchmark_icd10_report.txt)
  - JSON KPIs               (results_bf16/benchmark_icd10_kpis.json)
"""

import os
import re
import json
import csv
import torch
import torch.multiprocessing as mp
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    hamming_loss,
)
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
MODEL_DIR = os.path.join(BASE_DIR, "model")
ADAPTER_DIR = os.path.join(BASE_DIR, "output_bf16", "final_model")
DATA_DIR = os.path.join(BASE_DIR, "data")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation_finetuning_llm.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results_bf16")

os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── register custom model configs (same as inference.py) ────────────────────
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

# ── regex for extracting ICD-10 codes ──────────────────────────────────────
ICD10_PATTERN = re.compile(r"\*\*ICD-10 Code:\*\*\s*([A-Z][A-Z0-9]+)")


def extract_icd10(text: str) -> str | None:
    """Return the first ICD-10 code found in *text*, or None."""
    m = ICD10_PATTERN.search(text)
    return m.group(1) if m else None


# ── load validation data ────────────────────────────────────────────────────
def load_validation_entries(path: str) -> list[dict]:
    """Parse the multi-line-pretty-printed JSONL file into a list of entries.

    Works regardless of line-ending style by tracking brace depth line-by-line
    and joining lines that belong to the same top-level JSON object.
    """
    entries = []
    buf: list[str] = []
    depth = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Count unescaped braces outside of JSON string values.
            in_string = False
            escape = False
            for ch in line:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            buf.append(line)
            if depth == 0 and buf:
                text = "".join(buf).strip()
                if text:
                    entries.append(json.loads(text))
                buf = []
    return entries


# ── load model (bf16, no quantisation) ─────────────────────────────────────
NUM_GPUS = torch.cuda.device_count() or 1


def load_model_and_tokenizer(gpu_id: int = 0):
    """Load a model replica pinned to a specific GPU."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, dialogue: str, max_new_tokens: int = 2048) -> str:
    messages = [{"role": "user", "content": dialogue}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── multi-GPU worker ────────────────────────────────────────────────────────
def _worker(gpu_id: int, work_items: list[tuple[int, str, str]], result_queue: mp.Queue):
    """Worker process: loads its own model on gpu_id, processes its shard."""
    model, tokenizer = load_model_and_tokenizer(gpu_id)
    for idx, dialogue, gt_code in work_items:
        pred_text = generate_response(model, tokenizer, dialogue)
        pred_code = extract_icd10(pred_text)
        result_queue.put((idx, gt_code, pred_code))
    # signal done
    result_queue.put(None)


# ── ICD-10 category helpers ─────────────────────────────────────────────────
def icd10_chapter(code: str) -> str:
    """Return the ICD-10 letter prefix (chapter-level), e.g. 'A' from 'A047'."""
    return code[0] if code else ""


def icd10_block(code: str) -> str:
    """Return the 3-character block, e.g. 'A04' from 'A047'."""
    return code[:3] if code and len(code) >= 3 else code or ""


# ── main benchmark ──────────────────────────────────────────────────────────
def main():
    print("Loading validation data …")
    entries = load_validation_entries(VALIDATION_FILE)
    print(f"  {len(entries)} entries loaded.")

    # Prepare work items: (index, dialogue, ground_truth_code)
    work_items: list[tuple[int, str, str]] = []
    skipped = 0
    for i, entry in enumerate(entries):
        msgs = entry["messages"]
        dialogue = next(m["content"] for m in msgs if m["role"] == "user")
        gt_text = next(m["content"] for m in msgs if m["role"] == "assistant")
        gt_code = extract_icd10(gt_text)
        if gt_code is None:
            print(f"  [WARN] No ground-truth ICD-10 found in entry {i}, skipping.")
            skipped += 1
            continue
        work_items.append((i, dialogue, gt_code))

    if skipped:
        print(f"  Skipped {skipped} entries without ground-truth codes.")

    num_gpus = NUM_GPUS
    print(f"Using {num_gpus} GPU(s) for inference …")

    # Split work items into per-GPU shards (round-robin for balanced load)
    shards: list[list[tuple[int, str, str]]] = [[] for _ in range(num_gpus)]
    for j, item in enumerate(work_items):
        shards[j % num_gpus].append(item)

    # Launch one worker process per GPU, collect results via a shared queue
    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue = mp.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=_worker, args=(gpu_id, shards[gpu_id], result_queue))
        p.start()
        processes.append(p)

    # Collect results with a progress bar
    results_by_idx: dict[int, tuple[str, str | None]] = {}
    done_count = 0
    with tqdm(total=len(work_items), desc="Benchmark (bf16)") as pbar:
        while done_count < num_gpus:
            item = result_queue.get()
            if item is None:
                done_count += 1
                continue
            idx, gt_code, pred_code = item
            results_by_idx[idx] = (gt_code, pred_code)
            pbar.update(1)

    for p in processes:
        p.join()

    # Build ordered output lists
    y_true = []
    y_pred = []
    rows = []
    for idx in sorted(results_by_idx.keys()):
        gt_code, pred_code = results_by_idx[idx]
        pred = pred_code if pred_code else "NONE"
        y_true.append(gt_code)
        y_pred.append(pred)
        rows.append({
            "index": idx,
            "ground_truth": gt_code,
            "predicted": pred,
            "exact_match": int(gt_code == (pred_code or "")),
            "block_match": int(icd10_block(gt_code) == icd10_block(pred_code or "")),
            "chapter_match": int(icd10_chapter(gt_code) == icd10_chapter(pred_code or "")),
        })

    # ── save per-sample CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "benchmark_icd10_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample results saved to {csv_path}")

    # ── compute metrics ─────────────────────────────────────────────────────
    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    n = len(y_true)
    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    block = sum(1 for t, p in zip(y_true, y_pred) if icd10_block(t) == icd10_block(p))
    chapter = sum(1 for t, p in zip(y_true, y_pred) if icd10_chapter(t) == icd10_chapter(p))
    no_code = sum(1 for p in y_pred if p == "NONE")

    log("=" * 70)
    log("ICD-10 BENCHMARK RESULTS (bf16)")
    log("=" * 70)
    log(f"Total samples evaluated:    {n}")
    log(f"No ICD-10 code extracted:   {no_code}  ({no_code/n*100:.1f}%)")
    log()

    # ── Accuracy at different granularities ──────────────────────────────
    log("--- Accuracy at different granularities ---")
    log(f"Exact code match accuracy:  {exact/n*100:.2f}%  ({exact}/{n})")
    log(f"Block-level match (3-char): {block/n*100:.2f}%  ({block}/{n})")
    log(f"Chapter-level match (1-ch): {chapter/n*100:.2f}%  ({chapter}/{n})")
    log()

    # ── Standard classification metrics (macro / micro / weighted) ───────
    labels = sorted(set(y_true + y_pred))

    log("--- Aggregate Classification Metrics ---")
    for avg in ("micro", "macro", "weighted"):
        p = precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        r = recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        f = f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        log(f"  [{avg:>8s}]  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
    log()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)

    log(f"Accuracy (sklearn):         {acc:.4f}")
    log(f"Balanced accuracy:          {bal_acc:.4f}")
    log(f"Cohen's Kappa:              {kappa:.4f}")
    log(f"Matthews Corr. Coeff (MCC): {mcc:.4f}")
    log(f"Hamming loss:               {h_loss:.4f}")
    log()

    # ── Per-class classification report ──────────────────────────────────
    log("--- Per-class Classification Report ---")
    log(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # ── Confusion matrix ─────────────────────────────────────────────────
    log("--- Confusion Matrix (top 30 most frequent codes) ---")
    code_counts = Counter(y_true)
    top_codes = [c for c, _ in code_counts.most_common(30)]
    cm = confusion_matrix(y_true, y_pred, labels=top_codes)
    header = "         " + " ".join(f"{c:>8s}" for c in top_codes)
    log(header)
    for label, row in zip(top_codes, cm):
        log(f"{label:>8s} " + " ".join(f"{v:>8d}" for v in row))
    log()

    # ── Block-level metrics ──────────────────────────────────────────────
    y_true_block = [icd10_block(c) for c in y_true]
    y_pred_block = [icd10_block(c) for c in y_pred]
    block_labels = sorted(set(y_true_block + y_pred_block))

    log("--- Block-level (3-char) Classification Metrics ---")
    for avg in ("micro", "macro", "weighted"):
        p = precision_score(y_true_block, y_pred_block, labels=block_labels, average=avg, zero_division=0)
        r = recall_score(y_true_block, y_pred_block, labels=block_labels, average=avg, zero_division=0)
        f = f1_score(y_true_block, y_pred_block, labels=block_labels, average=avg, zero_division=0)
        log(f"  [{avg:>8s}]  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
    log(f"  Block accuracy:           {accuracy_score(y_true_block, y_pred_block):.4f}")
    log(f"  Block Cohen's Kappa:      {cohen_kappa_score(y_true_block, y_pred_block):.4f}")
    log(f"  Block MCC:                {matthews_corrcoef(y_true_block, y_pred_block):.4f}")
    log()

    # ── Chapter-level metrics ────────────────────────────────────────────
    y_true_ch = [icd10_chapter(c) for c in y_true]
    y_pred_ch = [icd10_chapter(c) for c in y_pred]
    ch_labels = sorted(set(y_true_ch + y_pred_ch))

    log("--- Chapter-level (1-char) Classification Metrics ---")
    for avg in ("micro", "macro", "weighted"):
        p = precision_score(y_true_ch, y_pred_ch, labels=ch_labels, average=avg, zero_division=0)
        r = recall_score(y_true_ch, y_pred_ch, labels=ch_labels, average=avg, zero_division=0)
        f = f1_score(y_true_ch, y_pred_ch, labels=ch_labels, average=avg, zero_division=0)
        log(f"  [{avg:>8s}]  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
    log(f"  Chapter accuracy:         {accuracy_score(y_true_ch, y_pred_ch):.4f}")
    log(f"  Chapter Cohen's Kappa:    {cohen_kappa_score(y_true_ch, y_pred_ch):.4f}")
    log()

    # ── Error analysis summary ───────────────────────────────────────────
    log("--- Error Analysis ---")
    mismatches = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    log(f"Total mismatches:           {len(mismatches)}")
    if mismatches:
        log("Most common (true -> pred) mismatches:")
        for (t, p), cnt in Counter(mismatches).most_common(20):
            log(f"  {t} -> {p}  ({cnt}x)")
    log()

    # ── save JSON KPIs ────────────────────────────────────────────────────
    kpi = {
        "total_samples": n,
        "correct": exact,
        "incorrect": n - exact,
        "no_code_extracted": no_code,
        "top1_accuracy": round(exact / n, 4),
        "macro_precision": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "micro_precision": round(precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0), 4),
        "micro_recall": round(recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0), 4),
        "micro_f1": round(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0), 4),
        "balanced_accuracy": round(bal_acc, 4),
        "cohen_kappa": round(kappa, 4),
        "matthews_corrcoef": round(mcc, 4),
        "hamming_loss": round(h_loss, 4),
        "block_accuracy": round(block / n, 4),
        "chapter_accuracy": round(chapter / n, 4),
    }

    json_path = os.path.join(RESULTS_DIR, "benchmark_icd10_kpis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kpi, f, indent=2)
    print(f"KPIs saved to {json_path}")

    # ── save report ──────────────────────────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, "benchmark_icd10_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Full report saved to {report_path}")


if __name__ == "__main__":
    main()
