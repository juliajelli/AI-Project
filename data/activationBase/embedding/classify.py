"""
Inference script: classify dialogues using the finetuned embedding model.

Docker paths:
  - Model directory: /tmp/knowledgeBase/embedding_model
  - Validation data: /tmp/learningBase/validation/embedding/validation_finetuning_embedding.json
"""

import argparse
import csv
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, top_k_accuracy_score


class EmbeddingClassifier(nn.Module):
    """Embedding model + classification head."""

    def __init__(self, encoder, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token embeddings
        token_embeddings = outputs.last_hidden_state  # (B, seq_len, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        embeddings = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)
        return logits, embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to finetuned_model directory")
    parser.add_argument("--input", type=str, default=None, help="JSON/CSV file with Dialogue entries or plain text file")
    parser.add_argument("--text", type=str, default=None, help="Direct dialogue text to classify")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for predictions")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on a JSON dataset with ground truth ICD10 labels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label encoder
    with open(f"{args.model_dir}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load model
    checkpoint = torch.load(f"{args.model_dir}/best_model.pt", map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_dir}/encoder", local_files_only=True, use_fast=False)
    encoder = AutoModel.from_pretrained(f"{args.model_dir}/encoder", local_files_only=True)

    model = EmbeddingClassifier(encoder, checkpoint["hidden_size"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load input
    if args.text:
        dialogues = [args.text]
        ground_truth = None
    elif args.input:
        if args.input.endswith(".json"):
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
            dialogues = [d["Dialogue"] for d in data]
            if args.evaluate:
                ground_truth = [d["ICD10"] for d in data]
            else:
                ground_truth = None
        elif args.input.endswith(".csv"):
            # CSV format: Note,Dialogue,ICD10,ICD10_desc
            with open(args.input, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            if not data:
                parser.error(f"CSV file {args.input} is empty.")
            dialogues = [d["Dialogue"] for d in data]
            if args.evaluate and "ICD10" in data[0]:
                ground_truth = [d["ICD10"] for d in data]
            else:
                ground_truth = None
        else:
            # Plain text file - one dialogue per line
            with open(args.input, "r", encoding="utf-8") as f:
                dialogues = [line.strip() for line in f if line.strip()]
            ground_truth = None
    else:
        parser.error("Either --input or --text must be provided.")

    if args.evaluate and ground_truth is None:
        parser.error("--evaluate requires a JSON/CSV --input file with 'ICD10' ground truth labels.")

    # Predict in batches
    all_probs = []
    all_preds = []
    results = []
    for i in range(0, len(dialogues), args.batch_size):
        batch_texts = dialogues[i : i + args.batch_size]
        encoding = tokenizer(
            batch_texts, max_length=args.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = probs.topk(args.top_k, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.extend(probs.argmax(dim=-1).cpu().tolist())

        for j in range(len(batch_texts)):
            preds = []
            for k in range(args.top_k):
                idx = top_k_indices[j, k].item()
                prob = top_k_probs[j, k].item()
                icd_code = label_encoder.inverse_transform([idx])[0]
                preds.append({"icd10": icd_code, "probability": round(prob, 4)})
            results.append({"dialogue": batch_texts[j][:200] + "...", "predictions": preds})

    # Evaluation mode
    if args.evaluate:
        all_probs = np.concatenate(all_probs, axis=0)
        y_true = label_encoder.transform(ground_truth)
        y_pred = np.array(all_preds)

        accuracy = accuracy_score(y_true, y_pred)
        top3_acc = top_k_accuracy_score(y_true, all_probs, k=3, labels=np.arange(all_probs.shape[1]))
        top5_acc = top_k_accuracy_score(y_true, all_probs, k=5, labels=np.arange(all_probs.shape[1]))
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        correct = int((y_true == y_pred).sum())
        incorrect = len(y_true) - correct

        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total samples:       {len(y_true)}")
        print(f"Correct predictions: {correct}")
        print(f"Incorrect predictions: {incorrect}")
        print("-" * 60)
        print(f"Top-1 Accuracy:      {accuracy:.4f}")
        print(f"Top-3 Accuracy:      {top3_acc:.4f}")
        print(f"Top-5 Accuracy:      {top5_acc:.4f}")
        print(f"Macro Precision:     {macro_precision:.4f}")
        print(f"Macro Recall:        {macro_recall:.4f}")
        print(f"Macro F1-Score:      {macro_f1:.4f}")
        print(f"Weighted F1-Score:   {weighted_f1:.4f}")
        print("=" * 60)
        print("\nPer-Class Classification Report:\n")
        target_names = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

        if args.output:
            eval_results = {
                "total_samples": len(y_true),
                "correct": correct,
                "incorrect": incorrect,
                "top1_accuracy": round(accuracy, 4),
                "top3_accuracy": round(top3_acc, 4),
                "top5_accuracy": round(top5_acc, 4),
                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "predictions": results,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            print(f"\nFull results saved to {args.output}")
    else:
        # Standard inference output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Predictions saved to {args.output}")
        else:
            for r in results[:10]:
                print(f"\nDialogue: {r['dialogue'][:100]}...")
                for p in r["predictions"]:
                    print(f"  {p['icd10']}: {p['probability']:.4f}")


if __name__ == "__main__":
    main()
