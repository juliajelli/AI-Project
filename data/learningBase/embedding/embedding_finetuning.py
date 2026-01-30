"""
Finetune google/embedding-gemma-300m for ICD-10 classification.

Strategy: Fine-tune the embedding model with a classification head on top.
Uses multi-GPU training via Accelerate/DataParallel.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import wandb
from tqdm import tqdm
import pickle


class ICDDataset(Dataset):
    def __init__(self, dialogues, labels, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.dialogues[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


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


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogues = [d["Dialogue"] for d in data]
    labels = [d["ICD10"] for d in data]
    return dialogues, labels


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            all_logits.append(logits.cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)

    acc = accuracy_score(all_labels_arr, all_preds_arr)
    avg_loss = total_loss / len(all_labels_arr)

    # Top-5 accuracy
    top5_acc = top_k_accuracy_score(
        all_labels_arr, all_logits, k=min(5, num_classes), labels=np.arange(num_classes)
    )

    return avg_loss, acc, top5_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to local embedding-gemma-300m model")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./finetuned_model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="icd10-embedding-finetune")
    args = parser.parse_args()

    # Offline wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.wandb_project, config=vars(args))

    # Load data
    print("Loading data...")
    train_dialogues, train_labels_raw = load_data(args.train_data)
    val_dialogues, val_labels_raw = load_data(args.val_data)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels_raw + val_labels_raw)
    train_labels = label_encoder.transform(train_labels_raw)
    val_labels = label_encoder.transform(val_labels_raw)
    num_classes = len(label_encoder.classes_)
    print(f"Number of ICD-10 classes: {num_classes}")

    # Save label encoder
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    # Load model & tokenizer from local path
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    hidden_size = encoder.config.hidden_size

    # Build classifier
    model = EmbeddingClassifier(encoder, hidden_size, num_classes)

    # Multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model.to(device)

    # Datasets & loaders
    train_dataset = ICDDataset(train_dialogues, train_labels, tokenizer, args.max_length)
    val_dataset = ICDDataset(val_dialogues, val_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, device, num_classes)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5_acc": val_top5,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "model_state_dict": save_model.state_dict(),
                "num_classes": num_classes,
                "hidden_size": hidden_size,
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_top5_acc": val_top5,
            }, os.path.join(args.output_dir, "best_model.pt"))
            # Also save the encoder weights and tokenizer for inference
            save_model.encoder.save_pretrained(os.path.join(args.output_dir, "encoder"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "encoder"))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
