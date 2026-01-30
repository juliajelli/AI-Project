"""
Fine-tune Ministral-3B with LoRA on medical consultation → SOAP note data.
Designed for offline, multi-GPU execution via accelerate + DeepSpeed.
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb

# Register missing ministral3 text config (not yet in transformers 4.57.6)
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig

# Register Mistral3 as a valid CausalLM model
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

# == paths ====================================================================
BASE_DIR = os.getenv("SLURM_SUBMIT_DIR", ".")
DATA_DIR = os.getenv(BASE_DIR, ".")
LLM_MODEL_IN_DIR = os.path.join(BASE_DIR, "llm_model_in")
LLM_MODEL_OUT_DIR = os.path.join(BASE_DIR, "llm_model_out")
WANDB_DIR = os.path.join(BASE_DIR, "wandb")

# == offline / wandb =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = WANDB_DIR

wandb.init(project=os.getenv("PROJECT_NAME", "Default"), dir=WANDB_DIR)

# == load tokenizer ===========================================================
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_IN_DIR, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# == quantisation config (4-bit QLoRA) ========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# == load model ===============================================================
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_IN_DIR,
    quantization_config=bnb_config,
    device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# == LoRA config ==============================================================
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# == dataset ==================================================================
MAX_SEQ_LENGTH = 4096

dataset = load_dataset(
    "json",
    data_files={
        "training": os.path.join(DATA_DIR, "training_finetuning_llm.jsonl"),
        "validation": os.path.join(DATA_DIR, "validation_finetuning_llm.jsonl"),
    },
)

def preprocess(examples):
    texts = []
    for msgs in examples["messages"]:
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["training"].column_names)

# == training config ==========================================================
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = SFTConfig(
    output_dir=LLM_MODEL_OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    max_seq_length=MAX_SEQ_LENGTH,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
)

# == trainer ==================================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["training"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    peft_config=None,  # already applied above
)

# == train & save =============================================================
trainer.train()
trainer.save_model(os.path.join(LLM_MODEL_OUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(LLM_MODEL_OUT_DIR, "final_model"))
wandb.finish()
print("Training complete. Model saved to", os.path.join(LLM_MODEL_OUT_DIR, "final_model"))
