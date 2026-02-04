"""
Run inference with the fine-tuned LoRA adapter on top of the base model.

Docker paths:
  - Base model: /tmp/knowledgeBase/base_model (or download at runtime)
  - LoRA adapter: /tmp/knowledgeBase/llm_lora_adapter
"""
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from peft import PeftModel

# Register missing configs
CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
CONFIG_MAPPING._extra_content["ministral3"] = MistralConfig
MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral3Config] = Mistral3ForConditionalGeneration

# == default paths (Docker) ===================================================
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/knowledgeBase/base_model")
DEFAULT_ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/tmp/knowledgeBase/llm_lora_adapter")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def load_model(model_dir, adapter_dir):
    """Load tokenizer and model with LoRA adapter."""
    # Load tokenizer from adapter directory (has chat template)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
    )

    # Load base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return tokenizer, model


def generate_summary(tokenizer, model, transcript: str, max_new_tokens: int = 2048) -> str:
    """Generate SOAP note with ICD-10 diagnosis from dialogue transcript."""
    messages = [
        {"role": "user", "content": transcript},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference for ICD-10 diagnosis")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to base model directory")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input file with dialogue transcript")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}")
    print(f"Loading adapter from {args.adapter_dir}")
    tokenizer, model = load_model(args.model_dir, args.adapter_dir)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        print("\nPaste consultation transcript (press Ctrl+D or Ctrl+Z when done):")
        import sys
        transcript = sys.stdin.read()

    print("\n--- Generated Summary ---\n")
    print(generate_summary(tokenizer, model, transcript, args.max_new_tokens))
