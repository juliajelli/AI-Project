"""
Run inference with the fine-tuned LoRA adapter on top of the base model.
"""
import os
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

# == paths ====================================================================
BASE_DIR = "/work/hanken/aibas_ft2"
MODEL_DIR = os.path.join(BASE_DIR, "model")
ADAPTER_DIR = os.path.join(BASE_DIR, "output", "final_model")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# == load tokenizer ===========================================================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True)

# == load base model + adapter ================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

# == inference ================================================================
def generate_summary(transcript: str, max_new_tokens: int = 2048) -> str:
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
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            transcript = f.read()
    else:
        transcript = input("Paste consultation transcript:\n")

    print("\n--- Generated Summary ---\n")
    print(generate_summary(transcript))