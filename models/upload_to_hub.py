import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

repo_name = "stykat/t5-squad2-small"

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "t5_squad2"))

checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]

if not checkpoints:
    raise ValueError(f"No checkpoints found in {model_dir}")

latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
latest_checkpoint_path = os.path.join(model_dir, latest_checkpoint)

print(f"Uploading latest checkpoint: {latest_checkpoint_path}")

tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint_path)

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model uploaded to: https://huggingface.co/{repo_name}")
