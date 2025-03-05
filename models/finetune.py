import argparse
import random

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb
from data.squad_loader import load_squad


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune T5 model on SQuAD 2.0")

    # Model & Data
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="t5-small",
        help="Pretrained model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "dev"],
        help="Dataset type",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples (None for full dataset)",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training (fp16)"
    )

    # Model Saving & Logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./t5_squad2",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save model checkpoint every X steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=250,
        help="Log training progress every X steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit on the number of saved checkpoints",
    )

    return parser.parse_args()


args = parse_args()

torch.backends.cuda.matmul.allow_tf32 = True

train_data = load_squad(squad_type=args.dataset, sample_size=args.sample_size)

contexts = list(set(train_data.contexts))
random.seed(42)
random.shuffle(contexts)

val_contexts = set(contexts[: int(0.1 * len(contexts))])

train_samples, val_samples = [], []
for q, c, a in zip(train_data.questions, train_data.contexts, train_data.answers):
    sample = {"question": q, "context": c, "answers": a}
    if c in val_contexts:
        val_samples.append(sample)
    else:
        train_samples.append(sample)

train_dataset = Dataset.from_list(train_samples)
val_dataset = Dataset.from_list(val_samples)

print(f"Final Train Size: {len(train_dataset)}")
print(f"Final Validation Size: {len(val_dataset)}")

wandb.init(project="squad2-reader-finetuning", name=args.model_checkpoint)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)


def preprocess_squad_data(data):
    inputs = [
        f"question: {q} context: {c}" for q, c in zip(data["question"], data["context"])
    ]
    targets = [a["text"] if a["text"] else "no answer" for a in data["answers"]]

    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=1024
    )
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=256)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Preprocessing datasets...")
train_dataset = train_dataset.map(preprocess_squad_data, batched=True, num_proc=8)
val_dataset = val_dataset.map(preprocess_squad_data, batched=True, num_proc=8)
print("Datasets processed")

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy="steps",
    eval_steps=args.save_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=args.logging_steps,
    report_to="wandb",
    fp16=args.fp16,
    fp16_full_eval=args.fp16,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    dataloader_num_workers=4,
    save_total_limit=args.save_total_limit,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
