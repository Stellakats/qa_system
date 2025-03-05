import os
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base_inference import BaseQAInference


class T5Inference(BaseQAInference):
    def __init__(
        self,
        model_dir: str = "./t5_squad2",
        use_hub: bool = False,
        hub_model: str = "stykat/t5-squad2-small",
        batch_size: int = 8,
    ) -> None:
        super().__init__(batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_hub:
            print(f"loading model from Hugging Face Hub: {hub_model}")
            self.model_path = hub_model
        else:
            self.model_path = self._find_latest_checkpoint(model_dir)
            print(f"loading model from local checkpoint: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(
            self.device
        )

    def _find_latest_checkpoint(self, model_dir: str) -> str:
        """Finds the latest checkpoint in the local directory."""
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {model_dir}")

        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        return os.path.join(model_dir, latest_checkpoint)

    def answer(self, questions: List[str], contexts: List[str]) -> List[str]:
        """Generates answers in batch given lists of questions and contexts."""
        inputs = [f"question: {q} context: {c}" for q, c in zip(questions, contexts)]

        # Tokenize all inputs in batch
        encodings = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        # Generate answers in batch
        with torch.no_grad():
            outputs = self.model.generate(
                **encodings, max_length=128, num_beams=5, early_stopping=True
            )

        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            .strip()
            .replace("no answer", "")
            for output in outputs
        ]
