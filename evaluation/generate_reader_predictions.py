import json
import os
from typing import Dict

import wandb
from tqdm import tqdm

from data.squad_loader import load_squad
from models import get_model

squad_data = load_squad("dev")

qa_model = get_model(name="t5", model_dir="./t5_squad2", batch_size=32)

wandb.init(project="squad2-t5-evaluation", name="evaluation-run")

batch_size: int = qa_model.batch_size
predictions: Dict[str, str] = {}

for i in tqdm(
    range(0, len(squad_data.ids), batch_size), total=len(squad_data.ids) // batch_size
):
    batch_ids = squad_data.ids[i : i + batch_size]
    batch_questions = squad_data.questions[i : i + batch_size]
    batch_contexts = squad_data.contexts[i : i + batch_size]

    batch_answers = qa_model.answer(batch_questions, batch_contexts)

    for qid, answer in zip(batch_ids, batch_answers):
        predictions[qid] = answer


def save_predictions(predictions: Dict[str, str], output_dir: str = "data") -> str:
    """Saves predictions to a JSON file and returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reader-predictions.json")

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Predictions saved to {output_path}")
    return output_path


output_path = save_predictions(predictions)

wandb.finish()
