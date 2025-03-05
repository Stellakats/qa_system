import argparse
import json
import os
from typing import List, Tuple

from tqdm import tqdm

from data.squad_loader import load_squad
from models import get_model
from retrievers import get_retriever


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using a QA model with retrieval."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["dev", "train"], help="Dataset type."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of titles to include (0 for full dataset).",
    )
    parser.add_argument(
        "--retriever",
        required=True,
        choices=["sparse", "dense", "hybrid", "chroma"],
        help="Retriever type (BM25, FAISS, Hybrid, ChromaDB).",
    )
    return parser.parse_args()


def get_file_paths(dataset: str, sample_size: int) -> Tuple[str, str]:
    """Generates dataset and prediction file paths based on sample size."""
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    dataset_name = f"sampled-{dataset}_{sample_size}" if sample_size > 0 else dataset
    dataset_file = os.path.join(DATA_DIR, f"{dataset_name}.json")
    prediction_file = os.path.join(DATA_DIR, f"predictions-{dataset_name}.json")

    return dataset_file, prediction_file


def retrieve_passages(retriever, batch_questions: List[str]) -> List[str]:
    """Retrieves relevant passages for each question."""
    batch_contexts = []

    for q in batch_questions:
        retrieved_docs, _ = retriever.retrieve(q, top_k=1)
        if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
            batch_contexts.append(retrieved_docs[0])
        else:
            batch_contexts.append("No relevant passage found.")

    return batch_contexts


def generate_predictions(dataset: str, sample_size: int, retriever_type: str) -> None:
    """Loads SQuAD questions, retrieves one passage per question 
    using the specified retriever and saves all answers provided by finetuned T5."""
    dataset_file, pred_file = get_file_paths(dataset, sample_size)

    squad_data = load_squad(dataset, sample_size if sample_size > 0 else None)

    retriever = get_retriever(retriever_type, squad_data.contexts)
    qa_model = get_model(name="t5", model_dir="./t5_squad2", batch_size=8)

    predictions = {}
    batch_size = qa_model.batch_size

    for i in tqdm(
        range(0, len(squad_data.ids), batch_size),
        total=max(1, len(squad_data.ids) // batch_size),
        desc="Processing Batches",
    ):
        batch_ids = squad_data.ids[i : i + batch_size]
        batch_questions = squad_data.questions[i : i + batch_size]
        batch_contexts = retrieve_passages(retriever, batch_questions)
        batch_answers = qa_model.answer(batch_questions, batch_contexts)

        for qid, answer in zip(batch_ids, batch_answers):
            predictions[qid] = answer

    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    print(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    args = parse_args()
    generate_predictions(args.dataset, args.sample_size, args.retriever)
