import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

SQUAD_URLS: Dict[str, str] = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

SQUAD_FILES: Dict[str, str] = {"train": "train.json", "dev": "dev.json"}


@dataclass
class SquadData:
    ids: List[str]
    questions: List[str]
    answers: List[Dict]
    contexts: List[str]


def download_squad(squad_type: str = "dev", sample_size: Optional[int] = None) -> str:
    """
    Download the SQuAD dataset and return the file path.

    Args:
        squad_type (str): 'train' for training data, 'dev' for development data.
        sample_size (Optional[int]): Number of titles to sample. If None, full dataset.

    Returns:
        str: The path to the downloaded (or sampled) dataset file.
    """
    if squad_type not in SQUAD_URLS:
        raise ValueError(f"Invalid squad_type: {squad_type}. Must be 'train' or 'dev'.")

    squad_url = SQUAD_URLS[squad_type]
    pwd = os.path.dirname(os.path.realpath(__file__))
    squad_file = os.path.join(pwd, SQUAD_FILES[squad_type])

    if not os.path.exists(squad_file):
        print(f"Downloading {squad_type} dataset from {squad_url} to {squad_file}...")
        response = requests.get(squad_url)
        with open(squad_file, "wb") as f:
            f.write(response.content)
        print("Download complete.")

    if sample_size is None:
        return squad_file

    sampled_file = os.path.join(pwd, f"sampled-{squad_type}_{sample_size}.json")

    if not os.path.exists(sampled_file):
        print(f"Creating a sampled dataset with {sample_size} titles...")

        with open(squad_file, "r") as f:
            squad_data = json.load(f)

        sampled_data = squad_data.copy()
        sampled_data["data"] = squad_data["data"][:sample_size]

        with open(sampled_file, "w") as f:
            json.dump(sampled_data, f)

        print(f"Sampled dataset saved to {sampled_file}")

    return sampled_file


def load_squad(squad_type: str = "dev", sample_size: Optional[int] = None) -> SquadData:
    """
    Load the SQuAD dataset and return structured SquadData.

    Args:
        squad_type (str): 'train' for training data, 'dev' for development data.
        sample_size (Optional[int]): Number of titles to sample. If None, full dataset.

    Returns:
        SquadData: A dataclass containing lists of questions, answers, contexts, and IDs.
    """
    file_path = download_squad(squad_type, sample_size)

    with open(file_path, "r") as f:
        squad_data = json.load(f)

    contexts, questions, answers, qids = [], [], [], []

    for group in squad_data["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                qid = qa["id"]

                if qa["is_impossible"] or not qa["answers"]:
                    answer_text = ""
                    answer_start = -1
                else:
                    # Take only the first answer (for the dev set)
                    answer_text = qa["answers"][0]["text"]
                    answer_start = qa["answers"][0]["answer_start"]

                contexts.append(context)
                questions.append(question)
                answers.append({"text": answer_text, "answer_start": answer_start})
                qids.append(qid)

    return SquadData(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ids=qids,
    )
