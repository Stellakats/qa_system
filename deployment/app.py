import os
import pickle
from typing import Dict

from fastapi import FastAPI

from data.squad_loader import load_squad
from models.t5_inference import T5Inference
from retrievers import get_retriever

app = FastAPI()

squad_data = load_squad("train")
documents = squad_data.contexts

cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache"))
os.makedirs(cache_dir, exist_ok=True)

USE_CHROMA_SERVER = os.getenv("USE_CHROMA_SERVER", "false").lower() == "true"
RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "hybrid")
PORT = os.getenv("PORT", "8000")
retriever_cache_path = os.path.join(cache_dir, f"retriever_cache_{RETRIEVER_TYPE}.pkl")

if RETRIEVER_TYPE == "chroma":
    print(f"Initializing retriever: {RETRIEVER_TYPE}")
    retriever = get_retriever(RETRIEVER_TYPE, documents, use_server=USE_CHROMA_SERVER)
else:
    if not os.path.exists(retriever_cache_path):
        print(f"Initializing retriever: {RETRIEVER_TYPE}")
        retriever = get_retriever(RETRIEVER_TYPE, documents)
        with open(retriever_cache_path, "wb") as f:
            pickle.dump(retriever, f)
    else:
        print(f"found cached retriever: {RETRIEVER_TYPE}. Loading...")
        with open(retriever_cache_path, "rb") as f:
            retriever = pickle.load(f)

qa_model = T5Inference(model_dir="./t5_squad2", use_hub=True)


@app.on_event("startup")
def startup_event() -> None:
    """Print API information on startup."""
    print(f"FastAPI Server Running with {RETRIEVER_TYPE} retriever")
    print(f"Access the API at http://127.0.0.1:{PORT}")
    print("Endpoints:")
    print(f" - http://127.0.0.1:{PORT}/ask?question=Your+question+here")


@app.get("/ask")
def ask(question: str, use_hub: bool = True) -> Dict[str, str]:
    """Retrieve a passage and generate an answer."""
    retrieved_docs, _ = retriever.retrieve(question, top_k=1)

    if retrieved_docs:
        context = retrieved_docs[0]
        print(f"Retrieved doc: {context[:200]}...")
    else:
        context = "No relevant context found."
        print("No relevant context found.")

    answer = qa_model.answer([question], [context])[0]

    return {
        "question": question,
        "answer": answer,
        "retriever": RETRIEVER_TYPE,
        "source": "Hugging Face Hub" if use_hub else "Local Checkpoint",
    }


@app.get("/info")
def info() -> Dict[str, str]:
    """Returns the current retriever being used."""
    return {
        "retriever": RETRIEVER_TYPE,
        "use_chroma_server": USE_CHROMA_SERVER,
    }
