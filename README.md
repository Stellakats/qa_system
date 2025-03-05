# **QA System**

## **Overview**
This QA system retrieves passages from **SQuAD v2** and generates answers using a **fine-tuned T5 model**.
It supports **multiple retrieval methods**, including:
- **BM25 (Sparse Retrieval)**
- **FAISS (Dense Retrieval)**
- **Hybrid (BM25 + FAISS)**
- **ChromaDB (Vector Database)**

The system runs as a **FastAPI web service**, and supports **dynamic retriever switching**.


## **Installation**
```sh
git clone https://github.com/Stellakats/qa_system.git
cd qa_system
make install
```

## **Running FastAPI**
```sh
make run_api RETRIEVER=<sparse|dense|hybrid|chroma> PORT=<port>
```
This command starts the FastAPI server with the specified retriever and port.

- The system retrieves passages using the SQuAD v2 `train` set as the corpus.
- It uses a fine-tuned T5-small model as the reader: [T5-SQuAD2-Small](https://huggingface.co/stykat/t5-squad2-small), which I fine-tuned and uploaded to the Hugging Face Model Hub.
- To recreate the model, you can run the fine-tuning script included in this repository.

## **Finetune Reader on SDquad**
To fine-tune t5 with default arguments, run:
```sh
make finetune
```

This will save a checkpoint under `t5_squad2`. Alternatively, to fine-tune with custom arguments, run a command similar to the following from project root:
```sh
PYTHONPATH=$(pwd) poetry run python models/finetune.py \
    --model_checkpoint t5-base \
    --dataset train \
    --sample_size 5000 \
    --batch_size 8 \
    --num_train_epochs 5 \
    --learning_rate 3e-5 \
    --fp16
```


## **Evaluate Reader**
After fine-tuning your model, evaluate it on the SQuAD v2 development set by running:

```sh
make evaluate_reader
```
This command:

- Generates predictions for the dev set using the locally saved fine-tuned checkpoint.
- Evaluates the predictions against the ground truth using the official SQuAD evaluation script.


## **Evaluate RAG Model**
To evaluate the RAG, run:

```sh
make evaluate_rag DATASET=<dev|train> SAMPLE_SIZE=<number> RETRIEVER=<sparse|dense|hybrid|chroma>
```
This command :

- Loads the SQuAD v2 dataset (dev or train). You can also sample the dataset using SAMPLE_SIZE which specifies the num of *titles* to include in the dataset (set 0 for the full dataset).
- Uses the specified retriever (sparse, dense, hybrid, or chroma) to fetch relevant passages.
- Passes the retrieved passages to the fine-tuned T5 model for answer generation.
- Saves predictions to data/predictions-<dataset>.json.
- Evaluates the predictions against the ground truth using the official SQuAD evaluation script.

*Note*: You can use either the train or dev set for evaluation, but since the train set was already used to fine-tune the T5 model, it's recommended to use the dev set to avoid data leakage and thereby obtain more objective results.
