SHELL := /bin/bash

.PHONY: install
install:
	poetry install --no-root
	poetry run pre-commit install

.PHONY: check_code
check_code:
	poetry run pre-commit run -a

DATASET ?= dev
SAMPLE_SIZE ?= 0

ifeq ($(SAMPLE_SIZE), 0)
	DATASET_FILE := data/$(DATASET).json
	PRED_FILE := data/predictions-$(DATASET).json
else
	DATASET_FILE := data/sampled-$(DATASET)_$(SAMPLE_SIZE).json
	PRED_FILE := data/predictions-sampled-$(DATASET)_$(SAMPLE_SIZE).json
endif

.PHONY: evaluate_rag
evaluate_rag: generate_rag_predictions
	@echo "Evaluating dataset: $(DATASET_FILE) with predictions $(PRED_FILE)"
	poetry run python evaluation/evaluate_predictions.py $(DATASET_FILE) $(PRED_FILE)

.PHONY: generate_rag_predictions
generate_rag_predictions:
	@echo "Generating predictions for $(DATASET) using $(RETRIEVER), sample size: $(SAMPLE_SIZE)"
	PYTHONPATH=$(shell pwd) poetry run python evaluation/generate_rag_predictions.py --dataset $(DATASET) --sample-size $(SAMPLE_SIZE) --retriever $(RETRIEVER)

.PHONY: evaluate_reader
evaluate_reader: generate_reader_predictions
	@echo "Evaluating reader predictions."
	poetry run python evaluation/evaluate_predictions.py data/dev.json data/reader-predictions.json

.PHONY: generate_reader_predictions
generate_reader_predictions:
	@echo "Generating predictions for dev set using the reader."
	PYTHONPATH=$(shell pwd) poetry run python evaluation/generate_reader_predictions.py

.PHONY: finetune
finetune:
	@echo "Starting T5 fine-tuning..."
	PYTHONPATH=$(shell pwd) poetry run python models/finetune.py

.PHONY: start_chroma
start_chroma:
	@echo "Checking if ChromaDB is running..."
	@if ! docker ps --format '{{.Names}}' | grep -q '^chromadb$$'; then \
		echo "Starting ChromaDB..."; \
		docker run -d --name chromadb -p 8000:8000 ghcr.io/chroma-core/chroma:latest; \
	else \
		echo "ChromaDB is already running."; \
	fi

PORT ?= 8000

.PHONY: run_api
run_api: $(if $(filter chroma,$(RETRIEVER)),start_chroma)
	@echo "Starting FastAPI server with retriever: $(RETRIEVER) on port $(PORT)"
	RETRIEVER_TYPE=$(RETRIEVER) USE_CHROMA_SERVER=true PYTHONPATH=$(shell pwd) $(HOME)/.local/bin/poetry run uvicorn deployment.app:app --host 0.0.0.0 --port $(PORT) --reload
