.PHONY: help setup-core-datasets-dir clean-core-datasets clean process-core process-core-test combine-core processing-pipeline-core label-core


LABEL_PORT ?= 5000
LABEL_PASSWORD ?= aaaaaaaaaa # <YOUR-LABEL-PASSWORD>
EVAL_INPUT_DIR ?= datasets/core/chats_processed
EVAL_OUTPUT_DIR ?= datasets/core/chats_evals
TRAIN_OUTPUT_DIR ?= datasets/core/chats_train
EVAL_RATIO ?= 0.1
MIN_EVAL_PER_PERSONA ?= 25
EVAL_SPLIT_SEED ?= 42

@echo "Available commands:"
	@echo ""
	@echo "Dataset Setup:"
	@echo "  make setup-core-datasets-dir - Create core datasets directory"
	@echo "  make clean-core-datasets     - Clean raw chat data in core datasets"
	@echo ""
	@echo "Data Cleaning:"
	@echo "  make clean                   - Clean raw chat data in core datasets"
	@echo ""
	@echo "Data Processing:"
	@echo "  make process-core            - Process cleaned data to ChatML format (per-persona files)"
	@echo "  make process-core-test       - Test processing (first 200 messages)"
	@echo "                                 LLM_ENDPOINT=$(LLM_ENDPOINT) LLM_MODEL=$(LLM_MODEL)"
	@echo "  make combine-core            - Combine persona files into single training file"
	@echo "  make processing-pipeline-core - Full pipeline: clean → process → combine"
	@echo "  make create-eval-datasets    - Split chats_processed into chats_train + chats_evals"
	@echo ""
	@echo "Human Labelling:"
	@echo "  make label-core              - Start core labelling dashboard (PORT=5000 LABEL_PASSWORD=<YOUR-LABEL-PASSWORD>)"

setup-core-datasets-dir:
	mkdir -p datasets/core/chats_raw

clean-core-datasets:
	@echo "Cleaning core chat data..."
	rm -rf datasets/core/chats_cleaned
	python preprocessing/cleaning/core/clean.py

clean:
	@echo "🧹 Cleaning core chat data..."
	rm -rf datasets/core/chats_cleaned
	python preprocessing/cleaning/core/clean.py

process-core:
	@echo "🔄 Processing cleaned data to ChatML format..."
	@echo "LLM Endpoint: $(LLM_ENDPOINT)"
	@echo "LLM Model: $(LLM_MODEL)"
	LLM_ENDPOINT=$(LLM_ENDPOINT) LLM_MODEL=$(LLM_MODEL) python -m preprocessing.process.core

process-core-test:
	@echo "🧪 Testing data processing (first 200 messages)..."
	@echo "LLM Endpoint: $(LLM_ENDPOINT)"
	@echo "LLM Model: $(LLM_MODEL)"
	LLM_ENDPOINT=$(LLM_ENDPOINT) LLM_MODEL=$(LLM_MODEL) python -m preprocessing.process.core --test

combine-core:
	@echo "🔗 Combining persona files into single training file..."
	python -m preprocessing.process.combine

processing-pipeline-core:
	@echo "Starting full core dataset pipeline..."
	@echo ""
	@echo "Step 1: Cleaning raw chat data..."
	$(MAKE) clean
	@echo ""
	@echo "Step 2: Processing cleaned data to ChatML format..."
	$(MAKE) process-core LLM_ENDPOINT=$(LLM_ENDPOINT) LLM_MODEL=$(LLM_MODEL)
	@echo ""
	@echo "Step 3: Combining persona files into single training file..."
	$(MAKE) combine-core
	@echo ""
	@echo "Pipeline complete! Source-of-truth data ready at datasets/core/chats_processed/conversations.jsonl"
	@echo "Next: run 'make create-eval-datasets' to generate datasets/core/chats_train/sft/conversations_train.jsonl"

create-eval-datasets:
	@echo "🧪 Creating train/eval datasets from processed ChatML..."
	@echo "Input: $(EVAL_INPUT_DIR)"
	@echo "Train output: $(TRAIN_OUTPUT_DIR)"
	@echo "Eval output: $(EVAL_OUTPUT_DIR)"
	@echo "Split: eval_ratio=$(EVAL_RATIO) min_eval_per_persona=$(MIN_EVAL_PER_PERSONA) seed=$(EVAL_SPLIT_SEED)"
	python preprocessing/evaluations/create_eval_datasets.py \
		--input_dir $(EVAL_INPUT_DIR) \
		--output_dir $(EVAL_OUTPUT_DIR) \
		--train_dir $(TRAIN_OUTPUT_DIR) \
		--eval_ratio $(EVAL_RATIO) \
		--min_eval_per_persona $(MIN_EVAL_PER_PERSONA) \
		--seed $(EVAL_SPLIT_SEED)

label-core:
	@echo "Starting core labelling dashboard"
	@echo "Visit http://localhost:$(LABEL_PORT) in your browser"
	@echo "Password: $(LABEL_PASSWORD)"
	@echo ""
	@echo "Keyboard shortcuts:"
	@echo "  A - Add to conversation"
	@echo "  S - Skip message"
	@echo "  U - Undo"
	@echo "  E - End conversation"
	@echo ""
	pip install -q -r preprocessing/labelling/requirements.txt
	LABELING_PASSWORD=$(LABEL_PASSWORD) gunicorn --bind 127.0.0.1:$(LABEL_PORT) --workers 1 --timeout 120 --access-logfile - \
		'preprocessing.labelling.core:app'