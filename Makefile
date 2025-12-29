.PHONY: help setup-core-datasets-dir clean-core-datasets

help:
	@echo "Available commands:"
	@echo ""
	@echo "Dataset Setup:"
	@echo "  make setup-core-datasets-dir - Create core datasets directory"
	@echo "  make clean-core-datasets     - Clean raw chat data in core datasets"

setup-core-datasets-dir:
	mkdir -p datasets/core/chats_raw

clean-core-datasets:
	@echo "Cleaning core chat data..."
	rm -rf datasets/core/chats_cleaned
	python preprocessing/cleaning/core/clean.py