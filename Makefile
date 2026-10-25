.PHONY: setup test lint prepare train-r8 train-r32 serve-api
setup:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check src tests

prepare:
	distill prepare --out data/processed

train-r8:
	python scripts/train.py --rank 8 --run-name banking77-qlora-r8

train-r32:
	python scripts/train.py --rank 32 --run-name banking77-qlora-r32

serve-api:
	uvicorn distill_lab.api:app --host 127.0.0.1 --port 8080
