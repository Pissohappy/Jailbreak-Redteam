# VLM Redteam Skeleton

A minimal runnable repository skeleton for a future LangGraph-based multi-round beam-search VLM red-team framework.

## Requirements

- Python 3.11+

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run

```bash
python -m vlm_redteam.cli.run --config configs/run.yaml
```

Expected output:

- `Loaded config OK`
- `Graph compiled OK`
