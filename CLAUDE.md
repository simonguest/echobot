# EchoBot

EchoBot is a Jupyter notebook project for fine-tuning a T5 transformer model to learn arbitrary text transformation tasks from small sets of input/output examples.

## What it does

Uses `t5-base` (HuggingFace Transformers) to learn text transformations defined as `(input, output)` tuple pairs. After training, the model can generalize the pattern to new inputs.

The task prefix used at training and inference time is `generate: {input}</s>`.

Example transformations demonstrated in the notebook:
- **Falsification** — replace adjectives/states with their antonyms
- **Reversal** — reverse word order in a sentence
- **Statement-to-Question** — convert declarative sentences to yes/no questions
- **Capitalizing Proper Nouns** — fix capitalization of names/places

## Architecture / key details

- Model: `t5-base` (encoder-decoder)
- Optimizer: AdamW, lr=3e-4
- Training: 10 epochs, batch size of 1 (iterates over tuples)
- Inference: beam search, 10 beams, returns top 3 sequences
- Device: auto-detected — CPU, CUDA, or MPS (Apple Silicon)

## Project structure

```
notebook/EchoBot.ipynb   # Main notebook — model loading, datasets, training loop, inference
pyproject.toml           # Python deps (uv-managed): torch, transformers, ipykernel, ipywidgets
logos/                   # Logo images
```

## Development setup

Dependencies are managed with `uv`. Python 3.13+ required.

```bash
uv sync          # Install dependencies
```

Run the notebook with Jupyter (kernel is in `.venv`).

## Context

Originally developed in Google Colab, moved to local (commit: "Moved from Colab"). The datasets are framed as student-contributed examples, suggesting this is used in a teaching context.
