# EchoBot

EchoBot is a Jupyter notebook project for fine-tuning a T5 transformer model to learn arbitrary text transformation tasks from small sets of input/output examples.

## What it does

Uses `t5-base` (HuggingFace Transformers) to learn text transformations defined as `(input, output)` tuple pairs. After training, the model can generalize the pattern to new inputs.

The task prefix used at training and inference time is `generate: {input}</s>`.

Available datasets (defined in `app/datasets.py`):
- **Falsification** — replace adjectives/states with their antonyms
- **Reversal** — reverse word order in a sentence
- **Statement-to-Question** — convert declarative sentences to yes/no questions
- **Past to Present Tense** — convert past-tense sentences to present tense
- **Formalization** — convert informal/colloquial language to formal English
- **Capitalizing Proper Nouns** — fix capitalization of names/places

## Architecture / key details

- Model: `t5-base` (encoder-decoder, ~250M parameters)
- Optimizer: AdamW, lr=3e-4 (adjustable in the UI)
- Training: configurable epochs (default 10), batch size of 1 (iterates over tuples)
- Inference: beam search, configurable beams (default 10), returns top 3 sequences
- Device: auto-detected — CPU, CUDA, or MPS (Apple Silicon)
- Per-session state: each browser session has its own independent model via `gr.State`
- GPU: uses `@spaces.GPU` (ZeroGPU) on HuggingFace Spaces

## Project structure

```
notebook/EchoBot.ipynb   # Original notebook — model loading, datasets, training loop, inference
app/                     # Gradio web app (deployed to HuggingFace Spaces)
  app.py                 # Main Gradio UI — event handlers, layout, session state
  model.py               # T5 model loading, training loop, inference
  datasets.py            # All (input, output) training pairs for each dataset
  logo_b64.py            # Base64-serialized logo for embedding in the UI
  Dockerfile             # For local Docker builds
  requirements.txt       # Pip deps for Docker / HF Spaces (not uv-managed)
  README.md              # HuggingFace Spaces card (title, sdk metadata, usage docs)
pyproject.toml           # Python deps (uv-managed): torch, transformers, ipykernel, ipywidgets, gradio
logos/                   # Logo source images
```

## Development setup

Dependencies are managed with `uv`. Python 3.13+ required.

```bash
uv sync          # Install dependencies
```

Run the notebook with Jupyter (kernel is in `.venv`).

To run the Gradio app locally:

```bash
cd app
python app.py    # Serves on http://0.0.0.0:7860
```

Or with Docker:

```bash
docker build -t simonguest/echobot app/
docker run -p 7860:7860 simonguest/echobot
```

## Deployment

The app is deployed to HuggingFace Spaces at `spaces/simonguest/echobot` via a `hf` git remote.

To deploy, push the `app/` subtree:

```bash
git push hf $(git commit-tree HEAD:app -m "Deploy"):main --force
```

## Git remotes

- `origin` — GitHub (`https://github.com/simonguest/echobot.git`)
- `hf` — HuggingFace Spaces (`git@hf.co:spaces/simonguest/echobot`)

## Context

Originally developed in Google Colab, moved to local (commit: "Moved from Colab"). The datasets are framed as student-contributed examples, suggesting this is used in a teaching context.
