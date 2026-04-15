---
title: EchoBot
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
---

# EchoBot

EchoBot lets you fine-tune a T5 transformer model to learn text transformation patterns from just a handful of examples — then chat with it to see the result.

## How to use

1. **Pick a dataset** from the dropdown on the left. The table shows the input/output pairs the model will learn from.
2. **Click "Train EchoBot"** — training progress is shown epoch by epoch. On CPU this takes a few minutes; on GPU it's under a minute.
3. **Chat** on the right to test the transformation. Try inputs similar to the training examples, or new ones to see how well it generalizes.
4. **Click "Reset EchoBot"** to wipe the fine-tuned weights and start fresh with another dataset.

Before training, EchoBot echoes your message back unchanged.

## Datasets included

| Dataset | What it learns |
|---|---|
| Falsification | Replace adjectives/states with their antonyms |
| Reversal | Reverse the word order of a sentence |
| Statement-to-Question | Convert declarative sentences to yes/no questions |
| Capitalizing Proper Nouns | Fix capitalization of names and places |

## Technical details

- Model: [`t5-base`](https://huggingface.co/t5-base) (encoder-decoder, ~250M parameters)
- Optimizer: AdamW, lr=3e-4
- Training: 10 epochs, batch size 1
- Inference: beam search, 10 beams
- Each browser session has its own independent model — students don't interfere with each other.

## Container builds

Build the image: `docker build -t simonguest/echobot .`

Run the image: `docker run -p 7860:7860 simonguest/echobot`