import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Tokenizer is stateless and read-only — load once at startup and share across sessions.
TOKENIZER = T5Tokenizer.from_pretrained("t5-base")


def load_fresh_model():
    """Return a fresh T5-base model initialized from pre-trained weights."""
    return T5ForConditionalGeneration.from_pretrained("t5-base")


def train_model(model, tokenizer, tuples, device, epochs=10, lr=3e-4):
    """
    Fine-tune model on the given (input, output) tuples.
    Yields a progress string after each epoch so the caller can stream updates.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=lr, eps=1e-8)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for input_text, output_text in tuples:
            input_sent = f"generate: {input_text}</s>"
            output_sent = f"{output_text}</s>"

            tokenized_inp = tokenizer(
                input_sent, max_length=96, padding="max_length", return_tensors="pt"
            )
            tokenized_out = tokenizer(
                output_sent, max_length=96, padding="max_length", return_tensors="pt"
            )

            input_ids = tokenized_inp["input_ids"].to(device)
            attention_mask = tokenized_inp["attention_mask"].to(device)
            labels = tokenized_out["input_ids"].to(device)
            decoder_attention_mask = tokenized_out["attention_mask"].to(device)

            result = model(
                input_ids=input_ids,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                attention_mask=attention_mask,
            )
            loss = result[0]
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        yield epoch + 1, epoch_loss / len(tuples)


def infer(model, tokenizer, text, device, num_beams=10, num_sequences=3):
    """Run beam-search inference and return the top candidate strings."""
    model.eval()

    input_text = f"generate: {text}</s>"
    input_tokens = tokenizer(input_text, return_tensors="pt").to(device)

    # num_sequences cannot exceed num_beams
    num_sequences = min(num_sequences, num_beams)

    with torch.no_grad():
        beam_outputs = model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            max_length=64,
            early_stopping=True,
            num_beams=num_beams,
            num_return_sequences=num_sequences,
            no_repeat_ngram_size=2,
        )

    return [
        tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for out in beam_outputs
    ]
