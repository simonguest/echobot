import math
import torch
import gradio as gr
import spaces
import pandas as pd

from datasets import DATASETS
from model import load_fresh_model, train_model, infer, TOKENIZER

# ---------------------------------------------------------------------------
# Per-session state factory
# ---------------------------------------------------------------------------

def make_state():
    """Called by gr.State for each new browser session."""
    return {"model": None, "trained_on": None}


def _detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def on_dataset_change(dataset_name):
    pairs = [[inp, out] for inp, out in DATASETS[dataset_name]]
    return pairs


def _overfitting_warning(loss_records):
    """Return a warning string if the final loss is extremely low, or None."""
    if not loss_records:
        return None
    final_loss = 10 ** loss_records[-1]["Log Loss"]
    if final_loss < 0.01:
        return (
            "> **Possible overfitting:** the loss is extremely low, which on a small "
            "dataset usually means the model has memorized the examples rather than "
            "learned the pattern. Try fewer epochs or a lower learning rate."
        )
    return None


@spaces.GPU(duration=300)
def on_train(dataset_name, epochs, lr, state):
    """Generator — yields (progress, state, status, train_btn, reset_btn) after each step."""
    device = _detect_device()
    state["device"] = device

    yield (
        None,
        state,
        "**Status:** Loading model...",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    model = load_fresh_model()
    model.to(device) # type:ignore
    tuples = DATASETS[dataset_name]

    loss_records = []
    for epoch_num, loss in train_model(model, TOKENIZER, tuples, device, epochs=epochs, lr=float(lr)):
        loss_records.append({"Epoch": epoch_num, "Log Loss": math.log10(loss)})
        df = pd.DataFrame(loss_records)
        yield (
            df,
            state,
            f"**Status:** Training... Epoch {epoch_num}/{epochs} | Loss: {loss:.4f}",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    state["model"] = model.cpu()
    state["trained_on"] = dataset_name

    status = f"**Status:** Trained on '{dataset_name}'"
    warning = _overfitting_warning(loss_records)
    if warning:
        status += f"\n\n{warning}"

    yield (
        pd.DataFrame(loss_records),
        state,
        status,
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def on_reset(state):
    state["model"] = None
    state["trained_on"] = None
    return (
        state,
        "**Status:** Untrained (echoing)",
        gr.update(interactive=True),
        gr.update(interactive=False),
        None,
    )


@spaces.GPU
def on_chat(message, history, num_beams, state):
    if not message.strip():
        return history, ""

    if state["model"] is None:
        response = message
    else:
        device = _detect_device()
        model = state["model"].to(device)
        results = infer(model, TOKENIZER, message, device, num_beams=num_beams)
        model.cpu()  # move back to CPU before ZeroGPU releases the allocation
        response = results[0]

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

first_dataset = list(DATASETS.keys())[0]

with gr.Blocks(title="EchoBot") as demo:
    state = gr.State(make_state)

    gr.HTML(
        '<div style="text-align:center">'
        '<img src="/gradio_api/file=logo.png" style="display:block;margin:0 auto;height:300px">'
        '<p>Select a dataset, train the model, then chat to see how EchoBot responds!</p>'
        '</div>'
    )

    with gr.Row():
        # ---- Column 1: dataset explorer ----
        with gr.Column(scale=1):
            gr.Markdown("## Dataset")
            dataset_dropdown = gr.Dropdown(
                choices=list(DATASETS.keys()),
                value=first_dataset,
                label="Select Dataset",
            )
            dataset_table = gr.Dataframe(
                value=[[inp, out] for inp, out in DATASETS[first_dataset]],
                headers=["Input", "Output"],
                interactive=False,
                label="Input / Output Pairs",
                wrap=True,
            )

        # ---- Column 2: training controls ----
        with gr.Column(scale=1):
            gr.Markdown("## Training")
            status_display = gr.Markdown("**Status:** Untrained (echoing)")
            epochs_slider = gr.Slider(
                minimum=1, maximum=50, step=1, value=10,
                label="Epochs",
            )
            lr_dropdown = gr.Dropdown(
                choices=[
                    ("1e-3 — high (aggressive)", "1e-3"),
                    ("3e-4 — medium (default)", "3e-4"),
                    ("1e-4 — low (cautious)", "1e-4"),
                    ("1e-5 — very low (stable)", "1e-5"),
                ],
                value="3e-4",
                label="Learning Rate",
            )
            num_beams_slider = gr.Slider(
                minimum=1, maximum=20, step=1, value=10,
                label="Inference Beams",
            )
            train_btn = gr.Button("Train EchoBot", variant="primary")
            loss_plot = gr.LinePlot(
                value=None,
                x="Epoch",
                y="Log Loss",
                label="Training Loss (log scale)",
                min_width=200,
            )
            reset_btn = gr.Button("Reset EchoBot", variant="secondary", interactive=False)

        # ---- Column 3: chat ----
        with gr.Column(scale=1):
            gr.Markdown("## Chat with EchoBot")
            chatbot = gr.Chatbot(type="messages", height=520)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type a message and press Enter...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1)

    # ---- Event wiring ----
    dataset_dropdown.change(
        fn=on_dataset_change,
        inputs=[dataset_dropdown],
        outputs=[dataset_table],
    )

    train_btn.click(
        fn=on_train,
        inputs=[dataset_dropdown, epochs_slider, lr_dropdown, state],
        outputs=[loss_plot, state, status_display, train_btn, reset_btn],
    )

    reset_btn.click(
        fn=on_reset,
        inputs=[state],
        outputs=[state, status_display, train_btn, reset_btn, loss_plot],
    )

    send_btn.click(
        fn=on_chat,
        inputs=[chat_input, chatbot, num_beams_slider, state],
        outputs=[chatbot, chat_input],
    )

    chat_input.submit(
        fn=on_chat,
        inputs=[chat_input, chatbot, num_beams_slider, state],
        outputs=[chatbot, chat_input],
    )

demo.queue()
demo.launch(server_name="0.0.0.0", allowed_paths=["."])
