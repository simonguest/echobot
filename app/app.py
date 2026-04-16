import torch
import gradio as gr
import spaces

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


@spaces.GPU(duration=300)
def on_train(dataset_name, state):
    """Generator — yields (progress, state, status, train_btn, reset_btn) after each step."""
    device = _detect_device()
    state["device"] = device

    yield (
        "Loading T5-base model...",
        state,
        "**Status:** Loading model...",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    model = load_fresh_model()
    model.to(device) # type:ignore
    tuples = DATASETS[dataset_name]

    progress_lines = ["Model loaded. Starting training...", ""]
    yield (
        "\n".join(progress_lines),
        state,
        "**Status:** Training...",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    for line in train_model(model, TOKENIZER, tuples, device):
        progress_lines.append(line)
        yield (
            "\n".join(progress_lines),
            state,
            "**Status:** Training...",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    state["model"] = model.cpu()
    state["trained_on"] = dataset_name
    progress_lines.append("\nTraining complete!")

    yield (
        "\n".join(progress_lines),
        state,
        f"**Status:** Trained on '{dataset_name}'",
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
    )


@spaces.GPU
def on_chat(message, history, state):
    if not message.strip():
        return history, ""

    if state["model"] is None:
        response = message
    else:
        device = _detect_device()
        model = state["model"].to(device)
        results = infer(model, TOKENIZER, message, device)
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
            train_btn = gr.Button("Train EchoBot", variant="primary")
            progress_display = gr.Textbox(
                label="Training Progress",
                lines=14,
                interactive=False,
                placeholder="Progress will appear here once training starts...",
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
        inputs=[dataset_dropdown, state],
        outputs=[progress_display, state, status_display, train_btn, reset_btn],
    )

    reset_btn.click(
        fn=on_reset,
        inputs=[state],
        outputs=[state, status_display, train_btn, reset_btn],
    )

    send_btn.click(
        fn=on_chat,
        inputs=[chat_input, chatbot, state],
        outputs=[chatbot, chat_input],
    )

    chat_input.submit(
        fn=on_chat,
        inputs=[chat_input, chatbot, state],
        outputs=[chatbot, chat_input],
    )

demo.queue()
demo.launch(server_name="0.0.0.0", allowed_paths=["."])
