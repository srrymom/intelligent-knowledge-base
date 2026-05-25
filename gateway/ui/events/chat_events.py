"""Event wiring for the chat tab."""

import gradio as gr
from gradio import ChatMessage

from services.chat import chat_respond, on_source_select


def wire_chat_events(chat, state):
    submit_outputs = [
        chat["chatbot"],
        chat["source_radio"],
        state["chat_source_ids"],
        state["chat_source_titles"],
        chat["chat_input"],
        chat["src_title"],
        chat["src_transcript"],
        chat["src_summary"],
    ]

    def on_submit(message, history):
        history = history or []
        history.append(ChatMessage(role="user", content=message))
        new_messages, choices, ids = chat_respond(message, history)
        for msg in new_messages:
            history.append(msg)
        return (
            history,
            gr.update(choices=choices, value=None),
            ids, choices,
            "",
            "", "", "",
        )

    chat["chat_btn"].click(
        fn=on_submit,
        inputs=[chat["chat_input"], chat["chatbot"]],
        outputs=submit_outputs,
        concurrency_limit=1,
        concurrency_id="rag_chat",
    )
    chat["chat_input"].submit(
        fn=on_submit,
        inputs=[chat["chat_input"], chat["chatbot"]],
        outputs=submit_outputs,
        concurrency_limit=1,
        concurrency_id="rag_chat",
    )

    chat["source_radio"].change(
        fn=on_source_select,
        inputs=[chat["source_radio"], state["chat_source_ids"], state["chat_source_titles"]],
        outputs=[chat["src_title"], chat["src_transcript"], chat["src_summary"]],
        queue=False,
        show_progress="hidden",
    )
