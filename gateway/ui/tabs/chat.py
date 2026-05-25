"""Chat tab UI."""

import gradio as gr


def build_chat_tab():
    with gr.Tab("Чат"):
        with gr.Row(elem_classes=["workspace-row"]):
            with gr.Column(scale=2, elem_classes=["workspace-panel", "chatbot-panel"]):
                chatbot = gr.Chatbot(
                    placeholder="Задайте вопрос по базе знаний...",
                    height=480,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Введите вопрос...",
                        show_label=False,
                        scale=4,
                    )
                    chat_btn = gr.Button("Отправить", variant="primary", scale=1)
            with gr.Column(scale=1, elem_classes=["workspace-panel", "source-panel"]):
                gr.Markdown("### Источники", elem_classes=["panel-title"])
                source_radio = gr.Radio(
                    choices=[], label="Найденные записи", interactive=True,
                )
                src_title = gr.Textbox(label="Название", interactive=False, lines=1)
                src_transcript = gr.Textbox(
                    label="Транскрипция",
                    interactive=False,
                    lines=8,
                    elem_classes=["reader-text"],
                )
                src_summary = gr.Textbox(
                    label="Конспект",
                    interactive=False,
                    lines=5,
                    elem_classes=["summary-text"],
                )

    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "chat_btn": chat_btn,
        "source_radio": source_radio,
        "src_title": src_title,
        "src_transcript": src_transcript,
        "src_summary": src_summary,
    }
