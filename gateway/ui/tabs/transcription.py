"""Task creation and task list tab UI."""

import gradio as gr

from runtime import format_activity_log
from shared.config import SUMMARIZATION_METHODS


def build_transcription_tab():
    with gr.Tab("Задания"):
        with gr.Row(elem_classes=["workspace-row"]):
            with gr.Column(scale=1, elem_classes=["workspace-panel", "task-create-panel"]):
                gr.Markdown("### Создать задание")
                with gr.Tabs():
                    with gr.Tab("Аудио"):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Запись или загрузка аудио",
                        )
                        audio_btn = gr.Button(
                            "Создать задание", variant="primary", elem_classes=["primary-action"]
                        )

                    with gr.Tab("Видео"):
                        video_input = gr.Video(
                            sources=["upload"],
                            label="Загрузка видео",
                        )
                        video_btn = gr.Button(
                            "Создать задание", variant="primary", elem_classes=["primary-action"]
                        )

                    with gr.Tab("Текст"):
                        text_input = gr.Textbox(
                            label="Вставьте текст",
                            lines=8,
                            placeholder="Введите или вставьте текст...",
                        )
                        text_file = gr.File(
                            label="Или загрузите файл",
                            file_types=[".txt"],
                        )
                        text_btn = gr.Button(
                            "Создать задание", variant="primary", elem_classes=["primary-action"]
                        )

                sum_method = gr.Radio(
                    SUMMARIZATION_METHODS,
                    value="Hierarchical",
                    label="Метод суммаризации",
                )
                report_mode = gr.Radio(
                    ["Конспект", "Структурированный отчёт"],
                    value="Конспект",
                    label="Тип результата",
                )
                task_action_status = gr.HTML()

            with gr.Column(scale=2, elem_classes=["workspace-panel", "task-list-panel"]):
                with gr.Row(elem_classes=["task-toolbar"]):
                    refresh_tasks_btn = gr.Button("Обновить список", size="sm")
                    clear_finished_btn = gr.Button("Очистить завершённые", size="sm")

                task_list_html = gr.HTML()

        with gr.Accordion("Мониторинг ресурсов", open=False):
            monitor_box = gr.Textbox(
                label="Состояние системы",
                lines=8,
                interactive=False,
            )
            gr.Markdown("Журнал активности")
            activity_log_box = gr.HTML(format_activity_log())
            refresh_btn = gr.Button("Обновить журнал", size="sm")

    return {
        "audio_input": audio_input,
        "audio_btn": audio_btn,
        "video_input": video_input,
        "video_btn": video_btn,
        "text_input": text_input,
        "text_file": text_file,
        "text_btn": text_btn,
        "report_mode": report_mode,
        "sum_method": sum_method,
        "task_action_status": task_action_status,
        "task_list_html": task_list_html,
        "refresh_tasks_btn": refresh_tasks_btn,
        "clear_finished_btn": clear_finished_btn,
        "monitor_box": monitor_box,
        "activity_log_box": activity_log_box,
        "refresh_btn": refresh_btn,
    }
