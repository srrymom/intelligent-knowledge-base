"""Top-level Gradio layout assembly."""

import time

import gradio as gr

from runtime import (
    logger,
    log_startup_environment,
    start_activity_log_console_tailer,
    start_workers,
    write_resource_event,
)
from ui.theme import APP_CSS, APP_HEADER_HTML, APP_THEME, FORCE_LIGHT_THEME_HEAD
from ui.tabs.transcription import build_transcription_tab
from ui.tabs.knowledge_base import build_kb_tab
from ui.tabs.chat import build_chat_tab
from ui.events.transcription_events import wire_transcription_events
from ui.events.knowledge_base_events import wire_kb_events
from ui.events.chat_events import wire_chat_events


def launch_ui():
    log_startup_environment()
    started_at = time.time()
    start_activity_log_console_tailer()
    start_workers()
    logger.info("Собираю Gradio UI")

    with gr.Blocks(title="Локальный ассистент по лекциям", fill_width=True) as demo:
        state = {
            "current_uuid":           gr.State(None),
            "cached_segments":        gr.State(None),
            "structured_report":      gr.State(None),
            "kb_uuid_list":           gr.State([]),
            "kb_selected_id":         gr.State(None),
            "kb_selected_segments":   gr.State(None),
            "kb_structured_report":   gr.State(None),
            "chat_source_ids":        gr.State([]),
            "chat_source_titles":     gr.State([]),
        }

        with gr.Tabs(elem_classes=["app-tabs"]):
            t = build_transcription_tab()
            kb = build_kb_tab()
            chat = build_chat_tab()

        timer = gr.Timer(value=2)
        monitor_timer = gr.Timer(value=5)

        wire_transcription_events(t, state, timer, monitor_timer)
        wire_kb_events(kb, state, demo)
        wire_chat_events(chat, state)

    logger.info("Gradio UI собран за %.1fс", time.time() - started_at)
    logger.info("Запускаю Gradio. Ожидаемый адрес: http://localhost:7860")
    write_resource_event("UI", "Перед запуском Gradio")
    demo.launch(show_error=True, css=APP_CSS, theme=APP_THEME, head=FORCE_LIGHT_THEME_HEAD)
