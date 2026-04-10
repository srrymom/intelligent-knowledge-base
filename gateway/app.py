import os
import subprocess
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import PROJECT_ROOT, SUMMARIZATION_METHODS
from shared.ollama_runtime import ensure_ollama_started
sys.path.insert(0, os.path.join(PROJECT_ROOT, "storage"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))
from kb import load_kb_list, load_kb_entry, delete_kb_entry, get_kb_stats, get_all_topics
from engine import ask as rag_ask, ensure_indexed

from monitor import format_status, register_workers
from shared.gpu_coord import reset_gpu_state

sys.path.insert(0, os.path.join(PROJECT_ROOT, "shared"))
from log import read_tail
from formatting import (
    format_segments, render_topics, render_word_stats,
    render_kb_stats,
)
from handlers import save_media, save_text, poll_outputs, reformat, make_export_file
from gradio import ChatMessage


def start_workers():
    reset_gpu_state()
    ensure_ollama_started()
    venv = "Scripts" if sys.platform == "win32" else "bin"
    python_name = "python.exe" if sys.platform == "win32" else "python"
    asr_python = os.path.join(PROJECT_ROOT, "asr", ".venv", venv, python_name)
    asr_worker = os.path.join(PROJECT_ROOT, "asr", "worker.py")
    llm_python = os.path.join(PROJECT_ROOT, "llm", ".venv", venv, python_name)
    llm_worker = os.path.join(PROJECT_ROOT, "llm", "worker.py")
    asr_proc = subprocess.Popen([asr_python, asr_worker])
    llm_proc = subprocess.Popen([llm_python, llm_worker])
    register_workers(asr_proc, llm_proc)


# Вкладка транскрипция 

def build_transcription_tab():
    with gr.Tab("Транскрипция"):
        with gr.Row():
            # — левая колонка: ввод
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Аудио"):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Запись или загрузка аудио",
                        )
                        audio_btn = gr.Button("Отправить")

                    with gr.Tab("Видео"):
                        video_input = gr.Video(
                            sources=["upload"],
                            label="Загрузка видео",
                        )
                        video_btn = gr.Button("Отправить")

                    with gr.Tab("Текст"):
                        text_input = gr.Textbox(
                            label="Вставьте текст",
                            lines=5,
                            placeholder="Введите или вставьте текст...",
                        )
                        text_file = gr.File(
                            label="Или загрузите файл",
                            file_types=[".txt"],
                        )
                        text_btn = gr.Button("Отправить")

            # — правая колонка: результат
            with gr.Column():
                mode = gr.Radio(
                    choices=["С временными метками", "Сплошной текст"],
                    value="С временными метками",
                    label="Режим вывода",
                )
                output = gr.Textbox(label="Транскрипция")
                stats_html = gr.HTML()
                report_mode = gr.Radio(
                    ["Конспект", "Структурированный отчёт"],
                    value="Конспект",
                    show_label=False,
                    container=False,
                )
                sum_method = gr.Radio(
                    SUMMARIZATION_METHODS,
                    value="Hierarchical",
                    label="Метод суммаризации",
                )
                summary_output = gr.Textbox(label="Конспект (LLM)")
                summary_report_md = gr.Markdown(visible=False)
                topics_html = gr.HTML()
                export_file = gr.File(
                    label="Скачать конспект (.md)",
                    visible=False,
                    interactive=False,
                )

        with gr.Accordion("Мониторинг ресурсов", open=False):
            monitor_box = gr.Textbox(
                label="Состояние системы",
                lines=8,
                interactive=False,
            )
            activity_log_box = gr.Textbox(
                label="Журнал активности",
                lines=6,
                interactive=False,
            )
            refresh_btn = gr.Button("Обновить", size="sm")

    return {
        "audio_input": audio_input,
        "audio_btn": audio_btn,
        "video_input": video_input,
        "video_btn": video_btn,
        "text_input": text_input,
        "text_file": text_file,
        "text_btn": text_btn,
        "mode": mode,
        "output": output,
        "stats_html": stats_html,
        "report_mode": report_mode,
        "sum_method": sum_method,
        "summary_output": summary_output,
        "summary_report_md": summary_report_md,
        "topics_html": topics_html,
        "export_file": export_file,
        "monitor_box": monitor_box,
        "activity_log_box": activity_log_box,
        "refresh_btn": refresh_btn,
    }


# вкладка база знаний
def build_kb_tab():
    with gr.Tab("База знаний") as kb_tab:
        # Шапка со статистикой
        kb_stats_html = gr.HTML()

        with gr.Row():
            # — левая колонка: поиск + список
            with gr.Column(scale=1):
                kb_search = gr.Textbox(
                    placeholder="Поиск по названию...",
                    show_label=False,
                    container=False,
                )
                kb_topic_filter = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Фильтр по теме",
                    interactive=True,
                    allow_custom_value=False,
                )
                kb_table = gr.Dataframe(
                    headers=["Дата", "Заголовок"],
                    datatype=["str", "str"],
                    type="array",
                    interactive=False,
                    label="Сохранённые лекции",
                    wrap=True,
                )
                with gr.Row():
                    kb_refresh_btn = gr.Button("Обновить", size="sm")
                    kb_delete_btn = gr.Button(
                        "Удалить выбранную", size="sm", variant="stop"
                    )

            # — правая колонка: детали записи
            with gr.Column(scale=2):
                kb_mode = gr.Radio(
                    choices=["С временными метками", "Сплошной текст"],
                    value="С временными метками",
                    label="Режим вывода",
                )
                kb_title = gr.Textbox(
                    label="Название", lines=1, interactive=False, container=False
                )
                kb_topics_html = gr.HTML()
                kb_transcript = gr.Textbox(
                    label="Транскрипция", lines=10, interactive=False,
                )
                kb_report_mode = gr.Radio(
                    ["Конспект", "Структурированный отчёт"],
                    value="Конспект",
                    show_label=False,
                    container=False,
                )
                kb_summary = gr.Textbox(
                    label="Конспект", lines=5, interactive=False,
                )
                kb_report_md = gr.Markdown(visible=False)
                kb_export_file = gr.File(
                    label="Скачать конспект (.md)",
                    visible=False,
                    interactive=False,
                )

    return {
        "kb_tab": kb_tab,
        "kb_stats_html": kb_stats_html,
        "kb_search": kb_search,
        "kb_topic_filter": kb_topic_filter,
        "kb_table": kb_table,
        "kb_refresh_btn": kb_refresh_btn,
        "kb_delete_btn": kb_delete_btn,
        "kb_mode": kb_mode,
        "kb_title": kb_title,
        "kb_topics_html": kb_topics_html,
        "kb_transcript": kb_transcript,
        "kb_report_mode": kb_report_mode,
        "kb_summary": kb_summary,
        "kb_report_md": kb_report_md,
        "kb_export_file": kb_export_file,
    }


# вкладка чат
def chat_respond(message, history):
    result = rag_ask(message)
    answer = result["answer"]
    sources = result["sources"]

    sources_text = "\n".join(f"- {s['title']}" for s in sources)
    new_messages = [
        ChatMessage(content=answer),
        ChatMessage(
            content=sources_text,
            metadata={"title": "Источники", "status": "done"},
        ),
    ]
    choices = [s["title"] for s in sources]
    ids = [s["id"] for s in sources]
    return new_messages, choices, ids


def on_source_select(title, source_ids, source_titles):
    if not title or not source_titles:
        return "", "", ""
    try:
        idx = source_titles.index(title)
    except ValueError:
        return "", "", ""
    entry = load_kb_entry(source_ids[idx])
    if entry is None:
        return "", "", ""
    segments = entry.get("segments", [])
    transcript = format_segments(segments, "Сплошной текст")
    return entry.get("title", ""), transcript, entry.get("summary", "")


def build_chat_tab():
    with gr.Tab("Чат"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    placeholder="Задайте вопрос по базе знаний...",
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Введите вопрос...",
                        show_label=False,
                        scale=4,
                    )
                    chat_btn = gr.Button("Отправить", scale=1)
            with gr.Column(scale=1):
                gr.Markdown("### Источники")
                source_radio = gr.Radio(
                    choices=[], label="Найденные записи", interactive=True,
                )
                src_title = gr.Textbox(label="Название", interactive=False, lines=1)
                src_transcript = gr.Textbox(label="Транскрипция", interactive=False, lines=8)
                src_summary = gr.Textbox(label="Конспект", interactive=False, lines=4)

    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "chat_btn": chat_btn,
        "source_radio": source_radio,
        "src_title": src_title,
        "src_transcript": src_transcript,
        "src_summary": src_summary,
    }


# обработчики KB

def _kb_stats_html():
    count, words, secs = get_kb_stats()
    return render_kb_stats(count, words, secs)


def refresh_kb_table():
    table_data, uuids = load_kb_list()
    stats = _kb_stats_html()
    topics = get_all_topics()
    return (
        table_data, uuids,
        None, None,                                          # selected_id, selected_segments
        "", "", "",                                          # transcript, summary, title
        "",                                                  # topics_html
        stats,                                               # kb_stats_html
        gr.update(visible=False),                            # kb_export_file
        gr.update(choices=topics, value=None),               # kb_topic_filter
    )


def filter_kb_table(search_text, topic_filter):
    """Фильтрует таблицу по тексту названия и/или выбранной теме."""
    table_data, uuids = load_kb_list(topic_filter=topic_filter or None)
    if not (search_text or "").strip():
        return table_data, uuids
    q = search_text.lower()
    filtered_data, filtered_uuids = [], []
    for row, uid in zip(table_data, uuids):
        if q in row[1].lower():
            filtered_data.append(row)
            filtered_uuids.append(uid)
    return filtered_data, filtered_uuids


def on_kb_select(evt: gr.SelectData, uuid_list):
    row_idx = evt.index[0]
    if row_idx >= len(uuid_list):
        return None, None, "", "", "", "", gr.update(visible=False), None

    entry_id = uuid_list[row_idx]
    entry = load_kb_entry(entry_id)
    if entry is None:
        return None, None, "", "", "", "", gr.update(visible=False), None

    segments = entry.get("segments", [])
    summary = entry.get("summary", "")
    title = entry.get("title", "")
    topics = entry.get("topics", [])
    structured_report = entry.get("structured_report", "")
    transcript_text = format_segments(segments, "С временными метками")
    topics_html = render_topics(topics)

    export_path = make_export_file(entry_id)
    export_update = gr.update(value=export_path, visible=bool(export_path))

    return entry_id, segments, transcript_text, summary, title, topics_html, export_update, structured_report


def on_kb_delete(selected_id):
    if selected_id:
        delete_kb_entry(selected_id)
    return refresh_kb_table()


def on_kb_mode_change(kb_segs, kb_m):
    if not kb_segs:
        return gr.update()
    return format_segments(kb_segs, kb_m)


# привязка событий: Транскрипция 

def wire_transcription_events(t, state, timer, monitor_timer):
    monitor_timer.tick(
        fn=lambda: (format_status(), read_tail(10)),
        outputs=[t["monitor_box"], t["activity_log_box"]],
    )
    t["refresh_btn"].click(
        fn=lambda: (format_status(), read_tail(10)),
        outputs=[t["monitor_box"], t["activity_log_box"]],
    )

    # Выходы кнопок «Отправить» — сбрасываем summary + stats при новой отправке
    send_outputs = [
        t["output"], state["current_uuid"],
        t["summary_output"], state["cached_segments"],
        t["stats_html"],
    ]

    t["audio_btn"].click(fn=save_media, inputs=[t["audio_input"], t["report_mode"], t["sum_method"]], outputs=send_outputs)
    t["video_btn"].click(fn=save_media, inputs=[t["video_input"], t["report_mode"], t["sum_method"]], outputs=send_outputs)
    t["text_btn"].click(
        fn=save_text,
        inputs=[t["text_input"], t["text_file"], t["report_mode"], t["sum_method"]],
        outputs=send_outputs,
    )

    # Поллинг: 8 выходов (см. handlers.py _N = 8)
    timer.tick(
        fn=poll_outputs,
        inputs=[state["current_uuid"], t["mode"]],
        outputs=[
            t["output"], t["summary_output"],
            state["current_uuid"], state["cached_segments"],
            t["topics_html"], t["stats_html"], t["export_file"],
            state["structured_report"],
        ],
    )

    t["mode"].change(
        fn=reformat,
        inputs=[state["cached_segments"], t["mode"]],
        outputs=t["output"],
    )

    def on_report_mode_change(mode, structured, current_uuid):
        if mode == "Структурированный отчёт":
            if structured:
                return gr.update(visible=False), gr.update(value=structured, visible=True)
            # отчёт ещё не сгенерирован (файл не отправлен или не готов)
            return gr.update(visible=False), gr.update(value="", visible=True)
        summary = ""
        if current_uuid:
            entry = load_kb_entry(current_uuid)
            if entry:
                summary = entry.get("summary", "")
        return gr.update(value=summary, visible=True), gr.update(visible=False)

    t["report_mode"].change(
        fn=on_report_mode_change,
        inputs=[t["report_mode"], state["structured_report"], state["current_uuid"]],
        outputs=[t["summary_output"], t["summary_report_md"]],
    )


# привязка событий: База знаний 

def wire_kb_events(kb, state, demo):
    # Полный список выходов при обновлении таблицы
    table_outputs = [
        kb["kb_table"], state["kb_uuid_list"],
        state["kb_selected_id"], state["kb_selected_segments"],
        kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
        kb["kb_topics_html"], kb["kb_stats_html"], kb["kb_export_file"],
        kb["kb_topic_filter"],
    ]

    kb["kb_refresh_btn"].click(fn=refresh_kb_table, outputs=table_outputs)

    # Автообновление при переключении на вкладку
    kb["kb_tab"].select(fn=refresh_kb_table, outputs=table_outputs)

    # Поиск + фильтр по теме — обновляют только таблицу и uuid-список
    kb["kb_search"].change(
        fn=filter_kb_table,
        inputs=[kb["kb_search"], kb["kb_topic_filter"]],
        outputs=[kb["kb_table"], state["kb_uuid_list"]],
    )
    kb["kb_topic_filter"].change(
        fn=filter_kb_table,
        inputs=[kb["kb_search"], kb["kb_topic_filter"]],
        outputs=[kb["kb_table"], state["kb_uuid_list"]],
    )

    kb["kb_table"].select(
        fn=on_kb_select,
        inputs=[state["kb_uuid_list"]],
        outputs=[
            state["kb_selected_id"], state["kb_selected_segments"],
            kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
            kb["kb_topics_html"], kb["kb_export_file"],
            state["kb_structured_report"],
        ],
    )

    def on_kb_report_mode(mode, structured, selected_id):
        if mode == "Структурированный отчёт":
            return gr.update(visible=False), gr.update(value=structured or "", visible=True)
        summary = ""
        if selected_id:
            entry = load_kb_entry(selected_id)
            if entry:
                summary = entry.get("summary", "")
        return gr.update(value=summary, visible=True), gr.update(visible=False)

    kb["kb_report_mode"].change(
        fn=on_kb_report_mode,
        inputs=[kb["kb_report_mode"], state["kb_structured_report"], state["kb_selected_id"]],
        outputs=[kb["kb_summary"], kb["kb_report_md"]],
    )

    kb["kb_mode"].change(
        fn=on_kb_mode_change,
        inputs=[state["kb_selected_segments"], kb["kb_mode"]],
        outputs=kb["kb_transcript"],
    )

    kb["kb_delete_btn"].click(
        fn=on_kb_delete,
        inputs=[state["kb_selected_id"]],
        outputs=table_outputs,
    )

    demo.load(fn=refresh_kb_table, outputs=table_outputs)


# привязка событий: Чат 

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
    )
    chat["chat_input"].submit(
        fn=on_submit,
        inputs=[chat["chat_input"], chat["chatbot"]],
        outputs=submit_outputs,
    )

    chat["source_radio"].change(
        fn=on_source_select,
        inputs=[chat["source_radio"], state["chat_source_ids"], state["chat_source_titles"]],
        outputs=[chat["src_title"], chat["src_transcript"], chat["src_summary"]],
    )


# сборка и запуск 

start_workers()
ensure_indexed()

with gr.Blocks() as demo:
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

    with gr.Tabs():
        t = build_transcription_tab()
        kb = build_kb_tab()
        chat = build_chat_tab()

    timer = gr.Timer(value=2)
    monitor_timer = gr.Timer(value=5)

    wire_transcription_events(t, state, timer, monitor_timer)
    wire_kb_events(kb, state, demo)
    wire_chat_events(chat, state)

demo.launch()
