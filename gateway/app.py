import os
import subprocess
import sys
import gradio as gr
sys.path.insert(0, os.path.join(PROJECT_ROOT, "storage"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import PROJECT_ROOT
from kb import load_kb_list, load_kb_entry, delete_kb_entry
from engine import ask as rag_ask, ensure_indexed


from monitor import format_status
from formatting import format_segments
from handlers import save_media, save_text, poll_outputs, reformat
from gradio import ChatMessage






def start_workers():
    # запуск  воркеров
    asr_python = os.path.join(PROJECT_ROOT, "asr", ".venv", "Scripts", "python.exe")
    asr_worker = os.path.join(PROJECT_ROOT, "asr", "worker.py")
    llm_python = os.path.join(PROJECT_ROOT, "llm", ".venv", "Scripts", "python.exe")
    llm_worker = os.path.join(PROJECT_ROOT, "llm", "worker.py")

    subprocess.Popen([asr_python, asr_worker])
    subprocess.Popen([llm_python, llm_worker])



def build_transcription_tab():
    # вкладка "Транскрипция"

    with gr.Tab("Транскрипция"):
        with gr.Row():
            # ввод
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

            # результат
            with gr.Column():
                mode = gr.Radio(
                    choices=["С временными метками", "Сплошной текст"],
                    value="С временными метками",
                    label="Режим вывода",
                )
                output = gr.Textbox(label="Транскрипция")
                summary_output = gr.Textbox(label="Резюме (LLM)")

        # панель мониторинга (свернута по умолчанию)
        with gr.Accordion("Мониторинг ресурсов", open=False):
            monitor_box = gr.Textbox(
                label="Состояние системы",
                lines=8,
                interactive=False,
            )
            refresh_btn = gr.Button("Обновить", size="sm")

    # возвращаем все компоненты, которые нужны для привязки событий
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
        "summary_output": summary_output,
        "monitor_box": monitor_box,
        "refresh_btn": refresh_btn,
    }




def chat_respond(message, history):
    # вызов RAG: поиск по KB + генерация ответа
    result = rag_ask(message)
    answer = result["answer"]
    sources = result["sources"]

    # источники списком для метадаты в чате
    sources_text = "\n".join(f"- {s['title']}" for s in sources)

    # два сообщения: ответ + свернутые источники
    new_messages = [
        ChatMessage(content=answer),
        ChatMessage(
            content=sources_text,
            metadata={"title": "Источники", "status": "done"},
        ),
    ]

    # варианты для Radio в правой панели
    choices = [s["title"] for s in sources]
    ids = [s["id"] for s in sources]
    return new_messages, choices, ids


def on_source_select(title, source_ids, source_titles):
    # клик по источнику — загрузить запись из KB
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
    # вкладка "Чат"
    # левая колонка - чат с RAG
    # правая колонка - кликабельные источники + детали
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
                # список источников (обновляется после каждого ответа)
                source_radio = gr.Radio(
                    choices=[], label="Найденные записи", interactive=True,
                )
                # детали выбранного источника
                src_title = gr.Textbox(label="Название", interactive=False, lines=1)
                src_transcript = gr.Textbox(label="Транскрипция", interactive=False, lines=8)
                src_summary = gr.Textbox(label="Резюме", interactive=False, lines=4)

    return {
        "chatbot": chatbot,
        "chat_input": chat_input,
        "chat_btn": chat_btn,
        "source_radio": source_radio,
        "src_title": src_title,
        "src_transcript": src_transcript,
        "src_summary": src_summary,
    }

def build_kb_tab():
    # вкладка "База знаний"
    # левая колонка - таблица записей и кнопки управления
    # правая колонка - просмотр выбранной записи
    with gr.Tab("База знаний"):
        with gr.Row():
            # список записей
            with gr.Column(scale=1):
                kb_table = gr.Dataframe(
                    headers=["Дата", "Заголовок"],
                    datatype=["str", "str"],
                    type="array",
                    interactive=False,
                    label="Сохранённые записи",
                    wrap=True,
                )
                with gr.Row():
                    kb_refresh_btn = gr.Button("Обновить список", size="sm")
                    kb_delete_btn = gr.Button(
                        "Удалить выбранную", size="sm", variant="stop"
                    )

            # детали выбранной записи
            with gr.Column(scale=2):
                kb_mode = gr.Radio(
                    choices=["С временными метками", "Сплошной текст"],
                    value="С временными метками",
                    label="Режим вывода",
                )
                kb_title = gr.Textbox(
                    label="Название", lines=1, interactive=False, container=False
                )
                kb_transcript = gr.Textbox(
                    label="Транскрипция", lines=12, interactive=False,
                )
                kb_summary = gr.Textbox(
                    label="Резюме", lines=5, interactive=False,
                )

    return {
        "kb_table": kb_table,
        "kb_refresh_btn": kb_refresh_btn,
        "kb_delete_btn": kb_delete_btn,
        "kb_mode": kb_mode,
        "kb_title": kb_title,
        "kb_transcript": kb_transcript,
        "kb_summary": kb_summary,
    }



def refresh_kb_table():
    # обработчики событий базы знаний
    # загрузить список записей и сбросить выделение
    table_data, uuids = load_kb_list()
    return table_data, uuids, None, None, "", "", ""


def on_kb_select(evt: gr.SelectData, uuid_list):
    # клик по строке таблицы - загрузить полную запись
    row_idx = evt.index[0]
    if row_idx >= len(uuid_list):
        return None, None, "", "", ""

    entry_id = uuid_list[row_idx]
    entry = load_kb_entry(entry_id)
    if entry is None:
        return None, None, "", "", ""

    segments = entry.get("segments", [])
    summary = entry.get("summary", "")
    title = entry.get("title", "")
    transcript_text = format_segments(segments, "С временными метками")
    return entry_id, segments, transcript_text, summary, title


def on_kb_delete(selected_id):
    # удалить запись и обновить таблицу
    if selected_id:
        delete_kb_entry(selected_id)
    table_data, uuids = load_kb_list()
    return table_data, uuids, None, None, "", "", ""


def on_kb_mode_change(kb_segs, kb_m):
    # переключение режима отображения транскрипции
    if not kb_segs:
        return gr.update()
    return format_segments(kb_segs, kb_m)


# привязка событий вкладки "Транскрипция"

def wire_transcription_events(t, state, timer, monitor_timer):
    # t     - компоненты из build_transcription_tab
    # state - общие gr.State (uuid, segments)

    # обновление монитора каждые 5 сек + кнопка
    monitor_timer.tick(fn=lambda: format_status(), outputs=[t["monitor_box"]])
    t["refresh_btn"].click(fn=lambda: format_status(), outputs=[t["monitor_box"]])

    # общие выходы для кнопок отправки
    send_outputs = [t["output"], state["current_uuid"],
                    t["summary_output"], state["cached_segments"]]

    # отправка аудио / видео / текста
    t["audio_btn"].click(
        fn=save_media, inputs=[t["audio_input"]], outputs=send_outputs,
    )
    t["video_btn"].click(
        fn=save_media, inputs=[t["video_input"]], outputs=send_outputs,
    )
    t["text_btn"].click(
        fn=save_text,
        inputs=[t["text_input"], t["text_file"]],
        outputs=send_outputs,
    )

    # поллинг результатов каждые 2 сек
    timer.tick(
        fn=poll_outputs,
        inputs=[state["current_uuid"], t["mode"]],
        outputs=[t["output"], t["summary_output"],
                 state["current_uuid"], state["cached_segments"]],
    )

    # переключение режима отображения
    t["mode"].change(
        fn=reformat,
        inputs=[state["cached_segments"], t["mode"]],
        outputs=t["output"],
    )


# привязка событий вкладки "База знаний"

def wire_kb_events(kb, state, demo):
    # kb    - компоненты из build_kb_tab
    # state - общие gr.State (kb_uuid_list, kb_selected_*)

    # выходы, которые обновляются при обновлении/удалении записи
    table_outputs = [
        kb["kb_table"], state["kb_uuid_list"], state["kb_selected_id"],
        state["kb_selected_segments"],
        kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
    ]

    kb["kb_refresh_btn"].click(fn=refresh_kb_table, outputs=table_outputs)

    kb["kb_table"].select(
        fn=on_kb_select,
        inputs=[state["kb_uuid_list"]],
        outputs=[state["kb_selected_id"], state["kb_selected_segments"],
                 kb["kb_transcript"], kb["kb_summary"], kb["kb_title"]],
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

    # загрузить список записей при старте
    demo.load(fn=refresh_kb_table, outputs=table_outputs)


# привязка событий вкладки "Чат"

def wire_chat_events(chat, state):
    submit_outputs = [
        chat["chatbot"],       # обновленная история
        chat["source_radio"],  # варианты источников
        state["chat_source_ids"],     # id источников
        state["chat_source_titles"],  # названия источников
        chat["chat_input"],    # очистка поля ввода
        chat["src_title"],     # сброс деталей
        chat["src_transcript"],
        chat["src_summary"],
    ]

    def on_submit(message, history):
        # добавляем вопрос пользователя в историю
        history.append(ChatMessage(role="user", content=message))
        new_messages, choices, ids = chat_respond(message, history)
        for msg in new_messages:
            history.append(msg)
        # обновить Radio с новыми источниками, сбросить детали
        return (
            history,
            gr.update(choices=choices, value=None),
            ids, choices,
            "",   # очистить поле ввода
            "", "", "",  # сбросить детали источника
        )

    # кнопка и enter отправляют вопрос
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

    # клик по источнику — показать детали
    chat["source_radio"].change(
        fn=on_source_select,
        inputs=[chat["source_radio"], state["chat_source_ids"], state["chat_source_titles"]],
        outputs=[chat["src_title"], chat["src_transcript"], chat["src_summary"]],
    )


# сборка и запуск

start_workers()
ensure_indexed()

with gr.Blocks() as demo:
    # скрытые состояния, общие для обеих вкладок
    state = {
        "current_uuid":       gr.State(None),  # uuid текущей задачи
        "cached_segments":    gr.State(None),  # сегменты для переформатирования
        "kb_uuid_list":       gr.State([]),    # список uuid записей в таблице
        "kb_selected_id":     gr.State(None),  # uuid выбранной записи
        "kb_selected_segments": gr.State(None),  # сегменты выбранной записи
        "chat_source_ids":    gr.State([]),    # id источников последнего ответа
        "chat_source_titles": gr.State([]),    # названия источников
    }

    with gr.Tabs():
        t = build_transcription_tab()
        kb = build_kb_tab()
        chat = build_chat_tab()

    # таймеры (живут вне вкладок)
    timer = gr.Timer(value=2)          # поллинг результатов
    monitor_timer = gr.Timer(value=5)  # обновление монитора

    wire_transcription_events(t, state, timer, monitor_timer)
    wire_kb_events(kb, state, demo)
    wire_chat_events(chat, state)

demo.launch()