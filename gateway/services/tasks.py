"""UI helpers for task list rendering and task actions."""

import os
import sys
from html import escape

GATEWAY_DIR = os.path.dirname(os.path.dirname(__file__))
if GATEWAY_DIR not in sys.path:
    sys.path.insert(0, GATEWAY_DIR)

from formatting import render_asr_progress, render_progress_bar, render_waiting_progress
from storage.tasks import clear_finished_tasks, list_tasks_with_runtime


def _short_id(task_id: str) -> str:
    return task_id[:8]


def _source_label(source_type: str) -> str:
    return {
        "audio": "Аудио",
        "video": "Видео",
        "text": "Текст",
    }.get(source_type or "", source_type or "Материал")


def _progress_done(label: str = "Готово") -> str:
    return (
        '<div class="task-progress">'
        f'<div class="task-progress__label">{escape(label)}</div>'
        '<div class="task-progress__track">'
        '<div class="task-progress__bar task-progress__bar--done" style="width:100%"></div>'
        '</div></div>'
    )


def _progress_error(label: str) -> str:
    return (
        '<div class="task-progress">'
        f'<div class="task-progress__label">{escape(label)}</div>'
        '<div class="task-progress__track">'
        '<div class="task-progress__bar task-progress__bar--error" style="width:100%"></div>'
        '</div></div>'
    )


def _progress_for_task(task: dict) -> str:
    status = task.get("status")
    current = task.get("progress_current")
    total = task.get("progress_total")

    if status == "asr_running":
        return render_asr_progress(current, total)
    if status == "llm_running":
        if total:
            return render_progress_bar(task.get("progress_stage") or "", current or 0, total)
        return render_waiting_progress("Суммаризация...")
    if status == "waiting_llm":
        return render_waiting_progress("Ожидание LLM...")
    if status == "queued_asr":
        return render_waiting_progress("Ожидание ASR...")
    if status == "done":
        return _progress_done("Запись готова в базе знаний")
    if status == "error":
        return _progress_error(task.get("error") or "Ошибка обработки")
    return _progress_error(task.get("error") or "Файлы задания не найдены")


def _task_title(task: dict) -> str:
    if task.get("result_title"):
        return task["result_title"]
    return task.get("source_name") or f"Задание {_short_id(task['id'])}"


def render_task_list(tasks: list[dict]) -> str:
    if not tasks:
        return (
            '<div class="task-empty-state">'
            "<strong>Заданий пока нет</strong>"
            "<span>Создайте обработку аудио, видео или текста, и она появится в списке.</span>"
            "</div>"
        )

    rows = []
    for task in tasks:
        task_id = task["id"]
        status = task.get("status") or "missing"
        label = task.get("status_label") or status
        title = _task_title(task)
        created = (task.get("created_at") or "")[:16].replace("T", " ")
        source = _source_label(task.get("source_type"))
        source_name = task.get("source_name") or ""
        method = task.get("sum_method") or ""
        report_mode = task.get("report_mode") or ""
        kb_id = task.get("kb_id") or ""
        error = task.get("error") or ""
        badge_class = f"task-status task-status--{escape(status)}"
        result_line = ""
        if kb_id:
            result_line = (
                '<span class="task-row__detail">'
                f'<b>База знаний:</b> {escape(kb_id)}'
                '</span>'
            )
        error_line = ""
        if error:
            error_line = (
                '<span class="task-row__detail task-row__detail--error">'
                f'<b>Ошибка:</b> {escape(error)}'
                '</span>'
            )

        rows.append(
            '<div class="task-row">'
            '<div class="task-row__head">'
            '<div>'
            f'<div class="task-row__title">{escape(title)}</div>'
            f'<div class="task-row__meta">{escape(created)} · {escape(source)}'
            f' · {escape(method)} · {escape(report_mode)} · #{escape(_short_id(task_id))}</div>'
            '</div>'
            f'<span class="{badge_class}">{escape(label)}</span>'
            '</div>'
            '<div class="task-row__details">'
            f'<span class="task-row__detail"><b>UUID:</b> {escape(task_id)}</span>'
            f'<span class="task-row__detail"><b>Источник:</b> {escape(source_name)}</span>'
            f"{result_line}{error_line}"
            '</div>'
            f'<div class="task-row__progress">{_progress_for_task(task)}</div>'
            '</div>'
        )

    return '<div class="task-list">' + "".join(rows) + "</div>"


def refresh_task_ui():
    return render_task_list(list_tasks_with_runtime())


def clear_finished_task_rows():
    list_tasks_with_runtime()
    count = clear_finished_tasks()
    message = f"Удалено завершённых записей: {count}."
    return refresh_task_ui(), _message_html(message)


def _message_html(message: str, task_id: str | None = None) -> str:
    extra = f"<span>UUID: {escape(task_id)}</span>" if task_id else ""
    return (
        '<div class="task-action-message">'
        f"<strong>{escape(message)}</strong>"
        f"{extra}"
        "</div>"
    )


def create_message(message: str, task_id: str | None) -> str:
    return _message_html(message, task_id)
