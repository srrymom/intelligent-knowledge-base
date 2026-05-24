"""Knowledge base UI service functions."""

import os
import sys

import gradio as gr

from shared.config import PROJECT_ROOT
from shared.log import write_event

sys.path.insert(0, os.path.join(PROJECT_ROOT, "storage"))
from kb import delete_kb_entry, get_all_topics, get_kb_stats, load_kb_entry, load_kb_list

from formatting import format_segments, render_kb_stats, render_topics
from handlers import make_export_file


def _kb_stats_html():
    count, words, secs = get_kb_stats()
    return render_kb_stats(count, words, secs)


def _kb_choices(table_data, uuids):
    choices = []
    for row, uid in zip(table_data, uuids):
        date_str = row[0] if len(row) > 0 else ""
        title = row[1] if len(row) > 1 else "(без названия)"
        label = f"{date_str} | {title}" if date_str else title
        choices.append((label, uid))
    return choices


def _filtered_kb_choices(search_text=None, topic_filter=None):
    table_data, uuids = load_kb_list(topic_filter=topic_filter or None)
    if (search_text or "").strip():
        q = search_text.lower()
        filtered_data, filtered_uuids = [], []
        for row, uid in zip(table_data, uuids):
            if q in row[1].lower():
                filtered_data.append(row)
                filtered_uuids.append(uid)
        table_data, uuids = filtered_data, filtered_uuids
    return _kb_choices(table_data, uuids), uuids


def refresh_kb_table():
    table_data, uuids = load_kb_list()
    choices = _kb_choices(table_data, uuids)
    stats = _kb_stats_html()
    topics = get_all_topics()
    is_empty = len(choices) == 0
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=[]),
        uuids,
        None, None,                                          # selected_id, selected_segments
        "", "", "",                                          # transcript, summary, title
        "",                                                  # topics_html
        stats,                                               # kb_stats_html
        gr.update(visible=is_empty),                         # kb_empty_html
        gr.update(visible=False),                            # kb_export_file
        gr.update(choices=topics, value=None),               # kb_topic_filter
        gr.update(visible=False),                            # kb_delete_one_btn
        gr.update(visible=False),                            # kb_delete_btn
        gr.update(visible=True),                             # kb_reader_content
        gr.update(value="Конспект"),                         # kb_report_mode
        gr.update(value="", visible=False),                  # kb_report_md
    )


def filter_kb_table(search_text, topic_filter):
    """Фильтрует список по тексту названия и/или выбранной теме."""
    choices, uuids = _filtered_kb_choices(search_text, topic_filter)
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=[]),
        uuids,
        None, None,                                          # selected_id, selected_segments
        "", "", "",                                          # transcript, summary, title
        "",                                                  # topics_html
        gr.update(value=None, visible=False),                 # kb_export_file
        None,                                                # structured_report
        gr.update(visible=False),                            # kb_delete_one_btn
        gr.update(visible=True),                             # kb_reader_content
        gr.update(value="Конспект"),                         # kb_report_mode
        gr.update(value="", visible=False),                  # kb_report_md
    )


def on_kb_select(entry_id):
    if not entry_id:
        return (
            None, None, "", gr.update(value="", visible=True), "", "",
            gr.update(value=None, visible=False), None,
            gr.update(visible=False), gr.update(visible=True),
            gr.update(value="Конспект"), gr.update(value="", visible=False),
        )

    entry = load_kb_entry(entry_id)
    if entry is None:
        return (
            None, None, "", gr.update(value="", visible=True), "", "",
            gr.update(value=None, visible=False), None,
            gr.update(visible=False), gr.update(visible=True),
            gr.update(value="Конспект"), gr.update(value="", visible=False),
        )

    segments = entry.get("segments", [])
    summary = entry.get("summary", "")
    title = entry.get("title", "")
    topics = entry.get("topics", [])
    structured_report = entry.get("structured_report", "")
    transcript_text = format_segments(segments, "С временными метками")
    topics_html = render_topics(topics)

    export_path = make_export_file(entry_id)
    export_update = gr.update(value=export_path, visible=bool(export_path))

    return (
        entry_id, segments, transcript_text, gr.update(value=summary, visible=True), title, topics_html,
        export_update, structured_report, gr.update(visible=True), gr.update(visible=True),
        gr.update(value="Конспект"), gr.update(value="", visible=False),
    )


def on_kb_mode_switch(mode, search_text=None, topic_filter=None):
    selection_mode = mode == "Выбор"
    choices, _ = _filtered_kb_choices(search_text, topic_filter)
    return (
        gr.update(choices=choices, visible=not selection_mode, value=None),
        gr.update(choices=choices, visible=selection_mode, value=[]),
        gr.update(visible=False),
        gr.update(visible=selection_mode),
    )


def on_kb_delete_one(selected_id):
    if selected_id:
        delete_kb_entry(selected_id)
        write_event("UI", "Удалена запись из базы знаний: 1")
    return refresh_kb_table()


def on_kb_delete_many(selected_ids):
    selected_ids = selected_ids or []
    for entry_id in selected_ids:
        delete_kb_entry(entry_id)
    if selected_ids:
        write_event("UI", f"Удалено записей из базы знаний: {len(selected_ids)}")
    return refresh_kb_table()


def on_kb_mode_change(kb_segs, kb_m):
    if not kb_segs:
        return gr.update()
    return format_segments(kb_segs, kb_m)
