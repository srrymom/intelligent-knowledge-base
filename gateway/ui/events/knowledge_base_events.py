"""Event wiring for the knowledge base tab."""

import gradio as gr

from formatting import format_segments
from services.knowledge_base import (
    filter_kb_table,
    load_kb_entry,
    on_kb_delete_many,
    on_kb_delete_one,
    on_kb_mode_change,
    on_kb_mode_switch,
    on_kb_select,
    refresh_kb_table,
)


def wire_kb_events(kb, state, demo):
    def collapse_sidebar():
        return gr.update(visible=True), gr.update(visible=False)

    def expand_sidebar():
        return gr.update(visible=False), gr.update(visible=True)

    # Полный список выходов при обновлении списка
    table_outputs = [
        kb["kb_entry_select"], kb["kb_delete_select"], state["kb_uuid_list"],
        state["kb_selected_id"], state["kb_selected_segments"],
        kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
        kb["kb_topics_html"], kb["kb_stats_html"], kb["kb_empty_html"],
        kb["kb_export_file"], kb["kb_topic_filter"],
        kb["kb_delete_one_btn"], kb["kb_delete_btn"],
        kb["kb_reader_content"], kb["kb_report_mode"], kb["kb_report_md"],
    ]

    # Автообновление при переключении на вкладку
    kb["kb_tab"].select(
        fn=refresh_kb_table,
        outputs=table_outputs,
        queue=False,
        show_progress="hidden",
    )

    kb["kb_filter_collapse_btn"].click(
        fn=collapse_sidebar,
        outputs=[kb["kb_filter_rail"], kb["kb_filter_panel"]],
        queue=False,
        show_progress="hidden",
    )
    kb["kb_filter_expand_btn"].click(
        fn=expand_sidebar,
        outputs=[kb["kb_filter_rail"], kb["kb_filter_panel"]],
        queue=False,
        show_progress="hidden",
    )

    # Поиск + фильтр по теме обновляют список открытия, список удаления и uuid-список
    kb["kb_search"].change(
        fn=filter_kb_table,
        inputs=[kb["kb_search"], kb["kb_topic_filter"]],
        outputs=[
            kb["kb_entry_select"], kb["kb_delete_select"], state["kb_uuid_list"],
            state["kb_selected_id"], state["kb_selected_segments"],
            kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
            kb["kb_topics_html"], kb["kb_export_file"],
            state["kb_structured_report"], kb["kb_delete_one_btn"],
            kb["kb_reader_content"], kb["kb_report_mode"], kb["kb_report_md"],
        ],
        queue=False,
        show_progress="hidden",
    )
    kb["kb_topic_filter"].change(
        fn=filter_kb_table,
        inputs=[kb["kb_search"], kb["kb_topic_filter"]],
        outputs=[
            kb["kb_entry_select"], kb["kb_delete_select"], state["kb_uuid_list"],
            state["kb_selected_id"], state["kb_selected_segments"],
            kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
            kb["kb_topics_html"], kb["kb_export_file"],
            state["kb_structured_report"], kb["kb_delete_one_btn"],
            kb["kb_reader_content"], kb["kb_report_mode"], kb["kb_report_md"],
        ],
        queue=False,
        show_progress="hidden",
    )
    kb["kb_work_mode"].change(
        fn=on_kb_mode_switch,
        inputs=[kb["kb_work_mode"], kb["kb_search"], kb["kb_topic_filter"]],
        outputs=[
            kb["kb_entry_select"],
            kb["kb_delete_select"],
            kb["kb_delete_one_btn"],
            kb["kb_delete_btn"],
        ],
        queue=False,
        show_progress="hidden",
    )

    kb["kb_entry_select"].change(
        fn=on_kb_select,
        inputs=[kb["kb_entry_select"]],
        outputs=[
            state["kb_selected_id"], state["kb_selected_segments"],
            kb["kb_transcript"], kb["kb_summary"], kb["kb_title"],
            kb["kb_topics_html"], kb["kb_export_file"],
            state["kb_structured_report"], kb["kb_delete_one_btn"],
            kb["kb_reader_content"], kb["kb_report_mode"], kb["kb_report_md"],
        ],
        queue=False,
        show_progress="hidden",
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
        queue=False,
        show_progress="hidden",
    )

    kb["kb_mode"].change(
        fn=on_kb_mode_change,
        inputs=[state["kb_selected_segments"], kb["kb_mode"]],
        outputs=kb["kb_transcript"],
        queue=False,
        show_progress="hidden",
    )

    kb["kb_delete_one_btn"].click(
        fn=on_kb_delete_one,
        inputs=[state["kb_selected_id"]],
        outputs=table_outputs,
        queue=False,
        show_progress="hidden",
    )

    kb["kb_delete_btn"].click(
        fn=on_kb_delete_many,
        inputs=[kb["kb_delete_select"]],
        outputs=table_outputs,
        queue=False,
        show_progress="hidden",
    )

    demo.load(
        fn=refresh_kb_table,
        outputs=table_outputs,
        queue=False,
        show_progress="hidden",
    )
