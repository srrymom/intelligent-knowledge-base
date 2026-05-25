"""Knowledge base tab UI."""

import gradio as gr

from ui.theme import KB_EMPTY_HTML


def build_kb_tab():
    with gr.Tab("База знаний") as kb_tab:
        with gr.Row(equal_height=False, elem_classes=["workspace-row"]):
            # — левая колонка: поиск + список
            with gr.Column(scale=0, min_width=56, visible=False, elem_classes=["kb-sidebar-rail"]) as kb_filter_rail:
                kb_filter_expand_btn = gr.Button("Поиск", size="sm", elem_classes=["kb-sidebar-rail-btn"])

            with gr.Column(scale=1, min_width=300, elem_classes=["kb-filter-panel", "kb-panel"]) as kb_filter_panel:
                with gr.Group(elem_classes=["kb-panel", "kb-sidebar-panel"]):
                    with gr.Row(elem_classes=["kb-sidebar-head"]):
                        kb_filter_collapse_btn = gr.Button("‹", size="sm", elem_classes=["kb-sidebar-toggle"])
                        with gr.Column(scale=1, min_width=0, elem_classes=["kb-sidebar-meta"]):
                            gr.Markdown("**Поиск и выбор**", elem_classes=["kb-sidebar-title"])
                            kb_stats_html = gr.HTML(elem_classes=["kb-sidebar-stats"])
                    kb_empty_html = gr.HTML(KB_EMPTY_HTML, visible=False)
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
                        elem_classes=["kb-topic-filter"],
                    )
                    kb_work_mode = gr.Radio(
                        choices=["Просмотр", "Выбор"],
                        value="Просмотр",
                        label="Режим",
                        interactive=True,
                    )
                    kb_entry_select = gr.Radio(
                        choices=[],
                        value=None,
                        label="Открыть запись",
                        interactive=True,
                        elem_classes=["kb-list"],
                    )
                    kb_delete_select = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="Выбрать для удаления",
                        interactive=True,
                        show_select_all=True,
                        elem_classes=["kb-list"],
                        visible=False,
                    )
                    with gr.Row(elem_classes=["kb-actions"]):
                        kb_delete_one_btn = gr.Button(
                            "Удалить текущую", size="sm", variant="stop", visible=False
                        )
                        kb_delete_btn = gr.Button(
                            "Удалить выбранные", size="sm", variant="stop", visible=False
                        )

            # — правая колонка: детали записи.
            with gr.Column(scale=4, elem_classes=["kb-panel", "kb-reader-panel"]):
                with gr.Group( elem_classes=["kb-reader-group",]) as kb_reader_content:
                    kb_report_mode = gr.Radio(
                        ["Конспект", "Структурированный отчёт"],
                        value="Конспект",
                        show_label=False,
                        container=False,
                        elem_classes=["kb-report-toggle"],
                    )
                    kb_title = gr.Textbox(
                        label="Название", lines=1, interactive=False, container=False
                    )
                    kb_summary = gr.Textbox(
                        label="Конспект",
                        lines=6,
                        interactive=False,
                        elem_classes=["summary-text"],
                    )
                    kb_report_md = gr.Markdown(visible=False)
                   
                    kb_topics_html = gr.HTML()
                    kb_export_file = gr.File(
                        label="Скачать конспект (.md)",
                        visible=False,
                        interactive=False,
                    )
                    with gr.Accordion("Транскрипция", open=False, elem_classes=["kb-transcript-box"]):
                        kb_mode = gr.Radio(
                            choices=["С временными метками", "Сплошной текст"],
                            value="С временными метками",
                            label="Режим вывода",
                        )
                        kb_transcript = gr.Textbox(
                            label="Транскрипция",
                            lines=10,
                            interactive=False,
                            elem_classes=["reader-text"],
                        )

    return {
        "kb_tab": kb_tab,
        "kb_filter_rail": kb_filter_rail,
        "kb_filter_panel": kb_filter_panel,
        "kb_filter_expand_btn": kb_filter_expand_btn,
        "kb_filter_collapse_btn": kb_filter_collapse_btn,
        "kb_stats_html": kb_stats_html,
        "kb_empty_html": kb_empty_html,
        "kb_search": kb_search,
        "kb_topic_filter": kb_topic_filter,
        "kb_work_mode": kb_work_mode,
        "kb_entry_select": kb_entry_select,
        "kb_delete_select": kb_delete_select,
        "kb_delete_one_btn": kb_delete_one_btn,
        "kb_delete_btn": kb_delete_btn,
        "kb_reader_content": kb_reader_content,
        "kb_mode": kb_mode,
        "kb_title": kb_title,
        "kb_topics_html": kb_topics_html,
        "kb_transcript": kb_transcript,
        "kb_report_mode": kb_report_mode,
        "kb_summary": kb_summary,
        "kb_report_md": kb_report_md,
        "kb_export_file": kb_export_file,
    }
