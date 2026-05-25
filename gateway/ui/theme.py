"""Shared Gradio theme and CSS for the gateway UI."""

import gradio as gr

APP_CSS = """
:root {
    --app-bg: #f7f8fb;
    --app-surface: #ffffff;
    --app-surface-muted: #f2f5f9;
    --app-border: #cfd8e6;
    --app-border-strong: #9fb0c5;
    --app-text: #172033;
    --app-muted: #526071;
    --app-accent: #1f5fbf;
    --app-accent-soft: #e8f1ff;
    --app-danger-soft: #fff1f2;
}

.gradio-container {
    background: var(--app-bg) !important;
    color: var(--app-text) !important;
    color-scheme: light;
}

.main {
    max-width: 1320px;
    margin: 0 auto;
}

.app-header {
    margin: 12px 0 14px;
    padding: 16px 20px;
    border: 1px solid var(--app-border);
    border-radius: 8px;
    background: var(--app-surface);
    box-shadow: 0 10px 26px rgba(23, 32, 51, 0.06);
}

.app-header__top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
}

.app-kicker {
    margin: 0 0 4px;
    color: var(--app-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
}

.app-title {
    margin: 0;
    color: var(--app-text);
    font-size: clamp(22px, 2.4vw, 30px);
    line-height: 1.12;
    letter-spacing: 0;
}

.app-subtitle {
    max-width: 760px;
    margin: 7px 0 0;
    color: var(--app-muted);
    font-size: 14px;
    line-height: 1.45;
}

.app-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
    min-width: 250px;
}

.app-badge {
    border: 1px solid var(--app-border);
    border-radius: 999px;
    background: var(--app-surface-muted);
    color: #334155;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 650;
    white-space: nowrap;
}

.app-tabs > .tab-nav {
    border-bottom: 1px solid var(--app-border);
}

.workspace-row {
    gap: 14px;
    align-items: stretch;
}

.workspace-panel {
    border: 1px solid var(--app-border-strong);
    border-radius: 8px;
    background: var(--app-surface);
    padding: 12px;
    box-shadow: 0 7px 18px rgba(23, 32, 51, 0.06);
    color: var(--app-text);
}

.gradio-container div[class*="styler"] {
    background: #ffffff !important;
}

.gradio-container div[class*="wrap-inner"] {
    border: 1px solid var(--app-border-strong) !important;
    border-radius: 8px !important;
}

.gradio-container label,
.gradio-container .block-label,
.gradio-container .form label,
.gradio-container span[data-testid="block-info"] {
    color: #334155 !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="textbox"],
.gradio-container [role="combobox"] {
    color: var(--app-text) !important;
    background: #ffffff !important;
}

.gradio-container [data-testid="dropdown"] *,
.gradio-container .dropdown * {
    color: var(--app-text) !important;
}

.gradio-container [data-testid="dropdown"] input::placeholder,
.gradio-container .dropdown input::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Gradio иногда почти сливает dropdown с фоном; для фильтра темы держим явную рамку. */
.kb-topic-filter,
.kb-topic-filter > div,
.kb-topic-filter > .styler,
.kb-topic-filter [data-testid="dropdown"],
.kb-topic-filter .wrap,
.kb-topic-filter .container {
    border: 1px solid var(--app-border-strong) !important;
    border-color: var(--app-border-strong) !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

.kb-topic-filter input,
.kb-topic-filter [role="combobox"] {
    border-color: var(--app-border-strong) !important;
}

/* Выбранные radio/checkbox должны отличаться не только маленькой точкой, но и всей плиткой. */
.gradio-container [role="radio"][aria-checked="true"],
.gradio-container [role="checkbox"][aria-checked="true"],
.gradio-container label:has(input[type="radio"]:checked),
.gradio-container label:has(input[type="checkbox"]:checked) {
    border-color: var(--app-accent) !important;
    background: var(--app-accent-soft) !important;
    color: #174ea6 !important;
    box-shadow: inset 0 0 0 1px var(--app-accent) !important;
}

.gradio-container [role="radio"][aria-checked="false"],
.gradio-container [role="checkbox"][aria-checked="false"],
.gradio-container label:has(input[type="radio"]:not(:checked)),
.gradio-container label:has(input[type="checkbox"]:not(:checked)) {
    border-color: #d7dee8 !important;
    background: #ffffff !important;
    color: var(--app-text) !important;
}

.kb-reader-panel:empty {
    min-height: 420px;
}

.primary-action {
    width: 100%;
}

.output-text textarea,
.reader-text textarea {
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    line-height: 1.55;
}

.output-text textarea {
    min-height: 220px;
}

.summary-text textarea {
    min-height: 150px;
    line-height: 1.55;
}

.chatbot-panel {
    min-height: 520px;
}

.source-panel {
    min-height: 520px;
}

.activity-log {
    max-height: 320px;
    overflow-y: auto;
    overflow-x: auto;
    padding: 10px 12px;
    border: 1px solid var(--app-border);
    border-radius: 6px;
    background: #f8fafc;
    color: #263244;
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    line-height: 1.45;
    white-space: pre-wrap;
}
.kb-list {
    max-height: 360px;
    overflow-y: auto;
}

.kb-list label,
.kb-list [role="radio"],
.kb-list [role="checkbox"] {
    min-height: 44px;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    align-items: center;
}

.kb-panel {
    border: 1px solid var(--app-border-strong);
    border-radius: 8px;
    background: var(--app-surface);
    padding: 12px;
    box-shadow: 0 7px 18px rgba(23, 32, 51, 0.06);
    color: var(--app-text);
}
.kb-reader-panel {
    min-height: 420px;
}

.topic-chip-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 6px 0 2px;
}

.topic-chip {
    display: inline-block;
    padding: 3px 11px;
    border: 1px solid color-mix(in srgb, var(--topic-color) 34%, #ffffff);
    border-radius: 999px;
    background: #ffffff;
    color: var(--topic-color);
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}

.kb-reader-panel div[class*="gr-group"] > div[class*="styler"],
.kb-reader-panel div[class*="gr-group"] div[class*="styler"] {
    border: 0 !important;
    background: #ffffff !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.kb-report-toggle,
.kb-report-toggle > div,
.kb-report-toggle .wrap {
    background: #ffffff !important;
}

.kb-report-toggle label,
.kb-report-toggle [role="radio"] {
    background: #ffffff !important;
}

.kb-report-toggle label:has(input[type="radio"]:checked),
.kb-report-toggle [role="radio"][aria-checked="true"] {
    background: #ffffff !important;
    border-color: var(--app-accent) !important;
    color: #174ea6 !important;
    box-shadow: inset 0 0 0 1px var(--app-accent) !important;
}

.kb-sidebar-rail {
    flex: 0 0 56px !important;
    min-width: 56px !important;
    max-width: 56px !important;
}

.kb-sidebar-rail .form,
.kb-sidebar-rail > .styler {
    height: 100%;
}

.kb-sidebar-rail-btn {
    min-width: 44px !important;
    width: 44px !important;
    min-height: 420px !important;
    padding: 12px 0 !important;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    white-space: nowrap;
}

.kb-filter-panel {
    flex: 0 1 360px !important;
    min-width: 300px !important;
    max-width: 390px;
    background: #ffffff !important;
}

.kb-filter-panel > .styler,
.kb-filter-panel .gr-form,
.kb-filter-panel .form {
    background: #ffffff !important;
}

.kb-sidebar-panel {
    background: #ffffff !important;
    min-height: 420px;
}

.kb-sidebar-panel > div[class*="styler"] {
    display: flex !important;
    flex-direction: column !important;
    min-height: 420px;
}

.kb-sidebar-head {
    position: relative;

    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}

.kb-sidebar-title {
    text-align: center;
    flex: 1 1 auto;
    padding-top: 8px;
}

.kb-sidebar-title p {
    margin: 0 !important;
    text-align: center;
}

.kb-sidebar-toggle {
    flex: 0 0 34px !important;
    min-width: 34px !important;
    width: 34px !important;
    min-height: 38px !important;
    height: 38px !important;
    padding: 0 !important;
}

.kb-actions {
    gap: 10px;
    margin-top: auto;
}

.task-create-panel,
.task-list-panel {
    align-self: flex-start;
}

.task-toolbar {
    gap: 10px;
    align-items: center;
}

.task-list {
    display: grid;
    gap: 10px;
    margin: 4px 0 12px;
}

.task-row {
    border: 1px solid var(--app-border);
    border-radius: 8px;
    background: #ffffff;
    padding: 11px 12px;
}

.task-row__head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 8px;
}

.task-row__title {
    color: var(--app-text);
    font-size: 14px;
    font-weight: 700;
    line-height: 1.25;
    overflow-wrap: anywhere;
}

.task-row__meta {
    margin-top: 3px;
    color: var(--app-muted);
    font-size: 12px;
    line-height: 1.35;
}

.task-row__details {
    display: grid;
    gap: 3px;
    margin: 0 0 8px;
}

.task-row__detail {
    color: var(--app-muted);
    font-size: 12px;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.task-row__detail b {
    color: #334155;
}

.task-row__detail--error,
.task-row__detail--error b {
    color: #a31220;
}

.task-status {
    flex: 0 0 auto;
    border: 1px solid var(--app-border);
    border-radius: 999px;
    background: var(--app-surface-muted);
    color: #334155;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

.task-status--done {
    border-color: #86c69a;
    background: #eaf7ee;
    color: #14632e;
}

.task-status--error,
.task-status--missing {
    border-color: #f1a1a8;
    background: #fff1f2;
    color: #a31220;
}

.task-status--asr_running,
.task-status--llm_running {
    border-color: #9fb7d7;
    background: #eef4ff;
    color: #174ea6;
}

.task-row__progress > div {
    margin: 0 !important;
}

.task-progress {
    margin: 0;
}

.task-progress__label {
    color: var(--app-muted);
    font-size: 12px;
    margin-bottom: 5px;
}

.task-progress__track {
    height: 6px;
    overflow: hidden;
    border-radius: 4px;
    background: rgba(128, 128, 128, 0.15);
}

.task-progress__bar {
    height: 100%;
    border-radius: 4px;
}

.task-progress__bar--done {
    background: #22a06b;
}

.task-progress__bar--error {
    background: #dc2626;
}

.task-empty-state,
.task-action-message {
    border: 1px solid var(--app-border);
    border-radius: 8px;
    background: #f8fafc;
    color: var(--app-muted);
    padding: 10px 12px;
    font-size: 13px;
    line-height: 1.45;
}

.task-empty-state strong,
.task-action-message strong {
    display: block;
    color: var(--app-text);
    margin-bottom: 3px;
}

.task-action-message span {
    display: block;
    color: var(--app-muted);
    overflow-wrap: anywhere;
}

.kb-empty-state {
    border: 1px dashed #c7d2e1;
    border-radius: 8px;
    background: #f8fafc;
    color: #526071;
    padding: 12px 14px;
    margin: 0 0 12px;
}

.kb-empty-state strong {
    display: block;
    color: #172033;
    font-size: 14px;
    margin-bottom: 3px;
}

.kb-empty-state span {
    font-size: 13px;
    line-height: 1.4;
}

button.primary-action {
    font-weight: 700;
}

.gradio-container button.secondary,
.gradio-container button[class*="secondary"] {
    border-color: #9fb7d7 !important;
    background: #eef4ff !important;
    color: #173f7a !important;
    font-weight: 650 !important;
}

.gradio-container button.secondary:hover,
.gradio-container button[class*="secondary"]:hover {
    background: #dfeaff !important;
}

@media (max-width: 760px) {
    .app-header {
        padding: 16px;
    }

    .app-header__top {
        display: block;
    }

    .app-badges {
        justify-content: flex-start;
        min-width: 0;
        margin-top: 14px;
    }

    .kb-filter-panel {
        flex: 1 1 100% !important;
        min-width: 0 !important;
        max-width: none !important;
    }

    .kb-sidebar-rail {
        flex: 1 1 100% !important;
        min-width: 0 !important;
        max-width: none !important;
    }

    .kb-sidebar-rail-btn {
        min-width: 100% !important;
        width: 100% !important;
        min-height: 40px !important;
        writing-mode: horizontal-tb;
    }
}
"""

APP_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="sm",
    text_size="md",
)

APP_HEADER_HTML = ""

FORCE_LIGHT_THEME_HEAD = """
<script>
(() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get("__theme") !== "light") {
    url.searchParams.set("__theme", "light");
    window.location.replace(url.toString());
  }
})();
</script>
"""

KB_EMPTY_HTML = """
<div class="kb-empty-state">
    <strong>База знаний пока пуста</strong>
    <span>После обработки лекций здесь появятся документы, темы, транскрипции и конспекты.</span>
</div>
"""
