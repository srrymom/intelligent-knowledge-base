# План нормального разрезания `gateway/app.py`

Сейчас `gateway/app.py` смешивает сразу несколько ответственностей: запуск runtime-части, CSS/тему, сборку трёх вкладок, обработчики событий, KB/RAG-логику и вспомогательные форматтеры. Это нормально для прототипа, но плохо для дальнейшей разработки: любое маленькое изменение UI начинает цеплять запуск воркеров и бизнес-логику.

## Целевая структура

```text
gateway/
  app.py                         # только точка входа: import launch_ui; launch_ui()
  runtime.py                     # proxy, логирование окружения, start_workers()
  ui/
    __init__.py
    layout.py                    # launch_ui(), общий state, Tabs, Timer
    theme.py                     # APP_CSS, APP_THEME, APP_HEADER_HTML, KB_EMPTY_HTML
    tabs/
      __init__.py
      transcription.py           # build_transcription_tab()
      knowledge_base.py          # build_kb_tab()
      chat.py                    # build_chat_tab()
    events/
      __init__.py
      transcription_events.py    # wire_transcription_events()
      knowledge_base_events.py   # wire_kb_events()
      chat_events.py             # wire_chat_events()
  services/
    __init__.py
    knowledge_base.py            # refresh/filter/select/delete KB
    chat.py                      # chat_respond(), on_source_select(), _load_rag_engine()
```

## Минимальный будущий `gateway/app.py`

```python
from runtime import ensure_localhost_bypasses_proxy
from ui.layout import launch_ui

if __name__ == "__main__":
    ensure_localhost_bypasses_proxy()
    launch_ui()
```

## Порядок миграции без поломки

1. Сначала вынести только `APP_CSS`, `APP_THEME`, `APP_HEADER_HTML`, `KB_EMPTY_HTML` в `gateway/ui/theme.py`.
2. Потом вынести `start_workers()`, `log_startup_environment()`, `_ensure_localhost_bypasses_proxy()` в `gateway/runtime.py`.
3. Потом вынести `build_transcription_tab()`, `build_kb_tab()`, `build_chat_tab()` в отдельные файлы `gateway/ui/tabs/`.
4. Потом вынести `wire_transcription_events()`, `wire_kb_events()`, `wire_chat_events()` в `gateway/ui/events/`.
5. В самом конце вынести KB/RAG функции в `gateway/services/`, потому что у них больше всего зависимостей от `storage`, `rag`, `handlers`, `formatting`.

## Что важно не менять при первом разрезании

- Не менять имена ключей в словарях `t`, `kb`, `chat`.
- Не менять порядок outputs в обработчиках Gradio.
- Не менять импорты из `shared`, `storage`, `handlers`, `formatting`, пока файл просто режется на модули.
- После каждого шага запускать хотя бы синтаксическую проверку:

```bash
python -m py_compile gateway/app.py
```

И затем ручной smoke-тест:

```bash
python gateway/app.py
```

## Промпт для Codex

```text
Нужно безопасно отрефакторить gateway/app.py без изменения поведения приложения.
Делай маленькими коммитами/патчами.
Сначала вынеси APP_CSS, APP_THEME, APP_HEADER_HTML, KB_EMPTY_HTML в gateway/ui/theme.py.
Потом вынеси runtime-функции в gateway/runtime.py.
Потом вынеси build_*_tab в gateway/ui/tabs/*.py.
Потом вынеси wire_*_events в gateway/ui/events/*.py.
Словари компонентов и порядок Gradio outputs не менять.
После каждого шага проверяй python -m py_compile gateway/app.py.
Кириллицу в файлах не повреждать, сохранять UTF-8.
```
