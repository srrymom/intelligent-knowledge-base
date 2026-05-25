# Разделение `gateway/app.py` на модули

Была проблема: `gateway/app.py` смешивал запуск воркеров, тему/CSS, сборку вкладок, обработчики событий и сервисную логику KB/RAG. Из-за этого любые изменения интерфейса приходилось делать в одном большом файле.

Сделали:

- создали `gateway/runtime.py` для proxy-настройки, логгера, `start_workers()`, `log_startup_environment()` и `format_activity_log()`;
- создали `gateway/ui/theme.py` для `APP_CSS`, `APP_THEME`, `APP_HEADER_HTML`, `KB_EMPTY_HTML`;
- создали `gateway/ui/layout.py` для `launch_ui()`, общего state, вкладок и таймеров;
- вынесли вкладки в `gateway/ui/tabs/transcription.py`, `gateway/ui/tabs/knowledge_base.py`, `gateway/ui/tabs/chat.py`;
- вынесли привязку событий в `gateway/ui/events/transcription_events.py`, `gateway/ui/events/knowledge_base_events.py`, `gateway/ui/events/chat_events.py`;
- вынесли KB-функции в `gateway/services/knowledge_base.py`;
- вынесли RAG/chat-функции в `gateway/services/chat.py`;
- оставили `gateway/app.py` тонкой точкой входа: настройка `sys.path`, proxy, импорт `launch_ui()`, запуск и логирование критической ошибки.

Возникла проблема: при механическом переносе русские строки, прописанные прямо в генераторе, PowerShell передал с битой кодировкой, из-за чего временно сломался строковый литерал в `gateway/app.py`.

Сделали: переписали `gateway/app.py` вручную через патч и заменили диагностическое сообщение о Gradio на ASCII-текст, чтобы не зависеть от кодировки оболочки.

Проверили:

- `.\gradio-env\Scripts\python.exe -m py_compile gateway\app.py gateway\runtime.py gateway\ui\theme.py gateway\ui\layout.py gateway\ui\tabs\transcription.py gateway\ui\tabs\knowledge_base.py gateway\ui\tabs\chat.py gateway\ui\events\transcription_events.py gateway\ui\events\knowledge_base_events.py gateway\ui\events\chat_events.py gateway\services\knowledge_base.py gateway\services\chat.py`
- dry-run импорта: `app import ok`
- dry-run сборки дерева Gradio без запуска сервера и воркеров: `ui build ok`
- проверили размеры возвращаемых наборов KB: `refresh outputs 18`, `filter outputs 15`, `empty select outputs 12`.

Итог: `gateway/app.py` больше не является монолитом. Структура теперь ближе к плану из `docs/gateway_refactor_plan.md`, а UI, события и сервисные функции разнесены по отдельным файлам.

Для отчета:

Проведен структурный рефакторинг gateway-слоя: точка входа, runtime-запуск, тема интерфейса, вкладки, обработчики событий и сервисная логика KB/RAG разделены по модулям без изменения пользовательских сценариев.
