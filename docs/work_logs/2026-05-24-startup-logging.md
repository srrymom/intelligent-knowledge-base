# Логирование запуска и диагностика зависаний

Была проблема: при запуске `python gateway\app.py` терминал мог долго ничего не показывать. Было непонятно, завис ли проект на импортах, Ollama, запуске воркеров или сборке Gradio UI.

Сделали:

- добавили общий runtime-логгер в `shared/log.py`;
- добавили `shared/resources.py` для компактных снимков CPU/RAM/GPU/VRAM и RAM текущего процесса;
- события `write_event()` теперь пишутся не только в `data/activity.log`, но и в консоль;
- ключевые события UI, ASR, LLM и RAG теперь дополнительно пишут строку `resources: ...`;
- `gateway/app.py` теперь логирует Python, платформу, рабочие папки, FFmpeg/Ollama в `PATH`, старт Ollama, старт ASR/LLM-воркеров, сборку UI и запуск Gradio;
- `gateway/handlers.py` теперь логирует отправку медиа и текста в очередь;
- журнал активности в UI заменен с маленького `Textbox` на скроллируемый HTML-блок с моноширинным выводом;
- UI теперь показывает хвост примерно из 120 строк лога вместо 10;
- вкладка "База знаний" больше не использует псевдотаблицу с выделением ячеек;
- добавлены два режима работы базы знаний: `Просмотр` для переключения по записям и удаления текущей записи, `Выбор` для выбора нескольких записей и пакетного удаления;
- интерфейс базы знаний визуально разделен на левую панель поиска/выбора и правую панель чтения;
- правая панель чтения сделана сворачиваемой через `Accordion`;
- если не найдены `asr/.venv` или `llm/.venv`, приложение явно пишет ошибку и падает с понятным сообщением;
- если `gateway/app.py` запущен не из `gradio-env` и не найден `gradio`, выводится подсказка по активации окружения;
- поправлен `sys.path` для `gateway`, чтобы приложение можно было импортировать в диагностических проверках;
- `demo.launch()` запускается с `show_error=True`;
- в протокол задач добавлено правило: задачи на работоспособность должны улучшать диагностируемость.

Проверили:

- `python -m py_compile gateway/app.py gateway/handlers.py shared/log.py`
- `python -c "import gateway.app; print('import OK')"` на системном Python показал `ModuleNotFoundError: No module named 'gradio'`, то есть текущий терминал не использует `gradio-env`.
- `.\gradio-env\Scripts\python.exe -c "import gateway.app; print(gateway.app.format_activity_log(3)[:80]); print('import OK')"`
- `.\gradio-env\Scripts\python.exe -c "from gateway.app import refresh_kb_table, filter_kb_table; r=refresh_kb_table(); f=filter_kb_table('', None); print(len(r), len(f))"`
- `.\gradio-env\Scripts\python.exe -c "import gateway.app as a; print(len(a.on_kb_mode_switch('Просмотр')), len(a.on_kb_mode_switch('Выбор'))); print(len(a.refresh_kb_table()))"`
- `.\gradio-env\Scripts\python.exe -c "import gradio as gr, inspect; print(inspect.signature(gr.Row)); print(inspect.signature(gr.Column))"`
- `python -m py_compile shared/resources.py shared/log.py gateway/app.py gateway/handlers.py rag/engine.py asr/worker.py llm/worker.py`
- `.\gradio-env\Scripts\python.exe -c "from shared.resources import format_resource_snapshot; print(format_resource_snapshot())"`

Итог: при запуске `python gateway\app.py` в терминале должны появляться строки вида `UI: INFO ...`, а события воркеров `ASR:` и `LLM:` должны быть видны и в терминале, и в `data/activity.log`.

Для отчета:

Улучшена сопровождаемость прототипа: добавлено централизованное логирование запуска, воркеров и критических проверок окружения, что упрощает диагностику зависаний и ошибок при локальном тестировании.
