# Разгрузка проекта и IDE

Дата: 2026-05-24  
Проект: `/home/emiro/projects/prototype`

## Симптом

Проект начал тяжело работать вместе с браузером: VS Code/WSL заметно грузили CPU и память, параллельно в терминале шла установка зависимостей.

## Диагностика

Проверены процессы:

```bash
ps -eo pid,ppid,pcpu,pmem,rss,comm,args --sort=-%mem
```

Что было видно:

- `pylance` держал примерно 580-640 МБ RAM;
- `extensionHost` VS Code держал примерно 400 МБ RAM и заметно грузил CPU;
- шёл процесс `python -m pip install -r requirements.txt`, который дополнительно грузил CPU;
- `ollama serve` был запущен, но занимал немного памяти без активной модели.

Проверен размер проекта:

```bash
du -h -d 2 . | sort -h | tail -40
```

Главная находка: сам код небольшой, но внутри рабочей папки лежат большие виртуальные окружения:

- `gradio-env/` - около 5.5 ГБ;
- `asr/.venv/` - около 7.4 ГБ;
- `llm/.venv/` - около 5.1 ГБ;
- `llm/gradio-env/` - около 902 МБ, лишнее случайное окружение.

Отдельно обнаружено, что `gradio-env` и `llm/.venv` содержат тяжёлые пакеты `torch`, `transformers`, `sentence-transformers` и наборы `nvidia-*`. Это объясняет размер окружений и нагрузку при индексации IDE.

## Что сделано

1. Удалено лишнее окружение:

```bash
rm -rf llm/gradio-env
```

Оно не используется кодом: UI запускает LLM через `llm/.venv/bin/python`, а не через `llm/gradio-env`.

2. Добавлены локальные настройки VS Code:

```text
.vscode/settings.json
```

Настройки исключают из файлового наблюдателя, поиска и Pylance:

- все `.venv`;
- все `gradio-env`;
- `data`;
- `__pycache__`;
- кеши тестов и mypy;
- тяжёлые тесты и notebook внутри `asr/GigaAM`.

Также Pylance переведён в более лёгкий режим:

```json
"python.analysis.diagnosticMode": "openFilesOnly",
"python.analysis.indexing": false
```

Эффект полностью проявится после перезагрузки окна VS Code.

После добавления настроек Pylance был мягко перезапущен через `kill -TERM` старого процесса, чтобы он начал перечитывать проект уже с исключениями.

3. Облегчён старт UI в `gateway/app.py`, `gateway/handlers.py` и `storage/kb.py`.

До правки RAG-движок импортировался сразу при запуске приложения:

- импортировался `rag/engine.py`;
- вместе с ним импортировались `chromadb`, `sentence-transformers`;
- `sentence-transformers` обычно тянет `torch`.

Теперь RAG импортируется лениво только когда он реально нужен: при первом вопросе в чате, при индексации готовой записи или при удалении записи из базы знаний. Индексация базы знаний тоже перенесена с запуска приложения на момент первого chat-запроса.

4. Облегчён idle-старт ASR-воркера в `asr/worker.py`.

До правки ASR-воркер импортировал `torch` сразу при запуске, даже когда просто ждал файлов. Теперь `torch` импортируется только когда реально нужна транскрипция или выгрузка модели.

## Проверка

Синтаксис проверен:

```bash
python3 -m py_compile gateway/app.py asr/worker.py
gradio-env/bin/python -m py_compile gateway/app.py gateway/handlers.py storage/kb.py asr/worker.py
asr/.venv/bin/python -m py_compile asr/worker.py
```

Все проверки прошли без ошибок.

Дополнительно проверено, что импорт лёгких UI-модулей больше не подтягивает RAG и ML-стек:

```text
engine_loaded= False
sentence_transformers_loaded= False
torch_loaded= False
```

Проверено, что случайное окружение удалено:

```bash
find llm -maxdepth 2 -type d -name 'gradio-env' -print
```

Команда ничего не вывела.

## Текущее состояние

Остались крупные, но ожидаемые окружения:

- `gradio-env/`;
- `asr/.venv/`;
- `llm/.venv/`.

Они большие из-за ML-зависимостей. Особенно заметны `torch`, `transformers`, `sentence-transformers`, `pyannote.audio`, `torchaudio`, `torchcodec` и CUDA/NVIDIA-пакеты.

## Что делать дальше

1. Перезагрузить окно VS Code: `Developer: Reload Window`.

Это нужно, чтобы Pylance и file watcher точно перечитали `.vscode/settings.json`.

2. Не запускать повторный `pip install -r requirements.txt` во время работы с браузером, если машина начинает задыхаться. Лучше запускать установку отдельно или с пониженным приоритетом:

```bash
nice -n 10 gradio-env/bin/python -m pip install -r requirements.txt
```

3. Если проект всё ещё тяжёлый, следующий кандидат на оптимизацию - убрать дублирование `torch` между `gradio-env` и `llm/.venv` или перевести embedding/RAG на отдельный более лёгкий процесс.

4. Важное правило для этого проекта: не создавать новые venv внутри подпапок без необходимости. Сейчас рабочая схема такая:

- UI/RAG: `gradio-env`;
- ASR: `asr/.venv`;
- LLM: `llm/.venv`.
