# Карта проекта — система обработки аудиолекций

## Что это

Веб-приложение на Gradio для обработки лекций: аудио → транскрипция → конспект + семантический поиск (RAG). Три вкладки: транскрипция, база знаний, чат.

Целевое железо: **NVIDIA RTX 3050, 4 GB VRAM**. Из-за этого ASR и LLM не работают одновременно — они делят одну GPU через файловый mutex.

---

## Структура каталогов

```
prototype/
├── gateway/        # UI + оркестрация (запускается через gateway/app.py)
├── asr/            # ASR-воркер (отдельный venv: asr/.venv)
├── llm/            # LLM-воркер (отдельный venv: llm/.venv)
├── rag/            # RAG-движок (ChromaDB + embedder)
├── shared/         # Конфиги, GPU-координация, логирование
├── storage/        # CRUD для базы знаний (JSON-файлы)
├── benchmark/      # Скрипты оценки качества
├── data/           # Все рабочие данные (создаётся автоматически)
│   ├── queue/          # Аудиофайлы ждут ASR-воркера
│   ├── transcript/     # Готовые транскрипты (UUID.json)
│   ├── summary/        # Прогресс-файлы (.progress, .asr.progress)
│   ├── knowledge_base/ # Финальные записи (UUID.json) — источник правды
│   ├── rag_db/         # ChromaDB (векторный индекс)
│   └── models/         # Кэш embedding-модели (e5-small)
└── gradio-env/     # venv для gateway (не путать с asr/.venv и llm/.venv)
```

---

## Три отдельных виртуальных окружения

| Окружение | Для чего | Запускает |
|-----------|----------|-----------|
| `gradio-env/` | UI, storage, rag, benchmark | `gateway/app.py` |
| `asr/.venv/` | GigaAM, torchcodec, torch | `asr/worker.py` |
| `llm/.venv/` | ollama, sentence-transformers, sklearn | `llm/worker.py` |

Это костыль: GigaAM конфликтует с gradio по зависимостям, поэтому ASR-воркер живёт в отдельном venv.

---

## Поток данных

```
Пользователь загружает файл
        ↓
gateway/handlers.py::save_media()
   → копирует в data/queue/UUID.ext
   → пишет data/transcript/UUID.meta (метод суммаризации, тип отчёта)
        ↓
asr/worker.py  (процесс-демон, смотрит queue/)
   → транскрибирует через GigaAM
   → пишет прогресс в data/summary/UUID.asr.progress
   → кладёт data/transcript/UUID.json  (список сегментов с таймкодами)
   → удаляет файл из queue/
        ↓
llm/worker.py  (процесс-демон, смотрит transcript/)
   → читает UUID.json + UUID.meta
   → суммаризирует через Ollama (Qwen2.5:3b)
   → пишет прогресс в data/summary/UUID.progress
   → кладёт data/knowledge_base/UUID.json  (полная запись: транскрипт + конспект + темы + отчёт)
   → удаляет UUID.json из transcript/
        ↓
gateway/handlers.py::poll_outputs()  (таймер 2 сек в UI)
   → видит knowledge_base/UUID.json → отдаёт результат в Gradio
   → попутно индексирует запись в ChromaDB если ещё не проиндексировано
```

Прямой путь для текстового ввода: `save_text()` пропускает ASR, сразу пишет в transcript/.

---

## Файлы по компонентам

### gateway/

**`app.py`** — точка входа. Запускает `asr/worker.py` и `llm/worker.py` как subprocess через `start_workers()`. Собирает Gradio-интерфейс из трёх вкладок. При старте вызывает `reset_gpu_state()` и `ensure_ollama_started()`.

**`handlers.py`** — логика кнопок и поллинга.
- `save_media(file_path, report_mode, sum_method)` — кладёт файл в очередь
- `save_text(text_input, text_file, ...)` — текст напрямую в transcript/
- `poll_outputs(file_uuid, mode)` — главная функция таймера, возвращает ровно 8 значений (`_N=8`), синхронизировать с app.py если менять
- `make_export_file(entry_id)` — генерирует .md файл для скачивания

**`formatting.py`** — HTML-рендеры. Чистые функции, никакого состояния.
- `is_valid_segment(text)` — фильтрует мусорные сегменты (⁇ артефакты GigaAM)
- `format_segments(segments, mode)` — "С временными метками" или сплошной текст
- `render_topics`, `render_word_stats`, `render_kb_stats` — HTML для UI
- `render_progress_bar`, `render_asr_progress`, `render_waiting_progress` — прогресс-бары

**`monitor.py`** — мониторинг, вызывается таймером каждые 5 сек.
- `format_status()` — строка со статусом GPU/CPU/Ollama/очередей для UI
- `get_worker_health()` — проверяет subprocess'ы воркеров, **перезапускает при падении**
- `register_workers(asr_proc, llm_proc)` — вызывается из app.py при старте

---

### asr/

**`worker.py`** — бесконечный цикл, смотрит `data/queue/`.
- Модель GigaAM загружается лениво при первом файле в очереди
- Выгружается после `ASR_IDLE_TIMEOUT_SEC` простоя (по умолчанию 45 сек) или если LLM-воркер просит GPU
- Перед загрузкой GigaAM выгружает Ollama из VRAM через HTTP-запрос (keep_alive=0)
- `LOCK_FILE` (data/asr.lock) существует пока модель в VRAM

**`GigaAM/`** — локальная копия библиотеки GigaAM от SberDevices. Русскоязычная ASR-модель, модификация v3_e2e_rnnt.

---

### llm/

**`summarization.py`** — вся логика суммаризации.

Метод `summarize_transcript(segments, method, ...)` — главная функция.

Поддерживаемые методы (REDUCE-фаза):
- `Hierarchical` — дерево слияний по 3, ceil(log₃(N)) уровней. Дефолт.
- `Map-Reduce` — плоское слияние, fallback на Hierarchical при > 4 чанках
- `Semantic-Cluster` — KMeans (k≈√N) на эмбеддингах пересказов → per-cluster stitch

Устаревшие (алиасы на Hierarchical):
- `Sequential`, `Coarse-to-Fine`, `STA` — в коде есть, в UI не показываются

Ключевые функции:
- `make_chunks(segments, limit)` — делит транскрипт по word count
- `summarize_chunk(chunk)` — MAP-фаза для одного чанка
- `dispatch_merge(method, summaries)` — выбирает REDUCE-стратегию
- `build_title(summary)`, `extract_topics(summary)` — отдельные LLM-вызовы
- `build_structured_report(summary)` — ещё один LLM-вызов, если запрошен
- `get_chunk_word_limit()` — считает лимит из context_length Ollama: `(ctx - 600) / 3`
- `llm_call(content)` — обёртка над ollama.chat с OOM-fallback на CPU и retry

**`worker.py`** — бесконечный цикл, смотрит `data/transcript/`.
- Читает UUID.meta для параметров (метод, тип отчёта)
- Пишет UUID.progress для прогресс-бара в UI
- Готовый результат → `data/knowledge_base/UUID.json`, потом удаляет UUID.json из transcript/
- GPU-координация зеркальная ASR-воркеру: уступает при запросе от ASR

---

### rag/

**`engine.py`** — ChromaDB + multilingual-e5-small + Ollama.
- Коллекция `knowledge_base` в `data/rag_db/`
- `index_entry(entry)` — чанкит транскрипт (300 слов, overlap 30), эмбеддит, кладёт в Chroma
- `ensure_indexed()` — при старте проверяет все KB-записи, индексирует пропущенные
- `is_indexed(entry_id)` — используется в handlers.py при получении результата
- `ask(question)` — embed вопроса → top-5 чанков → LLM с контекстом → ответ + источники
- `remove_entry(entry_id)` — при удалении записи из KB

Embedding модель: `intfloat/multilingual-e5-small`, кэш в `data/models/`. При повреждённом кэше — сносит и перекачивает.

---

### shared/

**`config.py`** — все константы. Создаёт директории при импорте.

Ключевые переменные:
```python
QUEUE_DIR       # data/queue/
TRANSCRIPT_DIR  # data/transcript/
SUMMARY_DIR     # data/summary/  (прогресс-файлы)
KB_DIR          # data/knowledge_base/
RAG_DB_DIR      # data/rag_db/
LOCK_FILE       # data/asr.lock
GPU_STATE_FILE  # data/gpu_state.json
ACTIVITY_LOG    # data/activity.log

LLM_MODEL       # "qwen2.5:3b" (env: LLM_MODEL)
LLM_NUM_CTX     # 2048 (env: LLM_NUM_CTX)
ASR_MODEL       # "v3_e2e_rnnt" (env: ASR_MODEL)
OLLAMA_URL      # "http://localhost:11434" (env: OLLAMA_URL)
FFMPEG_PATH     # "D:\ffmpeg\bin" (env: FFMPEG_PATH)

ASR_IDLE_TIMEOUT_SEC   # 45
LLM_IDLE_TIMEOUT_SEC   # 90
SUMMARIZATION_METHODS  # ["Map-Reduce", "Hierarchical", "Semantic-Cluster"]
```

**`gpu_coord.py`** — файловый mutex для GPU.

Функции:
- `request_gpu(worker)` — заявить что хочешь GPU
- `acquire_gpu(worker)` → bool — захватить если свободно
- `release_gpu(worker)` — освободить
- `clear_gpu_request(worker)` — снять заявку
- `read_gpu_state()` → `{"owner": "asr"|"llm"|None, "requests": {...}}`
- `reset_gpu_state()` — вызывается при старте UI чтобы сбросить зависший стейт

Состояние в `data/gpu_state.json`. Запись атомарная (tmp + os.replace).

**`log.py`** — append-лог в `data/activity.log`.
- `write_event(source, message)` — используется воркерами ("ASR", "LLM")
- `read_tail(n)` — UI читает последние строки для отображения
- Ротация при > 500 строк

**`ollama_runtime.py`**
- `is_ollama_available()` — HTTP-запрос к /api/ps
- `ensure_ollama_started()` — если Ollama не отвечает, запускает `ollama serve`

---

### storage/

**`kb.py`** — CRUD для `data/knowledge_base/`.

Формат записи (UUID.json):
```json
{
  "id": "uuid",
  "created_at": "2024-01-01T12:00:00",
  "title": "Название лекции",
  "segments": [{"transcription": "...", "boundaries": [0.0, 5.2]}],
  "summary": "Конспект...",
  "topics": ["тема 1", "тема 2"],
  "structured_report": "## Заголовок\n...",
  "sum_method": "Hierarchical"
}
```

Функции:
- `load_kb_list(topic_filter)` → (table_data, uuid_list)
- `load_kb_entry(entry_id)` → dict
- `delete_kb_entry(entry_id)` — удаляет JSON + чистит ChromaDB
- `get_kb_stats()` → (count, total_words, total_audio_sec)
- `get_all_topics()` → список тем по частоте

---

### benchmark/

Все скрипты запускаются из корня проекта. Результаты в `data/benchmark/results/`.

| Файл | Что делает | Как запускать |
|------|-----------|----------------|
| `metrics.py` | WER, CER, faithfulness, term_coverage, compression_ratio, P@K, MRR | импортируется другими |
| `eval_asr.py` | GigaAM на датасете Golos | из `asr/.venv` |
| `eval_summary.py` | метрики суммаризации на KB-записях | из `gradio-env` |
| `eval_rag.py` | RAG через синтетические QA-пары | из `gradio-env` |
| `compare_asr.py` | GigaAM vs Whisper medium/large | из `asr/.venv` |
| `compare_methods.py` | сравнение методов суммаризации | из `gradio-env` |
| `cascade_analysis.py` | деградация суммаризации от WER | из `gradio-env` |
| `run_all.py` | запускает всё, генерирует BENCHMARK_REPORT.md | из `gradio-env` |

**`install_golos.py`** — одноразовый скрипт, скачивает 50 wav из Golos для бенчмарка. Запускать один раз из `gradio-env`.

---

## Зависимости между файлами

```
app.py
  ├── handlers.py        (poll_outputs, save_media, save_text)
  ├── formatting.py      (render_*, format_segments)
  ├── monitor.py         (format_status, register_workers)
  ├── storage/kb.py      (load_kb_list, load_kb_entry, delete_kb_entry)
  ├── rag/engine.py      (ask, ensure_indexed)
  └── shared/
        ├── config.py
        ├── gpu_coord.py (reset_gpu_state)
        ├── log.py       (read_tail)
        └── ollama_runtime.py (ensure_ollama_started)

handlers.py
  ├── formatting.py
  ├── storage/kb.py      (load_kb_entry)
  └── rag/engine.py      (index_entry, is_indexed)

asr/worker.py
  └── shared/ (config, gpu_coord, log)

llm/worker.py
  ├── llm/summarization.py
  └── shared/ (config, gpu_coord, log)

llm/summarization.py
  └── shared/ (config, ollama_runtime)

rag/engine.py
  └── shared/config.py

storage/kb.py
  └── rag/engine.py      (remove_entry при удалении)
```

---

## Нюансы и грабли

**Три venv — это важно.** asr/worker.py запускается через `asr/.venv/Scripts/python`, llm/worker.py — через `llm/.venv/Scripts/python`. Если запустить через gradio-env — упадёт на import torchcodec или gigaam.

**poll_outputs возвращает ровно 8 значений (_N=8).** Если добавить новый Gradio-компонент и поменять _N, надо синхронизировать в handlers.py и app.py одновременно.

**asr.lock** — признак что GigaAM в VRAM. handlers.py смотрит на него чтобы показывать "Транскрипция аудио..." вместо "Ожидание GPU". Если воркер упал не по-хорошему, lock может остаться и UI зависнет. monitor.py при перезапуске воркера освобождает GPU-стейт.

**knowledge_base/UUID.json — источник правды.** Как только файл появился — запись готова. LLM-воркер пишет напрямую туда, никакого shutil.move в UI. handlers.py просто poll'ит наличие файла.

**GPU_STATE_FILE** сбрасывается при каждом запуске app.py через `reset_gpu_state()`. Это защита от зависшего стейта при аварийном завершении.

**FFMPEG_PATH** захардкожен на `D:\ffmpeg\bin`. Если разворачивать на другой машине — менять в config.py или через env.

**Semantic-Cluster** требует `sentence-transformers` и `sklearn` в llm/.venv. Если не установлены — тихо деградирует до Hierarchical.
