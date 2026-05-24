# Аудит RAM/VRAM

Дата: 2026-05-24

## Была проблема

Нужно разобраться, что именно потребляет RAM и VRAM в проекте, почему работа с проектом и браузером становится тяжёлой, и где возможны конфликты между ASR, LLM и RAG.

## Диагностика

Проверены RAM, процессы, GPU и Ollama:

```bash
free -h
ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm,args --sort=-rss
nvidia-smi
ollama ps
ollama list
```

Фактическое состояние без активной модели:

- RAM: около `1.7 GiB used`, `5.8 GiB available`;
- swap: около `780 MiB used`;
- GPU: `NVIDIA GeForce GTX 1650`, всего `4096 MiB VRAM`;
- базовая VRAM без compute-моделей: примерно `930-1025 MiB`;
- `ollama ps` пустой, то есть Qwen не был загружен;
- главный RAM-потребитель в idle-состоянии: VS Code/Pylance, а не прототип.

## RAM-замеры импортов

Проверено через `/usr/bin/time -v`.

```bash
/usr/bin/time -v gradio-env/bin/python -c "import sys"
/usr/bin/time -v gradio-env/bin/python -c "import gradio"
/usr/bin/time -v gradio-env/bin/python -c "import torch"
/usr/bin/time -v gradio-env/bin/python -c "import sentence_transformers"
/usr/bin/time -v asr/.venv/bin/python -c "import torch"
```

Пиковый RSS:

- пустой Python в `gradio-env`: около `9 MiB`;
- `import gradio`: около `127 MiB`;
- `import torch` в `gradio-env`: около `602 MiB`;
- `import sentence_transformers`: около `890 MiB`;
- `import torch` в `asr/.venv`: около `603 MiB`.

Вывод: резкий рост RAM даёт не сам UI, а импорт ML-стека, особенно `torch` и `sentence-transformers`.

## Idle-воркеры

Проверено через:

```bash
/usr/bin/time -v timeout 6s asr/.venv/bin/python asr/worker.py
/usr/bin/time -v timeout 6s llm/.venv/bin/python llm/worker.py
```

Пиковый RSS:

- ASR worker без задач: около `12 MiB`;
- LLM worker без задач: около `45 MiB`.

Вывод: сами idle-воркеры лёгкие. Память начинает расти при загрузке моделей и ML-библиотек.

## Qwen/Ollama VRAM и RAM

Проверен короткий запрос:

```bash
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:3b","prompt":"Ответь одним словом: тест","stream":false,"options":{"num_predict":1}}'
```

Во время загрузки/ответа:

- VRAM выросла примерно до `3314-3319 MiB`;
- свободной VRAM осталось около `617-622 MiB`;
- `ollama runner` держал около `2.55 GiB RSS`;
- `ollama ps` после ответа показал `qwen2.5:3b`, размер около `2.6 GB`, размещение `10%/90% CPU/GPU`, контекст `4096`.

После выгрузки:

```bash
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:3b","keep_alive":0}'
```

VRAM вернулась примерно к `931 MiB`, `ollama ps` снова пустой.

Вывод: Qwen на GTX 1650 почти заполняет VRAM. После загрузки Qwen остаётся слишком мало места для параллельной GigaAM-модели.

## ASR/GigaAM

Локального кеша GigaAM сейчас не найдено:

```bash
find ~/.cache -maxdepth 3 -type d -iname '*giga*'
find ~/.cache -maxdepth 4 -type f -name '*.ckpt'
```

`asr/GigaAM/gigaam/__init__.py` показывает, что модель `v3_e2e_rnnt` скачивается в `~/.cache/gigaam` с CDN как `v3_e2e_rnnt.ckpt`, затем грузится через `torch.load(..., map_location="cpu")` и переносится на `cuda`, если CUDA доступна.

Фактический VRAM-замер GigaAM не делался, потому что модель ещё не закеширована и для замера пришлось бы скачивать большой checkpoint.

## Найденные риски

### 1. RAG мог загружать Qwen поверх ASR

`llm/worker.py` и `asr/worker.py` используют `shared/gpu_coord.py`, но `rag/engine.py` напрямую вызывал `ollama.chat()` без GPU-координации и без выгрузки модели.

Риск: пользователь задаёт вопрос в чате во время транскрипции, RAG загружает Qwen, VRAM резко растёт до ~3.3 GiB, ASR/GigaAM получает OOM или деградацию.

Сделано:

- в `rag/engine.py` добавлен GPU-slot `rag`;
- RAG ждёт, пока GPU не свободна от ASR/LLM;
- RAG вызывает `ollama.chat(..., keep_alive=0)`;
- после ответа всегда вызывает `unload_from_vram()` и `release_gpu("rag")`.

### 2. SentenceTransformer мог занимать VRAM

`SentenceTransformer(...)` без явного `device` может выбрать CUDA.

Сделано:

- в `rag/engine.py` embedding-модель теперь создаётся с `device="cpu"`;
- в `llm/summarization.py` метод `Semantic-Cluster` тоже создаёт `SentenceTransformer(..., device="cpu")`.

Это сохраняет VRAM для Qwen/GigaAM. RAM при RAG всё равно будет расти, но это лучше, чем конкуренция за 4 GB VRAM.

### 3. GPU-state раньше терял нестандартные request-ключи

`shared/gpu_coord.py` нормализовал только `asr` и `llm`, поэтому новый request `rag` не сохранялся при чтении состояния.

Сделано:

- `read_gpu_state()` теперь сохраняет дополнительные request-ключи, включая `rag`.

## Проверки

Синтаксис:

```bash
gradio-env/bin/python -m py_compile rag/engine.py llm/summarization.py shared/gpu_coord.py gateway/app.py gateway/handlers.py storage/kb.py
llm/.venv/bin/python -m py_compile llm/summarization.py shared/gpu_coord.py
asr/.venv/bin/python -m py_compile asr/worker.py shared/gpu_coord.py
```

Проверка GPU-state:

```text
request_gpu('rag') -> requests содержит rag=True
clear_gpu_request('rag') -> requests содержит rag=False
acquire_gpu('rag') -> owner='rag'
release_gpu('rag') -> owner=None
```

Проверка текущего состояния после выгрузки Qwen:

```text
VRAM: около 1000 MiB used / 2936 MiB free
ollama ps: пусто
```

## Итог

Основная причина нехватки VRAM: Qwen `qwen2.5:3b` после загрузки занимает примерно `2.3 GiB` сверх базовой VRAM WSL/desktop и оставляет около `600 MiB` свободной VRAM на GTX 1650.

Основная причина скачков RAM: импорт `torch` даёт около `600 MiB RSS`, импорт `sentence-transformers` даёт около `900 MiB RSS`, а `ollama runner` во время Qwen держит около `2.55 GiB RSS`.

После правок:

- RAG больше не должен грузить Qwen параллельно с ASR/LLM;
- RAG/semantic embeddings больше не должны занимать VRAM;
- Qwen после RAG-чата выгружается сразу;
- idle-воркеры остаются лёгкими.

## Что осталось

1. Сделать отдельный фактический замер GigaAM после скачивания `~/.cache/gigaam/v3_e2e_rnnt.ckpt`.
2. Проверить полный сценарий `текст -> LLM -> KB -> RAG` с мониторингом `nvidia-smi` во время работы.
3. Проверить полный аудио-сценарий после появления тестового аудио и кеша GigaAM.
4. Возможно добавить в UI отдельную строку `GPU owner: rag`, чтобы пользователь видел, что чат ждёт GPU.

## Для отчета

Проведён аудит потребления RAM/VRAM. Установлено, что idle-процессы проекта потребляют мало памяти, а основные пики создают загрузка Qwen через Ollama, импорт PyTorch/SentenceTransformers и потенциальная загрузка GigaAM. Для предотвращения конфликтов на GPU добавлена координация RAG-запросов через общий GPU-state, принудительный CPU-режим для embedding-моделей и немедленная выгрузка Qwen после RAG-чата.
