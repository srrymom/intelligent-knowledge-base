# Интеллектуальная база знаний аудиолекций

Дипломный прототип для обработки аудио- и видеолекций на локальной машине с ограниченной VRAM. Система распознаёт русскую речь, строит конспект, сохраняет лекцию в базе знаний и позволяет задавать вопросы по содержимому через RAG.

## Возможности

- загрузка аудио, видео или готового текста через Gradio UI;
- транскрибация русской речи через GigaAM;
- суммаризация через Ollama и `qwen2.5:3b`;
- методы суммаризации: `Map-Reduce`, `Hierarchical`, `Semantic-Cluster`;
- сохранение конспекта, тем и транскрипта в локальной базе знаний;
- семантический поиск и чат по сохранённым лекциям;
- мониторинг ASR/LLM-воркеров, GPU и Ollama;
- бенчмарки для ASR, суммаризации и RAG.

## Архитектура

Проект разделён на независимые процессы:

```text
gateway/app.py
  Gradio UI, загрузка файлов, база знаний, чат, мониторинг

asr/worker.py
  очередь медиафайлов -> GigaAM -> JSON-транскрипт

llm/worker.py
  JSON-транскрипт -> чанки -> конспект, темы, отчёт

rag/engine.py
  ChromaDB + multilingual-e5-small -> поиск и ответы по лекциям

storage/kb.py
  чтение, запись, удаление и экспорт записей базы знаний
```

Рабочие данные лежат в `data/` и не хранятся в git:

```text
data/
  queue/            входная очередь ASR
  transcript/       готовые транскрипты
  summary/          готовые конспекты
  knowledge_base/   сохранённые лекции
  rag_db/           индекс ChromaDB
  processed/        архив обработанных задач
  activity.log      журнал воркеров
```

Подробная диаграмма есть в [docs/architecture.md](docs/architecture.md).

## Быстрое восстановление репозитория

```bash
mkdir -p ~/projects
git clone https://github.com/srrymom/intelligent-knowledge-basee.git ~/projects/prototype
cd ~/projects/prototype
git submodule update --init --recursive
```

Проверка состояния:

```bash
git status --short --branch
git log --oneline -n 5
```

На момент восстановления актуальная ветка: `master`, актуальный удалённый репозиторий: `origin`.

## Требования

- Python 3.10+;
- NVIDIA GPU с CUDA, желательно 4 GB VRAM или больше;
- FFmpeg;
- Ollama;
- модель Ollama `qwen2.5:3b`.

Установить модель:

```bash
ollama pull qwen2.5:3b
```

Если FFmpeg не в `PATH`, задайте переменную окружения:

```bash
export FFMPEG_PATH=/path/to/ffmpeg/bin
```

## Установка в WSL/Linux

Основное окружение для UI, RAG и хранилища:

```bash
cd ~/projects/prototype
python -m venv gradio-env
source gradio-env/bin/activate
pip install -U pip
pip install -r requirements.txt
deactivate
```

ASR-воркер:

```bash
cd ~/projects/prototype/asr
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
deactivate
```

LLM-воркер:

```bash
cd ~/projects/prototype/llm
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
deactivate
```

## Запуск

```bash
cd ~/projects/prototype
source gradio-env/bin/activate
python gateway/app.py
```

Интерфейс откроется на `http://localhost:7860`. При старте приложение запускает ASR- и LLM-воркеры отдельными процессами и пытается поднять Ollama, если он доступен в системе.

## Конфигурация

Основные переменные окружения:

| Переменная | Значение по умолчанию | Назначение |
| --- | --- | --- |
| `DATA_DIR` | `./data` | рабочие данные проекта |
| `OLLAMA_URL` | `http://localhost:11434` | адрес Ollama API |
| `LLM_MODEL` | `qwen2.5:3b` | модель для суммаризации и QA |
| `LLM_NUM_CTX` | `2048` | размер контекстного окна |
| `ASR_MODEL` | `v3_e2e_rnnt` | модель GigaAM |
| `FFMPEG_PATH` | `D:\ffmpeg\bin` | путь к FFmpeg |

## Методы суммаризации

`Map-Reduce` суммаризирует чанки отдельно и объединяет их. При большом числе чанков автоматически уходит в иерархическое слияние, чтобы не переполнить контекст.

`Hierarchical` объединяет резюме деревом по группам, поэтому размер каждого LLM-запроса остаётся ограниченным. Это основной стабильный метод.

`Semantic-Cluster` строит эмбеддинги чанков, кластеризует их через KMeans и делает итоговые разделы по смысловым группам. При проблемах с зависимостями или коротком тексте откатывается к `Hierarchical`.

## Бенчмарки

```bash
cd ~/projects/prototype
source gradio-env/bin/activate
python benchmark/run_all.py
```

Отдельные сценарии:

```bash
python benchmark/eval_asr.py
python benchmark/eval_summary.py
python benchmark/eval_rag.py
python benchmark/compare_methods.py
python benchmark/cascade_analysis.py
```

Результаты сохраняются в `data/benchmark/results/`, сводный отчёт проекта лежит в [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

## Полезные файлы

- [shared/config.py](shared/config.py) - пути, модели и переменные окружения;
- [gateway/app.py](gateway/app.py) - точка входа UI;
- [asr/worker.py](asr/worker.py) - транскрибация;
- [llm/summarization.py](llm/summarization.py) - методы суммаризации;
- [rag/engine.py](rag/engine.py) - RAG и ChromaDB;
- [docs/architecture.md](docs/architecture.md) - схема компонентов.
