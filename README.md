# Intelligent Knowledge Base

Прототип интеллектуальной базы знаний с автоматической транскрипцией аудио/видео, суммаризацией и RAG-чатом.

> **Статус:** ранний прототип, работает только в локальном окружении.

## Возможности

- **Транскрипция** — распознавание речи из аудио и видео (GigaAM ASR)
- **Суммаризация** — автоматическое резюме транскрипций (Ollama + Qwen 2.5)
- **База знаний** — хранение, просмотр и удаление записей
- **RAG-чат** — вопросы по базе знаний с указанием источников

## Архитектура

```
gateway/        — Gradio UI (веб-интерфейс)
asr/            — ASR-воркер (GigaAM, распознавание речи)
llm/            — LLM-воркер (Ollama, суммаризация)
rag/            — RAG-движок (поиск по базе знаний)
storage/        — работа с базой знаний (JSON-файлы)
shared/         — общая конфигурация
diarization/    — диаризация (определение спикеров)
image-generator/— генерация иконок (эксперимент)
data/           — рабочие данные (очередь, транскрипты, БЗ) — не в git
```

## Стек

- **ASR:** GigaAM (SberDevices)
- **LLM:** Qwen 2.5 3B через Ollama
- **RAG:** multilingual-e5-small + ChromaDB
- **UI:** Gradio
- **Язык:** Python

## Требования

- Python 3.10+
- [Ollama](https://ollama.com/) с моделью `qwen2.5:3b`
- FFmpeg
- ~4 GB RAM для моделей

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/<username>/intelligent-knowledge-base.git
cd intelligent-knowledge-base

# 2. Создать виртуальные окружения и установить зависимости
#    (у каждого воркера своё окружение)

# Основное окружение (gateway + rag + storage)
python -m venv gradio-env
source gradio-env/bin/activate        # Linux/Mac
# gradio-env\Scripts\activate         # Windows
pip install gradio chromadb sentence-transformers

# ASR-воркер
cd asr
python -m venv .venv
source .venv/bin/activate
pip install -r GigaAM/requirements.txt
cd ..

# LLM-воркер
cd llm
python -m venv .venv
source .venv/bin/activate
pip install requests
cd ..

# 3. Запустить Ollama
ollama pull qwen2.5:3b
ollama serve

# 4. Запустить приложение
python gateway/app.py
```

## Конфигурация

Переменные окружения (опционально):

| Переменная    | По умолчанию              | Описание                  |
|---------------|---------------------------|---------------------------|
| `OLLAMA_URL`  | `http://localhost:11434`  | Адрес Ollama              |
| `LLM_MODEL`   | `qwen2.5:3b`             | Модель для суммаризации   |
| `ASR_MODEL`   | `v3_e2e_rnnt`            | Модель GigaAM             |
| `FFMPEG_PATH` | `D:\ffmpeg\bin`           | Путь к FFmpeg             |
| `DATA_DIR`    | `./data`                  | Директория данных         |
