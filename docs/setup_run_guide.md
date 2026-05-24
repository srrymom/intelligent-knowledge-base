# Установка и запуск проекта

Инструкция рассчитана на текущую архитектуру проекта: три виртуальных окружения, локальный Ollama, ASR-воркер GigaAM и основной Gradio-интерфейс.

## 0. Что нужно заранее

Минимум:

- Python 3.10 или новее;
- Git;
- FFmpeg;
- Ollama;
- модель Ollama `qwen2.5:3b`;
- желательно NVIDIA GPU с CUDA и 4 GB VRAM или больше.

Проект можно частично использовать без GPU, но ASR и LLM будут работать медленнее или потребуют ручной настройки.

## 1. Системные зависимости

Для WSL/Ubuntu:

```bash
sudo apt update
sudo apt install -y git ffmpeg python3 python3-venv python3-pip
```

Проверка:

```bash
python3 --version
ffmpeg -version
```

Для Windows нужно установить FFmpeg отдельно и добавить папку `bin` в `PATH`, либо задать переменную:

```powershell
$env:FFMPEG_PATH="D:\ffmpeg\bin"
```

В WSL/Linux обычно достаточно, чтобы `ffmpeg` был доступен из `PATH`.

## 2. Ollama и модель

Установить Ollama обычным способом для своей ОС, затем скачать модель:

```bash
ollama pull qwen2.5:3b
ollama list
```

Можно отдельно проверить сервер:

```bash
ollama serve
```

Если `ollama serve` уже запущен как сервис, второй раз запускать не нужно. Приложение само пытается поднять Ollama, если бинарник доступен.

## 3. Репозиторий

Если проект уже есть:

```bash
cd /home/emiro/projects/prototype
```

Если ставить заново:

```bash
mkdir -p ~/projects
git clone <repo-url> ~/projects/prototype
cd ~/projects/prototype
git submodule update --init --recursive
```

Если `asr/GigaAM` уже лежит в репозитории, submodule-команда может ничего не изменить.

## 4. Основное окружение `gradio-env`

Это окружение запускает UI, RAG, базу знаний и часть benchmark-скриптов.

```bash
cd /home/emiro/projects/prototype
python3 -m venv gradio-env
source gradio-env/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
deactivate
```

Проверка:

```bash
cd /home/emiro/projects/prototype
source gradio-env/bin/activate
python - <<'PY'
import gradio, chromadb, ollama, sentence_transformers, httpx, psutil
print("gradio-env OK")
PY
deactivate
```

## 5. ASR-окружение `asr/.venv`

Это окружение нужно для GigaAM и транскрипции.

```bash
cd /home/emiro/projects/prototype/asr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
deactivate
```

Проверка импортов:

```bash
cd /home/emiro/projects/prototype/asr
source .venv/bin/activate
python - <<'PY'
import torch
import torchcodec
import gigaam
print("asr env OK")
print("cuda:", torch.cuda.is_available())
PY
deactivate
```

Если GigaAM/VAD попросит доступ к `pyannote/segmentation-3.0`, нужно принять условия модели на Hugging Face и задать токен:

```bash
export HF_TOKEN="hf_..."
```

Для постоянной настройки можно добавить эту строку в `~/.bashrc`.

## 6. LLM-окружение `llm/.venv`

Это окружение запускает LLM-воркер суммаризации.

```bash
cd /home/emiro/projects/prototype/llm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
deactivate
```

Проверка:

```bash
cd /home/emiro/projects/prototype/llm
source .venv/bin/activate
python - <<'PY'
import ollama
import sentence_transformers
import sklearn
import numpy
print("llm env OK")
PY
deactivate
```

`sentence-transformers`, `scikit-learn` и `numpy` нужны для метода `Semantic-Cluster`. Без них этот метод откатывается к `Hierarchical`.

## 7. Переменные окружения

Минимальный набор обычно не нужен, но полезно явно задать параметры:

```bash
cd /home/emiro/projects/prototype
export DATA_DIR="$PWD/data"
export OLLAMA_URL="http://localhost:11434"
export LLM_MODEL="qwen2.5:3b"
export LLM_NUM_CTX="2048"
```

Если FFmpeg не найден автоматически:

```bash
export FFMPEG_PATH="/usr/bin"
```

Для Windows путь обычно похож на:

```powershell
$env:FFMPEG_PATH="D:\ffmpeg\bin"
```

## 8. Запуск приложения

Запускать нужно только основной UI. Он сам стартует ASR- и LLM-воркеры как subprocess.

```bash
cd /home/emiro/projects/prototype
source gradio-env/bin/activate
python gateway/app.py
```

Открыть в браузере:

```text
http://localhost:7860
```

Если Gradio выберет другой порт, он напечатает адрес в терминале.

## 9. Как проверить, что всё живо

1. Открой вкладку "Транскрипция".
2. Сначала проверь путь "Текст": вставь небольшой русский текст и нажми "Отправить".
3. Дождись конспекта.
4. Проверь, что запись появилась во вкладке "База знаний".
5. Задай вопрос во вкладке "Чат".
6. После текстового сценария проверяй аудио.

Текстовый путь полезен тем, что он пропускает ASR и сразу проверяет LLM, базу знаний и RAG.

## 10. Что сейчас не считать полностью готовым

Видео в UI есть, но текущий ASR-воркер не подхватывает `.mp4/.mkv/.avi/.mov` из очереди. Для диплома это нужно либо доработать через извлечение аудиодорожки FFmpeg, либо временно тестировать только аудио и текст.

`BENCHMARK_REPORT.md` сейчас не является реальным финальным отчётом качества. Для диплома нужно запускать новые эксперименты после подготовки корпуса.

## 11. Benchmark-зависимости

Для подготовки ASR-датасета Golos:

```bash
cd /home/emiro/projects/prototype
source gradio-env/bin/activate
pip install -r benchmark/requirements.txt
python install_golos.py
deactivate
```

ASR-оценку запускать из ASR-окружения:

```bash
cd /home/emiro/projects/prototype
source asr/.venv/bin/activate
python benchmark/eval_asr.py --n 10
deactivate
```

Оценку суммаризации и RAG запускать из основного окружения:

```bash
cd /home/emiro/projects/prototype
source gradio-env/bin/activate
python benchmark/eval_summary.py
python benchmark/eval_rag.py
deactivate
```

`benchmark/compare_methods.py` сейчас требует доработки под актуальный модуль `llm/summarization.py`, поэтому его результаты лучше не использовать как финальные.

## 12. Частые проблемы

### `ModuleNotFoundError: ollama`

Переустановить основное окружение или поставить пакет:

```bash
source gradio-env/bin/activate
pip install ollama
```

### `ModuleNotFoundError: torchcodec`

Поставить пакет в ASR-окружение:

```bash
cd asr
source .venv/bin/activate
pip install torchcodec
```

Если после этого конфликтует `torch`, нужно согласовать версии `torch`, `torchaudio`, `torchcodec`.

### `Semantic-Cluster` работает как `Hierarchical`

Проверить LLM-окружение:

```bash
cd llm
source .venv/bin/activate
pip install sentence-transformers scikit-learn numpy
```

### Ollama не отвечает

Проверить:

```bash
ollama list
ollama serve
```

И в другом терминале:

```bash
curl http://localhost:11434/api/ps
```

### Не хватает VRAM

Оставить `LLM_NUM_CTX=2048`, закрыть лишние GPU-процессы и сначала проверять короткие тексты. Система пытается выгружать модели и координировать GPU, но 4 GB VRAM всё равно остаётся жёстким ограничением.

