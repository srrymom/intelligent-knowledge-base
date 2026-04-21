"""
Все пути и константы проекта: директории, модели, таймауты, URL. Создаёт рабочие директории при импорте -- просто import shared.config уже достаточно для инициализации.
"""

import os

# Корень проекта: директория на уровень выше shared/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Корень для всех рабочих данных. Можно переопределить через DATA_DIR=...
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# Очередь файлов для транскрипции: asr/worker.py забирает файлы отсюда
QUEUE_DIR = os.path.join(DATA_DIR, "queue")

# Готовые транскрипты (.json): llm/worker.py читает отсюда, gateway/handlers.py тоже
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcript")

# Готовые резюме (.json): gateway/handlers.py забирает результат отсюда
SUMMARY_DIR = os.path.join(DATA_DIR, "summary")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

# База знаний: полные записи (транскрипт, конспект, темы), каждая как UUID.json.
# Используется UI для отображения и экспорта. ChromaDB хранит отдельно только эмбеддинги.
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")

# ChromaDB: векторный индекс чанков транскрипта для семантического поиска (RAG).
# Не дублирует KB — хранит только тексты чанков + эмбеддинги для collection.query()
RAG_DB_DIR = os.path.join(DATA_DIR, "rag_db")

# Lock-файл: ASR-воркер создаёт его пока занят, удаляет когда освободил VRAM
LOCK_FILE = os.path.join(DATA_DIR, "asr.lock")

# Состояние владения GPU между ASR и LLM-воркерами
GPU_STATE_FILE = os.path.join(DATA_DIR, "gpu_state.json")

# Архив: обработанные транскрипты и резюме перемещаются сюда после сохранения в KB
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Журнал активности воркеров (ASR + LLM), отображается в интерфейсе
ACTIVITY_LOG = os.path.join(DATA_DIR, "activity.log")

# URL Ollama API. По умолчанию — локальный экземпляр
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Модель Ollama для суммаризации и QA. qwen2.5:3b укладывается в 4 ГБ VRAM
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

# Размер контекстного окна LLM (в токенах).
# Влияет на размер чанка: chunk_word_limit ≈ (LLM_NUM_CTX - 600) / 3
# При 2048: ~480 слов/чанк → ~40 вызовов на лекцию 20 000 слов
# При 4096: ~1160 слов/чанк → ~17 вызовов (доп. VRAM ~30-50 МБ — незначительно)
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "2048"))

# Модель GigaAM для транскрипции. v3_e2e_rnnt — основная русскоязычная модель
ASR_MODEL = os.environ.get("ASR_MODEL", "v3_e2e_rnnt")

# Сколько держать модель в памяти после последней задачи, если другой этап не просит GPU
ASR_IDLE_TIMEOUT_SEC = int(os.environ.get("ASR_IDLE_TIMEOUT_SEC", "45"))
LLM_IDLE_TIMEOUT_SEC = int(os.environ.get("LLM_IDLE_TIMEOUT_SEC", "90"))

# Путь к бинарникам ffmpeg (нужен для конвертации видео и нестандартных аудиоформатов)
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", r"D:\ffmpeg\bin")

# Публично оставляем только самые предсказуемые режимы.
# Сложные/экспериментальные методы ниже в коде могут встречаться в старых meta/KB записях,
# но новые задачи UI больше не предлагает.
DEFAULT_SUMMARIZATION_METHOD = "Hierarchical"
SUMMARIZATION_METHODS = ["Map-Reduce", "Hierarchical", "Semantic-Cluster"]

BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
BENCHMARK_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")

# Создать рабочие директории если не существуют
for _d in [QUEUE_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, EXPORT_DIR, KB_DIR, RAG_DB_DIR, BENCHMARK_RESULTS_DIR, PROCESSED_DIR]:
    os.makedirs(_d, exist_ok=True)
