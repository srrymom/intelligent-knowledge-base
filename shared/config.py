import os

# Project root = parent of shared/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

QUEUE_DIR = os.path.join(DATA_DIR, "queue")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcript")
SUMMARY_DIR = os.path.join(DATA_DIR, "summary")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
RAG_DB_DIR = os.path.join(DATA_DIR, "rag_db")
LOCK_FILE = os.path.join(DATA_DIR, "asr.lock")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ACTIVITY_LOG = os.path.join(DATA_DIR, "activity.log")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
# Размер контекста LLM. 2048 достаточно для суммаризации и экономит ~0.5 ГБ VRAM.
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "2048"))
ASR_MODEL = os.environ.get("ASR_MODEL", "v3_e2e_rnnt")

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", r"D:\ffmpeg\bin")

SUMMARIZATION_METHODS = ["Map-Reduce", "Sequential", "Hierarchical"]

BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
BENCHMARK_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")

# Ensure data directories exist
for _d in [QUEUE_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, KB_DIR, RAG_DB_DIR, BENCHMARK_RESULTS_DIR, PROCESSED_DIR]:
    os.makedirs(_d, exist_ok=True)
