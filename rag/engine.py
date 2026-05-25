"""
RAG-движок: ChromaDB + multilingual-e5-small + Ollama. Индексирует лекции из KB по чанкам, отвечает на вопросы через семантический поиск + LLM.

Embedder грузится лениво. При повреждённом кэше модели -- сносит папку и перекачивает.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import KB_DIR, RAG_DB_DIR, LLM_MODEL, DATA_DIR, OLLAMA_URL
from shared.gpu_coord import acquire_gpu, clear_gpu_request, release_gpu, request_gpu
from shared.log import write_event, write_resource_event
from llm.summarization import unload_from_vram

import chromadb
import ollama

MODELS_CACHE = os.path.join(DATA_DIR, "models")
EMBEDDER_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDER_REPO_CACHE = os.path.join(
    MODELS_CACHE, "models--intfloat--multilingual-e5-small"
)
os.makedirs(MODELS_CACHE, exist_ok=True)

embedder = None

chroma_client = chromadb.PersistentClient(path=RAG_DB_DIR)
collection = chroma_client.get_or_create_collection("knowledge_base")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
RAG_GPU_WAIT_TIMEOUT_SEC = int(os.environ.get("RAG_GPU_WAIT_TIMEOUT_SEC", "30"))
OLLAMA_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "120"))


def _load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDER_MODEL_NAME,
        cache_folder=MODELS_CACHE,
        device="cpu",
    )


def _acquire_rag_gpu_slot(timeout_sec: int = RAG_GPU_WAIT_TIMEOUT_SEC) -> None:
    import time

    request_gpu("rag")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if acquire_gpu("rag"):
            clear_gpu_request("rag")
            return
        time.sleep(0.5)
    clear_gpu_request("rag")
    raise RuntimeError("GPU занята ASR/LLM, RAG-запрос не получил слот")


def _get_embedder():
    global embedder
    if embedder is not None:
        return embedder
    try:
        write_resource_event("RAG", "Перед загрузкой embedding-модели")
        embedder = _load_embedder()
        write_resource_event("RAG", "После загрузки embedding-модели")
        return embedder
    except OSError as e:
        if os.path.exists(EMBEDDER_REPO_CACHE):
            shutil.rmtree(EMBEDDER_REPO_CACHE, ignore_errors=True)
            embedder = _load_embedder()
            return embedder
        raise RuntimeError(f"Не удалось загрузить embedding-модель: {e}") from e


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def index_entry(entry):
    entry_id = entry["id"]

    try:
        collection.delete(where={"meeting_id": entry_id})
    except Exception:
        pass

    full_text = " ".join(seg["transcription"] for seg in entry.get("segments", []))
    if not full_text.strip():
        return

    chunks = _chunk_text(full_text)
    embeddings = _get_embedder().encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{entry_id}_{i}" for i in range(len(chunks))],
        metadatas=[
            {
                "meeting_id": entry_id,
                "title": entry.get("title", ""),
                "created_at": entry.get("created_at", ""),
            }
            for _ in chunks
        ],
    )


def remove_entry(entry_id):
    try:
        collection.delete(where={"meeting_id": entry_id})
    except Exception:
        pass


def is_indexed(entry_id: str) -> bool:
    try:
        results = collection.get(where={"meeting_id": entry_id}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def ensure_indexed():
    kb_files = [fname for fname in os.listdir(KB_DIR) if fname.endswith(".json")]
    if not kb_files:
        write_event("RAG", "База знаний пуста, индексировать нечего")
        return

    try:
        _get_embedder()
    except Exception as e:
        write_event("RAG", f"Embedding model unavailable: {e}")
        write_resource_event("RAG", "Ошибка загрузки embedding-модели")
        return

    existing = set()
    try:
        all_meta = collection.get()["metadatas"]
        for m in all_meta:
            existing.add(m["meeting_id"])
    except Exception:
        pass

    for fname in kb_files:
        entry_id = fname[:-5]
        if entry_id in existing:
            continue
        path = os.path.join(KB_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            index_entry(entry)
            write_event("RAG", f"Indexed: {entry.get('title', entry_id)}")
        except Exception as e:
            write_event("RAG", f"Ошибка индексации {fname}: {e}")

    write_event("RAG", f"Индекс готов ({collection.count()} чанков)")


SYSTEM_PROMPT = (
    "Ты помощник для анализа транскриптов речи. "
    "Отвечай только на русском. Пиши кратко и по делу, без лишних слов."
)


def ask(question):
    write_resource_event("RAG", "Перед RAG-запросом")
    if collection.count() == 0:
        return {
            "answer": "База знаний пока пуста или ещё не проиндексирована.",
            "sources": [],
        }

    q_vec = _get_embedder().encode([question]).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = "\n\n".join(documents)

    seen = set()
    sources = []
    for m in metadatas:
        mid = m["meeting_id"]
        if mid not in seen:
            seen.add(mid)
            sources.append({"id": mid, "title": m.get("title", "")})

    prompt = (
        "Используй только контекст ниже для ответа.\n\n"
        f"Контекст из базы знаний:\n{context}\n\n"
        f"Вопрос: {question}\nОтвет:"
    )

    _acquire_rag_gpu_slot()
    write_resource_event("RAG", "GPU захвачена для RAG-запроса")
    try:
        client = ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT_SEC)
        response = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
            keep_alive=0,
        )
        answer = response.message.content
    finally:
        unload_from_vram()
        release_gpu("rag")
        write_resource_event("RAG", "После RAG-запроса")

    return {"answer": answer, "sources": sources}
