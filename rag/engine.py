import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import KB_DIR, RAG_DB_DIR, LLM_MODEL, DATA_DIR

import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# кеш модели хранится внутри проекта (data/models/)
MODELS_CACHE = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_CACHE, exist_ok=True)

embedder = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    cache_folder=MODELS_CACHE,
)

# персистентное хранилище векторов
chroma_client = chromadb.PersistentClient(path=RAG_DB_DIR)
collection = chroma_client.get_or_create_collection("knowledge_base")

CHUNK_SIZE = 300    # слов в чанке
CHUNK_OVERLAP = 30  # слов перекрытия


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # разбить текст на чанки с перекрытием
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def index_entry(entry):
    # добавить запись KB в векторный индекс
    entry_id = entry["id"]

    # удаляем старые чанки этой записи (на случай переиндексации)
    try:
        collection.delete(where={"meeting_id": entry_id})
    except Exception:
        pass

    # склеиваем текст из всех сегментов
    full_text = " ".join(
        seg["transcription"] for seg in entry.get("segments", [])
    )
    if not full_text.strip():
        return

    chunks = _chunk_text(full_text)
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{entry_id}_{i}" for i in range(len(chunks))],
        metadatas=[{
            "meeting_id": entry_id,
            "title": entry.get("title", ""),
            "created_at": entry.get("created_at", ""),
        } for _ in chunks],
    )


def remove_entry(entry_id):
    # удалить все чанки записи из индекса
    try:
        collection.delete(where={"meeting_id": entry_id})
    except Exception:
        pass


def is_indexed(entry_id: str) -> bool:
    """Проверяет есть ли запись в ChromaDB."""
    try:
        results = collection.get(where={"meeting_id": entry_id}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def ensure_indexed():
    # доиндексировать записи KB, которых ещё нет в ChromaDB
    # собираем id, которые уже есть в индексе
    existing = set()
    try:
        all_meta = collection.get()["metadatas"]
        for m in all_meta:
            existing.add(m["meeting_id"])
    except Exception:
        pass

    # проходим по файлам KB
    for fname in os.listdir(KB_DIR):
        if not fname.endswith(".json"):
            continue
        entry_id = fname[:-5]
        if entry_id in existing:
            continue
        path = os.path.join(KB_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            index_entry(entry)
            print(f"[RAG] indexed: {entry.get('title', entry_id)}")
        except Exception as e:
            print(f"[RAG] ошибка индексации {fname}: {e}")

    print(f"[RAG] индекс готов ({collection.count()} чанков)")


SYSTEM_PROMPT = (
    "Ты помощник для анализа транскриптов речи. "
    "Отвечай только на русском. Пиши кратко и по делу, без лишних слов."
)


def ask(question):
    # поиск по KB + генерация ответа
    q_vec = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)

    # уникальные источники (по id), сохраняя порядок
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

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    answer = response.message.content

    return {"answer": answer, "sources": sources}
