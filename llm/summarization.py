"""
Вся логика суммаризации текста через Ollama. Режет транскрипт на чанки, суммаризирует каждый, сливает по выбранной стратегии: Hierarchical (дерево слияний), Map-Reduce (плоское), Semantic-Cluster (KMeans + stitch).

Sequential и Coarse-to-Fine -- заглушки, алиасы на Hierarchical. Semantic-Cluster требует sentence-transformers и sklearn.
"""

import json
import math
import re
import time
from typing import Callable, Optional

import numpy as np
import ollama
from shared.ollama_runtime import is_ollama_available

from shared.config import (
    DEFAULT_SUMMARIZATION_METHOD,
    LLM_MODEL,
    LLM_NUM_CTX,
    OLLAMA_URL,
    SUMMARIZATION_METHODS,
)

# ─── Константы ────────────────────────────────────────────────────────────────

# Максимум слов в накопительном резюме (Sequential) — чтобы не переполнить контекст
_SEQ_MAX_RUNNING_WORDS = 500

# Flat merge переключается на hierarchical при большем числе чанков
_FLAT_MAX_CHUNKS = 4

# Дефолтный лимит слов на чанк если Ollama недоступна
CHUNK_WORD_LIMIT_DEFAULT = 500
LEGACY_METHOD_ALIASES = {
    "Sequential": "Hierarchical",
    "Coarse-to-Fine": "Hierarchical",
    "STA": "Hierarchical",
}

SYSTEM_PROMPT = (
    "Ты помощник для анализа транскриптов речи. "
    "Отвечай только на русском. Пиши кратко и по делу, без лишних слов."
)


def check_ollama_available() -> bool:
    """Возвращает True если Ollama запущена и отвечает на запросы."""
    return is_ollama_available()


def get_chunk_word_limit():
    """chunk_word_limit ≈ (context_length - 600) / 3.
    600 токенов резервируем под промпт и вывод."""
    try:
        resp = ollama.ps()
        if resp.models:
            ctx = resp.models[0].context_length
            limit = max(200, (ctx - 600) // 3)
            print(f"  context_length={ctx}, chunk_word_limit={limit}")
            return limit
    except Exception:
        pass
    return CHUNK_WORD_LIMIT_DEFAULT


def llm_call(user_content, num_gpu=None, _attempt=0):
    """Вызов LLM с OOM-fallback на CPU и retry при сетевых ошибках."""
    opts = {"temperature": 0.1, "num_ctx": LLM_NUM_CTX}
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            options=opts,
        )
        return response.message.content
    except Exception as e:
        err = str(e).lower()
        if "cudamalloc" in err or "out of memory" in err or "runner process has terminated" in err:
            if num_gpu != 0:
                print("GPU OOM — повтор на CPU (num_gpu=0)...")
                return llm_call(user_content, num_gpu=0)
            raise
        transient = ("connect", "eof", "reset", "timeout", "refused", "broken pipe")
        if _attempt < 2 and any(k in err for k in transient):
            wait = 2 ** _attempt
            print(f"LLM connection error (попытка {_attempt + 1}): {e}. Повтор через {wait}с...")
            time.sleep(wait)
            return llm_call(user_content, num_gpu=num_gpu, _attempt=_attempt + 1)
        raise


def unload_from_vram():
    """Выгружает LLM из VRAM (keep_alive=0) чтобы освободить память для ASR."""
    try:
        import urllib.request

        data = json.dumps({"model": LLM_MODEL, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=8)
        print("Ollama: LLM выгружена из VRAM.")
    except Exception as e:
        print(f"Ollama unload: {e} (игнорируем)")


def make_chunks(segments, chunk_word_limit):
    chunks, current, word_count = [], [], 0
    for seg in segments:
        text = seg["transcription"]
        if not text.replace("⁇", "").replace(" ", "").strip():
            continue
        w = len(text.split())
        if current and word_count + w > chunk_word_limit:
            chunks.append(current)
            current, word_count = [], 0
        current.append(seg)
        word_count += w
    if current:
        chunks.append(current)
    return chunks


def fmt_time(seconds):
    t = int(seconds)
    return f"{t // 60:02d}:{t % 60:02d}"


def fmt_chunk(segs):
    start = fmt_time(segs[0]["boundaries"][0])
    end = fmt_time(segs[-1]["boundaries"][1])
    text = " ".join(s["transcription"] for s in segs)
    return f"[{start} — {end}]\n{text}"


def clean_summary(text: str) -> str:
    """Убирает артефакты которые модель копирует из промпта."""
    text = re.sub(r"\[?\d{2}:\d{2}\s*[—\-]\s*\d{2}:\d{2}\]?", "", text)
    for prefix in (
        "Временные метки:",
        "Контекст:",
        "Фрагмент:",
        "Резюме:",
        "Краткое содержание:",
    ):
        text = re.sub(rf"(?m)^{re.escape(prefix)}\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def summarize_chunk(chunk: list) -> str:
    summary = llm_call(
        "Перескажи содержание этого фрагмента транскрипта в 2–3 предложения. "
        "Пиши только само содержание — без временных меток и служебных меток.\n\n"
        + fmt_chunk(chunk)
    )
    return clean_summary(summary)


def normalize_method(method: Optional[str]) -> str:
    if method in SUMMARIZATION_METHODS:
        return method
    if method in LEGACY_METHOD_ALIASES:
        return LEGACY_METHOD_ALIASES[method]
    return DEFAULT_SUMMARIZATION_METHOD


def _sequential_merge(chunk_summaries: list) -> str:
    """Накопительный refine. Running summary обрезается до _SEQ_MAX_RUNNING_WORDS
    чтобы не переполнить контекст на длинных лекциях."""
    if not chunk_summaries:
        return ""
    running = chunk_summaries[0]
    for new_chunk in chunk_summaries[1:]:
        words = running.split()
        if len(words) > _SEQ_MAX_RUNNING_WORDS:
            running = " ".join(words[:_SEQ_MAX_RUNNING_WORDS])
        running = clean_summary(llm_call(
            "Обнови конспект, добавив информацию из нового фрагмента. "
            "Сохрани все ключевые мысли, убери повторы. "
            "Итоговый конспект — не более 400 слов.\n\n"
            f"Текущий конспект:\n{running}\n\n"
            f"Новый фрагмент:\n{new_chunk}"
        ))
    return running


def _hierarchical_merge(summaries: list, group_size: int = 3) -> str:
    """Дерево слияний: ceil(log_3(N)) уровней, контекст каждого вызова ограничен."""
    while len(summaries) > 1:
        next_level = []
        for i in range(0, len(summaries), group_size):
            group = summaries[i:i + group_size]
            if len(group) == 1:
                next_level.append(group[0])
                continue
            joined = "\n\n---\n\n".join(f"Часть {j + 1}:\n{s}" for j, s in enumerate(group))
            merged = clean_summary(llm_call(
                "Объедини эти части в один связный конспект. "
                "Сохрани все ключевые мысли, убери повторы:\n\n" + joined
            ))
            next_level.append(merged)
        summaries = next_level
    return summaries[0]


def _flat_merge(chunk_summaries: list) -> str:
    """Плоское слияние. При > _FLAT_MAX_CHUNKS автоматически переключается
    на hierarchical чтобы не переполнить контекст."""
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    if len(chunk_summaries) > _FLAT_MAX_CHUNKS:
        return _hierarchical_merge(chunk_summaries)
    joined = "\n\n---\n\n".join(f"Часть {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries))
    return clean_summary(llm_call(
        "Объедини эти части резюме в одно связное резюме:\n\n" + joined
    ))


def _coarse_to_fine_merge(chunk_summaries: list) -> str:
    """Временная безопасная заглушка для сложного метода."""
    return _hierarchical_merge(chunk_summaries)


# ─── Semantic-Cluster ─────────────────────────────────────────────────────────

_SC_EMBEDDER_MODEL = "intfloat/multilingual-e5-small"
_SC_K_MIN, _SC_K_MAX = 2, 8


def _truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "..."


def _is_bad_output(text: str) -> bool:
    """Детектирует отказ LLM или петлю повторений."""
    low = text.lower()
    refusal = [r"\?\s*$", r"какую часть", r"если вы хотите",
               r"мне нужно видеть", r"предоставьте", r"уточните",
               r"не могу", r"хотите начать"]
    if any(re.search(p, low) for p in refusal):
        return True
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    if len(sentences) >= 2:
        seen: set = set()
        for s in sentences:
            if s in seen:
                return True
            seen.add(s)
    return False


def _boundary_sentences(text: str) -> tuple:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text[:150], text[-150:]
    return sentences[0], sentences[-1]


def _make_transition(text_a: str, text_b: str) -> str:
    """Генерирует 1-2 связующих предложения между двумя блоками."""
    _, last_a = _boundary_sentences(text_a)
    first_b, _ = _boundary_sentences(text_b)
    prompt = (
        "Напиши 1-2 связующих предложения-перехода между двумя частями лекции. "
        "Только переход, без пересказа содержания. Начинай сразу с текста.\n\n"
        f"Суть первого блока: {_truncate_to_words(text_a, 80)}\n"
        f"Конец первого блока: {last_a}\n\n"
        f"Суть второго блока: {_truncate_to_words(text_b, 80)}\n"
        f"Начало второго блока: {first_b}"
    )
    result = clean_summary(llm_call(prompt))
    return result if not _is_bad_output(result) else ""


def _cluster_stitch_merge(sums: list) -> str:
    """Stitch: оригинальные резюме сохраняются, между группами генерируются переходы."""
    if len(sums) == 1:
        return sums[0]

    if len(sums) <= 3:
        joined = "\n\n---\n\n".join(
            f"Часть {j+1}:\n{_truncate_to_words(s, 350 // len(sums))}"
            for j, s in enumerate(sums)
        )
        result = clean_summary(llm_call(
            "Объедини эти фрагменты в связный абзац-конспект. "
            "Начинай сразу с содержания, без вступлений:\n\n" + joined
        ))
        return result if not _is_bad_output(result) else " ".join(sums)

    groups = [sums[i:i + 3] for i in range(0, len(sums), 3)]
    group_texts = [" ".join(g) for g in groups]

    parts = [group_texts[0]]
    for i in range(1, len(groups)):
        transition = _make_transition(group_texts[i - 1], group_texts[i])
        if transition:
            parts.append(f"*{transition}*")
        parts.append(group_texts[i])

    return "\n\n".join(parts)


def _extract_cluster_topic(raw: str) -> str:
    first_line = next((l.strip() for l in raw.splitlines() if l.strip()), raw.strip())
    first_line = re.sub(
        r"^[\*_]*\s*(тема|ключевая тема|раздел)\s*[\*_]*\s*[:：]\s*",
        "", first_line, flags=re.IGNORECASE,
    ).strip().strip("«»\"'.,;:!?")
    words = first_line.split()
    return " ".join(words[:5]) if len(words) > 6 else (first_line or "Тема")


def _semantic_cluster_merge(chunk_summaries: list) -> str:
    """Embed → KMeans → per-cluster stitch. Fallback на Hierarchical если нет зависимостей."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError:
        print("sentence-transformers/sklearn не установлены — fallback на Hierarchical")
        return _hierarchical_merge(chunk_summaries)

    N = len(chunk_summaries)
    if N <= 3:
        return _hierarchical_merge(chunk_summaries)

    embedder = SentenceTransformer(_SC_EMBEDDER_MODEL)
    vectors = embedder.encode(chunk_summaries)

    k = max(_SC_K_MIN, min(_SC_K_MAX, round(math.sqrt(N))))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)

    clusters: dict = {}
    for idx, (label, summary) in enumerate(zip(labels, chunk_summaries)):
        clusters.setdefault(int(label), []).append((idx, summary))

    ordered = sorted(clusters.items(), key=lambda item: np.mean([i for i, _ in item[1]]))
    for _, members in ordered:
        members.sort(key=lambda x: x[0])

    sections = []
    for _, members in ordered:
        sums = [s for _, s in members]
        topic_sample = "\n\n".join(_truncate_to_words(s, 100) for s in sums[:3])
        topic_raw = llm_call(
            "Напиши ТОЛЬКО 3-5 слов — название темы. "
            "Никаких предложений, только слова:\n\n" + topic_sample
        )
        topic = _extract_cluster_topic(topic_raw)
        sections.append(f"## {topic}\n\n{_cluster_stitch_merge(sums)}")

    return "\n\n".join(sections)


def _parse_sta_clusters(raw: str, n: int) -> list:
    """Парсит JSON-кластеры от LLM, robust fallback на единый кластер."""
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            seen, valid = set(), []
            for c in data:
                if not (isinstance(c, dict) and "topic" in c and "indices" in c):
                    continue
                idxs = [
                    int(i) for i in c["indices"]
                    if isinstance(i, (int, float)) and 0 <= int(i) < n and int(i) not in seen
                ]
                seen.update(idxs)
                if idxs:
                    valid.append({"topic": str(c["topic"]), "indices": idxs})
            leftover = [i for i in range(n) if i not in seen]
            if leftover:
                valid.append({"topic": "Дополнительно", "indices": leftover})
            if valid:
                return valid
        except Exception:
            pass
    return [{"topic": "Содержание лекции", "indices": list(range(n))}]


def _sta_merge(chunk_summaries: list) -> str:
    """Временная безопасная заглушка для сложного метода."""
    return _hierarchical_merge(chunk_summaries)


def dispatch_merge(method: str, chunk_summaries: list) -> str:
    method = normalize_method(method)
    if method == "Hierarchical":
        return _hierarchical_merge(chunk_summaries)
    if method == "Map-Reduce":
        return _flat_merge(chunk_summaries)
    if method == "Semantic-Cluster":
        return _semantic_cluster_merge(chunk_summaries)
    return _flat_merge(chunk_summaries)


def build_title(summary_text: str) -> str:
    return llm_call(
        "Придумай краткое техническое название для базы знаний (3–5 слов). "
        "Главные термины из текста, без метафор. Только название, без кавычек:\n\n"
        + summary_text
    ).strip().strip('"«»')


def extract_topics(summary_text: str) -> list:
    topics_raw = llm_call(
        "Выдели 5–7 ключевых тем (словосочетания 2–4 слова, строчные буквы, через запятую):\n\n"
        + summary_text
    )
    return [t.strip().lower().strip(".,;:") for t in topics_raw.split(",") if t.strip()][:7]


def build_structured_report(summary_text: str) -> str:
    return clean_summary(llm_call(
        "Составь структурированный отчёт на русском. "
        "3–5 разделов с заголовками (## Заголовок), маркированные списки. "
        "В конце — '## Ключевые выводы' с 3–5 тезисами. "
        "Начинай сразу с первого заголовка:\n\n" + summary_text
    ))


def summarize_transcript(
    segments: list,
    word_limit: int = None,
    method: str = "Hierarchical",
    want_structured_report: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    event_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Суммаризирует транскрипт. Возвращает {summary, title, topics, structured_report}."""
    method = normalize_method(method)
    chunk_word_limit = word_limit or get_chunk_word_limit()
    chunks = make_chunks(segments, chunk_word_limit)
    if not chunks:
        raise ValueError("Не удалось разбить на чанки")

    total_steps = len(chunks) + (3 if want_structured_report else 2)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback("chunk", i + 1, total_steps)
        if event_callback:
            event_callback(f"Чанк {i + 1}/{len(chunks)}...")
        chunk_summaries.append(summarize_chunk(chunk))

    if progress_callback:
        progress_callback("merge", len(chunks) + 1, total_steps)
    if event_callback:
        event_callback(f"Merge ({method})...")
    try:
        summary_text = dispatch_merge(method, chunk_summaries)
    except Exception:
        summary_text = _hierarchical_merge(chunk_summaries)

    if progress_callback:
        progress_callback("title", len(chunks) + 2, total_steps)
    if event_callback:
        event_callback("Заголовок и темы...")
    try:
        title = build_title(summary_text)
    except Exception:
        title = "Без названия"
    try:
        topics = extract_topics(summary_text)
    except Exception:
        topics = []

    structured_report = ""
    if want_structured_report:
        if progress_callback:
            progress_callback("report", total_steps, total_steps)
        if event_callback:
            event_callback("Структурированный отчёт...")
        try:
            structured_report = build_structured_report(summary_text)
        except Exception:
            structured_report = ""

    return {
        "summary": summary_text,
        "title": title,
        "topics": topics,
        "structured_report": structured_report,
        "chunks": chunks,
        "method": method,
    }
