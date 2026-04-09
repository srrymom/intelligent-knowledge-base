import json
import re
import time
from typing import Callable, Optional

import ollama

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
        if "cudamalloc" in err or "out of memory" in err:
            if num_gpu != 0:
                print("GPU OOM — повтор на CPU (num_gpu=0)...")
                return llm_call(user_content, num_gpu=0)
            raise
        transient = ("connection", "eof", "reset", "timeout", "refused", "broken pipe")
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
