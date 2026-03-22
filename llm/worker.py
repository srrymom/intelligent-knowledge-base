import os
import re
import sys
import time
import glob
import json
import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import TRANSCRIPT_DIR, SUMMARY_DIR, LOCK_FILE, LLM_MODEL, LLM_NUM_CTX, OLLAMA_URL
from shared.log import write_event

CHUNK_WORD_LIMIT_DEFAULT = 500

SYSTEM_PROMPT = (
    "Ты помощник для анализа транскриптов речи. "
    "Отвечай только на русском. Пиши кратко и по делу, без лишних слов."
)


def get_chunk_word_limit():
    try:
        resp = ollama.ps()
        if resp.models:
            ctx = resp.models[0].context_length
            input_tokens = ctx - 600
            limit = max(200, input_tokens // 3)
            print(f"  context_length={ctx}, chunk_word_limit={limit}")
            return limit
    except Exception:
        pass
    return CHUNK_WORD_LIMIT_DEFAULT


def llm_call(user_content, num_gpu=None):
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
                print("GPU OOM при вызове LLM, повтор на CPU (num_gpu=0)...")
                return llm_call(user_content, num_gpu=0)
        raise


def unload_from_vram():
    """Выгружает LLM из VRAM через Ollama API (keep_alive=0).
    Вызывается после обработки всей очереди, чтобы освободить VRAM для ASR."""
    try:
        import urllib.request
        import json as _json
        data = _json.dumps({"model": LLM_MODEL, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=8)
        print("Ollama: LLM выгружена из VRAM, VRAM свободна для ASR.")
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
    """Убирает артефакты, которые qwen2.5:3b копирует из форматированного промпта."""
    # Временные метки вида [00:21 — 00:56] или 00:21 — 00:56
    text = re.sub(r'\[?\d{2}:\d{2}\s*[—\-]\s*\d{2}:\d{2}\]?', '', text)
    # Служебные заголовки которые модель добавляет сама
    for prefix in ('Временные метки:', 'Контекст:', 'Фрагмент:', 'Резюме:', 'Краткое содержание:'):
        text = re.sub(rf'(?m)^{re.escape(prefix)}\s*', '', text)
    # Схлопнуть множественные пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _sequential_merge(chunk_summaries: list) -> str:
    """
    Последовательная суммаризация с накоплением: каждый вызов получает
    текущий конспект + новый фрагмент. O(N) вызовов, O(1) контекст на вызов.
    """
    if not chunk_summaries:
        return ""
    running = chunk_summaries[0]
    for new_chunk in chunk_summaries[1:]:
        running = clean_summary(llm_call(
            "Обнови конспект, добавив информацию из нового фрагмента. "
            "Сохрани все ключевые мысли, убери повторы. "
            "Пиши кратко и по делу.\n\n"
            f"Текущий конспект:\n{running}\n\n"
            f"Новый фрагмент:\n{new_chunk}"
        ))
    return running


def _hierarchical_merge(summaries: list, group_size: int = 3) -> str:
    """
    Иерархическая суммаризация Map-Reduce: дерево merge-вызовов по group_size штук.
    ceil(log_{group_size}(N)) уровней. Контекст каждого вызова ограничен: group_size × размер_пересказа.
    """
    while len(summaries) > 1:
        next_level = []
        for i in range(0, len(summaries), group_size):
            group = summaries[i:i + group_size]
            if len(group) == 1:
                next_level.append(group[0])
                continue
            joined = "\n\n---\n\n".join(
                f"Часть {j + 1}:\n{s}" for j, s in enumerate(group)
            )
            merged = clean_summary(llm_call(
                "Объедини эти части в один связный конспект. "
                "Сохрани все ключевые мысли, убери повторы:\n\n" + joined
            ))
            next_level.append(merged)
        summaries = next_level
    return summaries[0]


def _flat_merge(chunk_summaries: list) -> str:
    """Плоское слияние: все пересказы в один LLM вызов. Исходный метод Map-Reduce."""
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    joined = "\n\n---\n\n".join(
        f"Часть {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    return clean_summary(llm_call(
        "Объедини эти части резюме в одно связное резюме:\n\n" + joined
    ))


def write_progress(file_uuid, stage, current=0, total=0):
    path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "current": current, "total": total}, f)


def remove_progress(file_uuid):
    path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    if os.path.exists(path):
        os.remove(path)


def summarize_transcript(segments: list, word_limit: int = None, method: str = "Hierarchical") -> dict:
    """
    Суммаризирует транскрипт, возвращает {'summary': str, 'title': str, 'topics': list, 'structured_report': str}.
    Используется напрямую из benchmark скриптов.
    method: 'Map-Reduce' | 'Sequential' | 'Hierarchical'
    """
    chunk_word_limit = word_limit or get_chunk_word_limit()
    chunks = make_chunks(segments, chunk_word_limit)

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = fmt_chunk(chunk)
        summary = llm_call(
            "Перескажи содержание этого фрагмента транскрипта в 2–3 предложениях. "
            "Пиши только само содержание — без временных меток, заголовков и служебных меток. "
            "Временные метки в начале — только ориентир, не включай их в ответ.\n\n"
            + chunk_text
        )
        chunk_summaries.append(clean_summary(summary))

    if method == "Sequential":
        summary_text = _sequential_merge(chunk_summaries)
    elif method == "Hierarchical":
        summary_text = _hierarchical_merge(chunk_summaries)
    else:  # Map-Reduce (flat)
        summary_text = _flat_merge(chunk_summaries)

    title = llm_call(
        "Проанализируй текст и придумай для него краткое, техническое название для базы знаний. "
        "Заголовок должен четко отражать тему (о чем идет речь) и быть понятным без контекста. "
        "Избегай метафор и поэтичных названий. Ограничься 3–5 словами. "
        "Название должно содержать главные термины из текста. "
        "Ответь только названием, без кавычек:\n\n" + summary_text
    ).strip().strip('"«»')

    topics_raw = llm_call(
        "Выдели 5–7 ключевых тем из этого текста. "
        "Каждая тема — словосочетание из 2–4 слов (не одиночные слова). "
        "Примеры хорошего формата: «машинное обучение», «обработка естественного языка». "
        "Перечисли через запятую, строчными буквами, без нумерации:\n\n" + summary_text
    )
    topics = [t.strip().lower().strip(".,;:") for t in topics_raw.split(",") if t.strip()][:7]

    structured_report = clean_summary(llm_call(
        "На основе этого конспекта составь структурированный отчёт на русском. "
        "Раздели на 3–5 тематических разделов с заголовками (## Заголовок). "
        "В каждом разделе — маркированный список ключевых мыслей (- пункт). "
        "В конце добавь раздел '## Ключевые выводы' с 3–5 главными тезисами. "
        "Начинай сразу с первого заголовка, без вводных фраз.\n\n" + summary_text
    ))

    return {"summary": summary_text, "title": title, "topics": topics, "structured_report": structured_report}


if __name__ == "__main__":
    write_event("LLM", "Воркер запущен, ожидание транскрипций...")

    while True:
        if os.path.exists(LOCK_FILE):
            time.sleep(2)
            continue

        json_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.json"))
        processed_any = False

        for json_path in json_files:
            file_uuid = os.path.splitext(os.path.basename(json_path))[0]
            summary_path = os.path.join(SUMMARY_DIR, f"{file_uuid}.json")

            if os.path.exists(summary_path):
                continue

            write_event("LLM", f"Обрабатываю транскрипт: {os.path.basename(json_path)}")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    segments = json.load(f)

                chunk_word_limit = get_chunk_word_limit()
                chunks = make_chunks(segments, chunk_word_limit)
                write_event("LLM", f"Разбито на {len(chunks)} чанк(ов)")

                meta_path = os.path.join(TRANSCRIPT_DIR, f"{file_uuid}.meta")
                want_structured = False
                sum_method = "Hierarchical"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            want_structured = meta.get("structured", False)
                            sum_method = meta.get("sum_method", "Hierarchical")
                        os.remove(meta_path)
                    except Exception:
                        pass

                write_event("LLM", f"Метод суммаризации: {sum_method}")

                chunk_summaries = []
                total_steps = len(chunks) + (3 if want_structured else 2)
                for i, chunk in enumerate(chunks):
                    write_progress(file_uuid, "chunk", i + 1, total_steps)
                    write_event("LLM", f"Чанк {i + 1}/{len(chunks)}...")
                    chunk_text = fmt_chunk(chunk)

                    summary = llm_call(
                        "Перескажи содержание этого фрагмента транскрипта в 2–3 предложениях. "
                        "Пиши только само содержание — без временных меток, заголовков и служебных меток. "
                        "Временные метки в начале — только ориентир, не включай их в ответ.\n\n"
                        + chunk_text
                    )
                    chunk_summaries.append(clean_summary(summary))

                write_progress(file_uuid, "merge", len(chunks) + 1, total_steps)
                if sum_method == "Sequential":
                    write_event("LLM", "Последовательное слияние чанков...")
                    summary_text = _sequential_merge(chunk_summaries)
                elif sum_method == "Hierarchical":
                    write_event("LLM", "Иерархическое слияние чанков...")
                    summary_text = _hierarchical_merge(chunk_summaries)
                else:  # Map-Reduce (flat)
                    write_event("LLM", "Плоское слияние чанков (Map-Reduce)...")
                    summary_text = _flat_merge(chunk_summaries)

                write_progress(file_uuid, "title", len(chunks) + 1, total_steps)
                write_event("LLM", "Генерирую заголовок и темы...")
                title = llm_call(
                    "Проанализируй текст и придумай для него краткое, техническое название для базы знаний. "
                    "Заголовок должен четко отражать тему (о чем идет речь) и быть понятным без контекста. "
                    "Избегай метафор и поэтичных названий. Ограничься 3–5 словами. "
                    "Название должно содержать главные термины из текста. "
                    "Ответь только названием, без кавычек:\n\n" + summary_text
                ).strip().strip('"«»')

                topics_raw = llm_call(
                    "Выдели 5–7 ключевых тем из этого текста. "
                    "Каждая тема — словосочетание из 2–4 слов (не одиночные слова). "
                    "Примеры хорошего формата: «машинное обучение», «обработка естественного языка». "
                    "Перечисли через запятую, строчными буквами, без нумерации:\n\n" + summary_text
                )
                topics = [t.strip().lower().strip(".,;:") for t in topics_raw.split(",") if t.strip()][:7]

                if want_structured:
                    write_progress(file_uuid, "report", total_steps, total_steps)
                    write_event("LLM", "Структурирую отчёт...")
                    structured_report = clean_summary(llm_call(
                        "На основе этого конспекта составь структурированный отчёт на русском. "
                        "Раздели на 3–5 тематических разделов с заголовками (## Заголовок). "
                        "В каждом разделе — маркированный список ключевых мыслей (- пункт). "
                        "В конце добавь раздел '## Ключевые выводы' с 3–5 главными тезисами. "
                        "Начинай сразу с первого заголовка, без вводных фраз.\n\n" + summary_text
                    ))
                else:
                    structured_report = ""

            except Exception as e:
                write_event("LLM", f"Ошибка обработки: {e}")
                summary_text = f"Ошибка обработки: {e}"
                title = "Ошибка"
                topics = []
                structured_report = ""
                sum_method = "unknown"

            remove_progress(file_uuid)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": summary_text,
                    "title": title,
                    "topics": topics,
                    "structured_report": structured_report,
                    "sum_method": sum_method,
                }, f, ensure_ascii=False, indent=2)

            write_event("LLM", f"Саммари готово: «{title}» [{sum_method}]")
            processed_any = True

        # После обработки всей очереди выгружаем LLM из VRAM,
        # чтобы ASR мог загрузиться без конкуренции за память.
        if processed_any:
            write_event("LLM", "Выгружаю модель из VRAM...")
            unload_from_vram()

        time.sleep(2)
