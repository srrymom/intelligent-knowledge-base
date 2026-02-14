import os
import sys
import time
import glob
import json
import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import TRANSCRIPT_DIR, SUMMARY_DIR, LOCK_FILE, LLM_MODEL

CHUNK_WORD_LIMIT_DEFAULT = 500

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


SYSTEM_PROMPT = (
    "Ты помощник для анализа транскриптов речи. "
    "Отвечай только на русском. Пиши кратко и по делу, без лишних слов."
)


def llm_call(user_content):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.1},
    )
    return response.message.content


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


def write_progress(file_uuid, stage, current=0, total=0):
    path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "current": current, "total": total}, f)


def remove_progress(file_uuid):
    path = os.path.join(SUMMARY_DIR, f"{file_uuid}.progress")
    if os.path.exists(path):
        os.remove(path)


print("LLM воркер запущен, ожидает транскрипций...")

while True:
    if os.path.exists(LOCK_FILE):
        print("ASR is running, waiting...")
        time.sleep(2)
        continue

    json_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.json"))

    for json_path in json_files:
        file_uuid = os.path.splitext(os.path.basename(json_path))[0]
        summary_path = os.path.join(SUMMARY_DIR, f"{file_uuid}.json")

        if os.path.exists(summary_path):
            continue

        print(f"Summarizing: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                segments = json.load(f)

            chunk_word_limit = get_chunk_word_limit()
            chunks = make_chunks(segments, chunk_word_limit)
            print(f"  {len(chunks)} chunk(s)")

            chunk_summaries = []
            total_steps = len(chunks) + 2
            for i, chunk in enumerate(chunks):
                write_progress(file_uuid, "chunk", i + 1, total_steps)
                print(f"  chunk {i + 1}/{len(chunks)}...")
                chunk_text = fmt_chunk(chunk)
                print(chunk_text)

                summary = llm_call(
                    "Кратко перескажи этот фрагмент транскрипта в паре предлложнеий"
                    "(временные метки для контекста):\n\n" + chunk_text
                )
                chunk_summaries.append(summary)

            if len(chunk_summaries) == 1:
                summary_text = chunk_summaries[0]
            else:
                write_progress(file_uuid, "merge", len(chunks) + 1, total_steps)
                joined = "\n\n---\n\n".join(
                    f"Часть {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
                )
                summary_text = llm_call(
                    "Объедини эти части резюме в одно связное резюме. В начале придумай и напиши название для текста:\n\n" + joined
                )

            write_progress(file_uuid, "title", total_steps, total_steps)
            title = llm_call(
                "Проанализируй текст и придумай для него краткое, техническое название для базы знаний. Заголовок должен четко отражать тему (о чем идет речь) и быть понятным без контекста. Избегай метафор и поэтичных названий. Ограничься 3–5 словами Название должно содержать главные термины из текста"
                "Ответь только названием, без кавычек:\n\n" + summary_text
            ).strip().strip('"«»')

        except Exception as e:
            summary_text = f"Ошибка обработки: {e}"
            title = "Ошибка"

        remove_progress(file_uuid)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_text, "title": title}, f, ensure_ascii=False, indent=2)

        print(f"Done: {file_uuid}.json -> summary")

    time.sleep(2)
