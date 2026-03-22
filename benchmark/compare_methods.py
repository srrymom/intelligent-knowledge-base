"""
Сравнение методов суммаризации: Map-Reduce, Sequential, Hierarchical.

Для каждого метода на одном и том же транскрипте измеряются:
  - faithfulness      — доля 3-грамм резюме из транскрипта (↑ лучше, прокси галлюцинаций)
  - term_coverage     — доля ключевых терминов транскрипта, попавших в резюме (↑ лучше)
  - compression_ratio — слов_в_резюме / слов_в_транскрипте (целевой диапазон 0.10–0.25)
  - llm_calls         — количество обращений к LLM (↓ дешевле)
  - time_sec          — реальное время выполнения merge-фазы в секундах (↓ быстрее)

Использование:
    python benchmark/compare_methods.py
    python benchmark/compare_methods.py --n 3
    python benchmark/compare_methods.py --entry <uuid>
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm"))

from shared.config import KB_DIR, SUMMARIZATION_METHODS
from benchmark.metrics import faithfulness, term_coverage, compression_ratio
import llm.worker as _worker

METHODS = SUMMARIZATION_METHODS  # ["Map-Reduce", "Sequential", "Hierarchical"]


# ---------------------------------------------------------------------------
# LLM call counter: monkey-patch llm_call чтобы считать вызовы без изменения логики
# ---------------------------------------------------------------------------

_call_counter = 0
_orig_llm_call = _worker.llm_call


def _counting_llm_call(user_content, num_gpu=None):
    global _call_counter
    _call_counter += 1
    return _orig_llm_call(user_content, num_gpu=num_gpu)


def _reset_counter():
    global _call_counter
    _call_counter = 0


def _get_count():
    return _call_counter


def _install_counter():
    """Заменяет llm_call во всех функциях модуля на версию со счётчиком."""
    _worker.llm_call = _counting_llm_call
    # Функции _flat_merge, _sequential_merge, _hierarchical_merge замыкаются
    # на llm.worker.llm_call через глобальный словарь модуля — патч работает автоматически.


# ---------------------------------------------------------------------------
# Запуск одного метода и замер метрик
# ---------------------------------------------------------------------------

def _run_method(method: str, chunk_summaries: list) -> tuple:
    """
    Запускает merge-фазу нужного метода.
    Возвращает (summary_text, llm_calls, elapsed_sec).
    """
    _install_counter()
    _reset_counter()
    t0 = time.monotonic()

    if method == "Sequential":
        result = _worker._sequential_merge(chunk_summaries)
    elif method == "Hierarchical":
        result = _worker._hierarchical_merge(chunk_summaries)
    else:  # Map-Reduce (flat)
        result = _worker._flat_merge(chunk_summaries)

    elapsed = time.monotonic() - t0
    calls = _get_count()
    return result, calls, elapsed


def compare_entry(entry: dict) -> list:
    """
    Прогоняет все 3 метода на одном KB-транскрипте.
    Возвращает список строк-результатов (по одной на метод).
    """
    segments = entry.get("segments", [])
    orig_transcript = " ".join(
        s["transcription"] for s in segments if s.get("transcription")
    ).strip()

    if not orig_transcript:
        print("    [пропущено: пустой транскрипт]")
        return []

    # MAP-фаза: одинакова для всех методов — строим chunk_summaries
    chunk_word_limit = _worker.get_chunk_word_limit()
    chunks = _worker.make_chunks(segments, chunk_word_limit)

    print(f"    Чанков: {len(chunks)}, слов транскрипта: {len(orig_transcript.split())}")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = _worker.fmt_chunk(chunk)
        s = _worker.llm_call(
            "Перескажи содержание этого фрагмента транскрипта в 2–3 предложениях. "
            "Пиши только само содержание — без временных меток, заголовков и служебных меток. "
            "Временные метки в начале — только ориентир, не включай их в ответ.\n\n"
            + chunk_text
        )
        chunk_summaries.append(_worker.clean_summary(s))
        print(f"    Чанк {i + 1}/{len(chunks)} готов", end="\r")
    print()

    rows = []
    for method in METHODS:
        print(f"    Метод: {method} ...", end="  ", flush=True)
        try:
            summary, merge_calls, elapsed = _run_method(method, list(chunk_summaries))
        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue

        fa = faithfulness(orig_transcript, summary, ngram=3)
        tc = term_coverage(orig_transcript, summary)
        cr = compression_ratio(orig_transcript, summary)

        print(
            f"faith={fa:.3f}  coverage={tc:.3f}  "
            f"ratio={cr:.3f}  calls={merge_calls}  {elapsed:.1f}с"
        )

        rows.append({
            "entry_id": entry["id"],
            "title": entry.get("title", ""),
            "n_chunks": len(chunks),
            "method": method,
            "faithfulness": fa,
            "term_coverage": tc,
            "compression_ratio": cr,
            "merge_calls": merge_calls,
            "time_sec": round(elapsed, 2),
            "summary_words": len(summary.split()),
        })

    return rows


# ---------------------------------------------------------------------------
# Агрегирование и отчёт
# ---------------------------------------------------------------------------

def aggregate(all_rows: list) -> dict:
    """Среднее по методам."""
    by_method = defaultdict(list)
    for row in all_rows:
        by_method[row["method"]].append(row)

    result = {}
    for method in METHODS:
        rows = by_method.get(method, [])
        if not rows:
            continue
        n = len(rows)
        result[method] = {
            "n": n,
            "avg_faithfulness":      round(sum(r["faithfulness"] for r in rows) / n, 4),
            "avg_term_coverage":     round(sum(r["term_coverage"] for r in rows) / n, 4),
            "avg_compression_ratio": round(sum(r["compression_ratio"] for r in rows) / n, 4),
            "avg_merge_calls":       round(sum(r["merge_calls"] for r in rows) / n, 1),
            "avg_time_sec":          round(sum(r["time_sec"] for r in rows) / n, 1),
            "avg_summary_words":     round(sum(r["summary_words"] for r in rows) / n, 0),
        }
    return result


def print_report(agg: dict, all_rows: list):
    if not agg:
        print("Нет результатов.")
        return

    W = 88
    print("\n" + "=" * W)
    print("СРАВНЕНИЕ МЕТОДОВ СУММАРИЗАЦИИ")
    print("=" * W)
    print(
        f"{'Метод':<16} {'Достов.':>8} {'Покрытие':>9} {'Сжатие':>7} "
        f"{'LLM calls':>10} {'Время,с':>8} {'Слов рез.':>10}"
    )
    print("-" * W)

    for method in METHODS:
        v = agg.get(method)
        if v is None:
            continue
        print(
            f"{method:<16} {v['avg_faithfulness']:>8.3f} {v['avg_term_coverage']:>9.3f} "
            f"{v['avg_compression_ratio']:>7.3f} {v['avg_merge_calls']:>10.1f} "
            f"{v['avg_time_sec']:>8.1f} {v['avg_summary_words']:>10.0f}"
        )

    print("=" * W)
    print("Достов. (faithfulness)  — доля 3-грамм резюме из транскрипта; ↑ меньше галлюцинаций")
    print("Покрытие (term_coverage)— доля ключевых слов транскрипта в резюме;  ↑ лучше")
    print("Сжатие                  — слов_резюме / слов_транскрипта; целевой диапазон 0.10–0.25")
    print("LLM calls               — вызовов LLM на фазе merge (MAP-фаза одинакова)")

    # Лучший по каждой метрике
    metrics_rank = {
        "avg_faithfulness":      ("Достов.",  max),
        "avg_term_coverage":     ("Покрытие", max),
        "avg_compression_ratio": ("Сжатие",   lambda vals: min(vals, key=lambda x: abs(x - 0.175))),
        "avg_merge_calls":       ("LLM calls",min),
    }
    methods_list = [m for m in METHODS if m in agg]
    print("\nЛучший по метрике:")
    for key, (label, selector) in metrics_rank.items():
        values = {m: agg[m][key] for m in methods_list}
        if key == "avg_compression_ratio":
            best = min(methods_list, key=lambda m: abs(agg[m][key] - 0.175))
        elif selector is max:
            best = max(methods_list, key=lambda m: values[m])
        else:
            best = min(methods_list, key=lambda m: values[m])
        print(f"  {label:<14}: {best}  ({values[best]:.3f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_entries(n: int = None, entry_id: str = None) -> list:
    if entry_id:
        path = Path(KB_DIR) / f"{entry_id}.json"
        if not path.exists():
            print(f"Запись {entry_id} не найдена в {KB_DIR}")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            return [json.load(f)]

    paths = sorted(Path(KB_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    entries = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                entries.append(json.load(f))
        except Exception:
            continue
        if n and len(entries) >= n:
            break
    return entries


def main():
    parser = argparse.ArgumentParser(description="Сравнение методов суммаризации")
    parser.add_argument("--n", type=int, default=2, help="Количество KB-записей (default: 2)")
    parser.add_argument("--entry", default=None, help="UUID конкретной записи")
    parser.add_argument("--output", default="data/benchmark/results/compare_methods.json")
    args = parser.parse_args()

    entries = load_entries(args.n, args.entry)
    if not entries:
        print(f"Нет записей в {KB_DIR}. Обработайте лекции через интерфейс.")
        sys.exit(1)

    print(f"Сравнение {len(METHODS)} методов на {len(entries)} записях...")
    print(f"Методы: {', '.join(METHODS)}\n")

    all_rows = []
    for i, entry in enumerate(entries):
        title = entry.get("title", entry["id"])
        print(f"\n[{i + 1}/{len(entries)}] {title}")
        rows = compare_entry(entry)
        all_rows.extend(rows)

    if not all_rows:
        print("Нет результатов.")
        sys.exit(1)

    agg = aggregate(all_rows)
    print_report(agg, all_rows)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"aggregated": agg, "raw": all_rows},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nРезультаты сохранены: {out}")

    return agg


if __name__ == "__main__":
    main()
