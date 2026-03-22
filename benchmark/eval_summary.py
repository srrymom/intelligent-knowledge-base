"""
Оценка качества суммаризации без эталонных резюме.

Метрики:
  - faithfulness      — доля n-грамм резюме, присутствующих в транскрипте
                        (прокси: чем ниже, тем больше вероятность галлюцинаций)
  - term_coverage     — доля ключевых терминов транскрипта, упомянутых в резюме
  - compression_ratio — длина резюме / длина транскрипта (слова)

Входные данные: записи из data/knowledge_base/ (созданные через интерфейс).

Использование:
    python benchmark/eval_summary.py
    python benchmark/eval_summary.py --n 10
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import KB_DIR
from benchmark.metrics import faithfulness, term_coverage, compression_ratio


def load_kb_entries(n: int = None) -> list:
    """Загружает записи из knowledge_base/ (сортировка по дате — новые первыми)."""
    entries = []
    paths = sorted(Path(KB_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                entries.append(json.load(f))
        except Exception:
            continue
        if n and len(entries) >= n:
            break
    return entries


def eval_entry(entry: dict) -> dict | None:
    """Оценивает одну KB-запись по трём метрикам."""
    segments = entry.get("segments", [])
    transcript = " ".join(
        s["transcription"] for s in segments if s.get("transcription")
    ).strip()
    summary = (entry.get("summary") or "").strip()

    if not transcript or not summary:
        return None

    return {
        "id": entry["id"],
        "title": entry.get("title", ""),
        "transcript_words": len(transcript.split()),
        "summary_words": len(summary.split()),
        "faithfulness": faithfulness(transcript, summary, ngram=3),
        "term_coverage": term_coverage(transcript, summary),
        "compression_ratio": compression_ratio(transcript, summary),
    }


def compute_summary_stats(results: list) -> dict:
    valid = [r for r in results if r]
    if not valid:
        return {}
    return {
        "n": len(valid),
        "avg_faithfulness": round(sum(r["faithfulness"] for r in valid) / len(valid), 4),
        "avg_term_coverage": round(sum(r["term_coverage"] for r in valid) / len(valid), 4),
        "avg_compression_ratio": round(sum(r["compression_ratio"] for r in valid) / len(valid), 4),
    }


def print_report(results: list, stats: dict):
    valid = [r for r in results if r]
    if not valid:
        print("Нет данных. Добавьте записи через интерфейс (вкладка «Транскрипция»).")
        return

    print("\n" + "=" * 78)
    print("ОЦЕНКА СУММАРИЗАЦИИ")
    print("=" * 78)
    print(f"{'Название':<36} {'Слов тр.':>8} {'Слов рез.':>9} {'Сжатие':>7} {'Покрытие':>9} {'Достов.':>8}")
    print("-" * 78)
    for r in valid:
        title = (r["title"] or r["id"])[:35]
        print(
            f"{title:<36} {r['transcript_words']:>8} {r['summary_words']:>9} "
            f"{r['compression_ratio']:>7.3f} {r['term_coverage']:>9.3f} {r['faithfulness']:>8.3f}"
        )
    print("-" * 78)
    print(
        f"{'Среднее  N=' + str(stats['n']):<36} {'':>8} {'':>9} "
        f"{stats['avg_compression_ratio']:>7.3f} {stats['avg_term_coverage']:>9.3f} "
        f"{stats['avg_faithfulness']:>8.3f}"
    )
    print("=" * 78)

    print("\nИнтерпретация:")
    cr = stats["avg_compression_ratio"]
    tc = stats["avg_term_coverage"]
    fa = stats["avg_faithfulness"]

    cr_note = "в норме" if 0.10 <= cr <= 0.25 else ("слишком кратко" if cr < 0.10 else "слишком подробно")
    tc_note = "хорошо" if tc >= 0.40 else ("приемлемо" if tc >= 0.25 else "низкое — термины теряются")
    fa_note = "галлюцинации маловероятны" if fa >= 0.70 else ("есть риск" if fa >= 0.50 else "высокий риск галлюцинаций")

    print(f"  Сжатие:        {cr:.3f}  → {cr_note}  (целевой диапазон: 0.10–0.25)")
    print(f"  Покрытие:      {tc:.3f}  → {tc_note}")
    print(f"  Достоверность: {fa:.3f}  → {fa_note}")


def main():
    parser = argparse.ArgumentParser(description="Оценка суммаризации")
    parser.add_argument("--n", type=int, default=None, help="Максимальное число записей")
    parser.add_argument(
        "--output", default="data/benchmark/results/summary_eval.json"
    )
    args = parser.parse_args()

    entries = load_kb_entries(args.n)
    if not entries:
        print(f"Нет записей в {KB_DIR}. Обработайте несколько лекций через интерфейс.")
        sys.exit(1)

    print(f"Оценка {len(entries)} записей...")
    results = [eval_entry(e) for e in entries]
    stats = compute_summary_stats(results)
    print_report(results, stats)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": stats, "results": [r for r in results if r]},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nРезультаты сохранены: {out}")

    return stats


if __name__ == "__main__":
    main()
