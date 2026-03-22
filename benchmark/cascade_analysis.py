"""
Каскадный анализ ошибок: влияние WER ASR на качество суммаризации.

Методология:
  1. Берём реальные транскрипты из data/knowledge_base/.
  2. Для каждого уровня WER (0%, 5%, 10%, 20%, 30%) искусственно
     вносим ошибки в транскрипт: замены, удаления, вставки слов.
  3. Запускаем LLM-суммаризацию на каждом испорченном транскрипте.
  4. Считаем faithfulness и term_coverage относительно оригинального транскрипта.
  5. Строим таблицу деградации метрик от уровня WER.

Результат: количественное подтверждение тезиса о критичности качества ASR
для downstream-задач (суммаризация, RAG).

Использование:
    python benchmark/cascade_analysis.py --n 3
"""

import sys
import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import KB_DIR
from benchmark.metrics import faithfulness, term_coverage, compression_ratio

WER_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]


def corrupt_segments(segments: list, target_wer: float, seed: int = 42) -> list:
    """
    Вносит ошибки в транскрипт с заданным приближённым WER.

    Операции (равномерно по target_wer):
      - замена (1/3): заменяем слово случайным словом из словаря
      - удаление (1/3): удаляем слово
      - вставка (1/3): вставляем случайное слово перед текущим

    Словарь строится из всех слов транскрипта (реалистичные «ошибки» ASR).
    """
    rng = random.Random(seed)
    all_words = [w for s in segments for w in s["transcription"].split() if w]
    vocab = list(set(all_words)) if all_words else ["слово"]

    corrupted = []
    for seg in segments:
        words = seg["transcription"].split()
        result = []
        for word in words:
            p = rng.random()
            if p < target_wer / 3:
                # замена
                result.append(rng.choice(vocab))
            elif p < target_wer * 2 / 3:
                # удаление
                pass
            elif p < target_wer:
                # вставка
                result.append(rng.choice(vocab))
                result.append(word)
            else:
                result.append(word)
        corrupted.append({**seg, "transcription": " ".join(result)})

    return corrupted


def analyze_entry(entry: dict) -> list:
    """
    Анализирует каскадный эффект для одной KB-записи.
    Возвращает список строк: [{wer_input, faithfulness, term_coverage, ...}, ...]
    """
    from llm.worker import summarize_transcript

    segments = entry.get("segments", [])
    orig_transcript = " ".join(
        s["transcription"] for s in segments if s.get("transcription")
    )
    if not orig_transcript.strip():
        print("    [пропущено: пустой транскрипт]")
        return []

    rows = []
    for wer_level in WER_LEVELS:
        corrupted = corrupt_segments(segments, wer_level) if wer_level > 0 else segments
        corrupted_text = " ".join(
            s["transcription"] for s in corrupted if s.get("transcription")
        )

        print(f"    WER={wer_level:.0%} ...", end="  ", flush=True)
        try:
            result = summarize_transcript(corrupted)
            summary = result["summary"]
        except Exception as e:
            print(f"ОШИБКА LLM: {e}")
            continue

        fa = faithfulness(corrupted_text, summary, ngram=3)
        tc = term_coverage(orig_transcript, summary)  # всегда vs оригинала
        cr = compression_ratio(corrupted_text, summary)

        print(f"faithfulness={fa:.3f}  term_coverage={tc:.3f}")

        rows.append({
            "entry_id": entry["id"],
            "wer_input": wer_level,
            "faithfulness": fa,
            "term_coverage": tc,
            "compression_ratio": cr,
        })

    return rows


def aggregate(all_rows: list) -> dict:
    """Агрегирует строки по уровням WER (среднее по всем записям)."""
    from collections import defaultdict
    by_wer = defaultdict(list)
    for row in all_rows:
        by_wer[row["wer_input"]].append(row)

    result = {}
    for wer_level in WER_LEVELS:
        rows = by_wer.get(wer_level, [])
        if not rows:
            continue
        result[wer_level] = {
            "faithfulness": round(sum(r["faithfulness"] for r in rows) / len(rows), 4),
            "term_coverage": round(sum(r["term_coverage"] for r in rows) / len(rows), 4),
            "n": len(rows),
        }

    # Считаем деградацию относительно baseline (WER=0%)
    if 0.0 in result:
        base_fa = result[0.0]["faithfulness"]
        base_tc = result[0.0]["term_coverage"]
        for wer_level, vals in result.items():
            vals["delta_faithfulness"] = round(vals["faithfulness"] - base_fa, 4)
            vals["delta_term_coverage"] = round(vals["term_coverage"] - base_tc, 4)

    return result


def print_report(agg: dict):
    print("\n" + "=" * 68)
    print("КАСКАДНЫЙ АНАЛИЗ: WER ASR → качество суммаризации")
    print("=" * 68)
    print(f"{'WER вход':>9} {'Достоверность':>14} {'Покрытие':>10} {'Δ Достов.':>12} {'Δ Покрытие':>12}")
    print("-" * 68)

    for wer_level in WER_LEVELS:
        vals = agg.get(wer_level)
        if vals is None:
            continue
        delta_fa = vals.get("delta_faithfulness", 0)
        delta_tc = vals.get("delta_term_coverage", 0)
        dfa_str = f"{delta_fa:+.3f}" if wer_level > 0 else "baseline"
        dtc_str = f"{delta_tc:+.3f}" if wer_level > 0 else "baseline"
        print(
            f"{wer_level:>8.0%}  {vals['faithfulness']:>14.3f} {vals['term_coverage']:>10.3f} "
            f"{dfa_str:>12} {dtc_str:>12}"
        )

    print("=" * 68)

    # Найдём порог деградации
    thresholds = []
    base_fa = agg.get(0.0, {}).get("faithfulness", 1.0)
    for wer_level in WER_LEVELS[1:]:
        vals = agg.get(wer_level)
        if vals and vals.get("delta_faithfulness", 0) < -0.05:
            thresholds.append(wer_level)
            break

    if thresholds:
        print(f"\nВывод: заметная деградация суммаризации (>5%) начинается при WER ≥ {thresholds[0]:.0%}.")
    else:
        print("\nВывод: существенной деградации суммаризации в диапазоне WER 0–30% не обнаружено.")

    print("Это подтверждает/не подтверждает критичность качества ASR для downstream-задач.")


def main():
    parser = argparse.ArgumentParser(description="Каскадный анализ ошибок ASR → суммаризация")
    parser.add_argument("--n", type=int, default=3, help="Количество KB-записей для анализа")
    parser.add_argument(
        "--output", default="data/benchmark/results/cascade_analysis.json"
    )
    args = parser.parse_args()

    # Загружаем записи из KB
    entries = []
    for path in sorted(Path(KB_DIR).glob("*.json"))[:args.n]:
        try:
            with open(path, encoding="utf-8") as f:
                entries.append(json.load(f))
        except Exception:
            continue

    if not entries:
        print(f"Нет записей в {KB_DIR}. Обработайте лекции через интерфейс.")
        sys.exit(1)

    print(f"Анализ {len(entries)} записей × {len(WER_LEVELS)} уровней WER...")
    print(f"Итого LLM-вызовов: {len(entries) * len(WER_LEVELS)}\n")

    all_rows = []
    for i, entry in enumerate(entries):
        print(f"\n  [{i + 1}/{len(entries)}] {entry.get('title', entry['id'])}")
        rows = analyze_entry(entry)
        all_rows.extend(rows)

    if not all_rows:
        print("Нет результатов.")
        sys.exit(1)

    agg = aggregate(all_rows)
    print_report(agg)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"aggregated": {str(k): v for k, v in agg.items()}, "raw": all_rows},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nРезультаты сохранены: {out}")

    return agg


if __name__ == "__main__":
    main()
