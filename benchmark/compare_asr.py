"""
Сравнение ASR-моделей на Common Voice RU.

Сравниваемые модели:
  - GigaAM v3_e2e_rnnt  (выбранная модель)
  - Whisper medium       (альтернатива, ~3 GB VRAM)
  - Whisper large-v3     (лучшее качество, но >4 GB VRAM — не помещается)

Цель: обоснование выбора GigaAM для RTX 3050 4 GB VRAM.

Использование:
    # Для GigaAM — запускать из asr/.venv
    asr\\.venv\\Scripts\\python benchmark/compare_asr.py --n 50

    # Whisper можно запустить отдельно из основного venv:
    python benchmark/compare_asr.py --n 50 --models whisper_medium,whisper_large

Зависимости:
    asr/.venv:    pip install datasets soundfile gigaam
    gradio-env:   pip install openai-whisper datasets soundfile
"""

import os
import sys
import json
import time
import tempfile
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.metrics import wer, cer
from benchmark.eval_asr import load_jsonl_samples, compute_summary

DEFAULT_JSONL = "data/benchmark/asr/samples.jsonl"

# Эмпирические оценки VRAM (МБ) для каждой модели
VRAM_MB = {
    "GigaAM v3_e2e_rnnt": 3500,
    "Whisper medium":       3000,
    "Whisper large-v3":     6200,
}

VRAM_LIMIT_MB = 4096  # RTX 3050


def _eval_gigaam(samples: list, model_name: str = None) -> dict:
    """Оценивает GigaAM на образцах. Возвращает агрегированные метрики."""
    try:
        import soundfile as sf
        from asr.worker import load_model, transcribe_file, unload_model
    except ImportError as e:
        print(f"  [GigaAM] импорт не удался: {e} — пропускаем")
        return None

    label = f"GigaAM {model_name or 'v3_e2e_rnnt'}"
    print(f"  Загрузка {label}...")
    model = load_model(model_name)

    wer_list, cer_list, rtf_list = [], [], []
    for sample in samples:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, sample["audio_array"], sample["sampling_rate"])
            dur = len(sample["audio_array"]) / sample["sampling_rate"]
            t0 = time.perf_counter()
            segs = transcribe_file(model, tmp_path)
            elapsed = time.perf_counter() - t0
            hyp = " ".join(s["transcription"] for s in segs if s.get("transcription"))
            wer_list.append(wer(sample["reference"], hyp))
            cer_list.append(cer(sample["reference"], hyp))
            rtf_list.append(elapsed / dur if dur > 0 else 0.0)
        except Exception:
            pass
        finally:
            os.unlink(tmp_path)

    unload_model(model)

    if not wer_list:
        return None

    return {
        "model": "GigaAM v3_e2e_rnnt",
        "wer": round(sum(wer_list) / len(wer_list), 4),
        "cer": round(sum(cer_list) / len(cer_list), 4),
        "rtf": round(sum(rtf_list) / len(rtf_list), 3),
        "vram_mb": VRAM_MB["GigaAM v3_e2e_rnnt"],
        "fits_4gb": VRAM_MB["GigaAM v3_e2e_rnnt"] <= VRAM_LIMIT_MB,
        "n": len(wer_list),
    }


def _eval_whisper(samples: list, model_size: str = "medium") -> dict:
    """Оценивает Whisper на образцах. Возвращает агрегированные метрики."""
    try:
        import whisper
        import soundfile as sf
    except ImportError:
        print(f"  [Whisper {model_size}] openai-whisper не установлен — пропускаем")
        return None

    label = f"Whisper {model_size}"
    print(f"  Загрузка {label}...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"  [Whisper {model_size}] ошибка загрузки: {e} — пропускаем")
        return None

    wer_list, cer_list, rtf_list = [], [], []
    for sample in samples:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, sample["audio_array"], sample["sampling_rate"])
            dur = len(sample["audio_array"]) / sample["sampling_rate"]
            t0 = time.perf_counter()
            out = model.transcribe(tmp_path, language="ru")
            elapsed = time.perf_counter() - t0
            hyp = out.get("text", "").strip()
            wer_list.append(wer(sample["reference"], hyp))
            cer_list.append(cer(sample["reference"], hyp))
            rtf_list.append(elapsed / dur if dur > 0 else 0.0)
        except Exception:
            pass
        finally:
            os.unlink(tmp_path)

    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    if not wer_list:
        return None

    model_key = f"Whisper {model_size}"
    return {
        "model": model_key,
        "wer": round(sum(wer_list) / len(wer_list), 4),
        "cer": round(sum(cer_list) / len(cer_list), 4),
        "rtf": round(sum(rtf_list) / len(rtf_list), 3),
        "vram_mb": VRAM_MB.get(model_key, "?"),
        "fits_4gb": VRAM_MB.get(model_key, 9999) <= VRAM_LIMIT_MB,
        "n": len(wer_list),
    }


def run_comparison(n: int = 50, models: list = None, jsonl: str = None) -> list:
    """
    Запускает сравнение моделей. models — список строк из:
      ['gigaam', 'whisper_medium', 'whisper_large']
    По умолчанию — все три.
    """
    if models is None:
        models = ["gigaam", "whisper_medium", "whisper_large"]

    from pathlib import Path
    jsonl_path = Path(jsonl) if jsonl else Path(__file__).parent.parent / DEFAULT_JSONL
    if not jsonl_path.exists():
        print(f"Файл не найден: {jsonl_path}")
        print("Сначала: gradio-env\\Scripts\\python install_golos.py")
        import sys; sys.exit(1)
    samples = load_jsonl_samples(str(jsonl_path), n)
    table = []

    if "gigaam" in models:
        print("\n[GigaAM v3_e2e_rnnt]")
        result = _eval_gigaam(samples)
        if result:
            table.append(result)

    if "whisper_medium" in models:
        print("\n[Whisper medium]")
        result = _eval_whisper(samples, "medium")
        if result:
            table.append(result)

    if "whisper_large" in models:
        print("\n[Whisper large-v3]")
        result = _eval_whisper(samples, "large-v3")
        if result:
            table.append(result)

    return table


def print_comparison(table: list):
    if not table:
        print("Нет результатов.")
        return

    print("\n" + "=" * 72)
    print("СРАВНЕНИЕ ASR-МОДЕЛЕЙ — Common Voice RU")
    print("=" * 72)
    print(f"{'Модель':<28} {'WER':>6} {'CER':>6} {'RTF':>6}  {'VRAM':>8}  {'RTX3050 4GB':>12}")
    print("-" * 72)

    for row in table:
        fits = "✓ да" if row.get("fits_4gb") else "✗ не влезает"
        vram = f"{row['vram_mb']} МБ" if isinstance(row.get("vram_mb"), int) else "?"
        print(
            f"{row['model']:<28} {row['wer']:>6.3f} {row['cer']:>6.3f} "
            f"{row['rtf']:>6.2f}  {vram:>8}  {fits:>12}"
        )

    print("=" * 72)

    # Автоматический вывод
    fits = [r for r in table if r.get("fits_4gb")]
    if fits:
        best = min(fits, key=lambda x: x["wer"])
        print(f"\nВывод: '{best['model']}' — оптимальный выбор для RTX 3050 4 GB:")
        print(f"  · Наименьший WER среди моделей, помещающихся в 4 GB VRAM")
        rtf_note = "быстрее" if best["rtf"] < 1.0 else "медленнее"
        print(f"  · RTF={best['rtf']:.2f} — обработка {rtf_note} реального времени")

    no_fit = [r for r in table if not r.get("fits_4gb")]
    for r in no_fit:
        print(f"  · '{r['model']}' исключена: требует {r['vram_mb']} МБ > 4096 МБ")


def main():
    parser = argparse.ArgumentParser(description="Сравнение ASR-моделей")
    parser.add_argument("--n", type=int, default=50, help="Количество образцов")
    parser.add_argument(
        "--jsonl", default=DEFAULT_JSONL,
        help="JSONL-файл с образцами Golos"
    )
    parser.add_argument(
        "--models", default="gigaam,whisper_medium,whisper_large",
        help="Модели для сравнения через запятую"
    )
    parser.add_argument(
        "--output", default="data/benchmark/results/compare_asr.json"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    table = run_comparison(args.n, models, args.jsonl)
    print_comparison(table)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {out}")

    return table


if __name__ == "__main__":
    main()
