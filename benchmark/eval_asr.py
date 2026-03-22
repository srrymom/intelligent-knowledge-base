"""
Оценка качества ASR (GigaAM) на датасете Golos (sberdevices_golos_10h_crowd).

Метрики: WER (Word Error Rate), CER (Character Error Rate), RTF (Real-time Factor).

Подготовка данных:
    gradio-env\\Scripts\\python install_golos.py
    → сохраняет 50 WAV-файлов + data/benchmark/asr/samples.jsonl

Запуск оценки (из asr/.venv):
    asr\\.venv\\Scripts\\python benchmark/eval_asr.py
    asr\\.venv\\Scripts\\python benchmark/eval_asr.py --jsonl data/benchmark/asr/samples.jsonl --n 50
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.metrics import wer, cer

DEFAULT_JSONL = "data/benchmark/asr/samples.jsonl"


def load_jsonl_samples(jsonl_path: str, n: int = None) -> list:
    """
    Загружает образцы из JSONL-файла (формат install_golos.py).
    Строка: {"id": ..., "audio": "путь/к/файлу.wav", "transcription": "эталон"}
    """
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = Path(item["audio"])
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            samples.append({
                "id": str(item.get("id", len(samples))),
                "reference": item["transcription"],
                "audio_path": str(audio_path),
            })
            if n and len(samples) >= n:
                break
    print(f"Загружено {len(samples)} образцов из {jsonl_path}")
    return samples


def run_asr_eval(samples: list, model_name: str = None) -> list:
    """Запускает GigaAM на образцах, возвращает результаты с метриками."""
    try:
        import soundfile as sf
        from asr.worker import load_model, transcribe_file, unload_model
    except ImportError as e:
        print(f"Импорт не удался: {e}")
        print("Запускайте из asr/.venv: asr\\.venv\\Scripts\\python benchmark/eval_asr.py")
        sys.exit(1)

    print(f"\nЗагрузка модели GigaAM ({model_name or 'default'})...")
    model = load_model(model_name)
    print("Модель загружена.\n")

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i + 1:>3}/{len(samples)}] {sample['id']}", end="  ", flush=True)

        audio_path = sample["audio_path"]
        try:
            info = sf.info(audio_path)
            audio_duration = info.duration
        except Exception:
            audio_duration = 1.0

        try:
            t0 = time.perf_counter()
            segments = transcribe_file(model, audio_path)
            elapsed = time.perf_counter() - t0

            hypothesis = " ".join(
                s["transcription"] for s in segments if s.get("transcription")
            )
            rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
            w = wer(sample["reference"], hypothesis)
            c = cer(sample["reference"], hypothesis)

            print(f"WER={w:.3f}  CER={c:.3f}  RTF={rtf:.2f}")
            results.append({
                "id": sample["id"],
                "reference": sample["reference"],
                "hypothesis": hypothesis,
                "wer": w,
                "cer": c,
                "rtf": round(rtf, 3),
                "audio_duration_sec": round(audio_duration, 2),
            })

        except Exception as e:
            print(f"ОШИБКА: {e}")
            results.append({"id": sample["id"], "error": str(e)})

    unload_model(model)
    return results


def compute_summary(results: list) -> dict:
    valid = [r for r in results if "wer" in r]
    if not valid:
        return {}
    avg_wer = sum(r["wer"] for r in valid) / len(valid)
    avg_cer = sum(r["cer"] for r in valid) / len(valid)
    rtf_vals = [r["rtf"] for r in valid if r.get("rtf") is not None]
    avg_rtf = sum(rtf_vals) / len(rtf_vals) if rtf_vals else None
    return {
        "n": len(valid),
        "avg_wer": round(avg_wer, 4),
        "avg_cer": round(avg_cer, 4),
        "avg_rtf": round(avg_rtf, 3) if avg_rtf is not None else None,
        "dataset": "Golos (sberdevices_golos_10h_crowd, test split)",
        "model": "GigaAM v3_e2e_rnnt",
    }


def print_report(results: list, summary: dict):
    valid = [r for r in results if "wer" in r]
    if not valid:
        print("Нет валидных результатов.")
        return

    W = 72
    print("\n" + "=" * W)
    print("ОЦЕНКА ASR — Golos (sberdevices_golos_10h_crowd)")
    print("=" * W)
    print(f"{'ID':<10} {'WER':>6} {'CER':>6} {'RTF':>6}")
    print("-" * W)
    for r in valid[:20]:
        rtf = f"{r['rtf']:.2f}" if r.get("rtf") is not None else "—"
        print(f"{str(r['id']):<10} {r['wer']:>6.3f} {r['cer']:>6.3f} {rtf:>6}")
        ref = r["reference"].replace("\n", " ")
        hyp = r["hypothesis"].replace("\n", " ")
        print(f"  Эталон:   {ref[:W - 12]}")
        print(f"  Гипотеза: {hyp[:W - 12]}")
    if len(valid) > 20:
        print(f"  ... ещё {len(valid) - 20} образцов")
    print("-" * W)
    rtf_str = f"{summary['avg_rtf']:.2f}" if summary.get("avg_rtf") else "—"
    print(f"{'Среднее  N=' + str(summary['n']):<10} {summary['avg_wer']:>6.3f} {summary['avg_cer']:>6.3f} {rtf_str:>6}")
    print("=" * W)
    print("\nПорог WER: ≤0.05 отлично | ≤0.15 хорошо | ≤0.30 приемлемо | >0.30 плохо")
    print("RTF < 1.0 — обработка быстрее реального времени")


def main():
    parser = argparse.ArgumentParser(description="Оценка ASR на датасете Golos")
    parser.add_argument(
        "--jsonl", default=DEFAULT_JSONL,
        help=f"JSONL-файл с образцами (по умолчанию: {DEFAULT_JSONL})"
    )
    parser.add_argument("--n", type=int, default=None, help="Ограничить число образцов")
    parser.add_argument("--model", default=None, help="Модель GigaAM (по умолчанию из config)")
    parser.add_argument("--output", default="data/benchmark/results/asr_eval.json")
    args = parser.parse_args()

    jsonl_path = ROOT / args.jsonl
    if not jsonl_path.exists():
        print(f"Файл не найден: {jsonl_path}")
        print("Сначала скачайте данные:\n  gradio-env\\Scripts\\python install_golos.py")
        sys.exit(1)

    samples = load_jsonl_samples(str(jsonl_path), args.n)
    results = run_asr_eval(samples, args.model)
    summary = compute_summary(results)
    print_report(results, summary)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {out}")

    return summary


if __name__ == "__main__":
    main()
