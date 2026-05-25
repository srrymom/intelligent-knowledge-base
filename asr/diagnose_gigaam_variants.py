"""
Диагностика вариантов GigaAM longform ASR без изменения production-кода.

Куда положить:
    prototype/asr/diagnose_gigaam_variants.py

Пример запуска из корня проекта:
    .\asr\.venv\Scripts\python.exe .\asr\diagnose_gigaam_variants.py "D:\desktop\короткая лекция о свободе.mp3" \
        --models v3_e2e_rnnt v3_e2e_ctc v3_rnnt v3_ctc \
        --plans existing existing_split raw_vad raw_vad_split retry_existing spans \
        --max-durations 8 12 15 20 \
        --spans 18.695-40.025 57.423-84.794 \
        --out-dir data/asr_debug/free_lecture_variants

Что делает:
    1. Читает аудио тем же способом, что GigaAM.
    2. Получает raw VAD-регионы через pyannote.
    3. Получает grouped-регионы через штатный gigaam.vad_utils.segment_audio_file().
    4. Прогоняет разные модели GigaAM и разные планы сегментации.
    5. Ловит мусор вида "⁇ ⁇ ⁇", пустые результаты, подозрительные unknown-токены.
    6. Для retry-планов пытается рекурсивно разбить проблемный кусок и распознать части.
    7. Пишет:
        - details.jsonl: все финальные распознанные сегменты;
        - attempts.jsonl: все попытки retry, включая провалившиеся родительские чанки;
        - summary.csv: агрегированная таблица по model + plan;
        - report.md: читаемый отчёт с проблемными интервалами.

Важно:
    Это диагностический скрипт. Он ничего не меняет в data/queue, transcript, knowledge_base.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal


# -----------------------------------------------------------------------------
# Project bootstrap
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "asr":
    PROJECT_ROOT = THIS_FILE.parents[1]
else:
    PROJECT_ROOT = THIS_FILE.parent

GIGAAM_DIR = PROJECT_ROOT / "asr" / "GigaAM"
for item in (PROJECT_ROOT, GIGAAM_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

try:
    from shared.config import ASR_MODEL, FFMPEG_PATH
except Exception:
    ASR_MODEL = "v3_e2e_rnnt"
    FFMPEG_PATH = r"D:\ffmpeg\bin"

if sys.platform == "win32":
    os.environ["PATH"] += os.path.pathsep + str(FFMPEG_PATH)
    try:
        os.add_dll_directory(str(FFMPEG_PATH))
    except (FileNotFoundError, OSError):
        pass


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

UNKNOWN_CHARS = {"⁇", "�"}


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    source: str = "unknown"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def clamped(self, duration: float) -> "Region":
        return Region(
            start=max(0.0, min(self.start, duration)),
            end=max(0.0, min(self.end, duration)),
            source=self.source,
        )


@dataclass
class TextQuality:
    is_valid: bool
    reason: str
    unknown_count: int
    cleaned_text: str
    compact_text: str
    char_count: int
    word_count: int


@dataclass
class DecodeRow:
    audio_path: str
    model: str
    plan: str
    segment_source: str
    start: float
    end: float
    duration: float
    samples: int
    text: str
    cleaned_text: str
    compact_text: str
    is_valid: bool
    invalid_reason: str
    unknown_count: int
    char_count: int
    word_count: int
    elapsed_sec: float
    retry_level: int = 0
    parent_start: float | None = None
    parent_end: float | None = None
    note: str = ""


@dataclass
class RunSummary:
    audio_path: str
    model: str
    plan: str
    input_region_count: int
    final_segment_count: int
    valid_segment_count: int
    invalid_segment_count: int
    input_coverage_sec: float
    valid_coverage_sec: float
    invalid_coverage_sec: float
    input_coverage_pct: float
    valid_coverage_pct: float
    invalid_coverage_pct: float
    total_words: int
    total_chars: int
    unknown_count: int
    elapsed_sec: float
    error: str = ""


@dataclass
class PlanSpec:
    name: str
    base: Literal["existing", "raw_vad", "fixed", "spans"]
    max_duration: float | None = None
    retry: bool = False
    overlap: float = 0.0


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def run_json(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return json.loads(proc.stdout or "{}")



def probe_media(path: Path) -> dict[str, Any]:
    try:
        return run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size,format_name:stream=index,codec_type,codec_name,duration,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ]
        )
    except Exception as exc:
        return {"error": repr(exc), "file_size": path.stat().st_size}



def get_media_duration(probe: dict[str, Any], fallback: float = 0.0) -> float:
    try:
        return float((probe.get("format") or {}).get("duration") or fallback)
    except (TypeError, ValueError):
        return fallback



def format_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    minutes, rem_ms = divmod(total_ms, 60_000)
    sec, ms = divmod(rem_ms, 1000)
    return f"{minutes:02d}:{sec:02d}.{ms:03d}"



def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "item"



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")



def append_jsonl(path: Path, rows: Iterable[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                payload = asdict(row)
            else:
                payload = row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")



def union_duration(regions: Iterable[Region], min_gap: float = 0.001) -> float:
    intervals = sorted(
        [(max(0.0, r.start), max(0.0, r.end)) for r in regions if r.end > r.start],
        key=lambda item: item[0],
    )
    if not intervals:
        return 0.0

    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + min_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)



def find_gaps(regions: list[Region], audio_duration: float, min_gap: float = 0.05) -> list[Region]:
    gaps: list[Region] = []
    cursor = 0.0
    for region in sorted(regions, key=lambda item: item.start):
        if region.start - cursor >= min_gap:
            gaps.append(Region(cursor, region.start, source="gap"))
        cursor = max(cursor, region.end)
    if audio_duration - cursor >= min_gap:
        gaps.append(Region(cursor, audio_duration, source="gap"))
    return gaps


# -----------------------------------------------------------------------------
# Text validation
# -----------------------------------------------------------------------------


def clean_asr_text(text: str) -> str:
    # Не делаем агрессивную нормализацию для вывода: только убираем unknown-символы.
    cleaned = text
    for ch in UNKNOWN_CHARS:
        cleaned = cleaned.replace(ch, "")
    return re.sub(r"\s+", " ", cleaned).strip()



def compact_text(text: str) -> str:
    cleaned = clean_asr_text(text)
    return re.sub(r"\s+", "", cleaned).strip()



def count_words_ru(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text))



def assess_text_quality(text: str, duration: float) -> TextQuality:
    unknown_count = sum(text.count(ch) for ch in UNKNOWN_CHARS)
    cleaned = clean_asr_text(text)
    compact = compact_text(text)
    char_count = len(compact)
    word_count = count_words_ru(cleaned)
    lowered = cleaned.lower()

    # Прямо встречающийся артефакт GigaAM в твоём логе.
    if "очень длинная строка" in lowered and unknown_count >= 3:
        return TextQuality(False, "gigaam_long_unknown_string", unknown_count, cleaned, compact, char_count, word_count)

    if not compact:
        if unknown_count > 0:
            return TextQuality(False, "only_unknown_tokens", unknown_count, cleaned, compact, char_count, word_count)
        return TextQuality(False, "empty_text", unknown_count, cleaned, compact, char_count, word_count)

    # Если unknown-токенов много, а нормального текста мало — почти наверняка мусор.
    if unknown_count >= 5 and char_count < 40:
        return TextQuality(False, "many_unknowns_low_text", unknown_count, cleaned, compact, char_count, word_count)

    # Для длинного аудиокуска 1-2 слова подозрительны. Для короткого куска это нормально.
    if duration >= 8.0 and word_count <= 2:
        return TextQuality(False, "too_few_words_for_long_audio", unknown_count, cleaned, compact, char_count, word_count)

    # Если половина вывода состоит из unknown-символов, лучше считать сегмент проблемным.
    if unknown_count >= 3:
        denom = max(1, unknown_count + char_count)
        unknown_ratio = unknown_count / denom
        if unknown_ratio >= 0.20:
            return TextQuality(False, f"high_unknown_ratio_{unknown_ratio:.2f}", unknown_count, cleaned, compact, char_count, word_count)

    return TextQuality(True, "ok", unknown_count, cleaned, compact, char_count, word_count)


# -----------------------------------------------------------------------------
# Audio/VAD/segmentation
# -----------------------------------------------------------------------------


def load_audio_tensor(audio_path: Path):
    from gigaam.preprocess import SAMPLE_RATE, load_audio

    audio = load_audio(str(audio_path))
    return audio, SAMPLE_RATE



def select_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)



def get_raw_vad_regions(audio_path: Path, decoded_duration: float, device) -> list[Region]:
    from gigaam.vad_utils import get_pipeline

    pipeline = get_pipeline(device)
    sad_segments = pipeline(str(audio_path))
    regions = [
        Region(max(0.0, segment.start), min(decoded_duration, segment.end), source="raw_vad")
        for segment in sad_segments.get_timeline().support()
    ]
    return [r for r in regions if r.duration > 0.001]



def get_existing_grouped_regions(audio_path: Path, sample_rate: int, device) -> list[Region]:
    from gigaam.vad_utils import segment_audio_file

    _chunks, boundaries = segment_audio_file(str(audio_path), sample_rate, device=device)
    return [Region(float(start), float(end), source="existing_grouped") for start, end in boundaries]



def parse_spans(values: list[str] | None, audio_duration: float) -> list[Region]:
    if not values:
        return []

    regions: list[Region] = []
    for raw in values:
        item = raw.strip().replace(",", ".")
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Bad span {raw!r}; expected START-END, e.g. 18.695-40.025")
        left, right = item.split("-", 1)
        start = float(left)
        end = float(right)
        regions.append(Region(start, end, source="manual_span").clamped(audio_duration))
    return [r for r in regions if r.duration > 0.001]



def split_region(region: Region, max_duration: float) -> list[Region]:
    if max_duration <= 0 or region.duration <= max_duration:
        return [region]

    part_count = int(math.ceil(region.duration / max_duration))
    part_duration = region.duration / part_count
    parts: list[Region] = []
    for idx in range(part_count):
        start = region.start + idx * part_duration
        end = region.end if idx == part_count - 1 else region.start + (idx + 1) * part_duration
        parts.append(Region(start, end, source=f"{region.source}_split_{max_duration:g}s"))
    return parts



def split_regions(regions: list[Region], max_duration: float | None) -> list[Region]:
    if max_duration is None:
        return regions
    out: list[Region] = []
    for region in regions:
        out.extend(split_region(region, max_duration))
    return out



def fixed_windows(audio_duration: float, window: float, overlap: float = 0.0) -> list[Region]:
    if window <= 0:
        raise ValueError("fixed window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must be >= 0 and < window")

    step = window - overlap
    regions: list[Region] = []
    start = 0.0
    while start < audio_duration - 0.001:
        end = min(audio_duration, start + window)
        regions.append(Region(start, end, source=f"fixed_{window:g}s"))
        if end >= audio_duration:
            break
        start += step
    return regions



def make_plan_specs(
    plan_names: list[str],
    max_durations: list[float],
    fixed_overlap: float,
) -> list[PlanSpec]:
    specs: list[PlanSpec] = []

    for plan in plan_names:
        if plan == "existing":
            specs.append(PlanSpec(name="existing", base="existing"))
        elif plan == "raw_vad":
            specs.append(PlanSpec(name="raw_vad", base="raw_vad"))
        elif plan == "spans":
            specs.append(PlanSpec(name="spans", base="spans"))
        elif plan == "existing_split":
            for max_dur in max_durations:
                specs.append(PlanSpec(name=f"existing_split_{max_dur:g}s", base="existing", max_duration=max_dur))
        elif plan == "raw_vad_split":
            for max_dur in max_durations:
                specs.append(PlanSpec(name=f"raw_vad_split_{max_dur:g}s", base="raw_vad", max_duration=max_dur))
        elif plan == "fixed":
            for max_dur in max_durations:
                overlap_label = f"_ov{fixed_overlap:g}s" if fixed_overlap else ""
                specs.append(
                    PlanSpec(
                        name=f"fixed_{max_dur:g}s{overlap_label}",
                        base="fixed",
                        max_duration=max_dur,
                        overlap=fixed_overlap,
                    )
                )
        elif plan == "retry_existing":
            for max_dur in max_durations:
                specs.append(
                    PlanSpec(name=f"retry_existing_{max_dur:g}s", base="existing", max_duration=max_dur, retry=True)
                )
        elif plan == "retry_raw_vad":
            for max_dur in max_durations:
                specs.append(PlanSpec(name=f"retry_raw_vad_{max_dur:g}s", base="raw_vad", max_duration=max_dur, retry=True))
        else:
            raise ValueError(f"Unknown plan: {plan}")

    # Убираем дубли при странных аргументах.
    unique: dict[str, PlanSpec] = {}
    for spec in specs:
        unique[spec.name] = spec
    return list(unique.values())



def regions_for_plan(
    spec: PlanSpec,
    existing_regions: list[Region],
    raw_vad_regions: list[Region],
    manual_spans: list[Region],
    audio_duration: float,
) -> list[Region]:
    if spec.base == "existing":
        base = existing_regions
    elif spec.base == "raw_vad":
        base = raw_vad_regions
    elif spec.base == "spans":
        base = manual_spans
    elif spec.base == "fixed":
        assert spec.max_duration is not None
        return fixed_windows(audio_duration, spec.max_duration, overlap=spec.overlap)
    else:
        raise AssertionError(f"Unexpected plan base: {spec.base}")

    return split_regions(base, spec.max_duration)


# -----------------------------------------------------------------------------
# GigaAM decoding
# -----------------------------------------------------------------------------


def load_gigaam_model(model_name: str):
    # torchcodec импортируем явно: в твоём окружении он нужен GigaAM для чтения аудио.
    import torchcodec  # noqa: F401
    import gigaam

    return gigaam.load_model(model_name)



def audio_slice(audio, region: Region, sample_rate: int):
    start_i = max(0, int(round(region.start * sample_rate)))
    end_i = min(int(audio.shape[0]), int(round(region.end * sample_rate)))
    return audio[start_i:end_i]



def decode_region(
    *,
    audio_path: Path,
    model_name: str,
    model,
    plan_name: str,
    audio,
    sample_rate: int,
    region: Region,
    retry_level: int = 0,
    parent: Region | None = None,
    note: str = "",
) -> DecodeRow:
    import torch

    chunk = audio_slice(audio, region, sample_rate)
    start_time = time.time()

    raw_text = ""
    error_note = note
    try:
        if int(chunk.shape[-1]) <= 0:
            raw_text = ""
            error_note = (note + "; " if note else "") + "empty_audio_slice"
        else:
            with torch.no_grad():
                wav = chunk.to(model._device).unsqueeze(0).to(model._dtype)
                length = torch.full([1], wav.shape[-1], device=model._device)
                encoded, encoded_len = model.forward(wav, length)
                raw_text = model.decoding.decode(model.head, encoded, encoded_len)[0]
    except Exception as exc:
        raw_text = ""
        error_note = (note + "; " if note else "") + f"decode_exception={type(exc).__name__}: {exc}"

    elapsed = time.time() - start_time
    quality = assess_text_quality(raw_text, region.duration)

    # Если был exception, он важнее текстовой причины.
    invalid_reason = quality.reason
    is_valid = quality.is_valid
    if "decode_exception=" in error_note or "empty_audio_slice" in error_note:
        is_valid = False
        invalid_reason = error_note

    return DecodeRow(
        audio_path=str(audio_path),
        model=model_name,
        plan=plan_name,
        segment_source=region.source,
        start=region.start,
        end=region.end,
        duration=region.duration,
        samples=int(chunk.shape[-1]),
        text=raw_text,
        cleaned_text=quality.cleaned_text,
        compact_text=quality.compact_text,
        is_valid=is_valid,
        invalid_reason="ok" if is_valid else invalid_reason,
        unknown_count=quality.unknown_count,
        char_count=quality.char_count,
        word_count=quality.word_count,
        elapsed_sec=elapsed,
        retry_level=retry_level,
        parent_start=parent.start if parent else None,
        parent_end=parent.end if parent else None,
        note=error_note,
    )



def decode_region_with_retry(
    *,
    audio_path: Path,
    model_name: str,
    model,
    plan_name: str,
    audio,
    sample_rate: int,
    region: Region,
    retry_max_duration: float,
    max_retry_depth: int,
    min_retry_duration: float,
    retry_level: int = 0,
    parent: Region | None = None,
) -> tuple[list[DecodeRow], list[DecodeRow]]:
    """
    Возвращает:
        final_rows — финальные сегменты, которые идут в итоговую статистику;
        attempt_rows — все попытки, включая родительские failed attempts.
    """
    row = decode_region(
        audio_path=audio_path,
        model_name=model_name,
        model=model,
        plan_name=plan_name,
        audio=audio,
        sample_rate=sample_rate,
        region=region,
        retry_level=retry_level,
        parent=parent,
    )

    attempts = [row]
    can_split = (
        not row.is_valid
        and retry_level < max_retry_depth
        and region.duration >= min_retry_duration
        and region.duration > 1.0
    )

    if not can_split:
        return [row], attempts

    # Если передан max_duration, делим не просто пополам, а на части <= max_duration.
    # Для совсем коротких проблемных кусков всё равно получится 2 части.
    if retry_max_duration > 0 and region.duration > retry_max_duration:
        children = split_region(region, retry_max_duration)
    else:
        mid = region.start + region.duration / 2.0
        children = [
            Region(region.start, mid, source=f"{region.source}_retry_left"),
            Region(mid, region.end, source=f"{region.source}_retry_right"),
        ]

    final_rows: list[DecodeRow] = []
    for child in children:
        child_final, child_attempts = decode_region_with_retry(
            audio_path=audio_path,
            model_name=model_name,
            model=model,
            plan_name=plan_name,
            audio=audio,
            sample_rate=sample_rate,
            region=child,
            retry_max_duration=retry_max_duration,
            max_retry_depth=max_retry_depth,
            min_retry_duration=min_retry_duration,
            retry_level=retry_level + 1,
            parent=region,
        )
        final_rows.extend(child_final)
        attempts.extend(child_attempts)

    return final_rows, attempts


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def summarize_run(
    *,
    audio_path: Path,
    model_name: str,
    plan_name: str,
    input_regions: list[Region],
    final_rows: list[DecodeRow],
    audio_duration: float,
    elapsed_sec: float,
    error: str = "",
) -> RunSummary:
    valid_regions = [Region(r.start, r.end, source="valid") for r in final_rows if r.is_valid]
    invalid_regions = [Region(r.start, r.end, source="invalid") for r in final_rows if not r.is_valid]

    input_cov = union_duration(input_regions)
    valid_cov = union_duration(valid_regions)
    invalid_cov = union_duration(invalid_regions)

    denom = audio_duration if audio_duration > 0 else 1.0
    return RunSummary(
        audio_path=str(audio_path),
        model=model_name,
        plan=plan_name,
        input_region_count=len(input_regions),
        final_segment_count=len(final_rows),
        valid_segment_count=sum(1 for r in final_rows if r.is_valid),
        invalid_segment_count=sum(1 for r in final_rows if not r.is_valid),
        input_coverage_sec=input_cov,
        valid_coverage_sec=valid_cov,
        invalid_coverage_sec=invalid_cov,
        input_coverage_pct=input_cov / denom * 100.0,
        valid_coverage_pct=valid_cov / denom * 100.0,
        invalid_coverage_pct=invalid_cov / denom * 100.0,
        total_words=sum(r.word_count for r in final_rows if r.is_valid),
        total_chars=sum(r.char_count for r in final_rows if r.is_valid),
        unknown_count=sum(r.unknown_count for r in final_rows),
        elapsed_sec=elapsed_sec,
        error=error,
    )



def write_summary_csv(path: Path, summaries: list[RunSummary]) -> None:
    ensure_dir(path.parent)
    if not summaries:
        path.write_text("", encoding="utf-8")
        return

    fields = list(asdict(summaries[0]).keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))



def short_text(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"



def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")
    return "\n".join(out)



def write_markdown_report(
    *,
    path: Path,
    audio_path: Path,
    media_probe: dict[str, Any],
    decoded_duration: float,
    raw_vad_regions: list[Region],
    existing_regions: list[Region],
    manual_spans: list[Region],
    summaries: list[RunSummary],
    detail_rows: list[DecodeRow],
    attempt_rows: list[DecodeRow],
) -> None:
    ensure_dir(path.parent)

    sorted_summaries = sorted(
        summaries,
        key=lambda s: (s.error != "", -s.valid_coverage_pct, s.invalid_coverage_pct, s.elapsed_sec),
    )

    summary_rows = []
    for s in sorted_summaries:
        summary_rows.append(
            [
                s.model,
                s.plan,
                f"{s.valid_coverage_pct:.1f}%",
                f"{s.invalid_coverage_pct:.1f}%",
                f"{s.valid_segment_count}/{s.final_segment_count}",
                s.total_words,
                f"{s.elapsed_sec:.1f}s",
                s.error or "",
            ]
        )

    invalid_rows = [r for r in detail_rows if not r.is_valid]
    invalid_rows = sorted(invalid_rows, key=lambda r: (r.model, r.plan, r.start))
    invalid_table_rows = []
    for r in invalid_rows[:100]:
        invalid_table_rows.append(
            [
                r.model,
                r.plan,
                f"{format_time(r.start)}–{format_time(r.end)}",
                f"{r.duration:.2f}s",
                r.invalid_reason,
                r.unknown_count,
                short_text(r.text, 120),
            ]
        )

    retry_failed_parent_rows = [
        r for r in attempt_rows if not r.is_valid and r.retry_level == 0 and r.plan.startswith("retry_")
    ]
    retry_table_rows = []
    for r in retry_failed_parent_rows[:100]:
        retry_table_rows.append(
            [
                r.model,
                r.plan,
                f"{format_time(r.start)}–{format_time(r.end)}",
                f"{r.duration:.2f}s",
                r.invalid_reason,
                short_text(r.text, 120),
            ]
        )

    raw_vad_cov = union_duration(raw_vad_regions)
    existing_cov = union_duration(existing_regions)
    span_cov = union_duration(manual_spans)
    denom = decoded_duration if decoded_duration > 0 else 1.0

    text = []
    text.append(f"# GigaAM ASR variants report\n")
    text.append(f"**Audio:** `{audio_path}`\n")
    text.append(f"**Decoded duration:** {decoded_duration:.3f}s\n")
    text.append("\n## Media probe\n")
    text.append("```json\n" + json.dumps(media_probe, ensure_ascii=False, indent=2) + "\n```\n")

    text.append("\n## Segmentation coverage\n")
    text.append(
        markdown_table(
            ["Source", "Regions", "Coverage", "Coverage %", "Gaps"],
            [
                ["raw_vad", len(raw_vad_regions), f"{raw_vad_cov:.3f}s", f"{raw_vad_cov / denom * 100:.1f}%", len(find_gaps(raw_vad_regions, decoded_duration))],
                ["existing_grouped", len(existing_regions), f"{existing_cov:.3f}s", f"{existing_cov / denom * 100:.1f}%", len(find_gaps(existing_regions, decoded_duration))],
                ["manual_spans", len(manual_spans), f"{span_cov:.3f}s", f"{span_cov / denom * 100:.1f}%", "-"],
            ],
        )
    )

    text.append("\n\n## Summary by model and plan\n")
    text.append(markdown_table(
        ["Model", "Plan", "Valid coverage", "Invalid coverage", "Valid segments", "Words", "Elapsed", "Error"],
        summary_rows,
    ))

    text.append("\n\n## Invalid final segments\n")
    if invalid_table_rows:
        text.append(markdown_table(
            ["Model", "Plan", "Time", "Dur", "Reason", "Unknowns", "Raw text"],
            invalid_table_rows,
        ))
    else:
        text.append("No invalid final segments.\n")

    text.append("\n\n## Retry parent failures\n")
    text.append(
        "Эта таблица показывает родительские чанки, которые сначала сломались, "
        "а затем retry-план пытался разбить на части.\n\n"
    )
    if retry_table_rows:
        text.append(markdown_table(
            ["Model", "Plan", "Parent time", "Dur", "Reason", "Raw text"],
            retry_table_rows,
        ))
    else:
        text.append("No retry parent failures.\n")

    text.append("\n\n## How to read this\n")
    text.append(
        "- `Valid coverage` — доля аудио, для которой финальные сегменты дали валидный текст.\n"
        "- `Invalid coverage` — доля аудио, которая дошла до ASR, но финально распознана как мусор/ошибка.\n"
        "- Если `existing` плохой, а `existing_split_12s` или `retry_existing_12s` хороший — проблема, вероятно, в длине/склейке чанков.\n"
        "- Если `v3_e2e_rnnt` плохой, а `v3_e2e_ctc` хороший — проблема, вероятно, в конкретном декодере RNNT.\n"
        "- Если raw VAD/fixed/spans тоже ломаются на одном и том же месте — надо отдельно слушать этот участок или проверять ffmpeg decode.\n"
    )

    path.write_text("\n".join(text), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def run_model_plan(
    *,
    audio_path: Path,
    model_name: str,
    model,
    plan_spec: PlanSpec,
    input_regions: list[Region],
    audio,
    sample_rate: int,
    audio_duration: float,
    max_retry_depth: int,
    min_retry_duration: float,
) -> tuple[RunSummary, list[DecodeRow], list[DecodeRow]]:
    started = time.time()
    final_rows: list[DecodeRow] = []
    attempt_rows: list[DecodeRow] = []

    for idx, region in enumerate(input_regions, start=1):
        print(
            f"    [{idx:03d}/{len(input_regions):03d}] "
            f"{format_time(region.start)}–{format_time(region.end)} "
            f"dur={region.duration:.2f}s",
            flush=True,
        )

        if plan_spec.retry:
            retry_max = float(plan_spec.max_duration or 0.0)
            rows, attempts = decode_region_with_retry(
                audio_path=audio_path,
                model_name=model_name,
                model=model,
                plan_name=plan_spec.name,
                audio=audio,
                sample_rate=sample_rate,
                region=region,
                retry_max_duration=retry_max,
                max_retry_depth=max_retry_depth,
                min_retry_duration=min_retry_duration,
            )
            final_rows.extend(rows)
            attempt_rows.extend(attempts)
        else:
            row = decode_region(
                audio_path=audio_path,
                model_name=model_name,
                model=model,
                plan_name=plan_spec.name,
                audio=audio,
                sample_rate=sample_rate,
                region=region,
            )
            final_rows.append(row)
            attempt_rows.append(row)

    elapsed = time.time() - started
    summary = summarize_run(
        audio_path=audio_path,
        model_name=model_name,
        plan_name=plan_spec.name,
        input_regions=input_regions,
        final_rows=final_rows,
        audio_duration=audio_duration,
        elapsed_sec=elapsed,
    )
    return summary, final_rows, attempt_rows



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GigaAM models and segmentation strategies on one audio file."
    )
    parser.add_argument("audio_path", help="Path to audio/video file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[ASR_MODEL, "v3_e2e_ctc"],
        help="GigaAM model names to test. Example: v3_e2e_rnnt v3_e2e_ctc v3_rnnt v3_ctc",
    )
    parser.add_argument(
        "--plans",
        nargs="+",
        default=["existing", "existing_split", "retry_existing", "spans"],
        choices=[
            "existing",
            "existing_split",
            "raw_vad",
            "raw_vad_split",
            "fixed",
            "retry_existing",
            "retry_raw_vad",
            "spans",
        ],
        help="Segmentation plans to run.",
    )
    parser.add_argument(
        "--max-durations",
        nargs="+",
        type=float,
        default=[8.0, 12.0, 15.0, 20.0],
        help="Max segment durations used by *_split, fixed, retry_* plans.",
    )
    parser.add_argument(
        "--fixed-overlap",
        type=float,
        default=0.0,
        help="Overlap in seconds for fixed-window plans.",
    )
    parser.add_argument(
        "--spans",
        nargs="*",
        default=[],
        help="Manual spans START-END in seconds. Example: --spans 18.695-40.025 57.423-84.794",
    )
    parser.add_argument(
        "--out-dir",
        default="data/asr_debug/gigaam_variants",
        help="Output directory for report files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for VAD and GigaAM.",
    )
    parser.add_argument(
        "--max-retry-depth",
        type=int,
        default=2,
        help="Recursive retry depth for retry_* plans.",
    )
    parser.add_argument(
        "--min-retry-duration",
        type=float,
        default=3.0,
        help="Do not split invalid chunks shorter than this duration.",
    )
    parser.add_argument(
        "--skip-vad-details",
        action="store_true",
        help="Still runs VAD, but prints less segmentation detail to console.",
    )
    parser.add_argument(
        "--only-segmentation",
        action="store_true",
        help="Stop after ffprobe/VAD/segment_audio_file diagnostics, no ASR decode.",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_path).resolve()
    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        return 2

    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    ensure_dir(out_dir)

    details_path = out_dir / "details.jsonl"
    attempts_path = out_dir / "attempts.jsonl"
    summary_path = out_dir / "summary.csv"
    report_path = out_dir / "report.md"
    segmentation_path = out_dir / "segmentation.json"

    # Перезаписываем старые результаты этого запуска.
    for p in (details_path, attempts_path, summary_path, report_path, segmentation_path):
        if p.exists():
            p.unlink()

    print("== Input ==")
    print(f"audio={audio_path}")
    print(f"out_dir={out_dir}")
    print(f"models={args.models}")
    print(f"plans={args.plans}")
    print(f"max_durations={args.max_durations}")

    media_probe = probe_media(audio_path)
    media_duration = get_media_duration(media_probe)

    print("\n== Decode audio ==")
    started = time.time()
    audio, sample_rate = load_audio_tensor(audio_path)
    decoded_duration = float(audio.shape[0]) / float(sample_rate)
    audio_duration = decoded_duration or media_duration
    print(
        f"samples={int(audio.shape[0])} sample_rate={sample_rate} "
        f"decoded_duration={decoded_duration:.3f}s media_duration={media_duration:.3f}s "
        f"elapsed={time.time() - started:.2f}s"
    )

    device = select_device(args.device)
    print("\n== VAD and existing grouping ==")
    print(f"device={device}")

    started = time.time()
    raw_vad_regions = get_raw_vad_regions(audio_path, audio_duration, device)
    print(f"raw_vad_count={len(raw_vad_regions)} elapsed={time.time() - started:.2f}s")

    started = time.time()
    existing_regions = get_existing_grouped_regions(audio_path, sample_rate, device)
    print(f"existing_grouped_count={len(existing_regions)} elapsed={time.time() - started:.2f}s")

    manual_spans = parse_spans(args.spans, audio_duration)

    if not args.skip_vad_details:
        for title, regions in [
            ("raw_vad", raw_vad_regions),
            ("existing_grouped", existing_regions),
            ("manual_spans", manual_spans),
        ]:
            cov = union_duration(regions)
            pct = cov / audio_duration * 100.0 if audio_duration > 0 else 0.0
            print(f"\n-- {title}: count={len(regions)} coverage={cov:.3f}s ({pct:.1f}%)")
            for i, r in enumerate(regions, start=1):
                print(f"{i:03d}. {format_time(r.start)} -> {format_time(r.end)} dur={r.duration:.3f}s")

    write_json(
        segmentation_path,
        {
            "audio_path": str(audio_path),
            "ffprobe": media_probe,
            "decoded": {
                "samples": int(audio.shape[0]),
                "sample_rate": sample_rate,
                "duration": audio_duration,
            },
            "raw_vad_regions": [asdict(r) for r in raw_vad_regions],
            "raw_vad_gaps": [asdict(r) for r in find_gaps(raw_vad_regions, audio_duration)],
            "existing_grouped_regions": [asdict(r) for r in existing_regions],
            "existing_grouped_gaps": [asdict(r) for r in find_gaps(existing_regions, audio_duration)],
            "manual_spans": [asdict(r) for r in manual_spans],
        },
    )
    print(f"\nSegmentation JSON: {segmentation_path}")

    if args.only_segmentation:
        print("Stopped because --only-segmentation was set.")
        return 0

    plan_specs = make_plan_specs(args.plans, args.max_durations, args.fixed_overlap)
    if any(spec.base == "spans" for spec in plan_specs) and not manual_spans:
        print("\nWARNING: plan 'spans' requested, but --spans is empty. It will produce empty runs.")

    summaries: list[RunSummary] = []
    all_detail_rows: list[DecodeRow] = []
    all_attempt_rows: list[DecodeRow] = []

    import torch

    for model_idx, model_name in enumerate(args.models, start=1):
        print(f"\n================ MODEL {model_idx}/{len(args.models)}: {model_name} ================")
        model = None
        try:
            started = time.time()
            model = load_gigaam_model(model_name)
            print(f"Loaded model {model_name} in {time.time() - started:.2f}s")

            for plan_idx, spec in enumerate(plan_specs, start=1):
                input_regions = regions_for_plan(
                    spec,
                    existing_regions=existing_regions,
                    raw_vad_regions=raw_vad_regions,
                    manual_spans=manual_spans,
                    audio_duration=audio_duration,
                )
                print(
                    f"\n== Plan {plan_idx}/{len(plan_specs)}: {spec.name} "
                    f"input_regions={len(input_regions)} =="
                )

                if not input_regions:
                    summary = summarize_run(
                        audio_path=audio_path,
                        model_name=model_name,
                        plan_name=spec.name,
                        input_regions=[],
                        final_rows=[],
                        audio_duration=audio_duration,
                        elapsed_sec=0.0,
                        error="no_input_regions",
                    )
                    summaries.append(summary)
                    continue

                try:
                    summary, detail_rows, attempt_rows = run_model_plan(
                        audio_path=audio_path,
                        model_name=model_name,
                        model=model,
                        plan_spec=spec,
                        input_regions=input_regions,
                        audio=audio,
                        sample_rate=sample_rate,
                        audio_duration=audio_duration,
                        max_retry_depth=args.max_retry_depth,
                        min_retry_duration=args.min_retry_duration,
                    )
                except Exception as exc:
                    summary = summarize_run(
                        audio_path=audio_path,
                        model_name=model_name,
                        plan_name=spec.name,
                        input_regions=input_regions,
                        final_rows=[],
                        audio_duration=audio_duration,
                        elapsed_sec=0.0,
                        error=f"plan_exception={type(exc).__name__}: {exc}",
                    )
                    detail_rows = []
                    attempt_rows = []
                    print(f"ERROR in plan {spec.name}: {summary.error}", file=sys.stderr)

                summaries.append(summary)
                all_detail_rows.extend(detail_rows)
                all_attempt_rows.extend(attempt_rows)
                append_jsonl(details_path, detail_rows)
                append_jsonl(attempts_path, attempt_rows)

                print(
                    f"Plan result: valid_cov={summary.valid_coverage_pct:.1f}% "
                    f"invalid_cov={summary.invalid_coverage_pct:.1f}% "
                    f"valid_segments={summary.valid_segment_count}/{summary.final_segment_count} "
                    f"words={summary.total_words} elapsed={summary.elapsed_sec:.1f}s"
                )

        except Exception as exc:
            err = f"model_exception={type(exc).__name__}: {exc}"
            print(f"ERROR loading/running model {model_name}: {err}", file=sys.stderr)
            for spec in plan_specs:
                input_regions = regions_for_plan(
                    spec,
                    existing_regions=existing_regions,
                    raw_vad_regions=raw_vad_regions,
                    manual_spans=manual_spans,
                    audio_duration=audio_duration,
                )
                summaries.append(
                    summarize_run(
                        audio_path=audio_path,
                        model_name=model_name,
                        plan_name=spec.name,
                        input_regions=input_regions,
                        final_rows=[],
                        audio_duration=audio_duration,
                        elapsed_sec=0.0,
                        error=err,
                    )
                )
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    write_summary_csv(summary_path, summaries)
    write_markdown_report(
        path=report_path,
        audio_path=audio_path,
        media_probe=media_probe,
        decoded_duration=audio_duration,
        raw_vad_regions=raw_vad_regions,
        existing_regions=existing_regions,
        manual_spans=manual_spans,
        summaries=summaries,
        detail_rows=all_detail_rows,
        attempt_rows=all_attempt_rows,
    )

    print("\n== Done ==")
    print(f"details:      {details_path}")
    print(f"attempts:     {attempts_path}")
    print(f"summary:      {summary_path}")
    print(f"report:       {report_path}")
    print(f"segmentation: {segmentation_path}")

    best = sorted(
        [s for s in summaries if not s.error],
        key=lambda s: (-s.valid_coverage_pct, s.invalid_coverage_pct, s.elapsed_sec),
    )[:5]
    if best:
        print("\nTop variants:")
        for s in best:
            print(
                f"  {s.model:14s} {s.plan:24s} "
                f"valid={s.valid_coverage_pct:5.1f}% invalid={s.invalid_coverage_pct:5.1f}% "
                f"segments={s.valid_segment_count}/{s.final_segment_count} words={s.total_words}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
