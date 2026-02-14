def fmt_time(seconds):
    seconds = float(seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def is_valid_segment(text):
    return bool(text.replace("⁇", "").replace(" ", "").strip())


def _has_timestamps(segments):
    return any(seg["boundaries"] != [0, 0] for seg in segments)


def format_segments(segments, mode):
    if mode == "С временными метками" and _has_timestamps(segments):
        lines = []
        for seg in segments:
            text = seg["transcription"]
            if not is_valid_segment(text):
                continue
            start = fmt_time(seg["boundaries"][0])
            lines.append(f"{start} — {text}")
        return "\n".join(lines)
    else:
        return " ".join(
            seg["transcription"] for seg in segments
            if is_valid_segment(seg["transcription"])
        )
