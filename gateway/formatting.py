def fmt_time(seconds):
    seconds = float(seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def is_valid_segment(text):
    return bool(text.replace("вЃ‡", "").replace(" ", "").strip())


def _has_timestamps(segments):
    return any(seg["boundaries"] != [0, 0] for seg in segments)


def format_segments(segments, mode):
    if mode == "РЎ РІСЂРµРјРµРЅРЅС‹РјРё РјРµС‚РєР°РјРё" and _has_timestamps(segments):
        lines = []
        for seg in segments:
            text = seg["transcription"]
            if not is_valid_segment(text):
                continue
            start = fmt_time(seg["boundaries"][0])
            lines.append(f"{start} вЂ” {text}")
        return "\n".join(lines)
    else:
        return " ".join(
            seg["transcription"] for seg in segments
            if is_valid_segment(seg["transcription"])
        )


# Р¦РІРµС‚РѕРІС‹Рµ РїР°СЂС‹ bg/fg РґР»СЏ С‚РµРіРѕРІ С‚РµРј (С†РёРєР»РёС‡РµСЃРєРё)
_TAG_COLORS = [
    ("#dbeafe", "#1d4ed8"),
    ("#dcfce7", "#15803d"),
    ("#fef9c3", "#854d0e"),
    ("#fce7f3", "#9d174d"),
    ("#ede9fe", "#6d28d9"),
    ("#ffedd5", "#c2410c"),
    ("#f1f5f9", "#475569"),
]


def render_topics(topics):
    """Р РµРЅРґРµСЂРёС‚ СЃРїРёСЃРѕРє С‚РµРј РєР°Рє С†РІРµС‚РЅС‹Рµ С‡РёРїС‹ (HTML)."""
    if not topics:
        return ""
    chips = []
    for i, topic in enumerate(topics):
        bg, fg = _TAG_COLORS[i % len(_TAG_COLORS)]
        chips.append(
            f'<span style="background:{bg};color:{fg};padding:3px 12px;'
            f'border-radius:999px;font-size:13px;font-weight:500;'
            f'white-space:nowrap;display:inline-block">{topic}</span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 2px 0">'
        + "".join(chips)
        + "</div>"
    )


def render_word_stats(segments):
    """Р РµРЅРґРµСЂРёС‚ СЃС‚СЂРѕРєСѓ СЃС‚Р°С‚РёСЃС‚РёРєРё: СЃР»РѕРІР°, РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ, РІСЂРµРјСЏ С‡С‚РµРЅРёСЏ (HTML)."""
    words = sum(
        len(s["transcription"].split())
        for s in segments
        if is_valid_segment(s["transcription"])
    )
    if words == 0:
        return ""

    read_min = max(1, words // 200)

    extra = ""
    if _has_timestamps(segments):
        total_sec = max(
            (s["boundaries"][1] for s in segments if s["boundaries"][1] > 0),
            default=0,
        )
        if total_sec > 0:
            m, s = divmod(int(total_sec), 60)
            extra = f' &nbsp;В·&nbsp; <b>{m}:{s:02d}</b> Р°СѓРґРёРѕ'

    return (
        f'<p style="font-size:13px;margin:4px 0 8px 0;opacity:0.75">'
        f'<b>{words:,}</b> СЃР»РѕРІ{extra} &nbsp;В·&nbsp; ~<b>{read_min}</b> РјРёРЅ С‡С‚РµРЅРёСЏ'
        f'</p>'
    )


def render_kb_stats(count, total_words, total_audio_sec):
    """Р РµРЅРґРµСЂРёС‚ С€Р°РїРєСѓ Р±Р°Р·С‹ Р·РЅР°РЅРёР№ СЃРѕ СЃС‚Р°С‚РёСЃС‚РёРєРѕР№ (HTML)."""
    if count == 0:
        return '<p style="font-size:13px;margin:0 0 10px 0;opacity:0.5">Р‘Р°Р·Р° Р·РЅР°РЅРёР№ РїСѓСЃС‚Р°</p>'

    words_fmt = f"{total_words:,}".replace(",", "\u202f")  # РЅРµСЂР°Р·СЂС‹РІРЅС‹Р№ РїСЂРѕР±РµР»

    audio_part = ""
    if total_audio_sec > 0:
        h = total_audio_sec // 3600
        m = (total_audio_sec % 3600) // 60
        if h > 0:
            audio_part = f' &nbsp;В·&nbsp; <b>{h} С‡ {m} РјРёРЅ</b> Р°СѓРґРёРѕ'
        else:
            audio_part = f' &nbsp;В·&nbsp; <b>{m} РјРёРЅ</b> Р°СѓРґРёРѕ'

    return (
        f'<div style="border:1px solid rgba(128,128,128,0.2);border-radius:8px;'
        f'padding:8px 14px;margin-bottom:10px;font-size:13px;opacity:0.85">'
        f'<b>{count}</b> {"Р»РµРєС†РёСЏ" if count == 1 else "Р»РµРєС†РёРё" if 2 <= count <= 4 else "Р»РµРєС†РёР№"}'
        f' &nbsp;В·&nbsp; <b>{words_fmt}</b> СЃР»РѕРІ'
        f'{audio_part}'
        f'</div>'
    )


def render_progress_bar(stage, cur, total):
    """РџСЂРѕРіСЂРµСЃСЃ-Р±Р°СЂ LLM-СЃСѓРјРјР°СЂРёР·Р°С†РёРё (determinate)."""
    pct = round(cur / total * 100) if total else 0
    if stage == "chunk":
        label = f"РЎСѓРјРјР°СЂРёР·Р°С†РёСЏ: С‡Р°СЃС‚СЊ {cur}/{max(total - 3, 1)}"
    elif stage == "merge":
        label = "РћР±СЉРµРґРёРЅРµРЅРёРµ С‡Р°СЃС‚РµР№..."
    elif stage == "title":
        label = "Р“РµРЅРµСЂР°С†РёСЏ Р·Р°РіРѕР»РѕРІРєР° Рё С‚РµРј..."
    elif stage == "report":
        label = "РЎС‚СЂСѓРєС‚СѓСЂРёСЂРѕРІР°РЅРёРµ РѕС‚С‡С‘С‚Р°..."
    else:
        label = "РћР±СЂР°Р±РѕС‚РєР°..."
    return (
        f'<div style="margin:4px 0 8px 0">'
        f'<div style="font-size:12px;opacity:0.7;margin-bottom:5px">{label} вЂ” {pct}%</div>'
        f'<div style="background:rgba(128,128,128,0.15);border-radius:4px;height:6px;overflow:hidden">'
        f'<div style="background:#3b82f6;height:100%;width:{pct}%;'
        f'transition:width 0.4s ease;border-radius:4px"></div>'
        f'</div></div>'
    )


def render_asr_progress(current=None, total=None):
    """Прогресс-бар транскрипции: determinate при current/total, иначе indeterminate."""
    if total and int(total) > 0:
        cur = max(0, int(current or 0))
        tot = max(1, int(total))
        pct = min(100, round(cur / tot * 100))
        return (
            '<div style="margin:4px 0 8px 0">'
            f'<div style="font-size:12px;opacity:0.7;margin-bottom:5px">Транскрипция аудио: сегмент {cur}/{tot} — {pct}%</div>'
            '<div style="background:rgba(128,128,128,0.15);border-radius:4px;height:6px;overflow:hidden">'
            f'<div style="background:#3b82f6;height:100%;width:{pct}%;transition:width 0.3s ease;border-radius:4px"></div>'
            '</div>'
            '</div>'
        )

    return (
        '<div style="margin:4px 0 8px 0">'
        '<div style="font-size:12px;opacity:0.7;margin-bottom:5px">Транскрипция аудио...</div>'
        '<div style="background:rgba(128,128,128,0.15);border-radius:4px;height:6px;overflow:hidden">'
        '<div style="background:#3b82f6;height:100%;width:100%;border-radius:4px;'
        'animation:_asr-pulse 1.6s ease-in-out infinite"></div>'
        '</div>'
        '<style>@keyframes _asr-pulse{0%,100%{opacity:0.3}50%{opacity:1}}</style>'
        '</div>'
    )
def render_waiting_progress(label="РћР¶РёРґР°РЅРёРµ GPU РґР»СЏ LLM..."):
    """РџСЂРѕРіСЂРµСЃСЃ-Р±Р°СЂ РѕР¶РёРґР°РЅРёСЏ РѕС‡РµСЂРµРґРё/РїРµСЂРµРєР»СЋС‡РµРЅРёСЏ GPU."""
    return (
        '<div style="margin:4px 0 8px 0">'
        f'<div style="font-size:12px;opacity:0.7;margin-bottom:5px">{label}</div>'
        '<div style="background:rgba(128,128,128,0.15);border-radius:4px;height:6px;overflow:hidden">'
        '<div style="background:#64748b;height:100%;width:100%;border-radius:4px;'
        'animation:_wait-pulse 1.8s ease-in-out infinite"></div>'
        '</div>'
        '<style>@keyframes _wait-pulse{0%,100%{opacity:0.25}50%{opacity:0.9}}</style>'
        '</div>'
    )

