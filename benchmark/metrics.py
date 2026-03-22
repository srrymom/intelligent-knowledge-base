"""
Метрики качества для оценки компонентов системы.

Реализованы без внешних зависимостей:
  - WER (Word Error Rate)           — для ASR
  - CER (Character Error Rate)      — для ASR
  - faithfulness                    — для суммаризации (прокси галлюцинаций)
  - term_coverage                   — для суммаризации (покрытие терминов)
  - compression_ratio               — для суммаризации
  - precision_at_k, MRR             — для RAG
"""

import re
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Нижний регистр, ё→е, удаление знаков препинания, нормализация пробелов."""
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a: list, b: list) -> int:
    """Расстояние Левенштейна для произвольных списков токенов."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# ASR-метрики
# ---------------------------------------------------------------------------

def wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate = (S + D + I) / N.
    S — замены, D — удаления, I — вставки, N — слов в эталоне.
    """
    ref = _normalize(reference).split()
    hyp = _normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — WER на уровне символов."""
    ref = list(_normalize(reference))
    hyp = list(_normalize(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_edit_distance(ref, hyp) / len(ref), 4)


# ---------------------------------------------------------------------------
# Метрики суммаризации (без эталонных резюме)
# ---------------------------------------------------------------------------

def compression_ratio(transcript: str, summary: str) -> float:
    """
    Коэффициент сжатия = слов_в_резюме / слов_в_транскрипте.
    Целевой диапазон: 0.10 – 0.25.
    """
    t = len(transcript.split())
    s = len(summary.split())
    if t == 0:
        return 0.0
    return round(s / t, 4)


def term_coverage(transcript: str, summary: str, min_len: int = 4) -> float:
    """
    Доля ключевых терминов транскрипта, попавших в резюме.
    Термины — слова длиной >= min_len (простая эвристика без стоп-слов).
    """
    terms = {w for w in _normalize(transcript).split() if len(w) >= min_len}
    if not terms:
        return 0.0
    summary_words = set(_normalize(summary).split())
    return round(len(terms & summary_words) / len(terms), 4)


def faithfulness(transcript: str, summary: str, ngram: int = 3) -> float:
    """
    Достоверность резюме: доля n-грамм резюме, встречающихся в транскрипте.
    Значение 1.0 означает отсутствие галлюцинаций на уровне n-грамм.

    Методология: предложения резюме разбиваются на n-граммы; для каждого
    предложения считается overlap с транскриптом; результат усредняется.
    """
    def ngrams(text: str) -> set:
        words = _normalize(text).split()
        return {tuple(words[i: i + ngram]) for i in range(len(words) - ngram + 1)}

    source_ng = ngrams(transcript)
    sentences = [s.strip() for s in re.split(r"[.!?]", summary) if s.strip()]
    if not sentences:
        return 0.0

    scores = []
    for sent in sentences:
        sent_ng = ngrams(sent)
        if not sent_ng:
            continue
        scores.append(len(sent_ng & source_ng) / len(sent_ng))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# RAG-метрики
# ---------------------------------------------------------------------------

def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Precision@K: доля релевантных документов среди первых K результатов."""
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return round(hits / k, 4)


def reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
    """Reciprocal Rank: 1 / ранг первого релевантного документа."""
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def mean_reciprocal_rank(queries: List[Tuple[List[str], List[str]]]) -> float:
    """MRR по набору запросов. queries: [(relevant_ids, retrieved_ids), ...]"""
    if not queries:
        return 0.0
    return round(sum(reciprocal_rank(rel, ret) for rel, ret in queries) / len(queries), 4)
