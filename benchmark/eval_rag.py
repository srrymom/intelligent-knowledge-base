"""
Оценка качества RAG-поиска через синтетические QA-пары.

Методология (self-retrieval / synthetic QA):
  1. Для каждого чанка из ChromaDB LLM генерирует вопрос по содержанию чанка.
  2. Ретривер получает этот вопрос и возвращает top-K документов.
  3. Проверяем, содержит ли top-K исходный документ (тот, из которого взят чанк).
  4. Метрики: Precision@1, Precision@3, Precision@5, MRR.

Подход обоснован в литературе как «self-retrieval evaluation» или
«synthetic QA evaluation» (Lewis et al., 2020; Gao et al., 2023).

Использование:
    python benchmark/eval_rag.py --n_chunks 30
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.metrics import precision_at_k, reciprocal_rank, mean_reciprocal_rank


def generate_question(chunk_text: str, llm_model: str) -> str:
    """LLM генерирует содержательный вопрос по тексту чанка."""
    import ollama
    resp = ollama.chat(
        model=llm_model,
        messages=[{
            "role": "user",
            "content": (
                "По следующему фрагменту лекции сформулируй один конкретный вопрос, "
                "ответ на который содержится в этом тексте. "
                "Вопрос должен касаться конкретного факта или понятия, а не быть общим. "
                "Ответь только вопросом, без пояснений.\n\n"
                f"{chunk_text[:600]}"
            ),
        }],
        options={"temperature": 0.3},
    )
    return resp.message.content.strip()


def retrieve(question: str, embedder, collection, n_results: int = 5) -> list:
    """Возвращает список meeting_id в порядке убывания релевантности."""
    query_emb = embedder.encode([f"query: {question}"])
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=min(n_results, collection.count()),
        include=["metadatas"],
    )
    return [m.get("meeting_id", "") for m in results["metadatas"][0]]


def run_eval(n_chunks: int = 30) -> tuple:
    """
    Запускает синтетическую QA-оценку.
    Возвращает (results_list, mrr_pairs).
    """
    from rag.engine import collection, embedder
    from shared.config import LLM_MODEL

    total = collection.count()
    if total == 0:
        print("ChromaDB пуста. Добавьте записи через интерфейс.")
        return None, None

    print(f"Всего чанков в ChromaDB: {total}")
    n = min(n_chunks, total)

    # Равномерная выборка чанков
    all_data = collection.get(include=["documents", "metadatas"])
    step = max(1, total // n)
    indices = list(range(0, total, step))[:n]

    chunks = [
        (all_data["ids"][i], all_data["documents"][i], all_data["metadatas"][i])
        for i in indices
    ]

    print(f"Оцениваем {len(chunks)} чанков (равномерная выборка)...\n")

    results = []
    mrr_pairs = []

    for i, (chunk_id, chunk_text, meta) in enumerate(chunks):
        doc_id = meta.get("meeting_id", "")
        title = meta.get("title", doc_id[:20])
        print(f"  [{i + 1:>3}/{len(chunks)}] {title[:35]}", end="  ", flush=True)

        try:
            question = generate_question(chunk_text, LLM_MODEL)
            retrieved_ids = retrieve(question, embedder, collection, n_results=5)

            p1 = precision_at_k([doc_id], retrieved_ids, 1)
            p3 = precision_at_k([doc_id], retrieved_ids, min(3, len(retrieved_ids)))
            p5 = precision_at_k([doc_id], retrieved_ids, min(5, len(retrieved_ids)))
            rr = reciprocal_rank([doc_id], retrieved_ids)
            mrr_pairs.append(([doc_id], retrieved_ids))

            print(f"P@1={p1:.2f}  P@3={p3:.2f}  RR={rr:.2f}")

            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
                "question": question,
                "retrieved_doc_ids": retrieved_ids,
                "p@1": p1,
                "p@3": p3,
                "p@5": p5,
                "rr": rr,
            })

        except Exception as e:
            print(f"ОШИБКА: {e}")

    return results, mrr_pairs


def compute_stats(results: list, mrr_pairs: list) -> dict:
    if not results:
        return {}
    return {
        "n": len(results),
        "p@1": round(sum(r["p@1"] for r in results) / len(results), 4),
        "p@3": round(sum(r["p@3"] for r in results) / len(results), 4),
        "p@5": round(sum(r["p@5"] for r in results) / len(results), 4),
        "mrr": mean_reciprocal_rank(mrr_pairs),
    }


def print_report(stats: dict):
    if not stats:
        return

    mrr = stats["mrr"]
    avg_rank = f"{1 / mrr:.1f}" if mrr > 0 else "∞"

    print("\n" + "=" * 55)
    print("ОЦЕНКА RAG — синтетические QA-пары")
    print("=" * 55)
    print(f"  N запросов:    {stats['n']}")
    print(f"  Precision@1:   {stats['p@1']:.3f}")
    print(f"  Precision@3:   {stats['p@3']:.3f}")
    print(f"  Precision@5:   {stats['p@5']:.3f}")
    print(f"  MRR:           {mrr:.3f}")
    print("=" * 55)
    print("\nИнтерпретация:")
    print(f"  P@1={stats['p@1']:.3f}: первый результат — нужный документ "
          f"в {stats['p@1'] * 100:.0f}% случаев")
    print(f"  MRR={mrr:.3f}: в среднем нужный документ на позиции {avg_rank}")

    p1 = stats["p@1"]
    note = "отлично" if p1 >= 0.8 else ("хорошо" if p1 >= 0.6 else ("приемлемо" if p1 >= 0.4 else "низко"))
    print(f"  Оценка: {note}")


def main():
    parser = argparse.ArgumentParser(description="Оценка RAG (синтетические QA)")
    parser.add_argument("--n_chunks", type=int, default=30, help="Количество чанков")
    parser.add_argument(
        "--output", default="data/benchmark/results/rag_eval.json"
    )
    args = parser.parse_args()

    results, mrr_pairs = run_eval(args.n_chunks)
    if results is None:
        sys.exit(1)

    stats = compute_stats(results, mrr_pairs)
    print_report(stats)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": stats, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {out}")

    return stats


if __name__ == "__main__":
    main()
