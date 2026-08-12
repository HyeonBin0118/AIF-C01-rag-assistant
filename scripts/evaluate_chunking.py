import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from app.database import engine

MODEL_NAME = "BAAI/bge-m3"
EVAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_questions.json"
)
STRATEGIES = ["structural", "fixed", "semantic"]
TOP_K_LIST = [3, 5]


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def search_chunks(conn, model, query, strategy, top_k):
    """특정 전략(strategy)의 청크들 중에서만 검색."""
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True).tolist()

    result = conn.execute(
        text(
            """
            SELECT c.content
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            JOIN documents d ON d.id = c.document_id
            WHERE d.chunking_strategy = :strategy
            ORDER BY e.embedding <=> (:qv)::vector
            LIMIT :k
            """
        ),
        {"strategy": strategy, "qv": str(query_vector), "k": top_k},
    )
    return [row[0] for row in result]


def is_hit(retrieved_contents, keywords):
    """검색된 청크들 중 하나라도 키워드를 전부(또는 하나라도) 포함하면 hit으로 판단."""
    combined = " ".join(retrieved_contents)
    return any(kw in combined for kw in keywords)


def evaluate():
    questions = load_eval_questions()
    print(f"평가 질문 수: {len(questions)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    results = {}

    with engine.connect() as conn:
        for strategy in STRATEGIES:
            for top_k in TOP_K_LIST:
                hits = 0
                for item in questions:
                    retrieved = search_chunks(conn, model, item["question"], strategy, top_k)
                    if is_hit(retrieved, item["keywords"]):
                        hits += 1

                recall_at_k = hits / len(questions)
                key = f"{strategy}_top{top_k}"
                results[key] = {
                    "strategy": strategy,
                    "top_k": top_k,
                    "hits": hits,
                    "total": len(questions),
                    "recall": round(recall_at_k, 4),
                }
                print(f"  {strategy} (top-{top_k}): {hits}/{len(questions)} = {recall_at_k:.2%}")

    print("\n" + "=" * 60)
    print(f"{'전략':<15} {'top-k':<8} {'Hit':<10} {'Recall'}")
    print("=" * 60)
    for key, r in results.items():
        print(f"{r['strategy']:<15} {r['top_k']:<8} {r['hits']}/{r['total']:<8} {r['recall']:.2%}")
    print("=" * 60)

    # CSV로도 저장
    import csv
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "chunking_evaluation.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "top_k", "hits", "total", "recall"])
        writer.writeheader()
        writer.writerows(results.values())
    print(f"\n결과 저장됨: {csv_path}")

    return results


if __name__ == "__main__":
    evaluate()