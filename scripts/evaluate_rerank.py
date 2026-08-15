import sys
import os
import json
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from scripts.hybrid_search import (
    load_chunks,
    build_bm25_index,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    STRATEGY,
)
from scripts.rerank_search import get_candidates, rerank, CANDIDATE_K

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EVAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_questions_hybrid.json"
)


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_rank_of_answer(retrieved_ids, chunks_by_id, keywords):
    """검색 결과 리스트에서 정답 키워드가 처음 등장하는 순위(1-indexed)를 반환. 없으면 None."""
    for rank, cid in enumerate(retrieved_ids, 1):
        content = chunks_by_id[cid]
        if any(kw in content for kw in keywords):
            return rank
    return None


def evaluate():
    questions = load_eval_questions()
    print(f"평가 질문 수: {len(questions)}")

    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)

    methods = ["hybrid_only", "hybrid_rerank"]
    rr_by_method = {m: [] for m in methods}
    detail_rows = []

    for item in questions:
        candidate_ids = get_candidates(
            item["question"], chunks, chunks_by_id, bm25, embed_model, top_k=CANDIDATE_K
        )
        reranked = rerank(item["question"], candidate_ids, chunks_by_id, reranker, top_k=CANDIDATE_K)
        reranked_ids = [cid for cid, _ in reranked]

        hybrid_rank = get_rank_of_answer(candidate_ids, chunks_by_id, item["keywords"])
        rerank_rank = get_rank_of_answer(reranked_ids, chunks_by_id, item["keywords"])

        hybrid_rr = 1 / hybrid_rank if hybrid_rank else 0
        rerank_rr = 1 / rerank_rank if rerank_rank else 0

        rr_by_method["hybrid_only"].append(hybrid_rr)
        rr_by_method["hybrid_rerank"].append(rerank_rr)

        detail_rows.append({
            "question": item["question"],
            "hybrid_rank": hybrid_rank or "-",
            "rerank_rank": rerank_rank or "-",
            "rank_change": (hybrid_rank - rerank_rank) if (hybrid_rank and rerank_rank) else "-",
        })

    print(f"\n{'질문':<45} {'하이브리드 순위':<15} {'Reranking 순위'}")
    print("=" * 85)
    for row in detail_rows:
        print(f"{row['question']:<45} {str(row['hybrid_rank']):<15} {row['rerank_rank']}")

    results = {}
    for method in methods:
        mrr = sum(rr_by_method[method]) / len(rr_by_method[method])
        results[method] = {"method": method, "mrr": round(mrr, 4)}
        print(f"\n{method} MRR: {mrr:.4f}")

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rerank_evaluation.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mrr"])
        writer.writeheader()
        writer.writerows(results.values())
    print(f"\n결과 저장됨: {csv_path}")

    detail_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rerank_evaluation_detail.csv"
    )
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "hybrid_rank", "rerank_rank", "rank_change"])
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"상세 결과 저장됨: {detail_csv_path}")

    return results


if __name__ == "__main__":
    evaluate()