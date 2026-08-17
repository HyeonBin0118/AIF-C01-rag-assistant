import sys
import os
import json
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from scripts.hybrid_search import (
    load_chunks,
    build_bm25_index,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    STRATEGY,
)
from scripts.query_rewrite_search import search_with_rewrites
from scripts.hyde_search import search_with_hyde

EMBED_MODEL_NAME = "BAAI/bge-m3"
EVAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_questions_hybrid.json"
)


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_rank_of_answer(retrieved_ids, chunks_by_id, keywords):
    for rank, cid in enumerate(retrieved_ids, 1):
        if any(kw in chunks_by_id[cid] for kw in keywords):
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

    methods = ["hybrid_only", "query_rewrite", "hyde"]
    rr_by_method = {m: [] for m in methods}
    detail_rows = []

    for i, item in enumerate(questions, 1):
        print(f"\n진행: {i}/{len(questions)} - {item['question']}")

        print("  [1/3] 하이브리드 검색 중...")
        vector_ranks = vector_search(embed_model, item["question"], top_k=15)
        bm25_ranks = bm25_search(bm25, chunks, item["question"], top_k=5)
        hybrid_fused = reciprocal_rank_fusion(vector_ranks, bm25_ranks)
        hybrid_ids = [cid for cid, _ in hybrid_fused]

        print("  [2/3] Query rewriting 중 (LLM 호출)...")
        rewrite_fused, _ = search_with_rewrites(item["question"], chunks, chunks_by_id, bm25, embed_model)
        rewrite_ids = [cid for cid, _ in rewrite_fused]

        print("  [3/3] HyDE 중 (LLM 호출)...")
        hyde_fused, _ = search_with_hyde(item["question"], chunks, chunks_by_id, bm25, embed_model)
        hyde_ids = [cid for cid, _ in hyde_fused]

        ranks = {
            "hybrid_only": get_rank_of_answer(hybrid_ids, chunks_by_id, item["keywords"]),
            "query_rewrite": get_rank_of_answer(rewrite_ids, chunks_by_id, item["keywords"]),
            "hyde": get_rank_of_answer(hyde_ids, chunks_by_id, item["keywords"]),
        }

        row = {"question": item["question"]}
        for method in methods:
            rank = ranks[method]
            rr = 1 / rank if rank else 0
            rr_by_method[method].append(rr)
            row[f"{method}_rank"] = rank or "-"
        detail_rows.append(row)

    results = {}
    print(f"\n{'='*50}")
    for method in methods:
        mrr = sum(rr_by_method[method]) / len(rr_by_method[method])
        results[method] = {"method": method, "mrr": round(mrr, 4)}
        print(f"{method} MRR: {mrr:.4f}")

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "query_optimization_evaluation.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mrr"])
        writer.writeheader()
        writer.writerows(results.values())
    print(f"\n결과 저장됨: {csv_path}")

    detail_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "query_optimization_detail.csv"
    )
    fieldnames = ["question"] + [f"{m}_rank" for m in methods]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"상세 결과 저장됨: {detail_path}")

    return results


if __name__ == "__main__":
    evaluate()