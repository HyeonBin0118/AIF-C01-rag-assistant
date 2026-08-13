import sys
import os
import json
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from app.database import engine
from scripts.hybrid_search import (
    load_chunks,
    build_bm25_index,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    STRATEGY,
)

MODEL_NAME = "BAAI/bge-m3"
EVAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_questions_hybrid.json"
)
TOP_K_LIST = [3, 5]


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_hit(retrieved_ids, chunks_by_id, keywords):
    combined = " ".join(chunks_by_id[cid] for cid in retrieved_ids)
    return any(kw in combined for kw in keywords)


def evaluate():
    questions = load_eval_questions()
    print(f"평가 질문 수: {len(questions)}")

    print("청크 로딩 중...")
    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}
    print(f"총 {len(chunks)}개 청크 로드됨 (strategy={STRATEGY})")

    print("BM25 인덱스 구축 중...")
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    results = {}
    methods = ["vector", "bm25", "hybrid"]

    for top_k in TOP_K_LIST:
        for method in methods:
            hits = 0
            for item in questions:
                vector_ranks = vector_search(model, item["question"], top_k=max(top_k * 3, 15))
                bm25_ranks = bm25_search(bm25, chunks, item["question"], top_k=max(top_k, 5))

                if method == "vector":
                    top_ids = [cid for cid, _ in vector_ranks[:top_k]]
                elif method == "bm25":
                    top_ids = [cid for cid, _ in bm25_ranks[:top_k]]
                else:  # hybrid
                    fused = reciprocal_rank_fusion(vector_ranks, bm25_ranks)
                    top_ids = [cid for cid, _ in fused[:top_k]]

                if is_hit(top_ids, chunks_by_id, item["keywords"]):
                    hits += 1

            recall = hits / len(questions)
            key = f"{method}_top{top_k}"
            results[key] = {
                "method": method,
                "top_k": top_k,
                "hits": hits,
                "total": len(questions),
                "recall": round(recall, 4),
            }
            print(f"  {method} (top-{top_k}): {hits}/{len(questions)} = {recall:.2%}")

    print("\n" + "=" * 60)
    print(f"{'방식':<12} {'top-k':<8} {'Hit':<10} {'Recall'}")
    print("=" * 60)
    for r in results.values():
        print(f"{r['method']:<12} {r['top_k']:<8} {r['hits']}/{r['total']:<8} {r['recall']:.2%}")
    print("=" * 60)

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "hybrid_evaluation.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "top_k", "hits", "total", "recall"])
        writer.writeheader()
        writer.writerows(results.values())
    print(f"\n결과 저장됨: {csv_path}")

    return results


if __name__ == "__main__":
    evaluate()