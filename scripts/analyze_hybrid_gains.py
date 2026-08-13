import sys
import os
import json

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

MODEL_NAME = "BAAI/bge-m3"
EVAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_questions_hybrid.json"
)
TOP_K = 3


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_hit(retrieved_ids, chunks_by_id, keywords):
    combined = " ".join(chunks_by_id[cid] for cid in retrieved_ids)
    return any(kw in combined for kw in keywords)


def analyze():
    questions = load_eval_questions()
    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"{'='*90}")
    print("하이브리드가 개별 방식 대비 개선한 질문 분석 (top-3 기준)")
    print(f"{'='*90}\n")

    found_any = False

    for item in questions:
        vector_ranks = vector_search(model, item["question"], top_k=15)
        bm25_ranks = bm25_search(bm25, chunks, item["question"], top_k=5)
        hybrid_ranks = reciprocal_rank_fusion(vector_ranks, bm25_ranks)

        vector_top3 = [cid for cid, _ in vector_ranks[:TOP_K]]
        bm25_top3 = [cid for cid, _ in bm25_ranks[:TOP_K]]
        hybrid_top3 = [cid for cid, _ in hybrid_ranks[:TOP_K]]

        vector_hit = is_hit(vector_top3, chunks_by_id, item["keywords"])
        bm25_hit = is_hit(bm25_top3, chunks_by_id, item["keywords"])
        hybrid_hit = is_hit(hybrid_top3, chunks_by_id, item["keywords"])

        # 하이브리드만 성공한 케이스 강조
        if hybrid_hit and not (vector_hit and bm25_hit):
            found_any = True
            print(f"질문: {item['question']}")
            print(f"  정답 키워드: {item['keywords']}")
            print(f"  벡터 단독:   {'적중' if vector_hit else '실패'}")
            print(f"  BM25 단독:   {'적중' if bm25_hit else '실패'}")
            print(f"  하이브리드:  {'적중' if hybrid_hit else '실패'}")
            print(f"  하이브리드 top-3 첫 결과: {chunks_by_id[hybrid_top3[0]][:80]}...")
            print()

    if not found_any:
        print("이번 질문셋에서는 하이브리드만 단독으로 성공한 케이스가 없었습니다.")


if __name__ == "__main__":
    analyze()