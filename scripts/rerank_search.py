import sys
import os

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

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CANDIDATE_K = 15  # reranker에 넘길 1차 후보 개수
FINAL_K = 5        # 최종 결과 개수


def get_candidates(query, chunks, chunks_by_id, bm25, embed_model, top_k=CANDIDATE_K):
    """하이브리드 검색으로 1차 후보군을 뽑음."""
    vector_ranks = vector_search(embed_model, query, top_k=top_k)
    bm25_ranks = bm25_search(bm25, chunks, query, top_k=top_k // 3)
    fused = reciprocal_rank_fusion(vector_ranks, bm25_ranks)

    candidate_ids = [cid for cid, _ in fused[:top_k]]
    return candidate_ids


def rerank(query, candidate_ids, chunks_by_id, reranker, top_k=FINAL_K):
    """Cross-encoder로 후보들을 재채점하고 재정렬."""
    pairs = [[query, chunks_by_id[cid]] for cid in candidate_ids]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


def run_search(query, chunks, chunks_by_id, bm25, embed_model, reranker):
    candidate_ids = get_candidates(query, chunks, chunks_by_id, bm25, embed_model)

    print(f"\n{'='*80}")
    print(f"쿼리: '{query}'")
    print(f"{'='*80}")

    print(f"\n[1차 후보 (하이브리드 top-{CANDIDATE_K})] - 순서대로")
    for i, cid in enumerate(candidate_ids[:5], 1):
        print(f"  {i}위: {chunks_by_id[cid][:60]}...")

    reranked = rerank(query, candidate_ids, chunks_by_id, reranker)

    print(f"\n[Reranking 후 top-{FINAL_K}]")
    for i, (cid, score) in enumerate(reranked, 1):
        print(f"  {i}위 (score={score:.4f}): {chunks_by_id[cid][:60]}...")


if __name__ == "__main__":
    print("청크 로딩 중...")
    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}
    print(f"총 {len(chunks)}개 청크 로드됨 (strategy={STRATEGY})")

    print("BM25 인덱스 구축 중...")
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    print(f"임베딩 모델 로딩 중: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    print(f"Reranker 모델 로딩 중: {RERANKER_MODEL_NAME} (처음 실행 시 다운로드)")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)

    import sys as _sys
    if len(_sys.argv) > 1:
        query = " ".join(_sys.argv[1:])
    else:
        query = input("검색할 질문을 입력하세요: ")

    run_search(query, chunks, chunks_by_id, bm25, embed_model, reranker)