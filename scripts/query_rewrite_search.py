import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from app.llm_client import chat_completion
from scripts.hybrid_search import (
    load_chunks,
    build_bm25_index,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    STRATEGY,
)

EMBED_MODEL_NAME = "BAAI/bge-m3"
N_REWRITES = 2  # 원본 포함 총 몇 개 버전으로 확장할지 (원본 + N_REWRITES)


def rewrite_query(original_query: str, n: int = N_REWRITES):
    """LLM으로 원본 질문을 의미는 같지만 표현이 다른 질문 n개로 확장."""
    prompt = f"""다음 질문을 의미는 그대로 유지하면서, 검색에 도움이 되도록 표현이 다른 버전으로 {n}개 만들어줘.
AWS 자격증(AIF-C01) 스터디 자료를 검색하기 위한 질문이야.

원본 질문: {original_query}

출력 형식: 한 줄에 하나씩, 번호나 설명 없이 질문만 출력해줘."""

    result = chat_completion(prompt)
    rewrites = [line.strip() for line in result.split("\n") if line.strip()]
    return rewrites[:n]


def search_with_rewrites(query, chunks, chunks_by_id, bm25, embed_model, top_k=15):
    """원본 + 재작성된 질문들 각각으로 검색 후 RRF로 통합."""
    all_queries = [query] + rewrite_query(query)

    rank_lists = []
    for q in all_queries:
        vector_ranks = vector_search(embed_model, q, top_k=top_k)
        bm25_ranks = bm25_search(bm25, chunks, q, top_k=top_k // 3)
        fused = reciprocal_rank_fusion(vector_ranks, bm25_ranks)
        rank_lists.append([(cid, rank + 1) for rank, (cid, _) in enumerate(fused)])

    final_fused = reciprocal_rank_fusion(*rank_lists)
    return final_fused, all_queries


if __name__ == "__main__":
    print("청크 로딩 중...")
    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}
    print(f"총 {len(chunks)}개 청크 로드됨 (strategy={STRATEGY})")

    print("BM25 인덱스 구축 중...")
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    import sys as _sys
    if len(_sys.argv) > 1:
        query = " ".join(_sys.argv[1:])
    else:
        query = input("검색할 질문을 입력하세요: ")

    results, all_queries = search_with_rewrites(query, chunks, chunks_by_id, bm25, embed_model)

    print(f"\n{'='*80}")
    print(f"원본 질문: {query}")
    print(f"재작성된 질문들: {all_queries[1:]}")
    print(f"{'='*80}")

    print("\n[최종 통합 검색 결과 top-5]")
    for rank, (cid, score) in enumerate(results[:5], 1):
        print(f"  {rank}위 (score={score:.4f}): {chunks_by_id[cid][:60]}...")