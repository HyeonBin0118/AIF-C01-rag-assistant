import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sqlalchemy import text
from app.database import engine

MODEL_NAME = "BAAI/bge-m3"
STRATEGY = "structural"
TOP_K = 5
RRF_K = 60  # RRF 상수

kiwi = Kiwi()


def tokenize_korean(text_str: str):
    """한국어 형태소 분석기로 명사/동사/영단어 위주 토큰 추출."""
    tokens = []
    for token in kiwi.tokenize(text_str):
        # 명사(N), 동사/형용사 어간(V), 영어(SL), 숫자(SN)만 사용
        if token.tag.startswith(("N", "V", "SL", "SN")):
            tokens.append(token.form.lower())
    return tokens


def load_chunks(strategy=STRATEGY):
    """DB에서 특정 전략의 청크를 전부 불러옴 (BM25는 메모리 기반이라 전체 로드 필요)."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT c.id, c.content
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.chunking_strategy = :strategy
                ORDER BY c.chunk_index
                """
            ),
            {"strategy": strategy},
        )
        rows = result.fetchall()
    return [(str(row[0]), row[1]) for row in rows]


def build_bm25_index(chunks):
    """청크 리스트로 BM25 인덱스 생성."""
    tokenized_corpus = [tokenize_korean(content) for _, content in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search(bm25, chunks, query, top_k=TOP_K):
    """BM25로 검색 후 (chunk_id, rank) 리스트 반환."""
    tokenized_query = tokenize_korean(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip([c[0] for c in chunks], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(chunk_id, rank + 1) for rank, (chunk_id, score) in enumerate(ranked[:top_k * 3])]
    # top_k보다 넉넉히 뽑아서 RRF 합산 후보군을 넓힘


def vector_search(model, query, strategy=STRATEGY, top_k=TOP_K * 3):
    """벡터 검색으로 (chunk_id, rank) 리스트 반환."""
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True).tolist()

    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT c.id
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
        rows = result.fetchall()

    return [(str(row[0]), rank + 1) for rank, row in enumerate(rows)]


def reciprocal_rank_fusion(*rank_lists, k=RRF_K):
    """
    여러 검색 결과(각각 [(chunk_id, rank), ...])를 RRF로 합쳐 최종 순위 반환.
    반환: [(chunk_id, rrf_score), ...] (점수 내림차순)
    """
    scores = {}
    for rank_list in rank_lists:
        for chunk_id, rank in rank_list:
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run_search(query, chunks_by_id, model, bm25, chunks):
    vector_ranks = vector_search(model, query)
    bm25_ranks = bm25_search(bm25, chunks, query)
    hybrid_ranks = reciprocal_rank_fusion(vector_ranks, bm25_ranks)

    print(f"\n{'='*80}")
    print(f"쿼리: '{query}'")
    print(f"{'='*80}")

    print("\n[벡터 검색 top-5]")
    for chunk_id, rank in vector_ranks[:5]:
        print(f"  {rank}위: {chunks_by_id[chunk_id][:60]}...")

    print("\n[BM25 검색 top-5]")
    for chunk_id, rank in bm25_ranks[:5]:
        print(f"  {rank}위: {chunks_by_id[chunk_id][:60]}...")

    print("\n[하이브리드(RRF) top-5]")
    for chunk_id, score in hybrid_ranks[:5]:
        print(f"  score={score:.4f}: {chunks_by_id[chunk_id][:60]}...")


if __name__ == "__main__":
    print("청크 로딩 중...")
    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}
    print(f"총 {len(chunks)}개 청크 로드됨")

    print("BM25 인덱스 구축 중...")
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    import sys as _sys
    if len(_sys.argv) > 1:
        query = " ".join(_sys.argv[1:])
    else:
        query = input("검색할 질문을 입력하세요: ")

    run_search(query, chunks_by_id, model, bm25, chunks)