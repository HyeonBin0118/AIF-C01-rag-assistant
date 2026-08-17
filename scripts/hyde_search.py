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
from sqlalchemy import text
from app.database import engine

EMBED_MODEL_NAME = "BAAI/bge-m3"


def generate_hypothetical_document(query: str) -> str:
    """질문에 대한 가상의 답변 문서를 LLM으로 생성."""
    prompt = f"""다음 질문에 대해, AWS AIF-C01 자격증 스터디 노트에 있을 법한 짧은 설명을 작성해줘.
정확한 사실인지 확신할 필요는 없고, 이 질문에 답하는 문서라면 어떤 내용과 형식일지를 그대로 흉내내면 돼.
2~3문장으로 간결하게.

질문: {query}"""

    return chat_completion(prompt)


def hyde_vector_search(model, hypothetical_doc, strategy=STRATEGY, top_k=15):
    """가상 문서를 임베딩해서 벡터 검색 (질문이 아닌 '답변'을 임베딩하는 게 핵심)."""
    doc_vector = model.encode(hypothetical_doc, normalize_embeddings=True).tolist()

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
            {"strategy": strategy, "qv": str(doc_vector), "k": top_k},
        )
        rows = result.fetchall()

    return [(str(row[0]), rank + 1) for rank, row in enumerate(rows)]


def search_with_hyde(query, chunks, chunks_by_id, bm25, embed_model, top_k=15):
    """HyDE 벡터 검색 + BM25(원본 질문 그대로)를 RRF로 통합."""
    hypothetical_doc = generate_hypothetical_document(query)

    hyde_ranks = hyde_vector_search(embed_model, hypothetical_doc, top_k=top_k)
    bm25_ranks = bm25_search(bm25, chunks, query, top_k=top_k // 3)  # BM25는 원본 질문 그대로 사용

    fused = reciprocal_rank_fusion(hyde_ranks, bm25_ranks)
    return fused, hypothetical_doc


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

    results, hypothetical_doc = search_with_hyde(query, chunks, chunks_by_id, bm25, embed_model)

    print(f"\n{'='*80}")
    print(f"원본 질문: {query}")
    print(f"생성된 가상 문서: {hypothetical_doc}")
    print(f"{'='*80}")

    print("\n[HyDE 기반 검색 결과 top-5]")
    for rank, (cid, score) in enumerate(results[:5], 1):
        print(f"  {rank}위 (score={score:.4f}): {chunks_by_id[cid][:60]}...")