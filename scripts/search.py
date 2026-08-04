import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from app.database import SessionLocal
from app.models import Chunk, Embedding, Document

MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5


def search(query: str, top_k: int = TOP_K):
    model = SentenceTransformer(MODEL_NAME)

    # BGE 계열은 쿼리 임베딩 시 지시문(instruction) 접두어를 붙이면 검색 정확도가 올라감
    query_with_prefix = f"query: {query}"
    query_vector = model.encode(query_with_prefix, normalize_embeddings=True).tolist()

    db = SessionLocal()

    try:
        # pgvector의 코사인 거리 연산자(<=>) 사용 - 값이 작을수록 유사
        results = (
            db.query(
                Chunk,
                Embedding.embedding.cosine_distance(query_vector).label("distance"),
                Document.title.label("part_title"),
            )
            .join(Embedding, Embedding.chunk_id == Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .order_by("distance")
            .limit(top_k)
            .all()
        )

        print(f"\n쿼리: '{query}'\n")
        print(f"{'='*80}")

        for rank, (chunk, distance, part_title) in enumerate(results, 1):
            similarity = 1 - distance  # 코사인 거리 → 코사인 유사도로 변환
            service = chunk.chunk_metadata.get("service") or "-"
            section = chunk.chunk_metadata.get("section") or "-"

            print(f"\n[{rank}위] 유사도: {similarity:.4f}")
            print(f"  PART: {part_title}")
            print(f"  섹션: {section} / 항목: {service}")
            print(f"  내용: {chunk.content[:150]}...")

        print(f"\n{'='*80}")

    finally:
        db.close()


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        query = " ".join(_sys.argv[1:])
    else:
        query = input("검색할 질문을 입력하세요: ")

    search(query)