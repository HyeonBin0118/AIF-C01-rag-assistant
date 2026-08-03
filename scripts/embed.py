import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from app.database import SessionLocal
from app.models import Chunk, Embedding
import uuid

MODEL_NAME = "BAAI/bge-m3"


def embed_all_chunks():
    print(f"모델 로딩 중: {MODEL_NAME} (처음 실행 시 다운로드 때문에 시간 걸림)")
    model = SentenceTransformer(MODEL_NAME)

    db = SessionLocal()

    try:
        chunks = db.query(Chunk).all()
        print(f"임베딩 대상 청크 수: {len(chunks)}")

        # 이미 임베딩된 chunk_id는 스킵 (재실행 대비)
        existing_ids = {e.chunk_id for e in db.query(Embedding).all()}
        target_chunks = [c for c in chunks if c.id not in existing_ids]
        print(f"신규로 임베딩할 청크 수: {len(target_chunks)}")

        if not target_chunks:
            print("이미 전부 임베딩되어 있음.")
            return

        texts = [c.content for c in target_chunks]

        # BGE 계열은 문서 임베딩 시 접두어 없이 그대로 인코딩 (쿼리 임베딩할 때만 별도 처리 예정)
        vectors = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,  # 코사인 유사도 계산 시 유리
        )

        for chunk, vector in zip(target_chunks, vectors):
            embedding = Embedding(
                id=uuid.uuid4(),
                chunk_id=chunk.id,
                embedding=vector.tolist(),
                model_name=MODEL_NAME,
            )
            db.add(embedding)

        db.commit()
        print(f"완료: {len(target_chunks)}개 청크 임베딩 저장됨")

    except Exception as e:
        db.rollback()
        print(f"에러 발생, 롤백함: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    embed_all_chunks()