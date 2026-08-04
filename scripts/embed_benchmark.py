import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from app.database import SessionLocal, engine
from app.models import BenchmarkVector
from sqlalchemy import text

MODEL_NAME = "BAAI/bge-m3"
CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "benchmark_corpus.txt"
)
BATCH_SIZE = 64


def embed_benchmark_corpus():
    if not os.path.exists(CORPUS_PATH):
        print(f"파일 없음: {CORPUS_PATH}")
        return

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"총 {len(lines)}개 문단 임베딩 예정")

    # 기존 벤치마크 데이터 초기화
    db = SessionLocal()
    db.query(BenchmarkVector).delete()
    db.commit()
    db.close()

    print("모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME)

    total_inserted = 0

    with engine.begin() as conn:
        for i in range(0, len(lines), BATCH_SIZE):
            batch = lines[i:i + BATCH_SIZE]
            vectors = model.encode(
                batch,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            rows = [
                {
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "embedding": str(vector.tolist()),
                }
                for content, vector in zip(batch, vectors)
            ]

            conn.execute(
                text(
                    "INSERT INTO benchmark_vectors (id, content, embedding) "
                    "VALUES (:id, :content, :embedding)"
                ),
                rows,
            )

            total_inserted += len(batch)
            if total_inserted % 1000 == 0:
                print(f"  진행: {total_inserted}/{len(lines)}")

    print(f"완료: {total_inserted}개 벡터 저장됨")


if __name__ == "__main__":
    embed_benchmark_corpus()