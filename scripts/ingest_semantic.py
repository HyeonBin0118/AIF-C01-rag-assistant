import sys
import os
import re
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from app.database import SessionLocal
from app.models import Document, Chunk, Embedding

RAW_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "aif_c01_terms.md"
)

MODEL_NAME = "BAAI/bge-m3"
BREAKPOINT_PERCENTILE = 90  # 거리 상위 10%를 경계로 판단
MIN_CHUNK_LINES = 2         # 청크가 너무 잘게 쪼개지는 것 방지
MAX_CHUNK_CHARS = 800       # 청크가 너무 커지는 것 방지 (안전장치)


def split_by_pattern(text: str, pattern: str):
    """주어진 정규식(헤더 패턴) 기준으로 텍스트를 분리. 반환: [(title_or_None, content), ...]"""
    lines = text.split("\n")
    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_title is not None or current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_title, content))
            current_title = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_title, content))

    return sections


def split_by_part(full_text: str):
    """'# PART N. 제목' 기준으로 최상위 문서 단위 분리."""
    pattern = r"^# PART \d+\.\s*(.+)$"
    results = split_by_pattern(full_text, pattern)
    return [(title, content) for title, content in results if title is not None]


def clean_markdown(text: str) -> str:
    """헤더 기호(#)만 제거."""
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text


def get_lines(text: str):
    """빈 줄 제거하고 의미 있는 줄만 추출."""
    lines = [line.strip() for line in text.split("\n")]
    return [line for line in lines if line]


def semantic_split(model, lines):
    """
    줄 단위 임베딩 후 인접 줄 간 거리가 퍼센타일 기준을 넘는 지점에서 분리.
    반환: [chunk_text, ...]
    """
    if len(lines) <= MIN_CHUNK_LINES:
        return ["\n".join(lines)] if lines else []

    embeddings = model.encode(lines, normalize_embeddings=True, show_progress_bar=False)

    # 인접 줄 간 코사인 거리 계산 (정규화된 벡터라 내적 = 코사인 유사도)
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = np.dot(embeddings[i], embeddings[i + 1])
        distances.append(1 - similarity)

    if not distances:
        return ["\n".join(lines)]

    threshold = np.percentile(distances, BREAKPOINT_PERCENTILE)

    # 경계 지점 찾기
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    chunks = []
    start = 0
    for bp in breakpoints:
        segment = lines[start:bp + 1]
        if len(segment) >= MIN_CHUNK_LINES or start == 0:
            chunks.append("\n".join(segment))
            start = bp + 1

    if start < len(lines):
        chunks.append("\n".join(lines[start:]))

    # 너무 큰 청크는 강제로 재분할 (안전장치)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHUNK_CHARS:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), MAX_CHUNK_CHARS):
                final_chunks.append(chunk[i:i + MAX_CHUNK_CHARS])

    return [c for c in final_chunks if c.strip()]


def reset_data(db, strategy):
    """해당 전략(strategy)의 documents/chunks/embeddings만 삭제 (다른 전략 데이터는 보존)."""
    target_doc_ids = [d.id for d in db.query(Document).filter(Document.chunking_strategy == strategy).all()]
    if target_doc_ids:
        target_chunk_ids = [c.id for c in db.query(Chunk).filter(Chunk.document_id.in_(target_doc_ids)).all()]
        db.query(Embedding).filter(Embedding.chunk_id.in_(target_chunk_ids)).delete(synchronize_session=False)
        db.query(Chunk).filter(Chunk.document_id.in_(target_doc_ids)).delete(synchronize_session=False)
        db.query(Document).filter(Document.chunking_strategy == strategy).delete(synchronize_session=False)
    db.commit()


def ingest_semantic():
    if not os.path.exists(RAW_FILE_PATH):
        print(f"파일을 찾을 수 없음: {RAW_FILE_PATH}")
        return

    with open(RAW_FILE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    parts = split_by_part(full_text)
    print(f"총 {len(parts)}개 PART 감지됨 (semantic 청킹 적용)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    print("모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    db = SessionLocal()
    reset_data(db, strategy="semantic")

    total_chunks = 0

    try:
        for part_title, part_content in parts:
            document = Document(
                id=uuid.uuid4(),
                title=part_title,
                source_type="part_note",
                chunking_strategy="semantic",
            )
            db.add(document)
            db.flush()

            cleaned = clean_markdown(part_content)
            lines = get_lines(cleaned)
            semantic_chunks = semantic_split(model, lines)

            for idx, chunk_text in enumerate(semantic_chunks):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=idx,
                    chunk_metadata={"section": None, "service": None},
                )
                db.add(chunk)
                total_chunks += 1

            print(f"  - '{part_title}': {len(semantic_chunks)}개 청크")

        db.commit()
        print(f"\n완료: 문서 {len(parts)}개, 청크 {total_chunks}개 저장됨 (strategy=semantic)")

    except Exception as e:
        db.rollback()
        print(f"에러 발생, 롤백함: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    ingest_semantic()