import sys
import os
import re
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Document, Chunk, Embedding

RAW_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "aif_c01_terms.md"
)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


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
    """헤더 기호(#)만 제거하고 나머지 텍스트는 그대로 유지 (fixed-size는 구조를 무시하므로)."""
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text


def split_fixed_size(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    순수 글자 수 기준으로 청크를 나눔. overlap만큼 이전 청크 끝부분을 다음 청크 앞에 겹침.
    반환: [content_str, ...]
    """
    text = re.sub(r"\n{2,}", "\n", text).strip()  # 빈 줄 정리
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def reset_data(db, strategy):
    """해당 전략(strategy)의 documents/chunks/embeddings만 삭제 (다른 전략 데이터는 보존)."""
    target_doc_ids = [d.id for d in db.query(Document).filter(Document.chunking_strategy == strategy).all()]
    if target_doc_ids:
        target_chunk_ids = [c.id for c in db.query(Chunk).filter(Chunk.document_id.in_(target_doc_ids)).all()]
        db.query(Embedding).filter(Embedding.chunk_id.in_(target_chunk_ids)).delete(synchronize_session=False)
        db.query(Chunk).filter(Chunk.document_id.in_(target_doc_ids)).delete(synchronize_session=False)
        db.query(Document).filter(Document.chunking_strategy == strategy).delete(synchronize_session=False)
    db.commit()


def ingest_fixed():
    if not os.path.exists(RAW_FILE_PATH):
        print(f"파일을 찾을 수 없음: {RAW_FILE_PATH}")
        return

    with open(RAW_FILE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    parts = split_by_part(full_text)
    print(f"총 {len(parts)}개 PART 감지됨 (fixed-size 청킹 적용)")

    db = SessionLocal()
    reset_data(db, strategy="fixed")

    total_chunks = 0

    try:
        for part_title, part_content in parts:
            document = Document(
                id=uuid.uuid4(),
                title=part_title,
                source_type="part_note",
                chunking_strategy="fixed",
            )
            db.add(document)
            db.flush()

            cleaned = clean_markdown(part_content)
            fixed_chunks = split_fixed_size(cleaned)

            for idx, chunk_text in enumerate(fixed_chunks):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=idx,
                    chunk_metadata={"section": None, "service": None},
                )
                db.add(chunk)
                total_chunks += 1

            print(f"  - '{part_title}': {len(fixed_chunks)}개 청크")

        db.commit()
        print(f"\n완료: 문서 {len(parts)}개, 청크 {total_chunks}개 저장됨 (strategy=fixed)")

    except Exception as e:
        db.rollback()
        print(f"에러 발생, 롤백함: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    ingest_fixed()