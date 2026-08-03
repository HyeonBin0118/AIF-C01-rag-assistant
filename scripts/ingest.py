import sys
import os
import re
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Document, Chunk

RAW_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "aif_c01_terms.md"
)


def split_by_pattern(text: str, pattern: str):
    """
    주어진 정규식(헤더 패턴) 기준으로 텍스트를 분리.
    반환: [(title_or_None, content), ...]
    """
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
    """
    '# PART N. 제목' 기준으로 최상위 문서 단위 분리.
    반환: [(part_title, part_content), ...]
    """
    pattern = r"^# PART \d+\.\s*(.+)$"
    results = split_by_pattern(full_text, pattern)
    # 최상단(첫 PART 이전) 텍스트는 제목 없는(None) 항목이라 제외
    return [(title, content) for title, content in results if title is not None]


def split_hierarchical(part_content: str):
    """
    PART 내부를 '##'(소주제) 기준으로 먼저 나누고,
    그 안에 '###'(서비스명) 헤더가 있으면 한 번 더 쪼갬.
    반환: [{"section": str|None, "service": str|None, "content": str}, ...]
    """
    section_pattern = r"^##\s+(.+)$"
    service_pattern = r"^###\s+(.+)$"

    sections = split_by_pattern(part_content, section_pattern)

    # PART 전체에 ## 헤더가 아예 없으면(=단일 섹션) 전체를 하나의 섹션으로 취급
    if not sections:
        sections = [(None, part_content)]

    chunks = []
    for section_title, section_content in sections:
        services = split_by_pattern(section_content, service_pattern)

        if len(services) == 1 and services[0][0] is None:
            # ### 헤더가 없는 섹션 → 섹션 자체를 청크로
            chunks.append({
                "section": section_title,
                "service": None,
                "content": services[0][1],
            })
        else:
            for service_title, service_content in services:
                if service_content.strip():
                    chunks.append({
                        "section": section_title,
                        "service": service_title,
                        "content": service_content,
                    })

    return chunks


def reset_data(db):
    """기존 documents/chunks/embeddings 전부 삭제 (재실행 대비)."""
    from app.models import Embedding
    db.query(Embedding).delete()
    db.query(Chunk).delete()
    db.query(Document).delete()
    db.commit()


def ingest():
    if not os.path.exists(RAW_FILE_PATH):
        print(f"파일을 찾을 수 없음: {RAW_FILE_PATH}")
        return

    with open(RAW_FILE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    parts = split_by_part(full_text)
    print(f"총 {len(parts)}개 PART 감지됨")

    db = SessionLocal()
    reset_data(db)  # 재실행 시 중복 방지

    total_chunks = 0

    try:
        for part_title, part_content in parts:
            document = Document(
                id=uuid.uuid4(),
                title=part_title,
                source_type="part_note",
            )
            db.add(document)
            db.flush()

            hierarchical_chunks = split_hierarchical(part_content)

            for idx, chunk_data in enumerate(hierarchical_chunks):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    content=chunk_data["content"],
                    chunk_index=idx,
                    chunk_metadata={
                        "section": chunk_data["section"],
                        "service": chunk_data["service"],
                    },
                )
                db.add(chunk)
                total_chunks += 1

            print(f"  - '{part_title}': {len(hierarchical_chunks)}개 청크")

        db.commit()
        print(f"\n완료: 문서 {len(parts)}개, 청크 {total_chunks}개 저장됨")

    except Exception as e:
        db.rollback()
        print(f"에러 발생, 롤백함: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    ingest()