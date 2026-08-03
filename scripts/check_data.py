import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Document, Chunk

db = SessionLocal()

doc_count = db.query(Document).count()
chunk_count = db.query(Chunk).count()

print(f"documents: {doc_count}개")
print(f"chunks: {chunk_count}개")

print("\n--- 샘플 청크 5개 ---")
for chunk in db.query(Chunk).limit(5).all():
    section = chunk.chunk_metadata.get("section") or "-"
    service = chunk.chunk_metadata.get("service") or "-"
    print(f"[section: {section} / service: {service}]")
    print(f"  {chunk.content[:100]}...")
    print()

db.close()