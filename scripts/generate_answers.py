import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.llm_client import chat_completion
from scripts.hybrid_search import (
    load_chunks,
    build_bm25_index,
    bm25_search,
    vector_search,
    reciprocal_rank_fusion,
    STRATEGY,
)
from scripts.rerank_search import get_candidates, rerank, CANDIDATE_K

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EVAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_questions_ragas.json"
)
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ragas_dataset.json"
)
FINAL_K = 3  # LLM에 넘길 최종 컨텍스트 개수


def load_eval_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_answer(question, contexts):
    """검색된 컨텍스트를 근거로 답변 생성. 문서에 없으면 모른다고 답하도록 유도."""
    context_text = "\n\n".join([f"[문서 {i+1}]\n{c}" for i, c in enumerate(contexts)])

    prompt = f"""다음은 AWS AIF-C01 스터디 노트에서 검색된 문서입니다. 이 문서만을 근거로 질문에 답해줘.
문서에 없는 내용은 추측하지 말고, 문서에서 찾을 수 없다면 "문서에서 찾을 수 없습니다"라고 답해줘.

{context_text}

질문: {question}

답변 (간결하게 1~2문장):"""

    return chat_completion(prompt, temperature=0.0)


def build_dataset():
    questions = load_eval_questions()
    print(f"질문 수: {len(questions)}")

    print("청크 로딩 중...")
    chunks = load_chunks()
    chunks_by_id = {chunk_id: content for chunk_id, content in chunks}

    print("BM25 인덱스 구축 중...")
    bm25 = build_bm25_index(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)

    dataset = []

    for i, item in enumerate(questions, 1):
        print(f"진행: {i}/{len(questions)} - {item['question']}")

        candidate_ids = get_candidates(
            item["question"], chunks, chunks_by_id, bm25, embed_model, top_k=CANDIDATE_K
        )
        reranked = rerank(item["question"], candidate_ids, chunks_by_id, reranker, top_k=FINAL_K)
        top_ids = [cid for cid, _ in reranked]
        contexts = [chunks_by_id[cid] for cid in top_ids]

        answer = generate_answer(item["question"], contexts)

        dataset.append({
            "question": item["question"],
            "contexts": contexts,
            "answer": answer,
            "reference": item["reference"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(dataset)}개 항목 저장됨 → {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()