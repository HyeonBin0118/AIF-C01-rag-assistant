import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DATASET_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ragas_dataset.json"
)
OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ragas_scores.csv"
)


def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # RAGAs가 요구하는 컬럼명에 맞춰 변환
    data = {
        "question": [item["question"] for item in raw],
        "contexts": [item["contexts"] for item in raw],
        "answer": [item["answer"] for item in raw],
        "reference": [item["reference"] for item in raw],
    }
    return Dataset.from_dict(data)


def run_evaluation():
    dataset = load_dataset()
    print(f"평가 데이터셋 크기: {len(dataset)}")

    # RAGAs 평가에 쓸 LLM/임베딩 (OpenAI 사용)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("RAGAs 평가 실행 중 (LLM 호출 다수 발생, 몇 분 소요)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "=" * 50)
    print("RAGAs 평가 결과")
    print("=" * 50)
    df = result.to_pandas()
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장됨: {OUTPUT_CSV}")

    return result


if __name__ == "__main__":
    run_evaluation()