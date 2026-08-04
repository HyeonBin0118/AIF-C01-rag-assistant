import os
from datasets import load_dataset

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "benchmark_corpus.txt"
)

N_SAMPLES = 50000
MIN_LENGTH = 50  # 너무 짧은 문단 제외
MAX_LENGTH = 500  # 너무 긴 문단은 잘라서 사용


def download_and_prepare():
    print("한국어 위키피디아 데이터셋 로딩 중 (스트리밍 모드)...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ko",
        split="train",
        streaming=True,
    )

    collected = []
    for item in dataset:
        text = item["text"]
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        for para in paragraphs:
            if MIN_LENGTH <= len(para) <= MAX_LENGTH:
                collected.append(para)

            if len(collected) >= N_SAMPLES:
                break

        if len(collected) >= N_SAMPLES:
            break

        if len(collected) % 5000 == 0 and len(collected) > 0:
            print(f"  수집 중: {len(collected)}/{N_SAMPLES}")

    print(f"총 {len(collected)}개 문단 수집 완료")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in collected:
            f.write(line.replace("\n", " ") + "\n")

    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    download_and_prepare()