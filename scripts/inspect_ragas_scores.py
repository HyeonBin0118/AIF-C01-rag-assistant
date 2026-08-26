import pandas as pd
import os

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ragas_scores.csv"
)

df = pd.read_csv(CSV_PATH)

print("=" * 100)
print("Answer Relevancy 하위 5개 질문")
print("=" * 100)
worst = df.sort_values("answer_relevancy").head(5)
for _, row in worst.iterrows():
    print(f"\n질문: {row['user_input']}")
    print(f"답변: {row['response']}")
    print(f"answer_relevancy: {row['answer_relevancy']:.4f}")