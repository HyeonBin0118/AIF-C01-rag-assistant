import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ragas_scores.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ragas_scores_comparison.png"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

METRIC_LABELS = {
    "faithfulness": "Faithfulness\n(충실도)",
    "answer_relevancy": "Answer Relevancy\n(답변 관련성)",
    "context_precision": "Context Precision\n(문맥 정밀도)",
    "context_recall": "Context Recall\n(문맥 재현율)",
}
COLORS = ["#3B5B92", "#D9691E", "#4C9A6E", "#8B5FBF"]


def load_scores():
    df = pd.read_csv(CSV_PATH)
    metrics = list(METRIC_LABELS.keys())
    return {m: df[m].mean() for m in metrics}


def plot_scores(scores):
    metrics = list(METRIC_LABELS.keys())
    values = [scores[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    bars = ax.bar(
        [METRIC_LABELS[m] for m in metrics], values,
        color=COLORS, width=0.55, edgecolor="white", linewidth=1.5, zorder=3,
    )

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{v:.3f}", ha="center", fontsize=12.5, fontweight="bold", color="#333333",
        )

    ax.set_ylabel("점수", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("RAGAs 파이프라인 평가 결과\n(하이브리드+Reranking 파이프라인, 평가 질문 20개)", fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.axhline(1.0, color="#cccccc", linestyle="--", linewidth=1, zorder=1)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"그래프 저장됨: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    scores = load_scores()
    plot_scores(scores)