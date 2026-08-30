import os
import csv
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rerank_evaluation.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rerank_comparison.png"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

METHOD_LABELS = {"hybrid_only": "하이브리드\n단독", "hybrid_rerank": "하이브리드\n+ Reranking"}
COLORS = {"hybrid_only": "#9AA5B1", "hybrid_rerank": "#4C9A6E"}


def load_results():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["method"]: float(row["mrr"]) for row in reader}


def plot_comparison(data):
    methods = ["hybrid_only", "hybrid_rerank"]
    values = [data[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 6.5))

    bars = ax.bar(
        [METHOD_LABELS[m] for m in methods], values,
        color=[COLORS[m] for m in methods],
        width=0.5, edgecolor="white", linewidth=1.5, zorder=3,
    )

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{v:.4f}", ha="center", fontsize=13, fontweight="bold", color="#333333",
        )

    ax.set_ylabel("MRR (Mean Reciprocal Rank)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Reranking 적용 전후 비교\n(평가 질문 20개, MRR 기준)", fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", labelsize=11)

    # 개선폭 화살표+텍스트
    improvement = values[1] - values[0]
    ax.annotate(
        f"+{improvement:.4f}",
        xy=(1, values[1]), xytext=(0.5, (values[0] + values[1]) / 2 + 0.05),
        fontsize=11, color="#2E7D32", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.3),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"그래프 저장됨: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    data = load_results()
    plot_comparison(data)