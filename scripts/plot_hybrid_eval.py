import os
import csv
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "hybrid_evaluation.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "hybrid_search_comparison.png"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

METHOD_LABELS = {"vector": "벡터 단독", "bm25": "BM25 단독", "hybrid": "하이브리드(RRF)"}
COLORS = {"vector": "#3B5B92", "bm25": "#D9691E", "hybrid": "#4C9A6E"}


def load_results():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_comparison(results):
    methods = ["vector", "bm25", "hybrid"]
    top_ks = ["3", "5"]

    data = {m: {} for m in methods}
    for row in results:
        data[row["method"]][row["top_k"]] = float(row["recall"])

    fig, ax = plt.subplots(figsize=(9, 6.5))

    x = np.arange(len(methods))
    width = 0.32

    for i, k in enumerate(top_ks):
        values = [data[m][k] * 100 for m in methods]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=f"Top-{k}",
            color=[COLORS[m] for m in methods],
            alpha=0.55 if k == "3" else 1.0,
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                f"{v:.0f}%", ha="center", fontsize=10, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=11)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_title("검색 방식별 정확도 비교\n(평가 질문 20개 기준)", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=10.5, loc="lower right", title="검색 범위")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"그래프 저장됨: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    plot_comparison(results)