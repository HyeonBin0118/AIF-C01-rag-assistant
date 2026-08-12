import os
import csv
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chunking_evaluation.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chunking_strategy_comparison.png"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

STRATEGY_LABELS = {"structural": "구조 기반\n(PART/섹션/서비스)", "fixed": "Fixed-size\n(400자 + overlap)", "semantic": "Semantic\n(임베딩 경계 감지)"}
COLORS = {"structural": "#3B5B92", "fixed": "#D9691E", "semantic": "#4C9A6E"}


def load_results():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_comparison(results):
    strategies = ["structural", "fixed", "semantic"]
    top_ks = ["3", "5"]

    data = {s: {} for s in strategies}
    for row in results:
        data[row["strategy"]][row["top_k"]] = float(row["recall"])

    fig, ax = plt.subplots(figsize=(9, 6.5))

    x = np.arange(len(strategies))
    width = 0.32

    for i, k in enumerate(top_ks):
        values = [data[s][k] * 100 for s in strategies]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=f"Top-{k}",
            color=[COLORS[s] for s in strategies],
            alpha=0.55 if k == "3" else 1.0,
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", fontsize=10, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=11)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_title("청킹 전략별 검색 정확도 비교\n(평가 질문 20개 기준)", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=10.5, loc="lower right", title="검색 범위")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"그래프 저장됨: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    plot_comparison(results)