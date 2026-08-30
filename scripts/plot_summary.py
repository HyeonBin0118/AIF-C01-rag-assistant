import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "summary_all_experiments.png")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_mrr_values():
    """실험 4, 5의 MRR 결과를 하나로 통합."""
    with open(os.path.join(DATA_DIR, "rerank_evaluation.csv"), "r", encoding="utf-8") as f:
        rerank = {row["method"]: float(row["mrr"]) for row in csv.DictReader(f)}

    with open(os.path.join(DATA_DIR, "query_optimization_evaluation.csv"), "r", encoding="utf-8") as f:
        query_opt = {row["method"]: float(row["mrr"]) for row in csv.DictReader(f)}

    # hybrid_only는 두 실험에서 동일하게 측정됨 (기준선)
    return {
        "하이브리드\n단독 (기준선)": rerank["hybrid_only"],
        "+ Query\nRewriting": query_opt["query_rewrite"],
        "+ Reranking": rerank["hybrid_rerank"],
        "+ HyDE": query_opt["hyde"],
    }


def load_ragas_scores():
    df = pd.read_csv(os.path.join(DATA_DIR, "ragas_scores.csv"))
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    return labels, [df[m].mean() for m in metrics]


def plot_summary():
    mrr_data = load_mrr_values()
    ragas_labels, ragas_values = load_ragas_scores()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.1, 1]})

    fig.suptitle(
        "AIF-C01-rag-assistant 전체 실험 결과 요약",
        fontsize=17, fontweight="bold", y=0.99,
    )

    # 왼쪽: 검색 단계 개선 흐름 (MRR)
    labels = list(mrr_data.keys())
    values = list(mrr_data.values())
    colors = ["#9AA5B1", "#D9691E", "#4C9A6E", "#3B5B92"]

    bars1 = ax1.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5, zorder=3)
    for bar, v in zip(bars1, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{v:.4f}", ha="center", fontsize=12, fontweight="bold", color="#333333",
        )
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("MRR", fontsize=12)
    ax1.set_title("검색 단계별 개선 흐름\n(실험 3~5, 기준선 대비)", fontsize=13.5, fontweight="bold", pad=12)
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)
    ax1.tick_params(axis="x", labelsize=10)

    # 오른쪽: 최종 파이프라인 RAGAs 점수
    ragas_colors = ["#3B5B92", "#D9691E", "#4C9A6E", "#8B5FBF"]
    bars2 = ax2.bar(ragas_labels, ragas_values, color=ragas_colors, width=0.55, edgecolor="white", linewidth=1.5, zorder=3)
    for bar, v in zip(bars2, ragas_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{v:.3f}", ha="center", fontsize=12, fontweight="bold", color="#333333",
        )
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("점수", fontsize=12)
    ax2.set_title("최종 파이프라인 RAGAs 평가\n(실험 6)", fontsize=13.5, fontweight="bold", pad=12)
    ax2.grid(True, axis="y", alpha=0.3, zorder=0)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.axhline(1.0, color="#cccccc", linestyle="--", linewidth=1, zorder=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"그래프 저장됨: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    plot_summary()