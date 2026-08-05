import os
import csv
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "benchmark_results.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "benchmark_recall_vs_latency.png"
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"HNSW": "#3B5B92", "IVFFlat": "#D9691E", "Brute-force": "#999999"}
MARKERS = {"HNSW": "o", "IVFFlat": "^", "Brute-force": "X"}


def load_results():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def short_label(row):
    """긴 파라미터 문자열을 그래프용 짧은 라벨로 변환."""
    if row["index_type"] == "HNSW":
        parts = dict(p.strip().split("=") for p in row["params"].split(","))
        return f"m={parts['m']}, ef={parts['ef_search']}"
    elif row["index_type"] == "IVFFlat":
        parts = dict(p.strip().split("=") for p in row["params"].split(","))
        return f"lists={parts['lists']}, probes={parts['probes']}"
    return row["params"]


def plot_recall_vs_latency(ax, results):
    # 파라미터 조합별로 라벨 위치를 수동 지정 (자동 배치의 겹침 문제 해결)
    manual_offsets = {
        "m=8, ef=20": (12, -30),
        "m=8, ef=100": (12, 16),
        "m=16, ef=20": (-110, 16),
        "m=16, ef=100": (12, 16),
        "m=32, ef=100": (-30, 45),
        "lists=100, probes=1": (12, -8),
        "lists=100, probes=10": (12, 30),
        "lists=200, probes=10": (12, -34),
        "lists=200, probes=20": (-125, 10),
    }

    # recall = 1.0 기준선
    ax.axhline(1.0, color="#cccccc", linestyle="--", linewidth=1, zorder=1)
    ax.text(
        0.72, 1.008, "recall = 1.0 (완전탐색과 동일한 정확도)",
        fontsize=8.5, color="#999999", transform=ax.get_yaxis_transform(),
    )

    for row in results:
        index_type = row["index_type"]
        recall = float(row["avg_recall"])
        latency = float(row["median_latency_ms"])

        ax.scatter(
            latency, recall,
            color=COLORS.get(index_type, "black"),
            marker=MARKERS.get(index_type, "o"),
            s=170,
            label=index_type,
            edgecolors="white",
            linewidths=1.8,
            zorder=4,
        )

        if index_type == "Brute-force":
            label_text = "Brute-force\n(인덱스 없음)"
            ox, oy = -100, -8
        else:
            label_text = short_label(row)
            ox, oy = manual_offsets.get(label_text, (12, 14))

        ax.annotate(
            label_text,
            (latency, recall),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=9,
            color="#2b2b2b",
            ha="left" if ox > 0 else "right",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="#dddddd",
                linewidth=0.8,
                alpha=0.92,
            ),
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8, zorder=2),
            zorder=5,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    legend = ax.legend(
        unique.values(), unique.keys(),
        fontsize=11, loc="lower right", frameon=True,
        title="인덱스 종류", title_fontsize=10.5,
    )
    legend.get_frame().set_alpha(0.95)

    ax.set_xlabel("중앙값 지연시간 (ms, 로그 스케일 · 낮을수록 좋음)", fontsize=12, labelpad=10)
    ax.set_ylabel("Recall@10 (높을수록 좋음)", fontsize=12, labelpad=10)
    ax.set_title("Recall vs Latency 트레이드오프", fontsize=14, fontweight="bold", pad=14)
    ax.set_xscale("log")

    all_latencies = [float(r["median_latency_ms"]) for r in results]
    ax.set_xlim(min(all_latencies) * 0.65, max(all_latencies) * 1.7)
    ax.set_ylim(0.58, 1.08)

    ax.grid(True, which="major", alpha=0.35, zorder=0)
    ax.grid(True, which="minor", alpha=0.12, zorder=0)
    ax.tick_params(labelsize=10.5)


def plot_build_time(ax, results):
    non_brute = [r for r in results if r["index_type"] != "Brute-force"]
    non_brute.sort(key=lambda r: float(r["build_time_sec"]))

    labels = [short_label(r) for r in non_brute]
    build_times = [float(r["build_time_sec"]) for r in non_brute]
    bar_colors = [COLORS[r["index_type"]] for r in non_brute]

    bars = ax.barh(labels, build_times, color=bar_colors, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, t in zip(bars, build_times):
        ax.text(
            bar.get_width() + max(build_times) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s",
            va="center", fontsize=9.5, color="#333333",
        )

    ax.set_xlabel("인덱스 빌드 시간 (초)", fontsize=12, labelpad=10)
    ax.set_title("인덱스 구축 비용", fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, max(build_times) * 1.2)


def plot_all(results):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    fig.suptitle(
        "벡터 인덱스 파라미터별 성능 비교 (AWS AIF-C01 벤치마크 코퍼스 5만개 기준)",
        fontsize=15.5, fontweight="bold", y=0.98,
    )

    plot_recall_vs_latency(ax1, results)
    plot_build_time(ax2, results)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"그래프 저장됨: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    plot_all(results)