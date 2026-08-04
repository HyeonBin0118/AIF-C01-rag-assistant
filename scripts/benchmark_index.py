import sys
import os
import time
import statistics
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.database import engine

N_QUERIES = 100
TOP_K = 10
WARMUP_QUERIES = 5


def get_sample_query_vectors(conn, n=N_QUERIES):
    """벤치마크 코퍼스에서 무작위로 n개를 뽑아 쿼리 벡터로 사용."""
    result = conn.execute(
        text(f"SELECT embedding FROM benchmark_vectors ORDER BY RANDOM() LIMIT {n}")
    )
    return [row[0] for row in result]


def get_ground_truth(conn, query_vectors, top_k=TOP_K):
    """인덱스 없이 brute-force로 진짜 정답 top-k ID 집합을 구함."""
    conn.execute(text("SET enable_seqscan = on"))

    ground_truths = []
    for qv in query_vectors:
        result = conn.execute(
            text(
                "SELECT id FROM benchmark_vectors "
                "ORDER BY embedding <=> (:qv)::vector LIMIT :k"
            ),
            {"qv": str(qv), "k": top_k},
        )
        ids = {str(row[0]) for row in result}
        ground_truths.append(ids)
    return ground_truths


def compute_recall(retrieved_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 0.0
    overlap = len(retrieved_ids & ground_truth_ids)
    return overlap / len(ground_truth_ids)


def drop_index(conn, index_name):
    conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
    conn.commit()


def measure_query_latency_and_recall(conn, query_vectors, ground_truths, force_index_scan=True):
    """EXPLAIN ANALYZE로 서버 내부 실행시간만 측정 (네트워크 왕복시간 제외)."""
    if force_index_scan:
        # 플래너가 인덱스 대신 Seq Scan으로 폴백하는 걸 방지 (실제 인덱스 성능만 측정)
        conn.execute(text("SET enable_seqscan = off"))
    else:
        conn.execute(text("SET enable_seqscan = on"))

    # Warm-up (캐시/JIT 영향 제거)
    for qv in query_vectors[:WARMUP_QUERIES]:
        conn.execute(
            text(
                "SELECT id FROM benchmark_vectors "
                "ORDER BY embedding <=> (:qv)::vector LIMIT :k"
            ),
            {"qv": str(qv), "k": TOP_K},
        )

    latencies = []
    recalls = []

    for qv, gt in zip(query_vectors, ground_truths):
        # 서버 실행시간(ms) 정확히 추출
        result = conn.execute(
            text(
                "EXPLAIN (ANALYZE, FORMAT JSON) "
                "SELECT id FROM benchmark_vectors "
                "ORDER BY embedding <=> (:qv)::vector LIMIT :k"
            ),
            {"qv": str(qv), "k": TOP_K},
        )
        plan = result.fetchone()[0]
        exec_time_ms = plan[0]["Execution Time"]
        latencies.append(exec_time_ms)

        # recall 계산용으로 실제 결과 조회
        result2 = conn.execute(
            text(
                "SELECT id FROM benchmark_vectors "
                "ORDER BY embedding <=> (:qv)::vector LIMIT :k"
            ),
            {"qv": str(qv), "k": TOP_K},
        )
        retrieved_ids = {str(row[0]) for row in result2}
        recalls.append(compute_recall(retrieved_ids, gt))

    sorted_latencies = sorted(latencies)
    p95_index = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)

    return {
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted_latencies[p95_index], 2),
        "avg_recall": round(sum(recalls) / len(recalls), 4),
    }


def benchmark_hnsw(conn, query_vectors, ground_truths, m, ef_construction, ef_search):
    index_name = "idx_benchmark_hnsw"
    drop_index(conn, index_name)

    build_start = time.time()
    conn.execute(
        text(
            f"CREATE INDEX {index_name} ON benchmark_vectors "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction})"
        )
    )
    conn.commit()
    build_time = time.time() - build_start

    conn.execute(text(f"SET hnsw.ef_search = {ef_search}"))

    metrics = measure_query_latency_and_recall(conn, query_vectors, ground_truths, force_index_scan=True)

    drop_index(conn, index_name)

    return {
        "index_type": "HNSW",
        "params": f"m={m}, ef_construction={ef_construction}, ef_search={ef_search}",
        "build_time_sec": round(build_time, 2),
        **metrics,
    }


def benchmark_ivfflat(conn, query_vectors, ground_truths, lists, probes):
    index_name = "idx_benchmark_ivfflat"
    drop_index(conn, index_name)

    build_start = time.time()
    conn.execute(
        text(
            f"CREATE INDEX {index_name} ON benchmark_vectors "
            f"USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {lists})"
        )
    )
    conn.commit()
    build_time = time.time() - build_start

    conn.execute(text(f"SET ivfflat.probes = {probes}"))

    metrics = measure_query_latency_and_recall(conn, query_vectors, ground_truths, force_index_scan=True)

    drop_index(conn, index_name)

    return {
        "index_type": "IVFFlat",
        "params": f"lists={lists}, probes={probes}",
        "build_time_sec": round(build_time, 2),
        **metrics,
    }


def benchmark_brute_force(conn, query_vectors, ground_truths):
    metrics = measure_query_latency_and_recall(conn, query_vectors, ground_truths, force_index_scan=False)
    return {
        "index_type": "Brute-force",
        "params": "-",
        "build_time_sec": 0,
        **metrics,
    }


def print_results_table(results):
    print("\n" + "=" * 115)
    header = f"{'인덱스':<12} {'파라미터':<40} {'빌드시간(s)':<12} {'중앙값 지연(ms)':<16} {'p95 지연(ms)':<14} {'recall'}"
    print(header)
    print("=" * 115)
    for r in results:
        print(
            f"{r['index_type']:<12} {r['params']:<40} {r['build_time_sec']:<12} "
            f"{r['median_latency_ms']:<16} {r['p95_latency_ms']:<14} {r['avg_recall']}"
        )
    print("=" * 115)


def save_results_csv(results, path="data/benchmark_results.csv"):
    full_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path
    )
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index_type", "params", "build_time_sec", "median_latency_ms", "p95_latency_ms", "avg_recall"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과 저장됨: {full_path}")


def run_benchmark():
    with engine.connect() as conn:
        print(f"쿼리 샘플 {N_QUERIES}개 추출 중...")
        query_vectors = get_sample_query_vectors(conn)

        print("Ground truth 계산 중 (brute-force)...")
        ground_truths = get_ground_truth(conn, query_vectors)

        results = []

        # Brute-force 베이스라인 먼저 측정 (인덱스 없는 상태, Seq Scan 허용)
        print("Brute-force(인덱스 없음) 베이스라인 측정 중...")
        result = benchmark_brute_force(conn, query_vectors, ground_truths)
        results.append(result)
        print(f"  → {result}")

        # HNSW 파라미터 조합
        hnsw_configs = [
            {"m": 8, "ef_construction": 32, "ef_search": 20},
            {"m": 8, "ef_construction": 32, "ef_search": 100},
            {"m": 16, "ef_construction": 64, "ef_search": 20},
            {"m": 16, "ef_construction": 64, "ef_search": 100},
            {"m": 32, "ef_construction": 128, "ef_search": 100},
        ]

        for cfg in hnsw_configs:
            print(f"HNSW 벤치마크: {cfg}")
            result = benchmark_hnsw(conn, query_vectors, ground_truths, **cfg)
            results.append(result)
            print(f"  → {result}")

        # IVFFlat 파라미터 조합
        ivfflat_configs = [
            {"lists": 100, "probes": 1},
            {"lists": 100, "probes": 10},
            {"lists": 200, "probes": 10},
            {"lists": 200, "probes": 20},
        ]

        for cfg in ivfflat_configs:
            print(f"IVFFlat 벤치마크: {cfg}")
            result = benchmark_ivfflat(conn, query_vectors, ground_truths, **cfg)
            results.append(result)
            print(f"  → {result}")

        # 마지막에 세션 설정 원복
        conn.execute(text("SET enable_seqscan = on"))

    print_results_table(results)
    return results


if __name__ == "__main__":
    results = run_benchmark()
    save_results_csv(results)