# AIF-C01-rag-assistant

AWS AIF-C01 자격증 스터디 중 정리한 노트(문제 오류 검증·정정 내용 포함)를 기반으로 한 RAG 시스템입니다.

이전에 만든 RAG 프로젝트들(ShopAI, ai-personal-assistant, ai-career-assistant)은 pgvector와 HNSW 인덱스를 "가져다 썼다"면, 이번엔 그 안에서 실제로 무슨 일이 일어나는지 — 인덱스 알고리즘의 동작 원리, 파라미터가 성능에 미치는 영향, 청킹 전략의 효과 — 를 직접 실험하고 숫자로 증명하는 데 목적을 뒀습니다.

## 왜 이 프로젝트를 시작했나

RAG(Retrieval-Augmented Generation)는 LLM이 답변할 때 학습하지 못한 지식(비공개 문서, 최신 정보 등)을 검색해서 참고하게 만드는 방식입니다. 파인튜닝 없이도 지식을 추가할 수 있고, 근거 문서를 기반으로 답하게 해서 환각(모델이 사실이 아닌 내용을 그럴듯하게 지어내는 현상)을 줄이는 효과도 있습니다.

기존 프로젝트에서 pgvector + HNSW 조합을 여러 번 썼지만, "왜 이 인덱스를 골랐는가", "파라미터를 이렇게 설정한 이유가 뭔가"를 숫자로 답할 수 있는 수준은 아니었습니다. 이 프로젝트는 그 공백을 메우기 위해 시작했습니다.

## 아키텍처

```
documents (PART 단위, 25개)
  └── chunks (섹션/서비스 단위, 172개)
        └── embeddings (청크당 벡터 1개, 1024차원)
```

세 테이블로 정규화한 이유는 임베딩 모델을 바꿔서 재실험할 때 chunks 이하 구조를 건드리지 않고 embeddings만 새로 채울 수 있게 하기 위함입니다. 실제로 임베딩 모델을 변경하며 이 구조 덕을 봤습니다.

## 기술 스택

- **DB**: PostgreSQL 16 + pgvector (Docker)
- **임베딩 모델**: BAAI/bge-m3 (1024차원, GPU 추론)
- **ORM**: SQLAlchemy
- **벤치마크**: 위키피디아 한국어 코퍼스 5만 건 (별도 테이블, 실 서비스 데이터와 분리)

## 핵심 의사결정

### 1. 데이터 소스: 단일 통합 문서 (PART 1~25)

벨로그에 정리한 [AWS AIF-C01 용어 정리](https://velog.io/@hyeonbin0118) 글 하나를 소스로 사용했습니다. 실제 문제를 풀며 발견한 오류를 검증하고 정정한 내용까지 포함된 최종본이라, 별도 오답노트 없이 이 문서 하나로 충분하다고 판단했습니다.

### 2. 임베딩 모델: BAAI/bge-m3

데이터가 "Amazon Comprehend", "SageMaker" 같은 영어 서비스명과 한글 설명이 섞인 형태라, 한국어 전용 모델보다 다국어를 함께 잘 다루는 모델이 유리하다고 판단했습니다. 무료·오프라인 모델 중 비교적 최신(2024)이라 추후 하이브리드 검색(dense + sparse) 확장 여지도 고려했습니다.

### 3. 인덱스 벤치마크 방법론: 합성 데이터 대신 공개 코퍼스

인덱스(HNSW/IVFFlat)는 데이터가 대량일 때 효과가 드러나는데, 실제 서비스 데이터는 172개뿐이라 벤치마크가 무의미했습니다. 처음엔 무작위로 생성한 텍스트를 대량으로 채워 넣는 방식을 고려했으나, "가짜 데이터로 결과를 부풀렸다"는 인상을 줄 수 있어 기각했습니다. 대신 위키피디아 한국어 코퍼스(공개 데이터셋)로 대체해, `ann-benchmarks` 같은 업계 표준 벤치마크 방법론을 참고했습니다.

## 실험 1: 벡터 인덱스 파라미터 벤치마크

### 방법

- 위키피디아 한국어 문단 5만 건을 BGE-M3로 임베딩 (실 서비스 데이터와 분리된 별도 테이블)
- 쿼리 100개에 대해 Brute-force로 정답(top-10)을 먼저 구해 Ground truth로 사용
- HNSW(`m`, `ef_construction`, `ef_search`)와 IVFFlat(`lists`, `probes`)을 각각 여러 파라미터 조합으로 인덱스 생성 후, 동일 쿼리에 대한 지연시간(중앙값·p95)과 recall을 측정
- `EXPLAIN (ANALYZE, FORMAT JSON)`으로 DB 서버 내부 실행시간만 측정해 네트워크 왕복시간 노이즈 제거

### 버그 발견: PostgreSQL 쿼리 플래너의 인덱스 무시 현상

벤치마크 초반, 특정 파라미터 조합에서만 지연시간이 Brute-force 수준(약 140ms)으로 튀는 이상 현상을 발견했습니다. 원인은 PostgreSQL 쿼리 플래너가 비용 추정을 잘못해 인덱스를 만들어두고도 Seq Scan(순차 탐색)으로 폴백하는 것이었습니다. `SET enable_seqscan = off`로 강제해 재측정한 결과 이상치가 사라지고 파라미터에 따른 일관된 트레이드오프가 드러났습니다.

### 결과

| 인덱스 | 파라미터 | 빌드 시간(s) | 중앙값 지연(ms) | Recall@10 |
|---|---|---|---|---|
| Brute-force | - | 0 | 136.45 | 1.0 |
| HNSW | m=8, ef_construction=32, ef_search=20 | 8.51 | 0.79 | 0.927 |
| HNSW | m=8, ef_construction=32, ef_search=100 | 9.05 | 1.81 | 0.978 |
| HNSW | m=16, ef_construction=64, ef_search=20 | 26.50 | 1.10 | 0.986 |
| HNSW | m=16, ef_construction=64, ef_search=100 | 26.35 | 3.12 | 1.0 |
| HNSW | m=32, ef_construction=128, ef_search=100 | 95.79 | 4.59 | 1.0 |
| IVFFlat | lists=100, probes=1 | 2.40 | 0.91 | 0.626 |
| IVFFlat | lists=100, probes=10 | 2.43 | 6.20 | 0.958 |
| IVFFlat | lists=200, probes=10 | 3.06 | 5.55 | 0.936 |
| IVFFlat | lists=200, probes=20 | 3.25 | 7.47 | 0.962 |

![인덱스 벤치마크 결과](data/benchmark_recall_vs_latency.png)

### 결론

- **검색 성능**: 동일 recall 기준으로 HNSW가 IVFFlat보다 일관되게 빠릅니다. 예를 들어 HNSW(m=16, ef_search=20)는 1.10ms에 recall 0.986을 내는데, IVFFlat(lists=200, probes=20)은 6배 느린 7.47ms에도 recall 0.962에 그칩니다.
- **빌드 비용**: 반대로 인덱스 구축 비용은 IVFFlat이 압도적으로 저렴합니다(2~3초 vs HNSW 최대 95.8초).
- **선택 기준**: 데이터가 자주 바뀌지 않는 서비스(본 프로젝트처럼 정적 스터디 자료 기반)라면 HNSW가 유리하고, 데이터가 실시간으로 추가·삭제되어 인덱스 재빌드가 잦은 서비스라면 빌드 비용이 낮은 IVFFlat이 더 실용적일 수 있습니다.

## 실행 방법

```bash
# 1. 환경 설정
conda create -n aif-rag python=3.11 -y
conda activate aif-rag
pip install -r requirements.txt

# 2. DB 실행
docker compose up -d

# 3. DB 초기화 및 데이터 적재
python scripts/init_db.py
python scripts/ingest.py
python scripts/embed.py

# 4. 검색 테스트
python scripts/search.py "RAG용 임베딩 모델은 뭘 써야해"

# 5. (선택) 인덱스 벤치마크 재현
python scripts/download_benchmark_data.py
python scripts/embed_benchmark.py
python scripts/benchmark_index.py
python scripts/plot_benchmark.py
```

## 프로젝트 구조

```
AIF-C01-rag-assistant/
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── database.py
│   └── models.py          # Document / Chunk / Embedding / BenchmarkVector
├── data/
│   ├── raw/                # 원본 마크다운, 벤치마크 코퍼스 (git 미포함)
│   ├── benchmark_results.csv
│   └── benchmark_recall_vs_latency.png
└── scripts/
    ├── init_db.py
    ├── ingest.py            # 계층적 청킹 (PART → 섹션 → 서비스)
    ├── embed.py
    ├── check_data.py
    ├── search.py
    ├── download_benchmark_data.py
    ├── embed_benchmark.py
    ├── benchmark_index.py
    └── plot_benchmark.py
```

## 진행 상황

- [x] Phase 1: DB 스키마 설계 및 인덱스 파라미터 벤치마크
- [ ] Phase 2: 청킹 전략 비교 (구조 기반 vs fixed-size vs semantic)
- [ ] Phase 3: 하이브리드 검색 (BM25 + 벡터, RRF 직접 구현)
- [ ] Phase 4: Reranking (Cross-encoder)
- [ ] Phase 5: 쿼리 최적화 (Query rewriting, HyDE)
- [ ] Phase 6: RAGAs 기반 정량 평가