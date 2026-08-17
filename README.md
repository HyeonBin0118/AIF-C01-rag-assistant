# AIF-C01-rag-assistant

AWS AIF-C01 자격증 스터디 중 정리한 노트(문제 오류 검증·정정 내용 포함)를 기반으로 한 RAG 시스템입니다.

이전에 만든 RAG 프로젝트들(ShopAI, ai-personal-assistant, ai-career-assistant)은 pgvector와 HNSW 인덱스를 "가져다 썼다"면, 이번엔 그 안에서 실제로 무슨 일이 일어나는지 — 인덱스 알고리즘의 동작 원리, 파라미터가 성능에 미치는 영향, 청킹 전략의 효과, 검색 방식 간의 상호보완, 쿼리 자체를 최적화하는 기법들 — 를 직접 실험하고 숫자로 증명하는 데 목적을 뒀습니다.

## 왜 이 프로젝트를 시작했나

RAG(Retrieval-Augmented Generation)는 LLM이 답변할 때 학습하지 못한 지식(비공개 문서, 최신 정보 등)을 검색해서 참고하게 만드는 방식입니다. 파인튜닝 없이도 지식을 추가할 수 있고, 근거 문서를 기반으로 답하게 해서 환각(모델이 사실이 아닌 내용을 그럴듯하게 지어내는 현상)을 줄이는 효과도 있습니다.

기존 프로젝트에서 pgvector + HNSW 조합을 여러 번 썼지만, "왜 이 인덱스를 골랐는가", "파라미터를 이렇게 설정한 이유가 뭔가"를 숫자로 답할 수 있는 수준은 아니었습니다. 이 프로젝트는 그 공백을 메우기 위해 시작했습니다.

## 아키텍처

```
documents (PART 단위, 청킹 전략별로 구분)
  └── chunks (청킹 전략에 따라 분리 방식이 다름)
        └── embeddings (청크당 벡터 1개, 1024차원)
```

세 테이블로 정규화한 이유는 임베딩 모델을 바꾸거나 청킹 전략을 추가할 때 기존 구조를 건드리지 않고 확장할 수 있게 하기 위함입니다. `documents.chunking_strategy` 컬럼으로 구조 기반(structural) / 고정 크기(fixed) / 의미 기반(semantic) 세 가지 청킹 결과를 같은 스키마 안에 공존시켜, 서로 다른 원본 데이터 없이도 전략 간 비교 실험이 가능하도록 설계했습니다.

## 기술 스택

- **DB**: PostgreSQL 16 + pgvector (Docker)
- **임베딩 모델**: BAAI/bge-m3 (1024차원, GPU 추론)
- **ORM**: SQLAlchemy
- **키워드 검색**: rank_bm25 + kiwipiepy (한국어 형태소 분석)
- **Reranking**: BAAI/bge-reranker-v2-m3 (Cross-encoder)
- **쿼리 최적화**: OpenAI API (gpt-4o-mini) — Query Rewriting, HyDE
- **벤치마크**: 위키피디아 한국어 코퍼스 5만 건 (별도 테이블, 실 서비스 데이터와 분리)

## 핵심 의사결정

### 1. 데이터 소스: 단일 통합 문서 (PART 1~25)

벨로그에 정리한 [AWS AIF-C01 용어 정리](https://velog.io/@hyeonbin0118) 글 하나를 소스로 사용했습니다. 실제 문제를 풀며 발견한 오류를 검증하고 정정한 내용까지 포함된 최종본이라, 별도 오답노트 없이 이 문서 하나로 충분하다고 판단했습니다.

### 2. 임베딩 모델: BAAI/bge-m3

데이터가 "Amazon Comprehend", "SageMaker" 같은 영어 서비스명과 한글 설명이 섞인 형태라, 한국어 전용 모델보다 다국어를 함께 잘 다루는 모델이 유리하다고 판단했습니다. 무료·오프라인 모델 중 비교적 최신(2024)이라 추후 하이브리드 검색(dense + sparse) 확장 여지도 고려했습니다.

### 3. 인덱스 벤치마크 방법론: 합성 데이터 대신 공개 코퍼스

인덱스(HNSW/IVFFlat)는 데이터가 대량일 때 효과가 드러나는데, 실제 서비스 데이터는 172개뿐이라 벤치마크가 무의미했습니다. 처음엔 무작위로 생성한 텍스트를 대량으로 채워 넣는 방식을 고려했으나, "가짜 데이터로 결과를 부풀렸다"는 인상을 줄 수 있어 기각했습니다. 대신 위키피디아 한국어 코퍼스(공개 데이터셋)로 대체해, `ann-benchmarks` 같은 업계 표준 벤치마크 방법론을 참고했습니다.

### 4. 쿼리 최적화 모델: 로컬 LLM 대신 OpenAI API

Query Rewriting과 HyDE는 텍스트 "생성"이 필요한 단계입니다. 로컬 GPU(RTX 3060)로 소형 LLM을 직접 돌리는 방법도 고려했으나, 평가 질문 20개 규모의 실험에서는 API 비용이 미미한 반면 생성 품질과 개발 속도 면에서 이점이 커 OpenAI API(gpt-4o-mini)를 선택했습니다.

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

## 실험 2: 청킹 전략 비교

### 방법

원본 문서(PART 1~25)를 세 가지 방식으로 각각 청킹해 별도로 저장했습니다.

- **구조 기반(Structural)**: 문서에 원래 있던 `##`/`###` 헤더를 그대로 따라 분리 (172개 청크, 실험 1에서 사용한 방식)
- **Fixed-size**: 헤더 무시하고 400자 단위로 분리, 앞뒤 50자 overlap (186개 청크)
- **Semantic**: 줄 단위 임베딩 후 인접 줄 간 코사인 거리가 급증하는 지점(상위 10% 퍼센타일)을 경계로 분리 (158개 청크)

평가는 질문 20개(`data/eval_questions.json`)를 직접 구성해, 각 전략별로 검색된 상위 청크에 정답 키워드가 포함되는지(Recall@3, Recall@5)로 측정했습니다.

### 결과

| 전략 | Recall@3 | Recall@5 |
|---|---|---|
| 구조 기반 | 100% (20/20) | 100% (20/20) |
| Fixed-size | 95% (19/20) | 100% (20/20) |
| Semantic | 90% (18/20) | 100% (20/20) |

![청킹 전략 비교](data/chunking_strategy_comparison.png)

### 결론

top-5까지 넉넉히 보면 세 전략 모두 정답을 놓치지 않지만, top-3처럼 검색 범위를 좁힌 조건에서는 **구조 기반 청킹이 가장 안정적**이었습니다. 원본 데이터가 "서비스 하나 = 설명 한 덩어리"로 이미 잘 구조화되어 있어, 그 경계를 그대로 따르는 방식이 질문-정답 매칭에 가장 유리했던 것으로 해석됩니다. Fixed-size와 semantic은 구조를 무시하고 자르는 과정에서 정답 내용이 청크 경계에서 분리되는 경우가 발생해 근소하게 낮은 결과를 보였습니다.

**한계**: 평가 질문이 20개로 표본이 작아, 95%/90%의 차이는 각각 1문제 차이에 불과합니다. 통계적으로 유의미한 차이라 단정하기보다는 "제한된 표본에서 관찰된 경향"으로 해석하는 것이 정확합니다. 다만 이 결과는 "원본 데이터가 이미 구조화되어 있다면, 그 구조를 활용하는 청킹이 일반적인 fixed-size 방식보다 우위를 가질 수 있다"는 방향성은 뒷받침합니다.

## 실험 3: 하이브리드 검색 (BM25 + 벡터, RRF)

### 방법

벡터 검색(의미 기반)만으로는 정확한 숫자·파라미터명·축약어 같은 "문자열 일치가 중요한" 질문에 약할 수 있다는 가설을 세우고, 통계 기반 키워드 검색(BM25)과 결합한 하이브리드 검색을 직접 구현했습니다.

- **BM25**: `rank_bm25` 라이브러리 사용, 한국어 형태소 분석은 `kiwipiepy`로 처리(명사/동사/영단어/숫자 토큰만 추출)
- **RRF(Reciprocal Rank Fusion)**: 벡터 검색과 BM25 검색 각각의 순위(점수 아님)를 `1 / (k + rank)` 공식으로 합산해 최종 순위 산출 (k=60)
- 구조 기반(structural) 청킹 172개를 대상으로 진행

평가 질문셋을 두 그룹으로 나눠 구성했습니다: 정확한 문자열이 중요한 질문 10개(예: "실시간 추론의 최대 페이로드 크기는?") + 의미 기반 질문 10개(예: "PII 탐지하는 서비스는?"). 처음엔 Phase 2에서 쓴 의미 기반 질문만으로 평가했으나 세 방식 모두 100%로 차이가 드러나지 않아, 벡터 검색이 약할 만한 유형을 의도적으로 추가해 재구성했습니다.

### 결과

| 방식 | Recall@3 | Recall@5 |
|---|---|---|
| 벡터 단독 | 85% (17/20) | 90% (18/20) |
| BM25 단독 | 85% (17/20) | 90% (18/20) |
| 하이브리드(RRF) | 90% (18/20) | 90% (18/20) |

**개선 사례** (top-3 기준, 한쪽만으로는 실패했을 질문을 하이브리드가 보완):

- *"XGBoost의 X는 무엇의 약자인가?"* — 벡터 검색은 적중했으나 BM25는 실패. 하이브리드는 벡터 결과를 살려 적중.
- *"BERT가 학습에 사용하는 방식의 약자는?"* — 반대로 BM25는 적중했으나 벡터 검색은 실패. 하이브리드는 BM25 결과를 살려 적중.

### 결론

top-3 기준으로 하이브리드가 단독 방식 대비 5%p 개선(85%→90%)을 보였고, 위 두 사례처럼 "한쪽이 놓친 것을 다른 쪽이 보완하는" 상호작용이 실제로 관찰됐습니다. 다만 top-5까지 넓히면 세 방식 모두 90%로 수렴해, 후보군을 넉넉히 볼 경우 개별 방식만으로도 충분할 수 있다는 점도 함께 확인했습니다. 즉 하이브리드 검색은 **검색 범위가 좁을수록(정확도가 중요한 상황일수록)** 효과가 커지는 경향을 보였습니다.

**한계**: 평가 질문 20개, 청크 172개라는 소규모 실험이라 이 경향이 대규모 데이터에서도 유지되는지는 별도 검증이 필요합니다.

## 실험 4: Reranking (Cross-encoder)

### 방법

하이브리드 검색(Bi-encoder 방식)으로 뽑은 1차 후보(top-15)를, `BAAI/bge-reranker-v2-m3`(Cross-encoder)로 질문과 문서를 함께 넣어 재채점해 순위를 재배열했습니다. Bi-encoder는 질문과 문서를 각각 독립적으로 벡터화해 빠르지만 정밀도에 한계가 있고, Cross-encoder는 둘을 함께 넣어 느리지만 훨씬 정밀하게 관련성을 판단합니다. 이 두 단계를 결합하는 것이 reranking의 표준적인 접근입니다.

### 지표 전환: Recall@k → MRR

처음엔 Phase 2·3과 동일하게 Recall@k로 평가했으나, 하이브리드 검색만으로 이미 대부분 정답이 top-3 안에 들어와 있어 reranking 전후로 Recall 값에 차이가 없었습니다. Recall@k는 "정답이 순위 안에 있는지"만 보는 이진 지표라, "5위였던 정답이 1위로 올라갔다"는 개선을 반영하지 못하는 한계가 있었습니다. 이에 정답이 정확히 몇 위에 나왔는지를 점수화하는 **MRR(Mean Reciprocal Rank, `1/정답순위`의 평균)**로 지표를 전환했습니다.

### 결과

| 방식 | MRR |
|---|---|
| 하이브리드 단독 | 0.8500 |
| 하이브리드 + Reranking | 0.9000 |

**개선 사례** (하이브리드 순위 → Reranking 후 순위):

- *"BERT가 학습에 사용하는 방식의 약자는?"* — 2위 → 1위
- *"Bedrock Guardrails의 필터 정책은 몇 가지인가?"* — 2위 → 1위

### 결론

Reranking은 "정답을 top-k 안에 들어오게 하는" 효과보다는, **이미 후보군 안에 있는 정답의 순위를 더 정확하게 끌어올리는** 효과가 뚜렷했습니다. 이는 실제 RAG 서비스에서 LLM에 넘기는 컨텍스트 개수가 제한적일 때(예: top-3만 사용) 특히 유효한데, 애매하게 2~3위에 있던 정답이 확실한 1위로 올라오면 LLM이 더 명확한 근거를 우선적으로 참고할 수 있기 때문입니다.

**한계**: 마찬가지로 질문 20개 규모의 소규모 평가이며, Cross-encoder는 후보 하나하나를 개별 채점하는 방식이라 후보 수·질문 수가 늘어날수록 지연시간 비용이 커집니다. 실서비스 적용 시 이 비용과 순위 개선 효과 사이의 트레이드오프를 고려해야 합니다.

## 실험 5: 쿼리 최적화 (Query Rewriting, HyDE)

### 방법

- **Query Rewriting (Multi-query)**: OpenAI(`gpt-4o-mini`)로 원본 질문을 표현이 다른 질문 2개로 확장한 뒤, 원본을 포함한 3개 질문 각각으로 검색해 RRF로 통합
- **HyDE (Hypothetical Document Embeddings)**: 질문을 그대로 임베딩하는 대신, LLM이 먼저 "그럴듯한 가상의 답변"을 생성하고 그 답변을 임베딩해 벡터 검색에 사용. "질문"과 "답변"의 문장 형태 차이로 인한 임베딩 공간 불일치를 완화하려는 의도

평가는 `eval_questions_hybrid.json`(20문항)을 사용했고, 실험 4에서 확인한 대로 Recall@k는 순위 개선을 못 잡아내므로 처음부터 MRR로 측정했습니다.

### 결과

| 방식 | MRR |
|---|---|
| 하이브리드 단독 | 0.8500 |
| Query Rewriting | 0.8750 |
| HyDE | 0.9000 |

### 결론

두 기법 모두 하이브리드 단독보다 개선됐고, 그중 HyDE가 가장 효과적이었습니다(Reranking과 동일한 0.9000 달성). "질문을 그대로 검색하기보다, 질문에 대한 가상의 답변 형태로 변환해서 검색하는 것"이 이 데이터셋에서는 질문을 여러 버전으로 확장하는 것보다 더 유효했다고 해석됩니다. 다만 두 기법 모두 LLM 호출 비용과 지연시간이 추가되므로(질문당 API 호출 1~3회), 실서비스에서는 Reranking처럼 검색 후 단계에 비용을 쓸지, HyDE처럼 검색 전 단계에 비용을 쓸지 트레이드오프를 고려해야 합니다.

**한계**: 동일하게 질문 20개 규모의 소규모 평가입니다.

## 실행 방법

```bash
# 1. 환경 설정
conda create -n aif-rag python=3.11 -y
conda activate aif-rag
pip install -r requirements.txt

# 2. .env 파일 생성 (DB 접속 정보 + OpenAI API 키)
# DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/aif_rag
# OPENAI_API_KEY=your_key_here

# 3. DB 실행
docker compose up -d

# 4. DB 초기화 및 데이터 적재 (세 가지 청킹 전략 모두)
python scripts/init_db.py
python scripts/ingest.py             # 구조 기반
python scripts/ingest_fixed.py       # Fixed-size
python scripts/ingest_semantic.py    # Semantic
python scripts/embed.py

# 5. 검색 테스트
python scripts/search.py "RAG용 임베딩 모델은 뭘 써야해"
python scripts/hybrid_search.py "SageMaker Ground Truth 라벨링 방법"
python scripts/rerank_search.py "SageMaker Ground Truth 라벨링 방법"
python scripts/query_rewrite_search.py "SageMaker Ground Truth 라벨링 방법"
python scripts/hyde_search.py "SageMaker Ground Truth 라벨링 방법"

# 6. (선택) 인덱스 벤치마크 재현
python scripts/download_benchmark_data.py
python scripts/embed_benchmark.py
python scripts/benchmark_index.py
python scripts/plot_benchmark.py

# 7. (선택) 청킹 전략 비교 재현
python scripts/evaluate_chunking.py
python scripts/plot_chunking_eval.py

# 8. (선택) 하이브리드 검색 평가 재현
python scripts/evaluate_hybrid.py
python scripts/analyze_hybrid_gains.py

# 9. (선택) Reranking 평가 재현
python scripts/evaluate_rerank.py

# 10. (선택) 쿼리 최적화 평가 재현
python scripts/evaluate_query_optimization.py
```

## 프로젝트 구조

```
AIF-C01-rag-assistant/
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── database.py
│   ├── models.py                  # Document / Chunk / Embedding / BenchmarkVector
│   └── llm_client.py               # OpenAI API 클라이언트
├── data/
│   ├── raw/                        # 원본 마크다운, 벤치마크 코퍼스 (git 미포함)
│   ├── eval_questions.json         # 청킹 전략 평가용 질문-정답 쌍
│   ├── eval_questions_hybrid.json  # 하이브리드/reranking/쿼리최적화 평가용 질문-정답 쌍
│   ├── benchmark_results.csv
│   ├── benchmark_recall_vs_latency.png
│   ├── chunking_evaluation.csv
│   ├── chunking_strategy_comparison.png
│   ├── hybrid_evaluation.csv
│   ├── rerank_evaluation.csv
│   ├── rerank_evaluation_detail.csv
│   ├── query_optimization_evaluation.csv
│   └── query_optimization_detail.csv
└── scripts/
    ├── init_db.py
    ├── ingest.py                   # 구조 기반 청킹 (PART → 섹션 → 서비스)
    ├── ingest_fixed.py             # Fixed-size 청킹
    ├── ingest_semantic.py          # Semantic 청킹
    ├── embed.py
    ├── check_data.py
    ├── search.py
    ├── download_benchmark_data.py
    ├── embed_benchmark.py
    ├── benchmark_index.py
    ├── plot_benchmark.py
    ├── evaluate_chunking.py
    ├── plot_chunking_eval.py
    ├── hybrid_search.py            # BM25 + 벡터 + RRF 하이브리드 검색
    ├── evaluate_hybrid.py
    ├── analyze_hybrid_gains.py
    ├── rerank_search.py            # Cross-encoder 기반 reranking
    ├── evaluate_rerank.py          # MRR 기반 정량 평가
    ├── query_rewrite_search.py     # Multi-query 쿼리 재작성
    ├── hyde_search.py              # HyDE
    └── evaluate_query_optimization.py
```

## 진행 상황

- [x] Phase 1: DB 스키마 설계 및 인덱스 파라미터 벤치마크
- [x] Phase 2: 청킹 전략 비교 (구조 기반 vs fixed-size vs semantic)
- [x] Phase 3: 하이브리드 검색 (BM25 + 벡터, RRF 직접 구현)
- [x] Phase 4: Reranking (Cross-encoder)
- [x] Phase 5: 쿼리 최적화 (Query rewriting, HyDE)
- [ ] Phase 6: RAGAs 기반 정량 평가