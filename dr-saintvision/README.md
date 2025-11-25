# DR-Saintvision

## Multi-Agent AI Debate System for Enhanced Reasoning

DR-Saintvision은 3개의 AI 모델이 협력하여 질문에 대한 종합적인 분석을 제공하는 다중 에이전트 토론 시스템입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DR-SAINTVISION                            │
│              Multi-Agent AI Debate System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
│   │  Search Agent │   │Reasoning Agent│   │Synthesis Agent│     │
│   │   (Mistral)   │   │    (Llama)    │   │    (Qwen)     │     │
│   │               │   │               │   │               │     │
│   │  웹 검색 &    │   │  심층 추론 &  │   │  종합 분석 &  │     │
│   │  RAG 분석     │   │  논리적 분석  │   │  최종 판단    │     │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘     │
│           │                   │                   │              │
│           └───────────────────┼───────────────────┘              │
│                               │                                  │
│                               ▼                                  │
│                    ┌─────────────────────┐                       │
│                    │    Final Answer     │                       │
│                    │  (Synthesized)      │                       │
│                    └─────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 주요 특징

- **다각도 분석**: 웹 검색, 논리 추론, 종합 분석의 조합
- **교차 검증**: 여러 모델의 결과를 비교하여 신뢰도 향상
- **투명성**: 각 단계의 추론 과정 확인 가능
- **병렬 처리**: 검색과 추론을 동시에 수행하여 시간 단축

## 시스템 요구사항

### 최소 사양
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 16GB
- **Python**: 3.10+

### 권장 사양
- **GPU**: NVIDIA RTX 4070 (12GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 50GB (모델 저장용)

## 설치 방법

### 1. 프로젝트 클론
```bash
git clone https://github.com/your-repo/dr-saintvision.git
cd dr-saintvision
```

### 2. 환경 설정

#### Windows
```batch
setup.bat
```

#### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

### 3. 모델 다운로드

#### 옵션 A: Ollama 사용 (권장)
```bash
# Ollama 설치: https://ollama.ai
ollama pull mistral:7b-instruct-v0.2-q4_0
ollama pull llama3.2:latest
ollama pull qwen2.5:7b-instruct-q4_0
```

#### 옵션 B: Hugging Face 사용
```bash
python download_models.py
```

### 4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 설정 조정
```

## 사용 방법

### 전체 시스템 시작
```bash
python main.py
# 또는
./run.sh  # Linux/Mac
run.bat   # Windows
```

### API 서버만 실행
```bash
python main.py --api-only
```

### Gradio UI만 실행
```bash
python main.py --gradio-only
```

### CLI 모드
```bash
python main.py --cli
```

### 단일 질문 처리
```bash
python main.py --query "인공지능의 미래는?"
```

## 접속 URL

- **Gradio UI**: http://localhost:7860
- **API 문서**: http://localhost:8000/docs
- **API ReDoc**: http://localhost:8000/redoc

## 프로젝트 구조

```
dr-saintvision/
├── models/                    # AI 에이전트
│   ├── base_agent.py         # 기본 에이전트 클래스
│   ├── search_agent.py       # Mistral 웹검색 에이전트
│   ├── reasoning_agent.py    # Llama 심층추론 에이전트
│   ├── synthesis_agent.py    # Qwen 종합분석 에이전트
│   └── debate_manager.py     # 토론 관리자
├── frontend/
│   └── app.py                # Gradio 인터페이스
├── backend/
│   ├── api.py                # FastAPI 서버
│   ├── database.py           # SQLite 데이터베이스
│   └── evaluation.py         # 평가 시스템
├── utils/
│   ├── web_search.py         # 웹 검색 유틸
│   ├── metrics.py            # 평가 메트릭
│   └── prompts.py            # 프롬프트 템플릿
├── tests/                    # 테스트 파일
├── main.py                   # 메인 실행 파일
├── config.py                 # 설정 관리
├── requirements.txt          # 패키지 의존성
├── setup.bat                 # Windows 설치
├── setup.sh                  # Linux/Mac 설치
├── run.bat                   # Windows 실행
└── run.sh                    # Linux/Mac 실행
```

## API 엔드포인트

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API 정보 |
| GET | `/health` | 헬스 체크 |
| POST | `/analyze` | 질문 분석 (전체 토론) |
| POST | `/quick` | 빠른 분석 |
| GET | `/debate/{query_id}` | 토론 결과 상세 |
| GET | `/history/{user_id}` | 사용자 히스토리 |
| GET | `/stats` | 시스템 통계 |
| GET | `/search` | 토론 검색 |

## 모델 정보

| 에이전트 | 모델 | 역할 | 메모리 (4-bit) |
|---------|------|------|---------------|
| Search | Mistral-7B-Instruct | 웹 검색 & RAG | ~4GB |
| Reasoning | Llama-3.2-7B | 심층 추론 | ~4GB |
| Synthesis | Qwen2.5-7B | 종합 분석 | ~4GB |

## 환경 변수

| 변수 | 설명 | 기본값 |
|-----|------|-------|
| `USE_OLLAMA` | Ollama 사용 여부 | `true` |
| `API_PORT` | API 서버 포트 | `8000` |
| `GRADIO_PORT` | Gradio 포트 | `7860` |
| `DEBUG` | 디버그 모드 | `false` |
| `LOG_LEVEL` | 로그 레벨 | `INFO` |

## 테스트 실행

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=. --cov-report=html

# 특정 테스트
pytest tests/test_agents.py -v
```

## 장단점

### 장점
1. **다각도 분석**: 여러 관점에서 문제를 분석
2. **교차 검증**: 여러 모델의 결과 비교로 신뢰도 향상
3. **투명성**: 각 단계의 추론 과정 확인 가능
4. **확장성**: 모델 추가/변경 용이

### 단점
1. **속도**: 단일 모델 대비 처리 시간 증가
2. **자원 소비**: 더 많은 GPU/RAM 필요
3. **복잡성**: 시스템 관리의 복잡도 증가

## 트러블슈팅

### CUDA 메모리 부족
```bash
# .env 파일에서 양자화 설정 확인
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
```

### Ollama 연결 오류
```bash
# Ollama 서비스 확인
ollama serve
# 모델 설치 확인
ollama list
```

### 모델 로딩 실패
```bash
# Hugging Face 로그인
huggingface-cli login

# 모델 캐시 정리
rm -rf ~/.cache/huggingface/
```

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 연락처

- Issues: GitHub Issues
- Email: support@dr-saintvision.ai
