## RFP PDF 프로토타입(Gradio + 페이지 RAG + FAISS)

컨설턴트가 RFP PDF에서 필요한 정보를 빠르게 찾고, 답변/요약의 **근거(페이지+원문 스니펫)** 를 바로 확인할 수 있도록 만든 로컬 Gradio 프로토타입입니다. (문서 선택 → 요약 탭 → Q&A 챗 탭)

</br>

### 1. 프로젝트 구성

- `app.py`: Gradio UI (문서 선택, 요약 탭, Q&A 탭)
- `rag_core.py`: 페이지 추출(C2) + 임베딩/FAISS(R2) + 요약/답변 생성 로직
- `data/pdfs/`: PDF 보관 폴더 (Git에 업로드하지 않음)
  - `.gitignore`에 `data/pdfs/*.pdf` 처리되어 있으므로, PDF는 **로컬에서만** 관리됩니다.

</br>

### 2. 사전 준비(로컬)

#### 1) Python 버전
- Python 3.11 권장/고정 (`pyproject.toml`의 `requires-python = "==3.11.*"`)

</br>

#### 2) 의존성 설치
아래 중 하나로 설치하세요.

- (권장) 가상환경 생성 후 설치
```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```
</br>

#### 3) API Key 설정 (.env)
프로젝트 루트에 `.env` 파일을 만들고 아래처럼 넣습니다.
python-dotenv의 load_dotenv()를 사용하면 .env를 읽어 환경변수로 로드할 수 있습니다.

```
OPENAI_API_KEY=sk-...
```
</br>

#### 4) PDF 넣기 (필수)
아래 경로에 PDF를 넣어야 드롭다운에 표시됩니다.

- data/pdfs/ 폴더에 *.pdf 파일 복사
    - 예: data/pdfs/doc1.pdf, data/pdfs/doc2.pdf ...

</br>

### 3. 실행 방법
```bash
python app.py
```
- 실행 후 터미널에 뜨는 로컬 URL(예: http://127.0.0.1:7860)로 접속합니다.
- UI는 탭 2개만 사용합니다: 요약, Q&A.

</br>

### 4. 데모 시나리오(3줄)
- 문서 선택: 드롭다운에서 PDF 1개 선택

- 요약 탭: 요약 생성 클릭 → 요약문 + Evidence(최대 5개) 확인

- Q&A 탭: 질문 2개 입력 → 답변 + Evidence(각 최대 3개) 확인 (Enter로 전송)

</br>

### 5. 출력 규칙(중요)
- 요약: summary_text + evidence 최대 5개 ({page:int, snippet:str})

- Q&A: answer + evidence 최대 3개 ({page:int, snippet:str})

- snippet은 해당 페이지 텍스트에서 그대로 발췌(추측/재작성 금지)

- 근거가 없으면 "NOT_FOUND"

- LLM 호출/파싱 실패는 내부 "GEN_FAIL"로 기록하고, UI에는 “재시도 안내”만 표시

</br>

### 6. 문제 해결(Troubleshooting)
- 드롭다운에 문서가 안 보임
    - `data/pdfs/` 경로에 *.pdf가 있는지 확인
    - 파일 확장자가 .pdf인지 확인

- OpenAI 에러: api_key 설정 관련
    - `.env`에 OPENAI_API_KEY가 있는지 확인

- 요약/답변이 “재시도 안내”만 나옴
    - 일시적인 생성 실패(GEN_FAIL)일 수 있으니 다시 시도
    - 네트워크/키/모델 접근 권한을 확인