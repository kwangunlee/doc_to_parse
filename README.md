# 문서 파싱 정합성 검증 (Streamlit)

원본 문서(PDF/텍스트)와 파서가 만든 결과(HTML)를 **파일명으로 자동 페어링**하고,
**OpenAI API(LLM)** 로 정합성을 판정해 정량 점수와 오류 목록을 뽑아주는 앱입니다.

- 여러 파일을 한 번에 업로드 → 확장자로 자동 분류
  - `.pdf` `.txt` `.md` → **원본**
  - `.html` `.htm` → **파싱 결과**
- 이름(확장자 제외)이 같은 원본↔파싱을 자동으로 묶음 (직접 지정도 가능)
- LLM 판정: 종합 점수 / 문자·내용 / 표 구조 점수 + 오인식·누락·추가·구조 오류 목록
- 보조 지표(API 불필요): difflib 문자 유사도, HTML 표 구조 점검

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

API 키는 사이드바에 직접 입력하거나, `.streamlit/secrets.toml` 에 저장합니다:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

## GitHub → Streamlit Community Cloud 배포

1. **GitHub 저장소 만들기**
   ```bash
   git init
   git add .
   git commit -m "파싱 정합성 검증 앱"
   git branch -M main
   git remote add origin https://github.com/<계정>/<저장소>.git
   git push -u origin main
   ```
   > `.gitignore` 가 `secrets.toml` 을 제외하므로 API 키는 올라가지 않습니다.

2. **Streamlit 배포**
   - https://share.streamlit.io 접속 → GitHub 로그인
   - **New app** → 저장소 / 브랜치(`main`) / **Main file path** `app.py` 선택 → Deploy

3. **API 키 등록**
   - 배포된 앱의 **⋮ → Settings → Secrets** 에 아래 입력 후 저장:
     ```toml
     OPENAI_API_KEY = "sk-..."
     ```
   - 저장하면 앱이 자동 재시작되고 사이드바 키가 자동 입력됩니다.

## 파일 구성

```
app.py                        # 메인 앱
requirements.txt              # 의존성
.streamlit/config.toml        # 테마/업로드 설정
.streamlit/secrets.toml.example  # 시크릿 템플릿
.gitignore
```

## 참고

- `.hwp` 원본은 Streamlit Cloud에서 안정적으로 파싱하기 어렵습니다. **PDF로 변환 후** 올리세요.
- 비전 판정은 PDF 페이지를 이미지로 렌더링해 표 레이아웃까지 비교합니다(더 정확, 비용↑).
  비용을 줄이려면 사이드바에서 비전 판정을 끄고 텍스트 비교만 사용하세요.
