# -*- coding: utf-8 -*-
"""
문서 파싱 정합성 검증 (Streamlit)
- 파일 여러 개를 한 번에 업로드하면 확장자로 자동 분류:
    .html/.htm            -> 파싱 결과
    .pdf/.txt/.md         -> 원본
- 파일명(확장자 제외)이 같은 원본 <-> 파싱결과를 자동 페어링
- OpenAI API(LLM)로 정합성을 판정하고 정량 점수 + 오류 목록을 반환
- difflib 기반 문자 유사도, 표 구조 점검은 API 없이도 계산되는 보조 지표
"""
import io
import re
import json
import base64
import difflib
import unicodedata

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

# ------------------------------------------------------------------ #
# 기본 설정
# ------------------------------------------------------------------ #
st.set_page_config(page_title="파싱 정합성 검증", page_icon="🔍", layout="wide")

ORIGIN_EXT = {"pdf", "txt", "md"}
PARSED_EXT = {"html", "htm"}


def stem(name: str) -> str:
    base = name.rsplit(".", 1)[0]
    return base.strip().lower().replace(" ", "")


def ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def smart_decode(data: bytes) -> str:
    """한글 문서 인코딩 자동 감지(UTF-8 / CP949 / EUC-KR 등). UTF-8 강제 디코딩으로 깨지는 문제 방지."""
    if data.startswith(b"\xef\xbb\xbf"):        # UTF-8 BOM
        return data[3:].decode("utf-8", "replace")
    # <meta charset=...> 선언 탐지
    head = data[:2048].decode("latin-1", "ignore").lower()
    m = re.search(r'charset\s*=\s*["\']?\s*([a-z0-9_\-]+)', head)
    aliases = {"ms949": "cp949", "euckr": "euc-kr", "ksc5601": "euc-kr",
               "ks_c_5601-1987": "euc-kr", "x-windows-949": "cp949"}
    candidates = []
    if m:
        enc = m.group(1)
        candidates.append(aliases.get(enc, enc))
    candidates += ["utf-8", "cp949", "euc-kr"]     # UTF-8 우선, 실패 시 한글 레거시
    for enc in candidates:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    try:                                            # 마지막: 통계 기반 감지
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best:
            return str(best)
    except Exception:
        pass
    return data.decode("utf-8", "replace")


# ------------------------------------------------------------------ #
# 텍스트 추출
# ------------------------------------------------------------------ #
def pdf_text(data: bytes) -> str:
    if not HAS_FITZ:
        return ""
    out = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return "\n".join(out)


def pdf_page_pngs(data: bytes, max_pages: int = 10, dpi: int = 140):
    """PDF 페이지를 PNG 바이트 목록으로 렌더링"""
    out = []
    if not HAS_FITZ:
        return out
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            out.append(pix.tobytes("png"))
    return out


def pdf_images(data: bytes, max_pages: int = 2, dpi: int = 140):
    """LLM 비전 판정용 페이지 이미지(base64 PNG) 목록"""
    return [base64.b64encode(p).decode() for p in pdf_page_pngs(data, max_pages, dpi)]


def html_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text("\n")


def analyze_tables(html: str):
    soup = BeautifulSoup(html, "lxml")
    result = []
    for idx, t in enumerate(soup.find_all("table"), 1):
        rows = t.find_all("tr")
        widths, total, cspan, rspan = [], 0, 0, 0
        for r in rows:
            w = 0
            for c in r.find_all(["td", "th"], recursive=False):
                cs = int(c.get("colspan", 1) or 1)
                rs = int(c.get("rowspan", 1) or 1)
                if cs > 1:
                    cspan += 1
                if rs > 1:
                    rspan += 1
                w += cs
                total += 1
            widths.append(w)
        mode = max(set(widths), key=widths.count) if widths else 0
        irregular = sum(1 for w in widths if w != mode)
        result.append({
            "표": f"#{idx}", "행": len(rows), "기준열": mode, "총셀": total,
            "colspan": cspan, "rowspan": rspan, "비정상행": irregular,
            "구조": "일관" if irregular == 0 else "불일치",
        })
    return result


# ------------------------------------------------------------------ #
# 보조 지표 (API 불필요)
# ------------------------------------------------------------------ #
def char_similarity(ref: str, hyp: str) -> float:
    a = nfc(ref).replace(" ", "").replace("\n", "")
    b = nfc(hyp).replace(" ", "").replace("\n", "")
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------ #
# OpenAI LLM 판정
# ------------------------------------------------------------------ #
JUDGE_SYSTEM = (
    "너는 한국 정부 정책문서의 파싱 결과를 검수하는 QA 평가자다. "
    "원본 문서와 파서가 생성한 결과를 비교해, 내용/문자/표 구조의 정합성을 엄격하게 평가한다. "
    "OCR 오인식(예: 급속→금속, 밀집도→밀질도), 내용 누락, 없는 내용 추가(환각), "
    "표의 행·열·병합 구조 붕괴를 특히 잘 잡아낸다. 반드시 JSON만 출력한다."
)

JUDGE_SCHEMA = """다음 JSON 스키마로만 답하라(설명 문장 금지):
{
  "overall_score": 0~100 정수(종합 정합성),
  "text_fidelity": 0~100 정수(문자/내용 일치도),
  "table_structure": 0~100 정수(표 구조 정합성, 표 없으면 100),
  "verdict": "좋음" | "보통" | "주의",
  "summary": "한두 문장 요약(한국어)",
  "errors": [
    {
      "type": "오인식" | "누락" | "추가" | "구조",
      "original": "원본의 해당 부분(없으면 빈 문자열)",
      "parsed": "파싱 결과의 해당 부분(없으면 빈 문자열)",
      "severity": "high" | "med" | "low",
      "note": "무엇이 어떻게 다른지(한국어)"
    }
  ]
}
errors는 심각도 높은 순으로 최대 %d개까지."""


def build_user_content(ref_text, parsed_html, images, max_err, use_vision):
    parts = []
    head = JUDGE_SCHEMA % max_err + "\n\n"
    head += "=== 파싱 결과(HTML/텍스트) ===\n" + parsed_html[:12000] + "\n\n"
    if use_vision and images:
        head += "=== 원본 ===\n아래 첨부한 원본 페이지 이미지를 기준(정답)으로 삼아 위 파싱 결과와 대조하라.\n"
        parts.append({"type": "text", "text": head})
        for b64 in images:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })
    else:
        head += "=== 원본(추출 텍스트) ===\n" + (ref_text or "(텍스트 없음)")[:12000]
        parts.append({"type": "text", "text": head})
    return parts


def llm_judge(client, model, ref_text, parsed_html, images, max_err, use_vision, temperature):
    content = build_user_content(ref_text, parsed_html, images, max_err, use_vision)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ],
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    return data


# ------------------------------------------------------------------ #
# 사이드바
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("설정")
    secret_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    user_key = st.text_input("OpenAI API Key", value="", type="password",
                             placeholder="비워두면 서버에 저장된 키 사용",
                             help="여기 입력한 키는 이 세션에서만 쓰이고 저장/노출되지 않습니다.")
    api_key = user_key or secret_key
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    use_vision = st.toggle("비전 판정(원본 이미지 사용)", value=True,
                           help="PDF 페이지를 이미지로 렌더링해 표 레이아웃까지 비교합니다. 끄면 추출 텍스트로만 비교(더 저렴).")
    max_err = st.slider("오류 목록 최대 개수", 3, 20, 8)
    temperature = st.slider("temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption("PDF/텍스트 = 원본 · HTML = 파싱 결과 (파일명으로 자동 페어링)")
    if not HAS_FITZ:
        st.warning("PyMuPDF 미설치: PDF 텍스트·이미지 추출 불가. requirements.txt를 확인하세요.")


# ------------------------------------------------------------------ #
# 메인
# ------------------------------------------------------------------ #
st.title("🔍 문서 파싱 정합성 검증")
st.write("파일을 한꺼번에 올리면 원본(PDF/텍스트)과 파싱 결과(HTML)를 파일명으로 짝지어 OpenAI로 검증합니다.")

files = st.file_uploader(
    "원본과 파싱 결과 파일들을 함께 올리세요",
    type=["pdf", "html", "htm", "txt", "md"],
    accept_multiple_files=True,
)

if not files:
    st.info("예: `D2009263-1_p5.pdf`(원본) + `D2009263-1_p5.html`(파싱) 처럼 이름이 같은 쌍을 같이 올리면 자동으로 묶입니다.")
    st.stop()

# --- 분류 & 페어링 ---
origins, parsed = {}, {}
for f in files:
    e = ext(f.name)
    data = f.getvalue()
    if e in PARSED_EXT:
        parsed[f.name] = data
    elif e in ORIGIN_EXT:
        origins[f.name] = data

origin_by_stem = {}
for name in origins:
    origin_by_stem.setdefault(stem(name), name)

st.subheader("페어링")
pairs = []
origin_names = ["(없음)"] + list(origins.keys())
for pname in parsed:
    guess = origin_by_stem.get(stem(pname), "(없음)")
    col1, col2 = st.columns([1, 1])
    col1.markdown(f"**파싱** · `{pname}`")
    sel = col2.selectbox(f"원본 선택 — {pname}", origin_names,
                         index=origin_names.index(guess) if guess in origin_names else 0,
                         key=f"pair_{pname}", label_visibility="collapsed")
    pairs.append((pname, None if sel == "(없음)" else sel))

unpaired_o = set(origins) - {o for _, o in pairs if o}
if unpaired_o:
    st.caption("짝을 못 찾은 원본: " + ", ".join(f"`{x}`" for x in unpaired_o))

run = st.button("검증 실행", type="primary", use_container_width=True)

# ------------------------------------------------------------------ #
# 실행
# ------------------------------------------------------------------ #
if run:
    if not api_key:
        st.error("OpenAI API Key를 입력하세요(사이드바).")
        st.stop()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        st.stop()

    rows = []
    progress = st.progress(0.0, text="검증 준비 중…")
    total = len(pairs)

    for i, (pname, oname) in enumerate(pairs):
        progress.progress(i / total, text=f"검증 중… {pname}")
        phtml = smart_decode(parsed[pname])
        ref_text, images, preview_pngs = "", [], []
        if oname:
            odata = origins[oname]
            if ext(oname) == "pdf":
                ref_text = pdf_text(odata)
                preview_pngs = pdf_page_pngs(odata, max_pages=10, dpi=130)
                if use_vision:
                    images = pdf_images(odata)
            else:
                ref_text = smart_decode(odata)

        sim = char_similarity(ref_text, html_text(phtml)) if ref_text else None
        tables = analyze_tables(phtml)

        judged, err = None, None
        try:
            judged = llm_judge(client, model, ref_text, phtml, images,
                               max_err, use_vision and bool(images), temperature)
        except Exception as e:
            err = str(e)

        # --- 결과 카드 ---
        title = f"{pname}  ↔  {oname or '(원본 없음)'}"
        with st.expander(title, expanded=(total <= 3)):
            if err:
                st.error(f"LLM 판정 실패: {err}")
            if judged:
                score = judged.get("overall_score", 0)
                verdict = judged.get("verdict", "-")
                color = "🟢" if score >= 90 else ("🟡" if score >= 75 else "🔴")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("종합 점수", f"{score}")
                m2.metric("문자/내용", judged.get("text_fidelity", "-"))
                m3.metric("표 구조", judged.get("table_structure", "-"))
                m4.metric("문자 유사도", f"{sim*100:.1f}%" if sim is not None else "N/A")
                st.markdown(f"{color} **판정: {verdict}** — {judged.get('summary','')}")

                errs = judged.get("errors", [])
                if errs:
                    df = pd.DataFrame(errs)
                    order = [c for c in ["type", "original", "parsed", "note"] if c in df.columns]
                    df = df[order].rename(columns={
                        "type": "유형", "original": "원본", "parsed": "파싱결과", "note": "설명"
                    })
                    st.markdown("""<style>
                    table.qa td,table.qa th{white-space:normal;word-break:break-word;vertical-align:top;padding:6px 8px;font-size:13px;border:1px solid #e4e8ee}
                    table.qa{border-collapse:collapse;width:100%;table-layout:fixed}
                    table.qa td:nth-child(1){width:4%}
                    table.qa td:nth-child(2),table.qa td:nth-child(3){width:40%}
                    table.qa td:nth-child(4){width:16%}
                    </style>""", unsafe_allow_html=True)
                    st.markdown(df.to_html(index=False, classes="qa", escape=True), unsafe_allow_html=True)
                else:
                    st.success("LLM이 보고한 오류 없음.")

            if tables:
                st.caption("표 구조 점검(파싱 결과 기준)")
                st.dataframe(pd.DataFrame(tables), use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.caption("원본 (PDF 그대로 보기)")
                if preview_pngs:
                    imgs = "".join(
                        f'<img src="data:image/png;base64,{base64.b64encode(p).decode()}" '
                        f'style="width:100%;display:block;margin:0 auto 8px">'
                        for p in preview_pngs
                    )
                    components.html(f"""
                      <label style="font:13px sans-serif">확대/축소
                        <input type="range" min="50" max="300" value="100"
                          oninput="pg.style.width=this.value+'%';zl.textContent=this.value+'%'">
                        <span id="zl">100%</span>
                      </label>
                      <div style="overflow:auto;max-height:640px;border:1px solid #ddd;padding:6px">
                        <div id="pg" style="width:100%">{imgs}</div>
                      </div>
                    """, height=720, scrolling=True)
                elif oname:
                    st.text_area("ref", ref_text or "(없음)", height=220,
                                 key=f"ref_{pname}", label_visibility="collapsed")
                else:
                    st.info("원본 파일이 없습니다.")
            with c2:
                st.caption("파싱 결과 렌더")
                components.html(phtml, height=600, scrolling=True)

        rows.append({
            "파싱파일": pname,
            "원본파일": oname or "",
            "종합점수": judged.get("overall_score") if judged else None,
            "문자/내용": judged.get("text_fidelity") if judged else None,
            "표구조": judged.get("table_structure") if judged else None,
            "판정": judged.get("verdict") if judged else "실패",
            "문자유사도(%)": round(sim * 100, 1) if sim is not None else None,
            "오류수": len(judged.get("errors", [])) if judged else None,
        })

    progress.progress(1.0, text="완료")

    # --- 요약 ---
    st.subheader("요약")
    summary = pd.DataFrame(rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    valid = summary["종합점수"].dropna()
    if len(valid):
        st.metric("평균 종합 점수", f"{valid.mean():.1f}")

    st.download_button("요약 CSV 다운로드", summary.to_csv(index=False).encode("utf-8-sig"),
                       "parsing_qa_summary.csv", "text/csv")
    st.download_button("전체 결과 JSON 다운로드",
                       json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
                       "parsing_qa_result.json", "application/json")
