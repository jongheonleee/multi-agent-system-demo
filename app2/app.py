"""
Streamlit 채팅 UI (Claude Agent SDK 버전)

실행: streamlit run app.py   (반드시 app2/ 디렉터리 안에서)

app/ 의 LangGraph 버전과 화면은 같고, 내부만 Claude Agent SDK 로 바뀌었다.
색상/폰트는 .streamlit/config.toml 에서 정의한다.
"""
import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 키는 app2/.env 를 우선 보고, 없으면 app/.env 를 그대로 쓴다.
_HERE = Path(__file__).resolve().parent
for candidate in (_HERE / ".env", _HERE.parent / "app" / ".env"):
    if candidate.is_file():
        load_dotenv(candidate)
        break

from agents import MODEL  # noqa: E402  (load_dotenv 이후에 읽어야 한다)
from runner import Event, run  # noqa: E402

st.set_page_config(
    page_title="IT Agent (SDK)",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "JWT와 세션 인증의 차이점이 뭔가?",
    "트래픽 급증 시 p99 지연이 튀면 무엇부터 봐야 하나?",
    "MSA에서 동기 REST와 Kafka 이벤트 중 무엇을 골라야 하나?",
]

CSS = """
<style>
header[data-testid="stHeader"] {background: transparent; height: 0;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

.block-container {padding-top: 3rem; padding-bottom: 6rem; max-width: 780px;}

.welcome-mark {font-size: 2rem; line-height: 1; color: #d97757; text-align: center; margin-bottom: 1rem;}
.welcome-title {
    font-size: 2.1rem; font-weight: 400; letter-spacing: -0.02em;
    text-align: center; color: #f5f4ef; margin: 0 0 2.25rem 0;
}
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    color: #8a867e; margin: 1.4rem 0 0.5rem 0;
}
.brand {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;
    color: #f5f4ef; padding: 0.25rem 0 0.5rem 0;
}
.brand .mark {color: #d97757; font-size: 1.15rem;}
.meta {font-size: 0.74rem; color: #8a867e; margin-top: 0.6rem;}

.stButton > button {
    border: 1px solid #3d3a36; background: transparent; color: #b8b4ac;
    font-size: 0.83rem; font-weight: 400; text-align: left;
    padding: 0.55rem 0.8rem; line-height: 1.35;
    transition: border-color 120ms ease, color 120ms ease;
}
.stButton > button:hover {border-color: #d97757; color: #f5f4ef; background: transparent;}

[data-testid="stChatMessage"] {background: transparent; padding: 0.4rem 0;}
[data-testid="stChatMessage"] p {line-height: 1.7;}

[data-testid="stExpander"] {border: 1px solid #343230; border-radius: 0.7rem; background: #232220;}
[data-testid="stExpander"] summary {font-size: 0.82rem; color: #a3a09a;}
[data-testid="stExpander"] code {font-size: 0.74rem; line-height: 1.5;}
[data-testid="stExpander"] p {font-size: 0.8rem; margin-bottom: 0.25rem;}

[data-testid="stChatInput"] {border: 1px solid #3d3a36; border-radius: 0.9rem; background: #302f2c;}
[data-testid="stChatInput"]:focus-within {border-color: #57534c;}
[data-testid="stBottomBlockContainer"] {padding-bottom: 1.5rem; max-width: 780px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def greeting() -> str:
    hour = datetime.datetime.now().hour
    if hour < 6:
        return "늦은 시간까지 고생이 많으시네요"
    if hour < 12:
        return "좋은 아침이에요"
    if hour < 18:
        return "좋은 오후예요"
    return "좋은 저녁이에요"


def submit(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.run_prompt = prompt
    st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "sdk_session_id" not in st.session_state:
    st.session_state.sdk_session_id = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="mark">✳</span>IT Agent <span style="color:#8a867e;font-weight:400">SDK</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">구성</div>', unsafe_allow_html=True)
    st.caption(f"모델 `{MODEL}`")
    st.caption("라우터 → 서브에이전트 2종 (논문 RAG / 웹 검색)")

    st.markdown('<div class="section-label">표시</div>', unsafe_allow_html=True)
    show_steps = st.toggle("도구 호출 과정", value=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("run_prompt", None)
        st.session_state.sdk_session_id = None
        st.rerun()

    if st.session_state.sdk_session_id:
        st.markdown(
            f'<div class="meta">session {st.session_state.sdk_session_id[:8]}</div>',
            unsafe_allow_html=True,
        )

is_welcome = not st.session_state.messages and "run_prompt" not in st.session_state

# ---------------- Welcome ----------------
if is_welcome:
    st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
    st.markdown('<div class="welcome-mark">✳</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="welcome-title">{greeting()}</h1>', unsafe_allow_html=True)

    with st.container():
        typed = st.chat_input("무엇이든 물어보세요", key="welcome_input")

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    for col, question in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if col.button(question, key=f"chip_{question}", use_container_width=True):
            submit(question)

    if typed:
        submit(typed)
    st.stop()

# ---------------- Chat ----------------
typed = st.chat_input("질문을 입력하세요", key="chat_input")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if "run_prompt" in st.session_state:
    prompt = st.session_state.pop("run_prompt")

    with st.chat_message("assistant"):
        status = st.status("에이전트 실행 중...", expanded=show_steps) if show_steps else None

        def on_event(event: Event) -> None:
            if status is None:
                return
            prefix = "  ↳ " if event.nested else ""
            if event.kind == "tool_use":
                status.markdown(f"{prefix}**{event.name}** 호출")
                if event.body:
                    status.code(event.body, language="json")
            elif event.kind == "tool_result":
                status.markdown(f"{prefix}{'⚠️ 오류' if event.is_error else '결과'}")
                status.code(event.body)
            elif event.kind == "text" and event.nested:
                status.markdown(f"{prefix}_{event.body[:300]}_")
            elif event.kind == "error":
                status.markdown(f"⚠️ {event.body}")

        outcome = run(
            prompt,
            on_event=on_event,
            resume=st.session_state.sdk_session_id,
            cwd=str(_HERE),
        )

        if outcome.session_id:
            st.session_state.sdk_session_id = outcome.session_id

        if status:
            if outcome.error:
                status.update(label="오류 발생", state="error")
            else:
                bits = []
                if outcome.num_turns:
                    bits.append(f"{outcome.num_turns}턴")
                if outcome.cost_usd:
                    bits.append(f"${outcome.cost_usd:.4f}")
                status.update(
                    label="완료" + (f" · {' · '.join(bits)}" if bits else ""),
                    state="complete",
                    expanded=False,
                )

        answer = outcome.answer or (
            f"⚠️ 오류가 발생했습니다: `{outcome.error}`" if outcome.error else ""
        )
        st.markdown(answer or "_(응답이 비어 있습니다)_")

    st.session_state.messages.append({"role": "assistant", "content": answer})

if typed:
    submit(typed)
