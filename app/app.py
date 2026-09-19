"""
Streamlit 채팅 UI

실행: streamlit run app.py   (반드시 app/ 디렉터리 안에서)

색상/폰트는 .streamlit/config.toml 에서 정의한다.
여기의 CSS 는 레이아웃·간격·위젯 형태만 다룬다.
"""
import datetime
import json
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

load_dotenv()

st.set_page_config(
    page_title="IT Agent",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "JWT와 세션 인증의 차이점이 뭔가?",
    "트래픽 급증 시 p99 지연이 튀면 무엇부터 봐야 하나?",
    "MSA에서 동기 REST와 Kafka 이벤트 중 무엇을 골라야 하나?",
]


# ---------------- Style ----------------
CSS = """
<style>
/* 기본 크롬 제거 — 상단 헤더 바와 푸터가 다크 톤을 깨뜨린다 */
header[data-testid="stHeader"] {background: transparent; height: 0;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

/* 본문 폭을 채팅에 맞게 좁히고 하단 입력창 공간을 확보 */
.block-container {padding-top: 3rem; padding-bottom: 6rem; max-width: 780px;}

/* ---- 웰컴 화면 ---- */
.welcome-mark {
    font-size: 2rem;
    line-height: 1;
    color: #d97757;
    text-align: center;
    margin-bottom: 1rem;
}
.welcome-title {
    font-size: 2.1rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    text-align: center;
    color: #f5f4ef;
    margin: 0 0 2.25rem 0;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8a867e;
    margin: 1.4rem 0 0.5rem 0;
}

/* ---- 사이드바 ---- */
.brand {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;
    color: #f5f4ef; padding: 0.25rem 0 0.5rem 0;
}
.brand .mark {color: #d97757; font-size: 1.15rem;}

/* 라디오를 버튼 나열이 아닌 목록처럼 */
[data-testid="stSidebar"] [role="radiogroup"] {gap: 0.15rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.42rem 0.6rem;
    border-radius: 0.55rem;
    transition: background 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: #2b2a27;}

/* ---- 예시 질문 칩 ---- */
.stButton > button {
    border: 1px solid #3d3a36;
    background: transparent;
    color: #b8b4ac;
    font-size: 0.83rem;
    font-weight: 400;
    text-align: left;
    padding: 0.55rem 0.8rem;
    line-height: 1.35;
    transition: border-color 120ms ease, color 120ms ease;
}
.stButton > button:hover {
    border-color: #d97757;
    color: #f5f4ef;
    background: transparent;
}

/* ---- 채팅 메시지 ---- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.4rem 0;
}
[data-testid="stChatMessage"] p {line-height: 1.7;}

/* ---- 도구 호출 과정 카드 ---- */
[data-testid="stExpander"] {
    border: 1px solid #343230;
    border-radius: 0.7rem;
    background: #232220;
}
[data-testid="stExpander"] summary {font-size: 0.82rem; color: #a3a09a;}
[data-testid="stExpander"] code {font-size: 0.74rem; line-height: 1.5;}
[data-testid="stExpander"] p {font-size: 0.8rem; margin-bottom: 0.25rem;}

/* ---- 입력창 ---- */
[data-testid="stChatInput"] {
    border: 1px solid #3d3a36;
    border-radius: 0.9rem;
    background: #302f2c;
}
[data-testid="stChatInput"]:focus-within {border-color: #57534c;}
[data-testid="stBottomBlockContainer"] {
    padding-bottom: 1.5rem;
    max-width: 780px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_graph():
    from langgraph.checkpoint.memory import InMemorySaver
    from main_agent import build_main_graph

    # interrupt(재질문)는 checkpointer 없이는 동작하지 않는다.
    # @st.cache_resource 가 프로세스 수명 동안 유지하므로 rerun 사이에도 상태가 남는다.
    # 앱을 재시작하면 진행 중이던 대화는 사라진다(토이프로젝트 범위에서 허용).
    return build_main_graph(checkpointer=InMemorySaver())


@st.cache_resource
def get_langfuse_handler():
    """LANGFUSE_PUBLIC_KEY가 있으면 Langfuse 콜백 핸들러를, 없으면 None을 반환"""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def to_text(content) -> str:
    """AIMessage.content가 str 또는 block list일 수 있으므로 텍스트만 추출"""
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
    )


def truncate(value, limit: int = 800) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


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
    """입력을 큐에 넣고 다시 그린다. 웰컴 화면 -> 채팅 화면 전환도 여기서 일어난다."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.run_prompt = prompt
    st.rerun()


graph = load_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="mark">✳</span>IT Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">구성</div>', unsafe_allow_html=True)
    st.caption("Orchestrator → 판단 Agent → 도메인 3종 병렬 → 병합")

    st.markdown('<div class="section-label">표시</div>', unsafe_allow_html=True)
    show_steps = st.toggle("도구 호출 과정", value=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("run_prompt", None)
        st.session_state.pop("resume_value", None)
        st.session_state.pending_clarification = None
        # thread_id 가 바뀌어야 checkpointer 의 이전 대화와 섞이지 않는다.
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

is_welcome = not st.session_state.messages and "run_prompt" not in st.session_state

# ---------------- Welcome ----------------
if is_welcome:
    st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
    st.markdown('<div class="welcome-mark">✳</div>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 class="welcome-title">{greeting()}</h1>',
        unsafe_allow_html=True,
    )

    # 컨테이너 안에 두면 하단 고정이 아니라 인라인으로 렌더된다.
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

    history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        status = st.status("생각하는 중...", expanded=show_steps) if show_steps else None
        answer = ""

        # Langfuse: 같은 채팅 세션의 질문들을 하나의 session으로 묶어서 기록
        config = {
            "run_name": "main_agent",
            # checkpointer 를 쓰면 thread_id 가 필수다.
            "configurable": {"thread_id": st.session_state.session_id},
            "metadata": {
                "langfuse_session_id": st.session_state.session_id,
                "langfuse_tags": ["streamlit", graph.name],
            },
        }
        langfuse_handler = get_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        # 재질문에 답한 경우엔 새 질문이 아니라 멈춘 지점부터 재개한다.
        resume_value = st.session_state.pop("resume_value", None)
        graph_input = (
            Command(resume=resume_value)
            if resume_value is not None
            else {"messages": history, "question": prompt}
        )

        try:
            # subgraphs=True 로 내부 ReAct 에이전트의 model/tools 단계까지 스트리밍
            for namespace, update in graph.stream(
                graph_input,
                config=config,
                stream_mode="updates",
                subgraphs=True,
            ):
                # Orchestrator 가 모호하다고 판단하면 여기서 멈춘다.
                if "__interrupt__" in update:
                    payload = update["__interrupt__"][0].value
                    st.session_state.pending_clarification = payload.get(
                        "clarifying_question", "추가 정보가 필요합니다."
                    )
                    break

                for node_update in update.values():
                    if not node_update:
                        continue
                    for msg in node_update.get("messages", []):
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                # 도구 호출 과정은 내부 ReAct 에이전트(namespace 존재)에서만 표시
                                if status and namespace:
                                    for tc in msg.tool_calls:
                                        status.markdown(f"**{tc['name']}**")
                                        status.code(truncate(tc["args"]), language="json")
                            elif to_text(msg.content).strip():
                                answer = to_text(msg.content)
                        elif isinstance(msg, ToolMessage) and status and namespace:
                            status.markdown(f"↳ {msg.name}")
                            status.code(truncate(to_text(msg.content)))

            if status:
                status.update(label="완료", state="complete", expanded=False)
        except Exception as e:
            if status:
                status.update(label="오류 발생", state="error")
            answer = f"⚠️ 오류가 발생했습니다: `{e}`"

        if st.session_state.pending_clarification:
            # 아직 답변이 없다. 되묻고 사용자의 답을 기다린다.
            answer = f"**확인이 필요합니다** — {st.session_state.pending_clarification}"

        st.markdown(answer or "_(응답이 비어 있습니다)_")

    st.session_state.messages.append({"role": "assistant", "content": answer})

if typed:
    if st.session_state.pending_clarification:
        # 재질문에 대한 답. 새 질문이 아니라 멈춘 지점을 재개한다.
        st.session_state.pending_clarification = None
        st.session_state.resume_value = typed
        st.session_state.messages.append({"role": "user", "content": typed})
        st.session_state.run_prompt = typed
        st.rerun()
    else:
        submit(typed)
