"""
Streamlit 채팅 UI (Claude Agent SDK 버전)

실행: streamlit run app.py   (반드시 app2/ 디렉터리 안에서)

화면과 사용법은 app/ 과 같다. 내부 차이는 두 가지다.
  - graph.stream(...) 대신 main_agent.stream_turn(...) 의 이벤트를 그린다.
  - 대화 이어가기는 체크포인터(thread_id) 대신 Claude Code 세션(session_id resume)이 맡는다.

색상/폰트는 .streamlit/config.toml 에서 정의한다.
여기의 CSS 는 레이아웃·간격·위젯 형태만 다룬다.
"""
import datetime
import json
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 키는 app2/.env 를 우선 보고, 없으면 app/.env 를 그대로 쓴다.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / ".env", _HERE.parent / "app" / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break

from async_utils import iterate_in_thread  # noqa: E402  (load_dotenv 이후에 읽어야 한다)
from main_agent import clarification_prompt, stream_turn  # noqa: E402

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
def configure_langsmith() -> bool:
    """LANGSMITH_TRACING=true 면 Agent SDK 호출을 LangSmith 로 보낸다(공식 연동)."""
    if os.getenv("LANGSMITH_TRACING", "").lower() not in ("true", "1"):
        return False
    from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk

    return configure_claude_agent_sdk(name="main_agent", project_name=os.getenv("LANGSMITH_PROJECT") or None)


@st.cache_resource
def get_langfuse():
    """LANGFUSE_PUBLIC_KEY 가 있으면 Langfuse 클라이언트를, 없으면 None 을 반환"""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    from langfuse import get_client

    return get_client()


class LangfuseTurn:
    """한 턴의 이벤트를 Langfuse trace 로 남긴다.

    app/ 은 LangChain CallbackHandler 가 그래프 노드를 span 으로 쌓는다. 여기서는
    메시지 스트림의 tool_use_id / parent_tool_use_id 로 같은 트리를 만든다.
      main_agent
        └ domain:server_infra (Agent 위임)
            └ vault_read ...   (서브에이전트의 도구 호출)
    """

    def __init__(self, client, root) -> None:
        self.client, self.root = client, root
        self.spans: dict[str, object] = {}

    def on_event(self, ev) -> None:
        if ev.kind == "tool_call":
            parent = self.spans.get(ev.parent_tool_use_id, self.root)
            name = ev.name
            if name in ("Agent", "Task"):
                name = f"domain:{(ev.body or {}).get('subagent_type', '')}"
            self.spans[ev.tool_use_id] = parent.start_observation(
                name=name, as_type="agent" if name.startswith("domain:") else "tool", input=ev.body
            )
        elif ev.kind == "tool_result" and ev.tool_use_id in self.spans:
            span = self.spans.pop(ev.tool_use_id)
            span.update(output=ev.body, level="ERROR" if ev.is_error else None)
            span.end()
        elif ev.kind == "done":
            usage = ev.data.get("usage") or {}
            self.root.update(
                output=ev.body,
                metadata={"session_id": ev.data.get("session_id"), "num_turns": ev.data.get("num_turns")},
            )
            gen = self.root.start_observation(
                name="claude-agent-sdk",
                as_type="generation",
                usage_details={
                    "input": int(usage.get("input_tokens") or 0),
                    "output": int(usage.get("output_tokens") or 0),
                },
                cost_details={"total": float(ev.data.get("cost_usd") or 0.0)},
            )
            gen.end()

    def close(self) -> None:
        for span in self.spans.values():
            span.end()
        self.spans.clear()


def to_text(content) -> str:
    """도구 결과 content 가 str 또는 block list 일 수 있으므로 텍스트만 추출"""
    if content is None:
        return ""
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


configure_langsmith()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "agent_session_id" not in st.session_state:
    # Claude Code 세션 ID. 첫 턴이 끝나면 채워지고, 다음 턴부터 resume 에 쓴다.
    st.session_state.agent_session_id = None
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="mark">✳</span>IT Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">구성</div>', unsafe_allow_html=True)
    st.caption("Orchestrator → 도메인 서브에이전트 병렬 → 병합 (Claude Agent SDK)")

    st.markdown('<div class="section-label">표시</div>', unsafe_allow_html=True)
    show_steps = st.toggle("도구 호출 과정", value=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("run_prompt", None)
        st.session_state.pop("resume_value", None)
        st.session_state.pending_clarification = None
        # 세션을 끊어야 이전 대화와 섞이지 않는다.
        st.session_state.agent_session_id = None
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

    with st.chat_message("assistant"):
        status = st.status("생각하는 중...", expanded=show_steps) if show_steps else None
        answer = ""

        # 재질문에 답한 경우엔 같은 세션에 사용자 추가 정보로 이어서 보낸다.
        resume_value = st.session_state.pop("resume_value", None)
        agent_prompt = clarification_prompt(resume_value) if resume_value is not None else prompt

        # Langfuse: 같은 채팅 세션의 질문들을 하나의 session 으로 묶어서 기록
        langfuse = get_langfuse()
        recorder = None
        root_cm = attrs_cm = None
        if langfuse:
            from langfuse import propagate_attributes

            root_cm = langfuse.start_as_current_observation(name="main_agent", as_type="span", input=agent_prompt)
            root = root_cm.__enter__()
            attrs_cm = propagate_attributes(
                session_id=st.session_state.session_id, tags=["streamlit", "main_agent"], trace_name="main_agent"
            )
            attrs_cm.__enter__()
            recorder = LangfuseTurn(langfuse, root)

        try:
            # st.session_state 는 스크립트 스레드에서만 읽을 수 있다. 값을 먼저 꺼내 넘긴다.
            agent_session_id = st.session_state.agent_session_id
            events = iterate_in_thread(lambda: stream_turn(agent_prompt, agent_session_id))
            for ev in events:
                if recorder:
                    recorder.on_event(ev)
                # 도구 호출 과정은 서브에이전트 안에서 난 것만 표시(app/ 의 namespace 조건과 같다)
                if ev.kind == "tool_call" and status and ev.nested:
                    status.markdown(f"**{ev.name}**")
                    status.code(truncate(ev.body), language="json")
                elif ev.kind == "tool_result" and status and ev.nested:
                    status.markdown(f"↳ {ev.name}")
                    status.code(truncate(to_text(ev.body)))
                elif ev.kind == "clarification":
                    # Orchestrator 가 모호하다고 판단하면 되묻고 멈춘다.
                    st.session_state.pending_clarification = ev.body
                elif ev.kind == "done":
                    answer = ev.body
                    st.session_state.agent_session_id = ev.data.get("session_id")
                elif ev.kind == "error":
                    raise RuntimeError(ev.body)

            if status:
                status.update(label="완료", state="complete", expanded=False)
        except Exception as e:
            if status:
                status.update(label="오류 발생", state="error")
            answer = f"⚠️ 오류가 발생했습니다: `{e}`"
        finally:
            if recorder:
                recorder.close()
                attrs_cm.__exit__(None, None, None)
                root_cm.__exit__(None, None, None)
                langfuse.flush()

        if st.session_state.pending_clarification:
            # 아직 답변이 없다. 되묻고 사용자의 답을 기다린다.
            answer = f"**확인이 필요합니다** — {st.session_state.pending_clarification}"

        st.markdown(answer or "_(응답이 비어 있습니다)_")

    st.session_state.messages.append({"role": "assistant", "content": answer})

if typed:
    if st.session_state.pending_clarification:
        # 재질문에 대한 답. 새 질문이 아니라 같은 세션에서 이어서 처리한다.
        st.session_state.pending_clarification = None
        st.session_state.resume_value = typed
        st.session_state.messages.append({"role": "user", "content": typed})
        st.session_state.run_prompt = typed
        st.rerun()
    else:
        submit(typed)
