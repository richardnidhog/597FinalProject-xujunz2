"""
app.py — Streamlit web UI for the UIUC Policy Agent.

Run:
    streamlit run app.py
"""

import contextlib
import io
import os
import re

import streamlit as st

from uiuc_policy_agent import ClaudeBackend, OpenAIBackend, UIUCPolicyAgent

st.set_page_config(
    page_title="UIUC Policy Agent",
    page_icon="🎓",
    layout="centered",
)

_BACKEND_LABELS = {
    "openai": "OpenAI (gpt-4.1-mini)",
    "claude": "Claude (claude-opus-4-6)",
}

_KEY_LABELS = {
    "openai": "OpenAI API Key",
    "claude": "Anthropic API Key",
}

_TOOL_ICONS = {
    "query_local_index": "🔍",
    "search_web": "🌐",
    "fetch_and_extract": "📄",
    "verify_claim": "✔️",
}


def _format_log_line(line: str) -> str | None:
    """Format a single stdout line for display. Returns None to skip the line."""
    m = re.match(r"\s*\[tool\]\s+(\w+)\((.*)\)\s*$", line)
    if m:
        name, args_raw = m.group(1), m.group(2)
        icon = _TOOL_ICONS.get(name, "🔧")
        first_val = re.match(r"\w+=(.+?)(?:,\s*\w+=|$)", args_raw)
        summary = first_val.group(1).strip().strip("'\"") if first_val else args_raw
        return f"{icon} **{name}** — `{summary}`"

    m = re.match(r"\s*\[(index|agent)\]\s+(.+)", line)
    if m:
        return f"*{m.group(2).strip()}*"

    return None


class _LiveWriter(io.StringIO):
    """StringIO that pushes each completed line to a Streamlit placeholder."""

    def __init__(self, placeholder) -> None:
        super().__init__()
        self._placeholder = placeholder
        self._items: list[str] = []
        self._buf = ""

    def write(self, s: str) -> int:
        result = super().write(s)
        self._buf += s
        # Process only complete lines so partial writes don't produce noise.
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            formatted = _format_log_line(line.rstrip())
            if formatted:
                self._items.append(formatted)
                self._placeholder.markdown("\n\n".join(self._items))
        return result


def _render_tool_calls(tool_log: str) -> None:
    """Render historical tool calls in a collapsed expander."""
    calls = []
    for line in tool_log.splitlines():
        m = re.match(r"\s*\[tool\]\s+(\w+)\((.*)\)\s*$", line)
        if not m:
            continue
        name, args_raw = m.group(1), m.group(2)
        icon = _TOOL_ICONS.get(name, "🔧")
        first_val = re.match(r"\w+=(.+?)(?:,\s*\w+=|$)", args_raw)
        summary = first_val.group(1).strip().strip("'\"") if first_val else args_raw
        calls.append({"icon": icon, "name": name, "summary": summary, "args": args_raw})

    if not calls:
        return

    with st.expander(f"Tool calls ({len(calls)})", expanded=False):
        for call in calls:
            st.markdown(f"{call['icon']} **{call['name']}** — `{call['summary']}`")
            if call["args"]:
                st.code(call["args"], language=None)


# --- Sidebar ---

with st.sidebar:
    st.title("UIUC Policy Agent")
    st.caption("Answers degree-requirement questions for LAS, Grainger Engineering, and the iSchool.")

    backend_choice = st.radio(
        "LLM Backend",
        options=["openai", "claude"],
        format_func=lambda x: _BACKEND_LABELS[x],
        key="backend_radio",
    )

    api_key = st.text_input(
        _KEY_LABELS[backend_choice],
        type="password",
        key=f"api_key_{backend_choice}",
        placeholder="Paste your API key here",
    )

    tavily_key = st.text_input(
        "Tavily API Key (optional)",
        type="password",
        key="api_key_tavily",
        placeholder="Enables live web search",
    )

    connect_clicked = st.button("Connect", use_container_width=True, type="primary")

    if st.button("New session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if "agent" in st.session_state:
        st.success("Connected")
    else:
        st.info("No agent connected.")

    st.caption("Sources indexed: LAS · Grainger Engineering · iSchool")

# --- Agent init on Connect ---

if connect_clicked:
    if not api_key:
        st.sidebar.error(f"Please enter your {_KEY_LABELS[backend_choice]}.")
    else:
        clean_key = "".join(c for c in api_key if ord(c) < 128).strip()
        clean_tavily = "".join(c for c in tavily_key if ord(c) < 128).strip()

        if clean_tavily:
            os.environ["TAVILY_API_KEY"] = clean_tavily
        else:
            os.environ.pop("TAVILY_API_KEY", None)

        try:
            if backend_choice == "claude":
                os.environ["ANTHROPIC_API_KEY"] = clean_key
                backend = ClaudeBackend()
            else:
                os.environ["OPENAI_API_KEY"] = clean_key
                backend = OpenAIBackend()

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                st.session_state.agent = UIUCPolicyAgent(backend)

            st.session_state.active_backend = backend_choice
            st.session_state.messages = []
            st.rerun()

        except Exception as exc:
            st.sidebar.error(f"Failed to connect: {exc}")

# --- Main area ---

st.title("UIUC Policy Agent")

if "agent" not in st.session_state:
    st.info("Enter your API key in the sidebar and click **Connect** to start.")
    st.stop()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_log"):
            _render_tool_calls(msg["tool_log"])

# Chat input
if prompt := st.chat_input("e.g. How many credit hours does LAS require to graduate?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Working…", expanded=True) as status:
            live = st.empty()
            buf = _LiveWriter(live)
            with contextlib.redirect_stdout(buf):
                answer = st.session_state.agent.chat(prompt)
            tool_log = buf.getvalue().strip()
            status.update(label="Done", state="complete", expanded=False)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_log": tool_log}
    )
