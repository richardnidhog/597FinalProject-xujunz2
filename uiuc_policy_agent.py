"""
uiuc_policy_agent.py — ReAct agent for UIUC degree requirement queries.

Tools
-----
  query_local_index  semantic search over the pre-built FAISS index
  search_web         live Tavily web search  (requires TAVILY_API_KEY)
  fetch_and_extract  fetch a URL and extract its main text
  verify_claim       LLM-based claim verification against source text

Backends
--------
  openai   gpt-5.4-mini        (requires OPENAI_API_KEY)
  claude   claude-opus-4-6     (requires ANTHROPIC_API_KEY)

Run
---
  python uiuc_policy_agent.py                   # choose backend interactively
  python uiuc_policy_agent.py --backend openai
  python uiuc_policy_agent.py --backend claude

Prerequisites
-------------
  The FAISS index must already exist in ./rag_db/ (run the notebook first).
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False

# Config

DB_DIR = "./rag_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-5.4-mini"
CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOOL_ROUNDS = 8

# Common types

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


# LLM backends

class OpenAIBackend:
    label = f"OpenAI ({OPENAI_MODEL})"

    def __init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI()

    def complete(
        self, messages: list[dict], system: str
    ) -> tuple[str | None, list[ToolCall], bool]:
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            tools=_OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in (msg.tool_calls or [])
        ]
        return msg.content, tool_calls, resp.choices[0].finish_reason != "tool_calls"

    def serialize_assistant_turn(
        self, content: str | None, tool_calls: list[ToolCall]
    ) -> dict[str, Any]:
        turn: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ]
        return turn

    def serialize_tool_results(
        self, results: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": id_, "content": content}
            for id_, content in results
        ]

    def complete_json(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


class ClaudeBackend:
    label = f"Claude ({CLAUDE_MODEL})"

    def __init__(self) -> None:
        import anthropic
        self.client = anthropic.Anthropic()
        # Stores the raw SDK content blocks from the last complete() call.
        # Claude requires thinking blocks to be returned verbatim (including
        # their cryptographic signature) in multi-step tool-use sequences.
        self._last_content: list = []

    def complete(
        self, messages: list[dict], system: str
    ) -> tuple[str | None, list[ToolCall], bool]:
        resp = self.client.messages.create(
            model=CLAUDE_MODEL,
            # System prompt as a cacheable block — stable across all turns.
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
            tools=_CLAUDE_TOOLS,
            max_tokens=16000,
            thinking={"type": "adaptive"},
        )
        self._last_content = resp.content

        content_text: str | None = None
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
        return content_text, tool_calls, resp.stop_reason != "tool_use"

    def serialize_assistant_turn(
        self, content: str | None, tool_calls: list[ToolCall]
    ) -> dict[str, Any]:
        # Return ALL raw content blocks (text + thinking + tool_use) verbatim
        # so the API receives thinking blocks with their original signatures.
        return {"role": "assistant", "content": self._last_content}

    def serialize_tool_results(
        self, results: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        # Claude requires all tool results in a single user message.
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": id_, "content": content}
                for id_, content in results
            ],
        }]

    def complete_json(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=CLAUDE_MODEL,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=1024,
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        return "{}"


# Module-level reference so verify_claim can use whichever backend is active.
_active_backend: OpenAIBackend | ClaudeBackend | None = None


# Tool implementations
# FAISS index is lazy-loaded once on first query.

_embed_model: SentenceTransformer | None = None
_faiss_index = None
_faiss_meta: list[dict] | None = None


def _load_index() -> None:
    global _embed_model, _faiss_index, _faiss_meta
    if _faiss_index is not None:
        return
    index_path = os.path.join(DB_DIR, "index_pages.faiss")
    meta_path = os.path.join(DB_DIR, "meta_pages.jsonl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run uiuc_policy_rag_demo.ipynb first to build the index."
        )
    print("[index] loading FAISS index and embedding model...")
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    _faiss_index = faiss.read_index(index_path)
    _faiss_meta = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            _faiss_meta.append(json.loads(line))
    print(f"[index] loaded {len(_faiss_meta)} documents")


def query_local_index(query: str, top_k: int = 2) -> str:
    _load_index()
    q_emb = _embed_model.encode([query], normalize_embeddings=True)
    scores, ids = _faiss_index.search(np.asarray(q_emb, dtype=np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        rec = _faiss_meta[idx]
        results.append({
            "college": rec["college"],
            "source_url": rec["source_url"],
            "similarity_score": round(float(score), 3),
            "text": rec["page_text"][:8000],
        })
    return json.dumps(results, ensure_ascii=False)


def search_web(query: str) -> str:
    if not _TAVILY_AVAILABLE:
        return json.dumps({"error": "tavily-python not installed. Run: pip install tavily-python"})
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY environment variable not set."})
    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query, max_results=5, search_depth="basic")
        return json.dumps([
            {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")[:600]}
            for r in resp.get("results", [])
        ], ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def fetch_and_extract(url: str) -> str:
    try:
        headers = {"User-Agent": "uiuc-policy-agent/1.0 (course project; polite fetch)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        return json.dumps({"error": str(exc), "url": url})
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    node = soup.find("main") or soup.body or soup
    text = re.sub(r"\s+", " ", node.get_text(" ")).strip()
    return json.dumps({"url": url, "text": text[:10000]}, ensure_ascii=False)


def verify_claim(claim: str, source_text: str) -> str:
    if _active_backend is None:
        return json.dumps({"error": "No active backend."})
    system = (
        "You are a strict fact-checker. "
        "Only mark supported=true if the claim is explicitly stated in the source."
    )
    user = (
        f"Does the source text support the claim?\n\n"
        f"Claim: {claim}\n\n"
        f"Source text:\n{source_text[:8000]}\n\n"
        f'Return JSON only: {{"supported": bool, "explanation": str, "relevant_quote": str}}'
    )
    return _active_backend.complete_json(system, user)


# Tool registry & dispatch

_TOOL_REGISTRY = {
    "query_local_index": query_local_index,
    "search_web": search_web,
    "fetch_and_extract": fetch_and_extract,
    "verify_claim": verify_claim,
}


def _dispatch(name: str, args: dict[str, Any]) -> str:
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# Tool schemas
# Single source of truth; generate both OpenAI and Claude formats from it.

_TOOL_DEFS = [
    {
        "name": "query_local_index",
        "description": (
            "Semantic search over the local FAISS index built from official UIUC degree "
            "requirement pages for LAS, Engineering, and iSchool. Use this first for "
            "questions about those three colleges."
        ),
        "params": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 2, "description": "Number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Live web search via Tavily. Use when the question covers other UIUC colleges, "
            "implies recency, or the local index similarity score is below 0.4."
        ),
        "params": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_and_extract",
        "description": "Fetch a URL and extract its main text. Use after search_web to read a full page.",
        "params": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "verify_claim",
        "description": (
            "Strictly verify whether a specific claim is explicitly supported by source text. "
            "Use before asserting any numerical requirement (credit counts, GPA thresholds, etc.)."
        ),
        "params": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "source_text": {"type": "string"},
            },
            "required": ["claim", "source_text"],
        },
    },
]

# OpenAI format: tools → function → parameters
_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["params"],
        },
    }
    for t in _TOOL_DEFS
]

# Anthropic format: flat object with input_schema
_CLAUDE_TOOLS = [
    {
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["params"],
    }
    for t in _TOOL_DEFS
]


# System prompt

_SYSTEM_PROMPT = """You are a knowledgeable assistant for UIUC degree requirements, helping \
students and advisors understand graduation requirements for the College of Liberal Arts & \
Sciences (LAS), Grainger College of Engineering, and the iSchool (Information Sciences).

You have four tools:
- query_local_index  Search the pre-built FAISS index for LAS, Engineering, and iSchool pages.
- search_web         Live Tavily web search for other colleges or time-sensitive questions.
- fetch_and_extract  Read the full content of a specific URL.
- verify_claim       Strict LLM verification of a single claim against source text.

Decision rules:
1. For LAS / Engineering / iSchool questions → start with query_local_index.
2. If similarity_score < 0.4, or the question covers other colleges → use search_web, \
then fetch_and_extract on the most relevant result.
3. Before stating any specific number (credit hours, GPA, course count) → call verify_claim.
4. Always cite the source URL in your final answer.
5. If evidence is genuinely insufficient, say so — do not guess.

You retain full conversation history. Refer back to earlier turns when relevant."""


# Agent 

class UIUCPolicyAgent:
    """ReAct agent with conversational memory and pluggable LLM backend."""

    def __init__(self, backend: OpenAIBackend | ClaudeBackend) -> None:
        global _active_backend
        self.backend = backend
        _active_backend = backend
        self.history: list[dict[str, Any]] = []
        print(f"[agent] Using backend: {backend.label}")
        if not _TAVILY_AVAILABLE or not os.getenv("TAVILY_API_KEY"):
            print(
                "[agent] Warning: search_web is disabled. "
                "Install tavily-python and set TAVILY_API_KEY to enable live web search."
            )

    def chat(self, user_input: str) -> str:
        """Send a user message and return the agent's final response."""
        self.history.append({"role": "user", "content": user_input})
        # Build messages fresh from history each turn (API is stateless).
        messages: list[dict[str, Any]] = list(self.history)

        for _ in range(MAX_TOOL_ROUNDS):
            content, tool_calls, is_final = self.backend.complete(messages, _SYSTEM_PROMPT)

            # Append the assistant turn (preserves thinking/tool_use blocks for
            # Claude; a plain dict with optional tool_calls for OpenAI).
            messages.append(self.backend.serialize_assistant_turn(content, tool_calls))

            if is_final:
                answer = content or ""
                self.history.append({"role": "assistant", "content": answer})
                return answer

            # Execute tools and collect results before the next LLM call.
            tool_results: list[tuple[str, str]] = []
            for tc in tool_calls:
                print(
                    f"  [tool] {tc.name}"
                    f"({', '.join(f'{k}={v!r}' for k, v in tc.arguments.items())})"
                )
                result = _dispatch(tc.name, tc.arguments)
                tool_results.append((tc.id, result))

            # Append tool results in the format expected by each backend.
            messages.extend(self.backend.serialize_tool_results(tool_results))

        fallback = (
            "I reached the tool-use limit without a conclusive answer. "
            "Please try a more specific question."
        )
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self) -> None:
        self.history.clear()
        print("[agent] Conversation history cleared.")


# Backend selection

def _pick_backend(choice: str | None) -> OpenAIBackend | ClaudeBackend:
    if choice is None:
        print("Select LLM backend:")
        print(f"  1. OpenAI  ({OPENAI_MODEL})   — requires OPENAI_API_KEY")
        print(f"  2. Claude  ({CLAUDE_MODEL}) — requires ANTHROPIC_API_KEY")
        raw = input("Enter 1 or 2 [default 1]: ").strip()
        choice = "claude" if raw == "2" else "openai"

    if choice == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        return ClaudeBackend()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAIBackend()


# CLI 

def main() -> None:
    parser = argparse.ArgumentParser(description="UIUC Policy Agent")
    parser.add_argument(
        "--backend",
        choices=["openai", "claude"],
        default=None,
        help="LLM backend (default: ask interactively)",
    )
    args = parser.parse_args()

    print("UIUC Policy Agent")
    print("Type 'reset' to clear history, 'exit' to quit.\n")

    try:
        backend = _pick_backend(args.backend)
    except EnvironmentError as exc:
        print(f"[error] {exc}")
        return

    agent = UIUCPolicyAgent(backend)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue
        answer = agent.chat(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
