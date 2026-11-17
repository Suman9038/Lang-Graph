import os
import sys
import uuid
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage

# Streamlit runs this file directly (not as a package), so make sure the
# folder containing backend_with_rag.py is on sys.path before we try
# to import it. This also lets the file be moved around without breaking.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from backend_with_rag import (
    ingest_pdf,
    get_retriever,
    is_rag_ready,
    rag_tool,
)

from backend_with_tools import (
    chatbot,
    retrieve_all_threads,
    delete_thread,
    load_thread_messages,
    thread_title,
)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="LangGraph RAG Chatbot",
    page_icon=":material/chat:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helpers
# ============================================================
def start_new_chat() -> str:
    """Create a brand-new thread_id and reset the chat view.

    Returns the new thread_id so callers can use it immediately.
    """
    new_thread_id = str(uuid.uuid4())
    st.session_state.thread_id = new_thread_id
    st.session_state.messages = []
    return new_thread_id


def _open_thread_clicked(thread_id: str) -> None:
    """Click callback for the per-thread 'open' button.

    Loaded via `on_click` so the switch happens before the script reruns and
    we never end up rendering the wrong thread in this pass.
    """
    st.session_state.thread_id = thread_id
    st.session_state.messages = load_thread_messages(thread_id)


def load_conversation(thread_id: str) -> None:
    """Switch the active view to the given thread, fetching its history
    from SQLite so it survives page refreshes / Streamlit restarts."""
    st.session_state.thread_id = thread_id
    st.session_state.messages = load_thread_messages(thread_id)


def delete_conversation(thread_id: str) -> None:
    """Remove a thread from the DB and clean up session state.

    If the user deletes the currently-active thread we drop them into a
    fresh chat so they're never left looking at a dead conversation.
    """
    delete_thread(thread_id)
    st.session_state.threads = [t for t in st.session_state.threads if t != thread_id]

    if st.session_state.thread_id == thread_id:
        # Drop any stale view and let the main render create a new chat
        st.session_state.pop("thread_id", None)
        st.session_state.pop("messages", None)

    # Title cache is keyed on thread_id; safest to drop the whole cache.
    get_thread_title.clear()


def extract_model_name(message_chunk, metadata):
    """Pull the actual model name out of a streamed chunk.

    Different providers store it in different keys, so check the common
    ones in both the chunk's response_metadata and the stream metadata.
    """
    for source in (
        getattr(message_chunk, "response_metadata", None) or {},
        metadata or {},
    ):
        for key in ("model_name", "model", "ls_model_name"):
            value = source.get(key)
            if value:
                return value
    return None


@st.cache_data(show_spinner=False)
def get_thread_title(thread_id: str) -> str:
    """Cached wrapper around `thread_title` so the sidebar doesn't re-read
    every conversation from SQLite on every rerun."""
    return thread_title(thread_id)


@st.cache_data(ttl="30s", show_spinner=False)
def get_threads() -> list[str]:
    """Cached thread list. Capped at 30s so a new chat shows up quickly
    without hitting SQLite on every rerun."""
    return retrieve_all_threads()


def ensure_active_thread() -> None:
    """Make sure there's a current thread to attach the chat to.

    On a brand-new browser session with no history we create one so the
    chat input has something to bind to.
    """
    if st.session_state.thread_id:
        return
    new_tid = start_new_chat()
    st.session_state.threads.append(new_tid)


# ============================================================
# Session State — initialize once per browser session
# ============================================================
# All thread_ids that exist in the database right now (sidebar source of truth)
st.session_state.setdefault("threads", get_threads())

# The currently active thread_id; lazily created on first send / New Chat
st.session_state.setdefault("thread_id", None)

# Messages of the active conversation, hydrated from the DB when switching
st.session_state.setdefault("messages", [])

# Tracks whether the assistant is mid-stream, so we can dim the chat input
st.session_state.setdefault("is_streaming", False)


# ============================================================
# RAG Initialization Sidebar
# ============================================================
with st.sidebar:
    st.markdown(":material/chat:  **LangGraph RAG Chatbot**", help="Persistent chat with LangGraph + SQLite + PDF RAG")
    st.caption("Powered by LangGraph · Persisted with SQLite · PDF RAG Enabled")

    st.space("medium")

    st.subheader("PDF Ingestion", anchor=False)
    st.caption("Upload a PDF document to enable RAG search")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload any PDF to index for RAG search"
    )

    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50)
    embedding_model = st.text_input("Embedding model", value="embeddinggemma:300m", help="Ollama embedding model name")

    if uploaded_file is not None:
        if st.button("Ingest PDF", type="primary", use_container_width=True):
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Processing PDF..."):
                    result = ingest_pdf(
                        pdf_path=tmp_path,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        embedding_model=embedding_model
                    )
                if result["status"] == "success":
                    st.success(f"✅ {result['message']}")
                    st.info(f"📄 Pages: {result['num_pages']} | 📦 Chunks: {result['num_chunks']} | 🤖 Model: {result['embedding_model']}")
                    st.session_state.rag_ready = True
                    st.session_state.uploaded_filename = uploaded_file.name
                else:
                    st.error(f"❌ {result['message']}")
                    st.session_state.rag_ready = False
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    st.space("small")

    # RAG status indicator
    if st.session_state.get("rag_ready", False):
        st.success("📚 RAG system ready - rag_tool available")
        if st.session_state.get("uploaded_filename"):
            st.caption(f"Loaded: {st.session_state.uploaded_filename}")
    else:
        st.warning("⚠️ RAG system not initialized - upload and ingest a PDF first")

    st.space("small")

    # Available tools info
    st.subheader("Available Tools", anchor=False)
    st.caption("• calculator • get_stock_price • search_tool • rag_tool")


# ============================================================
# Sidebar: Conversations
# ============================================================
with st.sidebar:
    st.space("medium")

    st.subheader("My conversations", anchor=False)

    threads = get_threads()
    if not threads:
        with st.container(border=True):
            st.caption("No conversations yet. Start one to get going.")
    else:
        with st.container(border=True, gap="small"):
            for tid in reversed(threads):
                is_active = tid == st.session_state.thread_id
                label = get_thread_title(tid)
                chat_label = f":material/forum: **{label}**" if is_active else f":material/chat_bubble: {label}"
                button_type = "tertiary" if is_active else "secondary"

                chat_col, del_col = st.columns([9, 1], gap="small", vertical_alignment="center")
                with chat_col:
                    st.button(
                        chat_label,
                        key=f"open_{tid}",
                        width="stretch",
                        type=button_type,
                        disabled=is_active,
                        help="Open this conversation" if not is_active else "Currently active",
                        on_click=_open_thread_clicked,
                        args=(tid,),
                    )
                with del_col:
                    if st.button(
                        " ",
                        icon=":material/delete:",
                        key=f"del_{tid}",
                        help="Delete this conversation",
                        width="content",
                        wrap=False,
                    ):
                        delete_conversation(tid)
                        st.rerun()

    st.space("small")
    with st.container(border=True):
        st.caption("Active thread")
        if st.session_state.thread_id:
            short_id = st.session_state.thread_id[:8]
            st.code(short_id, language=None)
        else:
            st.caption("(none yet)")


# ============================================================
# Main chat area
# ============================================================
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    with st.container(horizontal=False, gap="small"):
        st.title("LangGraph RAG Chatbot", anchor=False)
        st.caption("Streaming chat · Multiple conversations · SQLite-backed history")
    if st.button("New chat", icon=":material/refresh:", width="content"):
        new_tid = start_new_chat()
        if new_tid not in st.session_state.threads:
            st.session_state.threads.append(new_tid)
        st.rerun()

st.space("small")

# Bootstrap a thread on first-ever load (no history in DB yet)
ensure_active_thread()

# LangGraph config bound to the active thread
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Render the active conversation's history
for msg in st.session_state.messages:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "ai" and msg.get("model"):
            st.caption(f"Generated by `{msg['model']}`")

        # Show tool calls if present
        if msg.get("tool_calls"):
            with st.expander(":material/build: Tool calls", expanded=False):
                for tc in msg["tool_calls"]:
                    st.code(f"{tc['name']}({tc['args']})", language="json")

        # Show tool results if present
        if msg.get("tool_results"):
            with st.expander(":material/check_circle: Tool results", expanded=False):
                for tr in msg["tool_results"]:
                    st.markdown(f"**{tr['name']}**")
                    st.code(tr["content"], language="text")


# ---------------- Suggestion chips (only on empty chats) ----------------
SUGGESTIONS = {
    ":blue[:material/rocket_launch:] Get started": "What can you help me with?",
    ":green[:material/lightbulb:] Brainstorm ideas": "Help me brainstorm 3 product names for a developer tool.",
    ":violet[:material/code:] Explain a concept": "Explain how a LangGraph checkpoint works in simple terms.",
    ":orange[:material/article:] Summarize": "Summarize the key differences between SQL and NoSQL databases.",
}

if not st.session_state.messages:
    st.caption("Try asking:")
    selected = st.pills(
        "Suggestions",
        options=list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
        key="suggestion_pills",
    )
    if selected:
        user_input = SUGGESTIONS[selected]
    else:
        user_input = None
else:
    user_input = None


# ---------------- Chat Input (pinned to bottom of the main area) ----------------
chat_input = st.chat_input(
    "Type your message...",
    accept_file=False,
    disabled=st.session_state.is_streaming,
)
if chat_input:
    user_input = chat_input


# ---------------- Handle the user turn ----------------
if user_input:
    # Store the user's message in session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Don't render immediately - let the history loop handle it after rerun
    # This prevents duplicate rendering

    # Stream the bot's reply from the LangGraph backend
    model_used = {"name": None}
    # Use dicts to allow mutation inside nested function (nonlocal workaround)
    tool_calls_collected = {"items": []}
    tool_results_collected = {"items": []}
    st.session_state.is_streaming = True

    # Create placeholders for streaming tool calls and results
    tool_calls_placeholder = st.empty()
    tool_results_placeholder = st.empty()

    def stream_chunks():
        # Use stream_mode="values" to get full state updates including tool calls/results
        for chunk in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
            # Get the latest message from the state
            messages = chunk.get("messages", [])
            if not messages:
                continue

            latest_msg = messages[-1]

            # Check for tool calls in AI messages
            if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                for tc in latest_msg.tool_calls:
                    if tc not in tool_calls_collected["items"]:
                        tool_calls_collected["items"].append(tc)
                        # Stream tool call info
                        with tool_calls_placeholder.container():
                            with st.expander(":material/build: Tool calls", expanded=True):
                                for collected_tc in tool_calls_collected["items"]:
                                    st.code(f"{collected_tc['name']}({collected_tc['args']})", language="json")

            # Check for tool messages (results)
            if isinstance(latest_msg, ToolMessage):
                tool_results_collected["items"].append({
                    "name": latest_msg.name,
                    "content": latest_msg.content,
                    "tool_call_id": latest_msg.tool_call_id
                })
                # Stream tool result info
                with tool_results_placeholder.container():
                    with st.expander(":material/check_circle: Tool results", expanded=True):
                        for tr in tool_results_collected["items"]:
                            st.markdown(f"**{tr['name']}**")
                            st.code(tr["content"], language="text")

            # Extract model name
            if model_used["name"] is None:
                model_used["name"] = extract_model_name(latest_msg, {})

            # Yield text content for streaming display
            if hasattr(latest_msg, 'content') and latest_msg.content:
                yield latest_msg.content

    try:
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            response = st.write_stream(stream_chunks())
            if model_used["name"]:
                st.caption(f"Generated by `{model_used['name']}`")
    except Exception as e:
        response = None
        st.error(f"Something went wrong: {e}")
    finally:
        st.session_state.is_streaming = False

    if response is not None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "model": model_used["name"],
                "tool_calls": tool_calls_collected["items"] if tool_calls_collected["items"] else None,
                "tool_results": tool_results_collected["items"] if tool_results_collected["items"] else None
            }
        )

    # Make sure the sidebar list reflects this thread (it may have just been
    # created on the very first message, or simply updated in title).
    if st.session_state.thread_id not in st.session_state.threads:
        st.session_state.threads.append(st.session_state.thread_id)
        get_threads.clear()  # so the sidebar re-pulls from SQLite next rerun

    # The active thread's title likely changed — drop its cache entry.
    get_thread_title.clear(st.session_state.thread_id)

    st.rerun()