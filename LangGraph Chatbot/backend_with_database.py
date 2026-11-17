from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

llm1 = ChatOllama(model="nemotron-3-nano:30b-cloud")
llm2 = ChatGroq(model="llama-3.3-70b-versatile")
llm3 = ChatOllama(model="smollm2:360m")

final_model = llm1.with_fallbacks(fallbacks=[llm2, llm3])


# -------------------- State --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = final_model.invoke(messages)
    return {"messages": [response]}


# -------------------- SQLite Checkpointer --------------------
# Persists every conversation into chatbot.db so threads survive page refreshes
# and Streamlit restarts.
connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

# -------------------- Graph --------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------- Thread helpers --------------------
def retrieve_all_threads() -> list[str]:
    """Return every thread_id that has a checkpoint stored in SQLite.

    We look at the LangGraph checkpoints table directly so we don't have to
    load every thread's full state. The table is internal to langgraph, but
    the column name `thread_id` is part of its stable schema.
    """
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception:
        # Table may not exist yet (no chats have been persisted)
        return []


def delete_thread(thread_id: str) -> None:
    """Remove every checkpoint row that belongs to the given thread."""
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        connection.commit()
    except Exception as e:
        print(f"Failed to delete thread {thread_id}: {e}")


def load_thread_messages(thread_id: str) -> list[dict]:
    """Rehydrate a thread's chat history from the checkpointer as plain dicts
    the frontend can render directly."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = chatbot.get_state(config)
        messages = state.values.get("messages", []) if state else []
    except Exception:
        return []

    history = []
    for msg in messages:
        # LangChain messages expose `type` ("human"/"ai"/...) and string content
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            role = "user"
        elif msg_type == "ai":
            role = "assistant"
        else:
            # Skip system / tool messages in the UI
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        history.append(
            {
                "role": role,
                "content": content,
                "model": getattr(msg, "response_metadata", {}).get("model_name")
                or getattr(msg, "response_metadata", {}).get("model"),
            }
        )
    return history


def thread_title(thread_id: str) -> str:
    """Return a short, human-friendly label for a thread.

    Fa lls back to the first user message; if the thread is empty we just show
    the truncated UUID.
    """
    history = load_thread_messages(thread_id)
    for msg in history:
        if msg["role"] == "user":
            label = msg["content"].strip().replace("\n", " ")
            return label[:40] + ("…" if len(label) > 40 else "")
    return f"Thread {thread_id[:8]}"