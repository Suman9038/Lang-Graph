from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3
import requests
import random

load_dotenv()

# tools

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression like "2 + 3 * 4" or "sqrt(16)" or "2**10"

    Returns:
        The result of the expression as a string.
    """
    import math

    # Allowed functions and constants
    allowed_names = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'pow': pow, 'sum': sum,
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10, 'exp': math.exp,
        'pi': math.pi, 'e': math.e,
        'ceil': math.ceil, 'floor': math.floor,
        'degrees': math.degrees, 'radians': math.radians,
    }

    try:
        # Safely evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_stock_price(symbol: str):
    """
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOG", "MSFT", "TCS")
        using alpha vantage with API key in the url
    Returns:
        The current stock price information as a human-readable string.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=JL20KU7U29PYDDR"
        r = requests.get(url)
        data = r.json()

        # Parse the Global Quote response
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            symbol = quote.get("01. symbol", "N/A")
            price = quote.get("05. price", "N/A")
            change = quote.get("09. change", "N/A")
            change_percent = quote.get("10. change percent", "N/A")
            prev_close = quote.get("08. previous close", "N/A")
            volume = quote.get("06. volume", "N/A")
            latest_day = quote.get("07. latest trading day", "N/A")

            return (
                f"Stock: {symbol}\n"
                f"Current Price: ${price}\n"
                f"Previous Close: ${prev_close}\n"
                f"Change: ${change} ({change_percent})\n"
                f"Volume: {volume}\n"
                f"Latest Trading Day: {latest_day}"
            )
        elif "Error Message" in data:
            return f"Error: {data['Error Message']}"
        elif "Note" in data:
            return f"API Limit: {data['Note']}"
        else:
            return f"Could not fetch data for {symbol}. Response: {data}"
    except Exception as e:
        return f"Error fetching stock price for {symbol.upper()}: {str(e)}"




llm1 = ChatOllama(model="nemotron-3-nano:30b-cloud")
llm2 = ChatGroq(model="llama-3.3-70b-versatile")
llm3 = ChatOllama(model="smollm2:360m")

# Bind tools to the models
tools = [calculator, get_stock_price, search_tool]
llm1_with_tools = llm1.bind_tools(tools)
llm2_with_tools = llm2.bind_tools(tools)
llm3_with_tools = llm3.bind_tools(tools)

final_model = llm1_with_tools.with_fallbacks(fallbacks=[llm2_with_tools, llm3_with_tools])


# -------------------- State --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = final_model.invoke(messages)
    return {"messages": [response]}


# -------------------- SQLite Checkpointer -------------------
# Persists every conversation into chatbot.db so threads survive page refreshes
# and Streamlit restarts.
connection = sqlite3.connect("chatbot_with_tools.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

# -------------------- Graph --------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
# if the llm asked for the tools_condition then the next node is tools node otherwise it will return to the chat node
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

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