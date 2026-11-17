from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, BaseMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
import requests
from langchain_core.tools import tool

load_dotenv()

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

@tool
def purchase_stock(symbol: str, quantity: int)-> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision to confirm or cancel ("yes/ anything else).
    """
    # This pause the graph and returns control to the caller
    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )
    if isinstance(decision, str) and decision.lower() == "yes":
        return{
            "status": "success",
            "messages": f"✅ Purchased {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity":quantity
        }
    else:
        return {
            "status": "cancelled",
            "messages": f"❌ Purchase of {quantity} shares of {symbol} was cancelled.",
            "symbol": symbol,
            "quantity":quantity
        }


llm1 = ChatOllama(model="nemotron-3-nano:30b-cloud")
llm2 = ChatGroq(model="llama-3.3-70b-versatile")
llm3 = ChatOllama(model="smollm2:360m")

# Bind tools to the models
tools = [get_stock_price, purchase_stock]
llm1_with_tools = llm1.bind_tools(tools)
llm2_with_tools = llm2.bind_tools(tools)
llm3_with_tools = llm3.bind_tools(tools)

final_model = llm1_with_tools.with_fallbacks(fallbacks=[llm2_with_tools, llm3_with_tools])

# -------------------- State --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = final_model.invoke
    (messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
# if the llm asked for the tools_condition then the next node is tools node otherwise it will return to the chat node
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("GoodBye!")
            break

        state = {"messages":[HumanMessage(content=user_input)]}
        
        config = {"configurable": {"thread_id":thread_id}}
        result = chatbot.invoke(state,config=config)

        interrupts = result.get("__interrupt__",[])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            answer = input("Your decision (yes/no): ").lower().strip()

            # Resume the graph with human decsion
            result = chatbot.invoke(
                Command(resume=answer),
                config=config
            )
        
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}")
            