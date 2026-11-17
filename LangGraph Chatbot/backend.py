from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import operator
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm1 = ChatOllama(model= "nemotron-3-nano:30b-cloud")
llm2= ChatGroq(model="llama-3.3-70b-versatile")
llm3 = ChatOllama(
    model="smollm2:360m"
)

final_model = llm1.with_fallbacks(fallbacks=[llm2,llm3])

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]

    response = final_model.invoke(messages)

    return{
        "messages": [response]
    }

# Checkpointer instance
checkpointer = InMemorySaver()

# Graph Creation
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge( "chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
