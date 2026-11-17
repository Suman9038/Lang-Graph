"""Probe: (1) is rag_tool reachable from the graph the RAG frontend actually uses?
          (2) is checkpoint_id time-ordered (can we sort threads by recency)?
          (3) does the `dotenv` shim vs python-dotenv matter?"""
import sys, os, uuid, sqlite3, tempfile
sys.path.insert(0, r"C:\CODER KA ADDA\LangGraph\LangGraph Chatbot")

print("=== 1. tools bound in each backend's graph ===")
import backend_with_tools as bt
import backend_with_rag as br
print("backend_with_tools tools:", [t.name for t in bt.tools])
print("backend_with_rag   tools:", [t.name for t in br.tools])
print("RAG tool present in tools-graph (used by frontend_with_rag)?",
      "rag_tool" in [t.name for t in bt.tools])
print("rag_tool imported by frontend_with_rag from backend_with_rag, but the")
print("chatbot it drives comes from backend_with_tools -> rag_tool unreachable.")

print()
print("=== 2. checkpoint_id ordering ===")
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class S(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def node(state: S):
    return {"messages": []}

tmp = os.path.join(tempfile.gettempdir(), "_probe_order.db")
if os.path.exists(tmp): os.remove(tmp)
conn = sqlite3.connect(tmp, check_same_thread=False)
app = StateGraph(S)
app.add_node("node", node)
app.add_edge(START, "node")
app = app.compile(checkpointer=SqliteSaver(conn=conn))

ids = []
for i in range(3):
    tid = str(uuid.uuid4())
    app.invoke({"messages": []}, config={"configurable": {"thread_id": tid}})
    row = conn.execute(
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? ORDER BY rowid DESC LIMIT 1", (tid,)
    ).fetchone()
    ids.append((tid[:8], row[0]))
for t, cid in ids:
    print(f"  thread {t}  checkpoint_id {cid}")
print("checkpoint_id is uuid7-style (time-ordered)?",
      all(uuid.UUID(c).version == 7 for _, c in ids))
conn.close(); os.remove(tmp)
