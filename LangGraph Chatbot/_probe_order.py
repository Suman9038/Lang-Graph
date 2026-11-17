"""Probe: ordering threads by rowid vs by thread_id (uuid4 random)."""
import os
import sqlite3
import tempfile
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class S(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def node(state: S):
    return {"messages": []}


tmp = os.path.join(tempfile.gettempdir(), "_probe_ord2.db")
if os.path.exists(tmp):
    os.remove(tmp)
conn = sqlite3.connect(tmp, check_same_thread=False)
app = StateGraph(S)
app.add_node("node", node)
app.add_edge(START, "node")
app = app.compile(checkpointer=SqliteSaver(conn=conn))

created = []
for _ in range(4):
    tid = str(uuid.uuid4())
    app.invoke({"messages": []}, config={"configurable": {"thread_id": tid}})
    created.append(tid)

print("creation order:", [t[:8] for t in created])
by_tid = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")]
by_rowid = [r[0] for r in conn.execute(
    "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC")]
print("ORDER BY thread_id :", [t[:8] for t in by_tid], "matches creation?",
      by_tid == list(reversed(created)))
print("ORDER BY MAX(rowid):", [t[:8] for t in by_rowid], "matches creation (newest first)?",
      by_rowid == list(reversed(created)))

conn.close()
os.remove(tmp)
