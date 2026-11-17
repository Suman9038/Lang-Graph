"""Probe: what tables does SqliteSaver create, and does delete_thread clean them all?"""
import os
import sqlite3
import tempfile

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class S(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def node(state: S):
    return {"messages": []}


tmp = os.path.join(tempfile.gettempdir(), "_probe_saver.db")
if os.path.exists(tmp):
    os.remove(tmp)

conn = sqlite3.connect(tmp, check_same_thread=False)
saver = SqliteSaver(conn=conn)

g = StateGraph(S)
g.add_node("node", node)
g.add_edge(START, "node")
app = g.compile(checkpointer=saver)

app.invoke({"messages": []}, config={"configurable": {"thread_id": "t1"}})

tables = sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
print("tables after write:", tables)
for t in tables:
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
    print(f"  {t}: thread_id={'thread_id' in cols} cols={cols}")
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"     rows={n}")

conn.close()
os.remove(tmp)
