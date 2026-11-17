import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()
print("ALPHA_VINTAGE_API_KEY =", repr(os.getenv("ALPHA_VINTAGE_API_KEY")))
print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))

for mod in ("backend", "backend_with_database", "backend_with_tools", "backend_with_rag"):
    try:
        __import__(mod)
        print(f"OK   import {mod}")
    except Exception as exc:
        print(f"FAIL import {mod}: {type(exc).__name__}: {exc}")
