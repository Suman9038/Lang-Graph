"""Probe: (1) does chatbot_with_hitl import?  (2) does interrupt() inside a @tool work?"""
import sys, os
sys.path.insert(0, r"C:\CODER KA ADDA\LangGraph")

print("=== 1. import chatbot_with_hitl ===")
try:
    import chatbot_with_hitl
    print("OK import")
except Exception as exc:
    print(f"FAIL import: {type(exc).__name__}: {exc}")

print()
print("=== 2. dotenv provenance ===")
import dotenv
print("dotenv.__file__:", dotenv.__file__)
import importlib.metadata as md
for dist in md.distributions():
    name = (dist.metadata['Name'] or '')
    if 'dotenv' in name.lower():
        files = dist.files or []
        owners = [str(f) for f in files if str(f).startswith('dotenv/')]
        print(f"  {name} {dist.version} -> claims {len(owners)} files under dotenv/")
        for o in owners[:8]:
            print("      ", o)
