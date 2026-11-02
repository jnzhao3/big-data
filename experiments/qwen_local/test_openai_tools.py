"""Check Ollama's OpenAI-compatible endpoint: plain chat + tool calling, per model."""
import json, sys, time, urllib.request

URL = "http://localhost:11434/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "get_column_stats",
    "description": "Return statistics about a column in a table.",
    "parameters": {"type": "object", "properties": {
        "table": {"type": "string", "enum": ["source", "target"]},
        "column": {"type": "string"}}, "required": ["table", "column"]}}}]

def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer ollama"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r), time.time() - t

for model in sys.argv[1:]:
    d, dt = post({"model": model, "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 50})
    print(f"[{model}] chat {dt:.1f}s ->", repr(d["choices"][0]["message"]["content"][:80]))
    msgs = [{"role": "user", "content": "Look up the stats of column 'age_at_diagnosis' in the source table, then tell me how many nulls it has."}]
    d, dt = post({"model": model, "messages": msgs, "tools": TOOLS})
    m = d["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    print(f"[{model}] tools {dt:.1f}s finish={d['choices'][0]['finish_reason']} calls=",
          [(c["function"]["name"], c["function"]["arguments"]) for c in calls], "content=", repr((m.get("content") or "")[:80]))
    if calls:  # second turn: feed the tool result back
        msgs += [m, {"role": "tool", "tool_call_id": calls[0].get("id", ""), "content": json.dumps({"nulls": 42, "dtype": "float"})}]
        d, dt = post({"model": model, "messages": msgs, "tools": TOOLS})
        print(f"[{model}] after tool result {dt:.1f}s ->", repr((d["choices"][0]["message"].get("content") or "")[:120]))
