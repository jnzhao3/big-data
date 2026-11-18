"""Timeline merging, the hook script, and the lockdown configuration."""

import json
import subprocess
import sys
from pathlib import Path

from cvm import HARNESS_DIR, timeline
from cvm.paths import SessionPaths

SID = "00000000-0000-4000-8000-000000000002"

# The built-in tool names listed in the Claude Code tools reference (code.claude.com/docs/en/tools-reference),
# checked 2026-11-18, plus older names. EndConversation cannot be denied by design; ToolSearch and
# WaitForMcpServers only load or wait for the matcher's own tools, so they stay available.
BUILT_IN = {"Agent", "Artifact", "AskUserQuestion", "Bash", "CronCreate", "CronDelete", "CronList", "Edit",
            "EnterPlanMode", "EnterWorktree", "ExitPlanMode", "ExitWorktree", "Glob", "Grep", "ListAgents",
            "ListMcpResourcesTool", "LSP", "Monitor", "NotebookEdit", "PowerShell", "PushNotification", "Read",
            "ReadMcpResourceTool", "RemoteTrigger", "ReportFindings", "ScheduleWakeup", "SendFeedback",
            "SendMessage", "SendUserFile", "ShareOnboardingGuide", "Skill", "SubagentHandback", "TaskCreate",
            "TaskGet", "TaskList", "TaskOutput", "TaskStop", "TaskUpdate", "TodoWrite", "WebFetch", "WebSearch",
            "Workflow", "Write", "MultiEdit", "LS", "NotebookRead", "Task", "BashOutput", "KillShell"}


def _jl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))


def test_timeline_merges_and_pairs(tmp_path):
    paths = SessionPaths(SID, tmp_path)
    _jl(paths.transcripts / "main.jsonl", [
        {"type": "user", "uuid": "u1", "timestamp": "2026-11-18T10:00:00.000Z",
         "message": {"role": "user", "content": "Start the task"}},
        {"type": "assistant", "uuid": "a1", "timestamp": "2026-11-18T10:00:01.000Z", "message": {
            "role": "assistant", "model": "m", "content": [
                {"type": "thinking", "thinking": "", "signature": "x"},
                {"type": "tool_use", "id": "toolu_1", "name": "mcp__matcher__list_values",
                 "input": {"column": "Race"}}]}},
        {"type": "user", "uuid": "u2", "timestamp": "2026-11-18T10:00:02.000Z", "message": {
            "role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "[...]"}]}},
        {"type": "assistant", "uuid": "a2", "timestamp": "2026-11-18T10:00:03.000Z", "message": {
            "role": "assistant", "content": [{"type": "thinking", "thinking": "I should list values"},
                                             {"type": "text", "text": "Done."}]}},
        {"type": "system", "subtype": "compact_boundary", "uuid": "s1", "timestamp": "2026-11-18T10:00:04.000Z"},
        {"type": "brand-new-entry-type", "uuid": "x1", "timestamp": "2026-11-18T10:00:05.000Z"},
        "{not json",
    ])
    _jl(paths.server_log, [
        {"ts": "2026-11-18T10:00:00.500Z", "event": "server_start"},
        {"ts": "2026-11-18T10:00:01.500Z", "event": "tool_call", "tool": "list_values", "arguments": {},
         "result_rows": 5, "result_bytes": 10, "duration_ms": 3, "error": None, "result": {}},
    ])
    _jl(paths.hook_events, [{"ts": "2026-11-18T10:00:06.000Z", "hook_event_name": "Stop"}])
    doc = timeline.build(paths, {"study": "clark"})
    kinds = [e["kind"] for e in doc["events"]]
    assert kinds == ["user_message", "server_log", "thinking", "tool_call", "server_log", "tool_result", "thinking",
                     "assistant_text", "compaction", "other", "hook_event"]
    call = next(e for e in doc["events"] if e["kind"] == "tool_call")
    res = next(e for e in doc["events"] if e["kind"] == "tool_result")
    srv = next(e for e in doc["events"] if e.get("tool") == "list_values" and e["source"] == "server")
    assert call["result_ref"] == res["id"] and call["server_ref"] == srv["id"] and srv["call_ref"] == call["id"]
    th = [e for e in doc["events"] if e["kind"] == "thinking"]
    assert [e["thinking_display"] for e in th] == ["redacted", "summary"]
    assert doc["summary"]["transcript"]["malformed_lines"] == 1
    assert doc["summary"]["tool_calls_by_tool"] == {"list_values": 1} and doc["summary"]["rows_returned"] == 5


def test_hook_archives_and_never_fails(tmp_path):
    hook = HARNESS_DIR / "hooks" / "archive_transcript.py"
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"user"}\n')
    env = {"MATCH_EVAL_LOG_ROOT": str(tmp_path / "logs"), "PATH": "/usr/bin:/bin"}
    for ev in ("Stop", "PreCompact", "SessionEnd"):
        p = subprocess.run([sys.executable, str(hook)], input=json.dumps(
            {"session_id": SID, "hook_event_name": ev, "transcript_path": str(t), "reason": "other"}),
            text=True, env=env, capture_output=True)
        assert p.returncode == 0, p.stderr
    paths = SessionPaths(SID, tmp_path / "logs")
    assert (paths.transcripts / "main.jsonl").read_text() == t.read_text()
    assert len(list(paths.transcripts.glob("precompact_*.jsonl"))) == 1
    assert [json.loads(x)["hook_event_name"] for x in paths.hook_events.read_text().splitlines()] == [
        "Stop", "PreCompact", "SessionEnd"]
    bad = subprocess.run([sys.executable, str(hook)], input="{broken", text=True, env=env, capture_output=True)
    assert bad.returncode == 0
    assert "JSONDecodeError" in (tmp_path / "logs" / "hooks_errors.log").read_text()


def test_lockdown_settings_template():
    s = json.loads((HARNESS_DIR / "eval_workspace" / ".claude" / "settings.json").read_text())
    perms = s["permissions"]
    assert perms["allow"] == ["mcp__matcher__*", "ToolSearch", "WaitForMcpServers"]
    assert BUILT_IN <= set(perms["deny"]), sorted(BUILT_IN - set(perms["deny"]))
    assert perms["defaultMode"] == "dontAsk"
    assert s["showThinkingSummaries"] is True and s["disableClaudeAiConnectors"] is True
    assert "cleanupPeriodDays" not in s
    assert set(s["hooks"]) == {"Stop", "SubagentStop", "PreCompact", "SessionEnd"}
    for groups in s["hooks"].values():
        assert groups[0]["hooks"][0]["command"] == "{{HOOK_COMMAND}}"
