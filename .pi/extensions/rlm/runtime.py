"""
RLM Runtime — Python-side helpers for the Recursive Language Model extension.

This module is injected into a persistent Python REPL subprocess. It provides:
  - `context`: the full prompt/document as a string variable
  - `llm_query(sub_context, instruction)`: bridge that sends a JSON request
    over stdout and reads the LLM response from stdin (synchronous from
    Python's perspective, async on the TypeScript side).
  - `Final`: a variable the LLM sets when the answer is ready.

Communication protocol (line-delimited JSON over stdio):
  -> stdout: {"type":"llm_query","sub_context":"...","instruction":"...","id":"..."}
  <- stdin:  {"type":"llm_result","id":"...","result":"..."}
  -> stdout: {"type":"exec_done","stdout":"...","stderr":"...","has_final":bool,"final_value":"..."|null}

All protocol I/O uses saved references to the original sys.stdout/sys.stdin
(_real_stdout, _real_stdin) so that exec'd code can freely redirect sys.stdout
for print() capture without breaking the JSON message channel.
"""

import json
import sys
import uuid
import io
import traceback

# Real stdio handles — saved before exec() can redirect sys.stdout/sys.stderr.
# llm_query() and protocol messages must always use these, not sys.stdout.
_real_stdout = sys.stdout
_real_stdin = sys.stdin

# Will be set by the TypeScript host before each execution
context: str = ""

# Sentinel — when the LLM sets this to a non-None value, the loop terminates
Final = None

# Counter for sub-queries within a single execution
_sub_query_count = 0

# Buffer for messages received during llm_query that belong to the main loop
_deferred_messages: list = []


def llm_query(sub_context: str, instruction: str) -> str:
    """Send a sub-context and instruction to the parent LLM and return the response."""
    global _sub_query_count
    _sub_query_count += 1
    request_id = uuid.uuid4().hex[:12]
    request = {
        "type": "llm_query",
        "sub_context": sub_context,
        "instruction": instruction,
        "id": request_id,
    }
    # Write request to the real stdout pipe (not the captured StringIO)
    _real_stdout.write(json.dumps(request) + "\n")
    _real_stdout.flush()

    # Block until the TypeScript host sends back the result on real stdin
    while True:
        line = _real_stdin.readline()
        if not line:
            raise RuntimeError("REPL stdin closed unexpectedly")
        line = line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if response.get("type") == "llm_result" and response.get("id") == request_id:
            return response.get("result", "")
        if response.get("type") == "shutdown":
            raise RuntimeError("Shutdown requested during llm_query")
        # Buffer any other messages for the main loop to process later
        _deferred_messages.append(response)


def _execute_code(code: str) -> None:
    """Execute a code snippet in the module's global scope, capturing output."""
    global Final
    # Track sub-queries made during this execution
    sub_queries_before = _sub_query_count

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        exec(code, globals())  # noqa: S102 — intentional exec for REPL
    except Exception:
        traceback.print_exc(file=captured_stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    stdout_val = captured_stdout.getvalue()
    stderr_val = captured_stderr.getvalue()

    # Calculate sub-queries made during this execution
    sub_query_count = _sub_query_count - sub_queries_before

    result = {
        "type": "exec_done",
        "stdout": stdout_val,
        "stderr": stderr_val,
        "has_final": Final is not None,
        "final_value": str(Final) if Final is not None else None,
        "sub_query_count": sub_query_count,
    }
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.flush()


def _dispatch(msg: dict) -> bool:
    """Handle a single message. Returns False if shutdown requested."""
    if msg.get("type") == "exec":
        _execute_code(msg.get("code", ""))
    elif msg.get("type") == "set_context":
        global context
        context = msg.get("value", "")
        _real_stdout.write(json.dumps({"type": "context_set"}) + "\n")
        _real_stdout.flush()
    elif msg.get("type") == "reset_final":
        global Final
        Final = None
        _real_stdout.write(json.dumps({"type": "final_reset"}) + "\n")
        _real_stdout.flush()
    elif msg.get("type") == "shutdown":
        return False
    return True


def _main_loop() -> None:
    """Read execution requests from stdin in a loop."""
    while True:
        # Drain any messages that were buffered during llm_query() calls
        while _deferred_messages:
            msg = _deferred_messages.pop(0)
            if not _dispatch(msg):
                return

        line = _real_stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not _dispatch(msg):
            break


if __name__ == "__main__":
    # Signal readiness
    ready = {"type": "ready"}
    _real_stdout.write(json.dumps(ready) + "\n")
    _real_stdout.flush()
    _main_loop()
