# RLM Extension — Recursive Language Model for pi

An extension that implements the Recursive Language Model approach from
[arXiv:2512.24601](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab;
MIT/Stanford, ICML 2025).

## What it does

Standard LLMs are limited by their context window — even large windows (128K–
272K tokens) can't handle truly massive inputs like full codebases, large log
files, or book-length documents.

RLM sidesteps this by treating the context as a **variable in a persistent
Python REPL** rather than feeding it into the prompt. The LLM writes Python
code to:

1. Inspect metadata (length, structure, first/last lines)
2. Slice and chunk the context programmatically
3. Call `llm_query(sub_context, instruction)` to recursively query the LLM on
   smaller pieces
4. Aggregate results and set `Final = "..."` when the answer is ready

This loop repeats until the LLM sets `Final` or hits the iteration limit.

## Architecture

```
index.ts      Extension entry point — registers rlm_query tool, handles rendering
rlm.ts        Core RLM loop — Algorithm 1 from the paper, event emission
repl.ts       Persistent Python subprocess manager (line-delimited JSON over stdio)
runtime.py    Python-side REPL (context variable, llm_query bridge, Final sentinel)
```

### How it works

Communication between TypeScript and Python uses **line-delimited JSON over
stdin/stdout**. The Python process stays alive across iterations, maintaining
state (variables, imports, intermediate results).

Two stdout channels exist in the Python runtime:
- `_real_stdout` — the actual pipe to TypeScript, used for JSON protocol messages
  (`llm_query` requests, `exec_done` results)
- `sys.stdout` — redirected to a `StringIO` buffer during code execution, so
  `print()` output is captured separately and fed back to the LLM

## Usage

Place this directory in your project's `.pi/extensions/rlm/` directory. The
tool becomes available as `rlm_query` in pi.

### Prerequisites

- Python 3 must be available as `python3` in your PATH.

### Tool parameters

| Parameter | Type   | Required | Description                                              |
|-----------|--------|----------|----------------------------------------------------------|
| `context` | string | no*      | The full text to process (can be arbitrarily large)      |
| `url`     | string | no*      | A URL or `file://` path to load content from             |
| `query`   | string | yes      | What to do with the context (summarize, extract, etc.)   |

\* At least one of `context` or `url` must be provided.

**`url` supports:**
- HTTP/HTTPS URLs — fetched automatically
- `file:///path/to/file` — reads a local file
- `file:///path/to/directory` — recursively reads `.ts`, `.js`, `.json`, `.md`,
  `.txt`, `.yaml`, `.yml` files (skips `node_modules`, `dist`, dotfiles)

### Example

```
rlm_query(
  url: "file:///path/to/codebase",
  query: "Summarize the architecture, key modules, and design patterns"
)
```

The RLM loop then:
1. Inspects the context size and structure
2. Splits it into manageable chunks
3. Queries the LLM on each chunk for a local summary
4. Aggregates the summaries into a final answer

### Trace output

While running, a compact status line shows the current step. When done, the
full execution trace is rendered inline showing:
- Each iteration's generated Python code (with line numbers)
- Sub-query details (instruction, sent/received sizes, timing)
- REPL output and errors
- Final result in a bordered box

## Configuration

Constants in `rlm.ts` control loop behavior:

| Constant                 | Default | Description                               |
|--------------------------|---------|-------------------------------------------|
| `MAX_ITERATIONS`         | 30      | Maximum outer loop iterations (paper default)           |
| `MAX_SUB_QUERIES`        | 500     | Safety cap on `llm_query()` calls (paper has no limit)  |
| `METADATA_PREVIEW_LINES` | 20      | Lines of context shown to LLM as metadata               |
| `MAX_STDOUT_CHARS`       | 20,000  | Maximum REPL output fed back to the LLM (paper default) |

## References

- Zhang, Kraska, Khattab. "Recursive Language Models." arXiv:2512.24601, 2025.
- [rLLM reference implementation](https://github.com/agentic-learning-ai-lab/rllm)
