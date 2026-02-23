/**
 * Persistent Python REPL manager for the RLM extension.
 *
 * Spawns a single Python subprocess running `runtime.py` and keeps it alive
 * across multiple RLM iterations. Communication uses line-delimited JSON
 * over stdin/stdout.
 *
 * The REPL provides:
 *   - `setContext(text)`:  inject the full prompt as the `context` variable
 *   - `execute(code)`:    run a code snippet and return stdout/stderr/Final
 *   - `resetFinal()`:     clear the Final sentinel between queries
 *   - `handleLlmQuery()`: callback for bridging `llm_query()` calls back
 *                          to the TypeScript LLM layer
 *   - `shutdown()`:       gracefully terminate the subprocess
 */

import { type ChildProcess, spawn } from "node:child_process";
import * as path from "node:path";
import * as readline from "node:readline";

/** Result of executing a code snippet in the REPL. */
export interface ExecResult {
	stdout: string;
	stderr: string;
	hasFinal: boolean;
	finalValue: string | null;
	/** Number of llm_query calls made in this execution */
	subQueryCount: number;
}

/** Callback the host provides to handle llm_query() calls from Python. */
export type LlmQueryHandler = (subContext: string, instruction: string) => Promise<string>;

interface ReadyMessage {
	type: "ready";
}

interface ExecDoneMessage {
	type: "exec_done";
	stdout: string;
	stderr: string;
	has_final: boolean;
	final_value: string | null;
	sub_query_count: number;
}

interface LlmQueryMessage {
	type: "llm_query";
	sub_context: string;
	instruction: string;
	id: string;
}

interface ContextSetMessage {
	type: "context_set";
}

interface FinalResetMessage {
	type: "final_reset";
}

type InboundMessage = ReadyMessage | ExecDoneMessage | LlmQueryMessage | ContextSetMessage | FinalResetMessage;

export class PythonRepl {
	private proc: ChildProcess | null = null;
	private rl: readline.Interface | null = null;
	private llmQueryHandler: LlmQueryHandler | null = null;
	private abortHandler: (() => void) | null = null;
	private abortSignal: AbortSignal | null = null;

	/**
	 * Pending resolvers for messages we're waiting on from Python.
	 * Each entry maps a message type to resolve/reject + timeout cleanup.
	 */
	private pending: Map<
		string,
		{ resolve: (msg: InboundMessage) => void; reject: (err: Error) => void; timeout: ReturnType<typeof setTimeout> }
	> = new Map();

	/** Whether the REPL subprocess is alive. */
	get isAlive(): boolean {
		return this.proc !== null && this.proc.exitCode === null;
	}

	/**
	 * Start the Python subprocess and wait for it to signal readiness.
	 * Throws if Python is not available or the runtime fails to start.
	 */
	async start(signal?: AbortSignal): Promise<void> {
		if (this.isAlive) return;

		const runtimePath = path.join(path.dirname(new URL(import.meta.url).pathname), "runtime.py");

		this.proc = spawn("python3", [runtimePath], {
			stdio: ["pipe", "pipe", "pipe"],
			env: { ...process.env, PYTHONUNBUFFERED: "1" },
		});

		this.rl = readline.createInterface({ input: this.proc.stdout! });
		this.rl.on("line", (line: string) => this.handleLine(line));

		// Collect stderr for diagnostics
		this.proc.stderr!.on("data", (chunk: Buffer) => {
			const text = chunk.toString();
			if (text.trim()) {
				process.stderr.write(`[rlm-repl-python] ${text}`);
			}
		});

		this.proc.on("close", () => {
			this.cleanup();
		});

		if (signal) {
			this.abortHandler = () => this.shutdown();
			this.abortSignal = signal;
			signal.addEventListener("abort", this.abortHandler, { once: true });
		}

		// Wait for the "ready" message
		await this.waitForMessage("ready");
	}

	/** Register the callback that handles llm_query() calls from Python. */
	setLlmQueryHandler(handler: LlmQueryHandler): void {
		this.llmQueryHandler = handler;
	}

	/** Inject the full context string into the Python REPL. */
	async setContext(text: string): Promise<void> {
		this.send({ type: "set_context", value: text });
		await this.waitForMessage("context_set");
	}

	/** Reset the Final sentinel variable. */
	async resetFinal(): Promise<void> {
		this.send({ type: "reset_final" });
		await this.waitForMessage("final_reset");
	}

	/**
	 * Execute a code snippet and return the result.
	 *
	 * If the code calls `llm_query()`, those calls are bridged back to the
	 * TypeScript LLM layer via the registered handler. This means `execute()`
	 * may take arbitrarily long depending on how many sub-queries the code
	 * issues.
	 */
	async execute(code: string): Promise<ExecResult> {
		this.send({ type: "exec", code });
		const msg = (await this.waitForMessage("exec_done")) as ExecDoneMessage;
		return {
			stdout: msg.stdout,
			stderr: msg.stderr,
			hasFinal: msg.has_final,
			finalValue: msg.final_value,
			subQueryCount: msg.sub_query_count ?? 0,
		};
	}

	/** Gracefully shut down the Python subprocess. */
	shutdown(): void {
		if (this.proc && this.proc.exitCode === null) {
			try {
				this.send({ type: "shutdown" });
			} catch {
				// stdin may already be closed
			}
			this.proc.kill("SIGTERM");
		}
		this.cleanup();
	}

	private send(msg: Record<string, unknown>): void {
		if (!this.proc || !this.proc.stdin || this.proc.stdin.destroyed) {
			throw new Error("REPL subprocess is not running");
		}
		this.proc.stdin.write(`${JSON.stringify(msg)}\n`);
	}

	private handleLine(line: string): void {
		const trimmed = line.trim();
		if (!trimmed) return;

		let msg: InboundMessage;
		try {
			msg = JSON.parse(trimmed) as InboundMessage;
		} catch {
			return;
		}

		// Handle llm_query asynchronously — Python is blocking on stdin
		if (msg.type === "llm_query") {
			this.handleLlmQueryMessage(msg as LlmQueryMessage);
			return;
		}

		// Check if anyone is waiting for this message type
		const entry = this.pending.get(msg.type);
		if (entry) {
			this.pending.delete(msg.type);
			clearTimeout(entry.timeout);
			entry.resolve(msg);
		}
	}

	private async handleLlmQueryMessage(msg: LlmQueryMessage): Promise<void> {
		if (!this.llmQueryHandler) {
			// No handler — return an error string to Python
			this.send({
				type: "llm_result",
				id: msg.id,
				result: "[ERROR] No LLM query handler registered",
			});
			return;
		}

		try {
			const result = await this.llmQueryHandler(msg.sub_context, msg.instruction);
			this.send({ type: "llm_result", id: msg.id, result });
		} catch (err) {
			const errorText = err instanceof Error ? err.message : String(err);
			this.send({
				type: "llm_result",
				id: msg.id,
				result: `[ERROR] LLM query failed: ${errorText}`,
			});
		}
	}

	private waitForMessage(type: string): Promise<InboundMessage> {
		return new Promise((resolve, reject) => {
			if (!this.isAlive) {
				reject(new Error(`REPL subprocess is not running (waiting for "${type}")`));
				return;
			}

			const timeout = setTimeout(() => {
				if (this.pending.has(type)) {
					this.pending.delete(type);
					reject(new Error(`Timeout waiting for "${type}" from Python REPL`));
				}
			}, 300_000); // 5 minutes — llm_query chains can be slow

			this.pending.set(type, { resolve, reject, timeout });
		});
	}

	private cleanup(): void {
		this.rl?.close();
		this.rl = null;
		this.proc = null;
		// Remove abort listener so this instance can be GC'd
		if (this.abortHandler && this.abortSignal) {
			this.abortSignal.removeEventListener("abort", this.abortHandler);
			this.abortHandler = null;
			this.abortSignal = null;
		}
		// Reject all pending promises so callers fail fast instead of hanging
		for (const [type, entry] of this.pending) {
			clearTimeout(entry.timeout);
			entry.reject(new Error(`REPL process died (waiting for "${type}")`));
		}
		this.pending.clear();
	}
}
