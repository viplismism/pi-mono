/**
 * RLM Loop — implements Algorithm 1 from "Recursive Language Models" (arXiv:2512.24601).
 *
 * The loop works as follows:
 *   1. Inject the full context into a persistent Python REPL as a variable.
 *   2. Send the LLM metadata about the context (length, first/last lines) plus
 *      the user's query. The LLM writes Python code that can:
 *        - Inspect/slice/filter `context`
 *        - Call `llm_query(sub_context, instruction)` to recursively query the
 *          LLM on sub-pieces of the context
 *        - Set `Final = "..."` when the answer is ready
 *   3. Execute the code in the REPL, capture stdout (which contains metadata
 *      the LLM can use in the next iteration).
 *   4. If `Final` is set, return it. Otherwise, feed stdout back into the LLM
 *      as the next observation and repeat from step 2.
 *
 * Sub-queries via `llm_query()` are handled by calling `completeSimple()` on
 * the TypeScript side, so they use the same model/provider as the parent call.
 */

import {
	type Api,
	type AssistantMessage,
	completeSimple,
	type Message,
	type Model,
	type TextContent,
	type UserMessage,
} from "@mariozechner/pi-ai";
import type { ExecResult, PythonRepl } from "./repl.js";

/** Maximum number of outer loop iterations before giving up (paper default: 30). */
const MAX_ITERATIONS = 30;

/**
 * Maximum number of recursive `llm_query()` calls per execution.
 * The paper does not cap sub-queries — this is a safety limit to prevent runaway costs.
 * Set high enough to never interfere with normal operation.
 */
const MAX_SUB_QUERIES = 500;

/** How many characters of context metadata to show the LLM. */
const METADATA_PREVIEW_LINES = 20;

/** Maximum characters of REPL stdout to feed back to the LLM (paper default: 20,000). */
const MAX_STDOUT_CHARS = 20_000;

export interface RlmOptions {
	context: string;
	query: string;
	model: Model<Api>;
	repl: PythonRepl;
	signal?: AbortSignal;
	/** API key to pass through to completeSimple for custom providers (e.g. grid). */
	apiKey?: string;
	/** Called with structured events as the RLM loop progresses. */
	onEvent?: (event: RlmEvent) => void;
}

/** Structured events emitted during the RLM loop. */
export type RlmEvent =
	| { type: "iteration_start"; iteration: number; maxIterations: number; totalSubQueries: number }
	| { type: "code_generated"; iteration: number; code: string; totalSubQueries: number }
	| {
			type: "sub_query_start";
			iteration: number;
			subQueryIndex: number;
			totalSubQueries: number;
			instruction: string;
			contextSize: number;
	  }
	| {
			type: "sub_query_done";
			iteration: number;
			subQueryIndex: number;
			totalSubQueries: number;
			instruction: string;
			contextSize: number;
			responseSize: number;
			responsePreview: string;
	  }
	| {
			type: "iteration_done";
			iteration: number;
			totalSubQueries: number;
			stdout: string;
			stderr: string;
			iterSubQueries: number;
			hasFinal: boolean;
	  }
	| { type: "error"; iteration: number; message: string; totalSubQueries: number };

export interface RlmResult {
	answer: string;
	iterations: number;
	totalSubQueries: number;
	completed: boolean;
}

function buildSystemPrompt(): string {
	return `You are a Recursive Language Model (RLM) agent. You process large contexts by writing Python code in a persistent REPL.

## Environment

- \`context\`: string variable holding the full input text (may be very large).
- \`llm_query(sub_context: str, instruction: str) -> str\`: sends a sub-piece to the LLM and returns the response. Use for summarization, extraction, classification, etc.
- \`Final\`: set this to your final answer string. The loop terminates immediately when set.
- **State persists** between iterations — variables, imports, and results carry over.
- Python 3 standard library is available.

## Strategy

1. **First iteration**: Always check \`len(context)\` and inspect structure (first/last lines, delimiters, file boundaries).
2. **Small context** (under ~10K chars): Process directly with Python or a single \`llm_query()\`. Set Final in one step.
3. **Medium context** (10K–100K chars): Split into logical chunks (by lines, paragraphs, file boundaries), \`llm_query()\` each, then aggregate.
4. **Large context** (over 100K chars): Use Python to extract structure first (headings, file paths, patterns), then targeted \`llm_query()\` calls on relevant sections only.
5. **Chunking**: 4K–8K chars per chunk works well for \`llm_query()\`. Overlap slightly at boundaries if continuity matters.
6. **Aggregation**: After collecting chunk results, use a final \`llm_query()\` to synthesize if needed, then set Final.

## Rules

1. Use \`print()\` to output intermediate results you'll see in the next iteration.
2. Be efficient — minimize \`llm_query()\` calls with smart chunking.
3. Do NOT set Final prematurely. If you need more iterations, print your state and continue.
4. Keep printed output concise. Focus on what you need for the next step.

## Format

Respond with ONLY a Python code block. No explanation before or after.

\`\`\`python
print(f"Context: {len(context)} chars")
print(context[:500])
\`\`\``;
}

function buildContextMetadata(context: string): string {
	const lines = context.split("\n");
	const charCount = context.length;
	const lineCount = lines.length;

	const previewStart = lines.slice(0, METADATA_PREVIEW_LINES).join("\n");
	const previewEnd = lines.slice(-METADATA_PREVIEW_LINES).join("\n");

	return [
		`Context statistics:`,
		`  - ${charCount.toLocaleString()} characters`,
		`  - ${lineCount.toLocaleString()} lines`,
		``,
		`First ${METADATA_PREVIEW_LINES} lines:`,
		previewStart,
		``,
		`Last ${METADATA_PREVIEW_LINES} lines:`,
		previewEnd,
	].join("\n");
}

function extractCodeFromResponse(response: AssistantMessage): string | null {
	for (const block of response.content) {
		if (block.type !== "text") continue;
		const text = (block as TextContent).text;

		const fenceMatch = text.match(/```(?:python)?\s*\n([\s\S]*?)```/);
		if (fenceMatch) return fenceMatch[1].trim();

		const trimmed = text.trim();
		if (
			trimmed &&
			!trimmed.startsWith("#") &&
			(trimmed.includes("=") ||
				trimmed.includes("print") ||
				trimmed.includes("import") ||
				trimmed.includes("for ") ||
				trimmed.includes("def "))
		) {
			return trimmed;
		}
	}
	return null;
}

function truncateStdout(stdout: string): string {
	if (stdout.length <= MAX_STDOUT_CHARS) return stdout;
	const half = Math.floor(MAX_STDOUT_CHARS / 2);
	return `${stdout.slice(0, half)}\n\n... [truncated ${(stdout.length - MAX_STDOUT_CHARS).toLocaleString()} characters] ...\n\n${stdout.slice(-half)}`;
}

export async function runRlmLoop(options: RlmOptions): Promise<RlmResult> {
	const { context, query, model, repl, signal, apiKey, onEvent } = options;

	await repl.setContext(context);
	await repl.resetFinal();

	let totalSubQueries = 0;
	let currentIteration = 0;

	// Set up the llm_query handler — emits sub-query events
	repl.setLlmQueryHandler(async (subContext: string, instruction: string) => {
		if (signal?.aborted) throw new Error("Aborted");
		if (totalSubQueries >= MAX_SUB_QUERIES) {
			return `[ERROR] Maximum sub-query limit (${MAX_SUB_QUERIES}) reached. Set Final with your best answer.`;
		}
		totalSubQueries++;
		const sqIndex = totalSubQueries;

		onEvent?.({
			type: "sub_query_start",
			iteration: currentIteration,
			subQueryIndex: sqIndex,
			totalSubQueries,
			instruction,
			contextSize: subContext.length,
		});

		for (let retry = 0; retry < 3; retry++) {
			try {
				const response = await completeSimple(
					model,
					{
						messages: [
							{
								role: "user",
								content: `Context:\n${subContext}\n\nInstruction: ${instruction}`,
								timestamp: Date.now(),
							},
						],
					},
					{ apiKey },
				);

				const textParts = response.content.filter((b): b is TextContent => b.type === "text").map((b) => b.text);
				const result = textParts.join("\n");

				onEvent?.({
					type: "sub_query_done",
					iteration: currentIteration,
					subQueryIndex: sqIndex,
					totalSubQueries,
					instruction,
					contextSize: subContext.length,
					responseSize: result.length,
					responsePreview: result.length > 300 ? `${result.slice(0, 300)}...` : result,
				});

				return result;
			} catch (err) {
				const errorMsg = err instanceof Error ? err.message : String(err);
				if (errorMsg.includes("overloaded") || errorMsg.includes("rate_limit") || errorMsg.includes("429")) {
					const waitTime = (retry + 1) * 2000;
					await new Promise((resolve) => setTimeout(resolve, waitTime));
					continue;
				}
				throw err;
			}
		}
		return "[Error] API still overloaded after retries";
	});

	const metadata = buildContextMetadata(context);
	const conversationHistory: Message[] = [
		{
			role: "user",
			content: `${metadata}\n\nQuery: ${query}`,
			timestamp: Date.now(),
		} satisfies UserMessage,
	];

	for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
		currentIteration = iteration;
		if (signal?.aborted) {
			return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
		}

		// Step 1: Ask the LLM to generate Python code
		onEvent?.({
			type: "iteration_start",
			iteration,
			maxIterations: MAX_ITERATIONS,
			totalSubQueries,
		});

		let response: AssistantMessage | undefined;
		let lastError: string | undefined;
		for (let retry = 0; retry < 3; retry++) {
			try {
				response = await completeSimple(
					model,
					{
						systemPrompt: buildSystemPrompt(),
						messages: conversationHistory,
					},
					{ apiKey },
				);
				break;
			} catch (err) {
				const errorMsg = err instanceof Error ? err.message : String(err);
				lastError = errorMsg;
				if (errorMsg.includes("overloaded") || errorMsg.includes("rate_limit") || errorMsg.includes("429")) {
					const waitTime = (retry + 1) * 2000;
					onEvent?.({
						type: "error",
						iteration,
						message: `API rate limited, retrying in ${waitTime}ms...`,
						totalSubQueries,
					});
					await new Promise((resolve) => setTimeout(resolve, waitTime));
					continue;
				}
				throw err;
			}
		}

		if (!response) {
			return {
				answer: `[API Error] Failed after 3 retries. Last error: ${lastError || "Unknown error"}`,
				iterations: iteration,
				totalSubQueries,
				completed: false,
			};
		}

		const code = extractCodeFromResponse(response);
		if (!code) {
			const textContent = response.content
				.filter((b): b is TextContent => b.type === "text")
				.map((b) => b.text)
				.join("\n");

			return {
				answer: textContent || "[No code or answer produced]",
				iterations: iteration,
				totalSubQueries,
				completed: !!textContent,
			};
		}

		// Step 2: Emit the code, then execute
		onEvent?.({
			type: "code_generated",
			iteration,
			code,
			totalSubQueries,
		});

		conversationHistory.push(response);

		let execResult: ExecResult;
		try {
			execResult = await repl.execute(code);
		} catch (err) {
			const errorMsg = err instanceof Error ? err.message : String(err);
			onEvent?.({
				type: "error",
				iteration,
				message: `Execution error: ${errorMsg}`,
				totalSubQueries,
			});
			conversationHistory.push({
				role: "user",
				content: `Execution error: ${errorMsg}\n\nPlease fix the code and try again.`,
				timestamp: Date.now(),
			});
			continue;
		}

		// Step 3: Emit iteration results
		onEvent?.({
			type: "iteration_done",
			iteration,
			totalSubQueries,
			stdout: execResult.stdout,
			stderr: execResult.stderr,
			iterSubQueries: execResult.subQueryCount,
			hasFinal: execResult.hasFinal,
		});

		if (execResult.hasFinal && execResult.finalValue !== null) {
			return {
				answer: execResult.finalValue,
				iterations: iteration,
				totalSubQueries,
				completed: true,
			};
		}

		// Step 4: Feed output back to LLM
		const parts: string[] = [];
		if (execResult.stdout) {
			parts.push(`REPL output:\n${truncateStdout(execResult.stdout)}`);
		}
		if (execResult.stderr) {
			parts.push(`REPL stderr:\n${execResult.stderr.slice(0, 5000)}`);
		}
		if (parts.length === 0) {
			parts.push("(No output produced. The code ran without printing anything.)");
		}
		parts.push(
			`\nIteration ${iteration}/${MAX_ITERATIONS}. Sub-queries used: ${totalSubQueries}/${MAX_SUB_QUERIES}.`,
		);
		parts.push("Continue processing or set Final when you have the answer.");

		conversationHistory.push({
			role: "user",
			content: parts.join("\n\n"),
			timestamp: Date.now(),
		});
	}

	return {
		answer: "[Maximum iterations reached without setting Final]",
		iterations: MAX_ITERATIONS,
		totalSubQueries,
		completed: false,
	};
}
