/** RLM Extension — registers `rlm_query` tool (arXiv:2512.24601). */

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { getMarkdownTheme } from "@mariozechner/pi-coding-agent";
import { Container, Markdown, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import { PythonRepl } from "./repl.js";
import { type RlmEvent, type RlmResult, runRlmLoop } from "./rlm.js";

interface IterationSummary {
	step: number;
	status: "running" | "done" | "error";
	subQueries: number;
	elapsed?: string;
	/** What's happening right now (only for the running step) */
	activity?: string;
}

interface RlmDetails {
	iterations: number;
	totalSubQueries: number;
	completed: boolean;
	queryPreview: string;
	contextLength: number;
	trace?: string;
	/** Brief status for collapsed view */
	liveStatus?: string;
	/** Per-iteration summaries for the compact progress view */
	iterationSummaries?: IterationSummary[];
	/** Path to the full untruncated trace log on disk */
	traceLogPath?: string;
}

const RlmParams = Type.Object({
	context: Type.Optional(
		Type.String({
			description: "The full context to process (document, code, data, etc.). Can be arbitrarily large.",
		}),
	),
	url: Type.Optional(
		Type.String({
			description:
				"A URL to fetch content from. The fetched text becomes the context. Use instead of context for web pages, raw files, etc.",
		}),
	),
	query: Type.String({
		description: "What to do with the context — summarize, extract, analyze, search, etc.",
	}),
});

function fmtSize(n: number): string {
	if (n > 1_000_000) return `${(n / 1_000_000).toFixed(1)}M chars`;
	if (n > 1_000) return `${(n / 1_000).toFixed(1)}K chars`;
	return `${n} chars`;
}

function fmtElapsed(ms: number): string {
	if (ms < 1000) return `${ms}ms`;
	return `${(ms / 1000).toFixed(1)}s`;
}

const LINE = "──────────────────────────────────────────────────────────────────────────";

/** Format a block with a left border: ╭ title ──╮ / │ lines / ╰────╯ */
function formatBlock(title: string, content: string): string[] {
	const lines: string[] = [];
	const contentLines = content.split("\n");

	lines.push(`  ╭ ${title} ${"─".repeat(Math.max(0, 68 - title.length))}╮`);
	for (const line of contentLines) {
		lines.push(`  │ ${line}`);
	}
	lines.push(`  ╰${"─".repeat(72)}╯`);
	return lines;
}

/** Format code with line numbers */
function formatCode(code: string): string[] {
	const codeLines = code.split("\n");
	const numWidth = String(codeLines.length).length;

	const lines: string[] = [];
	lines.push(`  ╭ Code ${"─".repeat(65)}╮`);
	for (let i = 0; i < codeLines.length; i++) {
		const num = String(i + 1).padStart(numWidth, " ");
		lines.push(`  │ ${num} ${codeLines[i]}`);
	}
	lines.push(`  ╰${"─".repeat(72)}╯`);
	return lines;
}

export default function rlmExtension(pi: ExtensionAPI): void {
	let repl: PythonRepl | null = null;
	let activeLock: Promise<void> | null = null;

	pi.on("session_shutdown", () => {
		if (repl) {
			repl.shutdown();
			repl = null;
		}
	});

	pi.registerTool<typeof RlmParams, RlmDetails>({
		name: "rlm_query",
		label: "RLM Query",
		description: [
			"Process contexts too large for a single prompt — entire codebases, massive logs, long documents, or data spanning many files.",
			"The context is kept external and explored programmatically via a Python REPL; the LLM never ingests it all at once.",
			"Accepts a file:// URL (file or directory), an HTTP/HTTPS URL, or inline text as context.",
			"USE when: you need to read/analyze 5+ files, the total content exceeds ~50K characters, or the task requires cross-file reasoning (architecture analysis, repo-wide refactors, pattern extraction, log aggregation).",
			"DO NOT use when: the task involves a single file, a small edit, or content that fits comfortably in one read() call.",
		].join(" "),
		parameters: RlmParams,

		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			// Serialize access — only one rlm_query can use the REPL at a time
			while (activeLock) {
				await activeLock;
			}
			let releaseLock!: () => void;
			activeLock = new Promise<void>((resolve) => {
				releaseLock = resolve;
			});

			try {
				const { query } = params;

				const traceLines: string[] = [];
				let liveStatus = "";
				let iterStartTime = 0;
				let iterSubQueryCount = 0;
				let sqStartTime = 0;
				const iterationSummaries: IterationSummary[] = [];

				// Emit helpers — throttled variant prevents rapid re-renders during
				// sub-query bursts which cause the terminal scroll to jump.
				let throttleTimer: ReturnType<typeof setTimeout> | null = null;
				let pendingEmit: (() => void) | null = null;
				const THROTTLE_MS = 250;

				const doEmit = (iterations: number, totalSubQueries: number) => {
					if (!onUpdate) return;
					onUpdate({
						content: [{ type: "text", text: liveStatus }],
						details: {
							iterations,
							totalSubQueries,
							completed: false,
							queryPreview: query.slice(0, 100),
							contextLength: context?.length ?? 0,
							trace: traceLines.join("\n"),
							liveStatus,
							iterationSummaries: [...iterationSummaries],
						},
					});
				};

				/** Emit immediately — use for key milestones (iteration start/done). */
				const emit = (iterations: number, totalSubQueries: number) => {
					if (throttleTimer) {
						clearTimeout(throttleTimer);
						throttleTimer = null;
						pendingEmit = null;
					}
					doEmit(iterations, totalSubQueries);
				};

				/** Throttled emit — use for rapid-fire events (sub-query start/done). */
				const emitThrottled = (iterations: number, totalSubQueries: number) => {
					pendingEmit = () => doEmit(iterations, totalSubQueries);
					if (!throttleTimer) {
						throttleTimer = setTimeout(() => {
							throttleTimer = null;
							if (pendingEmit) {
								pendingEmit();
								pendingEmit = null;
							}
						}, THROTTLE_MS);
					}
				};

				// Resolve context
				let context: string;
				if (params.url) {
					try {
						if (params.url.startsWith("file://")) {
							const filePath = decodeURIComponent(new URL(params.url).pathname);

							const stat = await fs.stat(filePath);
							if (stat.isDirectory()) {
								const files: string[] = [];
								const extensions = [".ts", ".js", ".json", ".md", ".txt", ".yaml", ".yml"];

								async function walkDir(dir: string) {
									const entries = await fs.readdir(dir, { withFileTypes: true });
									for (const entry of entries) {
										const fullPath = path.join(dir, entry.name);
										if (
											entry.isDirectory() &&
											!entry.name.startsWith(".") &&
											entry.name !== "node_modules" &&
											entry.name !== "dist"
										) {
											await walkDir(fullPath);
										} else if (entry.isFile() && extensions.some((ext) => entry.name.endsWith(ext))) {
											files.push(fullPath);
										}
									}
								}

								await walkDir(filePath);
								traceLines.push(`  Loaded ${files.length} files from: ${filePath}`);

								const contents: string[] = [];
								for (const file of files) {
									const relativePath = path.relative(filePath, file);
									const content = await fs.readFile(file, "utf-8");
									contents.push(`// ${relativePath}\n\n${content}`);
								}
								context = contents.join("\n\n");
							} else {
								context = await fs.readFile(filePath, "utf-8");
								traceLines.push(`  Loaded file: ${filePath}`);
							}
						} else {
							const response = await fetch(params.url, { signal });
							if (!response.ok) {
								return {
									content: [
										{
											type: "text" as const,
											text: `Failed to fetch URL: ${response.status} ${response.statusText}`,
										},
									],
									details: {
										iterations: 0,
										totalSubQueries: 0,
										completed: false,
										queryPreview: query.slice(0, 100),
										contextLength: 0,
										trace: "",
									},
								};
							}
							context = await response.text();
							traceLines.push(`  Fetched: ${params.url} (${fmtSize(context.length)})`);
						}
					} catch (err) {
						const msg = err instanceof Error ? err.message : String(err);
						return {
							content: [{ type: "text" as const, text: `Failed to load context: ${msg}` }],
							details: {
								iterations: 0,
								totalSubQueries: 0,
								completed: false,
								queryPreview: query.slice(0, 100),
								contextLength: 0,
								trace: "",
							},
						};
					}
				} else if (params.context) {
					context = params.context;
					traceLines.push(`  Inline context: ${fmtSize(context.length)}`);
				} else {
					return {
						content: [{ type: "text" as const, text: "Either 'context' or 'url' must be provided." }],
						details: {
							iterations: 0,
							totalSubQueries: 0,
							completed: false,
							queryPreview: query.slice(0, 100),
							contextLength: 0,
							trace: "",
						},
					};
				}

				if (!repl || !repl.isAlive) {
					repl = new PythonRepl();
					try {
						await repl.start(signal);
					} catch (err) {
						const errorMsg = err instanceof Error ? err.message : String(err);
						return {
							content: [
								{
									type: "text",
									text: `Failed to start Python REPL: ${errorMsg}. Ensure python3 is available.`,
								},
							],
							details: {
								iterations: 0,
								totalSubQueries: 0,
								completed: false,
								queryPreview: query.slice(0, 100),
								contextLength: context.length,
								trace: "",
							},
						};
					}
				}

				const model = ctx.model;
				if (!model) {
					return {
						content: [{ type: "text", text: "No model configured. Please set a model first." }],
						details: {
							iterations: 0,
							totalSubQueries: 0,
							completed: false,
							queryPreview: query.slice(0, 100),
							contextLength: context.length,
							trace: "",
						},
					};
				}

				const handleEvent = (event: RlmEvent) => {
					switch (event.type) {
						case "iteration_start": {
							iterStartTime = Date.now();
							iterSubQueryCount = 0;
							if (event.iteration > 1) traceLines.push("");
							traceLines.push(`  Step ${event.iteration}/${event.maxIterations}`);
							traceLines.push(LINE);
							liveStatus = `Step ${event.iteration}/${event.maxIterations} — generating code...`;
							iterationSummaries.push({
								step: event.iteration,
								status: "running",
								subQueries: 0,
								activity: "generating code...",
							});
							emit(event.iteration, event.totalSubQueries);
							break;
						}
						case "code_generated": {
							traceLines.push(...formatCode(event.code));
							liveStatus = `Step ${event.iteration} — executing...`;
							const current = iterationSummaries[iterationSummaries.length - 1];
							if (current) current.activity = "executing...";
							// No emit — trace builds in memory, visible on Ctrl+O or at iteration_done
							break;
						}
						case "sub_query_start": {
							iterSubQueryCount++;
							sqStartTime = Date.now();
							traceLines.push(`  ┌─ Sub-query #${event.subQueryIndex}  sending ${fmtSize(event.contextSize)}`);
							traceLines.push(`  │`);
							traceLines.push(`  │  ${event.instruction}`);
							liveStatus = `Step ${event.iteration} — sub-query #${iterSubQueryCount}...`;
							const cur = iterationSummaries[iterationSummaries.length - 1];
							if (cur) {
								cur.subQueries = iterSubQueryCount;
								cur.activity = `sub-query #${iterSubQueryCount}...`;
							}
							emitThrottled(event.iteration, event.totalSubQueries);
							break;
						}
						case "sub_query_done": {
							const sqElapsed = fmtElapsed(Date.now() - sqStartTime);
							const respLines = event.responsePreview.split("\n");
							traceLines.push(`  │`);
							for (const line of respLines) {
								traceLines.push(`  │  → ${line}`);
							}
							traceLines.push(`  └── ${sqElapsed} · ${fmtSize(event.responseSize)} received`);
							liveStatus = `Step ${event.iteration} — sub-query #${iterSubQueryCount} done (${sqElapsed})`;
							const curDone = iterationSummaries[iterationSummaries.length - 1];
							if (curDone) curDone.activity = `sub-query #${iterSubQueryCount} done`;
							emitThrottled(event.iteration, event.totalSubQueries);
							break;
						}
						case "iteration_done": {
							const elapsed = fmtElapsed(Date.now() - iterStartTime);

							if (event.stdout) {
								traceLines.push(...formatBlock("Output", event.stdout));
							}

							if (event.stderr) {
								traceLines.push(...formatBlock("Errors", event.stderr));
							}

							const sqNote =
								iterSubQueryCount > 0
									? ` · ${iterSubQueryCount} sub-quer${iterSubQueryCount !== 1 ? "ies" : "y"}`
									: "";
							traceLines.push(`  ${elapsed}${sqNote}`);
							traceLines.push("");

							liveStatus = `Step ${event.iteration} done — ${elapsed}${sqNote}`;
							const curIter = iterationSummaries[iterationSummaries.length - 1];
							if (curIter) {
								curIter.status = "done";
								curIter.elapsed = elapsed;
								curIter.subQueries = iterSubQueryCount;
								curIter.activity = undefined;
							}
							emit(event.iteration, event.totalSubQueries);
							break;
						}
						case "error": {
							traceLines.push(`  ⚠ ${event.message}`);
							liveStatus = event.message;
							const curErr = iterationSummaries[iterationSummaries.length - 1];
							if (curErr) {
								curErr.status = "error";
								curErr.activity = event.message;
							}
							emit(event.iteration, event.totalSubQueries);
							break;
						}
					}
				};

				let result: RlmResult;
				try {
					result = await runRlmLoop({
						context,
						query,
						model,
						repl,
						signal,
						onEvent: handleEvent,
					});
				} catch (err) {
					const errorMsg = err instanceof Error ? err.message : String(err);
					return {
						content: [{ type: "text", text: `RLM loop failed: ${errorMsg}` }],
						details: {
							iterations: 0,
							totalSubQueries: 0,
							completed: false,
							queryPreview: query.slice(0, 100),
							contextLength: context.length,
							trace: traceLines.join("\n"),
						},
					};
				}

				// Flush any pending throttled emit
				if (throttleTimer) {
					clearTimeout(throttleTimer);
					throttleTimer = null;
					pendingEmit = null;
				}

				// Write full trace to disk so it's always reviewable
				const fullTrace = traceLines.join("\n");
				let traceLogPath: string | undefined;
				try {
					const logDir = path.join(os.tmpdir(), "rlm-traces");
					await fs.mkdir(logDir, { recursive: true });
					const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
					traceLogPath = path.join(logDir, `rlm-${timestamp}.log`);
					const logHeader = [
						`RLM Query Trace`,
						`Date: ${new Date().toISOString()}`,
						`Query: ${query}`,
						`Context: ${fmtSize(context.length)}`,
						`Iterations: ${result.iterations}`,
						`Sub-queries: ${result.totalSubQueries}`,
						`Completed: ${result.completed}`,
						`${"─".repeat(74)}`,
						"",
					].join("\n");
					await fs.writeFile(traceLogPath, `${logHeader}${fullTrace}\n\n── Answer ──\n\n${result.answer}\n`);
				} catch {
					// Best-effort logging — don't fail the query
				}

				return {
					content: [{ type: "text", text: result.answer }],
					details: {
						iterations: result.iterations,
						totalSubQueries: result.totalSubQueries,
						completed: result.completed,
						queryPreview: query.slice(0, 100),
						contextLength: context.length,
						trace: fullTrace,
						iterationSummaries: [...iterationSummaries],
						traceLogPath,
					},
				};
			} finally {
				activeLock = null;
				releaseLock();
			}
		},

		renderCall(args, theme) {
			const queryPreview = args.query.length > 80 ? `${args.query.slice(0, 80)}...` : args.query;

			let sourceLabel: string;
			if (args.url) {
				const urlPreview = args.url.length > 60 ? `${args.url.slice(0, 60)}...` : args.url;
				sourceLabel = `url: ${urlPreview}`;
			} else if (args.context) {
				sourceLabel = fmtSize(args.context.length);
			} else {
				sourceLabel = "(no context)";
			}

			const text =
				theme.fg("toolTitle", theme.bold("rlm_query ")) +
				theme.fg("accent", sourceLabel) +
				`\n  ${theme.fg("dim", queryPreview)}`;
			return new Text(text, 0, 0);
		},

		renderResult(result, { expanded, isPartial }, theme) {
			const details = result.details as RlmDetails | undefined;
			if (!details) {
				const text = result.content[0];
				return new Text(text?.type === "text" ? text.text : "(no output)", 0, 0);
			}

			const isRunning = isPartial || (!details.completed && details.iterations > 0);

			let icon: string;
			if (details.completed) {
				icon = theme.fg("success", "[OK]");
			} else if (details.iterations > 0) {
				icon = theme.fg("accent", "[~]");
			} else {
				icon = theme.fg("muted", "[...]");
			}

			// Use the higher of the reported total and the sum from iteration summaries
			const summarySubQueries = details.iterationSummaries?.reduce((sum, s) => sum + s.subQueries, 0) ?? 0;
			const totalSQ = Math.max(details.totalSubQueries, summarySubQueries);

			const stats = [
				`${details.iterations} step${details.iterations !== 1 ? "s" : ""}`,
				`${totalSQ} sub-quer${totalSQ !== 1 ? "ies" : "y"}`,
			].join(", ");

			const header = `${icon} ${theme.fg("toolTitle", theme.bold("RLM Query"))} ${theme.fg("dim", stats)} ${theme.fg("dim", "·")} ${theme.fg("dim", fmtSize(details.contextLength))}`;

			const answerText = result.content
				.filter((b): b is { type: "text"; text: string } => b.type === "text")
				.map((b) => b.text)
				.join("\n");

			/** Full trace is written to disk — show everything in Ctrl+O view. */

			/** Build compact progress timeline from iteration summaries. */
			const buildProgressLines = (summaries: IterationSummary[]): string => {
				if (!summaries.length) return "";
				const lines: string[] = [];
				for (const s of summaries) {
					const stepLabel = theme.fg("dim", `Step ${String(s.step).padStart(2, " ")}`);
					if (s.status === "done") {
						const check = theme.fg("success", "✓");
						const time = theme.fg("dim", s.elapsed ?? "");
						const sq =
							s.subQueries > 0
								? theme.fg("accent", ` ${s.subQueries} sub-quer${s.subQueries !== 1 ? "ies" : "y"}`)
								: "";
						lines.push(`  ${stepLabel}  ${check}  ${time}${sq}`);
					} else if (s.status === "running") {
						const spinner = theme.fg("accent", "◐");
						const activity = s.activity ? theme.fg("toolOutput", ` ${s.activity}`) : "";
						const sq =
							s.subQueries > 0
								? theme.fg("accent", ` (${s.subQueries} sub-quer${s.subQueries !== 1 ? "ies" : "y"} so far)`)
								: "";
						lines.push(`  ${stepLabel}  ${spinner}${activity}${sq}`);
					} else {
						const errIcon = theme.fg("error", "✗");
						const activity = s.activity ? theme.fg("error", ` ${s.activity}`) : "";
						lines.push(`  ${stepLabel}  ${errIcon}${activity}`);
					}
				}
				return lines.join("\n");
			};

			if (isRunning) {
				if (expanded && details.trace) {
					// Ctrl+O while running — show full trace
					const container = new Container();
					container.addChild(new Text(header, 0, 0));
					container.addChild(new Spacer(1));
					container.addChild(new Text(theme.fg("toolOutput", details.trace), 0, 0));
					return container;
				}
				// Collapsed — show compact progress timeline
				const container = new Container();
				container.addChild(new Text(header, 0, 0));
				if (details.iterationSummaries?.length) {
					container.addChild(new Spacer(1));
					container.addChild(new Text(buildProgressLines(details.iterationSummaries), 0, 0));
				}
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "Ctrl+O to see full trace"), 0, 0));
				return container;
			}

			// Done — show compact progress + answer; full trace only on Ctrl+O
			const mdTheme = getMarkdownTheme();
			const container = new Container();
			container.addChild(new Text(header, 0, 0));

			if (expanded && details.trace) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("toolOutput", details.trace), 0, 0));
			} else if (details.iterationSummaries?.length) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(buildProgressLines(details.iterationSummaries), 0, 0));
			}

			if (details.completed && answerText) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(`╭ ${theme.fg("success", "Result")} ${"─".repeat(66)}╮`, 0, 0));
				container.addChild(new Spacer(1));
				container.addChild(new Markdown(answerText.trim(), 0, 0, mdTheme));
				container.addChild(new Text(`╰${"─".repeat(74)}╯`, 0, 0));
			} else if (!details.completed && answerText) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("toolOutput", answerText), 0, 0));
			}

			if (!expanded && details.trace) {
				container.addChild(new Text(theme.fg("muted", "Ctrl+O to see full trace"), 0, 0));
			}

			if (details.traceLogPath) {
				container.addChild(new Text(theme.fg("dim", `Full trace: ${details.traceLogPath}`), 0, 0));
			}

			return container;
		},
	});
}
