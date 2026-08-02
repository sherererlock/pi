/**
 * TowerAI Provider Extension
 *
 * Adds gpt-5.5 via the company TowerAI gateway (tower-ai.yottastudios.com).
 *
 * Auth: reads the token pair from the local state file that TowerAI's own
 * tooling keeps refreshed (default: ~/.towerai/state.json, same file the
 * Python SDK's browser-login flow writes to). This extension never logs in
 * or refreshes credentials itself — if the token has expired, run whatever
 * TowerAI flow you normally use to refresh it, then retry.
 *
 * The gateway's chat endpoint does not speak plain OpenAI Chat Completions
 * or OpenAI Responses SSE. It wraps the real OpenAI Responses API events in
 * its own envelope and additionally emits simplified `event: text` /
 * `event: tool_calls` / `event: usage` / `event: stop` events (see
 * TowerAI's own `src/stream.ts` / `towerai/_sse.py`). We parse only those
 * simplified events and ignore the raw `event: data` passthrough.
 */

import { homedir } from "node:os";
import { join } from "node:path";
import { readFile } from "node:fs/promises";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	calculateCost,
	type Context,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type Tool,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_BASE_URL = "https://tower-ai.yottastudios.com";
const CHAT_ENDPOINT = "/zi/webapi/chat/openai";
const STATE_FILE = process.env.TOWERAI_STATE_FILE ?? join(homedir(), ".towerai", "state.json");

interface TowerAIState {
	token?: string;
	auth_token?: string;
	base_url?: string;
}

async function loadState(): Promise<TowerAIState> {
	let raw: string;
	try {
		raw = await readFile(STATE_FILE, "utf8");
	} catch {
		throw new Error(
			`TowerAI: could not read state file at ${STATE_FILE}. Log in via your normal TowerAI flow first.`,
		);
	}
	let state: TowerAIState;
	try {
		state = JSON.parse(raw);
	} catch {
		throw new Error(`TowerAI: state file at ${STATE_FILE} is not valid JSON.`);
	}
	if (!state.token) {
		throw new Error(
			`TowerAI: no token in ${STATE_FILE}. Refresh your TowerAI login, then retry.`,
		);
	}
	return state;
}

interface OpenAIChatMessage {
	role: "system" | "user" | "assistant" | "tool";
	content?: string | { type: string; [key: string]: unknown }[];
	tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
	tool_call_id?: string;
}

function convertMessages(context: Context): OpenAIChatMessage[] {
	const out: OpenAIChatMessage[] = [];
	if (context.systemPrompt) {
		out.push({ role: "system", content: context.systemPrompt });
	}
	for (const message of context.messages) {
		out.push(...convertMessage(message));
	}
	return out;
}

interface TextBlock {
	type: "text";
	text: string;
}
interface ImageBlock {
	type: "image";
	data: string;
	mimeType: string;
}
interface ToolCallBlock {
	type: "toolCall";
	id: string;
	name: string;
	arguments?: Record<string, unknown>;
}

function convertMessage(message: Message): OpenAIChatMessage[] {
	if (message.role === "user") {
		if (typeof message.content === "string") {
			return [{ role: "user", content: message.content }];
		}
		const parts = (message.content as (TextBlock | ImageBlock)[]).map((block) =>
			block.type === "text"
				? { type: "text", text: block.text }
				: { type: "image_url", image_url: { url: `data:${block.mimeType};base64,${block.data}` } },
		);
		return [{ role: "user", content: parts }];
	}

	if (message.role === "assistant") {
		const blocks = message.content as (TextBlock | ToolCallBlock | { type: "thinking" })[];
		const text = blocks
			.filter((block): block is TextBlock => block.type === "text")
			.map((block) => block.text)
			.join("");
		const toolCalls = blocks
			.filter((block): block is ToolCallBlock => block.type === "toolCall")
			.map((block) => ({
				id: block.id,
				type: "function" as const,
				function: { name: block.name, arguments: JSON.stringify(block.arguments ?? {}) },
			}));
		const out: OpenAIChatMessage = { role: "assistant" };
		if (text) out.content = text;
		if (toolCalls.length > 0) out.tool_calls = toolCalls;
		return [out];
	}

	// toolResult
	const text = (message.content as TextBlock[])
		.filter((block): block is TextBlock => block.type === "text")
		.map((block) => block.text)
		.join("\n");
	return [{ role: "tool", tool_call_id: message.toolCallId, content: text || (message.isError ? "Error" : "") }];
}

// The gateway routes gpt-5.5 tool calls to OpenAI's real Responses API backend,
// which strictly validates JSON Schema and rejects regex features it doesn't
// support (e.g. lookaround). Some built-in pi tools (e.g. subagent_start's
// `model` pattern) use lookaround, so strip/loosen unsupported `pattern`
// fields recursively before sending.
const UNSUPPORTED_PATTERN_RE = /\(\?[=!<]/;

function sanitizeSchema(schema: unknown): unknown {
	if (Array.isArray(schema)) return schema.map(sanitizeSchema);
	if (!schema || typeof schema !== "object") return schema;
	const out: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(schema as Record<string, unknown>)) {
		if (key === "pattern" && typeof value === "string" && UNSUPPORTED_PATTERN_RE.test(value)) {
			continue; // drop the constraint rather than fail the whole request
		}
		out[key] = sanitizeSchema(value);
	}
	return out;
}

function convertTools(tools: Tool[] | undefined) {
	if (!tools || tools.length === 0) return undefined;
	return tools.map((tool) => ({
		type: "function" as const,
		function: { name: tool.name, description: tool.description, parameters: sanitizeSchema(tool.parameters) },
	}));
}

interface TowerSSEEvent {
	event?: string;
	data?: string;
}

function parseSSEBlock(block: string): TowerSSEEvent {
	const event: TowerSSEEvent = {};
	for (const line of block.split("\n")) {
		const colon = line.indexOf(":");
		if (colon === -1) continue;
		const field = line.slice(0, colon).trim();
		const value = line.slice(colon + 1).trim();
		if (field === "event") event.event = value;
		else if (field === "data") event.data = value;
	}
	return event;
}

export function streamTowerAI(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	const output: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "pending",
		timestamp: Date.now(),
	};

	(async () => {
		try {
			const state = await loadState();
			const baseUrl = model.baseUrl ?? state.base_url ?? DEFAULT_BASE_URL;
			const url = `${baseUrl}${CHAT_ENDPOINT}`;

			const body = {
				model: model.id,
				messages: convertMessages(context),
				tools: convertTools(context.tools),
				stream: true,
				temperature: 1,
			};

			const response = await fetch(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Token: state.token ?? "",
					"X-lobe-chat-auth": state.auth_token ?? "",
					accept: "text/event-stream",
				},
				body: JSON.stringify(body),
				signal: options?.signal,
			});

			if (!response.ok || !response.body) {
				const text = await response.text().catch(() => "");
				throw new Error(`TowerAI request failed (${response.status}): ${text.slice(0, 500)}`);
			}

			stream.push({ type: "start", partial: output });

			let textIndex = -1;
			const toolIndexToContentIndex = new Map<number, number>();
			const toolRawArgs = new Map<number, string>();

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";

			const handleEvent = (evt: TowerSSEEvent) => {
				if (evt.event === "text" && evt.data) {
					let delta: string | null;
					try {
						delta = JSON.parse(evt.data);
					} catch {
						return;
					}
					if (!delta) return;
					if (textIndex === -1) {
						output.content.push({ type: "text", text: "" });
						textIndex = output.content.length - 1;
						stream.push({ type: "text_start", contentIndex: textIndex, partial: output });
					}
					const block = output.content[textIndex];
					if (block.type === "text") {
						block.text += delta;
						stream.push({ type: "text_delta", contentIndex: textIndex, delta, partial: output });
					}
					return;
				}

				if (evt.event === "tool_calls" && evt.data) {
					let calls: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[];
					try {
						calls = JSON.parse(evt.data);
					} catch {
						return;
					}
					for (const tc of calls) {
						const idx = tc.index ?? 0;
						let contentIndex = toolIndexToContentIndex.get(idx);
						if (contentIndex === undefined) {
							output.content.push({ type: "toolCall", id: tc.id ?? `call_${idx}`, name: tc.function?.name ?? "", arguments: {} });
							contentIndex = output.content.length - 1;
							toolIndexToContentIndex.set(idx, contentIndex);
							toolRawArgs.set(idx, "");
							stream.push({ type: "toolcall_start", contentIndex, partial: output });
						}
						const block = output.content[contentIndex];
						if (block.type !== "toolCall") continue;
						if (tc.function?.name) block.name = tc.function.name;
						if (tc.id) block.id = tc.id;
						const argsDelta = tc.function?.arguments ?? "";
						const raw = (toolRawArgs.get(idx) ?? "") + argsDelta;
						toolRawArgs.set(idx, raw);
						try {
							block.arguments = JSON.parse(raw);
						} catch {
							// incomplete JSON, keep accumulating
						}
						stream.push({ type: "toolcall_delta", contentIndex, delta: argsDelta, partial: output });
					}
					return;
				}

				if (evt.event === "usage" && evt.data) {
					let raw: Record<string, number>;
					try {
						raw = JSON.parse(evt.data);
					} catch {
						return;
					}
					output.usage.input = raw.inputTextTokens ?? raw.totalInputTokens ?? output.usage.input;
					output.usage.output = raw.outputTextTokens ?? raw.totalOutputTokens ?? output.usage.output;
					output.usage.reasoning = raw.outputReasoningTokens;
					output.usage.cacheRead = raw.inputCacheHitTokens ?? output.usage.cacheRead;
					output.usage.totalTokens =
						raw.totalTokens ?? output.usage.input + output.usage.output + output.usage.cacheRead;
					calculateCost(model, output.usage);
					return;
				}

				if (evt.event === "stop") {
					finalize();
				}
			};

			// Not every response includes an explicit `event: stop` (only text, no
			// tool calls, ends by closing the connection). Finalize open blocks and
			// set stopReason exactly once, whether triggered by `stop` or by EOF.
			const finalize = () => {
				if (output.stopReason !== "pending") return;
				if (textIndex !== -1) {
					const block = output.content[textIndex];
					if (block.type === "text") {
						stream.push({ type: "text_end", contentIndex: textIndex, content: block.text, partial: output });
					}
					textIndex = -1;
				}
				for (const [idx, contentIndex] of toolIndexToContentIndex) {
					const block = output.content[contentIndex];
					if (block.type !== "toolCall") continue;
					try {
						block.arguments = JSON.parse(toolRawArgs.get(idx) ?? "{}");
					} catch {
						block.arguments = {};
					}
					stream.push({ type: "toolcall_end", contentIndex, toolCall: block, partial: output });
				}
				output.stopReason = toolIndexToContentIndex.size > 0 ? "toolUse" : "stop";
			};

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const parts = buffer.split("\n\n");
				buffer = parts.pop() ?? "";
				for (const part of parts) {
					if (part.trim()) handleEvent(parseSSEBlock(part));
				}
			}
			if (buffer.trim()) handleEvent(parseSSEBlock(buffer));

			// Gateway closed the connection without an explicit `event: stop`
			// (normal for plain text replies) — treat EOF as completion.
			finalize();

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}

export default function (pi: ExtensionAPI) {
	pi.registerProvider("towerai", {
		name: "TowerAI",
		baseUrl: DEFAULT_BASE_URL,
		apiKey: "towerai-state-file", // unused literal; streamTowerAI reads the state file directly
		api: "towerai-api",
		models: [
			{
				id: "gpt-5.5",
				name: "GPT-5.5 (TowerAI)",
				reasoning: true,
				input: ["text"],
				cost: { input: 5, output: 30, cacheRead: 0.5, cacheWrite: 0 },
				contextWindow: 400000,
				maxTokens: 128000,
			},
		],
		streamSimple: streamTowerAI,
	});
}
