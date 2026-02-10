/**
 * LLMDecisionMaker – uses a chat model to analyze conversation deltas
 * and decide what should be stored/updated/removed from long-term memory.
 *
 * The decision model is an internal "Mini-Agent" with private tools
 * (store_memory, forget_memory, update_memory) that operate directly
 * on LanceDB. Errors are fed back to the model for retry.
 *
 * Supports two API backends: OpenAI completions and Anthropic messages.
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import type { ChatApi, MemoryCategory } from "./config.js";
import type { CleanMessage, JournalData, MemoryLogEntry } from "./session-journal.js";

// ============================================================================
// Types
// ============================================================================

export type DecisionResult = {
  logs: MemoryLogEntry[];
};

/** Low-level memory operations provided by the plugin host */
export type MemoryOps = {
  /** Embed + store. Returns the new memory ID. Throws on duplicate. */
  store(text: string, category: MemoryCategory, importance: number): Promise<string>;
  /** Delete by ID. Throws if ID is invalid. */
  forget(memoryId: string): Promise<boolean>;
  /** Delete + re-embed + insert. Returns the new memory ID. */
  update(
    memoryId: string,
    text: string,
    category: MemoryCategory,
    importance: number,
  ): Promise<string>;
};

type PluginLogger = {
  info: (msg: string) => void;
  warn: (msg: string) => void;
};

// ============================================================================
// Internal tool schemas (shared between adapters)
// ============================================================================

type InternalToolDef = {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required: string[];
  };
};

const TOOL_DEFS: InternalToolDef[] = [
  {
    name: "store_memory",
    description: "Store a new piece of information in long-term memory.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "The original conversation text to store" },
        category: {
          type: "string",
          enum: ["preference", "fact", "decision", "entity", "other"],
          description: "Memory category",
        },
        importance: { type: "number", description: "Importance score from 0 to 1" },
      },
      required: ["text", "category", "importance"],
    },
  },
  {
    name: "forget_memory",
    description: "Delete a memory by its ID. Only use when the user explicitly asks to forget.",
    parameters: {
      type: "object",
      properties: {
        memoryId: { type: "string", description: "UUID of the memory to delete" },
      },
      required: ["memoryId"],
    },
  },
  {
    name: "update_memory",
    description:
      "Update an existing memory (replaces it with new text). Use when new information contradicts or refines a previously stored memory.",
    parameters: {
      type: "object",
      properties: {
        memoryId: { type: "string", description: "UUID of the memory to update" },
        text: { type: "string", description: "Updated memory text" },
        category: {
          type: "string",
          enum: ["preference", "fact", "decision", "entity", "other"],
          description: "Memory category",
        },
        importance: { type: "number", description: "Updated importance score 0-1" },
      },
      required: ["memoryId", "text", "category", "importance"],
    },
  },
];

// ============================================================================
// Chat adapter interface
// ============================================================================

/** A pending tool call returned by the model. */
type ToolCallRequest = {
  id: string;
  name: string;
  args: Record<string, unknown>;
};

/** Result of a single chat completion round. */
type AdapterResponse = {
  toolCalls: ToolCallRequest[];
  /** True when the model signalled it is done (no more tool calls). */
  done: boolean;
};

interface ChatAdapter {
  /** Run one completion round and return parsed tool calls. */
  complete(): Promise<AdapterResponse>;
  /** Push a tool result into the conversation. */
  pushToolResult(toolCallId: string, content: string, isError: boolean): void;
}

// ============================================================================
// OpenAI adapter
// ============================================================================

class OpenAIAdapter implements ChatAdapter {
  private client: OpenAI;
  private messages: OpenAI.ChatCompletionMessageParam[];
  private tools: OpenAI.ChatCompletionTool[];

  constructor(
    private readonly model: string,
    apiKey: string,
    baseUrl: string,
    systemPrompt: string,
    userPrompt: string,
  ) {
    this.client = new OpenAI({ apiKey, baseURL: baseUrl });
    this.messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ];
    this.tools = TOOL_DEFS.map((t) => ({
      type: "function" as const,
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }

  async complete(): Promise<AdapterResponse> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: this.messages,
      tools: this.tools,
      tool_choice: "auto",
    });

    const choice = response.choices[0];
    if (!choice) return { toolCalls: [], done: true };

    this.messages.push(choice.message);

    const toolCalls: ToolCallRequest[] = [];
    if (choice.message.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        if (tc.type !== "function") continue;
        toolCalls.push({
          id: tc.id,
          name: tc.function.name,
          args: JSON.parse(tc.function.arguments) as Record<string, unknown>,
        });
      }
    }

    return {
      toolCalls,
      done: toolCalls.length === 0 || choice.finish_reason === "stop",
    };
  }

  pushToolResult(toolCallId: string, content: string, _isError: boolean): void {
    this.messages.push({ role: "tool", tool_call_id: toolCallId, content });
  }
}

// ============================================================================
// Anthropic adapter
// ============================================================================

class AnthropicAdapter implements ChatAdapter {
  private client: Anthropic;
  private messages: Anthropic.Messages.MessageParam[];
  private tools: Anthropic.Messages.Tool[];
  private systemPrompt: string;

  constructor(
    private readonly model: string,
    apiKey: string,
    baseUrl: string,
    systemPrompt: string,
    userPrompt: string,
  ) {
    this.client = new Anthropic({ apiKey, baseURL: baseUrl });
    this.systemPrompt = systemPrompt;
    this.messages = [{ role: "user", content: userPrompt }];
    this.tools = TOOL_DEFS.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters as Anthropic.Messages.Tool.InputSchema,
    }));
  }

  async complete(): Promise<AdapterResponse> {
    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      system: this.systemPrompt,
      messages: this.messages,
      tools: this.tools,
      tool_choice: { type: "auto" },
    });

    // Append assistant reply to conversation
    this.messages.push({ role: "assistant", content: response.content });

    const toolCalls: ToolCallRequest[] = [];
    for (const block of response.content) {
      if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id,
          name: block.name,
          args: block.input as Record<string, unknown>,
        });
      }
    }

    return {
      toolCalls,
      done: toolCalls.length === 0 || response.stop_reason !== "tool_use",
    };
  }

  pushToolResult(toolCallId: string, content: string, isError: boolean): void {
    // Anthropic expects tool results as content blocks inside a user message
    this.messages.push({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: toolCallId, content, is_error: isError }],
    });
  }
}

// ============================================================================
// System prompt
// ============================================================================

const SYSTEM_PROMPT = `You are a memory management system for an AI assistant. Your job is to analyze new conversation messages and decide what should be stored in, updated in, or removed from long-term memory.

You have three tools: store_memory, forget_memory, update_memory.

RULES:
1. ONLY analyze messages in [NEW DELTA]. Use [CONTEXT] for background and [MEMORY LOGS] to avoid duplicates.
2. Store the ORIGINAL text from the conversation, not your own summary or rephrasing.
3. SKIP: greetings, acknowledgments, one-word replies, code blocks (unless they contain configuration or preference decisions), system-injected XML tags.
4. STORE: explicit preferences ("I prefer…", "I like…", "I always…"), personal facts (name, email, phone), architectural/technical decisions, named entities.
5. If new information contradicts or refines an existing memory (check MEMORY LOGS), use update_memory with the existing memory's ID.
6. Only use forget_memory when the user explicitly asks to forget something.
7. If there is nothing worth storing, respond with a brief text message and NO tool calls.
8. Importance guide: 0.9 explicit "remember this", 0.8 preferences/decisions, 0.6 facts/entities, 0.5 other.`;

// ============================================================================
// LLMDecisionMaker
// ============================================================================

const MAX_TOOL_ITERATIONS = 5;

export class LLMDecisionMaker {
  constructor(
    private readonly api: ChatApi,
    private readonly model: string,
    private readonly apiKey: string,
    private readonly baseUrl: string,
    private readonly ops: MemoryOps,
    private readonly logger: PluginLogger,
  ) {}

  /**
   * Evaluate a journal's delta and execute memory operations via tool calls.
   * Returns logs of all operations performed.
   */
  async evaluate(journal: JournalData, delta: CleanMessage[]): Promise<DecisionResult> {
    if (delta.length === 0) {
      return { logs: [] };
    }

    const userPrompt = this.buildPrompt(journal, delta);
    const logs: MemoryLogEntry[] = [];
    const now = Date.now();

    // Create the appropriate adapter
    const adapter: ChatAdapter =
      this.api === "anthropic-messages"
        ? new AnthropicAdapter(this.model, this.apiKey, this.baseUrl, SYSTEM_PROMPT, userPrompt)
        : new OpenAIAdapter(this.model, this.apiKey, this.baseUrl, SYSTEM_PROMPT, userPrompt);

    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      const response = await adapter.complete();

      if (response.toolCalls.length === 0 || response.done) {
        // Execute any final tool calls before breaking
        for (const tc of response.toolCalls) {
          const log = await this.handleToolCall(tc, journal.cleanContext.length, now, adapter);
          if (log) logs.push(log);
        }
        break;
      }

      for (const tc of response.toolCalls) {
        const log = await this.handleToolCall(tc, journal.cleanContext.length, now, adapter);
        if (log) logs.push(log);
      }
    }

    return { logs };
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

  private async handleToolCall(
    tc: ToolCallRequest,
    deltaIndex: number,
    now: number,
    adapter: ChatAdapter,
  ): Promise<MemoryLogEntry | null> {
    try {
      const log = await this.executeTool(tc.name, tc.args, deltaIndex);
      if (log) log.timestamp = now;
      const resultText = log
        ? `Success: ${log.action} memory ${log.memoryId}`
        : "No action taken.";
      adapter.pushToolResult(tc.id, resultText, false);
      return log;
    } catch (err) {
      const resultText = `Error: ${String(err instanceof Error ? err.message : err)}`;
      this.logger.warn(`memory-lancedb: tool ${tc.name} failed: ${resultText}`);
      adapter.pushToolResult(tc.id, resultText, true);
      return null;
    }
  }

  private async executeTool(
    name: string,
    args: Record<string, unknown>,
    deltaIndex: number,
  ): Promise<MemoryLogEntry | null> {
    switch (name) {
      case "store_memory": {
        const text = String(args.text ?? "");
        const category = String(args.category ?? "other") as MemoryCategory;
        const importance = Number(args.importance ?? 0.7);
        if (!text) throw new Error("text is required for store_memory");
        const memoryId = await this.ops.store(text, category, importance);
        this.logger.info(`memory-lancedb: stored ${memoryId.slice(0, 8)}: "${text.slice(0, 60)}"`);
        return { deltaIndex, action: "store", memoryId, text, timestamp: 0 };
      }

      case "forget_memory": {
        const memoryId = String(args.memoryId ?? "");
        if (!memoryId) throw new Error("memoryId is required for forget_memory");
        await this.ops.forget(memoryId);
        this.logger.info(`memory-lancedb: forgot ${memoryId.slice(0, 8)}`);
        return { deltaIndex, action: "forget", memoryId, text: "", timestamp: 0 };
      }

      case "update_memory": {
        const memoryId = String(args.memoryId ?? "");
        const text = String(args.text ?? "");
        const category = String(args.category ?? "other") as MemoryCategory;
        const importance = Number(args.importance ?? 0.7);
        if (!memoryId) throw new Error("memoryId is required for update_memory");
        if (!text) throw new Error("text is required for update_memory");
        const newId = await this.ops.update(memoryId, text, category, importance);
        this.logger.info(`memory-lancedb: updated ${memoryId.slice(0, 8)} → ${newId.slice(0, 8)}`);
        return { deltaIndex, action: "update", memoryId: newId, text, timestamp: 0 };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  private buildPrompt(journal: JournalData, delta: CleanMessage[]): string {
    const parts: string[] = [];

    // [CONTEXT] – full conversation history for background understanding
    if (journal.cleanContext.length > 0) {
      const lines = journal.cleanContext.map((m) => `[${m.role}]: ${m.text}`);
      parts.push(`[CONTEXT]\n${lines.join("\n")}`);
    }

    // [MEMORY LOGS] – actions already taken in this session
    if (journal.memoryLogs.length > 0) {
      const lines = journal.memoryLogs.map(
        (l) => `- ${l.action} [${l.memoryId.slice(0, 8)}]: ${l.text.slice(0, 80)}`,
      );
      parts.push(`[MEMORY LOGS]\n${lines.join("\n")}`);
    } else {
      parts.push("[MEMORY LOGS]\nNo memory operations in this session yet.");
    }

    // [NEW DELTA] – only these messages should be analyzed for decisions
    const deltaLines = delta.map((m) => `[${m.role}]: ${m.text}`);
    parts.push(`[NEW DELTA]\n${deltaLines.join("\n")}`);

    return parts.join("\n\n");
  }
}
