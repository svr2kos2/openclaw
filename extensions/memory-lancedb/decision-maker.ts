/**
 * LLMDecisionMaker – uses a chat model to analyze conversation deltas
 * and decide what should be stored/updated/removed from long-term memory.
 *
 * The decision model is an internal "Mini-Agent" with private tools
 * (store_memory, forget_memory, update_memory) that operate directly
 * on LanceDB. Errors are fed back to the model for retry.
 */

import OpenAI from "openai";
import type { MemoryCategory } from "./config.js";
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
// Internal tool definitions (OpenAI function-calling format)
// ============================================================================

const INTERNAL_TOOLS: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
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
          importance: {
            type: "number",
            description: "Importance score from 0 to 1",
          },
        },
        required: ["text", "category", "importance"],
      },
    },
  },
  {
    type: "function",
    function: {
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
  },
  {
    type: "function",
    function: {
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
  },
];

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
  private client: OpenAI;

  constructor(
    private readonly model: string,
    apiKey: string,
    baseUrl: string,
    private readonly ops: MemoryOps,
    private readonly logger: PluginLogger,
  ) {
    this.client = new OpenAI({ apiKey, baseURL: baseUrl });
  }

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

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ];

    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        tools: INTERNAL_TOOLS,
        tool_choice: "auto",
      });

      const choice = response.choices[0];
      if (!choice) break;

      const assistantMsg = choice.message;
      // Push the assistant response (including any tool_calls) back into history
      messages.push(assistantMsg);

      // No tool calls → model is done
      if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
        break;
      }

      // Execute each tool call and feed results back
      for (const toolCall of assistantMsg.tool_calls) {
        // Only handle function-type tool calls
        if (toolCall.type !== "function") continue;
        const { name, arguments: argsStr } = toolCall.function;
        let resultText: string;

        try {
          const args = JSON.parse(argsStr) as Record<string, unknown>;
          const log = await this.executeTool(name, args, journal.cleanContext.length);
          if (log) {
            log.timestamp = now;
            logs.push(log);
          }
          resultText = log ? `Success: ${log.action} memory ${log.memoryId}` : "No action taken.";
        } catch (err) {
          // Return error to model so it can retry or correct
          resultText = `Error: ${String(err instanceof Error ? err.message : err)}`;
          this.logger.warn(`memory-lancedb: tool ${name} failed: ${resultText}`);
        }

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: resultText,
        });
      }

      // If model indicated it's done, stop iterating
      if (choice.finish_reason === "stop") break;
    }

    return { logs };
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

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
