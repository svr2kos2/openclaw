/**
 * OpenClaw Memory (LanceDB) Plugin
 *
 * Long-term memory with vector search for AI conversations.
 * Uses LanceDB for storage and OpenAI-compatible embedding APIs.
 * Provides seamless auto-recall and auto-capture via lifecycle hooks.
 */

import type * as LanceDB from "@lancedb/lancedb";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import { randomUUID } from "node:crypto";
import OpenAI from "openai";
import { stringEnum } from "openclaw/plugin-sdk";
import { MEMORY_CATEGORIES, type MemoryCategory, memoryConfigSchema } from "./config.js";
import { LLMDecisionMaker, type MemoryOps } from "./decision-maker.js";
import { SessionJournal, type CleanMessage } from "./session-journal.js";

// ============================================================================
// Types
// ============================================================================

let lancedbImportPromise: Promise<typeof import("@lancedb/lancedb")> | null = null;
const loadLanceDB = async (): Promise<typeof import("@lancedb/lancedb")> => {
  if (!lancedbImportPromise) {
    lancedbImportPromise = import("@lancedb/lancedb");
  }
  try {
    return await lancedbImportPromise;
  } catch (err) {
    // Common on macOS today: upstream package may not ship darwin native bindings.
    throw new Error(`memory-lancedb: failed to load LanceDB. ${String(err)}`, { cause: err });
  }
};

type MemoryEntry = {
  id: string;
  text: string;
  vector: number[];
  importance: number;
  category: MemoryCategory;
  createdAt: number;
};

type MemorySearchResult = {
  entry: MemoryEntry;
  score: number;
};

// ============================================================================
// LanceDB Provider
// ============================================================================

const TABLE_NAME = "memories";

class MemoryDB {
  private db: LanceDB.Connection | null = null;
  private table: LanceDB.Table | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly dbPath: string,
    private readonly vectorDim: number,
  ) {}

  private async ensureInitialized(): Promise<void> {
    if (this.table) {
      return;
    }
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    const lancedb = await loadLanceDB();
    this.db = await lancedb.connect(this.dbPath);
    const tables = await this.db.tableNames();

    if (tables.includes(TABLE_NAME)) {
      this.table = await this.db.openTable(TABLE_NAME);
    } else {
      this.table = await this.db.createTable(TABLE_NAME, [
        {
          id: "__schema__",
          text: "",
          vector: Array.from({ length: this.vectorDim }).fill(0),
          importance: 0,
          category: "other",
          createdAt: 0,
        },
      ]);
      await this.table.delete('id = "__schema__"');
    }
  }

  async store(entry: Omit<MemoryEntry, "id" | "createdAt">): Promise<MemoryEntry> {
    await this.ensureInitialized();

    const fullEntry: MemoryEntry = {
      ...entry,
      id: randomUUID(),
      createdAt: Date.now(),
    };

    await this.table!.add([fullEntry]);
    return fullEntry;
  }

  async search(vector: number[], limit = 5, minScore = 0.5): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();

    const results = await this.table!.vectorSearch(vector).limit(limit).toArray();

    // LanceDB uses L2 distance by default; convert to similarity score
    const mapped = results.map((row) => {
      const distance = row._distance ?? 0;
      // Use inverse for a 0-1 range: sim = 1 / (1 + d)
      const score = 1 / (1 + distance);
      return {
        entry: {
          id: row.id as string,
          text: row.text as string,
          vector: row.vector as number[],
          importance: row.importance as number,
          category: row.category as MemoryEntry["category"],
          createdAt: row.createdAt as number,
        },
        score,
      };
    });

    return mapped.filter((r) => r.score >= minScore);
  }

  async delete(id: string): Promise<boolean> {
    await this.ensureInitialized();
    // Validate UUID format to prevent injection
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(id)) {
      throw new Error(`Invalid memory ID format: ${id}`);
    }
    await this.table!.delete(`id = '${id}'`);
    return true;
  }

  async count(): Promise<number> {
    await this.ensureInitialized();
    return this.table!.countRows();
  }
}

// ============================================================================
// OpenAI-Compatible Embeddings
// ============================================================================

class Embeddings {
  private client: OpenAI;

  constructor(
    apiKey: string,
    private readonly model: string,
    baseUrl: string,
  ) {
    this.client = new OpenAI({ apiKey, baseURL: baseUrl });
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.data[0].embedding;
  }
}

// ============================================================================
// Message extraction helper
// ============================================================================

/** Extract user/assistant plain-text messages from an agent event's message array. */
function extractCleanMessages(messages: unknown[]): CleanMessage[] {
  const result: CleanMessage[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const msgObj = msg as Record<string, unknown>;
    const role = msgObj.role;
    if (role !== "user" && role !== "assistant") continue;

    const content = msgObj.content;

    if (typeof content === "string") {
      const text = content.trim();
      if (text.length > 0 && !text.includes("<relevant-memories>")) {
        result.push({ role: role as "user" | "assistant", text });
      }
      continue;
    }

    if (Array.isArray(content)) {
      for (const block of content) {
        if (
          block &&
          typeof block === "object" &&
          "type" in block &&
          (block as Record<string, unknown>).type === "text" &&
          "text" in block &&
          typeof (block as Record<string, unknown>).text === "string"
        ) {
          const text = ((block as Record<string, unknown>).text as string).trim();
          if (text.length > 0 && !text.includes("<relevant-memories>")) {
            result.push({ role: role as "user" | "assistant", text });
          }
        }
      }
    }
  }
  return result;
}

// ============================================================================
// Plugin Definition
// ============================================================================

const memoryPlugin = {
  id: "memory-lancedb",
  name: "Memory (LanceDB)",
  description: "LanceDB-backed long-term memory with auto-recall/capture",
  kind: "memory" as const,
  configSchema: memoryConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = memoryConfigSchema.parse(api.pluginConfig);
    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    const db = new MemoryDB(resolvedDbPath, cfg.embedding.dimensions);
    const embeddings = new Embeddings(
      cfg.embedding.apiKey,
      cfg.embedding.model,
      cfg.embedding.baseUrl,
    );

    api.logger.info(
      `memory-lancedb: plugin registered (db: ${resolvedDbPath}, model: ${cfg.embedding.model}, baseUrl: ${cfg.embedding.baseUrl}, lazy init)`,
    );

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: "memory_recall",
        label: "Memory Recall",
        description:
          "Search through long-term memories. Use when you need context about user preferences, past decisions, or previously discussed topics.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          limit: Type.Optional(Type.Number({ description: "Max results (default: 5)" })),
        }),
        async execute(_toolCallId, params) {
          const { query, limit = 5 } = params as { query: string; limit?: number };

          const vector = await embeddings.embed(query);
          const results = await db.search(vector, limit, 0.1);

          if (results.length === 0) {
            return {
              content: [{ type: "text", text: "No relevant memories found." }],
              details: { count: 0 },
            };
          }

          const text = results
            .map(
              (r, i) =>
                `${i + 1}. [${r.entry.category}] ${r.entry.text} (${(r.score * 100).toFixed(0)}%)`,
            )
            .join("\n");

          // Strip vector data for serialization (typed arrays can't be cloned)
          const sanitizedResults = results.map((r) => ({
            id: r.entry.id,
            text: r.entry.text,
            category: r.entry.category,
            importance: r.entry.importance,
            score: r.score,
          }));

          return {
            content: [{ type: "text", text: `Found ${results.length} memories:\n\n${text}` }],
            details: { count: results.length, memories: sanitizedResults },
          };
        },
      },
      { name: "memory_recall" },
    );

    api.registerTool(
      {
        name: "memory_store",
        label: "Memory Store",
        description:
          "Save important information in long-term memory. Use for preferences, facts, decisions.",
        parameters: Type.Object({
          text: Type.String({ description: "Information to remember" }),
          importance: Type.Optional(Type.Number({ description: "Importance 0-1 (default: 0.7)" })),
          category: Type.Optional(stringEnum(MEMORY_CATEGORIES)),
        }),
        async execute(_toolCallId, params) {
          const {
            text,
            importance = 0.7,
            category = "other",
          } = params as {
            text: string;
            importance?: number;
            category?: MemoryEntry["category"];
          };

          const vector = await embeddings.embed(text);

          // Check for duplicates
          const existing = await db.search(vector, 1, 0.95);
          if (existing.length > 0) {
            return {
              content: [
                {
                  type: "text",
                  text: `Similar memory already exists: "${existing[0].entry.text}"`,
                },
              ],
              details: {
                action: "duplicate",
                existingId: existing[0].entry.id,
                existingText: existing[0].entry.text,
              },
            };
          }

          const entry = await db.store({
            text,
            vector,
            importance,
            category,
          });

          return {
            content: [{ type: "text", text: `Stored: "${text.slice(0, 100)}..."` }],
            details: { action: "created", id: entry.id },
          };
        },
      },
      { name: "memory_store" },
    );

    api.registerTool(
      {
        name: "memory_forget",
        label: "Memory Forget",
        description: "Delete specific memories. GDPR-compliant.",
        parameters: Type.Object({
          query: Type.Optional(Type.String({ description: "Search to find memory" })),
          memoryId: Type.Optional(Type.String({ description: "Specific memory ID" })),
        }),
        async execute(_toolCallId, params) {
          const { query, memoryId } = params as { query?: string; memoryId?: string };

          if (memoryId) {
            await db.delete(memoryId);
            return {
              content: [{ type: "text", text: `Memory ${memoryId} forgotten.` }],
              details: { action: "deleted", id: memoryId },
            };
          }

          if (query) {
            const vector = await embeddings.embed(query);
            const results = await db.search(vector, 5, 0.7);

            if (results.length === 0) {
              return {
                content: [{ type: "text", text: "No matching memories found." }],
                details: { found: 0 },
              };
            }

            if (results.length === 1 && results[0].score > 0.9) {
              await db.delete(results[0].entry.id);
              return {
                content: [{ type: "text", text: `Forgotten: "${results[0].entry.text}"` }],
                details: { action: "deleted", id: results[0].entry.id },
              };
            }

            const list = results
              .map((r) => `- [${r.entry.id.slice(0, 8)}] ${r.entry.text.slice(0, 60)}...`)
              .join("\n");

            // Strip vector data for serialization
            const sanitizedCandidates = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              score: r.score,
            }));

            return {
              content: [
                {
                  type: "text",
                  text: `Found ${results.length} candidates. Specify memoryId:\n${list}`,
                },
              ],
              details: { action: "candidates", candidates: sanitizedCandidates },
            };
          }

          return {
            content: [{ type: "text", text: "Provide query or memoryId." }],
            details: { error: "missing_param" },
          };
        },
      },
      { name: "memory_forget" },
    );

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const memory = program.command("ltm").description("LanceDB memory plugin commands");

        memory
          .command("list")
          .description("List memories")
          .action(async () => {
            const count = await db.count();
            console.log(`Total memories: ${count}`);
          });

        memory
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "5")
          .action(async (query, opts) => {
            const vector = await embeddings.embed(query);
            const results = await db.search(vector, parseInt(opts.limit), 0.3);
            // Strip vectors for output
            const output = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              importance: r.entry.importance,
              score: r.score,
            }));
            console.log(JSON.stringify(output, null, 2));
          });

        memory
          .command("stats")
          .description("Show memory statistics")
          .action(async () => {
            const count = await db.count();
            console.log(`Total memories: ${count}`);
          });
      },
      { commands: ["ltm"] },
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // Auto-recall: inject relevant memories before agent starts
    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event) => {
        if (!event.prompt || event.prompt.length < 5) {
          return;
        }

        try {
          const vector = await embeddings.embed(event.prompt);
          const results = await db.search(vector, 3, 0.3);

          if (results.length === 0) {
            return;
          }

          const memoryContext = results
            .map((r) => `- [${r.entry.category}] ${r.entry.text}`)
            .join("\n");

          api.logger.info?.(`memory-lancedb: injecting ${results.length} memories into context`);

          return {
            prependContext: `<relevant-memories>\nThe following memories may be relevant to this conversation:\n${memoryContext}\n</relevant-memories>`,
          };
        } catch (err) {
          api.logger.warn(`memory-lancedb: recall failed: ${String(err)}`);
        }
      });
    }

    // Auto-capture: LLM-based analysis of conversations for memory capture
    if (cfg.autoCapture) {
      if (!cfg.chat) {
        api.logger.warn(
          "memory-lancedb: autoCapture enabled but no chat config provided. " +
            "Add a 'chat' section with apiKey/model/baseUrl to enable LLM-based capture.",
        );
      } else {
        const journal = new SessionJournal(resolvedDbPath);

        // Build memory operations for the decision maker
        const ops: MemoryOps = {
          async store(text, category, importance) {
            const vector = await embeddings.embed(text);
            // Check for near-duplicates
            const existing = await db.search(vector, 1, 0.95);
            if (existing.length > 0) {
              throw new Error(
                `Similar memory already exists: "${existing[0].entry.text.slice(0, 60)}" ` +
                  `(id: ${existing[0].entry.id}). Use update_memory to modify it.`,
              );
            }
            const entry = await db.store({ text, vector, importance, category });
            return entry.id;
          },
          async forget(memoryId) {
            return db.delete(memoryId);
          },
          async update(memoryId, text, category, importance) {
            // delete + re-embed + insert (holding the session queue lock)
            await db.delete(memoryId);
            const vector = await embeddings.embed(text);
            const entry = await db.store({ text, vector, importance, category });
            return entry.id;
          },
        };

        const decisionMaker = new LLMDecisionMaker(
          cfg.chat.api,
          cfg.chat.model,
          cfg.chat.apiKey,
          cfg.chat.baseUrl,
          ops,
          api.logger,
        );

        api.on("agent_end", async (event, ctx) => {
          api.logger.info(
            `memory-lancedb: agent_end triggered, success=${event.success}, messages=${event.messages?.length ?? 0}`,
          );

          if (!event.success || !event.messages || event.messages.length === 0) {
            api.logger.info("memory-lancedb: skipping - no success or no messages");
            return;
          }

          const sessionKey = ctx.sessionId ?? ctx.sessionKey;
          if (!sessionKey) {
            api.logger.warn(
              "memory-lancedb: no sessionId/sessionKey in agent_end context, skipping capture",
            );
            return;
          }

          api.logger.info(
            `memory-lancedb: using sessionId=${ctx.sessionId ?? "(none)"} (fallback sessionKey=${ctx.sessionKey ?? "(none)"})`,
          );

          try {
            // Extract user/assistant plain text from the event messages
            const cleanMsgs = extractCleanMessages(event.messages);
            api.logger.info(
              `memory-lancedb: extracted ${cleanMsgs.length} clean messages from ${event.messages.length} raw messages`,
            );
            if (cleanMsgs.length === 0) {
              api.logger.info("memory-lancedb: no clean messages, skipping");
              return;
            }

            // Append to journal and get unprocessed delta
            const { journal: journalData, delta } = await journal.appendMessages(
              sessionKey,
              cleanMsgs,
            );
            api.logger.info(
              `memory-lancedb: journal state - context=${journalData.cleanContext.length}, processedIndex=${journalData.processedIndex}, delta=${delta.length}`,
            );
            if (delta.length === 0) {
              api.logger.info("memory-lancedb: delta is empty, skipping LLM call");
              return;
            }

            // Run LLM decision maker on the delta
            api.logger.info(
              `memory-lancedb: calling LLM decision maker with ${delta.length} new messages...`,
            );
            const { logs } = await decisionMaker.evaluate(journalData, delta);
            api.logger.info(`memory-lancedb: LLM returned ${logs.length} memory operations`);

            // Commit decisions (advance processedIndex + save logs)
            await journal.commitDecisions(sessionKey, logs, journalData.cleanContext.length);
            api.logger.info(
              `memory-lancedb: committed decisions, new processedIndex=${journalData.cleanContext.length}`,
            );

            if (logs.length > 0) {
              api.logger.info(
                `memory-lancedb: auto-captured ${logs.length} memory operations via LLM`,
              );
            }
          } catch (err) {
            api.logger.warn(`memory-lancedb: LLM capture failed: ${String(err)}`);
            // Log full stack if available
            if (err instanceof Error && err.stack) {
              api.logger.warn(`memory-lancedb: stack: ${err.stack}`);
            }
          }
        });
      }
    }

    // ========================================================================
    // Service
    // ========================================================================

    api.registerService({
      id: "memory-lancedb",
      start: () => {
        api.logger.info(
          `memory-lancedb: initialized (db: ${resolvedDbPath}, model: ${cfg.embedding.model}, baseUrl: ${cfg.embedding.baseUrl})`,
        );
      },
      stop: () => {
        api.logger.info("memory-lancedb: stopped");
      },
    });
  },
};

export default memoryPlugin;
