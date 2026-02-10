import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// ============================================================================
// Types
// ============================================================================

export type EmbeddingConfig = {
  provider: "openai";
  model: string;
  apiKey: string;
  baseUrl: string;
  dimensions: number;
};

export type ChatConfig = {
  model: string;
  apiKey: string;
  baseUrl: string;
};

export type MemoryConfig = {
  embedding: EmbeddingConfig;
  chat?: ChatConfig;
  dbPath: string;
  autoCapture: boolean;
  autoRecall: boolean;
};

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

// ============================================================================
// Defaults
// ============================================================================

const DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";
const DEFAULT_CHAT_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_CHAT_MODEL = "gpt-4o-mini";
const LEGACY_STATE_DIRS: string[] = [];

function resolveDefaultDbPath(): string {
  const home = homedir();
  const preferred = join(home, ".openclaw", "memory", "lancedb");
  try {
    if (fs.existsSync(preferred)) {
      return preferred;
    }
  } catch {
    // best-effort
  }

  for (const legacy of LEGACY_STATE_DIRS) {
    const candidate = join(home, legacy, "memory", "lancedb");
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // best-effort
    }
  }

  return preferred;
}

const DEFAULT_DB_PATH = resolveDefaultDbPath();

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
};

// ============================================================================
// Helpers
// ============================================================================

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

/** Resolve a string config value with env-var expansion and a fallback default. */
function resolveString(raw: unknown, fallback: string): string {
  if (typeof raw === "string" && raw.trim().length > 0) {
    const resolved = resolveEnvVars(raw.trim());
    return resolved.length > 0 ? resolved : fallback;
  }
  return fallback;
}

// ============================================================================
// Config resolvers
// ============================================================================

function resolveEmbeddingConfig(embedding: Record<string, unknown>): EmbeddingConfig {
  const apiKeyRaw = embedding.apiKey;
  if (typeof apiKeyRaw !== "string" || apiKeyRaw.trim().length === 0) {
    throw new Error("embedding.apiKey is required");
  }

  const model = resolveString(embedding.model, DEFAULT_EMBEDDING_MODEL);
  const baseUrl = resolveString(embedding.baseUrl, DEFAULT_EMBEDDING_BASE_URL);

  // Validate dimensions if provided
  if (typeof embedding.dimensions !== "undefined") {
    if (
      typeof embedding.dimensions !== "number" ||
      !Number.isFinite(embedding.dimensions) ||
      embedding.dimensions <= 0
    ) {
      throw new Error("embedding.dimensions must be a positive number");
    }
  }

  const providedDimensions =
    typeof embedding.dimensions === "number" ? Math.floor(embedding.dimensions) : undefined;
  const knownDimensions = EMBEDDING_DIMENSIONS[model];

  if (
    typeof providedDimensions === "number" &&
    typeof knownDimensions === "number" &&
    providedDimensions !== knownDimensions
  ) {
    throw new Error(
      `embedding.dimensions (${providedDimensions}) does not match expected (${knownDimensions}) for model ${model}`,
    );
  }

  const dimensions = providedDimensions ?? knownDimensions;
  if (!dimensions) {
    throw new Error(
      `embedding.dimensions is required for unknown model: ${model}. Set the vector size returned by your embedding endpoint.`,
    );
  }

  return {
    provider: "openai",
    model,
    apiKey: resolveEnvVars(apiKeyRaw.trim()),
    baseUrl,
    dimensions,
  };
}

function resolveChatConfig(chat: Record<string, unknown>): ChatConfig {
  const apiKeyRaw = chat.apiKey;
  if (typeof apiKeyRaw !== "string" || apiKeyRaw.trim().length === 0) {
    throw new Error("chat.apiKey is required");
  }

  return {
    model: resolveString(chat.model, DEFAULT_CHAT_MODEL),
    apiKey: resolveEnvVars(apiKeyRaw.trim()),
    baseUrl: resolveString(chat.baseUrl, DEFAULT_CHAT_BASE_URL),
  };
}

// ============================================================================
// Schema
// ============================================================================

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["embedding", "chat", "dbPath", "autoCapture", "autoRecall"],
      "memory config",
    );

    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding) {
      throw new Error("embedding config required");
    }
    assertAllowedKeys(embedding, ["apiKey", "model", "baseUrl", "dimensions"], "embedding config");

    let chat: ChatConfig | undefined;
    if (cfg.chat && typeof cfg.chat === "object" && !Array.isArray(cfg.chat)) {
      const chatRaw = cfg.chat as Record<string, unknown>;
      assertAllowedKeys(chatRaw, ["apiKey", "model", "baseUrl"], "chat config");
      chat = resolveChatConfig(chatRaw);
    }

    return {
      embedding: resolveEmbeddingConfig(embedding),
      chat,
      dbPath:
        typeof cfg.dbPath === "string" && cfg.dbPath.trim().length > 0
          ? resolveEnvVars(cfg.dbPath.trim())
          : DEFAULT_DB_PATH,
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
    };
  },
  uiHints: {
    "embedding.apiKey": {
      label: "Embedding API Key",
      sensitive: true,
      placeholder: "sk-...",
      help: "API key for your OpenAI-compatible embeddings endpoint (supports ${ENV_VAR})",
    },
    "embedding.baseUrl": {
      label: "Embedding Base URL",
      placeholder: DEFAULT_EMBEDDING_BASE_URL,
      help: "Base URL for an OpenAI-compatible embeddings API",
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_EMBEDDING_MODEL,
      help: "Embedding model name to call on the configured endpoint",
    },
    "embedding.dimensions": {
      label: "Vector Dimensions",
      placeholder: "1536",
      help: "Optional for known models; required for custom models",
      advanced: true,
    },
    "chat.apiKey": {
      label: "Chat API Key",
      sensitive: true,
      placeholder: "sk-...",
      help: "API key for the LLM used for memory decisions (supports ${ENV_VAR})",
    },
    "chat.baseUrl": {
      label: "Chat Base URL",
      placeholder: DEFAULT_CHAT_BASE_URL,
      help: "Base URL for an OpenAI-compatible chat API",
    },
    "chat.model": {
      label: "Chat Model",
      placeholder: DEFAULT_CHAT_MODEL,
      help: "Chat model for analyzing conversations and making memory decisions",
    },
    dbPath: {
      label: "Database Path",
      placeholder: "~/.openclaw/memory/lancedb",
      advanced: true,
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information via LLM analysis (requires chat config)",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
  },
};
