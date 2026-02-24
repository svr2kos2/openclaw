import { describe, expect, test } from "vitest";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "test-key";

describe("memory-teto config + exports", () => {
  test("plugin exports basic metadata", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(memoryPlugin.id).toBe("memory-teto");
    expect(memoryPlugin.name).toBe("Memory (Teto)");
    expect(memoryPlugin.kind).toBe("memory");
    expect(memoryPlugin.configSchema).toBeDefined();
  });

  test("config schema parses embedding defaults", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const config = memoryPlugin.configSchema?.parse?.({
      embedding: {
        apiKey: OPENAI_API_KEY,
      },
    });

    expect(config).toBeDefined();
    expect(config?.embedding?.baseUrl).toBe("https://api.openai.com/v1");
    expect(config?.embedding?.model).toBe("text-embedding-3-small");
    expect(config?.embedding?.dimensions).toBe(1536);
  });

  test("config schema supports custom embedding endpoint + model", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const config = memoryPlugin.configSchema?.parse?.({
      embedding: {
        apiKey: OPENAI_API_KEY,
        baseUrl: "https://custom-embeddings.example.com/v1",
        model: "my-custom-embedding-model",
        dimensions: 1024,
      },
    });

    expect(config?.embedding?.baseUrl).toBe("https://custom-embeddings.example.com/v1");
    expect(config?.embedding?.model).toBe("my-custom-embedding-model");
    expect(config?.embedding?.dimensions).toBe(1024);
  });

  test("config schema requires dimensions for unknown embedding model", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(() => {
      memoryPlugin.configSchema?.parse?.({
        embedding: {
          apiKey: OPENAI_API_KEY,
          model: "unknown-embedding",
        },
      });
    }).toThrow("embedding.dimensions is required");
  });

  test("config schema parses anthropic chat config defaults", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    const config = memoryPlugin.configSchema?.parse?.({
      embedding: { apiKey: OPENAI_API_KEY },
      chat: { api: "anthropic-messages", apiKey: "sk-ant-xxx" },
    });

    expect(config?.chat?.api).toBe("anthropic-messages");
    expect(config?.chat?.model).toBe("claude-sonnet-4-5-20250929");
    expect(config?.chat?.baseUrl).toBe("https://api.anthropic.com");
  });

  test("config schema rejects invalid chat.api", async () => {
    const { default: memoryPlugin } = await import("./index.js");

    expect(() => {
      memoryPlugin.configSchema?.parse?.({
        embedding: { apiKey: OPENAI_API_KEY },
        chat: { api: "invalid-api", apiKey: "x" },
      });
    }).toThrow("chat.api must be one of");
  });

  test("legacy shouldCapture/detectCategory exports are removed", async () => {
    const mod = await import("./index.js");
    expect((mod as Record<string, unknown>).shouldCapture).toBeUndefined();
    expect((mod as Record<string, unknown>).detectCategory).toBeUndefined();
  });
});
