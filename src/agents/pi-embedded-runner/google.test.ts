import type { AgentTool } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import { sanitizeToolsForGoogle } from "./google.js";

describe("sanitizeToolsForGoogle", () => {
  const createTool = (parameters: Record<string, unknown>) =>
    ({
      name: "test",
      description: "test",
      parameters,
      execute: async () => ({ ok: true, content: [] }),
    }) as unknown as AgentTool;

  const expectFormatRemoved = (
    sanitized: AgentTool,
    key: "additionalProperties" | "patternProperties",
  ) => {
    const params = sanitized.parameters as {
      additionalProperties?: unknown;
      patternProperties?: unknown;
      properties?: Record<string, { format?: unknown }>;
    };
    expect(params[key]).toBeUndefined();
    expect(params.properties?.foo?.format).toBeUndefined();
  };

  it("strips unsupported schema keywords when modelId contains gemini", () => {
    const tool = createTool({
      type: "object",
      additionalProperties: false,
      properties: {
        foo: {
          type: "string",
          format: "uuid",
        },
      },
    });
    const [sanitized] = sanitizeToolsForGoogle({
      tools: [tool],
      modelId: "gemini-2.5-pro",
    });
    expectFormatRemoved(sanitized, "additionalProperties");
  });

  it("matches modelId case-insensitively", () => {
    const tool = createTool({
      type: "object",
      patternProperties: {
        "^foo$": { type: "string" },
      },
      properties: {
        foo: {
          type: "string",
          format: "uuid",
        },
      },
    });
    const [sanitized] = sanitizeToolsForGoogle({
      tools: [tool],
      modelId: "GeMiNi-Flash-2.0",
    });
    expectFormatRemoved(sanitized, "patternProperties");
  });

  it("returns original tools for non-gemini model ids", () => {
    const tool = createTool({
      type: "object",
      additionalProperties: false,
      properties: {
        foo: {
          type: "string",
          format: "uuid",
        },
      },
    });
    const sanitized = sanitizeToolsForGoogle({
      tools: [tool],
      modelId: "gpt-5",
    });

    expect(sanitized).toEqual([tool]);
    expect(sanitized[0]).toBe(tool);
  });
});
