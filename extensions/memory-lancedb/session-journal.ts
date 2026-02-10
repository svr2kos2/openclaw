/**
 * SessionJournal – per-session conversation tracking with atomic persistence.
 *
 * Each session maintains a "clean context" (user/assistant plain-text messages)
 * and a log of memory operations performed during the session. The journal is
 * the single source of truth for incremental LLM evaluation (delta tracking).
 */

import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import fs from "node:fs/promises";
import { join, dirname } from "node:path";

// ============================================================================
// Types
// ============================================================================

export type CleanMessage = {
  role: "user" | "assistant";
  text: string;
};

export type MemoryLogEntry = {
  /** Index in cleanContext at the time this action was decided */
  deltaIndex: number;
  action: "store" | "forget" | "update";
  memoryId: string;
  text: string;
  timestamp: number;
};

export type JournalData = {
  /** How far into cleanContext the LLM has already evaluated */
  processedIndex: number;
  /** Full user/assistant plain-text message history for the session */
  cleanContext: CleanMessage[];
  /** Memory operations performed during this session */
  memoryLogs: MemoryLogEntry[];
};

// ============================================================================
// SessionJournal
// ============================================================================

export class SessionJournal {
  /** Per-session serialization queues (Promise chains) */
  private queues = new Map<string, Promise<unknown>>();
  /** In-memory cache of loaded journals */
  private cache = new Map<string, JournalData>();
  private readonly sessionsDir: string;

  constructor(dbPath: string) {
    this.sessionsDir = join(dbPath, "sessions");
    if (!existsSync(this.sessionsDir)) {
      mkdirSync(this.sessionsDir, { recursive: true });
    }
  }

  /**
   * Append new messages (deduped) and return the unprocessed delta.
   * Runs inside the session's serialization queue.
   */
  async appendMessages(
    sessionKey: string,
    messages: CleanMessage[],
  ): Promise<{ journal: JournalData; delta: CleanMessage[] }> {
    return this.enqueue(sessionKey, async () => {
      const journal = await this.load(sessionKey);

      // Deduplicate: skip messages already in the clean context
      const existing = new Set(journal.cleanContext.map((m) => `${m.role}:${m.text}`));
      const fresh = messages.filter((m) => !existing.has(`${m.role}:${m.text}`));

      if (fresh.length > 0) {
        journal.cleanContext.push(...fresh);
        await this.save(sessionKey, journal);
      }

      const delta = journal.cleanContext.slice(journal.processedIndex);
      return { journal, delta };
    });
  }

  /**
   * Atomically commit decision results: new memory logs + advance processedIndex.
   */
  async commitDecisions(
    sessionKey: string,
    logs: MemoryLogEntry[],
    newProcessedIndex: number,
  ): Promise<void> {
    return this.enqueue(sessionKey, async () => {
      const journal = await this.load(sessionKey);
      journal.memoryLogs.push(...logs);
      journal.processedIndex = newProcessedIndex;
      await this.save(sessionKey, journal);
    });
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

  private enqueue<T>(sessionKey: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.queues.get(sessionKey) ?? Promise.resolve();
    // Always run fn regardless of whether previous operation failed
    const next = prev.then(fn, fn);
    // Swallow errors in the chain to prevent propagation to next enqueued op
    this.queues.set(
      sessionKey,
      next.catch(() => {}),
    );
    return next;
  }

  private async load(sessionKey: string): Promise<JournalData> {
    const cached = this.cache.get(sessionKey);
    if (cached) return cached;

    const filePath = this.journalPath(sessionKey);
    try {
      const raw = await fs.readFile(filePath, "utf-8");
      const data = JSON.parse(raw) as JournalData;
      this.cache.set(sessionKey, data);
      return data;
    } catch {
      // File doesn't exist or is corrupt → start fresh
      const fresh: JournalData = {
        processedIndex: 0,
        cleanContext: [],
        memoryLogs: [],
      };
      this.cache.set(sessionKey, fresh);
      return fresh;
    }
  }

  /** Atomic write: write to temp file, then rename */
  private async save(sessionKey: string, data: JournalData): Promise<void> {
    this.cache.set(sessionKey, data);
    const filePath = this.journalPath(sessionKey);
    const tmpPath = `${filePath}.${randomUUID().slice(0, 8)}.tmp`;

    const dir = dirname(filePath);
    await fs.mkdir(dir, { recursive: true });

    await fs.writeFile(tmpPath, JSON.stringify(data, null, 2), "utf-8");
    await fs.rename(tmpPath, filePath);
  }

  /** Sanitize session key for use as a safe filename */
  private journalPath(sessionKey: string): string {
    const safe = sessionKey.replace(/[^a-zA-Z0-9_-]/g, "_");
    return join(this.sessionsDir, `${safe}.json`);
  }
}
