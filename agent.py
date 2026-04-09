"""BAU agent — Plan-Execute-ReAct loop + AgentMemory (Phase 2-3)."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Context budget helper
# ------------------------------------------------------------------

def allocate_context(total_budget: int) -> dict[str, int]:
    """Allocate context window tokens across system components.

    Args:
        total_budget: Total available tokens.

    Returns:
        Dict with token budgets for system_prompt, tools, memory, conversation.
    """
    return {
        "system_prompt": int(total_budget * 0.40),
        "tools":         int(total_budget * 0.30),
        "memory":        int(total_budget * 0.20),
        "conversation":  int(total_budget * 0.10),
    }


# ------------------------------------------------------------------
# RRF merge (standalone, used by archive_query)
# ------------------------------------------------------------------

def rrf_merge(
    bm25_results: list[tuple[int, float]],
    vec_results: list[tuple[int, float]],
    k: int = 60,
    w_bm25: float = 0.4,
    w_vec: float = 0.6,
) -> list[tuple[int, float]]:
    """Merge BM25 and vector results via Reciprocal Rank Fusion.

    Args:
        bm25_results: List of (memory_id, score) from FTS5.
        vec_results: List of (memory_id, distance) from sqlite-vec.
        k: RRF constant (default 60).
        w_bm25: Weight for BM25 results.
        w_vec: Weight for vector results.

    Returns:
        Sorted list of (memory_id, fused_score) descending by score.
    """
    scores: dict[int, float] = {}
    for rank, (mid, _) in enumerate(bm25_results):
        scores[mid] = scores.get(mid, 0) + w_bm25 / (k + rank + 1)
    for rank, (mid, _) in enumerate(vec_results):
        scores[mid] = scores.get(mid, 0) + w_vec / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ------------------------------------------------------------------
# FTS5 trigger SQL
# ------------------------------------------------------------------

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, memory_type)
    VALUES (new.id, new.content, new.memory_type);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
    VALUES ('delete', old.id, old.content, old.memory_type);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
    VALUES ('delete', old.id, old.content, old.memory_type);
    INSERT INTO memories_fts(rowid, content, memory_type)
    VALUES (new.id, new.content, new.memory_type);
END;
"""


# ------------------------------------------------------------------
# AgentMemory
# ------------------------------------------------------------------

class AgentMemory:
    """Hierarchical memory backed by SQLite + sqlite-vec + FTS5.

    Three layers:
    - Core memory: always-in-context blocks (persona, goals, task_ledger).
    - Recall memory: full conversation log, searchable and summarizable.
    - Archival memory: unlimited long-term with hybrid BM25 + vector retrieval.

    Args:
        db_path: Path to the SQLite database file.
        embedding_model: FastEmbed model name for vector embeddings.
        dim: Embedding dimensionality (default 384 for bge-small-en-v1.5).
    """

    def __init__(self, db_path: str, embedding_model: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        self._db_path = db_path
        self._embedding_model_name = embedding_model
        self._dim = dim
        self._embed_model: Any = None  # lazy-loaded FastEmbed instance

        self._conn = self._open_connection(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Connection & schema
    # ------------------------------------------------------------------

    @staticmethod
    def _open_connection(db_path: str) -> sqlite3.Connection:
        """Open a SQLite connection and load sqlite-vec extension.

        Args:
            db_path: Path to database file.

        Returns:
            Configured sqlite3.Connection.
        """
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)

        import sqlite_vec
        sqlite_vec.load(conn)

        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    def _init_db(self) -> None:
        """Create all tables, indexes, virtual tables, and FTS5 triggers."""
        cur = self._conn.cursor()

        cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_accessed REAL,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                memory_id INTEGER PRIMARY KEY,
                embedding float[{self._dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, memory_type,
                tokenize='porter unicode61',
                content=memories, content_rowid=id
            );

            CREATE TABLE IF NOT EXISTS core_memory (
                label TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                max_tokens INTEGER DEFAULT 500,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls JSON,
                timestamp REAL NOT NULL,
                summarized INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY,
                state_json TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tools (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                file_path TEXT NOT NULL,
                parameters_json TEXT,
                version INTEGER DEFAULT 1,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                source_hash TEXT NOT NULL,
                approved INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL,
                deprecated INTEGER DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS tool_embeddings USING vec0(
                rowid INTEGER PRIMARY KEY,
                embedding float[{self._dim}]
            );

            {_FTS_TRIGGERS}
        """)

        self._conn.commit()

    # ------------------------------------------------------------------
    # Embeddings (lazy-loaded)
    # ------------------------------------------------------------------

    def _get_embed_model(self) -> Any:
        """Return the cached FastEmbed model, loading on first use.

        Returns:
            FastEmbed TextEmbedding instance.
        """
        if self._embed_model is None:
            from fastembed import TextEmbedding
            self._embed_model = TextEmbedding(model_name=self._embedding_model_name)
        return self._embed_model

    def _embed(self, text: str) -> list[float]:
        """Compute embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats (length = self._dim).
        """
        model = self._get_embed_model()
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()

    def _pack_vector(self, vec: list[float]) -> bytes:
        """Pack a float vector into bytes for sqlite-vec.

        Args:
            vec: List of floats.

        Returns:
            Packed bytes blob.
        """
        return struct.pack(f"{len(vec)}f", *vec)

    # ------------------------------------------------------------------
    # Core memory (always-in-context blocks)
    # ------------------------------------------------------------------

    def core_get(self, label: str) -> str:
        """Get a core memory block by label.

        Args:
            label: Block label (e.g. 'persona', 'goals', 'task_ledger').

        Returns:
            The block value, or empty string if not set.
        """
        row = self._conn.execute(
            "SELECT value FROM core_memory WHERE label = ?", (label,)
        ).fetchone()
        return row[0] if row else ""

    def core_set(self, label: str, value: str) -> None:
        """Set or overwrite a core memory block.

        Args:
            label: Block label.
            value: New value for the block.
        """
        self._conn.execute(
            """INSERT INTO core_memory (label, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(label) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (label, value, time.time()),
        )
        self._conn.commit()

    def core_replace(self, label: str, old: str, new: str) -> bool:
        """Replace a substring in a core memory block (MemGPT pattern).

        Args:
            label: Block label.
            old: Substring to find.
            new: Replacement string.

        Returns:
            True if the replacement was made, False if old not found.
        """
        current = self.core_get(label)
        if old not in current:
            return False
        updated = current.replace(old, new, 1)
        self.core_set(label, updated)
        return True

    def core_render(self) -> str:
        """Render all core memory blocks as a formatted string for the system prompt.

        Returns:
            Formatted string with all core memory blocks.
        """
        rows = self._conn.execute(
            "SELECT label, value FROM core_memory ORDER BY label"
        ).fetchall()
        if not rows:
            return "(no core memories)"
        parts: list[str] = []
        for label, value in rows:
            parts.append(f"<{label}>\n{value}\n</{label}>")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Recall memory (conversation log)
    # ------------------------------------------------------------------

    def recall_append(self, role: str, content: str, tool_calls: list[dict] | None = None) -> int:
        """Append a message to the conversation log.

        Args:
            role: Message role ('user', 'assistant', 'tool').
            content: Message content.
            tool_calls: Optional tool call data.

        Returns:
            Row id of the inserted message.
        """
        tc_json = json.dumps(tool_calls) if tool_calls else None
        cur = self._conn.execute(
            "INSERT INTO conversation (role, content, tool_calls, timestamp) VALUES (?, ?, ?, ?)",
            (role, content, tc_json, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def recall_recent(self, n: int = 20) -> list[dict]:
        """Retrieve the most recent conversation messages.

        Args:
            n: Number of recent messages to return.

        Returns:
            List of message dicts ordered oldest-first.
        """
        rows = self._conn.execute(
            "SELECT id, role, content, tool_calls, timestamp, summarized FROM conversation "
            "ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        rows.reverse()
        return [
            {
                "id": r[0], "role": r[1], "content": r[2],
                "tool_calls": json.loads(r[3]) if r[3] else None,
                "timestamp": r[4], "summarized": bool(r[5]),
            }
            for r in rows
        ]

    def recall_search(self, query: str, k: int = 5) -> list[dict]:
        """Search conversation history by keyword.

        Args:
            query: Search query.
            k: Max results to return.

        Returns:
            List of matching message dicts.
        """
        # Use LIKE for simple substring search on conversation table
        rows = self._conn.execute(
            "SELECT id, role, content, tool_calls, timestamp FROM conversation "
            "WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", k),
        ).fetchall()
        return [
            {
                "id": r[0], "role": r[1], "content": r[2],
                "tool_calls": json.loads(r[3]) if r[3] else None,
                "timestamp": r[4],
            }
            for r in rows
        ]

    async def recall_summarize_old(self, llm: Any, keep_recent: int = 20) -> None:
        """Summarize older conversation messages to save context.

        Keeps the most recent `keep_recent` messages intact,
        summarizes the rest via an LLM call, and marks them as summarized.

        Args:
            llm: LLMClient instance for summarization.
            keep_recent: Number of recent messages to preserve.
        """
        # Find messages eligible for summarization
        rows = self._conn.execute(
            "SELECT id, role, content FROM conversation "
            "WHERE summarized = 0 ORDER BY id",
        ).fetchall()

        if len(rows) <= keep_recent:
            return

        to_summarize = rows[:-keep_recent]
        text_block = "\n".join(f"{r[1]}: {r[2]}" for r in to_summarize)

        result = await llm.complete([
            {"role": "system", "content": "Summarize this conversation segment concisely. Keep key facts, decisions, and tool results. Output only the summary."},
            {"role": "user", "content": text_block},
        ])

        # Mark old messages as summarized
        ids = [r[0] for r in to_summarize]
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"UPDATE conversation SET summarized = 1 WHERE id IN ({placeholders})", ids
        )

        # Store summary as an archival memory
        self.archive_store(
            content=f"[Conversation summary] {result.text}",
            memory_type="episodic",
            metadata={"source": "recall_summarize", "message_ids": ids},
            importance=0.6,
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Archival memory (long-term, vector + BM25)
    # ------------------------------------------------------------------

    def archive_store(
        self,
        content: str,
        memory_type: str = "semantic",
        metadata: dict | None = None,
        importance: float = 0.5,
    ) -> int:
        """Store a new memory in archival storage with embedding.

        Args:
            content: Memory content text.
            memory_type: One of 'episodic', 'semantic', 'procedural', 'task_result'.
            metadata: Optional JSON metadata.
            importance: Importance score 0-1 (default 0.5).

        Returns:
            Row id of the stored memory.
        """
        vec = self._embed(content)
        blob = self._pack_vector(vec)
        now = time.time()

        cur = self._conn.execute(
            "INSERT INTO memories (memory_type, content, metadata, importance, created_at, last_accessed, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_type, content, json.dumps(metadata) if metadata else None, importance, now, now, blob),
        )
        mem_id = cur.lastrowid

        # Insert into vec_memories for vector search
        self._conn.execute(
            "INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)",
            (mem_id, blob),
        )

        self._conn.commit()
        return mem_id  # type: ignore[return-value]

    def archive_query(
        self,
        query: str,
        k: int = 5,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Query archival memory using hybrid BM25 + vector retrieval with RRF.

        Pipeline: BM25 top-20 + Vector top-20 → RRF merge → recency decay
        → optional FlashRank rerank → top k.

        Args:
            query: Search query text.
            k: Number of results to return.
            memory_type: Optional filter by memory type.

        Returns:
            List of memory dicts sorted by relevance.
        """
        # 1. BM25 search via FTS5
        #    Use OR between terms for better recall; quote each token to
        #    avoid FTS5 syntax errors from special characters.
        tokens = query.split()
        fts_query = " OR ".join(f'"{t}"' for t in tokens) if tokens else query

        try:
            if memory_type:
                bm25_rows = self._conn.execute(
                    "SELECT m.id, -bm25(memories_fts) AS score "
                    "FROM memories_fts f JOIN memories m ON f.rowid = m.id "
                    "WHERE memories_fts MATCH ? AND m.memory_type = ? "
                    "ORDER BY score DESC LIMIT 20",
                    (fts_query, memory_type),
                ).fetchall()
            else:
                bm25_rows = self._conn.execute(
                    "SELECT m.id, -bm25(memories_fts) AS score "
                    "FROM memories_fts f JOIN memories m ON f.rowid = m.id "
                    "WHERE memories_fts MATCH ? "
                    "ORDER BY score DESC LIMIT 20",
                    (fts_query,),
                ).fetchall()
        except sqlite3.OperationalError:
            bm25_rows = []  # malformed query — fall back to vector-only

        bm25_results: list[tuple[int, float]] = [(r[0], r[1]) for r in bm25_rows]

        # 2. Vector search via sqlite-vec
        query_vec = self._embed(query)
        query_blob = self._pack_vector(query_vec)

        vec_rows = self._conn.execute(
            "SELECT memory_id, distance FROM vec_memories "
            "WHERE embedding MATCH ? AND k = 20",
            (query_blob,),
        ).fetchall()

        vec_results: list[tuple[int, float]] = [(r[0], r[1]) for r in vec_rows]

        # Filter by memory_type if requested (vec_memories doesn't have the column)
        if memory_type and vec_results:
            type_ids = {r[0] for r in self._conn.execute(
                "SELECT id FROM memories WHERE memory_type = ?", (memory_type,)
            ).fetchall()}
            vec_results = [(mid, d) for mid, d in vec_results if mid in type_ids]

        # 3. Reciprocal Rank Fusion
        fused = rrf_merge(bm25_results, vec_results)
        if not fused:
            return []

        # 4. Recency decay: score *= exp(-age_days / 30) * importance
        now = time.time()
        mem_ids = [mid for mid, _ in fused]
        placeholders = ",".join("?" * len(mem_ids))
        meta_rows = self._conn.execute(
            f"SELECT id, created_at, importance FROM memories WHERE id IN ({placeholders})",
            mem_ids,
        ).fetchall()
        meta_map = {r[0]: (r[1], r[2]) for r in meta_rows}

        decayed: list[tuple[int, float]] = []
        for mid, score in fused:
            if mid in meta_map:
                created_at, importance = meta_map[mid]
                age_days = (now - created_at) / 86400.0
                decay = math.exp(-age_days / 30.0) * importance
                decayed.append((mid, score * decay))

        decayed.sort(key=lambda x: -x[1])

        # 5. Optional FlashRank rerank (top 10 candidates)
        candidates = decayed[:max(10, k)]
        try:
            from flashrank import Ranker, RerankRequest

            cand_ids = [mid for mid, _ in candidates]
            cand_rows = self._conn.execute(
                f"SELECT id, content FROM memories WHERE id IN ({','.join('?' * len(cand_ids))})",
                cand_ids,
            ).fetchall()
            cand_content = {r[0]: r[1] for r in cand_rows}

            passages = [{"id": mid, "text": cand_content.get(mid, "")} for mid, _ in candidates]
            ranker = Ranker()
            reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
            candidates = [(int(r["id"]), r["score"]) for r in reranked]
        except (ImportError, Exception):
            pass  # FlashRank is optional — skip silently

        # Take top k
        top_ids = [mid for mid, _ in candidates[:k]]
        if not top_ids:
            return []

        # Fetch full rows
        placeholders = ",".join("?" * len(top_ids))
        rows = self._conn.execute(
            f"SELECT id, memory_type, content, metadata, importance, access_count, "
            f"created_at, last_accessed FROM memories WHERE id IN ({placeholders})",
            top_ids,
        ).fetchall()
        row_map = {r[0]: r for r in rows}

        # Update access_count and last_accessed
        self._conn.execute(
            f"UPDATE memories SET access_count = access_count + 1, last_accessed = ? "
            f"WHERE id IN ({placeholders})",
            [now, *top_ids],
        )
        self._conn.commit()

        # Return in ranked order
        results: list[dict] = []
        for mid in top_ids:
            if mid in row_map:
                r = row_map[mid]
                results.append({
                    "id": r[0],
                    "memory_type": r[1],
                    "content": r[2],
                    "metadata": json.loads(r[3]) if r[3] else None,
                    "importance": r[4],
                    "access_count": r[5],
                    "created_at": r[6],
                    "last_accessed": r[7],
                })
        return results

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate(self, new_content: str, llm: Any) -> str:
        """Check for similar memories and decide: skip / keep / merge / replace / update.

        Searches for the top 3 similar existing memories, then asks the LLM
        to arbitrate what action to take. Executes the chosen action atomically.

        Args:
            new_content: The new memory content to potentially store.
            llm: LLMClient instance for arbitration.

        Returns:
            The action taken: 'skip', 'keep', 'merge', 'replace', or 'update'.
        """
        # Find similar existing memories
        query_vec = self._embed(new_content)
        query_blob = self._pack_vector(query_vec)

        similar_rows = self._conn.execute(
            "SELECT memory_id, distance FROM vec_memories "
            "WHERE embedding MATCH ? AND k = 3",
            (query_blob,),
        ).fetchall()

        if not similar_rows:
            # No existing memories — just store
            self.archive_store(new_content)
            return "keep"

        # Fetch content of similar memories
        sim_ids = [r[0] for r in similar_rows]
        placeholders = ",".join("?" * len(sim_ids))
        existing = self._conn.execute(
            f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
            sim_ids,
        ).fetchall()

        if not existing:
            self.archive_store(new_content)
            return "keep"

        existing_block = "\n".join(f"[ID {r[0]}] {r[1]}" for r in existing)

        prompt = (
            "You are a memory consolidation agent. Given a NEW memory and EXISTING similar memories, "
            "choose exactly one action.\n\n"
            f"<existing_memories>\n{existing_block}\n</existing_memories>\n\n"
            f"<new_memory>\n{new_content}\n</new_memory>\n\n"
            'Respond with ONLY a JSON object: {"action": "skip|keep|merge|replace|update", '
            '"target_id": <id or null>, "merged_content": "<text or null>"}\n'
            "- skip: new memory is redundant, discard it\n"
            "- keep: new memory is distinct, store it alongside existing\n"
            "- merge: combine new + target into merged_content, replace target\n"
            "- replace: new supersedes target, delete target and store new\n"
            "- update: enrich target with info from new, use merged_content"
        )

        result = await llm.complete([
            {"role": "user", "content": prompt},
        ], response_format={"type": "json_object"})

        try:
            decision = json.loads(result.text)
        except (json.JSONDecodeError, TypeError):
            # Fallback: just keep
            self.archive_store(new_content)
            return "keep"

        action = decision.get("action", "keep")
        target_id = decision.get("target_id")
        merged = decision.get("merged_content")

        match action:
            case "skip":
                pass
            case "keep":
                self.archive_store(new_content)
            case "merge" if target_id and merged:
                self._delete_memory(target_id)
                self.archive_store(merged)
            case "replace" if target_id:
                self._delete_memory(target_id)
                self.archive_store(new_content)
            case "update" if target_id and merged:
                self._delete_memory(target_id)
                self.archive_store(merged)
            case _:
                self.archive_store(new_content)
                action = "keep"

        return action

    def _delete_memory(self, memory_id: int) -> None:
        """Delete a memory from all tables (memories, vec_memories; FTS via trigger).

        Args:
            memory_id: The memory row id to delete.
        """
        self._conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint(self, state: dict, iteration: int) -> None:
        """Save an agent state checkpoint.

        Args:
            state: Serializable state dict.
            iteration: Current iteration number.
        """
        self._conn.execute(
            "INSERT INTO checkpoints (state_json, iteration, timestamp) VALUES (?, ?, ?)",
            (json.dumps(state), iteration, time.time()),
        )
        self._conn.commit()

    def latest_checkpoint(self) -> dict | None:
        """Retrieve the most recent checkpoint.

        Returns:
            State dict from the latest checkpoint, or None.
        """
        row = self._conn.execute(
            "SELECT state_json FROM checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return json.loads(row[0]) if row else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return memory system statistics.

        Returns:
            Dict with counts, db_size_bytes, and oldest_entry timestamp.
        """
        counts: dict[str, int] = {}
        for table in ("memories", "conversation", "checkpoints", "core_memory"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0] if row else 0

        import os
        db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0

        oldest = self._conn.execute(
            "SELECT MIN(created_at) FROM memories"
        ).fetchone()

        return {
            "counts": counts,
            "db_size_bytes": db_size,
            "oldest_entry": oldest[0] if oldest and oldest[0] else None,
        }

    # ------------------------------------------------------------------
    # Tool metadata
    # ------------------------------------------------------------------

    def tool_register_meta(
        self,
        name: str,
        description: str,
        file_path: str,
        params_json: str,
        source_hash: str,
    ) -> None:
        """Register a generated tool's metadata and embed its description.

        Args:
            name: Tool function name.
            description: One-line description.
            file_path: Path to the .py file.
            params_json: JSON Schema of parameters.
            source_hash: SHA-256 hash of the source code.
        """
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO tools "
            "(name, description, file_path, parameters_json, source_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, description, file_path, params_json, source_hash, now),
        )
        # Embed description into tool_embeddings
        vec = self._embed(description)
        blob = self._pack_vector(vec)
        # Use a stable rowid derived from the name hash
        rowid = abs(hash(name)) % (2**31)
        self._conn.execute(
            "INSERT OR REPLACE INTO tool_embeddings (rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        # Store the rowid mapping so we can look up name from rowid
        self._conn.execute(
            "UPDATE tools SET version = version WHERE name = ?", (name,)
        )
        self._conn.commit()

    def tool_search(self, query: str, k: int = 5) -> list[str]:
        """Search generated tools by description similarity with success-rate weighting.

        Args:
            query: Natural language query.
            k: Number of results.

        Returns:
            List of tool names ranked by weighted similarity.
        """
        vec = self._embed(query)
        blob = self._pack_vector(vec)

        # Vector search on tool embeddings
        vec_rows = self._conn.execute(
            "SELECT rowid, distance FROM tool_embeddings "
            "WHERE embedding MATCH ? AND k = ?",
            (blob, min(k * 2, 20)),
        ).fetchall()

        if not vec_rows:
            return []

        # Fetch all non-deprecated tools with their rowid mapping
        tools_rows = self._conn.execute(
            "SELECT name, description, success_count, failure_count FROM tools "
            "WHERE deprecated = 0"
        ).fetchall()
        # Build rowid → tool mapping
        rowid_to_tool: dict[int, tuple[str, int, int]] = {}
        for tname, _desc, sc, fc in tools_rows:
            rid = abs(hash(tname)) % (2**31)
            rowid_to_tool[rid] = (tname, sc, fc)

        # Apply success-rate weighting: score = similarity * (1 + 0.1 * success_rate)
        scored: list[tuple[str, float]] = []
        for rowid, distance in vec_rows:
            if rowid not in rowid_to_tool:
                continue
            tname, sc, fc = rowid_to_tool[rowid]
            similarity = max(0.0, 1.0 - distance)
            success_rate = sc / max(1, sc + fc)
            final_score = similarity * (1 + 0.1 * success_rate)
            scored.append((tname, final_score))

        scored.sort(key=lambda x: -x[1])
        return [name for name, _ in scored[:k]]

    def tool_update_stats(self, name: str, success: bool) -> None:
        """Update usage stats for a generated tool.

        Args:
            name: Tool name.
            success: Whether the call succeeded.
        """
        col = "success_count" if success else "failure_count"
        self._conn.execute(
            f"UPDATE tools SET {col} = {col} + 1, usage_count = usage_count + 1, "
            f"last_used = ? WHERE name = ?",
            (time.time(), name),
        )
        self._conn.commit()

    def list_tools(self, deprecated: bool = False) -> list[dict]:
        """List generated tools from the metadata table.

        Args:
            deprecated: If True, include deprecated tools.

        Returns:
            List of tool metadata dicts.
        """
        if deprecated:
            rows = self._conn.execute(
                "SELECT name, description, file_path, parameters_json, "
                "success_count, failure_count, usage_count, approved, deprecated "
                "FROM tools"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT name, description, file_path, parameters_json, "
                "success_count, failure_count, usage_count, approved, deprecated "
                "FROM tools WHERE deprecated = 0"
            ).fetchall()
        return [
            {
                "name": r[0], "description": r[1], "file_path": r[2],
                "parameters_json": r[3], "success_count": r[4],
                "failure_count": r[5], "usage_count": r[6],
                "approved": bool(r[7]), "deprecated": bool(r[8]),
            }
            for r in rows
        ]

    def tool_count_active(self) -> int:
        """Count non-deprecated generated tools.

        Returns:
            Number of active generated tools.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM tools WHERE deprecated = 0"
        ).fetchone()
        return row[0] if row else 0

    def tool_deprecate_stale(self, max_age_days: int = 30) -> int:
        """Deprecate tools unused for max_age_days with zero usage since creation.

        Args:
            max_age_days: Age threshold in days.

        Returns:
            Number of tools deprecated.
        """
        cutoff = time.time() - (max_age_days * 86400)
        cur = self._conn.execute(
            "UPDATE tools SET deprecated = 1 "
            "WHERE deprecated = 0 AND usage_count = 0 AND created_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def tool_set_approved(self, name: str) -> None:
        """Mark a tool as approved.

        Args:
            name: Tool name.
        """
        self._conn.execute(
            "UPDATE tools SET approved = 1 WHERE name = ?", (name,),
        )
        self._conn.commit()

    def tool_get(self, name: str) -> dict | None:
        """Get a single tool's metadata.

        Args:
            name: Tool name.

        Returns:
            Tool metadata dict, or None.
        """
        row = self._conn.execute(
            "SELECT name, description, file_path, parameters_json, "
            "success_count, failure_count, usage_count, approved, deprecated, source_hash "
            "FROM tools WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            return None
        return {
            "name": row[0], "description": row[1], "file_path": row[2],
            "parameters_json": row[3], "success_count": row[4],
            "failure_count": row[5], "usage_count": row[6],
            "approved": bool(row[7]), "deprecated": bool(row[8]),
            "source_hash": row[9],
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ------------------------------------------------------------------
# Cosine similarity helper
# ------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ------------------------------------------------------------------
# Agent state
# ------------------------------------------------------------------

@dataclass(slots=True)
class AgentState:
    """Mutable state for a single agent run."""

    goal: str
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    completed: list[dict] = field(default_factory=list)
    iteration: int = 0
    scratchpad: list[dict] = field(default_factory=list)
    seen_actions: dict[str, int] = field(default_factory=dict)
    drift_baseline: list[float] | None = None
    started_at: float = field(default_factory=time.time)
    cost_so_far: float = 0.0
    status: str = "running"


# ------------------------------------------------------------------
# Agent — hybrid Plan-Execute-ReAct loop
# ------------------------------------------------------------------

class Agent:
    """Hybrid Plan-Execute-ReAct autonomous agent.

    Event vocabulary (emitted via on_event callback):
        goal:        {"goal": str}
        plan:        {"steps": list[str]}
        replan:      {"reason": str}
        step_start:  {"index": int, "description": str}
        step_done:   {"index": int, "result": dict}
        thinking:    {"text": str}
        tool_call:   {"name": str, "args": dict}
        tool_result: {"name": str, "result": str}
        memory_op:   {"op": str, "key": str}
        warning:     {"message": str}
        error:       {"message": str}
        done:        {"answer": str, "status": str}

    Args:
        llm: LLMClient instance.
        memory: AgentMemory instance.
        tools: ToolRegistry instance.
        prompts: Dict mapping prompt name to template string.
        config: Parsed config dict.
        on_event: Callback ``f(event_type: str, payload: dict)``.
    """

    def __init__(
        self,
        llm: Any,
        memory: Any,
        tools: Any,
        prompts: dict[str, str],
        config: dict,
        on_event: Any = lambda *a, **kw: None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._tools = tools
        self._prompts = prompts
        self._config = config
        self._on_event = on_event

        limits = config.get("limits", {})
        self._max_iterations: int = limits.get("max_iterations", 30)
        self._warning_ratio: float = limits.get("iteration_warning_at", 0.7)
        self._max_inner: int = limits.get("max_inner_iterations", 8)
        self._max_replans: int = limits.get("max_replans", 3)
        self._interrupted: bool = False

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict) -> None:
        """Emit an event to the UI callback."""
        self._on_event(event_type, payload)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render(template: str, **variables: str) -> str:
        """Replace ``{{key}}`` placeholders in a template."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _build_system_prompt(self) -> str:
        """Render the system prompt with core memory and tool names."""
        from datetime import date as _date

        core_mem = self._memory.core_render()
        tool_names = ", ".join(self._tools.names()) or "(none)"
        template = self._prompts.get("system", "You are BAU, an autonomous AI agent.")
        return self._render(
            template, tools=tool_names, date=str(_date.today()), memory_block=core_mem,
        )

    def _format_plan(self, state: AgentState) -> str:
        """Format the plan with a marker on the active step."""
        lines: list[str] = []
        for i, step in enumerate(state.plan):
            if i < state.current_step:
                lines.append(f"  [done] {step}")
            elif i == state.current_step:
                lines.append(f"  -> {i + 1}. {step}")
            else:
                lines.append(f"     {i + 1}. {step}")
        return "\n".join(lines)

    @staticmethod
    def _format_completed(state: AgentState) -> str:
        """Format completed steps as a bullet list."""
        if not state.completed:
            return "(none yet)"
        return "\n".join(
            f"- {c['step']}: {c['result'].get('text', 'done')}"
            for c in state.completed
        )

    # ------------------------------------------------------------------
    # Plan parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(text: str) -> list[str]:
        """Parse a plan from JSON or numbered-list format.

        Args:
            text: Raw LLM output (JSON preferred, numbered list fallback).

        Returns:
            List of step description strings.
        """
        # Try JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "steps" in data:
                steps = data["steps"]
                if isinstance(steps, list):
                    return [str(s) for s in steps if s]
            if isinstance(data, list):
                return [str(s) for s in data if s]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: numbered list
        lines = text.strip().splitlines()
        steps: list[str] = []
        for line in lines:
            m = re.match(r"^\d+[.)]\s*(.+)", line.strip())
            if m:
                steps.append(m.group(1))
        return steps if steps else [text.strip()]

    # ------------------------------------------------------------------
    # State serialization (JSON-safe subset)
    # ------------------------------------------------------------------

    @staticmethod
    def _state_to_dict(state: AgentState) -> dict:
        """Convert AgentState to a JSON-serializable dict."""
        return {
            "goal": state.goal,
            "plan": state.plan,
            "current_step": state.current_step,
            "completed": state.completed,
            "iteration": state.iteration,
            "started_at": state.started_at,
            "cost_so_far": state.cost_so_far,
            "status": state.status,
        }

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _check_iteration_cap(self, state: AgentState) -> str:
        """Check outer iteration budget.

        Returns:
            ``"stop"`` if hard limit reached, ``"warn"`` if approaching, else ``""``.
        """
        if state.iteration >= self._max_iterations:
            return "stop"
        if state.iteration >= int(self._max_iterations * self._warning_ratio):
            return "warn"
        return ""

    @staticmethod
    def _check_loop(state: AgentState, action_hash: str) -> str:
        """Track repeated actions within a step.

        Returns:
            ``"replan"`` at 5+, ``"nudge"`` at 3+, else ``""``.
        """
        state.seen_actions[action_hash] = state.seen_actions.get(action_hash, 0) + 1
        count = state.seen_actions[action_hash]
        if count >= 5:
            return "replan"
        if count >= 3:
            return "nudge"
        return ""

    def _check_drift(self, state: AgentState, latest_text: str) -> bool:
        """Detect goal drift via cosine distance from baseline embedding.

        Returns:
            True if drift exceeds 0.30.
        """
        try:
            vec = self._memory._embed(latest_text)
        except Exception:
            return False
        if state.drift_baseline is None:
            state.drift_baseline = vec
            return False
        sim = _cosine_similarity(state.drift_baseline, vec)
        return (1.0 - sim) > 0.30

    def _validate_tool_call(self, name: str, args: dict) -> str | None:
        """Validate a tool call against the registry.

        Returns:
            Error message if invalid, None if OK.
        """
        if name not in self._tools.names():
            available = ", ".join(self._tools.names())
            return f"Unknown tool '{name}'. Available tools: {available}"
        return None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def _plan(self, goal: str) -> list[str]:
        """Create an initial plan for the goal.

        Args:
            goal: User's high-level goal.

        Returns:
            Ordered list of step descriptions (3-8 typical).
        """
        tool_names = ", ".join(self._tools.names()) or "(none)"
        rendered = self._render(
            self._prompts.get("planner", "Break the goal into steps."),
            tools=tool_names,
        )
        messages = [
            {"role": "system", "content": rendered},
            {"role": "user", "content": (
                f"Goal: {goal}\n\n"
                'Respond with a JSON object: {"steps": ["step 1", "step 2", ...]}'
            )},
        ]
        try:
            result = await self._llm.complete(
                messages, response_format={"type": "json_object"},
            )
            steps = self._parse_plan(result.text)
            return steps if steps else [goal]
        except Exception as e:
            self._emit("error", {"message": f"Planning failed: {e}"})
            return [goal]

    async def _replan(
        self, goal: str, completed: list[dict], reason: str,
    ) -> list[str]:
        """Create a revised plan accounting for completed work and failure.

        Args:
            goal: Original goal.
            completed: List of completed step dicts.
            reason: Why replanning was triggered.

        Returns:
            New ordered list of step descriptions.
        """
        tool_names = ", ".join(self._tools.names()) or "(none)"
        rendered = self._render(
            self._prompts.get("planner", "Break the goal into steps."),
            tools=tool_names,
        )
        completed_text = "\n".join(
            f"- {c['step']}: {c['result'].get('text', 'done')}"
            for c in completed
        ) or "(none)"
        messages = [
            {"role": "system", "content": rendered},
            {"role": "user", "content": (
                f"Goal: {goal}\n\n"
                f"Already completed:\n{completed_text}\n\n"
                f"Reason for replanning: {reason}\n\n"
                "Create a new plan for the remaining work. "
                'Respond with a JSON object: {"steps": ["step 1", "step 2", ...]}'
            )},
        ]
        try:
            result = await self._llm.complete(
                messages, response_format={"type": "json_object"},
            )
            steps = self._parse_plan(result.text)
            return steps if steps else [goal]
        except Exception as e:
            self._emit("error", {"message": f"Replanning failed: {e}"})
            return [goal]

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(self, state: AgentState) -> str:
        """Summarize completed work into a final answer.

        Args:
            state: Final agent state.

        Returns:
            Human-readable answer string.
        """
        if not state.completed:
            return f"Task {state.status}. No steps were completed."

        completed_text = "\n".join(
            f"Step {i + 1} ({c['step']}): {c['result'].get('text', 'done')}"
            for i, c in enumerate(state.completed)
        )
        messages = [
            {"role": "system", "content": (
                "Summarize the completed work into a concise final answer "
                "for the user."
            )},
            {"role": "user", "content": (
                f"Goal: {state.goal}\n\n"
                f"Completed work:\n{completed_text}\n\n"
                "Provide a clear, concise answer."
            )},
        ]
        try:
            result = await self._llm.complete(messages)
            return result.text
        except Exception:
            return f"Task {state.status}. {len(state.completed)} steps completed."

    # ------------------------------------------------------------------
    # Tool RAG (filter tools per turn when library is large)
    # ------------------------------------------------------------------

    # Core tools always visible to the LLM regardless of library size
    ALWAYS_LOADED = frozenset({
        "web_search", "web_fetch", "memory_query", "memory_store",
        "todo_write", "ask_user", "create_tool",
    })

    def _select_tools(self, query: str) -> list[dict]:
        """Select relevant tool schemas for the current turn.

        When the registry has <= 15 tools, returns all schemas.
        Otherwise, returns always-loaded tools + top-K relevant via Tool RAG.

        Args:
            query: The current step/query to match tools against.

        Returns:
            List of OpenAI-format tool schema dicts.
        """
        all_names = self._tools.names()
        if len(all_names) <= 15:
            return self._tools.schemas()

        # Always-loaded core tools
        always = self._tools.schemas(
            names=[n for n in all_names if n in self.ALWAYS_LOADED],
        )
        seen = {s["function"]["name"] for s in always}

        # Tool RAG: search generated tools by description similarity
        relevant_names = self._memory.tool_search(query, k=8)
        relevant = self._tools.schemas(names=relevant_names)

        return always + [s for s in relevant if s["function"]["name"] not in seen]

    # ------------------------------------------------------------------
    # ReAct sub-loop (one plan step)
    # ------------------------------------------------------------------

    async def _execute_step_react(self, state: AgentState, step: str) -> dict:
        """Execute a single plan step via a ReAct sub-loop.

        Args:
            state: Current agent state (scratchpad and seen_actions are reset).
            step: Step description to execute.

        Returns:
            Result dict with keys ``text``, ``needs_replan``, ``failed``,
            and optionally ``reason``.
        """
        state.scratchpad = []
        state.seen_actions = {}

        system_prompt = self._build_system_prompt()
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Execute this step: {step}\n\n"
                f"Full plan:\n{self._format_plan(state)}\n\n"
                f"Completed so far:\n{self._format_completed(state)}"
            )},
        ]
        tool_schemas = self._select_tools(step) or None

        for _inner in range(self._max_inner):
            messages = base_messages + state.scratchpad

            try:
                result = await self._llm.complete(messages, tools=tool_schemas)
            except Exception as e:
                self._emit("error", {"message": f"LLM error: {e}"})
                return {"failed": True, "text": str(e), "needs_replan": False}

            state.cost_so_far += result.cost

            # Text-only response → step is done
            if not result.tool_calls:
                if result.text:
                    self._emit("thinking", {"text": result.text})
                return {
                    "text": result.text, "needs_replan": False, "failed": False,
                }

            # Build assistant message for scratchpad (arguments as JSON strings)
            tc_for_msg = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": (
                            json.dumps(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], dict)
                            else tc["function"]["arguments"]
                        ),
                    },
                }
                for tc in result.tool_calls
            ]
            state.scratchpad.append({
                "role": "assistant",
                "content": result.text or None,
                "tool_calls": tc_for_msg,
            })

            # Process each tool call
            for tc in result.tool_calls:
                tc_id = tc["id"]
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                self._emit("tool_call", {"name": name, "args": args})

                # Validate against registry
                error = self._validate_tool_call(name, args)
                if error:
                    state.scratchpad.append({
                        "role": "tool", "tool_call_id": tc_id, "content": error,
                    })
                    self._emit("tool_result", {"name": name, "result": error})
                    continue

                # Loop guard
                action_hash = f"{name}:{json.dumps(args, sort_keys=True)}"
                loop_status = self._check_loop(state, action_hash)
                if loop_status == "replan":
                    return {
                        "needs_replan": True,
                        "reason": "repeated action detected 5+ times",
                        "failed": False,
                    }

                nudge = ""
                if loop_status == "nudge":
                    nudge = (
                        "\n[System: You repeated the same action. "
                        "Try a different approach or call a different tool.]"
                    )

                # Execute the tool (ToolRegistry.call is async)
                try:
                    tool_result = await self._tools.call(name, args)
                    result_str = str(tool_result) + nudge
                except Exception as e:
                    result_str = (
                        f"Tool {name} raised {type(e).__name__}: {e}" + nudge
                    )

                state.scratchpad.append({
                    "role": "tool", "tool_call_id": tc_id, "content": result_str,
                })
                self._emit("tool_result", {"name": name, "result": result_str})

            # Drift check on reasoning text
            if result.text and self._check_drift(state, result.text):
                state.scratchpad.append({
                    "role": "user",
                    "content": (
                        f"[System: Drift detected. Refocus on your goal: "
                        f"{state.goal}\nCurrent step: {step}]"
                    ),
                })
                self._emit("warning", {"message": "drift detected, re-anchoring"})

        # Inner iteration cap hit
        return {
            "needs_replan": True,
            "reason": "step exceeded inner iteration limit",
            "failed": False,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> str:
        """Execute the full Plan-Execute-ReAct loop for a goal.

        Args:
            goal: The user's high-level goal.

        Returns:
            Final answer string. Never raises — failures surface via
            ``state.status`` and the ``done`` event.
        """
        state = AgentState(goal=goal)
        replan_count = 0

        try:
            self._emit("goal", {"goal": goal})

            state.plan = await self._plan(goal)
            self._emit("plan", {"steps": state.plan})

            while state.current_step < len(state.plan):
                # Interrupt check
                if self._interrupted:
                    self._interrupted = False
                    state.status = "interrupted"
                    self._emit("warning", {"message": "interrupted by user"})
                    break

                # Iteration cap check
                cap = self._check_iteration_cap(state)
                if cap == "stop":
                    state.status = "aborted"
                    break
                if cap == "warn":
                    self._emit("warning", {"message": "approaching iteration cap"})

                step = state.plan[state.current_step]
                self._emit("step_start", {
                    "index": state.current_step,
                    "description": step,
                    "total": len(state.plan),
                })

                result = await self._execute_step_react(state, step)

                if result.get("needs_replan"):
                    replan_count += 1
                    if replan_count > self._max_replans:
                        state.status = "failed"
                        self._emit("error", {"message": "exceeded maximum replans"})
                        break
                    self._emit("replan", {"reason": result["reason"]})
                    state.plan = await self._replan(
                        goal, state.completed, result["reason"],
                    )
                    state.current_step = 0
                    continue

                if result.get("failed"):
                    state.status = "failed"
                    break

                state.completed.append({"step": step, "result": result})
                self._memory.checkpoint(
                    self._state_to_dict(state), state.iteration,
                )
                state.current_step += 1
                state.iteration += 1
                self._emit("step_done", {
                    "index": state.current_step - 1, "result": result,
                })

            if state.status == "running":
                state.status = "done"

        except Exception as e:
            state.status = "failed"
            self._emit("error", {"message": f"Agent error: {e}"})

        try:
            answer = await self._synthesize(state)
        except Exception:
            answer = f"Task {state.status}. {len(state.completed)} steps completed."

        self._emit("done", {"answer": answer, "status": state.status})
        return answer

    # ------------------------------------------------------------------
    # Interrupt & stats (Phase 5 — TUI wiring)
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """Signal the agent to stop after the current step."""
        self._interrupted = True

    def stats(self) -> dict:
        """Collect combined stats from LLM, memory, and tools.

        Returns:
            Dict with llm, memory, and tools sub-dicts.
        """
        return {
            "llm": self._llm.get_cost(),
            "memory": self._memory.stats(),
            "tools": self._tools.stats(),
        }
