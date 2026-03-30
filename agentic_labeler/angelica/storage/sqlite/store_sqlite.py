from __future__ import annotations

"""SQLite storage for documents and labels.

Design:
- Always store the full label as JSON (`StoreSpec.json_column`).
- Optionally also store selected top-level fields as extra columns (`StoreSpec.index_fields`).

Tables:
- documents(doc_id, source, text, created_at)
- labels(label_id, doc_id, agent_id, label_json, <optional index fields>, created_at)
- final_labels(doc_id, decided_by, label_json, <optional index fields>, created_at)
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Type

import pandas as pd
from pydantic import BaseModel

from angelica.models.config import StoreSpec


def _now_iso() -> str:
    """UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    """SQLite store that works for any Pydantic schema."""

    def __init__(self, db_path: str, schema: Type[BaseModel], store_spec: Optional[StoreSpec] = None):
        self.db_path = db_path
        self.schema = schema
        self.store_spec = store_spec or StoreSpec()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        # Extra columns for index fields (stored as TEXT)
        idx_fields = [f for f in self.store_spec.index_fields if f]
        extra_cols = "".join([f", {f} TEXT" for f in idx_fields])

        with self._conn() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS labels (
                    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    {self.store_spec.json_column} TEXT NOT NULL
                    {extra_cols},
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    UNIQUE(doc_id, agent_id)
                );

                CREATE TABLE IF NOT EXISTS final_labels (
                    doc_id INTEGER PRIMARY KEY,
                    decided_by TEXT NOT NULL,
                    {self.store_spec.json_column} TEXT NOT NULL
                    {extra_cols},
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
                """
            )

    def add_document(self, text: str, source: Optional[str] = None) -> int:
        """Insert a document and return its doc_id."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO documents(source, text, created_at) VALUES (?, ?, ?)",
                (source, text, _now_iso()),
            )
            return int(cur.lastrowid)

    def _index_field_values(self, label: BaseModel) -> Dict[str, Any]:
        """Extract configured index fields from a label for storage."""
        out: Dict[str, Any] = {}
        for f in self.store_spec.index_fields:
            if not f:
                continue
            val = getattr(label, f, None)
            # Enums: store their `.value`
            if hasattr(val, "value"):
                val = val.value
            out[f] = val
        return out

    def save_label(self, doc_id: int, agent_id: str, label: BaseModel) -> None:
        """Upsert a per-agent label for a document."""
        payload = json.dumps(label.model_dump(), ensure_ascii=False)
        idx = self._index_field_values(label)

        cols = ["doc_id", "agent_id", self.store_spec.json_column] + list(idx.keys()) + ["created_at"]
        vals = [doc_id, agent_id, payload] + list(idx.values()) + [_now_iso()]
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join(cols)

        update_cols = [f"{self.store_spec.json_column}=excluded.{self.store_spec.json_column}"]
        update_cols += [f"{k}=excluded.{k}" for k in idx.keys()]
        update_cols += ["created_at=excluded.created_at"]
        update_sql = ", ".join(update_cols)

        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO labels({col_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT(doc_id, agent_id) DO UPDATE SET {update_sql}",
                vals,
            )

    def save_final_label(self, doc_id: int, decided_by: str, label: BaseModel) -> None:
        """Upsert the final label for a document."""
        payload = json.dumps(label.model_dump(), ensure_ascii=False)
        idx = self._index_field_values(label)

        cols = ["doc_id", "decided_by", self.store_spec.json_column] + list(idx.keys()) + ["created_at"]
        vals = [doc_id, decided_by, payload] + list(idx.values()) + [_now_iso()]
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join(cols)

        update_cols = ["decided_by=excluded.decided_by", f"{self.store_spec.json_column}=excluded.{self.store_spec.json_column}"]
        update_cols += [f"{k}=excluded.{k}" for k in idx.keys()]
        update_cols += ["created_at=excluded.created_at"]
        update_sql = ", ".join(update_cols)

        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO final_labels({col_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT(doc_id) DO UPDATE SET {update_sql}",
                vals,
            )

    def fetch_examples_for_doc_ids(self, doc_ids: Sequence[int]) -> pd.DataFrame:
        """Fetch documents + final labels for a list of doc_ids (used for retrieval examples)."""
        if not doc_ids:
            return pd.DataFrame()

        placeholders = ",".join(["?"] * len(doc_ids))
        with self._conn() as conn:
            return pd.read_sql_query(
                f"""
                SELECT d.doc_id, d.text, f.decided_by, f.{self.store_spec.json_column} as label_json, f.created_at as labeled_at
                FROM final_labels f
                JOIN documents d ON d.doc_id = f.doc_id
                WHERE d.doc_id IN ({placeholders})
                ORDER BY f.created_at DESC
                """,
                conn,
                params=list(doc_ids),
            )

    def fetch_agent_pairwise_json(self, agent_a: str, agent_b: str) -> pd.DataFrame:
        """Return one row per doc where both agents labeled, including each label's JSON."""
        with self._conn() as conn:
            return pd.read_sql_query(
                f"""
                SELECT
                    a.doc_id as doc_id,
                    CASE WHEN a.created_at > b.created_at THEN a.created_at ELSE b.created_at END AS created_at,
                    a.{self.store_spec.json_column} AS a_label_json,
                    b.{self.store_spec.json_column} AS b_label_json
                FROM labels a
                JOIN labels b ON a.doc_id=b.doc_id
                WHERE a.agent_id = ?
                  AND b.agent_id = ?
                ORDER BY created_at ASC
                """,
                conn,
                params=(agent_a, agent_b),
            )
