import sqlite3
import json
import os
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

class RouterDB:
    def __init__(self, db_path: str = "agent_router.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initializes SQLite database tables if they do not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Cache Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prompt_hash TEXT UNIQUE,
                prompt TEXT,
                response TEXT,
                embedding BLOB,
                provider TEXT,
                model TEXT
            )
        """)

        # Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prompt TEXT,
                complexity_score REAL,
                routed_model TEXT,
                requested_model TEXT,
                provider TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                input_cost REAL,
                output_cost REAL,
                tier_selected INTEGER,
                routing_reason TEXT,
                duration_ms INTEGER,
                cache_hit TEXT
            )
        """)

        conn.commit()
        conn.close()

    def get_exact_cache(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached response by its exact prompt SHA-256 hash."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response, provider, model FROM cache WHERE prompt_hash = ?", 
            (prompt_hash,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "response": json.loads(row["response"]),
                "provider": row["provider"],
                "model": row["model"]
            }
        return None

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieves all cache entries containing embeddings for semantic search."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT prompt, response, embedding, provider, model FROM cache WHERE embedding IS NOT NULL"
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            try:
                # Convert blob back to float list
                embedding_bytes = row["embedding"]
                embedding = list(float(v) for v in json.loads(embedding_bytes.decode('utf-8')))
                results.append({
                    "prompt": row["prompt"],
                    "response": json.loads(row["response"]),
                    "embedding": embedding,
                    "provider": row["provider"],
                    "model": row["model"]
                })
            except Exception as e:
                # Log error or pass if corrupt
                pass
        return results

    def add_cache(self, prompt_hash: str, prompt: str, response: Dict[str, Any], 
                  embedding: Optional[List[float]] = None, provider: str = "", model: str = ""):
        """Stores a query and response in the cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = None
        if embedding:
            embedding_blob = json.dumps(embedding).encode('utf-8')

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO cache (prompt_hash, prompt, response, embedding, provider, model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (prompt_hash, prompt, json.dumps(response), embedding_blob, provider, model)
            )
            conn.commit()
        except sqlite3.Error as e:
            # Let it fail gracefully
            print(f"Database error writing to cache: {e}")
        finally:
            conn.close()

    def log_request(self, prompt: str, complexity_score: float, routed_model: str, 
                    requested_model: str, provider: str, input_tokens: int, 
                    output_tokens: int, input_cost: float, output_cost: float, 
                    tier_selected: int, routing_reason: str, duration_ms: int, 
                    cache_hit: str):
        """Logs request analytics into the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO request_logs (
                    prompt, complexity_score, routed_model, requested_model, provider,
                    input_tokens, output_tokens, input_cost, output_cost, 
                    tier_selected, routing_reason, duration_ms, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (prompt, complexity_score, routed_model, requested_model, provider,
                 input_tokens, output_tokens, input_cost, output_cost,
                 tier_selected, routing_reason, duration_ms, cache_hit)
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error writing to log: {e}")
        finally:
            conn.close()

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates aggregate metrics for the dashboard."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # General counts
        cursor.execute("SELECT COUNT(*) as total_reqs FROM request_logs")
        total_reqs = cursor.fetchone()["total_reqs"] or 0

        # Cache statistics
        cursor.execute("SELECT COUNT(*) as exact_hits FROM request_logs WHERE cache_hit = 'exact'")
        exact_hits = cursor.fetchone()["exact_hits"] or 0

        cursor.execute("SELECT COUNT(*) as semantic_hits FROM request_logs WHERE cache_hit = 'semantic'")
        semantic_hits = cursor.fetchone()["semantic_hits"] or 0

        # Token usage & Cost
        cursor.execute("""
            SELECT 
                SUM(input_tokens) as total_in_tokens,
                SUM(output_tokens) as total_out_tokens,
                SUM(input_cost + output_cost) as total_actual_cost
            FROM request_logs
        """)
        row = cursor.fetchone()
        total_in_tokens = row["total_in_tokens"] or 0
        total_out_tokens = row["total_out_tokens"] or 0
        total_actual_cost = row["total_actual_cost"] or 0.0

        # Calculate "Hypothetical cost" if all requests had gone to the high-quality model directly
        # Let's say high-quality is Claude 3.5 Sonnet ($3.00/$15.00 per M tokens).
        # We will assume a baseline of $3.00/1M input and $15.00/1M output for estimating savings.
        # Plus, for cached hits, the savings is 100%.
        cursor.execute("""
            SELECT 
                SUM(input_tokens + output_tokens) as total_saved_tokens,
                SUM(CASE 
                    WHEN cache_hit IN ('exact', 'semantic') THEN (input_tokens * 3.0 / 1000000.0) + (output_tokens * 15.0 / 1000000.0)
                    WHEN tier_selected = 1 THEN ((input_tokens * 3.0 / 1000000.0) + (output_tokens * 15.0 / 1000000.0)) - (input_cost + output_cost)
                    ELSE 0 
                END) as cost_savings
            FROM request_logs
        """)
        savings_row = cursor.fetchone()
        cost_savings = savings_row["cost_savings"] or 0.0

        # Get average latency
        cursor.execute("SELECT AVG(duration_ms) as avg_latency FROM request_logs")
        avg_latency = cursor.fetchone()["avg_latency"] or 0.0

        # Get routing distribution (T1 vs T2)
        cursor.execute("SELECT COUNT(*) as t1_count FROM request_logs WHERE tier_selected = 1")
        t1_count = cursor.fetchone()["t1_count"] or 0
        
        cursor.execute("SELECT COUNT(*) as t2_count FROM request_logs WHERE tier_selected = 2")
        t2_count = cursor.fetchone()["t2_count"] or 0

        # Time-series logs (last 50 requests)
        cursor.execute("""
            SELECT timestamp, requested_model, routed_model, provider, 
                   (input_tokens + output_tokens) as tokens, (input_cost + output_cost) as cost, 
                   duration_ms, cache_hit, tier_selected, routing_reason 
            FROM request_logs 
            ORDER BY timestamp DESC 
            LIMIT 50
        """)
        history = [dict(r) for r in cursor.fetchall()]

        conn.close()

        cache_hit_rate = ((exact_hits + semantic_hits) / total_reqs * 100) if total_reqs > 0 else 0.0

        return {
            "total_requests": total_reqs,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "total_tokens": total_in_tokens + total_out_tokens,
            "total_cost": round(total_actual_cost, 6),
            "cost_savings": round(cost_savings, 6),
            "average_latency_ms": round(avg_latency, 2),
            "tier1_count": t1_count,
            "tier2_count": t2_count,
            "history": history
        }
