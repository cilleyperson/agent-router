import hashlib
import json
import httpx
import os
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from router.db import RouterDB

class CacheEngine:
    def __init__(self, db: RouterDB, config: Dict[str, Any]):
        self.db = db
        self.config = config
        self.caching_config = config.get("caching", {})
        self.enabled = self.caching_config.get("enabled", True)
        self.semantic_threshold = self.caching_config.get("semantic_threshold", 0.88)
        self.embedding_provider = self.caching_config.get("embedding_provider", "none")
        self.embedding_model = self.caching_config.get("embedding_model", "text-embedding-3-small")

    def hash_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Generates a stable SHA-256 hash of the chat messages."""
        # Convert messages to a canonical JSON string (with sorted keys)
        serialized = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_query_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extracts the main user query text from the chat messages."""
        # Typically the last user message represents the current query
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multimodal or list of contents
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates embedding for a text using the configured provider."""
        if not text or self.embedding_provider == "none":
            return None

        try:
            if self.embedding_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "input": text,
                            "model": self.embedding_model
                        },
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data["data"][0]["embedding"]

            elif self.embedding_provider == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    return None
                model = self.embedding_model or "text-embedding-004"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}",
                        headers={"Content-Type": "application/json"},
                        json={
                            "content": {
                                "parts": [{"text": text}]
                            }
                        },
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data["embedding"]["values"]
        except Exception as e:
            # Silently fail and return None for embedding to bypass semantic cache
            print(f"Embedding generation error: {e}")
            
        return None

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        try:
            arr_a = np.array(a)
            arr_b = np.array(b)
            dot = np.dot(arr_a, arr_b)
            norm_a = np.linalg.norm(arr_a)
            norm_b = np.linalg.norm(arr_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        except Exception:
            # Pure Python fallback
            try:
                dot = sum(x*y for x,y in zip(a, b))
                norm_a = sum(x*x for x in a) ** 0.5
                norm_b = sum(x*x for x in b) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)
            except Exception:
                return 0.0

    async def lookup(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Looks up a response in the cache first by exact hash, then by semantic similarity.
        Returns: Tuple of (cached_response, cache_hit_type)
        """
        if not self.enabled:
            return None, "none"

        # 1. Exact Match Check
        prompt_hash = self.hash_prompt(messages)
        exact_hit = self.db.get_exact_cache(prompt_hash)
        if exact_hit:
            return exact_hit["response"], "exact"

        # 2. Semantic Match Check (if enabled)
        if self.embedding_provider != "none":
            query_text = self.get_query_text(messages)
            if not query_text:
                return None, "none"
            
            # Fetch embedding for current query
            query_embedding = await self.get_embedding(query_text)
            if not query_embedding:
                return None, "none"

            # Retrieve all cached embeddings
            candidates = self.db.get_all_embeddings()
            if not candidates:
                return None, "none"

            best_sim = -1.0
            best_candidate = None

            for candidate in candidates:
                sim = self.cosine_similarity(query_embedding, candidate["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = candidate

            if best_sim >= self.semantic_threshold and best_candidate:
                print(f"Semantic cache hit! Similarity: {best_sim:.4f}")
                return best_candidate["response"], "semantic"

        return None, "none"

    async def save(self, messages: List[Dict[str, Any]], response: Dict[str, Any], provider: str = "", model: str = ""):
        """Saves a query-response pair in the cache."""
        if not self.enabled:
            return

        prompt_hash = self.hash_prompt(messages)
        query_text = self.get_query_text(messages)

        # Generate embedding for the new entry to enable future semantic hits
        embedding = None
        if self.embedding_provider != "none" and query_text:
            embedding = await self.get_embedding(query_text)

        self.db.add_cache(
            prompt_hash=prompt_hash,
            prompt=query_text,
            response=response,
            embedding=embedding,
            provider=provider,
            model=model
        )
