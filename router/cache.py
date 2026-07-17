import hashlib
import json
import httpx
import os
import re
import math
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from router.db import RouterDB

class LocalTFIDF:
    def __init__(self):
        self.idf = {}
        self.vocabulary = {}

    def _tokenize(self, text: str) -> List[str]:
        # Simple word tokenization, removing single character tokens and converting to lowercase
        tokens = re.findall(r'\b\w\w+\b', text.lower())
        return tokens

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build vocabulary
        vocab = set()
        for doc in tokenized_docs:
            vocab.update(doc)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}
        
        # Calculate Inverse Document Frequency (IDF)
        num_docs = len(documents)
        doc_counts = Counter()
        for doc in tokenized_docs:
            unique_words = set(doc)
            for word in unique_words:
                doc_counts[word] += 1
                
        self.idf = {}
        for word, count in doc_counts.items():
            # Smoothed IDF formula
            self.idf[word] = math.log((1 + num_docs) / (1 + count)) + 1.0
            
        # Calculate TF-IDF vectors
        vectors = []
        for doc in tokenized_docs:
            vectors.append(self._vectorize_tokens(doc))
        return vectors

    def transform(self, doc: str) -> List[float]:
        tokens = self._tokenize(doc)
        return self._vectorize_tokens(tokens)

    def _vectorize_tokens(self, tokens: List[str]) -> List[float]:
        if not self.vocabulary:
            return []
            
        vector = [0.0] * len(self.vocabulary)
        tf = Counter(tokens)
        
        # Multiply Term Frequency by IDF
        l2_sum = 0.0
        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                val = count * self.idf.get(word, 1.0)
                vector[idx] = val
                l2_sum += val * val
                
        # Normalize vector (L2 norm)
        l2_norm = math.sqrt(l2_sum)
        if l2_norm > 0:
            vector = [v / l2_norm for v in vector]
            
        return vector

class CacheEngine:
    def __init__(self, db: RouterDB, config: Dict[str, Any]):
        self.db = db
        self.config = config
        self.caching_config = config.get("caching", {})
        self.enabled = self.caching_config.get("enabled", True)
        self.semantic_threshold = self.caching_config.get("semantic_threshold", 0.82)
        self.embedding_provider = self.caching_config.get("embedding_provider", "none")
        self.embedding_model = self.caching_config.get("embedding_model", "none")

    def hash_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Generates a stable SHA-256 hash of the chat messages."""
        serialized = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_query_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extracts the main user query text from the chat messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates embedding for a text using configured remote provider."""
        if not text or self.embedding_provider in ["none", "local"]:
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
                            "model": self.embedding_model or "text-embedding-3-small"
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
            print(f"Embedding generation error: {e}")
            
        return None

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
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

        # 2. Semantic Match Check
        query_text = self.get_query_text(messages)
        if not query_text:
            return None, "none"

        # 2.a Local TF-IDF Caching
        if self.embedding_provider == "local":
            candidates = self.db.get_all_cache_entries()
            if not candidates:
                return None, "none"
            
            # Fit and project with LocalTFIDF
            prompts = [c["prompt"] for c in candidates]
            all_docs = prompts + [query_text]
            
            tfidf = LocalTFIDF()
            vectors = tfidf.fit_transform(all_docs)
            
            query_vector = vectors[-1]
            candidate_vectors = vectors[:-1]
            
            best_sim = -1.0
            best_candidate = None
            
            for candidate, vector in zip(candidates, candidate_vectors):
                sim = self.cosine_similarity(query_vector, vector)
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = candidate
                    
            if best_sim >= self.semantic_threshold and best_candidate:
                print(f"Local TF-IDF Semantic Cache Hit! Similarity: {best_sim:.4f}")
                return best_candidate["response"], "semantic"

        # 2.b Remote Provider Caching (OpenAI/Gemini)
        elif self.embedding_provider != "none":
            query_embedding = await self.get_embedding(query_text)
            if not query_embedding:
                return None, "none"

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
                print(f"Remote Semantic Cache Hit! Similarity: {best_sim:.4f}")
                return best_candidate["response"], "semantic"

        return None, "none"

    async def save(self, messages: List[Dict[str, Any]], response: Dict[str, Any], provider: str = "", model: str = ""):
        """Saves a query-response pair in the cache."""
        if not self.enabled:
            return

        prompt_hash = self.hash_prompt(messages)
        query_text = self.get_query_text(messages)

        # Generate embedding only for remote providers
        embedding = None
        if self.embedding_provider not in ["none", "local"] and query_text:
            embedding = await self.get_embedding(query_text)

        self.db.add_cache(
            prompt_hash=prompt_hash,
            prompt=query_text,
            response=response,
            embedding=embedding,
            provider=provider,
            model=model
        )
