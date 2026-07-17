import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

# Add main workspace to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from main import app, load_config
from router.classifier import ComplexityClassifier
from router.cache import CacheEngine, LocalTFIDF
from router.db import RouterDB

TEST_DB_PATH = "test_agent_router.db"

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_code_compression():
    classifier = ComplexityClassifier({"routing": {"compress_context": True}})
    
    # Python code block with comments
    python_block = (
        "Here is the code:\n"
        "```python\n"
        "#!/usr/bin/env python\n"
        "# This is a helper comment\n"
        "def hello():\n"
        "    # Print hello world\n"
        "    print('hello')\n"
        "\n"
        "\n"
        "```"
    )
    compressed = classifier.compress_code_context(python_block)
    assert "#!/usr/bin/env python" in compressed
    assert "This is a helper comment" not in compressed
    assert "Print hello world" not in compressed
    assert "def hello():" in compressed
    # verify empty lines are collapsed
    assert "\n\n\n" not in compressed

def test_message_canonicalization():
    classifier = ComplexityClassifier({"routing": {"compress_context": False}})
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "system", "content": "Date is 2026-07-17 time is 19:40:00."}
    ]
    canonical = classifier.canonicalize_messages(messages)
    
    # System should be first
    assert canonical[0]["role"] == "system"
    # Dynamic values should be replaced
    assert "2026-07-17" not in canonical[0]["content"]
    assert "<canonical_date>" in canonical[0]["content"]
    assert "<canonical_time>" in canonical[0]["content"]

def test_local_tfidf_vectorizer():
    # Test L2 norm normalization and vocabulary matching
    tfidf = LocalTFIDF()
    docs = [
        "Explain binary search in Python.",
        "What is time complexity of binary search?",
        "Refactor authentication token logic."
    ]
    vectors = tfidf.fit_transform(docs)
    assert len(vectors) == 3
    # Check L2 norm is approximately 1.0 for non-empty vectors
    for vec in vectors:
        norm = sum(v*v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    # Check transform on a similar query matches the correct candidate
    query = "How to write binary search?"
    query_vec = tfidf.transform(query)
    
    # Cosine similarities
    sims = [sum(x*y for x,y in zip(query_vec, v)) for v in vectors]
    # Similarity with first doc (about binary search) should be higher than third (about auth token)
    assert sims[0] > sims[2]
    assert sims[1] > sims[2]

@patch("httpx.AsyncClient.send")
def test_cascade_routing_and_validation(mock_send):
    db = RouterDB(TEST_DB_PATH)
    
    # Scenario: Tier 1 output has mismatched braces (fails validation) -> triggers escalation to Tier 2
    # Mock response 1: Tier 1 failure (unmatched curly braces)
    mock_resp_t1 = MagicMock()
    mock_resp_t1.status_code = 200
    mock_resp_t1.json.return_value = {
        "id": "chatcmpl-t1-failed",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Broken Code: { "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    
    # Mock response 2: Tier 2 success (clean, valid output)
    mock_resp_t2 = MagicMock()
    mock_resp_t2.status_code = 200
    mock_resp_t2.json.return_value = {
        "id": "chatcmpl-t2-success",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Corrected code: { ok }"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8}
    }
    
    mock_send.side_effect = [mock_resp_t1, mock_resp_t2]

    # Patch config to enable cascades
    with patch("main.load_config") as mock_load_config:
        mock_load_config.return_value = {
            "server": {"host": "127.0.0.1", "port": 8000},
            "caching": {"enabled": True, "database_path": TEST_DB_PATH, "embedding_provider": "local", "semantic_threshold": 0.82},
            "tiers": {
                "tier1": {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "TEST_KEY"},
                "tier2": {"provider": "openai", "model": "gpt-4o", "api_key_env": "TEST_KEY"}
            },
            "routing": {
                "default_tier": 1,
                "cascade_enabled": True,
                "compress_context": True,
                "max_read_context_threshold": 30000,
                "tier2_keywords": ["fix", "debug"],
                "tier1_keywords": ["explain"]
            }
        }
        os.environ["TEST_KEY"] = "mock-key"
        
        with TestClient(app) as client:
            # Send a request that classifies as Tier 2 (uses multiple tier2 keywords)
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "fix and debug and refactor the socket block compilation"}]
            }
            
            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            # Should return the Tier 2 response since Tier 1 failed validation
            assert resp.json()["choices"][0]["message"]["content"] == "Corrected code: { ok }"
            # Verify mock client was called twice (Tier 1 then Tier 2)
            assert mock_send.call_count == 2

            # Verify metrics tracks the logs
            metrics_resp = client.get("/api/metrics")
            metrics = metrics_resp.json()
            assert metrics["total_requests"] == 2 # 1 failed cascade attempt + 1 escalated Tier 2 completion
            
def test_feedback_endpoint():
    db = RouterDB(TEST_DB_PATH)
    
    with patch("main.load_config") as mock_load_config:
        mock_load_config.return_value = {
            "server": {"host": "127.0.0.1", "port": 8000},
            "caching": {"enabled": True, "database_path": TEST_DB_PATH, "embedding_provider": "none"},
            "routing": {"default_tier": 1}
        }
        
        with patch("main.router_db", db):
            # Seed a request log in database
            db.log_request(
                prompt="test prompt query text",
                complexity_score=1.0,
                routed_model="gpt-4o-mini",
                requested_model="gpt-4o-mini",
                provider="openai",
                input_tokens=10,
                output_tokens=15,
                input_cost=0.00001,
                output_cost=0.00002,
                tier_selected=1,
                routing_reason="Test routing",
                duration_ms=200,
                cache_hit="none",
                success=None
            )
            
            with TestClient(app) as client:
                # Post feedback
                feedback_payload = {
                    "prompt": "test prompt query text",
                    "success": True
                }
                
                resp = client.post("/v1/feedback", json=feedback_payload)
                assert resp.status_code == 200
                assert resp.json()["status"] == "success"
                
                # Check DB updated success rate
                metrics_resp = client.get("/api/metrics")
                metrics = metrics_resp.json()
                assert metrics["success_rate"] == 100.0
                assert metrics["feedback_total"] == 1
