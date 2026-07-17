import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Add main workspace to sys.path so we can import modules
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from main import app, load_config
from router.classifier import ComplexityClassifier
from router.db import RouterDB

# Use a test database
TEST_DB_PATH = "test_agent_router.db"

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    # Cleanup DB if exists before test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield
    # Cleanup DB after test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_complexity_classifier():
    config = {
        "routing": {
            "default_tier": 1,
            "max_read_context_threshold": 500,
            "tier2_keywords": ["fix", "debug", "refactor", "implement", "modify"],
            "tier1_keywords": ["explain", "what is", "summarize", "list"]
        }
    }
    
    classifier = ComplexityClassifier(config)
    
    # 1. Simple query -> Tier 1
    messages_simple = [{"role": "user", "content": "Explain how a link list works in Python."}]
    tier, score, reason = classifier.analyze_request(messages_simple)
    assert tier == 1
    assert "Low complexity" in reason
    
    # 2. Complex query -> Tier 2
    messages_complex = [{"role": "user", "content": "Debug and fix the segfault memory issue in the socket listener."}]
    tier, score, reason = classifier.analyze_request(messages_complex)
    assert tier == 2
    assert "High complexity" in reason

    # 3. Large read context -> Tier 1
    messages_large = [
        {"role": "system", "content": "You are a reader helper. " * 50}, # ~300 chars
        {"role": "user", "content": "Explain what this document is about. " * 30} # ~1100 chars
    ]
    tier, score, reason = classifier.analyze_request(messages_large)
    assert tier == 1
    assert "Large read context" in reason

@patch("httpx.AsyncClient.send")
def test_proxy_flow(mock_send):
    from unittest.mock import MagicMock
    # Mock upstream API response (httpx.Response.json is synchronous)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-mock123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock reply from upstream LLM."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20
        }
    }
    mock_send.return_value = mock_response

    # Force using test database and test configuration
    with patch("main.load_config") as mock_load_config:
        mock_load_config.return_value = {
            "server": {"host": "127.0.0.1", "port": 8000},
            "caching": {
                "enabled": True,
                "database_path": TEST_DB_PATH,
                "embedding_provider": "none" # Avoid remote embedding api calls
            },
            "tiers": {
                "tier1": {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "TEST_KEY"},
                "tier2": {"provider": "openai", "model": "gpt-4o", "api_key_env": "TEST_KEY"}
            },
            "routing": {
                "default_tier": 1,
                "max_read_context_threshold": 30000,
                "tier2_keywords": ["fix", "debug", "refactor"],
                "tier1_keywords": ["explain", "what is"]
            }
        }
        
        # Set dummy key for environment
        os.environ["TEST_KEY"] = "mock-key-value"
        
        # Instantiate test client
        with TestClient(app) as client:
            # 1. First Request (Cache Miss)
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "explain links"}]
            }
            
            resp1 = client.post("/v1/chat/completions", json=payload)
            assert resp1.status_code == 200
            assert resp1.json()["choices"][0]["message"]["content"] == "This is a mock reply from upstream LLM."
            assert mock_send.call_count == 1 # Upstream was called

            # 2. Second Request (Cache Hit)
            resp2 = client.post("/v1/chat/completions", json=payload)
            assert resp2.status_code == 200
            assert resp2.json()["choices"][0]["message"]["content"] == "This is a mock reply from upstream LLM."
            assert mock_send.call_count == 1 # Upstream was NOT called again (cache hit)

            # 3. Check metrics endpoint
            metrics_resp = client.get("/api/metrics")
            assert metrics_resp.status_code == 200
            metrics_data = metrics_resp.json()
            assert metrics_data["total_requests"] == 2
            assert metrics_data["exact_hits"] == 1
            assert metrics_data["cache_hit_rate"] == 50.0

if __name__ == "__main__":
    # If running directly, run pytest
    import pytest
    sys.exit(pytest.main([__file__]))
