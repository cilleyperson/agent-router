# 🤖 AI Agent Router Proxy

An efficient, cost-optimized API gateway and proxy designed for local AI agent CLIs and IDE extensions (e.g., Antigravity, Aider, Continue, Cursor, Roo Code). By intercepting outgoing model calls, the router optimizes token costs and latency while preserving execution quality.

---

## 🚀 Core Optimization Engine

*   **Dynamic Complexity Classifier**: Evaluates keywords, system instruction lengths, code block markers, file extensions, and error stack traces to route simple reads to Tier 1 and complex refactoring tasks to Tier 2.
*   **Cheap-First Cascades & Validators**: Routes Tier 2 queries to the cheaper Tier 1 model first, validates the output locally (checking for indentation failures, compilation warnings, mismatched braces, or abrupt truncation), and escalates to Tier 2 only if validation fails.
*   **Local Offline Semantic Caching**: Projects prompt queries into TF-IDF vectors locally and matches against cache hits using cosine similarity. Runs in `<10ms` for `$0.00` with zero remote network calls.
*   **Context Compaction**: Strips single-line and block code comments (`#`, `//`) and collapses consecutive empty lines from active developer contexts, reducing context token costs by 20–40%.
*   **Message Canonicalizer**: Canonicalizes dynamic runtime variables (dates, times, user paths) and groups system prompts first to stabilize context prefixes and guarantee prompt caching hits on providers (Anthropic, DeepSeek, OpenAI).
*   **Telemetry Feedback Loop**: Exposes a telemetry route (`/v1/feedback`) for local agents to report compiler or test success, feeding back into the observability metrics.

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the standard dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Define the upstream API keys in your shell:
```bash
# Windows (CMD)
set OPENAI_API_KEY=your-openai-key
set ANTHROPIC_API_KEY=your-anthropic-key
set GEMINI_API_KEY=your-gemini-key

# Linux / MacOS / PowerShell
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 3. Run the Server
Start the local FastAPI proxy server:
```bash
python main.py run
```
The server will boot by default on `http://127.0.0.1:8000`.

### 4. Metrics Dashboard
Open `http://localhost:8000/dashboard` in any browser to access the glowing dark-mode observability panel tracking accumulated savings, cache hits, requests, latencies, and compiler success rates.

### 5. CLI Statistics Command
To view aggregates directly in the terminal:
```bash
python main.py stats
```

---

## 🔌 Integration Guide

Configure your developer environment to route requests through the local proxy.

### 1. Aider CLI
To use the router with Aider, tell it to route through the local endpoint:

**Via CLI command:**
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key dummy --model openai/agent-router-auto
```

**Via environment variables (in your project `.env`):**
```env
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy
AIDER_MODEL=openai/agent-router-auto
```

---

### 2. Continue (VS Code & JetBrains Extension)
Edit your Continue configuration (`~/.continue/config.json`) to register the router as an OpenAI-compatible provider:

```json
{
  "models": [
    {
      "title": "Agent Router (Auto)",
      "provider": "openai",
      "model": "agent-router-auto",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "dummy"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Agent Router Tab Fill",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "dummy"
  }
}
```

---

### 3. Cursor IDE
Set up the router as an override base URL:
1.  Navigate to **Cursor Settings** -> **Models**.
2.  Under **OpenAI API**, toggle on the override option.
3.  Set the API Key to `dummy`.
4.  Override the base URL to: `http://localhost:8000/v1`.
5.  Turn on the models you wish to route (e.g., `gpt-4o-mini` or leave it to standard selections).

---

### 4. Roo Code / Claude Dev (VS Code Extension)
1.  Open the Roo Code settings panel.
2.  Select **OpenAI Compatible** as the Provider.
3.  Set **Base URL** to `http://localhost:8000/v1`.
4.  Set **API Key** to `dummy`.
5.  Set **Model ID** to `agent-router-auto`.

---

### 5. Antigravity IDE & agy CLI
Configure your Antigravity settings or Python SDK setup to point its base URL to the local proxy:

```python
from antigravity import AntigravityClient

client = AntigravityClient(
    api_base="http://localhost:8000/v1",
    api_key="dummy"
)
```

---

### 6. Sending Compiler Feedback (Automation Script)
In your test runner or build script, notify the router if a generated file compiled successfully or failed tests:

**Bash example:**
```bash
# Run compiler/tests
pytest tests/
SUCCESS=$?

# Report feedback to proxy
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$(git log -1 --pretty=%B)\", \"success\": $([ $SUCCESS -eq 0 ] && echo "true" || echo "false")}"
```
This loop ensures the router continuously refines its routing scoring mechanics!
