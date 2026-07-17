# 🤖 AI Agent Router Proxy

An efficient and cost-optimized API proxy for local AI agent CLIs and IDEs (e.g., Antigravity, Aider, Continue, Cursor). It intercepts requests, determines prompt complexity, executes dynamic model routing (Tier 1 vs. Tier 2), performs semantic/exact caching to save cost/latency, and aggregates real-time observability metrics.

## 🚀 Key Features

*   **Dynamic Complexity Classification**: Routes simple questions to Tier 1 models (e.g. Gemini 1.5 Flash, GPT-4o-Mini) and complex requests to Tier 2 models (e.g. Claude 3.5 Sonnet, GPT-4o).
*   **Exact & Semantic Caching**: Caches completions using prompt hashing and semantic similarity (via embeddings). Semantic similarity returns responses for structurally similar queries, saving 100% of upstream costs.
*   **Unified API Compatibility**: Accepts standard OpenAI `/v1/chat/completions` formats and translates calls seamlessly to Anthropic or Gemini under the hood.
*   **Observability Dashboard**: Exposes a gorgeous, dark-mode dashboard with real-time savings tracking, latency records, cache breakdown, and request log audits.

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Set your API keys in your environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 3. Start the Server
To launch the FastAPI proxy server locally:
```bash
python main.py run
```
By default, the server runs on `http://127.0.0.1:8000`.

### 4. Open the Observability Dashboard
Navigate to `http://localhost:8000/dashboard` in your browser. You will see:
*   Real-time savings counters and token tracking.
*   Interactive donut and bar charts showing model shares and cache hits.
*   A scrollable historical audit table of all routed requests, latencies, and reasons.
*   "Clear Cache" and "Refresh" tools.

### 5. CLI Metrics Stats
To view proxy metrics directly in your terminal:
```bash
python main.py stats
```

## ⚙️ Routing Configuration
Modify `config.yaml` to change model tier assignments, caching thresholds, and classification keywords.
