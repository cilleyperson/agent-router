import os
import sys
import yaml
import uvicorn
import argparse
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from router.db import RouterDB
from router.cache import CacheEngine
from router.classifier import ComplexityClassifier
from router.proxy import ProxyHandler

app = FastAPI(title="AI Agent Router Proxy", version="0.1.0")

# Enable CORS for local integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for router components
router_db = None
cache_engine = None
classifier = None
proxy_handler = None
config = {}

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: config file '{config_path}' not found. Using defaults.")
        return {
            "server": {"host": "127.0.0.1", "port": 8000},
            "caching": {"enabled": True, "database_path": "agent_router.db"},
            "routing": {"default_tier": 1}
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Initialization on startup
@app.on_event("startup")
async def startup_event():
    global router_db, cache_engine, classifier, proxy_handler, config
    config = load_config()
    
    db_path = config.get("caching", {}).get("database_path", "agent_router.db")
    router_db = RouterDB(db_path)
    cache_engine = CacheEngine(router_db, config)
    classifier = ComplexityClassifier(config)
    proxy_handler = ProxyHandler(router_db, cache_engine, classifier, config)
    
    # Ensure dashboard static directory exists
    os.makedirs("dashboard", exist_ok=True)
    
    print("--------------------------------------------------")
    print(f"AI Agent Router Proxy successfully initialized.")
    print(f"Listening for OpenAI-compatible client requests on port {config.get('server', {}).get('port', 8000)}...")
    print(f"Metrics dashboard available at: http://localhost:{config.get('server', {}).get('port', 8000)}/dashboard")
    print("--------------------------------------------------")

# Core OpenAI-compatible proxy completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not proxy_handler:
        raise HTTPException(status_code=500, detail="Proxy Handler not initialized")
    return await proxy_handler.handle_request(request)

# OpenAI models list endpoint (for tool compatibility)
@app.get("/v1/models")
async def list_models():
    # Return standard list of available models mapped to routing options
    return {
        "object": "list",
        "data": [
            {"id": "agent-router-auto", "object": "model", "created": 1700000000, "owned_by": "agent-router"},
            {"id": "gpt-4o-mini", "object": "model", "created": 1700000000, "owned_by": "system"},
            {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "system"},
            {"id": "claude-3-5-sonnet-20241022", "object": "model", "created": 1700000000, "owned_by": "system"},
            {"id": "claude-3-5-haiku-20241022", "object": "model", "created": 1700000000, "owned_by": "system"},
            {"id": "gemini-1.5-flash", "object": "model", "created": 1700000000, "owned_by": "system"},
            {"id": "gemini-1.5-pro", "object": "model", "created": 1700000000, "owned_by": "system"}
        ]
    }

# API Endpoint for Dashboard metrics
@app.get("/api/metrics")
async def get_metrics():
    if not router_db:
        return JSONResponse({"error": "Database not initialized"}, status_code=500)
    return router_db.get_metrics()

# API Endpoint to clear cache
@app.post("/api/clear_cache")
async def clear_cache():
    if not router_db:
        return JSONResponse({"error": "Database not initialized"}, status_code=500)
    try:
        conn = router_db.connect() # wait, router_db uses local connections inside methods, let's execute query directly
    except AttributeError:
        import sqlite3
        conn = sqlite3.connect(router_db.db_path)
        
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        return {"status": "success", "message": "Cache successfully cleared"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        conn.close()

# Serves Dashboard page
@app.get("/dashboard")
async def serve_dashboard():
    dashboard_file = Path("dashboard/index.html")
    if not dashboard_file.exists():
        return HTMLResponse(
            "<h3>Dashboard files are missing. Please run compilation / setup.</h3>", 
            status_code=404
        )
    return FileResponse(dashboard_file)

# Mount dashboard directory as static files for css/js loading
app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard_static")

# Serve a basic welcome page at root redirecting to dashboard
@app.get("/")
async def root():
    return HTMLResponse(
        """
        <html>
            <head>
                <title>AI Agent Router Proxy</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; padding-top: 15%; }
                    a { color: #3b82f6; text-decoration: none; font-weight: bold; border: 1px solid #3b82f6; padding: 10px 20px; border-radius: 6px; transition: background 0.2s; }
                    a:hover { background: #1e3a8a; }
                    h1 { margin-bottom: 30px; }
                </style>
            </head>
            <body>
                <h1>🤖 AI Agent Router Proxy is Running</h1>
                <a href="/dashboard">Open Observability Dashboard</a>
            </body>
        </html>
        """
    )

def main():
    parser = argparse.ArgumentParser(description="AI Agent Router Proxy CLI")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "stats", "clear-cache"], help="Command to run")
    parser.add_argument("--host", default=None, help="Host to run server on")
    parser.add_argument("--port", type=int, default=None, help="Port to run server on")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    args = parser.parse_args()
    
    cfg = load_config(args.config)
    host = args.host or cfg.get("server", {}).get("host", "127.0.0.1")
    port = args.port or cfg.get("server", {}).get("port", 8000)

    if args.command == "run":
        uvicorn.run("main:app", host=host, port=port, reload=True)
        
    elif args.command == "stats":
        db_path = cfg.get("caching", {}).get("database_path", "agent_router.db")
        db = RouterDB(db_path)
        metrics = db.get_metrics()
        print("\n=== AI Agent Router Proxy Stats ===")
        print(f"Total Requests Processed: {metrics['total_requests']}")
        print(f"Cache Hit Rate:          {metrics['cache_hit_rate']}%")
        print(f"  Exact Hits:            {metrics['exact_hits']}")
        print(f"  Semantic Hits:         {metrics['semantic_hits']}")
        print(f"Total Tokens Saved:      {metrics['total_tokens']}")
        print(f"Total Model Cost:        ${metrics['total_cost']:.4f}")
        print(f"Accumulated Cost Saved:  ${metrics['cost_savings']:.4f}")
        print(f"Average Request Latency: {metrics['average_latency_ms']} ms")
        print(f"Routed Model Shares:")
        print(f"  Tier 1 (Low Cost):     {metrics['tier1_count']} requests")
        print(f"  Tier 2 (High Quality):  {metrics['tier2_count']} requests")
        print("===================================\n")
        
    elif args.command == "clear-cache":
        db_path = cfg.get("caching", {}).get("database_path", "agent_router.db")
        db = RouterDB(db_path)
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        print("Agent Router cache cleared successfully.")

if __name__ == "__main__":
    main()
