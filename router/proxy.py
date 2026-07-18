import os
import json
import time
import httpx
import re
from typing import Dict, Any, List, Tuple, AsyncGenerator, Optional
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

from router.db import RouterDB
from router.cache import CacheEngine
from router.classifier import ComplexityClassifier

# Standard pricing mapping per 1,000,000 tokens
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
}

class ProxyHandler:
    def __init__(self, db: RouterDB, cache: CacheEngine, classifier: ComplexityClassifier, config: Dict[str, Any]):
        self.db = db
        self.cache = cache
        self.classifier = classifier
        self.config = config
        self.tiers = config.get("tiers", {})
        self.cascade_enabled = config.get("routing", {}).get("cascade_enabled", True)

    def get_tier_settings(self, tier: int) -> Tuple[str, str, str]:
        """Returns provider, model, and api_key env variable for a given tier."""
        tier_key = f"tier{tier}"
        settings = self.tiers.get(tier_key, {})
        provider = settings.get("provider", "openai")
        model = settings.get("model", "")
        
        if not model:
            if tier == 1:
                model = "gpt-4o-mini"
            else:
                model = "gpt-4o"
                
        api_key_env = settings.get("api_key_env", f"{provider.upper()}_API_KEY")
        return provider, model, api_key_env

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Tuple[float, float]:
        """Calculates input and output costs based on model pricing."""
        local_routing_config = self.config.get("local_routing", {})
        local_enabled = local_routing_config.get("enabled", False)
        t1_local_model = local_routing_config.get("tier1", {}).get("model", "")
        t2_local_model = local_routing_config.get("tier2", {}).get("model", "")

        if local_enabled and (model == t1_local_model or model == t2_local_model or "ollama" in model.lower()):
            return 0.0, 0.0

        match_model = "gpt-4o-mini"
        for key in PRICING:
            if key in model.lower():
                match_model = key
                break
                
        rates = PRICING.get(match_model, {"input": 0.15, "output": 0.60})
        input_cost = (input_tokens * rates["input"]) / 1_000_000.0
        output_cost = (output_tokens * rates["output"]) / 1_000_000.0
        return input_cost, output_cost

    def parse_openai_usage(self, data: Dict[str, Any]) -> Tuple[int, int]:
        """Extracts input and output tokens from standard OpenAI response."""
        usage = data.get("usage", {})
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def validate_output(self, text: str) -> Tuple[bool, str]:
        """Validates LLM output for common syntax errors and code failures."""
        if not text or len(text.strip()) < 10:
            return False, "Response is too short or empty"
            
        blocks = re.findall(r'```[a-zA-Z0-9#\+\-]*\n(.*?)\n```', text, re.DOTALL)
        for block in blocks:
            # Check Python syntax errors
            if "IndentationError:" in block or "TabError:" in block or "SyntaxError:" in block:
                return False, "Contains code compilation error indicators"
            
            # Check curly brackets match
            if block.count('{') != block.count('}'):
                return False, "Mismatched curly braces in code block"
                
        # Check truncation
        last_char = text.strip()[-1]
        if last_char not in ['.', '!', '?', '"', "'", '}', ']', ')', '`', ';', '>']:
            return False, "Response appears truncated (ends abruptly)"
            
        return True, "Validation passed"

    def openai_to_anthropic_req(self, openai_req: Dict[str, Any], target_model: str) -> Dict[str, Any]:
        """Converts OpenAI request payload to Anthropic structure, adding prompt caching control if enabled."""
        messages = openai_req.get("messages", [])
        system_parts = []
        filtered_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            system_parts.append(p.get("text", ""))
            else:
                filtered_messages.append(msg)

        normalized_messages = []
        for msg in filtered_messages:
            role = msg.get("role")
            if role not in ["user", "assistant"]:
                role = "user"
                
            content = msg.get("content")
            if not content:
                continue
                
            if normalized_messages and normalized_messages[-1]["role"] == role:
                old_content = normalized_messages[-1]["content"]
                if isinstance(old_content, str) and isinstance(content, str):
                    normalized_messages[-1]["content"] = old_content + "\n" + content
                else:
                    normalized_messages[-1]["content"] = str(old_content) + "\n" + str(content)
            else:
                normalized_messages.append({"role": role, "content": content})

        if normalized_messages and normalized_messages[0]["role"] != "user":
            normalized_messages.insert(0, {"role": "user", "content": "Continuing context..."})

        anthropic_req = {
            "model": target_model,
            "messages": normalized_messages,
            "max_tokens": openai_req.get("max_tokens", 4096),
        }
        
        prompt_caching_enabled = self.config.get("caching", {}).get("prompt_caching_enabled", True)
        if system_parts:
            system_text = "\n".join(system_parts)
            if prompt_caching_enabled:
                anthropic_req["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                anthropic_req["system"] = system_text
            
        if "temperature" in openai_req:
            anthropic_req["temperature"] = openai_req["temperature"]
            
        if "stream" in openai_req:
            anthropic_req["stream"] = openai_req["stream"]
            
        return anthropic_req

    async def handle_request(self, request: Request) -> Any:
        """Core handler for proxying completions with caching and complexity routing."""
        start_time = time.time()
        
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        messages = body.get("messages", [])
        requested_model = body.get("model", "default")
        stream = body.get("stream", False)

        # Optimization 2: Canonicalize system and user messages to stabilize prompt caching
        canonical_messages = self.classifier.canonicalize_messages(messages)
        body["messages"] = canonical_messages

        # 1. Lookup in Cache (using local TF-IDF if configured)
        cached_resp, cache_hit_type = await self.cache.lookup(canonical_messages)
        if cached_resp:
            duration_ms = int((time.time() - start_time) * 1000)
            print(f"Cache Hit ({cache_hit_type})! Serving immediately.")

            input_tokens = self.classifier.estimate_tokens(str(canonical_messages))
            output_tokens = self.classifier.estimate_tokens(str(cached_resp))
            
            self.db.log_request(
                prompt=self.cache.get_query_text(canonical_messages),
                complexity_score=0.0,
                routed_model="CACHED",
                requested_model=requested_model,
                provider="cache",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=0.0,
                output_cost=0.0,
                tier_selected=0,
                routing_reason=f"Served from {cache_hit_type} cache",
                duration_ms=duration_ms,
                cache_hit=cache_hit_type,
                success=1
            )

            if stream:
                async def stream_cache_replay() -> AsyncGenerator[bytes, None]:
                    chunk = {
                        "id": "chatcmpl-cached",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": cached_resp.get("choices", [{}])[0].get("message", {}).get("content", "")},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                return StreamingResponse(stream_cache_replay(), media_type="text/event-stream")
            
            return cached_resp

        # 2. Analyze complexity
        tier, complexity_score, reason = self.classifier.analyze_request(canonical_messages)
        
        # Optimization 1: Cascade cheap-first routing
        is_cascade = self.cascade_enabled and tier == 2
        
        if is_cascade:
            print(f"Cascade Active: Attempting cheap Tier 1 model first for complexity {complexity_score:.2f}.")
            run_tier = 1
            routing_reason = f"Cascade (Tier 1 Attempt); {reason}"
        else:
            run_tier = tier
            routing_reason = reason

        # Check local model routing overrides
        local_routing_config = self.config.get("local_routing", {})
        local_enabled = local_routing_config.get("enabled", False)
        
        is_local_routed = False
        local_provider = ""
        local_model = ""
        local_base_url = ""
        
        if local_enabled:
            max_threshold = local_routing_config.get("max_complexity_threshold", 3.0)
            if complexity_score >= max_threshold:
                print(f"Bypassing local routing: complexity score ({complexity_score:.2f}) exceeds local threshold ({max_threshold:.2f}).")
            else:
                if run_tier == 1:
                    is_local_routed = True
                    local_model = local_routing_config.get("tier1", {}).get("model", "qwen2.5-coder:7b")
                elif run_tier == 2 and local_routing_config.get("tier2", {}).get("enabled", False):
                    is_local_routed = True
                    local_model = local_routing_config.get("tier2", {}).get("model", "qwen2.5-coder:32b")
                    
                if is_local_routed:
                    local_provider = local_routing_config.get("provider", "ollama")
                    local_base_url = local_routing_config.get("base_url", "http://localhost:11434")

        if is_local_routed:
            provider = local_provider
            routed_model = local_model
            api_key = "local"
            routing_reason = f"Local Routing Override; {routing_reason}"
        else:
            provider, routed_model, api_key_env = self.get_tier_settings(run_tier)
            api_key = os.getenv(api_key_env)

            if not api_key:
                # Check backup env variables
                for key_env in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]:
                    fallback_key = os.getenv(key_env)
                    if fallback_key:
                        api_key = fallback_key
                        if "GEMINI" in key_env:
                            provider, routed_model = "gemini", "gemini-1.5-flash"
                        elif "ANTHROPIC" in key_env:
                            provider, routed_model = "anthropic", "claude-3-5-haiku-20241022"
                        else:
                            provider, routed_model = "openai", "gpt-4o-mini"
                        break
                
                if not api_key:
                    raise HTTPException(
                        status_code=500, 
                        detail="No API Keys found. Please set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY."
                    )

        body["model"] = routed_model

        if provider in ["ollama", "openai_compatible", "local"]:
            url = f"{local_base_url}/v1/chat/completions"
            headers = {"Authorization": "Bearer local", "Content-Type": "application/json"}
            return await self._dispatch_request(provider, url, headers, body, canonical_messages, complexity_score, routed_model, requested_model, run_tier, routing_reason, start_time, is_cascade)

        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            return await self._dispatch_request("openai", url, headers, body, canonical_messages, complexity_score, routed_model, requested_model, run_tier, routing_reason, start_time, is_cascade)

        elif provider == "gemini":
            url = "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            return await self._dispatch_request("gemini", url, headers, body, canonical_messages, complexity_score, routed_model, requested_model, run_tier, routing_reason, start_time, is_cascade)

        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            anthropic_body = self.openai_to_anthropic_req(body, routed_model)
            return await self._dispatch_request("anthropic", url, headers, anthropic_body, canonical_messages, complexity_score, routed_model, requested_model, run_tier, routing_reason, start_time, is_cascade)

        else:
            raise HTTPException(status_code=500, detail=f"Unsupported provider: {provider}")

    async def _dispatch_request(self, provider: str, url: str, headers: Dict[str, str], body: Dict[str, Any], 
                                  messages: List[Dict[str, Any]], complexity_score: float, 
                                  routed_model: str, requested_model: str, tier: int, 
                                  reason: str, start_time: float, is_cascade: bool) -> Any:
        """Dispatches the API request, and executes cascade fallback logic if enabled and validation fails."""
        stream = body.get("stream", False)

        if not stream:
            # Non-streaming cascade
            res_data, err_msg = await self._execute_call(provider, url, headers, body, routed_model)
            
            if err_msg:
                if is_cascade:
                    print(f"Tier 1 call failed ({err_msg}). Escalating to Tier 2 immediately.")
                    return await self._escalate_to_tier2(body, messages, complexity_score, requested_model, reason, start_time)
                else:
                    raise HTTPException(status_code=500, detail=f"Upstream API error: {err_msg}")

            # Parse content text
            content_text = ""
            if provider in ["openai", "gemini", "ollama", "openai_compatible", "local"]:
                content_text = res_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider == "anthropic":
                content_text = ""
                for block in res_data.get("content", []):
                    if block.get("type") == "text":
                        content_text += block.get("text", "")

            # Run Cascade Validation Check
            if is_cascade:
                valid, val_msg = self.validate_output(content_text)
                if not valid:
                    print(f"Tier 1 validation failed: {val_msg}. Escalating to Tier 2.")
                    # Log failure in database for tracking
                    in_tokens = self.classifier.estimate_tokens(str(messages))
                    out_tokens = self.classifier.estimate_tokens(content_text)
                    in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)
                    self.db.log_request(
                        prompt=self.cache.get_query_text(messages),
                        complexity_score=complexity_score,
                        routed_model=routed_model,
                        requested_model=requested_model,
                        provider=provider,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        input_cost=in_cost,
                        output_cost=out_cost,
                        tier_selected=1,
                        routing_reason=f"Cascade Try (Failed validation: {val_msg})",
                        duration_ms=int((time.time() - start_time) * 1000),
                        cache_hit="none",
                        success=0 # failed
                    )
                    # Escalate
                    return await self._escalate_to_tier2(body, messages, complexity_score, requested_model, f"Escalated (Tier 1 failed: {val_msg})", start_time)
                else:
                    print("Tier 1 validation passed. Returning Cascade response.")
                    # Succeeded! Let's return and log success
                    in_tokens = self.classifier.estimate_tokens(str(messages))
                    out_tokens = self.classifier.estimate_tokens(content_text)
                    in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Wrap Anthropic to OpenAI format if needed
                    if provider == "anthropic":
                        res_data = self._translate_anthropic_to_openai_res(res_data, routed_model)

                    self.db.log_request(
                        prompt=self.cache.get_query_text(messages),
                        complexity_score=complexity_score,
                        routed_model=routed_model,
                        requested_model=requested_model,
                        provider=provider,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        input_cost=in_cost,
                        output_cost=out_cost,
                        tier_selected=1,
                        routing_reason=f"Cascade Success; {reason}",
                        duration_ms=duration_ms,
                        cache_hit="none",
                        success=1 # success
                    )
                    await self.cache.save(messages, res_data, provider, routed_model)
                    return res_data

            # Normal Non-Streaming
            else:
                duration_ms = int((time.time() - start_time) * 1000)
                if provider == "anthropic":
                    res_data = self._translate_anthropic_to_openai_res(res_data, routed_model)
                
                in_tokens, out_tokens = self.parse_openai_usage(res_data)
                in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)

                self.db.log_request(
                    prompt=self.cache.get_query_text(messages),
                    complexity_score=complexity_score,
                    routed_model=routed_model,
                    requested_model=requested_model,
                    provider=provider,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    input_cost=in_cost,
                    output_cost=out_cost,
                    tier_selected=tier,
                    routing_reason=reason,
                    duration_ms=duration_ms,
                    cache_hit="none",
                    success=None # pending agent feedback
                )
                await self.cache.save(messages, res_data, provider, routed_model)
                return res_data

        # Streaming Request logic
        else:
            if is_cascade:
                # For streaming cascade: we must buffer chunks internally to check validation.
                # If valid -> yield buffered chunks. If invalid -> discard and stream Tier 2.
                print("Cascade Streaming: buffering Tier 1 stream for validation...")
                buffered_chunks = []
                accumulated_text = ""
                
                client = httpx.AsyncClient(timeout=60.0)
                if provider == "anthropic":
                    # For Anthropic streaming, we must yield OpenAI format chunks, let's parse them
                    req = client.build_request("POST", url, headers=headers, json=body)
                    try:
                        response = await client.send(req, stream=True)
                        response.raise_for_status()
                    except Exception as e:
                        await client.aclose()
                        return await self._escalate_to_tier2(body, messages, complexity_score, requested_model, f"Tier 1 stream error: {str(e)}", start_time)
                    
                    in_tokens = self.classifier.estimate_tokens(str(messages))
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip().startswith("data:"):
                                data_str = line.replace("data:", "").strip()
                                try:
                                    data_json = json.loads(data_str)
                                    if data_json.get("type") == "content_block_delta":
                                        txt = data_json.get("delta", {}).get("text", "")
                                        accumulated_text += txt
                                        
                                        openai_chunk = {
                                            "id": "chatcmpl-cascade",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": routed_model,
                                            "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}]
                                        }
                                        buffered_chunks.append(f"data: {json.dumps(openai_chunk)}\n\n".encode("utf-8"))
                                except Exception:
                                    pass
                    await response.aclose()
                    await client.aclose()

                else: # OpenAI / Gemini streaming
                    req = client.build_request("POST", url, headers=headers, json=body)
                    try:
                        response = await client.send(req, stream=True)
                        response.raise_for_status()
                    except Exception as e:
                        await client.aclose()
                        return await self._escalate_to_tier2(body, messages, complexity_score, requested_model, f"Tier 1 stream error: {str(e)}", start_time)

                    async for chunk in response.aiter_bytes():
                        buffered_chunks.append(chunk)
                        lines = chunk.decode("utf-8", errors="ignore").split("\n")
                        for line in lines:
                            if line.strip().startswith("data:"):
                                data_str = line.replace("data:", "").strip()
                                if data_str != "[DONE]":
                                    try:
                                        data_json = json.loads(data_str)
                                        txt = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if txt:
                                            accumulated_text += txt
                                    except Exception:
                                        pass
                    await response.aclose()
                    await client.aclose()

                # Validate buffered output
                valid, val_msg = self.validate_output(accumulated_text)
                if not valid:
                    print(f"Tier 1 stream validation failed: {val_msg}. Discarding and escalating to Tier 2.")
                    # Log failure in DB
                    in_tokens = self.classifier.estimate_tokens(str(messages))
                    out_tokens = self.classifier.estimate_tokens(accumulated_text)
                    in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)
                    self.db.log_request(
                        prompt=self.cache.get_query_text(messages),
                        complexity_score=complexity_score,
                        routed_model=routed_model,
                        requested_model=requested_model,
                        provider=provider,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        input_cost=in_cost,
                        output_cost=out_cost,
                        tier_selected=1,
                        routing_reason=f"Cascade Stream Try (Failed: {val_msg})",
                        duration_ms=int((time.time() - start_time) * 1000),
                        cache_hit="none",
                        success=0
                    )
                    # Escalate to Tier 2 stream
                    return await self._escalate_to_tier2(body, messages, complexity_score, requested_model, f"Escalated (Tier 1 failed: {val_msg})", start_time)
                else:
                    print("Tier 1 stream validation passed. Streaming buffered chunks to client.")
                    # Stream the buffered chunks
                    async def replay_buffered() -> AsyncGenerator[bytes, None]:
                        for chunk in buffered_chunks:
                            yield chunk
                        yield b"data: [DONE]\n\n"
                        
                        # Log success in DB
                        in_tokens = self.classifier.estimate_tokens(str(messages))
                        out_tokens = self.classifier.estimate_tokens(accumulated_text)
                        in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)
                        self.db.log_request(
                            prompt=self.cache.get_query_text(messages),
                            complexity_score=complexity_score,
                            routed_model=routed_model,
                            requested_model=requested_model,
                            provider=provider,
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            input_cost=in_cost,
                            output_cost=out_cost,
                            tier_selected=1,
                            routing_reason=f"Cascade Stream Success; {reason}",
                            duration_ms=int((time.time() - start_time) * 1000),
                            cache_hit="none",
                            success=1
                        )
                        # Save in cache
                        mock_res = {
                            "id": "chatcmpl-completed",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": routed_model,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": accumulated_text}, "finish_reason": "stop"}]
                        }
                        await self.cache.save(messages, mock_res, provider, routed_model)

                    return StreamingResponse(replay_buffered(), media_type="text/event-stream")

            # Normal streaming routing
            else:
                if provider in ["openai", "gemini", "ollama", "openai_compatible", "local"]:
                    return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model, requested_model, provider, tier, reason, start_time)
                else:
                    return await self._proxy_anthropic(url, headers, body, messages, complexity_score, routed_model, requested_model, provider, tier, reason, start_time)

    async def _execute_call(self, provider: str, url: str, headers: Dict[str, str], body: Dict[str, Any], routed_model: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Executes a synchronous POST API call returning json data or error string."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, headers=headers, json=body)
                if response.status_code != 200:
                    return None, f"Status code {response.status_code}: {response.text}"
                return response.json(), None
            except Exception as e:
                return None, str(e)

    async def _escalate_to_tier2(self, body: Dict[str, Any], messages: List[Dict[str, Any]], 
                                  complexity_score: float, requested_model: str, 
                                  reason: str, start_time: float) -> Any:
        """Escalates request execution to Tier 2 model."""
        local_routing_config = self.config.get("local_routing", {})
        local_enabled = local_routing_config.get("enabled", False)
        
        max_threshold = local_routing_config.get("max_complexity_threshold", 3.0)
        if local_enabled and local_routing_config.get("tier2", {}).get("enabled", False) and complexity_score < max_threshold:
            provider2 = local_routing_config.get("provider", "ollama")
            routed_model2 = local_routing_config.get("tier2", {}).get("model", "qwen2.5-coder:32b")
            api_key2 = "local"
            local_base_url = local_routing_config.get("base_url", "http://localhost:11434")
            
            body["model"] = routed_model2
            url = f"{local_base_url}/v1/chat/completions"
            headers = {"Authorization": "Bearer local", "Content-Type": "application/json"}
            return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model2, requested_model, provider2, 2, reason, start_time)
        else:
            provider2, routed_model2, api_key_env2 = self.get_tier_settings(2)
            api_key2 = os.getenv(api_key_env2)
            
            if not api_key2:
                raise HTTPException(status_code=500, detail=f"No API Key found for Tier 2: {api_key_env2}")

            body["model"] = routed_model2
            print(f"Escalation executing Tier 2 model: {provider2}/{routed_model2}")

            if provider2 == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key2}", "Content-Type": "application/json"}
                return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model2, requested_model, provider2, 2, reason, start_time)

            elif provider2 == "gemini":
                url = "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key2}", "Content-Type": "application/json"}
                return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model2, requested_model, provider2, 2, reason, start_time)

            elif provider2 == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": api_key2,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                anthropic_body = self.openai_to_anthropic_req(body, routed_model2)
                return await self._proxy_anthropic(url, headers, anthropic_body, messages, complexity_score, routed_model2, requested_model, provider2, 2, reason, start_time)

    def _translate_anthropic_to_openai_res(self, res_data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Translates Anthropic JSON response to OpenAI JSON format."""
        content_blocks = res_data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage = res_data.get("usage", {})
        return {
            "id": f"chatcmpl-{res_data.get('id', 'msg')}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text_content},
                "finish_reason": "stop" if res_data.get("stop_reason") == "end_turn" else res_data.get("stop_reason")
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
        }

    async def _proxy_openai(self, url: str, headers: Dict[str, str], body: Dict[str, Any], 
                            messages: List[Dict[str, Any]], complexity_score: float, 
                            routed_model: str, requested_model: str, provider: str, 
                            tier: int, reason: str, start_time: float) -> Any:
        client = httpx.AsyncClient(timeout=60.0)
        stream = body.get("stream", False)

        if not stream:
            try:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                res_data = response.json()
            except Exception as e:
                await client.aclose()
                raise HTTPException(status_code=500, detail=f"Upstream OpenAI API Error: {str(e)}")
            finally:
                await client.aclose()

            duration_ms = int((time.time() - start_time) * 1000)
            in_tokens, out_tokens = self.parse_openai_usage(res_data)
            in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)

            self.db.log_request(
                prompt=self.cache.get_query_text(messages),
                complexity_score=complexity_score,
                routed_model=routed_model,
                requested_model=requested_model,
                provider=provider,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                input_cost=in_cost,
                output_cost=out_cost,
                tier_selected=tier,
                routing_reason=reason,
                duration_ms=duration_ms,
                cache_hit="none",
                success=None
            )

            await self.cache.save(messages, res_data, provider, routed_model)
            return res_data
        else:
            req = client.build_request("POST", url, headers=headers, json=body)
            try:
                response = await client.send(req, stream=True)
                response.raise_for_status()
            except Exception as e:
                await client.aclose()
                raise HTTPException(status_code=500, detail=f"Upstream OpenAI API Stream Error: {str(e)}")

            async def openai_stream_generator() -> AsyncGenerator[bytes, None]:
                accumulated_text = ""
                input_tokens = self.classifier.estimate_tokens(str(messages))
                
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        
                        lines = chunk.decode("utf-8", errors="ignore").split("\n")
                        for line in lines:
                            if line.strip().startswith("data:"):
                                data_str = line.strip().replace("data:", "").strip()
                                if data_str == "[DONE]":
                                    continue
                                try:
                                    data_json = json.loads(data_str)
                                    choices = data_json.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            accumulated_text += content
                                except Exception:
                                    pass
                finally:
                    await response.aclose()
                    await client.aclose()

                    duration_ms = int((time.time() - start_time) * 1000)
                    output_tokens = self.classifier.estimate_tokens(accumulated_text)
                    in_cost, out_cost = self.calculate_cost(routed_model, input_tokens, output_tokens)

                    mock_res = {
                        "id": "chatcmpl-completed",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": routed_model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": accumulated_text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
                    }

                    self.db.log_request(
                        prompt=self.cache.get_query_text(messages),
                        complexity_score=complexity_score,
                        routed_model=routed_model,
                        requested_model=requested_model,
                        provider=provider,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        input_cost=in_cost,
                        output_cost=out_cost,
                        tier_selected=tier,
                        routing_reason=reason,
                        duration_ms=duration_ms,
                        cache_hit="none",
                        success=None
                    )
                    await self.cache.save(messages, mock_res, provider, routed_model)

            return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")

    async def _proxy_anthropic(self, url: str, headers: Dict[str, str], body: Dict[str, Any], 
                               messages: List[Dict[str, Any]], complexity_score: float, 
                               routed_model: str, requested_model: str, provider: str, 
                               tier: int, reason: str, start_time: float) -> Any:
        client = httpx.AsyncClient(timeout=60.0)
        stream = body.get("stream", False)

        if not stream:
            try:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                res_data = response.json()
            except Exception as e:
                await client.aclose()
                raise HTTPException(status_code=500, detail=f"Upstream Anthropic API Error: {str(e)}")
            finally:
                await client.aclose()

            duration_ms = int((time.time() - start_time) * 1000)
            
            usage = res_data.get("usage", {})
            in_tokens = usage.get("input_tokens", 0)
            out_tokens = usage.get("output_tokens", 0)
            
            openai_res = self._translate_anthropic_to_openai_res(res_data, routed_model)
            in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)

            self.db.log_request(
                prompt=self.cache.get_query_text(messages),
                complexity_score=complexity_score,
                routed_model=routed_model,
                requested_model=requested_model,
                provider=provider,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                input_cost=in_cost,
                output_cost=out_cost,
                tier_selected=tier,
                routing_reason=reason,
                duration_ms=duration_ms,
                cache_hit="none",
                success=None
            )

            await self.cache.save(messages, openai_res, provider, routed_model)
            return openai_res
        else:
            req = client.build_request("POST", url, headers=headers, json=body)
            try:
                response = await client.send(req, stream=True)
                response.raise_for_status()
            except Exception as e:
                await client.aclose()
                raise HTTPException(status_code=500, detail=f"Upstream Anthropic Stream Error: {str(e)}")

            async def anthropic_stream_generator() -> AsyncGenerator[bytes, None]:
                accumulated_text = ""
                in_tokens = self.classifier.estimate_tokens(str(messages))
                out_tokens = 0
                msg_id = "msg-" + str(int(time.time()))
                
                try:
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or line.startswith("event:"):
                                continue
                            
                            if line.startswith("data:"):
                                data_str = line.replace("data:", "").strip()
                                try:
                                    data_json = json.loads(data_str)
                                    event_type = data_json.get("type")
                                    
                                    text_delta = ""
                                    if event_type == "message_start":
                                        msg_id = data_json.get("message", {}).get("id", msg_id)
                                        usage = data_json.get("message", {}).get("usage", {})
                                        in_tokens = usage.get("input_tokens", in_tokens)
                                    elif event_type == "content_block_delta":
                                        delta = data_json.get("delta", {})
                                        if delta.get("type") == "text_delta":
                                            text_delta = delta.get("text", "")
                                            accumulated_text += text_delta
                                    elif event_type == "message_delta":
                                        usage = data_json.get("usage", {})
                                        out_tokens = usage.get("output_tokens", out_tokens)

                                    if text_delta:
                                        openai_chunk = {
                                            "id": f"chatcmpl-{msg_id}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": routed_model,
                                            "choices": [{"index": 0, "delta": {"content": text_delta}, "finish_reason": None}]
                                        }
                                        yield f"data: {json.dumps(openai_chunk)}\n\n".encode("utf-8")
                                except Exception:
                                    pass
                finally:
                    await response.aclose()
                    await client.aclose()
                    yield b"data: [DONE]\n\n"

                    duration_ms = int((time.time() - start_time) * 1000)
                    if out_tokens == 0:
                        out_tokens = self.classifier.estimate_tokens(accumulated_text)
                    
                    in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)
                    mock_res = {
                        "id": f"chatcmpl-{msg_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": routed_model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": accumulated_text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
                    }

                    self.db.log_request(
                        prompt=self.cache.get_query_text(messages),
                        complexity_score=complexity_score,
                        routed_model=routed_model,
                        requested_model=requested_model,
                        provider=provider,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        input_cost=in_cost,
                        output_cost=out_cost,
                        tier_selected=tier,
                        routing_reason=reason,
                        duration_ms=duration_ms,
                        cache_hit="none",
                        success=None
                    )
                    await self.cache.save(messages, mock_res, provider, routed_model)

            return StreamingResponse(anthropic_stream_generator(), media_type="text/event-stream")
