import os
import json
import time
import httpx
from typing import Dict, Any, List, Tuple, AsyncGenerator
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

    def get_tier_settings(self, tier: int) -> Tuple[str, str, str]:
        """Returns provider, model, and api_key env variable for a given tier."""
        tier_key = f"tier{tier}"
        settings = self.tiers.get(tier_key, {})
        provider = settings.get("provider", "openai")
        model = settings.get("model", "")
        
        # Fallback to defaults if model is empty
        if not model:
            if tier == 1:
                model = "gpt-4o-mini"
            else:
                model = "gpt-4o"
                
        api_key_env = settings.get("api_key_env", f"{provider.upper()}_API_KEY")
        return provider, model, api_key_env

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Tuple[float, float]:
        """Calculates input and output costs based on model pricing."""
        # Find match in pricing dictionary (handling prefixes)
        match_model = "gpt-4o-mini" # default fallback
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

    def openai_to_anthropic_req(self, openai_req: Dict[str, Any], target_model: str) -> Dict[str, Any]:
        """Converts OpenAI request payload to Anthropic structure."""
        messages = openai_req.get("messages", [])
        
        # Extract system messages
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

        # Normalize messages for Anthropic (must alternate user/assistant, must start with user)
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
        
        if system_parts:
            anthropic_req["system"] = "\n".join(system_parts)
            
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

        # 1. Lookup in Cache
        cached_resp, cache_hit_type = await self.cache.lookup(messages)
        if cached_resp:
            duration_ms = int((time.time() - start_time) * 1000)
            print(f"Cache Hit ({cache_hit_type})! Serving immediately.")

            # Record metrics for cache hit
            input_tokens = self.classifier.estimate_tokens(str(messages))
            output_tokens = self.classifier.estimate_tokens(str(cached_resp))
            
            # Cached response is free
            self.db.log_request(
                prompt=self.cache.get_query_text(messages),
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
                cache_hit=cache_hit_type
            )

            # Return cached response (supporting streaming replay if client requested stream)
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
        tier, complexity_score, reason = self.classifier.analyze_request(messages)
        provider, routed_model, api_key_env = self.get_tier_settings(tier)
        api_key = os.getenv(api_key_env)

        if not api_key:
            # Fallback to other tiers or default keys
            print(f"Warning: API Key {api_key_env} not found in environment. Trying fallbacks.")
            # Search alternative key
            for key_env in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]:
                fallback_key = os.getenv(key_env)
                if fallback_key:
                    api_key = fallback_key
                    # Update provider & model based on what key we found
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

        print(f"Routing logic: Routed to Tier {tier} ({provider}/{routed_model}). Reason: {reason}")

        # Override request body model
        body["model"] = routed_model

        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model, requested_model, provider, tier, reason, start_time)

        elif provider == "gemini":
            # Use Gemini's OpenAI-compatible endpoint
            url = "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model, requested_model, provider, tier, reason, start_time)

        elif provider == "deepseek":
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            return await self._proxy_openai(url, headers, body, messages, complexity_score, routed_model, requested_model, provider, tier, reason, start_time)

        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            anthropic_body = self.openai_to_anthropic_req(body, routed_model)
            return await self._proxy_anthropic(url, headers, anthropic_body, messages, complexity_score, routed_model, requested_model, provider, tier, reason, start_time)

        else:
            raise HTTPException(status_code=500, detail=f"Unsupported provider: {provider}")

    async def _proxy_openai(self, url: str, headers: Dict[str, str], body: Dict[str, Any], 
                            messages: List[Dict[str, Any]], complexity_score: float, 
                            routed_model: str, requested_model: str, provider: str, 
                            tier: int, reason: str, start_time: float) -> Any:
        """Proxies requests to OpenAI-compatible endpoints."""
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

            # Log request
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
                cache_hit="none"
            )

            # Cache response
            await self.cache.save(messages, res_data, provider, routed_model)
            return res_data

        # Streaming Response
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
                        # Yield the raw chunk to client
                        yield chunk
                        
                        # Process chunk for logging & caching
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

                    # Save complete response to DB log and cache
                    duration_ms = int((time.time() - start_time) * 1000)
                    output_tokens = self.classifier.estimate_tokens(accumulated_text)
                    in_cost, out_cost = self.calculate_cost(routed_model, input_tokens, output_tokens)

                    # Create a mock completion response to save in cache
                    mock_res = {
                        "id": "chatcmpl-completed",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": routed_model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": accumulated_text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
                        }
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
                        cache_hit="none"
                    )

                    await self.cache.save(messages, mock_res, provider, routed_model)

            return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")

    async def _proxy_anthropic(self, url: str, headers: Dict[str, str], body: Dict[str, Any], 
                               messages: List[Dict[str, Any]], complexity_score: float, 
                               routed_model: str, requested_model: str, provider: str, 
                               tier: int, reason: str, start_time: float) -> Any:
        """Proxies requests to Anthropic and translates to OpenAI response structure."""
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
            
            # Extract usage
            usage = res_data.get("usage", {})
            in_tokens = usage.get("input_tokens", 0)
            out_tokens = usage.get("output_tokens", 0)
            
            # Extract content text
            content_blocks = res_data.get("content", [])
            text_content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")

            # Translate to OpenAI structure
            openai_res = {
                "id": f"chatcmpl-{res_data.get('id', 'msg')}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": routed_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_content
                    },
                    "finish_reason": "stop" if res_data.get("stop_reason") == "end_turn" else res_data.get("stop_reason")
                }],
                "usage": {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens
                }
            }

            in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)

            # Log request
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
                cache_hit="none"
            )

            # Cache response
            await self.cache.save(messages, openai_res, provider, routed_model)
            return openai_res

        # Streaming Response
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
                            if not line:
                                continue
                            
                            # Anthropic streaming events format:
                            # event: <event_name>
                            # data: <json>
                            if line.startswith("event:"):
                                continue
                            
                            if line.startswith("data:"):
                                data_str = line.replace("data:", "").strip()
                                try:
                                    data_json = json.loads(data_str)
                                    event_type = data_json.get("type")
                                    
                                    # Extract metadata or content delta
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
                                        # Yield converted OpenAI-style SSE chunk
                                        openai_chunk = {
                                            "id": f"chatcmpl-{msg_id}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": routed_model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": text_delta},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(openai_chunk)}\n\n".encode("utf-8")
                                except Exception:
                                    pass
                finally:
                    await response.aclose()
                    await client.aclose()

                    # Yield the standard [DONE] chunk
                    yield b"data: [DONE]\n\n"

                    # Complete metrics and caching
                    duration_ms = int((time.time() - start_time) * 1000)
                    if out_tokens == 0:
                        out_tokens = self.classifier.estimate_tokens(accumulated_text)
                    
                    in_cost, out_cost = self.calculate_cost(routed_model, in_tokens, out_tokens)

                    # Mock complete response
                    mock_res = {
                        "id": f"chatcmpl-{msg_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": routed_model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": accumulated_text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": in_tokens,
                            "completion_tokens": out_tokens,
                            "total_tokens": in_tokens + out_tokens
                        }
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
                        cache_hit="none"
                    )

                    await self.cache.save(messages, mock_res, provider, routed_model)

            return StreamingResponse(anthropic_stream_generator(), media_type="text/event-stream")
