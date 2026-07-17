import re
from typing import Dict, Any, List, Tuple

class ComplexityClassifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.routing_config = config.get("routing", {})
        self.default_tier = self.routing_config.get("default_tier", 1)
        self.max_read_context_threshold = self.routing_config.get("max_read_context_threshold", 30000)
        self.tier2_keywords = self.routing_config.get("tier2_keywords", [])
        self.tier1_keywords = self.routing_config.get("tier1_keywords", [])

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count based on character length (approx 4 chars per token)."""
        return len(text) // 4

    def analyze_request(self, messages: List[Dict[str, Any]]) -> Tuple[int, float, str]:
        """
        Analyzes the chat history to determine complexity and recommend a model tier.
        Returns:
            Tuple of (tier_selected: int, score: float, reasoning: str)
        """
        if not messages:
            return self.default_tier, 0.0, "Empty messages. Defaulted to tier."

        # Extract textual content from system and user roles
        system_text = ""
        user_text = ""
        assistant_text = ""
        total_text = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Content can be list (multimodal) or string
            text_val = ""
            if isinstance(content, str):
                text_val = content
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                text_val = " ".join(parts)

            total_text += "\n" + text_val
            if role == "system":
                system_text += "\n" + text_val
            elif role == "user":
                user_text += "\n" + text_val
            elif role == "assistant":
                assistant_text += "\n" + text_val

        score = 0.0
        reason_parts = []

        # 1. Evaluate user instruction keywords (case-insensitive)
        user_text_lower = user_text.lower()
        
        # Check Tier 2 keywords (high complexity indicators)
        t2_matches = [kw for kw in self.tier2_keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', user_text_lower)]
        if t2_matches:
            score += len(t2_matches) * 0.8
            reason_parts.append(f"Matched high-complexity keywords: {t2_matches}")

        # Check Tier 1 keywords (low complexity indicators)
        t1_matches = [kw for kw in self.tier1_keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', user_text_lower)]
        if t1_matches:
            score -= len(t1_matches) * 0.6
            reason_parts.append(f"Matched low-complexity keywords: {t1_matches}")

        # 2. Check for code structures, file paths, and diffs (suggests modification or debugging)
        # Check for diff symbols (+ / - lines) or patch references
        if "diff --git" in user_text or ("+++" in user_text and "---" in user_text):
            score += 2.0
            reason_parts.append("Contains file diff patterns")

        # Check for error stack traces or exceptions
        exception_patterns = [r"exception", r"traceback", r"error\:", r"fatal", r"failed at", r"nullpointer"]
        trace_matches = [p for p in exception_patterns if re.search(p, user_text_lower)]
        if trace_matches:
            score += 1.5
            reason_parts.append(f"Detected stack trace or error keywords: {trace_matches}")

        # Check if the user is asking to modify code blocks or write new scripts
        if "```" in user_text and score >= 0.0:
            score += 0.5
            reason_parts.append("Prompt contains code blocks")

        # Check for file path indicators (.py, .js, .cpp, etc.)
        file_extensions = [r"\.py\b", r"\.js\b", r"\.ts\b", r"\.json\b", r"\.html\b", r"\.css\b", r"\.java\b", r"\.cpp\b", r"\.go\b"]
        ext_matches = [ext for ext in file_extensions if re.search(ext, user_text_lower)]
        if ext_matches:
            score += min(len(ext_matches) * 0.4, 1.2)
            reason_parts.append(f"Contains source file extensions: {ext_matches}")

        # 3. Context Length Routing Logic
        total_len = len(total_text)
        token_estimate = self.estimate_tokens(total_text)
        
        # If context is very large (e.g., above 30,000 characters / ~8,000 tokens)
        # but complexity score indicates mostly explanation or search (score < 1.0)
        # Force Tier 1 to save massive token costs on large reads.
        if total_len > self.max_read_context_threshold and score < 1.2:
            tier_selected = 1
            reason_parts.append(f"Large read context ({token_estimate} est. tokens) and low score. Routed to Tier 1 to save cost.")
        elif score >= 1.2:
            tier_selected = 2
            reason_parts.append(f"High complexity score ({score:.2f}). Routed to Tier 2.")
        else:
            tier_selected = 1
            reason_parts.append(f"Low complexity score ({score:.2f}). Routed to Tier 1.")

        reasoning = "; ".join(reason_parts) if reason_parts else "Default routing."
        return tier_selected, score, reasoning
