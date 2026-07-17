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
        self.compress_context_enabled = self.routing_config.get("compress_context", True)

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count based on character length (approx 4 chars per token)."""
        return len(text) // 4

    def compress_code_context(self, text: str) -> str:
        """Strips comments and collapses white spaces from code blocks in the text to optimize tokens."""
        if not self.compress_context_enabled:
            return text

        # Match triple backtick code blocks: ```lang ... ```
        pattern = r"(```[a-zA-Z0-9#\+\-]*\n)(.*?)(```)"
        
        def replace_block(match):
            header = match.group(1)
            code = match.group(2)
            footer = match.group(3)
            
            lang = header.replace("```", "").strip().lower()
            lines = code.split("\n")
            cleaned_lines = []
            
            for line in lines:
                stripped = line.strip()
                # Strip Python, Ruby, Shell comments
                if lang in ["python", "py", "ruby", "rb", "bash", "sh", "yaml", "yml", "dockerfile"]:
                    if stripped.startswith("#") and not stripped.startswith("#!"):
                        continue
                # Strip C-style comments (//)
                elif lang in ["javascript", "js", "typescript", "ts", "go", "java", "c", "cpp", "h", "hpp", "rust", "rs", "css"]:
                    if stripped.startswith("//"):
                        continue
                        
                cleaned_lines.append(line.rstrip())
                
            # Collapse multiple empty lines
            collapsed_lines = []
            prev_empty = False
            for line in cleaned_lines:
                if not line.strip():
                    if not prev_empty:
                        collapsed_lines.append("")
                        prev_empty = True
                else:
                    collapsed_lines.append(line)
                    prev_empty = False
                    
            return header + "\n".join(collapsed_lines) + "\n" + footer

        try:
            return re.sub(pattern, replace_block, text, flags=re.DOTALL)
        except Exception:
            return text

    def canonicalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groups system messages first and normalizes dynamic dates/times to stabilize prefix caching."""
        system_msgs = []
        other_msgs = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                if isinstance(content, str):
                    # Replace standard YYYY-MM-DD and HH:MM:SS patterns
                    content = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<canonical_date>', content)
                    content = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<canonical_time>', content)
                elif isinstance(content, list):
                    # Handle multimodal/block format
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            txt = part.get("text", "")
                            txt = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<canonical_date>', txt)
                            txt = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<canonical_time>', txt)
                            part["text"] = txt
                            
                system_msgs.append({"role": "system", "content": content})
            else:
                # If compression is enabled and content is user text, compress internal code blocks
                if self.compress_context_enabled and role == "user":
                    if isinstance(content, str):
                        content = self.compress_code_context(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                part["text"] = self.compress_code_context(part.get("text", ""))
                                
                other_msgs.append(msg)
                
        return system_msgs + other_msgs

    def analyze_request(self, messages: List[Dict[str, Any]]) -> Tuple[int, float, str]:
        """
        Analyzes the chat history to determine complexity and recommend a model tier.
        Returns:
            Tuple of (tier_selected: int, score: float, reasoning: str)
        """
        if not messages:
            return self.default_tier, 0.0, "Empty messages. Defaulted to tier."

        system_text = ""
        user_text = ""
        total_text = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
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

        score = 0.0
        reason_parts = []

        # 1. Keywords evaluation
        user_text_lower = user_text.lower()
        
        t2_matches = [kw for kw in self.tier2_keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', user_text_lower)]
        if t2_matches:
            score += len(t2_matches) * 0.8
            reason_parts.append(f"Matched high-complexity keywords: {t2_matches}")

        t1_matches = [kw for kw in self.tier1_keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', user_text_lower)]
        if t1_matches:
            score -= len(t1_matches) * 0.6
            reason_parts.append(f"Matched low-complexity keywords: {t1_matches}")

        # 2. Structure evaluation
        if "diff --git" in user_text or ("+++" in user_text and "---" in user_text):
            score += 2.0
            reason_parts.append("Contains file diff patterns")

        exception_patterns = [r"exception", r"traceback", r"error\:", r"fatal", r"failed at", r"nullpointer"]
        trace_matches = [p for p in exception_patterns if re.search(p, user_text_lower)]
        if trace_matches:
            score += 1.5
            reason_parts.append(f"Detected stack trace or error keywords: {trace_matches}")

        if "```" in user_text and score >= 0.0:
            score += 0.5
            reason_parts.append("Prompt contains code blocks")

        file_extensions = [r"\.py\b", r"\.js\b", r"\.ts\b", r"\.json\b", r"\.html\b", r"\.css\b", r"\.java\b", r"\.cpp\b", r"\.go\b"]
        ext_matches = [ext for ext in file_extensions if re.search(ext, user_text_lower)]
        if ext_matches:
            score += min(len(ext_matches) * 0.4, 1.2)
            reason_parts.append(f"Contains source file extensions: {ext_matches}")

        # 3. Context Length Check
        total_len = len(total_text)
        token_estimate = self.estimate_tokens(total_text)
        
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
