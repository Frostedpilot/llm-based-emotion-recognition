import os
import re
import inspect
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try importing injectors, handle case where it's not yet discoverable
try:
    import backend.prompt_injectors as injectors
except ImportError:
    try:
        import prompt_injectors as injectors
    except ImportError:
        injectors = None

class PromptRegistry:
    """
    Registry for loading .txt prompt templates and filling them with 
    dynamic content from 'injector' functions.
    """
    def __init__(self, prompts_dir: Optional[str] = None):
        if not prompts_dir:
            # Detect prompts directory relative to this file
            base_dir = Path(__file__).resolve().parent.parent
            self.prompts_dir = base_dir / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
            
        self.injectors = {}
        self._discover_injectors()

    def _discover_injectors(self):
        """Auto-discovers injector functions from prompt_injectors.py."""
        if not injectors:
            return
            
        for name, func in inspect.getmembers(injectors, inspect.isfunction):
            # Relax the module check: as long as it contains 'prompt_injectors' 
            # (handles 'backend.prompt_injectors', 'prompt_injectors', etc)
            if 'prompt_injectors' in func.__module__:
                self.injectors[name] = func

    def list_available_injectors(self) -> List[Dict[str, str]]:
        """Returns a list of injection function names and their docstrings."""
        return [
            {"name": name, "description": (func.__doc__ or "").strip()}
            for name, func in self.injectors.items()
        ]

    def render(self, template_name_or_path: str, **context) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Loads a template (by name like 'erc_cot.txt' or full path),
        fills placeholders, and Returns (OpenAI-style messages list, metadata).
        """
        # 1. Load template text
        template_text = self._load_template_text(template_name_or_path)
        
        # 2. Extract metadata (lines starting with # at the very beginning)
        metadata = {"media": []}
        clean_lines = []
        for line in template_text.splitlines():
            if line.strip().startswith("# MEDIA:"):
                media_str = line.strip().replace("# MEDIA:", "").strip()
                metadata["media"] = [m.strip().lower() for m in media_str.split(",") if m.strip()]
            elif not clean_lines and line.strip().startswith("#"):
                # Generic metadata or comments at top
                pass
            else:
                clean_lines.append(line)
        
        final_template = "\n".join(clean_lines)

        # 3. Identify placeholders and fill them
        def _get_replacement(match):
            p = match.group(1)
            if p in context:
                return str(context[p])
            if p in self.injectors:
                # print(f"[DEBUG] Triggering injector: {p}")
                return str(self.injectors[p](**context))
            return match.group(0) # Return as-is (literal brace)

        filled = re.sub(r"\{(\w+)\}", _get_replacement, final_template)
        
        # 4. Parse into messages
        messages = self._parse_to_messages(filled)
        
        # Debug: check if agent_history was actually filled
        history_marker = "### MODALITY AGENT REASONING HISTORY"
        if history_marker in filled:
            after_marker = filled.split(history_marker)[1].split("\n")[1:5] # next few lines
            # if "(No agent history recorded yet)" in "".join(after_marker):
            #    print(f"[DEBUG] History for {template_name_or_path} is EMPTY. Context keys: {list(context.keys())}")
        
        return messages, metadata

    def _load_template_text(self, name_or_path: str) -> str:
        # Check if it's a full path first
        if os.path.isfile(name_or_path):
            with open(name_or_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # Check in prompts_dir
        path = self.prompts_dir / name_or_path
        if not path.is_file():
            # Try adding .txt if missing
            path = self.prompts_dir / (name_or_path + ".txt")
            
        if not path.is_file():
            # Fallback to default if everything fails
            raise FileNotFoundError(f"Template not found: {name_or_path} (checked {self.prompts_dir})")
            
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_to_messages(self, text: str) -> List[Dict[str, str]]:
        """
        Splits text into messages based on ## SYSTEM and ## USER indicators.
        If no indicators are found, treats whole text as a user message.
        """
        # re.split with capture group to keep the delimiters
        parts = re.split(r"(##\s*SYSTEM|##\s*USER)", text, flags=re.IGNORECASE)
        
        messages = []
        current_role = "user" # default
        
        for i in range(len(parts)):
            p = parts[i].strip()
            if not p: continue
            
            if re.match(r"##\s*SYSTEM", p, re.IGNORECASE):
                current_role = "system"
            elif re.match(r"##\s*USER", p, re.IGNORECASE):
                current_role = "user"
            else:
                # This is content
                messages.append({"role": current_role, "content": p})
                
        if not messages:
            return [{"role": "user", "content": text.strip()}]
            
        return messages

# --- Compatibility Layer ---
# These functions will be deprecated in favor of using PromptRegistry(dir).render(...) directly

_global_registry = None

def get_registry() -> PromptRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry()
    return _global_registry

def get_erc_messages(dataset_name, utterances, target_index, window_size=5, soft_label=False):
    """
    Backward-compatible wrapper for get_erc_messages.
    """
    template = "erc_cot" if "cot" in dataset_name.lower() else "erc_default"
    
    context = {
        "dataset_name": dataset_name,
        "utterances": utterances,
        "target_index": target_index,
        "window_size": window_size
    }
    
    reg = get_registry()
    messages, _metadata = reg.render(template, **context)
    
    if soft_label:
        # Append instruction to the last system or user message
        messages[-1]["content"] += "\n\nPlease provide a probability distribution over the emotions."
        
    return messages

def format_conversation_context(utterances, target_index, window_size=5):
    """Deprecated: use prompt_injectors.context_and_target instead."""
    import backend.prompt_injectors as p_inj
    return p_inj.context_and_target(utterances, target_index, window_size)

def get_emotion_detection_system_prompt(dataset_name="MELD"):
    """Deprecated: logic moved to prompt_injectors.emotions."""
    import backend.prompt_injectors as p_inj
    emotions = p_inj.emotions(dataset_name)
    return f"You are an expert in ERC. AVAILABLE EMOTIONS: {emotions}"
