import re
import json
import numpy as np

EMOTION_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happiness",
    "sad": "sadness",
    "ang": "anger",
    "fru": "frustration",
    "exc": "excitement",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": "other"
}

def unify_label(label: str) -> str:
    """Converts dataset-specific shorthands (e.g., 'neu') to full names (e.g., 'neutral')."""
    if not label:
        return ""
    low = label.lower().strip()
    return EMOTION_LABEL_MAP.get(low, low)

def normalize_prediction(raw: str, valid_labels: list) -> str:
    """
    Extracts and normalizes an emotion label from raw LLM output.
    Priority:
    1. Look for "Emotion: [label]" pattern (case-insensitive).
    2. If not found, look for any standalone valid label in the text.
    3. Fallback to the original loose matching logic.
    """
    raw_clean = raw.strip()
    
    # 1. Try regex for "Emotion: <label>"
    # This handles "Emotion: joy", "Emotion: **joy**", etc.
    # We look for a word that matches one of the valid labels after "Emotion:"
    match = re.search(r"(?i)Emotion:\s*[*_`\"']*([a-zA-Z]+)[*_`\"']*", raw_clean)
    if match:
        candidate = match.group(1).lower()
        for label in valid_labels:
            if label.lower() == candidate:
                return unify_label(label)

    # 2. If no "Emotion:" prefix, try looking at the last line 
    # (often models put the final answer at the end)
    lines = [l.strip() for l in raw_clean.split('\n') if l.strip()]
    if lines:
        last_line = re.sub(r"[*_`\"'.]", "", lines[-1]).lower()
        for label in valid_labels:
            if label.lower() == last_line:
                return unify_label(label)
                
    # 3. Fallback: current loose matching (containment)
    # but prioritize exact match first
    raw_lower = re.sub(r"[*_`\"'.]", "", raw_clean).lower()
    for label in valid_labels:
        if label.lower() == raw_lower:
            return unify_label(label)
            
    # Check if any label is contained in the raw output using word boundaries
    for label in valid_labels:
        if re.search(rf"\b{re.escape(label.lower())}\b", raw_lower):
            return unify_label(label)
            
    return unify_label(raw_clean) # Return simplified raw if no match found

def parse_soft_prediction(raw: str, valid_labels: list) -> dict:
    """
    Parses LLM output as a probability distribution dictionary.
    Handles nested keys (like "Soft Labels": {...}) and non-standard JSON.
    Returns None if parsing fails.
    """
    try:
        clean_raw = raw.strip()
        
        # 1. Try to find the specific "Soft Labels" dictionary block
        # This handles cases where the model wraps it in "Soft Labels": { ... }
        block_match = re.search(r"(?i)Soft\s*Labels\s*[:\s]*({.+?})", clean_raw, re.DOTALL)
        if block_match:
            dict_str = block_match.group(1)
        else:
            # Fallback: look for the last/only { } block in the entire text
            all_blocks = re.findall(r"({.+?})", clean_raw, re.DOTALL)
            dict_str = all_blocks[-1] if all_blocks else clean_raw

        # 2. Try standard JSON load first
        try:
            data = json.loads(dict_str)
        except json.JSONDecodeError:
            # 3. Regex Fallback: manual pair extraction (handles missing quotes etc.)
            # Looks for "label": 0.5 or label: 0.5
            pairs = re.findall(r"['\"`]?(\w+)['\"`]?\s*:\s*(\d*\.?\d+)", dict_str)
            data = {k: float(v) for k, v in pairs}

        if not isinstance(data, dict) or not data:
            return None
            
        # Normalize keys and values
        result = {}
        total = 0.0
        
        # We prioritize labels in valid_labels
        for label in valid_labels:
            # Case-insensitive match for keys
            key = next((k for k in data.keys() if k.lower() == label.lower()), None)
            val = float(data[key]) if key is not None else 0.0
            result[label.lower()] = max(0.0, val)
            total += result[label.lower()]
            
        if total > 0:
            for label in result:
                result[label] /= total
        else:
            return None # Force fallback to neutral in caller if sum is 0
                
        return result
        
    except Exception:
        return None

def calculate_js_divergence(p_dist: dict, g_dist: dict) -> float:
    """
    Calculates Jensen-Shannon Divergence between two probability distributions.
    0 means identical, 1 means completely disjoint.
    """
    # Align probability spaces
    all_labels = sorted(list(set(p_dist.keys()) | set(g_dist.keys())))
    
    p = np.array([p_dist.get(l, 0.0) for l in all_labels], dtype=np.float64)
    g = np.array([g_dist.get(l, 0.0) for l in all_labels], dtype=np.float64)
    
    # Re-normalize to ensure they sum to 1
    p /= (p.sum() + 1e-12)
    g /= (g.sum() + 1e-12)
    
    # Mixture distribution
    m = 0.5 * (p + g)
    
    # Helper for KL Divergence
    def _kld(a, b):
        # Only compute where a > 0
        idx = a > 0
        return np.sum(a[idx] * np.log(a[idx] / (b[idx] + 1e-12)))
    
    # JS Divergence (raw)
    jsd = 0.5 * _kld(p, m) + 0.5 * _kld(g, m)
    return float(max(0.0, jsd))

def get_argmax_label(dist: dict) -> str:
    """Returns the key with the highest probability in a distribution."""
    if not dist:
        return ""
    # Return the label with the max value. If ties, returns the first one encountered.
    return max(dist, key=dist.get)
