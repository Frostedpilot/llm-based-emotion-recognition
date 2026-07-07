from typing import List, Dict, Any

# --- Data Configuration ---
EMOTION_SETS = {
    "meld":    "neutral, surprise, fear, sadness, joy, disgust, anger",
    "camer":   "neutral, surprise, fear, sadness, joy, disgust, anger",
    "iemocap_test_6class": "neutral, frustration, excitement, sadness, anger, happiness",
    "iemocap": "neutral, happiness, sadness, anger, fear, frustration, excitement, surprise, disgust",
}

def emotions(dataset_name: str, **kwargs) -> str:
    """
    Returns a comma-separated list of valid emotion labels for the current dataset.
    Derived from 'dataset_name' (e.g., MELD, IEMOCAP, CA-MER).
    """
    name_lower = dataset_name.lower()
    # Sort keys by length descending to prevent substring collisions (e.g., 'iemocap' matching 'iemocap_test_6class')
    for key in sorted(EMOTION_SETS.keys(), key=len, reverse=True):
        if key in name_lower:
            return EMOTION_SETS[key]
    # Default to MELD set
    return EMOTION_SETS["meld"]

def context_and_target(utterances: List[Dict[str, Any]], target_index: int, window_size: int = 5, **kwargs) -> str:
    """
    Formats the conversation context (prior utterances) and the target utterance.
    Includes any multimodal analysis if available in the utterance metadata.
    """
    start_index = max(0, target_index - window_size)
    context_turns = utterances[start_index : target_index]
    
    lines = ["CONVERSATION CONTEXT:"]
    if not context_turns:
        lines.append("(No previous context)")
    else:
        for utt in context_turns:
            lines.append(f"  {utt['speaker']}: {utt['text']}")
    
    target_utt = utterances[target_index]
    lines.append(f"\nTARGET UTTERANCE:\n  {target_utt['speaker']}: {target_utt['text']}")
    
    # Optional: Include multi-modal metadata if available
    metadata = target_utt.get('metadata', {})
    if metadata:
        lines.append("\nMULTI-MODAL ANALYSIS (from expert models):")
        if metadata.get('video_analysis'): 
            lines.append(f"  Video: {metadata['video_analysis']}")
        if metadata.get('audio_analysis'): 
            lines.append(f"  Audio: {metadata['audio_analysis']}")
            
    return "\n".join(lines)

def available_metadata(utterances: List[Dict[str, Any]], target_index: int, **kwargs) -> str:
    """
    Summarizes which multimodal features are actually present for the target.
    """
    m = utterances[target_index].get('metadata', {})
    features = [k for k, v in m.items() if v]
    if not features:
        return "No expert model analysis available."
    return f"Available features: {', '.join(features)}"

def meld_target_speaker(utterances: List[Dict[str, Any]], target_index: int, **kwargs) -> str:
    """
    Returns the name of the target speaker in the MELD dataset.
    """
    target_utt = utterances[target_index]
    speaker = target_utt.get('speaker', 'Unknown')
    return speaker

def meld_target_text(utterances: List[Dict[str, Any]], target_index: int, **kwargs) -> str:
    """
    Returns the text of the target utterance in the MELD dataset.
    """
    target_utt = utterances[target_index]
    text = target_utt.get('text', 'Unknown')
    return text

# ── Acoustic Baselines (MELD Full Dataset) ──────────────────────────────────

GLOBAL_ACOUSTIC_STATS = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": {"mean": 35.959, "std": 8.110},
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": {"mean": 0.114, "std": 0.070},
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": {"mean": 32.727, "std": 8.172},
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": {"mean": 35.840, "std": 8.477},
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": {"mean": 39.095, "std": 8.854},
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": {"mean": 6.367, "std": 4.456},
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope": {"mean": 120.783, "std": 169.899},
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope": {"mean": 106.382, "std": 185.477},
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope": {"mean": 64.445, "std": 116.012},
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope": {"mean": 61.886, "std": 115.224},
    "loudness_sma3_amean": {"mean": 0.519, "std": 0.217},
    "loudness_sma3_stddevNorm": {"mean": 0.553, "std": 0.171},
    "loudness_sma3_percentile20.0": {"mean": 0.267, "std": 0.170},
    "loudness_sma3_percentile50.0": {"mean": 0.477, "std": 0.240},
    "loudness_sma3_percentile80.0": {"mean": 0.764, "std": 0.296},
    "loudness_sma3_pctlrange0-2": {"mean": 0.498, "std": 0.213},
    "loudness_sma3_meanRisingSlope": {"mean": 5.764, "std": 2.819},
    "loudness_sma3_stddevRisingSlope": {"mean": 3.364, "std": 1.881},
    "loudness_sma3_meanFallingSlope": {"mean": 4.312, "std": 2.509},
    "loudness_sma3_stddevFallingSlope": {"mean": 2.634, "std": 1.402},
    "spectralFlux_sma3_amean": {"mean": 0.179, "std": 0.104},
    "spectralFlux_sma3_stddevNorm": {"mean": 0.777, "std": 0.246},
    "mfcc1_sma3_amean": {"mean": 15.952, "std": 5.398},
    "mfcc1_sma3_stddevNorm": {"mean": 0.711, "std": 6.834},
    "mfcc2_sma3_amean": {"mean": -3.849, "std": 8.778},
    "mfcc2_sma3_stddevNorm": {"mean": -0.216, "std": 84.089},
    "mfcc3_sma3_amean": {"mean": 4.606, "std": 6.727},
    "mfcc3_sma3_stddevNorm": {"mean": -1.671, "std": 225.520},
    "mfcc4_sma3_amean": {"mean": -10.111, "std": 6.646},
    "mfcc4_sma3_stddevNorm": {"mean": -1.615, "std": 112.713},
    "jitterLocal_sma3nz_amean": {"mean": 0.024, "std": 0.013},
    "jitterLocal_sma3nz_stddevNorm": {"mean": 1.153, "std": 0.476},
    "shimmerLocaldB_sma3nz_amean": {"mean": 1.174, "std": 0.325},
    "shimmerLocaldB_sma3nz_stddevNorm": {"mean": 0.590, "std": 0.171},
    "HNRdBACF_sma3nz_amean": {"mean": 6.067, "std": 2.232},
    "HNRdBACF_sma3nz_stddevNorm": {"mean": 0.677, "std": 11.144},
    "logRelF0-H1-H2_sma3nz_amean": {"mean": 4.091, "std": 5.169},
    "logRelF0-H1-H2_sma3nz_stddevNorm": {"mean": -0.030, "std": 126.858},
    "logRelF0-H1-A3_sma3nz_amean": {"mean": 15.689, "std": 6.551},
    "logRelF0-H1-A3_sma3nz_stddevNorm": {"mean": 0.478, "std": 21.047},
    "F1frequency_sma3nz_amean": {"mean": 741.492, "std": 154.983},
    "F1frequency_sma3nz_stddevNorm": {"mean": 0.304, "std": 0.084},
    "F1bandwidth_sma3nz_amean": {"mean": 1272.837, "std": 203.723},
    "F1bandwidth_sma3nz_stddevNorm": {"mean": 0.185, "std": 0.061},
    "F1amplitudeLogRelF0_sma3nz_amean": {"mean": -124.686, "std": 35.022},
    "F1amplitudeLogRelF0_sma3nz_stddevNorm": {"mean": -0.750, "std": 0.668},
    "F2frequency_sma3nz_amean": {"mean": 1722.845, "std": 262.570},
    "F2frequency_sma3nz_stddevNorm": {"mean": 0.136, "std": 0.035},
    "F2bandwidth_sma3nz_amean": {"mean": 1048.330, "std": 180.734},
    "F2bandwidth_sma3nz_stddevNorm": {"mean": 0.268, "std": 0.092},
    "F2amplitudeLogRelF0_sma3nz_amean": {"mean": -118.500, "std": 34.574},
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": {"mean": -0.771, "std": 0.353},
    "F3frequency_sma3nz_amean": {"mean": 2708.996, "std": 391.193},
    "F3frequency_sma3nz_stddevNorm": {"mean": 0.088, "std": 0.023},
    "F3bandwidth_sma3nz_amean": {"mean": 937.673, "std": 162.664},
    "F3bandwidth_sma3nz_stddevNorm": {"mean": 0.308, "std": 0.101},
    "F3amplitudeLogRelF0_sma3nz_amean": {"mean": -120.283, "std": 33.818},
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": {"mean": -0.730, "std": 0.340},
    "alphaRatioV_sma3nz_amean": {"mean": -7.193, "std": 4.696},
    "alphaRatioV_sma3nz_stddevNorm": {"mean": -1.041, "std": 20.240},
    "hammarbergIndexV_sma3nz_amean": {"mean": 16.987, "std": 5.352},
    "hammarbergIndexV_sma3nz_stddevNorm": {"mean": 0.468, "std": 0.707},
    "slopeV0-500_sma3nz_amean": {"mean": 0.044, "std": 0.020},
    "slopeV0-500_sma3nz_stddevNorm": {"mean": 0.614, "std": 10.691},
    "slopeV500-1500_sma3nz_amean": {"mean": -0.009, "std": 0.008},
    "slopeV500-1500_sma3nz_stddevNorm": {"mean": -3.783, "std": 364.176},
    "spectralFluxV_sma3nz_amean": {"mean": 0.229, "std": 0.129},
    "spectralFluxV_sma3nz_stddevNorm": {"mean": 0.528, "std": 0.169},
    "mfcc1V_sma3nz_amean": {"mean": 20.318, "std": 7.435},
    "mfcc1V_sma3nz_stddevNorm": {"mean": 0.687, "std": 24.382},
    "mfcc2V_sma3nz_amean": {"mean": -7.910, "std": 10.366},
    "mfcc2V_sma3nz_stddevNorm": {"mean": -1.696, "std": 181.744},
    "mfcc3V_sma3nz_amean": {"mean": 3.084, "std": 9.023},
    "mfcc3V_sma3nz_stddevNorm": {"mean": 4.212, "std": 224.850},
    "mfcc4V_sma3nz_amean": {"mean": -14.092, "std": 8.786},
    "mfcc4V_sma3nz_stddevNorm": {"mean": -1.050, "std": 17.233},
    "alphaRatioUV_sma3nz_amean": {"mean": -4.876, "std": 3.652},
    "hammarbergIndexUV_sma3nz_amean": {"mean": 12.980, "std": 4.196},
    "slopeUV0-500_sma3nz_amean": {"mean": 0.026, "std": 0.021},
    "slopeUV500-1500_sma3nz_amean": {"mean": -0.001, "std": 0.007},
    "spectralFluxUV_sma3nz_amean": {"mean": 0.139, "std": 0.102},
    "loudnessPeaksPerSec": {"mean": 4.215, "std": 1.367},
    "VoicedSegmentsPerSec": {"mean": 2.621, "std": 1.373},
    "MeanVoicedSegmentLengthSec": {"mean": 0.172, "std": 0.101},
    "StddevVoicedSegmentLengthSec": {"mean": 0.112, "std": 0.081},
    "MeanUnvoicedSegmentLength": {"mean": 0.228, "std": 0.178},
    "StddevUnvoicedSegmentLength": {"mean": 0.194, "std": 0.184},
    "equivalentSoundLevel_dBp": {"mean": -30.978, "std": 4.551}
}

def get_acoustic_descriptor(value: float, feature_key: str) -> str:
    """Maps a raw feature value to a Z-score based natural language descriptor."""
    stats = GLOBAL_ACOUSTIC_STATS.get(feature_key)
    if not stats or stats['std'] == 0:
        return f"{value:.3f}"
        
    z = (value - stats['mean']) / stats['std']
    
    if z > 1.5:  return "significantly higher than typical"
    if z > 0.5:  return "somewhat higher than typical"
    if z < -1.5: return "significantly lower than typical"
    if z < -0.5: return "somewhat lower than typical"
    return "within the average range"

def egemaps_features(utterances: List[Dict[str, Any]], target_index: int, **kwargs) -> str:
    """
    Extracts and formats eGeMAPS acoustic features for the target utterance.
    Returns DISCRETIZED text descriptors based on global MELD baselines.
    """
    try:
        from backend.media_utils import extract_egemaps
    except ImportError:
        from media_utils import extract_egemaps
    
    utt = utterances[target_index]
    # Check both video_path and audio_path
    video_path = utt.get("video_path") or utt.get("audio_path")
    
    if not video_path:
        return "No audio/video path found for acoustic analysis."
        
    features = extract_egemaps(video_path)
    if not features:
        return "Acoustic features unavailable for this turn."
        
    parts = []
    # Inject all 88 eGeMAPS features present in GLOBAL_ACOUSTIC_STATS
    for key, stats in GLOBAL_ACOUSTIC_STATS.items():
        if key in features:
            desc = get_acoustic_descriptor(features[key], key)
            parts.append(f"- {key}: {desc}")
            
    if not parts:
        return "Acoustic feature extraction yielded empty results."
        
    return "FULL eGeMAPS ACOUSTIC ANALYSIS (Global Baseline Comparison):\n" + "\n".join(parts)


# ── Agentic QoL Injectors ────────────────────────────────────────────────────

def last_output(**kwargs) -> str:
    """
    Returns the most recent agent output found in the context.
    Looks for keys ending in '_output' or '_out'.
    """
    # Filter for output keys, excluding 'last_output' itself to avoid recursion
    outputs = {k: v for k, v in kwargs.items() if (k.endswith('_output') or k.endswith('_out')) and k != 'last_output'}
    if not outputs:
        return "(No previous agent output found)"
    
    # Python dicts preserve insertion order. Get the last one.
    last_key = list(outputs.keys())[-1]
    return f"[{last_key.upper()}]:\n{outputs[last_key]}"

def agent_history(**kwargs) -> str:
    """
    Formats all previous agent steps into a clean chronological trace.
    Useful for 'Finalizer' agents who need the full picture.
    """
    # Look for any keys that store model outputs, ignoring duplicate acoustic/egemaps outputs
    outputs = {
        k: v for k, v in kwargs.items()
        if (k.endswith('_output') or k.endswith('_out')) 
        and k != 'last_output'
        and not any(x in k.lower() for x in ['acoustic', 'egemaps'])
    }
    
    if not outputs:
        # Fallback: check if they are nestled in a 'ctx' key
        if 'ctx' in kwargs and isinstance(kwargs['ctx'], dict):
            outputs = {
                k: v for k, v in kwargs['ctx'].items()
                if (k.endswith('_output') or k.endswith('_out'))
                and not any(x in k.lower() for x in ['acoustic', 'egemaps'])
            }

    if not outputs:
        return "(No agent history recorded yet)"
    
    lines = ["--- AGENT REASONING HISTORY ---"]
    for key, val in outputs.items():
        step_name = key.replace('_output', '').replace('_out', '').upper()
        # Clean up tags and reasoning trace if present
        import re
        clean_val = re.sub(r"<thought>.*?</thought>", "", str(val), flags=re.DOTALL).strip()
        lines.append(f"STEP: {step_name}\n{clean_val}\n")
    
    return "\n".join(lines)

def target_text(utterances: List[Dict[str, Any]], target_index: int, **kwargs) -> str:
    """
    Returns just the speaker and text of the target utterance.
    """
    utt = utterances[target_index]
    return f"{utt['speaker']}: {utt['text']}"
