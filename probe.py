#!/usr/bin/env python3
"""
probe.py — Quick one-off LLM tester for ERC experiments.

Usage examples:
  python probe.py                                          # interactive selector
  python probe.py -d meld_dev -c 3 -u 1                   # pick dataset/conversation/utterance
  python probe.py -d meld_dev -c 3 -u 1 -p prompts/erc_cot.txt
  python probe.py -d meld_dev -c 3 -u 1 -m qwen3:4b --no-think
  python probe.py -d iemocap -c 0 -u 2 --provider openrouter -m anthropic/claude-3.5-sonnet
  python probe.py -d meld_dev -c 3 -u 1 --window 3 --list-utterances
  python probe.py -d meld_dev -c 3 -u 1 --agent --workflow tva_theory_soft --soft-label   # Run internal multi-agent flow
  python probe.py -d meld_dev -c 3 -u 1 --agent --text-results results/text_run.json    # Run agent flow using pre-computed text predictions
"""

import json
import os
import time
import sys
import argparse
import re
import glob
import urllib.request
import urllib.error
from typing import Optional

# Ensure project root is in path for backend imports
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.utils import (
    normalize_prediction,
    unify_label,
    get_argmax_label,
    calculate_js_divergence,
)
from backend.prompt_templates import get_registry

# ── ANSI colours (degrade gracefully on Windows without VT support) ────────────
try:
    import colorama

    colorama.init()
    C_RESET = "\033[0m"
    C_BOLD = "\033[1m"
    C_DIM = "\033[2m"
    C_CYAN = "\033[96m"
    C_YELLOW = "\033[93m"
    C_GREEN = "\033[92m"
    C_MAGENTA = "\033[95m"
    C_BLUE = "\033[94m"
    C_RED = "\033[91m"
    C_GRAY = "\033[90m"
except ImportError:
    C_RESET = C_BOLD = C_DIM = C_CYAN = C_YELLOW = C_GREEN = ""
    C_MAGENTA = C_BLUE = C_RED = C_GRAY = ""

RULE = C_GRAY + "-" * 70 + C_RESET


def safe_print(text: str, color: str = "", end: str = "", flush: bool = True):
    """
    Prints text to the console while:
    1. Handling UnicodeEncodeError (sanitizing to ASCII if needed).
    2. Sanitizing carriage returns (\r) which cause line overwriting in terminal.
    3. Ensuring ANSI color codes are wrapped correctly.
    """
    if not text:
        return

    # Sanitize control characters that mess up terminal cursor
    text = text.replace("\r\n", "\n").replace("\r", " ")

    try:
        print(f"{color}{text}{C_RESET if color else ''}", end=end, flush=flush)
    except UnicodeEncodeError:
        # Fallback to ASCII for limited terminals (cp1252 etc)
        sanitized = text.encode("ascii", "replace").decode("ascii")
        print(f"{color}{sanitized}{C_RESET if color else ''}", end=end, flush=flush)


# ── Helpers ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "processed"
)
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DATASET_ALIASES = {
    "meld_dev": "meld_dev_processed.json",
    "meld_test": "meld_test_processed.json",
    "meld_train": "meld_train_processed.json",
    "iemocap": "iemocap_processed.json",
    "camer": "camer_processed.json",
    "meld_subset": "subsets/meld_subset_100.json",
    "iemocap_subset": "subsets/iemocap_subset_100.json",
    "camer_subset": "subsets/camer_subset_100.json",
    "meld_subset_200": "subsets/meld_subset_200_proportional_origdist.json",
    "iemocap_subset_200": "subsets/iemocap_subset_200_proportional_origdist.json",
    "meld_subset_500": "subsets/meld_subset_500_proportional_origdist.json",
    "iemocap_subset_500": "subsets/iemocap_subset_500_proportional_origdist.json",
}

DEFAULT_MODELS = {
    "local": "qwen3.5:4b",
    "openrouter": "qwen/qwen3.6-plus:free",
}

EMOTION_LABELS = {
    "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
    "camer": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
    "iemocap": [
        "neutral",
        "happiness",
        "sadness",
        "anger",
        "fear",
        "frustration",
        "excitement",
    ],
}


def get_labels(ds: str) -> list:
    for k, v in EMOTION_LABELS.items():
        if k in ds.lower():
            return v
    return EMOTION_LABELS["meld"]


def resolve_dataset(name: str) -> str:
    """Resolve short alias or file path to absolute path."""
    key = name.lower().replace("-", "_").replace(".json", "")
    if key in DATASET_ALIASES:
        return os.path.join(PROCESSED_DIR, DATASET_ALIASES[key])
    if os.path.isfile(name):
        return os.path.abspath(name)

    # Check subsets directory
    subset_matches = glob.glob(os.path.join(PROCESSED_DIR, "subsets", f"*{key}*"))
    if subset_matches:
        return subset_matches[0]

    # Fuzzy match inside processed dir
    pattern = os.path.join(PROCESSED_DIR, f"*{key}*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Dataset not found: '{name}'.\nAvailable: {list(DATASET_ALIASES)}"
    )


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_dataset_name(path: str) -> str:
    base = os.path.basename(path).lower()
    if "iemocap" in base:
        return "iemocap"
    if "camer" in base:
        return "camer"
    return "meld"


def list_datasets() -> list:
    files = glob.glob(os.path.join(PROCESSED_DIR, "*.json"))
    return sorted(files)


def list_prompt_templates() -> list:
    files = glob.glob(os.path.join(PROMPTS_DIR, "*.txt"))
    return sorted(files)


def interactive_pick(options: list, prompt: str, display_fn=None) -> int:
    """Show a numbered list, let user pick by index."""
    print()
    for i, opt in enumerate(options):
        label = display_fn(opt) if display_fn else str(opt)
        print(f"  {C_CYAN}{i}{C_RESET}  {label}")
    print()
    while True:
        try:
            raw = input(f"{C_BOLD}{prompt}{C_RESET} [0-{len(options)-1}]: ").strip()
            idx = int(raw)
            if 0 <= idx < len(options):
                return idx
            print(f"  Enter a number between 0 and {len(options)-1}.")
        except (ValueError, EOFError):
            print("  Please enter a valid number.")


def render_messages_as_prompt(messages: list, media_reqs: Optional[list] = None) -> str:
    """Pretty-print the messages as they would be sent."""
    media_reqs = [m.lower() for m in (media_reqs or [])]
    media_is_video = ("video" in media_reqs) or ("vision" in media_reqs)
    media_is_image = "image" in media_reqs
    media_is_audio = "audio" in media_reqs

    parts = []
    for m in messages:
        role_label = m["role"].upper()
        content = m["content"]
        if isinstance(content, list):
            # Multimodal content
            text_parts = []
            img_count = 0
            has_video = False
            has_audio = False
            for item in content:
                if item["type"] == "text":
                    text_parts.append(item["text"])
                elif item["type"] == "image_url":
                    img_count += 1
                elif item["type"] == "video_url":
                    has_video = True
                elif item["type"] == "input_audio":
                    has_audio = True

            if media_is_video:
                if has_video:
                    text_parts.append(f"{C_MAGENTA}[VIDEO INCLUDED IN PROMPT]{C_RESET}")
                elif img_count > 0:
                    text_parts.append(
                        f"{C_YELLOW}[VIDEO REQUESTED, SENT AS {img_count} KEYFRAME(S)]{C_RESET}"
                    )
                else:
                    text_parts.append(
                        f"{C_YELLOW}[VIDEO REQUESTED, NO VIDEO ATTACHED]{C_RESET}"
                    )
            elif media_is_image or img_count > 0:
                text_parts.append(f"{C_MAGENTA}[IMAGES INCLUDED: {img_count}]{C_RESET}")

            if media_is_audio:
                if has_audio:
                    text_parts.append(f"{C_MAGENTA}[AUDIO INCLUDED IN PROMPT]{C_RESET}")
                else:
                    text_parts.append(
                        f"{C_DIM}[AUDIO REQUESTED, NOT INCLUDED]{C_RESET}"
                    )

            content_str = "\n".join(text_parts)
        else:
            content_str = content

        parts.append(f"{C_DIM}[{role_label}]{C_RESET}\n{content_str}")
    return ("\n\n" + RULE + "\n\n").join(parts)


def print_section(title: str, content: str, color: str = C_CYAN):
    print()
    safe_print(RULE)
    safe_print(f" {title} ", color + C_BOLD, end="\n")
    safe_print(RULE, end="\n")
    safe_print(content, end="\n")


def stream_and_capture(generator) -> tuple[str, str]:
    """
    Stream chunks from the generator, printing in realtime.
    Returns (thinking_text, answer_text).
    """
    thinking_buf = []
    answer_buf = []

    in_thinking = False
    buffer = ""

    for chunk in generator:
        buffer += chunk

        # Process complete <thought>...</thought> tags from the buffer
        while True:
            if not in_thinking:
                open_pos = buffer.find("<thought>")
                if open_pos == -1:
                    # Flush everything before any potential partial tag
                    safe = buffer
                    # Keep last 8 chars in case of partial tag
                    if len(buffer) > 8:
                        safe = buffer[:-8]
                        buffer = buffer[-8:]
                    else:
                        buffer = ""
                    if safe:
                        print(safe, end="", flush=True)
                        answer_buf.append(safe)
                    break
                else:
                    # Flush answer text before <thought>
                    before = buffer[:open_pos]
                    if before:
                        print(before, end="", flush=True)
                        answer_buf.append(before)
                    buffer = buffer[open_pos + len("<thought>") :]
                    in_thinking = True
                    print(f"\n{C_MAGENTA}{C_DIM}[thinking...]{C_RESET}\n", flush=True)
            else:
                close_pos = buffer.find("</thought>")
                if close_pos == -1:
                    # Safe to flush all but partial closing tag
                    safe = buffer
                    if len(buffer) > 10:
                        safe = buffer[:-10]
                        buffer = buffer[-10:]
                    else:
                        buffer = ""
                    if safe:
                        print(f"{C_MAGENTA}{C_DIM}{safe}{C_RESET}", end="", flush=True)
                        thinking_buf.append(safe)
                    break
                else:
                    thinking_chunk = buffer[:close_pos]
                    if thinking_chunk:
                        print(
                            f"{C_MAGENTA}{C_DIM}{thinking_chunk}{C_RESET}",
                            end="",
                            flush=True,
                        )
                        thinking_buf.append(thinking_chunk)
                    buffer = buffer[close_pos + len("</thought>") :]
                    in_thinking = False
                    print(f"\n{C_MAGENTA}{C_DIM}[/thinking]{C_RESET}\n", flush=True)

    # Flush remainder
    if buffer:
        if in_thinking:
            print(f"{C_MAGENTA}{C_DIM}{buffer}{C_RESET}", end="", flush=True)
            thinking_buf.append(buffer)
        else:
            print(buffer, end="", flush=True)
            answer_buf.append(buffer)

    print()  # final newline

    return "".join(thinking_buf), "".join(answer_buf)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="probe.py — Quick one-off LLM tester for ERC experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="NAME",
        help="Dataset alias (meld_dev, meld_test, iemocap, camer) or file path",
    )
    parser.add_argument(
        "-c",
        "--conversation",
        metavar="IDX",
        type=int,
        help="Dialogue index within dataset (0-based)",
    )
    parser.add_argument(
        "-u",
        "--utterance",
        metavar="IDX",
        type=int,
        help="Utterance index within dialogue (0-based)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="FILE",
        help="Path to prompt template .txt file (default: prompts/erc_default.txt)",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        help="Model ID override (e.g. qwen3:4b, llama3.1, anthropic/claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openrouter"],
        default="local",
        help="Inference provider (default: local)",
    )
    parser.add_argument(
        "--window",
        metavar="N",
        type=int,
        default=5,
        help="Context window size — number of prior turns (default: 5)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking tokens (Ollama /no_think)",
    )
    parser.add_argument(
        "--max-tokens",
        metavar="N",
        type=int,
        default=65536,
        help="Max generation tokens incl. thinking (default: 65536)",
    )
    parser.add_argument(
        "--think-budget",
        metavar="N",
        type=int,
        default=None,
        help="Max tokens allowed for the thinking block (Ollama only, requires Ollama ≥0.6.4)",
    )
    parser.add_argument(
        "--temperature",
        metavar="F",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--agent", action="store_true", help="Use multi-agent mode via the backend"
    )
    parser.add_argument(
        "--workflow",
        metavar="NAME",
        default="reasoner_verifier",
        help="Select agentic workflow (reasoner_verifier, self_correction)",
    )
    parser.add_argument(
        "--backend-url",
        metavar="URL",
        default="http://localhost:8283",
        help="Backend server URL (default: http://localhost:8283)",
    )
    parser.add_argument(
        "--list-utterances",
        action="store_true",
        help="Print all utterances in the selected conversation and exit",
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="Print available datasets and exit"
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="Print available prompt templates and exit",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision (I-frame extraction) for the session",
    )
    parser.add_argument(
        "--vision-frames",
        metavar="N",
        type=int,
        default=3,
        help="Max number of vision frames to extract (default: 3)",
    )
    parser.add_argument(
        "--audio", action="store_true", help="Enable audio extraction for the session"
    )
    parser.add_argument(
        "--soft-label",
        action="store_true",
        help="Request probability distribution and calculate JS-divergence",
    )
    parser.add_argument("--text-results", type=str, help="Path to pre-computed text modality results JSON")
    parser.add_argument("--vision-results", type=str, help="Path to pre-computed vision modality results JSON")
    parser.add_argument("--audio-results", type=str, help="Path to pre-computed audio modality results JSON")

    args = parser.parse_args()

    # ── Quick list commands ───────────────────────────────────────────────────
    if args.list_datasets:
        print(f"\n{C_BOLD}Available datasets:{C_RESET}")
        for alias, fname in DATASET_ALIASES.items():
            path = os.path.join(PROCESSED_DIR, fname)
            exists = "[OK]" if os.path.isfile(path) else "[X]"
            print(f"  {C_CYAN}{exists}{C_RESET}  {C_BOLD}{alias:<18}{C_RESET}  {fname}")
        return

    if args.list_prompts:
        templates = list_prompt_templates()
        print(f"\n{C_BOLD}Available prompt templates:{C_RESET}")
        for t in templates:
            print(f"  {C_CYAN}{os.path.relpath(t)}{C_RESET}")
        return

    # ── Resolve dataset ───────────────────────────────────────────────────────
    dataset_path = None
    if args.dataset:
        try:
            dataset_path = resolve_dataset(args.dataset)
        except FileNotFoundError as e:
            print(f"{C_RED}Error: {e}{C_RESET}")
            sys.exit(1)
    else:
        files = list_datasets()
        if not files:
            print(f"{C_RED}No datasets found in {PROCESSED_DIR}{C_RESET}")
            sys.exit(1)
        print(f"\n{C_BOLD}Select dataset:{C_RESET}")
        idx = interactive_pick(
            files, "Dataset", display_fn=lambda p: os.path.basename(p)
        )
        dataset_path = files[idx]

    print(f"\n{C_DIM}Loading {os.path.basename(dataset_path)}...{C_RESET}")
    dialogues = load_dataset(dataset_path)
    dataset_name = infer_dataset_name(dataset_path)  # short: 'meld', 'iemocap', 'camer'
    dataset_alias = (
        args.dataset
        if args.dataset
        else os.path.splitext(os.path.basename(dataset_path))[0].replace(
            "_processed", ""
        )
    )
    print(f"{C_GREEN}[OK] Loaded {len(dialogues)} dialogues{C_RESET}")

    # --- Load Agentic Simulation Results ---
    simulated_results = {}

    def load_sim_results(path, modality_key):
        if not path:
            return
        if not os.path.isfile(path):
            # Try results dir if relative
            if not os.path.isabs(path) and not path.startswith("results"):
                 alt_path = os.path.join(RESULTS_DIR, path)
                 if os.path.isfile(alt_path):
                     path = alt_path

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for res in data.get("results", []):
                    d_id = res.get("dialogue_id")
                    u_id = res.get("utterance_id")
                    if d_id is not None and u_id is not None:
                        try:
                            u_id_parsed = int(u_id)
                        except (ValueError, TypeError):
                            u_id_parsed = str(u_id)
                        lookup_key = (str(d_id), u_id_parsed)
                        if lookup_key not in simulated_results:
                            simulated_results[lookup_key] = {}
                        simulated_results[lookup_key][modality_key] = res.get("prediction_raw", "")
            print(f"{C_DIM}  Loaded simulated {modality_key} results from {path}{C_RESET}")
        except Exception as e:
            print(f"{C_RED}Error loading {modality_key} results: {e}{C_RESET}")

    load_sim_results(args.text_results, "text_output")
    load_sim_results(args.vision_results, "vision_output")
    load_sim_results(args.audio_results, "audio_output")

    is_simulation = bool(simulated_results)

    # ── Resolve conversation ──────────────────────────────────────────────────
    conv_idx = args.conversation
    if conv_idx is None:
        print(f"\n{C_BOLD}Select conversation:{C_RESET}")
        conv_idx = interactive_pick(
            dialogues,
            "Conversation index",
            display_fn=lambda d: f"{d['dialogue_id']}  ({len(d['utterances'])} utterances)",
        )

    if conv_idx < 0 or conv_idx >= len(dialogues):
        print(
            f"{C_RED}Conversation index {conv_idx} out of range (0-{len(dialogues)-1}).{C_RESET}"
        )
        sys.exit(1)

    dialogue = dialogues[conv_idx]
    utterances = dialogue["utterances"]

    # ── List utterances mode ──────────────────────────────────────────────────
    if args.list_utterances:
        print(
            f"\n{C_BOLD}Dialogue {dialogue['dialogue_id']} - {len(utterances)} utterances:{C_RESET}"
        )
        for i, u in enumerate(utterances):
            label = f"{C_YELLOW}[{u['emotion']}]{C_RESET}"
            print(f"  {C_CYAN}{i:3}{C_RESET}  {label:30}  {u['speaker']}: {u['text']}")
        return

    # ── Resolve utterance ─────────────────────────────────────────────────────
    utt_idx = args.utterance
    if utt_idx is None:
        if "target_index" in dialogue:
            utt_idx = dialogue["target_index"]
            print(
                f"{C_DIM}Subsetting target: auto-selected utterance {utt_idx}{C_RESET}"
            )
        else:
            print(f"\n{C_BOLD}Select utterance in {dialogue['dialogue_id']}:{C_RESET}")
            utt_idx = interactive_pick(
                utterances,
                "Utterance index",
                display_fn=lambda u: f"{C_YELLOW}[{u['emotion']}]{C_RESET}  {u['speaker']}: {u['text'][:60]}",
            )

    if utt_idx < 0 or utt_idx >= len(utterances):
        print(
            f"{C_RED}Utterance index {utt_idx} out of range (0-{len(utterances)-1}).{C_RESET}"
        )
        sys.exit(1)

    target_utt = utterances[utt_idx]

    # ── Resolve prompt template ───────────────────────────────────────────────
    if args.prompt:
        prompt_path = args.prompt
    elif args.agent:
        # In multi-agent mode the agents have their own system instructions;
        # the template is only used to format the conversation context, so we
        # silently fall back to erc_default without asking the user.
        default_tpl = os.path.join(PROMPTS_DIR, "erc_default.txt")
        templates = list_prompt_templates()
        prompt_path = (
            default_tpl
            if os.path.isfile(default_tpl)
            else (templates[0] if templates else default_tpl)
        )
    else:
        default_tpl = os.path.join(PROMPTS_DIR, "erc_default.txt")
        templates = list_prompt_templates()
        if os.path.isfile(default_tpl) and not templates:
            prompt_path = default_tpl
        elif templates and not args.prompt:
            # If only one, use it silently; otherwise ask
            if len(templates) == 1:
                prompt_path = templates[0]
                print(f"{C_DIM}Using template: {os.path.relpath(prompt_path)}{C_RESET}")
            else:
                print(f"\n{C_BOLD}Select prompt template:{C_RESET}")
                idx = interactive_pick(
                    templates, "Template", display_fn=lambda p: os.path.relpath(p)
                )
                prompt_path = templates[idx]
        else:
            prompt_path = default_tpl

    if not os.path.isfile(prompt_path):
        print(f"{C_RED}Prompt template not found: {prompt_path}{C_RESET}")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    # ── Build prompt ──────────────────────────────────────────────────────────
    registry = get_registry()

    context = {
        "dataset_name": dataset_name,
        "utterances": utterances,
        "target_index": utt_idx,
        "window_size": args.window,
    }
    messages, metadata = registry.render(prompt_path, **context)
    labels = get_labels(dataset_name)
    uni_gt = unify_label(target_utt["emotion"])

    # ── Multimodal Handling ──────────────────────────────────────────────────
    from backend.media_utils import prepare_multimodal_content, get_absolute_media_path, extract_iframes

    media_reqs = metadata.get("media", [])
    wants_video = "video" in media_reqs
    wants_image = "image" in media_reqs or "vision" in media_reqs
    wants_audio = "audio" in media_reqs
    include_v = args.vision or wants_video or wants_image
    include_a = args.audio or wants_audio
    visual_mode = "video" if wants_video else "image"

    if (include_v or include_a) and args.provider == "openrouter":
        media_path = target_utt.get("video_path") or target_utt.get("audio_path")
        abs_media_path = get_absolute_media_path(media_path)

        if not abs_media_path and media_path:
            print(
                f"{C_YELLOW}  [!] Warning: Could not resolve media path: {media_path}{C_RESET}"
            )
            print(
                f"{C_DIM}      (Expected inside data/raw/ or relative to root){C_RESET}"
            )

        last_msg = messages[-1]
        if last_msg["role"] == "user":
            u_id = target_utt.get("utterance_id")
            last_msg["content"] = prepare_multimodal_content(
                last_msg["content"],
                media_path,
                include_v,
                include_audio=include_a,
                max_vision_frames=args.vision_frames,
                visual_mode=visual_mode,
                utterance_id=u_id,
            )

    # ── Resolve model ─────────────────────────────────────────────────────────
    model_id = args.model or DEFAULT_MODELS[args.provider]

    # Handle @presets
    from backend.agents_logic import MODEL_PRESETS

    if model_id in MODEL_PRESETS:
        model_id = MODEL_PRESETS[model_id]

    include_thinking = not args.no_think
    think_budget = args.think_budget  # None = unlimited

    # ── Print header ──────────────────────────────────────────────────────────
    print()
    print(RULE)
    header_info = f"{dataset_name.upper()}  -  conv {conv_idx}  -  utt {utt_idx}"
    if "original_dialogue_id" in dialogue:
        header_info = f"{header_info} (origin: {dialogue['original_dialogue_id']})"
    print(
        f"{C_BOLD}  probe.py{C_RESET}  {C_DIM}|  {header_info}  -  {model_id}  -  {args.provider}{C_RESET}"
    )
    print(RULE)
    uni_gt = unify_label(target_utt["emotion"])
    print(f"  {C_BOLD}Ground truth:{C_RESET} {C_YELLOW}{uni_gt}{C_RESET}")
    if "soft_labels" in target_utt and target_utt["soft_labels"]:
        sl = target_utt["soft_labels"]
        sl_str = ", ".join(f"{k}={v:.2f}" for k, v in sl.items())
        print(f"  {C_BOLD}Soft labels: {C_RESET}{C_DIM}{sl_str}{C_RESET}")
    if not args.agent:
        print(
            f"  {C_BOLD}Template:    {C_RESET}{C_DIM}{os.path.relpath(prompt_path)}{C_RESET}"
        )
    print(f"  {C_BOLD}Window:      {C_RESET}{args.window} prior turns")
    if not args.agent:
        thinking_label = (
            "off"
            if not include_thinking
            else (
                f"on  (budget: {think_budget} tokens)"
                if think_budget
                else "on  (unlimited – raise --max-tokens if answer is cut off)"
            )
        )
        print(f"  {C_BOLD}Thinking:    {C_RESET}{thinking_label}")
        print(f"  {C_BOLD}Max tokens:  {C_RESET}{args.max_tokens}")

        vision_status = (
            f"{C_GREEN}active{C_RESET}" if include_v else f"{C_DIM}inactive{C_RESET}"
        )
        audio_status = (
            f"{C_GREEN}active{C_RESET}" if include_a else f"{C_DIM}inactive{C_RESET}"
        )
        print(f"  {C_BOLD}Vision:      {C_RESET}{vision_status}")
        print(f"  {C_BOLD}Audio:       {C_RESET}{audio_status}")

    # ── Media Path Display ────────────────────────────────────────────────────
    if include_v or include_a:
        media_path = target_utt.get("video_path") or target_utt.get("audio_path")
        abs_media_path = get_absolute_media_path(media_path)
        if abs_media_path:
            print(f"  {C_BOLD}Media Path:  {C_RESET}{C_MAGENTA}{abs_media_path}{C_RESET}")
            
            if include_v and visual_mode == "image":
                kf_paths = extract_iframes(str(abs_media_path), max_frames=args.vision_frames)
                if kf_paths:
                    print(f"  {C_BOLD}Keyframes:   {C_RESET}")
                    for kf in kf_paths:
                        print(f"               {C_DIM}- {kf}{C_RESET}")
        else:
            print(f"  {C_BOLD}Media Path:  {C_RESET}{C_RED}[NOT FOUND]{C_RESET}")

    print(
        f"  {C_BOLD}Soft-Label:  {C_RESET}{C_GREEN if args.soft_label else C_DIM}{'active' if args.soft_label else 'inactive'}{C_RESET}"
    )

    # ── Section 1: Actual prompt ──────────────────────────────────────────────
    if not args.agent:
        print_section(
            "ACTUAL PROMPT SENT TO LLM",
            render_messages_as_prompt(messages, media_reqs),
            C_BLUE,
        )

    # ── Section 2 & 3: LLM response ──────────────────────────────────────────
    from backend.utils import normalize_prediction

    BACKEND_URL = args.backend_url.rstrip("/")

    if args.agent:
        # ── Multi-agent path (Internal Python Execution) ───────────────────
        # Instead of calling the backend server via HTTP, we run the workflow
        # internally within the python process using run_multiagent_stream.
        import asyncio
        from backend.agents_logic import run_multiagent_stream

        # Context contains references to the dialogue structures
        original_context = {
            "dataset_name": dataset_name,
            "dialogue_id": dialogue["dialogue_id"],
            "utterances": utterances,
            "target_index": utt_idx,
            "window_size": args.window,
            "vision_frames": args.vision_frames,
            "simulated_results": simulated_results, # Pass loaded pre-computed dictionary
            "soft_label": args.soft_label,
        }

        thinking_text = ""  # Reasoner output (used in Summary)
        answer_text = ""  # Verifier/Finalizer output
        final_label = ""
        finalizer_out_dist = {}

        AGENT_COLORS = {
            "TextAgent": C_BLUE,
            "VisionAgent": C_MAGENTA,
            "AcousticAgent": C_YELLOW,
            "ResolverAgent": C_GREEN,
            "ReasonerAgent": C_CYAN,
            "VerifierAgent": C_MAGENTA,
            "CriticAgent": C_YELLOW,
            "FinalizerAgent": C_GREEN,
        }
        response_header_printed = set()

        async def run_local_agent():
            nonlocal thinking_text, answer_text, final_label, finalizer_out_dist
            stream_gen = run_multiagent_stream(
                messages=messages,
                valid_emotions=", ".join(labels),
                model=model_id,
                provider=args.provider,
                workflow=args.workflow,
                original_context=original_context,
            )
            async for line in stream_gen:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    ev = json.loads(line_str)
                except json.JSONDecodeError:
                    safe_print(line_str, end="\n")
                    continue

                agent_name = ev.get("agent", "")
                color = AGENT_COLORS.get(agent_name, C_CYAN)
                event_type = ev.get("event", "")

                if event_type == "prompt":
                    print()
                    safe_print(RULE, end="\n")
                    safe_print(
                        f" >> {agent_name} - PROMPT ", color + C_BOLD, end="\n"
                    )
                    safe_print(RULE, end="\n")
                    prompt_msgs = ev.get("messages", [])
                    safe_print(render_messages_as_prompt(prompt_msgs), end="\n")
                    print()

                elif event_type == "chunk":
                    text = ev.get("text", "")
                    if text:
                        if agent_name not in response_header_printed:
                            response_header_printed.add(agent_name)
                            print()
                            safe_print(
                                f" >> {agent_name} - RESPONSE ",
                                color + C_BOLD,
                                end="\n",
                            )
                            safe_print(RULE, end="\n")

                        display_color = color
                        process_text = text

                        if "<thought>" in text:
                            parts = text.split("<thought>", 1)
                            safe_print(parts[0], color)
                            safe_print("<thought>", C_GRAY)
                            display_color = C_GRAY
                            process_text = parts[1]
                        elif "</thought>" in text:
                            parts = text.split("</thought>", 1)
                            safe_print(parts[0], C_GRAY)
                            safe_print("</thought>", C_GRAY)
                            display_color = color
                            process_text = parts[1]

                        safe_print(process_text, display_color)

                        if (
                            agent_name == "ReasonerAgent"
                            or "TextAgent" in agent_name
                        ):
                            thinking_text += text
                        else:
                            answer_text += text

                elif event_type == "done":
                    print()  # newline after streaming
                    safe_print(RULE, end="\n")
                    sys.stdout.flush()

                elif event_type == "final":
                    final_label = ev.get("label", "")
                    answer_text = ev.get("raw_verifier", answer_text)
                    if "soft_labels" in ev:
                        finalizer_out_dist = ev["soft_labels"]

                elif event_type == "error":
                    msg = ev.get("message", "Unknown backend error")
                    safe_print(
                        f"\n[!] AGENT ERROR ({agent_name}): {msg}", C_RED, end="\n"
                    )

        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_local_agent())
        except Exception as e:
            safe_print(f"\nUnexpected error during local agent run: {e}", C_RED, end="\n")
            sys.exit(1)
    else:
        # ── Single-agent path (direct InferenceBridge) ─────────────────────
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from backend.llm_interface import InferenceBridge

        bridge = InferenceBridge(provider=args.provider)
        print()
        print(RULE)
        print(
            f"{C_CYAN}{C_BOLD} STREAMING RESPONSE {C_RESET}{C_DIM}(model: {model_id}){C_RESET}"
        )
        print(RULE)

        # ── Execution with retry ──────────────────────────────────────────
        retries = 3
        for attempt in range(retries):
            try:
                stream = bridge.chat(
                    model=model_id,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=True,
                    include_thinking=include_thinking,
                    soft_label=args.soft_label,
                )
                thinking_text, answer_text = stream_and_capture(stream)
                
                if answer_text.startswith("Error:"):
                    raise RuntimeError(answer_text)
                break
            except Exception as e:
                if attempt == retries - 1:
                    thinking_text = ""
                    answer_text = str(e)
                else:
                    wait_time = 2 ** attempt
                    print(f"\n{C_YELLOW}  [API Error: {e}. Retrying in {wait_time}s... ({attempt+1}/{retries})]{C_RESET}")
                    time.sleep(wait_time)

        # Retry logic for soft-labels
        if args.soft_label:
            from backend.utils import parse_soft_prediction

            pred_soft = parse_soft_prediction(answer_text, labels)
            if pred_soft is None:
                print(
                    f"\n{C_YELLOW}  [Invalid JSON, retrying with temp=0.0...]{C_RESET}"
                )
                stream = bridge.chat(
                    model=model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    stream=True,
                    include_thinking=include_thinking,
                    soft_label=True,
                )
                thinking_text, answer_text = stream_and_capture(stream)
                pred_soft = parse_soft_prediction(answer_text, labels)

            if pred_soft is None:
                print(
                    f"{C_RED}  [Retry failed, defaulting to neutral distribution]{C_RESET}"
                )
                pred_soft = {
                    l.lower(): (1.0 if l.lower() == "neutral" else 0.0) for l in labels
                }

            # Store in a way that Summary can access
            finalizer_out_dist = pred_soft
        final_label = ""

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(RULE)
    print(f"{C_GREEN}{C_BOLD} SUMMARY {C_RESET}")
    print(RULE)
    if thinking_text.strip():
        thinking_preview = thinking_text.strip()[:200].replace("\n", " ")
        print(
            f"  {C_BOLD}Thinking tokens:{C_RESET} {C_MAGENTA}{C_DIM}{thinking_preview}…{C_RESET}"
        )

    if args.soft_label:
        norm = get_argmax_label(finalizer_out_dist)
    else:
        norm = final_label if final_label else normalize_prediction(answer_text, labels)
    match_mark = (
        f"{C_GREEN}[MATCH]{C_RESET}"
        if norm == unify_label(target_utt["emotion"])
        else f"{C_RED}[MISMATCH]{C_RESET}"
    )

    print(f"  {C_BOLD}Answer:         {C_RESET}{C_GREEN}{answer_text.strip()}{C_RESET}")
    print(f"  {C_BOLD}Normalized:     {C_RESET}{C_CYAN}{norm}{C_RESET}  {match_mark}")

    if args.soft_label:
        gt_soft = target_utt.get("soft_labels", {})
        if not gt_soft:
            gt_soft = {
                l.lower(): (1.0 if l.lower() == target_utt["emotion"].lower() else 0.0)
                for l in labels
            }

        jsd = calculate_js_divergence(finalizer_out_dist, gt_soft)
        print(f"  {C_BOLD}JS-Divergence:  {C_RESET}{C_CYAN}{jsd:.4f}{C_RESET}")
        print(
            f"  {C_BOLD}Pred Dist:      {C_RESET}{C_DIM}{finalizer_out_dist}{C_RESET}"
        )

    uni_gt = unify_label(target_utt["emotion"])
    print(
        f"  {C_BOLD}Ground truth:   {C_RESET}{C_YELLOW}{uni_gt}{C_RESET}{C_DIM} (original: {target_utt['emotion']}){C_RESET}"
    )
    print(RULE)
    print()


if __name__ == "__main__":
    main()
