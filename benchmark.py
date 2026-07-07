#!/usr/bin/env python3
"""
benchmark.py — Systematically run a model through a dataset subset and track
               live accuracy + weighted-F1 score after every utterance.

Every run is saved as {"meta": {...}, "results": [...]} so that settings are
always stored alongside data. When --resume is used, the saved meta is compared
against the current settings and mismatches are flagged before any inference runs.

Usage examples:
  python benchmark.py                                    # fully interactive
  python benchmark.py -d meld_dev -n 10                 # first 10 conversations
  python benchmark.py -d meld_dev -n 10 --random        # 10 random conversations
  python benchmark.py -d meld_dev -n 10 -p prompts/erc_cot.txt
  python benchmark.py -d meld_dev -n 10 -m qwen3.5:4b --no-think
  python benchmark.py -d iemocap  -n 5  --provider openrouter -m anthropic/claude-3.5-haiku
  python benchmark.py -d meld_dev -n 10 -o results/my_run.json --resume
"""

import json
import os
import time
import re
import sys
import glob
import random
import hashlib
import argparse
import datetime
import urllib.request
import urllib.error

# ── Optional deps ───────────────────────────────────────────────────────────────
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

try:
    from sklearn.metrics import f1_score, classification_report

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

RULE = C_GRAY + "-" * 70 + C_RESET
RULE2 = C_CYAN + "=" * 70 + C_RESET

# ── Project paths ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
PROMPTS_DIR = os.path.join(ROOT, "prompts")
RESULTS_DIR = os.path.join(ROOT, "results")

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
    "iemocap_test_6class": ["neutral", "frustration", "excitement", "sadness", "anger", "happiness"],
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

# Settings that directly affect which predictions the model makes.
# If any of these differ between a saved run and the current invocation,
# results cannot safely be merged.
RESULT_AFFECTING_KEYS = [
    "dataset_file",  # which data file
    "model",  # which model
    "provider",  # local vs openrouter
    "template_sha256",  # exact template content (not just filename)
    "window",  # context window size
    "temperature",  # sampling temperature
    "include_thinking",  # thinking on/off changes the output distribution
    "agent_mode",  # single vs multi-agent
    "soft_label",  # distribution vs hard label
    # n_conversations / selection / seed are NOT in this list
]


# ── Helpers ──────────────────────────────────────────────────────────────────────


def resolve_dataset(name: str) -> str:
    key = name.lower().replace("-", "_").replace(".json", "")
    if key in DATASET_ALIASES:
        return os.path.join(PROCESSED_DIR, DATASET_ALIASES[key])
    if os.path.isfile(name):
        return os.path.abspath(name)

    # Check subsets directory
    subset_matches = glob.glob(os.path.join(PROCESSED_DIR, "subsets", f"*{key}*"))
    if subset_matches:
        return subset_matches[0]

    matches = glob.glob(os.path.join(PROCESSED_DIR, f"*{key}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Dataset not found: '{name}'.\nAvailable: {list(DATASET_ALIASES)}"
    )


def infer_dataset_name(path: str) -> str:
    base = os.path.basename(path).lower()
    if "iemocap_test_6class" in base:
        return "iemocap_test_6class"
    if "iemocap" in base:
        return "iemocap"
    if "camer" in base:
        return "camer"
    return "meld"


def get_labels(ds: str) -> list:
    ds_lower = ds.lower()
    # Sort keys by length descending to prevent substring collisions (e.g. 'iemocap' matching 'iemocap_test_6class')
    for k in sorted(EMOTION_LABELS.keys(), key=len, reverse=True):
        if k in ds_lower:
            return EMOTION_LABELS[k]
    return EMOTION_LABELS["meld"]


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]  # first 16 hex chars is plenty


def list_datasets() -> list:
    return sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.json")))


def list_prompt_templates() -> list:
    return sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))


def interactive_pick(options: list, prompt: str, display_fn=None) -> int:
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


def ask_int(prompt: str, default: int, min_val: int = 1, max_val: int = 9999) -> int:
    while True:
        try:
            raw = input(f"{C_BOLD}{prompt}{C_RESET} [{default}]: ").strip()
            if not raw:
                return default
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Enter a number between {min_val} and {max_val}.")
        except (ValueError, EOFError):
            return default


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"{C_BOLD}{prompt}{C_RESET} [{hint}]: ").strip().lower()
            if not raw:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
        except EOFError:
            return default


def collect_full_response(stream) -> tuple[str, str]:
    """Consume a stream generator; return (thinking_text, answer_text)."""
    thinking_buf, answer_buf = [], []
    in_thinking = False
    buf = ""
    for chunk in stream:
        buf += chunk
        while True:
            if not in_thinking:
                p = buf.find("<thought>")
                if p == -1:
                    safe, buf = (buf[:-8], buf[-8:]) if len(buf) > 8 else (buf, "")
                    answer_buf.append(safe)
                    break
                answer_buf.append(buf[:p])
                buf = buf[p + len("<thought>") :]
                in_thinking = True
            else:
                p = buf.find("</thought>")
                if p == -1:
                    safe, buf = (buf[:-10], buf[-10:]) if len(buf) > 10 else (buf, "")
                    thinking_buf.append(safe)
                    break
                thinking_buf.append(buf[:p])
                buf = buf[p + len("</thought>") :]
                in_thinking = False
    if buf:
        (thinking_buf if in_thinking else answer_buf).append(buf)
    return "".join(thinking_buf), "".join(answer_buf)


# ── Metadata & resume logic ──────────────────────────────────────────────────────


def build_run_meta(
    dataset_path: str,
    prompt_path: str,
    model: str,
    args,
    include_thinking: bool,
    selection_desc: str,
    n_conv: int,
) -> dict:
    """Build the metadata dict that is persisted alongside results."""
    return {
        # ── Result-affecting settings ──────────────────────────────────────
        "dataset_file": os.path.basename(dataset_path),
        "model": model,
        "provider": args.provider,
        "template_path": os.path.relpath(prompt_path),
        "template_sha256": sha256_of_file(prompt_path),
        "window": args.window,
        "temperature": args.temperature,
        "include_thinking": include_thinking,
        "agent_mode": "multi" if args.agent else "single",
        "soft_label": args.soft_label,
        # ── Informational ─────────────────────────────────────────────────
        "max_tokens": args.max_tokens,
        "n_conversations": n_conv,
        "selection": selection_desc,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def check_meta_compatibility(saved: dict, current: dict) -> list[str]:
    """
    Compare saved run meta against current settings.
    Returns a list of human-readable mismatch descriptions (empty = all OK).
    """
    mismatches = []
    for key in RESULT_AFFECTING_KEYS:
        sv = saved.get(key)
        cv = current.get(key)
        if sv != cv:
            label = key.replace("_", " ")
            mismatches.append(
                f"  {C_BOLD}{label}{C_RESET}: saved={C_YELLOW}{sv}{C_RESET}  current={C_RED}{cv}{C_RESET}"
            )
    return mismatches


def load_run_file(path: str) -> tuple[dict, list]:
    """
    Load a results file. Handles both the new {"meta": ..., "results": [...]}
    format and the old flat-list format (for backward compatibility).
    Returns (meta_dict, results_list).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # Old flat-list format — no metadata available
        return {}, data
    elif isinstance(data, dict) and "results" in data:
        return data.get("meta", {}), data["results"]
    else:
        return {}, []


def save_run_file(path: str, meta: dict, results: list):
    """Save results with metadata to the output file."""
    meta["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)


# ── Live metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(
    truths: list, preds: list, labels: list, results: list = None
) -> dict:
    """Calculates common classification metrics (Acc, F1) + optional JSD."""
    results = results or []
    n = len(truths)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "f1_macro": 0.0,
            "correct": 0,
            "avg_jsd": 0.0,
        }

    correct = sum(t == p for t, p in zip(truths, preds))
    acc = correct / n
    if HAS_SKLEARN and n >= 2:
        try:
            f1_w = f1_score(
                truths, preds, labels=labels, average="weighted", zero_division=0
            )
            f1_m = f1_score(
                truths, preds, labels=labels, average="macro", zero_division=0
            )
        except Exception:
            f1_w = f1_m = 0.0
    else:
        f1_w = f1_m = 0.0

    avg_jsd = (
        sum(r.get("js_divergence", 0.0) for r in results if "js_divergence" in r) / n
        if n
        else 0.0
    )

    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "avg_jsd": avg_jsd,
    }


def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return C_GREEN + "#" * filled + C_DIM + "-" * (width - filled) + C_RESET


def print_live_metrics(metrics: dict, last_result: dict, conv_id: str, utt_idx: int):
    acc = metrics["accuracy"]
    f1w = metrics["f1_weighted"]
    f1m = metrics["f1_macro"]
    jsd = last_result.get("js_divergence", 0.0)
    n = metrics["n"]
    ok = metrics["correct"]

    pred = last_result["prediction_raw"]
    norm = last_result["prediction"]
    gt = last_result["ground_truth"]
    mark = f"{C_GREEN}[MATCH]{C_RESET}" if norm == gt else f"{C_RED}[MISMATCH]{C_RESET}"
    pred_color = C_GREEN if norm == gt else C_RED

    # Add JSD to display if it's a soft-label run
    jsd_display = (
        f"  JSD={C_CYAN}{jsd:.3f}{C_RESET}" if "js_divergence" in last_result else ""
    )

    print(
        f"  {C_DIM}{conv_id}[{utt_idx}]{C_RESET}  "
        f"{mark}  "
        f"pred={pred_color}{norm:<12}{C_RESET}  gt={C_YELLOW}{gt:<12}{C_RESET}  "
        f"acc={C_CYAN}{acc:.1%}{C_RESET} {bar(acc, 12)}  "
        f"F1w={C_MAGENTA}{f1w:.3f}{C_RESET}{jsd_display}  "
        f"{C_DIM}({ok}/{n}){C_RESET}"
    )
    if norm != pred.lower().strip():
        print(f'           {C_DIM}>> raw: "{pred[:60]}"{C_RESET}')


def print_final_report(
    metrics: dict, labels: list, truths: list, preds: list, meta: dict, output_path: str
):
    print()
    print(RULE2)
    print(f"{C_BOLD}{C_CYAN}  BENCHMARK COMPLETE{C_RESET}")
    print(RULE2)
    print(f"  {C_BOLD}Dataset:       {C_RESET}{meta.get('dataset_file', '?')}")
    print(
        f"  {C_BOLD}Model:         {C_RESET}{meta.get('model', '?')}  ({meta.get('provider', '?')})"
    )
    print(
        f"  {C_BOLD}Template:      {C_RESET}{meta.get('template_path', '?')}  {C_DIM}(sha256: {meta.get('template_sha256', '?')}){C_RESET}"
    )
    print(f"  {C_BOLD}Window:        {C_RESET}{meta.get('window', '?')}  prior turns")
    print(f"  {C_BOLD}Temperature:   {C_RESET}{meta.get('temperature', '?')}")
    print(
        f"  {C_BOLD}Thinking:      {C_RESET}{'on' if meta.get('include_thinking') else 'off'}"
    )
    print(
        f"  {C_BOLD}Conversations: {C_RESET}{meta.get('n_conversations', '?')}  ({meta.get('selection', '?')})"
    )
    print(
        f"  {C_BOLD}Utterances:    {C_RESET}{metrics['n']}  ({metrics['correct']} correct)"
    )
    print()
    print(
        f"  {C_BOLD}Accuracy:      {C_RESET}{C_CYAN}{metrics['accuracy']:.4f}  ({metrics['accuracy']:.1%}){C_RESET}  {bar(metrics['accuracy'], 24)}"
    )
    print(
        f"  {C_BOLD}F1 Weighted:   {C_RESET}{C_MAGENTA}{metrics['f1_weighted']:.4f}{C_RESET}"
    )
    print(
        f"  {C_BOLD}F1 Macro:      {C_RESET}{C_BLUE}{metrics['f1_macro']:.4f}{C_RESET}"
    )
    if "avg_jsd" in metrics and metrics["avg_jsd"] > 0:
        print(
            f"  {C_BOLD}Avg JS-Diverg: {C_RESET}{C_CYAN}{metrics['avg_jsd']:.4f}{C_RESET}"
        )

    if HAS_SKLEARN and truths:
        print()
        print(RULE)
        print(f"{C_BOLD}  PER-CLASS BREAKDOWN{C_RESET}")
        print(RULE)
        try:
            report = classification_report(
                truths, preds, labels=labels, zero_division=0, output_dict=False
            )
            for line in report.splitlines():
                print(f"  {line}")
        except Exception:
            pass

    print()
    print(RULE)
    print(f"  {C_BOLD}Results saved to:{C_RESET} {C_DIM}{output_path}{C_RESET}")
    print(RULE)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="benchmark.py — Run a model on a dataset subset with live accuracy/F1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d", "--dataset", metavar="NAME", help="Dataset alias or file path"
    )
    parser.add_argument(
        "-n",
        "--n-conversations",
        metavar="N",
        type=int,
        help="Number of conversations to run (default: interactive)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick N conversations randomly (default: first N)",
    )
    parser.add_argument(
        "--seed",
        metavar="S",
        type=int,
        default=42,
        help="Random seed when using --random (default: 42)",
    )
    parser.add_argument(
        "-p", "--prompt", metavar="FILE", help="Prompt template .txt file"
    )
    parser.add_argument(
        "-m", "--model", metavar="MODEL", help="Model ID (e.g. qwen3.5:4b)"
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
        help="Context window size (default: 5)",
    )
    parser.add_argument(
        "--no-think", action="store_true", help="Disable thinking tokens"
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
        help="Max tokens allowed for the thinking block (Ollama only, or reasoning_max_tokens for OpenRouter)",
    )
    parser.add_argument(
        "--temperature",
        metavar="F",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a previous run — skips already-processed utterances "
        "after verifying all result-affecting settings match",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output JSON path (auto-named by default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --resume, ignore setting mismatches and merge anyway",
    )
    parser.add_argument(
        "--agent", action="store_true", help="Use multi-agent mode via the backend"
    )
    parser.add_argument(
        "--workflow",
        metavar="WF",
        default="reasoner_verifier",
        help="Workflow name (e.g. modality_tva, reasoner_verifier)",
    )
    parser.add_argument(
        "--backend-url",
        metavar="URL",
        default="http://localhost:8283",
        help="Backend server URL (default: http://localhost:8283)",
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="List available datasets and exit"
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt templates and exit",
    )
    parser.add_argument(
        "--vision", action="store_true", help="Enable vision (I-frame extraction)"
    )
    parser.add_argument(
        "--vision-frames",
        metavar="N",
        type=int,
        default=3,
        help="Max number of vision frames to extract (default: 3)",
    )
    parser.add_argument("--audio", action="store_true", help="Enable audio extraction")
    parser.add_argument(
        "--soft-label",
        action="store_true",
        help="Request probability distribution and calculate JS-divergence",
    )

    # Agentic Simulation Arguments
    parser.add_argument("--text-results", type=str, help="Path to pre-computed text modality results JSON")
    parser.add_argument("--vision-results", type=str, help="Path to pre-computed vision modality results JSON")
    parser.add_argument("--audio-results", type=str, help="Path to pre-computed audio modality results JSON")

    args = parser.parse_args()

    # ── Quick-list helpers ────────────────────────────────────────────────────
    if args.list_datasets:
        print(f"\n{C_BOLD}Available datasets:{C_RESET}")
        for alias, fname in DATASET_ALIASES.items():
            exists = (
                "[OK]" if os.path.isfile(os.path.join(PROCESSED_DIR, fname)) else "[X]"
            )
            print(f"  {C_CYAN}{exists}{C_RESET}  {C_BOLD}{alias:<18}{C_RESET}  {fname}")
        return

    if args.list_prompts:
        templates = list_prompt_templates()
        print(f"\n{C_BOLD}Available prompt templates:{C_RESET}")
        for t in templates:
            print(f"  {C_CYAN}{os.path.relpath(t)}{C_RESET}")
        return

    # ── Resolve dataset ───────────────────────────────────────────────────────
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
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_dialogues = json.load(f)
    dataset_name = infer_dataset_name(dataset_path)  # short: 'meld', 'iemocap', 'camer'
    dataset_alias = (
        args.dataset
        if args.dataset
        else os.path.splitext(os.path.basename(dataset_path))[0].replace(
            "_processed", ""
        )
    )
    print(f"{C_GREEN}[OK] Loaded {len(all_dialogues)} dialogues{C_RESET}")

    # ── Resolve n_conversations ───────────────────────────────────────────────
    n_conv = args.n_conversations
    if n_conv is None:
        n_conv = ask_int(
            f"How many conversations to benchmark? (max {len(all_dialogues)})",
            default=min(10, len(all_dialogues)),
            min_val=1,
            max_val=len(all_dialogues),
        )
    n_conv = min(n_conv, len(all_dialogues))

    if args.random:
        random.seed(args.seed)
        dialogues = random.sample(all_dialogues, n_conv)
        selection_desc = f"{n_conv} random (seed={args.seed})"
    else:
        dialogues = all_dialogues[:n_conv]
        selection_desc = f"first {n_conv}"

    total_utterances = sum(len(d["utterances"]) for d in dialogues)

    # Track the original sequence order to keep output results sorted during resumes
    key_to_index = {}
    global_idx = 0
    for diag in dialogues:
        utterances = diag["utterances"]
        if "target_index" in diag:
            indices = [diag["target_index"]]
        else:
            indices = range(len(utterances))
        for utt_idx in indices:
            utt = utterances[utt_idx]
            u_id = utt.get("utterance_id", utt_idx)
            key_to_index[(diag["dialogue_id"], u_id)] = global_idx
            global_idx += 1

    # ── Resolve prompt template ───────────────────────────────────────────────
    if args.prompt:
        prompt_path = args.prompt
    else:
        templates = list_prompt_templates()
        default_tpl = os.path.join(PROMPTS_DIR, "erc_default.txt")
        if len(templates) == 0:
            prompt_path = default_tpl
        elif len(templates) == 1:
            prompt_path = templates[0]
            print(f"{C_DIM}Using template: {os.path.relpath(prompt_path)}{C_RESET}")
        else:
            print(f"\n{C_BOLD}Select prompt template:{C_RESET}")
            idx = interactive_pick(
                templates, "Template", display_fn=lambda p: os.path.relpath(p)
            )
            prompt_path = templates[idx]

    if not os.path.isfile(prompt_path):
        print(f"{C_RED}Prompt template not found: {prompt_path}{C_RESET}")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    # ── Resolve model & other settings ───────────────────────────────────────
    model = args.model or DEFAULT_MODELS[args.provider]
    include_thinking = not args.no_think
    labels = get_labels(dataset_name)

    # ── Resolve output path ───────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tpl_name = os.path.splitext(os.path.basename(prompt_path))[0]
    if args.output:
        output_path = args.output
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        safe_model = model.replace(":", "_").replace("/", "-")
        output_path = os.path.join(
            RESULTS_DIR,
            f"bench_{dataset_name}_{safe_model}_{tpl_name}_n{n_conv}_{ts}.json",
        )

    # ── Build current run metadata ────────────────────────────────────────────
    current_meta = build_run_meta(
        dataset_path, prompt_path, model, args, include_thinking, selection_desc, n_conv
    )

    # ── Resume: load + validate existing results ──────────────────────────────
    results: list = []
    processed_ids: set = set()

    if args.resume and os.path.isfile(output_path):
        saved_meta, results = load_run_file(output_path)

        if not saved_meta:
            print(
                f"{C_YELLOW}[!] Existing file has no metadata (old format). "
                f"Cannot verify settings - treating as compatible.{C_RESET}"
            )
        else:
            mismatches = check_meta_compatibility(saved_meta, current_meta)
            if mismatches:
                print()
                print(
                    f"{C_RED}{C_BOLD}  [!] SETTINGS MISMATCH - results may not be comparable{C_RESET}"
                )
                print(RULE)
                for m in mismatches:
                    print(m)
                print(RULE)
                print()
                if args.force:
                    print(f"{C_YELLOW}  --force specified - merging anyway.{C_RESET}")
                else:
                    ok = ask_yes_no(
                        "Settings differ from saved run. Merge anyway? (results will be mixed)",
                        default=False,
                    )
                    if not ok:
                        print(f"{C_RED}Aborted.{C_RESET}")
                        sys.exit(0)
            else:
                print(
                    f"{C_GREEN}[OK] Settings match saved run - safe to resume.{C_RESET}"
                )

        # Filter out failed/error entries so they are not skipped and can be rerun
        original_count = len(results)
        results = [
            r for r in results
            if not (r.get("prediction_raw", "").startswith("Error") or "402" in r.get("prediction_raw", ""))
        ]
        rerun_count = original_count - len(results)
        if rerun_count > 0:
            print(f"{C_YELLOW}[!] Identified {rerun_count} failed/error entries in the previous run. These will be rerun.{C_RESET}")

        processed_ids = {(r["dialogue_id"], r["utterance_id"]) for r in results}
        print(f"{C_DIM}  {len(results)} successful utterances already recorded.{C_RESET}")

    elif args.resume and not os.path.isfile(output_path):
        print(
            f"{C_YELLOW}⚠  --resume specified but output file not found — starting fresh.{C_RESET}"
        )

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
    # ---------------------------------------

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(RULE2)
    print(
        f"{C_BOLD}{C_CYAN}  BENCHMARK{C_RESET}  {C_DIM}|  {dataset_name.upper()}  ·  {model}  ·  {args.provider}{C_RESET}"
    )
    print(RULE2)
    print(
        f"  {C_BOLD}Conversations: {C_RESET}{selection_desc}  ->  {total_utterances} utterances"
    )
    print(
        f"  {C_BOLD}Template:      {C_RESET}{C_DIM}{os.path.relpath(prompt_path)}{C_RESET}  "
        f"{C_GRAY}sha256:{current_meta['template_sha256']}{C_RESET}"
    )
    print(f"  {C_BOLD}Window:        {C_RESET}{args.window} prior turns")
    print(
        f"  {C_BOLD}Thinking:      {C_RESET}{'off' if not include_thinking else 'on'}"
    )
    print(f"  {C_BOLD}Temperature:   {C_RESET}{args.temperature}")
    print(f"  {C_BOLD}Max tokens:    {C_RESET}{args.max_tokens}")
    if is_simulation:
        sim_mods = []
        if args.text_results: sim_mods.append("Text")
        if args.vision_results: sim_mods.append("Vision")
        if args.audio_results: sim_mods.append("Audio")
        print(f"  {C_BOLD}Simulation:    {C_RESET}{C_CYAN}Enabled ({', '.join(sim_mods)}){C_RESET}")

    # These flags need to be derived for the header display
    with open(prompt_path, "r", encoding="utf-8") as f:
        first_lines = "".join([f.readline() for _ in range(10)])

    include_v_any = (
        args.vision
        or "# MEDIA: video" in first_lines
        or "# MEDIA: vision" in first_lines
        or "# MEDIA: image" in first_lines
    )

    vision_status = (
        f"{C_GREEN}active{C_RESET}" if include_v_any else f"{C_DIM}inactive{C_RESET}"
    )
    print(f"  {C_BOLD}Vision:        {C_RESET}{vision_status}")
    print(
        f"  {C_BOLD}Soft-Label:    {C_RESET}{C_GREEN if args.soft_label else C_DIM}{'active' if args.soft_label else 'inactive'}{C_RESET}"
    )
    print(f"  {C_BOLD}Output:        {C_RESET}{C_DIM}{output_path}{C_RESET}")
    print()
    print(RULE)
    print(
        f"  {'CONV[UTT]':<16} {'':2} {'PREDICTED':<14} {'GROUND TRUTH':<14} "
        f"{'ACC':>6}  {'BAR':>14}  {'F1w':>7}  {'F1m':>7}  {'n':>6}"
    )
    print(RULE)

    # ── Init bridge / backend client ──────────────────────────────────────────
    sys.path.insert(0, ROOT)
    from backend.prompt_templates import get_registry
    from backend.utils import (
        normalize_prediction,
        parse_soft_prediction,
        calculate_js_divergence,
        get_argmax_label,
        unify_label,
    )

    # Resolve @presets
    from backend.agents_logic import MODEL_PRESETS

    model = args.model or "qwen3.5:4b"  # ensure model is defined
    if model in MODEL_PRESETS:
        model = MODEL_PRESETS[model]

    registry = get_registry()
    BACKEND_URL = args.backend_url.rstrip("/")

    def call_backend_single(
        dialogue, utt_idx, utterances, temperature=None
    ) -> tuple[str, str]:
        """
        Call the /inference endpoint (single-agent, streaming).
        Returns (thinking_text, answer_text).
        """
        from backend.llm_interface import InferenceBridge

        # Build messages locally and call the bridge directly
        bridge = InferenceBridge(provider=args.provider)
        context = {
            "dataset_name": dataset_name,
            "utterances": utterances,
            "target_index": utt_idx,
            "window_size": args.window,
        }

        # Inject simulated modality results if available
        if is_simulation:
            d_id = dialogue.get("dialogue_id")
            u_id = utterances[utt_idx].get("utterance_id")
            try:
                u_id_parsed = int(u_id)
            except (ValueError, TypeError):
                u_id_parsed = str(u_id)
            lookup_key = (str(d_id), u_id_parsed)
            if lookup_key in simulated_results:
                context.update(simulated_results[lookup_key])

        messages, metadata = registry.render(prompt_path, **context)

        # ── Multimodal Handling ──────────────────────────────────────────────
        from backend.media_utils import (
            prepare_multimodal_content,
            get_absolute_media_path,
        )

        media_reqs = metadata.get("media", [])
        wants_video = "video" in media_reqs
        wants_image = "image" in media_reqs or "vision" in media_reqs
        wants_audio = "audio" in media_reqs
        include_v = args.vision or wants_video or wants_image
        include_a = args.audio or wants_audio
        visual_mode = "video" if wants_video else "image"

        if include_v or include_a:
            # Check both video_path and audio_path
            m_path = utterances[utt_idx].get("video_path") or utterances[utt_idx].get(
                "audio_path"
            )
            abs_m_path = get_absolute_media_path(m_path)

            if not abs_m_path and m_path:
                # Only print once per benchmark run to avoid spamming
                if not getattr(args, "_warned_media", False):
                    print(
                        f"\n{C_YELLOW}  [!] Warning: Could not resolve media path for some utterances (e.g. {m_path}){C_RESET}"
                    )
                    print(
                        f"{C_DIM}      Continuing with text-only for those cases.{C_RESET}\n"
                    )
                    setattr(args, "_warned_media", True)

            last_msg = messages[-1]
            if last_msg["role"] == "user":
                u_id = utterances[utt_idx].get("utterance_id")
                last_msg["content"] = prepare_multimodal_content(
                    last_msg["content"],
                    m_path,
                    include_v,
                    include_audio=include_a,
                    max_vision_frames=args.vision_frames,
                    visual_mode=visual_mode,
                    utterance_id=u_id,
                )

        actual_temp = temperature if temperature is not None else args.temperature

        stream = bridge.chat(
            model=model,
            messages=messages,
            temperature=actual_temp,
            max_tokens=args.max_tokens,
            stream=True,
            include_thinking=include_thinking,
            soft_label=args.soft_label,
            reasoning_max_tokens=args.think_budget if args.think_budget is not None else 10000,
        )
        return collect_full_response(stream)

    def call_backend_multi(dialogue, utt_idx, utterances) -> tuple[str, str, dict]:
        """
        Runs the multi-agent pipeline internally in Python.
        Returns (raw_answer_thoughts, final_label, soft_labels).
        """
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
            "max_tokens": args.max_tokens if args.max_tokens is not None else 40000,
            "reasoning_max_tokens": args.think_budget if args.think_budget is not None else 10000,
        }

        final_label = ""
        raw_verifier = ""
        soft_labels = {}

        async def run_local():
            nonlocal final_label, raw_verifier, soft_labels
            stream_gen = run_multiagent_stream(
                messages=[],
                valid_emotions=", ".join(labels),
                model=model,
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
                except:
                    continue

                if ev.get("event") == "final":
                    final_label = ev.get("label", "")
                    raw_verifier = ev.get("raw_verifier", "")
                    if "soft_labels" in ev:
                        soft_labels = ev["soft_labels"]

        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_local())
        except Exception as e:
            print(f"{C_RED}Local agent run error: {e}{C_RESET}")
            return "", "error", {}

        return raw_verifier, final_label, soft_labels

    if args.agent:
        bridge = None
    else:
        bridge = None  # will use call_backend_single which creates bridge internally

    # Seed truths/preds from previously recorded results
    truths: list = [r["ground_truth"] for r in results]
    preds: list = [r["prediction"] for r in results]

    # ── Main inference loop ───────────────────────────────────────────────────
    skipped = 0
    try:
        for diag in dialogues:
            utterances = diag["utterances"]

            # If it's a subset sample, target_index tells us which one to benchmark.
            if "target_index" in diag:
                indices = [diag["target_index"]]
            else:
                indices = range(len(utterances))

            for utt_idx in indices:
                utt = utterances[utt_idx]
                key = (diag["dialogue_id"], utt.get("utterance_id", utt_idx))

                if key in processed_ids:
                    skipped += 1
                    continue

                if args.agent:
                    # ── Multi-agent mode ──────────────────────────────────────
                    raw_answer, norm, pred_soft = call_backend_multi(diag, utt_idx, utterances)
                    thinking_text = ""
                else:
                    # ── Single-agent mode ─────────────────────────────────────
                    retries = 3
                    for attempt in range(retries):
                        try:
                            thinking_text, raw_answer = call_backend_single(
                                diag, utt_idx, utterances
                            )
                            if raw_answer.startswith("Error:"):
                                raise RuntimeError(raw_answer)
                            break
                        except Exception as e:
                            if attempt == retries - 1:
                                thinking_text = ""
                                raw_answer = str(e)
                            else:
                                wait_time = 2 ** attempt
                                print(f"{C_YELLOW}  [API Error: {e}. Retrying in {wait_time}s... ({attempt+1}/{retries})]{C_RESET}")
                                time.sleep(wait_time)

                    if args.soft_label:
                        pred_soft = parse_soft_prediction(raw_answer, labels)
                        if pred_soft is None:
                            # Retry with temperature 0.0
                            print(
                                f"{C_YELLOW}  [Invalid JSON, retrying with temp=0.0...]{C_RESET}"
                            )
                            thinking_text, raw_answer = call_backend_single(
                                diag, utt_idx, utterances, temperature=0.0
                            )
                            pred_soft = parse_soft_prediction(raw_answer, labels)

                        if pred_soft is None:
                            # Fallback: neutral=1.0
                            print(
                                f"{C_RED}  [Retry failed, defaulting to neutral distribution]{C_RESET}"
                            )
                            pred_soft = {
                                l.lower(): (1.0 if l.lower() == "neutral" else 0.0)
                                for l in labels
                            }

                    # Determine "hard" label for standard metrics (Acc, F1)
                    if args.soft_label:
                        norm = get_argmax_label(pred_soft)
                    else:
                        norm = normalize_prediction(raw_answer, labels)

                gt = unify_label(utt["emotion"])

                jsd = None
                if args.soft_label:
                    gt_soft = utt.get("soft_labels", {})
                    if not gt_soft:
                        # Convert hard GT to soft if missing
                        gt_soft = {
                            l.lower(): (1.0 if l.lower() == gt.lower() else 0.0)
                            for l in labels
                        }
                    jsd = calculate_js_divergence(pred_soft, gt_soft)

                result_entry = {
                    "dialogue_id": diag["dialogue_id"],
                    "utterance_id": utt.get("utterance_id", utt_idx),
                    "speaker": utt["speaker"],
                    "text": utt["text"],
                    "ground_truth": gt,
                    "prediction": norm,
                    "prediction_raw": raw_answer.strip(),
                    "mode": "multi" if args.agent else "single",
                }
                if args.soft_label:
                    result_entry["prediction_soft"] = pred_soft
                    result_entry["js_divergence"] = jsd
                    result_entry["ground_truth_soft"] = utt.get("soft_labels", {})

                if "original_dialogue_id" in diag:
                    result_entry["original_dialogue_id"] = diag["original_dialogue_id"]

                results.append(result_entry)
                # Sort results by original dataset order to prevent sequence scrambling
                results.sort(key=lambda r: key_to_index.get((r["dialogue_id"], r["utterance_id"]), 999999))
                
                # Rebuild truths and preds to ensure they match the sorted results order perfectly
                truths = [r["ground_truth"] for r in results]
                preds = [r["prediction"] for r in results]

                save_run_file(output_path, current_meta, results)

                metrics = compute_metrics(truths, preds, labels, results)
                print_live_metrics(metrics, result_entry, diag["dialogue_id"], utt_idx)

    except KeyboardInterrupt:
        print(f"\n\n{C_YELLOW}  Interrupted - partial results saved.{C_RESET}")

    # ── Final report ──────────────────────────────────────────────────────────
    if truths:
        final_metrics = compute_metrics(truths, preds, labels)
        print_final_report(
            final_metrics, labels, truths, preds, current_meta, output_path
        )
    else:
        if skipped:
            print(f"\n{C_YELLOW}All {skipped} utterances already processed.{C_RESET}\n")


if __name__ == "__main__":
    main()
