#!/usr/bin/env python3
"""
calculate_metrics.py — Recalculates Accuracy, F1 Scores, JS-Divergence, and
                       Classification reports for any benchmark results JSON file.
                       Supports specifying a limit (e.g., first 100 rows).
"""

import os
import sys
import json
import argparse
import datetime
import glob

# Try loading sklearn metrics
try:
    from sklearn.metrics import f1_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try loading colorama for pretty terminals
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
RULE2 = C_CYAN + "=" * 70 + C_RESET

# Standard labels dictionary
EMOTION_LABELS = {
    "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
    "camer": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
    "iemocap_test_6class": ["neutral", "frustration", "excitement", "sadness", "anger", "happiness"],
    "iemocap": ["neutral", "happiness", "sadness", "anger", "fear", "frustration", "excitement"]
}

def bar(value: float, width: int = 24) -> str:
    filled = max(0, min(width, round(value * width)))
    return C_GREEN + "#" * filled + C_DIM + "-" * (width - filled) + C_RESET

def compute_metrics(truths, preds, labels, results=None):
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

    correct = sum(t.lower() == p.lower() for t, p in zip(truths, preds))
    acc = correct / n

    if HAS_SKLEARN and n >= 1:
        try:
            f1_w = f1_score(truths, preds, labels=labels, average="weighted", zero_division=0)
            f1_m = f1_score(truths, preds, labels=labels, average="macro", zero_division=0)
        except Exception:
            try:
                # Fallback in case of custom labels not in standard set
                f1_w = f1_score(truths, preds, average="weighted", zero_division=0)
                f1_m = f1_score(truths, preds, average="macro", zero_division=0)
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

def print_metrics_table(title, metrics, compare_metrics=None):
    """Prints a comparison or single summary of computed metrics."""
    print(RULE)
    print(f"  {C_BOLD}{title}{C_RESET}")
    print(RULE)
    
    n_str = f"{metrics['n']}"
    corr_str = f"{metrics['correct']}"
    acc_str = f"{metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})"
    f1w_str = f"{metrics['f1_weighted']:.4f}"
    f1m_str = f"{metrics['f1_macro']:.4f}"
    jsd_str = f"{metrics['avg_jsd']:.4f}" if metrics["avg_jsd"] > 0 else "N/A"

    if compare_metrics:
        # Printable side-by-side table
        print(f"  {C_BOLD}{'Metric':<20} | {'Subset':<18} | {'Overall Total':<18}{C_RESET}")
        print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*18}")
        print(f"  {'Sample Size (N)':<20} | {n_str:<18} | {compare_metrics['n']:<18}")
        print(f"  {'Correct Class.':<20} | {corr_str:<18} | {compare_metrics['correct']:<18}")
        print(f"  {'Accuracy':<20} | {C_CYAN}{acc_str:<18}{C_RESET} | {C_CYAN}{compare_metrics['accuracy']:.4f} ({compare_metrics['accuracy']:.1%}){C_RESET}")
        print(f"  {'F1 Weighted':<20} | {C_MAGENTA}{f1w_str:<18}{C_RESET} | {C_MAGENTA}{compare_metrics['f1_weighted']:.4f}{C_RESET}")
        print(f"  {'F1 Macro':<20} | {C_BLUE}{f1m_str:<18}{C_RESET} | {C_BLUE}{compare_metrics['f1_macro']:.4f}{C_RESET}")
        if metrics["avg_jsd"] > 0 or compare_metrics["avg_jsd"] > 0:
            comp_jsd = f"{compare_metrics['avg_jsd']:.4f}" if compare_metrics["avg_jsd"] > 0 else "N/A"
            print(f"  {'Avg JS-Divergence':<20} | {C_CYAN}{jsd_str:<18}{C_RESET} | {C_CYAN}{comp_jsd:<18}{C_RESET}")
    else:
        # Standard summary
        print(f"  {C_BOLD}Sample Size (N): {C_RESET}{n_str}")
        print(f"  {C_BOLD}Correct Class.:  {C_RESET}{corr_str} / {n_str}")
        print(f"  {C_BOLD}Accuracy:        {C_RESET}{C_CYAN}{acc_str}{C_RESET}  {bar(metrics['accuracy'], 20)}")
        print(f"  {C_BOLD}F1 Weighted:     {C_RESET}{C_MAGENTA}{f1w_str}{C_RESET}")
        print(f"  {C_BOLD}F1 Macro:        {C_RESET}{C_BLUE}{f1m_str}{C_RESET}")
        if metrics["avg_jsd"] > 0:
            print(f"  {C_BOLD}Avg JS-Diverg:   {C_RESET}{C_CYAN}{jsd_str}{C_RESET}")

def load_run_file(path: str) -> tuple[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {}, data
    elif isinstance(data, dict) and "results" in data:
        return data.get("meta", {}), data["results"]
    else:
        return {}, []

def main():
    parser = argparse.ArgumentParser(
        description="calculate_metrics.py — Recalculates metrics for a benchmark run, with limit support."
    )
    parser.add_argument("file", nargs="?", help="Path to the JSON result file in results/")
    parser.add_argument("-l", "--limit", type=int, default=0, help="Limit to first N results (e.g. 100)")
    parser.add_argument("--no-sklearn", action="store_true", help="Force fallback, skip sklearn classification report")
    
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # If no file is specified, display interactive filterable file picker
    target_file = args.file
    if not target_file:
        files = glob.glob(os.path.join(results_dir, "*.json"))
        if not files:
            print(f"{C_RED}Error: No results JSON files found in {results_dir}{C_RESET}")
            sys.exit(1)
        
        # Sort files by modified time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        current_query = ""
        while True:
            # Re-filter files based on current query if any
            if current_query:
                filtered_files = [f for f in files if current_query.lower() in os.path.basename(f).lower()]
            else:
                filtered_files = list(files)

            if not filtered_files:
                print(f"\n  {C_RED}No files matching '{current_query}'. Filter cleared.{C_RESET}")
                current_query = ""
                continue

            print()
            print(RULE2)
            if current_query:
                print(f"{C_BOLD}{C_CYAN}  SELECT RESULTS FILE (Filtered by: '{current_query}')  [{len(filtered_files)} files]{C_RESET}")
            else:
                print(f"{C_BOLD}{C_CYAN}  SELECT RESULTS FILE TO ANALYZE (Newest first)  [{len(filtered_files)} files]{C_RESET}")
            print(RULE2)

            limit_show = min(30, len(filtered_files))
            for i in range(limit_show):
                filepath = filtered_files[i]
                filename = os.path.basename(filepath)
                size_kb = os.path.getsize(filepath) / 1024
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  [{C_GREEN}{i+1}{C_RESET}] {C_BOLD}{filename:<45}{C_RESET} {C_DIM}({size_kb:5.1f} KB - {mtime}){C_RESET}")

            if len(filtered_files) > limit_show:
                print(f"  {C_GRAY}... and {len(filtered_files) - limit_show} more files. Type a search query to filter them down.{C_RESET}")

            print(RULE)
            try:
                prompt_msg = f"  Select [1-{limit_show}], type query to search, or press Enter for [1]: "
                choice = input(prompt_msg).strip()
                if not choice:
                    target_file = filtered_files[0]
                    break
                
                if choice.isdigit():
                    val = int(choice)
                    if 1 <= val <= limit_show:
                        target_file = filtered_files[val - 1]
                        break
                    else:
                        print(f"\n  {C_RED}Error: Number out of range. Choose between 1 and {limit_show}.{C_RESET}")
                else:
                    current_query = choice
            except (ValueError, IndexError, KeyboardInterrupt, EOFError):
                print(f"\n{C_YELLOW}Selection canceled. Exiting.{C_RESET}")
                sys.exit(0)

    # Resolve target file path
    if not os.path.isfile(target_file):
        # Try joining with results/ dir
        resolved = os.path.join(results_dir, target_file)
        if not os.path.isfile(resolved):
            print(f"{C_RED}Error: File not found: '{target_file}'{C_RESET}")
            sys.exit(1)
        target_file = resolved

    # Load results
    try:
        meta, results = load_run_file(target_file)
    except Exception as e:
        print(f"{C_RED}Error loading JSON file: {e}{C_RESET}")
        sys.exit(1)

    if not results:
        print(f"{C_RED}Error: Loaded result file contains no results data.{C_RESET}")
        sys.exit(1)

    total_count = len(results)
    
    # Slice the results if limit is set
    slice_limit = args.limit
    if slice_limit > 0:
        results_slice = results[:slice_limit]
        slice_str = f"First {slice_limit} Rows"
    else:
        results_slice = results
        slice_str = "All Rows"

    # Infer dataset name for labels
    dataset = "meld"
    dataset_file = meta.get("dataset_file", "").lower()
    target_lower = target_file.lower()
    
    # Sort keys by length descending to prevent substring collisions (e.g., 'iemocap' matching 'iemocap_test_6class')
    for key in sorted(EMOTION_LABELS.keys(), key=len, reverse=True):
        if key in dataset_file or key in target_lower:
            dataset = key
            break

    labels = EMOTION_LABELS.get(dataset, EMOTION_LABELS["meld"])

    # Extract truths and predictions
    truths_total = [r["ground_truth"] for r in results]
    preds_total = [r["prediction"] for r in results]
    
    truths_slice = [r["ground_truth"] for r in results_slice]
    preds_slice = [r["prediction"] for r in results_slice]

    # Calculate metrics
    metrics_slice = compute_metrics(truths_slice, preds_slice, labels, results_slice)
    metrics_total = compute_metrics(truths_total, preds_total, labels, results) if slice_limit > 0 else None

    # Print metadata
    print()
    print(RULE2)
    print(f"{C_BOLD}{C_CYAN}  METRICS REPORT: {os.path.basename(target_file)}{C_RESET}")
    print(RULE2)
    print(f"  {C_BOLD}Dataset:       {C_RESET}{meta.get('dataset_file', '?')}")
    print(f"  {C_BOLD}Model:         {C_RESET}{meta.get('model', '?')}  ({meta.get('provider', '?')})")
    print(f"  {C_BOLD}Template:      {C_RESET}{meta.get('template_path', '?')}")
    print(f"  {C_BOLD}Window:        {C_RESET}{meta.get('window', '?')}  prior turns")
    print(f"  {C_BOLD}Temperature:   {C_RESET}{meta.get('temperature', '?')}")
    print(f"  {C_BOLD}Thinking:      {C_RESET}{'on' if meta.get('include_thinking') else 'off'}")
    print(f"  {C_BOLD}Total Logged:  {C_RESET}{total_count} turns")
    if slice_limit > 0:
        print(f"  {C_BOLD}Slice Limit:   {C_RESET}{slice_limit} turns")

    # Print main table
    print_metrics_table(f"Summary Table ({slice_str})", metrics_slice, metrics_total)

    # Classification report breakdown
    use_sklearn = HAS_SKLEARN and not args.no_sklearn
    if use_sklearn:
        print()
        print(RULE)
        print(f"{C_BOLD}  PER-CLASS BREAKDOWN ({slice_str}){C_RESET}")
        print(RULE)
        try:
            report = classification_report(
                truths_slice, preds_slice, labels=labels, zero_division=0, output_dict=False
            )
            for line in report.splitlines():
                print(f"  {line}")
        except Exception as e:
            # Fallback report using unique classes in case of strict label failures
            try:
                unique_found = sorted(list(set(truths_slice) | set(preds_slice)))
                report = classification_report(
                    truths_slice, preds_slice, labels=unique_found, zero_division=0, output_dict=False
                )
                for line in report.splitlines():
                    print(f"  {line}")
            except Exception:
                print(f"  {C_RED}Could not generate per-class report: {e}{C_RESET}")
                
    print(RULE)
    print()

if __name__ == "__main__":
    main()
