# LLM-Based Emotion Recognition in Conversation (ERC) Framework

This repository contains a framework for evaluating and experimenting with Large Language Models (LLMs) on **Emotion Recognition in Conversation (ERC)** tasks. It supports both **local models** (via Ollama) and **cloud models** (via OpenRouter), single-agent and multi-agent workflows, soft-label probability distributions, and multimodal inputs (text, vision keyframe extraction, acoustic features).

---

## 🚀 Overview

The framework is organized around three main CLI utilities located in the root directory:

1. **`probe.py`**: A debugging tool to test the LLM's response and behavior on a *single utterance* within a selected conversation. It prints the full rendered prompt sent to the model and streams the raw and structured responses in real-time.
2. **`benchmark.py`**: An automated benchmarking tool to run systematic evaluations across a dataset (or subsets of conversations). It provides a real-time progress bar, live accuracy and $F_1$-weighted metrics, and resume support.
3. **`calculate_metrics.py`**: A stats post-processor that allows you to recalculate metrics, generate detailed classification reports (precision, recall, F1 per class), slice results (e.g., analyze the first $N$ entries), and compare overall performance.

---

## 🛠️ Setup & Requirements

### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install ollama openai scikit-learn colorama python-dotenv
```

### 2. Environment Configuration
Create a `.env` file in the root directory based on `.env.example`:
```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
OLLAMA_API_BASE=http://127.0.0.1:11434
```

### 3. Folder Structure & Data Prep
For detailed instructions on how to download, structure, and ingest the raw MELD and IEMOCAP data, or extract experimental subsets, please refer to the [Dataset Preparation Guide](data_preparation.md).

The framework expects the following directories:
* **`data/processed/`**: Cleaned JSON datasets (e.g., `meld_dev_processed.json`, `iemocap_processed.json`).
* **`data/processed/subsets/`**: Smaller conversation subsets (e.g., `meld_subset_100.json`, `iemocap_subset_200_proportional_origdist.json`) for quick benchmarking.
* **`prompts/`**: TXT templates specifying system instructions and formatting rules for ERC prompts.
* **`results/`**: Output directory where benchmark runs are automatically stored as JSON files with extensive run-level metadata.

---

## 🔍 Detailed Usage Guide

### 1. `probe.py` — Interactive Debugger & Single-Utterance Tester
Use `probe.py` to examine exactly what prompt is constructed and stream the model's exact output (including intermediate chain-of-thought `<thought>` blocks).

#### Interactive Mode
Simply run the script with no arguments to pick the dataset, conversation, and target utterance interactively via a numbered list:
```bash
python probe.py
```

#### Command-Line Arguments
```bash
# Pick specific dataset alias, conversation index (0-based), and utterance index (0-based)
python probe.py -d meld_dev -c 3 -u 1

# List all utterances in dialogue index 3 and exit (useful to find target utterances)
python probe.py -d meld_dev -c 3 --list-utterances

# Use a custom prompt template
python probe.py -d meld_dev -c 3 -u 1 -p prompts/erc_cot.txt

# Run via a cloud provider (OpenRouter) using Claude 3.5 Sonnet
python probe.py -d iemocap -c 0 -u 2 --provider openrouter -m anthropic/claude-3.5-sonnet

# Override context window (number of previous turns included in history)
python probe.py -d meld_dev -c 3 -u 1 --window 3

# Disable thinking/reasoning blocks for deep-seek or qwen-style reasoning models
python probe.py -d meld_dev -c 3 -u 1 -m qwen3.5:4b --no-think

# Enable multimodal vision / audio extraction
python probe.py -d meld_dev -c 3 -u 1 --vision --vision-frames 3 --audio

# Run with soft labels (request JSON probability distribution and calculate JS-Divergence)
python probe.py -d meld_dev -c 3 -u 1 --soft-label
```

---

### 2. `benchmark.py` — Systematic Dataset Evaluator
Use `benchmark.py` to run comprehensive evaluations on large cohorts of dialogues, track metrics in real-time, and log outputs safely with full config compatibility checks.

#### Interactive Mode
Run the script with no parameters to choose a dataset and configure the run size interactively:
```bash
python benchmark.py
```

#### Command-Line Arguments
```bash
# Run first 10 dialogues of MELD dev set
python benchmark.py -d meld_dev -n 10

# Run 10 random dialogues using a fixed seed
python benchmark.py -d meld_dev -n 10 --random --seed 42

# Run using a specific local Ollama model and custom COT prompt
python benchmark.py -d meld_dev -n 50 -m qwen3.5:4b -p prompts/erc_cot.txt

# Run using OpenRouter with Claude 3.5 Haiku
python benchmark.py -d iemocap -n 15 --provider openrouter -m anthropic/claude-3.5-haiku

# Request probability distributions (soft labels) to log Jensen-Shannon Divergence
python benchmark.py -d meld_dev -n 20 --soft-label
```

#### Caching & Resuming Runs
Every benchmark run is stored inside `results/` in a structured format containing both a `"meta"` dictionary (specifying all configuration variables) and a `"results"` array. 

If a benchmark run is interrupted (e.g., due to an API timeout, system crash, or rate limit), you can resume it using `--resume`:
```bash
python benchmark.py -d meld_dev -n 100 -o results/my_run.json --resume
```
> [!IMPORTANT]
> The framework performs strict sanity checks on resume. If any result-affecting settings (`dataset_file`, `model`, `provider`, `template_sha256`, `window`, `temperature`, `include_thinking`, `soft_label`, or agent configuration) differ from the original run, the script flags a settings mismatch and prompts you to merge or abort.
>
> To bypass these checks and force-resume, append the `--force` flag:
> ```bash
> python benchmark.py -d meld_dev -n 100 -o results/my_run.json --resume --force
> ```

---

### 3. `calculate_metrics.py` — Analytical Report Generator
Use `calculate_metrics.py` to perform deep-dives on your benchmark results without running new API inferences. It recalculates overall metrics and outputs a clean, command-line report.

#### Interactive Selector
Running the script without arguments displays a list of your JSON result files inside `results/` sorted by modification date (newest first). You can select a file by index or type a query to filter them down interactively:
```bash
python calculate_metrics.py
```

#### Command-Line Arguments
```bash
# Calculate metrics for a specific result file
python calculate_metrics.py results/bench_meld_dev_qwen3.5_4b_erc_default_n10_20260601_080000.json

# Analyze and print stats using only the first N logged turns (useful for early performance checks)
python calculate_metrics.py results/my_run.json -l 100

# Force standard fallback metrics (bypasses importing scikit-learn metrics)
python calculate_metrics.py results/my_run.json --no-sklearn
```

#### Generated Report Breakdown
The output report displays:
* **Configuration Metadata**: Model, dataset, prompt path, context window size, etc.
* **Side-by-Side Comparison Table**: If `--limit` is specified, it prints a clean ASCII comparison showing metrics for the limited subset alongside the overall total (including Sample Size, Correct Classifications, Accuracy, Weighted $F_1$, Macro $F_1$, and Avg JS-Divergence).
* **Per-Class Breakdown**: Detailed classification reports showing precision, recall, f1-score, and support for each emotion label (e.g., `joy`, `anger`, `sadness`, `neutral`, etc.).

---

## ⚙️ Core Configuration Flags

The CLI tools share several foundational configuration parameters that tune prediction dynamics:

| Flag | Default | Description |
| :--- | :---: | :--- |
| `-d`, `--dataset` | *None* | Short alias (e.g., `meld_dev`, `iemocap`) or full path to the processed JSON dataset. |
| `-m`, `--model` | *Preset* | The model identifier. Defaults to `qwen3.5:4b` locally and `qwen/qwen3.6-plus:free` on OpenRouter. |
| `--provider` | `local` | `local` (Ollama connection on `127.0.0.1:11434`) or `openrouter` (OpenRouter API). |
| `--window` | `5` | Context window size. Includes up to $N$ prior conversational turns as history in the prompt. |
| `--temperature`| `0.0` | Sampling temperature. `0.0` ensures greedy decoding for high reproducibility. |
| `--no-think` | *False* | Disables Qwen/DeepSeek chain-of-thought tags (Ollama `/no_think`). |
| `--soft-label` | *False* | Requests probability distribution outputs and calculates JS-Divergence against ground truth. |
| `--vision` | *False* | Extracts visual features (I-frames) from raw video paths for multi-modal model analysis. |
| `--audio` | *False* | Extracts acoustic features from raw audio paths for multi-modal model analysis. |

---

## 📊 Dataset Ground Truth Emotion Labels

The benchmark automatic metrics and per-class reports automatically align labels depending on the target dataset:

* **MELD & CAMER**:
  `["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]`
* **IEMOCAP**:
  `["neutral", "happiness", "sadness", "anger", "fear", "frustration", "excitement"]`
* **IEMOCAP (6-class test)**:
  `["neutral", "frustration", "excitement", "sadness", "anger", "happiness"]`
