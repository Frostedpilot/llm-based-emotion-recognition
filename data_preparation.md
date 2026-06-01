# Dataset Preparation Guide

This guide details how to structure, ingest, and subset raw data for the **MELD** and **IEMOCAP** datasets. The ingestion process converts raw labels, transcriptions, and media pointers into the unified JSON format used by our LLM ERC evaluation pipelines.

---

## 📂 Raw Data Folder Structure

Before running any preprocessing scripts, you must place the raw dataset files in the `data/raw/` directory according to the following layout:

```text
repo_root/
├── raw/
│   ├── MELD.Raw/
│   │   ├── train_sent_emo.csv
│   │   ├── dev_sent_emo.csv
│   │   ├── test_sent_emo.csv
│   │   └── clips/
│   │       ├── train/           # e.g., dia0_utt0.mp4, etc.
│   │       ├── dev/
│   │       └── test/
│   │
│   └── IEMOCAP_full_release/
│       ├── Session1/
│       │   ├── dialog/
│       │   │   ├── avi/DivX/       # avi dialogue recordings
│       │   │   ├── transcription/  # transcription .txt files
│       │   │   └── EmoEvaluation/  # evaluator .txt files (labels)
│       │   └── sentences/
│       │       └── wav/            # sentence-sliced .wav audio
│       ├── Session2/
│       ├── Session3/
│       ├── Session4/
│       └── Session5/
└── processed/                      # target directory for output JSONs
```

---

## ⚙️ Data Ingestion Scripts

We provide dedicated python scripts under `scripts/ingestion/` to clean and structure raw transcriptions, align timestamps, build emotion ground truths, and define relative paths to audio/video media clips.

### 1. Preprocessing MELD
The MELD preprocessing script reads the standard split CSV files, groups utterances by dialogue ID in strict chronological sequence, and maps media relative paths.

* **Script Path**: `scripts/ingestion/process_meld.py`
* **Execution**:
  ```bash
  python scripts/ingestion/process_meld.py
  ```
* **What it does**:
  1. Loads `train_sent_emo.csv`, `dev_sent_emo.csv`, and `test_sent_emo.csv` from `data/raw/MELD.Raw/`.
  2. Sorts utterances chronologically using `Dialogue_ID` and `Utterance_ID`.
  3. Maps relative video/audio clip pointers: `MELD.Raw/clips/{split}/dia{Dialogue_ID}_utt{Utterance_ID}.mp4`.
  4. Formats discrete emotion labels as 1.0 soft labels.
  5. Saves output files in the unified format at `data/processed/meld_{split}_processed.json`.

---

### 2. Preprocessing IEMOCAP
The IEMOCAP preprocessing script integrates raw turn-level transcriptions, multiannotator categorical soft-labels (based on evaluation agreement), and acoustic VAD (Valence, Arousal, Dominance) scores across all 5 Sessions.

* **Script Path**: `scripts/ingestion/process_iemocap.py`
* **Execution**:
  ```bash
  python scripts/ingestion/process_iemocap.py
  ```
* **What it does**:
  1. Scans folders `Session1` to `Session5` inside `data/raw/IEMOCAP_full_release/`.
  2. Parses transcription `.txt` files to extract utterance ID, speaker identity (`M`/`F`), and the spoken text block.
  3. Parses multiannotator categorical evaluation files under `EmoEvaluation/` to capture:
     * **Hard Emotion**: Consensus categorical label.
     * **Soft Labels**: Reconstructed probability distribution across annotators (e.g., if 2 annotators chose `neutral` and 1 chose `frustration`, it builds `{ "neutral": 0.67, "frustration": 0.33 }`).
     * **VAD (Valence, Arousal, Dominance)**: Floating-point dimensional scoring vector.
  4. Resolves relative paths to sentence-sliced `.wav` files and dialogue `.avi` videos.
  5. Saves aggregated multi-session data into `data/processed/iemocap_processed.json`.

---

## ✂️ Subsetting & Sampling Experiments

For rapid prototyping and prompt tuning, running evaluations on full datasets can be computationally expensive and time-consuming. We provide a helper utility `scripts/utils/extract_subset.py` to compile balanced or distribution-proportional subsets.

### How to Extract Subsets

Run the script from the root directory specifying the target sample count, sampling strategy, and target datasets:

```bash
# Extract balanced 100-sample subsets for both datasets (saves to data/processed/subsets/)
python scripts/utils/extract_subset.py --count 100 --strategy balanced --datasets MELD IEMOCAP

# Extract 200-sample subsets preserving the original dataset's emotion label distribution
python scripts/utils/extract_subset.py --count 200 --strategy proportional --suffix origdist --datasets MELD IEMOCAP
```

### Parameter Reference
* `--count`: Total number of target utterance samples to output in the subset (e.g., `100`, `200`, `500`).
* `--strategy`: 
  * `balanced`: Pulls an equal number of samples per active emotion category.
  * `proportional`: Uses a largest-remainder target allocation to match the original emotion skew as closely as possible.
* `--datasets`: Space-separated list of datasets to process (`MELD` and/or `IEMOCAP`).
* `--suffix`: Optional suffix appended to the final filename (e.g., `origdist`).
* `--seed`: Seed value (default `42`) to guarantee reproducible random splits.

> [!NOTE]
> During subset creation for **IEMOCAP**, the script automatically ignores uninformative turn labels such as `oth` (other) and `xxx` (undefined/no agreement) to maximize test-run relevance.

---

## 📊 Processed JSON Structure

Both preprocessing scripts output a clean, unified schema that is ingested directly by `probe.py` and `benchmark.py`.

```json
[
  {
    "dialogue_id": "MELD_123",
    "source": "MELD",
    "split": "dev",
    "utterances": [
      {
        "utterance_id": 0,
        "speaker": "Ross",
        "text": "Hey, what are you guys doing?",
        "emotion": "neutral",
        "sentiment": "neutral",
        "audio_path": "MELD.Raw/clips/dev/dia123_utt0.mp4",
        "video_path": "MELD.Raw/clips/dev/dia123_utt0.mp4",
        "soft_labels": {
          "neutral": 1.0
        }
      }
    ]
  }
]
```

* **`dialogue_id`**: A unique ID indicating the dialogue session.
* **`utterances`**: Chronologically ordered list of conversational turns.
* **`target_index`** *(subsets only)*: Appended to subset json records to tell the benchmark runner exactly which utterance index is the evaluation target (while keeping preceding utterances in the array as historical context).
* **`soft_labels`**: Numerical probability distribution mapping annotator consensus, utilized during `--soft-label` runs to compute JSD (Jensen-Shannon Divergence).
* **`audio_path` / `video_path`**: Relative workspace paths used by the media loaders for frame/feature extraction during multimodal runs.
