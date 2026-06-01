import json
import os
import random
import argparse
import math
from collections import defaultdict


def _build_sample(dialogue, utterances, target_index):
    return {
        "dialogue_id": f"{dialogue['dialogue_id']}_turn_{target_index}",
        "original_dialogue_id": dialogue["dialogue_id"],
        "target_index": target_index,
        "utterances": utterances[: target_index + 1],
    }


def _load_emotion_groups(input_path, exclude_emotions):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    emotion_groups = defaultdict(list)
    for dialogue in data:
        utterances = dialogue.get("utterances", [])
        for i, utt in enumerate(utterances):
            emotion = utt.get("emotion", "unknown")
            if emotion in exclude_emotions:
                continue
            emotion_groups[emotion].append(_build_sample(dialogue, utterances, i))

    return emotion_groups


def _save_subset(output_path, selected_samples):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, indent=2, ensure_ascii=False)


def _print_subset_distribution(prefix, samples):
    dist = defaultdict(int)
    for sample in samples:
        dist[sample["utterances"][-1]["emotion"]] += 1
    print(f"   {prefix}: {dict(sorted(dist.items()))}")


def _select_balanced_samples(emotion_groups, target_count):
    num_classes = len(emotion_groups)
    samples_per_class = target_count // num_classes
    remainder = target_count % num_classes

    selected = []
    shortfall = 0
    sorted_emotions = sorted(emotion_groups.keys())

    for emotion in sorted_emotions:
        group = emotion_groups[emotion]
        random.shuffle(group)

        take = samples_per_class + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        actual_take = min(len(group), take)
        selected.extend(group[:actual_take])
        if len(group) < take:
            shortfall += take - len(group)

    if shortfall <= 0:
        return selected

    selected_ids = {s["dialogue_id"] for s in selected}
    available_extra = []
    for emotion in sorted_emotions:
        for sample in emotion_groups[emotion]:
            if sample["dialogue_id"] not in selected_ids:
                available_extra.append(sample)

    random.shuffle(available_extra)
    selected.extend(available_extra[:shortfall])
    return selected


def extract_balanced_subset(
    input_path, output_path, target_count=100, seed=42, exclude_emotions=None
):
    """
    Extracts a balanced subset of 'target_count' utterances from the dataset.
    Each selected utterance is stored with its dialogue context (history).
    """
    if exclude_emotions is None:
        exclude_emotions = []

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    random.seed(seed)
    emotion_groups = _load_emotion_groups(input_path, exclude_emotions)

    # 2. Balance classes
    num_classes = len(emotion_groups)
    if num_classes == 0:
        print(f"No emotions found in {input_path}")
        return
    temp_selected = _select_balanced_samples(emotion_groups, target_count)

    # Final shuffle and limit to exactly target_count
    random.shuffle(temp_selected)
    final_output = temp_selected[:target_count]

    _save_subset(output_path, final_output)

    print(f"Extracted {len(final_output)} samples to {output_path}")
    _print_subset_distribution("Distribution", final_output)


def _largest_remainder_targets(emotion_groups, target_count):
    """Allocate per-class sample counts that preserve source distribution."""
    total_available = sum(len(v) for v in emotion_groups.values())
    if total_available == 0:
        return {}

    safe_target = min(target_count, total_available)
    raw = {}
    base = {}
    for emotion, group in emotion_groups.items():
        expected = safe_target * (len(group) / total_available)
        raw[emotion] = expected
        base[emotion] = min(len(group), int(math.floor(expected)))

    assigned = sum(base.values())
    remainder = safe_target - assigned

    if remainder > 0:
        ranking = sorted(
            raw.keys(),
            key=lambda e: (raw[e] - math.floor(raw[e]), len(emotion_groups[e]), e),
            reverse=True,
        )
        i = 0
        while remainder > 0 and ranking:
            emotion = ranking[i % len(ranking)]
            if base[emotion] < len(emotion_groups[emotion]):
                base[emotion] += 1
                remainder -= 1
            i += 1
            # Guard to avoid pathological infinite loops if all classes are full.
            if i > len(ranking) * (safe_target + 1):
                break

    return base


def extract_proportional_subset(
    input_path, output_path, target_count=100, seed=42, exclude_emotions=None
):
    """
    Extract a subset preserving the source emotion distribution as closely as possible.
    Each selected utterance is stored with its dialogue context (history).
    """
    if exclude_emotions is None:
        exclude_emotions = []

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    random.seed(seed)
    emotion_groups = _load_emotion_groups(input_path, exclude_emotions)

    if not emotion_groups:
        print(f"No emotions found in {input_path}")
        return

    targets = _largest_remainder_targets(emotion_groups, target_count)

    selected = []
    for emotion in sorted(emotion_groups.keys()):
        group = emotion_groups[emotion]
        random.shuffle(group)
        selected.extend(group[: targets.get(emotion, 0)])

    # Fallback fill in case rounding/cap interactions leave a shortfall.
    total_available = sum(len(v) for v in emotion_groups.values())
    final_target = min(target_count, total_available)
    if len(selected) < final_target:
        selected_ids = {s["dialogue_id"] for s in selected}
        pool = []
        for emotion in sorted(emotion_groups.keys()):
            for sample in emotion_groups[emotion]:
                if sample["dialogue_id"] not in selected_ids:
                    pool.append(sample)
        random.shuffle(pool)
        selected.extend(pool[: final_target - len(selected)])

    random.shuffle(selected)
    final_output = selected[:final_target]

    _save_subset(output_path, final_output)

    source_dist = {k: len(v) for k, v in sorted(emotion_groups.items())}

    print(f"Extracted {len(final_output)} samples to {output_path}")
    print(f"   Source distribution: {source_dist}")
    _print_subset_distribution("Subset distribution", final_output)


def main():
    parser = argparse.ArgumentParser(
        description="Extract balanced subset of datasets for experimentation."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Total number of samples to extract per dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/subsets",
        help="Directory to save subsets.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="balanced",
        choices=["balanced", "proportional"],
        help="Sampling strategy: balanced per class or proportional to original distribution.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MELD", "IEMOCAP", "CA-MER"],
        help="Dataset names to process. Example: --datasets MELD IEMOCAP",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional filename suffix (without extension), e.g. --suffix origdist",
    )

    args = parser.parse_args()

    all_datasets = {
        "MELD": ("data/processed/meld_train_processed.json", []),
        "IEMOCAP": ("data/processed/iemocap_processed.json", ["oth", "xxx"]),
        "CA-MER": ("data/processed/camer_processed.json", []),
    }

    requested = {d.strip().upper() for d in args.datasets}
    datasets = {k: v for k, v in all_datasets.items() if k in requested}
    if not datasets:
        print(
            f"No matching datasets in --datasets {args.datasets}. Available: {list(all_datasets.keys())}"
        )
        return

    for name, (path, exclude) in datasets.items():
        suffix_part = f"_{args.suffix}" if args.suffix else ""
        output_file = os.path.join(
            args.output_dir,
            f"{name.lower()}_subset_{args.count}_{args.strategy}{suffix_part}.json",
        )

        if args.strategy == "proportional":
            extract_proportional_subset(
                path,
                output_file,
                target_count=args.count,
                seed=args.seed,
                exclude_emotions=exclude,
            )
        else:
            extract_balanced_subset(
                path,
                output_file,
                target_count=args.count,
                seed=args.seed,
                exclude_emotions=exclude,
            )


if __name__ == "__main__":
    main()
