import pandas as pd
import json
import os

def process_meld_file(csv_path, output_json_path, split_name):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort by Dialogue_ID and Utterance_ID to ensure chronological order
    df = df.sort_values(by=['Dialogue_ID', 'Utterance_ID'])
    
    dialogues = []
    
    for diag_id, group in df.groupby('Dialogue_ID'):
        utterances = []
        for _, row in group.iterrows():
            utt_id = int(row['Utterance_ID'])
            # Construct relative path
            # Standard: MELD.Raw/clips/{split}/dia{D}_utt{U}.mp4
            clip_name = f"dia{diag_id}_utt{utt_id}.mp4"
            rel_clip_path = f"MELD.Raw/clips/{split_name}/{clip_name}"
            
            utterance = {
                "utterance_id": utt_id,
                "speaker": str(row['Speaker']),
                "text": str(row['Utterance']),
                "emotion": str(row['Emotion']).lower(),
                "sentiment": str(row['Sentiment']).lower(),
                "audio_path": rel_clip_path,
                "video_path": rel_clip_path,
                "soft_labels": {
                    str(row['Emotion']).lower(): 1.0
                }
            }
            utterances.append(utterance)
        
        dialogues.append({
            "dialogue_id": f"MELD_{diag_id}",
            "source": "MELD",
            "split": split_name,
            "utterances": utterances
        })
    
    # Verify at least one file exists
    if dialogues and utterances:
        test_path = utterances[0]["audio_path"]
        if not os.path.exists(test_path):
             print(f"Warning: Clip not found at {test_path}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_json_path}")

if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    base_path = BASE_DIR / "data" / "raw" / "MELD.Raw"
    
    files = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv"
    }
    
    for split, filename in files.items():
        full_path = base_path / filename
        if full_path.exists():
            output_name = BASE_DIR / "data" / "processed" / f"meld_{split}_processed.json"
            process_meld_file(str(full_path), str(output_name), split)
        else:
            print(f"Warning: {full_path} not found.")
