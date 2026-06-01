import os
import re
import json
from collections import Counter

def parse_iemocap_evaluation(eval_path):
    labels = {}
    with open(eval_path, "r") as f:
        lines = f.readlines()
    
    current_turn = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('['):
            match = re.search(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]', line)
            if match:
                start, end, turn_id, hard_emo, v, a, d = match.groups()
                current_turn = turn_id
                labels[current_turn] = {
                    "hard_emotion": hard_emo.lower(),
                    "soft_labels_list": [],
                    "v_a_d": [float(v), float(a), float(d)]
                }
        elif line.startswith('C-') and current_turn:
            parts = line.split(':')
            if len(parts) > 1:
                sub_labels = parts[1].split(';')
                for sl in sub_labels:
                    sl = sl.strip().lower()
                    if sl and sl != '()' and not sl.startswith('('):
                        labels[current_turn]["soft_labels_list"].append(sl)
    
    for turn_id in labels:
        list_labels = labels[turn_id]["soft_labels_list"]
        if not list_labels:
            labels[turn_id]["soft_labels"] = {labels[turn_id]["hard_emotion"]: 1.0}
        else:
            counts = Counter(list_labels)
            total = sum(counts.values())
            labels[turn_id]["soft_labels"] = {k: v/total for k, v in counts.items()}
        del labels[turn_id]["soft_labels_list"]
        
    return labels

def parse_iemocap_transcription(trans_path):
    turns = []
    with open(trans_path, "r") as f:
        for line in f:
            match = re.search(r'^(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s+(.*)$', line.strip())
            if match:
                turn_id, text = match.groups()
                speaker = "F" if "_F" in turn_id else "M"
                turns.append({
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "text": text.strip()
                })
    return turns

def process_iemocap(root_dir, output_json_path):
    all_data = []
    sessions = [f"Session{i}" for i in range(1, 6)]
    
    # Workspace root for relative paths
    workspace_root = os.getcwd()

    for session in sessions:
        print(f"Processing {session}...")
        session_path = os.path.join(root_dir, session)
        trans_dir = os.path.join(session_path, "dialog", "transcriptions")
        eval_dir = os.path.join(session_path, "dialog", "EmoEvaluation")
        
        # Audio/Video directories for linking
        # Typical IEMOCAP release: sentences/wav/ Ses_ID / turn_ID.wav
        # Typical IEMOCAP release: dialog/avi/DivX/ dialog_name.avi
        wav_base_dir = os.path.join(session_path, "sentences", "wav")
        video_dir = os.path.join(session_path, "dialog", "avi", "DivX")
        
        if not os.path.exists(trans_dir):
            continue
            
        files_processed = 0
        for trans_file in os.listdir(trans_dir):
            if not trans_file.endswith(".txt"):
                continue
            
            dialog_name = trans_file.replace(".txt", "")
            trans_path = os.path.join(trans_dir, trans_file)
            eval_path = os.path.join(eval_dir, trans_file)
            
            if not os.path.exists(eval_path):
                continue
            
            trans_data = parse_iemocap_transcription(trans_path)
            eval_data = parse_iemocap_evaluation(eval_path)
            
            utterances = []
            for t_item in trans_data:
                turn_id = t_item["turn_id"]
                e_info = eval_data.get(turn_id)
                if e_info:
                    # Construct relative paths
                    # Audio: IEMOCAP_full_release/{session}/sentences/wav/{dialog_name}/{turn_id}.wav
                    rel_audio_path = f"IEMOCAP_full_release/{session}/sentences/wav/{dialog_name}/{turn_id}.wav"
                    # Video: IEMOCAP_full_release/{session}/dialog/avi/DivX/{dialog_name}.avi
                    rel_video_path = f"IEMOCAP_full_release/{session}/dialog/avi/DivX/{dialog_name}.avi"
                    
                    utterance = {
                        "utterance_id": turn_id,
                        "speaker": t_item["speaker"],
                        "text": t_item["text"],
                        "emotion": e_info["hard_emotion"],
                        "audio_path": rel_audio_path,
                        "video_path": rel_video_path,
                        "soft_labels": e_info["soft_labels"],
                        "v_a_d": e_info["v_a_d"]
                    }
                    utterances.append(utterance)
            
            if utterances:
                all_data.append({
                    "dialogue_id": f"IEMOCAP_{dialog_name}",
                    "source": "IEMOCAP",
                    "session": session,
                    "utterances": utterances
                })
                files_processed += 1
        print(f"Processed {files_processed} files in {session}")
                
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_data)} dialogues to {output_json_path}")

if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    root = BASE_DIR / "data" / "raw" / "IEMOCAP_full_release"
    output_path = BASE_DIR / "data" / "processed" / "iemocap_processed.json"
    process_iemocap(str(root), str(output_path))
