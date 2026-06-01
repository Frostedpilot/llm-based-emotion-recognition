import base64
import hashlib
import mimetypes
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any

import json

# Root directory detection
BASE_DIR = Path(__file__).resolve().parent.parent
MIN_VIDEO_DURATION_SECONDS = 1.0


def extract_egemaps(media_path: str) -> Dict[str, float]:
    """
    Extracts eGeMAPS v02 features from the given media path using openSMILE.
    Uses caching to avoid redundant extractions.
    """
    import opensmile

    # 1. Resolve absolute path
    abs_path = get_absolute_media_path(media_path)
    if not abs_path:
        return {}

    media_path_str = str(abs_path)
    media_hash = hashlib.md5(media_path_str.encode()).hexdigest()
    cache_dir = BASE_DIR / "tmp" / "egemaps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{media_hash}.json"

    # 2. Check cache
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except:
            pass

    # 3. Extract audio first (OpenSMILE works best on WAV)
    tmp_wav = BASE_DIR / "tmp" / f"audio_{media_hash}.wav"
    if not tmp_wav.exists():
        if not extract_audio(media_path_str, str(tmp_wav)):
            return {}

    # 4. Run OpenSMILE
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_file(str(tmp_wav))

        # Convert first row of DataFrame to dict
        features = df.iloc[0].to_dict()

        # 5. Save to cache
        # Convert float32/64 to standard float for JSON serialization
        features = {k: float(v) for k, v in features.items()}
        with open(cache_file, "w") as f:
            json.dump(features, f, indent=2)

        return features
    except Exception as e:
        print(f"[ERROR] eGeMAPS extraction failed: {e}")
        return {}


def encode_file_to_base64(file_path: str) -> str:
    """Encodes a file to a base64 string with appropriate MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def extract_iframes(video_path: str, max_frames: int = 3) -> List[str]:
    """
    Extracts only I-frames (keyframes) from video for efficient vision analysis.
    Returns a list of paths to the extracted images.
    """
    tmp_parent = BASE_DIR / "tmp" / "keyframes"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    shot_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
    tmp_dir = tmp_parent / shot_hash
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Extract only keyframes (I-frames)
        # -vsync vfr ensures we dont duplicate frames
        out_pattern = str(tmp_dir / "f_%03d.jpg")
        cmd = [
            "ffmpeg",
            "-y",
            "-skip_frame",
            "nokey",
            "-i",
            video_path,
            "-vsync",
            "vfr",
            out_pattern,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        frames = sorted(list(tmp_dir.glob("f_*.jpg")))
        if not frames:
            return []

        # Sample max_frames uniformly
        if len(frames) <= max_frames:
            selected = frames
        elif max_frames == 1:
            # For 1 frame, take the middle one (usually the most meaningful)
            selected = [frames[len(frames) // 2]]
        else:
            # Linear interpolation to pick max_frames indices
            indices = [
                int(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)
            ]
            # Ensure unique indices and sort them
            selected = [frames[i] for i in sorted(list(set(indices)))]

        return [str(p) for p in selected]
    except Exception as e:
        print(f"[ERROR] Extracting iframes from {video_path}: {e}")
        return []


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extracts audio from video as a 16kHz mono WAV file."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[ERROR] Extracting audio from {video_path}: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"        FFmpeg stderr: {e.stderr.decode()}")
        return False


def strip_audio(video_path: str, output_path: str) -> bool:
    """
    Removes audio from video using ffmpeg.
    Uses '-vcodec copy' to avoid re-encoding the video stream.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-an",  # Disable audio
            "-vcodec",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[ERROR] Stripping audio from {video_path}: {e}")
        return False


def get_video_duration_seconds(video_path: str) -> Optional[float]:
    """Returns video duration in seconds using ffprobe, or None if unavailable."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw = (res.stdout or "").strip()
        if not raw:
            return None
        return float(raw)
    except Exception:
        return None


def get_absolute_media_path(relative_path: str) -> Optional[Path]:
    """Resolves a relative dataset media path to an absolute local path."""
    if not relative_path:
        return None

    # 1. Try direct path (e.g. if path is already absolute or relative to root)
    full_path = BASE_DIR / relative_path
    if full_path.exists():
        return full_path

    # 2. Try inside data/raw/
    raw_path = BASE_DIR / "data" / "raw" / relative_path
    if raw_path.exists():
        return raw_path

    # 3. Special case for MELD if the path doesn't have the data/raw prefix
    # but refers to the MELD.Raw folder directly
    if "MELD" in relative_path and not relative_path.startswith("data/raw"):
        meld_path = BASE_DIR / "data" / "raw" / relative_path
        if meld_path.exists():
            return meld_path

    return None


def get_utterance_bounds(utterance_id: str) -> Optional[tuple[float, float]]:
    """Parses IEMOCAP EmoEvaluation files to extract exact start and end times for an utterance."""
    import re
    if not utterance_id:
        return None
    try:
        parts = utterance_id.rsplit('_', 1)
        if len(parts) < 2:
            return None
        dialog_name = parts[0]
        # Extract Session number (e.g. Ses05 -> 5)
        match = re.search(r'Ses(\d{2})', dialog_name)
        if not match:
            return None
        session_num = int(match.group(1))
        session_name = f"Session{session_num}"
        
        eval_path = BASE_DIR / "data" / "raw" / "IEMOCAP_full_release" / session_name / "dialog" / "EmoEvaluation" / f"{dialog_name}.txt"
        if not eval_path.exists():
            # Fallback search across all sessions
            for i in range(1, 6):
                eval_path = BASE_DIR / "data" / "raw" / "IEMOCAP_full_release" / f"Session{i}" / "dialog" / "EmoEvaluation" / f"{dialog_name}.txt"
                if eval_path.exists():
                    break
            else:
                return None
        
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if utterance_id in line and line.startswith("["):
                    time_match = re.search(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]', line)
                    if time_match:
                        return float(time_match.group(1)), float(time_match.group(2))
    except Exception as e:
        print(f"[media_utils] Warning: Error parsing bounds for {utterance_id}: {e}")
    return None


def cut_video_snippet(original_video_path: str, start: float, end: float, output_path: str) -> bool:
    """Uses FFmpeg to precisely cut a video snippet between start and end times, re-encoding it to MP4."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.4f}",
            "-to", f"{end:.4f}",
            "-i", original_video_path,
            "-c:v", "libx264",
            "-preset", "superfast",
            "-crf", "23",
            "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[media_utils] Error cutting video snippet from {start} to {end}: {e}")
        if hasattr(e, "stderr") and e.stderr:
            print(f"[media_utils] FFmpeg stderr: {e.stderr.decode()}")
        return False


def prepare_multimodal_content(
    text_content: str,
    video_path: Optional[str],
    include_video: bool,
    include_audio: bool,
    max_vision_frames: int = 3,
    visual_mode: str = "image",
    utterance_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Transforms a text string into an OpenRouter-compatible multimodal content list
    if video or audio modalities are requested. Supports dynamic snippet cutting for IEMOCAP.
    """
    if not include_video and not include_audio:
        return [{"type": "text", "text": text_content}]

    content_list = [{"type": "text", "text": text_content}]
    abs_v_path = get_absolute_media_path(video_path) if video_path else None

    # Precise video cutting for IEMOCAP if utterance_id is provided
    if abs_v_path and utterance_id and "IEMOCAP_full_release" in str(abs_v_path):
        bounds = get_utterance_bounds(utterance_id)
        if bounds:
            start, end = bounds
            snippets_dir = BASE_DIR / "tmp" / "snippets"
            snippets_dir.mkdir(parents=True, exist_ok=True)
            snippet_path = snippets_dir / f"{utterance_id}.mp4"
            if not snippet_path.exists():
                print(f"      [IEMOCAP Snippet] Cutting video for {utterance_id} ({start:.2f}s - {end:.2f}s)...")
                success = cut_video_snippet(str(abs_v_path), start, end, str(snippet_path))
                if success:
                    abs_v_path = snippet_path
                else:
                    print(f"      [IEMOCAP Snippet] Fallback to entire dialogue video due to cutting error.")
            else:
                abs_v_path = snippet_path

    if abs_v_path:
        (BASE_DIR / "tmp").mkdir(exist_ok=True)

        if include_video:
            if visual_mode == "video":
                # Strip audio to avoid context leak
                v_hash = hashlib.md5(str(abs_v_path).encode()).hexdigest()[:12]
                stripped_v_path = (
                    BASE_DIR / "tmp" / f"vision_{v_hash}{abs_v_path.suffix}"
                )
                if not stripped_v_path.exists():
                    if not strip_audio(str(abs_v_path), str(stripped_v_path)):
                        # Fallback to original if stripping fails
                        stripped_v_path = abs_v_path

                content_list.append(
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": encode_file_to_base64(str(stripped_v_path))
                        },
                    }
                )
                print(
                    f"      [Vision] Added 1 video file to prompt (audio stripped: {stripped_v_path != abs_v_path})."
                )
            else:
                kf_paths = extract_iframes(
                    str(abs_v_path), max_frames=max_vision_frames
                )
                for kf in kf_paths:
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_file_to_base64(kf)},
                        }
                    )
                print(f"      [Vision] Added {len(kf_paths)} image frame(s) to prompt.")

        if include_audio:
            tmp_wav = (
                BASE_DIR
                / "tmp"
                / f"audio_{hashlib.md5(str(abs_v_path).encode()).hexdigest()}.wav"
            )
            if extract_audio(str(abs_v_path), str(tmp_wav)):
                # OpenRouter expects raw base64 (without prefix) for audio
                b64_audio = encode_file_to_base64(str(tmp_wav)).split(",")[-1]
                content_list.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_audio, "format": "wav"},
                    }
                )
                print(f"      [Audio] Added 1 audio track to prompt ({len(b64_audio)} base64 chars).")
            else:
                print(f"      [Audio] Extraction failed for {abs_v_path}")

    return content_list
