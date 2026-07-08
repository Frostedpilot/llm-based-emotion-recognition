import json
import os
import re
import random
import datetime
import hashlib
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Generator
import uvicorn
import subprocess
from pathlib import Path
from utils import (
    normalize_prediction,
    parse_soft_prediction,
    calculate_js_divergence,
    get_argmax_label,
    unify_label,
)
from llm_interface import InferenceBridge
from media_utils import prepare_multimodal_content, encode_file_to_base64
from prompt_templates import get_registry

try:
    from sklearn.metrics import f1_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Multi-agent ADK orchestration
try:
    from agents_logic import run_multiagent_stream, MODEL_PRESETS

    HAS_AGENTS = True
except ImportError as _e:
    HAS_AGENTS = False
    print(f"[WARN] agents_logic not available: {_e}")

def get_clean_prompt_text(content):
    """
    Extracts only the user prompt text block, omitting large base64 binary files
    and replacing them with visual placeholders to keep logs and previews clean.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif "video_url" in item or "image_url" in item or "audio_url" in item:
                    media_type = item.get("type", "media")
                    texts.append(f"\n[{media_type.upper()} DATA APPENDED]")
        return "\n".join(texts)
    return str(content)

app = FastAPI(title="Emotion Detection Research Bench")
BASE_DIR = Path(__file__).resolve().parent.parent

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading ---
DATA_DIR = BASE_DIR / "data" / "processed"
DATASETS = {
    "meld_train": DATA_DIR / "meld_train_processed.json",
    "meld_dev": DATA_DIR / "meld_dev_processed.json",
    "meld_test": DATA_DIR / "meld_test_processed.json",
    "iemocap": DATA_DIR / "iemocap_processed.json",
    "camer": DATA_DIR / "camer_processed.json",
}

cached_data = {}
prompt_registry = get_registry()


def load_dataset(name: str):
    if name in cached_data:
        return cached_data[name]

    # Try direct mapping first
    file_path = None
    if name in DATASETS:
        file_path = DATASETS[name]
    else:
        # Try finding in subsets/ folder with common suffixes
        for suffix in ["", "_100", "_processed"]:
            possible_subset = DATA_DIR / "subsets" / f"{name}{suffix}.json"
            if possible_subset.exists():
                file_path = possible_subset
                break

        if not file_path:
            # Fallback for name directly in DATA_DIR
            for suffix in ["", "_processed", "_100"]:
                fallback = DATA_DIR / f"{name}{suffix}.json"
                if fallback.exists():
                    file_path = fallback
                    break

    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cached_data[name] = data
            return data
    return None


# Serve data files (MELD, IEMOCAP, CA-MER)
app.mount("/data", StaticFiles(directory=str(BASE_DIR / "data")), name="data")
app.mount(
    "/MELD.Raw",
    StaticFiles(directory=str(BASE_DIR / "data" / "raw" / "MELD.Raw")),
    name="meld",
)
app.mount(
    "/IEMOCAP_full_release",
    StaticFiles(directory=str(BASE_DIR / "data" / "raw" / "IEMOCAP_full_release")),
    name="iemocap",
)


# --- Video Transcoding Endpoint for AVI files (e.g. IEMOCAP) ---
@app.get("/video/transcode")
async def transcode_video(path: str):
    """
    On-the-fly transcoder that converts unsupported AVI files to MP4.
    Caches the results to avoid duplicate conversions.
    """
    # path is like "IEMOCAP_full_release/Session1/dialog/avi/DivX/Ses01F_impro01.avi"
    raw_path = BASE_DIR / "data" / "raw" / path
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Source video not found.")

    # Create cached directory
    cache_dir = BASE_DIR / "data" / "processed" / "transcoded_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    out_filename = raw_path.with_suffix(".mp4").name
    out_path = cache_dir / out_filename

    # If cache doesn't exist, transcode it using async subprocess
    if not out_path.exists():
        cmd = [
            "ffmpeg",
            "-i", str(raw_path),
            "-c:v", "libx264",
            "-preset", "superfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            str(out_path)
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcoding failed: {stderr.decode()}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error executing ffmpeg: {str(e)}"
            )

    return FileResponse(out_path)


# --- Models ---
class InferenceRequest(BaseModel):
    dataset_name: str
    dialogue_id: str
    target_index: int
    model_id: str
    provider: str
    window_size: int = 5
    soft_label: bool = False
    stream: bool = False
    template_name: Optional[str] = "erc_default"
    custom_system_prompt: Optional[str] = None
    disable_thinking: bool = True
    include_video: bool = False
    include_audio: bool = False
    vision_frames: int = 3
    max_tokens: Optional[int] = None
    reasoning_max_tokens: Optional[int] = None


class PromptUpdate(BaseModel):
    content: str


class PreloadRequest(BaseModel):
    model_id: str
    provider: str = "local"


class AgentChatRequest(BaseModel):
    dataset_name: str
    dialogue_id: str
    target_index: int
    model_id: str
    provider: str = "openrouter"
    window_size: int = 5
    template_name: Optional[str] = "erc_default"
    workflow: str = "reasoner_verifier"
    vision_frames: int = 3
    soft_label: bool = False
    max_tokens: Optional[int] = None
    reasoning_max_tokens: Optional[int] = None


class BenchmarkRequest(BaseModel):
    dataset_name: str
    n_conversations: int = 5
    selection: str = "first"  # 'first' | 'random'
    seed: int = 42
    model_id: str = "qwen3.5:4b"
    provider: str = "local"
    template_name: str = "erc_default"
    window_size: int = 5
    disable_thinking: bool = True
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    include_video: bool = False
    include_audio: bool = False
    vision_frames: int = 3
    agent_mode: bool = False
    workflow: str = "reasoner_verifier"
    output_name: Optional[str] = None
    soft_label: bool = False


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


def infer_dataset_key(name: str) -> str:
    nl = name.lower()
    if "iemocap" in nl:
        return "iemocap"
    if "camer" in nl:
        return "camer"
    return "meld"


# --- Media Helpers (Most logic moved to media_utils.py) ---


def compute_metrics_bench(
    truths: list, preds: list, labels: list, results: list = None
) -> dict:
    results = results or []
    n = len(truths)
    correct = sum(t == p for t, p in zip(truths, preds))
    acc = correct / n if n else 0.0
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


# --- API Endpoints ---


@app.get("/datasets")
async def get_datasets_info():
    info = []
    for key, path in DATASETS.items():
        if os.path.exists(path):
            info.append(
                {"id": key, "name": key.replace("_", " ").title(), "status": "ready"}
            )
        else:
            info.append(
                {"id": key, "name": key.replace("_", " ").title(), "status": "missing"}
            )
    return info


@app.get("/dialogues/{dataset_id}")
async def get_dialogues(dataset_id: str):
    data = load_dataset(dataset_id)
    if not data:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    result = []
    for diag in data:
        result.append(
            {
                "id": diag["dialogue_id"],
                "turns": len(diag["utterances"]),
                "first_line": (
                    diag["utterances"][0]["text"][:50] + "..."
                    if diag["utterances"]
                    else ""
                ),
            }
        )
    return result


@app.get("/dialogue/{dataset_id}/{dialogue_id}")
async def get_dialogue_content(dataset_id: str, dialogue_id: str):
    data = load_dataset(dataset_id)
    if not data:
        raise HTTPException(status_code=404, detail="Dataset not found or missing.")
    for diag in data:
        if (
            diag.get("dialogue_id") == dialogue_id
            or diag.get("original_dialogue_id") == dialogue_id
        ):
            return diag
    raise HTTPException(
        status_code=404,
        detail=f"Dialogue {dialogue_id} not found in dataset {dataset_id}.",
    )


@app.post("/preload")
async def preload_model(req: PreloadRequest):
    bridge = InferenceBridge(provider=req.provider)
    bridge.preload_model(req.model_id)
    return {"status": "success", "message": f"Preloading {req.model_id}..."}


@app.post("/unload")
async def unload_model(req: PreloadRequest):
    bridge = InferenceBridge(provider=req.provider)
    bridge.unload_model(req.model_id)
    return {"status": "success", "message": f"Unloading {req.model_id}..."}


@app.post("/inference")
async def run_inference(req: InferenceRequest):
    data = load_dataset(req.dataset_name)
    if not data:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    dialogue = next((d for d in data if d["dialogue_id"] == req.dialogue_id), None)
    if not dialogue:
        raise HTTPException(status_code=404, detail="Dialogue not found.")

    utterances = dialogue["utterances"]
    if req.target_index >= len(utterances):
        raise HTTPException(status_code=400, detail="Invalid target index.")

    bridge = InferenceBridge(provider=req.provider)

    # Use current dataset context for injectors
    context = {
        "dataset_name": "MELD" if "meld" in req.dataset_name else "IEMOCAP",
        "utterances": utterances,
        "target_index": req.target_index,
        "window_size": req.window_size,
        "vision_frames": req.vision_frames,
    }

    # Determine template
    template = req.template_name or "erc_default"

    try:
        messages, metadata = prompt_registry.render(template, **context)
    except Exception as e:
        # Fallback if template doesn't exist
        messages, metadata = prompt_registry.render("erc_default", **context)

    if req.soft_label:
        messages[-1][
            "content"
        ] += "\n\nPlease provide a probability distribution over the emotions."

    if req.custom_system_prompt:
        messages[0]["content"] = req.custom_system_prompt

    # Inject override instruction to system prompt and prefix user message if disable_thinking is active
    if req.disable_thinking:
        if len(messages) > 0:
            messages[0]["content"] += "\n\nIMPORTANT: Thinking is disabled. Do NOT include any 'Thoughts:' or 'Thoughts' reasoning block in your output. Skip the 'Thoughts:' step. Output ONLY the 'Soft Labels:' or final dictionary directly."
        if len(messages) > 1:
            content = messages[1]["content"]
            if isinstance(content, str) and not content.startswith("/no_think"):
                messages[1]["content"] = "/no_think\n" + content

    # Multimodal handling via request or template metadata
    media_reqs = metadata.get("media", [])
    wants_video = "video" in media_reqs
    wants_image = "image" in media_reqs or "vision" in media_reqs
    wants_audio = "audio" in media_reqs
    include_v = req.include_video or wants_video or wants_image
    include_a = req.include_audio or wants_audio
    visual_mode = "video" if wants_video else "image"

    if (include_v or include_a) and req.provider == "openrouter":
        utt = utterances[req.target_index]
        raw_media_path = utt.get("video_path") or utt.get("audio_path")
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            u_id = utt.get("utterance_id")
            last_msg["content"] = prepare_multimodal_content(
                last_msg["content"],
                raw_media_path,
                include_v,
                include_audio=include_a,
                max_vision_frames=req.vision_frames,
                visual_mode=visual_mode,
                utterance_id=u_id,
            )

    if req.stream:

        def generate_inference_stream():
            try:
                # Yield metadata first (Exact prompt sent to LLM)
                yield f"<system>{messages[0]['content']}</system>"
                yield f"<prompt>{get_clean_prompt_text(messages[1]['content'])}</prompt>"

                # Synchronous generator called in a thread-safe way by StreamingResponse
                chunk_generator = bridge.chat(
                    model=req.model_id,
                    messages=messages,
                    stream=True,
                    include_thinking=not req.disable_thinking,
                    soft_label=req.soft_label,
                    max_tokens=req.max_tokens,
                    reasoning_max_tokens=req.reasoning_max_tokens or 10000,
                )
                for chunk in chunk_generator:
                    yield chunk
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(
            generate_inference_stream(),
            media_type="text/plain",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Encoding": "identity",
            },
        )
    else:
        prediction = bridge.chat(
            model=req.model_id,
            messages=messages,
            soft_label=req.soft_label,
            include_thinking=not req.disable_thinking,
            max_tokens=req.max_tokens,
            reasoning_max_tokens=req.reasoning_max_tokens or 10000,
        )
        return {
            "prediction": prediction,
            "prompt": get_clean_prompt_text(messages[1]["content"]),
            "system": messages[0]["content"],
        }


@app.get("/results")
async def list_results():
    results_dir = BASE_DIR / "results"
    if not results_dir.exists():
        return []
    # List all .json files in results folder
    return [f for f in os.listdir(results_dir) if f.endswith(".json")]


@app.get("/results/{filename}")
async def get_result_file(filename: str):
    file_path = BASE_DIR / "results" / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Result file not found.")


class RenderPromptRequest(BaseModel):
    dataset_name: str
    dialogue_id: str
    target_index: int
    template_name: str
    window_size: int = 5
    vision_frames: int = 3
    soft_label: bool = False


@app.post("/render_prompt_preview")
async def render_prompt_preview(req: RenderPromptRequest):
    data = load_dataset(req.dataset_name)
    if not data:
        # Try stripping extension just in case
        clean_name = req.dataset_name.replace(".json", "")
        data = load_dataset(clean_name)
        if not data:
            raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_name} not found.")

    dialogue = next((d for d in data if d.get("dialogue_id") == req.dialogue_id or d.get("original_dialogue_id") == req.dialogue_id), None)
    if not dialogue:
        raise HTTPException(status_code=404, detail=f"Dialogue {req.dialogue_id} not found.")

    utterances = dialogue["utterances"]
    if req.target_index >= len(utterances):
        raise HTTPException(status_code=400, detail="Invalid target index.")

    context = {
        "dataset_name": "MELD" if "meld" in req.dataset_name.lower() else "IEMOCAP",
        "utterances": utterances,
        "target_index": req.target_index,
        "window_size": req.window_size,
        "vision_frames": req.vision_frames,
    }

    template = req.template_name or "erc_default"
    try:
        messages, metadata = prompt_registry.render(template, **context)
    except Exception:
        messages, metadata = prompt_registry.render("erc_default", **context)

    if req.soft_label:
        messages[-1]["content"] += "\n\nPlease provide a probability distribution over the emotions."

    return {
        "system": messages[0]["content"],
        "prompt": get_clean_prompt_text(messages[1]["content"]) if len(messages) > 1 else "",
        "messages": messages
    }


@app.get("/results/gather_agents")
async def gather_agents(
    dataset_file: str,
    model: str,
    dialogue_id: str,
    utterance_id: str,
    current_filename: str
):
    results_dir = BASE_DIR / "results"
    if not results_dir.exists():
        return {}

    gathered = {}
    
    def normalize_model(m):
        return m.lower().replace(":", "_").replace("/", "-")
        
    target_model_norm = normalize_model(model)
    target_ds_norm = dataset_file.lower().replace(".json", "")

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json") or filename == current_filename:
            continue
            
        file_path = results_dir / filename
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            meta = data.get("meta", {})
            meta_model = meta.get("model", "")
            meta_ds = meta.get("dataset_file", "")
            
            if not meta_model or not meta_ds:
                continue
                
            if normalize_model(meta_model) != target_model_norm:
                continue
            if meta_ds.lower().replace(".json", "") != target_ds_norm:
                continue
                
            # Check meta to determine agent_mode
            agent_mode = meta.get("agent_mode", "single")
            if agent_mode == "multi":
                continue # Skip multi-agent files (we only gather single-agent inputs)

            template_path = meta.get("template_path", "").lower()
            agent_name = None
            
            if "resolver" in template_path or "resolver" in filename.lower():
                continue # Skip resolver agent files
            elif "multimodal" in template_path or "multimodal" in filename.lower():
                agent_name = "MultimodalFusion"
            elif "vision" in template_path or "image" in template_path or "vision" in filename.lower():
                agent_name = "VisionAgent"
            elif "audio" in template_path or "egemaps" in template_path or "audio" in filename.lower() or "egemaps" in filename.lower():
                agent_name = "AcousticAgent"
            elif "cot" in template_path or "text" in template_path or "default" in template_path or "text" in filename.lower() or "cot" in filename.lower():
                agent_name = "TextAgent"
                
            if not agent_name:
                continue
                
            for res in data.get("results", []):
                res_d_id = res.get("dialogue_id")
                res_u_id = res.get("utterance_id")
                
                if res_d_id == dialogue_id and str(res_u_id) == str(utterance_id):
                    # Find model prediction
                    prediction_text = res.get("prediction_raw", res.get("prediction", ""))
                    
                    gathered[agent_name] = {
                        "prediction": res.get("prediction", ""),
                        "prediction_raw": prediction_text,
                        "prediction_soft": res.get("prediction_soft", {}),
                        "filename": filename,
                        "template": meta.get("template_path", "")
                    }
                    break
        except Exception as e:
            # Skip invalid files silently
            pass
            
    return gathered



# --- Prompt Lab Endpoints ---


@app.get("/prompts")
async def list_prompts():
    prompts_dir = BASE_DIR / "prompts"
    if not prompts_dir.exists():
        return []
    return [f for f in os.listdir(prompts_dir) if f.endswith(".txt")]


@app.get("/prompts/{name}")
async def get_prompt(name: str):
    prompts_dir = BASE_DIR / "prompts"
    file_path = prompts_dir / name
    if not file_path.name.endswith(".txt"):
        file_path = file_path.with_suffix(".txt")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Prompt template not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        return {"name": file_path.name, "content": f.read()}


@app.post("/prompts/{name}")
async def save_prompt(name: str, update: PromptUpdate):
    prompts_dir = BASE_DIR / "prompts"
    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True)

    file_path = prompts_dir / name
    if not file_path.name.endswith(".txt"):
        file_path = file_path.with_suffix(".txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(update.content)

    return {"status": "success", "message": f"Prompt {file_path.name} saved."}


@app.delete("/prompts/{name}")
async def delete_prompt(name: str):
    prompts_dir = BASE_DIR / "prompts"
    file_path = prompts_dir / name
    if not file_path.name.endswith(".txt"):
        file_path = file_path.with_suffix(".txt")

    if file_path.exists():
        os.remove(file_path)
        return {"status": "success", "message": f"Prompt {file_path.name} deleted."}
    raise HTTPException(status_code=404, detail="Prompt template not found.")


@app.get("/injectors")
async def list_injectors():
    return prompt_registry.list_available_injectors()


# --- Multi-Agent Endpoint ---


@app.post("/agent/chat")
async def agent_chat(req: AgentChatRequest):
    """
    Run the Reasoner → Verifier ADK multi-agent pipeline for a single utterance.
    Streams newline-delimited JSON events:
      {"agent": "ReasonerAgent", "start": true}
      {"agent": "ReasonerAgent", "chunk": "..."}
      {"agent": "ReasonerAgent", "done": true}
      {"agent": "VerifierAgent", "start": true}
      {"agent": "VerifierAgent", "chunk": "..."}
      {"agent": "VerifierAgent", "done": true}
      {"final": "joy", "raw_verifier": "Emotion: joy"}
    """
    if not HAS_AGENTS:
        raise HTTPException(status_code=503, detail="agents_logic not available")

    data = load_dataset(req.dataset_name)
    if not data:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    dialogue = next((d for d in data if d["dialogue_id"] == req.dialogue_id), None)
    if not dialogue:
        raise HTTPException(status_code=404, detail="Dialogue not found.")

    utterances = dialogue["utterances"]
    if req.target_index >= len(utterances):
        raise HTTPException(status_code=400, detail="Invalid target index.")

    # Build ERC prompt messages via existing registry
    dataset_lower = req.dataset_name.lower()
    if "iemocap" in dataset_lower:
        ds_key = "iemocap"
        valid_emotions = (
            "neutral, happiness, sadness, anger, fear, frustration, excitement"
        )
    elif "camer" in dataset_lower:
        ds_key = "camer"
        valid_emotions = "neutral, surprise, fear, sadness, joy, disgust, anger"
    else:
        ds_key = "meld"
        valid_emotions = "neutral, surprise, fear, sadness, joy, disgust, anger"

    context = {
        "dataset_name": ds_key,
        "utterances": utterances,
        "target_index": req.target_index,
        "window_size": req.window_size,
        "vision_frames": req.vision_frames,
        "soft_label": req.soft_label,
        "max_tokens": req.max_tokens,
        "reasoning_max_tokens": req.reasoning_max_tokens,
    }
    template = req.template_name or "erc_default"
    try:
        messages, metadata = prompt_registry.render(template, **context)
    except Exception:
        messages, metadata = prompt_registry.render("erc_default", **context)

    async def event_stream():
        async for chunk in run_multiagent_stream(
            messages=messages,
            valid_emotions=valid_emotions,
            model=req.model_id,
            provider=req.provider,
            workflow=req.workflow,
            original_context=context,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        },
    )


# --- Benchmark Run Endpoint ---


@app.post("/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    """
    Streams an automated benchmark run as NDJSON events:
      {"event": "start", "total": N, "meta": {...}}
      {"event": "result", "dialogue_id": ..., "utterance_id": ..., "speaker": ..., "text": ...,
       "ground_truth": ..., "prediction": ..., "correct": bool, "metrics": {...}, "done_count": N}
      {"event": "done", "metrics": {...}, "output_path": "..."}
    """
    data = load_dataset(req.dataset_name)
    if not data:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    ds_key = infer_dataset_key(req.dataset_name)
    labels = EMOTION_LABELS.get(ds_key, EMOTION_LABELS["meld"])
    valid_emotions_str = ", ".join(labels)

    # Select conversations
    n = min(req.n_conversations, len(data))
    if req.selection == "random":
        rng = random.Random(req.seed)
        dialogues = rng.sample(data, n)
        selection_desc = f"{n} random (seed={req.seed})"
    else:
        dialogues = data[:n]
        selection_desc = f"first {n}"

    total_utterances = sum(
        1 if "target_index" in d else len(d["utterances"]) for d in dialogues
    )

    # Resolve output path
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tpl_name = req.template_name.replace(".txt", "")
    safe_model = req.model_id.replace(":", "_").replace("/", "-")
    if req.output_name:
        output_path = results_dir / req.output_name
    else:
        mode_tag = "multi" if req.agent_mode else "single"
        output_path = results_dir / (
            f"bench_{ds_key}_{safe_model}_{tpl_name}_n{n}_{mode_tag}_{ts}.json"
        )

    meta = {
        "dataset_file": req.dataset_name,
        "model": req.model_id,
        "provider": req.provider,
        "template_path": req.template_name,
        "window": req.window_size,
        "temperature": req.temperature,
        "include_thinking": not req.disable_thinking,
        "agent_mode": "multi" if req.agent_mode else "single",
        "soft_label": req.soft_label,
        "n_conversations": n,
        "selection": selection_desc,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    async def benchmark_stream():
        yield json.dumps(
            {"event": "start", "total": total_utterances, "meta": meta}
        ) + "\n"

        results = []
        truths: list = []
        preds: list = []
        done_count = 0

        bridge = InferenceBridge(provider=req.provider)

        for diag in dialogues:
            utterances = diag["utterances"]
            indices = (
                [diag["target_index"]]
                if "target_index" in diag
                else range(len(utterances))
            )

            for utt_idx in indices:
                utt = utterances[utt_idx]
                gt = unify_label(utt["emotion"])

                context = {
                    "dataset_name": ds_key,
                    "utterances": utterances,
                    "target_index": utt_idx,
                    "window_size": req.window_size,
                    "max_tokens": req.max_tokens,
                }
                template = req.template_name
                try:
                    messages, metadata = prompt_registry.render(template, **context)
                except Exception:
                    messages, metadata = prompt_registry.render(
                        "erc_default", **context
                    )

                raw_answer = ""

                if req.agent_mode and HAS_AGENTS:
                    # Multi-agent path — collect final label
                    try:
                        async for line in run_multiagent_stream(
                            messages=messages,
                            valid_emotions=valid_emotions_str,
                            model=req.model_id,
                            provider=req.provider,
                            workflow=req.workflow,
                            original_context=context,
                        ):
                            ev = json.loads(line)
                            if ev.get("event") == "final":
                                raw_answer = ev.get("label", "")
                            # stream agent events to frontend
                            yield json.dumps({"event": "agent_event", **ev}) + "\n"
                    except Exception as e:
                        raw_answer = "error"
                        yield json.dumps(
                            {"event": "agent_error", "error": str(e)}
                        ) + "\n"
                else:
                    # Single-agent path
                    if req.disable_thinking and len(messages) > 1:
                        content = messages[1]["content"]
                        if isinstance(content, str) and not content.startswith("/no_think"):
                            messages[1]["content"] = "/no_think\n" + content

                    # Multimodal handling via request or template metadata
                    media_reqs = metadata.get("media", [])
                    wants_video = "video" in media_reqs
                    wants_image = "image" in media_reqs or "vision" in media_reqs
                    wants_audio = "audio" in media_reqs
                    include_v = req.include_video or wants_video or wants_image
                    include_a = req.include_audio or wants_audio
                    visual_mode = "video" if wants_video else "image"

                    if (include_v or include_a) and req.provider == "openrouter":
                        raw_media_path = utt.get("video_path") or utt.get("audio_path")
                        last_msg = messages[-1]
                        if last_msg["role"] == "user":
                            u_id = utt.get("utterance_id")
                            last_msg["content"] = prepare_multimodal_content(
                                last_msg["content"],
                                raw_media_path,
                                include_v,
                                include_audio=include_a,
                                max_vision_frames=req.vision_frames,
                                visual_mode=visual_mode,
                                utterance_id=u_id,
                            )

                        try:
                            stream_gen = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda m=messages: bridge.chat(
                                    model=req.model_id,
                                    messages=m,
                                    temperature=req.temperature,
                                    max_tokens=req.max_tokens,
                                    stream=True,
                                    include_thinking=not req.disable_thinking,
                                    soft_label=req.soft_label,
                                ),
                            )
                            raw_answer = ""
                            for chunk in stream_gen:
                                raw_answer += chunk
                        except Exception as e:
                            raw_answer = "error"

                    if req.soft_label:
                        pred_soft = parse_soft_prediction(raw_answer, labels)
                        if pred_soft is None:
                            # Retry with temperature 0.0
                            try:
                                stream_gen = (
                                    await asyncio.get_event_loop().run_in_executor(
                                        None,
                                        lambda m=messages: bridge.chat(
                                            model=req.model_id,
                                            messages=m,
                                            temperature=0.0,
                                            max_tokens=req.max_tokens,
                                            stream=True,
                                            include_thinking=not req.disable_thinking,
                                            soft_label=True,
                                        ),
                                    )
                                )
                                raw_answer = ""
                                for chunk in stream_gen:
                                    raw_answer += chunk
                                pred_soft = parse_soft_prediction(raw_answer, labels)
                            except:
                                pred_soft = None

                        if pred_soft is None:
                            # Fallback
                            pred_soft = {
                                l.lower(): (1.0 if l.lower() == "neutral" else 0.0)
                                for l in labels
                            }

                if req.soft_label:
                    norm = get_argmax_label(pred_soft)
                else:
                    norm = normalize_prediction(raw_answer.strip(), labels)

                correct = norm == gt

                jsd = None
                if req.soft_label:
                    gt_soft = utt.get("soft_labels", {})
                    if not gt_soft:
                        gt_soft = {
                            l.lower(): (1.0 if l.lower() == gt.lower() else 0.0)
                            for l in labels
                        }
                    jsd = calculate_js_divergence(pred_soft, gt_soft)

                truths.append(gt)
                preds.append(norm)
                done_count += 1

                result_entry = {
                    "dialogue_id": diag["dialogue_id"],
                    "utterance_id": utt.get("utterance_id", utt_idx),
                    "speaker": utt["speaker"],
                    "text": utt["text"],
                    "ground_truth": gt,
                    "prediction": norm,
                    "prediction_raw": raw_answer.strip(),
                    "mode": "multi" if req.agent_mode else "single",
                }
                if req.soft_label:
                    result_entry["prediction_soft"] = pred_soft
                    result_entry["js_divergence"] = jsd
                    result_entry["ground_truth_soft"] = utt.get("soft_labels", {})

                if "original_dialogue_id" in diag:
                    result_entry["original_dialogue_id"] = diag["original_dialogue_id"]
                results.append(result_entry)

                metrics = compute_metrics_bench(truths, preds, labels, results)

                # Save incrementally
                meta["updated_at"] = datetime.datetime.now().isoformat(
                    timespec="seconds"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"meta": meta, "results": results},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                yield json.dumps(
                    {
                        "event": "result",
                        "dialogue_id": diag["dialogue_id"],
                        "utterance_id": utt.get("utterance_id", utt_idx),
                        "speaker": utt["speaker"],
                        "text": utt["text"],
                        "ground_truth": gt,
                        "prediction": norm,
                        "correct": correct,
                        "metrics": metrics,
                        "done_count": done_count,
                    }
                ) + "\n"

        final_metrics = compute_metrics_bench(truths, preds, labels, results)
        yield json.dumps(
            {
                "event": "done",
                "metrics": final_metrics,
                "output_path": str(output_path.name),
            }
        ) + "\n"

    return StreamingResponse(
        benchmark_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        },
    )


# Serve static frontend UI
app.mount("/", StaticFiles(directory=str(BASE_DIR / "backend" / "static"), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8283)
