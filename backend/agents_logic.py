"""
backend/agents_logic.py
=======================
Modular multi-agent orchestration for Emotion Recognition in Conversation (ERC).

This version uses the PromptRegistry to load agent instructions from the
prompts/ directory, allowing for easy editing and dynamic injection.
"""

import os
import re
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ── Registry & Bridge Imports ────────────────────────────────────────────────
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from llm_interface import InferenceBridge
from prompt_templates import get_registry
from media_utils import prepare_multimodal_content

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PRESETS: Dict[str, str] = {
    "@preset/qwen-3-5-9b": "@preset/qwen-3-5-9b",
}

# ── Core Agent Step Runner ───────────────────────────────────────────────────


async def run_agent_step(
    agent_name: str,
    template_name: str,
    context: Dict[str, Any],
    model: str,
    provider: str,
    bridge: InferenceBridge,
) -> AsyncGenerator[str, None]:
    """
    Runs a single agent step by:
    1. Rendering the template from the registry with the provided context.
    2. Streaming the response and yielding NDJSON events.
    """
    registry = get_registry()

    # ── Rendering & Prompt Injection ─────────────────────────────────────────
    # The registry automatically calls injectors for things like {context_and_target}
    # and fills placeholders like {reasoner_output} from the context dict.
    messages, metadata = registry.render(template_name, **context)

    # ── Media Handling from Template ─────────────────────────────────────────
    # If the template specifies media requirements, we attach them here.
    media_reqs = metadata.get("media", [])
    wants_video = "video" in media_reqs
    wants_image = "image" in media_reqs or "vision" in media_reqs
    wants_audio = "audio" in media_reqs
    include_v = wants_video or wants_image
    include_a = wants_audio
    visual_mode = "video" if wants_video else "image"

    if (
        (include_v or include_a)
        and context.get("utterances")
        and context.get("target_index") is not None
    ):
        utts = context["utterances"]
        utt = utts[context["target_index"]]
        raw_media_path = utt.get("video_path") or utt.get("audio_path")

        # Transform the last user message content into a multimodal list
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            u_id = utt.get("utterance_id")
            last_msg["content"] = prepare_multimodal_content(
                last_msg["content"],
                raw_media_path,
                include_v,
                include_audio=include_a,
                max_vision_frames=context.get("vision_frames", 3),
                visual_mode=visual_mode,
                utterance_id=u_id,
            )

    yield json.dumps(
        {"agent": agent_name, "event": "prompt", "messages": messages}
    ) + "\n"

    full_response = ""
    try:
        # bridge.chat is likely synchronous/blocking in some implementations,
        # so we run it in an executor to avoid blocking the async loop.
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: bridge.chat(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                stream=True,
            ),
        )

        if isinstance(stream, str) and stream.startswith("Error:"):
            yield json.dumps(
                {"agent": agent_name, "event": "error", "message": stream}
            ) + "\n"
            return

        for chunk in stream:
            full_response += chunk
            yield json.dumps(
                {"agent": agent_name, "event": "chunk", "text": chunk}
            ) + "\n"

        yield json.dumps(
            {"agent": agent_name, "event": "done", "full": full_response.strip()}
        ) + "\n"

    except Exception as e:
        yield json.dumps(
            {"agent": agent_name, "event": "error", "message": str(e)}
        ) + "\n"


# ── Workflow 1: Reasoner → Verifier ──────────────────────────────────────────


async def reasoner_verifier_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """
    Standard handoff:
    1. Reasoner (generates CoT)
    2. Verifier (validates label against global emotion list)
    """
    # Context contains 'utterances', 'target_index', 'window_size', 'dataset_name'
    ctx = original_context.copy()

    # 1. Reasoner
    reasoner_out = ""
    async for ev in run_agent_step(
        "ReasonerAgent", "agent_reasoner", ctx, model, provider, bridge
    ):
        yield ev
        data = json.loads(ev)
        if data.get("event") == "done":
            reasoner_out = data["full"]

    # 2. Verifier
    ctx["reasoner_output"] = reasoner_out
    verifier_out = ""
    async for ev in run_agent_step(
        "VerifierAgent", "agent_verifier", ctx, model, provider, bridge
    ):
        yield ev
        data = json.loads(ev)
        if data.get("event") == "done":
            verifier_out = data["full"]

    yield format_final_event(verifier_out, valid_emotions)


# ── Workflow 2: Self-Correction ──────────────────────────────────────────────


async def self_correction_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """
    Extended handoff: Reasoner → Critic → Finalizer
    """
    ctx = original_context.copy()

    # 1. Reasoner
    reasoner_out = ""
    async for ev in run_agent_step(
        "ReasonerAgent", "agent_reasoner", ctx, model, provider, bridge
    ):
        yield ev
        if json.loads(ev).get("event") == "done":
            reasoner_out = json.loads(ev)["full"]

    # 2. Critic
    ctx["reasoner_output"] = reasoner_out
    critic_out = ""
    async for ev in run_agent_step(
        "CriticAgent", "agent_critic", ctx, model, provider, bridge
    ):
        yield ev
        if json.loads(ev).get("event") == "done":
            critic_out = json.loads(ev)["full"]

    # 3. Finalizer
    ctx["critic_output"] = critic_out
    # Note: ctx already has reasoner_output
    finalizer_out = ""
    async for ev in run_agent_step(
        "FinalizerAgent", "agent_finalizer", ctx, model, provider, bridge
    ):
        yield ev
        if json.loads(ev).get("event") == "done":
            finalizer_out = json.loads(ev)["full"]

    yield format_final_event(finalizer_out, valid_emotions)


# ── Shared Response Normalization ────────────────────────────────────────────


def format_final_event(text, valid_emotions):
    try:
        from utils import normalize_prediction, parse_soft_prediction, get_argmax_label
    except ImportError:
        from backend.utils import normalize_prediction, parse_soft_prediction, get_argmax_label
    valid_list = [e.strip() for e in valid_emotions.split(",")]
    
    # Try parsing soft labels first if it is a soft verifier output
    soft_labels = parse_soft_prediction(text, valid_list)
    if soft_labels:
        label = get_argmax_label(soft_labels)
        return (
            json.dumps({
                "event": "final",
                "label": label,
                "raw_verifier": text.strip(),
                "soft_labels": soft_labels
            })
            + "\n"
        )

    # Extract "Emotion: <label>" using regex
    match = re.search(r"(?i)Emotion:\s*[*_`\"']*([a-zA-Z]+)[*_`\"']*", text)
    label = ""
    if match:
        candidate = match.group(1).lower()
        label = next((l for l in valid_list if l.lower() == candidate), "")

    if not label:
        label = normalize_prediction(text, valid_list)

    return (
        json.dumps({"event": "final", "label": label, "raw_verifier": text.strip()})
        + "\n"
    )


# ── New Modality-Specific Workflows ──────────────────────────────────────────


async def multimodal_tv_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """Workflow: Text Agent + Vision Agent -> Resolver"""
    ctx = original_context.copy()

    # 1. Text Agent
    try:
        async for ev in run_agent_step(
            "TextAgent", "agent_text", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["text_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["text_output"] = (
                    f"[ERROR]: Text analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["text_output"] = f"[ERROR]: Text analysis crashed: {str(e)}"

    # 2. Vision Agent
    try:
        async for ev in run_agent_step(
            "VisionAgent", "agent_vision", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["vision_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["vision_output"] = (
                    f"[ERROR]: Vision analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["vision_output"] = f"[ERROR]: Vision analysis crashed: {str(e)}"

    # 3. Resolver
    resolver_out = ""
    async for ev in run_agent_step(
        "ResolverAgent", "agent_resolver", ctx, model, provider, bridge
    ):
        yield ev
        try:
            data = json.loads(ev)
            if data.get("event") == "done":
                resolver_out = data.get("full", "")
        except:
            continue

    yield format_final_event(resolver_out, valid_emotions)


async def multimodal_ta_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """Workflow: Text Agent + Acoustic Agent -> Resolver"""
    ctx = original_context.copy()

    # 1. Text Agent
    try:
        async for ev in run_agent_step(
            "TextAgent", "agent_text", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["text_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["text_output"] = (
                    f"[ERROR]: Text analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["text_output"] = f"[ERROR]: Text analysis crashed: {str(e)}"

    # 2. Acoustic Agent
    try:
        async for ev in run_agent_step(
            "AcousticAgent", "agent_egemaps", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["egemaps_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["egemaps_output"] = (
                    f"[ERROR]: Acoustic analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["egemaps_output"] = f"[ERROR]: Acoustic analysis crashed: {str(e)}"

    # 3. Resolver
    resolver_out = ""
    async for ev in run_agent_step(
        "ResolverAgent", "agent_resolver", ctx, model, provider, bridge
    ):
        yield ev
        try:
            data = json.loads(ev)
            if data.get("event") == "done":
                resolver_out = data.get("full", "")
        except:
            continue

    yield format_final_event(resolver_out, valid_emotions)


async def multimodal_tva_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """Workflow: Text Agent + Vision Agent + Acoustic Agent -> Resolver"""
    ctx = original_context.copy()

    # 1. Text Agent
    try:
        async for ev in run_agent_step(
            "TextAgent", "agent_text", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["text_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["text_output"] = (
                    f"[ERROR]: Text analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["text_output"] = f"[ERROR]: Text analysis crashed: {str(e)}"

    # 2. Vision Agent
    try:
        async for ev in run_agent_step(
            "VisionAgent", "agent_vision", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["vision_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["vision_output"] = (
                    f"[ERROR]: Vision analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["vision_output"] = f"[ERROR]: Vision analysis crashed: {str(e)}"

    # 3. Acoustic Agent
    try:
        async for ev in run_agent_step(
            "AcousticAgent", "agent_egemaps", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["egemaps_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["egemaps_output"] = (
                    f"[ERROR]: Acoustic analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["egemaps_output"] = f"[ERROR]: Acoustic analysis crashed: {str(e)}"

    # 4. Resolver
    resolver_out = ""
    async for ev in run_agent_step(
        "ResolverAgent", "agent_resolver", ctx, model, provider, bridge
    ):
        yield ev
        try:
            data = json.loads(ev)
            if data.get("event") == "done":
                resolver_out = data.get("full", "")
        except:
            continue

    yield format_final_event(resolver_out, valid_emotions)


async def multimodal_tva_theory_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """
    Workflow: Text Agent (erc_cot) + Vision Agent (vision_only_cot) + Acoustic Agent (audio_only) -> Resolver (agent_resolver_theory)
    """
    ctx = original_context.copy()

    # 1. Text Agent (erc_cot)
    try:
        async for ev in run_agent_step(
            "TextAgent", "erc_cot.txt", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["text_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["text_output"] = (
                    f"[ERROR]: Text analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["text_output"] = f"[ERROR]: Text analysis crashed: {str(e)}"

    # 2. Vision Agent (vision_only_cot)
    try:
        async for ev in run_agent_step(
            "VisionAgent", "vision_only_cot.txt", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["vision_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["vision_output"] = (
                    f"[ERROR]: Vision analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["vision_output"] = f"[ERROR]: Vision analysis crashed: {str(e)}"

    # 3. Acoustic Agent (audio_only)
    try:
        async for ev in run_agent_step(
            "AcousticAgent", "audio_only.txt", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["acoustic_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["acoustic_output"] = (
                    f"[ERROR]: Acoustic analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["acoustic_output"] = f"[ERROR]: Acoustic analysis crashed: {str(e)}"

    # 4. Resolver (agent_resolver_theory)
    resolver_out = ""
    async for ev in run_agent_step(
        "ResolverAgent", "agent_resolver_theory.txt", ctx, model, provider, bridge
    ):
        yield ev
        try:
            data = json.loads(ev)
            if data.get("event") == "done":
                resolver_out = data.get("full", "")
        except:
            continue

    yield format_final_event(resolver_out, valid_emotions)


async def multimodal_tva_theory_soft_workflow(
    messages, valid_emotions, model, provider, bridge, original_context
):
    """
    Workflow: Text Agent (erc_cot_soft_label) + Vision Agent (vision_only_soft_label_cot) + Acoustic Agent (audio_only_soft_label_cot) -> Resolver (agent_resolver_theory_soft)
    """
    ctx = original_context.copy()

    # 1. Text Agent (erc_cot_soft_label)
    try:
        async for ev in run_agent_step(
            "TextAgent", "erc_cot_soft_label.txt", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["text_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["text_output"] = (
                    f"[ERROR]: Text analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["text_output"] = f"[ERROR]: Text analysis crashed: {str(e)}"

    # 2. Vision Agent (vision_only_soft_label_cot)
    try:
        async for ev in run_agent_step(
            "VisionAgent", "vision_only_soft_label_cot.txt", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["vision_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["vision_output"] = (
                    f"[ERROR]: Vision analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["vision_output"] = f"[ERROR]: Vision analysis crashed: {str(e)}"

    # 3. Acoustic Agent (audio_only_soft_label_cot)
    try:
        async for ev in run_agent_step(
            "AcousticAgent", "audio_only_soft_label_cot.txt", ctx, model, provider, bridge
        ):
            yield ev
            data = json.loads(ev)
            if data.get("event") == "done":
                ctx["acoustic_output"] = data.get("full", "")
            elif data.get("event") == "error":
                ctx["acoustic_output"] = (
                    f"[ERROR]: Acoustic analysis failed: {data.get('message')}"
                )
    except Exception as e:
        ctx["acoustic_output"] = f"[ERROR]: Acoustic analysis crashed: {str(e)}"

    # 4. Resolver (agent_resolver_theory_soft)
    resolver_out = ""
    async for ev in run_agent_step(
        "ResolverAgent", "agent_resolver_theory_soft.txt", ctx, model, provider, bridge
    ):
        yield ev
        try:
            data = json.loads(ev)
            if data.get("event") == "done":
                resolver_out = data.get("full", "")
        except:
            continue

    yield format_final_event(resolver_out, valid_emotions)


# ── Module API ───────────────────────────────────────────────────────────────

WORKFLOW_REGISTRY: Dict[str, Callable] = {
    "reasoner_verifier": reasoner_verifier_workflow,
    "self_correction": self_correction_workflow,
    "modality_tv": multimodal_tv_workflow,
    "modality_ta": multimodal_ta_workflow,
    "modality_tva": multimodal_tva_workflow,
    "tva_theory": multimodal_tva_theory_workflow,
    "tva_theory_soft": multimodal_tva_theory_soft_workflow,
}


async def run_multiagent_stream(
    messages: List[Dict[str, str]],
    valid_emotions: str,
    model: str,
    provider: str,
    workflow: str = "reasoner_verifier",
    original_context: Dict[str, Any] = None,
) -> AsyncGenerator[str, None]:
    """The entry point for the FastAPI server/CLI."""
    bridge = InferenceBridge(provider=provider)
    wf_func = WORKFLOW_REGISTRY.get(workflow, reasoner_verifier_workflow)

    # We pass 'original_context' which contains utterances, target_index, etc.
    # so the injectors like context_and_target can work inside the agent steps.
    async for event in wf_func(
        messages, valid_emotions, model, provider, bridge, original_context or {}
    ):
        yield event


def run_multiagent_sync(
    messages,
    valid_emotions,
    model,
    provider,
    workflow="reasoner_verifier",
    original_context=None,
):
    """Sync wrapper for CLI batch processing."""
    final = ""
    loop = asyncio.new_event_loop()

    async def _collect():
        nonlocal final
        async for line in run_multiagent_stream(
            messages, valid_emotions, model, provider, workflow, original_context
        ):
            ev = json.loads(line)
            if ev.get("event") == "final":
                final = ev["label"]

    loop.run_until_complete(_collect())
    loop.close()
    return final
