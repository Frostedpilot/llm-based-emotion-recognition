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
    soft_label: bool = False,
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
                soft_label=soft_label,
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


async def run_multimodal_resolver_internal(
    messages,
    valid_emotions: str,
    model: str,
    provider: str,
    bridge: InferenceBridge,
    original_context: dict,
    workflow_name: str,
) -> AsyncGenerator[str, None]:
    # Extract config
    simulated_results = original_context.get("simulated_results")
    utterances = original_context.get("utterances", [])
    target_idx = original_context.get("target_index")
    window_size = original_context.get("window_size", 5)
    vision_frames = original_context.get("vision_frames", 3)
    dataset_name = original_context.get("dataset_name", "meld")

    INTERNAL_WORKFLOW_TEMPLATES = {
        "tva_theory_soft": {
            "text": "erc_cot_soft_label.txt",
            "vision_default": "vision_only_soft_label_cot.txt",
            "vision_iemocap": "vision_only_soft_label_cot_iemocap.txt",
            "acoustic": "audio_only_soft_label_cot.txt",
            "resolver": "agent_resolver_theory_soft.txt",
        },
        "tva_theory": {
            "text": "erc_cot.txt",
            "vision_default": "vision_only_cot.txt",
            "vision_iemocap": "vision_only_cot_iemocap.txt",
            "acoustic": "audio_only.txt",
            "resolver": "agent_resolver_theory.txt",
        },
        "modality_tva": {
            "text": "agent_text.txt",
            "vision_default": "agent_vision.txt",
            "vision_iemocap": "agent_vision.txt",
            "acoustic": "agent_egemaps.txt",
            "resolver": "agent_resolver.txt",
        },
        "modality_tv": {
            "text": "agent_text.txt",
            "vision_default": "agent_vision.txt",
            "vision_iemocap": "agent_vision.txt",
            "resolver": "agent_resolver.txt",
        },
        "modality_ta": {
            "text": "agent_text.txt",
            "acoustic": "agent_egemaps.txt",
            "resolver": "agent_resolver.txt",
        }
    }

    workflow_key = workflow_name if workflow_name in INTERNAL_WORKFLOW_TEMPLATES else "tva_theory_soft"
    tpls = INTERNAL_WORKFLOW_TEMPLATES[workflow_key]

    ctx = original_context.copy()

    # Get lookup key for simulated results
    lookup_key = None
    if simulated_results and target_idx is not None and utterances:
        d_id = original_context.get("dialogue_id")
        u_id = utterances[target_idx].get("utterance_id", target_idx)
        try:
            u_id_parsed = int(u_id)
        except (ValueError, TypeError):
            u_id_parsed = str(u_id)
        lookup_key = (str(d_id), u_id_parsed)

    sim_data = simulated_results.get(lookup_key, {}) if (simulated_results and lookup_key) else {}

    # 1. Text Agent
    if tpls.get("text"):
        if "text_output" in sim_data:
            ctx["text_output"] = sim_data["text_output"]
            yield json.dumps({
                "agent": "TextAgent",
                "event": "done",
                "full": sim_data["text_output"]
            }) + "\n"
        else:
            text_out = ""
            is_soft = "soft" in tpls["text"]
            async for ev in run_agent_step("TextAgent", tpls["text"], ctx, model, provider, bridge, soft_label=is_soft):
                yield ev
                try:
                    data = json.loads(ev)
                    if data.get("event") == "done":
                        text_out = data["full"]
                except:
                    pass
            ctx["text_output"] = text_out

    # 2. Vision Agent
    if tpls.get("vision_default"):
        if "vision_output" in sim_data:
            ctx["vision_output"] = sim_data["vision_output"]
            yield json.dumps({
                "agent": "VisionAgent",
                "event": "done",
                "full": sim_data["vision_output"]
            }) + "\n"
        else:
            v_tpl = tpls["vision_iemocap"] if "iemocap" in dataset_name.lower() else tpls["vision_default"]
            vision_out = ""
            is_soft = "soft" in v_tpl
            async for ev in run_agent_step("VisionAgent", v_tpl, ctx, model, provider, bridge, soft_label=is_soft):
                yield ev
                try:
                    data = json.loads(ev)
                    if data.get("event") == "done":
                        vision_out = data["full"]
                except:
                    pass
            ctx["vision_output"] = vision_out

    # 3. Acoustic Agent
    if tpls.get("acoustic"):
        sim_acoustic = sim_data.get("audio_output") or sim_data.get("acoustic_output") or sim_data.get("egemaps_output")
        if sim_acoustic:
            ctx["audio_output"] = ctx["acoustic_output"] = ctx["egemaps_output"] = sim_acoustic
            yield json.dumps({
                "agent": "AcousticAgent",
                "event": "done",
                "full": sim_acoustic
            }) + "\n"
        else:
            acoustic_out = ""
            is_soft = "soft" in tpls["acoustic"]
            async for ev in run_agent_step("AcousticAgent", tpls["acoustic"], ctx, model, provider, bridge, soft_label=is_soft):
                yield ev
                try:
                    data = json.loads(ev)
                    if data.get("event") == "done":
                        acoustic_out = data["full"]
                except:
                    pass
            ctx["audio_output"] = ctx["acoustic_output"] = ctx["egemaps_output"] = acoustic_out

    # 4. Resolver Agent
    resolver_out = ""
    resolver_soft = original_context.get("soft_label", False)
    async for ev in run_agent_step("ResolverAgent", tpls["resolver"], ctx, model, provider, bridge, soft_label=resolver_soft):
        yield ev
        try:
            data = json.loads(ev)
            if data.get("event") == "done":
                resolver_out = data["full"]
        except:
            pass

    yield format_final_event(resolver_out, valid_emotions)


async def multimodal_tv_workflow(messages, valid_emotions, model, provider, bridge, original_context):
    async for ev in run_multimodal_resolver_internal(messages, valid_emotions, model, provider, bridge, original_context, "modality_tv"):
        yield ev


async def multimodal_ta_workflow(messages, valid_emotions, model, provider, bridge, original_context):
    async for ev in run_multimodal_resolver_internal(messages, valid_emotions, model, provider, bridge, original_context, "modality_ta"):
        yield ev


async def multimodal_tva_workflow(messages, valid_emotions, model, provider, bridge, original_context):
    async for ev in run_multimodal_resolver_internal(messages, valid_emotions, model, provider, bridge, original_context, "modality_tva"):
        yield ev


async def multimodal_tva_theory_workflow(messages, valid_emotions, model, provider, bridge, original_context):
    async for ev in run_multimodal_resolver_internal(messages, valid_emotions, model, provider, bridge, original_context, "tva_theory"):
        yield ev


async def multimodal_tva_theory_soft_workflow(messages, valid_emotions, model, provider, bridge, original_context):
    async for ev in run_multimodal_resolver_internal(messages, valid_emotions, model, provider, bridge, original_context, "tva_theory_soft"):
        yield ev


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
