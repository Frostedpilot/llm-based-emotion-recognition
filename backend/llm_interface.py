import os
import time
import base64
import mimetypes
from typing import List, Dict, Optional, Generator, Union, Any
from dotenv import load_dotenv

load_dotenv()
try:
    import ollama
except ImportError:
    print("Warning: 'ollama' library not found. Please run 'pip install ollama'")
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    print("Warning: 'openai' library not found. Please run 'pip install openai'")
    OpenAI = None


def encode_file_to_base64(file_path: str) -> str:
    """
    Encodes a file to a base64 data URL with proper MIME type.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class InferenceBridge:
    """
    Unified interface for local (Ollama) and cloud (OpenRouter) LLM inference.
    """

    def __init__(
        self,
        provider: str = "local",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()

        if self.provider == "local":
            self.base_url = base_url or "http://127.0.0.1:11434"
            self.api_key = api_key or "ollama"
        elif self.provider == "openrouter":
            self.base_url = base_url or "https://openrouter.ai/api/v1"
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OpenRouter API key is required.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize Ollama client for local provider
        if self.provider == "local" and ollama:
            self.client = ollama.Client(host=self.base_url)
        else:
            self.client = None

        # Initialize OpenAI client for OpenRouter provider
        if self.provider == "openrouter" and OpenAI:
            self.openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.openai_client = None

    def preload_model(self, model_id: str):
        """
        Force Ollama to load the model and keep it in memory indefinitely.
        """
        if self.provider != "local":
            print(f"Preloading not supported for provider: {self.provider}")
            return

        if not self.client:
            print("Error: Ollama client not initialized.")
            return

        print(f"Preloading Ollama model: {model_id}...")
        try:
            # Use a 1-token request with keep_alive=-1 to force memory residency
            # Use /no_think directive for Qwen 3.5 models
            self.client.chat(
                model=model_id,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
                keep_alive=-1,
            )
            print(f"Model {model_id} preloaded successfully.")
        except Exception as e:
            print(f"Preloading failed: {e}")

    def unload_model(self, model_id: str):
        """
        Tell Ollama to release the model from memory.
        """
        if self.provider != "local":
            return

        if not self.client:
            return

        print(f"Unloading model: {model_id}...")
        try:
            self.client.chat(
                model=model_id,
                messages=[{"role": "user", "content": "bye"}],
                options={"num_predict": 1},
                keep_alive=0,
            )
        except:
            pass

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 36384,
        stream: bool = False,
        include_thinking: bool = True,
        soft_label: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Sends a chat completion request. Returns string or text generator.
        """
        if self.provider == "local":
            if not self.client:
                raise RuntimeError("Ollama client not initialized.")
            return self._ollama_chat(
                model,
                messages,
                temperature,
                max_tokens,
                stream,
                include_thinking,
                soft_label,
            )
        elif self.provider == "openrouter":
            if not self.openai_client:
                raise RuntimeError("OpenAI client not initialized.")
            return self._openrouter_chat(
                model,
                messages,
                temperature,
                max_tokens,
                stream,
                include_thinking,
                soft_label,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _ollama_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        include_thinking: bool,
        soft_label: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Ollama-native chat implementation using the official SDK.
        """
        # Ensure 'format="json"' is set for soft-label mode
        chat_opts = {"temperature": temperature, "num_predict": max_tokens}
        json_fmt = "json" if soft_label else None

        try:
            # Pre-process messages for Ollama multimodal format
            processed_messages = []
            for msg in messages:
                new_msg = {"role": msg["role"], "content": ""}
                images = []

                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            new_msg["content"] += item["text"]
                        elif item["type"] == "image_url":
                            # Extract base64 skip the data: prefix
                            b64 = item["image_url"]["url"].split(",")[1]
                            images.append(b64)
                    if images:
                        new_msg["images"] = images
                else:
                    new_msg["content"] = msg["content"]
                processed_messages.append(new_msg)

            if stream:

                def stream_generator():
                    # print(f"[DEBUG] Ollama Request: {processed_messages}")
                    response = self.client.chat(
                        model=model,
                        messages=processed_messages,
                        options=chat_opts,
                        format=json_fmt,
                        keep_alive=-1,
                        stream=True,
                        think=include_thinking,
                    )

                    in_thinking = False
                    for chunk in response:
                        # Official SDK returns objects with .message.thinking and .message.content
                        thinking = getattr(chunk.message, "thinking", None)
                        content = getattr(chunk.message, "content", None)

                        if thinking:
                            if not in_thinking:
                                yield "<thought>"
                                in_thinking = True
                            yield thinking

                        if content:
                            if in_thinking:
                                yield "</thought>"
                                in_thinking = False
                            yield content

                    # Ensure tag is closed if stream ends abruptly
                    if in_thinking:
                        yield "</thought>"

                return stream_generator()
            else:
                response = self.client.chat(
                    model=model,
                    messages=processed_messages,
                    options=chat_opts,
                    format=json_fmt,
                    keep_alive=-1,
                    think=include_thinking,
                )

                thinking = getattr(response.message, "thinking", None)
                content = getattr(response.message, "content", "")

                if thinking and include_thinking:
                    return f"<thought>{thinking}</thought>{content.strip()}"
                return content.strip()

        except Exception as e:
            return f"Error: {str(e)}"

    def _openrouter_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        include_thinking: bool,
        soft_label: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        OpenRouter chat implementation using OpenAI SDK.
        """
        try:
            resp_format = {"type": "json_object"} if soft_label else None

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                response_format=resp_format,
                extra_body={"reasoning": {"enabled": include_thinking}},
            )

            if stream:

                def stream_generator():
                    try:
                        for chunk in response:
                            if (
                                chunk.choices
                                and chunk.choices[0].delta.content is not None
                            ):
                                content = chunk.choices[0].delta.content
                                if content:
                                    yield content
                    except Exception as e:
                        yield f"Error: {str(e)}"

                return stream_generator()
            else:
                return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Test streaming with thinking tokens
    bridge = InferenceBridge(provider="local")
    # bridge.preload_model("qwen3.5:4b")
    print("Streaming response with thinking tags:")

    # Using a model that supports thinking
    model_name = "qwen3.5:4b"

    for chunk in bridge.chat(
        model_name,
        [{"role": "user", "content": "What is 17 * 23? Show your work."}],
        stream=True,
        include_thinking=True,
    ):
        print(chunk, end="", flush=True)
    print("\nDone.")
