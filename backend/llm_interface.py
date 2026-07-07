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
        max_tokens: Optional[int] = None,
        stream: bool = False,
        include_thinking: bool = True,
        soft_label: bool = False,
        max_retries: int = 3,
        reasoning_max_tokens: int = 10000,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Sends a chat completion request with exponential backoff retries.
        """
        import re
        
        attempts = max_retries + 1
        last_error = None
        
        for attempt in range(attempts):
            try:
                if self.provider == "local":
                    if not self.client:
                        raise RuntimeError("Ollama client not initialized.")
                    res = self._ollama_chat(
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
                    res = self._openrouter_chat(
                        model,
                        messages,
                        temperature,
                        max_tokens,
                        stream,
                        include_thinking,
                        soft_label,
                        reasoning_max_tokens=reasoning_max_tokens,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                
                # If non-streaming, strip thinking block if disabled
                if not stream:
                    if not include_thinking:
                        res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
                    return res
                else:
                    # If streaming and thinking is disabled, wrap in our stream stripper
                    if not include_thinking:
                        return self._strip_thinking_stream(res)
                    return res
            except Exception as e:
                last_error = e
                if attempt == attempts - 1:
                    break
                
                sleep_time = 2 ** attempt
                print(f"[RETRY] Chat attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                
        err_msg = f"Error: Request failed after {attempts} attempts. Last error: {str(last_error)}"
        if stream:
            def error_gen():
                yield err_msg
            return error_gen()
        return err_msg

    def _strip_thinking_stream(self, gen: Generator[str, None, None]) -> Generator[str, None, None]:
        """
        Stream wrapper to filter out '<think>...</think>' tags and their content.
        """
        buffer = ""
        in_think = False
        try:
            for chunk in gen:
                buffer += chunk
                while True:
                    if not in_think:
                        idx = buffer.find("<think>")
                        if idx != -1:
                            if idx > 0:
                                yield buffer[:idx]
                            buffer = buffer[idx:]
                            in_think = True
                        else:
                            # Keep the last few characters in case they are a partial '<think>'
                            if len(buffer) > 7:
                                yield buffer[:-7]
                                buffer = buffer[-7:]
                            break
                    else:
                        idx = buffer.find("</think>")
                        if idx != -1:
                            buffer = buffer[idx + 8:]
                            in_think = False
                        else:
                            # Keep the last few characters in case they are a partial '</think>'
                            if len(buffer) > 9:
                                buffer = buffer[-9:]
                            break
            if not in_think and buffer:
                yield buffer
        except Exception as e:
            yield f"Error in stream: {str(e)}"

    def _ollama_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        include_thinking: bool,
        soft_label: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Ollama-native chat implementation. Propagates exceptions to allow retry handling.
        """
        chat_opts = {"temperature": temperature}
        if max_tokens is not None:
            chat_opts["num_predict"] = max_tokens
        json_fmt = "json" if soft_label else None

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
                        b64 = item["image_url"]["url"].split(",")[1]
                        images.append(b64)
                if images:
                    new_msg["images"] = images
            else:
                new_msg["content"] = msg["content"]
            processed_messages.append(new_msg)

        if stream:
            response = self.client.chat(
                model=model,
                messages=processed_messages,
                options=chat_opts,
                format=json_fmt,
                keep_alive=-1,
                stream=True,
                think=include_thinking,
            )

            def stream_generator():
                in_thinking = False
                for chunk in response:
                    thinking = getattr(chunk.message, "thinking", None)
                    content = getattr(chunk.message, "content", None)

                    if thinking and include_thinking:
                        if not in_thinking:
                            yield "<thought>"
                            in_thinking = True
                        yield thinking

                    if content:
                        if in_thinking:
                            yield "</thought>"
                            in_thinking = False
                        yield content

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

    def _openrouter_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        include_thinking: bool,
        soft_label: bool = False,
        reasoning_max_tokens: int = 10000,
    ) -> Union[str, Generator[str, None, None]]:
        """
        OpenRouter chat implementation using OpenAI SDK. Propagates exceptions for retries.
        """
        resp_format = {"type": "json_object"} if soft_label else None

        reasoning_body = {}
        if include_thinking:
            reasoning_body = {"enabled": True, "max_tokens": reasoning_max_tokens}
        else:
            reasoning_body = {"enabled": False}

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "response_format": resp_format,
            "extra_body": {"reasoning": reasoning_body},
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.openai_client.chat.completions.create(**params)

        if stream:
            def stream_generator():
                in_thinking = False
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    content = getattr(delta, "content", None)

                    if reasoning and include_thinking:
                        if not in_thinking:
                            yield "<thought>"
                            in_thinking = True
                        yield reasoning

                    if content:
                        if in_thinking:
                            yield "</thought>"
                            in_thinking = False
                        yield content

                if in_thinking:
                    yield "</thought>"

            return stream_generator()
        else:
            message = response.choices[0].message
            content = getattr(message, "content", "") or ""
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)

            if reasoning and include_thinking:
                return f"<thought>{reasoning}</thought>{content.strip()}"
            return content.strip()


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
