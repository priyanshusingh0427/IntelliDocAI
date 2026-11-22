"""Local LLM providers for answer generation.

- "transformers": runs a Hugging Face chat model inside this process on the CPU
  (default Qwen/Qwen2.5-0.5B-Instruct, ~1 GB download on first use). Works with no
  extra software, but a 0.5B model gives short, simple answers.
- "ollama": calls a local Ollama server (https://ollama.com). Better quality with
  bigger models such as llama3.2:3b, and can use a GPU. Needs Ollama installed and
  `ollama pull <model>` run once.
- "extractive": no LLM. The RAG pipeline returns the best passages with citations.
"""
import logging
import threading
from functools import lru_cache
from typing import Protocol

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The configured LLM cannot be reached/loaded. Mapped to HTTP 503."""


class LLM(Protocol):
    model_name: str
    is_generative: bool

    def generate(self, system: str, user: str) -> str: ...

    def check(self) -> tuple[bool, str]:
        """Readiness probe: (ready, human-readable detail). Must be fast (no generation)."""
        ...


class TransformersLLM:
    is_generative = True

    def __init__(
        self, model_name: str, cache_dir: str, max_new_tokens: int, temperature: float
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()  # one generation at a time on the CPU

    def _load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        import torch
                        from transformers import AutoModelForCausalLM, AutoTokenizer

                        logger.info("Loading LLM %s (first time downloads it) ...", self.model_name)
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name, cache_dir=self._cache_dir
                        )
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_name, cache_dir=self._cache_dir, torch_dtype=torch.float32
                        )
                        self._model.eval()
                        logger.info("LLM loaded")
                    except Exception as exc:  # noqa: BLE001
                        raise LLMUnavailable(f"Could not load LLM {self.model_name}: {exc}") from exc
        return self._tokenizer, self._model

    def check(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, f"{self.model_name} loaded"
        return True, f"{self.model_name} not loaded yet (loads on the first question)"

    def generate(self, system: str, user: str) -> str:
        import torch

        tokenizer, model = self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        sampling = self._temperature > 0
        with self._generate_lock, torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=sampling,
                temperature=self._temperature if sampling else None,
                top_p=0.9 if sampling else None,
                top_k=None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OllamaLLM:
    is_generative = True

    def __init__(
        self,
        base_url: str,
        model: str,
        max_new_tokens: int,
        temperature: float,
        timeout: int,
        num_ctx: int = 4096,
    ) -> None:
        self.model_name = f"ollama/{model}"
        self._model = model
        self._url = base_url.rstrip("/")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._num_ctx = num_ctx

    def check(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self._url}/api/tags", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return False, f"Ollama is not reachable at {self._url} ({exc.__class__.__name__})"
        names = {m.get("name") for m in response.json().get("models", [])}
        wanted = self._model if ":" in self._model else f"{self._model}:latest"
        if wanted not in names:
            return False, f"Ollama is running but model {self._model!r} is missing. Run: ollama pull {self._model}"
        return True, f"Ollama ready with {self._model}"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_new_tokens,
                # Ollama silently truncates prompts longer than its context window.
                "num_ctx": self._num_ctx,
            },
        }
        try:
            response = httpx.post(f"{self._url}/api/chat", json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(
                f"Ollama is not reachable at {self._url}. Install/start Ollama, or set "
                f"LLM_PROVIDER=transformers. ({exc})"
            ) from exc
        if response.status_code == 404:
            raise LLMUnavailable(
                f"Ollama model {self._model!r} not found. Run: ollama pull {self._model}"
            )
        if response.status_code >= 400:
            raise LLMUnavailable(f"Ollama error {response.status_code}: {response.text[:200]}")
        return response.json().get("message", {}).get("content", "").strip()


class ExtractiveLLM:
    """Marker provider: the pipeline answers with the retrieved passages, no generation."""

    model_name = "extractive"
    is_generative = False

    def check(self) -> tuple[bool, str]:
        return True, "no LLM (answers are the best passages)"

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - never called
        raise NotImplementedError


def build_llm(settings: Settings) -> LLM:
    provider = settings.LLM_PROVIDER.lower()
    if provider == "transformers":
        return TransformersLLM(
            settings.LLM_MODEL,
            cache_dir=str(settings.embedding_cache_path),
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
    if provider == "ollama":
        return OllamaLLM(
            settings.OLLAMA_URL,
            settings.OLLAMA_MODEL,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            num_ctx=settings.OLLAMA_NUM_CTX,
        )
    if provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


@lru_cache
def get_llm() -> LLM:
    """FastAPI dependency: one shared LLM (model loaded once)."""
    return build_llm(get_settings())
