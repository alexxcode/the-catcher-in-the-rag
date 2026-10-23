from __future__ import annotations

import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from catcher_in_the_rag.providers.base import EmbeddingProvider, LLMProvider
from catcher_in_the_rag.providers.tracing import get_tracer

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """Real GCP generative AI provider using Vertex AI (ADC credentials, no API keys).

    Adheres to Non-negotiable Principle #7 (ADC auth) and #8 (provider-agnostic core).
    Model versions and retirement dates must be verified against:
    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning
    """

    provider_type: str = "vertex_ai"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> None:
        self.config = config or {}
        self.project = os.getenv(
            "GOOGLE_CLOUD_PROJECT", self.config.get("project", "the-catcher-in-the-rag")
        )
        ext_cfg = self.config.get("extraction_llm", {})
        # Configured model or GA default candidate
        model_candidate = model_id or ext_cfg.get("model_id")
        if not model_candidate or "TODO" in str(model_candidate):
            model_candidate = "gemini-3.5-flash"
        self.model_id = model_candidate
        self.temperature = float(ext_cfg.get("temperature", 0.0))

        # Model location routing: Gemini 3.5 & 3.8 Flash GA endpoints are hosted in global / multi-region
        default_region = self.config.get("region", "us-central1")
        if self.model_id in ("gemini-3.5-flash", "gemini-3.8-flash") and default_region == "us-central1" and not os.getenv("GOOGLE_CLOUD_REGION"):
            default_region = "global"
        self.region = os.getenv("GOOGLE_CLOUD_REGION", default_region)

        self._client: Any = None
        self._genai_client: Any = None
        self._use_genai: bool = False

    def _ensure_init(self) -> None:
        if self._client is not None or self._genai_client is not None:
            return

        # 1. Try modern Google Gen AI SDK first (replaces deprecated vertexai.generative_models)
        try:
            from google import genai

            self._genai_client = genai.Client(
                vertexai=True, project=self.project, location=self.region
            )
            self._use_genai = True
            return
        except Exception:
            self._use_genai = False

        # 2. Fall back to vertexai with deprecation warning suppressed
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
                import vertexai
                from vertexai.generative_models import GenerativeModel

                vertexai.init(project=self.project, location=self.region)
                self._client = GenerativeModel(self.model_id)
        except ImportError as err:
            msg = (
                "google-genai or google-cloud-aiplatform is required to run GeminiProvider. "
                "Ensure google-genai is installed and ADC is configured "
                "via `gcloud auth application-default login`, or use LLM_PROVIDER=stub for offline mode."
            )
            raise ImportError(msg) from err

    def _call_with_retry(
        self,
        call_fn: Any,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> Any:
        """Execute Vertex AI generation with bounded exponential backoff on transient 429 / 503 errors."""
        for attempt in range(max_retries):
            try:
                return call_fn()
            except Exception as err:
                err_str = str(err).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "resource exhausted" in err_str
                    or "resourceexhausted" in err_str
                    or "quota" in err_str
                )
                is_transient = (
                    "503" in err_str
                    or "service unavailable" in err_str
                    or "timeout" in err_str
                    or "deadline" in err_str
                )
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Vertex AI LLM call failed after {attempt + 1} attempt(s): {err}"
                ) from err

    def extract_structured(
        self,
        text: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Extract structured data conforming to a Pydantic schema using Gemini structured outputs."""
        self._ensure_init()

        prompt = (
            f"{system_prompt or 'Extract the structured document fields accurately from the input text.'}\n\n"
            f"Input text:\n{text}\n\n"
            f"Respond strictly in valid JSON adhering to schema: {schema.model_json_schema()}"
        )

        def _invoke_extract() -> Any:
            if self._use_genai:
                return self._genai_client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config={
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    },
                )
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
                from vertexai.generative_models import GenerationConfig

                gen_config = GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                )
                return self._client.generate_content(prompt, generation_config=gen_config)

        t0 = time.perf_counter()
        response = self._call_with_retry(_invoke_extract)
        raw_json = (response.text or "").strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract usage metadata if available
        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total": getattr(response.usage_metadata, "total_token_count", 0),
            }

        tracer = get_tracer()
        tracer.trace_generation(
            name=f"gemini_extract_{schema.__name__}",
            model=self.model_id,
            prompt=prompt,
            completion=raw_json,
            latency_ms=elapsed_ms,
            usage=usage,
        )

        if raw_json.startswith("```json"):
            raw_json = raw_json.removeprefix("```json").removesuffix("```").strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json.removeprefix("```").removesuffix("```").strip()
        return schema.model_validate_json(raw_json)

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text completion from Gemini."""
        self._ensure_init()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        def _invoke_generate() -> Any:
            if self._use_genai:
                return self._genai_client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config={"temperature": self.temperature},
                )
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
                from vertexai.generative_models import GenerationConfig

                gen_config = GenerationConfig(temperature=self.temperature)
                return self._client.generate_content(full_prompt, generation_config=gen_config)

        t0 = time.perf_counter()
        response = self._call_with_retry(_invoke_generate)
        res_text = response.text or ""
        elapsed_ms = (time.perf_counter() - t0) * 1000

        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total": getattr(response.usage_metadata, "total_token_count", 0),
            }

        tracer = get_tracer()
        tracer.trace_generation(
            name="gemini_generate_text",
            model=self.model_id,
            prompt=full_prompt,
            completion=res_text,
            latency_ms=elapsed_ms,
            usage=usage,
        )

        return res_text

    def check_model_health(self) -> dict[str, Any]:
        """Perform a startup health-check against Vertex AI to detect model retirement/deprecation.

        Fails fast with a clear explanation if the pinned model has been retired or is unavailable.
        """
        self._ensure_init()
        t0 = time.perf_counter()
        try:
            if self._use_genai and self._genai_client is not None:
                self._genai_client.models.generate_content(
                    model=self.model_id,
                    contents="ping",
                    config={"max_output_tokens": 1, "temperature": 0.0},
                )
            else:
                self._client.generate_content(
                    "ping",
                    generation_config={"max_output_tokens": 1, "temperature": 0.0},
                )
            dt_ms = (time.perf_counter() - t0) * 1000
            return {
                "status": "healthy",
                "model_id": self.model_id,
                "provider": "vertex_ai",
                "latency_ms": round(dt_ms, 2),
            }
        except Exception as err:
            err_msg = str(err).lower()
            if any(
                term in err_msg
                for term in (
                    "not found",
                    "404",
                    "410",
                    "deprecated",
                    "shutdown",
                    "retired",
                    "invalid model",
                    "does not exist",
                )
            ):
                fatal_msg = (
                    f"FATAL: Vertex AI model '{self.model_id}' is unavailable or has reached retirement! "
                    f"Google Vertex AI shut down this model version. "
                    f"Please update config/models.yaml or set VERTEX_MODEL_ID to an active GA model. "
                    f"Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning. "
                    f"Underlying error: {err}"
                )
                raise RuntimeError(fatal_msg) from err
            raise RuntimeError(
                f"Vertex AI health check failed for model '{self.model_id}': {err}"
            ) from err


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Dense embedding provider using Vertex AI text-embedding models."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.project = os.getenv(
            "GOOGLE_CLOUD_PROJECT", self.config.get("project", "the-catcher-in-the-rag")
        )
        self.region = os.getenv("GOOGLE_CLOUD_REGION", self.config.get("region", "us-central1"))
        emb_cfg = self.config.get("embeddings", {})
        self.model_id = emb_cfg.get("model_id") or "text-embedding-005"
        self.dimensionality = int(emb_cfg.get("output_dimensionality", 768))
        self._model: Any = None
        self._genai_client: Any = None
        self._use_genai: bool = False

    def _ensure_init(self) -> None:
        if self._model is not None or self._genai_client is not None:
            return

        # 1. Modern Google Gen AI SDK
        try:
            from google import genai

            self._genai_client = genai.Client(
                vertexai=True, project=self.project, location=self.region
            )
            self._use_genai = True
            return
        except Exception:
            self._use_genai = False

        # 2. Legacy vertexai fallback
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
                import vertexai
                from vertexai.language_models import TextEmbeddingModel

                vertexai.init(project=self.project, location=self.region)
                self._model = TextEmbeddingModel.from_pretrained(self.model_id)
        except ImportError as err:
            msg = (
                "google-cloud-aiplatform or google-genai is required to run GeminiEmbeddingProvider. "
                "Ensure google-genai is installed, or use EMBEDDING_PROVIDER=stub for offline mode."
            )
            raise ImportError(msg) from err

    def embed_texts(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Embed a batch of document texts in chunks to respect Vertex AI token limits."""
        if not texts:
            return []
        self._ensure_init()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:6000] for t in texts[i : i + batch_size]]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if self._use_genai:
                        res = self._genai_client.models.embed_content(
                            model=self.model_id,
                            contents=batch,
                            config={"output_dimensionality": self.dimensionality},
                        )
                        all_embeddings.extend([list(e.values) for e in res.embeddings])
                    else:
                        from vertexai.language_models import TextEmbeddingInput

                        inputs = [
                            TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT")
                            for t in batch
                        ]
                        embeddings = self._model.get_embeddings(
                            inputs, output_dimensionality=self.dimensionality
                        )
                        all_embeddings.extend([list(e.values) for e in embeddings])
                    time.sleep(0.1)
                    break
                except Exception as err:
                    err_str = str(err).lower()
                    if (
                        "429" in err_str or "quota" in err_str or "resource" in err_str
                    ) and attempt < max_retries - 1:
                        backoff = 2.0 * (2**attempt)
                        time.sleep(backoff)
                        continue
                    raise
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        self._ensure_init()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if self._use_genai:
                    res = self._genai_client.models.embed_content(
                        model=self.model_id,
                        contents=query[:6000],
                        config={"output_dimensionality": self.dimensionality},
                    )
                    return list(res.embeddings[0].values)
                from vertexai.language_models import TextEmbeddingInput

                inputs = [TextEmbeddingInput(text=query[:6000], task_type="RETRIEVAL_QUERY")]
                embeddings = self._model.get_embeddings(
                    inputs, output_dimensionality=self.dimensionality
                )
                return list(embeddings[0].values)
            except Exception as err:
                err_str = str(err).lower()
                if (
                    "429" in err_str or "quota" in err_str or "resource" in err_str
                ) and attempt < max_retries - 1:
                    backoff = 2.0 * (2**attempt)
                    time.sleep(backoff)
                    continue
                raise
        return []
