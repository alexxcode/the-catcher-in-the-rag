from __future__ import annotations

import os
from typing import Any, TypeVar

from pydantic import BaseModel

from catcher_in_the_rag.providers.base import EmbeddingProvider, LLMProvider

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(LLMProvider):
    """Real GCP generative AI provider using Vertex AI (ADC credentials, no API keys).

    Adheres to Non-negotiable Principle #7 (ADC auth) and #8 (provider-agnostic core).
    Model versions and retirement dates must be verified against:
    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> None:
        self.config = config or {}
        self.project = os.getenv(
            "GOOGLE_CLOUD_PROJECT", self.config.get("project", "the-catcher-in-the-rag")
        )
        self.region = os.getenv("GOOGLE_CLOUD_REGION", self.config.get("region", "us-central1"))
        ext_cfg = self.config.get("extraction_llm", {})
        # Configured model or GA default candidate
        model_candidate = model_id or ext_cfg.get("model_id")
        if not model_candidate or "TODO" in str(model_candidate):
            model_candidate = "gemini-3.5-flash"
        self.model_id = model_candidate
        self.temperature = float(ext_cfg.get("temperature", 0.0))
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

        if self._use_genai:
            response = self._genai_client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )
            raw_json = (response.text or "").strip()
        else:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
                from vertexai.generative_models import GenerationConfig

                gen_config = GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                )
                response = self._client.generate_content(prompt, generation_config=gen_config)
                raw_json = (response.text or "").strip()

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

        if self._use_genai:
            response = self._genai_client.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config={"temperature": self.temperature},
            )
            return response.text or ""

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
            from vertexai.generative_models import GenerationConfig

            gen_config = GenerationConfig(temperature=self.temperature)
            response = self._client.generate_content(full_prompt, generation_config=gen_config)
            return response.text or ""


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

    def _ensure_init(self) -> None:
        if self._model is not None:
            return
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(project=self.project, location=self.region)
            self._model = TextEmbeddingModel.from_pretrained(self.model_id)
        except ImportError as err:
            msg = (
                "google-cloud-aiplatform is required to run GeminiEmbeddingProvider. "
                "Ensure google-cloud-aiplatform is installed, or use EMBEDDING_PROVIDER=stub for offline mode."
            )
            raise ImportError(msg) from err

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        if not texts:
            return []
        self._ensure_init()
        from vertexai.language_models import TextEmbeddingInput

        inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
        embeddings = self._model.get_embeddings(inputs, output_dimensionality=self.dimensionality)
        return [list(e.values) for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        self._ensure_init()
        from vertexai.language_models import TextEmbeddingInput

        inputs = [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
        embeddings = self._model.get_embeddings(inputs, output_dimensionality=self.dimensionality)
        return list(embeddings[0].values)
