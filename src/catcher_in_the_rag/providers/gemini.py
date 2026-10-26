from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from catcher_in_the_rag.providers.base import EmbeddingProvider, LLMProvider
from catcher_in_the_rag.providers.tracing import get_tracer

T = TypeVar("T", bound=BaseModel)


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


_live_calls_used = 0


def _consume_live_call() -> None:
    """Hard cap on real Vertex network calls per process; MAX_LIVE_CALLS defaults to 0 (blocked)."""
    global _live_calls_used
    limit = _safe_int(os.getenv("MAX_LIVE_CALLS", "0"))
    if _live_calls_used >= limit:
        raise RuntimeError(f"MAX_LIVE_CALLS={limit} reached; live call blocked")
    _live_calls_used += 1


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
        thinking_budget: int | None = None,
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

        # Configurable thinking budget (default to GEMINI_THINKING_BUDGET env var if set)
        tb = thinking_budget if thinking_budget is not None else ext_cfg.get("thinking_budget")
        if tb is None and os.getenv("GEMINI_THINKING_BUDGET") is not None:
            try:
                tb = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
            except ValueError:
                tb = None
        self.thinking_budget = tb

        # Model location routing: Gemini 3.5 & 3.8 Flash GA endpoints are hosted in global / multi-region
        default_region = self.config.get("region", "us-central1")
        if (
            self.model_id in ("gemini-3.5-flash", "gemini-3.8-flash")
            and default_region == "us-central1"
            and not os.getenv("GOOGLE_CLOUD_REGION")
        ):
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
            _consume_live_call()
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

    def _log_raw_call(
        self,
        prompt: str,
        response: Any,
        latency_ms: float,
        extra: dict[str, Any],
    ) -> None:
        run_id = os.getenv("RUN_ID") or getattr(self, "run_id", None)
        if not run_id:
            run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            self.run_id = run_id

        raw_dir = Path(os.getenv("RAW_LOG_DIR", "evals/reports/raw"))
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"{run_id}.jsonl"

        usage = getattr(response, "usage_metadata", None)
        p_tok = _safe_int(getattr(usage, "prompt_token_count", 0)) if usage else 0
        c_tok = _safe_int(getattr(usage, "candidates_token_count", 0)) if usage else 0
        th_tok = (
            _safe_int(
                getattr(usage, "thoughts_token_count", getattr(usage, "thinking_token_count", 0))
            )
            if usage
            else 0
        )

        raw_text = getattr(response, "text", "")
        resp_text = raw_text if isinstance(raw_text, str) else ""
        raw_id = getattr(response, "response_id", None)
        resp_id = raw_id if isinstance(raw_id, str) else None
        raw_ver = getattr(response, "model_version", None)
        m_ver = raw_ver if isinstance(raw_ver, str) else None

        record = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_text": resp_text,
            "response_id": resp_id,
            "model_version": m_ver,
            "prompt_tokens": p_tok,
            "candidate_tokens": c_tok,
            "thought_tokens": th_tok,
            "latency_ms": round(latency_ms, 2),
            "query": extra.get("query"),
            "category": extra.get("category"),
            "contract_id": extra.get("contract_id"),
            "passage_ids": extra.get("passage_ids"),
        }
        with raw_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def extract_structured(
        self,
        text: str,
        schema: type[T],
        system_prompt: str | None = None,
        **kwargs: Any,
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
                cfg: dict[str, Any] = {
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                }
                if self.thinking_budget is not None:
                    cfg["thinking_config"] = {"thinking_budget": self.thinking_budget}
                return self._genai_client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=cfg,
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
        self._log_raw_call(prompt, response, elapsed_ms, kwargs)

        # Extract usage metadata if available (including thoughts tokens billed as output)
        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = _safe_int(getattr(response.usage_metadata, "prompt_token_count", 0))
            c_tok = _safe_int(getattr(response.usage_metadata, "candidates_token_count", 0))
            tot_tok = _safe_int(getattr(response.usage_metadata, "total_token_count", 0))
            raw_th = getattr(response.usage_metadata, "thoughts_token_count", None)
            if raw_th is None:
                raw_th = getattr(response.usage_metadata, "thinking_token_count", None)
            th_tok = _safe_int(raw_th) if raw_th is not None else 0
            if not th_tok and tot_tok > (p_tok + c_tok):
                th_tok = tot_tok - (p_tok + c_tok)

            # In Vertex AI pricing, reasoning/thinking tokens are billed at the candidate token rate
            billable_output = c_tok if (c_tok >= (tot_tok - p_tok)) else (c_tok + th_tok)

            usage = {
                "input": p_tok,
                "output": billable_output,
                "thoughts": th_tok,
                "total": tot_tok,
            }
            self.total_prompt_tokens = getattr(self, "total_prompt_tokens", 0) + p_tok
            self.total_candidate_tokens = (
                getattr(self, "total_candidate_tokens", 0) + billable_output
            )
            self.total_thoughts_tokens = getattr(self, "total_thoughts_tokens", 0) + th_tok

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

    def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Alias for extract_structured for API symmetry."""
        return self.extract_structured(
            text=prompt, schema=response_schema, system_prompt=system_prompt, **kwargs
        )

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text completion from Gemini."""
        self._ensure_init()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        def _invoke_generate() -> Any:
            if self._use_genai:
                cfg: dict[str, Any] = {"temperature": self.temperature}
                if self.thinking_budget is not None:
                    cfg["thinking_config"] = {"thinking_budget": self.thinking_budget}
                return self._genai_client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=cfg,
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
        self._log_raw_call(full_prompt, response, elapsed_ms, kwargs)

        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = _safe_int(getattr(response.usage_metadata, "prompt_token_count", 0))
            c_tok = _safe_int(getattr(response.usage_metadata, "candidates_token_count", 0))
            tot_tok = _safe_int(getattr(response.usage_metadata, "total_token_count", 0))
            raw_th = getattr(response.usage_metadata, "thoughts_token_count", None)
            if raw_th is None:
                raw_th = getattr(response.usage_metadata, "thinking_token_count", None)
            th_tok = _safe_int(raw_th) if raw_th is not None else 0
            if not th_tok and tot_tok > (p_tok + c_tok):
                th_tok = tot_tok - (p_tok + c_tok)

            billable_output = c_tok if (c_tok >= (tot_tok - p_tok)) else (c_tok + th_tok)

            usage = {
                "input": p_tok,
                "output": billable_output,
                "thoughts": th_tok or 0,
                "total": tot_tok,
            }
            self.total_prompt_tokens = getattr(self, "total_prompt_tokens", 0) + p_tok
            self.total_candidate_tokens = (
                getattr(self, "total_candidate_tokens", 0) + billable_output
            )
            self.total_thoughts_tokens = getattr(self, "total_thoughts_tokens", 0) + (th_tok or 0)

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
        _consume_live_call()
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
    """Dense embedding provider using Vertex AI text-embedding models with disk caching."""

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
        self._init_cache()

    def _init_cache(self) -> None:
        cache_dir = Path(".cache/embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_db_path = cache_dir / "embeddings.db"
        try:
            with sqlite3.connect(self._cache_db_path, timeout=10.0) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings_cache (
                        model_id TEXT,
                        text_sha256 TEXT,
                        embedding TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (model_id, text_sha256)
                    )
                    """
                )
                conn.commit()
        except Exception:
            pass

    def _get_cached(self, text: str) -> list[float] | None:
        if not hasattr(self, "_cache_db_path") or not self._cache_db_path.exists():
            return None
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with sqlite3.connect(self._cache_db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding FROM embeddings_cache WHERE model_id = ? AND text_sha256 = ?",
                    (self.model_id, sha),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            return None
        return None

    def _put_cached(self, text: str, embedding: list[float]) -> None:
        if not hasattr(self, "_cache_db_path"):
            return
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with sqlite3.connect(self._cache_db_path, timeout=10.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings_cache (model_id, text_sha256, embedding) VALUES (?, ?, ?)",
                    (self.model_id, sha, json.dumps(embedding)),
                )
                conn.commit()
        except Exception:
            pass

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

    def embed_texts(
        self, texts: list[str], batch_size: int = 10, pace_delay_seconds: float = 0.2
    ) -> list[list[float]]:
        """Embed a batch of document texts with persistent disk caching and quota pacing."""
        if not texts:
            return []

        # 1. Check disk cache for each text
        results: list[list[float] | None] = [self._get_cached(t) for t in texts]
        missing_indices = [i for i, r in enumerate(results) if r is None]

        if not missing_indices:
            # Complete cache hit! 0 external API calls needed
            return [r for r in results if r is not None]

        self._ensure_init()

        # 2. Fetch missing texts in batches with pacing
        for chunk_start in range(0, len(missing_indices), batch_size):
            chunk_indices = missing_indices[chunk_start : chunk_start + batch_size]
            batch_texts = [texts[idx][:6000] for idx in chunk_indices]
            max_retries = 5
            batch_embeddings: list[list[float]] = []

            for attempt in range(max_retries):
                _consume_live_call()
                try:
                    if self._use_genai:
                        res = self._genai_client.models.embed_content(
                            model=self.model_id,
                            contents=batch_texts,
                            config={"output_dimensionality": self.dimensionality},
                        )
                        batch_embeddings = [list(e.values) for e in res.embeddings]
                    else:
                        from vertexai.language_models import TextEmbeddingInput

                        inputs = [
                            TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT")
                            for t in batch_texts
                        ]
                        embeddings = self._model.get_embeddings(
                            inputs, output_dimensionality=self.dimensionality
                        )
                        batch_embeddings = [list(e.values) for e in embeddings]
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

            # Store in cache and update results
            for idx, emb in zip(chunk_indices, batch_embeddings, strict=False):
                results[idx] = emb
                self._put_cached(texts[idx], emb)

            if chunk_start + batch_size < len(missing_indices):
                time.sleep(pace_delay_seconds)

        return [r or [] for r in results]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query with persistent disk caching."""
        cache_key = f"query:{query}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._ensure_init()
        max_retries = 5
        for attempt in range(max_retries):
            _consume_live_call()
            try:
                if self._use_genai:
                    res = self._genai_client.models.embed_content(
                        model=self.model_id,
                        contents=query[:6000],
                        config={"output_dimensionality": self.dimensionality},
                    )
                    emb = list(res.embeddings[0].values)
                else:
                    from vertexai.language_models import TextEmbeddingInput

                    inputs = [TextEmbeddingInput(text=query[:6000], task_type="RETRIEVAL_QUERY")]
                    embeddings = self._model.get_embeddings(
                        inputs, output_dimensionality=self.dimensionality
                    )
                    emb = list(embeddings[0].values)

                self._put_cached(cache_key, emb)
                return emb
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
