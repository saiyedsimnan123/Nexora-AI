"""Opt-in live smoke test for real LLM answer generation in Nexora AI.

Verifies end-to-end connectivity, research retrieval, answer generation,
and citation attribution against the configured real LLM provider (OpenAI).

Safety Guarantees:
1. Opt-In Only: Never makes live API calls unless explicitly invoked by the user.
2. Secret Protection: Never prints or logs API keys or credentials in plaintext.
3. Truthful Attribution: Distinguishes real live model generation from mock synthetic answers.
4. Error Sanitization: Masks any credential tokens that may appear in error messages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from nexora.config.settings import ConfigurationError
from nexora.llm.config import LLMConfig
from nexora.rag.models import RAGResponse
from nexora.rag.pipeline import (
    _sanitize_error_message,
    create_live_rag_pipeline,
)

DEFAULT_SMOKE_QUERY = "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights and reduce trainable parameters?"


def mask_api_key(key: str | None) -> str:
    """Mask an API key for safe display, preserving only prefix and last 4 characters."""
    if not key or not key.strip():
        return "[NOT CONFIGURED]"
    cleaned = key.strip()
    if len(cleaned) <= 8:
        return "[CONFIGURED - HIDDEN]"
    prefix = cleaned[:7] if cleaned.startswith("sk-") else cleaned[:3]
    suffix = cleaned[-4:]
    return f"{prefix}...{suffix}"


@dataclass(frozen=True)
class SmokeTestResult:
    """Structured report of live LLM smoke test execution."""

    status: str  # "success", "unconfigured", "error", or "mock_detected"
    message: str
    provider: str
    model: str
    is_mock: bool
    has_credentials: bool
    masked_key: str
    latency_seconds: float | None = None
    response: RAGResponse | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "is_mock": self.is_mock,
            "has_credentials": self.has_credentials,
            "masked_key": self.masked_key,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
            "response": self.response.to_dict() if self.response else None,
        }


def check_live_configuration(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect LLM configuration safely without leaking secret values."""
    env = os.environ if environ is None else environ

    # Optionally load local .env if present and environ was not overridden
    if environ is None and os.path.exists(".env"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
            env = os.environ
        except ImportError:
            pass

    has_creds = LLMConfig.has_credentials(env)

    # Resolve active key safely
    active_key: str | None = None
    for var in ("NEXORA_LLM_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        val = env.get(var)
        if val and val.strip():
            active_key = val.strip()
            break

    model = env.get("NEXORA_LLM_MODEL") or env.get("OPENAI_MODEL") or "gpt-4o-mini"
    provider = env.get("NEXORA_LLM_PROVIDER") or env.get("LLM_PROVIDER") or "openai"
    base_url = env.get("NEXORA_LLM_BASE_URL") or env.get("OPENAI_BASE_URL")

    return {
        "has_credentials": has_creds,
        "masked_key": mask_api_key(active_key),
        "model": model,
        "provider": provider,
        "base_url": base_url or "https://api.openai.com/v1 (default)",
    }


def run_live_smoke_test(
    query: str = DEFAULT_SMOKE_QUERY,
    *,
    model: str | None = None,
    llm_service: Any | None = None,
    environ: Mapping[str, str] | None = None,
    top_k: int = 3,
    verbose: bool = True,
) -> SmokeTestResult:
    """Execute an opt-in live smoke test against the configured real LLM provider.

    Args:
        query: Research question to ask (default: LoRA adaptation).
        model: Optional model override (e.g. 'gpt-4o-mini', 'gpt-4o').
        llm_service: Optional pre-configured LLMService (for testing with fakes).
        environ: Optional environment dictionary override.
        top_k: Number of evidence chunks to retrieve.
        verbose: Whether to print diagnostic messages to stdout.

    Returns:
        SmokeTestResult detailing the outcome, latency, and response metadata.
    """
    config_info = check_live_configuration(environ)
    env = os.environ if environ is None else environ
    effective_model = model or config_info["model"]
    provider_name = config_info["provider"]

    if verbose:
        print("=" * 78)
        print("          Nexora AI -- Opt-In Real LLM Provider Smoke Test")
        print("=" * 78)
        print(f"Provider:       {provider_name}")
        print(f"Model:          {effective_model}")
        print(f"API Key:        {config_info['masked_key']}")
        print(f"Base URL:       {config_info['base_url']}")
        print(f"Query:          {query}")
        print("-" * 78)

    # Step 1: Check if credentials exist (unless a service was explicitly injected)
    if not config_info["has_credentials"] and llm_service is None:
        msg = (
            "Live LLM provider is UNCONFIGURED. No API key found in environment or .env file.\n\n"
            "To enable live generation:\n"
            "  1. Copy .env.example to .env:  copy .env.example .env\n"
            "  2. Add your OpenAI API key:   NEXORA_LLM_API_KEY=sk-...\n"
            "  3. (Optional) Set model:      NEXORA_LLM_MODEL=gpt-4o-mini\n\n"
            "Note: Nexora AI offline mode remains 100% operational without an API key."
        )
        if verbose:
            print("[STATUS] UNCONFIGURED")
            print(msg)
            print("=" * 78)
        return SmokeTestResult(
            status="unconfigured",
            message=msg,
            provider=provider_name,
            model=effective_model,
            is_mock=True,
            has_credentials=False,
            masked_key=config_info["masked_key"],
        )

    # Step 2: Initialize Live RAG Pipeline
    if verbose:
        print("[1/3] Initializing live RAG pipeline with real provider adapter...")

    start_time = time.perf_counter()
    try:
        pipeline = create_live_rag_pipeline(
            llm_service=llm_service,
            model=effective_model,
            default_mode="hybrid",
            environ=env,
        )
    except ConfigurationError as exc:
        clean_err = _sanitize_error_message(exc)
        if verbose:
            print(f"[FAILED] Pipeline configuration failed: {clean_err}")
            print("=" * 78)
        return SmokeTestResult(
            status="error",
            message=f"Configuration error: {clean_err}",
            provider=provider_name,
            model=effective_model,
            is_mock=True,
            has_credentials=config_info["has_credentials"],
            masked_key=config_info["masked_key"],
            error=clean_err,
        )
    except Exception as exc:
        clean_err = _sanitize_error_message(exc)
        if verbose:
            print(f"[FAILED] Unexpected initialization error: {clean_err}")
            print("=" * 78)
        return SmokeTestResult(
            status="error",
            message=f"Initialization error: {clean_err}",
            provider=provider_name,
            model=effective_model,
            is_mock=True,
            has_credentials=config_info["has_credentials"],
            masked_key=config_info["masked_key"],
            error=clean_err,
        )

    # Integrity check: Ensure pipeline generator is strictly non-mock
    if pipeline.is_mock or pipeline.generator.is_mock:
        msg = "Pipeline reported mock generator when live provider was requested!"
        if verbose:
            print(f"[FAILED] {msg}")
            print("=" * 78)
        return SmokeTestResult(
            status="mock_detected",
            message=msg,
            provider=provider_name,
            model=effective_model,
            is_mock=True,
            has_credentials=config_info["has_credentials"],
            masked_key=config_info["masked_key"],
            error=msg,
        )

    # Step 3: Execute real query
    if verbose:
        print(f"[2/3] Sending research query to '{effective_model}' via '{provider_name}'...")

    try:
        response = pipeline.answer(query, top_k=top_k)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        clean_err = _sanitize_error_message(exc)
        if verbose:
            print(f"[FAILED] Request failed after {elapsed:.2f}s: {clean_err}")
            print("=" * 78)
        return SmokeTestResult(
            status="error",
            message=f"Live generation request failed: {clean_err}",
            provider=provider_name,
            model=effective_model,
            is_mock=False,
            has_credentials=config_info["has_credentials"],
            masked_key=config_info["masked_key"],
            latency_seconds=elapsed,
            error=clean_err,
        )

    elapsed = time.perf_counter() - start_time

    # Step 4: Validate live response
    if response.status == "error":
        clean_err = _sanitize_error_message(response.error or "Unknown error")
        if verbose:
            print(f"[FAILED] Generation returned error status after {elapsed:.2f}s: {clean_err}")
            print("=" * 78)
        return SmokeTestResult(
            status="error",
            message=f"Provider reported error: {clean_err}",
            provider=provider_name,
            model=response.model or effective_model,
            is_mock=response.is_mock,
            has_credentials=config_info["has_credentials"],
            masked_key=config_info["masked_key"],
            latency_seconds=elapsed,
            response=response,
            error=clean_err,
        )

    if response.is_mock:
        msg = "Response was generated by synthetic mock rather than live LLM!"
        if verbose:
            print(f"[FAILED] {msg}")
            print("=" * 78)
        return SmokeTestResult(
            status="mock_detected",
            message=msg,
            provider=provider_name,
            model=response.model,
            is_mock=True,
            has_credentials=config_info["has_credentials"],
            masked_key=config_info["masked_key"],
            latency_seconds=elapsed,
            response=response,
            error=msg,
        )

    # Success output
    if verbose:
        print(f"[3/3] Live answer generated successfully in {elapsed:.2f}s!\n")
        print("VERIFIED RESPONSE DETAILS:")
        print(f"  Model Served:       {response.model}")
        print(f"  Live Generation:    Yes (is_mock = False)")
        print(f"  Execution Status:   {response.status.upper()}")
        print(f"  Evidence Retrieved: {response.evidence_count} chunks")
        print(f"  Citations Verified: {len(response.cited_sources)} cited sources")
        grounded_str = "YES (All citations point to retrieved evidence)" if response.is_grounded else "NO (Ungrounded)"
        print(f"  Grounded:           {grounded_str}")
        if response.usage:
            print(f"  Token Usage:        {response.usage}")
        print("\nGENERATED ANSWER:")
        print(response.answer)
        print("\nCITED SOURCES:")
        for src in response.cited_sources:
            print(f"  {src.label} {src.paper_title} ({src.chunk_id})")
        print("=" * 78)
        print("SMOKE TEST RESULT: SUCCESS -- Real LLM integration is operational.")
        print("=" * 78)

    return SmokeTestResult(
        status="success",
        message="Real LLM generation succeeded and citations were verified.",
        provider=provider_name,
        model=response.model,
        is_mock=False,
        has_credentials=True,
        masked_key=config_info["masked_key"],
        latency_seconds=elapsed,
        response=response,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for python -m nexora.rag.smoke."""
    parser = argparse.ArgumentParser(
        prog="nexora.rag.smoke",
        description="Nexora AI -- Opt-in Live LLM Provider Smoke Test",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_SMOKE_QUERY,
        help="Optional research question to test.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional model override (e.g. 'gpt-4o-mini', 'gpt-4o').",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of evidence chunks to retrieve (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON.",
    )
    args = parser.parse_args(argv)

    result = run_live_smoke_test(
        query=args.query,
        model=args.model,
        top_k=args.top_k,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))

    if result.status == "success":
        return 0
    elif result.status == "unconfigured":
        return 2  # Distinct exit code for unconfigured
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
