"""Application boundary between API requests and the research pipeline."""

from __future__ import annotations

from typing import Any

from nexora.api.errors import (
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
    VerificationServiceError,
)
from nexora.api.models import (
    ResearchRequest,
    ResearchResponse,
    ResearchVerifyRequest,
    ResearchVerifyResponse,
)


class ResearchApplicationService:
    """Converts a ResearchRequest into a ResearchResponse via ResearchQueryService.

    Pipeline failures are re-raised as ResearchServiceError (original chained),
    never turned into a successful response. Provider details are not copied
    into the error message.
    """

    def __init__(self, query_service: Any, *, verifier: Any = None) -> None:
        if not callable(getattr(query_service, "ask", None)):
            raise InvalidConfigurationError("query_service must provide ask(query, ...)")
        self._query_service = query_service
        self._verifier = verifier

    def ask(
        self,
        request: ResearchRequest,
        *,
        collection_ids: list[str] | None = None,
        collection_id: str | None = None,
    ) -> ResearchResponse:
        if not isinstance(request, ResearchRequest):
            raise InvalidRequestError("request must be a ResearchRequest")
        try:
            ask_kwargs: dict[str, Any] = {
                "model": request.model,
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
            }
            if collection_ids is not None:
                ask_kwargs["collection_ids"] = collection_ids
            if collection_id is not None:
                ask_kwargs["collection_id"] = collection_id
            try:
                answer = self._query_service.ask(request.query, **ask_kwargs)
            except TypeError:
                answer = self._query_service.ask(
                    request.query,
                    model=request.model,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
        except Exception as exc:
            raise ResearchServiceError(
                f"Research request failed ({type(exc).__name__})"
            ) from exc
        return ResearchResponse.from_answer(answer)

    def verify(self, request: ResearchVerifyRequest) -> ResearchVerifyResponse:
        if not isinstance(request, ResearchVerifyRequest):
            raise InvalidRequestError("request must be a ResearchVerifyRequest")
        try:
            # If explicit answer text is supplied, verify it against retrieved context
            if request.text is not None:
                if hasattr(self._query_service, "get_evidence_context"):
                    context = self._query_service.get_evidence_context(request.query)
                elif hasattr(self._query_service, "_evidence_builder"):
                    context = self._query_service._evidence_builder.build(request.query)
                else:
                    raise InvalidConfigurationError("query_service cannot build evidence context")

                verifier = self._verifier
                if verifier is None:
                    from nexora.research.verification import ClaimToEvidenceVerifier
                    verifier = ClaimToEvidenceVerifier(expected_paper_id=request.expected_paper_id)

                v_res = verifier.verify_answer(request.text, context, query=request.query)
                return ResearchVerifyResponse(
                    text=request.text,
                    query=request.query,
                    model=request.model or "verification-only",
                    usage=None,
                    evidence_count=len(context.items),
                    verification=v_res.to_dict(),
                )

            # Otherwise generate the answer with context then verify
            if hasattr(self._query_service, "ask_with_context"):
                answer, context = self._query_service.ask_with_context(
                    request.query,
                    model=request.model,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
            else:
                answer = self._query_service.ask(
                    request.query,
                    model=request.model,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
                if hasattr(self._query_service, "get_evidence_context"):
                    context = self._query_service.get_evidence_context(request.query)
                elif hasattr(self._query_service, "_evidence_builder"):
                    context = self._query_service._evidence_builder.build(request.query)
                else:
                    raise InvalidConfigurationError("query_service cannot build evidence context")

            verifier = self._verifier
            if verifier is None:
                from nexora.research.verification import ClaimToEvidenceVerifier
                verifier = ClaimToEvidenceVerifier(expected_paper_id=request.expected_paper_id)

            v_res = verifier.verify_answer(answer.text, context, query=request.query)
            return ResearchVerifyResponse(
                text=answer.text,
                query=answer.query,
                model=answer.model,
                usage=answer.usage,
                evidence_count=answer.evidence_count,
                verification=v_res.to_dict(),
            )
        except InvalidRequestError:
            raise
        except Exception as exc:
            raise VerificationServiceError(
                f"Research verification failed ({type(exc).__name__})"
            ) from exc

