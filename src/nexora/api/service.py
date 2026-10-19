"""Application boundary between API requests and the research pipeline."""

from __future__ import annotations

from typing import Any, Sequence

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

    def verify(
        self,
        request: ResearchVerifyRequest,
        *,
        collection_ids: Sequence[str] | None = None,
        paper_id: str | None = None,
    ) -> ResearchVerifyResponse:
        if not isinstance(request, ResearchVerifyRequest):
            raise InvalidRequestError("request must be a ResearchVerifyRequest")
        try:
            eff_col_ids = (
                collection_ids
                if collection_ids is not None
                else getattr(request, "collection_ids", None)
            )
            eff_paper_id = (
                paper_id
                if paper_id is not None
                else (getattr(request, "paper_id", None) or request.expected_paper_id)
            )

            # Fail closed immediately if collection_ids is explicitly empty
            if eff_col_ids is not None and len(eff_col_ids) == 0:
                verifier = self._verifier
                if verifier is None:
                    from nexora.research.verification import ClaimToEvidenceVerifier

                    verifier = ClaimToEvidenceVerifier(expected_paper_id=request.expected_paper_id)
                elif (
                    request.expected_paper_id is not None
                    and getattr(verifier, "_expected_paper_id", None) != request.expected_paper_id
                ):
                    from nexora.research.verification import ClaimToEvidenceVerifier

                    verifier = ClaimToEvidenceVerifier(
                        getattr(verifier, "_llm", None),
                        prompt_builder=getattr(verifier, "_prompt_builder", None),
                        expected_paper_id=request.expected_paper_id,
                    )

                from nexora.retrieval.evidence import EvidenceContext

                empty_ctx = EvidenceContext(query=request.query, items=[], total_candidates=0)

                if request.text is not None:
                    v_res = verifier.verify_answer(request.text, empty_ctx, query=request.query)
                    return ResearchVerifyResponse(
                        text=request.text,
                        query=request.query,
                        model=request.model or "verification-only",
                        usage=None,
                        evidence_count=0,
                        verification=v_res.to_dict(),
                    )
                else:
                    ans_text = "No evidence found in authorized research collections."
                    v_res = verifier.verify_answer(ans_text, empty_ctx, query=request.query)
                    return ResearchVerifyResponse(
                        text=ans_text,
                        query=request.query,
                        model=request.model or "unknown",
                        usage=None,
                        evidence_count=0,
                        verification=v_res.to_dict(),
                    )

            context_kwargs: dict[str, Any] = {}
            if eff_col_ids is not None:
                context_kwargs["collection_ids"] = eff_col_ids
            if eff_paper_id is not None:
                context_kwargs["paper_id"] = eff_paper_id

            # If explicit answer text is supplied, verify it against retrieved context
            if request.text is not None:
                if hasattr(self._query_service, "get_evidence_context"):
                    try:
                        context = self._query_service.get_evidence_context(request.query, **context_kwargs)
                    except TypeError:
                        context = self._query_service.get_evidence_context(request.query)
                elif hasattr(self._query_service, "_evidence_builder"):
                    try:
                        context = self._query_service._evidence_builder.build(request.query, **context_kwargs)
                    except TypeError:
                        context = self._query_service._evidence_builder.build(request.query)
                else:
                    raise InvalidConfigurationError("query_service cannot build evidence context")

                verifier = self._verifier
                if verifier is None:
                    from nexora.research.verification import ClaimToEvidenceVerifier

                    verifier = ClaimToEvidenceVerifier(expected_paper_id=request.expected_paper_id)
                elif (
                    request.expected_paper_id is not None
                    and getattr(verifier, "_expected_paper_id", None) != request.expected_paper_id
                ):
                    from nexora.research.verification import ClaimToEvidenceVerifier

                    verifier = ClaimToEvidenceVerifier(
                        getattr(verifier, "_llm", None),
                        prompt_builder=getattr(verifier, "_prompt_builder", None),
                        expected_paper_id=request.expected_paper_id,
                    )

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
            ask_kwargs: dict[str, Any] = {
                "model": request.model,
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
            }
            if eff_col_ids is not None:
                ask_kwargs["collection_ids"] = eff_col_ids
            if eff_paper_id is not None:
                ask_kwargs["paper_id"] = eff_paper_id

            if hasattr(self._query_service, "ask_with_context"):
                try:
                    answer, context = self._query_service.ask_with_context(
                        request.query,
                        **ask_kwargs,
                    )
                except TypeError:
                    answer, context = self._query_service.ask_with_context(
                        request.query,
                        model=request.model,
                        temperature=request.temperature,
                        max_output_tokens=request.max_output_tokens,
                    )
            else:
                try:
                    answer = self._query_service.ask(
                        request.query,
                        **ask_kwargs,
                    )
                except TypeError:
                    answer = self._query_service.ask(
                        request.query,
                        model=request.model,
                        temperature=request.temperature,
                        max_output_tokens=request.max_output_tokens,
                    )
                if hasattr(self._query_service, "get_evidence_context"):
                    try:
                        context = self._query_service.get_evidence_context(request.query, **context_kwargs)
                    except TypeError:
                        context = self._query_service.get_evidence_context(request.query)
                elif hasattr(self._query_service, "_evidence_builder"):
                    try:
                        context = self._query_service._evidence_builder.build(request.query, **context_kwargs)
                    except TypeError:
                        context = self._query_service._evidence_builder.build(request.query)
                else:
                    raise InvalidConfigurationError("query_service cannot build evidence context")

            verifier = self._verifier
            if verifier is None:
                from nexora.research.verification import ClaimToEvidenceVerifier

                verifier = ClaimToEvidenceVerifier(expected_paper_id=request.expected_paper_id)
            elif (
                request.expected_paper_id is not None
                and getattr(verifier, "_expected_paper_id", None) != request.expected_paper_id
            ):
                from nexora.research.verification import ClaimToEvidenceVerifier

                verifier = ClaimToEvidenceVerifier(
                    getattr(verifier, "_llm", None),
                    prompt_builder=getattr(verifier, "_prompt_builder", None),
                    expected_paper_id=request.expected_paper_id,
                )

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

