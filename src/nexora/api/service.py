"""Application boundary between API requests and the research pipeline."""

from __future__ import annotations

from typing import Any

from nexora.api.errors import (
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
)
from nexora.api.models import ResearchRequest, ResearchResponse


class ResearchApplicationService:
    """Converts a ResearchRequest into a ResearchResponse via ResearchQueryService.

    Pipeline failures are re-raised as ResearchServiceError (original chained),
    never turned into a successful response. Provider details are not copied
    into the error message.
    """

    def __init__(self, query_service: Any) -> None:
        if not callable(getattr(query_service, "ask", None)):
            raise InvalidConfigurationError("query_service must provide ask(query, ...)")
        self._query_service = query_service

    def ask(self, request: ResearchRequest) -> ResearchResponse:
        if not isinstance(request, ResearchRequest):
            raise InvalidRequestError("request must be a ResearchRequest")
        try:
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
