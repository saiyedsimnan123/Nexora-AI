"""Composition root for the Nexora application."""

from nexora.bootstrap.application import (
    build_analysis_service,
    build_comparison_service,
    build_http_app,
    build_ingestion_service,
    build_research_application,
    build_verification_service,
)

__all__ = [
    "build_analysis_service",
    "build_comparison_service",
    "build_http_app",
    "build_ingestion_service",
    "build_research_application",
    "build_verification_service",
]

