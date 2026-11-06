"""Knowledge Graph package (Stage 5).

Provides scientific entity and relationship models, extraction pipelines,
and collection-scoped graph intelligence.
"""

from nexora.graph.analytics import (
    AuthorCollaborationRecord,
    AuthorConnectivityMetrics,
    AuthorPaperConnection,
    CollectionAnalyticsResult,
    CollectionGraphSummary,
    EntityTypeCount,
    RelationshipTypeCount,
)
from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService
from nexora.graph.models import (
    ExtractedEntityCandidate,
    ExtractedGraphPayload,
    ExtractedRelationshipCandidate,
    FencingTokenError,
    GraphExtractionError,
    GraphExtractionResult,
    InvalidExtractionPayloadError,
)
from nexora.graph.prompt import (
    GRAPH_EXTRACTION_SYSTEM_INSTRUCTION,
    GraphExtractionPromptBuilder,
)
from nexora.graph.service import KnowledgeGraphExtractionService
from nexora.graph.traversal import KnowledgeGraphTraversalService
from nexora.graph.traversal_models import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphSubgraph,
)
from nexora.graph.worker import GraphExtractionWorker, KnowledgeGraphWorker

__all__ = [
    "AuthorCollaborationRecord",
    "AuthorConnectivityMetrics",
    "AuthorPaperConnection",
    "CollectionAnalyticsResult",
    "CollectionGraphSummary",
    "EntityTypeCount",
    "ExtractedEntityCandidate",
    "ExtractedGraphPayload",
    "ExtractedRelationshipCandidate",
    "FencingTokenError",
    "GRAPH_EXTRACTION_SYSTEM_INSTRUCTION",
    "GraphEdge",
    "GraphExtractionError",
    "GraphExtractionPromptBuilder",
    "GraphExtractionResult",
    "GraphExtractionWorker",
    "GraphNode",
    "GraphPath",
    "GraphSubgraph",
    "InvalidExtractionPayloadError",
    "KnowledgeGraphAnalyticsService",
    "KnowledgeGraphExtractionService",
    "KnowledgeGraphTraversalService",
    "KnowledgeGraphWorker",
    "RelationshipTypeCount",
]
