/**
 * Nexora AI API Client Wire Types.
 * Matches backend Pydantic models in `nexora.api.routes`.
 */

export interface AttributedSource {
  label: string; // e.g. '[E1]'
  document_id: string;
  chunk_id: string;
  paper_id: string | null;
  paper_title: string;
  score: number;
  text: string;
  text_excerpt: string;
  metadata: Record<string, unknown>;
  page_number: number | null;
}

export interface RAGQueryRequest {
  query: string;
  retrieval_mode?: 'bm25' | 'vector' | 'hybrid';
  top_k?: number;
  min_score?: number | null;
  temperature?: number | null;
  max_output_tokens?: number | null;
  collection_id?: string | null;
  paper_id?: string | null;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  sources: AttributedSource[];
  cited_sources: AttributedSource[];
  is_grounded: boolean;
  is_mock: boolean;
  model: string;
  status: 'success' | 'insufficient_evidence' | 'error';
  error: string | null;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  } | null;
  retrieval_mode: string;
  evidence_count: number;
}

export interface HealthResponse {
  status: string;
}

export interface ReadinessCheck {
  status: string;
  error?: string;
  enforced?: boolean;
}

export interface ReadinessResponse {
  status: string;
  checks: Record<string, ReadinessCheck>;
}
