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

export interface EvidenceByPaperItem {
  paper_id: string | null;
  paper_title: string;
  findings: string[];
  excerpts: string[];
  citations: string[];
  sources: AttributedSource[];
}

export interface ComparisonOfFindings {
  agreements: string[];
  differences: string[];
  relationships: string[];
  direct_evidence: string[];
  interpretation: string[];
}

export interface LimitationsAndConflicts {
  limitations: string[];
  conflicts: string[];
  evidence_gaps: string[];
  insufficient_evidence_notes?: string;
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
  evidence_by_paper?: EvidenceByPaperItem[];
  comparison_of_findings?: ComparisonOfFindings | null;
  limitations_and_conflicts?: LimitationsAndConflicts | null;
  conclusion?: 'Demonstrated improvement' | 'Limited evidence' | 'Conflicting evidence' | 'Insufficient evidence' | string | null;
  conclusion_reasoning?: string | null;
}

export interface PaperItem {
  id: string;
  title: string;
  doi?: string | null;
  publication_year?: number | null;
  authors: string[];
  abstract?: string | null;
  is_benchmark: boolean;
  source: string;
}

export interface PaperSearchResponse {
  papers: PaperItem[];
  total: number;
  offset: number;
  limit: number;
}

export interface ComparisonTableCell {
  value: string;
  status: 'reported' | 'not_reported' | 'uncertain';
  citations: string[];
}

export interface ComparisonRow {
  dimension: string;
  key: string;
  per_paper: Record<string, ComparisonTableCell>;
}

export interface ComparisonTableData {
  rows: ComparisonRow[];
  key_similarities: string[];
  key_differences: string[];
  direct_evidence_findings: string[];
  evidence_gaps: string[];
  conclusion: string;
  conclusion_reasoning: string;
  comparability_status: string;
}

export interface ResearchCompareRequest {
  paper_ids: string[];
  query?: string | null;
  model?: string | null;
  temperature?: number | null;
  max_output_tokens?: number | null;
}

export interface ResearchCompareResponse {
  paper_ids: string[];
  paper_titles: Record<string, string | null>;
  matrix: Record<string, unknown>;
  conclusion: string;
  conclusion_reasoning: string;
  comparability_status: string;
  overall_summary: string;
  model: string;
  usage?: Record<string, number> | null;
  comparison_table?: ComparisonTableData | null;
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
