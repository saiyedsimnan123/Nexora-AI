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
  reported_findings?: string[];
  unestablished_claims?: string[];
  supported_metrics?: string[];
  unsupported_metrics?: string[];
}

export interface LimitationsAndConflicts {
  limitations: string[];
  conflicts: string[];
  evidence_gaps: string[];
  what_cannot_be_concluded?: string[];
  unsupported_metrics?: string[];
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
  takeaway?: string | null;
}

export interface ResearchJobSubmitResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  created_at: string;
  poll_url: string;
}

export interface ResearchJobStatusResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  query: string;
  result?: RAGQueryResponse | null;
  error?: string | null;
  attempts?: number;
  created_at: string;
  completed_at?: string | null;
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

export interface DocumentIngestResponse {
  document_id: string;
  paper_id: string;
  chunks_count: number;
  indexed_chunks_count: number;
  graph_job_id?: string | null;
  graph_status?: string | null;
}

export interface PaperAnalysisDimension {
  summary: string;
  evidence_labels?: string[];
  evidence_citations?: string[];
  status?: 'reported' | 'uncertain' | 'not_reported' | string;
  is_missing?: boolean;
  details?: Record<string, unknown>;
}

export interface PaperAnalysisResponse {
  paper_id: string;
  title: string | null;
  problem_and_objectives: PaperAnalysisDimension | Record<string, any>;
  methodology_and_algorithms: PaperAnalysisDimension | Record<string, any>;
  models_and_datasets: PaperAnalysisDimension | Record<string, any>;
  experiments_and_metrics: PaperAnalysisDimension | Record<string, any>;
  results_and_findings: PaperAnalysisDimension | Record<string, any>;
  limitations: PaperAnalysisDimension | Record<string, any>;
  future_work: PaperAnalysisDimension | Record<string, any>;
  evidence_count: number;
  model: string;
  usage?: Record<string, number> | null;
}

export interface DocumentedGapItem {
  title: string;
  description: string;
  category?: string;
  paper_id?: string;
  evidence_citations?: string[];
  limitations?: string[];
  is_missing?: boolean;
}

export interface InferredGapItem {
  title: string;
  description: string;
  category?: string;
  paper_ids?: string[];
  rationale?: string;
  confidence_score?: number;
}

export interface ResearchGapsRequest {
  paper_ids?: string[];
  collection_id?: string;
  model?: string;
  temperature?: number;
  max_output_tokens?: number;
}

export interface ResearchGapsResponse {
  paper_ids: string[];
  collection_id?: string | null;
  documented_gaps: DocumentedGapItem[];
  inferred_gaps: InferredGapItem[];
  total_gaps_count: number;
  summary: string;
  limitations: string[];
  model: string;
  usage?: Record<string, number> | null;
}

export interface ResearchTrendItem {
  topic: string;
  direction: string;
  description: string;
  evidence?: string[];
}

export interface ResearchTrendsRequest {
  collection_id: string;
  paper_ids?: string[];
  time_unit?: string;
  model?: string;
  temperature?: number;
  max_output_tokens?: number;
}

export interface ResearchTrendsResponse {
  collection_id: string;
  paper_ids: string[];
  time_unit: string;
  temporal_buckets: Record<string, any>[];
  trends: ResearchTrendItem[];
  missing_date_paper_ids: string[];
  summary: string;
  limitations: string[];
  model: string;
  usage?: Record<string, number> | null;
}
