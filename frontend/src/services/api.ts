/**
 * Nexora AI API Client Service.
 * Connects the frontend to the backend REST API endpoints.
 */

import type {
  RAGQueryRequest,
  RAGQueryResponse,
  ResearchJobSubmitResponse,
  ResearchJobStatusResponse,
  HealthResponse,
  ReadinessResponse,
  PaperSearchResponse,
  ResearchCompareRequest,
  ResearchCompareResponse,
  DocumentIngestResponse,
  PaperAnalysisResponse,
  ResearchGapsResponse,
  ResearchTrendsResponse,
} from '../types/api';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');
const DEFAULT_TIMEOUT_MS = 20000;
const GUEST_STORAGE_KEY = 'nexora_guest_session_token';
export const DEFAULT_GUEST_TOKEN = 'nx_guest_preview_session';
export const ACTIVE_RESEARCH_JOB_KEY = 'nexora_active_research_job_id';

export class ApiError extends Error {
  public statusCode?: number;
  public details?: unknown;

  constructor(
    message: string,
    statusCode?: number,
    details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Retrieve cached guest token or fetch a temporary guest session token.
 */
export async function getOrFetchGuestToken(signal?: AbortSignal): Promise<string> {
  if (typeof window === 'undefined') return DEFAULT_GUEST_TOKEN;

  try {
    const cached =
      window.sessionStorage?.getItem(GUEST_STORAGE_KEY) ||
      window.localStorage?.getItem(GUEST_STORAGE_KEY);
    if (cached && cached.trim()) {
      return cached.trim();
    }

    const res = await fetch(`${API_BASE_URL}/api/v1/auth/guest-session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      signal,
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.token) {
        window.sessionStorage?.setItem(GUEST_STORAGE_KEY, data.token);
        return data.token;
      }
    }
  } catch {
    // If endpoint is unreachable or in non-browser context, fallback gracefully
  }

  try {
    window.sessionStorage?.setItem(GUEST_STORAGE_KEY, DEFAULT_GUEST_TOKEN);
  } catch {}
  return DEFAULT_GUEST_TOKEN;
}

/**
 * Fetch with automatic timeout handling.
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const headers = new Headers(options.headers || {});
    if (!headers.has('Authorization') && !headers.has('X-API-Key')) {
      let token = DEFAULT_GUEST_TOKEN;
      if (typeof window !== 'undefined') {
        const cached =
          window.sessionStorage?.getItem(GUEST_STORAGE_KEY) ||
          window.localStorage?.getItem(GUEST_STORAGE_KEY);
        if (cached && cached.trim()) {
          token = cached.trim();
        }
      }
      headers.set('Authorization', `Bearer ${token}`);
    }

    const response = await fetch(url, {
      ...options,
      headers,
      signal: options.signal || controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(`Request timed out after ${timeoutMs / 1000}s`, 504);
    }
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new ApiError('Unable to connect to the backend server. Is the API service running?', 503);
    }
    throw error;
  } finally {
    clearTimeout(id);
  }
}

/**
 * Check backend liveness via GET /health.
 */
export async function checkBackendHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/health`,
      { method: 'GET', signal },
      4000
    );
    if (!response.ok) return false;
    const data: HealthResponse = await response.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

/**
 * Check backend readiness via GET /ready.
 */
export async function checkBackendReadiness(signal?: AbortSignal): Promise<ReadinessResponse | null> {
  try {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/ready`,
      { method: 'GET', signal },
      4000
    );
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Retrieve cached active research job ID if one is currently pending.
 */
export function getActiveResearchJobId(): string | null {
  if (typeof window === 'undefined') return null;
  return window.sessionStorage?.getItem(ACTIVE_RESEARCH_JOB_KEY) || null;
}

/**
 * Clear cached active research job ID.
 */
export function clearActiveResearchJobId(): void {
  if (typeof window === 'undefined') return;
  window.sessionStorage?.removeItem(ACTIVE_RESEARCH_JOB_KEY);
}

/**
 * Submit an asynchronous durable research query job via POST /api/v1/research/jobs.
 */
export async function submitResearchJob(
  request: RAGQueryRequest,
  signal?: AbortSignal
): Promise<ResearchJobSubmitResponse> {
  if (!request.query || !request.query.trim()) {
    throw new ApiError('Research query must not be empty.', 400);
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/research/jobs`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        query: request.query.trim(),
        retrieval_mode: request.retrieval_mode || 'hybrid',
        top_k: request.top_k || 5,
        min_score: request.min_score,
        temperature: request.temperature,
        max_output_tokens: request.max_output_tokens,
        collection_id: request.collection_id,
        paper_id: request.paper_id,
      }),
      signal,
    },
    15000
  );

  if (!response.ok) {
    let errorMsg = `Server error (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  const data: ResearchJobSubmitResponse = await response.json();
  if (typeof window !== 'undefined') {
    window.sessionStorage?.setItem(ACTIVE_RESEARCH_JOB_KEY, data.job_id);
  }
  return data;
}

/**
 * Poll research job status and result via GET /api/v1/research/jobs/{job_id}.
 */
export async function getResearchJob(
  jobId: string,
  signal?: AbortSignal
): Promise<ResearchJobStatusResponse> {
  if (!jobId || !jobId.trim()) {
    throw new ApiError('Job ID must not be empty.', 400);
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/research/jobs/${encodeURIComponent(jobId.trim())}`,
    {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal,
    },
    10000
  );

  if (!response.ok) {
    let errorMsg = `Server error (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}

/**
 * Execute research question query synchronously via POST /api/v1/research/query (fallback).
 */
export async function queryResearchRAGDirect(
  request: RAGQueryRequest,
  signal?: AbortSignal
): Promise<RAGQueryResponse> {
  if (!request.query || !request.query.trim()) {
    throw new ApiError('Research query must not be empty.', 400);
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/research/query`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        query: request.query.trim(),
        retrieval_mode: request.retrieval_mode || 'hybrid',
        top_k: request.top_k || 5,
        min_score: request.min_score,
        temperature: request.temperature,
        max_output_tokens: request.max_output_tokens,
      }),
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!response.ok) {
    let errorMsg = `Server error (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}

/**
 * Poll an existing research job until completed or failed.
 */
export async function pollResearchJob(
  jobId: string,
  signal?: AbortSignal,
  onProgress?: (status: string, job?: ResearchJobStatusResponse) => void
): Promise<RAGQueryResponse> {
  const maxPollTimeMs = 120000;
  const startTime = Date.now();
  let delayMs = 1000;

  while (Date.now() - startTime < maxPollTimeMs) {
    if (signal?.aborted) {
      throw new ApiError('Request aborted by user.', 499);
    }

    await new Promise((resolve) => setTimeout(resolve, delayMs));

    const jobStatus = await getResearchJob(jobId, signal);
    onProgress?.(jobStatus.status, jobStatus);

    if (jobStatus.status === 'completed' && jobStatus.result) {
      clearActiveResearchJobId();
      return jobStatus.result;
    }

    if (jobStatus.status === 'failed') {
      clearActiveResearchJobId();
      throw new ApiError(
        jobStatus.error || 'Research generation failed in background worker.',
        500
      );
    }

    delayMs = Math.min(2500, Math.floor(delayMs * 1.3));
  }

  clearActiveResearchJobId();
  throw new ApiError('Research synthesis timed out after 120 seconds.', 504);
}

/**
 * Execute research question query using durable background jobs with polling.
 * Prevents Vercel timeouts and serverless request boundary drops.
 */
export async function queryResearchRAG(
  request: RAGQueryRequest,
  signal?: AbortSignal,
  onProgress?: (status: string, job?: ResearchJobStatusResponse) => void
): Promise<RAGQueryResponse> {
  if (!request.query || !request.query.trim()) {
    throw new ApiError('Research query must not be empty.', 400);
  }

  try {
    // 1. Submit durable research job
    const jobSubmit = await submitResearchJob(request, signal);
    const jobId = jobSubmit.job_id;
    onProgress?.('queued');

    // 2. Poll until completed or failed (up to 120s)
    return await pollResearchJob(jobId, signal, onProgress);
  } catch (err: unknown) {
    if (err instanceof ApiError && err.statusCode === 404) {
      // Fallback to direct synchronous query if background job route is unmounted
      return await queryResearchRAGDirect(request, signal);
    }
    throw err;
  }
}

/**
 * Search indexed and benchmark scholarly papers via GET /api/v1/papers.
 */
export async function searchPapers(
  query: string = '',
  limit: number = 10,
  offset: number = 0,
  signal?: AbortSignal
): Promise<PaperSearchResponse> {
  const params = new URLSearchParams();
  if (query.trim()) params.set('query', query.trim());
  params.set('limit', String(limit));
  params.set('offset', String(offset));

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/papers?${params.toString()}`,
    {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!response.ok) {
    let errorMsg = `Failed to search papers (HTTP ${response.status})`;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) errorMsg = errJson.error.message;
    } catch {}
    throw new ApiError(errorMsg, response.status);
  }

  return await response.json();
}

/**
 * Execute direct cross-paper comparison via POST /api/v1/research/compare.
 */
export async function compareResearchPapers(
  request: ResearchCompareRequest,
  signal?: AbortSignal
): Promise<ResearchCompareResponse> {
  if (!request.paper_ids || request.paper_ids.length < 2) {
    throw new ApiError('At least 2 paper IDs are required for comparison.', 400);
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/research/compare`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        paper_ids: request.paper_ids,
        query: request.query || null,
        model: request.model || null,
        temperature: request.temperature || null,
        max_output_tokens: request.max_output_tokens || null,
      }),
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!response.ok) {
    let errorMsg = `Comparison failed (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}

/**
 * Upload and ingest a research paper PDF via POST /api/v1/documents/ingest.
 */
export async function uploadDocument(
  paperId: string,
  file: File,
  signal?: AbortSignal
): Promise<DocumentIngestResponse> {
  if (!paperId || !paperId.trim()) {
    throw new ApiError('paper_id is required for document ingestion.', 400);
  }
  if (!file) {
    throw new ApiError('A PDF file must be provided for upload.', 400);
  }
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    throw new ApiError('Only PDF files are supported.', 400);
  }

  const formData = new FormData();
  formData.append('paper_id', paperId.trim());
  formData.append('file', file);

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/documents/ingest`,
    {
      method: 'POST',
      body: formData,
      signal,
    },
    60000 // Ingestion may take up to 60s for PDF parsing, chunking, and embedding
  );

  if (!response.ok) {
    let errorMsg = `Document ingestion failed (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}

/**
 * Perform structured 7-dimension analysis on a paper via POST /api/v1/papers/analyze.
 */
export async function analyzePaper(
  paperId: string,
  signal?: AbortSignal
): Promise<PaperAnalysisResponse> {
  if (!paperId || !paperId.trim()) {
    throw new ApiError('paper_id is required for paper analysis.', 400);
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/papers/analyze`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        paper_id: paperId.trim(),
      }),
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!response.ok) {
    let errorMsg = `Paper analysis failed (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}

/**
 * Discover evidence-backed research gaps via POST /api/v1/research/gaps.
 */
export async function getResearchGaps(
  paperIds?: string[],
  collectionId?: string,
  signal?: AbortSignal
): Promise<ResearchGapsResponse> {
  const body: Record<string, unknown> = {};
  if (paperIds && paperIds.length > 0) {
    body.paper_ids = paperIds;
  } else if (collectionId) {
    body.collection_id = collectionId;
  } else {
    throw new ApiError('Either paperIds or collectionId must be provided for gap discovery.', 400);
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/research/gaps`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!response.ok) {
    let errorMsg = `Research gaps discovery failed (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}

/**
 * Analyze temporal research trends via POST /api/v1/papers/trends.
 */
export async function getResearchTrends(
  collectionId: string,
  paperIds?: string[],
  timeUnit: string = 'year',
  signal?: AbortSignal
): Promise<ResearchTrendsResponse> {
  const body: Record<string, unknown> = {
    collection_id: collectionId,
    time_unit: timeUnit,
  };
  if (paperIds && paperIds.length > 0) {
    body.paper_ids = paperIds;
  }

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/papers/trends`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!response.ok) {
    let errorMsg = `Research trends analysis failed (HTTP ${response.status})`;
    let details: unknown = null;
    try {
      const errJson = await response.json();
      if (errJson.error?.message) {
        errorMsg = errJson.error.message;
      } else if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
      details = errJson;
    } catch {}
    throw new ApiError(errorMsg, response.status, details);
  }

  return await response.json();
}
