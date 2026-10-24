/**
 * Nexora AI API Client Service.
 * Connects the frontend to the backend REST API endpoints.
 */

import type { RAGQueryRequest, RAGQueryResponse, HealthResponse, ReadinessResponse } from '../types/api';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');
const DEFAULT_TIMEOUT_MS = 20000;

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
    const response = await fetch(url, {
      ...options,
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
 * Execute research question query via POST /api/v1/research/query.
 */
export async function queryResearchRAG(
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
    } catch {
      // Body not JSON
    }
    throw new ApiError(errorMsg, response.status, details);
  }

  const data: RAGQueryResponse = await response.json();
  return data;
}
