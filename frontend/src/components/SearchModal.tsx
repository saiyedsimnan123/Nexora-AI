import React, { useState, useEffect, useCallback, useId } from 'react';
import {
  X,
  Search,
  Sparkles,
  CheckCircle2,
  FileText,
  AlertCircle,
  RefreshCw,
  Layers,
  ShieldCheck,
  Loader2,
  Database,
  ExternalLink
} from 'lucide-react';
import { queryResearchRAG } from '../services/api';
import type { RAGQueryResponse, AttributedSource } from '../types/api';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  query: string;
}

// Curated reference scenario for offline demonstration / fallback
const CURATED_DEMO_SCENARIO: RAGQueryResponse = {
  query: 'How do FlashAttention IO optimizations achieve 4.3x speedup on A100 GPUs?',
  answer:
    'Standard transformer self-attention stores intermediate N x N attention matrices in High Bandwidth Memory (HBM), scaling quadratically as O(N^2) with sequence length [E2]. FlashAttention eliminates this memory bandwidth bottleneck by tiling the computation into GPU SRAM blocks and incrementally computing online softmax without materializing full attention matrices, achieving a verified 2.4x to 4.3x wall-clock speedup across BERT-large and GPT-style architectures [E1].',
  is_grounded: true,
  is_mock: true,
  model: 'curated-offline-benchmark',
  status: 'success',
  error: null,
  retrieval_mode: 'hybrid',
  evidence_count: 2,
  sources: [
    {
      label: '[E1]',
      document_id: 'doc_dao2022_flashattention',
      chunk_id: 'chunk_dao_io_sram_04',
      paper_id: 'paper_arxiv_2205_14135',
      paper_title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)',
      score: 0.0320,
      text: 'We evaluate BERT-large on MLPerf 1.1... FlashAttention yields a 2.4x speedup over standard HuggingFace attention while computing exact attention without approximation. On GPT-2 models with sequence length up to 4K, FlashAttention achieves up to 4.3x speedup on A100 GPUs by optimizing SRAM read/write access.',
      text_excerpt: 'FlashAttention achieves up to 4.3x speedup on A100 GPUs by optimizing SRAM read/write access without materializing intermediate N x N attention matrices.',
      metadata: { arxiv_id: '2205.14135', page: 4 },
      page_number: 4,
    },
    {
      label: '[E2]',
      document_id: 'doc_vaswani2017_transformer',
      chunk_id: 'chunk_vaswani_complexity_05',
      paper_id: 'paper_arxiv_1706_03762',
      paper_title: 'Attention Is All You Need (Vaswani et al., 2017)',
      score: 0.0315,
      text: 'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.',
      text_excerpt: 'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.',
      metadata: { arxiv_id: '1706.03762', page: 6 },
      page_number: 6,
    },
  ],
  cited_sources: [],
};
CURATED_DEMO_SCENARIO.cited_sources = [...CURATED_DEMO_SCENARIO.sources];

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, query }) => {
  const defaultQuery = 'How do FlashAttention IO optimizations achieve 4.3x speedup on A100 GPUs?';
  const initialQuery = query.trim() || defaultQuery;
  const [activeQuery, setActiveQuery] = useState<string>(initialQuery);
  const [prevQuery, setPrevQuery] = useState<string>(query);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [liveResponse, setLiveResponse] = useState<RAGQueryResponse | null>(null);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);
  const [selectedCitationLabel, setSelectedCitationLabel] = useState<string>('[E1]');
  const [pipelineStage, setPipelineStage] = useState<number>(5);
  const titleId = useId();

  // Sync state if query prop changes from parent
  if (query !== prevQuery) {
    setPrevQuery(query);
    setActiveQuery(query.trim() || defaultQuery);
  }

  // Execute live RAG query against the backend API
  const handleExecuteSearch = useCallback(async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed) return;

    setIsLoading(true);
    setError(null);
    setIsDemoMode(false);
    setPipelineStage(1);

    const stageTimer1 = setTimeout(() => setPipelineStage(2), 250);
    const stageTimer2 = setTimeout(() => setPipelineStage(3), 550);
    const stageTimer3 = setTimeout(() => setPipelineStage(4), 850);

    try {
      const data = await queryResearchRAG({
        query: trimmed,
        retrieval_mode: 'hybrid',
        top_k: 5,
      });

      setPipelineStage(5);
      setLiveResponse(data);

      // Select first citation label if available
      const firstLabel = data.sources?.[0]?.label || data.cited_sources?.[0]?.label || '[E1]';
      setSelectedCitationLabel(firstLabel);
    } catch (err: unknown) {
      setPipelineStage(0);
      const message = err instanceof Error ? err.message : 'Unknown network error.';
      setError(message);
    } finally {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      clearTimeout(stageTimer3);
      setIsLoading(false);
    }
  }, []);

  // Trigger search on modal opening
  useEffect(() => {
    if (!isOpen) return;
    const targetQuery = query.trim() || defaultQuery;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (!cancelled) {
        handleExecuteSearch(targetQuery);
      }
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [isOpen, query, defaultQuery, handleExecuteSearch]);

  // Keyboard accessibility
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Active dataset for rendering
  const currentData: RAGQueryResponse = isDemoMode
    ? CURATED_DEMO_SCENARIO
    : liveResponse || CURATED_DEMO_SCENARIO;

  const currentSources: AttributedSource[] = currentData.sources && currentData.sources.length > 0
    ? currentData.sources
    : currentData.cited_sources || [];

  const selectedSource =
    currentSources.find((s) => s.label === selectedCitationLabel) || currentSources[0] || null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto"
    >
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-[#04060c]/85 backdrop-blur-md transition-opacity"
      />

      {/* Modal Dialog Box */}
      <div className="relative w-full max-w-3xl glass-panel-glow bg-[#080d21] rounded-2xl border border-cyan-500/40 shadow-2xl p-6 sm:p-8 z-10 space-y-6 animate-in fade-in zoom-in-95 duration-200 max-h-[90vh] overflow-y-auto">

        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-400/40 flex items-center justify-center text-cyan-300">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 id={titleId} className="text-base font-bold text-white">
                  Research Pipeline Execution
                </h3>
                {isDemoMode ? (
                  <span className="text-[10px] font-mono uppercase bg-purple-950/70 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded">
                    Curated Scenario
                  </span>
                ) : (
                  <span className="text-[10px] font-mono uppercase bg-cyan-950/70 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    Live Grounded RAG
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 font-mono">
                5-Stage Evidence-Grounded Verification Flow
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/60 transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search Query Input Bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleExecuteSearch(activeQuery);
          }}
          className="space-y-1.5"
        >
          <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-slate-400">
            <span>Active Research Inquiry:</span>
            {isLoading && (
              <span className="text-cyan-400 flex items-center gap-1 normal-case font-sans text-xs">
                <Loader2 className="w-3 h-3 animate-spin" />
                Querying hybrid pipeline...
              </span>
            )}
          </div>

          <div className="relative flex items-center bg-slate-900/90 border border-white/10 rounded-xl p-1.5 shadow-inner">
            <div className="pl-3 pr-2 text-cyan-400">
              <Search className="w-4 h-4" />
            </div>
            <input
              type="text"
              value={activeQuery}
              onChange={(e) => setActiveQuery(e.target.value)}
              placeholder="Ask a scientific question..."
              className="w-full bg-transparent text-sm text-white placeholder-slate-400 focus:outline-none py-1.5 px-1 font-sans min-w-0"
            />
            <button
              type="submit"
              disabled={isLoading || !activeQuery.trim()}
              className="shrink-0 flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-medium text-xs text-slate-900 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 transition-all font-mono"
            >
              {isLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
              <span>Query</span>
            </button>
          </div>
        </form>

        {/* 5-Stage Pipeline Progress Indicator */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Pipeline Architecture:</span>
            <span className={isLoading ? 'text-cyan-300' : 'text-emerald-400 font-semibold'}>
              {isLoading
                ? `Stage ${pipelineStage} / 5 In Progress...`
                : 'Stage 5 / 5 Execution Complete'}
            </span>
          </div>

          <div className="grid grid-cols-5 gap-1.5 sm:gap-2 text-center text-[9px] sm:text-[10px] font-mono">
            <div className={`p-2 rounded border transition-all ${
              pipelineStage >= 1
                ? 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300'
                : 'bg-slate-900/40 border-white/5 text-slate-500'
            }`}>
              1. Intent
            </div>
            <div className={`p-2 rounded border transition-all ${
              pipelineStage >= 2
                ? 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300'
                : 'bg-slate-900/40 border-white/5 text-slate-500'
            }`}>
              2. Hybrid RRF
            </div>
            <div className={`p-2 rounded border transition-all ${
              pipelineStage >= 3
                ? 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300'
                : 'bg-slate-900/40 border-white/5 text-slate-500'
            }`}>
              3. Rerank
            </div>
            <div className={`p-2 rounded border transition-all ${
              pipelineStage >= 4
                ? 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300'
                : 'bg-slate-900/40 border-white/5 text-slate-500'
            }`}>
              4. Attributed RAG
            </div>
            <div className={`p-2 rounded border transition-all ${
              pipelineStage >= 5
                ? 'bg-emerald-950/70 border-emerald-500/40 text-emerald-300 font-semibold shadow-[0_0_10px_rgba(52,211,153,0.2)]'
                : 'bg-slate-900/40 border-white/5 text-slate-500'
            }`}>
              5. Claim Grounding
            </div>
          </div>
        </div>

        {/* Error / Offline Alert State */}
        {error && !isDemoMode && (
          <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-500/30 space-y-3 animate-in fade-in duration-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-xs font-bold text-amber-200 font-mono">
                  Backend Unreachable or Request Failed
                </h4>
                <p className="text-xs text-amber-200/90 leading-relaxed font-sans">
                  {error}
                </p>
              </div>
            </div>

            <div className="bg-black/50 p-3 rounded-lg border border-amber-500/20 text-xs font-mono text-slate-300 space-y-1">
              <div className="text-[11px] text-slate-400">To start the local offline Nexora backend:</div>
              <code className="text-cyan-300 block select-all">
                set PYTHONPATH=src &amp;&amp; python -m uvicorn nexora.api.app:app --host 127.0.0.1 --port 8000
              </code>
            </div>

            <div className="flex items-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => handleExecuteSearch(activeQuery || defaultQuery)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-400 text-slate-900 text-xs font-semibold hover:bg-amber-300 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Live Query</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setError(null);
                  setIsDemoMode(true);
                  setSelectedCitationLabel('[E1]');
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg glass-panel text-slate-200 hover:text-white text-xs border border-white/10 transition-colors"
              >
                <Layers className="w-3.5 h-3.5 text-purple-400" />
                <span>View Curated Demo Scenario</span>
              </button>
            </div>
          </div>
        )}

        {/* Loading Skeleton */}
        {isLoading && (
          <div className="p-6 rounded-xl bg-[#04060d] border border-cyan-500/25 space-y-4 animate-pulse">
            <div className="h-4 bg-cyan-950/60 rounded w-3/4" />
            <div className="h-4 bg-slate-900 rounded w-full" />
            <div className="h-4 bg-slate-900 rounded w-5/6" />
            <div className="pt-2 flex gap-2">
              <div className="h-6 w-16 bg-cyan-950/80 rounded" />
              <div className="h-6 w-16 bg-cyan-950/80 rounded" />
            </div>
          </div>
        )}

        {/* Synthesized Research Answer with [E#] Citations */}
        {!isLoading && (!error || isDemoMode) && (
          <div className="space-y-4">

            {/* Answer Box */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-slate-400">
                <span>Attributed Answer Synthesis:</span>
                <div className="flex items-center gap-2">
                  <span className="text-slate-500 font-mono">Model: {currentData.model}</span>
                  <span className="text-slate-500">•</span>
                  <span className="text-emerald-400 flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    {currentData.is_grounded ? 'Grounded in Evidence' : 'Ungrounded'}
                  </span>
                </div>
              </div>

              <div className="p-5 rounded-xl bg-[#04060d] border border-cyan-500/25 space-y-3 text-xs sm:text-sm text-slate-200 leading-relaxed font-sans">
                <p className="whitespace-pre-wrap">{currentData.answer}</p>
              </div>
            </div>

            {/* Citation Switcher Chips */}
            {currentSources.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>Available Citations ({currentSources.length}):</span>
                  <span className="text-slate-500">Click to inspect chunk excerpt</span>
                </div>

                <div className="flex flex-wrap gap-2">
                  {currentSources.map((source) => {
                    const isSelected = source.label === selectedCitationLabel;
                    return (
                      <button
                        key={source.chunk_id || source.label}
                        type="button"
                        onClick={() => setSelectedCitationLabel(source.label)}
                        className={`text-xs px-3 py-1.5 rounded-lg border font-mono transition-all flex items-center gap-2 ${
                          isSelected
                            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_12px_rgba(0,245,212,0.3)]'
                            : 'bg-slate-900/60 border-white/10 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <span className="font-bold">{source.label}</span>
                        <span className="text-[11px] truncate max-w-[200px]">
                          {source.paper_title.split(':')[0]}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Selected Citation Inspector */}
            {selectedSource ? (
              <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2 text-xs font-mono">
                <div className="flex items-center justify-between text-slate-400">
                  <span className="text-cyan-300 flex items-center gap-1.5 font-semibold">
                    <FileText className="w-3.5 h-3.5" />
                    Inspecting Citation {selectedSource.label}:
                  </span>
                  <span className="text-emerald-400 flex items-center gap-1 font-sans">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Retrieved Evidence Match
                  </span>
                </div>

                <p className="text-slate-200 italic bg-black/40 p-3 rounded border border-white/[0.04] leading-relaxed">
                  "{selectedSource.text_excerpt || selectedSource.text}"
                </p>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-slate-400 pt-1 gap-1">
                  <span className="truncate max-w-md">
                    <strong>Source:</strong> {selectedSource.paper_title}
                  </span>
                  <div className="flex items-center gap-3 shrink-0">
                    {selectedSource.score > 0 && (
                      <span className="text-cyan-400 font-mono">
                        Score: {selectedSource.score.toFixed(4)}
                      </span>
                    )}
                    <span className="text-slate-500">
                      Chunk: {selectedSource.chunk_id.slice(0, 16)}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-slate-900/40 border border-white/5 text-xs text-slate-400 text-center font-mono">
                No individual citation selected.
              </div>
            )}

            {/* Scientific Rigor Disclaimer */}
            <div className="p-3 rounded-lg bg-slate-900/40 border border-white/[0.06] flex items-start gap-2.5 text-xs text-slate-400">
              <Database className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
              <p>
                <strong>Scientific Rigor:</strong> Citations reflect passages directly retrieved from peer-reviewed literature. Citation of an excerpt indicates lexical or semantic relevance; researchers should evaluate experimental methodology and limitations in the cited paper.
              </p>
            </div>

          </div>
        )}

        {/* Modal Action Footer */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3 border-t border-white/[0.08]">
          <div className="text-xs text-slate-500 font-mono flex items-center gap-2">
            <span>Retrieval: Hybrid BM25 + Dense</span>
            <span>•</span>
            <span>Top-5 Candidates</span>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
            <button
              onClick={onClose}
              className="px-5 py-2.5 rounded-xl font-medium text-xs sm:text-sm text-slate-300 hover:text-white glass-panel border-white/10"
            >
              Close
            </button>
            <a
              href="https://github.com/saiyedsimnan123/Nexora-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-xs sm:text-sm text-slate-900 bg-cyan-400 hover:bg-cyan-300 font-semibold shadow-[0_0_15px_rgba(0,245,212,0.3)]"
            >
              <span>GitHub Specification</span>
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>

      </div>
    </div>
  );
};
