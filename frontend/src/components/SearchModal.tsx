import React, { useState, useEffect, useCallback, useId } from 'react';
import {
  X,
  Search,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  ShieldCheck,
  Loader2,
  GitCompare,
  HelpCircle,
  Scale,
  BookOpen,
  ArrowRight
} from 'lucide-react';
import { queryResearchRAG } from '../services/api';
import type {
  RAGQueryResponse,
  AttributedSource
} from '../types/api';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  query: string;
  onComparePapers?: (paperIds: string[]) => void;
}

// Curated reference scenario for offline demonstration / fallback
const CURATED_DEMO_SCENARIO: RAGQueryResponse = {
  query: 'How do FlashAttention IO optimizations achieve 4.3x speedup on A100 GPUs?',
  answer:
    '### Direct Answer\n\nStandard Transformer self-attention stores intermediate $N \\times N$ attention matrices in High Bandwidth Memory (HBM), scaling quadratically as $O(N^2)$ with sequence length [E2]. FlashAttention eliminates this memory bandwidth bottleneck by tiling the computation into GPU on-chip SRAM blocks and incrementally computing online softmax without materializing full attention matrices [E1]. This IO-aware execution achieves verified 2.4x to 4.3x wall-clock speedups across BERT-large and GPT-style architectures on A100 GPUs while computing mathematically exact attention [E1].',
  is_grounded: true,
  is_mock: true,
  model: 'curated-offline-benchmark',
  status: 'success',
  error: null,
  retrieval_mode: 'hybrid',
  evidence_count: 2,
  conclusion: 'Demonstrated improvement',
  conclusion_reasoning:
    'Empirical evaluations verify a 2.4x to 4.3x wall-clock speedup on A100 GPUs by optimizing SRAM read/write access without materializing intermediate quadratic matrices.',
  sources: [
    {
      label: '[E1]',
      document_id: 'doc_dao2022_flashattention',
      chunk_id: 'chunk_dao_io_sram_04',
      paper_id: 'a1877e5e-6536-51f6-ae8e-ca92c6bf7be5',
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
      paper_id: '541c4d4f-4d6b-592f-981f-7e9b04f3ce37',
      paper_title: 'Attention Is All You Need (Vaswani et al., 2017)',
      score: 0.0315,
      text: 'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.',
      text_excerpt: 'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.',
      metadata: { arxiv_id: '1706.03762', page: 6 },
      page_number: 6,
    },
  ],
  cited_sources: [],
  evidence_by_paper: [
    {
      paper_id: 'a1877e5e-6536-51f6-ae8e-ca92c6bf7be5',
      paper_title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
      findings: [
        'FlashAttention achieves up to 4.3x speedup on A100 GPUs by optimizing SRAM read/write access without intermediate matrix materialization.'
      ],
      excerpts: [
        'FlashAttention yields a 2.4x speedup over standard HuggingFace attention while computing exact attention without approximation. On GPT-2 models with sequence length up to 4K, FlashAttention achieves up to 4.3x speedup on A100 GPUs.'
      ],
      citations: ['[E1]'],
      sources: []
    },
    {
      paper_id: '541c4d4f-4d6b-592f-981f-7e9b04f3ce37',
      paper_title: 'Attention Is All You Need',
      findings: [
        'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.'
      ],
      excerpts: [
        'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.'
      ],
      citations: ['[E2]'],
      sources: []
    }
  ],
  comparison_of_findings: {
    agreements: [
      'Both papers analyze self-attention computational scaling and memory access characteristics.'
    ],
    differences: [
      'Attention Is All You Need establishes the foundational multi-head self-attention formulation with quadratic complexity, whereas FlashAttention redesigns the execution algorithm to optimize GPU SRAM IO transfers.'
    ],
    relationships: [
      'FlashAttention directly executes exact multi-head attention without approximating the underlying mathematical formulation introduced in Attention Is All You Need.'
    ],
    direct_evidence: [
      'FlashAttention: Achieves 2.4x to 4.3x speedup on A100 GPUs by optimizing SRAM reads/writes.',
      'Attention Is All You Need: Standard self-attention yields O(N^2) memory footprint per layer.'
    ],
    interpretation: [
      'Empirical wall-clock speedup directly correlates with reduced High Bandwidth Memory accesses rather than mathematical approximations.'
    ]
  },
  limitations_and_conflicts: {
    limitations: [
      'FlashAttention gains require GPU architectures with high-performance on-chip SRAM bandwidth (e.g. NVIDIA Ampere A100 GPUs).',
      'Standard Transformer self-attention is bandwidth-bound on sequences exceeding 2K tokens.'
    ],
    conflicts: [
      'No direct contradiction detected: FlashAttention preserves mathematical equivalence while resolving hardware memory bandwidth bottlenecks.'
    ],
    evidence_gaps: [
      'Inference energy consumption and edge-device hardware deployment metrics were not reported in retrieved evidence.'
    ],
    insufficient_evidence_notes: 'All claims are strictly bounded by retrieved excerpts; unmentioned dimensions are flagged as not reported.'
  }
};
CURATED_DEMO_SCENARIO.cited_sources = [...CURATED_DEMO_SCENARIO.sources];

export const SearchModal: React.FC<SearchModalProps> = ({
  isOpen,
  onClose,
  query,
  onComparePapers
}) => {
  const defaultQuery = 'How do FlashAttention IO optimizations achieve 4.3x speedup on A100 GPUs?';
  const initialQuery = query.trim() || defaultQuery;
  const [activeQuery, setActiveQuery] = useState<string>(initialQuery);
  const [prevQuery, setPrevQuery] = useState<string>(query);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [liveResponse, setLiveResponse] = useState<RAGQueryResponse | null>(null);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);
  const [selectedCitationLabel, setSelectedCitationLabel] = useState<string>('[E1]');
  const [activeTab, setActiveTab] = useState<'answer' | 'evidence' | 'comparison' | 'limitations'>('answer');
  const titleId = useId();

  // Sync state if query prop changes from parent
  if (query !== prevQuery) {
    setPrevQuery(query);
    setActiveQuery(query.trim() || defaultQuery);
  }

  // Execute live RAG query against backend API
  const handleExecuteSearch = useCallback(async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed) return;

    setIsLoading(true);
    setError(null);
    setIsDemoMode(false);

    try {
      const data = await queryResearchRAG({
        query: trimmed,
        retrieval_mode: 'hybrid',
        top_k: 5,
      });

      setLiveResponse(data);
      const firstLabel = data.sources?.[0]?.label || data.cited_sources?.[0]?.label || '[E1]';
      setSelectedCitationLabel(firstLabel);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown network error.';
      setError(message);
    } finally {
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

  const evidenceByPaper = currentData.evidence_by_paper || [];
  const compFindings = currentData.comparison_of_findings;
  const limConflicts = currentData.limitations_and_conflicts;
  const conclusion = currentData.conclusion || 'Demonstrated improvement';
  const takeaway = currentData.takeaway || currentData.conclusion_reasoning || '';

  // Extract paper IDs for direct comparison handoff
  const distinctPaperIds = Array.from(
    new Set(
      evidenceByPaper
        .map((p) => p.paper_id)
        .filter((id): id is string => Boolean(id))
    )
  );

  const handleLaunchCompare = () => {
    if (onComparePapers && distinctPaperIds.length >= 2) {
      onComparePapers(distinctPaperIds.slice(0, 2));
      onClose();
    } else if (onComparePapers) {
      onComparePapers([]);
      onClose();
    }
  };

  // Format main answer text into clean paragraphs
  const renderFormattedAnswer = (rawText: string) => {
    const cleanText = rawText.replace(/^### Direct Answer\s*/i, '').trim();
    const paragraphs = cleanText.split(/\n\n+/).filter((p) => p.trim());

    return (
      <div className="space-y-3.5 text-sm sm:text-base text-slate-200 leading-relaxed font-sans">
        {paragraphs.map((p, idx) => {
          // Highlight [E#] citation pills interactively
          const parts = p.split(/(\[E\d+\])/g);
          return (
            <p key={idx}>
              {parts.map((part, pIdx) => {
                if (/^\[E\d+\]$/.test(part)) {
                  const isSelected = selectedCitationLabel === part;
                  return (
                    <button
                      key={pIdx}
                      onClick={() => setSelectedCitationLabel(part)}
                      className={`inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-xs font-mono font-bold transition-all ${
                        isSelected
                          ? 'bg-cyan-400 text-slate-950 shadow-[0_0_10px_#00f5d4]'
                          : 'bg-cyan-950/70 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-900/60'
                      }`}
                    >
                      {part}
                    </button>
                  );
                }
                return part;
              })}
            </p>
          );
        })}
      </div>
    );
  };

  // Render conclusion badge styling
  const renderConclusionBadge = (conc: string) => {
    let colorClasses = 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30';
    if (conc.includes('Demonstrated improvement')) {
      colorClasses = 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40';
    } else if (conc.includes('Conflicting evidence')) {
      colorClasses = 'bg-purple-500/15 text-purple-300 border-purple-500/40';
    } else if (conc.includes('Limited evidence')) {
      colorClasses = 'bg-amber-500/15 text-amber-300 border-amber-500/40';
    } else if (conc.includes('Insufficient evidence')) {
      colorClasses = 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }

    return (
      <span
        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono uppercase tracking-wider border ${colorClasses}`}
      >
        <ShieldCheck className="w-3.5 h-3.5" />
        {conc}
      </span>
    );
  };

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
      <div className="relative w-full max-w-4xl glass-panel-glow bg-[#080d21] rounded-2xl border border-cyan-500/40 shadow-2xl p-6 sm:p-8 z-10 space-y-6 animate-in fade-in zoom-in-95 duration-200 max-h-[92vh] overflow-y-auto">

        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-400/40 flex items-center justify-center text-cyan-300">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 id={titleId} className="text-base font-bold text-white">
                  Research Intelligence Answer Experience
                </h3>
                {isDemoMode ? (
                  <span className="text-[10px] font-mono uppercase bg-purple-950/70 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded">
                    Curated Scenario
                  </span>
                ) : (
                  <span className="text-[10px] font-mono uppercase bg-cyan-950/70 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                    Live Grounded RAG
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 font-mono">
                5-Section Evidence-Grounded Research Synthesis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {onComparePapers && (
              <button
                onClick={handleLaunchCompare}
                className="px-3 py-1.5 rounded-lg text-xs font-mono uppercase tracking-wider text-cyan-300 hover:text-white bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 transition-colors flex items-center gap-1.5"
                title="Open Direct Paper Comparison"
              >
                <GitCompare className="w-3.5 h-3.5 text-cyan-400" />
                Compare Papers
              </button>
            )}

            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/60 transition-colors"
              aria-label="Close dialog"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
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
            <span>Research Question:</span>
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
              className="px-4 py-1.5 bg-gradient-to-r from-cyan-400 to-teal-300 text-slate-900 text-xs font-bold rounded-lg hover:opacity-95 transition-opacity disabled:opacity-50 flex items-center gap-1 font-sans shrink-0"
            >
              <span>Query</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </form>

        {/* Error Notification Bar */}
        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-center justify-between gap-3 text-xs text-rose-300">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => {
                setIsDemoMode(true);
                setError(null);
              }}
              className="px-2.5 py-1 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 rounded text-[11px] font-mono shrink-0 transition-colors"
            >
              Load Curated Scenario
            </button>
          </div>
        )}

        {/* Section Navigation Tabs */}
        <div className="flex items-center gap-1 border-b border-white/[0.08] pb-1 overflow-x-auto text-xs font-mono uppercase tracking-wider">
          <button
            onClick={() => setActiveTab('answer')}
            className={`px-3 py-2 rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
              activeTab === 'answer'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            Direct Answer
          </button>
          <button
            onClick={() => setActiveTab('evidence')}
            className={`px-3 py-2 rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
              activeTab === 'evidence'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Evidence by Paper ({evidenceByPaper.length || currentSources.length})
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            className={`px-3 py-2 rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
              activeTab === 'comparison'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'
            }`}
          >
            <Scale className="w-3.5 h-3.5" />
            Comparison of Findings
          </button>
          <button
            onClick={() => setActiveTab('limitations')}
            className={`px-3 py-2 rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
              activeTab === 'limitations'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'
            }`}
          >
            <HelpCircle className="w-3.5 h-3.5" />
            Limitations & Conflicts
          </button>
        </div>

        {/* Tab 1: Direct Answer */}
        {activeTab === 'answer' && (
          <div className="space-y-6 animate-in fade-in duration-200">
            {/* Direct Answer Container */}
            <div className="p-5 sm:p-6 rounded-2xl bg-slate-900/90 border border-white/10 space-y-4">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                  Synthesized Direct Answer
                </span>
                {renderConclusionBadge(conclusion)}
              </div>

              {renderFormattedAnswer(currentData.answer)}

              {/* Evidence-Grounded Takeaway Banner */}
              {takeaway && (
                <div className="p-3.5 rounded-xl bg-[#09102c] border border-cyan-500/20 text-xs text-slate-300 space-y-1">
                  <span className="font-mono uppercase tracking-wider text-cyan-400 font-bold text-[11px] block">
                    Evidence-Grounded Takeaway:
                  </span>
                  <p className="leading-relaxed">{takeaway}</p>
                </div>
              )}
            </div>

            {/* Interactive Selected Citation Inspector */}
            {selectedSource && (
              <div className="p-4 rounded-xl bg-slate-900/60 border border-cyan-500/30 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/70 border border-cyan-500/40 px-2 py-0.5 rounded">
                      {selectedSource.label}
                    </span>
                    <span className="text-xs font-semibold text-white truncate max-w-md">
                      {selectedSource.paper_title}
                    </span>
                  </div>
                  {selectedSource.page_number && (
                    <span className="text-[11px] font-mono text-slate-400">
                      Page {selectedSource.page_number}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-300 italic bg-black/30 p-2.5 rounded-lg border border-white/[0.05] leading-relaxed">
                  "{selectedSource.text_excerpt}"
                </p>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Evidence by Paper (Strict Separation) */}
        {activeTab === 'evidence' && (
          <div className="space-y-4 animate-in fade-in duration-200">
            <div className="text-xs text-slate-400 flex items-center justify-between">
              <span>Evidence excerpts are strictly partitioned by paper. No mixing across studies.</span>
              {distinctPaperIds.length >= 2 && onComparePapers && (
                <button
                  onClick={handleLaunchCompare}
                  className="text-cyan-400 hover:text-cyan-300 underline font-mono text-[11px] flex items-center gap-1"
                >
                  <GitCompare className="w-3.5 h-3.5" />
                  Compare these papers side-by-side &rarr;
                </button>
              )}
            </div>

            {evidenceByPaper.length > 0 ? (
              evidenceByPaper.map((paperGroup, pIdx) => (
                <div
                  key={pIdx}
                  className="p-5 rounded-2xl bg-slate-900/80 border border-white/10 space-y-3"
                >
                  <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                    <h4 className="text-sm font-bold text-white flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-cyan-400" />
                      {paperGroup.paper_title}
                    </h4>
                    <div className="flex items-center gap-1.5">
                      {paperGroup.citations.map((c) => (
                        <span
                          key={c}
                          className="text-[11px] font-mono font-bold text-cyan-300 bg-cyan-950/70 border border-cyan-500/30 px-1.5 py-0.5 rounded"
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Supporting Evidence Excerpts */}
                  {paperGroup.sources && paperGroup.sources.length > 0 ? (
                    <div className="space-y-3 pt-1">
                      <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-bold block">
                        Supporting Evidence:
                      </span>
                      {paperGroup.sources.map((src, sIdx) => (
                        <div key={sIdx} className="space-y-1.5 p-3 rounded-xl bg-black/30 border border-white/[0.05]">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono font-bold text-cyan-300 bg-cyan-950/70 border border-cyan-500/30 px-1.5 py-0.5 rounded">
                              {src.label}
                            </span>
                            {paperGroup.findings[sIdx] && (
                              <span className="text-xs text-slate-200 font-medium">
                                {paperGroup.findings[sIdx]}
                              </span>
                            )}
                          </div>
                          <blockquote className="text-xs text-slate-300 italic leading-relaxed pl-1">
                            "{src.text_excerpt || src.text}"
                          </blockquote>
                        </div>
                      ))}
                    </div>
                  ) : paperGroup.excerpts.length > 0 ? (
                    <div className="space-y-2 pt-1">
                      <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-bold block">
                        Supporting Excerpts:
                      </span>
                      {paperGroup.excerpts.map((exc, eIdx) => (
                        <blockquote key={eIdx} className="text-xs text-slate-300 italic bg-black/40 p-3 rounded-xl border border-white/[0.05] leading-relaxed">
                          "{exc}"
                        </blockquote>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))
            ) : (
              // Fallback to sources list
              currentSources.map((source) => (
                <div
                  key={source.chunk_id}
                  className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/70 border border-cyan-500/40 px-2 py-0.5 rounded">
                      {source.label}
                    </span>
                    <span className="text-xs font-semibold text-white">
                      {source.paper_title}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 italic bg-black/30 p-2.5 rounded-lg border border-white/[0.05]">
                    "{source.text_excerpt}"
                  </p>
                </div>
              ))
            )}
          </div>
        )}

        {/* Tab 3: Comparison of Findings */}
        {activeTab === 'comparison' && (
          <div className="space-y-5 animate-in fade-in duration-200">
            {compFindings ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Agreements Card */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-bold flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                    Agreements Across Papers
                  </h4>
                  <ul className="space-y-1.5 text-xs text-slate-300">
                    {compFindings.agreements.map((item, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <span className="text-cyan-400">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Differences Card */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold flex items-center gap-1.5">
                    <Scale className="w-4 h-4 text-purple-400" />
                    Distinct Approaches & Differences
                  </h4>
                  <ul className="space-y-1.5 text-xs text-slate-300">
                    {compFindings.differences.map((item, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <span className="text-purple-400">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Direct Evidence Claims */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-emerald-300 font-bold flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    Supported by Direct Evidence
                  </h4>
                  <ul className="space-y-1.5 text-xs text-slate-300">
                    {compFindings.direct_evidence.map((item, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <span className="text-emerald-400">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Interpretation vs Extrapolation */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-amber-300 font-bold flex items-center gap-1.5">
                    <HelpCircle className="w-4 h-4 text-amber-400" />
                    Interpretative Distinctions
                  </h4>
                  <ul className="space-y-1.5 text-xs text-slate-300">
                    {compFindings.interpretation.map((item, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <span className="text-amber-400">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Claims Not Established in Retrieved Evidence */}
                {compFindings.unestablished_claims && compFindings.unestablished_claims.length > 0 && (
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-rose-500/30 space-y-2">
                    <h4 className="text-xs font-mono uppercase tracking-wider text-rose-300 font-bold flex items-center gap-1.5">
                      <AlertCircle className="w-4 h-4 text-rose-400" />
                      Claims Not Established in Evidence
                    </h4>
                    <ul className="space-y-1.5 text-xs text-slate-300">
                      {compFindings.unestablished_claims.map((item, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-rose-400">•</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-400 italic bg-slate-900/60 rounded-xl border border-white/10">
                Single-paper context retrieved. Cross-paper comparative synthesis activates when multiple studies are queried.
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Limitations and Conflicting Evidence */}
        {activeTab === 'limitations' && (
          <div className="space-y-4 animate-in fade-in duration-200">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Paper Limitations */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                <h4 className="text-xs font-mono uppercase tracking-wider text-amber-300 font-bold flex items-center gap-1.5">
                  <HelpCircle className="w-4 h-4 text-amber-400" />
                  Paper-Supported Limitations
                </h4>
                <ul className="space-y-1.5 text-xs text-slate-300">
                  {limConflicts?.limitations?.map((item, i) => (
                    <li key={i} className="flex items-start gap-1.5">
                      <span className="text-amber-400">•</span>
                      <span>{item}</span>
                    </li>
                  )) || (
                    <li className="text-slate-500 italic">No specific architectural limitations reported in retrieved excerpts.</li>
                  )}
                </ul>
              </div>

              {/* Conflicting Findings or Trade-Offs */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                <h4 className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold flex items-center gap-1.5">
                  <Scale className="w-4 h-4 text-purple-400" />
                  Trade-Offs & Potential Conflicts
                </h4>
                <ul className="space-y-1.5 text-xs text-slate-300">
                  {limConflicts?.conflicts?.map((item, i) => (
                    <li key={i} className="flex items-start gap-1.5">
                      <span className="text-purple-400">•</span>
                      <span>{item}</span>
                    </li>
                  )) || (
                    <li className="text-slate-500 italic">No conflicting findings detected across retrieved evidence.</li>
                  )}
                </ul>
              </div>
            </div>

            {/* Evidence Gaps Banner */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-white/10 space-y-2">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400 font-bold block">
                Evidence Gaps & Unreported Aspects:
              </span>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">
                {limConflicts?.insufficient_evidence_notes ||
                  'All synthesized claims are strictly bounded by retrieved document excerpts. Unmentioned experimental configurations, energy footprints, or downstream tasks are truthfully flagged as not reported rather than fabricated.'}
              </p>
            </div>
          </div>
        )}

        {/* Modal Footer */}
        <div className="pt-3 border-t border-white/[0.08] flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-3">
            <span>Model: <span className="text-slate-200 font-mono">{currentData.model}</span></span>
            <span>•</span>
            <span>Evidence Chunks: <span className="text-cyan-300 font-mono">{currentSources.length}</span></span>
          </div>

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors"
          >
            Close View
          </button>
        </div>

      </div>
    </div>
  );
};
