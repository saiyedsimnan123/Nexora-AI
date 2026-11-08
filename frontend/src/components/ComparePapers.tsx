import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  GitCompare,
  Search,
  Plus,
  Trash2,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Sparkles,
  Scale,
  RefreshCw,
  ShieldCheck,
  FileText,
  UploadCloud,
  Loader2,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { searchPapers, compareResearchPapers, getResearchGaps } from '../services/api';
import type {
  PaperItem,
  ResearchCompareResponse,
  ComparisonTableData,
  ComparisonRow,
  ResearchGapsResponse
} from '../types/api';

interface ComparePapersProps {
  initialPaperIds?: string[];
  onBackToResearch?: () => void;
  onAnalyzePaper?: (paperId: string, title?: string) => void;
  onOpenUpload?: () => void;
}

// Curated benchmark papers for quick 1-click selection
const QUICK_BENCHMARK_PAPERS = [
  {
    id: '454c9c86-28c0-5fa5-a15d-cbc6a2e72234',
    canonicalKey: 'arxiv_1706.03762',
    title: 'Attention Is All You Need',
    year: 2017,
    doi: '10.48550/arXiv.1706.03762',
    badge: 'Foundation (2017)'
  },
  {
    id: 'f92484a7-8fa9-580d-a3a8-fb0807c11c30',
    canonicalKey: 'arxiv_2106.09685',
    title: 'LoRA: Low-Rank Adaptation of Large Language Models',
    year: 2021,
    doi: '10.48550/arXiv.2106.09685',
    badge: 'Adaptation (2021)'
  },
  {
    id: '2ef30728-fe34-53ba-9664-c16acaf3c364',
    canonicalKey: 'arxiv_2205.14135',
    title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
    year: 2022,
    doi: '10.48550/arXiv.2205.14135',
    badge: 'IO-Efficiency (2022)'
  }
];

export const ComparePapers: React.FC<ComparePapersProps> = ({
  initialPaperIds = [],
  onBackToResearch,
  onAnalyzePaper,
  onOpenUpload,
}) => {
  const [selectedPapers, setSelectedPapers] = useState<PaperItem[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<PaperItem[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [comparisonResult, setComparisonResult] = useState<ResearchCompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deepGaps, setDeepGaps] = useState<ResearchGapsResponse | null>(null);
  const [isLoadingDeepGaps, setIsLoadingDeepGaps] = useState<boolean>(false);
  const [deepGapsError, setDeepGapsError] = useState<string | null>(null);
  const [showDeepGaps, setShowDeepGaps] = useState<boolean>(false);

  // Load initial papers if provided or default to top two benchmark papers
  useEffect(() => {
    let isMounted = true;
    const fetchInitial = async () => {
      try {
        const resp = await searchPapers('');
        if (!isMounted) return;
        const available = resp.papers || [];

        if (initialPaperIds && initialPaperIds.length >= 2) {
          const selected = available.filter(
            (p) => initialPaperIds.includes(p.id) || initialPaperIds.includes(p.doi || '')
          );
          if (selected.length >= 2) {
            setSelectedPapers(selected.slice(0, 2));
            return;
          }
        }

        // Default to first two benchmark papers for instant value
        if (available.length >= 2) {
          setSelectedPapers([available[0], available[1]]);
        }
      } catch (err) {
        // Fallback to static benchmark papers if API search is cold
        if (isMounted) {
          setSelectedPapers([
            {
              id: QUICK_BENCHMARK_PAPERS[0].id,
              title: QUICK_BENCHMARK_PAPERS[0].title,
              doi: QUICK_BENCHMARK_PAPERS[0].doi,
              publication_year: QUICK_BENCHMARK_PAPERS[0].year,
              authors: ['Vaswani et al.'],
              is_benchmark: true,
              source: 'benchmark'
            },
            {
              id: QUICK_BENCHMARK_PAPERS[1].id,
              title: QUICK_BENCHMARK_PAPERS[1].title,
              doi: QUICK_BENCHMARK_PAPERS[1].doi,
              publication_year: QUICK_BENCHMARK_PAPERS[1].year,
              authors: ['Hu et al.'],
              is_benchmark: true,
              source: 'benchmark'
            }
          ]);
        }
      }
    };

    fetchInitial();
    return () => {
      isMounted = false;
    };
  }, [initialPaperIds]);

  // Execute paper search when query changes
  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      setSearchResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const resp = await searchPapers(trimmed, 8);
        setSearchResults(resp.papers || []);
      } catch {
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  const hasAttemptedRef = useRef<string | null>(null);

  // Handle adding a paper to comparison selection
  const handleSelectPaper = (paper: PaperItem) => {
    if (selectedPapers.some((p) => p.id === paper.id)) return;
    setError(null);
    setComparisonResult(null);
    hasAttemptedRef.current = null;
    if (selectedPapers.length >= 2) {
      // Replace the second paper if already have 2 selected
      setSelectedPapers([selectedPapers[0], paper]);
    } else {
      setSelectedPapers([...selectedPapers, paper]);
    }
    setSearchQuery('');
    setSearchResults([]);
  };

  // Handle removing a paper from selection
  const handleRemovePaper = (id: string) => {
    setError(null);
    setComparisonResult(null);
    hasAttemptedRef.current = null;
    setSelectedPapers(selectedPapers.filter((p) => p.id !== id));
  };

  // Run direct comparison
  const handleExecuteComparison = useCallback(async () => {
    if (selectedPapers.length < 2) {
      setError('Please select exactly two papers to compare.');
      return;
    }

    setIsComparing(true);
    setError(null);
    setDeepGaps(null);
    setDeepGapsError(null);

    try {
      const resp = await compareResearchPapers({
        paper_ids: [selectedPapers[0].id, selectedPapers[1].id]
      });
      setComparisonResult(resp);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Comparison request failed.';
      setError(msg);
    } finally {
      setIsComparing(false);
    }
  }, [selectedPapers]);

  // Fetch evidence-backed research gaps across selected papers
  const handleFetchDeepGaps = async () => {
    if (selectedPapers.length < 2) return;
    setIsLoadingDeepGaps(true);
    setDeepGapsError(null);
    setShowDeepGaps(true);

    try {
      const res = await getResearchGaps([selectedPapers[0].id, selectedPapers[1].id]);
      setDeepGaps(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to discover research gaps.';
      setDeepGapsError(msg);
    } finally {
      setIsLoadingDeepGaps(false);
    }
  };

  // Auto-run comparison when two papers are selected on load (guarded against re-trigger loop)
  useEffect(() => {
    if (selectedPapers.length === 2) {
      const pairKey = `${selectedPapers[0].id}:${selectedPapers[1].id}`;
      if (hasAttemptedRef.current !== pairKey && !comparisonResult && !isComparing && !error) {
        hasAttemptedRef.current = pairKey;
        handleExecuteComparison();
      }
    } else {
      hasAttemptedRef.current = null;
    }
  }, [selectedPapers, comparisonResult, isComparing, error, handleExecuteComparison]);

  const p1 = selectedPapers[0];
  const p2 = selectedPapers[1];
  const tableData: ComparisonTableData | undefined = comparisonResult?.comparison_table || undefined;

  // Render conclusion badge styling
  const renderConclusionBadge = (conclusion: string) => {
    let colorClasses = 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30';
    if (conclusion.includes('Demonstrated improvement')) {
      colorClasses = 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40';
    } else if (conclusion.includes('Conflicting evidence')) {
      colorClasses = 'bg-purple-500/15 text-purple-300 border-purple-500/40';
    } else if (conclusion.includes('Limited evidence')) {
      colorClasses = 'bg-amber-500/15 text-amber-300 border-amber-500/40';
    } else if (conclusion.includes('Insufficient evidence')) {
      colorClasses = 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }

    return (
      <span
        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono uppercase tracking-wider border ${colorClasses}`}
      >
        <ShieldCheck className="w-3.5 h-3.5" />
        {conclusion}
      </span>
    );
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 animate-in fade-in duration-300">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[11px] font-mono uppercase tracking-widest px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
              Direct Synthesis Engine
            </span>
            <span className="text-[11px] font-mono uppercase tracking-widest text-slate-400">
              11-Dimension Alignment
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
            <GitCompare className="w-7 h-7 text-cyan-400" />
            Head-to-Head Paper Comparison
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl font-sans">
            Select two scholarly papers to perform a rigorous, evidence-grounded head-to-head comparison
            across 11 standardized dimensions without hallucinated superiority claims.
          </p>
        </div>

        {onBackToResearch && (
          <button
            onClick={onBackToResearch}
            className="self-start md:self-auto px-4 py-2 rounded-xl text-xs font-mono uppercase tracking-wider text-slate-300 hover:text-white bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 transition-colors flex items-center gap-1.5"
          >
            &larr; Back to Research
          </button>
        )}
      </div>

      {/* Paper Selection Control Strip */}
      <div className="glass-panel p-5 sm:p-6 rounded-2xl border border-white/10 bg-[#080d21] space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Scale className="w-4 h-4 text-cyan-400" />
              Active Papers to Compare ({selectedPapers.length} / 2)
            </h3>
            {onOpenUpload && (
              <button
                onClick={onOpenUpload}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono text-cyan-300 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 transition-colors"
              >
                <UploadCloud className="w-3.5 h-3.5" />
                <span>Upload PDF</span>
              </button>
            )}
          </div>
          <span className="text-xs text-slate-400">
            Select two papers from search or quick benchmark chips below.
          </span>
        </div>

        {/* Selected Paper Chips */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Paper A Slot */}
          <div className="p-4 rounded-xl border border-cyan-500/30 bg-slate-900/80 relative">
            <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400 font-bold block mb-1">
              Paper A (Baseline)
            </span>
            {p1 ? (
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h4 className="text-sm font-semibold text-white truncate" title={p1.title}>
                    {p1.title}
                  </h4>
                  <div className="flex flex-wrap items-center gap-2 mt-1 text-[11px] font-mono text-slate-400">
                    {p1.publication_year && <span>Year: {p1.publication_year}</span>}
                    {p1.doi && <span className="truncate max-w-[200px]">DOI: {p1.doi}</span>}
                    {p1.is_benchmark && (
                      <span className="text-cyan-300 bg-cyan-950/70 border border-cyan-500/30 px-1.5 rounded">
                        Benchmark
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {onAnalyzePaper && (
                    <button
                      onClick={() => onAnalyzePaper(p1.id, p1.title)}
                      className="p-1 text-slate-400 hover:text-cyan-300 rounded hover:bg-slate-800 transition-colors"
                      title="Analyze 7 dimensions of Paper A"
                    >
                      <Sparkles className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => handleRemovePaper(p1.id)}
                    className="p-1 text-slate-400 hover:text-rose-400 rounded hover:bg-slate-800 transition-colors"
                    title="Remove paper"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500 italic py-2">
                No paper selected. Click a benchmark chip or search below.
              </div>
            )}
          </div>

          {/* Paper B Slot */}
          <div className="p-4 rounded-xl border border-purple-500/30 bg-slate-900/80 relative">
            <span className="text-[10px] font-mono uppercase tracking-wider text-purple-400 font-bold block mb-1">
              Paper B (Comparison)
            </span>
            {p2 ? (
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h4 className="text-sm font-semibold text-white truncate" title={p2.title}>
                    {p2.title}
                  </h4>
                  <div className="flex flex-wrap items-center gap-2 mt-1 text-[11px] font-mono text-slate-400">
                    {p2.publication_year && <span>Year: {p2.publication_year}</span>}
                    {p2.doi && <span className="truncate max-w-[200px]">DOI: {p2.doi}</span>}
                    {p2.is_benchmark && (
                      <span className="text-purple-300 bg-purple-950/70 border border-purple-500/30 px-1.5 rounded">
                        Benchmark
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {onAnalyzePaper && (
                    <button
                      onClick={() => onAnalyzePaper(p2.id, p2.title)}
                      className="p-1 text-slate-400 hover:text-purple-300 rounded hover:bg-slate-800 transition-colors"
                      title="Analyze 7 dimensions of Paper B"
                    >
                      <Sparkles className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => handleRemovePaper(p2.id)}
                    className="p-1 text-slate-400 hover:text-rose-400 rounded hover:bg-slate-800 transition-colors"
                    title="Remove paper"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500 italic py-2">
                No paper selected. Click a benchmark chip or search below.
              </div>
            )}
          </div>
        </div>

        {/* Quick Benchmark Selector Chips */}
        <div className="space-y-2 pt-2 border-t border-white/[0.06]">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
            Quick-Select Benchmark Papers:
          </span>
          <div className="flex flex-wrap items-center gap-2">
            {QUICK_BENCHMARK_PAPERS.map((bp) => {
              const isSelected = selectedPapers.some(
                (p) => p.id === bp.id || p.title.toLowerCase().includes(bp.title.toLowerCase().slice(0, 15))
              );
              return (
                <button
                  key={bp.id}
                  onClick={() =>
                    handleSelectPaper({
                      id: bp.id,
                      title: bp.title,
                      doi: bp.doi,
                      publication_year: bp.year,
                      authors: ['Benchmark Author'],
                      is_benchmark: true,
                      source: 'benchmark'
                    })
                  }
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all flex items-center gap-1.5 ${
                    isSelected
                      ? 'bg-cyan-500/20 text-cyan-200 border-cyan-400/50 shadow-[0_0_12px_rgba(0,245,212,0.25)]'
                      : 'bg-slate-900/90 text-slate-300 border-slate-700 hover:bg-slate-800 hover:border-slate-600'
                  }`}
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>{bp.title}</span>
                  <span className="text-[10px] font-mono text-slate-400">({bp.year})</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Search Paper Input */}
        <div className="relative pt-2">
          <div className="flex items-center bg-slate-900/90 border border-white/10 rounded-xl p-1.5">
            <div className="pl-3 pr-2 text-cyan-400">
              <Search className="w-4 h-4" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search other papers by title, DOI (e.g. 10.48550/arXiv...), or keyword..."
              className="w-full bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none py-1.5 px-1 font-sans"
            />
            {isSearching && (
              <span className="pr-3 text-xs text-cyan-400 font-mono animate-pulse">
                Searching...
              </span>
            )}
          </div>

          {/* Autocomplete Dropdown */}
          {searchResults.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 z-30 bg-[#0d1430] border border-cyan-500/40 rounded-xl shadow-2xl overflow-hidden max-h-60 overflow-y-auto">
              {searchResults.map((paper) => (
                <div
                  key={paper.id}
                  onClick={() => handleSelectPaper(paper)}
                  className="px-4 py-2.5 hover:bg-slate-800/80 cursor-pointer border-b border-white/[0.05] last:border-b-0 flex items-center justify-between gap-3 group"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-white group-hover:text-cyan-300 truncate">
                      {paper.title}
                    </p>
                    <p className="text-[11px] font-mono text-slate-400">
                      {paper.publication_year ? `Year: ${paper.publication_year} • ` : ''}
                      {paper.doi ? `DOI: ${paper.doi}` : ''}
                    </p>
                  </div>
                  <span className="text-xs text-cyan-400 opacity-0 group-hover:opacity-100 flex items-center gap-1 font-mono uppercase">
                    Select <Plus className="w-3.5 h-3.5" />
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Execute Comparison Button */}
        <div className="flex items-center justify-between pt-2">
          {error && (
            <div className="text-xs text-rose-300 flex items-center gap-2 font-sans bg-rose-500/15 border border-rose-500/30 px-3.5 py-2.5 rounded-xl max-w-xl">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
          )}
          <div className="ml-auto">
            <button
              onClick={handleExecuteComparison}
              disabled={selectedPapers.length < 2 || isComparing}
              className={`px-5 py-2.5 rounded-xl font-medium text-sm transition-all flex items-center gap-2 ${
                selectedPapers.length === 2 && !isComparing
                  ? 'bg-gradient-to-r from-cyan-400 via-teal-300 to-cyan-400 text-slate-900 shadow-[0_0_20px_rgba(0,245,212,0.35)] hover:shadow-[0_0_28px_rgba(0,245,212,0.5)] active:scale-[0.98]'
                  : 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
              }`}
            >
              {isComparing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-slate-900" />
                  <span>Synthesizing Matrix...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-slate-900" />
                  <span>{error ? 'Retry Comparison' : 'Run Comparison'}</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Comparison Results Section */}
      {tableData && (
        <div className="space-y-8 animate-in fade-in duration-300">
          {/* Comparison Table Section */}
          <div className="glass-panel rounded-2xl border border-white/10 bg-[#080d21] overflow-hidden shadow-2xl">
            <div className="p-5 border-b border-white/[0.08] flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-[#090f26]">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  11-Dimension Aligned Research Matrix
                </h3>
                <p className="text-xs text-slate-400">
                  Every reported cell is traceable to retrieved excerpts. Missing fields are explicitly reported as not found.
                </p>
              </div>

              {renderConclusionBadge(tableData.conclusion)}
            </div>

            {/* Responsive Table Container */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse min-w-[700px]">
                <thead>
                  <tr className="bg-slate-900/90 border-b border-white/[0.08] text-[11px] font-mono uppercase tracking-wider text-slate-400">
                    <th className="py-3.5 px-4 w-1/5">Research Dimension</th>
                    <th className="py-3.5 px-4 w-2/5 text-cyan-300">
                      {p1?.title || 'Paper A'}
                    </th>
                    <th className="py-3.5 px-4 w-2/5 text-purple-300">
                      {p2?.title || 'Paper B'}
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.06]">
                  {tableData.rows.map((row: ComparisonRow, idx: number) => {
                    const c1 = row.per_paper[p1?.id || ''] || Object.values(row.per_paper)[0];
                    const c2 = row.per_paper[p2?.id || ''] || Object.values(row.per_paper)[1];

                    const isNotReported1 = c1?.status === 'not_reported' || c1?.value.includes('Not reported');
                    const isNotReported2 = c2?.status === 'not_reported' || c2?.value.includes('Not reported');

                    return (
                      <tr
                        key={row.key}
                        className={idx % 2 === 0 ? 'bg-slate-900/30' : 'bg-transparent'}
                      >
                        {/* Dimension Label */}
                        <td className="py-3.5 px-4 font-medium text-slate-300 text-xs font-mono uppercase tracking-wider align-top bg-slate-950/20">
                          {row.dimension}
                        </td>

                        {/* Paper A Value */}
                        <td className="py-3.5 px-4 text-xs text-slate-200 align-top leading-relaxed">
                          {isNotReported1 ? (
                            <span className="text-slate-500 italic bg-slate-800/50 px-2 py-0.5 rounded text-[11px] font-mono">
                              Not reported in retrieved evidence.
                            </span>
                          ) : (
                            <div>
                              <span>{c1?.value}</span>
                              {c1?.citations && c1.citations.length > 0 && (
                                <span className="ml-1.5 text-[10px] font-mono font-bold text-cyan-300 bg-cyan-950/60 border border-cyan-500/40 px-1 py-0.5 rounded">
                                  {c1.citations[0]}
                                </span>
                              )}
                            </div>
                          )}
                        </td>

                        {/* Paper B Value */}
                        <td className="py-3.5 px-4 text-xs text-slate-200 align-top leading-relaxed">
                          {isNotReported2 ? (
                            <span className="text-slate-500 italic bg-slate-800/50 px-2 py-0.5 rounded text-[11px] font-mono">
                              Not reported in retrieved evidence.
                            </span>
                          ) : (
                            <div>
                              <span>{c2?.value}</span>
                              {c2?.citations && c2.citations.length > 0 && (
                                <span className="ml-1.5 text-[10px] font-mono font-bold text-purple-300 bg-purple-950/60 border border-purple-500/40 px-1 py-0.5 rounded">
                                  {c2.citations[0]}
                                </span>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Below-Table Comparative Insights Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Key Similarities Card */}
            <div className="glass-panel p-5 rounded-2xl border border-white/10 bg-[#080d21] space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-cyan-300 font-bold flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                Key Similarities
              </h4>
              <ul className="space-y-2 text-xs text-slate-300">
                {tableData.key_similarities.map((item: string, i: number) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-cyan-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Key Differences Card */}
            <div className="glass-panel p-5 rounded-2xl border border-white/10 bg-[#080d21] space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-purple-300 font-bold flex items-center gap-1.5">
                <Scale className="w-4 h-4 text-purple-400" />
                Key Differences
              </h4>
              <ul className="space-y-2 text-xs text-slate-300">
                {tableData.key_differences.map((item: string, i: number) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-purple-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Direct Evidence vs Extrapolation Card */}
            <div className="glass-panel p-5 rounded-2xl border border-white/10 bg-[#080d21] space-y-3">
              <h4 className="text-xs font-mono uppercase tracking-wider text-emerald-300 font-bold flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Direct Evidence Claims
              </h4>
              <ul className="space-y-2 text-xs text-slate-300">
                {tableData.direct_evidence_findings.map((item: string, i: number) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-emerald-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Evidence Gaps Card */}
            <div className="glass-panel p-5 rounded-2xl border border-white/10 bg-[#080d21] space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className="text-xs font-mono uppercase tracking-wider text-amber-300 font-bold flex items-center gap-1.5">
                  <HelpCircle className="w-4 h-4 text-amber-400" />
                  Evidence Gaps & Unreported Aspects
                </h4>
                <button
                  type="button"
                  onClick={handleFetchDeepGaps}
                  disabled={isLoadingDeepGaps}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-mono text-amber-300 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 transition-colors disabled:opacity-50"
                >
                  {isLoadingDeepGaps ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Discovering Gaps...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>{deepGaps ? 'Refresh Gaps' : 'Deep Gaps Analysis'}</span>
                    </>
                  )}
                </button>
              </div>

              <ul className="space-y-2 text-xs text-slate-300">
                {tableData.evidence_gaps.map((item: string, i: number) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-amber-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>

              {/* Deep Gaps Error */}
              {deepGapsError && (
                <div className="p-2.5 rounded-lg bg-red-950/50 border border-red-500/30 text-xs text-red-200 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
                  <span>{deepGapsError}</span>
                </div>
              )}

              {/* Deep Research Gaps Panel */}
              {deepGaps && (
                <div className="pt-3 border-t border-white/[0.08] space-y-3 animate-in fade-in duration-200">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono uppercase tracking-wider text-amber-400 font-bold">
                      Evidence-Grounded Research Gaps ({deepGaps.total_gaps_count} Found)
                    </span>
                    <button
                      onClick={() => setShowDeepGaps(!showDeepGaps)}
                      className="text-slate-400 hover:text-white p-1"
                    >
                      {showDeepGaps ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>

                  {showDeepGaps && (
                    <div className="space-y-3">
                      {deepGaps.documented_gaps.length > 0 && (
                        <div className="space-y-1.5">
                          <span className="text-[10px] font-mono uppercase text-slate-400">
                            Author-Documented Limitations:
                          </span>
                          <div className="space-y-1.5">
                            {deepGaps.documented_gaps.map((gap, idx) => (
                              <div key={idx} className="p-2.5 rounded-lg bg-slate-900/60 border border-amber-500/20 text-xs text-slate-200 space-y-1">
                                <div className="font-semibold text-white flex items-center justify-between">
                                  <span>{gap.title}</span>
                                  {gap.evidence_citations && gap.evidence_citations.length > 0 && (
                                    <span className="text-[10px] font-mono text-cyan-400">
                                      {gap.evidence_citations.join(' ')}
                                    </span>
                                  )}
                                </div>
                                <p className="text-slate-300 text-[11px]">{gap.description}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {deepGaps.inferred_gaps.length > 0 && (
                        <div className="space-y-1.5">
                          <span className="text-[10px] font-mono uppercase text-slate-400">
                            AI-Inferred Candidate Gaps:
                          </span>
                          <div className="space-y-1.5">
                            {deepGaps.inferred_gaps.map((gap, idx) => (
                              <div key={idx} className="p-2.5 rounded-lg bg-purple-950/30 border border-purple-500/20 text-xs text-slate-200 space-y-1">
                                <div className="font-semibold text-purple-200 flex items-center justify-between">
                                  <span>{gap.title}</span>
                                  {gap.confidence_score !== undefined && (
                                    <span className="text-[10px] font-mono text-purple-400 bg-purple-950/60 px-1.5 py-0.5 rounded border border-purple-500/30">
                                      Conf: {(gap.confidence_score * 100).toFixed(0)}%
                                    </span>
                                  )}
                                </div>
                                <p className="text-slate-300 text-[11px]">{gap.description}</p>
                                {gap.rationale && (
                                  <p className="text-[10px] text-slate-400 italic">
                                    Rationale: {gap.rationale}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Overall Comparative Conclusion Banner */}
          <div className="glass-panel p-6 rounded-2xl border border-cyan-500/30 bg-[#09102c] space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold">
                  Overall Comparative Conclusion:
                </span>
                {renderConclusionBadge(tableData.conclusion)}
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                Status: {tableData.comparability_status}
              </span>
            </div>
            <p className="text-sm text-slate-200 leading-relaxed font-sans">
              {tableData.conclusion_reasoning}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
