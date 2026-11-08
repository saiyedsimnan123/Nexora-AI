import React, { useState, useEffect, useId } from 'react';
import {
  X,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Loader2,
  ShieldCheck,
  Scale,
  Target,
  Cpu,
  Database,
  BarChart2,
  Flame,
  Compass
} from 'lucide-react';
import { analyzePaper, searchPapers } from '../services/api';
import type { PaperAnalysisResponse, PaperItem } from '../types/api';

interface AnalysisModalProps {
  isOpen: boolean;
  onClose: () => void;
  paperId: string | null;
  paperTitle?: string | null;
  onComparePaper?: (paperId: string) => void;
}

const DIMENSION_CONFIG = [
  {
    key: 'problem_and_objectives',
    label: 'Problem & Objectives',
    icon: Target,
    color: 'text-cyan-400',
    borderColor: 'border-cyan-500/30',
  },
  {
    key: 'methodology_and_algorithms',
    label: 'Methodology & Algorithms',
    icon: Cpu,
    color: 'text-purple-400',
    borderColor: 'border-purple-500/30',
  },
  {
    key: 'models_and_datasets',
    label: 'Models & Datasets',
    icon: Database,
    color: 'text-blue-400',
    borderColor: 'border-blue-500/30',
  },
  {
    key: 'experiments_and_metrics',
    label: 'Experiments & Metrics',
    icon: BarChart2,
    color: 'text-teal-400',
    borderColor: 'border-teal-500/30',
  },
  {
    key: 'results_and_findings',
    label: 'Results & Findings',
    icon: Flame,
    color: 'text-emerald-400',
    borderColor: 'border-emerald-500/30',
  },
  {
    key: 'limitations',
    label: 'Limitations & Constraints',
    icon: AlertCircle,
    color: 'text-amber-400',
    borderColor: 'border-amber-500/30',
  },
  {
    key: 'future_work',
    label: 'Future Work & Open Questions',
    icon: Compass,
    color: 'text-indigo-400',
    borderColor: 'border-indigo-500/30',
  },
] as const;

export const AnalysisModal: React.FC<AnalysisModalProps> = ({
  isOpen,
  onClose,
  paperId,
  paperTitle,
  onComparePaper,
}) => {
  const [activePaperId, setActivePaperId] = useState<string | null>(paperId);
  const [analysis, setAnalysis] = useState<PaperAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [benchmarkPapers, setBenchmarkPapers] = useState<PaperItem[]>([]);
  const titleId = useId();

  // Keep activePaperId in sync with prop
  useEffect(() => {
    setActivePaperId(paperId);
  }, [paperId]);

  // Load benchmark papers for quick selection if no paper specified
  useEffect(() => {
    if (isOpen && !activePaperId) {
      searchPapers('', 5).then((res) => {
        setBenchmarkPapers(res.papers || []);
      }).catch(() => {});
    }
  }, [isOpen, activePaperId]);

  // Fetch analysis when paperId is set
  useEffect(() => {
    if (!isOpen || !activePaperId) return;

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    analyzePaper(activePaperId)
      .then((data) => {
        if (isMounted) setAnalysis(data);
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to analyze paper.';
          setError(msg);
        }
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen, activePaperId]);

  if (!isOpen) return null;

  const renderStatusBadge = (status?: string) => {
    const s = (status || '').toLowerCase();
    if (s === 'reported') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-emerald-950/70 text-emerald-300 border border-emerald-500/30">
          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
          <span>Reported</span>
        </span>
      );
    }
    if (s === 'uncertain') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-amber-950/70 text-amber-300 border border-amber-500/30">
          <HelpCircle className="w-3 h-3 text-amber-400" />
          <span>Uncertain</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-800 text-slate-400 border border-slate-700">
        <span>Not Reported</span>
      </span>
    );
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200"
    >
      <div className="relative w-full max-w-4xl bg-[#090d21] border border-cyan-500/30 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/[0.08] bg-[#070a1a]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-300">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 id={titleId} className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                Structured Research Paper Analysis
                <span className="text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  7 Dimensions
                </span>
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                {paperTitle || analysis?.title || (activePaperId ? `Paper ID: ${activePaperId}` : 'Select a Paper to Analyze')}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close analysis dialog"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Quick Paper Selector if no paper active */}
          {!activePaperId && (
            <div className="space-y-4">
              <p className="text-sm text-slate-300">
                Choose an indexed benchmark paper or input a paper UUID to perform structured 7-dimension analysis:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {benchmarkPapers.map((p) => (
                  <div
                    key={p.id}
                    onClick={() => setActivePaperId(p.id)}
                    className="p-3.5 rounded-xl glass-panel glass-panel-hover border-white/10 cursor-pointer text-left space-y-1 transition-all"
                  >
                    <div className="text-xs font-semibold text-white line-clamp-1">{p.title}</div>
                    <div className="text-[11px] font-mono text-cyan-400 flex items-center justify-between">
                      <span>{p.authors?.[0] || 'Scholarly'} ({p.publication_year || 2022})</span>
                      <span className="text-[10px] text-slate-500 uppercase">Benchmark</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="py-16 text-center space-y-3">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin mx-auto" />
              <p className="text-sm font-mono text-slate-300">
                Extracting 7 standardized research dimensions from paper evidence...
              </p>
              <p className="text-xs text-slate-500 font-mono">
                Grounding claims against extracted source chunks
              </p>
            </div>
          )}

          {/* Error State */}
          {error && !isLoading && (
            <div className="p-4 rounded-xl bg-red-950/60 border border-red-500/30 text-xs text-red-200 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-semibold text-red-300">Analysis Error</p>
                <p>{error}</p>
                <button
                  onClick={() => {
                    if (activePaperId) {
                      setIsLoading(true);
                      setError(null);
                      analyzePaper(activePaperId)
                        .then(setAnalysis)
                        .catch((e) => setError(e instanceof Error ? e.message : 'Error'))
                        .finally(() => setIsLoading(false));
                    }
                  }}
                  className="mt-2 text-xs font-mono text-cyan-300 underline"
                >
                  Retry Analysis
                </button>
              </div>
            </div>
          )}

          {/* Successful Analysis Results */}
          {analysis && !isLoading && (
            <div className="space-y-6">
              {/* Meta strip */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-[#060818] border border-white/10 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Paper:</span>
                  <span className="text-white font-semibold truncate max-w-md">
                    {analysis.title || activePaperId}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-slate-400">
                  <span>Evidence: <strong className="text-cyan-300">{analysis.evidence_count}</strong></span>
                  <span>Model: <strong className="text-purple-300">{analysis.model}</strong></span>
                  {onComparePaper && activePaperId && (
                    <button
                      onClick={() => {
                        onClose();
                        onComparePaper(activePaperId);
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30 transition-colors"
                    >
                      <Scale className="w-3.5 h-3.5" />
                      <span>Compare</span>
                    </button>
                  )}
                </div>
              </div>

              {/* 7 Dimensions Cards Grid */}
              <div className="space-y-4">
                {DIMENSION_CONFIG.map(({ key, label, icon: Icon, color, borderColor }) => {
                  const dim = (analysis as unknown as Record<string, Record<string, unknown>>)[key] || {};
                  const summary = (dim.summary as string) || (dim.text as string) || 'No explicit findings reported in analyzed text.';
                  const status = (dim.status as string) || 'reported';
                  const citations = (dim.evidence_labels as string[]) || (dim.evidence_citations as string[]) || [];

                  return (
                    <div
                      key={key}
                      className={`glass-panel p-4 sm:p-5 rounded-xl border ${borderColor} bg-[#080d21] space-y-2.5 transition-all`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <Icon className={`w-4 h-4 ${color}`} />
                          <h4 className="text-sm font-bold text-white tracking-tight">{label}</h4>
                        </div>
                        <div className="flex items-center gap-2">
                          {citations.length > 0 && (
                            <div className="flex items-center gap-1">
                              {citations.map((c, i) => (
                                <span
                                  key={i}
                                  className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-500/30"
                                >
                                  {c}
                                </span>
                              ))}
                            </div>
                          )}
                          {renderStatusBadge(status)}
                        </div>
                      </div>

                      <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
                        {summary}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3.5 border-t border-white/[0.06] bg-[#060917] flex items-center justify-between text-[11px] font-mono text-slate-400">
          <div className="flex items-center gap-1.5 text-cyan-300">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>Structured Extraction Anchored in Paper Evidence</span>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1 rounded-lg text-slate-400 hover:text-white transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
