import React, { useState } from 'react';
import {
  Search,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Layers,
  FileCheck,
  X,
  ShieldCheck,
  ChevronRight
} from 'lucide-react';
import { Hero3D } from './Hero3D';
import type { SampleQuery } from '../types';

interface HeroSectionProps {
  searchQuery: string;
  onSearchChange: (val: string) => void;
  onSearchSubmit: (query: string) => void;
  onViewArchitecture: () => void;
}

const SAMPLE_QUERIES: SampleQuery[] = [
  {
    id: '1',
    query: 'How do FlashAttention IO optimizations achieve 4.3x speedup on A100 GPUs?',
    category: 'IO Complexity',
    targetPaper: 'Dao et al. (2022)',
  },
  {
    id: '2',
    query: 'What are the verified empirical scaling bounds for dense transformer attention?',
    category: 'Scaling Laws',
    targetPaper: 'Kaplan et al. / Vaswani et al.',
  },
  {
    id: '3',
    query: 'Does hybrid dense + BM25 reciprocal rank fusion outperform dense-only retrieval?',
    category: 'Retrieval RAG',
    targetPaper: 'Cormack et al. / Karpukhin et al.',
  },
  {
    id: '4',
    query: 'How does sentence-level claim verification eliminate hallucinated superiority claims?',
    category: 'Evidence Grounding',
    targetPaper: 'Nexora Core Verification',
  },
];

export const HeroSection: React.FC<HeroSectionProps> = ({
  searchQuery,
  onSearchChange,
  onSearchSubmit,
  onViewArchitecture,
}) => {
  const [activeChip, setActiveChip] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onSearchSubmit(searchQuery);
    }
  };

  const handleChipClick = (sample: SampleQuery) => {
    setActiveChip(sample.id);
    onSearchChange(sample.query);
  };

  return (
    <section className="relative pt-8 pb-16 md:pt-14 md:pb-24 overflow-hidden">
      {/* Background radial gradients constrained inside overflow-hidden bounds */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-[600px] bg-radial-ambient opacity-80" />
        <div className="absolute top-1/4 -left-20 w-80 h-80 bg-cyan-500/10 rounded-full blur-[128px]" />
        <div className="absolute top-1/3 -right-20 w-80 h-80 bg-purple-600/15 rounded-full blur-[140px]" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">

        {/* Top Eyebrow Badge */}
        <div className="flex justify-center md:justify-start mb-6">
          <div className="inline-flex items-center gap-2 sm:gap-2.5 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full glass-panel border-cyan-500/25 shadow-[0_0_20px_rgba(0,245,212,0.12)] max-w-full">
            <span className="flex h-2 w-2 relative shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400" />
            </span>
            <span className="text-[11px] sm:text-xs font-mono uppercase tracking-wider text-cyan-300 font-semibold truncate">
              <span className="hidden sm:inline">Research Intelligence Engine • </span>Milestone MVP
            </span>
            <span className="text-slate-500 hidden sm:inline">•</span>
            <span className="text-xs font-medium text-slate-300 hidden sm:inline">
              Evidence-Grounded RAG
            </span>
          </div>
        </div>

        {/* Hero Grid: Left Content & Right 3D Visual Centerpiece */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-8 items-center">

          {/* Left Column: Headlines & Search Form (7 cols) */}
          <div className="lg:col-span-7 space-y-6 text-center md:text-left">

            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-[1.15] break-words">
              Go Beyond{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-teal-200 to-cyan-400 text-glow-cyan">
                Reading
              </span>{' '}
              <br className="block sm:hidden" />
              Research.
            </h1>

            <p className="text-base sm:text-xl text-slate-300 max-w-2xl font-normal leading-relaxed">
              Connect scientific discoveries, compare research methods, and explore the
              evidence behind what we know.
            </p>

            {/* Interactive Research Query Search Form */}
            <form onSubmit={handleSubmit} className="pt-2 max-w-2xl">
              <div className="relative group">
                <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-cyan-500/30 via-violet-500/25 to-blue-500/30 blur-md opacity-60 group-hover:opacity-100 transition-opacity duration-300" />

                <div className="relative flex items-center bg-[#090d1f]/90 backdrop-blur-xl border border-cyan-500/35 rounded-2xl p-2 shadow-2xl">
                  <div className="pl-3.5 pr-2 text-cyan-400">
                    <Search className="w-5 h-5" />
                  </div>

                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => {
                      onSearchChange(e.target.value);
                      if (activeChip) setActiveChip(null);
                    }}
                    placeholder="Ask a scientific question (e.g. FlashAttention IO complexity vs standard attention)..."
                    className="min-w-0 w-full bg-transparent text-sm sm:text-base text-white placeholder-slate-400 focus:outline-none py-2 px-1 font-sans"
                  />

                  {searchQuery && (
                    <button
                      type="button"
                      onClick={() => {
                        onSearchChange('');
                        setActiveChip(null);
                      }}
                      className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/60 mr-1"
                      aria-label="Clear query input"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}

                  <button
                    type="submit"
                    className="shrink-0 flex items-center gap-1.5 sm:gap-2 px-3.5 sm:px-5 py-2.5 sm:py-3 rounded-xl font-medium text-xs sm:text-sm text-slate-900 bg-gradient-to-r from-cyan-400 via-teal-300 to-cyan-400 hover:from-cyan-300 hover:to-teal-200 shadow-[0_0_20px_rgba(0,245,212,0.4)] active:scale-[0.98] transition-all"
                  >
                    <span className="hidden sm:inline font-semibold">Explore Research</span>
                    <span className="sm:hidden font-semibold">Explore</span>
                    <ArrowRight className="w-4 h-4 text-slate-900" />
                  </button>
                </div>
              </div>
            </form>

            {/* Sample Research Query Chips */}
            <div className="space-y-2.5 pt-1">
              <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-400">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span>Sample Research Inquiries:</span>
              </div>

              <div className="flex flex-wrap gap-2">
                {SAMPLE_QUERIES.map((sample) => {
                  const isSelected = activeChip === sample.id;
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      onClick={() => handleChipClick(sample)}
                      className={`text-xs px-3 py-1.5 rounded-lg border font-mono transition-all text-left flex items-center gap-1.5 ${
                        isSelected
                          ? 'bg-cyan-500/20 border-cyan-400/80 text-cyan-300 shadow-[0_0_12px_rgba(0,245,212,0.3)]'
                          : 'glass-panel border-white/10 text-slate-300 hover:border-cyan-500/40 hover:text-white'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      <span>{sample.category}</span>
                      <ChevronRight className="w-3 h-3 text-slate-500" />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Secondary Action Buttons & Trust Line */}
            <div className="pt-3 flex flex-wrap items-center gap-4 justify-center md:justify-start">
              <button
                type="button"
                onClick={onViewArchitecture}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl glass-panel glass-panel-hover text-sm font-medium text-slate-200 border-white/10"
              >
                <ShieldCheck className="w-4 h-4 text-cyan-400" />
                <span>View Research Standards</span>
              </button>

              <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Zero Hallucination Grounding Gated</span>
              </div>
            </div>

          </div>

          {/* Right Column: Signature 3D Centerpiece with Floating Research Badges (5 cols) */}
          <div className="lg:col-span-5 relative flex items-center justify-center">

            {/* 3D Scene Container */}
            <div className="relative w-full aspect-square max-w-[500px]">

              {/* Central Three.js Visualization */}
              <Hero3D />

              {/* Floating Scientific Metadata Badges around 3D Core */}

              {/* Badge 1: Top-Left Citation Node */}
              <div className="absolute top-2 -left-2 sm:left-2 glass-panel-glow rounded-xl p-3 border-cyan-500/30 text-xs font-mono max-w-[220px] pointer-events-none hidden sm:block animate-float">
                <div className="flex items-center gap-1.5 text-cyan-300 font-semibold mb-1">
                  <FileCheck className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Evidence Node [E1]</span>
                </div>
                <p className="text-[11px] text-slate-300 leading-tight">
                  Vaswani et al. (2017)
                </p>
                <div className="mt-1 flex items-center gap-1 text-[10px] text-emerald-400 font-semibold">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>Entailed Claim (100%)</span>
                </div>
              </div>

              {/* Badge 2: Bottom-Right Cross-Paper Comparison Metric */}
              <div className="absolute bottom-4 -right-2 sm:right-2 glass-panel-glow rounded-xl p-3 border-purple-500/30 text-xs font-mono max-w-[230px] pointer-events-none hidden sm:block animate-float [animation-delay:2s]">
                <div className="flex items-center gap-1.5 text-purple-300 font-semibold mb-1">
                  <Layers className="w-3.5 h-3.5 text-purple-400" />
                  <span>Cross-Paper Matrix</span>
                </div>
                <p className="text-[11px] text-slate-300 leading-tight">
                  FlashAttention vs Multi-Head
                </p>
                <div className="mt-1 flex items-center gap-1.5 text-[10px] text-cyan-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span>Speedup: 4.3x Verified</span>
                </div>
              </div>

            </div>

          </div>

        </div>

      </div>
    </section>
  );
};
