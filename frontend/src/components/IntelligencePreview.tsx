import React, { useState } from 'react';
import {
  FileCheck,
  Layers,
  CheckCircle2,
  AlertCircle,
  ChevronRight,
  BarChart3,
  Network
} from 'lucide-react';
import type { PreviewTab, VerifiedClaimItem } from '../types';

const CLAIMS_SAMPLE: VerifiedClaimItem[] = [
  {
    id: 'c1',
    claim: 'FlashAttention achieves 2x-4x wall-clock speedup on BERT-large compared to standard exact attention [E1].',
    status: 'SUPPORTED',
    citations: ['[E1]'],
    paperTitle: 'FlashAttention (Dao et al., 2022)',
    excerpt:
      'We evaluate BERT-large on MLPerf 1.1... FlashAttention yields a 2.4x speedup over standard HuggingFace attention while computing exact attention without approximation.',
  },
  {
    id: 'c2',
    claim: 'Standard self-attention memory overhead scales quadratically O(N²) with sequence length N [E2].',
    status: 'SUPPORTED',
    citations: ['[E2]'],
    paperTitle: 'Attention Is All You Need (Vaswani et al., 2017)',
    excerpt:
      'Self-attention layers require computing an N by N attention matrix, yielding an O(N^2) computational and memory footprint per layer.',
  },
  {
    id: 'c3',
    claim: 'The proposed architecture vastly outperforms all existing deep learning models across every domain [E1].',
    status: 'LIMITED_EVIDENCE',
    citations: ['[E1]'],
    paperTitle: 'Adversarial Extrapolation Benchmark',
    excerpt:
      'Evaluation was conducted solely on natural language question answering benchmarks (SQuAD v2). Cross-domain vision and audio baselines were not evaluated.',
  },
];

export const IntelligencePreview: React.FC = () => {
  const [activeTab, setActiveTab] = useState<PreviewTab>('verification');
  const [selectedClaim, setSelectedClaim] = useState<string>('c1');

  const currentClaim = CLAIMS_SAMPLE.find((c) => c.id === selectedClaim) || CLAIMS_SAMPLE[0];

  return (
    <section id="preview" className="py-20 md:py-28 relative bg-[#060814]/60 border-y border-white/[0.06]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full glass-panel border-purple-500/25 text-xs font-mono text-purple-300">
              <Network className="w-3.5 h-3.5 text-purple-400" />
              <span>Interactive Product Preview</span>
            </div>

            <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Research Intelligence{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-300 to-cyan-400 text-glow-violet">
                Matrix
              </span>
            </h2>

            <p className="text-sm sm:text-base text-slate-300 max-w-2xl">
              An illustrative preview of how Nexora AI extracts claims, connects citation lineages, and
              validates empirical research findings.
            </p>
          </div>

          {/* Tab Switcher Buttons */}
          <div className="flex overflow-x-auto max-w-full w-full sm:w-auto p-1.5 rounded-xl glass-panel border-white/10">
            <button
              type="button"
              onClick={() => setActiveTab('verification')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono shrink-0 transition-all ${
                activeTab === 'verification'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/50 shadow-[0_0_15px_rgba(0,245,212,0.2)]'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <FileCheck className="w-3.5 h-3.5" />
              <span>Claim Verification</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('lineage')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono shrink-0 transition-all ${
                activeTab === 'lineage'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-400/50 shadow-[0_0_15px_rgba(168,85,247,0.2)]'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Concept Lineage</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('benchmark')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono shrink-0 transition-all ${
                activeTab === 'benchmark'
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-400/50 shadow-[0_0_15px_rgba(59,130,246,0.2)]'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Benchmark Signal</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Claim Verification Matrix */}
        {activeTab === 'verification' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

            {/* Left Column: Claims List (5 cols) */}
            <div className="lg:col-span-5 space-y-3">
              <div className="text-xs font-mono uppercase tracking-wider text-slate-400 px-1 flex items-center justify-between">
                <span>Extracted Research Claims:</span>
                <span className="text-cyan-400">3 Cases</span>
              </div>

              {CLAIMS_SAMPLE.map((item) => {
                const isSelected = item.id === selectedClaim;
                return (
                  <div
                    key={item.id}
                    onClick={() => setSelectedClaim(item.id)}
                    className={`p-4 rounded-xl cursor-pointer transition-all border ${
                      isSelected
                        ? 'glass-panel-glow border-cyan-400/50 -translate-x-1 shadow-[0_0_20px_rgba(0,245,212,0.15)]'
                        : 'glass-panel border-white/10 hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
                        {item.citations.join(' ')}
                      </span>

                      <span
                        className={`text-[10px] font-mono font-semibold uppercase px-2 py-0.5 rounded ${
                          item.status === 'SUPPORTED'
                            ? 'bg-emerald-950/70 text-emerald-300 border border-emerald-500/30'
                            : 'bg-amber-950/70 text-amber-300 border border-amber-500/30'
                        }`}
                      >
                        {item.status.replace('_', ' ')}
                      </span>
                    </div>

                    <p className="text-xs sm:text-sm text-slate-200 line-clamp-2 leading-relaxed">
                      {item.claim}
                    </p>

                    <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                      <span>{item.paperTitle}</span>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Right Column: Citation Inspector & Source Grounding (7 cols) */}
            <div className="lg:col-span-7 glass-panel rounded-2xl p-6 sm:p-8 border-cyan-500/30 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4 mb-6">
                <div className="flex items-center gap-2.5 text-xs font-mono text-cyan-300">
                  <FileCheck className="w-4 h-4 text-cyan-400" />
                  <span>Grounding Inspector • Evidence Lineage</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-slate-400">Status:</span>
                  <span
                    className={`text-xs font-mono font-semibold px-2.5 py-0.5 rounded ${
                      currentClaim.status === 'SUPPORTED'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/40'
                        : 'bg-amber-500/20 text-amber-300 border border-amber-400/40'
                    }`}
                  >
                    {currentClaim.status.replace('_', ' ')}
                  </span>
                </div>
              </div>

              {/* Selected Claim Box */}
              <div className="space-y-2 mb-6">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
                  Target Research Claim:
                </span>
                <p className="text-sm sm:text-base font-medium text-white bg-slate-900/80 p-4 rounded-xl border border-white/10 leading-relaxed">
                  "{currentClaim.claim}"
                </p>
              </div>

              {/* Source Excerpt Grounding */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>Cited Source Excerpt ({currentClaim.citations[0]}):</span>
                  <span className="text-cyan-400 font-mono">{currentClaim.paperTitle}</span>
                </div>

                <div className="bg-[#04060d] p-4 sm:p-5 rounded-xl border border-cyan-500/20 text-xs sm:text-sm font-mono text-slate-300 leading-relaxed space-y-3">
                  <p className="border-l-2 border-cyan-400 pl-3 italic text-cyan-100/90">
                    "{currentClaim.excerpt}"
                  </p>

                  <div className="pt-2 flex items-center justify-between text-[11px] border-t border-white/[0.05] text-slate-400">
                    <span>Document: arXiv:2205.14135 • Page 4</span>
                    <span className="text-emerald-400 flex items-center gap-1 font-sans">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Entailment Confirmed
                    </span>
                  </div>
                </div>
              </div>

              {/* Guardrail Policy Notice */}
              <div className="mt-6 p-3 rounded-lg bg-slate-900/60 border border-white/[0.06] flex items-start gap-2.5 text-xs text-slate-400">
                <AlertCircle className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                <p>
                  Nexora AI prohibits superiority assertions ("outperforms", "better than", "SOTA")
                  unless empirical comparative numbers appear explicitly in the cited literature excerpt.
                </p>
              </div>

            </div>

          </div>
        )}

        {/* Tab 2: Connected Concept Lineage */}
        {activeTab === 'lineage' && (
          <div className="glass-panel rounded-2xl p-6 sm:p-10 border-purple-500/30">
            <div className="max-w-4xl mx-auto space-y-8">

              <div className="text-center space-y-2">
                <h3 className="text-xl font-bold text-white">
                  Multi-Hop Scholarly Concept Lineage
                </h3>
                <p className="text-xs sm:text-sm text-slate-300">
                  How Nexora connects fundamental mathematical constraints to modern empirical breakthroughs.
                </p>
              </div>

              {/* Interactive Lineage Flow */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">

                {/* Node 1 */}
                <div className="glass-panel rounded-xl p-4 border-cyan-500/20 text-center space-y-2">
                  <div className="w-8 h-8 mx-auto rounded-full bg-cyan-500/10 border border-cyan-400/40 flex items-center justify-center text-cyan-300 font-mono text-xs">
                    01
                  </div>
                  <h4 className="text-sm font-bold text-white">Attention Complexity</h4>
                  <p className="text-xs text-slate-400 font-mono">O(N²) Quadratic Memory</p>
                  <span className="inline-block text-[10px] text-cyan-300 bg-cyan-950/70 px-2 py-0.5 rounded border border-cyan-500/30 font-mono">
                    Vaswani et al. (2017)
                  </span>
                </div>

                {/* Node 2 */}
                <div className="glass-panel rounded-xl p-4 border-violet-500/20 text-center space-y-2">
                  <div className="w-8 h-8 mx-auto rounded-full bg-violet-500/10 border border-violet-400/40 flex items-center justify-center text-violet-300 font-mono text-xs">
                    02
                  </div>
                  <h4 className="text-sm font-bold text-white">GPU SRAM Bottleneck</h4>
                  <p className="text-xs text-slate-400 font-mono">HBM Read/Write Overhead</p>
                  <span className="inline-block text-[10px] text-violet-300 bg-violet-950/70 px-2 py-0.5 rounded border border-violet-500/30 font-mono">
                    IO-Aware Complexity
                  </span>
                </div>

                {/* Node 3 */}
                <div className="glass-panel rounded-xl p-4 border-cyan-500/20 text-center space-y-2">
                  <div className="w-8 h-8 mx-auto rounded-full bg-cyan-500/10 border border-cyan-400/40 flex items-center justify-center text-cyan-300 font-mono text-xs">
                    03
                  </div>
                  <h4 className="text-sm font-bold text-white">Tiling & Recomputation</h4>
                  <p className="text-xs text-slate-400 font-mono">Online Softmax Fusing</p>
                  <span className="inline-block text-[10px] text-cyan-300 bg-cyan-950/70 px-2 py-0.5 rounded border border-cyan-500/30 font-mono">
                    Dao et al. (2022)
                  </span>
                </div>

                {/* Node 4 */}
                <div className="glass-panel rounded-xl p-4 border-emerald-500/20 text-center space-y-2">
                  <div className="w-8 h-8 mx-auto rounded-full bg-emerald-500/10 border border-emerald-400/40 flex items-center justify-center text-emerald-300 font-mono text-xs">
                    04
                  </div>
                  <h4 className="text-sm font-bold text-white">Verified 4.3x Speedup</h4>
                  <p className="text-xs text-slate-400 font-mono">Exact Math • Zero Quality Loss</p>
                  <span className="inline-block text-[10px] text-emerald-300 bg-emerald-950/70 px-2 py-0.5 rounded border border-emerald-500/30 font-mono">
                    Grounding Entailed
                  </span>
                </div>

              </div>

            </div>
          </div>
        )}

        {/* Tab 3: Retrieval Benchmark Signal */}
        {activeTab === 'benchmark' && (
          <div className="glass-panel rounded-2xl p-6 sm:p-10 border-blue-500/30">
            <div className="max-w-4xl mx-auto space-y-8">

              <div className="text-center space-y-2">
                <h3 className="text-xl font-bold text-white">
                  Deterministic Retrieval & Grounding Evaluation
                </h3>
                <p className="text-xs sm:text-sm text-slate-300">
                  Verified offline evaluation metrics across academic benchmark datasets and calibration audit suites.
                </p>
              </div>

              {/* Benchmark Stat Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

                <div className="glass-panel p-5 rounded-xl border-cyan-500/20 text-center space-y-1">
                  <span className="text-2xl sm:text-3xl font-extrabold font-mono text-cyan-400">
                    0.8462
                  </span>
                  <p className="text-xs text-slate-400 font-mono">Mean Reciprocal Rank</p>
                  <span className="text-[10px] text-slate-500">13 Core Query Suite</span>
                </div>

                <div className="glass-panel p-5 rounded-xl border-cyan-500/20 text-center space-y-1">
                  <span className="text-2xl sm:text-3xl font-extrabold font-mono text-cyan-400">
                    0.8462
                  </span>
                  <p className="text-xs text-slate-400 font-mono">Recall@3</p>
                  <span className="text-[10px] text-slate-500">Hybrid Dense + BM25</span>
                </div>

                <div className="glass-panel p-5 rounded-xl border-purple-500/20 text-center space-y-1">
                  <span className="text-2xl sm:text-3xl font-extrabold font-mono text-purple-400">
                    9 / 9
                  </span>
                  <p className="text-xs text-slate-400 font-mono">False Superiority Gated</p>
                  <span className="text-[10px] text-slate-500">0% Leaked Buzzwords</span>
                </div>

                <div className="glass-panel p-5 rounded-xl border-emerald-500/20 text-center space-y-1">
                  <span className="text-2xl sm:text-3xl font-extrabold font-mono text-emerald-400">
                    100%
                  </span>
                  <p className="text-xs text-slate-400 font-mono">4-Class Calibration</p>
                  <span className="text-[10px] text-slate-500">12/12 Scenarios Verified</span>
                </div>

              </div>

            </div>
          </div>
        )}

      </div>
    </section>
  );
};
