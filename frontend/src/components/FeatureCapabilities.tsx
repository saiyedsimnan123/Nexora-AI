import React, { useState } from 'react';
import {
  Compass,
  GitCompare,
  ShieldCheck,
  Layers,
  CheckCircle
} from 'lucide-react';
import type { FeatureCapability } from '../types';

const FEATURES: FeatureCapability[] = [
  {
    id: 'discovery',
    title: 'Research Discovery',
    tagline: 'Hybrid Semantic & Lexical Fusion',
    badge: 'Dual-Engine Retrieval',
    iconName: 'Compass',
    description:
      'Moves beyond keyword search with reciprocal rank fusion (RRF) combining dense vector embeddings and BM25 lexical precision. Maps structural chunk lineage directly back to source PDF pages.',
    metrics: 'MRR 0.8462 • Recall@3 0.8462',
    highlights: [
      'Structure-aware PDF chunking preserving tables & equations',
      'Dual vector store (Qdrant) and lexical index (BM25)',
      'Cross-encoder re-ranking with quality-score gating',
    ],
  },
  {
    id: 'comparison',
    title: 'Cross-Paper Comparison',
    tagline: 'Multi-Dimensional Empirical Matrix',
    badge: '4 Calibrated Outcomes',
    iconName: 'GitCompare',
    description:
      'Aligns two or more research papers across standardized dimensions: methods, datasets, baselines, metrics, and limitations. Automatically detects metric inversions and noise-level deltas.',
    metrics: '100% Calibration Accuracy (12/12)',
    highlights: [
      'Distinguishes Demonstrated, Limited, Conflicting, & Insufficient evidence',
      'Detects metric inversions (e.g. higher accuracy vs 10x latency)',
      'Downgrades single-run or non-overlapping benchmark claims',
    ],
  },
  {
    id: 'evidence',
    title: 'Evidence Intelligence',
    tagline: 'Sentence-Level Claim Grounding',
    badge: 'Zero Hallucination Gating',
    iconName: 'ShieldCheck',
    description:
      'Extracts individual sentence claims and verifies entailment against cited [E#] excerpts. Strictly prohibits unsupported superiority buzzwords without comparative baseline metrics.',
    metrics: '9/9 False Superiority Claims Blocked',
    highlights: [
      'Fail-closed answer grounding on unverified assertions',
      'Automatic contradiction detection across cited chunks',
      'Honest abstention when literature evidence is insufficient',
    ],
  },
];

export const FeatureCapabilities: React.FC = () => {
  const [activeFeature, setActiveFeature] = useState<string>('discovery');

  return (
    <section id="capabilities" className="py-20 md:py-28 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full glass-panel border-cyan-500/20 text-xs font-mono text-cyan-300">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Core Scholarly Capabilities</span>
          </div>

          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white tracking-tight">
            Engineered for{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-purple-400 text-glow-cyan">
              Scientific Rigor
            </span>
          </h2>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            Conventional LLMs hallucinate citations and exaggerate performance. Nexora AI
            enforces evidence-first architecture: retrieve verified sources first, verify
            claims second, conclude last.
          </p>
        </div>

        {/* 3 Signature Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {FEATURES.map((feat) => {
            const isSelected = activeFeature === feat.id;

            return (
              <div
                key={feat.id}
                onClick={() => setActiveFeature(feat.id)}
                className={`relative rounded-2xl p-7 transition-all duration-300 cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? 'glass-panel-glow border-cyan-400/50 -translate-y-1.5'
                    : 'glass-panel glass-panel-hover border-white/10'
                }`}
              >
                {/* Top Icon & Badge */}
                <div>
                  <div className="flex items-center justify-between mb-5">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center border transition-all ${
                        feat.id === 'discovery'
                          ? 'bg-cyan-500/10 border-cyan-400/30 text-cyan-300 shadow-[0_0_15px_rgba(0,245,212,0.2)]'
                          : feat.id === 'comparison'
                          ? 'bg-purple-500/10 border-purple-400/30 text-purple-300 shadow-[0_0_15px_rgba(168,85,247,0.2)]'
                          : 'bg-emerald-500/10 border-emerald-400/30 text-emerald-300 shadow-[0_0_15px_rgba(52,211,153,0.2)]'
                      }`}
                    >
                      {feat.id === 'discovery' && <Compass className="w-6 h-6" />}
                      {feat.id === 'comparison' && <GitCompare className="w-6 h-6" />}
                      {feat.id === 'evidence' && <ShieldCheck className="w-6 h-6" />}
                    </div>

                    <span className="text-[11px] font-mono uppercase tracking-wider px-2.5 py-1 rounded-full glass-panel border-white/10 text-slate-300">
                      {feat.badge}
                    </span>
                  </div>

                  {/* Title & Tagline */}
                  <h3 className="text-xl font-bold text-white mb-1.5 tracking-tight">
                    {feat.title}
                  </h3>
                  <p className="text-xs font-mono uppercase tracking-wider text-cyan-400 mb-3.5">
                    {feat.tagline}
                  </p>

                  {/* Description */}
                  <p className="text-sm text-slate-300 leading-relaxed mb-6">
                    {feat.description}
                  </p>
                </div>

                {/* Highlights List & Metric Footer */}
                <div className="pt-4 border-t border-white/[0.08] space-y-4">
                  <div className="space-y-2">
                    {feat.highlights.map((item, idx) => (
                      <div key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                        <CheckCircle className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>

                  {/* Verified Metric Badge */}
                  <div className="flex items-center justify-between pt-2">
                    <span className="text-[11px] font-mono text-slate-400">
                      Offline Benchmark:
                    </span>
                    <span className="text-xs font-mono font-semibold text-cyan-300 bg-cyan-950/60 px-2.5 py-1 rounded border border-cyan-500/30">
                      {feat.metrics}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
};
