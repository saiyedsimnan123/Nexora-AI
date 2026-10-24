import React from 'react';
import { ShieldCheck, BookOpen, Layers, CheckCircle2, FileCheck } from 'lucide-react';

export const ArchitectureBanner: React.FC = () => {
  return (
    <section id="standards" className="py-16 md:py-24 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Banner Container */}
        <div className="glass-panel rounded-3xl p-8 sm:p-12 border-cyan-500/30 relative overflow-hidden bg-gradient-to-b from-[#090e24]/90 to-[#060914]/90">
          <div className="absolute top-0 right-0 w-96 h-96 bg-purple-600/10 rounded-full blur-[120px] pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />

          <div className="relative z-10 max-w-4xl mx-auto text-center space-y-4 mb-10">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-xs font-mono text-cyan-300">
              <ShieldCheck className="w-4 h-4 text-cyan-400" />
              <span>Research Integrity &amp; Verification Standards</span>
            </div>

            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-white tracking-tight">
              Engineered for Scientific Rigor and Verification
            </h2>

            <p className="text-sm sm:text-base text-slate-300 max-w-2xl mx-auto leading-relaxed">
              Every query, comparative synthesis, and citation in Nexora AI is anchored
              in peer-reviewed literature and gated against ungrounded speculation.
            </p>
          </div>

          {/* 4 Core Pillars Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">

            {/* Pillar 1: Literature Grounding */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-300">
                <BookOpen className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Peer-Reviewed Grounding</h3>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Direct indexing and semantic retrieval over verified research papers, preprints, and open-access scientific literature.
              </p>
              <div className="text-[11px] font-mono text-cyan-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Curated Academic Corpus</span>
              </div>
            </div>

            {/* Pillar 2: Hybrid Retrieval */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-400/30 flex items-center justify-center text-purple-300">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Multi-Stage Retrieval</h3>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Reciprocal rank fusion uniting high-dimensional semantic embeddings with exact lexical BM25 keyword matching.
              </p>
              <div className="text-[11px] font-mono text-purple-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Dense + Lexical Fusion</span>
              </div>
            </div>

            {/* Pillar 3: Claim Entailment */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-400/30 flex items-center justify-center text-blue-300">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Claim Verification</h3>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Sentence-level Natural Language Inference gating every asserted claim against extracted citation evidence contexts.
              </p>
              <div className="text-[11px] font-mono text-blue-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Zero Unsupported Claims</span>
              </div>
            </div>

            {/* Pillar 4: Transparent Attribution */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-400/30 flex items-center justify-center text-emerald-300">
                <FileCheck className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Scientific Attribution</h3>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Complete citation lineage tracing, evidence span highlighting, and comparative trade-off matrices across papers.
              </p>
              <div className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Verifiable Citation Trace</span>
              </div>
            </div>

          </div>

        </div>

      </div>
    </section>
  );
};
