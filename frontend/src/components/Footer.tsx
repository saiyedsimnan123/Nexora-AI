import React from 'react';
import { ShieldCheck, BookOpen, Layers } from 'lucide-react';

interface FooterProps {
  onNavigate: (sectionId: string) => void;
}

const CURRENT_YEAR = 2026;

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  return (
    <footer className="border-t border-white/[0.08] bg-[#05070e] relative text-slate-400 text-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-14">

        <div className="grid grid-cols-1 md:grid-cols-12 gap-10 pb-12 border-b border-white/[0.06]">

          {/* Column 1: Brand & Mission (5 cols) */}
          <div className="md:col-span-5 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500/20 to-purple-600/30 border border-cyan-400/40 p-1.5 flex items-center justify-center">
                <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#00f5d4]" />
              </div>
              <span className="font-extrabold text-lg text-white tracking-tight">
                NEXORA <span className="text-cyan-400 font-mono text-xs">AI</span>
              </span>
            </div>

            <p className="text-sm text-slate-400 max-w-sm leading-relaxed">
              An AI-powered research intelligence platform connecting scientific discoveries,
              benchmarking empirical trade-offs, and gating conclusions on verified evidence.
            </p>

            {/* Verification Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-cyan-400">
              <ShieldCheck className="w-4 h-4 text-cyan-400" />
              <span>Evidence-Grounded Research Verification</span>
            </div>
          </div>

          {/* Column 2: Platform Links (3 cols) */}
          <div className="md:col-span-3 space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-semibold">
              Research Platform
            </h4>
            <ul className="space-y-2 text-xs font-mono">
              <li>
                <button
                  onClick={() => onNavigate('capabilities')}
                  className="hover:text-cyan-400 transition-colors"
                >
                  Dual-Engine Retrieval
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('preview')}
                  className="hover:text-cyan-400 transition-colors"
                >
                  Cross-Paper Matrix
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('preview')}
                  className="hover:text-cyan-400 transition-colors"
                >
                  Claim Grounding Inspector
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('standards')}
                  className="hover:text-cyan-400 transition-colors"
                >
                  Research Standards
                </button>
              </li>
            </ul>
          </div>

          {/* Column 3: Standards & Integrity (4 cols) */}
          <div className="md:col-span-4 space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-semibold">
              Integrity &amp; Standards
            </h4>
            <ul className="space-y-2 text-xs font-mono">
              <li className="flex items-center gap-2 text-slate-400">
                <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
                <span>Peer-Reviewed Literature Grounding</span>
              </li>
              <li className="flex items-center gap-2 text-slate-400">
                <Layers className="w-3.5 h-3.5 text-purple-400" />
                <span>Multi-Stage Hybrid Retrieval (Dense + BM25)</span>
              </li>
              <li className="flex items-center gap-2 text-slate-400">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>Sentence-Level Natural Language Inference</span>
              </li>
              <li className="text-slate-500 pt-1">
                Zero unsupported superiority claims. Verifiable citations only.
              </li>
            </ul>
          </div>

        </div>

        {/* Bottom Bar: Copyright & Platform Policy */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono text-slate-500">
          <div>
            &copy; {CURRENT_YEAR} Nexora AI. Research Intelligence Platform. All rights reserved.
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span>Rigorous Scientific Verification</span>
            <span>&bull;</span>
            <span>Privacy &amp; Security Grounded</span>
          </div>
        </div>

      </div>
    </footer>
  );
};
