import React from 'react';
import { Database, ShieldCheck, Activity, Cpu, Server, CheckCircle2 } from 'lucide-react';

export const ArchitectureBanner: React.FC = () => {
  return (
    <section id="architecture" className="py-16 md:py-24 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Banner Container */}
        <div className="glass-panel rounded-3xl p-8 sm:p-12 border-cyan-500/30 relative overflow-hidden bg-gradient-to-b from-[#090e24]/90 to-[#060914]/90">
          <div className="absolute top-0 right-0 w-96 h-96 bg-purple-600/10 rounded-full blur-[120px] pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />

          <div className="relative z-10 max-w-4xl mx-auto text-center space-y-4 mb-10">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-xs font-mono text-cyan-300">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>Production Backend Foundation</span>
            </div>

            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-white tracking-tight">
              Backed by a Fully Verified, Fail-Closed Architecture
            </h2>

            <p className="text-sm sm:text-base text-slate-300 max-w-2xl mx-auto leading-relaxed">
              Every query, comparison matrix, and citation verification in Nexora AI is anchored
              in a resilient, container-ready microservices foundation.
            </p>
          </div>

          {/* 4 Core Pillars Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">

            {/* Pillar 1 */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-300">
                <Database className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">PostgreSQL 16</h3>
              <p className="text-xs text-slate-400 font-mono leading-relaxed">
                10 relational tables, Alembic head 044c24ad4c33, SHA-256 key hashing & strict multi-tenant scoping.
              </p>
              <div className="text-[11px] font-mono text-cyan-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Fail-Closed Schema Readiness</span>
              </div>
            </div>

            {/* Pillar 2 */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-400/30 flex items-center justify-center text-purple-300">
                <Server className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Qdrant Vector Store</h3>
              <p className="text-xs text-slate-400 font-mono leading-relaxed">
                Cosine similarity clustering, dimension parity validation, and in-memory BM25 lexical fusion index.
              </p>
              <div className="text-[11px] font-mono text-purple-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Dual-Store RRF Retrieval</span>
              </div>
            </div>

            {/* Pillar 3 */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-400/30 flex items-center justify-center text-blue-300">
                <Activity className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">Redis 7 Rate Limiter</h3>
              <p className="text-xs text-slate-400 font-mono leading-relaxed">
                Atomic sliding-window Lua script, tamper-proof hashed client tracking & 503 readiness probe integration.
              </p>
              <div className="text-[11px] font-mono text-blue-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Distributed Abuse Gating</span>
              </div>
            </div>

            {/* Pillar 4 */}
            <div className="glass-panel p-5 rounded-2xl border-white/[0.08] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-400/30 flex items-center justify-center text-emerald-300">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white">FastAPI ASGI Engine</h3>
              <p className="text-xs text-slate-400 font-mono leading-relaxed">
                Request correlation IDs, structured JSON logs, zero secret leakage, and graceful resource lifespan teardown.
              </p>
              <div className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>2,659 Regression Tests Passed</span>
              </div>
            </div>

          </div>

        </div>

      </div>
    </section>
  );
};
