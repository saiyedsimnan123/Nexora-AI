import React, { useState, useRef, useId } from 'react';
import {
  X,
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Sparkles,
  ArrowRight,
  RefreshCw,
  Search,
  ShieldCheck
} from 'lucide-react';
import { uploadDocument } from '../services/api';
import type { DocumentIngestResponse } from '../types/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPaperUploaded?: (documentId: string, paperId: string, filename: string) => void;
  onAnalyzePaper?: (paperId: string, title?: string) => void;
  onSearchPaper?: (query: string) => void;
}

function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onPaperUploaded,
  onAnalyzePaper,
  onSearchPaper,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [paperId, setPaperId] = useState<string>(() => generateUUID());
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadResult, setUploadResult] = useState<DocumentIngestResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const titleId = useId();

  if (!isOpen) return null;

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (file: File) => {
    setErrorMessage(null);
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage('Only PDF research documents are supported.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 50MB maximum limit.');
      return;
    }
    setSelectedFile(file);
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMessage('Please select a PDF research paper to upload.');
      return;
    }

    setIsUploading(true);
    setErrorMessage(null);

    try {
      const activePaperId = paperId.trim() || generateUUID();
      const result = await uploadDocument(activePaperId, selectedFile);
      setUploadResult(result);
      if (onPaperUploaded) {
        onPaperUploaded(result.document_id, result.paper_id, selectedFile.name);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed. Please verify the document format.';
      setErrorMessage(msg);
    } finally {
      setIsUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setUploadResult(null);
    setErrorMessage(null);
    setPaperId(generateUUID());
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200"
    >
      <div className="relative w-full max-w-2xl bg-[#090d21] border border-cyan-500/30 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/[0.08] bg-[#070a1a]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-300">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div>
              <h3 id={titleId} className="text-base font-bold text-white tracking-tight">
                Upload Research Document
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Full-Text PDF Parsing • Relational Storage • Vector Indexing
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close upload dialog"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {uploadResult ? (
            /* Success State */
            <div className="space-y-5 text-center py-4">
              <div className="w-16 h-16 mx-auto rounded-2xl bg-emerald-500/10 border border-emerald-400/40 flex items-center justify-center text-emerald-400 shadow-[0_0_24px_rgba(52,211,153,0.3)] animate-in zoom-in-95 duration-200">
                <CheckCircle2 className="w-8 h-8" />
              </div>

              <div className="space-y-1">
                <h4 className="text-lg font-bold text-white">Paper Ingestion Complete!</h4>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  The document has been parsed into chunks, assigned persistent provenance IDs, and indexed into the vector store.
                </p>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-lg mx-auto text-left">
                <div className="glass-panel p-3 rounded-xl border-white/10 bg-[#060818]">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block">Paper ID</span>
                  <span className="text-xs font-mono font-semibold text-cyan-300 truncate block" title={uploadResult.paper_id}>
                    {uploadResult.paper_id.slice(0, 8)}...
                  </span>
                </div>
                <div className="glass-panel p-3 rounded-xl border-white/10 bg-[#060818]">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block">Document ID</span>
                  <span className="text-xs font-mono font-semibold text-purple-300 truncate block" title={uploadResult.document_id}>
                    {uploadResult.document_id.slice(0, 8)}...
                  </span>
                </div>
                <div className="glass-panel p-3 rounded-xl border-white/10 bg-[#060818]">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block">Chunks</span>
                  <span className="text-sm font-mono font-bold text-emerald-400 block">
                    {uploadResult.chunks_count}
                  </span>
                </div>
                <div className="glass-panel p-3 rounded-xl border-white/10 bg-[#060818]">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block">Indexed</span>
                  <span className="text-sm font-mono font-bold text-cyan-400 block">
                    {uploadResult.indexed_chunks_count}
                  </span>
                </div>
              </div>

              {/* Next Action Options */}
              <div className="pt-3 flex flex-wrap items-center justify-center gap-3">
                {onAnalyzePaper && (
                  <button
                    onClick={() => {
                      onClose();
                      onAnalyzePaper(uploadResult.paper_id, selectedFile?.name);
                    }}
                    className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-xs font-mono text-slate-900 bg-gradient-to-r from-cyan-400 to-teal-300 hover:from-cyan-300 hover:to-teal-200 shadow-[0_0_16px_rgba(0,245,212,0.3)] transition-all"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>Analyze 7 Dimensions</span>
                  </button>
                )}

                {onSearchPaper && (
                  <button
                    onClick={() => {
                      onClose();
                      const query = selectedFile ? selectedFile.name.replace(/\.pdf$/i, '') : 'research paper findings';
                      onSearchPaper(query);
                    }}
                    className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-xs font-mono text-slate-200 glass-panel hover:bg-slate-800 border-white/10 transition-all"
                  >
                    <Search className="w-4 h-4 text-cyan-400" />
                    <span>Ask Research Question</span>
                  </button>
                )}

                <button
                  onClick={handleReset}
                  className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-xs font-mono text-slate-400 hover:text-white transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Upload Another</span>
                </button>
              </div>
            </div>
          ) : (
            /* Upload Form State */
            <form onSubmit={handleUploadSubmit} className="space-y-5">
              {/* Dropzone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragging
                    ? 'border-cyan-400 bg-cyan-500/10 shadow-[0_0_24px_rgba(0,245,212,0.2)]'
                    : selectedFile
                    ? 'border-emerald-400/50 bg-emerald-500/5'
                    : 'border-white/15 hover:border-cyan-500/40 hover:bg-slate-900/40'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={handleFileChange}
                  className="hidden"
                  aria-label="Upload PDF file"
                />

                {selectedFile ? (
                  <div className="space-y-2">
                    <div className="w-12 h-12 mx-auto rounded-xl bg-emerald-500/10 border border-emerald-400/40 flex items-center justify-center text-emerald-400">
                      <FileText className="w-6 h-6" />
                    </div>
                    <p className="text-sm font-semibold text-white truncate max-w-md mx-auto">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs font-mono text-slate-400">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for ingestion
                    </p>
                    <span className="inline-block text-[11px] font-mono text-cyan-400 underline pt-1">
                      Click to choose different file
                    </span>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="w-12 h-12 mx-auto rounded-xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-400 shadow-[0_0_16px_rgba(0,245,212,0.15)]">
                      <UploadCloud className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-white">
                        Drop research paper PDF here, or <span className="text-cyan-400 underline">browse</span>
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        Accepts standard research paper PDFs up to 50MB
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Paper ID Setting (Optional Customization) */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <label htmlFor="paper-id-input">Paper Provenance UUID:</label>
                  <button
                    type="button"
                    onClick={() => setPaperId(generateUUID())}
                    className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Regenerate ID</span>
                  </button>
                </div>
                <input
                  id="paper-id-input"
                  type="text"
                  value={paperId}
                  onChange={(e) => setPaperId(e.target.value)}
                  placeholder="3fa85f64-5717-4562-b3fc-2c963f66afa6"
                  className="w-full bg-[#060818] border border-white/10 rounded-xl px-3.5 py-2 text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              {/* Error Message */}
              {errorMessage && (
                <div className="p-3 rounded-xl bg-red-950/60 border border-red-500/30 text-xs text-red-200 flex items-start gap-2 animate-in fade-in duration-150">
                  <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <span>{errorMessage}</span>
                </div>
              )}

              {/* Submit Button */}
              <div className="pt-2 flex items-center justify-end gap-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2.5 rounded-xl text-xs font-mono text-slate-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!selectedFile || isUploading}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-xs font-mono text-slate-900 bg-gradient-to-r from-cyan-400 to-teal-300 hover:from-cyan-300 hover:to-teal-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(0,245,212,0.3)] transition-all"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Ingesting Paper...</span>
                    </>
                  ) : (
                    <>
                      <span>Upload & Ingest Paper</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </form>
          )}
        </div>

        {/* Footer Guarantee */}
        <div className="p-3.5 border-t border-white/[0.06] bg-[#060917] flex items-center justify-between text-[11px] font-mono text-slate-400">
          <div className="flex items-center gap-1.5 text-cyan-300">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>Strict Attribution Provenance Tracked</span>
          </div>
          <span className="hidden sm:inline text-slate-500">Max size 50MB • PDF Only</span>
        </div>
      </div>
    </div>
  );
};
