import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { HeroSection } from './components/HeroSection';
import { FeatureCapabilities } from './components/FeatureCapabilities';
import { IntelligencePreview } from './components/IntelligencePreview';
import { ArchitectureBanner } from './components/ArchitectureBanner';
import { Footer } from './components/Footer';
import { SearchModal } from './components/SearchModal';
import { ComparePapers } from './components/ComparePapers';
import { UploadModal } from './components/UploadModal';
import { AnalysisModal } from './components/AnalysisModal';

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<'home' | 'compare'>('home');
  const [comparePaperIds, setComparePaperIds] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState<boolean>(false);
  const [analysisPaperId, setAnalysisPaperId] = useState<string | null>(null);
  const [analysisPaperTitle, setAnalysisPaperTitle] = useState<string | null>(null);

  const handleSearchSubmit = (query: string) => {
    setSearchQuery(query);
    setIsModalOpen(true);
  };

  const handleExploreClick = () => {
    setIsModalOpen(true);
  };

  const handleOpenUpload = () => {
    setIsUploadModalOpen(true);
  };

  const handleAnalyzePaper = (paperId: string, title?: string) => {
    setAnalysisPaperId(paperId);
    setAnalysisPaperTitle(title || null);
    setIsAnalysisModalOpen(true);
  };

  const handleNavigate = (sectionId: string) => {
    if (currentView !== 'home') {
      setCurrentView('home');
      setTimeout(() => {
        executeScroll(sectionId);
      }, 50);
      return;
    }
    executeScroll(sectionId);
  };

  const executeScroll = (sectionId: string) => {
    if (sectionId === 'hero') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    const targetId = sectionId === 'architecture' ? 'standards' : sectionId;
    const element = document.getElementById(targetId) || document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleCompareHandoff = (paperIds: string[]) => {
    setComparePaperIds(paperIds);
    setCurrentView('compare');
    setIsModalOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#070913] text-slate-100 flex flex-col font-sans bg-grid-pattern relative overflow-x-hidden w-full">
      {/* Top Navbar */}
      <Navbar
        onExploreClick={handleExploreClick}
        onNavigate={handleNavigate}
        currentView={currentView}
        onViewChange={(view) => {
          setCurrentView(view);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
        onUploadClick={handleOpenUpload}
      />

      {/* Main Content Sections */}
      <main className="flex-1">
        {currentView === 'compare' ? (
          <ComparePapers
            initialPaperIds={comparePaperIds}
            onBackToResearch={() => {
              setCurrentView('home');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            onAnalyzePaper={handleAnalyzePaper}
            onOpenUpload={handleOpenUpload}
          />
        ) : (
          <>
            {/* Section 1: Hero with Signature 3D Centerpiece */}
            <HeroSection
              searchQuery={searchQuery}
              onSearchChange={setSearchQuery}
              onSearchSubmit={handleSearchSubmit}
              onViewArchitecture={() => handleNavigate('standards')}
              onUploadClick={handleOpenUpload}
            />

            {/* Section 2: Research Capabilities (3 Signature Panels) */}
            <FeatureCapabilities />

            {/* Section 3: Research Intelligence Matrix Preview */}
            <IntelligencePreview />

            {/* Section 4: Production Architecture Foundation */}
            <ArchitectureBanner />
          </>
        )}
      </main>

      {/* Section 5: Professional Footer */}
      <Footer onNavigate={handleNavigate} />

      {/* Interactive Search / Pipeline Preview Modal */}
      <SearchModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        query={searchQuery}
        onComparePapers={handleCompareHandoff}
        onAnalyzePaper={handleAnalyzePaper}
        onOpenUpload={handleOpenUpload}
      />

      {/* Document Upload Modal */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onAnalyzePaper={handleAnalyzePaper}
        onSearchPaper={handleSearchSubmit}
      />

      {/* Structured 7-Dimension Paper Analysis Modal */}
      <AnalysisModal
        isOpen={isAnalysisModalOpen}
        onClose={() => setIsAnalysisModalOpen(false)}
        paperId={analysisPaperId}
        paperTitle={analysisPaperTitle}
        onComparePaper={(pid) => handleCompareHandoff([pid])}
      />
    </div>
  );
};

export default App;
