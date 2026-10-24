import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { HeroSection } from './components/HeroSection';
import { FeatureCapabilities } from './components/FeatureCapabilities';
import { IntelligencePreview } from './components/IntelligencePreview';
import { ArchitectureBanner } from './components/ArchitectureBanner';
import { Footer } from './components/Footer';
import { SearchModal } from './components/SearchModal';

export const App: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const handleSearchSubmit = (query: string) => {
    setSearchQuery(query);
    setIsModalOpen(true);
  };

  const handleExploreClick = () => {
    setIsModalOpen(true);
  };

  const handleNavigate = (sectionId: string) => {
    if (sectionId === 'hero') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#070913] text-slate-100 flex flex-col font-sans bg-grid-pattern relative overflow-x-hidden w-full">
      {/* Top Navbar */}
      <Navbar onExploreClick={handleExploreClick} onNavigate={handleNavigate} />

      {/* Main Content Sections */}
      <main className="flex-1">
        {/* Section 1: Hero with Signature 3D Centerpiece */}
        <HeroSection
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onSearchSubmit={handleSearchSubmit}
          onViewArchitecture={() => handleNavigate('architecture')}
        />

        {/* Section 2: Research Capabilities (3 Signature Panels) */}
        <FeatureCapabilities />

        {/* Section 3: Research Intelligence Matrix Preview */}
        <IntelligencePreview />

        {/* Section 4: Production Architecture Foundation */}
        <ArchitectureBanner />
      </main>

      {/* Section 5: Professional Footer */}
      <Footer onNavigate={handleNavigate} />

      {/* Interactive Search / Pipeline Preview Modal */}
      <SearchModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        query={searchQuery}
      />
    </div>
  );
};

export default App;
