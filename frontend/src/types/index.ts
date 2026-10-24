export interface SampleQuery {
  id: string;
  query: string;
  category: string;
  targetPaper?: string;
}

export interface FeatureCapability {
  id: string;
  title: string;
  tagline: string;
  description: string;
  badge: string;
  iconName: 'Compass' | 'GitCompare' | 'ShieldCheck';
  metrics: string;
  highlights: string[];
}

export type PreviewTab = 'verification' | 'lineage' | 'benchmark';

export interface VerifiedClaimItem {
  id: string;
  claim: string;
  status: 'SUPPORTED' | 'CONTRADICTED' | 'LIMITED_EVIDENCE' | 'INSUFFICIENT_EVIDENCE';
  citations: string[];
  excerpt: string;
  paperTitle: string;
}
