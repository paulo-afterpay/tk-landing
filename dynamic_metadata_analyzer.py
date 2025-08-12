#!/usr/bin/env python3
"""
Dynamic Metadata Analyzer for Knowledge Zone
Step 1: Simple metadata field detection and enhancement

This script analyzes your existing documents and suggests additional metadata fields
that could help connect similar documents together.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import re
from collections import Counter

# You'll need to set your OpenAI API key for LLM features (optional for Step 1)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# For Step 1, we'll focus on pattern-based extraction
# LLM features will be added in later steps

@dataclass
class DocumentAnalysis:
    """Results from analyzing a document for metadata"""
    document_name: str
    current_metadata: Dict[str, Any]
    suggested_metadata: Dict[str, Any]
    confidence_scores: Dict[str, float]
    connection_opportunities: List[str]

class DynamicMetadataAnalyzer:
    """
    Analyzes documents to suggest additional metadata fields
    that could improve document connections
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def analyze_document_content(self, document_title: str, document_url: str = None) -> DocumentAnalysis:
        """
        Analyze a document and suggest additional metadata fields
        
        For now, we'll work with just the document title and any content we can extract.
        In the future, this could be enhanced to read the actual document content.
        """
        
        # For Step 1, we'll analyze based on the document title
        # Later we can enhance this to read actual document content
        
        suggested_metadata = self._extract_metadata_from_title(document_title)
        confidence_scores = self._calculate_confidence_scores(suggested_metadata)
        connection_opportunities = self._identify_connection_opportunities(document_title, suggested_metadata)
        
        return DocumentAnalysis(
            document_name=document_title,
            current_metadata={},  # We'll populate this with existing metadata
            suggested_metadata=suggested_metadata,
            confidence_scores=confidence_scores,
            connection_opportunities=connection_opportunities
        )
    
    def _extract_metadata_from_title(self, title: str) -> Dict[str, Any]:
        """Extract metadata fields from document title using pattern matching and LLM"""
        
        metadata = {}
        
        # Pattern-based extraction (fast, reliable)
        metadata.update(self._extract_patterns(title))
        
        # LLM-based extraction (more comprehensive)
        if self.api_key:
            llm_metadata = self._extract_with_llm(title)
            metadata.update(llm_metadata)
        
        return metadata
    
    def _extract_patterns(self, title: str) -> Dict[str, Any]:
        """Extract metadata using regex patterns"""
        
        metadata = {}
        
        # Time periods (Q1, Q2, 2024, 2025, etc.)
        time_patterns = [
            r'Q[1-4]\s*20\d{2}',  # Q1 2024, Q2 2025
            r'20\d{2}',           # 2024, 2025
            r'H[1-2]\s*20\d{2}',  # H1 2024, H2 2025
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            if matches:
                metadata['time_period'] = matches[0]
                break
        
        # Document types
        doc_type_keywords = {
            'experiment': ['experiment', 'test', 'a/b', 'trial'],
            'research': ['research', 'study', 'analysis', 'insights'],
            'campaign': ['campaign', 'promo', 'promotion', 'marketing'],
            'tech_doc': ['tech doc', 'technical', 'spec', 'architecture'],
            'results': ['results', 'outcomes', 'findings', 'report'],
            'planning': ['plan', 'strategy', 'roadmap', 'preparation']
        }
        
        title_lower = title.lower()
        for doc_type, keywords in doc_type_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                metadata['document_type'] = doc_type
                break
        
        # Product surfaces
        product_keywords = {
            'cash_app_card': ['card', 'cash app card'],
            'afterpay': ['afterpay', 'bnpl'],
            'cash_app_afterpay': ['caap', 'cash app afterpay'],
            'banking': ['banking', 'bank', 'savings'],
            'p2p': ['p2p', 'peer to peer', 'send', 'payment']
        }
        
        for product, keywords in product_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                metadata['product_surface'] = product
                break
        
        # Status indicators
        status_keywords = {
            'completed': ['completed', 'done', 'finished', 'results'],
            'in_progress': ['wip', 'in progress', 'ongoing', 'current'],
            'planned': ['planned', 'upcoming', 'future', 'preparation']
        }
        
        for status, keywords in status_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                metadata['status'] = status
                break
        
        return metadata
    
    def _extract_with_llm(self, title: str) -> Dict[str, Any]:
        """Use LLM to extract additional metadata fields (disabled for Step 1)"""
        # For Step 1, we'll skip LLM extraction to avoid dependencies
        # This will be implemented in Step 2
        return {}
    
    def _call_openai(self, prompt: str) -> str:
        """Make API call to OpenAI (disabled for Step 1)"""
        # For Step 1, we'll skip LLM calls to avoid dependencies
        # This will be implemented in Step 2
        return ""
    
    def _calculate_confidence_scores(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for extracted metadata"""
        
        confidence_scores = {}
        
        for field, value in metadata.items():
            # Higher confidence for pattern-based extraction
            if field in ['time_period', 'document_type', 'product_surface', 'status']:
                confidence_scores[field] = 0.9
            else:
                # Lower confidence for LLM-based extraction
                confidence_scores[field] = 0.7
        
        return confidence_scores
    
    def _identify_connection_opportunities(self, title: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify opportunities to connect this document to others"""
        
        opportunities = []
        
        # Based on extracted metadata, suggest connection types
        if 'time_period' in metadata:
            opportunities.append(f"Connect to other documents from {metadata['time_period']}")
        
        if 'product_surface' in metadata:
            opportunities.append(f"Connect to other {metadata['product_surface']} documents")
        
        if 'document_type' in metadata:
            opportunities.append(f"Connect to other {metadata['document_type']} documents")
        
        if 'teams' in metadata:
            for team in metadata['teams']:
                opportunities.append(f"Connect to other documents involving {team}")
        
        return opportunities

def analyze_existing_documents():
    """Analyze the existing documents from shape-knowledge.html"""
    
    # Documents from your current system
    documents = [
        "Card Habituation Q2 Experiment",
        "Afterpay Goes Green Promo", 
        "A$AP Giveaway Research",
        "CAAP GTM Campaign Tech Doc",
        "Banking Product Tracker Results - Q3 2024",
        "Holiday Campaign Preparation"
    ]
    
    analyzer = DynamicMetadataAnalyzer(OPENAI_API_KEY)
    
    results = []
    
    print("🔍 Analyzing existing documents for dynamic metadata extraction...")
    print("=" * 60)
    
    for doc_title in documents:
        print(f"\nAnalyzing: {doc_title}")
        
        analysis = analyzer.analyze_document_content(doc_title)
        results.append(analysis)
        
        print(f"  Suggested metadata: {analysis.suggested_metadata}")
        print(f"  Connection opportunities: {len(analysis.connection_opportunities)}")
        
        # Show top connection opportunities
        for opportunity in analysis.connection_opportunities[:2]:
            print(f"    • {opportunity}")
    
    return results

def generate_enhanced_metadata_schema(analyses: List[DocumentAnalysis]) -> Dict[str, Any]:
    """Generate an enhanced metadata schema based on analysis results"""
    
    # Collect all suggested fields
    all_fields = {}
    field_frequencies = Counter()
    
    for analysis in analyses:
        for field, value in analysis.suggested_metadata.items():
            field_frequencies[field] += 1
            if field not in all_fields:
                all_fields[field] = []
            all_fields[field].append(value)
    
    # Create enhanced schema
    enhanced_schema = {
        "current_fields": [
            "productSurface",
            "status", 
            "documentType",
            "timePeriod"
        ],
        "suggested_new_fields": {},
        "field_statistics": dict(field_frequencies)
    }
    
    # Add fields that appear in multiple documents
    for field, frequency in field_frequencies.items():
        if frequency >= 2:  # Appears in at least 2 documents
            unique_values = list(set(all_fields[field]))
            enhanced_schema["suggested_new_fields"][field] = {
                "frequency": frequency,
                "sample_values": unique_values[:5],  # Show up to 5 sample values
                "total_unique_values": len(unique_values)
            }
    
    return enhanced_schema

def main():
    """Main function to run the dynamic metadata analysis"""
    
    if not OPENAI_API_KEY:
        print("⚠️  OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        print("   For now, running with pattern-based extraction only...")
    
    # Analyze existing documents
    analyses = analyze_existing_documents()
    
    # Generate enhanced schema
    enhanced_schema = generate_enhanced_metadata_schema(analyses)
    
    # Save results
    results = {
        "analyses": [asdict(analysis) for analysis in analyses],
        "enhanced_schema": enhanced_schema,
        "summary": {
            "total_documents_analyzed": len(analyses),
            "new_fields_suggested": len(enhanced_schema["suggested_new_fields"]),
            "total_connection_opportunities": sum(len(a.connection_opportunities) for a in analyses)
        }
    }
    
    output_file = "dynamic_metadata_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"✅ Analysis complete!")
    print(f"📄 Results saved to: {output_file}")
    print(f"📊 Summary:")
    print(f"   • {results['summary']['total_documents_analyzed']} documents analyzed")
    print(f"   • {results['summary']['new_fields_suggested']} new metadata fields suggested")
    print(f"   • {results['summary']['total_connection_opportunities']} connection opportunities identified")
    
    # Show top suggested fields
    if enhanced_schema["suggested_new_fields"]:
        print(f"\n🎯 Top suggested new metadata fields:")
        for field, info in enhanced_schema["suggested_new_fields"].items():
            print(f"   • {field}: appears in {info['frequency']} documents")
            print(f"     Sample values: {', '.join(info['sample_values'])}")

if __name__ == "__main__":
    main()
