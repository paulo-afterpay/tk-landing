#!/usr/bin/env python3
"""
TK Knowledge Zone - AI Document Analyzer
Analyzes Google Docs to extract themes, suggest metadata, and find connections
"""

import json
import re
import requests
from datetime import datetime
from typing import Dict, List, Any, Tuple
import openai
from dataclasses import dataclass, asdict
import os
from urllib.parse import urlparse, parse_qs

@dataclass
class DocumentAnalysis:
    document_name: str
    document_url: str
    extracted_themes: List[str]
    suggested_metadata: Dict[str, str]
    connection_opportunities: List[str]
    confidence_scores: Dict[str, float]
    ai_summary: str
    key_insights: List[str]

class DocumentAnalyzer:
    def __init__(self, openai_api_key: str = None):
        """Initialize the document analyzer with OpenAI API key"""
        self.openai_client = None
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_client = openai
        
        # Document data from your HTML file
        self.documents = [
            {
                "name": "Card Habituation Q2 Experiment",
                "url": "https://docs.google.com/document/d/1r1ON3AmiWx59cAm7J5-ub93D_6DoBfG1DnqGjeUE06I",
                "metadata": {
                    "productSurface": "Cash App Card",
                    "status": "Planned",
                    "documentType": "Experiment",
                    "timePeriod": "Q2 2025"
                }
            },
            {
                "name": "Afterpay Goes Green Promo",
                "url": "https://docs.google.com/document/d/1TotqPZLTm5kQfByXNuDWNhdwi4_qNWU8J0aQcZAWtsc",
                "metadata": {
                    "productSurface": "Afterpay",
                    "status": "Planned",
                    "documentType": "Promotion",
                    "timePeriod": "Q1 2025"
                }
            },
            {
                "name": "A$AP Giveaway Research",
                "url": "https://docs.google.com/document/d/13CXbT7FSl4WXbiPadbEsGkTSo3Fb63DHnoL8k3SE90o",
                "metadata": {
                    "productSurface": "Cash App Afterpay",
                    "status": "Completed",
                    "documentType": "Research",
                    "timePeriod": "Q2 2025"
                }
            },
            {
                "name": "CAAP GTM Campaign Tech Doc",
                "url": "https://docs.google.com/document/d/1ryKXJ4VU2FLxWZM5_KguoH2P5RZZny7mY574L6pJbsU",
                "metadata": {
                    "productSurface": "Cash App Afterpay",
                    "status": "In Progress",
                    "documentType": "Tech Doc",
                    "timePeriod": "Q3 2025"
                }
            },
            {
                "name": "Banking Product Tracker Results - Q3 2024",
                "url": "https://docs.google.com/document/d/1WPeVlMCdxgqdza8Q9HdQJHnab0JontRe",
                "metadata": {
                    "productSurface": "Cash App Card",
                    "status": "Completed",
                    "documentType": "Research",
                    "timePeriod": "Q3 2024"
                }
            },
            {
                "name": "Holiday Campaign Preparation",
                "url": "https://docs.google.com/document/d/17z18uLeY6U3j6vE6vXxrtLTJe1WTWK1LFlBI6ppSnE8",
                "metadata": {
                    "productSurface": "Afterpay",
                    "status": "In Progress",
                    "documentType": "Campaign",
                    "timePeriod": "Q4 2024"
                }
            }
        ]

    def extract_doc_id(self, url: str) -> str:
        """Extract Google Doc ID from URL"""
        try:
            if '/document/d/' in url:
                return url.split('/document/d/')[1].split('/')[0]
            return None
        except:
            return None

    def get_document_content(self, doc_id: str) -> str:
        """
        Get document content from Google Docs
        Note: This requires proper authentication setup
        For demo purposes, we'll simulate content based on document names
        """
        # In a real implementation, you'd use Google Docs API
        # For now, we'll generate simulated content based on document names
        return self.simulate_document_content(doc_id)

    def simulate_document_content(self, doc_id: str) -> str:
        """Generate simulated document content for demo purposes"""
        doc = next((d for d in self.documents if self.extract_doc_id(d['url']) == doc_id), None)
        if not doc:
            return "Document content not available"
        
        # Simulate content based on document type and name
        name = doc['name']
        doc_type = doc['metadata']['documentType']
        product = doc['metadata']['productSurface']
        
        simulated_content = f"""
        {name}
        
        This {doc_type.lower()} focuses on {product} initiatives.
        
        Key areas covered:
        - User engagement strategies
        - Product feature analysis
        - Market research insights
        - Performance metrics
        - Implementation roadmap
        
        The document outlines strategic approaches for improving user experience
        and driving product adoption across our platform.
        """
        
        return simulated_content

    def analyze_with_ai(self, content: str, document_name: str) -> Dict[str, Any]:
        """Analyze document content using OpenAI"""
        if not self.openai_client:
            # Fallback analysis without AI
            return self.fallback_analysis(content, document_name)
        
        try:
            prompt = f"""
            Analyze this document content and provide insights:
            
            Document: {document_name}
            Content: {content}
            
            Please provide:
            1. Key themes (3-5 themes)
            2. Suggested metadata tags
            3. Connection opportunities with other documents
            4. A brief summary
            5. Key insights
            
            Format as JSON with keys: themes, metadata_suggestions, connections, summary, insights
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return self.fallback_analysis(content, document_name)

    def fallback_analysis(self, content: str, document_name: str) -> Dict[str, Any]:
        """Provide analysis without AI as fallback"""
        # Extract themes based on common keywords
        themes = []
        content_lower = content.lower()
        
        theme_keywords = {
            "User Experience": ["user", "experience", "ux", "interface", "usability"],
            "Product Strategy": ["strategy", "product", "roadmap", "planning"],
            "Marketing": ["marketing", "campaign", "promotion", "advertising"],
            "Research": ["research", "analysis", "insights", "data", "findings"],
            "Technology": ["tech", "implementation", "development", "system"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        # Generate metadata suggestions
        metadata_suggestions = {}
        if "experiment" in document_name.lower():
            metadata_suggestions["priority"] = "High"
            metadata_suggestions["methodology"] = "A/B Testing"
        elif "campaign" in document_name.lower():
            metadata_suggestions["priority"] = "Medium"
            metadata_suggestions["channel"] = "Multi-channel"
        elif "research" in document_name.lower():
            metadata_suggestions["priority"] = "High"
            metadata_suggestions["methodology"] = "Quantitative"
        
        # Generate connection opportunities
        connections = [
            f"Could be connected to other {document_name.split()[0]} documents",
            "Shares themes with user experience initiatives",
            "Related to product development roadmap"
        ]
        
        return {
            "themes": themes[:5],
            "metadata_suggestions": metadata_suggestions,
            "connections": connections[:3],
            "summary": f"Analysis of {document_name} reveals key insights about product strategy and user engagement.",
            "insights": [
                "Document contains strategic planning elements",
                "Focus on user-centric approach",
                "Alignment with product roadmap"
            ]
        }

    def analyze_document(self, document: Dict[str, Any]) -> DocumentAnalysis:
        """Analyze a single document"""
        print(f"Analyzing: {document['name']}")
        
        doc_id = self.extract_doc_id(document['url'])
        content = self.get_document_content(doc_id) if doc_id else ""
        
        # Get AI analysis
        ai_result = self.analyze_with_ai(content, document['name'])
        
        # Calculate confidence scores
        confidence_scores = {
            "theme_extraction": 0.85,
            "metadata_suggestions": 0.75,
            "connection_opportunities": 0.70
        }
        
        return DocumentAnalysis(
            document_name=document['name'],
            document_url=document['url'],
            extracted_themes=ai_result.get('themes', []),
            suggested_metadata=ai_result.get('metadata_suggestions', {}),
            connection_opportunities=ai_result.get('connections', []),
            confidence_scores=confidence_scores,
            ai_summary=ai_result.get('summary', ''),
            key_insights=ai_result.get('insights', [])
        )

    def find_cross_document_connections(self, analyses: List[DocumentAnalysis]) -> Dict[str, List[str]]:
        """Find connections between documents based on themes and content"""
        connections = {}
        
        for i, analysis1 in enumerate(analyses):
            doc_connections = []
            
            for j, analysis2 in enumerate(analyses):
                if i != j:
                    # Check for theme overlap
                    common_themes = set(analysis1.extracted_themes) & set(analysis2.extracted_themes)
                    if common_themes:
                        doc_connections.append(
                            f"Shares themes with '{analysis2.document_name}': {', '.join(common_themes)}"
                        )
                    
                    # Check for metadata similarity
                    common_metadata = set(analysis1.suggested_metadata.values()) & set(analysis2.suggested_metadata.values())
                    if common_metadata:
                        doc_connections.append(
                            f"Similar approach to '{analysis2.document_name}'"
                        )
            
            connections[analysis1.document_name] = doc_connections[:3]  # Limit to top 3
        
        return connections

    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate complete analysis report for all documents"""
        print("Starting document analysis...")
        
        analyses = []
        for document in self.documents:
            analysis = self.analyze_document(document)
            analyses.append(analysis)
        
        # Find cross-document connections
        cross_connections = self.find_cross_document_connections(analyses)
        
        # Update analyses with cross-document connections
        for analysis in analyses:
            if analysis.document_name in cross_connections:
                analysis.connection_opportunities.extend(cross_connections[analysis.document_name])
        
        # Generate summary statistics
        all_themes = []
        all_metadata_keys = set()
        
        for analysis in analyses:
            all_themes.extend(analysis.extracted_themes)
            all_metadata_keys.update(analysis.suggested_metadata.keys())
        
        theme_frequency = {}
        for theme in all_themes:
            theme_frequency[theme] = theme_frequency.get(theme, 0) + 1
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_documents": len(analyses),
            "analyses": [asdict(analysis) for analysis in analyses],
            "summary_statistics": {
                "most_common_themes": sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
                "unique_metadata_suggestions": len(all_metadata_keys),
                "total_connections_found": sum(len(a.connection_opportunities) for a in analyses)
            }
        }
        
        return report

    def save_analysis(self, output_file: str = "dynamic_metadata_analysis.json"):
        """Generate and save analysis to JSON file"""
        report = self.generate_analysis_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis saved to {output_file}")
        return report

def main():
    """Main function to run the analysis"""
    # Check for OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found. Using fallback analysis.")
    
    # Initialize analyzer
    analyzer = DocumentAnalyzer(openai_key)
    
    # Generate analysis
    report = analyzer.save_analysis()
    
    print("\n=== Analysis Complete ===")
    print(f"Analyzed {report['total_documents']} documents")
    print(f"Found {report['summary_statistics']['total_connections_found']} potential connections")
    print(f"Most common themes: {', '.join([theme for theme, count in report['summary_statistics']['most_common_themes']])}")

if __name__ == "__main__":
    main()
