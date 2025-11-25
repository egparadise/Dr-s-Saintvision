"""
Metrics Calculator for DR-Saintvision
Provides evaluation metrics for debate quality and accuracy
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import re

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for debate evaluation"""

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._encoder = None

        if use_embeddings:
            self._init_encoder()

    def _init_encoder(self):
        """Initialize sentence encoder for semantic similarity"""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence encoder initialized")
        except ImportError:
            logger.warning("sentence-transformers not installed, semantic metrics disabled")
            self.use_embeddings = False
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            self.use_embeddings = False

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self._encoder or not text1 or not text2:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            embeddings = self._encoder.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def keyword_coverage(
        self,
        text: str,
        keywords: List[str],
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Calculate keyword coverage in text"""
        if not text or not keywords:
            return {"coverage": 0.0, "found": [], "missing": keywords}

        if not case_sensitive:
            text = text.lower()
            keywords = [k.lower() for k in keywords]

        found = [k for k in keywords if k in text]
        missing = [k for k in keywords if k not in text]

        coverage = len(found) / len(keywords) if keywords else 0.0

        return {
            "coverage": coverage,
            "found": found,
            "missing": missing,
            "total_keywords": len(keywords),
            "found_count": len(found)
        }

    def text_coherence(self, text: str) -> float:
        """
        Estimate text coherence based on various factors

        Returns a score from 0 to 1
        """
        if not text:
            return 0.0

        score = 0.5  # Base score

        # Check sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 3:
            score += 0.1

        # Check for transition words (indicates logical flow)
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover',
            'additionally', 'consequently', 'nevertheless', 'thus',
            'in contrast', 'similarly', 'as a result', 'for example'
        ]

        text_lower = text.lower()
        transition_count = sum(1 for tw in transition_words if tw in text_lower)
        score += min(transition_count * 0.05, 0.2)

        # Check for paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 2:
            score += 0.1

        # Penalize very short or very repetitive text
        words = text.split()
        if len(words) < 50:
            score -= 0.1

        unique_words = set(w.lower() for w in words)
        vocabulary_ratio = len(unique_words) / len(words) if words else 0
        if vocabulary_ratio < 0.3:  # Very repetitive
            score -= 0.1

        return max(0.0, min(1.0, score))

    def completeness_score(
        self,
        text: str,
        expected_sections: List[str]
    ) -> Dict[str, Any]:
        """Check if text contains expected sections/topics"""
        if not text or not expected_sections:
            return {"score": 0.0, "found_sections": [], "missing_sections": expected_sections}

        text_lower = text.lower()
        found = []
        missing = []

        for section in expected_sections:
            if section.lower() in text_lower:
                found.append(section)
            else:
                missing.append(section)

        score = len(found) / len(expected_sections)

        return {
            "score": score,
            "found_sections": found,
            "missing_sections": missing
        }

    def calculate_debate_metrics(
        self,
        debate_result: Dict[str, Any],
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a debate result

        Args:
            debate_result: Result from DebateManager
            reference_answer: Optional ground truth answer

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Extract components
        search_analysis = debate_result.get('search_analysis', {})
        reasoning_analysis = debate_result.get('reasoning_analysis', {})
        final_synthesis = debate_result.get('final_synthesis', {})

        final_answer = final_synthesis.get('final_answer', '')

        # Time metrics
        metrics['total_time'] = debate_result.get('debate_time', 0)
        metrics['search_time'] = search_analysis.get('processing_time', 0)
        metrics['reasoning_time'] = reasoning_analysis.get('processing_time', 0)
        metrics['synthesis_time'] = final_synthesis.get('processing_time', 0)

        # Confidence metrics
        metrics['confidence'] = {
            'search': search_analysis.get('confidence', 0),
            'reasoning': reasoning_analysis.get('confidence', 0),
            'synthesis': final_synthesis.get('confidence', 0),
            'overall': debate_result.get('confidence_scores', {}).get('overall', 0)
        }

        # Quality metrics
        metrics['answer_length'] = len(final_answer)
        metrics['coherence'] = self.text_coherence(final_answer)

        # Check for expected synthesis sections
        expected_sections = ['agreement', 'disagreement', 'conclusion', 'confidence']
        synthesis_output = final_synthesis.get('synthesis_output', '')
        metrics['synthesis_completeness'] = self.completeness_score(
            synthesis_output, expected_sections
        )['score']

        # Reference comparison if provided
        if reference_answer:
            metrics['reference_similarity'] = self.semantic_similarity(
                final_answer, reference_answer
            )

        # Agreement between agents
        if search_analysis.get('analysis') and reasoning_analysis.get('conclusion'):
            metrics['agent_agreement'] = self.semantic_similarity(
                search_analysis['analysis'][:1000],
                reasoning_analysis['conclusion'][:1000]
            )

        return metrics

    def compare_responses(
        self,
        response1: str,
        response2: str,
        query: str
    ) -> Dict[str, Any]:
        """Compare two responses to the same query"""
        return {
            "similarity": self.semantic_similarity(response1, response2),
            "length_ratio": len(response1) / len(response2) if response2 else 0,
            "query_relevance_1": self.semantic_similarity(query, response1),
            "query_relevance_2": self.semantic_similarity(query, response2),
            "coherence_1": self.text_coherence(response1),
            "coherence_2": self.text_coherence(response2)
        }

    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate metrics from multiple debates"""
        if not metrics_list:
            return {}

        aggregated = {
            "total_debates": len(metrics_list),
            "avg_time": np.mean([m.get('total_time', 0) for m in metrics_list]),
            "avg_coherence": np.mean([m.get('coherence', 0) for m in metrics_list]),
            "avg_confidence": np.mean([
                m.get('confidence', {}).get('overall', 0) for m in metrics_list
            ]),
            "avg_answer_length": np.mean([m.get('answer_length', 0) for m in metrics_list]),
            "min_time": min(m.get('total_time', 0) for m in metrics_list),
            "max_time": max(m.get('total_time', 0) for m in metrics_list)
        }

        return aggregated

    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable report"""
        lines = [
            "=" * 50,
            "DEBATE METRICS REPORT",
            "=" * 50,
            "",
            "TIME METRICS:",
            f"  Total Time: {metrics.get('total_time', 0):.2f}s",
            f"  Search Time: {metrics.get('search_time', 0):.2f}s",
            f"  Reasoning Time: {metrics.get('reasoning_time', 0):.2f}s",
            f"  Synthesis Time: {metrics.get('synthesis_time', 0):.2f}s",
            "",
            "CONFIDENCE METRICS:",
        ]

        conf = metrics.get('confidence', {})
        lines.extend([
            f"  Search Agent: {conf.get('search', 0):.1%}",
            f"  Reasoning Agent: {conf.get('reasoning', 0):.1%}",
            f"  Synthesis Agent: {conf.get('synthesis', 0):.1%}",
            f"  Overall: {conf.get('overall', 0):.1%}",
            "",
            "QUALITY METRICS:",
            f"  Answer Length: {metrics.get('answer_length', 0)} chars",
            f"  Coherence Score: {metrics.get('coherence', 0):.2f}",
            f"  Synthesis Completeness: {metrics.get('synthesis_completeness', 0):.1%}",
        ])

        if 'agent_agreement' in metrics:
            lines.append(f"  Agent Agreement: {metrics['agent_agreement']:.1%}")

        if 'reference_similarity' in metrics:
            lines.append(f"  Reference Similarity: {metrics['reference_similarity']:.1%}")

        lines.append("=" * 50)

        return "\n".join(lines)
