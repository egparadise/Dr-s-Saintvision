"""
Evaluation System for DR-Saintvision
Comprehensive evaluation of debate quality and accuracy
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class EvaluationSystem:
    """
    Comprehensive evaluation system for multi-agent debates
    Evaluates accuracy, quality, and compares with benchmarks
    """

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._encoder = None
        self._benchmark_data = []

        if use_embeddings:
            self._init_encoder()

        self._load_benchmarks()

    def _init_encoder(self):
        """Initialize sentence transformer for semantic similarity"""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence encoder initialized for evaluation")
        except ImportError:
            logger.warning("sentence-transformers not available")
            self.use_embeddings = False

    def _load_benchmarks(self):
        """Load benchmark data for evaluation"""
        # Default benchmark questions with expected key points
        self._benchmark_data = [
            {
                "query": "What causes climate change?",
                "expected_answer": "Climate change is primarily caused by human activities that release greenhouse gases",
                "key_points": [
                    "greenhouse gases",
                    "human activities",
                    "CO2 emissions",
                    "fossil fuels",
                    "deforestation"
                ],
                "category": "science"
            },
            {
                "query": "How does machine learning work?",
                "expected_answer": "Machine learning algorithms learn patterns from data to make predictions",
                "key_points": [
                    "data",
                    "patterns",
                    "training",
                    "model",
                    "predictions"
                ],
                "category": "technology"
            },
            {
                "query": "What is the theory of relativity?",
                "expected_answer": "Einstein's theory describes how space and time are interconnected",
                "key_points": [
                    "Einstein",
                    "space",
                    "time",
                    "gravity",
                    "speed of light"
                ],
                "category": "physics"
            }
        ]

    def evaluate_accuracy(
        self,
        query: str,
        answer: str,
        reference: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the accuracy of an answer

        Args:
            query: The original question
            answer: The generated answer
            reference: Optional reference answer

        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {}

        # Find matching benchmark if no reference provided
        benchmark = None
        if not reference:
            for b in self._benchmark_data:
                if self._is_similar_query(query, b['query']):
                    benchmark = b
                    reference = b['expected_answer']
                    break

        # Semantic similarity with query (relevance)
        if self._encoder and answer:
            from sklearn.metrics.pairwise import cosine_similarity

            query_emb = self._encoder.encode([query])
            answer_emb = self._encoder.encode([answer])
            metrics['query_relevance'] = float(
                cosine_similarity(query_emb, answer_emb)[0][0]
            )

            # Semantic similarity with reference if available
            if reference:
                ref_emb = self._encoder.encode([reference])
                metrics['reference_similarity'] = float(
                    cosine_similarity(answer_emb, ref_emb)[0][0]
                )

        # Keyword coverage if benchmark available
        if benchmark:
            coverage = self._evaluate_keyword_coverage(
                answer,
                benchmark['key_points']
            )
            metrics['keyword_coverage'] = coverage['score']
            metrics['keywords_found'] = coverage['found']
            metrics['keywords_missing'] = coverage['missing']

        # Coherence score
        metrics['coherence'] = self._evaluate_coherence(answer)

        # Completeness score
        metrics['completeness'] = self._evaluate_completeness(answer)

        # Overall accuracy (weighted average)
        weights = {
            'query_relevance': 0.25,
            'reference_similarity': 0.25,
            'keyword_coverage': 0.20,
            'coherence': 0.15,
            'completeness': 0.15
        }

        total_weight = 0
        weighted_sum = 0
        for metric, weight in weights.items():
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                weighted_sum += metrics[metric] * weight
                total_weight += weight

        metrics['overall_accuracy'] = weighted_sum / total_weight if total_weight > 0 else 0

        return metrics

    def _is_similar_query(self, query1: str, query2: str) -> bool:
        """Check if two queries are similar"""
        if not self._encoder:
            return query1.lower() == query2.lower()

        from sklearn.metrics.pairwise import cosine_similarity
        emb1 = self._encoder.encode([query1])
        emb2 = self._encoder.encode([query2])
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity > 0.8

    def _evaluate_keyword_coverage(
        self,
        text: str,
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Evaluate keyword coverage in text"""
        text_lower = text.lower()
        found = []
        missing = []

        for keyword in keywords:
            if keyword.lower() in text_lower:
                found.append(keyword)
            else:
                missing.append(keyword)

        score = len(found) / len(keywords) if keywords else 0

        return {
            'score': score,
            'found': found,
            'missing': missing,
            'total': len(keywords)
        }

    def _evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence"""
        if not text:
            return 0.0

        score = 0.5  # Base score

        # Check for sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) >= 3:
            score += 0.1

        # Check for transition words
        transitions = [
            'however', 'therefore', 'furthermore', 'moreover',
            'additionally', 'consequently', 'thus', 'hence'
        ]
        text_lower = text.lower()
        transition_count = sum(1 for t in transitions if t in text_lower)
        score += min(transition_count * 0.05, 0.2)

        # Check for paragraph structure
        if '\n\n' in text:
            score += 0.1

        # Check for logical markers
        logical_markers = ['because', 'since', 'as a result', 'due to']
        if any(m in text_lower for m in logical_markers):
            score += 0.1

        return min(score, 1.0)

    def _evaluate_completeness(self, text: str) -> float:
        """Evaluate answer completeness"""
        if not text:
            return 0.0

        score = 0.0

        # Length-based scoring
        word_count = len(text.split())
        if word_count >= 50:
            score += 0.2
        if word_count >= 100:
            score += 0.2
        if word_count >= 200:
            score += 0.1

        # Structure-based scoring
        if '##' in text or '**' in text:  # Markdown headers/bold
            score += 0.1

        # Check for multiple aspects
        aspect_indicators = [
            'first', 'second', 'third',
            'on one hand', 'on the other hand',
            'additionally', 'furthermore',
            'in conclusion', 'to summarize'
        ]
        text_lower = text.lower()
        aspect_count = sum(1 for a in aspect_indicators if a in text_lower)
        score += min(aspect_count * 0.05, 0.2)

        # Check for examples
        if 'for example' in text_lower or 'such as' in text_lower:
            score += 0.1

        return min(score, 1.0)

    def compare_with_single_model(
        self,
        debate_result: Dict[str, Any],
        single_model_result: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Compare multi-agent debate result with single model result

        Args:
            debate_result: Full debate result dictionary
            single_model_result: Response from single model
            query: Original query

        Returns:
            Comparison analysis
        """
        debate_answer = debate_result.get('final_synthesis', {}).get('final_answer', '')

        # Evaluate both
        debate_metrics = self.evaluate_accuracy(query, debate_answer)
        single_metrics = self.evaluate_accuracy(query, single_model_result)

        # Length comparison
        debate_length = len(debate_answer)
        single_length = len(single_model_result)

        # Vocabulary diversity
        debate_vocab = set(debate_answer.lower().split())
        single_vocab = set(single_model_result.lower().split())

        comparison = {
            'query': query,
            'debate_metrics': debate_metrics,
            'single_metrics': single_metrics,
            'length_comparison': {
                'debate': debate_length,
                'single': single_length,
                'ratio': debate_length / single_length if single_length > 0 else 0
            },
            'vocabulary_comparison': {
                'debate_unique_words': len(debate_vocab),
                'single_unique_words': len(single_vocab),
                'diversity_ratio': len(debate_vocab) / len(single_vocab) if single_vocab else 0
            },
            'accuracy_improvement': debate_metrics.get('overall_accuracy', 0) - single_metrics.get('overall_accuracy', 0),
            'winner': 'debate' if debate_metrics.get('overall_accuracy', 0) > single_metrics.get('overall_accuracy', 0) else 'single'
        }

        # Qualitative advantages
        comparison['debate_advantages'] = []
        comparison['debate_disadvantages'] = []

        if debate_metrics.get('coherence', 0) > single_metrics.get('coherence', 0):
            comparison['debate_advantages'].append("Better coherence")
        else:
            comparison['debate_disadvantages'].append("Lower coherence")

        if debate_metrics.get('completeness', 0) > single_metrics.get('completeness', 0):
            comparison['debate_advantages'].append("More complete answer")
        else:
            comparison['debate_disadvantages'].append("Less complete")

        if debate_length > single_length * 1.5:
            comparison['debate_advantages'].append("More detailed response")

        # Always note the time tradeoff
        comparison['debate_disadvantages'].append("Longer processing time")
        comparison['debate_disadvantages'].append("Higher resource usage")

        return comparison

    def evaluate_debate_quality(
        self,
        debate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive quality evaluation of a debate

        Args:
            debate_result: Full debate result

        Returns:
            Quality metrics and analysis
        """
        search = debate_result.get('search_analysis', {})
        reasoning = debate_result.get('reasoning_analysis', {})
        synthesis = debate_result.get('final_synthesis', {})

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': debate_result.get('query', ''),
            'debate_time': debate_result.get('debate_time', 0)
        }

        # Agent-specific metrics
        metrics['agent_metrics'] = {
            'search': {
                'confidence': search.get('confidence', 0),
                'processing_time': search.get('processing_time', 0),
                'search_results_count': len(search.get('search_results', [])),
                'analysis_length': len(search.get('analysis', ''))
            },
            'reasoning': {
                'confidence': reasoning.get('confidence', 0),
                'processing_time': reasoning.get('processing_time', 0),
                'reasoning_steps': len(reasoning.get('reasoning_steps', [])),
                'conclusion_length': len(reasoning.get('conclusion', ''))
            },
            'synthesis': {
                'confidence': synthesis.get('confidence', 0),
                'processing_time': synthesis.get('processing_time', 0),
                'final_answer_length': len(synthesis.get('final_answer', '')),
                'has_agreements': bool(synthesis.get('agreements')),
                'has_disagreements': bool(synthesis.get('disagreements'))
            }
        }

        # Agreement analysis
        search_text = search.get('analysis', '')[:500]
        reasoning_text = reasoning.get('conclusion', '')[:500]

        if self._encoder and search_text and reasoning_text:
            from sklearn.metrics.pairwise import cosine_similarity
            emb1 = self._encoder.encode([search_text])
            emb2 = self._encoder.encode([reasoning_text])
            metrics['agent_agreement'] = float(cosine_similarity(emb1, emb2)[0][0])
        else:
            metrics['agent_agreement'] = 0

        # Overall quality scores
        final_answer = synthesis.get('final_answer', '')
        metrics['quality_scores'] = {
            'coherence': self._evaluate_coherence(final_answer),
            'completeness': self._evaluate_completeness(final_answer),
            'confidence_consistency': self._evaluate_confidence_consistency(debate_result)
        }

        # Calculate overall quality
        quality_values = list(metrics['quality_scores'].values())
        metrics['overall_quality'] = np.mean(quality_values) if quality_values else 0

        # Grade
        metrics['grade'] = self._calculate_grade(metrics['overall_quality'])

        return metrics

    def _evaluate_confidence_consistency(self, result: Dict) -> float:
        """Evaluate how consistent confidence scores are across agents"""
        confidences = result.get('confidence_scores', {})

        if len(confidences) < 2:
            return 0.5

        values = [v for v in confidences.values() if isinstance(v, (int, float))]

        if not values:
            return 0.5

        # Lower standard deviation = higher consistency
        std = np.std(values)

        # Convert to 0-1 score (lower std = higher score)
        consistency = 1 - min(std * 2, 1)

        return consistency

    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        elif score >= 0.4:
            return "D"
        else:
            return "F"

    def run_benchmark(
        self,
        debate_manager,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark evaluation on predefined questions

        Args:
            debate_manager: DebateManager instance
            categories: Optional list of categories to test

        Returns:
            Benchmark results
        """
        import asyncio

        results = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': 0,
            'results_by_category': {},
            'overall_metrics': {}
        }

        questions = self._benchmark_data
        if categories:
            questions = [q for q in questions if q.get('category') in categories]

        results['total_questions'] = len(questions)

        all_accuracies = []
        all_times = []

        for question in questions:
            category = question.get('category', 'general')

            if category not in results['results_by_category']:
                results['results_by_category'][category] = []

            try:
                # Run debate
                debate_result = asyncio.run(
                    debate_manager.conduct_debate(question['query'])
                )

                # Evaluate
                final_answer = debate_result.final_synthesis.get('final_answer', '')
                accuracy = self.evaluate_accuracy(
                    question['query'],
                    final_answer,
                    question['expected_answer']
                )

                result_entry = {
                    'query': question['query'],
                    'accuracy': accuracy,
                    'debate_time': debate_result.debate_time,
                    'status': debate_result.status.value
                }

                results['results_by_category'][category].append(result_entry)

                all_accuracies.append(accuracy.get('overall_accuracy', 0))
                all_times.append(debate_result.debate_time)

            except Exception as e:
                logger.error(f"Benchmark error for '{question['query']}': {e}")
                results['results_by_category'][category].append({
                    'query': question['query'],
                    'error': str(e)
                })

        # Calculate overall metrics
        if all_accuracies:
            results['overall_metrics'] = {
                'average_accuracy': np.mean(all_accuracies),
                'min_accuracy': min(all_accuracies),
                'max_accuracy': max(all_accuracies),
                'average_time': np.mean(all_times),
                'total_time': sum(all_times)
            }

        return results

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        lines = [
            "=" * 60,
            "DR-SAINTVISION EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        if 'query' in evaluation_results:
            lines.append(f"Query: {evaluation_results['query']}")
            lines.append("")

        # Quality scores
        if 'quality_scores' in evaluation_results:
            lines.append("QUALITY SCORES:")
            for metric, value in evaluation_results['quality_scores'].items():
                lines.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
            lines.append("")

        # Agent metrics
        if 'agent_metrics' in evaluation_results:
            lines.append("AGENT PERFORMANCE:")
            for agent, metrics in evaluation_results['agent_metrics'].items():
                lines.append(f"\n  {agent.upper()}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"    {metric}: {value:.2f}")
                    else:
                        lines.append(f"    {metric}: {value}")
            lines.append("")

        # Overall
        if 'overall_quality' in evaluation_results:
            lines.append(f"OVERALL QUALITY: {evaluation_results['overall_quality']:.2f}")
            lines.append(f"GRADE: {evaluation_results.get('grade', 'N/A')}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
