"""
DR-Saintvision Research Module
Multi-Agent Debate System for Academic Research

This module contains:
- Novel debate algorithms (Confidence-Weighted Synthesis, Iterative Refinement)
- Benchmark datasets and evaluation frameworks
- Experimental comparison tools
- Statistical analysis utilities
- Single model vs Multi-agent comparison experiments
"""

from .algorithms import (
    ConfidenceWeightedSynthesis,
    IterativeDebate,
    AdversarialDebate,
    ConsensusBuilder
)
from .experiments import ExperimentRunner, ExperimentConfig
from .benchmarks import BenchmarkDataset, BenchmarkRunner
from .analysis import StatisticalAnalyzer, ResultVisualizer, ReportGenerator
from .comparison import (
    ComparisonConfig,
    ComparisonExperiment,
    SingleModelEvaluator,
    AblationStudy
)

__all__ = [
    # Algorithms
    'ConfidenceWeightedSynthesis',
    'IterativeDebate',
    'AdversarialDebate',
    'ConsensusBuilder',
    # Experiments
    'ExperimentRunner',
    'ExperimentConfig',
    # Benchmarks
    'BenchmarkDataset',
    'BenchmarkRunner',
    # Analysis
    'StatisticalAnalyzer',
    'ResultVisualizer',
    'ReportGenerator',
    # Comparison
    'ComparisonConfig',
    'ComparisonExperiment',
    'SingleModelEvaluator',
    'AblationStudy'
]
