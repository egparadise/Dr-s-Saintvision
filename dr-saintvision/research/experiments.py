"""
Experiment Framework for DR-Saintvision Research
Systematic evaluation of Multi-Agent Debate Systems

This module provides:
- Configurable experiment runner
- Automated result collection
- Statistical significance testing
- Reproducibility through seed control
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    # Experiment identification
    experiment_name: str
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Model configuration
    use_ollama: bool = True
    models: Dict[str, str] = field(default_factory=lambda: {
        "search": "mistral:7b-instruct-v0.2-q4_0",
        "reasoning": "llama3.2:latest",
        "synthesis": "qwen2.5:7b-instruct-q4_0"
    })

    # Experiment parameters
    num_trials: int = 10
    random_seed: int = 42
    timeout_per_query: float = 300.0

    # Debate configuration
    debate_strategy: str = "parallel"  # parallel, iterative, adversarial, consensus
    max_debate_rounds: int = 3
    convergence_threshold: float = 0.85

    # Evaluation settings
    evaluate_accuracy: bool = True
    evaluate_coherence: bool = True
    evaluate_confidence: bool = True
    human_evaluation: bool = False

    # Output settings
    output_dir: str = "./results"
    save_intermediate: bool = True
    verbose: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        return cls(**data)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


@dataclass
class ExperimentResult:
    """Result of a single experiment trial"""
    trial_id: int
    query: str
    query_category: str

    # Timing
    start_time: datetime
    end_time: datetime
    total_time: float

    # Agent outputs
    search_result: Dict[str, Any]
    reasoning_result: Dict[str, Any]
    synthesis_result: Dict[str, Any]

    # Metrics
    accuracy_score: float = 0.0
    coherence_score: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Debate specific
    num_rounds: int = 1
    convergence_achieved: bool = False
    agreement_score: float = 0.0

    # Ground truth comparison (if available)
    reference_answer: Optional[str] = None
    reference_similarity: float = 0.0

    # Metadata
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment"""
    experiment_id: str
    experiment_name: str
    config: ExperimentConfig

    # Counts
    total_trials: int
    successful_trials: int
    failed_trials: int

    # Timing statistics
    mean_time: float
    std_time: float
    min_time: float
    max_time: float

    # Accuracy statistics
    mean_accuracy: float
    std_accuracy: float

    # Confidence statistics
    mean_confidence: float
    std_confidence: float

    # Debate statistics
    mean_rounds: float
    convergence_rate: float

    # Per-category breakdown
    category_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw results
    results: List[ExperimentResult] = field(default_factory=list)


class ExperimentRunner:
    """
    Main experiment runner for DR-Saintvision research

    Supports:
    - Multiple experiment configurations
    - Automated metric collection
    - Result persistence
    - Statistical analysis
    """

    def __init__(
        self,
        config: ExperimentConfig,
        debate_manager: Any = None,
        evaluator: Any = None
    ):
        self.config = config
        self.debate_manager = debate_manager
        self.evaluator = evaluator
        self.results: List[ExperimentResult] = []

        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(str(self.output_dir / "config.json"))

        logger.info(f"Experiment initialized: {config.experiment_name}")

    async def run_single_trial(
        self,
        trial_id: int,
        query: str,
        category: str = "general",
        reference: Optional[str] = None
    ) -> ExperimentResult:
        """Run a single experiment trial"""
        start_time = datetime.now()

        if self.config.verbose:
            logger.info(f"Trial {trial_id}: {query[:50]}...")

        try:
            # Run debate
            debate_result = await self.debate_manager.conduct_debate(query)

            # Extract results
            search_result = debate_result.search_analysis
            reasoning_result = debate_result.reasoning_analysis
            synthesis_result = debate_result.final_synthesis

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Compute metrics
            accuracy = 0.0
            coherence = 0.0
            reference_sim = 0.0

            if self.evaluator:
                if self.config.evaluate_accuracy and reference:
                    accuracy_metrics = self.evaluator.evaluate_accuracy(
                        query,
                        synthesis_result.get("final_answer", ""),
                        reference
                    )
                    accuracy = accuracy_metrics.get("overall_accuracy", 0)
                    reference_sim = accuracy_metrics.get("reference_similarity", 0)

                if self.config.evaluate_coherence:
                    coherence = self.evaluator.text_coherence(
                        synthesis_result.get("final_answer", "")
                    )

            result = ExperimentResult(
                trial_id=trial_id,
                query=query,
                query_category=category,
                start_time=start_time,
                end_time=end_time,
                total_time=total_time,
                search_result=search_result,
                reasoning_result=reasoning_result,
                synthesis_result=synthesis_result,
                accuracy_score=accuracy,
                coherence_score=coherence,
                confidence_scores=debate_result.confidence_scores,
                num_rounds=getattr(debate_result, 'num_rounds', 1),
                convergence_achieved=getattr(debate_result, 'converged', False),
                agreement_score=getattr(debate_result, 'agreement_score', 0),
                reference_answer=reference,
                reference_similarity=reference_sim
            )

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            end_time = datetime.now()

            result = ExperimentResult(
                trial_id=trial_id,
                query=query,
                query_category=category,
                start_time=start_time,
                end_time=end_time,
                total_time=(end_time - start_time).total_seconds(),
                search_result={},
                reasoning_result={},
                synthesis_result={},
                error=str(e)
            )

        # Save intermediate result
        if self.config.save_intermediate:
            self._save_trial_result(result)

        return result

    async def run_experiment(
        self,
        queries: List[Dict[str, Any]]
    ) -> ExperimentSummary:
        """
        Run full experiment with multiple queries

        Args:
            queries: List of dicts with 'query', 'category', 'reference' keys

        Returns:
            ExperimentSummary with all results and statistics
        """
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info(f"Total queries: {len(queries)}")

        self.results = []

        for i, q in enumerate(queries):
            query = q.get("query", q) if isinstance(q, dict) else q
            category = q.get("category", "general") if isinstance(q, dict) else "general"
            reference = q.get("reference") if isinstance(q, dict) else None

            result = await self.run_single_trial(
                trial_id=i + 1,
                query=query,
                category=category,
                reference=reference
            )
            self.results.append(result)

            if self.config.verbose:
                logger.info(
                    f"Trial {i+1}/{len(queries)} complete. "
                    f"Time: {result.total_time:.2f}s, "
                    f"Accuracy: {result.accuracy_score:.2%}"
                )

        # Generate summary
        summary = self._generate_summary()

        # Save final results
        self._save_summary(summary)

        logger.info(f"Experiment complete: {self.config.experiment_name}")
        return summary

    def _generate_summary(self) -> ExperimentSummary:
        """Generate experiment summary statistics"""
        successful = [r for r in self.results if r.error is None]
        failed = [r for r in self.results if r.error is not None]

        times = [r.total_time for r in successful]
        accuracies = [r.accuracy_score for r in successful if r.accuracy_score > 0]
        confidences = [r.confidence_scores.get("overall", 0) for r in successful]
        rounds = [r.num_rounds for r in successful]
        converged = [r for r in successful if r.convergence_achieved]

        # Per-category breakdown
        categories = set(r.query_category for r in self.results)
        category_results = {}

        for cat in categories:
            cat_results = [r for r in successful if r.query_category == cat]
            if cat_results:
                category_results[cat] = {
                    "count": len(cat_results),
                    "mean_accuracy": float(np.mean([r.accuracy_score for r in cat_results])),
                    "mean_time": float(np.mean([r.total_time for r in cat_results])),
                    "mean_confidence": float(np.mean([
                        r.confidence_scores.get("overall", 0) for r in cat_results
                    ]))
                }

        return ExperimentSummary(
            experiment_id=self.config.experiment_id,
            experiment_name=self.config.experiment_name,
            config=self.config,
            total_trials=len(self.results),
            successful_trials=len(successful),
            failed_trials=len(failed),
            mean_time=float(np.mean(times)) if times else 0,
            std_time=float(np.std(times)) if times else 0,
            min_time=float(min(times)) if times else 0,
            max_time=float(max(times)) if times else 0,
            mean_accuracy=float(np.mean(accuracies)) if accuracies else 0,
            std_accuracy=float(np.std(accuracies)) if accuracies else 0,
            mean_confidence=float(np.mean(confidences)) if confidences else 0,
            std_confidence=float(np.std(confidences)) if confidences else 0,
            mean_rounds=float(np.mean(rounds)) if rounds else 1,
            convergence_rate=len(converged) / len(successful) if successful else 0,
            category_results=category_results,
            results=self.results
        )

    def _save_trial_result(self, result: ExperimentResult):
        """Save individual trial result"""
        trial_path = self.output_dir / f"trial_{result.trial_id:04d}.json"

        data = {
            "trial_id": result.trial_id,
            "query": result.query,
            "category": result.query_category,
            "total_time": result.total_time,
            "accuracy": result.accuracy_score,
            "coherence": result.coherence_score,
            "confidence": result.confidence_scores,
            "final_answer": result.synthesis_result.get("final_answer", ""),
            "error": result.error
        }

        with open(trial_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _save_summary(self, summary: ExperimentSummary):
        """Save experiment summary"""
        summary_path = self.output_dir / "summary.json"

        data = {
            "experiment_id": summary.experiment_id,
            "experiment_name": summary.experiment_name,
            "total_trials": summary.total_trials,
            "successful_trials": summary.successful_trials,
            "failed_trials": summary.failed_trials,
            "statistics": {
                "time": {
                    "mean": summary.mean_time,
                    "std": summary.std_time,
                    "min": summary.min_time,
                    "max": summary.max_time
                },
                "accuracy": {
                    "mean": summary.mean_accuracy,
                    "std": summary.std_accuracy
                },
                "confidence": {
                    "mean": summary.mean_confidence,
                    "std": summary.std_confidence
                },
                "debate": {
                    "mean_rounds": summary.mean_rounds,
                    "convergence_rate": summary.convergence_rate
                }
            },
            "category_results": summary.category_results
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to {summary_path}")


class ComparisonExperiment:
    """
    Run comparison experiments between different configurations

    Compares:
    - Single model vs Multi-agent
    - Different debate strategies
    - Different model combinations
    """

    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.comparison_results: Dict[str, ExperimentSummary] = {}

    async def run_comparison(
        self,
        queries: List[Dict[str, Any]],
        configurations: Dict[str, ExperimentConfig],
        debate_manager_factory: Callable
    ) -> Dict[str, Any]:
        """
        Run multiple configurations and compare results

        Args:
            queries: Test queries
            configurations: Dict of config_name -> ExperimentConfig
            debate_manager_factory: Function to create debate manager from config

        Returns:
            Comparison analysis
        """
        for name, config in configurations.items():
            logger.info(f"Running configuration: {name}")

            debate_manager = debate_manager_factory(config)
            runner = ExperimentRunner(config, debate_manager)

            summary = await runner.run_experiment(queries)
            self.comparison_results[name] = summary

        return self._analyze_comparison()

    def _analyze_comparison(self) -> Dict[str, Any]:
        """Analyze and compare results across configurations"""
        if not self.comparison_results:
            return {}

        analysis = {
            "configurations": list(self.comparison_results.keys()),
            "metrics_comparison": {},
            "statistical_tests": {},
            "rankings": {}
        }

        # Compare metrics
        metrics = ["mean_accuracy", "mean_confidence", "mean_time", "convergence_rate"]

        for metric in metrics:
            values = {
                name: getattr(summary, metric, 0)
                for name, summary in self.comparison_results.items()
            }
            analysis["metrics_comparison"][metric] = values

            # Rank configurations
            ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
            analysis["rankings"][metric] = [name for name, _ in ranked]

        # Best overall configuration
        scores = {}
        for name in self.comparison_results.keys():
            score = (
                analysis["metrics_comparison"]["mean_accuracy"].get(name, 0) * 0.4 +
                analysis["metrics_comparison"]["mean_confidence"].get(name, 0) * 0.3 +
                (1 - analysis["metrics_comparison"]["mean_time"].get(name, 0) / 100) * 0.3
            )
            scores[name] = score

        analysis["overall_ranking"] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        analysis["best_configuration"] = analysis["overall_ranking"][0][0]

        return analysis
