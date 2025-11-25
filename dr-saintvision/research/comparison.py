"""
Single Model vs Multi-Agent Comparison Experiments
Core experimental framework for validating multi-agent debate effectiveness

This module provides:
- Baseline single-model evaluation
- Multi-agent system evaluation
- Head-to-head comparison experiments
- Ablation studies for agent contributions
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

from .experiments import ExperimentConfig, ExperimentResult, ExperimentRunner
from .analysis import StatisticalAnalyzer, ResultVisualizer, ReportGenerator
from .benchmarks import BenchmarkDataset, QueryCategory

logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for comparison experiments"""
    experiment_name: str = "single_vs_multi_agent"

    # Models for single-agent baselines
    baseline_models: List[str] = field(default_factory=lambda: [
        "mistral:7b-instruct-v0.2-q4_0",
        "llama3.2:latest",
        "qwen2.5:7b-instruct-q4_0"
    ])

    # Multi-agent configurations
    multi_agent_strategies: List[str] = field(default_factory=lambda: [
        "parallel",
        "iterative",
        "adversarial",
        "consensus"
    ])

    # Experiment parameters
    num_trials_per_query: int = 3
    random_seed: int = 42
    timeout_per_query: float = 300.0

    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "coherence",
        "confidence",
        "response_time"
    ])

    # Output
    output_dir: str = "./comparison_results"
    generate_visualizations: bool = True
    generate_latex: bool = True


@dataclass
class SingleModelResult:
    """Result from single model evaluation"""
    model_name: str
    query: str
    response: str
    accuracy: float
    coherence: float
    confidence: float
    response_time: float
    error: Optional[str] = None


@dataclass
class MultiAgentResult:
    """Result from multi-agent evaluation"""
    strategy: str
    query: str
    final_response: str

    # Per-agent outputs
    search_response: str
    reasoning_response: str
    synthesis_response: str

    # Metrics
    accuracy: float
    coherence: float
    confidence: float
    response_time: float

    # Debate-specific
    num_rounds: int = 1
    convergence_achieved: bool = False
    agreement_score: float = 0.0

    error: Optional[str] = None


class SingleModelEvaluator:
    """
    Evaluates single model performance as baseline

    Tests each model independently without debate/collaboration
    """

    def __init__(
        self,
        model_loader: Callable,
        evaluator: Any = None,
        use_ollama: bool = True
    ):
        self.model_loader = model_loader
        self.evaluator = evaluator
        self.use_ollama = use_ollama
        self.models: Dict[str, Any] = {}

    async def evaluate_model(
        self,
        model_name: str,
        query: str,
        reference: Optional[str] = None
    ) -> SingleModelResult:
        """Evaluate a single model on one query"""
        start_time = datetime.now()

        try:
            # Load model if not cached
            if model_name not in self.models:
                self.models[model_name] = await self.model_loader(model_name)

            model = self.models[model_name]

            # Generate response
            if self.use_ollama:
                response = await self._generate_ollama(model_name, query)
            else:
                response = await self._generate_hf(model, query)

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            # Calculate metrics
            accuracy = 0.0
            coherence = 0.0
            confidence = 0.5  # Default confidence

            if self.evaluator:
                if reference:
                    accuracy_metrics = self.evaluator.evaluate_accuracy(
                        query, response, reference
                    )
                    accuracy = accuracy_metrics.get("overall_accuracy", 0)

                coherence = self.evaluator.text_coherence(response)

            return SingleModelResult(
                model_name=model_name,
                query=query,
                response=response,
                accuracy=accuracy,
                coherence=coherence,
                confidence=confidence,
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            end_time = datetime.now()

            return SingleModelResult(
                model_name=model_name,
                query=query,
                response="",
                accuracy=0.0,
                coherence=0.0,
                confidence=0.0,
                response_time=(end_time - start_time).total_seconds(),
                error=str(e)
            )

    async def _generate_ollama(self, model_name: str, query: str) -> str:
        """Generate response using Ollama"""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": query,
                        "stream": False
                    }
                )
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def _generate_hf(self, model: Any, query: str) -> str:
        """Generate response using HuggingFace model"""
        # Placeholder - implement based on actual model interface
        return ""

    async def run_baseline_evaluation(
        self,
        queries: List[Dict[str, Any]],
        model_names: List[str]
    ) -> Dict[str, List[SingleModelResult]]:
        """Run baseline evaluation across all models and queries"""
        results: Dict[str, List[SingleModelResult]] = {
            model: [] for model in model_names
        }

        total = len(queries) * len(model_names)
        completed = 0

        for model_name in model_names:
            logger.info(f"Evaluating baseline model: {model_name}")

            for q in queries:
                query = q.get("query", q) if isinstance(q, dict) else q
                reference = q.get("reference") if isinstance(q, dict) else None

                result = await self.evaluate_model(model_name, query, reference)
                results[model_name].append(result)

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        return results


class ComparisonExperiment:
    """
    Main comparison experiment between single models and multi-agent system

    Provides:
    - Head-to-head comparison
    - Statistical significance testing
    - Visualization generation
    - LaTeX table export
    """

    def __init__(
        self,
        config: ComparisonConfig,
        debate_manager_factory: Callable,
        single_model_loader: Callable,
        evaluator: Any = None
    ):
        self.config = config
        self.debate_manager_factory = debate_manager_factory
        self.single_model_loader = single_model_loader
        self.evaluator = evaluator

        # Initialize components
        self.single_evaluator = SingleModelEvaluator(
            single_model_loader,
            evaluator
        )
        self.analyzer = StatisticalAnalyzer()

        # Create output directory
        self.output_dir = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.visualizer = ResultVisualizer(str(self.output_dir / "figures"))

        # Results storage
        self.single_model_results: Dict[str, List[SingleModelResult]] = {}
        self.multi_agent_results: Dict[str, List[MultiAgentResult]] = {}

        logger.info(f"Comparison experiment initialized: {config.experiment_name}")

    async def run_full_comparison(
        self,
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run complete comparison experiment

        1. Evaluate single model baselines
        2. Evaluate multi-agent strategies
        3. Perform statistical comparison
        4. Generate visualizations and reports
        """
        logger.info("="*60)
        logger.info("Starting Full Comparison Experiment")
        logger.info(f"Queries: {len(queries)}")
        logger.info(f"Baseline models: {self.config.baseline_models}")
        logger.info(f"Multi-agent strategies: {self.config.multi_agent_strategies}")
        logger.info("="*60)

        # Phase 1: Single model baselines
        logger.info("\n[Phase 1] Evaluating Single Model Baselines")
        self.single_model_results = await self.single_evaluator.run_baseline_evaluation(
            queries,
            self.config.baseline_models
        )

        # Phase 2: Multi-agent strategies
        logger.info("\n[Phase 2] Evaluating Multi-Agent Strategies")
        for strategy in self.config.multi_agent_strategies:
            logger.info(f"Testing strategy: {strategy}")

            results = await self._run_multi_agent_strategy(queries, strategy)
            self.multi_agent_results[strategy] = results

        # Phase 3: Statistical analysis
        logger.info("\n[Phase 3] Performing Statistical Analysis")
        analysis = self._perform_analysis()

        # Phase 4: Generate outputs
        logger.info("\n[Phase 4] Generating Reports and Visualizations")

        if self.config.generate_visualizations:
            self._generate_visualizations(analysis)

        if self.config.generate_latex:
            self._generate_latex_tables(analysis)

        # Save complete results
        self._save_results(analysis)

        logger.info("\nComparison experiment complete!")
        logger.info(f"Results saved to: {self.output_dir}")

        return analysis

    async def _run_multi_agent_strategy(
        self,
        queries: List[Dict[str, Any]],
        strategy: str
    ) -> List[MultiAgentResult]:
        """Run multi-agent system with specific strategy"""
        results = []

        # Create debate manager with strategy
        config = ExperimentConfig(
            experiment_name=f"multi_agent_{strategy}",
            debate_strategy=strategy
        )
        debate_manager = self.debate_manager_factory(config)

        for q in queries:
            query = q.get("query", q) if isinstance(q, dict) else q
            reference = q.get("reference") if isinstance(q, dict) else None

            start_time = datetime.now()

            try:
                # Run debate
                debate_result = await debate_manager.conduct_debate(query)

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()

                # Calculate metrics
                accuracy = 0.0
                coherence = 0.0

                if self.evaluator and reference:
                    accuracy_metrics = self.evaluator.evaluate_accuracy(
                        query,
                        debate_result.final_synthesis.get("final_answer", ""),
                        reference
                    )
                    accuracy = accuracy_metrics.get("overall_accuracy", 0)
                    coherence = self.evaluator.text_coherence(
                        debate_result.final_synthesis.get("final_answer", "")
                    )

                result = MultiAgentResult(
                    strategy=strategy,
                    query=query,
                    final_response=debate_result.final_synthesis.get("final_answer", ""),
                    search_response=str(debate_result.search_analysis),
                    reasoning_response=str(debate_result.reasoning_analysis),
                    synthesis_response=str(debate_result.final_synthesis),
                    accuracy=accuracy,
                    coherence=coherence,
                    confidence=debate_result.confidence_scores.get("overall", 0.5),
                    response_time=response_time,
                    num_rounds=getattr(debate_result, 'num_rounds', 1),
                    convergence_achieved=getattr(debate_result, 'converged', False),
                    agreement_score=getattr(debate_result, 'agreement_score', 0)
                )

            except Exception as e:
                logger.error(f"Multi-agent evaluation failed: {e}")
                end_time = datetime.now()

                result = MultiAgentResult(
                    strategy=strategy,
                    query=query,
                    final_response="",
                    search_response="",
                    reasoning_response="",
                    synthesis_response="",
                    accuracy=0.0,
                    coherence=0.0,
                    confidence=0.0,
                    response_time=(end_time - start_time).total_seconds(),
                    error=str(e)
                )

            results.append(result)

        return results

    def _perform_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        analysis = {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "baseline_models": self.config.baseline_models,
                "multi_agent_strategies": self.config.multi_agent_strategies,
                "metrics": self.config.metrics
            },
            "single_model_stats": {},
            "multi_agent_stats": {},
            "comparisons": {},
            "rankings": {}
        }

        # Single model statistics
        for model, results in self.single_model_results.items():
            successful = [r for r in results if r.error is None]
            if successful:
                analysis["single_model_stats"][model] = {
                    "accuracy": {
                        "mean": float(np.mean([r.accuracy for r in successful])),
                        "std": float(np.std([r.accuracy for r in successful]))
                    },
                    "coherence": {
                        "mean": float(np.mean([r.coherence for r in successful])),
                        "std": float(np.std([r.coherence for r in successful]))
                    },
                    "response_time": {
                        "mean": float(np.mean([r.response_time for r in successful])),
                        "std": float(np.std([r.response_time for r in successful]))
                    },
                    "success_rate": len(successful) / len(results)
                }

        # Multi-agent statistics
        for strategy, results in self.multi_agent_results.items():
            successful = [r for r in results if r.error is None]
            if successful:
                analysis["multi_agent_stats"][strategy] = {
                    "accuracy": {
                        "mean": float(np.mean([r.accuracy for r in successful])),
                        "std": float(np.std([r.accuracy for r in successful]))
                    },
                    "coherence": {
                        "mean": float(np.mean([r.coherence for r in successful])),
                        "std": float(np.std([r.coherence for r in successful]))
                    },
                    "response_time": {
                        "mean": float(np.mean([r.response_time for r in successful])),
                        "std": float(np.std([r.response_time for r in successful]))
                    },
                    "convergence_rate": float(np.mean([
                        1 if r.convergence_achieved else 0 for r in successful
                    ])),
                    "mean_rounds": float(np.mean([r.num_rounds for r in successful])),
                    "success_rate": len(successful) / len(results)
                }

        # Statistical comparisons: Multi-agent vs each single model
        best_single_accuracy = []
        for model, results in self.single_model_results.items():
            successful = [r for r in results if r.error is None]
            if successful:
                best_single_accuracy.extend([r.accuracy for r in successful])

        for strategy, results in self.multi_agent_results.items():
            successful = [r for r in results if r.error is None]
            if successful:
                multi_accuracy = [r.accuracy for r in successful]

                # Compare against best single model baseline
                comparison = self.analyzer.compare_multi_agent_vs_single(
                    multi_accuracy,
                    best_single_accuracy
                )
                analysis["comparisons"][f"{strategy}_vs_single"] = comparison

        # Rankings
        all_configs = {}

        for model, stats in analysis["single_model_stats"].items():
            all_configs[f"single_{model}"] = stats["accuracy"]["mean"]

        for strategy, stats in analysis["multi_agent_stats"].items():
            all_configs[f"multi_{strategy}"] = stats["accuracy"]["mean"]

        # Sort by accuracy
        rankings = sorted(all_configs.items(), key=lambda x: x[1], reverse=True)
        analysis["rankings"]["by_accuracy"] = [
            {"config": name, "accuracy": acc} for name, acc in rankings
        ]

        return analysis

    def _generate_visualizations(self, analysis: Dict[str, Any]):
        """Generate all visualizations"""
        # 1. Accuracy comparison bar chart
        accuracy_data = {}
        accuracy_errors = {}

        for model, stats in analysis["single_model_stats"].items():
            accuracy_data[f"Single: {model.split(':')[0]}"] = stats["accuracy"]["mean"]
            accuracy_errors[f"Single: {model.split(':')[0]}"] = stats["accuracy"]["std"]

        for strategy, stats in analysis["multi_agent_stats"].items():
            accuracy_data[f"Multi: {strategy}"] = stats["accuracy"]["mean"]
            accuracy_errors[f"Multi: {strategy}"] = stats["accuracy"]["std"]

        self.visualizer.bar_chart_comparison(
            accuracy_data,
            accuracy_errors,
            title="Accuracy: Single Model vs Multi-Agent",
            ylabel="Accuracy Score",
            filename="accuracy_comparison.png"
        )

        # 2. Response time comparison
        time_data = {}

        for model, stats in analysis["single_model_stats"].items():
            time_data[f"Single: {model.split(':')[0]}"] = stats["response_time"]["mean"]

        for strategy, stats in analysis["multi_agent_stats"].items():
            time_data[f"Multi: {strategy}"] = stats["response_time"]["mean"]

        self.visualizer.bar_chart_comparison(
            time_data,
            title="Response Time Comparison",
            ylabel="Time (seconds)",
            filename="response_time_comparison.png"
        )

        # 3. Box plot for accuracy distributions
        if self.single_model_results and self.multi_agent_results:
            box_data = {}

            for model, results in self.single_model_results.items():
                successful = [r for r in results if r.error is None]
                if successful:
                    box_data[f"S:{model.split(':')[0]}"] = [r.accuracy for r in successful]

            for strategy, results in self.multi_agent_results.items():
                successful = [r for r in results if r.error is None]
                if successful:
                    box_data[f"M:{strategy}"] = [r.accuracy for r in successful]

            self.visualizer.box_plot_comparison(
                box_data,
                title="Accuracy Distribution",
                ylabel="Accuracy Score",
                filename="accuracy_boxplot.png"
            )

        # 4. Multi-metric radar chart
        radar_data = {}

        for strategy, stats in analysis["multi_agent_stats"].items():
            radar_data[strategy] = {
                "Accuracy": stats["accuracy"]["mean"],
                "Coherence": stats["coherence"]["mean"],
                "Convergence": stats.get("convergence_rate", 0),
                "Speed": 1 / (stats["response_time"]["mean"] / 60 + 0.01)  # Inverse of time
            }

        if radar_data:
            self.visualizer.multi_metric_radar(
                radar_data,
                title="Multi-Agent Strategy Comparison",
                filename="strategy_radar.png"
            )

        logger.info("Visualizations generated")

    def _generate_latex_tables(self, analysis: Dict[str, Any]):
        """Generate LaTeX tables for paper"""
        latex_dir = self.output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)

        # Main results table
        table_data = {}

        for model, stats in analysis["single_model_stats"].items():
            short_name = model.split(':')[0]
            table_data[f"Single ({short_name})"] = {
                "Accuracy": f"{stats['accuracy']['mean']:.3f}±{stats['accuracy']['std']:.3f}",
                "Coherence": f"{stats['coherence']['mean']:.3f}",
                "Time (s)": f"{stats['response_time']['mean']:.1f}"
            }

        for strategy, stats in analysis["multi_agent_stats"].items():
            table_data[f"Multi ({strategy})"] = {
                "Accuracy": f"{stats['accuracy']['mean']:.3f}±{stats['accuracy']['std']:.3f}",
                "Coherence": f"{stats['coherence']['mean']:.3f}",
                "Time (s)": f"{stats['response_time']['mean']:.1f}"
            }

        latex_table = self.visualizer.generate_latex_table(
            table_data,
            caption="Performance Comparison: Single Model vs Multi-Agent",
            label="tab:main_results"
        )

        with open(latex_dir / "main_results.tex", 'w', encoding='utf-8') as f:
            f.write(latex_table)

        # Statistical comparison table
        for name, comparison in analysis["comparisons"].items():
            latex_comp = self.visualizer.generate_latex_comparison_table(
                comparison,
                caption=f"Statistical Comparison: {name}",
                label=f"tab:{name}"
            )

            with open(latex_dir / f"{name}.tex", 'w', encoding='utf-8') as f:
                f.write(latex_comp)

        logger.info(f"LaTeX tables saved to {latex_dir}")

    def _save_results(self, analysis: Dict[str, Any]):
        """Save all results to files"""
        # Save analysis
        with open(self.output_dir / "analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

        # Save raw results
        raw_results = {
            "single_model": {},
            "multi_agent": {}
        }

        for model, results in self.single_model_results.items():
            raw_results["single_model"][model] = [
                {
                    "query": r.query,
                    "accuracy": r.accuracy,
                    "coherence": r.coherence,
                    "response_time": r.response_time,
                    "error": r.error
                }
                for r in results
            ]

        for strategy, results in self.multi_agent_results.items():
            raw_results["multi_agent"][strategy] = [
                {
                    "query": r.query,
                    "accuracy": r.accuracy,
                    "coherence": r.coherence,
                    "response_time": r.response_time,
                    "num_rounds": r.num_rounds,
                    "convergence": r.convergence_achieved,
                    "error": r.error
                }
                for r in results
            ]

        with open(self.output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)

        # Generate markdown report
        report_gen = ReportGenerator(
            self.analyzer,
            self.visualizer,
            str(self.output_dir)
        )

        report_gen.generate_experiment_report(
            self.config.experiment_name,
            {
                "total_queries": sum(len(r) for r in self.single_model_results.values()) // len(self.single_model_results) if self.single_model_results else 0,
                "baseline_models": len(self.config.baseline_models),
                "multi_agent_strategies": len(self.config.multi_agent_strategies),
                "best_config": analysis["rankings"]["by_accuracy"][0] if analysis["rankings"]["by_accuracy"] else None
            }
        )


class AblationStudy:
    """
    Ablation study to measure contribution of each agent

    Tests:
    - Full system (all 3 agents)
    - Without Search Agent
    - Without Reasoning Agent
    - Without Synthesis Agent (direct combination)
    """

    def __init__(
        self,
        debate_manager_factory: Callable,
        evaluator: Any = None
    ):
        self.debate_manager_factory = debate_manager_factory
        self.evaluator = evaluator
        self.results: Dict[str, List[Dict]] = {}

    async def run_ablation(
        self,
        queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run ablation study"""
        configurations = [
            ("full_system", {"use_search": True, "use_reasoning": True, "use_synthesis": True}),
            ("no_search", {"use_search": False, "use_reasoning": True, "use_synthesis": True}),
            ("no_reasoning", {"use_search": True, "use_reasoning": False, "use_synthesis": True}),
            ("no_synthesis", {"use_search": True, "use_reasoning": True, "use_synthesis": False}),
            ("search_only", {"use_search": True, "use_reasoning": False, "use_synthesis": False}),
            ("reasoning_only", {"use_search": False, "use_reasoning": True, "use_synthesis": False}),
        ]

        for config_name, config_params in configurations:
            logger.info(f"Running ablation: {config_name}")

            results = await self._run_configuration(queries, config_params)
            self.results[config_name] = results

        return self._analyze_ablation()

    async def _run_configuration(
        self,
        queries: List[Dict[str, Any]],
        config_params: Dict[str, bool]
    ) -> List[Dict]:
        """Run experiment with specific agent configuration"""
        results = []

        exp_config = ExperimentConfig(
            experiment_name="ablation",
            **config_params
        )
        debate_manager = self.debate_manager_factory(exp_config)

        for q in queries:
            query = q.get("query", q) if isinstance(q, dict) else q
            reference = q.get("reference") if isinstance(q, dict) else None

            start_time = datetime.now()

            try:
                debate_result = await debate_manager.conduct_debate(query)
                end_time = datetime.now()

                accuracy = 0.0
                if self.evaluator and reference:
                    accuracy_metrics = self.evaluator.evaluate_accuracy(
                        query,
                        debate_result.final_synthesis.get("final_answer", ""),
                        reference
                    )
                    accuracy = accuracy_metrics.get("overall_accuracy", 0)

                results.append({
                    "query": query,
                    "accuracy": accuracy,
                    "response_time": (end_time - start_time).total_seconds(),
                    "error": None
                })

            except Exception as e:
                end_time = datetime.now()
                results.append({
                    "query": query,
                    "accuracy": 0.0,
                    "response_time": (end_time - start_time).total_seconds(),
                    "error": str(e)
                })

        return results

    def _analyze_ablation(self) -> Dict[str, Any]:
        """Analyze ablation study results"""
        analysis = {
            "configurations": {},
            "agent_contributions": {}
        }

        for config, results in self.results.items():
            successful = [r for r in results if r["error"] is None]
            if successful:
                analysis["configurations"][config] = {
                    "mean_accuracy": float(np.mean([r["accuracy"] for r in successful])),
                    "std_accuracy": float(np.std([r["accuracy"] for r in successful])),
                    "mean_time": float(np.mean([r["response_time"] for r in successful])),
                    "success_rate": len(successful) / len(results)
                }

        # Calculate agent contributions
        full = analysis["configurations"].get("full_system", {}).get("mean_accuracy", 0)

        if "no_search" in analysis["configurations"]:
            no_search = analysis["configurations"]["no_search"]["mean_accuracy"]
            analysis["agent_contributions"]["search_agent"] = full - no_search

        if "no_reasoning" in analysis["configurations"]:
            no_reasoning = analysis["configurations"]["no_reasoning"]["mean_accuracy"]
            analysis["agent_contributions"]["reasoning_agent"] = full - no_reasoning

        if "no_synthesis" in analysis["configurations"]:
            no_synthesis = analysis["configurations"]["no_synthesis"]["mean_accuracy"]
            analysis["agent_contributions"]["synthesis_agent"] = full - no_synthesis

        return analysis


# Example usage and main runner
async def run_comparison_experiment():
    """Example function to run comparison experiment"""
    from .benchmarks import BenchmarkDataset

    # Load benchmark queries
    dataset = BenchmarkDataset()
    queries = dataset.get_queries_by_category(QueryCategory.FACTUAL, limit=20)
    queries.extend(dataset.get_queries_by_category(QueryCategory.REASONING, limit=20))

    # Create config
    config = ComparisonConfig(
        experiment_name="main_comparison",
        num_trials_per_query=1
    )

    # Note: These factories need to be implemented based on your actual model setup
    def debate_manager_factory(exp_config):
        # Return configured debate manager
        pass

    def single_model_loader(model_name):
        # Return loaded model
        pass

    # Create experiment
    experiment = ComparisonExperiment(
        config=config,
        debate_manager_factory=debate_manager_factory,
        single_model_loader=single_model_loader
    )

    # Run experiment
    results = await experiment.run_full_comparison(queries)

    return results


if __name__ == "__main__":
    asyncio.run(run_comparison_experiment())
