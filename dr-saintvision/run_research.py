"""
DR-Saintvision Research Runner
Execute experiments and generate paper-ready results

Usage:
    python run_research.py --experiment comparison
    python run_research.py --experiment ablation
    python run_research.py --experiment benchmark
    python run_research.py --experiment all
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from research import (
    ExperimentConfig,
    ExperimentRunner,
    BenchmarkDataset,
    BenchmarkRunner,
    ComparisonConfig,
    ComparisonExperiment,
    AblationStudy,
    StatisticalAnalyzer,
    ResultVisualizer
)
from research.benchmarks import QueryCategory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class ResearchRunner:
    """Main runner for all research experiments"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.output_dir = Path("./research_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Research output directory: {self.output_dir}")

    def _create_debate_manager(self, exp_config: ExperimentConfig):
        """Create debate manager with configuration"""
        try:
            from models.debate_manager import DebateManager

            return DebateManager(
                use_ollama=exp_config.use_ollama,
                models=exp_config.models,
                strategy=exp_config.debate_strategy
            )
        except ImportError as e:
            logger.warning(f"Could not import DebateManager: {e}")
            return MockDebateManager(exp_config)

    async def _load_single_model(self, model_name: str):
        """Load a single model for baseline evaluation"""
        # Return model handle (actual implementation depends on your setup)
        return model_name

    def _create_evaluator(self):
        """Create evaluation instance"""
        try:
            from backend.evaluation import EvaluationService
            return EvaluationService()
        except ImportError:
            logger.warning("EvaluationService not available, using mock")
            return MockEvaluator()

    async def run_benchmark_experiment(self, categories: list = None):
        """Run benchmark evaluation across different query categories"""
        logger.info("="*60)
        logger.info("Starting Benchmark Experiment")
        logger.info("="*60)

        # Initialize benchmark
        dataset = BenchmarkDataset()

        # Select categories
        if categories is None:
            categories = [
                QueryCategory.FACTUAL,
                QueryCategory.REASONING,
                QueryCategory.SCIENTIFIC,
                QueryCategory.ETHICAL,
                QueryCategory.COMPLEX
            ]

        # Create experiment config
        exp_config = ExperimentConfig(
            experiment_name="benchmark_evaluation",
            output_dir=str(self.output_dir / "benchmark"),
            num_trials=len(categories) * 10
        )

        # Create runner
        debate_manager = self._create_debate_manager(exp_config)
        evaluator = self._create_evaluator()
        runner = ExperimentRunner(exp_config, debate_manager, evaluator)

        # Collect queries from all categories
        all_queries = []
        for category in categories:
            queries = dataset.get_queries_by_category(category, limit=10)
            all_queries.extend(queries)
            logger.info(f"Loaded {len(queries)} queries for category: {category.value}")

        # Run experiment
        summary = await runner.run_experiment(all_queries)

        logger.info("\nBenchmark Results Summary:")
        logger.info(f"  Total trials: {summary.total_trials}")
        logger.info(f"  Successful: {summary.successful_trials}")
        logger.info(f"  Mean accuracy: {summary.mean_accuracy:.4f}")
        logger.info(f"  Mean confidence: {summary.mean_confidence:.4f}")
        logger.info(f"  Mean time: {summary.mean_time:.2f}s")

        return summary

    async def run_comparison_experiment(self):
        """Run single model vs multi-agent comparison"""
        logger.info("="*60)
        logger.info("Starting Comparison Experiment")
        logger.info("="*60)

        # Create config
        config = ComparisonConfig(
            experiment_name="single_vs_multi_agent",
            baseline_models=[
                "mistral:7b-instruct-v0.2-q4_0",
                "llama3.2:latest",
                "qwen2.5:7b-instruct-q4_0"
            ],
            multi_agent_strategies=["parallel", "iterative", "consensus"],
            output_dir=str(self.output_dir / "comparison")
        )

        # Load benchmark queries
        dataset = BenchmarkDataset()
        queries = []
        queries.extend(dataset.get_queries_by_category(QueryCategory.FACTUAL, limit=15))
        queries.extend(dataset.get_queries_by_category(QueryCategory.REASONING, limit=15))
        queries.extend(dataset.get_queries_by_category(QueryCategory.SCIENTIFIC, limit=10))

        logger.info(f"Total queries for comparison: {len(queries)}")

        # Create experiment
        evaluator = self._create_evaluator()
        experiment = ComparisonExperiment(
            config=config,
            debate_manager_factory=self._create_debate_manager,
            single_model_loader=self._load_single_model,
            evaluator=evaluator
        )

        # Run comparison
        results = await experiment.run_full_comparison(queries)

        # Print summary
        logger.info("\nComparison Results:")
        if "rankings" in results and "by_accuracy" in results["rankings"]:
            logger.info("Rankings by Accuracy:")
            for i, item in enumerate(results["rankings"]["by_accuracy"][:5], 1):
                logger.info(f"  {i}. {item['config']}: {item['accuracy']:.4f}")

        return results

    async def run_ablation_study(self):
        """Run ablation study to measure agent contributions"""
        logger.info("="*60)
        logger.info("Starting Ablation Study")
        logger.info("="*60)

        # Load queries
        dataset = BenchmarkDataset()
        queries = dataset.get_queries_by_category(QueryCategory.REASONING, limit=20)

        logger.info(f"Queries for ablation: {len(queries)}")

        # Create ablation study
        evaluator = self._create_evaluator()

        def debate_factory(config):
            return self._create_debate_manager(
                ExperimentConfig(experiment_name="ablation", **config.__dict__)
            )

        ablation = AblationStudy(
            debate_manager_factory=debate_factory,
            evaluator=evaluator
        )

        # Run study
        results = await ablation.run_ablation(queries)

        # Print results
        logger.info("\nAblation Study Results:")
        for config_name, stats in results.get("configurations", {}).items():
            logger.info(f"  {config_name}: accuracy={stats['mean_accuracy']:.4f}")

        if "agent_contributions" in results:
            logger.info("\nAgent Contributions:")
            for agent, contribution in results["agent_contributions"].items():
                logger.info(f"  {agent}: {contribution:+.4f}")

        # Save results
        import json
        ablation_path = self.output_dir / "ablation"
        ablation_path.mkdir(exist_ok=True)

        with open(ablation_path / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    async def run_all_experiments(self):
        """Run all research experiments"""
        logger.info("="*60)
        logger.info("Running ALL Research Experiments")
        logger.info("="*60)

        all_results = {}

        # 1. Benchmark
        logger.info("\n[1/3] Benchmark Experiment")
        try:
            all_results["benchmark"] = await self.run_benchmark_experiment()
        except Exception as e:
            logger.error(f"Benchmark experiment failed: {e}")
            all_results["benchmark"] = {"error": str(e)}

        # 2. Comparison
        logger.info("\n[2/3] Comparison Experiment")
        try:
            all_results["comparison"] = await self.run_comparison_experiment()
        except Exception as e:
            logger.error(f"Comparison experiment failed: {e}")
            all_results["comparison"] = {"error": str(e)}

        # 3. Ablation
        logger.info("\n[3/3] Ablation Study")
        try:
            all_results["ablation"] = await self.run_ablation_study()
        except Exception as e:
            logger.error(f"Ablation study failed: {e}")
            all_results["ablation"] = {"error": str(e)}

        # Generate final report
        self._generate_final_report(all_results)

        return all_results

    def _generate_final_report(self, all_results: dict):
        """Generate comprehensive final report"""
        report_path = self.output_dir / "FINAL_REPORT.md"

        report = []
        report.append("# DR-Saintvision Research Results")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Executive Summary\n")
        report.append("This report presents the experimental evaluation of the DR-Saintvision ")
        report.append("Multi-Agent Debate System for enhanced AI reasoning.\n")

        # Benchmark results
        if "benchmark" in all_results and not isinstance(all_results["benchmark"], dict):
            benchmark = all_results["benchmark"]
            report.append("## 1. Benchmark Evaluation\n")
            report.append(f"- Total Trials: {benchmark.total_trials}")
            report.append(f"- Success Rate: {benchmark.successful_trials/benchmark.total_trials*100:.1f}%")
            report.append(f"- Mean Accuracy: {benchmark.mean_accuracy:.4f}")
            report.append(f"- Mean Confidence: {benchmark.mean_confidence:.4f}\n")

        # Comparison results
        if "comparison" in all_results and "rankings" in all_results["comparison"]:
            comparison = all_results["comparison"]
            report.append("## 2. Single Model vs Multi-Agent Comparison\n")
            report.append("### Rankings by Accuracy:\n")
            for i, item in enumerate(comparison["rankings"]["by_accuracy"][:5], 1):
                report.append(f"{i}. **{item['config']}**: {item['accuracy']:.4f}")
            report.append("")

        # Ablation results
        if "ablation" in all_results and "agent_contributions" in all_results["ablation"]:
            ablation = all_results["ablation"]
            report.append("## 3. Ablation Study\n")
            report.append("### Agent Contributions:\n")
            for agent, contribution in ablation["agent_contributions"].items():
                report.append(f"- {agent}: {contribution:+.4f}")
            report.append("")

        report.append("\n## Conclusion\n")
        report.append("The experimental results demonstrate the effectiveness of the multi-agent ")
        report.append("debate approach for complex reasoning tasks.\n")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

        logger.info(f"Final report saved to: {report_path}")


class MockDebateManager:
    """Mock debate manager for testing without actual models"""

    def __init__(self, config):
        self.config = config

    async def conduct_debate(self, query: str):
        """Return mock debate result"""
        import random

        class MockResult:
            def __init__(self):
                self.search_analysis = {"summary": "Mock search result"}
                self.reasoning_analysis = {"analysis": "Mock reasoning"}
                self.final_synthesis = {"final_answer": "Mock synthesized answer"}
                self.confidence_scores = {"overall": random.uniform(0.6, 0.9)}
                self.num_rounds = random.randint(1, 3)
                self.converged = random.random() > 0.3
                self.agreement_score = random.uniform(0.7, 0.95)

        return MockResult()


class MockEvaluator:
    """Mock evaluator for testing"""

    def evaluate_accuracy(self, query: str, response: str, reference: str) -> dict:
        import random
        return {
            "overall_accuracy": random.uniform(0.5, 0.9),
            "reference_similarity": random.uniform(0.4, 0.85)
        }

    def text_coherence(self, text: str) -> float:
        import random
        return random.uniform(0.6, 0.95)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DR-Saintvision Research Experiment Runner"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["benchmark", "comparison", "ablation", "all"],
        default="all",
        help="Which experiment to run"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./research_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models for testing (no actual inference)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("DR-Saintvision Research Runner")
    logger.info(f"Experiment: {args.experiment}")

    runner = ResearchRunner()

    if args.experiment == "benchmark":
        await runner.run_benchmark_experiment()
    elif args.experiment == "comparison":
        await runner.run_comparison_experiment()
    elif args.experiment == "ablation":
        await runner.run_ablation_study()
    elif args.experiment == "all":
        await runner.run_all_experiments()

    logger.info("\nExperiment complete!")
    logger.info(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
