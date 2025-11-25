"""
Benchmark Datasets and Evaluation for DR-Saintvision Research
Standard benchmarks for evaluating Multi-Agent Debate Systems

Includes:
- Question Answering benchmarks
- Reasoning benchmarks
- Fact verification benchmarks
- Custom debate-specific benchmarks
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class BenchmarkCategory(Enum):
    """Categories of benchmark questions"""
    FACTUAL = "factual"           # Fact-based questions
    REASONING = "reasoning"        # Logical reasoning
    OPINION = "opinion"           # Subjective/opinion questions
    SCIENTIFIC = "scientific"      # Science-based questions
    ETHICAL = "ethical"           # Ethical dilemmas
    CURRENT_EVENTS = "current"    # Current events
    COMPLEX = "complex"           # Multi-faceted questions


@dataclass
class BenchmarkQuestion:
    """Single benchmark question"""
    id: str
    question: str
    category: BenchmarkCategory
    reference_answer: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "custom"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """Collection of benchmark questions"""
    name: str
    description: str
    version: str
    questions: List[BenchmarkQuestion]
    categories: List[BenchmarkCategory] = field(default_factory=list)

    def __len__(self):
        return len(self.questions)

    def filter_by_category(self, category: BenchmarkCategory) -> 'BenchmarkDataset':
        """Filter questions by category"""
        filtered = [q for q in self.questions if q.category == category]
        return BenchmarkDataset(
            name=f"{self.name}_{category.value}",
            description=f"Filtered: {category.value}",
            version=self.version,
            questions=filtered,
            categories=[category]
        )

    def filter_by_difficulty(self, difficulty: str) -> 'BenchmarkDataset':
        """Filter questions by difficulty"""
        filtered = [q for q in self.questions if q.difficulty == difficulty]
        return BenchmarkDataset(
            name=f"{self.name}_{difficulty}",
            description=f"Filtered: {difficulty}",
            version=self.version,
            questions=filtered
        )

    def sample(self, n: int, seed: int = 42) -> 'BenchmarkDataset':
        """Random sample of questions"""
        random.seed(seed)
        sampled = random.sample(self.questions, min(n, len(self.questions)))
        return BenchmarkDataset(
            name=f"{self.name}_sample_{n}",
            description=f"Sampled {n} questions",
            version=self.version,
            questions=sampled
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "num_questions": len(self.questions),
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "category": q.category.value,
                    "reference_answer": q.reference_answer,
                    "key_points": q.key_points,
                    "difficulty": q.difficulty
                }
                for q in self.questions
            ]
        }

    def save(self, path: str):
        """Save benchmark to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkDataset':
        """Load benchmark from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = [
            BenchmarkQuestion(
                id=q["id"],
                question=q["question"],
                category=BenchmarkCategory(q["category"]),
                reference_answer=q.get("reference_answer"),
                key_points=q.get("key_points", []),
                difficulty=q.get("difficulty", "medium")
            )
            for q in data["questions"]
        ]

        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            questions=questions
        )


class StandardBenchmarks:
    """
    Standard benchmark datasets for Multi-Agent evaluation

    Includes diverse questions across multiple categories
    for comprehensive system evaluation
    """

    @staticmethod
    def create_reasoning_benchmark() -> BenchmarkDataset:
        """Create benchmark for reasoning capabilities"""
        questions = [
            BenchmarkQuestion(
                id="reason_001",
                question="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                category=BenchmarkCategory.REASONING,
                reference_answer="No, we cannot conclude this. While all roses are flowers, the flowers that fade quickly might not include any roses.",
                key_points=["syllogism", "logical fallacy", "some vs all"],
                difficulty="medium"
            ),
            BenchmarkQuestion(
                id="reason_002",
                question="A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                category=BenchmarkCategory.REASONING,
                reference_answer="The ball costs $0.05. If the ball costs $0.05, the bat costs $1.05 ($1.00 more), and together they cost $1.10.",
                key_points=["algebra", "intuition trap", "careful calculation"],
                difficulty="easy"
            ),
            BenchmarkQuestion(
                id="reason_003",
                question="Three people check into a hotel room that costs $30. They each contribute $10. Later, the manager realizes the room only costs $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each guest. Now each guest has paid $9 (totaling $27), and the bellboy has $2. Where is the missing dollar?",
                category=BenchmarkCategory.REASONING,
                reference_answer="There is no missing dollar. The $27 paid by guests includes the $25 for the room and $2 kept by the bellboy. The question misleadingly adds $27 + $2 instead of subtracting.",
                key_points=["accounting", "misdirection", "correct framing"],
                difficulty="medium"
            ),
            BenchmarkQuestion(
                id="reason_004",
                question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                category=BenchmarkCategory.REASONING,
                reference_answer="5 minutes. Each machine makes 1 widget in 5 minutes. So 100 machines would make 100 widgets in the same 5 minutes.",
                key_points=["rate problem", "parallel processing", "proportional reasoning"],
                difficulty="easy"
            ),
            BenchmarkQuestion(
                id="reason_005",
                question="A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
                category=BenchmarkCategory.REASONING,
                reference_answer="9 sheep. 'All but 9' means 9 remain.",
                key_points=["language interpretation", "attention to wording"],
                difficulty="easy"
            ),
        ]

        return BenchmarkDataset(
            name="reasoning_benchmark",
            description="Benchmark for logical reasoning and problem-solving",
            version="1.0",
            questions=questions,
            categories=[BenchmarkCategory.REASONING]
        )

    @staticmethod
    def create_factual_benchmark() -> BenchmarkDataset:
        """Create benchmark for factual knowledge"""
        questions = [
            BenchmarkQuestion(
                id="fact_001",
                question="What causes the seasons on Earth?",
                category=BenchmarkCategory.FACTUAL,
                reference_answer="Earth's seasons are caused by the tilt of Earth's axis (23.5 degrees) relative to its orbital plane around the Sun, not by the distance from the Sun.",
                key_points=["axial tilt", "23.5 degrees", "orbital plane", "not distance"],
                difficulty="medium"
            ),
            BenchmarkQuestion(
                id="fact_002",
                question="How do vaccines work to protect against diseases?",
                category=BenchmarkCategory.FACTUAL,
                reference_answer="Vaccines work by training the immune system to recognize and fight pathogens. They contain weakened or inactive parts of a pathogen, triggering an immune response that creates memory cells for future protection.",
                key_points=["immune system", "antibodies", "memory cells", "pathogens"],
                difficulty="medium"
            ),
            BenchmarkQuestion(
                id="fact_003",
                question="What is the difference between weather and climate?",
                category=BenchmarkCategory.FACTUAL,
                reference_answer="Weather refers to short-term atmospheric conditions (hours to days), while climate refers to long-term average weather patterns over 30+ years in a region.",
                key_points=["short-term vs long-term", "atmospheric conditions", "patterns", "30 years"],
                difficulty="easy"
            ),
            BenchmarkQuestion(
                id="fact_004",
                question="How does photosynthesis work?",
                category=BenchmarkCategory.FACTUAL,
                reference_answer="Photosynthesis converts light energy, water, and CO2 into glucose and oxygen. It occurs in chloroplasts using chlorophyll to capture light energy.",
                key_points=["light energy", "CO2", "water", "glucose", "oxygen", "chlorophyll"],
                difficulty="medium"
            ),
            BenchmarkQuestion(
                id="fact_005",
                question="What causes earthquakes?",
                category=BenchmarkCategory.FACTUAL,
                reference_answer="Earthquakes are caused by the sudden release of energy in the Earth's crust, typically due to tectonic plate movements along fault lines.",
                key_points=["tectonic plates", "fault lines", "energy release", "crust"],
                difficulty="easy"
            ),
        ]

        return BenchmarkDataset(
            name="factual_benchmark",
            description="Benchmark for factual knowledge and explanation",
            version="1.0",
            questions=questions,
            categories=[BenchmarkCategory.FACTUAL]
        )

    @staticmethod
    def create_scientific_benchmark() -> BenchmarkDataset:
        """Create benchmark for scientific reasoning"""
        questions = [
            BenchmarkQuestion(
                id="sci_001",
                question="Why is the sky blue during the day but red/orange during sunset?",
                category=BenchmarkCategory.SCIENTIFIC,
                reference_answer="Blue light has a shorter wavelength and scatters more in the atmosphere (Rayleigh scattering). At sunset, sunlight travels through more atmosphere, scattering away blue light and allowing longer wavelengths (red/orange) to reach our eyes.",
                key_points=["Rayleigh scattering", "wavelength", "atmosphere thickness", "blue shorter wavelength"],
                difficulty="medium"
            ),
            BenchmarkQuestion(
                id="sci_002",
                question="How do black holes form and why can't light escape from them?",
                category=BenchmarkCategory.SCIENTIFIC,
                reference_answer="Black holes form when massive stars collapse under their own gravity after exhausting nuclear fuel. Their gravitational pull is so strong that the escape velocity exceeds the speed of light, preventing anything, including light, from escaping.",
                key_points=["stellar collapse", "escape velocity", "event horizon", "gravitational singularity"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="sci_003",
                question="What is quantum entanglement and why did Einstein call it 'spooky action at a distance'?",
                category=BenchmarkCategory.SCIENTIFIC,
                reference_answer="Quantum entanglement is when particles become correlated so that the quantum state of one instantly affects the other, regardless of distance. Einstein found this 'spooky' because it seemed to violate locality, but it doesn't transmit information faster than light.",
                key_points=["quantum correlation", "instantaneous", "non-locality", "no FTL information"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="sci_004",
                question="Why do we age? What biological processes drive aging?",
                category=BenchmarkCategory.SCIENTIFIC,
                reference_answer="Aging results from multiple factors: telomere shortening, accumulated DNA damage, cellular senescence, mitochondrial dysfunction, and declining stem cell function. These lead to reduced cellular repair and increased disease susceptibility.",
                key_points=["telomeres", "DNA damage", "cellular senescence", "mitochondria", "stem cells"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="sci_005",
                question="How does CRISPR gene editing work?",
                category=BenchmarkCategory.SCIENTIFIC,
                reference_answer="CRISPR uses a guide RNA to direct the Cas9 protein to a specific DNA sequence. Cas9 cuts the DNA at that location, allowing genes to be removed, modified, or inserted through the cell's natural repair mechanisms.",
                key_points=["guide RNA", "Cas9", "DNA cutting", "gene modification", "cell repair"],
                difficulty="hard"
            ),
        ]

        return BenchmarkDataset(
            name="scientific_benchmark",
            description="Benchmark for scientific knowledge and explanation",
            version="1.0",
            questions=questions,
            categories=[BenchmarkCategory.SCIENTIFIC]
        )

    @staticmethod
    def create_ethical_benchmark() -> BenchmarkDataset:
        """Create benchmark for ethical reasoning"""
        questions = [
            BenchmarkQuestion(
                id="eth_001",
                question="Is it ethical for AI to make life-or-death decisions in autonomous vehicles?",
                category=BenchmarkCategory.ETHICAL,
                reference_answer="This is a complex ethical issue involving multiple perspectives: utilitarian (minimize total harm), deontological (rights-based duties), and virtue ethics considerations. Key concerns include accountability, transparency, and the trolley problem scenarios.",
                key_points=["accountability", "transparency", "trolley problem", "utilitarian", "deontological"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="eth_002",
                question="Should genetic engineering be used to eliminate hereditary diseases in embryos?",
                category=BenchmarkCategory.ETHICAL,
                reference_answer="This involves weighing benefits (disease prevention) against concerns (designer babies, inequality, consent). Key ethical frameworks include autonomy, beneficence, non-maleficence, and justice.",
                key_points=["consent", "designer babies", "inequality", "autonomy", "beneficence"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="eth_003",
                question="Is universal basic income (UBI) ethically justified?",
                category=BenchmarkCategory.ETHICAL,
                reference_answer="UBI arguments involve freedom (providing basic security), justice (fair distribution), dignity (unconditional support) vs concerns about work incentives, funding, and economic effects.",
                key_points=["freedom", "dignity", "work incentives", "economic justice", "sustainability"],
                difficulty="medium"
            ),
        ]

        return BenchmarkDataset(
            name="ethical_benchmark",
            description="Benchmark for ethical reasoning and moral philosophy",
            version="1.0",
            questions=questions,
            categories=[BenchmarkCategory.ETHICAL]
        )

    @staticmethod
    def create_complex_benchmark() -> BenchmarkDataset:
        """Create benchmark for complex multi-faceted questions"""
        questions = [
            BenchmarkQuestion(
                id="complex_001",
                question="How might artificial intelligence transform the job market over the next 20 years, and what should society do to prepare?",
                category=BenchmarkCategory.COMPLEX,
                reference_answer="AI will likely automate routine tasks while creating new jobs requiring creativity and social skills. Preparation should include education reform, social safety nets, and policies for workforce transition.",
                key_points=["automation", "job displacement", "new jobs", "education", "policy", "transition"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="complex_002",
                question="What are the most effective strategies to address climate change at global, national, and individual levels?",
                category=BenchmarkCategory.COMPLEX,
                reference_answer="Effective strategies span multiple levels: global cooperation (Paris Agreement), national policies (carbon pricing, renewable investment), and individual actions (consumption changes, advocacy).",
                key_points=["international cooperation", "carbon pricing", "renewable energy", "individual action", "policy"],
                difficulty="hard"
            ),
            BenchmarkQuestion(
                id="complex_003",
                question="What are the implications of social media on democracy, mental health, and social cohesion?",
                category=BenchmarkCategory.COMPLEX,
                reference_answer="Social media has mixed effects: democratizing information but enabling misinformation; connecting people but potentially increasing isolation; empowering movements but creating polarization.",
                key_points=["misinformation", "polarization", "mental health", "democratic participation", "echo chambers"],
                difficulty="hard"
            ),
        ]

        return BenchmarkDataset(
            name="complex_benchmark",
            description="Benchmark for complex multi-faceted analysis",
            version="1.0",
            questions=questions,
            categories=[BenchmarkCategory.COMPLEX]
        )

    @classmethod
    def create_full_benchmark(cls) -> BenchmarkDataset:
        """Create comprehensive benchmark combining all categories"""
        reasoning = cls.create_reasoning_benchmark()
        factual = cls.create_factual_benchmark()
        scientific = cls.create_scientific_benchmark()
        ethical = cls.create_ethical_benchmark()
        complex_q = cls.create_complex_benchmark()

        all_questions = (
            reasoning.questions +
            factual.questions +
            scientific.questions +
            ethical.questions +
            complex_q.questions
        )

        return BenchmarkDataset(
            name="dr_saintvision_benchmark",
            description="Comprehensive benchmark for Multi-Agent Debate System evaluation",
            version="1.0",
            questions=all_questions,
            categories=list(BenchmarkCategory)
        )


class BenchmarkRunner:
    """Runner for benchmark evaluation"""

    def __init__(self, debate_manager: Any, evaluator: Any):
        self.debate_manager = debate_manager
        self.evaluator = evaluator
        self.results: List[Dict] = []

    async def run_benchmark(
        self,
        benchmark: BenchmarkDataset,
        max_questions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run benchmark and collect results"""
        questions = benchmark.questions[:max_questions] if max_questions else benchmark.questions

        self.results = []

        for i, question in enumerate(questions):
            logger.info(f"Benchmark {i+1}/{len(questions)}: {question.id}")

            try:
                result = await self.debate_manager.conduct_debate(question.question)

                # Evaluate
                final_answer = result.final_synthesis.get("final_answer", "")

                metrics = {}
                if question.reference_answer:
                    metrics = self.evaluator.evaluate_accuracy(
                        question.question,
                        final_answer,
                        question.reference_answer
                    )

                # Check key points coverage
                if question.key_points:
                    coverage = self.evaluator.keyword_coverage(
                        final_answer,
                        question.key_points
                    )
                    metrics["key_point_coverage"] = coverage

                self.results.append({
                    "question_id": question.id,
                    "category": question.category.value,
                    "difficulty": question.difficulty,
                    "metrics": metrics,
                    "confidence": result.confidence_scores,
                    "time": result.debate_time,
                    "success": True
                })

            except Exception as e:
                logger.error(f"Failed on {question.id}: {e}")
                self.results.append({
                    "question_id": question.id,
                    "success": False,
                    "error": str(e)
                })

        return self._summarize_results(benchmark.name)

    def _summarize_results(self, benchmark_name: str) -> Dict[str, Any]:
        """Summarize benchmark results"""
        successful = [r for r in self.results if r.get("success")]

        summary = {
            "benchmark": benchmark_name,
            "total_questions": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "overall_accuracy": 0,
            "by_category": {},
            "by_difficulty": {}
        }

        if successful:
            accuracies = [
                r["metrics"].get("overall_accuracy", 0)
                for r in successful if "metrics" in r
            ]
            summary["overall_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0

            # By category
            categories = set(r.get("category") for r in successful if r.get("category"))
            for cat in categories:
                cat_results = [r for r in successful if r.get("category") == cat]
                cat_accs = [r["metrics"].get("overall_accuracy", 0) for r in cat_results if "metrics" in r]
                summary["by_category"][cat] = {
                    "count": len(cat_results),
                    "accuracy": sum(cat_accs) / len(cat_accs) if cat_accs else 0
                }

            # By difficulty
            difficulties = set(r.get("difficulty") for r in successful if r.get("difficulty"))
            for diff in difficulties:
                diff_results = [r for r in successful if r.get("difficulty") == diff]
                diff_accs = [r["metrics"].get("overall_accuracy", 0) for r in diff_results if "metrics" in r]
                summary["by_difficulty"][diff] = {
                    "count": len(diff_results),
                    "accuracy": sum(diff_accs) / len(diff_accs) if diff_accs else 0
                }

        return summary
