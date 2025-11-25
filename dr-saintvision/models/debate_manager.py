"""
Debate Manager - Orchestrates the multi-agent debate process
Coordinates SearchAgent, ReasoningAgent, and SynthesisAgent
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .search_agent import SearchAgent
from .reasoning_agent import ReasoningAgent
from .synthesis_agent import SynthesisAgent

logger = logging.getLogger(__name__)


class DebateStatus(Enum):
    """Status of the debate process"""
    PENDING = "pending"
    SEARCHING = "searching"
    REASONING = "reasoning"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DebateConfig:
    """Configuration for debate sessions"""
    use_ollama: bool = False
    parallel_initial: bool = True
    max_search_results: int = 5
    enable_cot: bool = True
    timeout_seconds: float = 300.0
    save_intermediate: bool = True


@dataclass
class DebateResult:
    """Result of a debate session"""
    query: str
    search_analysis: Dict[str, Any] = field(default_factory=dict)
    reasoning_analysis: Dict[str, Any] = field(default_factory=dict)
    final_synthesis: Dict[str, Any] = field(default_factory=dict)
    debate_time: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    status: DebateStatus = DebateStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class DebateManager:
    """
    Manages the multi-agent debate process

    The debate follows this flow:
    1. SearchAgent performs web search and initial analysis (parallel)
    2. ReasoningAgent performs deep logical reasoning (parallel with search)
    3. SynthesisAgent combines both analyses for final answer
    """

    def __init__(
        self,
        config: Optional[DebateConfig] = None,
        on_status_change: Optional[Callable[[DebateStatus, str], None]] = None
    ):
        self.config = config or DebateConfig()
        self.on_status_change = on_status_change

        # Initialize agents
        self.search_agent = SearchAgent(
            use_ollama=self.config.use_ollama,
            use_quantization=not self.config.use_ollama
        )
        self.reasoning_agent = ReasoningAgent(
            use_ollama=self.config.use_ollama,
            use_quantization=not self.config.use_ollama
        )
        self.synthesis_agent = SynthesisAgent(
            use_ollama=self.config.use_ollama,
            use_quantization=not self.config.use_ollama
        )

        self._debate_history: List[DebateResult] = []
        logger.info("DebateManager initialized")

    def _update_status(self, status: DebateStatus, message: str = ""):
        """Update debate status and notify callback"""
        if self.on_status_change:
            self.on_status_change(status, message)
        logger.info(f"Debate status: {status.value} - {message}")

    async def conduct_debate(
        self,
        query: str,
        additional_context: str = ""
    ) -> DebateResult:
        """
        Conduct a full multi-agent debate on the given query

        Args:
            query: The question/topic to analyze
            additional_context: Optional additional context

        Returns:
            DebateResult containing all analyses and final synthesis
        """
        start_time = datetime.now()
        result = DebateResult(query=query)

        try:
            logger.info(f"Starting debate for query: {query[:100]}...")
            self._update_status(DebateStatus.PENDING, "Initializing debate")

            # Phase 1: Parallel search and reasoning
            if self.config.parallel_initial:
                search_result, reasoning_result = await self._parallel_analysis(
                    query, additional_context
                )
            else:
                search_result = await self._sequential_search(query, additional_context)
                reasoning_result = await self._sequential_reasoning(
                    query, additional_context, search_result
                )

            result.search_analysis = search_result
            result.reasoning_analysis = reasoning_result

            # Phase 2: Synthesis
            self._update_status(DebateStatus.SYNTHESIZING, "Combining analyses")
            synthesis_result = await self.synthesis_agent.process(
                query=query,
                search_result=search_result,
                reasoning_result=reasoning_result
            )
            result.final_synthesis = synthesis_result

            # Calculate final metrics
            result.debate_time = (datetime.now() - start_time).total_seconds()
            result.confidence_scores = {
                "search": search_result.get('confidence', 0),
                "reasoning": reasoning_result.get('confidence', 0),
                "synthesis": synthesis_result.get('confidence', 0),
                "overall": self._calculate_overall_confidence(
                    search_result, reasoning_result, synthesis_result
                )
            }
            result.status = DebateStatus.COMPLETED

            # Store in history
            self._debate_history.append(result)

            self._update_status(
                DebateStatus.COMPLETED,
                f"Debate completed in {result.debate_time:.2f}s"
            )

            logger.info(f"Debate completed successfully in {result.debate_time:.2f}s")
            return result

        except asyncio.TimeoutError:
            result.status = DebateStatus.FAILED
            result.error = "Debate timed out"
            logger.error(f"Debate timed out for query: {query[:50]}")
            self._update_status(DebateStatus.FAILED, "Timeout")
            return result

        except Exception as e:
            result.status = DebateStatus.FAILED
            result.error = str(e)
            logger.error(f"Debate failed: {e}")
            self._update_status(DebateStatus.FAILED, str(e))
            return result

    async def _parallel_analysis(
        self,
        query: str,
        context: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run search and reasoning in parallel"""
        self._update_status(DebateStatus.SEARCHING, "Starting parallel analysis")

        # Create tasks for parallel execution
        search_task = asyncio.create_task(
            self.search_agent.process(
                query=query,
                context=context,
                max_search_results=self.config.max_search_results
            )
        )

        reasoning_task = asyncio.create_task(
            self.reasoning_agent.process(
                query=query,
                context=context
            )
        )

        # Wait for both to complete
        search_result, reasoning_result = await asyncio.gather(
            search_task, reasoning_task
        )

        self._update_status(DebateStatus.REASONING, "Initial analyses complete")
        return search_result, reasoning_result

    async def _sequential_search(
        self,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """Run search sequentially"""
        self._update_status(DebateStatus.SEARCHING, "Performing web search")
        return await self.search_agent.process(
            query=query,
            context=context,
            max_search_results=self.config.max_search_results
        )

    async def _sequential_reasoning(
        self,
        query: str,
        context: str,
        search_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run reasoning sequentially with search results"""
        self._update_status(DebateStatus.REASONING, "Performing deep reasoning")
        return await self.reasoning_agent.process(
            query=query,
            context=context,
            search_results=search_result
        )

    def _calculate_overall_confidence(
        self,
        search: Dict[str, Any],
        reasoning: Dict[str, Any],
        synthesis: Dict[str, Any]
    ) -> float:
        """Calculate weighted overall confidence"""
        search_conf = search.get('confidence', 0.5)
        reasoning_conf = reasoning.get('confidence', 0.5)
        synthesis_conf = synthesis.get('confidence', 0.5)

        # Weighted average: synthesis has highest weight as final arbiter
        weights = {
            'search': 0.25,
            'reasoning': 0.30,
            'synthesis': 0.45
        }

        overall = (
            search_conf * weights['search'] +
            reasoning_conf * weights['reasoning'] +
            synthesis_conf * weights['synthesis']
        )

        return round(overall, 4)

    async def quick_debate(self, query: str) -> str:
        """
        Perform a quick debate and return just the final answer

        Args:
            query: The question to answer

        Returns:
            The final synthesized answer
        """
        result = await self.conduct_debate(query)

        if result.status == DebateStatus.COMPLETED:
            return result.final_synthesis.get('final_answer', 'No answer generated')
        else:
            return f"Debate failed: {result.error}"

    def get_debate_history(
        self,
        limit: int = 10,
        status_filter: Optional[DebateStatus] = None
    ) -> List[DebateResult]:
        """Get debate history with optional filtering"""
        history = self._debate_history

        if status_filter:
            history = [d for d in history if d.status == status_filter]

        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get debate statistics"""
        if not self._debate_history:
            return {
                "total_debates": 0,
                "average_time": 0,
                "average_confidence": 0,
                "success_rate": 0
            }

        completed = [d for d in self._debate_history if d.status == DebateStatus.COMPLETED]

        return {
            "total_debates": len(self._debate_history),
            "completed_debates": len(completed),
            "failed_debates": len(self._debate_history) - len(completed),
            "average_time": sum(d.debate_time for d in completed) / len(completed) if completed else 0,
            "average_confidence": sum(
                d.confidence_scores.get('overall', 0) for d in completed
            ) / len(completed) if completed else 0,
            "success_rate": len(completed) / len(self._debate_history) if self._debate_history else 0
        }

    def clear_history(self):
        """Clear debate history"""
        self._debate_history.clear()
        logger.info("Debate history cleared")

    async def compare_single_vs_debate(
        self,
        query: str,
        single_agent: str = "synthesis"
    ) -> Dict[str, Any]:
        """
        Compare single agent response vs full debate

        Args:
            query: Query to compare
            single_agent: Which agent to use for single response

        Returns:
            Comparison results
        """
        # Get single agent response
        if single_agent == "search":
            single_result = await self.search_agent.process(query)
            single_answer = single_result.get('analysis', '')
        elif single_agent == "reasoning":
            single_result = await self.reasoning_agent.process(query)
            single_answer = single_result.get('conclusion', '')
        else:
            # Use synthesis agent with minimal context
            single_answer = await self.synthesis_agent.quick_synthesize(
                [f"Query: {query}"], query
            )

        # Get full debate response
        debate_result = await self.conduct_debate(query)
        debate_answer = debate_result.final_synthesis.get('final_answer', '')

        return {
            "query": query,
            "single_agent": single_agent,
            "single_answer": single_answer,
            "single_length": len(single_answer),
            "debate_answer": debate_answer,
            "debate_length": len(debate_answer),
            "debate_time": debate_result.debate_time,
            "debate_confidence": debate_result.confidence_scores.get('overall', 0),
            "improvement_ratio": len(debate_answer) / len(single_answer) if single_answer else 0
        }

    def format_result_for_display(self, result: DebateResult) -> str:
        """Format debate result for human-readable display"""
        if result.status != DebateStatus.COMPLETED:
            return f"Debate failed: {result.error}"

        output = []
        output.append("=" * 60)
        output.append("DR-SAINTVISION DEBATE RESULT")
        output.append("=" * 60)
        output.append(f"\nQuery: {result.query}\n")

        # Search Analysis
        output.append("-" * 40)
        output.append("SEARCH AGENT (Mistral)")
        output.append("-" * 40)
        output.append(result.search_analysis.get('analysis', 'N/A')[:500])
        output.append(f"\nConfidence: {result.search_analysis.get('confidence', 0):.1%}")

        # Reasoning Analysis
        output.append("\n" + "-" * 40)
        output.append("REASONING AGENT (Llama)")
        output.append("-" * 40)
        output.append(result.reasoning_analysis.get('conclusion', 'N/A')[:500])
        output.append(f"\nConfidence: {result.reasoning_analysis.get('confidence', 0):.1%}")

        # Final Synthesis
        output.append("\n" + "-" * 40)
        output.append("SYNTHESIS AGENT (Qwen) - FINAL ANSWER")
        output.append("-" * 40)
        output.append(result.final_synthesis.get('final_answer', 'N/A'))
        output.append(f"\nConfidence: {result.final_synthesis.get('confidence', 0):.1%}")

        # Summary
        output.append("\n" + "=" * 60)
        output.append(f"Total Time: {result.debate_time:.2f} seconds")
        output.append(f"Overall Confidence: {result.confidence_scores.get('overall', 0):.1%}")
        output.append("=" * 60)

        return "\n".join(output)
