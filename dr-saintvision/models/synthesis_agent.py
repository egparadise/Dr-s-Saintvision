"""
Synthesis Agent - Qwen 2.5 based synthesis and final judgment agent
Responsible for combining analyses and providing final comprehensive answers
"""

import torch
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
import re

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent using Qwen2.5-7B-Instruct
    Handles combining multiple analyses and producing final judgments
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:7b-instruct-q4_0",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model

    def _create_synthesis_prompt(
        self,
        query: str,
        search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any]
    ) -> str:
        """Create the synthesis prompt combining both analyses"""

        search_analysis = search_result.get('analysis', 'No search analysis available')
        search_confidence = search_result.get('confidence', 0)

        reasoning_output = reasoning_result.get('reasoning_output', '')
        reasoning_conclusion = reasoning_result.get('conclusion', 'No conclusion available')
        reasoning_confidence = reasoning_result.get('confidence', 0)

        return f"""<|im_start|>system
You are a master synthesizer and final arbiter of truth. Your role is to analyze multiple AI agents' outputs and produce a comprehensive, balanced final answer. You excel at:
- Identifying agreements and disagreements between analyses
- Resolving contradictions with evidence-based reasoning
- Combining complementary insights
- Providing nuanced, well-supported conclusions
<|im_end|>

<|im_start|>user
Original Query: {query}

I have received analyses from two specialized AI agents. Please synthesize their findings:

=== SEARCH AGENT ANALYSIS (Mistral - Web Search & RAG) ===
Confidence: {search_confidence:.1%}

{search_analysis}

=== REASONING AGENT ANALYSIS (Llama - Deep Reasoning) ===
Confidence: {reasoning_confidence:.1%}

Conclusion: {reasoning_conclusion}

Full Reasoning:
{reasoning_output[:2000]}

---

Please provide your synthesis in the following format:

## Agreement Analysis
What do both agents agree on?

## Disagreement Analysis
Where do the agents differ, and why might that be?

## Contradiction Resolution
If there are contradictions, how do you resolve them based on evidence?

## Complementary Insights
What unique insights does each agent provide that the other doesn't?

## Final Comprehensive Answer
Provide a clear, definitive answer to the original query.

## Confidence Assessment
Rate your overall confidence (0-100%) and explain the basis for this rating.

## Limitations and Caveats
What are the limitations of this analysis? What additional information would be helpful?
<|im_end|>

<|im_start|>assistant
"""

    async def process(
        self,
        query: str,
        search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize results from search and reasoning agents

        Args:
            query: Original query
            search_result: Results from SearchAgent
            reasoning_result: Results from ReasoningAgent

        Returns:
            Synthesized analysis and final answer
        """
        start_time = datetime.now()

        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(query, search_result, reasoning_result)

        # Generate synthesis
        if self.use_ollama:
            synthesis_output = await self._generate_with_ollama(prompt)
        else:
            synthesis_output = self.generate_response(
                prompt,
                max_new_tokens=2000,
                temperature=0.5,  # Lower temperature for more consistent synthesis
                top_p=0.9
            )

        # Parse synthesis components
        agreements = self._extract_section(synthesis_output, "Agreement Analysis")
        disagreements = self._extract_section(synthesis_output, "Disagreement Analysis")
        resolution = self._extract_section(synthesis_output, "Contradiction Resolution")
        complementary = self._extract_section(synthesis_output, "Complementary Insights")
        final_answer = self._extract_section(synthesis_output, "Final Comprehensive Answer")
        limitations = self._extract_section(synthesis_output, "Limitations and Caveats")

        # Calculate overall confidence
        confidence = self._calculate_synthesis_confidence(
            synthesis_output,
            search_result.get('confidence', 0.5),
            reasoning_result.get('confidence', 0.5)
        )

        elapsed_time = (datetime.now() - start_time).total_seconds()

        return {
            "agent": "synthesis",
            "agent_name": "Qwen Synthesis Agent",
            "query": query,
            "synthesis_output": synthesis_output,
            "final_answer": final_answer,
            "agreements": agreements,
            "disagreements": disagreements,
            "resolution": resolution,
            "complementary_insights": complementary,
            "limitations": limitations,
            "confidence": confidence,
            "input_confidences": {
                "search": search_result.get('confidence', 0),
                "reasoning": reasoning_result.get('confidence', 0)
            },
            "processing_time": elapsed_time,
            "model": self.model_name if not self.use_ollama else self.ollama_model
        }

    async def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "top_p": 0.9,
                            "num_predict": 2000,
                            "num_ctx": 8192  # Larger context for synthesis
                        }
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error generating response: {e}"

    def _extract_section(self, output: str, section_name: str) -> str:
        """Extract a specific section from the synthesis output"""
        # Try to find section with ## header
        pattern = rf"##\s*{re.escape(section_name)}[:\s]*(.*?)(?=##|$)"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # Try without ## header
        pattern = rf"{re.escape(section_name)}[:\s]*(.*?)(?=\n[A-Z]|\n##|$)"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return ""

    def _calculate_synthesis_confidence(
        self,
        output: str,
        search_confidence: float,
        reasoning_confidence: float
    ) -> float:
        """Calculate overall synthesis confidence"""
        # Start with weighted average of input confidences
        base_confidence = (search_confidence * 0.4 + reasoning_confidence * 0.4)

        # Look for explicit confidence in output
        confidence_match = re.search(
            r"(?:confidence|certainty)[:\s]*(\d+)[%]?",
            output,
            re.IGNORECASE
        )
        if confidence_match:
            stated = int(confidence_match.group(1))
            if stated <= 100:
                base_confidence = (base_confidence + stated / 100) / 2

        # Adjust based on synthesis quality indicators
        quality_indicators = [
            ("agreement", 0.05),
            ("disagree", 0.03),  # Having disagreements shows thorough analysis
            ("evidence", 0.05),
            ("however", 0.03),
            ("limitation", 0.03),
            ("conclusion", 0.05)
        ]

        for indicator, bonus in quality_indicators:
            if indicator in output.lower():
                base_confidence += bonus

        # Penalty if synthesis is too short
        if len(output) < 500:
            base_confidence -= 0.1

        # Bonus if synthesis has all sections
        sections = ["Agreement", "Disagreement", "Final", "Confidence", "Limitation"]
        sections_found = sum(1 for s in sections if s.lower() in output.lower())
        base_confidence += sections_found * 0.02

        return min(max(base_confidence, 0.0), 1.0)

    async def quick_synthesize(
        self,
        analyses: List[str],
        query: str
    ) -> str:
        """
        Quick synthesis of multiple analyses without full structure

        Args:
            analyses: List of analysis texts
            query: Original query

        Returns:
            Synthesized answer
        """
        combined_analyses = "\n\n---\n\n".join([
            f"Analysis {i+1}:\n{analysis}"
            for i, analysis in enumerate(analyses)
        ])

        prompt = f"""<|im_start|>system
You are an expert at synthesizing multiple perspectives into a clear, unified answer.
<|im_end|>

<|im_start|>user
Query: {query}

Multiple analyses:
{combined_analyses}

Provide a unified, comprehensive answer that incorporates the best insights from all analyses.
<|im_end|>

<|im_start|>assistant
"""

        if self.use_ollama:
            return await self._generate_with_ollama(prompt)
        else:
            return self.generate_response(prompt, max_new_tokens=800)

    async def evaluate_debate_quality(
        self,
        debate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a debate/synthesis session

        Args:
            debate_result: Full debate result dictionary

        Returns:
            Quality evaluation metrics
        """
        prompt = f"""<|im_start|>system
You are an expert at evaluating multi-agent AI debates.
<|im_end|>

<|im_start|>user
Evaluate the quality of this AI debate:

Query: {debate_result.get('query', 'Unknown')}

Search Agent Confidence: {debate_result.get('search_analysis', {}).get('confidence', 0):.1%}
Reasoning Agent Confidence: {debate_result.get('reasoning_analysis', {}).get('confidence', 0):.1%}
Synthesis Confidence: {debate_result.get('final_synthesis', {}).get('confidence', 0):.1%}

Final Answer:
{debate_result.get('final_synthesis', {}).get('final_answer', 'No answer')}

Rate the following on a scale of 1-10:
1. Comprehensiveness
2. Logical Consistency
3. Evidence Quality
4. Clarity
5. Objectivity

Provide brief explanations for each rating.
<|im_end|>

<|im_start|>assistant
"""

        if self.use_ollama:
            evaluation = await self._generate_with_ollama(prompt)
        else:
            evaluation = self.generate_response(prompt, max_new_tokens=600)

        # Parse ratings
        ratings = {}
        for metric in ["Comprehensiveness", "Logical Consistency", "Evidence Quality", "Clarity", "Objectivity"]:
            match = re.search(rf"{metric}[:\s]*(\d+)", evaluation, re.IGNORECASE)
            if match:
                ratings[metric.lower().replace(" ", "_")] = int(match.group(1))

        overall_score = sum(ratings.values()) / len(ratings) if ratings else 5

        return {
            "evaluation_text": evaluation,
            "ratings": ratings,
            "overall_score": overall_score,
            "grade": self._score_to_grade(overall_score)
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B+"
        elif score >= 6:
            return "B"
        elif score >= 5:
            return "C"
        elif score >= 4:
            return "D"
        else:
            return "F"
