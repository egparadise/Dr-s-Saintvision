"""
Reasoning Agent - Llama 3.2 based deep reasoning agent
Responsible for Chain-of-Thought reasoning and logical analysis
"""

import torch
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
import re

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """
    Deep Reasoning Agent using Llama-3.2-7B
    Handles complex reasoning, Chain-of-Thought, and logical analysis
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-7B-Instruct",
        use_ollama: bool = False,
        ollama_model: str = "llama3.2:latest",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.reasoning_patterns = [
            "deductive",    # From general to specific
            "inductive",    # From specific to general
            "abductive",    # Best explanation inference
            "analogical"    # Reasoning by analogy
        ]

    def _create_reasoning_prompt(
        self,
        query: str,
        context: str = "",
        reasoning_type: str = "comprehensive"
    ) -> str:
        """Create the reasoning prompt for the model"""

        context_section = f"\nAdditional Context:\n{context}\n" if context else ""

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert reasoning agent with exceptional analytical capabilities. Your task is to analyze problems using systematic, deep thinking. Apply multiple reasoning approaches to ensure thorough analysis.

<|start_header_id|>user<|end_header_id|>

Analyze the following query using deep, systematic reasoning:

Query: {query}
{context_section}

Please follow this structured reasoning process:

## Step 1: Problem Decomposition
Break down the problem into its fundamental components and sub-questions.

## Step 2: Assumptions and Constraints
Identify key assumptions being made and any constraints on the problem.

## Step 3: Multi-perspective Analysis
Apply different reasoning approaches:
- **Deductive Reasoning**: What conclusions can we draw from established facts?
- **Inductive Reasoning**: What patterns can we observe and generalize?
- **Abductive Reasoning**: What is the most likely explanation?

## Step 4: Evidence Evaluation
Assess the strength of evidence and arguments for each perspective.

## Step 5: Synthesis and Conclusion
Combine insights into a coherent, well-reasoned conclusion.

## Confidence Assessment
Rate your confidence in the conclusion and explain why.

<|start_header_id|>assistant<|end_header_id|>

"""

    async def process(
        self,
        query: str,
        context: str = "",
        search_results: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query with deep reasoning

        Args:
            query: The query to analyze
            context: Additional context
            search_results: Optional search results from SearchAgent

        Returns:
            Dictionary containing reasoning analysis
        """
        start_time = datetime.now()

        # Build context from search results if available
        if search_results and 'analysis' in search_results:
            context = f"{context}\n\nSearch Analysis:\n{search_results['analysis']}"

        # Create reasoning prompt
        prompt = self._create_reasoning_prompt(query, context)

        # Generate reasoning
        if self.use_ollama:
            reasoning_output = await self._generate_with_ollama(prompt)
        else:
            reasoning_output = self.generate_response(
                prompt,
                max_new_tokens=1500,
                temperature=0.6,  # Lower temperature for more focused reasoning
                top_p=0.9
            )

        # Parse reasoning steps
        reasoning_steps = self._parse_reasoning_steps(reasoning_output)

        # Extract conclusion
        conclusion = self._extract_conclusion(reasoning_output)

        # Calculate confidence
        confidence = self._calculate_confidence(reasoning_output, reasoning_steps)

        elapsed_time = (datetime.now() - start_time).total_seconds()

        return {
            "agent": "reasoning",
            "agent_name": "Llama Reasoning Agent",
            "query": query,
            "reasoning_output": reasoning_output,
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
            "confidence": confidence,
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
                            "temperature": 0.6,
                            "top_p": 0.9,
                            "num_predict": 1500
                        }
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error generating response: {e}"

    def _parse_reasoning_steps(self, output: str) -> List[Dict[str, str]]:
        """Parse reasoning steps from the output"""
        steps = []

        # Define step patterns to look for
        step_patterns = [
            (r"## Step 1:.*?Problem Decomposition(.*?)(?=## Step|$)", "Problem Decomposition"),
            (r"## Step 2:.*?Assumptions(.*?)(?=## Step|$)", "Assumptions and Constraints"),
            (r"## Step 3:.*?Analysis(.*?)(?=## Step|$)", "Multi-perspective Analysis"),
            (r"## Step 4:.*?Evidence(.*?)(?=## Step|$)", "Evidence Evaluation"),
            (r"## Step 5:.*?(?:Synthesis|Conclusion)(.*?)(?=## Confidence|$)", "Synthesis"),
        ]

        for pattern, step_name in step_patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                steps.append({
                    "step": step_name,
                    "content": content
                })

        # If no structured steps found, create a single step with full output
        if not steps:
            steps.append({
                "step": "Analysis",
                "content": output
            })

        return steps

    def _extract_conclusion(self, output: str) -> str:
        """Extract the conclusion from reasoning output"""
        # Look for conclusion section
        patterns = [
            r"## Step 5:.*?(?:Synthesis|Conclusion)(.*?)(?=## Confidence|$)",
            r"(?:Conclusion|Final Conclusion|Summary)[:\s]*(.*?)(?=## |$)",
            r"(?:Therefore|Thus|In conclusion)[,:\s]*(.*?)(?=\n\n|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                if len(conclusion) > 50:  # Ensure meaningful conclusion
                    return conclusion

        # If no conclusion found, return last paragraph
        paragraphs = output.strip().split('\n\n')
        if paragraphs:
            return paragraphs[-1].strip()

        return output[:500]

    def _calculate_confidence(
        self,
        output: str,
        reasoning_steps: List[Dict]
    ) -> float:
        """Calculate confidence score based on reasoning quality"""
        confidence = 0.5  # Base confidence

        # Check for completeness of reasoning steps
        if len(reasoning_steps) >= 4:
            confidence += 0.15
        elif len(reasoning_steps) >= 2:
            confidence += 0.08

        # Check for explicit confidence statement
        confidence_match = re.search(
            r"(?:confidence|certainty)[:\s]*(\d+)[%]?",
            output,
            re.IGNORECASE
        )
        if confidence_match:
            stated_confidence = int(confidence_match.group(1))
            if stated_confidence <= 100:
                # Weight stated confidence
                confidence = (confidence + stated_confidence / 100) / 2

        # Check for reasoning indicators
        reasoning_indicators = [
            "because", "therefore", "thus", "hence",
            "evidence", "suggests", "indicates",
            "however", "although", "considering"
        ]

        indicator_count = sum(
            1 for indicator in reasoning_indicators
            if indicator in output.lower()
        )
        confidence += min(indicator_count * 0.02, 0.15)

        # Check for multiple perspectives
        perspective_keywords = ["deductive", "inductive", "abductive", "perspective"]
        for keyword in perspective_keywords:
            if keyword in output.lower():
                confidence += 0.03

        return min(confidence, 1.0)

    async def chain_of_thought(
        self,
        query: str,
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Perform explicit Chain-of-Thought reasoning

        Args:
            query: Query to reason about
            max_steps: Maximum reasoning steps

        Returns:
            CoT reasoning results
        """
        cot_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a logical reasoning expert. Think through problems step by step.

<|start_header_id|>user<|end_header_id|>

Think through this problem step by step:

{query}

For each step, clearly state:
1. What you're considering
2. Your reasoning
3. Your intermediate conclusion

Then provide your final answer.

<|start_header_id|>assistant<|end_header_id|>

Let me think through this step by step:

"""
        if self.use_ollama:
            response = await self._generate_with_ollama(cot_prompt)
        else:
            response = self.generate_response(
                cot_prompt,
                max_new_tokens=1200,
                temperature=0.5
            )

        # Parse steps from response
        steps = []
        step_matches = re.findall(
            r"(?:Step \d+|First|Second|Third|Fourth|Fifth|Next|Finally)[:\s]*(.*?)(?=(?:Step \d+|First|Second|Third|Fourth|Fifth|Next|Finally|$))",
            response,
            re.DOTALL | re.IGNORECASE
        )

        for i, match in enumerate(step_matches[:max_steps]):
            steps.append({
                "step_number": i + 1,
                "content": match.strip()
            })

        return {
            "query": query,
            "cot_response": response,
            "steps": steps,
            "num_steps": len(steps)
        }

    async def evaluate_argument(
        self,
        argument: str,
        counter_argument: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the strength of an argument

        Args:
            argument: The argument to evaluate
            counter_argument: Optional counter-argument

        Returns:
            Argument evaluation results
        """
        counter_section = f"\n\nCounter-argument to consider:\n{counter_argument}" if counter_argument else ""

        eval_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in logical analysis and argumentation.

<|start_header_id|>user<|end_header_id|>

Evaluate the following argument:

{argument}
{counter_section}

Provide:
1. **Logical Structure**: Is the argument logically valid?
2. **Premise Analysis**: Are the premises true or reasonable?
3. **Fallacy Check**: Are there any logical fallacies?
4. **Strength Rating**: Rate the argument strength (1-10)
5. **Improvements**: How could the argument be strengthened?

<|start_header_id|>assistant<|end_header_id|>

"""
        if self.use_ollama:
            evaluation = await self._generate_with_ollama(eval_prompt)
        else:
            evaluation = self.generate_response(
                eval_prompt,
                max_new_tokens=800,
                temperature=0.5
            )

        # Extract strength rating
        strength_match = re.search(r"(?:strength|rating)[:\s]*(\d+)", evaluation, re.IGNORECASE)
        strength = int(strength_match.group(1)) if strength_match else 5

        return {
            "argument": argument,
            "counter_argument": counter_argument,
            "evaluation": evaluation,
            "strength_rating": strength
        }
