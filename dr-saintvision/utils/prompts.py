"""
Prompt Templates for DR-Saintvision
Centralized prompt management for all agents
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A prompt template with metadata"""
    name: str
    template: str
    description: str
    model_type: str  # mistral, llama, qwen


class PromptTemplates:
    """Collection of prompt templates for different agents"""

    # Search Agent Prompts (Mistral)
    SEARCH_ANALYSIS = """<s>[INST] You are a research assistant specialized in web search analysis. Analyze the following search results and provide a comprehensive response.

Query: {query}

Search Results:
{search_results}

{context}

Please provide your analysis in the following format:

## Key Findings
- List the main points discovered from the search

## Relevant Facts
- Extract specific facts and data points

## Source Credibility Assessment
- Evaluate the reliability of the sources

## Summary
- Provide a concise summary answering the query

Be thorough but concise. Focus on factual information from the search results. [/INST]"""

    SEARCH_QUERY_EXPANSION = """<s>[INST] Given the following query, generate 3 alternative search queries that could help find more comprehensive information:

Original Query: {query}

Generate variations that:
1. Use different keywords
2. Approach the topic from different angles
3. Be more specific or more general as needed

Format: One query per line, no numbering. [/INST]"""

    # Reasoning Agent Prompts (Llama)
    DEEP_REASONING = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert reasoning agent with exceptional analytical capabilities. Your task is to analyze problems using systematic, deep thinking. Apply multiple reasoning approaches to ensure thorough analysis.

<|start_header_id|>user<|end_header_id|>

Analyze the following query using deep, systematic reasoning:

Query: {query}

{context}

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

    CHAIN_OF_THOUGHT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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

    ARGUMENT_EVALUATION = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in logical analysis and argumentation.

<|start_header_id|>user<|end_header_id|>

Evaluate the following argument:

{argument}

{counter_argument}

Provide:
1. **Logical Structure**: Is the argument logically valid?
2. **Premise Analysis**: Are the premises true or reasonable?
3. **Fallacy Check**: Are there any logical fallacies?
4. **Strength Rating**: Rate the argument strength (1-10)
5. **Improvements**: How could the argument be strengthened?

<|start_header_id|>assistant<|end_header_id|>

"""

    # Synthesis Agent Prompts (Qwen)
    SYNTHESIS_FULL = """<|im_start|>system
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
Confidence: {search_confidence}

{search_analysis}

=== REASONING AGENT ANALYSIS (Llama - Deep Reasoning) ===
Confidence: {reasoning_confidence}

{reasoning_analysis}

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

    QUICK_SYNTHESIS = """<|im_start|>system
You are an expert at synthesizing multiple perspectives into a clear, unified answer.
<|im_end|>

<|im_start|>user
Query: {query}

Multiple analyses:
{analyses}

Provide a unified, comprehensive answer that incorporates the best insights from all analyses.
<|im_end|>

<|im_start|>assistant
"""

    QUALITY_EVALUATION = """<|im_start|>system
You are an expert at evaluating multi-agent AI debates.
<|im_end|>

<|im_start|>user
Evaluate the quality of this AI debate:

Query: {query}

Search Agent Confidence: {search_confidence}
Reasoning Agent Confidence: {reasoning_confidence}
Synthesis Confidence: {synthesis_confidence}

Final Answer:
{final_answer}

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

    @classmethod
    def get_template(cls, name: str) -> Optional[str]:
        """Get a template by name"""
        return getattr(cls, name.upper(), None)

    @classmethod
    def format_template(cls, name: str, **kwargs) -> str:
        """Get and format a template with provided values"""
        template = cls.get_template(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """List all available templates with descriptions"""
        templates = {}
        for name in dir(cls):
            if name.isupper() and not name.startswith('_'):
                template = getattr(cls, name)
                if isinstance(template, str):
                    # Extract first line as description
                    first_line = template.split('\n')[0][:100]
                    templates[name] = first_line
        return templates

    @classmethod
    def get_korean_system_prompt(cls) -> str:
        """Get Korean language system prompt"""
        return """당신은 전문적인 AI 분석 시스템입니다.
다음 지침을 따라주세요:
1. 한국어로 응답하세요
2. 명확하고 구조화된 분석을 제공하세요
3. 증거와 논리에 기반한 결론을 도출하세요
4. 불확실한 부분은 명시적으로 표현하세요"""


# Pre-built prompt configurations for different use cases
PROMPT_CONFIGS = {
    "research": {
        "search": PromptTemplates.SEARCH_ANALYSIS,
        "reasoning": PromptTemplates.DEEP_REASONING,
        "synthesis": PromptTemplates.SYNTHESIS_FULL
    },
    "quick_answer": {
        "search": PromptTemplates.SEARCH_ANALYSIS,
        "reasoning": PromptTemplates.CHAIN_OF_THOUGHT,
        "synthesis": PromptTemplates.QUICK_SYNTHESIS
    },
    "debate_evaluation": {
        "evaluation": PromptTemplates.QUALITY_EVALUATION
    }
}
