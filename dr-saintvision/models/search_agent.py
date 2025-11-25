"""
Search Agent - Mistral 7B based web search and analysis agent
Responsible for web search and RAG (Retrieval-Augmented Generation)
"""

import torch
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    Web Search Agent using Mistral-7B-Instruct
    Handles web search, information retrieval, and initial analysis
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_ollama: bool = False,
        ollama_model: str = "mistral:7b-instruct-v0.2-q4_0",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.search_tool = None
        self._init_search_tool()

    def _init_search_tool(self):
        """Initialize web search tool"""
        try:
            from duckduckgo_search import DDGS
            self.search_tool = DDGS()
            logger.info("DuckDuckGo search tool initialized")
        except ImportError:
            logger.warning("duckduckgo-search not installed. Web search disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize search tool: {e}")

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform web search and return results"""
        if self.search_tool is None:
            logger.warning("Search tool not available")
            return []

        try:
            results = []
            search_results = self.search_tool.text(query, max_results=max_results)

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", "")
                })

            logger.info(f"Found {len(results)} search results for: {query}")
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for the prompt"""
        if not results:
            return "No search results found."

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"""
[Result {i}]
Title: {result['title']}
Content: {result['body']}
Source: {result['href']}
""")
        return "\n".join(formatted)

    def _create_analysis_prompt(self, query: str, search_results: str) -> str:
        """Create the analysis prompt for the model"""
        return f"""<s>[INST] You are a research assistant specialized in web search analysis. Analyze the following search results and provide a comprehensive response.

Query: {query}

Search Results:
{search_results}

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

    async def process(
        self,
        query: str,
        context: str = "",
        max_search_results: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query with web search and analysis

        Args:
            query: The search query
            context: Additional context
            max_search_results: Maximum number of search results

        Returns:
            Dictionary containing search results and analysis
        """
        start_time = datetime.now()

        # Perform web search
        search_results = self.web_search(query, max_results=max_search_results)
        formatted_results = self._format_search_results(search_results)

        # Create analysis prompt
        prompt = self._create_analysis_prompt(query, formatted_results)

        # Generate analysis
        if self.use_ollama:
            analysis = await self._generate_with_ollama(prompt)
        else:
            analysis = self.generate_response(
                prompt,
                max_new_tokens=1024,
                temperature=0.7
            )

        # Calculate confidence based on search results quality
        confidence = self._calculate_confidence(search_results, analysis)

        elapsed_time = (datetime.now() - start_time).total_seconds()

        return {
            "agent": "search",
            "agent_name": "Mistral Search Agent",
            "query": query,
            "search_results": search_results,
            "formatted_results": formatted_results,
            "analysis": analysis,
            "confidence": confidence,
            "processing_time": elapsed_time,
            "model": self.model_name if not self.use_ollama else self.ollama_model
        }

    async def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "num_predict": 1024
                        }
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error generating response: {e}"

    def _calculate_confidence(
        self,
        search_results: List[Dict],
        analysis: str
    ) -> float:
        """Calculate confidence score based on results quality"""
        confidence = 0.5  # Base confidence

        # Adjust based on number of search results
        if len(search_results) >= 5:
            confidence += 0.2
        elif len(search_results) >= 3:
            confidence += 0.1

        # Adjust based on analysis length (indicates depth)
        if len(analysis) > 500:
            confidence += 0.1
        if len(analysis) > 1000:
            confidence += 0.1

        # Check for key sections in analysis
        key_sections = ["Key Findings", "Relevant Facts", "Summary"]
        for section in key_sections:
            if section.lower() in analysis.lower():
                confidence += 0.03

        return min(confidence, 1.0)

    async def search_and_summarize(
        self,
        query: str,
        summarize_each: bool = False
    ) -> Dict[str, Any]:
        """
        Search and optionally summarize each result

        Args:
            query: Search query
            summarize_each: Whether to summarize each result individually

        Returns:
            Search results with optional summaries
        """
        results = self.web_search(query, max_results=5)

        if summarize_each and results:
            for result in results:
                summary_prompt = f"""<s>[INST] Summarize this text in 2-3 sentences:

{result['body']}
[/INST]"""
                result['summary'] = self.generate_response(
                    summary_prompt,
                    max_new_tokens=100,
                    temperature=0.5
                )

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
