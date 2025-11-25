"""
Web Search Utilities for DR-Saintvision
Provides enhanced web search capabilities and content extraction
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class WebSearchUtils:
    """Utilities for web search and content extraction"""

    def __init__(self, max_results: int = 10, timeout: float = 30.0):
        self.max_results = max_results
        self.timeout = timeout
        self._search_engine = None
        self._init_search_engine()

    def _init_search_engine(self):
        """Initialize search engine"""
        try:
            from duckduckgo_search import DDGS
            self._search_engine = DDGS()
            logger.info("DuckDuckGo search engine initialized")
        except ImportError:
            logger.warning("duckduckgo-search not installed")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        time_filter: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Perform web search

        Args:
            query: Search query
            max_results: Maximum results to return
            region: Region code (wt-wt for worldwide)
            time_filter: d (day), w (week), m (month), y (year)

        Returns:
            List of search results
        """
        if not self._search_engine:
            logger.warning("Search engine not available")
            return []

        max_results = max_results or self.max_results

        try:
            results = []
            search_results = self._search_engine.text(
                query,
                max_results=max_results,
                region=region,
                timelimit=time_filter
            )

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", ""),
                    "source": self._extract_domain(result.get("href", ""))
                })

            logger.info(f"Search returned {len(results)} results for: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        return match.group(1) if match else url

    async def async_search(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Async version of search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query, max_results)
        )

    def news_search(
        self,
        query: str,
        max_results: int = 5,
        time_filter: str = "w"
    ) -> List[Dict[str, str]]:
        """Search for news articles"""
        if not self._search_engine:
            return []

        try:
            results = []
            news_results = self._search_engine.news(
                query,
                max_results=max_results,
                timelimit=time_filter
            )

            for result in news_results:
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "url": result.get("url", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", "")
                })

            return results

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []

    async def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and extract content from a webpage"""
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Get text content
                text = soup.get_text(separator='\n', strip=True)

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)

                return text[:5000]  # Limit content length

        except Exception as e:
            logger.error(f"Failed to fetch page content from {url}: {e}")
            return None

    def format_results_for_prompt(
        self,
        results: List[Dict[str, str]],
        include_urls: bool = True
    ) -> str:
        """Format search results for LLM prompt"""
        if not results:
            return "No search results found."

        formatted = []
        for i, result in enumerate(results, 1):
            entry = f"[{i}] {result['title']}\n{result['body']}"
            if include_urls:
                entry += f"\nSource: {result.get('href', result.get('url', 'N/A'))}"
            formatted.append(entry)

        return "\n\n".join(formatted)

    def deduplicate_results(
        self,
        results: List[Dict[str, str]],
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, str]]:
        """Remove duplicate or very similar results"""
        if not results:
            return []

        unique_results = []
        seen_titles = set()

        for result in results:
            title_lower = result['title'].lower()

            # Check for exact duplicates
            if title_lower in seen_titles:
                continue

            # Check for similar titles (simple approach)
            is_duplicate = False
            for seen in seen_titles:
                if self._simple_similarity(title_lower, seen) > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_titles.add(title_lower)

        return unique_results

    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings"""
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def search_multiple_queries(
        self,
        queries: List[str],
        max_per_query: int = 3
    ) -> List[Dict[str, str]]:
        """Search multiple queries and combine results"""
        all_results = []

        for query in queries:
            results = self.search(query, max_results=max_per_query)
            all_results.extend(results)

        # Deduplicate combined results
        return self.deduplicate_results(all_results)

    async def search_with_content(
        self,
        query: str,
        max_results: int = 3,
        fetch_content: bool = True
    ) -> List[Dict[str, Any]]:
        """Search and optionally fetch full content from top results"""
        results = self.search(query, max_results=max_results)

        if fetch_content:
            for result in results:
                if 'href' in result:
                    content = await self.fetch_page_content(result['href'])
                    result['full_content'] = content

        return results
