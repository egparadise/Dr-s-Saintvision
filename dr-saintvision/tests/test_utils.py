"""
Test suite for DR-Saintvision Utilities
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWebSearchUtils:
    """Tests for WebSearchUtils"""

    def test_extract_domain(self):
        """Test domain extraction from URL"""
        from utils.web_search import WebSearchUtils

        utils = WebSearchUtils()

        assert utils._extract_domain("https://www.example.com/page") == "example.com"
        assert utils._extract_domain("http://test.org/path") == "test.org"

    def test_simple_similarity(self):
        """Test simple string similarity"""
        from utils.web_search import WebSearchUtils

        utils = WebSearchUtils()

        # Identical strings
        sim = utils._simple_similarity("hello world", "hello world")
        assert sim == 1.0

        # Partially similar
        sim = utils._simple_similarity("hello world", "hello there")
        assert 0 < sim < 1

        # Completely different
        sim = utils._simple_similarity("abc", "xyz")
        assert sim == 0.0

    def test_format_results_for_prompt(self):
        """Test result formatting"""
        from utils.web_search import WebSearchUtils

        utils = WebSearchUtils()

        results = [
            {"title": "Test 1", "body": "Content 1", "href": "http://test1.com"},
            {"title": "Test 2", "body": "Content 2", "href": "http://test2.com"}
        ]

        formatted = utils.format_results_for_prompt(results)
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "Test 1" in formatted

    def test_format_empty_results(self):
        """Test formatting empty results"""
        from utils.web_search import WebSearchUtils

        utils = WebSearchUtils()
        formatted = utils.format_results_for_prompt([])
        assert "No search results" in formatted

    def test_deduplicate_results(self):
        """Test result deduplication"""
        from utils.web_search import WebSearchUtils

        utils = WebSearchUtils()

        results = [
            {"title": "Same Title", "body": "Content 1"},
            {"title": "Same Title", "body": "Content 2"},
            {"title": "Different Title", "body": "Content 3"}
        ]

        deduped = utils.deduplicate_results(results)
        assert len(deduped) == 2


class TestMetricsCalculator:
    """Tests for MetricsCalculator"""

    def test_keyword_coverage(self):
        """Test keyword coverage calculation"""
        from utils.metrics import MetricsCalculator

        calc = MetricsCalculator(use_embeddings=False)

        text = "This is a test about artificial intelligence and machine learning."
        keywords = ["artificial", "machine", "quantum"]

        result = calc.keyword_coverage(text, keywords)

        assert result["coverage"] == 2/3
        assert "artificial" in result["found"]
        assert "quantum" in result["missing"]

    def test_text_coherence(self):
        """Test text coherence scoring"""
        from utils.metrics import MetricsCalculator

        calc = MetricsCalculator(use_embeddings=False)

        # Coherent text with transitions
        coherent = """
        First, we need to understand the problem. Therefore, we analyze the data.
        Furthermore, we apply multiple methods. In conclusion, we get results.
        """
        score = calc.text_coherence(coherent)
        assert score > 0.5

        # Less coherent text
        incoherent = "Random words here."
        score2 = calc.text_coherence(incoherent)
        assert score2 < score

    def test_completeness_score(self):
        """Test completeness scoring"""
        from utils.metrics import MetricsCalculator

        calc = MetricsCalculator(use_embeddings=False)

        expected = ["introduction", "analysis", "conclusion"]
        text = "This is the introduction. Here is the analysis. Finally, the conclusion."

        result = calc.completeness_score(text, expected)
        assert result["score"] == 1.0
        assert len(result["missing_sections"]) == 0

    def test_format_metrics_report(self):
        """Test metrics report formatting"""
        from utils.metrics import MetricsCalculator

        calc = MetricsCalculator(use_embeddings=False)

        metrics = {
            "total_time": 5.5,
            "search_time": 2.0,
            "reasoning_time": 2.5,
            "synthesis_time": 1.0,
            "confidence": {
                "search": 0.8,
                "reasoning": 0.7,
                "synthesis": 0.9,
                "overall": 0.8
            },
            "answer_length": 500,
            "coherence": 0.75,
            "synthesis_completeness": 0.9
        }

        report = calc.format_metrics_report(metrics)
        assert "METRICS REPORT" in report
        assert "5.5" in report


class TestPromptTemplates:
    """Tests for PromptTemplates"""

    def test_get_template(self):
        """Test template retrieval"""
        from utils.prompts import PromptTemplates

        search = PromptTemplates.get_template("SEARCH_ANALYSIS")
        assert search is not None
        assert "{query}" in search

        reasoning = PromptTemplates.get_template("DEEP_REASONING")
        assert reasoning is not None

    def test_format_template(self):
        """Test template formatting"""
        from utils.prompts import PromptTemplates

        formatted = PromptTemplates.format_template(
            "CHAIN_OF_THOUGHT",
            query="What is AI?"
        )

        assert "What is AI?" in formatted

    def test_list_templates(self):
        """Test listing all templates"""
        from utils.prompts import PromptTemplates

        templates = PromptTemplates.list_templates()
        assert len(templates) > 0
        assert "SEARCH_ANALYSIS" in templates

    def test_format_missing_variable(self):
        """Test error on missing template variable"""
        from utils.prompts import PromptTemplates

        with pytest.raises(ValueError):
            PromptTemplates.format_template(
                "SEARCH_ANALYSIS",
                # Missing required variables
            )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
