"""
Test suite for DR-Saintvision Agents
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSearchAgent:
    """Tests for SearchAgent"""

    def test_search_agent_initialization(self):
        """Test SearchAgent can be initialized"""
        from models.search_agent import SearchAgent

        # Use mock to avoid loading actual model
        with patch.object(SearchAgent, 'load_model'):
            agent = SearchAgent(use_ollama=True)
            assert agent is not None
            assert agent.use_ollama == True

    def test_web_search_returns_list(self):
        """Test web search returns a list"""
        from models.search_agent import SearchAgent

        with patch.object(SearchAgent, 'load_model'):
            agent = SearchAgent(use_ollama=True)

            # Mock the search tool
            mock_search = Mock()
            mock_search.text = Mock(return_value=[
                {"title": "Test", "body": "Test content", "href": "http://test.com"}
            ])
            agent.search_tool = mock_search

            results = agent.web_search("test query", max_results=1)
            assert isinstance(results, list)

    def test_format_search_results(self):
        """Test search results formatting"""
        from models.search_agent import SearchAgent

        with patch.object(SearchAgent, 'load_model'):
            agent = SearchAgent(use_ollama=True)

            results = [
                {"title": "Title 1", "body": "Body 1", "href": "http://test1.com"},
                {"title": "Title 2", "body": "Body 2", "href": "http://test2.com"}
            ]

            formatted = agent._format_search_results(results)
            assert "Title 1" in formatted
            assert "Title 2" in formatted

    def test_format_empty_results(self):
        """Test formatting of empty results"""
        from models.search_agent import SearchAgent

        with patch.object(SearchAgent, 'load_model'):
            agent = SearchAgent(use_ollama=True)
            formatted = agent._format_search_results([])
            assert "No search results" in formatted


class TestReasoningAgent:
    """Tests for ReasoningAgent"""

    def test_reasoning_agent_initialization(self):
        """Test ReasoningAgent can be initialized"""
        from models.reasoning_agent import ReasoningAgent

        with patch.object(ReasoningAgent, 'load_model'):
            agent = ReasoningAgent(use_ollama=True)
            assert agent is not None

    def test_parse_reasoning_steps(self):
        """Test parsing of reasoning steps from output"""
        from models.reasoning_agent import ReasoningAgent

        with patch.object(ReasoningAgent, 'load_model'):
            agent = ReasoningAgent(use_ollama=True)

            test_output = """
## Step 1: Problem Decomposition
Breaking down the problem...

## Step 2: Assumptions
Key assumptions are...

## Step 3: Multi-perspective Analysis
Analyzing from different angles...
"""
            steps = agent._parse_reasoning_steps(test_output)
            assert len(steps) >= 1

    def test_extract_conclusion(self):
        """Test conclusion extraction"""
        from models.reasoning_agent import ReasoningAgent

        with patch.object(ReasoningAgent, 'load_model'):
            agent = ReasoningAgent(use_ollama=True)

            test_output = """
Some analysis here...

## Step 5: Synthesis and Conclusion
The final conclusion is that AI will continue to evolve.
"""
            conclusion = agent._extract_conclusion(test_output)
            assert len(conclusion) > 0


class TestSynthesisAgent:
    """Tests for SynthesisAgent"""

    def test_synthesis_agent_initialization(self):
        """Test SynthesisAgent can be initialized"""
        from models.synthesis_agent import SynthesisAgent

        with patch.object(SynthesisAgent, 'load_model'):
            agent = SynthesisAgent(use_ollama=True)
            assert agent is not None

    def test_extract_section(self):
        """Test section extraction from synthesis output"""
        from models.synthesis_agent import SynthesisAgent

        with patch.object(SynthesisAgent, 'load_model'):
            agent = SynthesisAgent(use_ollama=True)

            test_output = """
## Agreement Analysis
Both agents agree on point A and point B.

## Final Comprehensive Answer
The answer is 42.
"""
            agreement = agent._extract_section(test_output, "Agreement Analysis")
            assert "Both agents agree" in agreement

            final = agent._extract_section(test_output, "Final Comprehensive Answer")
            assert "42" in final

    def test_score_to_grade(self):
        """Test score to grade conversion"""
        from models.synthesis_agent import SynthesisAgent

        with patch.object(SynthesisAgent, 'load_model'):
            agent = SynthesisAgent(use_ollama=True)

            assert agent._score_to_grade(9.5) == "A+"
            assert agent._score_to_grade(8.5) == "A"
            assert agent._score_to_grade(7.5) == "B+"
            assert agent._score_to_grade(5.0) == "C"
            assert agent._score_to_grade(3.0) == "F"


class TestDebateManager:
    """Tests for DebateManager"""

    def test_debate_manager_initialization(self):
        """Test DebateManager can be initialized"""
        from models.debate_manager import DebateManager, DebateConfig

        with patch('models.debate_manager.SearchAgent'), \
             patch('models.debate_manager.ReasoningAgent'), \
             patch('models.debate_manager.SynthesisAgent'):

            config = DebateConfig(use_ollama=True)
            manager = DebateManager(config=config)
            assert manager is not None

    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation"""
        from models.debate_manager import DebateManager, DebateConfig

        with patch('models.debate_manager.SearchAgent'), \
             patch('models.debate_manager.ReasoningAgent'), \
             patch('models.debate_manager.SynthesisAgent'):

            manager = DebateManager(DebateConfig(use_ollama=True))

            search = {"confidence": 0.8}
            reasoning = {"confidence": 0.7}
            synthesis = {"confidence": 0.9}

            overall = manager._calculate_overall_confidence(search, reasoning, synthesis)
            assert 0 <= overall <= 1

    def test_get_statistics_empty(self):
        """Test statistics with empty history"""
        from models.debate_manager import DebateManager, DebateConfig

        with patch('models.debate_manager.SearchAgent'), \
             patch('models.debate_manager.ReasoningAgent'), \
             patch('models.debate_manager.SynthesisAgent'):

            manager = DebateManager(DebateConfig(use_ollama=True))
            stats = manager.get_statistics()

            assert stats["total_debates"] == 0
            assert stats["average_time"] == 0


class TestDebateStatus:
    """Tests for DebateStatus enum"""

    def test_debate_status_values(self):
        """Test DebateStatus enum values"""
        from models.debate_manager import DebateStatus

        assert DebateStatus.PENDING.value == "pending"
        assert DebateStatus.COMPLETED.value == "completed"
        assert DebateStatus.FAILED.value == "failed"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
