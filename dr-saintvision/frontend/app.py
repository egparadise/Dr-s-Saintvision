"""
Gradio Frontend for DR-Saintvision
Interactive web interface for the multi-agent debate system
"""

import gradio as gr
import asyncio
import json
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import debate manager
from models.debate_manager import DebateManager, DebateConfig, DebateStatus

# Global debate manager
debate_manager: Optional[DebateManager] = None


def get_manager(use_ollama: bool = False) -> DebateManager:
    """Get or create debate manager"""
    global debate_manager

    config = DebateConfig(use_ollama=use_ollama)

    if debate_manager is None:
        debate_manager = DebateManager(config=config)

    return debate_manager


async def process_query_async(
    query: str,
    use_ollama: bool = False
) -> tuple[str, str, str, str, str]:
    """Process query asynchronously and return formatted results"""
    if not query.strip():
        return ("Please enter a query.", "", "", "", "")

    manager = get_manager(use_ollama)

    try:
        result = await manager.conduct_debate(query)

        if result.status != DebateStatus.COMPLETED:
            return (
                f"Debate failed: {result.error}",
                "", "", "", ""
            )

        # Format search analysis
        search_output = format_search_result(result.search_analysis)

        # Format reasoning analysis
        reasoning_output = format_reasoning_result(result.reasoning_analysis)

        # Format synthesis
        synthesis_output = format_synthesis_result(result.final_synthesis)

        # Format summary
        summary_output = format_summary(result)

        # Format confidence
        confidence_output = format_confidence(result.confidence_scores)

        return (
            synthesis_output,
            search_output,
            reasoning_output,
            summary_output,
            confidence_output
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return (f"Error: {str(e)}", "", "", "", "")


def process_query(
    query: str,
    use_ollama: bool = False
) -> tuple[str, str, str, str, str]:
    """Synchronous wrapper for async query processing"""
    return asyncio.run(process_query_async(query, use_ollama))


def format_search_result(result: dict) -> str:
    """Format search agent result for display"""
    if not result:
        return "No search results available."

    output = []
    output.append("### Search Agent Analysis (Mistral)")
    output.append(f"**Confidence:** {result.get('confidence', 0):.1%}")
    output.append(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")
    output.append("")

    # Search results
    search_results = result.get('search_results', [])
    if search_results:
        output.append("#### Web Search Results")
        for i, sr in enumerate(search_results[:5], 1):
            output.append(f"**{i}. {sr.get('title', 'N/A')}**")
            output.append(f"   {sr.get('body', '')[:200]}...")
            output.append("")

    # Analysis
    analysis = result.get('analysis', '')
    if analysis:
        output.append("#### Analysis")
        output.append(analysis)

    return "\n".join(output)


def format_reasoning_result(result: dict) -> str:
    """Format reasoning agent result for display"""
    if not result:
        return "No reasoning results available."

    output = []
    output.append("### Reasoning Agent Analysis (Llama)")
    output.append(f"**Confidence:** {result.get('confidence', 0):.1%}")
    output.append(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")
    output.append("")

    # Reasoning steps
    steps = result.get('reasoning_steps', [])
    if steps:
        output.append("#### Reasoning Steps")
        for step in steps:
            output.append(f"**{step.get('step', 'Step')}**")
            output.append(step.get('content', '')[:500])
            output.append("")

    # Conclusion
    conclusion = result.get('conclusion', '')
    if conclusion:
        output.append("#### Conclusion")
        output.append(conclusion)

    return "\n".join(output)


def format_synthesis_result(result: dict) -> str:
    """Format synthesis agent result for display"""
    if not result:
        return "No synthesis available."

    output = []
    output.append("## Final Answer (Synthesis Agent - Qwen)")
    output.append(f"**Overall Confidence:** {result.get('confidence', 0):.1%}")
    output.append("")

    # Final answer
    final_answer = result.get('final_answer', '')
    if final_answer:
        output.append("### Answer")
        output.append(final_answer)
        output.append("")

    # Agreements
    agreements = result.get('agreements', '')
    if agreements:
        output.append("### Agreements Between Agents")
        output.append(agreements)
        output.append("")

    # Disagreements
    disagreements = result.get('disagreements', '')
    if disagreements:
        output.append("### Points of Disagreement")
        output.append(disagreements)
        output.append("")

    # Limitations
    limitations = result.get('limitations', '')
    if limitations:
        output.append("### Limitations")
        output.append(limitations)

    return "\n".join(output)


def format_summary(result) -> str:
    """Format debate summary"""
    output = []
    output.append("## Debate Summary")
    output.append(f"**Query:** {result.query}")
    output.append(f"**Total Time:** {result.debate_time:.2f} seconds")
    output.append(f"**Status:** {result.status.value}")
    output.append(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(output)


def format_confidence(scores: dict) -> str:
    """Format confidence scores"""
    if not scores:
        return "No confidence data."

    output = []
    output.append("## Confidence Scores")
    output.append("")
    output.append("| Agent | Confidence |")
    output.append("|-------|------------|")
    output.append(f"| Search (Mistral) | {scores.get('search', 0):.1%} |")
    output.append(f"| Reasoning (Llama) | {scores.get('reasoning', 0):.1%} |")
    output.append(f"| Synthesis (Qwen) | {scores.get('synthesis', 0):.1%} |")
    output.append(f"| **Overall** | **{scores.get('overall', 0):.1%}** |")

    return "\n".join(output)


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface"""

    with gr.Blocks() as app:

        # Header
        gr.Markdown("""
        # DR-Saintvision: AI Multi-Agent Debate System

        This system uses three AI models working together to provide comprehensive, well-reasoned answers:

        | Agent | Model | Role |
        |-------|-------|------|
        | **Search Agent** | Mistral-7B | Web search and information retrieval |
        | **Reasoning Agent** | Llama-3.2-7B | Deep logical reasoning and analysis |
        | **Synthesis Agent** | Qwen2.5-7B | Final synthesis and judgment |

        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What would you like to analyze? (e.g., 'What are the implications of AI in healthcare?')",
                    lines=3,
                    max_lines=5
                )

                with gr.Row():
                    use_ollama = gr.Checkbox(
                        label="Use Ollama (local models)",
                        value=False,
                        info="Check if you have Ollama installed with the required models"
                    )

                submit_btn = gr.Button(
                    "Analyze",
                    variant="primary",
                    size="lg"
                )

                # Example queries
                gr.Examples(
                    examples=[
                        ["What are the potential risks and benefits of artificial general intelligence?"],
                        ["How might climate change affect global food security in the next 50 years?"],
                        ["Can quantum computers break current encryption methods?"],
                        ["What is the relationship between gut microbiome and mental health?"],
                        ["How will autonomous vehicles transform urban planning?"]
                    ],
                    inputs=query_input,
                    label="Example Questions"
                )

        # Output section
        with gr.Tabs():
            with gr.TabItem("Final Answer"):
                synthesis_output = gr.Markdown(label="Synthesized Answer")

            with gr.TabItem("Search Analysis"):
                search_output = gr.Markdown(label="Search Agent Results")

            with gr.TabItem("Reasoning Analysis"):
                reasoning_output = gr.Markdown(label="Reasoning Agent Results")

            with gr.TabItem("Summary"):
                summary_output = gr.Markdown(label="Debate Summary")

            with gr.TabItem("Confidence"):
                confidence_output = gr.Markdown(label="Confidence Scores")

        # Footer
        gr.Markdown("""
        ---
        ### How It Works

        1. **Search Phase**: The Search Agent queries the web and analyzes relevant information
        2. **Reasoning Phase**: The Reasoning Agent applies logical analysis to the query
        3. **Synthesis Phase**: The Synthesis Agent combines both analyses into a final answer

        All three phases can run in parallel for faster results. The system provides confidence scores
        to help you assess the reliability of the answer.

        ---
        *DR-Saintvision v1.0 - Multi-Agent AI Debate System*
        """)

        # Event handler
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, use_ollama],
            outputs=[
                synthesis_output,
                search_output,
                reasoning_output,
                summary_output,
                confidence_output
            ]
        )

        # Also trigger on Enter key
        query_input.submit(
            fn=process_query,
            inputs=[query_input, use_ollama],
            outputs=[
                synthesis_output,
                search_output,
                reasoning_output,
                summary_output,
                confidence_output
            ]
        )

    return app


# Main entry point
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
