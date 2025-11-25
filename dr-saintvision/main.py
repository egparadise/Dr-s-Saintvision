"""
DR-Saintvision Main Entry Point
Multi-Agent AI Debate System for Enhanced Reasoning

Usage:
    python main.py                    # Start both API and Gradio
    python main.py --api-only         # Start API server only
    python main.py --gradio-only      # Start Gradio interface only
    python main.py --cli              # Interactive CLI mode
    python main.py --query "question" # Single query mode
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    banner = """
    ================================================================
    |                                                              |
    |              DR-SAINTVISION                                  |
    |         Multi-Agent AI Debate System                         |
    |                                                              |
    ================================================================

    Models:
    - Search Agent: Mistral-7B-Instruct (Web Search & RAG)
    - Reasoning Agent: Llama-3.2 (Deep Reasoning)
    - Synthesis Agent: Qwen2.5-7B (Final Synthesis)

    """
    print(banner)


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start FastAPI server"""
    import uvicorn
    from backend.api import app

    logger.info(f"Starting API server at http://{host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)


def start_gradio_server(share: bool = False, port: int = 7860):
    """Start Gradio interface"""
    from frontend.app import create_interface

    logger.info(f"Starting Gradio interface at http://localhost:{port}")

    app = create_interface()
    app.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )


def start_both_servers():
    """Start both API and Gradio servers"""
    import threading

    print_banner()
    logger.info("Starting DR-Saintvision servers...")

    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=start_api_server,
        kwargs={"port": 8000},
        daemon=True
    )
    api_thread.start()

    # Start Gradio in main thread
    start_gradio_server(port=7860)


async def interactive_cli():
    """Interactive CLI mode"""
    from models.debate_manager import DebateManager, DebateConfig

    print_banner()
    print("\nInteractive CLI Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("-" * 50)

    config = DebateConfig(use_ollama=False)
    manager = DebateManager(config=config)

    while True:
        try:
            query = input("\n[Query] > ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if query.lower() == 'help':
                print("""
Commands:
  quit/exit/q  - Exit the program
  help         - Show this help message
  stats        - Show debate statistics
  history      - Show recent debate history
  ollama       - Toggle Ollama mode

Enter any question to start a debate.
                """)
                continue

            if query.lower() == 'stats':
                stats = manager.get_statistics()
                print(f"\nStatistics:")
                print(f"  Total debates: {stats['total_debates']}")
                print(f"  Average time: {stats['average_time']:.2f}s")
                print(f"  Average confidence: {stats['average_confidence']:.1%}")
                continue

            if query.lower() == 'history':
                history = manager.get_debate_history(limit=5)
                if not history:
                    print("No debate history yet.")
                else:
                    print("\nRecent debates:")
                    for i, h in enumerate(history, 1):
                        print(f"  {i}. {h.query[:50]}... ({h.status.value})")
                continue

            if query.lower() == 'ollama':
                current = config.use_ollama
                config.use_ollama = not current
                manager = DebateManager(config=config)
                print(f"Ollama mode: {'ON' if config.use_ollama else 'OFF'}")
                continue

            # Conduct debate
            print(f"\nAnalyzing: {query}")
            print("Please wait...")

            result = await manager.conduct_debate(query)

            # Print results
            print("\n" + "=" * 60)
            print("DEBATE RESULT")
            print("=" * 60)

            print(f"\nStatus: {result.status.value}")
            print(f"Time: {result.debate_time:.2f} seconds")

            print("\n--- Final Answer ---")
            final_answer = result.final_synthesis.get('final_answer', 'No answer generated')
            print(final_answer[:1000])
            if len(final_answer) > 1000:
                print("... [truncated]")

            print("\n--- Confidence Scores ---")
            for agent, score in result.confidence_scores.items():
                print(f"  {agent}: {score:.1%}")

            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


async def single_query(query: str, use_ollama: bool = False):
    """Run a single query and print results"""
    from models.debate_manager import DebateManager, DebateConfig

    print_banner()
    print(f"\nQuery: {query}")
    print("Processing...")

    config = DebateConfig(use_ollama=use_ollama)
    manager = DebateManager(config=config)

    result = await manager.conduct_debate(query)

    print("\n" + manager.format_result_for_display(result))

    return result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DR-Saintvision: Multi-Agent AI Debate System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Start full application
  python main.py --api-only               Start API server only
  python main.py --gradio-only            Start Gradio UI only
  python main.py --cli                    Interactive CLI mode
  python main.py --query "What is AI?"    Single query
  python main.py --query "..." --ollama   Use Ollama for inference
        """
    )

    parser.add_argument(
        '--api-only',
        action='store_true',
        help='Start API server only'
    )

    parser.add_argument(
        '--gradio-only',
        action='store_true',
        help='Start Gradio interface only'
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Interactive CLI mode'
    )

    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to analyze'
    )

    parser.add_argument(
        '--ollama',
        action='store_true',
        help='Use Ollama for local model inference'
    )

    parser.add_argument(
        '--api-port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )

    parser.add_argument(
        '--gradio-port',
        type=int,
        default=7860,
        help='Gradio server port (default: 7860)'
    )

    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio link'
    )

    args = parser.parse_args()

    try:
        if args.query:
            # Single query mode
            asyncio.run(single_query(args.query, args.ollama))

        elif args.cli:
            # Interactive CLI mode
            asyncio.run(interactive_cli())

        elif args.api_only:
            # API server only
            print_banner()
            start_api_server(port=args.api_port)

        elif args.gradio_only:
            # Gradio only
            print_banner()
            start_gradio_server(share=args.share, port=args.gradio_port)

        else:
            # Start both servers (default)
            start_both_servers()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
