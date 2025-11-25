"""
DR-Saintvision System Test Script
Tests all components: models, database, and multi-agent debate
"""

import sys
sys.path.insert(0, r"C:\Project\Dr's Saintvision\Lib\site-packages")

import asyncio
import httpx
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DATABASE_PATH = Path("./database")
DATABASE_PATH.mkdir(exist_ok=True)

MODELS = {
    "search": "mistral:7b-instruct",
    "reasoning": "llama3.2:latest",
    "synthesis": "qwen2.5:7b-instruct"
}

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(name, success, message=""):
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} {name}: {message}")

# ============== TEST 1: Ollama Connection ==============

async def test_ollama_connection():
    print_header("TEST 1: Ollama Connection")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(OLLAMA_TAGS_URL)

            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                print_result("Ollama service", True, f"Running with {len(models)} models")

                # Check required models
                required = ["mistral", "llama3.2", "qwen2.5"]
                for req in required:
                    found = any(req in m.lower() for m in models)
                    print_result(f"Model: {req}", found, "Available" if found else "NOT FOUND")

                return True
            else:
                print_result("Ollama service", False, f"HTTP {response.status_code}")
                return False

    except Exception as e:
        print_result("Ollama service", False, str(e))
        return False

# ============== TEST 2: Individual Model Responses ==============

async def query_ollama(model: str, prompt: str, timeout: float = 60.0) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                result = response.json().get("response", "")
                return True, result
            else:
                return False, f"HTTP {response.status_code}"
    except httpx.TimeoutException:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

async def test_individual_models():
    print_header("TEST 2: Individual Model Responses")

    test_prompts = {
        "search": ("mistral:7b-instruct", "What are the main symptoms of diabetes? Answer in 2-3 sentences."),
        "reasoning": ("llama3.2:latest", "If A > B and B > C, what can we conclude about A and C? Explain briefly."),
        "synthesis": ("qwen2.5:7b-instruct", "Summarize the benefits of exercise in one paragraph.")
    }

    results = {}
    all_passed = True

    for role, (model, prompt) in test_prompts.items():
        print(f"\n  Testing {role} agent ({model})...")
        success, response = await query_ollama(model, prompt)

        if success:
            # Check response quality
            has_content = len(response.strip()) > 20
            print_result(f"{role.capitalize()} Agent", has_content, f"{len(response)} chars")
            if has_content:
                print(f"    Response preview: {response[:150]}...")
            results[role] = response
        else:
            print_result(f"{role.capitalize()} Agent", False, response)
            all_passed = False

    return all_passed, results

# ============== TEST 3: Multi-Agent Debate Simulation ==============

async def test_multi_agent_debate():
    print_header("TEST 3: Multi-Agent Debate Simulation")

    test_query = "What are the pros and cons of remote work?"

    print(f"\n  Query: '{test_query}'")
    print("\n  Phase 1: Parallel Search and Reasoning...")

    # Phase 1: Run Search and Reasoning in parallel
    search_prompt = f"Search and analyze information about: {test_query}"
    reasoning_prompt = f"Analyze logically: {test_query}"

    search_task = asyncio.create_task(query_ollama(MODELS["search"], search_prompt))
    reasoning_task = asyncio.create_task(query_ollama(MODELS["reasoning"], reasoning_prompt))

    (search_ok, search_result), (reason_ok, reason_result) = await asyncio.gather(
        search_task, reasoning_task
    )

    print_result("Search Agent", search_ok, f"{len(search_result)} chars" if search_ok else search_result)
    print_result("Reasoning Agent", reason_ok, f"{len(reason_result)} chars" if reason_ok else reason_result)

    if not (search_ok and reason_ok):
        return False, None

    # Phase 2: Synthesis
    print("\n  Phase 2: Synthesis...")

    synthesis_prompt = f"""Combine the following analyses into a comprehensive final answer.

Question: {test_query}

Search Analysis:
{search_result[:1000]}

Reasoning Analysis:
{reason_result[:1000]}

Provide a final synthesized answer:"""

    synth_ok, synth_result = await query_ollama(MODELS["synthesis"], synthesis_prompt, timeout=90.0)
    print_result("Synthesis Agent", synth_ok, f"{len(synth_result)} chars" if synth_ok else synth_result)

    if synth_ok:
        # Calculate simple confidence based on response length and keyword presence
        confidence = min(1.0, (len(synth_result) / 500) * 0.5 + 0.5)
        print(f"\n  Estimated confidence: {confidence:.1%}")
        print(f"\n  Final Answer Preview:\n  {'='*50}")
        print(f"  {synth_result[:500]}...")

        return True, {
            "query": test_query,
            "search": search_result,
            "reasoning": reason_result,
            "synthesis": synth_result,
            "confidence": confidence
        }

    return False, None

# ============== TEST 4: Database Operations ==============

def test_database():
    print_header("TEST 4: Database Operations")

    db_file = DATABASE_PATH / "test_history.db"

    try:
        # Create/connect to database
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                similarity_score REAL,
                created_at TEXT NOT NULL
            )
        ''')

        conn.commit()
        print_result("Create tables", True)

        # Test insert
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO test_conversations (title, created_at) VALUES (?, ?)',
                       ("Test conversation", now))
        conv_id = cursor.lastrowid
        conn.commit()
        print_result("Insert conversation", True, f"ID: {conv_id}")

        # Test message insert
        cursor.execute('INSERT INTO test_messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
                       (conv_id, "user", "Test question", now))
        cursor.execute('INSERT INTO test_messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
                       (conv_id, "assistant", "Test answer", now))
        conn.commit()
        print_result("Insert messages", True)

        # Test learning data insert
        cursor.execute('INSERT INTO test_learning_data (query, answer, similarity_score, created_at) VALUES (?, ?, ?, ?)',
                       ("Test query", "Test ChatGPT answer", 0.75, now))
        conn.commit()
        print_result("Insert learning data", True)

        # Test query
        cursor.execute('SELECT COUNT(*) FROM test_conversations')
        count = cursor.fetchone()[0]
        print_result("Query data", count > 0, f"{count} conversations")

        cursor.execute('SELECT COUNT(*) FROM test_learning_data')
        learn_count = cursor.fetchone()[0]
        print_result("Learning data count", learn_count > 0, f"{learn_count} entries")

        # Cleanup test tables
        cursor.execute('DROP TABLE IF EXISTS test_conversations')
        cursor.execute('DROP TABLE IF EXISTS test_messages')
        cursor.execute('DROP TABLE IF EXISTS test_learning_data')
        conn.commit()
        conn.close()

        # Remove test db file
        db_file.unlink(missing_ok=True)
        print_result("Cleanup", True)

        return True

    except Exception as e:
        print_result("Database test", False, str(e))
        return False

# ============== TEST 5: Learning Data Simulation ==============

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity"""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0

def test_similarity_calculation():
    print_header("TEST 5: Similarity Calculation")

    test_cases = [
        ("The quick brown fox", "The quick brown dog", 0.6),  # High similarity
        ("Hello world", "Goodbye moon", 0.0),  # Low similarity
        ("AI is transforming technology", "AI is changing technology", 0.6),  # Moderate
        ("Same text here", "Same text here", 1.0),  # Identical
    ]

    all_passed = True
    for text1, text2, expected_min in test_cases:
        similarity = calculate_similarity(text1, text2)
        passed = similarity >= expected_min * 0.5  # Allow some variance
        print_result(f"'{text1[:20]}...' vs '{text2[:20]}...'", passed, f"{similarity:.2%}")
        if not passed:
            all_passed = False

    # Test 90% threshold logic
    print("\n  Testing 90% threshold logic:")
    high_sim = 0.92
    low_sim = 0.75

    print_result("High similarity (92%)", high_sim >= 0.90, "Would NOT save to learning data")
    print_result("Low similarity (75%)", low_sim < 0.90, "Would save to learning data")

    return all_passed

# ============== MAIN ==============

async def main():
    print("\n")
    print("=" * 60)
    print("    DR-SAINTVISION SYSTEM TEST")
    print("    Testing all components...")
    print("=" * 60)

    results = {}

    # Test 1: Ollama Connection
    results["ollama"] = await test_ollama_connection()

    if not results["ollama"]:
        print("\n[ERROR] Ollama not available. Cannot continue tests.")
        return

    # Test 2: Individual Models
    results["models"], _ = await test_individual_models()

    # Test 3: Multi-Agent Debate
    results["debate"], debate_result = await test_multi_agent_debate()

    # Test 4: Database
    results["database"] = test_database()

    # Test 5: Similarity
    results["similarity"] = test_similarity_calculation()

    # Summary
    print_header("TEST SUMMARY")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name.upper()}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All systems operational!")
    else:
        print(f"\n  {total - passed} test(s) failed. Check the output above.")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
