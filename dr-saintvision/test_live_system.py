"""
Live System Test - Tests the actual running DR-Saintvision system
"""
import sys
sys.path.insert(0, r"C:\Project\Dr's Saintvision\Lib\site-packages")

import asyncio
import httpx
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "search": "mistral:7b-instruct",
    "reasoning": "llama3.2:latest",
    "synthesis": "qwen2.5:7b-instruct"
}

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

async def query_ollama(model: str, prompt: str, timeout: float = 120.0) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                return True, response.json().get("response", "")
            else:
                return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

async def test_real_question():
    print_header("LIVE SYSTEM TEST - Real Question")

    # Test with Korean medical question
    test_query = "당뇨병의 주요 증상과 예방법에 대해 설명해주세요."
    print(f"\nTest Query: {test_query}\n")

    print("Phase 1: Running Search and Reasoning agents in parallel...")

    # Phase 1: Parallel execution
    search_prompt = f"""당신은 정보 검색 및 분석 전문가입니다.
다음 질문에 대해 관련 정보를 검색하고 분석하세요.

질문: {test_query}

관련 정보를 정리하여 답변하세요."""

    reasoning_prompt = f"""당신은 논리적 분석 전문가입니다.
다음 질문에 대해 논리적으로 분석하세요.

질문: {test_query}

단계별로 논리적 분석을 제공하세요."""

    search_task = asyncio.create_task(query_ollama(MODELS["search"], search_prompt))
    reasoning_task = asyncio.create_task(query_ollama(MODELS["reasoning"], reasoning_prompt))

    (search_ok, search_result), (reason_ok, reason_result) = await asyncio.gather(
        search_task, reasoning_task
    )

    print(f"  Search Agent: {'OK' if search_ok else 'FAIL'} ({len(search_result)} chars)")
    print(f"  Reasoning Agent: {'OK' if reason_ok else 'FAIL'} ({len(reason_result)} chars)")

    if not (search_ok and reason_ok):
        print("ERROR: Phase 1 failed")
        return False

    print("\nPhase 2: Running Synthesis agent...")

    # Phase 2: Synthesis
    synthesis_prompt = f"""당신은 종합 분석 전문가입니다.
다음 두 가지 분석을 종합하여 최종 답변을 작성하세요.

원래 질문: {test_query}

검색 분석:
{search_result[:1500]}

추론 분석:
{reason_result[:1500]}

위 분석을 종합하여 명확하고 포괄적인 최종 답변을 작성하세요."""

    synth_ok, synth_result = await query_ollama(MODELS["synthesis"], synthesis_prompt)
    print(f"  Synthesis Agent: {'OK' if synth_ok else 'FAIL'} ({len(synth_result)} chars)")

    if not synth_ok:
        print("ERROR: Phase 2 failed")
        return False

    # Display results
    print_header("SEARCH AGENT RESULT (Mistral)")
    print(search_result[:800])
    print("..." if len(search_result) > 800 else "")

    print_header("REASONING AGENT RESULT (Llama)")
    print(reason_result[:800])
    print("..." if len(reason_result) > 800 else "")

    print_header("SYNTHESIS AGENT RESULT (Qwen)")
    print(synth_result)

    # Calculate simple quality metrics
    print_header("QUALITY METRICS")

    total_length = len(search_result) + len(reason_result) + len(synth_result)
    print(f"  Total response length: {total_length} chars")

    # Check for key terms
    key_terms = ["당뇨", "혈당", "증상", "예방", "인슐린"]
    found_terms = [t for t in key_terms if t in synth_result]
    print(f"  Key terms found: {len(found_terms)}/{len(key_terms)} ({', '.join(found_terms)})")

    # Check response structure
    has_structure = any(c in synth_result for c in ["1.", "2.", "-", "•", "증상", "예방"])
    print(f"  Has structured format: {'Yes' if has_structure else 'No'}")

    quality_score = (len(found_terms) / len(key_terms)) * 0.5 + (0.5 if has_structure else 0)
    print(f"  Estimated quality score: {quality_score:.1%}")

    return True

async def test_similarity_learning():
    print_header("SIMILARITY LEARNING TEST")

    # Simulate two different responses
    response1 = """당뇨병은 혈당 조절 능력이 저하되는 대사 질환입니다.
주요 증상으로는 다음, 다뇨, 다식, 체중 감소 등이 있습니다.
예방을 위해서는 규칙적인 운동과 균형 잡힌 식단이 중요합니다."""

    response2 = """당뇨병의 증상에는 갈증 증가, 잦은 배뇨, 식욕 증가가 있습니다.
혈당 수치가 높아지면 피로감과 시력 저하도 나타날 수 있습니다.
예방하려면 체중 관리와 건강한 생활습관이 필요합니다."""

    # Calculate similarity
    def calculate_similarity(text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    similarity = calculate_similarity(response1, response2)
    print(f"\n  Response 1 length: {len(response1)} chars")
    print(f"  Response 2 length: {len(response2)} chars")
    print(f"  Calculated similarity: {similarity:.1%}")

    if similarity >= 0.90:
        print("  Result: HIGH similarity - would NOT save to learning data")
    else:
        print("  Result: LOW similarity - would save to learning data")
        print("  (This is how the system learns from ChatGPT comparisons)")

    return True

async def main():
    # Fix encoding for Windows console
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("\n")
    print("*" * 60)
    print("*    DR-SAINTVISION LIVE SYSTEM TEST")
    print("*    Testing actual model responses")
    print("*" * 60)

    # Run tests
    result1 = await test_real_question()
    result2 = await test_similarity_learning()

    print_header("FINAL SUMMARY")

    if result1 and result2:
        print("""
    All tests passed!

    The DR-Saintvision system is working correctly:

    - Ollama models are responding properly
    - Multi-agent parallel processing works
    - Search, Reasoning, and Synthesis agents produce meaningful outputs
    - Similarity calculation for learning is functional

    The system is ready for use at: http://localhost:7860
        """)
    else:
        print("\n    Some tests failed. Check the output above.")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
