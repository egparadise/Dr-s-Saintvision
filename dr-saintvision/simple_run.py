"""
DR-Saintvision v2.3
- ChatGPT-style UI with Agent Status Animation
- API key status indicator
- 90% similarity comparison logic
- Auto-save learning data for <90% similarity
"""

import sys
sys.path.insert(0, r"C:\Project\Dr's Saintvision\Lib\site-packages")

import gradio as gr
import asyncio
import httpx
from datetime import datetime
import logging
import base64
import os
import json
import sqlite3
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Models
MODELS = {
    "search": "mistral:7b-instruct",
    "reasoning": "llama3.2:latest",
    "synthesis": "qwen2.5:7b-instruct",
    "vision": "llava:7b",
    "chatgpt": "gpt-4o"
}

# Trained model names
TRAINED_MODELS = {
    "search": "dr-saintvision-search",
    "reasoning": "dr-saintvision-reasoning",
    "synthesis": "dr-saintvision-synthesis"
}

# Paths
DATABASE_PATH = Path("./database")
DATABASE_PATH.mkdir(exist_ok=True)
LEARNING_DATA_PATH = Path("./learning_data")
LEARNING_DATA_PATH.mkdir(exist_ok=True)
KNOWLEDGE_BASE_PATH = Path("./knowledge_base")
KNOWLEDGE_BASE_PATH.mkdir(exist_ok=True)
MODELFILE_PATH = Path("./modelfiles")
MODELFILE_PATH.mkdir(exist_ok=True)
CONFIG_FILE = Path("./config.json")

SUPPORTED_FILES = {
    "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"],
    "pdf": [".pdf"],
    "excel": [".xlsx", ".xls", ".csv"],
    "text": [".txt", ".md", ".json"]
}


# ============== DATABASE ==============

def init_database():
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')

    # Learning data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            chatgpt_answer TEXT NOT NULL,
            local_answer TEXT,
            similarity_score REAL,
            learned INTEGER DEFAULT 0,
            learned_models TEXT,
            created_at TEXT NOT NULL,
            learned_at TEXT
        )
    ''')

    conn.commit()
    conn.close()


def create_conversation(title: str) -> int:
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)',
                   (title, now, now))

    conv_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conv_id


def save_message(conv_id: int, role: str, content: str):
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
                   (conv_id, role, content, now))
    cursor.execute('UPDATE conversations SET updated_at = ? WHERE id = ?', (now, conv_id))

    conn.commit()
    conn.close()


def get_conversations(limit: int = 20):
    db_file = DATABASE_PATH / "history.db"
    if not db_file.exists():
        return []

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?', (limit,))
    results = cursor.fetchall()
    conn.close()
    return results


def get_conversation_messages(conv_id: int):
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute('SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC', (conv_id,))
    results = cursor.fetchall()
    conn.close()
    return results


# ============== LEARNING DATA ==============

def save_learning_data(query: str, chatgpt_answer: str, local_answer: str, similarity_score: float):
    """Save learning data to database"""
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO learning_data (query, chatgpt_answer, local_answer, similarity_score, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (query, chatgpt_answer, local_answer, similarity_score, now))

    conn.commit()
    conn.close()
    logger.info(f"Learning data saved: similarity={similarity_score:.2%}")


def get_learning_data(learned_filter: str = "all", limit: int = 50):
    """Get learning data from database"""
    db_file = DATABASE_PATH / "history.db"
    if not db_file.exists():
        return []

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    if learned_filter == "pending":
        cursor.execute('SELECT id, query, chatgpt_answer, local_answer, similarity_score, learned, created_at FROM learning_data WHERE learned = 0 ORDER BY created_at DESC LIMIT ?', (limit,))
    elif learned_filter == "completed":
        cursor.execute('SELECT id, query, chatgpt_answer, local_answer, similarity_score, learned, created_at FROM learning_data WHERE learned = 1 ORDER BY created_at DESC LIMIT ?', (limit,))
    else:
        cursor.execute('SELECT id, query, chatgpt_answer, local_answer, similarity_score, learned, created_at FROM learning_data ORDER BY created_at DESC LIMIT ?', (limit,))

    results = cursor.fetchall()
    conn.close()
    return results


def get_learning_data_by_id(data_id: int):
    """Get specific learning data"""
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM learning_data WHERE id = ?', (data_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def mark_as_learned(data_id: int, models: list):
    """Mark learning data as learned"""
    db_file = DATABASE_PATH / "history.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('UPDATE learning_data SET learned = 1, learned_models = ?, learned_at = ? WHERE id = ?',
                   (','.join(models), now, data_id))

    conn.commit()
    conn.close()


def get_learning_stats():
    """Get learning statistics"""
    db_file = DATABASE_PATH / "history.db"
    if not db_file.exists():
        return {"total": 0, "pending": 0, "completed": 0}

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM learning_data')
    total = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM learning_data WHERE learned = 0')
    pending = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM learning_data WHERE learned = 1')
    completed = cursor.fetchone()[0]

    conn.close()
    return {"total": total, "pending": pending, "completed": completed}


init_database()


# ============== CONFIG ==============

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"openai_api_key": "", "use_trained_models": False}


def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def get_api_key():
    return load_config().get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")


def save_api_key(api_key: str):
    config = load_config()
    config["openai_api_key"] = api_key
    save_config(config)
    return "✓ API 키가 저장되었습니다."


def get_use_trained_models():
    return load_config().get("use_trained_models", False)


def set_use_trained_models(use: bool):
    config = load_config()
    config["use_trained_models"] = use
    save_config(config)
    return f"✓ 학습된 모델 사용: {'ON' if use else 'OFF'}"


# ============== API & MODEL STATUS ==============

async def check_openai_api_status() -> dict:
    """Check if OpenAI API key is valid"""
    api_key = get_api_key()
    if not api_key:
        return {"status": "no_key", "message": "API 키 없음"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if response.status_code == 200:
                return {"status": "active", "message": "활성화됨"}
            elif response.status_code == 401:
                return {"status": "invalid", "message": "유효하지 않음"}
            else:
                return {"status": "error", "message": f"오류 ({response.status_code})"}
    except Exception as e:
        return {"status": "error", "message": f"연결 실패"}


def check_openai_api_status_sync() -> dict:
    """Sync wrapper for API check"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(check_openai_api_status())
        loop.close()
        return result
    except:
        return {"status": "error", "message": "확인 실패"}


async def check_ollama_status() -> dict:
    """Check Ollama service and available models"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(OLLAMA_TAGS_URL)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"status": "running", "models": models}
            else:
                return {"status": "error", "models": []}
    except:
        return {"status": "offline", "models": []}


def check_ollama_status_sync() -> dict:
    """Sync wrapper for Ollama check"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(check_ollama_status())
        loop.close()
        return result
    except:
        return {"status": "offline", "models": []}


def get_system_status_html():
    """Generate system status HTML"""
    openai_status = check_openai_api_status_sync()
    ollama_status = check_ollama_status_sync()

    # OpenAI status styling
    if openai_status["status"] == "active":
        openai_color = "#27ae60"
        openai_icon = "✓"
    elif openai_status["status"] == "no_key":
        openai_color = "#f39c12"
        openai_icon = "⚠"
    else:
        openai_color = "#e74c3c"
        openai_icon = "✗"

    # Ollama status styling
    if ollama_status["status"] == "running":
        ollama_color = "#27ae60"
        ollama_icon = "✓"
        model_count = len(ollama_status["models"])
    else:
        ollama_color = "#e74c3c"
        ollama_icon = "✗"
        model_count = 0

    # Check required models
    required_models = ["mistral", "llama3.2", "qwen2.5"]
    available = ollama_status.get("models", [])
    model_status = []
    for req in required_models:
        found = any(req in m.lower() for m in available)
        model_status.append(f"{'✓' if found else '✗'} {req}")

    html = f"""
    <div style="background: #f8f9fa; padding: 12px; border-radius: 10px; margin-bottom: 10px;">
        <h4 style="margin: 0 0 10px 0; font-size: 14px;">🔌 시스템 상태</h4>
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            <div style="background: white; padding: 8px 12px; border-radius: 8px; border-left: 4px solid {openai_color};">
                <span style="font-weight: 600;">ChatGPT API</span>
                <span style="color: {openai_color}; margin-left: 8px;">{openai_icon} {openai_status["message"]}</span>
            </div>
            <div style="background: white; padding: 8px 12px; border-radius: 8px; border-left: 4px solid {ollama_color};">
                <span style="font-weight: 600;">Ollama</span>
                <span style="color: {ollama_color}; margin-left: 8px;">{ollama_icon} {ollama_status["status"]}</span>
                <span style="color: #666; margin-left: 8px; font-size: 12px;">({model_count}개 모델)</span>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 11px; color: #666;">
            모델: {' | '.join(model_status)}
        </div>
    </div>
    """
    return html


# ============== KNOWLEDGE BASE (RAG) ==============

def load_knowledge_base():
    kb_file = KNOWLEDGE_BASE_PATH / "knowledge.json"
    if kb_file.exists():
        with open(kb_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_knowledge_base(knowledge: list):
    kb_file = KNOWLEDGE_BASE_PATH / "knowledge.json"
    with open(kb_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)


def add_to_knowledge_base(query: str, answer: str):
    """Add to knowledge base for RAG"""
    knowledge = load_knowledge_base()
    knowledge.append({
        "query": query,
        "answer": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_knowledge_base(knowledge)
    return len(knowledge)


def find_relevant_knowledge(query: str, top_k: int = 3) -> str:
    """Find relevant knowledge for RAG context"""
    knowledge = load_knowledge_base()
    if not knowledge:
        return ""

    query_words = set(query.lower().split())
    scored = []

    for entry in knowledge:
        entry_words = set(entry["query"].lower().split())
        if query_words and entry_words:
            intersection = query_words.intersection(entry_words)
            score = len(intersection) / len(query_words) if query_words else 0
            if score > 0.2:
                scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return ""

    context = "\n\n[학습된 지식 참고]\n"
    for _, entry in scored[:top_k]:
        context += f"Q: {entry['query']}\nA: {entry['answer'][:500]}\n\n"

    return context


# ============== MODEL TRAINING ==============

def create_modelfile(model_type: str, base_model: str, training_data: list) -> str:
    """Create Ollama Modelfile with training data"""

    # Build training examples
    examples = ""
    for data in training_data[:10]:  # Limit to 10 examples
        examples += f"""
### 질문:
{data['query']}

### 답변:
{data['answer'][:1000]}

"""

    modelfile_content = f"""FROM {base_model}

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM \"\"\"당신은 DR-Saintvision의 {model_type} 에이전트입니다.
다음 학습 데이터를 참고하여 답변하세요:

{examples}

위 예시를 참고하여 정확하고 상세하게 한글로 답변하세요.
\"\"\"
"""

    # Save modelfile
    modelfile_path = MODELFILE_PATH / f"Modelfile.{model_type}"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    return str(modelfile_path)


def train_model(model_type: str, data_ids: list) -> str:
    """Train a model using Ollama create command"""
    try:
        # Get training data
        training_data = []
        for data_id in data_ids:
            data = get_learning_data_by_id(data_id)
            if data:
                training_data.append({
                    "query": data[1],
                    "answer": data[2]
                })

        if not training_data:
            return "❌ 학습 데이터가 없습니다."

        # Get base model
        base_models = {
            "search": MODELS["search"],
            "reasoning": MODELS["reasoning"],
            "synthesis": MODELS["synthesis"]
        }
        base_model = base_models.get(model_type, MODELS["search"])

        # Create modelfile
        modelfile_path = create_modelfile(model_type, base_model, training_data)

        # Run ollama create
        trained_model_name = TRAINED_MODELS[model_type]
        result = subprocess.run(
            ["ollama", "create", trained_model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            # Mark data as learned
            for data_id in data_ids:
                mark_as_learned(data_id, [model_type])

            # Also add to knowledge base for RAG
            for data in training_data:
                add_to_knowledge_base(data["query"], data["answer"])

            return f"✅ {trained_model_name} 모델 학습 완료!\n\n학습된 데이터: {len(training_data)}개"
        else:
            return f"❌ 학습 실패: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "❌ 학습 시간 초과 (5분)"
    except Exception as e:
        return f"❌ 오류: {str(e)}"


def train_all_models(data_ids: list) -> str:
    """Train all three models"""
    results = []

    for model_type in ["search", "reasoning", "synthesis"]:
        result = train_model(model_type, data_ids)
        results.append(f"**{model_type}:** {result}")

    return "\n\n".join(results)


# ============== FILE PROCESSING ==============

def extract_text_from_file(file_path: str) -> tuple[str, str]:
    if not file_path:
        return "", ""

    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name

    try:
        if file_ext in SUPPORTED_FILES["image"]:
            return "", f"[이미지: {file_name}]"
        elif file_ext in SUPPORTED_FILES["pdf"]:
            try:
                import fitz
                doc = fitz.open(file_path)
                text = "".join([page.get_text() for page in doc])
                doc.close()
                return text[:5000], f"[PDF: {file_name}]"
            except:
                return "", "[PDF 읽기 실패]"
        elif file_ext in SUPPORTED_FILES["text"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()[:5000], f"[Text: {file_name}]"
        return "", f"[지원 안됨: {file_ext}]"
    except Exception as e:
        return "", f"[오류: {str(e)}]"


def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============== AI FUNCTIONS ==============

# Cache for available models
_available_models_cache = None
_cache_time = None

def get_available_ollama_models() -> list:
    """Get list of available Ollama models (cached)"""
    global _available_models_cache, _cache_time
    import time

    # Cache for 60 seconds
    if _available_models_cache is not None and _cache_time and (time.time() - _cache_time) < 60:
        return _available_models_cache

    try:
        import requests
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            _available_models_cache = [m["name"] for m in data.get("models", [])]
            _cache_time = time.time()
            return _available_models_cache
    except:
        pass

    return []


def get_active_model(model_type: str) -> str:
    """Get active model (trained or base), with fallback to base model"""
    base_model = MODELS.get(model_type)

    # Ensure we always have a valid base model
    if not base_model:
        logger.warning(f"Unknown model type: {model_type}, defaulting to mistral")
        base_model = MODELS.get("search", "mistral:7b-instruct")

    if get_use_trained_models():
        trained_model = TRAINED_MODELS.get(model_type)
        if trained_model:
            # Check if trained model actually exists
            available = get_available_ollama_models()
            if any(trained_model in m for m in available):
                return trained_model
            # Fallback to base model if trained model doesn't exist
            logger.info(f"Trained model {trained_model} not found, using base model {base_model}")

    return base_model


async def query_ollama(model: str, prompt: str, timeout: float = 120.0) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False})
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"[Ollama 오류: {response.status_code}]"
    except httpx.TimeoutException:
        return "[Ollama 시간 초과]"
    except httpx.ConnectError:
        return "[Ollama 연결 실패 - 서비스가 실행 중인지 확인하세요]"
    except Exception as e:
        return f"[Ollama 오류: {str(e)}]"


async def query_ollama_vision(model: str, prompt: str, image_path: str) -> str:
    try:
        image_base64 = encode_image_to_base64(image_path)
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(OLLAMA_URL, json={
                "model": model, "prompt": prompt, "images": [image_base64], "stream": False
            })
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"[Vision 오류: {response.status_code}]"
    except Exception as e:
        return f"[Vision 오류: {str(e)}]"


async def query_chatgpt(prompt: str, medical_mode: bool = False) -> str:
    api_key = get_api_key()
    if not api_key:
        return "[ChatGPT 오류] OpenAI API 키가 설정되지 않았습니다. 설정에서 API 키를 입력해주세요."

    system_prompt = """당신은 의료 전문 AI입니다. 의학 용어(영문 병기)와 해부학적 용어를 사용하세요.""" if medical_mode else """당신은 도움이 되는 AI 어시스턴트입니다. 정확하고 상세하게 한글로 답변하세요."""

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                OPENAI_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            )
            result = response.json()
            if "error" in result:
                return f"[ChatGPT API 오류] {result['error'].get('message', '')}"
            return result["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "[ChatGPT 오류] 요청 시간 초과"
    except Exception as e:
        return f"[ChatGPT 오류] {str(e)}"


async def query_chatgpt_vision(prompt: str, image_path: str, medical_mode: bool = False) -> str:
    """Query ChatGPT with image (GPT-4o Vision)"""
    api_key = get_api_key()
    if not api_key:
        return "[ChatGPT Vision 오류] OpenAI API 키가 설정되지 않았습니다."

    system_prompt = """당신은 의료 영상 분석 전문 AI입니다. 의학 용어(영문 병기)와 해부학적 용어를 사용하여 이미지를 분석하세요.""" if medical_mode else """당신은 이미지 분석 전문 AI입니다. 이미지를 자세히 분석하고 한글로 설명하세요."""

    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)

        # Determine image type
        file_ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        mime_type = mime_types.get(file_ext, "image/jpeg")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OPENAI_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            )
            result = response.json()
            if "error" in result:
                return f"[ChatGPT Vision 오류] {result['error'].get('message', '')}"
            return result["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "[ChatGPT Vision 오류] 요청 시간 초과"
    except Exception as e:
        return f"[ChatGPT Vision 오류] {str(e)}"


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    if not text1 or not text2:
        return 0.0

    # Clean and tokenize
    words1 = set(text1.lower().replace('\n', ' ').split())
    words2 = set(text2.lower().replace('\n', ' ').split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


# ============== STATUS HTML ==============

def create_agent_status_html(search="waiting", reasoning="waiting", synthesis="waiting", chatgpt="waiting"):
    """Create agent status animation HTML"""

    def get_status_style(status):
        if status == "running":
            return "background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); animation: pulse 1.5s infinite;"
        elif status == "done":
            return "background: #27ae60;"
        elif status == "error":
            return "background: #e74c3c;"
        else:
            return "background: #95a5a6;"

    def get_status_icon(status):
        if status == "running":
            return '<div class="spinner"></div>'
        elif status == "done":
            return "✓"
        elif status == "error":
            return "✗"
        else:
            return "○"

    use_trained = get_use_trained_models()
    trained_badge = '<span style="font-size:8px;background:#e74c3c;padding:1px 4px;border-radius:3px;">학습됨</span>' if use_trained else ''

    html = f"""
    <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.05); }}
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .agent-status-container {{
            display: flex;
            gap: 12px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 12px;
            justify-content: center;
        }}
        .agent-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 12px 16px;
            border-radius: 10px;
            color: white;
            min-width: 80px;
            font-size: 12px;
        }}
        .agent-icon {{ font-size: 24px; margin-bottom: 4px; }}
        .agent-name {{ font-weight: 600; margin-bottom: 2px; }}
        .agent-model {{ font-size: 10px; opacity: 0.8; }}
        .spinner {{
            width: 16px; height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        .status-indicator {{ margin-top: 4px; font-size: 14px; }}
    </style>

    <div class="agent-status-container">
        <div class="agent-box" style="{get_status_style(search)}">
            <div class="agent-icon">🔍</div>
            <div class="agent-name">검색 {trained_badge}</div>
            <div class="agent-model">{get_active_model('search')[:15]}</div>
            <div class="status-indicator">{get_status_icon(search)}</div>
        </div>
        <div class="agent-box" style="{get_status_style(reasoning)}">
            <div class="agent-icon">🧠</div>
            <div class="agent-name">추론 {trained_badge}</div>
            <div class="agent-model">{get_active_model('reasoning')[:15]}</div>
            <div class="status-indicator">{get_status_icon(reasoning)}</div>
        </div>
        <div class="agent-box" style="{get_status_style(synthesis)}">
            <div class="agent-icon">⚡</div>
            <div class="agent-name">통합 {trained_badge}</div>
            <div class="agent-model">{get_active_model('synthesis')[:15]}</div>
            <div class="status-indicator">{get_status_icon(synthesis)}</div>
        </div>
        <div class="agent-box" style="{get_status_style(chatgpt)}">
            <div class="agent-icon">🤖</div>
            <div class="agent-name">ChatGPT</div>
            <div class="agent-model">GPT-4o</div>
            <div class="status-indicator">{get_status_icon(chatgpt)}</div>
        </div>
    </div>
    """
    return html


# ============== MAIN CHAT PROCESSING ==============

async def process_chat_async(query: str, file_path: str, use_chatgpt: bool, medical_mode: bool):
    """Process chat with streaming status updates"""

    results = {
        "search": "", "reasoning": "", "synthesis": "", "chatgpt": "",
        "status": {"search": "waiting", "reasoning": "waiting", "synthesis": "waiting", "chatgpt": "waiting"}
    }

    yield create_agent_status_html(**results["status"]), "", "", "", "", ""

    # Process file
    file_content = ""
    is_image = False
    if file_path:
        file_ext = Path(file_path).suffix.lower()
        is_image = file_ext in SUPPORTED_FILES["image"]
        if not is_image:
            file_content, _ = extract_text_from_file(file_path)

    full_query = f"{query}\n\n{file_content}" if file_content else query
    medical_inst = "의학 용어와 해부학적 용어를 사용하세요." if medical_mode else ""

    # Add RAG context
    rag_context = find_relevant_knowledge(query)
    if rag_context:
        full_query += rag_context

    # Start ChatGPT if enabled
    chatgpt_task = None
    if use_chatgpt:
        results["status"]["chatgpt"] = "running"
        yield create_agent_status_html(**results["status"]), "", "", "", "", ""
        # Use ChatGPT Vision for images
        if is_image and file_path:
            chatgpt_task = asyncio.create_task(query_chatgpt_vision(query, file_path, medical_mode))
        else:
            chatgpt_task = asyncio.create_task(query_chatgpt(full_query, medical_mode))

    # Start Search and Reasoning in parallel
    results["status"]["search"] = "running"
    results["status"]["reasoning"] = "running"
    yield create_agent_status_html(**results["status"]), "", "", "", "", ""

    search_model = get_active_model("search") if not is_image else MODELS["vision"]
    reasoning_model = get_active_model("reasoning")
    synthesis_model = get_active_model("synthesis")

    if is_image:
        search_prompt = f"이미지를 자세히 분석하세요. {medical_inst}\n질문: {query}"
        search_task = asyncio.create_task(query_ollama_vision(MODELS["vision"], search_prompt, file_path))
        # For images, reasoning agent also analyzes the image
        reasoning_prompt = f"이미지를 논리적으로 분석하고 설명하세요. {medical_inst}\n질문: {query}"
        reasoning_task = asyncio.create_task(query_ollama_vision(MODELS["vision"], reasoning_prompt, file_path))
    else:
        search_prompt = f"정보를 검색하고 분석하세요. {medical_inst}\n질문: {full_query}"
        search_task = asyncio.create_task(query_ollama(search_model, search_prompt))
        reasoning_prompt = f"논리적으로 분석하세요. {medical_inst}\n질문: {full_query}"
        reasoning_task = asyncio.create_task(query_ollama(reasoning_model, reasoning_prompt))

    results["search"], results["reasoning"] = await asyncio.gather(search_task, reasoning_task)

    # Check for errors
    results["status"]["search"] = "error" if results["search"].startswith("[") else "done"
    results["status"]["reasoning"] = "error" if results["reasoning"].startswith("[") else "done"

    yield create_agent_status_html(**results["status"]), results["search"], results["reasoning"], "", "", ""

    # Synthesis
    results["status"]["synthesis"] = "running"
    yield create_agent_status_html(**results["status"]), results["search"], results["reasoning"], "", "", ""

    synthesis_prompt = f"""다음 분석을 통합하여 최종 답변을 작성하세요.
{medical_inst}

질문: {query}

검색 분석:
{results["search"][:1500]}

추론 분석:
{results["reasoning"][:1500]}

최종 답변:"""

    results["synthesis"] = await query_ollama(synthesis_model, synthesis_prompt)
    results["status"]["synthesis"] = "error" if results["synthesis"].startswith("[") else "done"

    # Wait for ChatGPT if enabled
    comparison_result = ""
    if chatgpt_task:
        results["chatgpt"] = await chatgpt_task
        results["status"]["chatgpt"] = "error" if results["chatgpt"].startswith("[") else "done"

        # Calculate similarity and save learning data if needed
        if not results["chatgpt"].startswith("[") and not results["synthesis"].startswith("["):
            similarity = calculate_similarity(results["chatgpt"], results["synthesis"])
            similarity_pct = similarity * 100

            if similarity >= 0.90:
                comparison_result = f"✅ **일치율: {similarity_pct:.1f}%** - ChatGPT와 로컬 AI 결과가 높은 일치율을 보입니다."
            else:
                comparison_result = f"⚠️ **일치율: {similarity_pct:.1f}%** - 결과가 다릅니다. 두 답변을 비교해보세요."
                # Save for learning (similarity < 90%)
                save_learning_data(query, results["chatgpt"], results["synthesis"], similarity)
                comparison_result += "\n\n📚 학습 데이터로 저장되었습니다."

    yield (create_agent_status_html(**results["status"]),
           results["search"], results["reasoning"],
           results["synthesis"], results["chatgpt"], comparison_result)


def run_chat_sync(query, history, file, use_chatgpt, medical_mode, conv_id):
    """Synchronous wrapper"""
    if not query.strip() and not file:
        return (history, conv_id,
                create_agent_status_html(), "", "", "", "", "", "")

    if not query.strip() and file:
        query = "이 파일을 분석해주세요."

    file_path = file.name if file and hasattr(file, 'name') else file

    if conv_id is None:
        title = query[:30] + "..." if len(query) > 30 else query
        conv_id = create_conversation(title)

    # Gradio 6.0 messages format: list of dicts with role and content
    history.append({"role": "user", "content": query})
    save_message(conv_id, "user", query)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        final_results = None
        async def run():
            nonlocal final_results
            async for result in process_chat_async(query, file_path, use_chatgpt, medical_mode):
                final_results = result

        loop.run_until_complete(run())

        status_html, search, reasoning, synthesis, chatgpt, comparison = final_results

        # Build response based on similarity
        if use_chatgpt and chatgpt and not chatgpt.startswith("["):
            similarity = calculate_similarity(chatgpt, synthesis)

            if similarity >= 0.90:
                # High similarity - show integrated result
                assistant_response = f"""### ✅ 통합 분석 결과 (일치율: {similarity*100:.1f}%)

ChatGPT와 로컬 AI의 분석이 일치합니다.

---

{synthesis}

---
*검색, 추론, 통합 분석을 종합한 결과입니다.*
"""
            else:
                # Low similarity - show both separately
                assistant_response = f"""### ⚠️ 비교 분석 결과 (일치율: {similarity*100:.1f}%)

**분석 결과가 다릅니다. 두 답변을 비교해보세요.**

---

### 🤖 ChatGPT 답변:
{chatgpt}

---

### 🏠 로컬 AI 통합 분석:
{synthesis}

---
*일치율이 90% 미만이므로 학습 데이터로 저장되었습니다.*
"""
        else:
            # ChatGPT not used or error
            assistant_response = f"""### 🏠 로컬 AI 분석 결과

{synthesis}
"""
            if chatgpt and chatgpt.startswith("["):
                assistant_response += f"\n\n---\n*ChatGPT 오류: {chatgpt}*"

        # Gradio 6.0: Add assistant response as separate message
        history.append({"role": "assistant", "content": assistant_response})
        save_message(conv_id, "assistant", assistant_response)

        return (history, conv_id, status_html,
                f"### 🔍 검색 분석\n{search}",
                f"### 🧠 추론 분석\n{reasoning}",
                f"### ⚡ 결과 통합\n{synthesis}",
                f"### 🤖 ChatGPT\n{chatgpt}" if chatgpt else "",
                comparison,
                "")
    except Exception as e:
        logger.error(f"Error: {e}")
        history.append({"role": "assistant", "content": f"오류: {str(e)}"})
        return (history, conv_id, create_agent_status_html(),
                "", "", f"오류: {str(e)}", "", "", "")
    finally:
        loop.close()


# ============== UI HELPERS ==============

def format_history_list():
    """Format history list as markdown (legacy)"""
    conversations = get_conversations(15)
    if not conversations:
        return "아직 대화가 없습니다."

    items = []
    for conv_id, title, updated in conversations:
        date_str = updated.split(" ")[0] if " " in updated else updated
        items.append(f"**{title[:25]}**\n_{date_str}_")

    return "\n\n".join(items)


def get_conversation_choices():
    """Get conversations as choices for dropdown/radio"""
    conversations = get_conversations(20)
    if not conversations:
        return []

    choices = []
    for conv_id, title, updated in conversations:
        date_str = updated.split(" ")[0] if " " in updated else updated
        # Format: "ID: title (date)"
        label = f"[{conv_id}] {title[:30]}  ({date_str})"
        choices.append((label, str(conv_id)))

    return choices


def load_conversation_by_selection(selection):
    """Load conversation when user clicks on history item"""
    if not selection:
        return [], None, create_agent_status_html(), "", "", "", "", ""

    try:
        # Extract conv_id from selection (it's the value part of the tuple)
        conv_id = int(selection)
        messages = get_conversation_messages(conv_id)
        history = []
        for role, content in messages:
            history.append({"role": role, "content": content})
        return history, conv_id, create_agent_status_html(), "", "", "", "", ""
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        return [], None, create_agent_status_html(), "", "", "", "", ""


def refresh_history_choices():
    """Refresh the conversation choices"""
    choices = get_conversation_choices()
    return gr.update(choices=choices, value=None)


def load_conversation_handler(conv_id_str):
    try:
        conv_id = int(conv_id_str)
        messages = get_conversation_messages(conv_id)
        history = []
        for role, content in messages:
            # Gradio 6.0 messages format
            history.append({"role": role, "content": content})
        return history, conv_id
    except:
        return [], None


def new_chat():
    return [], None, create_agent_status_html(), "", "", "", "", "", "", gr.update(value=None)


def format_learning_data_table(filter_type: str = "all"):
    """Format learning data as markdown table"""
    data = get_learning_data(filter_type, 30)

    if not data:
        return "학습 데이터가 없습니다.\n\nChatGPT 비교 기능을 사용하면 자동으로 학습 데이터가 수집됩니다."

    stats = get_learning_stats()
    header = f"""### 📊 학습 데이터 현황
- **전체:** {stats['total']}개
- **대기 중:** {stats['pending']}개
- **학습 완료:** {stats['completed']}개

---

| ID | 질문 | 유사도 | 상태 | 날짜 |
|:--:|:-----|:------:|:----:|:----:|
"""

    rows = []
    for item in data:
        data_id, query, chatgpt_ans, local_ans, similarity, learned, created = item
        status = "✅ 완료" if learned else "⏳ 대기"
        sim_pct = f"{similarity*100:.0f}%" if similarity else "-"
        query_short = query[:30] + "..." if len(query) > 30 else query
        date_short = created.split(" ")[0] if " " in created else created
        rows.append(f"| {data_id} | {query_short} | {sim_pct} | {status} | {date_short} |")

    return header + "\n".join(rows)


def view_learning_detail(data_id_str: str):
    """View detailed learning data"""
    try:
        data_id = int(data_id_str)
        data = get_learning_data_by_id(data_id)

        if not data:
            return "데이터를 찾을 수 없습니다."

        _, query, chatgpt_ans, local_ans, similarity, learned, learned_models, created, learned_at = data

        return f"""## 학습 데이터 상세 (ID: {data_id})

### 질문
{query}

### ChatGPT 답변
{chatgpt_ans}

### 로컬 AI 답변
{local_ans or "없음"}

### 정보
- **유사도:** {similarity*100:.1f}%
- **상태:** {"✅ 학습 완료" if learned else "⏳ 대기 중"}
- **학습된 모델:** {learned_models or "없음"}
- **생성일:** {created}
- **학습일:** {learned_at or "미학습"}
"""
    except:
        return "유효한 ID를 입력하세요."


def run_training(data_ids_str: str, model_type: str):
    """Run training on selected data"""
    try:
        if not data_ids_str.strip():
            return "❌ 학습할 데이터 ID를 입력하세요.\n예: 1,2,3 또는 1-5"

        # Parse data IDs
        data_ids = []
        for part in data_ids_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                data_ids.extend(range(start, end + 1))
            else:
                data_ids.append(int(part))

        if not data_ids:
            return "❌ 유효한 ID가 없습니다."

        if model_type == "all":
            return train_all_models(data_ids)
        else:
            return train_model(model_type, data_ids)

    except ValueError:
        return "❌ ID 형식 오류. 숫자를 입력하세요.\n예: 1,2,3 또는 1-5"
    except Exception as e:
        return f"❌ 오류: {str(e)}"


def refresh_system_status():
    """Refresh system status"""
    return get_system_status_html()


# ============== GRADIO INTERFACE ==============

def create_interface():
    with gr.Blocks(title="DR-Saintvision v2.3") as app:

        chat_history_state = gr.State([])
        conv_id_state = gr.State(None)

        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 28px;">🧠 DR-Saintvision</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.9;">멀티 에이전트 AI 토론 시스템 v2.3</p>
        </div>
        """)

        with gr.Tabs():
            # ========== CHAT TAB ==========
            with gr.TabItem("💬 대화"):
                with gr.Row():
                    # LEFT SIDEBAR
                    with gr.Column(scale=1, min_width=250):
                        gr.HTML("<h3>💬 대화</h3>")
                        new_chat_btn = gr.Button("✏️ 새 채팅", variant="primary", size="lg")

                        gr.HTML("<hr><h4 style='color:#666;'>📜 이전 대화</h4>")
                        # Clickable conversation list
                        history_radio = gr.Radio(
                            choices=get_conversation_choices(),
                            label="",
                            interactive=True,
                            elem_classes=["history-list"]
                        )
                        refresh_btn = gr.Button("🔄 새로고침", size="sm")

                        gr.HTML("<hr>")
                        with gr.Accordion("⚙️ 설정", open=True):
                            # System status
                            system_status_html = gr.HTML(value=get_system_status_html())
                            refresh_status_btn = gr.Button("🔄 상태 새로고침", size="sm")

                            gr.HTML("<hr>")
                            api_key_input = gr.Textbox(label="OpenAI API 키", type="password", placeholder="sk-...")
                            save_api_btn = gr.Button("💾 API 키 저장", size="sm")
                            api_result = gr.Markdown()

                            gr.HTML("<br>")
                            use_trained_cb = gr.Checkbox(
                                label="학습된 모델 사용",
                                value=get_use_trained_models()
                            )
                            trained_result = gr.Markdown()

                    # MAIN CONTENT
                    with gr.Column(scale=3):
                        gr.HTML("<h4 style='text-align:center;color:#666;'>🤖 에이전트 상태</h4>")
                        agent_status = gr.HTML(value=create_agent_status_html())

                        gr.HTML("<hr>")
                        chatbot = gr.Chatbot(value=[], height=350, show_label=False)

                        with gr.Row():
                            query_input = gr.Textbox(placeholder="질문을 입력하세요...", show_label=False, scale=6, lines=2)
                            submit_btn = gr.Button("전송 ➤", variant="primary", scale=1)

                        with gr.Row():
                            file_input = gr.File(label="📎 파일", file_types=[".png",".jpg",".pdf",".txt",".csv"], type="filepath")
                            with gr.Column():
                                use_chatgpt_cb = gr.Checkbox(label="🤖 ChatGPT 비교", value=True)
                                medical_mode_cb = gr.Checkbox(label="🏥 의료 모드", value=False)

                        # Comparison result
                        comparison_result = gr.Markdown()

                        gr.HTML("<hr><h4 style='color:#666;'>📊 상세 결과</h4>")
                        with gr.Tabs():
                            with gr.TabItem("🔍 검색"):
                                search_result = gr.Markdown()
                            with gr.TabItem("🧠 추론"):
                                reasoning_result = gr.Markdown()
                            with gr.TabItem("⚡ 통합"):
                                synthesis_result = gr.Markdown()
                            with gr.TabItem("🤖 ChatGPT"):
                                chatgpt_result = gr.Markdown()

            # ========== LEARNING TAB ==========
            with gr.TabItem("📚 데이터 학습"):
                gr.HTML("""
                <div style="padding: 16px; background: #e8f4f8; border-radius: 12px; margin-bottom: 16px;">
                    <h2 style="margin: 0;">📚 데이터 학습</h2>
                    <p style="margin: 8px 0 0 0; color: #666;">
                        ChatGPT와 로컬 AI의 일치율이 90% 미만일 때 자동으로 학습 데이터가 저장됩니다.<br>
                        저장된 데이터를 사용하여 로컬 AI 모델을 개선할 수 있습니다.
                    </p>
                </div>
                """)

                with gr.Row():
                    # Learning data list
                    with gr.Column(scale=2):
                        gr.HTML("<h3>📋 학습 데이터 목록</h3>")

                        with gr.Row():
                            filter_dropdown = gr.Dropdown(
                                choices=[("전체", "all"), ("대기 중", "pending"), ("완료", "completed")],
                                value="all",
                                label="필터",
                                scale=1
                            )
                            refresh_learning_btn = gr.Button("🔄 새로고침", scale=1)

                        learning_table = gr.Markdown(value=format_learning_data_table("all"))

                        gr.HTML("<hr>")
                        gr.HTML("<h4>🔍 상세 보기</h4>")
                        with gr.Row():
                            detail_id_input = gr.Textbox(label="데이터 ID", placeholder="예: 1")
                            view_detail_btn = gr.Button("보기")

                        learning_detail = gr.Markdown()

                    # Training controls
                    with gr.Column(scale=1):
                        gr.HTML("<h3>🎯 모델 학습</h3>")

                        gr.Markdown("""
**학습 방법:**
1. 왼쪽에서 학습할 데이터 ID 확인
2. 아래에 ID 입력 (예: 1,2,3 또는 1-5)
3. 학습할 모델 선택
4. "학습 시작" 클릭
""")

                        train_ids_input = gr.Textbox(
                            label="학습할 데이터 ID",
                            placeholder="예: 1,2,3 또는 1-5"
                        )

                        train_model_dropdown = gr.Dropdown(
                            choices=[
                                ("전체 모델 (3개)", "all"),
                                ("검색 모델 (Mistral)", "search"),
                                ("추론 모델 (Llama)", "reasoning"),
                                ("통합 모델 (Qwen)", "synthesis")
                            ],
                            value="all",
                            label="학습할 모델"
                        )

                        train_btn = gr.Button("🚀 학습 시작", variant="primary", size="lg")
                        training_result = gr.Markdown()

                        gr.HTML("<hr>")
                        gr.HTML("<h4>ℹ️ 학습 안내</h4>")
                        gr.Markdown("""
**자동 학습 데이터 수집:**
- ChatGPT 비교 활성화 시 자동 수집
- 일치율 90% 미만 결과만 저장
- 저장된 데이터로 모델 학습 가능

**학습 방식:**

1. **Modelfile 방식**
   - Ollama 커스텀 모델 생성
   - 시스템 프롬프트에 학습 데이터 포함

2. **RAG (지식 베이스)**
   - 학습 데이터를 지식 베이스에 저장
   - 질문 시 관련 지식 자동 참조
""")

        # ========== EVENT HANDLERS ==========

        # Chat handlers
        outputs = [chatbot, conv_id_state, agent_status,
                   search_result, reasoning_result, synthesis_result, chatgpt_result, comparison_result, query_input]

        submit_btn.click(
            fn=run_chat_sync,
            inputs=[query_input, chat_history_state, file_input, use_chatgpt_cb, medical_mode_cb, conv_id_state],
            outputs=outputs
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[chat_history_state]
        ).then(fn=refresh_history_choices, outputs=[history_radio])

        query_input.submit(
            fn=run_chat_sync,
            inputs=[query_input, chat_history_state, file_input, use_chatgpt_cb, medical_mode_cb, conv_id_state],
            outputs=outputs
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[chat_history_state]
        ).then(fn=refresh_history_choices, outputs=[history_radio])

        new_chat_btn.click(
            fn=new_chat,
            outputs=[chatbot, conv_id_state, agent_status, search_result, reasoning_result, synthesis_result, chatgpt_result, comparison_result, query_input, history_radio]
        ).then(fn=lambda: [], outputs=[chat_history_state])

        # History radio selection - load conversation when clicked
        history_radio.change(
            fn=load_conversation_by_selection,
            inputs=[history_radio],
            outputs=[chatbot, conv_id_state, agent_status, search_result, reasoning_result, synthesis_result, chatgpt_result, comparison_result]
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[chat_history_state])

        # Refresh history list
        refresh_btn.click(fn=refresh_history_choices, outputs=[history_radio])
        refresh_status_btn.click(fn=refresh_system_status, outputs=[system_status_html])

        save_api_btn.click(fn=save_api_key, inputs=[api_key_input], outputs=[api_result])
        use_trained_cb.change(fn=set_use_trained_models, inputs=[use_trained_cb], outputs=[trained_result])

        # Learning handlers
        refresh_learning_btn.click(
            fn=format_learning_data_table,
            inputs=[filter_dropdown],
            outputs=[learning_table]
        )

        filter_dropdown.change(
            fn=format_learning_data_table,
            inputs=[filter_dropdown],
            outputs=[learning_table]
        )

        view_detail_btn.click(
            fn=view_learning_detail,
            inputs=[detail_id_input],
            outputs=[learning_detail]
        )

        train_btn.click(
            fn=run_training,
            inputs=[train_ids_input, train_model_dropdown],
            outputs=[training_result]
        )

    return app


if __name__ == "__main__":
    print("""
    ================================================================
    |              DR-SAINTVISION v2.3                             |
    |     ChatGPT + Local AI Comparison System                     |
    ================================================================

    Features:
    - System status indicator (API key, Ollama, Models)
    - 90% similarity comparison logic
    - Auto-save learning data for <90% similarity
    - ChatGPT comparison enabled by default

    Access: http://localhost:7860
    """)

    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
