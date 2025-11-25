@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   DR-Saintvision 모델 다운로드 스크립트
echo ============================================================
echo.

:: Check if Ollama is installed
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Ollama가 설치되어 있지 않습니다.
    echo.
    echo Ollama를 먼저 설치해주세요:
    echo   1. https://ollama.ai 에서 다운로드
    echo   2. 또는 아래 명령어로 자동 설치:
    echo.
    echo      winget install Ollama.Ollama
    echo.

    choice /C YN /M "Ollama를 자동으로 설치하시겠습니까?"
    if !errorlevel! equ 1 (
        echo.
        echo Ollama 설치 중...
        winget install Ollama.Ollama
        if !errorlevel! neq 0 (
            echo [오류] Ollama 설치 실패
            echo https://ollama.ai 에서 수동으로 설치해주세요.
            pause
            exit /b 1
        )
        echo Ollama 설치 완료. 터미널을 다시 시작한 후 이 스크립트를 다시 실행해주세요.
        pause
        exit /b 0
    ) else (
        echo 설치를 취소했습니다.
        pause
        exit /b 1
    )
)

:: Check if Ollama service is running
echo Ollama 서비스 확인 중...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama 서비스 시작 중...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

echo.
echo ============================================================
echo   다운로드할 모델 목록:
echo   1. mistral:7b-instruct-v0.2-q4_0  (Search Agent)    ~4GB
echo   2. llama3.2:latest                (Reasoning Agent) ~4GB
echo   3. qwen2.5:7b-instruct-q4_0       (Synthesis Agent) ~4GB
echo ============================================================
echo   총 예상 용량: 약 12GB
echo ============================================================
echo.

choice /C YN /M "모델 다운로드를 시작하시겠습니까?"
if %errorlevel% neq 1 (
    echo 다운로드를 취소했습니다.
    exit /b 0
)

echo.
echo [1/3] Mistral 7B 모델 다운로드 중...
echo --------------------------------------------------------
ollama pull mistral:7b-instruct-v0.2-q4_0
if %errorlevel% neq 0 (
    echo [경고] mistral 다운로드 실패, 기본 버전으로 재시도...
    ollama pull mistral:7b-instruct
)

echo.
echo [2/3] Llama 3.2 모델 다운로드 중...
echo --------------------------------------------------------
ollama pull llama3.2:latest
if %errorlevel% neq 0 (
    echo [경고] llama3.2 다운로드 실패, 대체 버전으로 재시도...
    ollama pull llama3.2:3b
)

echo.
echo [3/3] Qwen 2.5 모델 다운로드 중...
echo --------------------------------------------------------
ollama pull qwen2.5:7b-instruct-q4_0
if %errorlevel% neq 0 (
    echo [경고] qwen2.5 다운로드 실패, 기본 버전으로 재시도...
    ollama pull qwen2.5:7b
)

echo.
echo ============================================================
echo   다운로드 완료!
echo ============================================================
echo.
echo 설치된 모델 목록:
ollama list

echo.
echo 모델 테스트를 진행하시겠습니까?
choice /C YN /M "테스트 실행"
if %errorlevel% equ 1 (
    echo.
    echo [테스트] Mistral 모델 테스트...
    echo "안녕하세요" | ollama run mistral:7b-instruct-v0.2-q4_0 "Say hello in Korean" 2>nul || echo "테스트 완료"

    echo.
    echo [테스트] Llama 모델 테스트...
    echo "test" | ollama run llama3.2:latest "What is 2+2?" 2>nul || echo "테스트 완료"

    echo.
    echo [테스트] Qwen 모델 테스트...
    echo "test" | ollama run qwen2.5:7b-instruct-q4_0 "Hello" 2>nul || echo "테스트 완료"
)

echo.
echo ============================================================
echo   모든 작업이 완료되었습니다!
echo   이제 다음 명령어로 DR-Saintvision을 실행할 수 있습니다:
echo.
echo      python main.py
echo ============================================================
pause
