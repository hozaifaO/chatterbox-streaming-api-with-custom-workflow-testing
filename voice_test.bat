@echo off
title Testing All Chatterbox Voices

echo ==========================================
echo 🎤 Testing All Chatterbox Voices
echo ==========================================
echo.

REM Check if TTS server is running
curl -s http://localhost:5001/health >nul 2>&1
if errorlevel 1 (
    echo ❌ TTS server not running on localhost:5001
    echo Start with: docker run -d --gpus all -p 5001:5001 --name chatterbox-tts-voices chatterbox-tts-voices
    pause
    exit /b 1
)

echo ✅ TTS server is running
echo.

REM Test each voice with unique phrases
echo 🔊 Testing voice: alloy
curl -s -X POST http://localhost:5001/v1/audio/speech ^
    -H "Content-Type: application/json" ^
    -d "{\"input\":\"Hello, I am Alloy, your professional and neutral voice assistant.\",\"voice\":\"alloy\",\"model\":\"tts-1\"}" ^
    --output test_alloy.wav
if exist test_alloy.wav (
    echo   ✅ Generated: test_alloy.wav
    start test_alloy.wav
    timeout /t 4 /nobreak >nul
)

echo.
echo 🔊 Testing voice: echo
curl -s -X POST http://localhost:5001/v1/audio/speech ^
    -H "Content-Type: application/json" ^
    -d "{\"input\":\"Hi there, I'm Echo, speaking with clarity and precision.\",\"voice\":\"echo\",\"model\":\"tts-1\"}" ^
    --output test_echo.wav
if exist test_echo.wav (
    echo   ✅ Generated: test_echo.wav
    start test_echo.wav
    timeout /t 4 /nobreak >nul
)

echo.
echo 🔊 Testing voice: fable
curl -s -X POST http://localhost:5001/v1/audio/speech ^
    -H "Content-Type: application/json" ^
    -d "{\"input\":\"Greetings, I'm Fable, ready to tell you wonderful stories.\",\"voice\":\"fable\",\"model\":\"tts-1\"}" ^
    --output test_fable.wav
if exist test_fable.wav (
    echo   ✅ Generated: test_fable.wav
    start test_fable.wav
    timeout /t 4 /nobreak >nul
)

echo.
echo 🔊 Testing voice: onyx
curl -s -X POST http://localhost:5001/v1/audio/speech ^
    -H "Content-Type: application/json" ^
    -d "{\"input\":\"Good day, I am Onyx, with a deep and commanding presence.\",\"voice\":\"onyx\",\"model\":\"tts-1\"}" ^
    --output test_onyx.wav
if exist test_onyx.wav (
    echo   ✅ Generated: test_onyx.wav
    start test_onyx.wav
    timeout /t 4 /nobreak >nul
)

echo.
echo 🔊 Testing voice: nova
curl -s -X POST http://localhost:5001/v1/audio/speech ^
    -H "Content-Type: application/json" ^
    -d "{\"input\":\"Hey! I'm Nova, bringing energy and enthusiasm to our chat.\",\"voice\":\"nova\",\"model\":\"tts-1\"}" ^
    --output test_nova.wav
if exist test_nova.wav (
    echo   ✅ Generated: test_nova.wav
    start test_nova.wav
    timeout /t 4 /nobreak >nul
)

echo.
echo 🔊 Testing voice: shimmer
curl -s -X POST http://localhost:5001/v1/audio/speech ^
    -H "Content-Type: application/json" ^
    -d "{\"input\":\"Hello, I'm Shimmer, speaking softly and gently.\",\"voice\":\"shimmer\",\"model\":\"tts-1\"}" ^
    --output test_shimmer.wav
if exist test_shimmer.wav (
    echo   ✅ Generated: test_shimmer.wav
    start test_shimmer.wav
    timeout /t 4 /nobreak >nul
)

echo.
echo ==========================================
echo 🎉 Voice testing complete!
echo ==========================================
echo.

echo 📁 Generated files:
dir test_*.wav 2>nul || echo No audio files found

echo.
echo 🔍 Available voices from API:
curl -s http://localhost:5001/v1/voices

echo.
echo.
echo 💡 To use different voices in your chat:
echo   python ultra_simple_chat.py --voice echo
echo   python ultra_simple_chat.py --voice fable  
echo   python ultra_simple_chat.py --voice onyx
echo   python ultra_simple_chat.py --voice nova
echo   python ultra_simple_chat.py --voice shimmer
echo.
pause