#!/bin/bash

# Test All Voices Script
# Tests each voice with a different phrase

echo "🎤 Testing All Chatterbox Voices"
echo "================================="

# Check if TTS server is running
if ! curl -s http://localhost:5001/health > /dev/null; then
    echo "❌ TTS server not running on localhost:5001"
    echo "Start with: docker run -d --gpus all -p 5001:5001 --name chatterbox-tts-voices chatterbox-tts-voices"
    exit 1
fi

echo "✅ TTS server is running"
echo ""

# Array of voices and test phrases
declare -A voice_phrases=(
    ["alloy"]="Hello, I am Alloy, your professional and neutral voice assistant."
    ["echo"]="Hi there, I'm Echo, speaking with clarity and precision."
    ["fable"]="Greetings, I'm Fable, ready to tell you wonderful stories."
    ["onyx"]="Good day, I am Onyx, with a deep and commanding presence."
    ["nova"]="Hey! I'm Nova, bringing energy and enthusiasm to our chat."
    ["shimmer"]="Hello, I'm Shimmer, speaking softly and gently."
)

# Test each voice
for voice in "${!voice_phrases[@]}"; do
    echo "🔊 Testing voice: $voice"
    phrase="${voice_phrases[$voice]}"
    
    # Generate audio
    curl -s -X POST http://localhost:5001/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d "{\"input\":\"$phrase\",\"voice\":\"$voice\",\"model\":\"tts-1\"}" \
        --output "test_${voice}.wav"
    
    if [ $? -eq 0 ] && [ -f "test_${voice}.wav" ]; then
        echo "  ✅ Generated: test_${voice}.wav"
        
        # Check file size (should be > 1KB for valid audio)
        size=$(stat -f%z "test_${voice}.wav" 2>/dev/null || stat -c%s "test_${voice}.wav" 2>/dev/null)
        if [ "$size" -gt 1000 ]; then
            echo "  📊 File size: ${size} bytes (good)"
            
            # Play on Windows (if available)
            if command -v cmd.exe >/dev/null 2>&1; then
                echo "  🎵 Playing on Windows..."
                cmd.exe /c start "test_${voice}.wav"
                sleep 3
            elif command -v afplay >/dev/null 2>&1; then
                echo "  🎵 Playing on macOS..."
                afplay "test_${voice}.wav"
            elif command -v aplay >/dev/null 2>&1; then
                echo "  🎵 Playing on Linux..."
                aplay "test_${voice}.wav"
            else
                echo "  💾 Audio saved (no player found)"
            fi
        else
            echo "  ❌ File too small (${size} bytes) - may be invalid"
        fi
    else
        echo "  ❌ Failed to generate audio"
    fi
    
    echo ""
done

echo "🎉 Voice testing complete!"
echo ""
echo "📁 Generated files:"
ls -la test_*.wav 2>/dev/null || echo "No audio files generated"

echo ""
echo "🔍 Check available voices:"
curl -s http://localhost:5001/v1/voices | python -m json.tool

echo ""
echo "💡 To use different voices in your chat:"
echo "  python ultra_simple_chat.py --voice echo"
echo "  python ultra_simple_chat.py --voice fable"
echo "  python ultra_simple_chat.py --voice onyx"