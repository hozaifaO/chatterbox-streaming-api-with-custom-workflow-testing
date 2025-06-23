#!/usr/bin/env python3
"""
Physics Book TTS Web Server
Handles requests from the HTML physics book reader frontend
Integrates with Ollama (Gemma3) and TTS services
"""

import json
import re
import argparse
import logging
import os
from typing import Optional
from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app for web server
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for configuration
OLLAMA_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "gemma3:latest"
TTS_URL = "http://localhost:5001"
TTS_VOICE = "alloy"

# Create directory for saving audio files
AUDIO_DIR = "/app/audio_outputs"
os.makedirs(AUDIO_DIR, exist_ok=True)

def clean_text_for_tts(text: str) -> str:
    """Remove emojis and clean text for TTS"""
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)
    
    # Replace common emojis with text
    replacements = {
        '😊': ' happy ', '😄': ' laughing ', '😢': ' sad ',
        '😎': ' cool ', '👍': ' thumbs up ', '❤️': ' heart '
    }
    
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_complete_ollama_response(prompt: str, model: str, base_url: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """Get complete response from Ollama (NO STREAMING)"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # NO STREAMING
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    logger.info(f"🤖 Sending request to Ollama: {base_url}/api/generate")
    logger.info(f"🤖 Model: {model}")
    logger.info(f"🤖 Prompt length: {len(prompt)} characters")
    
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=120  # Longer timeout for complete response
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Ollama API error: {response.status_code}")
            logger.error(f"❌ Response: {response.text}")
            return None
            
        data = response.json()
        llm_response = data.get('response', '')
        
        logger.info(f"✅ Ollama response received!")
        logger.info(f"📝 LLM Response length: {len(llm_response)} characters")
        logger.info(f"📝 LLM Response: {llm_response}")
        
        return llm_response                    
    except Exception as e:
        logger.error(f"❌ Error getting response from Ollama: {e}")
        return None

def get_complete_tts_audio(text: str, voice: str, tts_url: str) -> Optional[bytes]:
    """Get complete audio from TTS service (NO STREAMING)"""
    clean_text = clean_text_for_tts(text)
    if not clean_text.strip():
        logger.warning("⚠️ No clean text for TTS")
        return None
    
    payload = {
        "input": clean_text,
        "voice": voice,
        "model": "tts-1"
    }
    
    logger.info(f"🎤 Sending to TTS: {tts_url}/v1/audio/speech")
    logger.info(f"🎤 Voice: {voice}")
    logger.info(f"🎤 Text length: {len(clean_text)} characters")
    logger.info(f"🎤 Text preview: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}")
    
    try:
        response = requests.post(
            f"{tts_url}/v1/audio/speech",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            audio_data = response.content
            logger.info(f"✅ TTS audio generated successfully!")
            logger.info(f"🎵 Audio size: {len(audio_data)} bytes")
            return audio_data
        else:
            logger.error(f"❌ TTS API error: {response.status_code}")
            logger.error(f"❌ Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Error getting audio from TTS: {e}")
        return None

# Web server endpoints
@app.route('/explain', methods=['POST'])
def explain_section():
    """Endpoint to handle section explanation requests from frontend"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        section_text = data.get('text', '')
        section_number = data.get('section', 'unknown')
        
        if not section_text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"📖 Processing section {section_number}")
        logger.info(f"📖 Section text length: {len(section_text)} characters")
        logger.info(f"📖 Section text: {section_text}")
        
        # System prompt for physics explanations
        system_prompt = f"""You are a knowledgeable physics professor creating audio explanations for students. 
        
        The user will provide you with a physics text section, and you should explain it in a clear, engaging way suitable for audio listening.
        
        Your explanation should:
        - Be thorough but accessible to students learning physics
        - Explain key concepts in simple terms
        - Provide intuitive examples and analogies when helpful
        - Walk through any equations step by step
        - Connect concepts to real-world applications
        - Be between 2-4 minutes when spoken aloud
        - Use natural, conversational language suitable for audio
        - Avoid overly long sentences - keep them audio-friendly
        - Do not use emojis, special characters, or visual references
        - Focus on helping the listener truly understand the concept
        
        Think of this as creating a mini physics lesson that someone could listen to while walking or driving.
        Make the physics come alive through your explanation!"""
        
        # Get complete response from LLM
        logger.info("🚀 Starting LLM processing...")
        llm_response = get_complete_ollama_response(
            section_text, 
            OLLAMA_MODEL, 
            OLLAMA_URL, 
            system_prompt
        )
        
        if not llm_response:
            return jsonify({'error': 'Failed to get explanation from LLM'}), 500
        
        # Generate complete audio
        logger.info("🚀 Starting TTS processing...")
        audio_data = get_complete_tts_audio(llm_response, TTS_VOICE, TTS_URL)
        
        if not audio_data:
            return jsonify({'error': 'Failed to generate audio'}), 500
        
        # Save audio file locally
        filename = f"section_{section_number.replace('.', '_')}_{len(llm_response)}_chars.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            logger.info(f"💾 Audio saved: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save audio file: {e}")
        
        # Return audio directly
        return Response(
            audio_data,
            mimetype='audio/wav',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Content-Length': str(len(audio_data))
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in explain_section endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'physics_book_server',
        'ollama_url': OLLAMA_URL,
        'tts_url': TTS_URL,
        'audio_dir': AUDIO_DIR
    })

@app.route('/test-services', methods=['GET'])
def test_services():
    """Test connectivity to Ollama and TTS services"""
    results = {}
    
    # Test Ollama
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        results['ollama'] = {
            'status': 'connected' if response.status_code == 200 else 'error',
            'url': OLLAMA_URL,
            'status_code': response.status_code
        }
    except Exception as e:
        results['ollama'] = {
            'status': 'error',
            'url': OLLAMA_URL,
            'error': str(e)
        }
    
    # Test TTS
    try:
        response = requests.get(f"{TTS_URL}/health", timeout=10)
        results['tts'] = {
            'status': 'connected' if response.status_code == 200 else 'error',
            'url': TTS_URL,
            'status_code': response.status_code
        }
    except Exception as e:
        results['tts'] = {
            'status': 'error',
            'url': TTS_URL,
            'error': str(e)
        }
    
    return jsonify(results)

def main():
    parser = argparse.ArgumentParser(description="Physics Book TTS Web Server")
    parser.add_argument("--ollama-url", default="http://host.docker.internal:11434", help="Ollama URL")
    parser.add_argument("--ollama-model", default="gemma3:latest", help="Ollama model")
    parser.add_argument("--tts-url", default="http://localhost:5001", help="TTS URL")
    parser.add_argument("--voice", default="alloy", help="TTS voice")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        app.debug = True
    
    # Set global configuration
    global OLLAMA_URL, OLLAMA_MODEL, TTS_URL, TTS_VOICE
    OLLAMA_URL = args.ollama_url
    OLLAMA_MODEL = args.ollama_model
    TTS_URL = args.tts_url
    TTS_VOICE = args.voice
    
    print("🔬 Physics Book TTS Web Server")
    print("=" * 40)
    print(f"Server running on: http://{args.host}:{args.port}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"TTS URL: {TTS_URL}")
    print(f"TTS Voice: {TTS_VOICE}")
    print(f"Audio files saved to: {AUDIO_DIR}")
    print()
    print("Endpoints:")
    print(f"  POST /explain - Process physics section explanations")
    print(f"  GET  /health - Health check")
    print(f"  GET  /test-services - Test Ollama and TTS connectivity")
    print()
    print("Open your physics_book_reader.html file in a browser")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Physics Book Server...")

if __name__ == "__main__":
    main()