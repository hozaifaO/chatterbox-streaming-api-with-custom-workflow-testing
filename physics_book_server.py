#!/usr/bin/env python3
"""
Physics Book TTS Web Server
Handles requests from the HTML physics book reader frontend
Integrates with Azure DeepSeek-V3 and TTS services
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
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app for web server
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for configuration - UPDATED FOR AZURE
AZURE_ENDPOINT = "https://aiiieou.services.ai.azure.com/models"
AZURE_MODEL = "DeepSeek-V3-0324"
AZURE_API_KEY = ""  # Replace with your actual API key
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

def get_complete_azure_response(prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """Get complete response from Azure DeepSeek-V3 (NO STREAMING)"""
    
    logger.info(f"🤖 Sending request to Azure DeepSeek-V3: {AZURE_ENDPOINT}")
    logger.info(f"🤖 Model: {AZURE_MODEL}")
    logger.info(f"🤖 Prompt length: {len(prompt)} characters")
    
    try:
        # Initialize Azure client
        client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY),
            api_version="2024-05-01-preview"
        )
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(UserMessage(content=prompt))
        
        # Get complete response (NO STREAMING)
        response = client.complete(
            stream=False,  # NO STREAMING
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            model=AZURE_MODEL
        )
        
        # Extract response text
        if response.choices and len(response.choices) > 0:
            llm_response = response.choices[0].message.content
            
            logger.info(f"✅ Azure DeepSeek-V3 response received!")
            logger.info(f"📝 LLM Response length: {len(llm_response)} characters")
            logger.info(f"📝 LLM Response: {llm_response}")
            
            client.close()
            return llm_response
        else:
            logger.error("❌ No choices in Azure response")
            client.close()
            return None                    
    except Exception as e:
        logger.error(f"❌ Error getting response from Azure DeepSeek-V3: {e}")
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
            timeout=300  # INCREASED TIMEOUT TO 5 MINUTES
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
        
        # Get complete response from Azure DeepSeek-V3
        logger.info("🚀 Starting Azure DeepSeek-V3 processing...")
        llm_response = get_complete_azure_response(section_text, system_prompt)
        
        if not llm_response:
            return jsonify({'error': 'Failed to get explanation from Azure DeepSeek-V3'}), 500
        
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
        'azure_endpoint': AZURE_ENDPOINT,
        'azure_model': AZURE_MODEL,
        'tts_url': TTS_URL,
        'audio_dir': AUDIO_DIR
    })

@app.route('/test-services', methods=['GET'])
def test_services():
    """Test connectivity to Azure and TTS services"""
    results = {}
    
    # Test Azure DeepSeek-V3
    try:
        test_response = get_complete_azure_response("Hello, this is a test.", "You are a helpful assistant.")
        results['azure'] = {
            'status': 'connected' if test_response else 'error',
            'endpoint': AZURE_ENDPOINT,
            'model': AZURE_MODEL,
            'test_response_length': len(test_response) if test_response else 0
        }
    except Exception as e:
        results['azure'] = {
            'status': 'error',
            'endpoint': AZURE_ENDPOINT,
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
    parser = argparse.ArgumentParser(description="Physics Book TTS Web Server with Azure DeepSeek-V3")
    parser.add_argument("--azure-endpoint", default="https://aiiieou.services.ai.azure.com/models", help="Azure endpoint")
    parser.add_argument("--azure-model", default="DeepSeek-V3-0324", help="Azure model")
    parser.add_argument("--azure-api-key", default="<YOUR_API_KEY_HERE>", help="Azure API key")
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
    global AZURE_ENDPOINT, AZURE_MODEL, AZURE_API_KEY, TTS_URL, TTS_VOICE
    AZURE_ENDPOINT = args.azure_endpoint
    AZURE_MODEL = args.azure_model
    AZURE_API_KEY = args.azure_api_key
    TTS_URL = args.tts_url
    TTS_VOICE = args.voice
    
    print("🔬 Physics Book TTS Web Server with Azure DeepSeek-V3")
    print("=" * 55)
    print(f"Server running on: http://{args.host}:{args.port}")
    print(f"Azure Endpoint: {AZURE_ENDPOINT}")
    print(f"Azure Model: {AZURE_MODEL}")
    print(f"TTS URL: {TTS_URL}")
    print(f"TTS Voice: {TTS_VOICE}")
    print(f"Audio files saved to: {AUDIO_DIR}")
    print()
    print("Endpoints:")
    print(f"  POST /explain - Process physics section explanations")
    print(f"  GET  /health - Health check")
    print(f"  GET  /test-services - Test Azure and TTS connectivity")
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