#!/usr/bin/env python3
"""
Physics Book TTS Web Server - STREAMING VERSION
Handles requests from the HTML physics book reader frontend
Integrates with Azure DeepSeek-V3 and TTS services with streaming audio
"""

import json
import re
import argparse
import logging
import os
import io
import threading
import queue
import wave
from typing import Optional, Generator
from flask import Flask, request, Response, jsonify, stream_template
from flask_cors import CORS

import requests
import numpy as np
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

class SimpleWAVParser:
    """Extract complete WAV files from a streaming HTTP response."""
    def __init__(self) -> None:
        self.buffer = bytearray()
    
    def add_data(self, data: bytes) -> list[bytes]:
        self.buffer.extend(data)
        out: list[bytes] = []
        while True:
            wav = self._next_wav()
            if wav:
                out.append(wav)
            else:
                break
        return out
    
    def _next_wav(self) -> Optional[bytes]:
        if len(self.buffer) < 12:
            return None
        riff = self.buffer.find(b"RIFF")
        if riff == -1:
            self.buffer.clear()
            return None
        if riff > 0:
            del self.buffer[:riff]
        if len(self.buffer) < 12:
            return None
        size = int.from_bytes(self.buffer[4:8], "little") + 8
        if len(self.buffer) < size:
            return None
        wav = bytes(self.buffer[:size])
        del self.buffer[:size]
        return wav

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

def get_streaming_tts_audio_with_buffer(text: str, voice: str, tts_url: str, save_path: Optional[str] = None) -> Generator[bytes, None, None]:
    """Get streaming audio from TTS service with smart buffering to prevent stuttering"""
    clean_text = clean_text_for_tts(text)
    if not clean_text.strip():
        logger.warning("⚠️ No clean text for TTS")
        return
    
    payload = {
        "input": clean_text,
        "voice": voice,
        "model": "tts-1",
        "stream": True,
        "chunk_size": 30,  # Smaller chunks for faster streaming
        "low_latency": True
    }
    
    logger.info(f"🎤 Starting buffered streaming TTS: {tts_url}/v1/audio/speech/stream")
    logger.info(f"🎤 Voice: {voice}")
    logger.info(f"🎤 Text length: {len(clean_text)} characters")
    logger.info(f"🎤 Text preview: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}")
    
    # For saving complete audio file
    all_audio_chunks = []
    
    # Buffer management - collect initial chunks before streaming starts
    audio_buffer = []
    buffer_target = 3  # Target 3 chunks before starting playback
    chunks_yielded = 0
    
    try:
        response = requests.post(
            f"{tts_url}/v1/audio/speech/stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=300
        )
        
        if response.status_code != 200:
            logger.error(f"❌ TTS API error: {response.status_code}")
            logger.error(f"❌ Response: {response.text}")
            return
        
        logger.info(f"✅ TTS streaming started successfully!")
        
        parser = SimpleWAVParser()
        chunk_count = 0
        
        for http_chunk in response.iter_content(chunk_size=4096):
            if not http_chunk:
                continue
                
            for wav_bytes in parser.add_data(http_chunk):
                chunk_count += 1
                logger.debug(f"🎵 Received chunk {chunk_count}: {len(wav_bytes)} bytes")
                
                # Save for complete file
                if save_path:
                    all_audio_chunks.append(wav_bytes)
                
                # Add to buffer
                audio_buffer.append(wav_bytes)
                
                # Start yielding once we have enough buffer OR if this is a later chunk
                if len(audio_buffer) >= buffer_target or chunks_yielded > 0:
                    # Yield buffered chunks
                    while audio_buffer:
                        chunk_to_yield = audio_buffer.pop(0)
                        chunks_yielded += 1
                        logger.info(f"🎵 Streaming chunk {chunks_yielded} (buffer size: {len(audio_buffer)})")
                        yield chunk_to_yield
        
        # Yield any remaining buffered chunks
        while audio_buffer:
            chunk_to_yield = audio_buffer.pop(0)
            chunks_yielded += 1
            logger.info(f"🎵 Final chunk {chunks_yielded}")
            yield chunk_to_yield
        
        logger.info(f"✅ TTS streaming completed! Total chunks: {chunk_count}, Yielded: {chunks_yielded}")
        
        # Save complete audio file if requested
        if save_path and all_audio_chunks:
            try:
                with open(save_path, 'wb') as f:
                    for chunk in all_audio_chunks:
                        f.write(chunk)
                logger.info(f"💾 Complete audio saved: {save_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save complete audio: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error during streaming TTS: {e}")

# Web server endpoints
@app.route('/explain', methods=['POST'])
def explain_section():
    """Endpoint to handle section explanation requests from frontend - STREAMING VERSION"""
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
        
        # Prepare save path for complete audio
        filename = f"section_{section_number.replace('.', '_')}_{len(llm_response)}_chars.wav"
        save_path = os.path.join(AUDIO_DIR, filename)
          # Stream audio generation
        logger.info("🚀 Starting streaming TTS processing...")
        
        def generate_streaming_audio():
            for audio_chunk in get_streaming_tts_audio_with_buffer(llm_response, TTS_VOICE, TTS_URL, save_path):
                yield audio_chunk
        
        # Return streaming response
        return Response(
            generate_streaming_audio(),
            mimetype='audio/wav',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Transfer-Encoding': 'chunked',
                'X-Physics-Section': section_number,
                'X-LLM-Response-Length': str(len(llm_response))
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
        'service': 'physics_book_server_streaming',
        'azure_endpoint': AZURE_ENDPOINT,
        'azure_model': AZURE_MODEL,
        'tts_url': TTS_URL,
        'audio_dir': AUDIO_DIR,
        'streaming': True
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
    
    # Test TTS (streaming endpoint)
    try:
        response = requests.get(f"{TTS_URL}/v1/streaming/info", timeout=10)
        results['tts'] = {
            'status': 'connected' if response.status_code == 200 else 'error',
            'url': TTS_URL,
            'status_code': response.status_code,
            'streaming_supported': True
        }
    except Exception as e:
        results['tts'] = {
            'status': 'error',
            'url': TTS_URL,
            'error': str(e)
        }
    
    return jsonify(results)

def main():
    parser = argparse.ArgumentParser(description="Physics Book TTS Web Server with Azure DeepSeek-V3 - STREAMING")
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
    
    print("🔬 Physics Book TTS Web Server with Azure DeepSeek-V3 - STREAMING")
    print("=" * 65)
    print(f"Server running on: http://{args.host}:{args.port}")
    print(f"Azure Endpoint: {AZURE_ENDPOINT}")
    print(f"Azure Model: {AZURE_MODEL}")
    print(f"TTS URL: {TTS_URL}")
    print(f"TTS Voice: {TTS_VOICE}")
    print(f"Audio files saved to: {AUDIO_DIR}")
    print(f"🎵 STREAMING MODE: Audio streams as it's generated!")
    print()
    print("Endpoints:")
    print(f"  POST /explain - Process physics section explanations (STREAMING)")
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