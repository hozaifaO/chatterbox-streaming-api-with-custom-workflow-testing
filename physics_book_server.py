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
from typing import Optional
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app for web server
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"
TTS_URL = "http://localhost:5001"
TTS_VOICE = "alloy"

def clean_text_for_tts(text: str) -> str:
    """Remove emojis and clean text for TTS"""
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
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
    
    return re.sub(r'\s+', ' ', text).strip()

def stream_from_ollama(prompt: str, model: str, base_url: str, system_prompt: Optional[str] = None):
    """Stream text from Ollama"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code}")
            return
            
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        logger.error(f"Error streaming from Ollama: {e}")

def get_tts_audio(text: str, voice: str, tts_url: str) -> Optional[bytes]:
    """Get audio from TTS service"""
    clean_text = clean_text_for_tts(text)
    if not clean_text.strip():
        return None
    
    logger.info(f"Converting to speech: '{clean_text[:50]}...'")
    
    payload = {
        "input": clean_text,
        "voice": voice,
        "model": "tts-1"
    }
    
    try:
        response = requests.post(
            f"{tts_url}/v1/audio/speech",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            audio_data = response.content
            if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                return audio_data
            else:
                logger.warning("Invalid WAV data from TTS")
                return None
        else:
            logger.error(f"TTS API error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"TTS request error: {e}")
        return None

class TextChunker:
    """Text chunker optimized for TTS streaming"""
    
    def __init__(self, min_size=20, max_size=150):
        self.buffer = ""
        self.min_size = min_size
        self.max_size = max_size
        
    def add_text(self, text: str) -> list[str]:
        """Add text and return complete chunks ready for TTS"""
        self.buffer += text
        chunks = []
        
        # Look for natural breaking points
        while True:
            # First priority: sentence endings
            sentence_match = re.search(r'([.!?])\s+', self.buffer)
            if sentence_match and sentence_match.end() >= self.min_size:
                end_pos = sentence_match.end()
                chunk = self.buffer[:end_pos].strip()
                
                if len(chunk) >= self.min_size and re.search(r'[a-zA-Z]', chunk):
                    chunks.append(chunk)
                    self.buffer = self.buffer[end_pos:].strip()
                    continue
            
            # Second priority: comma breaks (for longer text)
            if len(self.buffer) > self.max_size:
                comma_match = re.search(r',\s+', self.buffer)
                if comma_match and comma_match.end() >= self.min_size:
                    end_pos = comma_match.end()
                    chunk = self.buffer[:end_pos].strip()
                    
                    if len(chunk) >= self.min_size and re.search(r'[a-zA-Z]', chunk):
                        chunks.append(chunk)
                        self.buffer = self.buffer[end_pos:].strip()
                        continue
            
            # No suitable break found
            break
        
        return chunks
    
    def flush(self) -> Optional[str]:
        """Get remaining text in buffer"""
        if self.buffer.strip() and len(self.buffer.strip()) >= self.min_size:
            if re.search(r'[a-zA-Z]', self.buffer):
                text = self.buffer.strip()
                self.buffer = ""
                return text
        return None

# Web server endpoints
@app.route('/explain', methods=['POST'])
def explain_section():
    """Endpoint to handle section explanation requests from frontend"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        section_text = data['text']
        section_number = data.get('section', 'Unknown')
        
        logger.info(f"Processing explanation request for section {section_number}")
        
        # System prompt for creating natural audio explanations
        system_prompt = """You are an expert physics teacher creating engaging audio explanations for students. 
        
        Your task is to take the provided physics content and transform it into a clear, conversational explanation 
        that works perfectly for audio narration.
        
        Guidelines:
        - Use simple, conversational language as if talking to a student
        - Explain technical terms when you first use them
        - Use analogies and real-world examples to make concepts relatable
        - Break down mathematical equations step by step
        - Structure your explanation logically with smooth transitions
        - Maintain an enthusiastic but professional teaching tone
        - Avoid overly long sentences - keep them audio-friendly
        - Do not use emojis, special characters, or visual references
        - Focus on helping the listener truly understand the concept
        
        Think of this as creating a mini physics lesson that someone could listen to while walking or driving.
        Make the physics come alive through your explanation!"""
        
        def generate_audio_stream():
            try:
                chunker = TextChunker(min_size=25, max_size=120)
                chunk_count = 0
                
                logger.info("Starting LLM processing...")
                
                # Stream from LLM and convert to audio in real-time
                for text_chunk in stream_from_ollama(
                    section_text, 
                    OLLAMA_MODEL, 
                    OLLAMA_URL, 
                    system_prompt
                ):
                    # Add to chunker and get ready chunks
                    ready_chunks = chunker.add_text(text_chunk)
                    
                    # Convert each ready chunk to audio and stream it
                    for chunk in ready_chunks:
                        chunk_count += 1
                        logger.info(f"Processing audio chunk {chunk_count}")
                        
                        audio_data = get_tts_audio(chunk, TTS_VOICE, TTS_URL)
                        if audio_data:
                            yield audio_data
                        else:
                            logger.warning(f"Failed to generate audio for chunk {chunk_count}")
                
                # Process any remaining text in buffer
                final_chunk = chunker.flush()
                if final_chunk:
                    chunk_count += 1
                    logger.info(f"Processing final audio chunk {chunk_count}")
                    
                    audio_data = get_tts_audio(final_chunk, TTS_VOICE, TTS_URL)
                    if audio_data:
                        yield audio_data
                
                logger.info(f"Completed processing {chunk_count} audio chunks")
                        
            except Exception as e:
                logger.error(f"Error in audio stream generation: {e}")
                # Return a simple error audio message
                error_message = "Sorry, there was an error generating the explanation."
                error_audio = get_tts_audio(error_message, TTS_VOICE, TTS_URL)
                if error_audio:
                    yield error_audio
        
        return Response(
            generate_audio_stream(),
            mimetype='audio/wav',
            headers={
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in explain_section endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Physics Book TTS Server is running',
        'services': {
            'ollama_url': OLLAMA_URL,
            'ollama_model': OLLAMA_MODEL,
            'tts_url': TTS_URL,
            'tts_voice': TTS_VOICE
        }
    })

@app.route('/test-services', methods=['GET'])
def test_services():
    """Test connectivity to Ollama and TTS services"""
    results = {}
    
    # Test Ollama
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        results['ollama'] = {
            'status': 'connected' if response.status_code == 200 else 'error',
            'status_code': response.status_code
        }
    except Exception as e:
        results['ollama'] = {'status': 'error', 'error': str(e)}
    
    # Test TTS
    try:
        test_payload = {"input": "test", "voice": TTS_VOICE, "model": "tts-1"}
        response = requests.post(
            f"{TTS_URL}/v1/audio/speech",
            json=test_payload,
            timeout=5
        )
        results['tts'] = {
            'status': 'connected' if response.status_code == 200 else 'error',
            'status_code': response.status_code
        }
    except Exception as e:
        results['tts'] = {'status': 'error', 'error': str(e)}
    
    return jsonify(results)

def main():
    parser = argparse.ArgumentParser(description="Physics Book TTS Web Server")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--ollama-model", default="gemma3:latest", help="Ollama model")
    parser.add_argument("--tts-url", default="http://localhost:5001", help="TTS URL")
    parser.add_argument("--voice", default="alloy", help="TTS voice")
    parser.add_argument("--host", default="localhost", help="Web server host")
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
        print("\n\nShutting down server. Goodbye!")

if __name__ == "__main__":
    main()