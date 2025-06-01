#!/usr/bin/env python3
"""
Ultra Simple Real-time LLM + TTS Chat
Fixed for Windows audio issues - uses winsound
"""

import json
import re
import argparse
import logging
import tempfile
import os
from typing import Optional

import requests

# Windows audio
import winsound

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            timeout=30
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

def get_tts_audio_and_play(text: str, voice: str, tts_url: str):
    """Get audio from TTS and play immediately"""
    clean_text = clean_text_for_tts(text)
    if not clean_text.strip():
        return
    
    # Debug: Show what's being sent to TTS
    print(f"\n[TTS: '{clean_text}']", end="", flush=True)
    
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
                # Play using Windows winsound (simple and reliable)
                play_audio_windows(audio_data)
            else:
                logger.warning("Invalid WAV data from TTS")
        else:
            logger.error(f"TTS API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"TTS request error: {e}")

def play_audio_windows(wav_data: bytes):
    """Play audio on Windows using winsound"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(wav_data)
            tmp_path = tmp_file.name
        
        # Play with winsound (built into Windows Python)
        winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Audio playback error: {e}")

class SimpleTextChunker:
    """Simple text chunker for TTS - improved logic"""
    
    def __init__(self, min_size=15, max_size=100):
        self.buffer = ""
        self.min_size = min_size
        self.max_size = max_size
        
    def add_text(self, text: str) -> list[str]:
        """Add text and return complete chunks"""
        self.buffer += text
        chunks = []
        
        # Only chunk on complete sentences to avoid weird fragments
        while True:
            # Look for sentence endings
            match = re.search(r'([.!?])\s+', self.buffer)
            if match:
                # Found a complete sentence
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                
                # Only send if it's a reasonable length and has actual words
                if len(sentence) >= self.min_size and re.search(r'[a-zA-Z]', sentence):
                    chunks.append(sentence)
                
                # Remove processed text from buffer
                self.buffer = self.buffer[end_pos:].strip()
            else:
                # No complete sentence found
                break
        
        # If buffer is getting very long, look for comma breaks
        if len(self.buffer) > self.max_size:
            comma_match = re.search(r',\s+', self.buffer)
            if comma_match:
                end_pos = comma_match.end()
                chunk = self.buffer[:end_pos].strip()
                if len(chunk) >= self.min_size and re.search(r'[a-zA-Z]', chunk):
                    chunks.append(chunk)
                self.buffer = self.buffer[end_pos:].strip()
        
        return chunks
    
    def flush(self) -> Optional[str]:
        """Get remaining text"""
        if self.buffer.strip() and len(self.buffer.strip()) >= self.min_size:
            # Only return if it has actual words and reasonable length
            if re.search(r'[a-zA-Z]', self.buffer):
                text = self.buffer.strip()
                self.buffer = ""
                return text
        return None

def main():
    parser = argparse.ArgumentParser(description="Ultra Simple Real-time LLM + TTS Chat")
    parser.add_argument("--llm-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--llm-model", default="gemma2:latest", help="LLM model")
    parser.add_argument("--tts-url", default="http://localhost:5001", help="TTS URL")
    parser.add_argument("--voice", default="alloy", help="TTS voice")
    parser.add_argument("--system-prompt", help="System prompt")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Default system prompt
    system_prompt = args.system_prompt or """You are a helpful AI assistant. 
    Keep responses natural and conversational. Do not use emojis. 
    Keep responses reasonably short."""
    
    print("🎤 Ultra Simple Real-time AI Chat")
    print("=" * 35)
    print("Type 'quit' to exit")
    print()
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            print("AI: ", end="", flush=True)
            
            # Process conversation
            chunker = SimpleTextChunker()
            
            # Stream from LLM and process in chunks
            for text_chunk in stream_from_ollama(user_input, args.llm_model, args.llm_url, system_prompt):
                print(text_chunk, end="", flush=True)
                
                # Add to chunker and get ready chunks
                ready_chunks = chunker.add_text(text_chunk)
                
                # Send each ready chunk to TTS and play
                for chunk in ready_chunks:
                    get_tts_audio_and_play(chunk, args.voice, args.tts_url)
            
            # Process any remaining text
            final_chunk = chunker.flush()
            if final_chunk:
                get_tts_audio_and_play(final_chunk, args.voice, args.tts_url)
            
            print()  # New line after response
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")

if __name__ == "__main__":
    main()