#!/usr/bin/env python3
"""
Real-time Streaming Emotional TTS Chat - FIXED VERSION
Properly handles chunked WAV streaming from ChatterboxTTS API
"""

import json
import re
import argparse
import logging
import tempfile
import os
import queue
import threading
import time
import wave
import io
from typing import Optional, Dict

import requests

# Try to import better audio playback
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
    print("✅ Using sounddevice for real-time audio playback")
except ImportError:
    try:
        import winsound
        AUDIO_AVAILABLE = "winsound"
        print("⚠️  Using winsound (install sounddevice for better streaming: pip install sounddevice)")
    except ImportError:
        AUDIO_AVAILABLE = False
        print("❌ No audio playback available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [Keep all the existing emotion and voice definitions...]
BASE_EMOTIONS = {
    "excited": {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 1.0},
    "happy": {"exaggeration": 0.7, "cfg_weight": 0.4, "temperature": 0.9},
    "enthusiastic": {"exaggeration": 0.75, "cfg_weight": 0.35, "temperature": 0.95},
    "sad": {"exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.6},
    "angry": {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 0.9},
    "frustrated": {"exaggeration": 0.7, "cfg_weight": 0.35, "temperature": 0.85},
    "calm": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.7},
    "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    "confused": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.7},
    "surprised": {"exaggeration": 0.6, "cfg_weight": 0.4, "temperature": 0.85},
    "tired": {"exaggeration": 0.2, "cfg_weight": 0.8, "temperature": 0.5},
    "worried": {"exaggeration": 0.35, "cfg_weight": 0.65, "temperature": 0.65},
}

VOICE_CHARACTERISTICS = {
    "nova": {"exaggeration_modifier": 0.05, "cfg_weight_modifier": -0.05, "temperature_modifier": 0.05},
    "shimmer": {"exaggeration_modifier": -0.1, "cfg_weight_modifier": 0.1, "temperature_modifier": -0.1},
    "onyx": {"exaggeration_modifier": 0.0, "cfg_weight_modifier": -0.02, "temperature_modifier": -0.05},
    "echo": {"exaggeration_modifier": -0.05, "cfg_weight_modifier": 0.05, "temperature_modifier": -0.05},
    "fable": {"exaggeration_modifier": 0.1, "cfg_weight_modifier": 0.0, "temperature_modifier": 0.1},
    "alloy": {"exaggeration_modifier": 0.0, "cfg_weight_modifier": 0.0, "temperature_modifier": 0.0}
}

DEFAULT_VOICE_MAPPING = {
    "excited": "nova", "happy": "nova", "enthusiastic": "nova",
    "sad": "shimmer", "tired": "shimmer", "worried": "shimmer",
    "angry": "onyx", "frustrated": "onyx",
    "confused": "echo", "surprised": "fable",
    "neutral": "alloy", "calm": "alloy"
}

class WAVChunkParser:
    """Parser to extract complete WAV files from chunked HTTP stream"""
    
    def __init__(self):
        self.buffer = bytearray()
        self.expecting_size = None
        
    def add_data(self, data: bytes) -> list:
        """Add data and return list of complete WAV files"""
        self.buffer.extend(data)
        complete_wavs = []
        
        while True:
            wav_file = self._extract_next_wav()
            if wav_file:
                complete_wavs.append(bytes(wav_file))
            else:
                break
                
        return complete_wavs
    
    def _extract_next_wav(self) -> Optional[bytearray]:
        """Extract the next complete WAV file from buffer"""
        if len(self.buffer) < 8:
            return None
            
        # Look for RIFF header
        riff_pos = self.buffer.find(b'RIFF')
        if riff_pos == -1:
            return None
            
        # Remove any data before RIFF header
        if riff_pos > 0:
            self.buffer = self.buffer[riff_pos:]
            
        # Check if we have enough data for the header
        if len(self.buffer) < 8:
            return None
            
        # Read the file size from RIFF header
        try:
            file_size = int.from_bytes(self.buffer[4:8], byteorder='little')
            total_size = file_size + 8  # +8 for RIFF header itself
            
            # Check if we have the complete file
            if len(self.buffer) >= total_size:
                complete_wav = self.buffer[:total_size]
                self.buffer = self.buffer[total_size:]
                return complete_wav
            else:
                return None
                
        except Exception:
            # If we can't read the size, remove the bad RIFF and try again
            self.buffer = self.buffer[4:]
            return None

class RealTimeAudioPlayer:
    """Real-time audio player that properly handles WAV chunks"""
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.playing = False
        self.player_thread = None
        self.chunk_count = 0
        self.total_duration = 0.0
        
    def start(self):
        """Start the real-time audio player thread"""
        if self.playing:
            return
            
        self.playing = True
        
        if AUDIO_AVAILABLE == True:  # sounddevice available
            self.player_thread = threading.Thread(target=self._sounddevice_player_worker)
        elif AUDIO_AVAILABLE == "winsound":  # winsound fallback
            self.player_thread = threading.Thread(target=self._winsound_player_worker)
        else:
            print("❌ No audio playback available")
            return
            
        self.player_thread.daemon = True
        self.player_thread.start()
        print("🎵 Real-time audio playback started!")
    
    def add_chunk(self, wav_data: bytes, debug: bool = False):
        """Add audio chunk for immediate playback"""
        if not self.playing:
            return
            
        self.chunk_count += 1
        
        # Calculate chunk duration
        try:
            duration = self._get_wav_duration(wav_data)
            self.total_duration += duration
            
            if debug:
                print(f"[🎵 Chunk {self.chunk_count}: {len(wav_data)} bytes, ~{duration:.3f}s duration]")
        except Exception as e:
            if debug:
                print(f"[🎵 Chunk {self.chunk_count}: {len(wav_data)} bytes, duration calc failed: {e}]")
        
        # Queue for immediate playback
        self.audio_queue.put(wav_data)
    
    def _get_wav_duration(self, wav_data: bytes) -> float:
        """Get duration of WAV audio data"""
        try:
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    return frames / sample_rate
        except Exception:
            return 0.0
    
    def _sounddevice_player_worker(self):
        """Worker thread for sounddevice playback (preferred)"""
        while self.playing:
            try:
                wav_data = self.audio_queue.get(timeout=1.0)
                if wav_data is None:  # Sentinel to stop
                    break
                    
                # Convert WAV data to numpy array for sounddevice
                audio_np = self._wav_to_numpy(wav_data)
                if audio_np is not None:
                    sd.play(audio_np, self.sample_rate, blocking=True)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def _winsound_player_worker(self):
        """Worker thread for winsound playback (fallback)"""
        while self.playing:
            try:
                wav_data = self.audio_queue.get(timeout=1.0)
                if wav_data is None:  # Sentinel to stop
                    break
                    
                # Play using winsound (blocking)
                self._play_with_winsound(wav_data)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def _wav_to_numpy(self, wav_data: bytes) -> Optional[np.ndarray]:
        """Convert WAV bytes to numpy array for sounddevice"""
        try:
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    # Handle both mono and stereo
                    if wav_file.getnchannels() == 1:
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32).reshape(-1, wav_file.getnchannels()) / 32768.0
                    return audio_np
        except Exception as e:
            print(f"WAV conversion error: {e}")
            return None
    
    def _play_with_winsound(self, wav_data: bytes):
        """Play audio using winsound (Windows only)"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(wav_data)
                tmp_file.flush()
                tmp_path = tmp_file.name
            
            import winsound
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            print(f"Winsound playback error: {e}")
    
    def stop(self):
        """Stop the audio player and wait for completion"""
        if not self.playing:
            return
            
        # Wait for current queue to finish
        print("🎵 Finishing audio playback...")
        self.audio_queue.join()
        
        # Signal stop
        self.audio_queue.put(None)
        self.playing = False
        
        if self.player_thread:
            self.player_thread.join(timeout=2.0)
        
        print(f"🎵 Playback complete! {self.chunk_count} chunks, {self.total_duration:.2f}s total")

def get_emotion_parameters(emotion: str, voice: str) -> Dict:
    """Generate emotion parameters for any voice/emotion combination"""
    if emotion not in BASE_EMOTIONS:
        emotion = "neutral"
    
    base_params = BASE_EMOTIONS[emotion].copy()
    
    if voice not in VOICE_CHARACTERISTICS:
        voice = "alloy"
    
    voice_char = VOICE_CHARACTERISTICS[voice]
    
    modified_params = {
        "voice": voice,
        "exaggeration": max(0.0, min(1.0, base_params["exaggeration"] + voice_char["exaggeration_modifier"])),
        "cfg_weight": max(0.0, min(1.0, base_params["cfg_weight"] + voice_char["cfg_weight_modifier"])),
        "temperature": max(0.1, min(1.5, base_params["temperature"] + voice_char["temperature_modifier"]))
    }
    
    return modified_params

def get_default_voice_for_emotion(emotion: str) -> str:
    """Get the default voice for an emotion"""
    return DEFAULT_VOICE_MAPPING.get(emotion, "alloy")

def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return re.sub(r'\s+', ' ', emoji_pattern.sub('', text)).strip()

def stream_from_ollama(prompt: str, model: str, base_url: str, system_prompt: Optional[str] = None):
    """Stream text from Ollama"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7, "top_p": 0.9}
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, stream=True, timeout=30)
        
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

def detect_emotion_from_text(text: str, debug: bool = False) -> str:
    """Detect emotion from text"""
    text_lower = text.lower()
    
    emotion_keywords = [
        ("excited", ['excited', 'amazing', 'incredible', 'fantastic', 'awesome']),
        ("angry", ['furious', 'outrageous', 'ridiculous']),
        ("sad", ['sad', 'sorry', 'unfortunate', 'terrible']),
        ("surprised", ['surprised', 'wow', 'unexpected']),
        ("confused", ['confused', "don't understand", 'unclear']),
        ("tired", ['tired', 'exhausted', 'weary']),
        ("happy", ['happy', 'wonderful', 'great', 'excellent']),
    ]
    
    for emotion, keywords in emotion_keywords:
        for keyword in keywords:
            if keyword in text_lower:
                if debug:
                    print(f"\n[🎭 Emotion: '{emotion}' ← '{keyword}']", end="")
                return emotion
    
    return "neutral"

def get_realtime_streaming_tts(text: str, tts_url: str, emotion: str = "neutral", voice_override: str = None, debug: bool = False):
    """Get real-time streaming TTS with proper WAV chunk handling"""
    clean_text = clean_text_for_tts(text)
    if not clean_text.strip():
        return
    
    # Determine voice and parameters
    voice_to_use = voice_override or get_default_voice_for_emotion(emotion)
    params = get_emotion_parameters(emotion, voice_to_use)
    
    if debug:
        print(f"\n[🎤 REAL-TIME STREAMING {voice_to_use.upper()}: {emotion}]")
        print(f"[📝 Text: '{clean_text}' ({len(clean_text)} chars)]")
        print(f"[⚙️  Params: E:{params['exaggeration']:.2f} C:{params['cfg_weight']:.2f} T:{params['temperature']:.2f}]")
    else:
        indicator = "*" if voice_override else ""
        print(f"\n[🎭 STREAMING {emotion}/{voice_to_use}{indicator}]", end="")
    
    # Setup real-time audio player and WAV parser
    audio_player = RealTimeAudioPlayer(sample_rate=24000)
    wav_parser = WAVChunkParser()
    
    # Request payload for streaming
    payload = {
        "input": clean_text,
        "model": "tts-1",
        "voice": params["voice"],
        "exaggeration": params["exaggeration"],
        "cfg_weight": params["cfg_weight"],
        "temperature": params["temperature"],
        "stream": True,
        "chunk_size": 25,  # Smaller chunks for lower latency
        "low_latency": True,
        "return_metrics": debug
    }
    
    try:
        if debug:
            print("[🚀 Starting real-time streaming TTS...]")
        
        # Start audio player
        audio_player.start()
        
        # Stream audio chunks
        response = requests.post(
            f"{tts_url}/v1/audio/speech/stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=120
        )
        
        if debug:
            print(f"[📡 Response: {response.status_code}]")
        
        if response.status_code == 200:
            if debug:
                print("[📦 Processing real-time audio chunks...]")
            
            complete_audio_chunks = []
            http_chunk_count = 0
            wav_chunk_count = 0
            
            # Process HTTP chunks and extract complete WAV files
            for http_chunk in response.iter_content(chunk_size=8192):
                if http_chunk:
                    http_chunk_count += 1
                    
                    # Parse HTTP chunk to extract complete WAV files
                    complete_wavs = wav_parser.add_data(http_chunk)
                    
                    # Play each complete WAV file immediately
                    for wav_data in complete_wavs:
                        wav_chunk_count += 1
                        complete_audio_chunks.append(wav_data)
                        
                        # Play chunk immediately
                        audio_player.add_chunk(wav_data, debug=debug)
                        
                        if wav_chunk_count == 1 and not debug:
                            print("🎵", end="", flush=True)  # Indicate audio started
            
            if debug:
                print(f"[✅ Streaming complete: {http_chunk_count} HTTP chunks, {wav_chunk_count} WAV chunks]")
            
            # Wait for all audio to finish playing
            audio_player.stop()
            
            # Save complete concatenated audio if requested
            if complete_audio_chunks and debug:
                total_bytes = sum(len(chunk) for chunk in complete_audio_chunks)
                print(f"[💾 Complete audio: {wav_chunk_count} WAV chunks, {total_bytes} bytes total]")
            
        else:
            audio_player.stop()
            logger.error(f"Streaming TTS error: {response.status_code}")
            if debug:
                print(f"[❌ HTTP {response.status_code} - stopping audio player]")
            
    except Exception as e:
        audio_player.stop()
        logger.error(f"Real-time streaming error: {e}")
        if debug:
            print(f"[❌ Error: {e}]")

def get_enhanced_emotional_system_prompt():
    """System prompt for emotional responses"""
    return """You are a helpful AI assistant with natural emotional responses.

Express emotions naturally using these keywords:
- Excitement: "amazing", "incredible", "fantastic", "awesome"
- Happiness: "wonderful", "great", "excellent" 
- Sadness: "sad", "sorry", "unfortunate", "terrible"
- Anger: "outrageous", "furious", "ridiculous"
- Surprise: "wow", "unexpected", "surprising"
- Confusion: "confused", "unclear", "don't understand"

Keep responses natural and conversational (1-2 sentences). Never use emojis.
Your voice will automatically adapt to match your emotions with real-time streaming audio!"""

def main():
    parser = argparse.ArgumentParser(description="Real-time Streaming Emotional TTS Chat - FIXED")
    parser.add_argument("--llm-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--llm-model", default="gemma2:latest", help="LLM model")
    parser.add_argument("--tts-url", default="http://localhost:5001", help="TTS URL")
    parser.add_argument("--voice", default=None, help="Override voice (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    voice_override = args.voice
    debug_mode = args.debug
    system_prompt = get_enhanced_emotional_system_prompt()
    
    print("🎭 Real-time Streaming Emotional TTS Chat (FIXED)")
    print("=" * 55)
    print(f"LLM Model: {args.llm_model}")
    print(f"Voice Mode: {'Auto-Selected' if not voice_override else f'{voice_override.upper()} Override'}")
    print(f"Audio: Real-time streaming with proper WAV parsing")
    print(f"Playback: {AUDIO_AVAILABLE}")
    print(f"Debug: {'ON' if debug_mode else 'OFF'}")
    print("🎵 Audio plays in real-time as complete WAV chunks arrive!")
    print("🔧 Fixed: Proper WAV chunk parsing and playback")
    print()
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
            
            # Handle simple commands
            if user_input.startswith('!'):
                if user_input == '!help':
                    print("Commands: !help, !debug on/off, !voice <name>, !voice auto, quit")
                    continue
                elif user_input.startswith('!debug'):
                    if 'on' in user_input:
                        debug_mode = True
                        print("🔍 Debug mode: ON")
                    else:
                        debug_mode = False
                        print("🔍 Debug mode: OFF")
                    continue
                elif user_input.startswith('!voice'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        if parts[1] == 'auto':
                            voice_override = None
                            print("🎤 Voice: AUTO")
                        else:
                            voice_override = parts[1]
                            print(f"🎤 Voice: {voice_override.upper()}")
                    continue
            
            print("AI: ", end="", flush=True)
            
            # Stream from LLM
            full_response = ""
            for text_chunk in stream_from_ollama(user_input, args.llm_model, args.llm_url, system_prompt):
                print(text_chunk, end="", flush=True)
                full_response += text_chunk
            
            # Real-time streaming TTS with fixed WAV handling
            if full_response.strip():
                detected_emotion = detect_emotion_from_text(full_response, debug_mode)
                
                # Start real-time streaming TTS
                get_realtime_streaming_tts(
                    full_response, 
                    args.tts_url,
                    detected_emotion, 
                    voice_override=voice_override, 
                    debug=debug_mode
                )
            
            print()  # New line after complete response
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main()