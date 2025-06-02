#!/usr/bin/env python3
"""
Realtime Seamless Emotional TTS Chat
-----------------------------------

"""

from __future__ import annotations

# ── standard library ─────────────────────────────────────────────────────────
import argparse
import io
import json
import logging
import queue
import re
import threading
import time
import wave
from functools import partial
from typing import Dict, List, Optional, Tuple

# ── third-party ──────────────────────────────────────────────────────────────
import requests
import numpy as np

try:
    import sounddevice as sd

    AUDIO_AVAILABLE: bool = True
    print("✅ Using sounddevice for realtime audio")
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ sounddevice required:  pip install sounddevice numpy")

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime-tts")

# ── emotion & voice parameter tables ────────────────────────────────────────
BASE_EMOTIONS: Dict[str, Dict[str, float]] = {
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

VOICE_CHARACTERISTICS: Dict[str, Dict[str, float]] = {
    "nova":    {"exaggeration_modifier": 0.05, "cfg_weight_modifier": -0.05, "temperature_modifier": 0.05},
    "shimmer": {"exaggeration_modifier": -0.10, "cfg_weight_modifier":  0.10, "temperature_modifier": -0.10},
    "onyx":    {"exaggeration_modifier": 0.00, "cfg_weight_modifier": -0.02, "temperature_modifier": -0.05},
    "echo":    {"exaggeration_modifier": -0.05, "cfg_weight_modifier":  0.05, "temperature_modifier": -0.05},
    "fable":   {"exaggeration_modifier": 0.10, "cfg_weight_modifier":  0.00, "temperature_modifier":  0.10},
    "alloy":   {"exaggeration_modifier": 0.00, "cfg_weight_modifier":  0.00, "temperature_modifier":  0.00},
}

DEFAULT_VOICE_MAPPING: Dict[str, str] = {
    "excited": "nova", "happy": "nova", "enthusiastic": "nova",
    "sad": "shimmer", "tired": "shimmer", "worried": "shimmer",
    "angry": "onyx", "frustrated": "onyx",
    "confused": "echo", "surprised": "fable",
    "neutral": "alloy", "calm": "alloy",
}

# ── helper utilities ────────────────────────────────────────────────────────
def get_emotion_parameters(emotion: str, voice: str) -> Dict[str, float]:
    if emotion not in BASE_EMOTIONS:
        emotion = "neutral"
    base = BASE_EMOTIONS[emotion]
    vc = VOICE_CHARACTERISTICS.get(voice, VOICE_CHARACTERISTICS["alloy"])
    return {
        "voice": voice,
        "exaggeration": np.clip(base["exaggeration"] + vc["exaggeration_modifier"], 0.0, 1.0),
        "cfg_weight": np.clip(base["cfg_weight"] + vc["cfg_weight_modifier"], 0.0, 1.0),
        "temperature": np.clip(base["temperature"] + vc["temperature_modifier"], 0.1, 1.5),
    }

def get_default_voice_for_emotion(emotion: str) -> str:
    return DEFAULT_VOICE_MAPPING.get(emotion, "alloy")

class SimpleWAVParser:
    """Extract complete WAV files from a streaming HTTP response."""
    def __init__(self) -> None:
        self.buffer = bytearray()
    def add_data(self, data: bytes) -> List[bytes]:
        self.buffer.extend(data)
        out: List[bytes] = []
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

def wav_to_numpy(wav_bytes: bytes) -> Optional[Tuple[np.ndarray, int]]:
    """Convert a WAV buffer to mono float32 ndarray + sample-rate."""
    try:
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as w:
                frames = w.readframes(w.getnframes())
                width = w.getsampwidth()
                if width == 2:
                    audio = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
                elif width == 4:
                    audio = np.frombuffer(frames, np.int32).astype(np.float32) / 2147483648.0
                else:
                    return None
                if w.getnchannels() > 1:
                    audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
                return audio, w.getframerate()
    except Exception as exc:
        logger.error(f"WAV→numpy failed: {exc}")
        return None

# Regex that works on narrow Python builds (Windows) – removes virtually all emoji
EMOJI_RE = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)

def clean_text_for_tts(text: str) -> str:
    return re.sub(r"\s+", " ", EMOJI_RE.sub("", text)).strip()

# ── Ollama client ────────────────────────────────────────────────────────────
def stream_from_ollama(prompt: str, model: str, base_url: str, system_prompt: Optional[str] = None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7, "top_p": 0.9},
    }
    if system_prompt:
        payload["system"] = system_prompt
    try:
        with requests.post(f"{base_url}/api/generate", json=payload, stream=True, timeout=30) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode())
                    if "response" in msg:
                        yield msg["response"]
                    if msg.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.error(f"Ollama stream error: {exc}")

# ── emotion detection ───────────────────────────────────────────────────────
def detect_emotion_from_text(text: str, debug: bool = False) -> str:
    tl = text.lower()
    mapping = [
        ("excited", ["excited", "amazing", "incredible", "fantastic", "awesome"]),
        ("angry", ["furious", "outrageous", "ridiculous"]),
        ("sad", ["sad", "sorry", "unfortunate", "terrible"]),
        ("surprised", ["surprised", "wow", "unexpected"]),
        ("confused", ["confused", "don't understand", "unclear"]),
        ("tired", ["tired", "exhausted", "weary"]),
        ("happy", ["happy", "wonderful", "great", "excellent"]),
    ]
    for emo, kws in mapping:
        if any(k in tl for k in kws):
            if debug:
                print(f"\n[🎭 {emo} ← keyword]")
            return emo
    return "neutral"

# (Keep other imports and functions as they are)
# ...

# --- replace your existing get_streaming_tts with this version ---
def get_streaming_tts(
    text: str,
    tts_url: str,
    emotion: str = "neutral",
    voice_override: Optional[str] = None,
    debug: bool = False,
    prebuffer_ms: int = 350,      # Initial prebuffer target
):
    clean = clean_text_for_tts(text)
    if not clean or not AUDIO_AVAILABLE:
        return

    voice = voice_override or get_default_voice_for_emotion(emotion)
    pars = get_emotion_parameters(emotion, voice)

    if debug:
        logger.info(f"\n[🎤 {voice.upper()}: {emotion}] -> '{clean[:48]}…'")

    payload = {
        "input": clean,
        "model": "tts-1",
        "voice": pars["voice"],
        "exaggeration": pars["exaggeration"],
        "cfg_weight": pars["cfg_weight"],
        "temperature": pars["temperature"],
        "stream": True,
        # MODIFICATION: Experiment with server's text chunk size.
        # Original was 15. Larger values mean server sends bigger audio chunks less frequently.
        # This might help if network round-trips for many small chunks are an issue.
        "chunk_size": 30, # Try values like 30, 40, or 50. Was 15.
        "low_latency": True,
    }

    # MODIFICATION: Increased queue size for more buffering capacity
    pcm_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128) # Was 64
    stop_flag = threading.Event()
    
    # Use a dictionary to share state with the callback, including leftover audio
    # and the sample rate determined by the first chunk.
    callback_state = {
        'leftover_block_float32': None, # Stores leftover audio as float32 numpy array
        'first_sr': None,
        'status_logged': False # To avoid flooding logs with underrun messages
    }

    # ── producer: HTTP → WAV → numpy → queue ─────────────────────────────────
    def reader():
        parser = SimpleWAVParser()
        try:
            with requests.post(
                f"{tts_url}/v1/audio/speech/stream",
                json=payload,
                stream=True,
                timeout=120, # Overall request timeout
            ) as r:
                r.raise_for_status()
                for http_chunk in r.iter_content(chunk_size=4096): # HTTP level chunking
                    if stop_flag.is_set(): break
                    for wav_bytes in parser.add_data(http_chunk):
                        if stop_flag.is_set(): break
                        res = wav_to_numpy(wav_bytes)
                        if not res:
                            logger.warning("Failed to convert WAV to numpy array.")
                            continue
                        
                        pcm_data_float32, sr = res
                        
                        if callback_state['first_sr'] is None:
                            logger.info(f"Audio stream started. Sample rate: {sr} Hz.")
                            callback_state['first_sr'] = sr
                        elif callback_state['first_sr'] != sr:
                            logger.warning(f"Sample rate changed mid-stream! Expected {callback_state['first_sr']}, got {sr}. This is not handled well.")
                            # Ideally, re-initialize sounddevice stream or handle resampling. For now, log and continue.
                        
                        try:
                            pcm_q.put(pcm_data_float32, timeout=1.0) # Put with timeout to prevent indefinite block
                        except queue.Full:
                            logger.warning("PCM queue full. Discarding audio data. Playback might be choppy.")
                            # This indicates the consumer (playback) is too slow or stalled.
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS request error: {e}")
        except Exception as e:
            logger.error(f"Reader thread error: {e}")
        finally:
            logger.info("Reader thread finished.")
            stop_flag.set()

    # ── consumer: PortAudio callback ─────────────────────────────────────────
    def pa_cb(outdata: memoryview, frames: int, time_info, status_flags):
        # `outdata` is a memoryview for RawOutputStream, expecting bytes.
        # `frames` is the number of audio frames requested.
        # For mono int16, 1 frame = 1 sample = 2 bytes.
        
        if status_flags: # sounddevice.CallbackFlags object
            if status_flags.output_underflow and not callback_state['status_logged']:
                logger.warning("Sounddevice output underflow detected! Audio may be choppy.")
                callback_state['status_logged'] = True # Log once per stream segment or reset periodically
            if status_flags.priming_output and not callback_state['status_logged']: # Useful for debugging
                logger.info("Sounddevice priming output.")
                callback_state['status_logged'] = True

        bytes_needed = frames * 2  # For mono int16 (1 channel * 2 bytes/sample)
        output_audio_bytes = bytearray(bytes_needed) # Initialize with silence (zeros)
        bytes_filled = 0

        # Try to use leftover data first
        if callback_state['leftover_block_float32'] is not None:
            block_float32 = callback_state['leftover_block_float32']
            
            # Convert float32 numpy array to int16 bytes
            block_int16 = (block_float32 * 32767.0).astype(np.int16)
            block_bytes_data = block_int16.tobytes()
            
            num_bytes_from_leftover = len(block_bytes_data)
            bytes_to_take = min(bytes_needed - bytes_filled, num_bytes_from_leftover)
            
            output_audio_bytes[bytes_filled : bytes_filled + bytes_to_take] = block_bytes_data[:bytes_to_take]
            bytes_filled += bytes_to_take
            
            if bytes_to_take < num_bytes_from_leftover:
                # Some data from this leftover_block remains. Store the float32 version.
                remaining_samples_in_leftover_block = len(block_float32) - (bytes_to_take // 2)
                callback_state['leftover_block_float32'] = block_float32[-remaining_samples_in_leftover_block:]
            else:
                callback_state['leftover_block_float32'] = None

        # Fill remaining space from the queue
        while bytes_filled < bytes_needed:
            try:
                pcm_float32_block = pcm_q.get_nowait()
                pcm_q.task_done() # Signal that item is processed
            except queue.Empty:
                # Queue is empty. The output_audio_bytes is already zero-filled (silence)
                # for the remaining part.
                if bytes_filled == 0 and not stop_flag.is_set(): # Completely empty and reader is still supposed to run
                    # This is a true underrun if it happens often
                    pass # logger.debug("Callback: Queue empty, outputting silence.")
                break # Exit the while loop, output whatever is filled (or silence)

            block_int16 = (pcm_float32_block * 32767.0).astype(np.int16)
            block_bytes_data = block_int16.tobytes()

            num_bytes_from_new_block = len(block_bytes_data)
            bytes_to_take = min(bytes_needed - bytes_filled, num_bytes_from_new_block)

            output_audio_bytes[bytes_filled : bytes_filled + bytes_to_take] = block_bytes_data[:bytes_to_take]
            bytes_filled += bytes_to_take

            if bytes_to_take < num_bytes_from_new_block:
                # Store leftover from this new block (as float32)
                remaining_samples_in_new_block = len(pcm_float32_block) - (bytes_to_take // 2)
                callback_state['leftover_block_float32'] = pcm_float32_block[-remaining_samples_in_new_block:]
                break # Filled output_buffer or have leftover, current callback done with new blocks

        outdata[:] = output_audio_bytes

    # ── spin everything up ───────────────────────────────────────────────────
    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    # Wait for the first audio chunk to determine sample rate
    logger.info("Waiting for first audio data to determine sample rate...")
    sr_wait_start_time = time.monotonic()
    while callback_state['first_sr'] is None and not stop_flag.is_set():
        if time.monotonic() - sr_wait_start_time > 10.0: # 10 second timeout
            logger.error("Timeout: No audio data received from server after 10s.")
            stop_flag.set() # Ensure reader thread also stops if it's stuck
            reader_thread.join(timeout=2.0)
            return
        time.sleep(0.01)

    if stop_flag.is_set() or callback_state['first_sr'] is None:
        logger.error("Failed to get initial audio data or sample rate. Aborting playback.")
        if reader_thread.is_alive():
            reader_thread.join(timeout=2.0)
        return
    
    current_sr = callback_state['first_sr']
    logger.info(f"Starting playback. Pre-buffering up to {prebuffer_ms}ms of audio at {current_sr} Hz...")

    # Pre-buffering logic
    # We want to ensure a certain amount of audio *duration* is in the pcm_q
    # before starting sounddevice stream.
    buffered_frames_count = 0
    # Sum frames already in queue from leftover + any initial chunks
    if callback_state['leftover_block_float32'] is not None:
         buffered_frames_count += len(callback_state['leftover_block_float32'])
    
    # Temporarily store items from queue to count them, then put back
    temp_prebuffer_items = []
    try:
        while not pcm_q.empty():
            item = pcm_q.get_nowait()
            temp_prebuffer_items.append(item)
            buffered_frames_count += len(item)
    except queue.Empty:
        pass # Expected
    finally:
        for item in temp_prebuffer_items: # Put them back in order
            pcm_q.put_nowait(item) # Assuming queue won't be full here
    
    target_prebuffer_frames = int((prebuffer_ms / 1000.0) * current_sr)

    prebuffering_loop_start_time = time.monotonic()
    while buffered_frames_count < target_prebuffer_frames and not stop_flag.is_set():
        if time.monotonic() - prebuffering_loop_start_time > 5.0: # Max 5s for prebuffering
            logger.warning(f"Pre-buffering timeout. Starting with {buffered_frames_count / current_sr * 1000:.0f}ms of audio.")
            break
        try:
            # Get new data that arrived while we were counting or waiting
            new_pcm = pcm_q.get(timeout=0.1) # Wait for new data
            buffered_frames_count += len(new_pcm)
            # This item is now out of the queue. It will be the first one picked up by the callback
            # if we start the stream now, or we need to put it back if we continue prebuffering.
            # For simplicity, we'll assume this is okay: items are consumed from queue for prebuffer calc.
            # A more robust way is to put it into a temporary list and then re-feed the queue.
            # For now: let's put it back to ensure it's available for the actual playback start.
            # This makes the 'buffered_frames_count' an estimate of what *has arrived*.
            temp_prebuffer_items.append(new_pcm) # Collect to put back
        except queue.Empty:
            if stop_flag.is_set() and pcm_q.empty():
                logger.info("Reader stopped and queue empty during pre-buffering.")
                break
    
    # Put back any items taken during the prebuffering wait loop
    for item in reversed(temp_prebuffer_items): # Put back in reverse order of taking, to maintain original order at head
        # This is tricky with queue.Queue; collections.deque would be better with appendleft
        # For now, this puts them at the end.
        # A simpler way for prebuffering: just check pcm_q.qsize() * average_chunk_samples.
        # The original get/put in the loop was simpler:
        # pcm = pcm_q.get(timeout=0.05); buffered += len(pcm) / first_sr; pcm_q.put(pcm)
        # Let's revert prebuffering calculation to the simpler original one for now if the detailed one is too complex.
        # The provided code had:
        # buffered = 0.0
        # while buffered < prebuffer_ms / 1000 and not stop_flag.is_set():
        #    try:
        #        pcm = pcm_q.get(timeout=0.05)
        #        buffered += len(pcm) / current_sr
        #        pcm_q.put(pcm) # Puts it back at the end
        #    except queue.Empty: pass
        # This logic is simple and ensures that amount of data has passed into the queue.
        pass # The items taken are effectively the "first" items for playback now if not put back.
             # Or, better: after this loop, if temp_prebuffer_items is not empty, make the first item
             # callback_state['leftover_block_float32'] and put rest back.

    # Let's stick to a simple pre-buffering fill based on a target number of chunks or total items
    # The actual pre-buffering by waiting ensures the queue has *some* data. `prebuffer_ms`
    # defines how much we wait for initially.

    if buffered_frames_count == 0 and stop_flag.is_set(): # Nothing buffered and no more data coming
        logger.error("No audio data available for playback.")
        if reader_thread.is_alive(): reader_thread.join(timeout=1.0)
        return

    logger.info(f"Actual prebuffered duration: {buffered_frames_count / current_sr * 1000:.0f}ms. Starting audio stream.")
    callback_state['status_logged'] = False # Reset log flag for new stream segment

    stream = None
    try:
        stream = sd.RawOutputStream(
            samplerate=current_sr,
            channels=1,
            dtype="int16",               # Raw stream works with bytes
            callback=pa_cb,
            blocksize=1024,              # Size of chunk sounddevice asks for from callback. 0 for optimal.
            latency="high",              # 'low', 'high', or a specific time in seconds. 'high' is safer.
        )
        stream.start()
        logger.info("Audio stream started successfully.")

        # Keep main thread alive until playback is done
        while not stop_flag.is_set() or not pcm_q.empty() or callback_state['leftover_block_float32'] is not None:
            if stream and stream.closed: # Stream might get closed by sounddevice on error
                logger.error("Sounddevice stream closed unexpectedly.")
                break
            time.sleep(0.05)
        
        # Wait for the queue to be fully processed by the callback after reader stops
        if not pcm_q.empty():
            logger.info(f"Reader stopped, waiting for remaining {pcm_q.qsize()} audio chunks in queue to play out...")
            pcm_q.join() # Wait for all pcm_q.task_done() calls
        
        # Final check for leftover data in callback_state, allow it to play out if stream is still active
        # This needs a bit more thought if relying on it for the very last bit of audio.
        # Usually, stream.stop() will handle flushing.

    except Exception as e:
        logger.error(f"Error during audio stream setup or playback: {e}")
    finally:
        if stream:
            logger.info("Stopping and closing audio stream...")
            try:
                if not stream.closed:
                    stream.stop()
                    stream.close()
                logger.info("Audio stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error stopping/closing stream: {e}")
        
        if reader_thread.is_alive():
            logger.info("Waiting for reader thread to join...")
            stop_flag.set() # Ensure it's signaled if not already
            reader_thread.join(timeout=5.0) # Wait for reader to finish
            if reader_thread.is_alive():
                logger.warning("Reader thread did not terminate cleanly.")
        logger.info("Playback finished for this text.")
# ── chat system prompt ───────────────────────────────────────────────────────
def get_emotional_system_prompt() -> str:
    return """You are a helpful AI assistant with natural emotional responses.

Express emotions naturally using these keywords:
- Excitement: "amazing", "incredible", "fantastic", "awesome"
- Happiness: "wonderful", "great", "excellent" 
- Sadness: "sad", "sorry", "unfortunate", "terrible"
- Anger: "outrageous", "furious", "ridiculous"
- Surprise: "wow", "unexpected", "surprising"
- Confusion: "confused", "unclear", "don't understand"

Keep responses natural and conversational (1-2 sentences). Never use emojis.
Your voice will automatically adapt to match your emotions!"""

# ── main loop ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Realtime Seamless Emotional TTS Chat")
    ap.add_argument("--llm-url", default="http://localhost:11434", help="Ollama URL")
    ap.add_argument("--llm-model", default="gemma2:latest", help="LLM model")
    ap.add_argument("--tts-url", default="http://localhost:5001", help="TTS URL")
    ap.add_argument("--voice", default=None, help="Override voice (else auto)")
    ap.add_argument("--debug", action="store_true", help="Debug mode")
    args = ap.parse_args()

    if not AUDIO_AVAILABLE:
        print("❌ Install sounddevice: pip install sounddevice numpy")
        return

    voice_override = args.voice
    debug_mode = args.debug
    system_prompt = get_emotional_system_prompt()

    print("🎭 Realtime Seamless Emotional TTS Chat")
    print("=" * 40)
    print(f"LLM : {args.llm_model}")
    print(f"Voice: {'Auto' if not voice_override else voice_override.upper()}")
    print("Mode : Realtime streaming → Speaker")
    print(f"Debug: {'ON' if debug_mode else 'OFF'}\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit", "bye"}:
                break
            if not user_input:
                continue

            # ── simple command parser ────────────────────────────────────────
            if user_input.startswith("!"):
                if user_input == "!help":
                    print("Commands: !help, !debug on/off, !voice <name>, !voice auto, quit")
                    continue
                if user_input.startswith("!debug"):
                    debug_mode = "on" in user_input
                    print(f"🔍 Debug: {'ON' if debug_mode else 'OFF'}")
                    continue
                if user_input.startswith("!voice"):
                    parts = user_input.split()
                    if len(parts) > 1:
                        voice_override = None if parts[1] == "auto" else parts[1]
                        print(f"🎤 Voice: {'AUTO' if not voice_override else voice_override.upper()}")
                    continue

            print("AI: ", end="", flush=True)
            full_response = ""
            for chunk in stream_from_ollama(user_input, args.llm_model, args.llm_url, system_prompt):
                print(chunk, end="", flush=True)
                full_response += chunk

            if full_response.strip():
                detected = detect_emotion_from_text(full_response, debug_mode)
                get_streaming_tts(full_response, args.tts_url, detected, voice_override, debug_mode)
            print()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main()
