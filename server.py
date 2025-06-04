#!/usr/bin/env python3
"""
Chatterbox Streaming TTS API Server
OpenAI-compatible API for text-to-speech with streaming support and emotional parameters
"""

import os
import io
import asyncio
import tempfile
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Generator, Union
import logging

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Response, Form, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from chatterbox.tts import ChatterboxTTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Models - Updated with emotional parameters
class TTSRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model to use for synthesis")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice to use for synthesis")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav or mp3)")
    speed: Optional[float] = Field(default=1.0, description="Speed of speech (0.25-4.0) - legacy parameter")
    # Chatterbox-specific emotional parameters
    exaggeration: Optional[float] = Field(default=0.5, description="Exaggeration factor (0-1) - higher values make speech more expressive")
    cfg_weight: Optional[float] = Field(default=0.5, description="CFG weight (0-1) - lower values make speech faster, higher values slower")
    temperature: Optional[float] = Field(default=0.8, description="Temperature for sampling (0.1-1.5) - controls randomness")
    
class TTSStreamRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model to use for synthesis")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice to use for synthesis")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav or mp3)")
    speed: Optional[float] = Field(default=1.0, description="Speed of speech (0.25-4.0) - legacy parameter")
    stream: bool = Field(default=True, description="Enable streaming response")
    # Chatterbox-specific emotional parameters
    exaggeration: Optional[float] = Field(default=0.5, description="Exaggeration factor (0-1) - higher values make speech more expressive")
    cfg_weight: Optional[float] = Field(default=0.5, description="CFG weight (0-1) - lower values make speech faster, higher values slower")
    temperature: Optional[float] = Field(default=0.8, description="Temperature for sampling (0.1-1.5) - controls randomness")
    chunk_size: Optional[int] = Field(default=50, description="Speech tokens per chunk for streaming")
    # Enhanced streaming parameters
    return_metrics: Optional[bool] = Field(default=False, description="Return streaming metrics")
    low_latency: Optional[bool] = Field(default=False, description="Optimize for low latency (smaller chunks)")
    custom_voice_path: Optional[str] = Field(default=None, description="Path to custom voice reference file")

# FastAPI app
app = FastAPI(
    title="Chatterbox Streaming TTS API",
    description="OpenAI-compatible text-to-speech API with streaming support and emotional parameters",
    version="1.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model: Optional[ChatterboxTTS] = None
voices_dir: Optional[Path] = None
supported_voices: Dict[str, Path] = {}
default_params = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "chunk_size": 50
}

def load_model(device: str = "cuda"):
    """Load the Chatterbox TTS model"""
    global model
    
    # Detect best device if cuda requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Loading model on device: {device}")
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback to CPU if GPU fails
        if device != "cpu":
            logger.info("Falling back to CPU...")
            model = ChatterboxTTS.from_pretrained(device="cpu")
            logger.info("Model loaded on CPU")
        else:
            raise

def load_voices(voices_path: Path, voice_list: list):
    """Load voice files from directory"""
    global supported_voices
    
    for voice_name in voice_list:
        voice_file = voices_path / f"{voice_name}.wav"
        if voice_file.exists():
            supported_voices[voice_name] = voice_file
            logger.info(f"Loaded voice: {voice_name}")
        else:
            logger.warning(f"Voice file not found: {voice_file}")
    
    if not supported_voices:
        logger.warning("No voice files loaded. Will use default voice.")

def validate_emotional_params(exaggeration: float, cfg_weight: float, temperature: float) -> tuple:
    """Validate and clamp emotional parameters to safe ranges"""
    # Clamp exaggeration to 0-1
    exaggeration = max(0.0, min(1.0, exaggeration))
    
    # Clamp cfg_weight to 0-1  
    cfg_weight = max(0.0, min(1.0, cfg_weight))
    
    # Clamp temperature to reasonable range
    temperature = max(0.1, min(1.5, temperature))
    
    return exaggeration, cfg_weight, temperature

import re  # ADD THIS IMPORT

def smart_text_split(text: str, max_words_per_chunk: int = 100):
    """
    Split text into chunks by sentences, keeping under word limit
    ChatterboxTTS has a practical limit of ~40 seconds (~100-120 words)
    """
    if len(text.split()) <= max_words_per_chunk:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Count words in potential new chunk
        test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
        word_count = len(test_chunk.split())
        
        if word_count <= max_words_per_chunk:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def audio_generator(text: str, voice_path: Optional[Path] = None, **kwargs) -> Generator[bytes, None, None]:
    """Generate complete audio with smart text chunking to handle unlimited length"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Extract and validate emotional parameters
    exaggeration = kwargs.get("exaggeration", default_params["exaggeration"])
    cfg_weight = kwargs.get("cfg_weight", default_params["cfg_weight"])
    temperature = kwargs.get("temperature", default_params["temperature"])
    chunk_size = kwargs.get("chunk_size", default_params["chunk_size"])
    
    exaggeration, cfg_weight, temperature = validate_emotional_params(exaggeration, cfg_weight, temperature)
    
    logger.info(f"Generating audio for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    logger.info(f"Text length: {len(text)} chars, {len(text.split())} words")
    logger.info(f"Parameters - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}, chunk_size: {chunk_size}")
    
    try:
        # Split text into manageable chunks for ChatterboxTTS
        text_chunks = smart_text_split(text, max_words_per_chunk=100)
        logger.info(f"Split into {len(text_chunks)} text chunks")
        
        all_audio_chunks = []
        total_chunks_processed = 0
        
        # Process each text chunk
        for text_idx, text_chunk in enumerate(text_chunks):
            logger.info(f"Processing text chunk {text_idx + 1}/{len(text_chunks)}: '{text_chunk[:60]}{'...' if len(text_chunk) > 60 else ''}'")
            logger.info(f"Text chunk {text_idx + 1} has {len(text_chunk.split())} words")
            
            # Process this text chunk with generate_stream
            chunk_audio_parts = []
            
            for audio_chunk, metrics in model.generate_stream(
                text_chunk,
                audio_prompt_path=str(voice_path) if voice_path else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                chunk_size=chunk_size,
                print_metrics=False
            ):
                total_chunks_processed += 1
                chunk_audio_parts.append(audio_chunk)
                
                # Log chunk info
                chunk_duration = audio_chunk.shape[-1] / model.sr
                logger.debug(f"Received audio chunk {total_chunks_processed}, shape: {audio_chunk.shape}, duration: {chunk_duration:.3f}s")
                
                # Log metrics if available
                if metrics:
                    if hasattr(metrics, 'rtf') and metrics.rtf:
                        logger.debug(f"Audio chunk {total_chunks_processed}, RTF: {metrics.rtf:.3f}")
                    if hasattr(metrics, 'latency_to_first_chunk') and metrics.latency_to_first_chunk and total_chunks_processed == 1:
                        logger.info(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
            
            # Concatenate audio parts for this text chunk
            if chunk_audio_parts:
                text_chunk_audio = torch.cat(chunk_audio_parts, dim=-1)
                all_audio_chunks.append(text_chunk_audio)
                text_chunk_duration = text_chunk_audio.shape[-1] / model.sr
                logger.info(f"Text chunk {text_idx + 1} complete: {len(chunk_audio_parts)} audio chunks, {text_chunk_duration:.3f}s duration")
            
        # Concatenate ALL audio chunks from ALL text chunks
        if all_audio_chunks:
            final_audio = torch.cat(all_audio_chunks, dim=-1)
            final_duration = final_audio.shape[-1] / model.sr
            logger.info(f"FINAL AUDIO: {len(text_chunks)} text chunks, {total_chunks_processed} audio chunks total")
            logger.info(f"Final audio shape: {final_audio.shape}, duration: {final_duration:.3f}s")
            
            # Convert to bytes and yield as single complete WAV file
            buffer = io.BytesIO()
            torchaudio.save(buffer, final_audio, model.sr, format="wav")
            buffer.seek(0)
            yield buffer.read()
        else:
            logger.error("No audio chunks were generated!")
            raise HTTPException(status_code=500, detail="No audio generated")
                    
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def audio_generator_enhanced(text: str, voice_path: Optional[Path] = None, **kwargs) -> Generator[bytes, None, None]:
    """Enhanced audio generator with smart text chunking to handle unlimited length"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Extract parameters
    exaggeration = kwargs.get("exaggeration", 0.5)
    cfg_weight = kwargs.get("cfg_weight", 0.5)
    temperature = kwargs.get("temperature", 0.8)
    chunk_size = kwargs.get("chunk_size", 50)
    return_metrics = kwargs.get("return_metrics", False)
    low_latency = kwargs.get("low_latency", False)
    custom_voice_path = kwargs.get("custom_voice_path", None)
    
    # Optimize chunk size for low latency
    if low_latency:
        chunk_size = min(25, chunk_size)  # Smaller speech token chunks for lower latency
    
    # Validate and clamp parameters
    exaggeration = max(0.0, min(1.0, exaggeration))
    cfg_weight = max(0.0, min(1.0, cfg_weight))
    temperature = max(0.1, min(1.5, temperature))
    chunk_size = max(10, min(200, chunk_size))
    
    # Use custom voice path if provided
    if custom_voice_path:
        voice_path = Path(custom_voice_path) if Path(custom_voice_path).exists() else voice_path
    
    logger.info(f"Enhanced generation for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    logger.info(f"Enhanced text length: {len(text)} chars, {len(text.split())} words")
    logger.info(f"Enhanced parameters - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}, chunk_size: {chunk_size}")
    
    try:
        # Split text into manageable chunks for ChatterboxTTS
        text_chunks = smart_text_split(text, max_words_per_chunk=80 if low_latency else 100)
        logger.info(f"Enhanced: Split into {len(text_chunks)} text chunks")
        
        all_audio_chunks = []
        total_chunks_processed = 0
        
        # Process each text chunk
        for text_idx, text_chunk in enumerate(text_chunks):
            logger.info(f"Enhanced processing text chunk {text_idx + 1}/{len(text_chunks)}: '{text_chunk[:60]}{'...' if len(text_chunk) > 60 else ''}'")
            
            # Process this text chunk with generate_stream
            chunk_audio_parts = []
            
            for audio_chunk, metrics in model.generate_stream(
                text_chunk,
                audio_prompt_path=str(voice_path) if voice_path else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                chunk_size=chunk_size
            ):
                total_chunks_processed += 1
                chunk_audio_parts.append(audio_chunk)
                
                chunk_duration = audio_chunk.shape[-1] / model.sr
                logger.debug(f"Enhanced received audio chunk {total_chunks_processed}, shape: {audio_chunk.shape}, duration: {chunk_duration:.3f}s")
                
                # Enhanced metrics logging
                if metrics:
                    if hasattr(metrics, 'rtf') and metrics.rtf:
                        logger.debug(f"Enhanced audio chunk {total_chunks_processed}, RTF: {metrics.rtf:.3f}")
                    if hasattr(metrics, 'latency_to_first_chunk') and metrics.latency_to_first_chunk and total_chunks_processed == 1:
                        logger.info(f"Enhanced first chunk latency: {metrics.latency_to_first_chunk:.3f}s")
            
            # Concatenate audio parts for this text chunk
            if chunk_audio_parts:
                text_chunk_audio = torch.cat(chunk_audio_parts, dim=-1)
                all_audio_chunks.append(text_chunk_audio)
                text_chunk_duration = text_chunk_audio.shape[-1] / model.sr
                logger.info(f"Enhanced text chunk {text_idx + 1} complete: {len(chunk_audio_parts)} audio chunks, {text_chunk_duration:.3f}s duration")
                    
        # Concatenate ALL audio chunks from ALL text chunks
        if all_audio_chunks:
            final_audio = torch.cat(all_audio_chunks, dim=-1)
            final_duration = final_audio.shape[-1] / model.sr
            logger.info(f"Enhanced FINAL AUDIO: {len(text_chunks)} text chunks, {total_chunks_processed} audio chunks total")
            logger.info(f"Enhanced final audio shape: {final_audio.shape}, duration: {final_duration:.3f}s")
            
            # Convert to bytes and yield as single complete WAV file
            buffer = io.BytesIO()
            torchaudio.save(buffer, final_audio, model.sr, format="wav")
            buffer.seek(0)
            yield buffer.read()
        else:
            logger.error("Enhanced: No audio chunks were generated!")
            raise HTTPException(status_code=500, detail="No audio generated")
                        
    except Exception as e:
        logger.error(f"Error in enhanced generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup if not already loaded"""
    if model is None:
        logger.info("Model not loaded on startup - will load on first request")

@app.get("/")
async def root():
    """Health check endpoint"""
    # Get device info safely
    device_info = "not loaded"
    if model is not None:
        try:
            if hasattr(model, 'device'):
                device_info = str(model.device)
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device_info = str(model.model.device)
            elif hasattr(model, '_device'):
                device_info = str(model._device)
            else:
                device_info = "unknown"
        except Exception:
            device_info = "unknown"
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "voices": list(supported_voices.keys()),
        "device": device_info,
        "features": {
            "emotional_parameters": True,
            "streaming": True,
            "enhanced_streaming": True,
            "voice_upload": True,
            "openai_compatible": True,
            "chatterbox_streaming": True,
            "complete_audio_generation": True
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """OpenAI-compatible text-to-speech endpoint with emotional parameters"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate voice
    voice_path = None
    if request.voice in supported_voices:
        voice_path = supported_voices[request.voice]
    elif supported_voices:
        # Use first available voice as fallback
        logger.warning(f"Voice '{request.voice}' not found, using default")
        voice_path = list(supported_voices.values())[0]
    
    # Validate and extract emotional parameters
    exaggeration, cfg_weight, temperature = validate_emotional_params(
        request.exaggeration, request.cfg_weight, request.temperature
    )
    
    try:
        # Generate complete audio with emotional parameters and smart chunking
        logger.info(f"Generating audio for text: {request.input[:50]}...")
        logger.info(f"Text length: {len(request.input)} chars, {len(request.input.split())} words")
        logger.info(f"Emotional params - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}")
        
        # Check if we need to use chunking for long text
        if len(request.input.split()) > 100:
            logger.info("Using smart text chunking for long text")
            text_chunks = smart_text_split(request.input, max_words_per_chunk=100)
            all_audio_chunks = []
            
            for chunk_idx, text_chunk in enumerate(text_chunks):
                logger.info(f"Generating chunk {chunk_idx + 1}/{len(text_chunks)}")
                wav_chunk = model.generate(
                    text_chunk,
                    audio_prompt_path=str(voice_path) if voice_path else None,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
                all_audio_chunks.append(wav_chunk)
            
            # Concatenate all chunks
            final_wav = torch.cat(all_audio_chunks, dim=-1)
            logger.info(f"Non-streaming: Combined {len(text_chunks)} chunks, final duration: {final_wav.shape[-1] / model.sr:.3f}s")
        else:
            # Short text, use single generation
            text_chunks = [request.input]  # For header consistency
            final_wav = model.generate(
                request.input,
                audio_prompt_path=str(voice_path) if voice_path else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # Convert to bytes
        buffer = io.BytesIO()
        
        if request.response_format == "mp3":
            # Save as WAV first, then convert to MP3
            # Note: This requires ffmpeg to be installed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                torchaudio.save(tmp_wav.name, final_wav, model.sr)
                tmp_wav_path = tmp_wav.name
            
            # Convert to MP3 using subprocess + ffmpeg
            try:
                import subprocess
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                    subprocess.run([
                        "ffmpeg", "-i", tmp_wav_path, "-acodec", "mp3", 
                        "-ab", "192k", tmp_mp3.name, "-y"
                    ], check=True, capture_output=True)
                    
                    with open(tmp_mp3.name, "rb") as f:
                        buffer.write(f.read())
                    
                    os.unlink(tmp_mp3.name)
                os.unlink(tmp_wav_path)
                
                media_type = "audio/mpeg"
            except Exception as e:
                logger.warning(f"MP3 conversion failed: {e}, falling back to WAV")
                buffer = io.BytesIO()
                torchaudio.save(buffer, final_wav, model.sr, format="wav")
                media_type = "audio/wav"
        else:
            # Save as WAV
            torchaudio.save(buffer, final_wav, model.sr, format="wav")
            media_type = "audio/wav"
        
        buffer.seek(0)
        
        return Response(
            content=buffer.read(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Chatterbox-Exaggeration": str(exaggeration),
                "X-Chatterbox-CFG-Weight": str(cfg_weight),
                "X-Chatterbox-Temperature": str(temperature),
                "X-Chatterbox-Text-Chunks": str(len(text_chunks)),
                "X-Chatterbox-Words": str(len(request.input.split())),
                "X-Chatterbox-Chunking-Used": str(len(text_chunks) > 1)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSStreamRequest):
    """Streaming text-to-speech endpoint with emotional parameters"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate voice
    voice_path = None
    if request.voice in supported_voices:
        voice_path = supported_voices[request.voice]
    elif supported_voices:
        voice_path = list(supported_voices.values())[0]
    
    # Validate emotional parameters
    exaggeration, cfg_weight, temperature = validate_emotional_params(
        request.exaggeration, request.cfg_weight, request.temperature
    )
    
    # Validate chunk size
    chunk_size = max(10, min(200, request.chunk_size or 50))
    
    try:
        # For streaming, we only support WAV format
        if request.response_format != "wav":
            logger.warning("Streaming only supports WAV format")
        
        logger.info(f"Streaming audio for text: {request.input[:50]}...")
        logger.info(f"Emotional params - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}")
        logger.info(f"Streaming options - chunk_size: {chunk_size}, low_latency: {request.low_latency}, metrics: {request.return_metrics}")
        
        # Use enhanced generator if advanced options are requested
        if request.return_metrics or request.low_latency or request.custom_voice_path:
            generator = audio_generator_enhanced(
                request.input,
                voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                chunk_size=chunk_size,
                return_metrics=request.return_metrics,
                low_latency=request.low_latency,
                custom_voice_path=request.custom_voice_path
            )
        else:
            generator = audio_generator(
                request.input,
                voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                chunk_size=chunk_size
            )
        
        return StreamingResponse(
            generator,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Cache-Control": "no-cache",
                "X-Chatterbox-Exaggeration": str(exaggeration),
                "X-Chatterbox-CFG-Weight": str(cfg_weight),
                "X-Chatterbox-Temperature": str(temperature),
                "X-Chatterbox-Chunk-Size": str(chunk_size),
                "X-Chatterbox-Low-Latency": str(request.low_latency),
                "X-Chatterbox-Metrics": str(request.return_metrics),
                "X-Chatterbox-Generation-Method": "smart-text-chunking",
                "X-Chatterbox-Words": str(len(request.input.split()))
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "chatterbox",
                "capabilities": {
                    "emotional_parameters": True,
                    "streaming": True,
                    "enhanced_streaming": True,
                    "formats": ["wav", "mp3"]
                }
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699000000,
                "owned_by": "chatterbox",
                "capabilities": {
                    "emotional_parameters": True,
                    "streaming": True,
                    "enhanced_streaming": True,
                    "formats": ["wav", "mp3"]
                }
            }
        ]
    }

@app.get("/v1/voices")
async def list_voices():
    """List available voices with emotional parameter suggestions"""
    voice_suggestions = {
        "alloy": {"description": "Neutral, professional", "suggested_emotions": ["neutral", "calm"]},
        "echo": {"description": "Clear, articulate", "suggested_emotions": ["confused", "worried"]},
        "fable": {"description": "Warm, storytelling", "suggested_emotions": ["surprised", "enthusiastic"]},
        "onyx": {"description": "Deep, authoritative", "suggested_emotions": ["angry", "frustrated"]},
        "nova": {"description": "Energetic, bright", "suggested_emotions": ["happy", "excited"]},
        "shimmer": {"description": "Soft, gentle", "suggested_emotions": ["sad", "tired"]}
    }
    
    return {
        "voices": [
            {
                "name": voice,
                "available": True,
                **voice_suggestions.get(voice, {"description": "Custom voice"})
            }
            for voice in supported_voices.keys()
        ],
        "emotional_parameters": {
            "exaggeration": {
                "range": [0.0, 1.0],
                "default": 0.5,
                "description": "Controls expressiveness - higher values make speech more dramatic"
            },
            "cfg_weight": {
                "range": [0.0, 1.0],
                "default": 0.5,
                "description": "Controls pacing - lower values speed up speech, higher values slow it down"
            },
            "temperature": {
                "range": [0.1, 1.5],
                "default": 0.8,
                "description": "Controls randomness and naturalness in generation"
            }
        }
    }

@app.post("/v1/voices/upload")
async def upload_voice(
    voice_name: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a new voice file"""
    if not voices_dir:
        raise HTTPException(status_code=500, detail="Voices directory not configured")
    
    # Validate file type
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Save the uploaded file
        voice_path = voices_dir / f"{voice_name}.wav"
        with open(voice_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Add to supported voices
        supported_voices[voice_name] = voice_path
        
        return {"message": f"Voice '{voice_name}' uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Error uploading voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/emotional-presets")
async def get_emotional_presets():
    """Get predefined emotional parameter presets"""
    return {
        "presets": {
            "happy": {"exaggeration": 0.7, "cfg_weight": 0.4, "temperature": 0.9, "suggested_voice": "nova"},
            "excited": {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 1.0, "suggested_voice": "nova"},
            "sad": {"exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.6, "suggested_voice": "shimmer"},
            "angry": {"exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 0.9, "suggested_voice": "onyx"},
            "calm": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.7, "suggested_voice": "alloy"},
            "confused": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.7, "suggested_voice": "echo"},
            "tired": {"exaggeration": 0.2, "cfg_weight": 0.8, "temperature": 0.5, "suggested_voice": "shimmer"},
            "enthusiastic": {"exaggeration": 0.75, "cfg_weight": 0.35, "temperature": 0.95, "suggested_voice": "fable"},
            "worried": {"exaggeration": 0.35, "cfg_weight": 0.65, "temperature": 0.65, "suggested_voice": "echo"},
            "surprised": {"exaggeration": 0.6, "cfg_weight": 0.4, "temperature": 0.85, "suggested_voice": "fable"},
            "frustrated": {"exaggeration": 0.7, "cfg_weight": 0.35, "temperature": 0.85, "suggested_voice": "onyx"},
            "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "suggested_voice": "alloy"}
        },
        "usage": "Use these presets as starting points for emotional speech generation"
    }

@app.get("/v1/streaming/info")
async def get_streaming_info():
    """Get information about streaming capabilities and performance"""
    # Get device info safely
    device_info = "not_loaded"
    sample_rate_info = "not_loaded"
    
    if model is not None:
        try:
            # Try different ways to get device info from ChatterboxTTS
            if hasattr(model, 'device'):
                device_info = str(model.device)
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device_info = str(model.model.device)
            elif hasattr(model, '_device'):
                device_info = str(model._device)
            else:
                device_info = "unknown"
        except Exception:
            device_info = "unknown"
        
        try:
            sample_rate_info = model.sr if hasattr(model, 'sr') else "unknown"
        except Exception:
            sample_rate_info = "unknown"
    
    return {
        "streaming_supported": True,
        "chunk_size_range": [10, 200],
        "default_chunk_size": 50,
        "low_latency_chunk_size": 25,
        "generation_method": "smart_text_chunking_with_streaming",
        "text_chunking": {
            "enabled": True,
            "max_words_per_chunk": 100,
            "method": "sentence_boundary_splitting",
            "reason": "ChatterboxTTS has ~40 second limit per generate_stream call"
        },
        "supported_parameters": [
            "exaggeration", "cfg_weight", "temperature", "chunk_size"
        ],
        "metrics_available": [
            "rtf", "latency_to_first_chunk", "chunk_count"
        ],
        "performance_modes": {
            "standard": "chunk_size=50, balanced quality/latency",
            "low_latency": "chunk_size=25, optimized for real-time",
            "high_quality": "chunk_size=100, optimized for quality"
        },
        "enhanced_features": {
            "return_metrics": "Get streaming performance metrics",
            "low_latency": "Optimize chunk size for real-time playback",
            "custom_voice_path": "Use custom voice reference files"
        },
        "model_info": {
            "sample_rate": sample_rate_info,
            "device": device_info,
            "model_loaded": model is not None,
            "model_type": "ChatterboxTTS"
        }
    }

@app.post("/v1/streaming/test")
async def test_streaming_performance(
    text: str = "This is a test of the streaming text-to-speech system.",
    voice: str = "alloy",
    chunk_size: int = 50,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8
):
    """Test streaming performance with specified parameters"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    voice_path = supported_voices.get(voice, list(supported_voices.values())[0] if supported_voices else None)
    
    # Performance test
    start_time = time.time()
    
    chunk_count = 0
    total_audio_duration = 0
    first_chunk_time = None
    
    try:
        # Test using the same method as the generators - collect all chunks
        streamed_chunks = []
        
        for audio_chunk, metrics in model.generate_stream(
            text,
            audio_prompt_path=str(voice_path) if voice_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            chunk_size=chunk_size
        ):
            chunk_count += 1
            streamed_chunks.append(audio_chunk)
            
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            # Calculate audio duration
            audio_duration = audio_chunk.shape[-1] / model.sr
            total_audio_duration += audio_duration
        
        total_time = time.time() - start_time
        rtf_overall = total_audio_duration / total_time if total_time > 0 else 0
        
        return {
            "test_results": {
                "text_length": len(text),
                "chunks_collected": len(streamed_chunks),
                "chunk_count": chunk_count,
                "total_generation_time": round(total_time, 3),
                "first_chunk_latency": round(first_chunk_time, 3) if first_chunk_time else None,
                "total_audio_duration": round(total_audio_duration, 3),
                "overall_rtf": round(rtf_overall, 3),
                "chunks_per_second": round(chunk_count / total_time, 2) if total_time > 0 else 0
            },
            "parameters_used": {
                "voice": voice,
                "chunk_size": chunk_size,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature
            },
            "performance_rating": "excellent" if rtf_overall > 5 else "good" if rtf_overall > 2 else "fair"
        }
        
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Chatterbox Streaming TTS API Server with Emotional Parameters")
    parser.add_argument("voices_dir", type=str, help="Path to the audio prompt files directory")
    parser.add_argument("supported_voices", type=str, help="Comma-separated list of supported voices")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Default exaggeration factor (0-1)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="Default CFG weight (0-1)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Default temperature for sampling")
    parser.add_argument("--chunk-size", type=int, default=50, help="Default chunk size for streaming")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set global variables
    global voices_dir, default_params
    voices_dir = Path(args.voices_dir)
    
    if not voices_dir.exists():
        logger.error(f"Voices directory not found: {voices_dir}")
        return
    
    # Update default parameters
    default_params.update({
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": args.temperature,
        "chunk_size": args.chunk_size
    })
    
    # Load voices
    voice_list = [v.strip() for v in args.supported_voices.split(",")]
    load_voices(voices_dir, voice_list)
    
    # Load model
    load_model(args.device)
    
    # Run the server
    import uvicorn
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
