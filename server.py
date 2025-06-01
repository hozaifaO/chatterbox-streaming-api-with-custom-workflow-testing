#!/usr/bin/env python3
"""
Chatterbox Streaming TTS API Server
OpenAI-compatible API for text-to-speech with streaming support
"""

import os
import io
import asyncio
import tempfile
import argparse
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

# API Models
class TTSRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model to use for synthesis")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice to use for synthesis")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav or mp3)")
    speed: Optional[float] = Field(default=1.0, description="Speed of speech (0.25-4.0)")
    
class TTSStreamRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model to use for synthesis")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice to use for synthesis")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav or mp3)")
    speed: Optional[float] = Field(default=1.0, description="Speed of speech (0.25-4.0)")
    stream: bool = Field(default=True, description="Enable streaming response")

# FastAPI app
app = FastAPI(
    title="Chatterbox Streaming TTS API",
    description="OpenAI-compatible text-to-speech API with streaming support",
    version="1.0.0"
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

def audio_generator(text: str, voice_path: Optional[Path] = None, **kwargs) -> Generator[bytes, None, None]:
    """Generate audio chunks for streaming"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Generate audio chunks
        for audio_chunk, metrics in model.generate_stream(
            text,
            audio_prompt_path=str(voice_path) if voice_path else None,
            exaggeration=kwargs.get("exaggeration", default_params["exaggeration"]),
            cfg_weight=kwargs.get("cfg_weight", default_params["cfg_weight"]),
            temperature=kwargs.get("temperature", default_params["temperature"]),
            chunk_size=kwargs.get("chunk_size", default_params["chunk_size"]),
            print_metrics=False
        ):
            # Convert tensor to bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_chunk, model.sr, format="wav")
            buffer.seek(0)
            yield buffer.read()
            
            # Log metrics if available
            if metrics and hasattr(metrics, 'rtf') and metrics.rtf:
                logger.debug(f"Chunk {getattr(metrics, 'chunk_count', '?')}, RTF: {metrics.rtf:.3f}")
                
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup if not already loaded"""
    if model is None:
        logger.info("Model not loaded on startup - will load on first request")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "voices": list(supported_voices.keys()),
        "device": str(next(model.parameters()).device) if model else "not loaded"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """OpenAI-compatible text-to-speech endpoint"""
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
    
    try:
        # Generate complete audio
        logger.info(f"Generating audio for text: {request.input[:50]}...")
        
        wav = model.generate(
            request.input,
            audio_prompt_path=str(voice_path) if voice_path else None,
            exaggeration=default_params["exaggeration"],
            cfg_weight=default_params["cfg_weight"],
            temperature=default_params["temperature"]
        )
        
        # Convert to bytes
        buffer = io.BytesIO()
        
        if request.response_format == "mp3":
            # Save as WAV first, then convert to MP3
            # Note: This requires ffmpeg to be installed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                torchaudio.save(tmp_wav.name, wav, model.sr)
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
                torchaudio.save(buffer, wav, model.sr, format="wav")
                media_type = "audio/wav"
        else:
            # Save as WAV
            torchaudio.save(buffer, wav, model.sr, format="wav")
            media_type = "audio/wav"
        
        buffer.seek(0)
        
        return Response(
            content=buffer.read(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: TTSStreamRequest):
    """Streaming text-to-speech endpoint"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate voice
    voice_path = None
    if request.voice in supported_voices:
        voice_path = supported_voices[request.voice]
    elif supported_voices:
        voice_path = list(supported_voices.values())[0]
    
    try:
        # For streaming, we only support WAV format
        if request.response_format != "wav":
            logger.warning("Streaming only supports WAV format")
        
        logger.info(f"Streaming audio for text: {request.input[:50]}...")
        
        return StreamingResponse(
            audio_generator(
                request.input,
                voice_path,
                exaggeration=default_params["exaggeration"],
                cfg_weight=default_params["cfg_weight"],
                temperature=default_params["temperature"],
                chunk_size=default_params["chunk_size"]
            ),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
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
                "owned_by": "chatterbox"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699000000,
                "owned_by": "chatterbox"
            }
        ]
    }

@app.get("/v1/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": list(supported_voices.keys())
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Chatterbox Streaming TTS API Server")
    parser.add_argument("voices_dir", type=str, help="Path to the audio prompt files directory")
    parser.add_argument("supported_voices", type=str, help="Comma-separated list of supported voices")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration factor (0-1)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight (0-1)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size for streaming")
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