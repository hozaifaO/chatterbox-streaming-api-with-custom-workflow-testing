# Chatterbox Streaming API Docker


A high-performance, text-to-speech API with streaming capabilities, featuring OpenAI-compatible endpoints and multiple voice support. Built with FastAPI and optimised for Docker deployment with GPU acceleration.

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's TTS API endpoints
- **Real-time Streaming**: Low-latency audio generation with chunked streaming
- **Multiple Voices**: Six distinct voices (alloy, echo, fable, onyx, nova, shimmer)
- **GPU Acceleration**: CUDA support for fast inference
- **Live Chat Integration**: Real-time LLM + TTS chat with Ollama
- **Cross-Platform**: Windows, macOS, and Linux support
- **Docker Ready**: Easy deployment with Docker Compose
- **Voice Upload**: Custom voice upload functionality


## 📋 Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Python 3.10+ (for local development)
- [Ollama](https://ollama.ai) (for chat functionality)

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dwain-barnes/chatterbox-streaming-api-docker.git
cd chatterbox-streaming-api-docker
```

### 2. Build and Run with Docker

```bash
# Build the image with voice support
docker build -f Dockerfile.voices -t chatterbox-tts-voices .

# Run the container
docker run -d --gpus all -p 5001:5001 --name chatterbox-tts-voices chatterbox-tts-voices
```

**Or use Docker Compose:**

```bash
docker-compose up -d
```

### 3. Verify Installation

```bash
# Check server health
curl http://localhost:5001/health

# List available voices
curl http://localhost:5001/v1/voices
```

### 4. Test Voice Generation

```bash
# Test with curl
curl -X POST http://localhost:5001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello, this is a test of the text-to-speech system.","voice":"alloy","model":"tts-1"}' \
  --output test.wav

# Or use the provided test scripts
./voice_test_script.sh        # Linux/macOS
voice_test.bat               # Windows
```

## 🎯 Usage Examples

### Basic TTS API Call

```python
import requests

response = requests.post(
    "http://localhost:5001/v1/audio/speech",
    json={
        "input": "Hello, world! This is Chatterbox speaking.",
        "voice": "nova",
        "model": "tts-1"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Streaming TTS

```python
import requests

response = requests.post(
    "http://localhost:5001/v1/audio/speech/stream",
    json={
        "input": "This is a streaming example with real-time audio generation.",
        "voice": "echo",
        "stream": True
    },
    stream=True
)

with open("streaming_output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)
```

### Real-time Chat with LLM + TTS

First, ensure Ollama is running:

```bash
# Install and run Ollama
ollama pull gemma3:latest
ollama serve
```

Then start the chat:

```bash
# Basic chat
python ultra_simple_chat.py

# With custom voice and model
python ultra_simple_chat.py --voice fable --llm-model gemma3:latest
```

## 🎙️ Available Voices

| Voice | Characteristics |
|-------|----------------|
| **alloy** | Professional, neutral, balanced |
| **echo** | Clear, articulate, precise |
| **fable** | Warm, friendly, storytelling |
| **onyx** | Deep, authoritative, commanding |
| **nova** | Energetic, enthusiastic, bright |
| **shimmer** | Soft, gentle, soothing |

## 🔧 Configuration

### Environment Variables

```bash
# Docker environment
NVIDIA_VISIBLE_DEVICES=all
PYTHONUNBUFFERED=1

# Model configuration
TORCH_HOME=/app/.cache
HF_HOME=/app/.cache
TRANSFORMERS_CACHE=/app/.cache
```

### Server Parameters

```bash
python server.py [voices_dir] [supported_voices] [options]

Options:
  --host          Host to bind to (default: 0.0.0.0)
  --port          Port to bind to (default: 5001)
  --device        Device to use: cuda/mps/cpu (default: cuda)
  --exaggeration  Exaggeration factor 0-1 (default: 0.5)
  --cfg-weight    CFG weight 0-1 (default: 0.5)
  --temperature   Temperature for sampling (default: 0.8)
  --chunk-size    Chunk size for streaming (default: 50)
```

### Chat Configuration

```bash
python ultra_simple_chat.py [options]

Options:
  --llm-url       Ollama URL (default: http://localhost:11434)
  --llm-model     LLM model (default: gemma2:latest)
  --tts-url       TTS URL (default: http://localhost:5001)
  --voice         TTS voice (default: alloy)
  --system-prompt Custom system prompt
  --debug         Enable debug mode
```

## 📚 API Reference

### OpenAI-Compatible Endpoints

#### POST `/v1/audio/speech`
Generate complete audio from text.

**Request:**
```json
{
  "model": "tts-1",
  "input": "Text to speak",
  "voice": "alloy",
  "response_format": "wav",
  "speed": 1.0
}
```

**Response:** Audio file (WAV/MP3)

#### POST `/v1/audio/speech/stream`
Generate streaming audio from text.

**Request:**
```json
{
  "model": "tts-1", 
  "input": "Text to speak",
  "voice": "nova",
  "stream": true
}
```

**Response:** Streaming audio chunks

#### GET `/v1/models`
List available models.

#### GET `/v1/voices`
List available voices.

#### POST `/v1/voices/upload`
Upload custom voice file.

## 🧪 Development

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Chatterbox TTS
pip install git+https://github.com/davidbrowne17/chatterbox-streaming.git

# Run server locally
python server.py ./voices alloy,echo,fable,onyx,nova,shimmer --device cpu
```

### Testing

```bash
# Test all voices
./voice_test_script.sh

# Debug mode (no chunking)
python debug_chat.py --debug

# Simple chat test
python ultra_simple_chat.py --debug
```

## 🐳 Docker Configuration

### Build Options

```bash
# Standard build
docker build -t chatterbox-tts .

# With voice support
docker build -f Dockerfile.voices -t chatterbox-tts-voices .

# Development build
docker build --target development -t chatterbox-tts-dev .
```

### Docker Compose

The `docker-compose.yml` includes:
- GPU support
- Volume mounting for voices
- Health checks
- Restart policies
- Environment configuration

## 🔍 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Fallback to CPU
docker run -p 5001:5001 chatterbox-tts-voices python server.py /app/voices alloy,echo --device cpu
```

**Audio playback issues on Windows:**
```bash
# Use debug version
python debug_chat.py

# Check audio format
file test.wav
```

**Ollama connection issues:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - Core TTS engine
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Ollama](https://ollama.ai) - Local LLM inference
- OpenAI - API compatibility standards
- [davidbrowne17](https://github.com/davidbrowne17/chatterbox-streaming) - Streaming version

## Disclaimer
Don't use this model to do bad things.

**⭐ Star this repository if you find it useful!**
