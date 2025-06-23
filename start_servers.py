#!/usr/bin/env python3
"""
Start both TTS server and Physics Book server in the same container
Updated for Azure DeepSeek-V3
"""
import subprocess
import sys
import time
import threading

def start_tts_server():
    """Start the TTS server"""
    print("🎤 Starting TTS Server on port 5001...")
    subprocess.run([
        sys.executable, "server.py", 
        "/app/voices", 
        "alloy,echo,fable,onyx,nova,shimmer", 
        "--host", "0.0.0.0", 
        "--port", "5001", 
        "--device", "cuda"
    ])

def start_physics_server():
    """Start the Physics Book server with Azure DeepSeek-V3"""
    print("🔬 Starting Physics Book Server on port 5000...")
    # Wait a moment for TTS server to initialize
    time.sleep(5)
    subprocess.run([
        sys.executable, "physics_book_server.py",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--tts-url", "http://localhost:5001",
        "--azure-endpoint", "https://aiiieou.services.ai.azure.com/models",
        "--azure-model", "DeepSeek-V3-0324",
        "--azure-api-key", "3D8J84afqQ4QYZTyx6ZzWKpit4REpMScPE2ofUfDjKNxwsIlBcqRJQQJ99BFACYeBjFXJ3w3AAAAACOG7oIT"  # Your API key
    ])

def main():
    print("🚀 Starting Combined TTS + Physics Book Server")
    print("=" * 50)
    
    # Start TTS server in background thread
    tts_thread = threading.Thread(target=start_tts_server, daemon=True)
    tts_thread.start()
    
    # Start physics server in main thread (this will block)
    start_physics_server()

if __name__ == "__main__":
    main()