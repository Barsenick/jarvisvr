#!/usr/bin/env python3
"""
Voice service setup checker.
- Prints current configuration.
- Runs the voice service self-test (pytest).
- Optionally tests connection to backend.
"""
import sys
import subprocess
import os
from pathlib import Path

# Add the voice-service package to path
sys.path.insert(0, str(Path(__file__).parent))

def print_config():
    try:
        from jarvis_voice.config import Config
        cfg = Config.from_env(load_dotenv=True)
        print("=== Voice Service Configuration ===")
        print(f"Backend URL: {cfg.backend_url}")
        print(f"Audio URL: {cfg.audio_url}")
        print(f"Wake engine: {cfg.wake_engine}")
        print(f"STT engine: {cfg.stt_engine}")
        print(f"TTS engine: {cfg.tts_engine}")
        print(f"Sample rate: {cfg.sample_rate} Hz")
        print(f"Frame size: {cfg.frame_ms} ms")
        print(f"Wake word: {cfg.wake_word}")
        print(f"STT language: {cfg.stt_language}")
        print(f"TTS language: {cfg.tts_language}")
        print(f"Ambient mode: {cfg.ambient_mode}")
        print(f"Sound events: {cfg.sound_events_engine}")
        print(f"Barge-in enabled: {cfg.barge_in_enabled}")
        print()
        print("=== Model Paths ===")
        print(f"Vosk model: {cfg.vosk_model or '<not set>'}")
        print(f"Piper model: {cfg.piper_model or '<not set>'}")
        print(f"Piper config: {cfg.piper_config or '<not set>'}")
        print(f"Whisper model: {cfg.whisper_model}")
        print(f"Whisper device: {cfg.whisper_device}")
        print(f"Whisper compute type: {cfg.whisper_compute_type}")
        print(f"Mock transcript: {cfg.mock_transcript}")
        print()
    except Exception as e:
        print(f"Error loading config: {e}")
        return False
    return True

def run_self_test():
    print("=== Running Voice Service Self-Test (pytest) ===")
    test_dir = Path(__file__).parent / "tests"
    if not test_dir.exists():
        print("No tests directory found.")
        return False
    # Run pytest on the tests directory
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_dir)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"Exit code: {result.returncode}")
    return result.returncode == 0

def test_backend_connection():
    print("=== Testing Backend Connection (optional) ===")
    try:
        import asyncio
        import json
        import websockets
        from jarvis_voice.config import Config
        cfg = Config.from_env(load_dotenv=True)
        uri = cfg.backend_url
        print(f"Connecting to {uri} ...")
        async def test():
            try:
                async with websockets.connect(uri) as ws:
                    # Send hello
                    hello = {
                        "type": "client.hello",
                        "payload": {
                            "device": "voice-service-checker",
                            "app_version": "0.0.0",
                            "locale": cfg.locale,
                            "capabilities": {
                                "passthrough": True,
                                "hand_tracking": True,
                                "controllers": True,
                                "mic": True,
                                "speaker": True,
                                "scene_understanding": True,
                                "camera_passthrough": False,
                                "ambient_audio": False,
                                "eye_tracking": False,
                                "on_device_vision": False,
                                "depth": False,
                            }
                        }
                    }
                    await ws.send(json.dumps(hello))
                    resp = await ws.recv()
                    print(f"Received: {resp}")
                    data = json.loads(resp)
                    if data.get("type") == "server.hello_ack":
                        print("✓ Backend hello_ack received.")
                        session = data.get("payload", {}).get("session")
                        if session:
                            print(f"  Session: {session}")
                        # Check perception support
                        perc = data.get("payload", {}).get("perception", {})
                        if perc:
                            print(f"  Perception support: vision={perc.get('vision')}, ambient_audio={perc.get('ambient_audio')}, gaze={perc.get('gaze')}, scene_objects={perc.get('scene_objects')}, annotations={perc.get('annotations')}")
                        return True
                    else:
                        print("✗ Unexpected response type.")
                        return False
            except Exception as e:
                print(f"✗ Connection error: {e}")
                return False
        return asyncio.run(test())
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Jarvis VR Voice Service Setup Checker\n")
    if not print_config():
        sys.exit(1)
    # Run self-test
    if not run_self_test():
        print("\nSelf-test failed. Check the output above.")
        # Don't exit; maybe continue to backend test
    # Optional backend test
    if "--test-backend" in sys.argv:
        if not test_backend_connection():
            print("\nBackend test failed.")
    else:
        print("\nTip: Re-run with --test-backend to also test connection to agent-backend.")
    print("\nDone.")

if __name__ == "__main__":
    main()