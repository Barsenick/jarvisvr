"""``jarvis-voice`` command-line interface.

Subcommands:

* ``demo``     — local loop: mic → wake → STT → print (mock-friendly; falls back
                 to an interactive "type to simulate speech" REPL with no mic).
* ``ambient``  — local demo of continuous ambient listening + sound events (v1.1).
* ``bridge``   — connect to the agent-backend and act as the voice front-end.
* ``say TEXT`` — TTS smoke test (optionally ``--out file.wav``).
* ``selftest`` — headless end-to-end check using the configured/fallback engines.
* ``devices``  — list audio devices (diagnostics).

Run ``python -m jarvis_voice ...`` or the installed ``jarvis-voice`` script.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import re
from typing import List, Optional, Iterable, Generator
from numpy import frombuffer

from . import __version__, audio, protocol
from .config import Config
from .pipeline import PipelineCallbacks, PipelineState, build_pipeline
from .stt import TranscriptResult, create_transcriber
from .tts import create_speaker
from .wakeword import create_wakeword

log = logging.getLogger("jarvis_voice")

SENTENCE_END = re.compile(r"[.!?。！？]\s+")

def sentences_from_stream(token_stream: Iterable[str]) -> Generator[str, None, None]:
    """Buffer LLM tokens into complete sentences to preserve intonation."""
    buffer = ""
    for token in token_stream:
        buffer += token
        while True:
            match = SENTENCE_END.search(buffer)
            if not match:
                break
            sentence = buffer[: match.end()].strip()
            if sentence:
                yield sentence
            buffer = buffer[match.end():]
    tail = buffer.strip()
    if tail:
        yield tail


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _banner(config: Config) -> None:
    print(f"JarvisVR voice-service v{__version__}")
    print(f"  config: {config.summary()}")


# --------------------------------------------------------------------------- #
# say
# --------------------------------------------------------------------------- #
def cmd_say(config: Config, args: argparse.Namespace) -> int:
    from .tts import create_speaker

    speaker = create_speaker(config)
    text = args.text
    print(f"[tts={speaker.name}] speaking: {text!r}")
    if args.out:
        wav = speaker.synthesize(text)
        with open(args.out, "wb") as fh:
            fh.write(wav)
        print(f"wrote {len(wav)} bytes of WAV → {args.out}")
        if not args.no_play:
            speaker.speak(text)
    else:
        speaker.speak(text)
    return 0


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #
def cmd_devices(config: Config, args: argparse.Namespace) -> int:
    print("audio backend available:", audio.audio_io_available())
    print(audio.list_devices())
    return 0


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def cmd_demo(config: Config, args: argparse.Namespace) -> int:
    callbacks = PipelineCallbacks(
        on_state_change=lambda s: log.debug("state: %s", s.value),
        on_wake=lambda: print("  [wake] 👂 listening…"),
        on_partial=lambda t: print(f"  … {t}", end="\r", flush=True),
        on_transcript=lambda r: print(f"  📝 transcript: {r.text!r} (conf={r.confidence:.2f})"),
        on_utterance_empty=lambda: print("  (no speech detected)"),
        on_speak_start=lambda t: None,
    )
    pipeline = build_pipeline(config, callbacks)
    _banner(config)
    print(
        f"engines → wake={pipeline.wake.name} stt={pipeline.stt.name} tts={pipeline.tts.name}"
    )

    use_mic = audio.audio_io_available() and not args.simulate
    if use_mic:
        print('\nListening. Say "Jarvis …". Press Ctrl+C to quit.\n')
        from .audio import AudioUnavailable, MicStream

        try:
            with MicStream(
                sample_rate=config.sample_rate,
                frame_samples=config.samples_per_frame,
                input_device=config.input_device,
            ) as mic:
                pipeline.run(mic)
        except AudioUnavailable as exc:
            print(f"(mic unavailable: {exc}) — switching to simulate mode\n")
            use_mic = False
        except KeyboardInterrupt:
            print("\nbye.")
            return 0

    if not use_mic:
        print("\n[simulate] No mic loop. Type what you'd say to Jarvis (after the wake word).")
        print("           Each line is treated as a recognized utterance. Ctrl+C/empty to quit.\n")
        try:
            while True:
                try:
                    line = input("you> ").strip()
                except EOFError:
                    break
                if not line:
                    break
                result = pipeline.simulate_utterance(line)
                pipeline.speak(f"You said: {result.text}")
        except KeyboardInterrupt:
            print()
    print("bye.")
    return 0


def cmd_demo_streaming(config: Config, args: argparse.Namespace) -> int:
    """Streaming demo: Mic -> STT -> Ollama (Stream) -> Sentence Buffer -> Piper -> Audio."""
    from .tts import create_speaker
    import numpy as np
    import httpx
    import json

    # 1. Setup the Speaker (for the output)
    speaker = create_speaker(config)
    
    # 2. Define how to handle the result once the Pipeline finishes STT
    def handle_transcript(result):
        print(f"  📝 transcript: {result.text!r}")
        
        # The Real LLM Stream (Ollama)
        async def ollama_stream(prompt: str):
            url = "http://localhost:11434/api/generate"
            payload = {"model": "gemma4:e2b", "prompt": prompt, "stream": True}
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        async for line in response.aiter_lines():
                            if not line: continue
                            chunk = json.loads(line)
                            if "response" in chunk: yield chunk["response"]
                            if chunk.get("done"): break
            except Exception as exc:
                log.error("Ollama stream failed: %s", exc)
                yield f"Sorry, I had a connection error: {exc}"

        async def run_tts_pipeline():
            print(f"  🤖 Jarvis: ", end="", flush=True)
            tokens = []
            async for token in ollama_stream(result.text):
                tokens.append(token)
            
            full_text = "".join(tokens)
            # Use the sentence buffer to keep the intonation natural
            for sentence in sentences_from_stream([full_text]):
                print(sentence, end="", flush=True)
                speaker.speak(sentence) 
            print("\n")

        # Run the async TTS pipeline in the background
        asyncio.run(run_tts_pipeline())

    # 3. Use the SAME Pipeline logic as cmd_demo
    callbacks = PipelineCallbacks(
        on_wake=lambda: print("\n[wake] 👂 listening..."),
        on_transcript=handle_transcript,
    )
    pipeline = build_pipeline(config, callbacks)
    _banner(config)
    print("\n[Streaming LLM Demo] Listening for wake word... (Ctrl+C to quit)")

    use_mic = audio.audio_io_available() and not args.simulate
    if use_mic:
        from .audio import AudioUnavailable, MicStream
        try:
            with MicStream(
                sample_rate=config.sample_rate,
                frame_samples=config.samples_per_frame,
                input_device=config.input_device,
            ) as mic:
                pipeline.run(mic) # <--- THIS uses the a-priori working VAD loop
        except AudioUnavailable as exc:
            print(f"Mic unavailable: {exc}")
        except KeyboardInterrupt:
            print("\nbye.")
            return 0
    else:
        print("\n[simulate] No mic. Type utterance to trigger streaming TTS.")
        try:
            while True:
                line = input("you> ").strip()
                if not line: break
                class Result: pass
                r = Result(); r.text = line
                handle_transcript(r)
        except KeyboardInterrupt:
            pass
    
    return 0

# --------------------------------------------------------------------------- #
# ambient (continuous listening + sound events)
# --------------------------------------------------------------------------- #
def cmd_ambient(config: Config, args: argparse.Namespace) -> int:
    from .ambient import AmbientCallbacks, build_ambient

    def on_scene(sc) -> None:
        sounds = ", ".join(f"{s['label']}:{s['confidence']:.2f}" for s in sc.sounds) or "—"
        if sc.ambient_transcript:
            body = f"[{sc.speaker}] {sc.ambient_transcript!r}"
        else:
            body = "(no speech)"
        print(f"  🎧 scene {sc.window_ms}ms {sc.loudness_db:6.1f} dBFS  sounds=[{sounds}]  {body}")

    def on_event(ev) -> None:
        print(f"  🔔 event: {ev.label}  (conf={ev.confidence:.2f}, {ev.loudness_db:.1f} dBFS)")

    ambient = build_ambient(config, AmbientCallbacks(on_audio_scene=on_scene, on_audio_event=on_event))
    _banner(config)
    print(f"ambient engines → stt={ambient.transcriber.name} sound_events={ambient.sounds.name}")

    use_mic = audio.audio_io_available() and not args.simulate
    if use_mic:
        print("\nAmbient listening on. Talk near the mic (no wake word). Ctrl+C to quit.\n")
        from .audio import AudioUnavailable, MicStream

        try:
            with MicStream(
                sample_rate=config.sample_rate,
                frame_samples=config.samples_per_frame,
                input_device=config.input_device,
            ) as mic:
                ambient.run(mic)
        except AudioUnavailable as exc:
            print(f"(mic unavailable: {exc}) — switching to simulate mode\n")
            use_mic = False
        except KeyboardInterrupt:
            print("\nbye.")
            return 0

    if not use_mic:
        print("\n[simulate] Generating synthetic room audio (no mic)…\n")
        try:
            ambient.sounds.set_canned(["doorbell", "music", "speech", "alarm"])
        except Exception:
            pass
        loud = audio.tone(300.0, config.frame_ms, sample_rate=config.sample_rate, amplitude=0.6)
        quiet = audio.silence(config.frame_ms, config.sample_rate)
        wpf = config.frames_for_ms(config.ambient_window_ms)
        for _ in range(3):
            for _ in range(max(1, wpf // 2)):
                ambient.process_frame(loud)
            for _ in range(wpf - wpf // 2 + 1):
                ambient.process_frame(quiet)
        print("\nType overheard speech to simulate a scene (empty/Ctrl+C to quit).\n")
        try:
            while True:
                try:
                    line = input("room> ").strip()
                except EOFError:
                    break
                if not line:
                    break
                ambient.simulate_scene(
                    transcript=line,
                    speaker=config.ambient_speaker,
                    sounds=[{"label": "speech", "confidence": 0.8}],
                    loudness_db=-26.0,
                )
        except KeyboardInterrupt:
            print()
    print("bye.")
    return 0


# --------------------------------------------------------------------------- #
# bridge
# --------------------------------------------------------------------------- #
def cmd_bridge(config: Config, args: argparse.Namespace) -> int:
    from .bridge import build_bridge

    _banner(config)
    print(f"bridging to {config.backend_url} …  (Ctrl+C to quit)")
    bridge = build_bridge(config, with_capture=not args.no_mic)

    try:
        asyncio.run(bridge.connect_and_run(max_retries=args.max_retries))
    except KeyboardInterrupt:
        print("\nbye.")
    return 0


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def cmd_selftest(config: Config, args: argparse.Namespace) -> int:
    print("=" * 60)
    print(f"JarvisVR voice-service selftest (v{__version__})")
    print("=" * 60)
    ok = True

    try:
        env = protocol.voice_transcript("hello jarvis", 0.91, session="S")
        parsed = protocol.Envelope.from_json(env.to_json())
        assert parsed.type == protocol.USER_VOICE_TRANSCRIPT
        assert parsed.text == "hello jarvis"
        assert parsed.is_version_compatible()
        print("[ok] protocol envelope build/parse")
    except Exception as exc:
        ok = False
        print(f"[FAIL] protocol: {exc}")

    pipeline = build_pipeline(config)
    print(
        f"[ok] engines selected: wake={pipeline.wake.name} "
        f"stt={pipeline.stt.name} tts={pipeline.tts.name}"
    )

    transcripts: List[TranscriptResult] = []
    wakes: List[bool] = []
    pipeline.cb = PipelineCallbacks(
        on_wake=lambda: wakes.append(True),
        on_transcript=lambda r: transcripts.append(r),
    )
    try:
        loud = audio.tone(300.0, config.frame_ms, sample_rate=config.sample_rate, amplitude=0.6)
        quiet = audio.silence(config.frame_ms, config.sample_rate)
        frames = [loud] * 8 + [quiet] * (config.frames_for_ms(config.silence_ms) + 3)
        for fr in frames:
            pipeline.process_frame(fr)
        if not transcripts:
            pipeline.simulate_utterance(config.mock_transcript)
        assert transcripts, "no transcript produced"
        print(f"[ok] pipeline produced transcript: {transcripts[-1].text!r}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] pipeline: {exc}")

    try:
        wav = pipeline.synthesize("Jarvis online.")
        pcm, sr, ch = audio.wav_to_pcm16(wav)
        assert len(pcm) > 0 and sr > 0 and ch >= 1
        print(f"[ok] tts synthesize → {len(wav)} byte WAV ({sr} Hz, {ch}ch)")
    except Exception as exc:
        ok = False
        print(f"[FAIL] tts: {exc}")

    try:
        hello = protocol.client_hello(mic=True, speaker=True, ambient_audio=True)
        caps = hello.payload["capabilities"]
        assert caps["mic"] is True and caps["speaker"] is True
        assert caps["ambient_audio"] is True
        print("[ok] bridge hello advertises mic+speaker+ambient_audio")
    except Exception as exc:
        ok = False
        print(f"[FAIL] bridge: {exc}")

    try:
        from .sound_events import create_sound_event_detector
        det = create_sound_event_detector(config)
        win = audio.tone(300.0, config.sound_event_window_ms, sample_rate=config.sample_rate, amplitude=0.6)
        sil = audio.silence(config.sound_event_window_ms, sample_rate=config.sample_rate)
        events = det.analyze(win)
        assert events, "no event on loud audio"
        assert det.analyze(sil) == [], "event on silence"
        e = events[0]
        print(f"[ok] sound events: {det.name} → {e.label!r} (conf={e.confidence:.2f}, {e.loudness_db:.1f} dBFS)")
    except Exception as exc:
        ok = False
        print(f"[FAIL] sound events: {exc}")

    try:
        from .ambient import build_ambient
        amb = build_ambient(config)
        spcm = audio.tone(300.0, config.ambient_window_ms, sample_rate=config.sample_rate, amplitude=0.6)
        scene = amb.analyze_window(spcm)
        assert scene.window_ms == config.ambient_window_ms
        assert scene.loudness_db < 0
        assert scene.sounds or scene.ambient_transcript
        print(
            f"[ok] ambient scene: speaker={scene.speaker} sounds={len(scene.sounds)} "
            f"transcript={scene.ambient_transcript!r}"
        )
    except Exception as exc:
        ok = False
        print(f"[FAIL] ambient: {exc}")

    try:
        bp = build_pipeline(config)
        fired: List[bool] = []
        bp.cb = PipelineCallbacks(on_barge_in=lambda: fired.append(True))
        bp._speaking = True
        loud_frame = audio.tone(300.0, config.frame_ms, sample_rate=config.sample_rate, amplitude=0.6)
        for _ in range(config.barge_in_min_frames + 2):
            bp.process_frame(loud_frame)
        assert fired, "barge-in interrupts TTS on user speech"
        assert not bp.is_speaking()
        print("[ok] barge-in interrupts TTS on user speech")
    except Exception as exc:
        ok = False
        print(f"[FAIL] barge-in: {exc}")

    try:
        sc_env = protocol.audio_scene(
            "overheard chatter", "other", [{"label": "music", "confidence": 0.6}], -30.0, 4000
        )
        ev_env = protocol.audio_event("doorbell", 0.82, -22.0)
        req = protocol.Envelope.from_json(
            protocol.Envelope.build(
                protocol.PERCEPTION_REQUEST, {"stream": "ambient_audio", "action": "start"}
            ).to_json()
        )
        assert protocol.PROTOCOL_VERSION == "1.1.0"
        assert protocol.Envelope.from_json(sc_env.to_json()).type == protocol.PERCEPTION_AUDIO_SCENE
        assert protocol.Envelope.from_json(ev_env.to_json()).payload["label"] == "doorbell"
        assert req.payload["stream"] == "ambient_audio"
        print("[ok] protocol v1.1 perception build/parse (audio_scene/event/request)")
    except Exception as exc:
        ok = False
        print(f"[FAIL] perception protocol: {exc}")

    print("-" * 60)
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    print("audio backend available:", audio.audio_io_available())
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #
def _add_common_overrides(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--wake", default=default, help="override JARVIS_WAKE (auto|openwakeword|porcupine|energy)"
    )
    parser.add_argument(
        "--stt", default=default, help="override JARVIS_STT (auto|faster-whisper|vosk|mock)"
    )
    parser.add_argument(
        "--tts", default=default, help="override JARVIS_TTS (auto|piper|pyttsx3|mock)"
    )
    parser.add_argument("--backend", default=default, help="override JARVIS_BACKEND_URL")
    parser.add_argument(
        "--ambient", default=default, help="override JARVIS_AMBIENT (auto|on|off)"
    )
    parser.add_argument(
        "--sound-events", default=default, help="override JARVIS_SOUND_EVENTS (auto|yamnet|heuristic|off)"
    )
    parser.add_argument(
        "--language", default=default, help="set STT+TTS language (multi-language hook)"
    )
    parser.add_argument(
        "--log-level", default=default, help="override JARVIS_LOG_LEVEL (DEBUG/INFO/…)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis-voice",
        description="JarvisVR voice service — wake word + STT + TTS.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_common_overrides(p, suppress=False)

    sub = p.add_subparsers(dest="command", required=True)

    sp_demo = sub.add_parser("demo", help="local mic → wake → STT loop (mock-friendly)")
    sp_demo.add_argument("--simulate", action="store_true", help="force the typed REPL (no mic)")
    _add_common_overrides(sp_demo, suppress=True)
    sp_demo.set_defaults(func=cmd_demo)

    sp_demo_stream = sub.add_parser("demo-stream", help="local mic → wake → STT → Streaming TTS loop")
    sp_demo_stream.add_argument("--simulate", action="store_true", help="force the typed REPL (no mic)")
    _add_common_overrides(sp_demo_stream, suppress=True)
    sp_demo_stream.set_defaults(func=cmd_demo_streaming)
    sp_ambient = sub.add_parser(
        "ambient", help="continuous ambient listening + sound events (mock-friendly)"
    )
    sp_ambient.add_argument(
        "--simulate", action="store_true", help="force synthetic audio + REPL (no mic)"
    )
    _add_common_overrides(sp_ambient, suppress=True)
    sp_ambient.set_defaults(func=cmd_ambient)

    sp_bridge = sub.add_parser("bridge", help="connect to the agent-backend WebSocket")
    sp_bridge.add_argument("--no-mic", action="store_true", help="speak-only (don't capture mic)")
    sp_bridge.add_argument(
        "--max-retries", type=int, default=0, help="0 = retry forever (default)"
    )
    _add_common_overrides(sp_bridge, suppress=True)
    sp_bridge.set_defaults(func=cmd_bridge)

    sp_say = sub.add_parser("say", help="speak TEXT via TTS")
    sp_say.add_argument("text", help="text to speak")
    sp_say.add_argument("--out", help="also write synthesized WAV to this path")
    sp_say.add_argument("--no-play", action="store_true", help="with --out, don't also play")
    _add_common_overrides(sp_say, suppress=True)
    sp_say.set_defaults(func=cmd_say)

    sp_self = sub.add_parser("selftest", help="headless end-to-end check using the configured/fallback engines")
    _add_common_overrides(sp_self, suppress=True)
    sp_self.set_defaults(func=cmd_selftest)

    sp_dev = sub.add_parser("devices", help="list audio devices")
    sp_dev.set_defaults(func=cmd_devices)

    return p


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.wake:
        config.wake_engine = args.wake.lower()
    if args.stt:
        config.stt_engine = args.stt.lower()
    if args.tts:
        config.tts_engine = args.tts.lower()
    if args.backend:
        config.backend_url = args.backend
    if getattr(args, "ambient", None):
        config.ambient_mode = args.ambient.lower()
    if getattr(args, "sound_events", None):
        config.sound_events_engine = args.sound_events.lower()
    if getattr(args, "language", None):
        config.stt_language = args.language
        config.tts_language = args.language
    if args.log_level:
        config.log_level = args.log_level.upper()
    return config


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config.from_env()
    config = _apply_overrides(config, args)
    _setup_logging(config.log_level)

    try:
        return int(args.func(config, args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nbye.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
