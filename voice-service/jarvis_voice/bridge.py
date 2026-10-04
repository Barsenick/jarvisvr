"""WebSocket bridge: the voice front-end of the agent-backend.
Implements a Gateway:
  Quest (Client) -> Bridge (Server, Port 8766) -> Backend (Client, Port 8765)
"""

from __future__ import annotations
import asyncio
import logging
import threading
import base64
from typing import Any, Optional

import websockets
from . import protocol
from .ambient import AmbientCallbacks, AmbientListener, build_ambient
from .audio import audio_io_available, rms_energy
from .config import Config
from .pipeline import PipelineCallbacks, VoicePipeline, build_pipeline
from .protocol import Envelope, ProtocolError
from .tts import Speaker, create_speaker

log = logging.getLogger(__name__)

class VoiceBridge:
    """Bridges the Quest audio stream to the agent-backend."""

    def __init__(
        self,
        config: Config,
        pipeline: Optional[VoicePipeline] = None,
        speaker: Optional[Speaker] = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.speaker = speaker or (pipeline.tts if pipeline else None) or create_speaker(config)
        
        self.session: Optional[str] = None
        self._backend_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._quest_ws: Optional[websockets.WebSocketServerProtocol] = None
        self._outbox: asyncio.Queue[Envelope] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_trigger_time = 0.0

        
        self._ambient: Optional[AmbientListener] = None
        self._ambient_active = False

    # --- Backend Connection (Client Mode) ---
    
    async def connect_to_backend(self):
        """Connects the bridge to the Agent Backend."""
        url = self.config.backend_url
        log.info("Connecting to backend at %s...", url)
        try:
            self._backend_ws = await websockets.connect(url)
            await self.send_hello()
            asyncio.create_task(self._backend_recv_loop())
            asyncio.create_task(self._sender_loop())
            asyncio.create_task(self._heartbeat_loop())
            log.info("Connected to backend.")
        except Exception as e:
            log.error("Failed to connect to backend: %s", e)
            raise

    async def send_hello(self):
        env = protocol.client_hello(
            mic=True, speaker=True, ambient_audio=not self.config.ambient_disabled,
            device=self.config.device_name, app_version=self.config.app_version, locale=self.config.locale
        )
        await self._backend_ws.send(env.to_json())

    async def _backend_recv_loop(self):
        async for message in self._backend_ws:
            await self.handle_backend_message(message)

    async def handle_backend_message(self, raw: Any):
        try:
            env = Envelope.from_json(raw)
            t = env.type
            if t == protocol.SERVER_HELLO_ACK:
                self.session = env.payload.get("session")
                log.info("← server.hello_ack (session=%s)", self.session)
            elif t == protocol.AGENT_SPEECH or t == protocol.AGENT_OBSERVATION:
                await self._stream_speech(env, kind=t)
            elif t == protocol.SERVER_HEARTBEAT:
                log.debug("← server.heartbeat")
        except Exception as e:
            log.debug("ignoring non-json message: %s", e)

    async def _sender_loop(self):
        while True:
            env = await self._outbox.get()
            try:
                if self._backend_ws:
                    await self._backend_ws.send(env.to_json())
            except Exception as e:
                log.error("backend send error: %s", e)
            finally:
                self._outbox.task_done()

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(protocol.HEARTBEAT_INTERVAL_S)
            if self._backend_ws:
                await self._backend_ws.send(protocol.client_heartbeat(self.session).to_json())

    def _enqueue(self, env: Envelope):
        if self._loop:
            self._loop.call_soon_threadsafe(self._outbox.put_nowait, env)

    # --- Quest Connection (Server Mode) ---

    async def start_quest_server(self):
        """Starts the server that the Quest 3 connects to."""
        port = 8766
        async with websockets.serve(self._handle_quest_connection, "0.0.0.0", port):
            log.info("🚀 Voice Bridge Server listening on port %d", port)
            await asyncio.Future()

    async def _handle_quest_connection(self, ws: websockets.WebSocketServerProtocol):
        log.info("Quest connected to Bridge!")
        self._quest_ws = ws
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    energy = rms_energy(message)
                    if energy > 0.01:
                        pass
                    if self.pipeline:
                        self.pipeline.process_frame(message)
                    
                    # MOCK TRIGGER for testing end-to-end flow
                    # MOCK TRIGGER: If audio is loud, force a response
                    # MOCK TRIGGER removed to enable real wake-word and STT pipeline
                else:
                    log.debug("Received JSON from Quest: %s", message)
        except Exception as e:
            log.error("Quest connection error: %s", e)
        finally:
            self._quest_ws = None
            log.info("Quest disconnected.")

    # --- Speech Synthesis ---

    async def _stream_speech(self, env: Envelope, kind: str = "agent.speech") -> None:
        """Synthesize speech and stream audio chunks back to the Quest."""
        text = env.text or ""
        log.info("← %s received: %s", kind, text)
        if not text: 
            log.warning("Speech received but text is empty!")
            return
        
        log.info("Synthesizing speech for Quest...")
        loop = asyncio.get_running_loop()
        try:
            wav_bytes = await loop.run_in_executor(None, self.speaker.synthesize, text)
            log.info("Synthesis complete. Bytes: %d", len(wav_bytes))
            from .audio import wav_to_pcm16
            pcm, sr, ch = wav_to_pcm16(wav_bytes)
            pcm_bytes = pcm.tobytes() if hasattr(pcm, 'tobytes') else pcm
            
            chunk_size = 3200 
            num_chunks = (len(pcm_bytes) + chunk_size - 1) // chunk_size
            log.info("Streaming %d audio chunks to Quest...", num_chunks)
            
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i : i + chunk_size]
                b64_audio = base64.b64encode(chunk).decode("utf-8")
                
                env_chunk = protocol.Envelope.build(
                    protocol.AGENT_AUDIO_CHUNK,
                    {"audio": b64_audio, "sr": sr, "ch": ch},
                    session=self.session,
                )
                if self._quest_ws:
                    await self._quest_ws.send(env_chunk.to_json())
                else:
                    log.warning("No active Quest connection to send audio to!")
                await asyncio.sleep(0.01)
            
            log.info("Finished streaming audio to Quest.")
            # PC playback completely removed to prevent WDM driver crashes
            # await loop.run_in_executor(None, self.speaker.speak, text)
        except Exception as exc:
            log.error("Streaming speech failed: %s", exc)
            log.error("Streaming speech failed: %s", exc)

    def setup_pipeline(self):
        if not self.pipeline: return
        self.pipeline.cb.on_partial = lambda text: self._enqueue(
            protocol.voice_partial(text, 0.0, session=self.session))
        self.pipeline.cb.on_transcript = lambda res: self._enqueue(
            protocol.voice_transcript(res.text, res.confidence, session=self.session))
        self.pipeline.cb.on_barge_in = lambda: self._enqueue(
            protocol.client_barge_in(session=self.session))

    async def connect_and_run(self, max_retries: int = 0):
        """Compatibility wrapper for __main__.py"""
        self._loop = asyncio.get_running_loop()
        self.setup_pipeline()
        await asyncio.gather(
            self.connect_to_backend(),
            self.start_quest_server()
        )

async def run_bridge(config: Config):
    pipeline = build_pipeline(config)
    bridge = VoiceBridge(config, pipeline=pipeline)
    bridge._loop = asyncio.get_running_loop()
    bridge.setup_pipeline()
    await asyncio.gather(
        bridge.connect_to_backend(),
        bridge.start_quest_server()
    )

def build_bridge(config: Config, with_capture: bool = True) -> VoiceBridge:
    return VoiceBridge(config, pipeline=build_pipeline(config) if with_capture else None)
