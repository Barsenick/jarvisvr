#!/usr/bin/env python3
"""
End-to-end test for Ollama vision integration.
This test connects to a running Jarvis server and tests the vision flow:
1. Connect to server via WebSocket
2. Send client.hello to establish session
3. Send perception.vision_frame with test image
4. Send user.text question about the image
5. Verify that perception.request (start) is emitted
6. Verify that we get an observation back
7. Verify that we get a vision_annotation hologram spawned
8. Verify that perception.request (stop) is emitted
"""

import asyncio
import json
import uuid
import logging
import sys
import base64
import os
from pathlib import Path

# Add the agent-backend directory to the path so we can import jarvis_backend modules
sys.path.insert(0, str(Path(__file__).parent))

from jarvis_backend import protocol
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRecorder:
    """Records messages received from the server for test verification."""
    
    def __init__(self):
        self.messages = []
        self.events_by_type = {}
    
    def emit(self, event_type: str, payload: dict):
        """Called by the agent when it emits an event."""
        logger.info(f"Received event: {event_type} - {payload}")
        self.messages.append((event_type, payload))
        if event_type not in self.events_by_type:
            self.events_by_type[event_type] = []
        self.events_by_type[event_type].append(payload)
    
    def of(self, event_type: str):
        """Get all events of a specific type."""
        return self.events_by_type.get(event_type, [])
    
    def clear(self):
        """Clear recorded messages."""
        self.messages.clear()
        self.events_by_type.clear()


async def test_ollama_vision_end_to_end():
    """Test Ollama vision integration end-to-end."""
    # Server connection details
    uri = "ws://127.0.0.1:8765/jarvis"
    
    logger.info(f"Connecting to Jarvis server at {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to server")
            
            # Create a recorder to capture events
            recorder = TestRecorder()
            
            # 1. Send client.hello to join the session
            hello_envelope = protocol.make(
                protocol.MsgType.CLIENT_HELLO,
                {"device": "quest3", "app_version": "0.0.0"}
            )
            hello_json = hello_envelope.to_json()
            await websocket.send(hello_json)
            logger.info("Sent CLIENT_HELLO message")
            
            # 2. Receive the hello_ack and get the session ID
            response = await websocket.recv()
            response_data = json.loads(response)
            logger.info(f"Received: {response_data}")
            
            # The server's response should be an envelope with type "server.hello_ack"
            session_id = response_data.get("session")
            if not session_id:
                # If the session is not in the response, we cannot proceed
                raise AssertionError("No session ID in server hello_ack")
            logger.info(f"Received session ID: {session_id}")
            
            # 3. Read the test image and send as a perception.vision_frame (inline transport)
            image_path = os.path.join(os.path.dirname(__file__), "..", "test_image_small.jpg")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Encode the image as base64 for inline transport
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Get image dimensions (we know our test image is 256x256)
            width = 256
            height = 256
            
            vision_frame_envelope = protocol.make(
                protocol.MsgType.PERCEPTION_VISION_FRAME,
                {
                    "frame_id": "test_frame_1",
                    "width": width,
                    "height": height,
                    "format": "jpeg",
                    "transport": "inline",
                    "data": image_base64,
                    "seq": 1,
                },
                session=session_id
            )
            vision_frame_json = vision_frame_envelope.to_json()
            await websocket.send(vision_frame_json)
            logger.info("Sent PERCEPTION_VISION_FRAME message (inline)")
            
            # 4. Send user.text message with attach_perception=true
            # Note: We don't actually need to set attach_perception=true because
            # sending the vision frame sets vision_active=True, which will cause
            # _resolve_attach to return true
            user_text_envelope = protocol.make(
                protocol.MsgType.USER_TEXT,
                {
                    "text": "What is in this image? Please describe the objects and their positions.",
                    # attach_perception: True,  # Not needed since vision frame makes vision_active=True
                },
                session=session_id
            )
            user_text_json = user_text_envelope.to_json()
            await websocket.send(user_text_json)
            logger.info("Sent USER_TEXT message")
            
            # 5. Wait for responses
            start_time = asyncio.get_event_loop().time()
            timeout = 60.0  # 60 seconds timeout for vision processing
            
            perception_request_start_received = False
            perception_request_stop_received = False
            observation_received = False
            vision_annotation_spawned = False
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    # Wait for a message with a short timeout to allow checking conditions
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    logger.info(f"Received: {response_data}")
                    
                    # Record the event using our recorder
                    event_type = response_data.get("type")
                    payload = response_data.get("payload", {})
                    recorder.emit(event_type, payload)
                    
                    # Check for specific events we're interested in
                    if event_type == protocol.MsgType.PERCEPTION_REQUEST:
                        action = payload.get("action")
                        if action == "start":
                            perception_request_start_received = True
                            logger.info("✓ Received perception.request (start)")
                        elif action == "stop":
                            perception_request_stop_received = True
                            logger.info("✓ Received perception.request (stop)")
                    
                    elif event_type == protocol.MsgType.AGENT_OBSERVATION:
                        observation_received = True
                        logger.info("✓ Received agent.observation")
                    
                    elif event_type == protocol.MsgType.HOLO_SPAWN:
                        widget_type = payload.get("widget_type")
                        if widget_type == "vision_annotation":
                            vision_annotation_spawned = True
                            logger.info("✓ Received vision_annotation hologram spawn")
                    
                    # If we've gotten all the expected events, we can break early
                    if perception_request_start_received and perception_request_stop_received and observation_received and vision_annotation_spawned:
                        logger.info("✓ All expected events received!")
                        break
                        
                except asyncio.TimeoutError:
                    # Continue checking conditions
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    break
            
            # 6. Report results
            logger.info("=== Test Results ===")
            logger.info(f"Perception request (start) received: {perception_request_start_received}")
            logger.info(f"Perception request (stop) received: {perception_request_stop_received}")
            logger.info(f"Observation received: {observation_received}")
            logger.info(f"Vision annotation spawned: {vision_annotation_spawned}")
            
            if perception_request_start_received and perception_request_stop_received and observation_received and vision_annotation_spawned:
                logger.info("✓ All tests PASSED!")
                return True
            else:
                logger.info("✗ Some tests FAILED!")
                return False
                
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_ollama_vision_end_to_end())
    sys.exit(0 if result else 1)