"""Vision Testing Client for JarvisVR.
Simulates a Quest 3 headset by sending a local image to the agent-backend.
"""
import asyncio
import json
import base64
import os
import sys
import websockets
from datetime import datetime

# Configuration
SERVER_URL = "ws://localhost:8765/jarvis"
IMAGE_PATH = "test_image.jpg" # Change this to your image path

async def test_vision():
    if not os.path.exists(IMAGE_PATH):
        print("Error: Image file not found at {}".format(IMAGE_PATH))
        print("Please place a JPG image named 'test_image.jpg' in this folder or update IMAGE_PATH in the script.")
        return

    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("Connected to Jarvis Backend at {}".format(SERVER_URL))

            # 1. Handshake (client.hello)
            hello = {
                "type": "client.hello",
                "payload": {
                    "user_id": "test_user",
                    "device": "PC_Tester"
                }
            }
            await ws.send(json.dumps(hello))
            
            # Wait for hello_ack
            resp = await ws.recv()
            print("Server Hello: {}".format(resp))

            # 2. Send Vision Frame
            print("Sending image: {}".format(IMAGE_PATH))
            with open(IMAGE_PATH, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

            vision_frame = {
                "type": "perception.vision_frame",
                "payload": {
                    "frame_id": "test_001",
                    "data": img_data,
                    "width": 1024,
                    "height": 1024,
                    "pose": {"position": [0, 1.6, 0], "rotation": [0, 0, 0, 1]}
                }
            }
            await ws.send(json.dumps(vision_frame))

            # 3. Ask a question
            question = "Jarvis, what do you see in this image? Please describe the objects and their positions."
            print("Asking: {}".format(question))
            speech = {
                "type": "user.speech",
                "payload": {
                    "text": question
                }
            }
            await ws.send(json.dumps(speech))

            # 4. Listen for responses
            print("\n--- Jarvis is thinking... ---")
            
            async def heartbeat():
                while True:
                    await asyncio.sleep(5)
                    print("... still thinking (waiting for Ollama) ...")

            heartbeat_task = asyncio.create_task(heartbeat())
            
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if msg_type == "agent.thinking":
                        print("Thinking: {}".format(payload.get('stage', '...')))
                    elif msg_type == "agent.speech":
                        print("Jarvis: {}".format(payload.get('text', '')))
                        if payload.get("final"):
                            print("\n--- Conversation Finished ---")
                            break
                    elif msg_type == "holo.spawn":
                        print("Hologram Spawned: {} at {}".format(payload.get('widget_type'), payload.get('transform', {}).get('position')))
                    else:
                        print("Message [{}]: {}".format(msg_type, payload))
            finally:
                heartbeat_task.cancel()

    except Exception as e:
        print("Connection Error: {}".format(e))

if __name__ == "__main__":
    asyncio.run(test_vision())