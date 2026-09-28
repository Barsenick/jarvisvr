import asyncio
import json
import websockets
import base64
import os

async def test_vision_verbose():
    # Configuration
    SERVER_URL = "ws://localhost:8765/jarvis"
    IMAGE_PATH = "test_image.jpg"
    
    if not os.path.exists(IMAGE_PATH):
        print("Error: Image file not found at {}".format(IMAGE_PATH))
        return

    try:
        print("[DEBUG] Attempting to connect to {}".format(SERVER_URL))
        async with websockets.connect(SERVER_URL) as ws:
            print("[INFO] Connected to Jarvis Backend at {}".format(SERVER_URL))
            
            # 1. Handshake (client.hello)
            hello = {
                "type": "client.hello",
                "payload": {
                    "user_id": "test_user",
                    "device": "PC_Tester"
                }
            }
            hello_json = json.dumps(hello)
            print("[DEBUG] Sending client.hello: {}".format(hello_json))
            await ws.send(hello_json)
            print("[INFO] Sent client.hello")
            
            # Wait for hello_ack
            print("[DEBUG] Waiting for hello_ack...")
            resp = await ws.recv()
            print("[DEBUG] Received raw response: {}".format(resp))
            data = json.loads(resp)
            print("[INFO] Server Hello: v{} agent={}".format(
                data.get('payload', {}).get('agent', {}).get('name', 'unknown'),
                data.get('payload', {}).get('agent', {}).get('model', 'unknown')
            ))
            
            session_id = data.get('payload', {}).get('session')
            print("[INFO] Session ID: {}".format(session_id))
            
            # 2. Send Vision Frame
            print("[DEBUG] Loading image: {}".format(IMAGE_PATH))
            with open(IMAGE_PATH, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            vision_frame = {
                "type": "perception.vision_frame",
                "payload": {
                    "frame_id": "test_001",
                    "data": img_data,
                    "width": 256,  # Using small image dimensions
                    "height": 256,
                    "pose": {"position": [0, 1.6, 0], "rotation": [0, 0, 0, 1]}
                }
            }
            vision_json = json.dumps(vision_frame)
            print("[DEBUG] Sending perception.vision_frame (size: {} bytes)".format(len(img_data)))
            print("[DEBUG] Vision frame JSON (first 100 chars): {}".format(vision_json[:100]))
            await ws.send(vision_json)
            print("[INFO] Sent perception.vision_frame")
            
            # 3. Ask a question
            question = "Jarvis, what do you see in this image? Describe briefly."
            print("[INFO] Asking: {}".format(question))
            speech = {
                "type": "user.speech",
                "payload": {
                    "text": question
                }
            }
            speech_json = json.dumps(speech)
            print("[DEBUG] Sending user.speech: {}".format(speech_json))
            await ws.send(speech_json)
            print("[INFO] Sent user.speech")
            
            # 4. Listen for responses with a longer timeout
            print("\n--- Jarvis is thinking... ---")
            
            # Wait for responses with a longer timeout
            try:
                response_received = False
                start_time = asyncio.get_event_loop().time()
                timeout = 120.0  # 2 minutes timeout for first inference
                
                while not response_received and (asyncio.get_event_loop().time() - start_time) < timeout:
                    # Try to receive a message with a timeout
                    try:
                        print("[DEBUG] Waiting for message (5s timeout)...")
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        print("[DEBUG] Received raw message: {}".format(msg))
                        data = json.loads(msg)
                        msg_type = data.get("type")
                        payload = data.get("payload", {})
                        
                        if msg_type == "agent.thinking":
                            print("[THINK] Thinking: {}".format(payload.get('stage', '...')))
                        elif msg_type == "agent.speech":
                            print("[SPEECH] Jarvis: {}".format(payload.get('text', '')))
                            if payload.get("final"):
                                print("\n[INFO] Conversation Finished")
                                response_received = True
                                break
                        elif msg_type == "holo.spawn":
                            print("[HOLO] Hologram Spawned: {} at {}".format(
                                payload.get('widget_type'), 
                                payload.get('transform', {}).get('position')
                            ))
                        elif msg_type == "agent.observation":
                            print("[OBSERV] Observation received")
                        else:
                            print("[MSG] Message [{}]: {}".format(msg_type, str(payload)[:200]))
                            
                    except asyncio.TimeoutError:
                        # No message received in 5 seconds, show we're still waiting
                        elapsed = asyncio.get_event_loop().time() - start_time
                        print("[WAIT] ... still thinking (waiting for Ollama) [{:.0f}s/{:.0f}s]".format(elapsed, timeout))
                        continue
                    except Exception as e:
                        print("[ERROR] Error receiving message: {}".format(e))
                        import traceback
                        traceback.print_exc()
                        break
                
                if not response_received:
                    print("[WARN] Timeout waiting for response after {} seconds".format(timeout))
                        
            except Exception as e:
                print("[ERROR] Error in response loop: {}".format(e))
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print("[ERROR] Connection Error: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vision_verbose())