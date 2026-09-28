import asyncio
import json
import websockets
import base64
import os

async def test_vision_detailed():
    # Configuration
    SERVER_URL = "ws://localhost:8765/jarvis"
    IMAGE_PATH = "test_image.jpg"
    
    if not os.path.exists(IMAGE_PATH):
        print("Error: Image file not found at {}".format(IMAGE_PATH))
        return

    try:
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
            await ws.send(json.dumps(hello))
            print("[INFO] Sent client.hello")
            
            # Wait for hello_ack
            resp = await ws.recv()
            data = json.loads(resp)
            print("[INFO] Server Hello: v{} agent={}".format(
                data.get('payload', {}).get('agent', {}).get('name', 'unknown'),
                data.get('payload', {}).get('agent', {}).get('model', 'unknown')
            ))
            
            session_id = data.get('payload', {}).get('session')
            print("[INFO] Session ID: {}".format(session_id))
            
            # 2. Send Vision Frame
            print("[INFO] Loading image: {}".format(IMAGE_PATH))
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
            print("[INFO] Sent perception.vision_frame (size: {} bytes)".format(len(img_data)))
            
            # 3. Ask a question
            question = "Jarvis, what do you see in this image? Please describe the objects and their positions."
            print("[INFO] Asking: {}".format(question))
            speech = {
                "type": "user.text",
                "payload": {
                    "text": question
                }
            }
            await ws.send(json.dumps(speech))
            print("[INFO] Sent user.speech")
            
            # 4. Listen for responses with timeout
            print("\n--- Jarvis is thinking... ---")
            
            # Set up a timeout for the entire operation
            try:
                # Wait for responses with a timeout
                response_received = False
                start_time = asyncio.get_event_loop().time()
                timeout = 30.0  # 30 seconds timeout
                
                while not response_received:
                    # Check if we've timed out
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        print("[WARN] Timeout waiting for response after {} seconds".format(timeout))
                        break
                    
                    # Try to receive a message with a short timeout
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
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
                            print("[MSG] Message [{}]: {}".format(msg_type, str(payload)[:100]))
                            
                    except asyncio.TimeoutError:
                        # No message received in 2 seconds, show we're still waiting
                        print("[WAIT] ... still thinking (waiting for Ollama) ...")
                        continue
                    except Exception as e:
                        print("[ERROR] Error receiving message: {}".format(e))
                        break
                        
            except Exception as e:
                print("[ERROR] Error in response loop: {}".format(e))
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print("[ERROR] Connection Error: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vision_detailed())