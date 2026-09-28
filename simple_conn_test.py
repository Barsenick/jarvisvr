import asyncio
import json
import websockets

async def test_connection():
    uri = "ws://localhost:8765/jarvis"
    try:
        async with websockets.connect(uri) as websocket:
            # Send a hello message
            hello_msg = {
                "type": "client.hello",
                "payload": {
                    "user_id": "test_user",
                    "device": "test_device"
                }
            }
            await websocket.send(json.dumps(hello_msg))
            # Wait for response
            response = await websocket.recv()
            print("Received: {}".format(response))
            
            # Send a simple speech message
            speech_msg = {
                "type": "user.speech",
                "payload": {
                    "text": "Hello Jarvis"
                }
            }
            await websocket.send(json.dumps(speech_msg))
            
            # Wait for response
            response = await websocket.recv()
            print("Received: {}".format(response))
            
    except Exception as e:
        print("Error: {}".format(e))

if __name__ == "__main__":
    asyncio.run(test_connection())