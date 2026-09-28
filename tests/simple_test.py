import asyncio
import json
import websockets

async def test_connection():
    uri = "ws://localhost:8765/jarvis"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server")
            
            # Send a hello message
            hello_msg = {
                "type": "client.hello",
                "payload": {
                    "user_id": "test_user",
                    "device": "test_device"
                }
            }
            await websocket.send(json.dumps(hello_msg))
            print("Sent hello message")
            
            # Wait for response
            response = await websocket.recv()
            print("Received: {}".format(response))
            
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())