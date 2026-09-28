import asyncio
import json
import websockets

async def test_connection_persist():
    uri = "ws://localhost:8765/jarvis"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected")
            
            # Send hello
            hello = {
                "type": "client.hello",
                "payload": {
                    "user_id": "test_user",
                    "device": "PC_Tester"
                }
            }
            await websocket.send(json.dumps(hello))
            print("Sent hello")
            
            # Wait for hello_ack
            response = await websocket.recv()
            print("Received: {}".format(response))
            
            # Keep connection alive for 10 seconds
            print("Keeping connection alive for 10 seconds...")
            await asyncio.sleep(10)
            print("Done")
            
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection_persist())