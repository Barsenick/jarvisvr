import httpx
import base64
import json

def test_ollama_vision_small():
    # Read and encode small image
    with open("test_image_small.jpg", "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare the request
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen3-vl:8b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image? Describe briefly."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 200
    }
    
    print("Sending vision request to Ollama (small image)...")
    try:
        with httpx.Client(timeout=120.0) as client:  # 2 minute timeout
            response = client.post(url, headers=headers, json=payload)
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("Response:")
                print(json.dumps(result, indent=2))
                
                # Extract the assistant's message content
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content', '')
                    if content:
                        print(f"\nAssistant says: {content}")
                    else:
                        print("\nAssistant returned empty content")
                        print(f"Full message: {message}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_vision_small()