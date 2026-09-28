import httpx
import base64
import json

# Test Ollama's OpenAI-compatible vision endpoint
def test_ollama_vision():
    # Read and encode image
    with open("test_image.jpg", "rb") as f:
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
                    {"type": "text", "text": "What is in this image?"},
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
        "max_tokens": 1000
    }
    
    print("Sending request to Ollama...")
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("Response:")
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_vision()