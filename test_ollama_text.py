import httpx
import json

def test_ollama_text():
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen3-vl:8b",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    
    print("Sending text request to Ollama...")
    try:
        with httpx.Client(timeout=30.0) as client:
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
    test_ollama_text()