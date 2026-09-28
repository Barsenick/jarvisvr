import base64
import os
import json
import urllib.request

def test_ollama_api_chat():
    # Read and encode image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'test_image_small.jpg')
    with open(image_path, 'rb') as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    # Create the format that we used in the working test
    data = json.dumps({
        'model': 'qwen3-vl:8b',
        'messages': [{
            'role': 'user',
            'content': 'What is in this image? Please describe the objects and their positions.',
            'images': [img_b64]  # This is the format that worked before
        }],
        'stream': False
    }).encode('utf-8')
    
    req = urllib.request.Request(
        'http://localhost:11434/api/chat',
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("Sending request to Ollama /api/chat endpoint...")
    try:
        r = urllib.request.urlopen(req)
        print('Status:', r.status)
        resp = json.loads(r.read().decode('utf-8'))
        print('Response keys:', list(resp.keys()))
        if 'message' in resp:
            message = resp['message']
            # Safely print the content by replacing problematic characters
            content = message.get('content', 'NO CONTENT')
            # Replace or remove emojis and other problematic chars for console output
            safe_content = content.encode('ascii', 'replace').decode('ascii')
            print('Message content (safe):', safe_content)
            print('Message content length:', len(content))
        else:
            print('No message in response')
            print('Full response:', resp)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_api_chat()