import base64
import os
import json
import urllib.request

def test_ollama_v1_format():
    # Read and encode image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'test_image_small.jpg')
    with open(image_path, 'rb') as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    # Create the standard OpenAI format
    data = json.dumps({
        'model': 'qwen3-vl:8b',
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful AI assistant.'
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'What is in this image? Please describe the objects and their positions.'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64:{img_b64}'
                        }
                    }
                ]
            }
        ],
        'temperature': 0
    }).encode('utf-8')
    
    req = urllib.request.Request(
        'http://localhost:11434/v1/chat/completions',
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("Sending request to Ollama /v1/chat/completions endpoint with OpenAI format...")
    try:
        r = urllib.request.urlopen(req)
        print('Status:', r.status)
        resp = json.loads(r.read().decode('utf-8'))
        print('Response keys:', list(resp.keys()))
        if 'choices' in resp and len(resp['choices']) > 0:
            choice = resp['choices'][0]
            if 'message' in choice:
                message = choice['message']
                # Safely print the content by replacing problematic characters
                content = message.get('content', 'NO CONTENT')
                # Replace or remove emojis and other problematic chars for console output
                safe_content = content.encode('ascii', 'replace').decode('ascii')
                print('Message content (safe):', safe_content)
                print('Message content length:', len(content))
            else:
                print('No message in choice')
        else:
            print('No choices in response')
            print('Full response:', resp)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_v1_format()