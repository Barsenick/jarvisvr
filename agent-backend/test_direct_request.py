import json
import urllib.request
import base64
import os

print('Current directory:', os.getcwd())

# Read and encode image
image_path = 'test_image_small.jpg'
print('Image path:', image_path)
print('Image path exists:', os.path.exists(image_path))

with open(image_path, 'rb') as f:
    img_data = f.read()
img_b64 = base64.b64encode(img_data).decode('utf-8')
print('Image encoded, length:', len(img_b64))

# Create the request exactly as in the working test
data = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [{
        'role': 'user',
        'content': 'What is in this image? Please describe the objects and their positions.',
        'images': [img_b64]
    }],
    'stream': False
}).encode('utf-8')

print('Request data length:', len(data))
print('Request data preview:', data[:200])

req = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=data,
    headers={'Content-Type': 'application/json'}
)

print('Sending vision request to Ollama...')
try:
    r = urllib.request.urlopen(req)
    print('Status:', r.status)
    resp = json.loads(r.read().decode('utf-8'))
    content = resp.get('message', {}).get('content', 'NO CONTENT')
    # Safe print for console
    safe_content = content.encode('ascii', 'replace').decode('ascii')
    print('Response:', safe_content)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()