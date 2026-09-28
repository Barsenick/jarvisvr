import base64
import urllib.request
import json

# Read and encode image
f = open('test_image_small.jpg', 'rb')
img_data = f.read()
f.close()
img_b64 = base64.b64encode(img_data).decode('utf-8')

# Prepare request
data = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [{
        'role': 'user',
        'content': 'What is in this image?',
        'images': [img_b64]
    }],
    'stream': False
}).encode('utf-8')

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
    print('Response keys:', list(resp.keys()))
    if 'message' in resp:
        print('Message content:', resp['message'].get('content', 'NO CONTENT'))
    if 'error' in resp:
        print('Error:', resp['error'])
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()