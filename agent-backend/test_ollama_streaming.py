import json
import urllib.request
import base64
import os

print('Testing Ollama streaming vs non-streaming...')

# Read and encode image
image_path = 'test_image_small.jpg'
with open(image_path, 'rb') as f:
    img_data = f.read()
img_b64 = base64.b64encode(img_data).decode('utf-8')

print(f"Image encoded, length: {len(img_b64)}")

# Test 1: Non-streaming (what we tried in agent)
print("\n=== Test 1: Non-streaming ===")
data_non_stream = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [{
        'role': 'user',
        'content': 'What is in this image? Please describe the objects and their positions.',
        'images': [img_b64]
    }],
    'stream': False
}).encode('utf-8')

req_non_stream = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=data_non_stream,
    headers={'Content-Type': 'application/json'}
)

try:
    print("Sending non-streaming request...")
    r = urllib.request.urlopen(req_non_stream, timeout=30)
    print(f'Status: {r.status}')
    resp = json.loads(r.read().decode('utf-8'))
    content = resp.get('message', {}).get('content', 'NO CONTENT')
    safe_content = content.encode('ascii', 'replace').decode('ascii')
    print(f'Response (first 200 chars): {safe_content[:200]}...')
except Exception as e:
    print(f'Error: {e}')

# Test 2: Streaming (what Ollama might prefer)
print("\n=== Test 2: Streaming ===")
data_stream = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [{
        'role': 'user',
        'content': 'What is in this image? Please describe the objects and their positions.',
        'images': [img_b64]
    }],
    'stream': True
}).encode('utf-8')

req_stream = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=data_stream,
    headers={'Content-Type': 'application/json'}
)

try:
    print("Sending streaming request...")
    r = urllib.request.urlopen(req_stream, timeout=30)
    print(f'Status: {r.status}')
    print("Streaming response:")
    full_content = ""
    for line in r:
        line = line.decode('utf-8').strip()
        if line:
            try:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content_piece = data['message']['content']
                    full_content += content_piece
                    print(content_piece, end='', flush=True)
                if data.get('done', False):
                    print("\n[Stream completed]")
                    break
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
    print(f"\nFull response ({len(full_content)} chars):")
    safe_content = full_content.encode('ascii', 'replace').decode('ascii')
    print(safe_content[:200] + "..." if len(safe_content) > 200 else safe_content)
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

# Test 3: With system message and streaming
print("\n=== Test 3: With system message, streaming ===")
data_stream_sys = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [
        {
            'role': 'system',
            'content': 'You are a helpful AI assistant.'
        },
        {
            'role': 'user',
            'content': 'What is in this image? Please describe the objects and their positions.',
            'images': [img_b64]
        }
    ],
    'stream': True
}).encode('utf-8')

req_stream_sys = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=data_stream_sys,
    headers={'Content-Type': 'application/json'}
)

try:
    print("Sending streaming request with system message...")
    r = urllib.request.urlopen(req_stream_sys, timeout=30)
    print(f'Status: {r.status}')
    print("Streaming response:")
    full_content = ""
    for line in r:
        line = line.decode('utf-8').strip()
        if line:
            try:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content_piece = data['message']['content']
                    full_content += content_piece
                    print(content_piece, end='', flush=True)
                if data.get('done', False):
                    print("\n[Stream completed]")
                    break
            except json.JSONDecodeError:
                continue
    print(f"\nFull response ({len(full_content)} chars):")
    safe_content = full_content.encode('ascii', 'replace').decode('ascii')
    print(safe_content[:200] + "..." if len(safe_content) > 200 else safe_content)
except Exception as e:
    print(f'Error: {e}')