import json
import urllib.request
import base64
import os

# Read and encode image
image_path = 'test_image_small.jpg'
with open(image_path, 'rb') as f:
    img_data = f.read()
img_b64 = base64.b64encode(img_data).decode('utf-8')

# Test the EXACT format that the agent is sending
data = json.dumps({
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
    'stream': True,
    'temperature': 0
}).encode('utf-8')

req = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=data,
    headers={'Content-Type': 'application/json'}
)

print("Sending request matching agent format (streaming)...")
print(f"Request JSON length: {len(data)} bytes")

try:
    r = urllib.request.urlopen(req, timeout=30)
    print(f'Status: {r.status}')
    print("Response stream:")
    full_content = ""
    chunk_count = 0
    for line in r:
        line = line.decode('utf-8').strip()
        if line:
            chunk_count += 1
            try:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content_piece = data['message']['content']
                    full_content += content_piece
                    # Print first few chunks to see progress
                    if chunk_count <= 5:
                        print(f"Chunk {chunk_count}: {repr(content_piece)}", end=' ')
                    elif chunk_count == 6:
                        print("... (more chunks) ...", end=' ')
                if data.get('done', False):
                    print(f"\n[Stream completed after {chunk_count} chunks]")
                    break
            except json.JSONDecodeError as e:
                print(f"\nJSON decode error on line: {line[:50]}...")
                continue
    
    print(f"\nTotal response length: {len(full_content)} characters")
    if full_content:
        # Safe print
        safe_content = full_content.encode('ascii', 'replace').decode('ascii')
        print(f"First 200 chars: {safe_content[:200]}...")
        print(f"Last 200 chars: ...{safe_content[-200:]}")
    else:
        print("No content received")
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()