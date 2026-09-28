import sys
sys.path.insert(0, '.')

import asyncio
import base64
from agent_backend.jarvis_backend.agent.llm import create_llm

async def test_llm_directly():
    # Create LLM instance
    llm = create_llm(
        provider="ollama",
        model="qwen3-vl:8b",
        base_url="http://localhost:11434/v1"
    )
    
    print(f"LLM created: {llm}")
    print(f"LLM supports vision: {getattr(llm, 'supports_vision', 'Unknown')}")
    
    # Read and encode image
    with open('test_image_small.jpg', 'rb') as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    # Create messages similar to what orchestrator would send
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "What is in this image? Please describe the objects and their positions.",
            # Note: In the actual implementation, images are passed separately
        }
    ]
    
    # Tools specs (empty for this test)
    tools = []
    
    # Images
    images = [img_b64]
    
    print("Calling LLM.complete...")
    try:
        result = await llm.complete(messages, tools, images=images)
        print(f"LLM Result: {result}")
        print(f"Content: {result.content[:200] if result.content else None}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm_directly())