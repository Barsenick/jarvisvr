import asyncio
import base64
import os
from jarvis_backend.agent.llm import OllamaLLM, LLMMessage, ImageInput

async def test_llm_directly():
    # Create LLM instance for Ollama
    llm = OllamaLLM(
        provider_id="ollama",
        model="qwen3-vl:8b",
        api_key=None,  # Ollama doesn't need an API key
        base_url="http://localhost:11434/v1",
        supports_vision=True  # Important: set this to True for vision models
    )
    
    print(f"LLM created: {llm}")
    print(f"LLM supports vision: {getattr(llm, 'supports_vision', 'Unknown')}")
    
    # Read and encode image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'test_image_small.jpg')
    with open(image_path, 'rb') as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    # Create ImageInput object
    image_input = ImageInput(b64=img_b64, media_type="image/jpeg")
    
    # Create messages similar to what orchestrator would send
    messages = [
        LLMMessage(
            role="system",
            content="You are a helpful AI assistant."
        ),
        LLMMessage(
            role="user",
            content="What is in this image? Please describe the objects and their positions.",
            # Note: In the actual implementation, images are passed separately
        )
    ]
    
    # Tools specs (empty for this test)
    tools = []
    
    # Images
    images = [image_input]
    
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