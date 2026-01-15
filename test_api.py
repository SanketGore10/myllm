"""
Simple test script to interact with MyLLM API.

This script demonstrates how to:
1. Check API health
2. List available models
3. Send chat messages (streaming)
4. Generate completions (non-streaming)
"""

import requests
import json
import time


# API Configuration
BASE_URL = "http://localhost:8000"
MODEL_NAME = "tinyllama-1.1b"  # Change to your model name


def check_health():
    """Check if the API is healthy."""
    print("🔍 Checking API health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(f"   Response: {response.json()}\n")
            return True
        else:
            print(f"❌ API returned status {response.status_code}\n")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start with: myllm serve\n")
        return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def list_models():
    """List available models."""
    print("📋 Listing available models...")
    try:
        response = requests.get(f"{BASE_URL}/api/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"✅ Found {len(models)} model(s):")
            for model in models:
                print(f"   - {model['name']} ({model.get('family', 'unknown')} family)")
            print()
            return models
        else:
            print(f"❌ Failed to list models: {response.status_code}\n")
            return []
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return []


def chat_streaming(message: str):
    """Send a chat message and stream the response."""
    print(f"💬 Sending chat message: '{message}'")
    print("🔄 Streaming response:\n")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "max_tokens": 200
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}\n")
            return
        
        print("   Assistant: ", end="", flush=True)
        full_response = ""
        
        # Parse SSE stream
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        token = data.get('content', '')
                        if token:
                            print(token, end="", flush=True)
                            full_response += token
                    except json.JSONDecodeError:
                        pass
        
        print("\n")
        print(f"✅ Received {len(full_response)} characters\n")
        return full_response
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None


def generate_completion(prompt: str):
    """Generate a completion (non-streaming)."""
    print(f"✨ Generating completion for: '{prompt}'")
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "max_tokens": 150
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            completion = data.get('response', '')
            print(f"✅ Generated {len(completion)} characters:")
            print(f"   {completion[:200]}{'...' if len(completion) > 200 else ''}\n")
            return completion
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}\n")
            return None
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("MyLLM API Test Script")
    print("=" * 60)
    print()
    
    # 1. Check health
    if not check_health():
        print("⚠️  Server is not running. Start it with:")
        print("   myllm serve")
        return
    
    # 2. List models
    models = list_models()
    if not models:
        print("⚠️  No models found. Pull a model with:")
        print("   myllm pull tinyllama-1.1b")
        return
    
    # Update MODEL_NAME if needed
    global MODEL_NAME
    if models:
        MODEL_NAME = models[0]['name']
        print(f"ℹ️  Using model: {MODEL_NAME}\n")
    
    # 3. Test chat (streaming)
    print("-" * 60)
    chat_streaming("Tell me a short joke")
    
    time.sleep(1)  # Small delay between requests
    
    # 4. Test chat with a follow-up
    print("-" * 60)
    chat_streaming("What is the capital of France?")
    
    time.sleep(1)
    
    # 5. Test generation (non-streaming)
    print("-" * 60)
    generate_completion("Once upon a time in a magical forest,")
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
