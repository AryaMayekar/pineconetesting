import time
import requests
import json

def test_llama3_speed():
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    model = "llama3.2:latest"
    prompt = "Tell me a story about a brave knight and a dragon."

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Get full response at once for easier timing
    }

    print(f"🚀 Testing {model} speed...")
    print(f"📝 Prompt: {prompt}\n")

    start_time = time.time()
    
    try:
        # Send request to local Ollama server
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Extract metrics from Ollama's response
        response_text = result.get("response", "")
        eval_count = result.get("eval_count", 0)        # Number of tokens generated
        eval_duration = result.get("eval_duration", 0)  # Time spent generating (nanoseconds)
        
        print("-" * 40)
        print(f"🤖 Response:\n{response_text.strip()}")
        print("-" * 40)
        
        print(f"\n⏱️  Total Request Time: {total_duration:.2f}s")
        
        if eval_duration > 0:
            # Convert nanoseconds to seconds
            eval_duration_sec = eval_duration / 1e9
            tps = eval_count / eval_duration_sec
            print(f"📊 Tokens Generated: {eval_count}")
            print(f"⚡ Generation Speed: {tps:.2f} tokens/second")
        else:
            print("⚠️  Could not calculate token speed (missing metrics).")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to Ollama.")
        print("👉 Make sure Ollama is running! (Run 'ollama serve' in a separate terminal)")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_llama3_speed()
