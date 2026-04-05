import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def probe_models():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    test_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    print("--- PROBING MODELS ---")
    for model_id in test_models:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model=model_id,
                max_tokens=5
            )
            print(f"✅ {model_id}: WORKS")
        except Exception as e:
            print(f"❌ {model_id}: FAILED ({e})")
    print("----------------------")

if __name__ == "__main__":
    probe_models()
