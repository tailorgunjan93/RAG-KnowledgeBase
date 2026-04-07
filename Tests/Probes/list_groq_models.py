import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def list_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        models = client.models.list()
        print("--- AVAILABLE GROQ MODELS ---")
        for model in models.data:
            print(f"- {model.id}")
        print("----------------------------")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_groq_models()
