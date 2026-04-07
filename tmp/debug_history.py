
import os
import sys

# Add Src to path
sys.path.append(os.getcwd())

# Mock environment variables needed for Groq/LangChain
os.environ["GROQ_API_KEY"] = "mock_key"

from Src.Agents.Graph_builder import app
from langchain_core.messages import HumanMessage, AIMessage

from unittest.mock import MagicMock, patch

def test_history():
    config = {"configurable": {"thread_id": "test_session_1"}}
    
    with patch("Src.Utils.llm_utils.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock responses
        res1_content = "Hi Gunjan! Nice to meet you."
        res2_content = "Your name is Gunjan."
        
        # We need to make invoke return different things for different turns
        mock_llm.invoke.side_effect = [
            MagicMock(content=res1_content), # Turn 1
            MagicMock(content=res2_content)  # Turn 2
        ]
        
        # First turn
        print("--- Turn 1 ---")
        res1 = app.invoke({"query": "Hi, I'm Gunjan"}, config=config)
        print(f"Response 1: {res1.get('response')}")
        print(f"Messages count: {len(res1.get('messages', []))}")
        for m in res1.get('messages', []):
            print(f"  {type(m).__name__}: {m.content}")

        # Second turn
        print("\n--- Turn 2 ---")
        res2 = app.invoke({"query": "What is my name?"}, config=config)
        print(f"Response 2: {res2.get('response')}")
        print(f"Messages count: {len(res2.get('messages', []))}")
        for m in res2.get('messages', []):
            print(f"  {type(m).__name__}: {m.content}")

if __name__ == "__main__":
    # We need to mock the LLM or provide a real key. 
    # Since I can't provide a real key here safely, I'll just check the message list growth 
    # by mocking the nodes if needed, or just seeing if the list appends.
    
    # Actually, I'll just run it and see if it fails on API key or if I can observe the state.
    try:
        test_history()
    except Exception as e:
        print(f"Caught expected or unexpected error: {e}")
