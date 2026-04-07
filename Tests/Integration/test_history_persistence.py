import os
import sys

# Support direct execution of this script by adding project root to paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from Src.Agents.Graph_builder import app

@pytest.fixture
def db_path():
    path = "memory_test.db"
    # Ensure clean start
    if os.path.exists(path):
        os.remove(path)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def thread_config():
    return {"configurable": {"thread_id": "test_history_thread"}}

@patch("Src.Utils.llm_utils.get_llm")
def test_history_persistence_across_invocations(mock_get_llm, thread_config):
    # Setup simple stub classes to avoid MagicMock serialization issues
    class StubResponse:
        def __init__(self, content):
            self.content = content
    
    class StubIntentResult:
        def __init__(self, intent):
            self.intent = intent

    class StubLLM:
        def __init__(self, responses, intent="greeting"):
            self.responses = responses
            self.intent = intent
            self.index = 0
            self.calls = []
        
        def invoke(self, input_data, config=None):
            self.calls.append(input_data)
            res = StubResponse(self.responses[self.index])
            self.index += 1
            return res
        
        def with_structured_output(self, schema):
            return self
        
        def __or__(self, other):
            # To handle prompt | llm
            return self

    # Mock responses for two turns
    res1_content = "Hello Gunjan! How can I help you today?"
    res2_content = "Your name is Gunjan."
    
    stub_llm = StubLLM([res1_content, res2_content])
    mock_get_llm.return_value = stub_llm
    
    # We also need to mock the intent Result when stub_llm is used in a chain
    # In intent_detector: (prompt | llm_structured).invoke(...)
    # Since stub_llm.__or__ returns self, invoke is called on stub_llm.
    # But wait, intent_detector expects an object with .intent attribute.
    # So we need to detect which call it is.
    
    original_invoke = stub_llm.invoke
    def patched_invoke(input_data, config=None):
        if isinstance(input_data, dict) and "query" in input_data and "Classify" in str(input_data.get("query", "")):
             # This is likely the intent detector call
             return StubIntentResult("greeting")
        return original_invoke(input_data, config)
    
    stub_llm.invoke = patched_invoke
    
    # --- TURN 1 ---
    input1 = {"query": "Hi, my name is Gunjan"}
    output1 = app.invoke(input1, config=thread_config)
    
    assert "Gunjan" in output1["response"]
    assert len(output1["messages"]) == 2 # Human + AI
    assert isinstance(output1["messages"][0], HumanMessage)
    assert isinstance(output1["messages"][1], AIMessage)
    
    # --- TURN 2 ---
    # In turn 2, the AI should have access to history
    input2 = {"query": "What is my name?"}
    output2 = app.invoke(input2, config=thread_config)
    
    assert "Gunjan" in output2["response"]
    # Messages should now be 4: [H1, A1, H2, A2]
    assert len(output2["messages"]) == 4
    
    # Verify that the history was passed to the LLM in the second turn
    # The last call to stub_llm.invoke should have received history with the first turn
    messages_sent_to_llm = stub_llm.calls[-1]
    
    # It should contain: System prompt + H1 + A1 + H2
    # In chat_bot_node, we have 1 system message.
    assert len(messages_sent_to_llm) >= 4
    assert any("my name is Gunjan" in m.content for m in messages_sent_to_llm if isinstance(m, HumanMessage))

def test_sqlite_file_creation():
    # Verify that memory.db is created
    assert os.path.exists("memory.db")
