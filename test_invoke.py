import traceback
from Src.Agents.Graph_builder import app

try:
    print("Invoking graph...")
    result = app.invoke({
        'query': 'Hi', 
        'query_evaluator': '', 
        'query_grader': '',     
        'response_checker': '', 
        'messages': [], 
        'response': '',  
        'document': []
    }, config={"configurable": {"thread_id": "test_123"}})
    print(f"Success! Agent says: {result.get('response', '')}")
except Exception as e:
    traceback.print_exc()
