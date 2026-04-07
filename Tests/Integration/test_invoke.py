import traceback
from Src.Agents.Graph_builder import app

try:
    print("Testing Graph with a general knowledge query...")
    thread_id = "test_srk_123"
    config = {"configurable": {"thread_id": thread_id}}
    
    query = 'encoder and decoder stack'
    result1 = app.invoke({
        'query': query, 
    }, config=config)
    
    response = result1.get('response', '')
    print(f"Response: {response}")

    if any(keyword in response.lower() for keyword in ["actor", "bollywood", "india", "king khan"]):
        print("\n✅ Success! The agent correctly identified Shah Rukh Khan.")
    else:
        print("\n❌ Failure! The response seems off.")

except Exception as e:
    traceback.print_exc()
