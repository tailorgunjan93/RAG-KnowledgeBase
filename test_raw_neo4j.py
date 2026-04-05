import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import traceback

load_dotenv(os.path.join(os.getcwd(), '.env'))

def test_raw_connection():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"--- RAW DRIVER DIAGNOSTIC ---")
    print(f"NEO4J_URI: {uri}")
    print(f"NEO4J_USERNAME: {username}")
    print(f"-----------------------------")

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("RETURN 'Raw Connection Successful' as msg")
            record = result.single()
            print(f"✅ RAW DRIVER SUCCESS: {record['msg']}")
        driver.close()
    except Exception as e:
        print("❌ RAW DRIVER FAILED.")
        traceback.print_exc()

if __name__ == "__main__":
    test_raw_connection()
