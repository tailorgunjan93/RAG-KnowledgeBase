import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import traceback

# Explicitly load .env from the current directory
load_dotenv(os.path.join(os.getcwd(), '.env'))

def test_connection():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"--- DIAGNOSTIC ---")
    print(f"NEO4J_URI: {uri}")
    print(f"NEO4J_USERNAME: {username}")
    print(f"NEO4J_PASSWORD: {'****' if password else 'None'}")
    print(f"------------------")

    if not uri or not username or not password:
        print("❌ ERROR: Missing Neo4j credentials in .env file!")
        return

    print(f"Attempting to connect to Neo4j at {uri}...")
    try:
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password
        )
        # Try a simple query
        result = graph.query("RETURN 'Connection Successful' as msg")
        print(f"✅ SUCCESS: {result[0]['msg']}")
        
        # Check node count
        count = graph.query("MATCH (n) RETURN count(n) as nodeCount")
        print(f"📊 Current node count in database: {count[0]['nodeCount']}")
        
    except Exception as e:
        print("❌ FAILED to connect to Neo4j.")
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()
