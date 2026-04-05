from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class CorrectQuery(BaseModel):
    new_query: str = Field(description="Generate new query if older query cannot generate good or relevant results")

def query_corrector(user_query, llm_response):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2")
    else:
        model_name = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
        llm = ChatGroq(model=model_name, api_key=api_key)

    llm_with_structure_output = llm.with_structured_output(CorrectQuery)

    system = """
            You are an expert prompt engineer specialist in prompts such that LLMs 
            create new enhanced prompts that are given by users.
             You receive:
                - A USER QUERY: what the user is asking.
                - LLM response: Response of user query from LLMs.
            Your job is to enhance the user query such that it will get a relevant or enhanced LLM response
            output strictly to give a new user query
            """
    query_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
                """### User query
                    {query}
                    ### LLM response
                    {response}
                    This response is not relevant so this is the user query - check 
                    and enhance the prompt so that the LLM will give a relevant result
                    """)
        ]
    )
    
    query_agent = query_prompt | llm_with_structure_output
    result = query_agent.invoke({"query": user_query, "response": llm_response})
    return result.new_query if result else user_query
