from nt import system
from langchain_ollama import ChatOllama
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate

class CorrectQuery(BaseModel):
    new_query : str = Field(description=
    "Generate new query if older query cannot generate good or relevant results")

llm  = ChatOllama(model="llama3.2")
llm_with_structure_ouput = llm.with_structured_output(CorrectQuery)

def query_corrector(user_query,llm_response):
    system = """
            You are an expert propmpt engineer specialist in prompt such that llm 
            create new enhance prompt that is given by user.
             You receive:
                - A USER QUERY: what the user is asking.
                - LLM response: Response of user query from LLMs.
            Your job is to enhance user query such that it will get relevant or enhance LLM response
            output strictly to give new user query
            """
    query_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            ("human",
                """### User query
                    {query}
                    ### LLm response
                    {response}
                    This response is not relevant so this is user query check 
                    and enhance prompt that llm will give relevent result
                    """)
        ]
    )
    
    query_agent = query_prompt|llm_with_structure_ouput
    result = query_agent.invoke({"query":user_query,"response":llm_response})
    return result.new_query if result else user_query
