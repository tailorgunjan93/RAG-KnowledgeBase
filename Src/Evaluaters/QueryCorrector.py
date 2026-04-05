from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os

class CorrectQuery(BaseModel):
    new_query: str = Field(description="Generate new query if older query cannot generate good or relevant results")

def query_corrector(user_query, llm_response):
    from Src.Utils.llm_utils import get_llm
    llm = get_llm(performance="standard")

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
    try:
        result = query_agent.invoke({"query": user_query, "response": llm_response})
        return result.new_query if result else user_query
    except Exception as e:
        print(f"QueryCorrector error: {e}")
        return user_query
