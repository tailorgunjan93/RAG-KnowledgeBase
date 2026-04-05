from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field

class GradeResults(BaseModel):
    binary_score :str = Field(
        description=
        "Exactly 'yes' or 'no' checking for result relevance for query submitted by use ")

def result_evaluater(user_query,output_response):
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2")
    else:
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
        llm = ChatGroq(model=model_name, api_key=api_key)
    llm_with_structured = llm.with_structured_output(GradeResults)
    system = """
        You are an specialist RAG Agent which evaluates results relevance with user
        query.
         You receive:
                - A USER QUERY: what the user is asking.
                - LLM response: Response of user query from LLMs.

                Your job: decide if LLM result is **relevant** to answering the user query.

                Answer **yes** when the passage directly discusses the same topic, entities, or concepts as the query, or would help a downstream answer.
                Answer **no** when the passage is off-topic, too generic to help, contradictory, empty, or only tangentially related.

                Output only the structured field binary_score as exactly "yes" or "no" (lowercase)."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """### User query
                    {query}
                    ### LLm response
                    {response}
                    Is this LLm response content relevant for answering the user query? Respond with binary_score: yes or no.""",
            ),
        ]
    )
    grader_agent = grade_prompt | llm_with_structured
    
    result = grader_agent.invoke(
        {"query": user_query, "response": output_response}
    )
    return result.binary_score if result else None
    

