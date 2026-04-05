from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class GradeQuery(BaseModel):
    binary_score: str = Field(
        description="Exactly 'yes' or 'no': whether the retrieved passage is relevant to the user query."
    )


def QueryGrader(user_query: str, documents):
    llm = ChatOllama(model="llama3.2")
    llm_structured = llm.with_structured_output(GradeQuery)
    system = """You grade retrieval quality for a RAG (vector database) pipeline.

                You receive:
                - A USER QUERY: what the user is asking.
                - RETRIEVED CONTENT: one text chunk returned by similarity search from the vector store.

                Your job: decide if this retrieved chunk is **relevant** to answering the user query.

                Answer **yes** when the passage directly discusses the same topic, entities, or concepts as the query, or would help a downstream answer.
                Answer **no** when the passage is off-topic, too generic to help, contradictory, empty, or only tangentially related.

                Output only the structured field binary_score as exactly "yes" or "no" (lowercase)."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """### User query
                    {question}
                    ### Retrieved content (from vector database)
                    {document}
                    Is this retrieved content relevant for answering the user query? Respond with binary_score: yes or no.""",
            ),
        ]
    )
    grader_agent = grade_prompt | llm_structured
    
    result = grader_agent.invoke(
        {"question": user_query, "document": documents}
    )
    return result.binary_score if result else None

