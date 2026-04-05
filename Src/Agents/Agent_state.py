from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class state(TypedDict):
    query: str
    query_evaluator: str
    query_grader: str
    response_checker: str
    messages: Annotated[list[BaseMessage], add_messages]
    response: str
    document: List[Document]

