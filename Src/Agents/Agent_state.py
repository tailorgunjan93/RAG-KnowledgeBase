from typing import List, TypedDict, Annotated
from operator import add
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str
    query_evaluator: str
    query_grader: str
    response_checker: str
    messages: Annotated[list[BaseMessage], add]
    response: str
    document: List[Document]

