from langchain_core.documents import Document

from Src.Evaluaters.QueryGrader import QueryGrader

from .Agent_state import state


def query_evaluation(state: state):
    document = state["document"]
    query = state["query"]
    route = "chat_bot"
    if not document:
        route = "web_search"
    else:
        for item in document:
            if isinstance(item, tuple):
                doc = item[0]
                doc_content = (
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                )
            elif hasattr(item, "page_content"):
                doc_content = item.page_content
            else:
                doc_content = str(item)

            if QueryGrader(query, doc_content) == "no":
                route = "web_search"
                break
    return {"query_evaluator": route}
