from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

duck_api_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
duck_search_tool = DuckDuckGoSearchResults(api_wrapper=duck_api_wrapper)


def create_search_results(query: str):
    response = duck_search_tool.invoke(query)
    return response
