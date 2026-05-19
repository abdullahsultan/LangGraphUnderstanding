from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float) -> float:
    """
    param num: A number to be tripled.
    Returns the triple of a number.
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]


llm = ChatOllama(model="gemma4", temperature=0.9).bind_tools(tools)
