from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflection_chain

load_dotenv()  # Load environment variables from .env file

REFLECT = "reflect"
GENERATE = "generate"

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def generation_node(state: MessageGraph):
    print("Generating tweet...")
    return{"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    print("Reflecting on tweet...")
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

graph = StateGraph(state_schema=MessageGraph)
graph.add_node(GENERATE, generation_node)
graph.add_node(REFLECT, reflection_node)

graph.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    print("Checking if we should continue reflecting...")
    if len(state["messages"]) > 6:
        return END
    else:
        return REFLECT

graph.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT: REFLECT})
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

def main():
    print("Hello from reflectionagent!")
    input_msg = {"messages": [HumanMessage(content="""

Write a tweet about skyrocketing RAM prices.

""")]}
    tweet = app.invoke(input_msg)
    print("Final tweet:")
    print(tweet["messages"][-1].content)


if __name__ == "__main__":
    main()
