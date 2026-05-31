from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import first_responder, revisor
from tool_executor import execute_tools

MAC_ITERATIONS = 2

def draft_node(state: MessagesState):
    """Draft initial response"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revide the answer based on tool result"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["excute_tools", END]:
    """Determine wether to continue or end based on iteration count"""
    count_tool_visit = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visit // 2
    if num_iterations > MAC_ITERATIONS:
        return END
    return "excute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)
# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)

def main():
    print("Hello from reflexionagent!")


if __name__ == "__main__":
    main()
