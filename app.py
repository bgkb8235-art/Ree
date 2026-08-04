from typing import TypedDict, List
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Get Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

# Define State
class AgentState(TypedDict):
    messages: List[HumanMessage]

# Initialize LLM
print("Initializing Groq LLM client...")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

print("Groq LLM client initialized successfully.")

# Node function
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

# Build Graph
graph = StateGraph(AgentState)

graph.add_node("process_node", process)

graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

agent = graph.compile()

# Chat Loop
if __name__ == "__main__":
    print("Type 'exit' to quit.\n")

    user_input = input("You: ")

    while user_input.lower() != "exit":
        agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        user_input = input("You: ")

    print("Goodbye!")
