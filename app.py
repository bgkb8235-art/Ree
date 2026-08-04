import streamlit as st
from typing import TypedDict, List

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Chatbot using Groq")
st.write("Ask any question!")

# -----------------------------
# Get API Key from Streamlit Secrets
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("❌ GROQ_API_KEY not found.")
    st.info("Go to **App Settings → Secrets** and add:")
    st.code("""
GROQ_API_KEY="your_groq_api_key"
""")
    st.stop()

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# -----------------------------
# LangGraph State
# -----------------------------
class AgentState(TypedDict):
    messages: List[HumanMessage]
    response: str

# -----------------------------
# Node
# -----------------------------
def process(state: AgentState):
    response = llm.invoke(state["messages"])
    state["response"] = response.content
    return state

# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("process_node", process)

graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

agent = graph.compile()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("Type your question...")

if prompt:

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke LangGraph
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "response": ""
        }
    )

    answer = result["response"]

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
