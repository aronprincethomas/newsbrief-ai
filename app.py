import streamlit as st
from src.query_engine import QueryEngine

st.title("📰 NewsBrief AI")
st.write("Ask questions about news articles.")

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    "NewsBrief AI retrieves relevant news articles and summarizes them using transformer models."
)

st.sidebar.write("Example queries:")
st.sidebar.write("- What is happening with Brexit?")
st.sidebar.write("- How to improve focus?")

@st.cache_resource
def load_engine():
    return QueryEngine("data/processed/news_clean.csv")

engine = load_engine()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
prompt = st.chat_input("Ask about news...")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    with st.spinner("Thinking..."):
        response = engine.ask(prompt)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # Expandable sources section
    with st.expander("Sources"):
        st.write("Relevant articles retrieved from dataset.")