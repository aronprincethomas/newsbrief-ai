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
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} article{'s' if len(msg['sources']) != 1 else ''})"):
                for src in msg["sources"]:
                    st.markdown(f"**📰 {src['title']}**")
                    st.caption(f"📁 {src['category']}  |  📅 {src['date']}")
                    st.markdown(f"🔗 [{src['source_url']}]({src['source_url']})")
                    st.divider()

# Chat input
prompt = st.chat_input("Ask about news...")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    with st.spinner("Thinking..."):
        result = engine.ask(prompt)

    summary = result["summary"]
    sources = result["sources"]

    # Save assistant response with sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": summary,
        "sources": sources
    })

    with st.chat_message("assistant"):
        st.write(summary)
        with st.expander(f"Sources ({len(sources)} article{'s' if len(sources) != 1 else ''})"):
            if sources:
                for src in sources:
                    st.markdown(f"**📰 {src['title']}**")
                    st.caption(f"📁 {src['category']}  |  📅 {src['date']}")
                    st.markdown(f"🔗 [{src['source_url']}]({src['source_url']})")
                    st.divider()
            else:
                st.write("No sources found.")