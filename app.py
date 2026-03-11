import streamlit as st
from src.query_engine import QueryEngine

st.title("📰 NewsBrief AI")
st.write("Ask questions about news articles.")

@st.cache_resource
def load_engine():
    return QueryEngine("data/processed/news_clean.csv")

engine = load_engine()

query = st.text_input("Enter your question")

if st.button("Ask"):
    if query:
        with st.spinner("Generating answer..."):
            answer = engine.ask(query)

        st.subheader("Summary")
        st.write(answer)