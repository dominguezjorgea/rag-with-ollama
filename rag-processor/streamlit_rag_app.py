import streamlit as st
import requests

st.set_page_config(page_title="RAG Assistant", page_icon="🧠")
st.title("🧠 RAG Assistant")
st.write("Ask anything based on your uploaded documents.")

question = st.text_input("Your question", "")

if question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://rag-processor:5050/rag", json={"question": question})
            result = response.json()
            st.success(result.get("answer", {}).get("result", "No result found."))

        except Exception as e:
            st.error(f"Error: {e}")
