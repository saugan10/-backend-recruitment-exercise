import streamlit as st
import requests
import os

BACKEND_URL = "http://aws_service:8002"

st.title("PDF RAG System")

st.sidebar.header("Options")
page = st.sidebar.radio("Select Page", ["Upload PDF", "Query RAG", "View Document"])

if page == "Upload PDF":
    st.header("Upload PDF Files")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Upload"):
        if uploaded_files:
            files = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
            response = requests.post(f"{BACKEND_URL}/pdf/upload", files=files)
            if response.status_code == 200:
                st.success("Upload successful!")
                st.json(response.json())
            else:
                st.error(f"Upload failed: {response.text}")

elif page == "Query RAG":
    st.header("Query RAG System")
    doc_ids = st.text_input("Enter Document IDs (comma-separated)", "doc1,doc2")
    question = st.text_input("Enter your question", "What is this about?")
    if st.button("Query"):
        doc_ids_list = [id.strip() for id in doc_ids.split(",") if id.strip()]
        if doc_ids_list and question:
            response = requests.post(f"{BACKEND_URL}/rag/query", json={"doc_ids": doc_ids_list, "question": question})
            if response.status_code == 200:
                result = response.json()
                st.success("Query successful!")
                st.write("Answer:", result["answer"])
                st.write("Metrics:", {k: v for k, v in result.items() if k != "answer"})
            else:
                st.error(f"Query failed: {response.text}")

elif page == "View Document":
    st.header("View Document Metadata")
    doc_id = st.text_input("Enter Document ID")
    if st.button("Fetch"):
        if doc_id:
            response = requests.get(f"{BACKEND_URL}/pdf/documents/{doc_id}")
            if response.status_code == 200:
                st.success("Document found!")
                st.json(response.json())
            else:
                st.error(f"Document not found: {response.text}")

if __name__ == "__main__":
    st.write("Current time:", os.popen("date -u").read().strip())