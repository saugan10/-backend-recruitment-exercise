import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import numpy as np
from sklearn.decomposition import PCA
import json
from PyPDF2 import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# Configure Gemini API key
gemini_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("Please set the GOOGLE_API_KEY in Streamlit secrets or environment variables.")
    st.stop()
genai.configure(api_key=gemini_api_key)

# Set up Streamlit app
st.title("PDF RAG System with Metadata and Document ID")
st.markdown("Upload a PDF, assign a Document ID, view metadata, and chat with content using Retrieval-Augmented Generation (RAG) powered by Gemini AI.")

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunk_texts" not in st.session_state:
    st.session_state.chunk_texts = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

# Sidebar for PDF upload, document ID, and metadata
with st.sidebar:
    st.header("PDF Management")
    
    # Document ID input
    doc_id = st.text_input("Enter Document ID", value=st.session_state.doc_id or "")
    if doc_id:
        st.session_state.doc_id = doc_id
    
    # PDF upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and doc_id:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF and generating embeddings..."):
                try:
                    # Step 1: Save uploaded PDF temporarily
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Step 2: Extract metadata using PyPDF2
                    pdf_reader = PdfReader("temp.pdf")
                    metadata = pdf_reader.metadata or {}
                    st.session_state.metadata = {
                        "Title": metadata.get("/Title", "Unknown"),
                        "Author": metadata.get("/Author", "Unknown"),
                        "Creator": metadata.get("/Creator", "Unknown"),
                        "CreationDate": metadata.get("/CreationDate", "Unknown"),
                        "PageCount": len(pdf_reader.pages)
                    }
                    
                    # Step 3: Load and extract text from PDF
                    loader = PyPDFLoader("temp.pdf")
                    documents = loader.load()
                    
                    # Step 4: Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_documents(documents)
                    st.session_state.chunk_texts = [chunk.page_content for chunk in chunks[:50]]
                    
                    # Step 5: Create embeddings using local SentenceTransformer model
                    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    chunk_embeddings = embeddings.embed_documents(st.session_state.chunk_texts)
                    st.session_state.embeddings = np.array(chunk_embeddings)
                    
                    # Step 6: Build vector store (FAISS)
                    st.session_state.vectorstore = FAISS.from_documents(chunks[:50], embeddings)
                    
                    # Step 7: Set up Gemini LLM and RetrievalQA chain
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0.7,
                        google_api_key=gemini_api_key
                    )
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    
                    # Custom prompt template for RAG
                    prompt_template = """Use the following context to answer the question. If you don't know the answer, say so.
                    Context: {context}
                    Question: {question}
                    Answer:"""
                    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": PROMPT}
                    )
                    
                    st.success(f"PDF (ID: {st.session_state.doc_id}) processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
    else:
        st.warning("Please enter a Document ID and upload a PDF.")

# Display document metadata
if st.session_state.metadata is not None:
    st.markdown("### Document Metadata")
    st.write(f"**Document ID**: {st.session_state.doc_id}")
    for key, value in st.session_state.metadata.items():
        st.write(f"**{key}**: {value}")

# Embedding visualization function
def visualize_embeddings(embeddings, title="Embedding Visualization"):
    if embeddings is None or len(embeddings) == 0:
        st.warning("No embeddings available to visualize.")
        return
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create scatter plot
    chart_data = {
        "type": "scatter",
        "data": {
            "datasets": [{
                "label": "Document Chunks",
                "data": [{"x": x, "y": y} for x, y in reduced_embeddings],
                "backgroundColor": "#1f77b4",
                "borderColor": "#1f77b4",
                "pointRadius": 5
            }]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "PCA Component 1"}},  # Changed true to True
                "y": {"title": {"display": True, "text": "PCA Component 2"}}   # Changed true to True
            },
            "plugins": {
                "title": {"display": True, "text": title}  # Changed true to True
            }
        }
    }
    
    st.markdown("### Embedding Visualization")
    st.markdown("2D projection of document chunk embeddings using PCA:")
    st.json({"chartjs": chart_data})

# Display embeddings if available
if st.session_state.embeddings is not None:
    st.markdown("### Document Embeddings")
    st.write(f"Number of chunks: {len(st.session_state.embeddings)}")
    st.write(f"Embedding dimension: {st.session_state.embeddings.shape[1]}")
    
    st.markdown("#### Sample Embeddings (First 5 Chunks, First 5 Dimensions)")
    for i, emb in enumerate(st.session_state.embeddings[:5]):
        st.write(f"Chunk {i+1}: {emb[:5].tolist()}...")
    
    visualize_embeddings(st.session_state.embeddings, f"Embeddings for Document ID: {st.session_state.doc_id}")

# Main chat interface
if st.session_state.vectorstore is not None and st.session_state.qa_chain is not None:
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_query = st.chat_input("Ask a question about the PDF...")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.conversation_history.append({"role": "user", "content": user_query})
        
        # Generate response and retrieve embeddings
        with st.spinner("Thinking..."):
            try:
                # Run RAG query
                result = st.session_state.qa_chain({"query": user_query})
                response = result["result"]
                
                # Get retrieved documents and their embeddings
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.get_relevant_documents(user_query)
                retrieved_texts = [doc.page_content for doc in retrieved_docs]
                retrieved_embeddings = np.array([st.session_state.embeddings[st.session_state.chunk_texts.index(text)]
                                               for text in retrieved_texts if text in st.session_state.chunk_texts])
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                
                # Show retrieved chunks and their embeddings
                st.markdown("### Retrieved Chunks and Embeddings")
                for i, (text, emb) in enumerate(zip(retrieved_texts, retrieved_embeddings)):
                    st.markdown(f"**Chunk {i+1}**:")
                    st.write(text[:200] + "..." if len(text) > 200 else text)
                    st.write(f"Embedding (first 5 dimensions): {emb[:5].tolist()}...")
                
                # Visualize retrieved embeddings
                if len(retrieved_embeddings) > 0:
                    visualize_embeddings(retrieved_embeddings, f"Retrieved Chunks for Query (Doc ID: {st.session_state.doc_id})")
            except Exception as e:
                response = f"Error generating response: {str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
else:
    st.info("Enter a Document ID, upload a PDF, and process it to start chatting and view embeddings/metadata.")