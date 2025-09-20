from abc import ABC, abstractmethod
from langchain.tools import tool
import os
import google.generativeai as genai
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import boto3
import uuid
import time
import requests

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
embed_model = 'models/embedding-001'
gen_model = genai.GenerativeModel('gemini-1.5-flash')

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index('rag-index')  # Assume created with dim=768

dynamodb = boto3.resource('dynamodb', endpoint_url=os.environ['DYNAMODB_ENDPOINT'])
s3 = boto3.client('s3', endpoint_url=os.environ['S3_ENDPOINT'])
docs_table = dynamodb.Table('DocumentsMetadata')
metrics_table = dynamodb.Table('AgentMetrics')
bucket = os.environ['S3_BUCKET']
lambda_url = os.environ['METRICS_LAMBDA_URL']

class BaseAgent(ABC):
    @abstractmethod
    def process(self, task):
        pass

@tool
def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "".join(page.extract_text() for page in reader.pages)

@tool
def upload_to_s3(doc_id: str, file_path: str):
    s3.upload_file(file_path, bucket, f"{doc_id}.pdf")

@tool
def store_metadata(doc_id: str, metadata: dict):
    docs_table.put_item(Item={'doc_id': doc_id, **metadata})

@tool
def chunk_and_embed(text: str, doc_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectors = [(f"{doc_id}_{i}", genai.embed_content(model=embed_model, content=chunk)['embedding'], {"doc_id": doc_id, "text": chunk}) for i, chunk in enumerate(chunks)]
    index.upsert(vectors)

@tool
def rag_query(question: str, doc_ids: list) -> dict:
    start = time.time()
    query_emb = genai.embed_content(model=embed_model, content=question)['embedding']
    results = index.query(vector=query_emb, top_k=5, filter={"doc_id": {"$in": doc_ids}}, include_metadata=True)
    context = "\n".join(m['metadata']['text'] for m in results['matches'])
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    token_count = gen_model.count_tokens(prompt)
    tokens_consumed = token_count.total_tokens
    response = gen_model.generate_content(prompt)
    answer = response.text
    tokens_generated = response.usage_metadata.candidates_token_count
    response_time_ms = int((time.time() - start) * 1000)
    confidence = sum(m['score'] for m in results['matches']) / len(results['matches']) if results['matches'] else 0.0
    return {
        "answer": answer, "tokens_consumed": tokens_consumed, "tokens_generated": tokens_generated,
        "response_time_ms": response_time_ms, "confidence_score": confidence
    }

@tool
def submit_metrics(run_id: str, metrics: dict):
    metrics['run_id'] = run_id
    metrics['agent_name'] = 'RAGAgent'
    metrics['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    metrics['status'] = 'completed'
    requests.post(lambda_url, json=metrics)