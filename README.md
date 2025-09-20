# Backend Recruitment Exercise

This project implements a PDF-based RAG (Retrieval-Augmented Generation) system with LocalStack, LangGraph agents, Gemini AI, and a Streamlit UI.

## Setup

1. **Prerequisites**:
   - Docker and Docker Compose installed.
   - Gemini API key and Pinecone API key/environment.

2. **Configuration**:
   - Copy `.env.example` to `.env` and fill in the required keys.

3. **Run the Project**:
   - Run `docker-compose up --build` to start all services.
   - Access the Streamlit UI at `http://localhost:8501`.

4. **Verification**:
   - Use `awslocal s3 ls s3://my-pdf-bucket` to check uploaded PDFs.
   - Use `awslocal dynamodb scan --table-name DocumentsMetadata` to view metadata.

## Usage
- **Upload PDF**: Go to the "Upload PDF" page and upload files.
- **Query RAG**: Enter document IDs and a question on the "Query RAG" page.
- **View Document**: Fetch metadata by document ID on the "View Document" page.

## Testing
- Run tests with `pytest` in the `tests/` directory (after mounting it into a container if needed).