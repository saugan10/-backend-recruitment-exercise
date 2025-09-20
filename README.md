\# Streamlit PDF RAG Chatbot



A comprehensive Streamlit-based application for uploading PDFs, extracting metadata, generating local embeddings, and enabling chat functionality using Retrieval-Augmented Generation (RAG) with Gemini AI. This project includes additional modules for PDF services, RAG processing, metrics/lambda functions, AWS integration, and a Docker setup for containerized deployment.



\## Overview



\- \*\*Purpose\*\*: Build a scalable PDF processing and chat system for document analysis.

\- \*\*Features\*\*:

&nbsp; - Upload and process PDF files with metadata extraction.

&nbsp; - Generate local embeddings using `sentence-transformers`.

&nbsp; - Implement RAG with Gemini AI for contextual chat.

&nbsp; - Integrate `pdf\_services` for file handling.

&nbsp; - Use `rag\_module` for retrieval logic.

&nbsp; - Include `matrics\_lambda` for metrics or AWS Lambda functions.

&nbsp; - Connect with `aws\_service` for cloud storage or processing.

&nbsp; - Deploy with Docker for consistency across environments.

\- \*\*Tech Stack\*\*: Python, Streamlit, LangChain, FAISS, `sentence-transformers`, Google Generative AI, Docker.



\## Prerequisites



\- \*\*Python 3.8+\*\*

\- \*\*Git\*\* (for version control)

\- \*\*Docker\*\* (for containerization)

\- \*\*Dependencies\*\*: Install via `requirements.txt` (see below)

\- \*\*API Key\*\*: Gemini API key (store in `.streamlit/secrets.toml`)



\## Setup



\### 1. Clone the Repository

```bash

git clone https://github.com/saugan10/-backend-recruitment-exercise.git

cd -backend-recruitment-exercise

2. Install Dependencies
Create a requirements.txt file if not present:
textnotepad requirements.txt
Paste:
textstreamlit
langchain
langchain-community
langchain-google-genai
pypdf
PyPDF2
faiss-cpu
google-generativeai
numpy
scikit-learn
sentence-transformers
Save, then install:
bashpip install -r requirements.txt

3. Set Up Environment

Create a virtual environment (optional but recommended):
bashpython -m venv myenv
myenv\Scripts\activate

Configure Gemini API key in .streamlit/secrets.toml (do not commit this file):
text[grok]
GOOGLE_API_KEY = "your_api_key_here"


4. Run the Application
bashstreamlit run app.py

Access at http://localhost:8501 in your browser.

5. Docker Setup

Build the Docker image:
bashdocker build -t pdf-rag .

Run the container:
bashdocker run -p 8501:8501 pdf-rag


Usage

Upload a PDF file via the Streamlit UI.
Assign a Document ID.
Process the PDF to view metadata, embeddings, and a 2D visualization.
Chat with the PDF content using the RAG interface.

File Structure

app.py: Main Streamlit application.
pdf_services.py: Handles PDF processing and metadata extraction.
rag_module.py: Implements RAG logic.
matrics_lambda.py: Contains metrics or AWS Lambda functions.
aws_service.py: Integrates with AWS services.
requirements.txt: Dependency list.
.gitignore: Excludes sensitive files (e.g., secrets.toml, myenv).

Contributing
Feel free to fork this repository, submit issues, or pull requests. Ensure you follow the coding standards and update the README.md if adding features.
License
[Add a license here, e.g., MIT] - (Optional: Specify or remove this section based on your preference.)
Acknowledgments

Built with guidance from xAI's Grok.
Utilizes open-source libraries from Hugging Face, LangChain, and Google.

Contact
For questions, reach out to [your email or GitHub handle, e.g., saugan10@github.com].