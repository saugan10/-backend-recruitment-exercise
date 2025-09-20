from .base import BaseAgent, extract_pdf_text, upload_to_s3, store_metadata

class PDFAgent(BaseAgent):
    def process(self, task):
        files = task['files']
        results = []
        for file in files:
            doc_id = str(uuid.uuid4())
            file_path = f"/tmp/{doc_id}.pdf"
            with open(file_path, 'wb') as f:
                f.write(file.file.read())
            text = extract_pdf_text(file_path)
            metadata = {"filename": file.filename, "upload_timestamp": time.time(), "text": text}
            upload_to_s3(doc_id, file_path)
            store_metadata(doc_id, metadata)
            results.append({"doc_id": doc_id, "metadata": metadata})
        return results