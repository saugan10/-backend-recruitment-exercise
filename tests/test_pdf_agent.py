import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch

client = TestClient(app)

@patch('agents.base.extract_pdf_text')
@patch('agents.base.upload_to_s3')
@patch('agents.base.store_metadata')
def test_pdf_upload(mock_store, mock_upload, mock_extract):
    mock_extract.return_value = "test text"
    files = [type('File', (), {'filename': 'test.pdf', 'file': type('Stream', (), {'read': lambda: b'%PDF-1.3'})})]
    state = {"task": {"files": files, "type": "upload"}}
    result = pdf_agent.process(state['task'])
    assert len(result) == 1
    assert 'doc_id' in result[0]