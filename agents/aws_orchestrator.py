from .base import BaseAgent
from boto3.dynamodb.conditions import Key

class AWSOrchestratorAgent(BaseAgent):
    def __init__(self, rag_agent):
        self.rag_agent = rag_agent

    def process(self, task):
        if task['type'] == 'create':
            store_metadata(task['doc_id'], task['metadata'])
        elif task['type'] == 'get':
            item = docs_table.get_item(Key={'doc_id': task['doc_id']}).get('Item')
            return item
        elif task['type'] == 'update':
            docs_table.update_item(Key={'doc_id': task['doc_id']}, UpdateExpression="SET tags = :tags", ExpressionAttributeValues={":tags": task['tags']})
        elif task['type'] == 'delete':
            docs_table.delete_item(Key={'doc_id': task['doc_id']})
            try:
                s3.delete_object(Bucket=bucket, Key=f"{task['doc_id']}.pdf")
            except:
                pass
        elif task['type'] == 'index':
            self.rag_agent.process({'type': 'index', 'doc_ids': [task['doc_id']], 'texts': task['texts']})
            return {"status": "indexed"}
        elif task['type'] == 'query':
            return self.rag_agent.process({'type': 'query', 'doc_ids': task['doc_ids'], 'question': task['question']})