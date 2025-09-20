from .base import BaseAgent, chunk_and_embed, rag_query, submit_metrics

class RAGAgent(BaseAgent):
    def process(self, task):
        if task['type'] == 'index':
            for doc_id in task['doc_ids']:
                text = task['texts'][doc_id]  # Assume fetched
                chunk_and_embed(text, doc_id)
            return {"status": "indexed"}
        elif task['type'] == 'query':
            run_id = str(uuid.uuid4())
            result = rag_query(task['question'], task['doc_ids'])
            submit_metrics(run_id, result)
            result['run_id'] = run_id
            return result