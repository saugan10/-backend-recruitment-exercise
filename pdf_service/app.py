from fastapi import FastAPI, UploadFile
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.pdf_agent import PDFAgent
from agents.rag_agent import RAGAgent
from agents.aws_orchestrator import AWSOrchestratorAgent

app = FastAPI()
pdf_agent = PDFAgent()
rag_agent = RAGAgent()
aws_orchestrator = AWSOrchestratorAgent(rag_agent)

class State(TypedDict):
    task: dict
    result: dict

def pdf_node(state: State):
    state['result'] = pdf_agent.process(state['task'])
    return state

def rag_node(state: State):
    state['result'] = rag_agent.process(state['task'])
    return state

def aws_node(state: State):
    state['result'] = aws_orchestrator.process(state['task'])
    return state

graph = StateGraph(State)
graph.add_node("pdf", pdf_node)
graph.add_node("rag", rag_node)
graph.add_node("aws", aws_node)
graph.add_conditional_edges("aws", lambda s: "rag" if s['task']['type'] in ['index', 'query'] else "pdf" if s['task']['type'] == 'upload' else END)
graph.set_entry_point("aws")
compiled_graph = graph.compile()

@app.post("/pdf/upload")
async def upload_pdf(files: List[UploadFile]):
    state = {"task": {"files": files, "type": "upload"}}
    return compiled_graph.invoke(state)['result']