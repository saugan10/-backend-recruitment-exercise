from fastapi import FastAPI
from pydantic import BaseModel
import os
import boto3

app = FastAPI()
dynamodb = boto3.resource('dynamodb', endpoint_url=os.environ['DYNAMODB_ENDPOINT'], region_name='us-east-1')
metrics_table = dynamodb.Table('AgentMetrics')

class Metrics(BaseModel):
    run_id: str
    agent_name: str
    tokens_consumed: int
    tokens_generated: int
    response_time_ms: int
    confidence_score: float
    timestamp: str
    status: str

@app.post("/metrics")
async def store_metrics(metrics: Metrics):
    metrics_table.put_item(Item=metrics.dict())
    return {"status": "success"}