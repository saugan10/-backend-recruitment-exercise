@echo off
awslocal s3 mb s3://my-pdf-bucket
awslocal dynamodb create-table --table-name DocumentsMetadata --attribute-definitions AttributeName=doc_id,AttributeType=S --key-schema AttributeName=doc_id,KeyType=HASH --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
awslocal dynamodb create-table --table-name AgentMetrics --attribute-definitions AttributeName=run_id,AttributeType=S AttributeName=timestamp,AttributeType=S --key-schema AttributeName=run_id,KeyType=HASH AttributeName=timestamp,KeyType=RANGE --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5