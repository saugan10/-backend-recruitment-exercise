#!/bin/bash
awslocal s3 mb s3://my-pdf-bucket
awslocal dynamodb create-table --table-name pdf-metadata --attribute-definitions AttributeName=id,AttributeType=S --key-schema AttributeName=id,KeyType=HASH --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
awslocal dynamodb create-table --table-name AgentMetrics --attribute-definitions AttributeName=timestamp,AttributeType=S --key-schema AttributeName=timestamp,KeyType=HASH --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5