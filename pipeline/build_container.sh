#!/bin/bash

# setup ECR repository uri
ACCOUNT_ID=$1
REGION=$2
ECR_REPO_NAME=$3

processing_repository_uri=$(sh ./pipeline/get_processing_repo_uri.sh ${ACCOUNT_ID} ${REGION} ${ECR_REPO_NAME})

echo Processing_repo_uri: $processing_repository_uri

# build docker container
docker build -t $ECR_REPO_NAME -f ./pipeline/ml_pipeline_preprocessing_Dockerfile .

# # Login and push the built docker image
$(aws ecr get-login --region ${REGION} --registry-ids ${ACCOUNT_ID} --no-include-email)
docker tag "${ECR_REPO_NAME}:latest" $processing_repository_uri
docker push $processing_repository_uri