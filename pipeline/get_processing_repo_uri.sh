#!/bin/bash

# setup ECR repository uri
ACCOUNT_ID=$1
REGION=$2
ECR_REPO_NAME=$3
TAG=':latest'

uri_suffix='amazonaws.com'
CHINA_REGIONS='cn-north-1 cn-northwest-1'
if [[ " $CHINA_REGIONS " =~ .*\ $REGION\ .* ]]; then
    uri_suffix='amazonaws.com.cn'
fi
echo "${ACCOUNT_ID}.dkr.ecr.${REGION}.${uri_suffix}/${ECR_REPO_NAME}${TAG}"