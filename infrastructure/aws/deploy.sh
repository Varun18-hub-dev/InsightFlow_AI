#!/bin/bash
# InsightFlow AI — AWS Deployment Script
# Prerequisites: AWS CLI configured, ECR repos created, ECS cluster running
# Usage: ./infrastructure/aws/deploy.sh [environment]

set -e

ENV=${1:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Starting deployment for $ENV environment in $AWS_REGION..."

ECR_BACKEND_REPO="insightflow-backend"
ECR_FRONTEND_REPO="insightflow-frontend"
ECS_CLUSTER="insightflow-cluster"
ECS_BACKEND_SERVICE="insightflow-backend-service"
ECS_FRONTEND_SERVICE="insightflow-frontend-service"

IMAGE_TAG=$(git rev-parse --short HEAD)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Logging into Amazon ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "Building Backend Image..."
docker build -t $ECR_BACKEND_REPO:$IMAGE_TAG ./backend
docker tag $ECR_BACKEND_REPO:$IMAGE_TAG $ECR_REGISTRY/$ECR_BACKEND_REPO:$IMAGE_TAG
docker tag $ECR_BACKEND_REPO:$IMAGE_TAG $ECR_REGISTRY/$ECR_BACKEND_REPO:latest

echo "Pushing Backend Image..."
docker push $ECR_REGISTRY/$ECR_BACKEND_REPO:$IMAGE_TAG
docker push $ECR_REGISTRY/$ECR_BACKEND_REPO:latest

echo "Building Frontend Image..."
docker build -t $ECR_FRONTEND_REPO:$IMAGE_TAG ./frontend
docker tag $ECR_FRONTEND_REPO:$IMAGE_TAG $ECR_REGISTRY/$ECR_FRONTEND_REPO:$IMAGE_TAG
docker tag $ECR_FRONTEND_REPO:$IMAGE_TAG $ECR_REGISTRY/$ECR_FRONTEND_REPO:latest

echo "Pushing Frontend Image..."
docker push $ECR_REGISTRY/$ECR_FRONTEND_REPO:$IMAGE_TAG
docker push $ECR_REGISTRY/$ECR_FRONTEND_REPO:latest

echo "Updating ECS Services..."
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_BACKEND_SERVICE --force-new-deployment --region $AWS_REGION > /dev/null
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_FRONTEND_SERVICE --force-new-deployment --region $AWS_REGION > /dev/null

echo "Waiting for services to become stable..."
aws ecs wait services-stable --cluster $ECS_CLUSTER --services $ECS_BACKEND_SERVICE $ECS_FRONTEND_SERVICE --region $AWS_REGION

echo "Deployment completed successfully!"
