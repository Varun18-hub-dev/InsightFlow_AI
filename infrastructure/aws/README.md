# AWS Infrastructure Setup for InsightFlow AI

This directory contains configuration templates for deploying InsightFlow AI to AWS.

## Prerequisites
- AWS CLI configured
- Domain name managed via Route 53 (optional, for custom domain)
- ACM certificates (for ALB and CloudFront HTTPS)

## 1. VPC and Networking
Create a VPC with:
- 2 Public Subnets (for ALB and NAT Gateway)
- 2 Private Subnets (for ECS tasks, RDS, ElastiCache)
- NAT Gateway in public subnets to allow ECS tasks to access the internet.

## 2. Secrets Management
Store the following in **AWS Systems Manager Parameter Store** (SecureString):
- `/insightflow/DATABASE_URL`
- `/insightflow/GEMINI_API_KEY`
- `/insightflow/SECRET_KEY`

## 3. Databases (RDS & ElastiCache)
- **RDS**: PostgreSQL 15 multi-AZ deployment in private subnets.
- **ElastiCache**: Redis 7 cluster in private subnets.

## 4. ECR (Elastic Container Registry)
Create two repositories:
- `insightflow-backend`
- `insightflow-frontend`

## 5. ECS (Elastic Container Service)
- Create an ECS Cluster named `insightflow-cluster`.
- Use the JSON templates in this directory to register task definitions. Replace placeholders (`${AWS_ACCOUNT_ID}`, etc.) with your actual values.
- Create Fargate services for both backend and frontend.

## 6. ALB (Application Load Balancer)
- Create an ALB in public subnets.
- Create Target Groups for backend (port 8000) and frontend (port 3000).
- Configure Listener Rules (see `alb-config.json`).

## 7. S3 and CloudFront
- Create an S3 bucket for document uploads (apply `s3-bucket-policy.json`).
- Create a CloudFront distribution pointing to the ALB (see `cloudfront-distribution.json`).

## Deployment
Use the included `deploy.sh` script or the GitHub Actions CD pipeline to deploy updates.
