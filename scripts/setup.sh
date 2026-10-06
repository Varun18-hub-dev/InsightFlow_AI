#!/bin/bash
# First-time setup for InsightFlow AI

set -e

echo "======================================"
echo "InsightFlow AI - Setup Script"
echo "======================================"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is required but it's not installed. Aborting."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || { echo >&2 "Docker Compose is required but it's not installed. Aborting."; exit 1; }

# Copy .env
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "IMPORTANT: Please update .env with your API keys!"
else
    echo ".env file already exists."
fi

# Build containers
echo "Building Docker containers..."
docker compose build

echo "Starting detached containers..."
docker compose up -d postgres redis mlflow

echo "Waiting for database to be ready..."
sleep 5

echo "Setup complete! You can now start the application with:"
echo "docker compose up"
