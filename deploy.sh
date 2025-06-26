#!/bin/bash

# Spotify Dashboard Cloud Run Deployment Script
# Make sure you have Google Cloud CLI installed and authenticated

# Configuration
PROJECT_ID="your-project-id"  # Replace with your Google Cloud Project ID
SERVICE_NAME="spotify-dashboard"
REGION="us-central1"  # Change to your preferred region
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🎵 Deploying Spotify Dashboard to Google Cloud Run..."

# Set the project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the Docker image
echo "Building Docker image..."
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --port 8080

echo "✅ Deployment complete!"
echo "🌐 Your Spotify Dashboard is now available at:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)' 