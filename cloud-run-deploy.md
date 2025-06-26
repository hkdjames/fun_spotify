# 🚀 Google Cloud Run Deployment Guide

Deploy your Spotify Dashboard to Google Cloud Run for scalable, serverless hosting.

## 📋 Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI** installed and authenticated
3. **Docker** installed locally (optional, uses Cloud Build)

## 🛠️ Setup Instructions

### 1. Install Google Cloud CLI
```bash
# macOS
brew install --cask google-cloud-sdk

# Windows
# Download from https://cloud.google.com/sdk/docs/install

# Linux
curl https://sdk.cloud.google.com | bash
```

### 2. Authenticate and Setup
```bash
# Login to Google Cloud
gcloud auth login

# Set your project (replace with your project ID)
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Quick Deployment
```bash
# Make deployment script executable
chmod +x deploy.sh

# Edit PROJECT_ID in deploy.sh
# Then run:
./deploy.sh
```

## 🔧 Manual Deployment Steps

### 1. Build and Push Image
```bash
# Set variables
PROJECT_ID="your-project-id"
SERVICE_NAME="spotify-dashboard"
REGION="us-central1"

# Build image using Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
```

### 2. Deploy to Cloud Run
```bash
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --port 8080
```

## 📊 Resource Configuration

- **Memory**: 2GB (handles large datasets)
- **CPU**: 2 vCPUs (for data processing)
- **Timeout**: 1 hour (for initial data loading)
- **Max Instances**: 10 (auto-scaling)
- **Port**: 8080 (Streamlit default for Cloud Run)

## 💰 Cost Optimization

### Free Tier Limits
- **2 million requests** per month
- **400,000 GB-seconds** of memory
- **200,000 vCPU-seconds**

### Cost-Saving Tips
1. **Reduce memory/CPU** if your dataset is smaller
2. **Set max-instances** to control scaling
3. **Add authentication** to limit usage
4. **Use caching** to reduce compute time

## 🔒 Security Options

### Add Authentication
```bash
# Deploy with authentication required
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --no-allow-unauthenticated

# Grant access to specific users
gcloud run services add-iam-policy-binding $SERVICE_NAME \
    --member="user:email@example.com" \
    --role="roles/run.invoker"
```

### Environment Variables
```bash
# Add environment variables during deployment
gcloud run deploy $SERVICE_NAME \
    --set-env-vars="ENV=production,LOG_LEVEL=info"
```

## 📈 Monitoring & Logs

### View Logs
```bash
# Real-time logs
gcloud run services logs tail $SERVICE_NAME --region $REGION

# Historical logs in Cloud Console
# Navigation: Cloud Run > Your Service > Logs
```

### Monitoring
- **Cloud Monitoring** - Automatic metrics
- **Error Reporting** - Crash detection
- **Cloud Trace** - Performance analysis

## 🔄 Updates & Rollbacks

### Update Deployment
```bash
# Rebuild and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --image gcr.io/$PROJECT_ID/$SERVICE_NAME
```

### Rollback
```bash
# List revisions
gcloud run revisions list --service $SERVICE_NAME

# Rollback to previous revision
gcloud run services update-traffic $SERVICE_NAME --to-revisions=REVISION-NAME=100
```

## 🌐 Custom Domain (Optional)

### Map Custom Domain
```bash
# Add domain mapping
gcloud run domain-mappings create \
    --service $SERVICE_NAME \
    --domain your-domain.com \
    --region $REGION
```

## 🚨 Troubleshooting

### Common Issues
1. **Out of Memory**: Increase memory allocation
2. **Timeout**: Increase timeout or optimize data loading
3. **Build Failures**: Check Dockerfile and dependencies
4. **Permission Errors**: Verify IAM roles

### Debug Deployment
```bash
# Check service status
gcloud run services describe $SERVICE_NAME --region $REGION

# View recent logs
gcloud run services logs read $SERVICE_NAME --region $REGION --limit 50
```

## 🎯 Production Recommendations

1. **Use Artifact Registry** instead of Container Registry
2. **Set up CI/CD** with GitHub Actions or Cloud Build triggers
3. **Enable VPC connector** for database access
4. **Configure health checks**
5. **Set up monitoring alerts**

## 💡 Alternative Options

### Streamlit Cloud (Easiest)
- Free hosting for public apps
- Direct GitHub integration
- Limited resources

### Google App Engine
- Alternative serverless option
- Different pricing model
- Built-in scaling

### Compute Engine
- Full VM control
- More cost-effective for constant usage
- Requires more management

---

Your Spotify Dashboard will be live at: `https://SERVICE_NAME-HASH-REGION.run.app` 