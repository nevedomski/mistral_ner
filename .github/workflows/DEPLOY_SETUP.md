# Deploy Workflow Setup

## Required GitHub Secrets

To enable the Google Cloud Run deployment, you need to set up the following secrets in your GitHub repository:

### 1. `GCP_PROJECT_ID`
Your Google Cloud Project ID where the Cloud Run service will be deployed.

### 2. `GCP_SA_KEY`
Service Account key in JSON format with the following permissions:
- Cloud Run Admin
- Service Account User
- Storage Admin (if using Cloud Storage for models)

## Setting up the Service Account

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "IAM & Admin" > "Service Accounts"
3. Click "Create Service Account"
4. Give it a name like "github-actions-deploy"
5. Grant the following roles:
   - Cloud Run Admin
   - Service Account User
   - Storage Admin (optional, if using GCS)
6. Click "Create Key" and choose JSON format
7. Copy the entire JSON content and add it as the `GCP_SA_KEY` secret in GitHub

## Adding Secrets to GitHub

1. Go to your repository on GitHub
2. Navigate to Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Add both secrets:
   - Name: `GCP_PROJECT_ID`, Value: your-project-id
   - Name: `GCP_SA_KEY`, Value: (paste the entire JSON key)

## Testing the Deployment

After setting up the secrets, the deployment will automatically trigger when:
- Code is pushed to the `main` branch
- Any of these paths are modified:
  - `api/**`
  - `src/**`
  - `Dockerfile`
  - `.github/workflows/deploy.yml`

You can also manually trigger the deployment using the "Run workflow" button in the Actions tab.