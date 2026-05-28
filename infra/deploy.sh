#!/bin/bash
# Deploy both repos to the VM and install dependencies
# Usage: bash infra/deploy.sh
#
# Prerequisites:
#   - VM is running (bash infra/create-vm.sh)
#   - ANTHROPIC_API_KEY is set in your local shell

set -e

PROJECT_ID="my-project-1556791654869"
ZONE="us-east1-b"
VM_NAME="apollena-train"
SSH="gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command"

echo "=== Creating directories on VM ==="

$SSH "mkdir -p ~/Projects"

echo "=== Copying repos to VM (excluding large/unnecessary files) ==="

# Get VM external IP for rsync
VM_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
SSH_KEY=~/.ssh/google_compute_engine

# Apollena — exclude venv and __pycache__
rsync -avz --progress \
  --exclude='venv/' --exclude='__pycache__/' --exclude='.venv/' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ~/Projects/apollena/ "${USER}@${VM_IP}:~/Projects/apollena/"

# FuxiCTR — only copy Frappe_x1 data, exclude Criteo and other large datasets
rsync -avz --progress \
  --exclude='data/Criteo/' --exclude='data/Criteo_x1/' \
  --exclude='data/tiny_*/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='checkpoints/' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ~/Projects/FuxiCTR/ "${USER}@${VM_IP}:~/Projects/FuxiCTR/"

echo "=== Installing Python dependencies ==="

$SSH "pip install --break-system-packages -r ~/Projects/apollena/requirements.txt"
$SSH "cd ~/Projects/FuxiCTR && pip install --break-system-packages -e . 2>/dev/null || pip install --break-system-packages -r requirements.txt 2>/dev/null || echo 'Install FuxiCTR deps manually if needed'"

echo "=== Setting API key ==="

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "WARNING: ANTHROPIC_API_KEY not set locally."
  echo "Set it on the VM manually:"
  echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
  echo "  export ANTHROPIC_API_KEY=your-key-here"
else
  $SSH "echo 'export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY' >> ~/.bashrc"
  echo "API key configured on VM."
fi

echo ""
echo "=== Deployment complete ==="
echo "Connect: gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "Then run:"
echo "  cd ~/Projects/apollena"
echo "  python3 orchestrator.py --domain statistical_physics --resume"
