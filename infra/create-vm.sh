#!/bin/bash
# Create a GCP VM for Apollena training
# Usage: bash infra/create-vm.sh

set -e

PROJECT_ID="my-project-1556791654869"
ZONE="us-east1-b"
VM_NAME="apollena-train"
MACHINE_TYPE="e2-standard-4"  # 4 vCPU, 16GB RAM — good for Frappe_x1 Optuna

echo "Creating VM: $VM_NAME ($MACHINE_TYPE) in $ZONE..."

gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --metadata-from-file=startup-script=infra/vm-startup.sh \
  --tags=apollena

echo ""
echo "VM created. To connect:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "To stop when done:"
echo "  gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "To delete when no longer needed:"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
