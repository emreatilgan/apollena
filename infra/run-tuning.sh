#!/bin/bash
# Run Optuna tuning on the VM, then auto-shutdown
# Usage: SSH into VM, then: bash ~/Projects/apollena/infra/run-tuning.sh
#
# The VM will shut itself down when training completes (or fails),
# so you won't pay for idle time.

set -e

FUXICTR_DIR=~/Projects/FuxiCTR
APOLLENA_DIR=~/Projects/apollena
RESULTS_DIR=$APOLLENA_DIR/results

echo "=== Starting Optuna tuning: TAPNet on Frappe_x1 ==="
echo "VM will auto-shutdown when complete."
echo "Start time: $(date)"
echo ""

cd $FUXICTR_DIR/model_zoo/TAPNet/TAPNet_torch

# Run the tuner
python3 run_expid.py \
  --config ./config/TAPNet_frappe_x1_tuner_config \
  --expid TAPNet_default \
  --gpu -1 \
  2>&1 | tee $RESULTS_DIR/tuning_log.txt

echo ""
echo "=== Tuning complete ==="
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR/tuning_log.txt"
echo "Tuner CSV: $FUXICTR_DIR/model_zoo/TAPNet/TAPNet_torch/TAPNet_frappe_x1_tuner_config.csv"

# Copy results back to apollena
cp $FUXICTR_DIR/model_zoo/TAPNet/TAPNet_torch/TAPNet_frappe_x1_tuner_config.csv \
   $RESULTS_DIR/tuner_results.csv 2>/dev/null || true

echo ""
echo "Shutting down VM in 60 seconds... (Ctrl+C to cancel)"
sleep 60
sudo shutdown -h now
