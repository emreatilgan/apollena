#!/bin/bash
# VM startup script — runs once on first boot
# Installs Python, pip, and base dependencies

set -e

apt-get update
apt-get install -y python3.10 python3-pip python3.10-venv git

echo "VM setup complete. Ready for repo deployment."
