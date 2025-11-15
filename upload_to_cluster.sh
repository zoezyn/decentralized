#!/bin/bash

# ==========================================
# 📦 Upload to Cluster Script
# ==========================================
# Usage: ./upload_to_cluster.sh <cluster-address>
# Example: ./upload_to_cluster.sh user@cluster.example.com

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if cluster address is provided
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Cluster address not provided${NC}"
    echo -e "Usage: $0 <cluster-address>"
    echo -e "Example: $0 user@cluster.example.com"
    exit 1
fi

CLUSTER_ADDR="$1"
LOCAL_DIR="/Users/marty/AIHack-Berlin/flwr_final_hack/berlin25-eurosat"
REMOTE_DIR="~/berlin25-eurosat"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          📦 Uploading to Cluster                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Source:${NC} $LOCAL_DIR"
echo -e "${YELLOW}Target:${NC} $CLUSTER_ADDR:$REMOTE_DIR"
echo ""

# Use rsync for efficient upload (only uploads changed files)
echo -e "${GREEN}→ Syncing files...${NC}"
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'outputs/*' \
    --exclude '.DS_Store' \
    "$LOCAL_DIR/" "$CLUSTER_ADDR:$REMOTE_DIR/"

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. SSH into cluster: ${YELLOW}ssh $CLUSTER_ADDR${NC}"
echo -e "  2. Install dependencies: ${YELLOW}cd ~/berlin25-eurosat && pip install -e .${NC}"
echo -e "  3. Submit job: ${YELLOW}./submit_job.sh fp32${NC}"
echo ""
