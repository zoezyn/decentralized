#!/bin/bash

# ==========================================
# 📤 Push to GitHub
# ==========================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          📤 Pushing to GitHub                             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if gh is authenticated
if ! gh auth status &>/dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI not authenticated${NC}"
    echo ""
    echo -e "${BLUE}Please authenticate with GitHub:${NC}"
    echo -e "  ${YELLOW}gh auth login${NC}"
    echo ""
    echo -e "Then run this script again."
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ GitHub CLI authenticated${NC}"
echo ""

# Create repository and push
REPO_NAME="berlin25-eurosat-deployment"
DESCRIPTION="Streamlined Flower FL deployment with time-based training for EuroSAT satellite imagery - Berlin Hackathon 2025"

echo -e "${YELLOW}→ Creating GitHub repository: $REPO_NAME${NC}"
echo -e "${YELLOW}→ Description: $DESCRIPTION${NC}"
echo ""

gh repo create "$REPO_NAME" \
    --public \
    --source=. \
    --description="$DESCRIPTION" \
    --push

echo ""
echo -e "${GREEN}✅ Repository created and code pushed!${NC}"
echo ""

# Get repository URL
REPO_URL=$(gh repo view --json url -q .url)

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ✅ Success!                                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Repository URL:${NC} $REPO_URL"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. On cluster: ${YELLOW}git clone $REPO_URL${NC}"
echo -e "  2. Navigate: ${YELLOW}cd berlin25-eurosat-deployment${NC}"
echo -e "  3. Follow: ${YELLOW}cat START_HERE.md${NC}"
echo ""
