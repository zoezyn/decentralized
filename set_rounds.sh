#!/bin/bash

# ==========================================
# ⚙️ Quick Round Configuration Tool
# ==========================================
# Usage: ./set_rounds.sh <rounds>
# Example: ./set_rounds.sh 10

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Number of rounds not specified${NC}"
    echo ""
    echo "Usage: $0 <rounds>"
    echo "Example: $0 10"
    echo ""
    exit 1
fi

ROUNDS="$1"

# Validate rounds is a number
if ! [[ "$ROUNDS" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}❌ Error: Rounds must be a positive number${NC}"
    exit 1
fi

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ⚙️  Setting Rounds for All Configs               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Setting num-server-rounds to: $ROUNDS${NC}"
echo ""

# Update all config files
for config in pyproject_fp32.toml pyproject_fp16.toml pyproject_int8.toml; do
    if [ -f "$config" ]; then
        # Use sed to replace the rounds value
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/num-server-rounds = .*/num-server-rounds = $ROUNDS      # ← CHANGE THIS (e.g., 3, 5, 10, 20)/" "$config"
        else
            # Linux
            sed -i "s/num-server-rounds = .*/num-server-rounds = $ROUNDS      # ← CHANGE THIS (e.g., 3, 5, 10, 20)/" "$config"
        fi
        echo -e "${GREEN}✅ Updated: $config${NC}"
    else
        echo -e "${RED}⚠️  Not found: $config${NC}"
    fi
done

echo ""
echo -e "${GREEN}✅ All configurations updated!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Review changes: ${YELLOW}cat pyproject_fp*.toml | grep num-server-rounds${NC}"
echo -e "  2. Upload to cluster: ${YELLOW}./upload_to_cluster.sh <cluster-address>${NC}"
echo ""
