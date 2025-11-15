#!/bin/bash

# ==========================================
# ✅ Validate Deployment Setup
# ==========================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ✅ Validating Deployment Setup                   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

ERRORS=0
WARNINGS=0

# Check required files exist
echo -e "${YELLOW}Checking files...${NC}"
FILES=(
    "upload_to_cluster.sh"
    "submit_job.sh"
    "set_rounds.sh"
    "pyproject_fp32.toml"
    "pyproject_fp16.toml"
    "pyproject_int8.toml"
    "eurosat/server_app.py"
    "eurosat/client_app.py"
    "eurosat/task.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file ${RED}(MISSING)${NC}"
        ((ERRORS++))
    fi
done
echo ""

# Check scripts are executable
echo -e "${YELLOW}Checking executables...${NC}"
SCRIPTS=("upload_to_cluster.sh" "submit_job.sh" "set_rounds.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script is executable"
    else
        echo -e "  ${RED}✗${NC} $script ${RED}(NOT EXECUTABLE)${NC}"
        echo -e "    Fix: ${YELLOW}chmod +x $script${NC}"
        ((ERRORS++))
    fi
done
echo ""

# Check wandb removed from server_app.py
echo -e "${YELLOW}Checking WandB removal...${NC}"
if grep -q "import wandb" eurosat/server_app.py; then
    echo -e "  ${RED}✗${NC} WandB still imported in server_app.py"
    ((ERRORS++))
else
    echo -e "  ${GREEN}✓${NC} WandB import removed"
fi

if grep -q "wandb.init" eurosat/server_app.py; then
    echo -e "  ${RED}✗${NC} wandb.init() still present"
    ((ERRORS++))
else
    echo -e "  ${GREEN}✓${NC} wandb.init() removed"
fi

if grep -q "wandb.log" eurosat/server_app.py; then
    echo -e "  ${RED}✗${NC} wandb.log() still present"
    ((ERRORS++))
else
    echo -e "  ${GREEN}✓${NC} wandb.log() removed"
fi
echo ""

# Check console logging added
echo -e "${YELLOW}Checking console logging...${NC}"
if grep -q "print(f\"Round" eurosat/server_app.py; then
    echo -e "  ${GREEN}✓${NC} Console logging added"
else
    echo -e "  ${YELLOW}⚠${NC} Console logging may be missing"
    ((WARNINGS++))
fi
echo ""

# Check configs have correct settings
echo -e "${YELLOW}Checking configurations...${NC}"
for config in pyproject_fp32.toml pyproject_fp16.toml pyproject_int8.toml; do
    if grep -q "num-server-rounds" "$config"; then
        ROUNDS=$(grep "num-server-rounds" "$config" | sed 's/.*= //' | sed 's/ .*//')
        echo -e "  ${GREEN}✓${NC} $config → $ROUNDS rounds"
    else
        echo -e "  ${RED}✗${NC} $config missing num-server-rounds"
        ((ERRORS++))
    fi
done
echo ""

# Check dependency simplification
echo -e "${YELLOW}Checking dependencies...${NC}"
for config in pyproject_fp32.toml pyproject_fp16.toml pyproject_int8.toml; do
    if grep -q "torch==" "$config"; then
        echo -e "  ${YELLOW}⚠${NC} $config has pinned torch version (should be unpinned for cluster)"
        ((WARNINGS++))
    else
        echo -e "  ${GREEN}✓${NC} $config uses unpinned dependencies"
    fi
done
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready for deployment.${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Set rounds: ${YELLOW}./set_rounds.sh 10${NC}"
    echo -e "  2. Upload: ${YELLOW}./upload_to_cluster.sh user@cluster.ai${NC}"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠  $WARNINGS warning(s) - deployment should work but review above${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found - fix before deploying${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠  $WARNINGS warning(s) also present${NC}"
    fi
    echo ""
    exit 1
fi
