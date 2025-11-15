#!/bin/bash

# ==========================================
# ⏱️ Submit Time-Based Comparison Jobs
# ==========================================
# Run all 3 precisions with same time limit
# Usage: ./submit_time_comparison.sh <time_limit_minutes>
# Example: ./submit_time_comparison.sh 10

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Time limit not specified${NC}"
    echo ""
    echo "Usage: $0 <time_limit_minutes>"
    echo "Example: $0 10    # Run all 3 precisions for 10 minutes each"
    echo ""
    exit 1
fi

TIME_LIMIT="$1"

# Validate time limit is a number
if ! [[ "$TIME_LIMIT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}❌ Error: Time limit must be a positive number${NC}"
    exit 1
fi

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      ⏱️  Submitting Time-Based Comparison Jobs            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Time Limit:${NC}      $TIME_LIMIT minutes"
echo -e "${YELLOW}Max Rounds:${NC}      100 (will stop at time limit)"
echo -e "${YELLOW}Jobs:${NC}            FP32, FP16, INT8"
echo ""
echo -e "${BLUE}This simulates:${NC} \"How many rounds can each precision complete in $TIME_LIMIT minutes?\""
echo ""

# Submit all 3 jobs with same time limit
echo -e "${GREEN}→ Submitting FP32 job...${NC}"
./submit_job.sh fp32 100 "$TIME_LIMIT"
echo ""

echo -e "${GREEN}→ Submitting FP16 job...${NC}"
./submit_job.sh fp16 100 "$TIME_LIMIT"
echo ""

echo -e "${GREEN}→ Submitting INT8 job...${NC}"
./submit_job.sh int8 100 "$TIME_LIMIT"
echo ""

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ All 3 jobs submitted!${NC}"
echo ""
echo -e "${BLUE}Monitor:${NC}"
echo -e "  squeue -u \$USER"
echo -e "  tail -f logs/fl-*-t${TIME_LIMIT}m-*.out"
echo ""
echo -e "${BLUE}Compare results after completion:${NC}"
echo -e "  Each job will show how many rounds completed in $TIME_LIMIT minutes"
echo -e "  Check accuracy at time limit for each precision"
echo ""
