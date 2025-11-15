#!/bin/bash

# ==========================================
# 🔬 Complete Comparison Workflow
# ==========================================
#
# This script runs the complete comparison:
# 1. Train FP32, FP16, INT8 for 10 minutes each
# 2. Quantize the best FP32 model with real PTQ
# 3. Generate comparison plots and cost analysis
#
# Usage: ./run_comparison.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔬 Federated Learning Quantization Comparison         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

TIME_LIMIT=10  # minutes

echo -e "${YELLOW}Step 1: Training with 10-minute time limit${NC}"
echo -e "  This shows: How far does each precision get in 10 minutes?"
echo ""

# Submit all 3 jobs
echo -e "${GREEN}→ Submitting FP32 job...${NC}"
JOB_FP32=$(./submit_job.sh fp32 100 $TIME_LIMIT | grep "Submitted batch job" | awk '{print $4}')
echo -e "  Job ID: $JOB_FP32"

echo -e "${GREEN}→ Submitting FP16 job...${NC}"
JOB_FP16=$(./submit_job.sh fp16 100 $TIME_LIMIT | grep "Submitted batch job" | awk '{print $4}')
echo -e "  Job ID: $JOB_FP16"

echo -e "${GREEN}→ Submitting INT8 job...${NC}"
JOB_INT8=$(./submit_job.sh int8 100 $TIME_LIMIT | grep "Submitted batch job" | awk '{print $4}')
echo -e "  Job ID: $JOB_INT8"

echo ""
echo -e "${YELLOW}Waiting for jobs to complete...${NC}"
echo -e "  Monitor with: ${BLUE}squeue -u \$USER${NC}"
echo ""

# Wait for all jobs to finish
while squeue -u $USER | grep -q "$JOB_FP32\|$JOB_FP16\|$JOB_INT8"; do
    sleep 10
done

echo -e "${GREEN}✅ All training jobs completed!${NC}"
echo ""

# Find the latest FP32 model
echo -e "${YELLOW}Step 2: Post-Training Quantization (PTQ)${NC}"
echo -e "  This shows: Real compression with actual accuracy measurement"
echo ""

FP32_MODEL=$(find outputs -name "final_model.pt" -path "*/fp32/*" | sort -r | head -1)

if [ -z "$FP32_MODEL" ]; then
    echo -e "${RED}❌ No FP32 model found!${NC}"
    exit 1
fi

echo -e "${GREEN}→ Found FP32 model: $FP32_MODEL${NC}"
echo -e "${GREEN}→ Running quantization comparison...${NC}"
echo ""

python compare_quantizations.py "$FP32_MODEL"

echo ""
echo -e "${YELLOW}Step 3: Generating comparison plots${NC}"
echo ""

python create_comparison_plots.py

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ✅ Comparison Complete!                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results:${NC}"
echo -e "  📊 Training comparison: Check logs/fl-*-r100-t10m-*.out"
echo -e "  📊 PTQ comparison: Check outputs/.../quantization_comparison.json"
echo -e "  📈 Plots: Check outputs/.../comparison_plots.png"
echo ""
