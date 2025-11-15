#!/bin/bash

# ==========================================
# 🚀 Submit Federated Learning Job
# ==========================================
# Usage: ./submit_job.sh <precision> [rounds] [time_limit_minutes]
# Examples:
#   ./submit_job.sh fp32              # Uses config file settings
#   ./submit_job.sh fp16 10           # Override: 10 rounds, config time limit
#   ./submit_job.sh int8 20 10        # Override: 20 rounds OR 10 minutes (whichever comes first)
#   ./submit_job.sh fp32 100 10       # Simulate: 100 max rounds but stop after 10 minutes

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Validate precision argument
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Precision not specified${NC}"
    echo ""
    echo "Usage: $0 <precision> [rounds]"
    echo ""
    echo "Available precisions:"
    echo "  fp32  - 32-bit floating point (baseline)"
    echo "  fp16  - 16-bit half precision"
    echo "  int8  - 8-bit quantized"
    echo ""
    echo "Examples:"
    echo "  $0 fp32        # Run FP32 with config file rounds"
    echo "  $0 fp16 10     # Run FP16 with 10 rounds"
    exit 1
fi

PRECISION="$1"
OVERRIDE_ROUNDS="$2"
OVERRIDE_TIME_LIMIT="$3"

# Validate precision
if [[ ! "$PRECISION" =~ ^(fp32|fp16|int8)$ ]]; then
    echo -e "${RED}❌ Invalid precision: $PRECISION${NC}"
    echo "Must be one of: fp32, fp16, int8"
    exit 1
fi

# Map precision to config file
CONFIG_FILE="pyproject_${PRECISION}.toml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Start with original config
cp "$CONFIG_FILE" pyproject.toml

# Override rounds if specified
if [ -n "$OVERRIDE_ROUNDS" ]; then
    echo -e "${YELLOW}⚠️  Overriding max rounds to: $OVERRIDE_ROUNDS${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/num-server-rounds = .*/num-server-rounds = $OVERRIDE_ROUNDS/" pyproject.toml
    else
        sed -i "s/num-server-rounds = .*/num-server-rounds = $OVERRIDE_ROUNDS/" pyproject.toml
    fi
fi

# Override time limit if specified
if [ -n "$OVERRIDE_TIME_LIMIT" ]; then
    echo -e "${YELLOW}⏱️  Overriding time limit to: $OVERRIDE_TIME_LIMIT minutes${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/time-limit-minutes = .*/time-limit-minutes = $OVERRIDE_TIME_LIMIT/" pyproject.toml
    else
        sed -i "s/time-limit-minutes = .*/time-limit-minutes = $OVERRIDE_TIME_LIMIT/" pyproject.toml
    fi
fi

# Extract values for display
ROUNDS=$(grep "num-server-rounds" pyproject.toml | sed 's/.*= //' | sed 's/ .*//')
TIME_LIMIT=$(grep "time-limit-minutes" pyproject.toml | sed 's/.*= //' | sed 's/ .*//')

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          🚀 Submitting Federated Learning Job             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Precision:${NC}       $PRECISION"
echo -e "${GREEN}Config File:${NC}     $CONFIG_FILE"
echo -e "${GREEN}Max Rounds:${NC}      $ROUNDS"
if [ "$TIME_LIMIT" != "0" ]; then
    echo -e "${YELLOW}⏱️  Time Limit:${NC}   $TIME_LIMIT minutes (stops when reached)"
else
    echo -e "${GREEN}Time Limit:${NC}      None (runs until $ROUNDS rounds complete)"
fi
echo -e "${GREEN}Clients:${NC}         10 satellites"
echo -e "${GREEN}Local Epochs:${NC}    2"
echo ""

# Create job script
if [ "$TIME_LIMIT" != "0" ]; then
    JOB_NAME="fl-${PRECISION}-r${ROUNDS}-t${TIME_LIMIT}m"
else
    JOB_NAME="fl-${PRECISION}-r${ROUNDS}"
fi
JOB_FILE="${JOB_NAME}.slurm"

cat > "$JOB_FILE" << 'EOF'
#!/bin/bash
#SBATCH --job-name=fl-JOB_PRECISION
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu_qos

# Print job info
echo "=========================================="
echo "Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Activate cluster's pre-configured environment with flwr-datasets
source ~/hackathon-venv-flwr-datasets/bin/activate

# Set MIOpen cache to writable directory (AMD GPU fix)
export MIOPEN_USER_DB_PATH=$SLURM_SUBMIT_DIR/.miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$SLURM_SUBMIT_DIR/.miopen_cache
mkdir -p $MIOPEN_USER_DB_PATH

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "MIOpen cache: $MIOPEN_USER_DB_PATH"
echo ""

# Run Flower
cd $SLURM_SUBMIT_DIR
echo "Running Flower with configuration:"
cat pyproject.toml | grep -A 6 "\[tool.flwr.app.config\]"
echo ""

flwr run . cluster-gpu

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Check outputs: ls -lh ~/berlin25-eurosat/outputs/"
echo "=========================================="
EOF

# Replace placeholders
sed -i.bak "s/JOB_PRECISION/$PRECISION/g" "$JOB_FILE"
rm "${JOB_FILE}.bak" 2>/dev/null || true

# Create logs directory
mkdir -p logs

# Submit job
echo -e "${YELLOW}→ Submitting job to cluster...${NC}"
sbatch "$JOB_FILE"

echo ""
echo -e "${GREEN}✅ Job submitted successfully!${NC}"
echo ""
echo -e "${BLUE}Monitor job:${NC}"
echo -e "  squeue -u \$USER                           # Check job status"
echo -e "  tail -f logs/${JOB_NAME}-<job-id>.out      # Watch live output"
echo ""
echo -e "${BLUE}Check outputs:${NC}"
echo -e "  ls -lh outputs/                            # View saved models"
echo ""
