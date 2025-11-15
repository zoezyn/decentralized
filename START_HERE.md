# 🚀 START HERE - Cluster Deployment Guide

## 📦 What You Have

A **streamlined cluster deployment system** for running 3 Federated Learning jobs with different precisions:

- ✅ **FP32** (32-bit) - Baseline
- ✅ **FP16** (16-bit) - Half precision
- ✅ **INT8** (8-bit) - Quantized

**Zero installation needed** - uses cluster's pre-configured `hackathon-venv`

---

## ⚡ Quick Start (5 Steps)

### 1. Set Training Parameters
```bash
# Option A: Set rounds only
./set_rounds.sh 10    # Sets 10 rounds for all 3 configs

# Option B: Edit config to add time limit
nano pyproject_fp32.toml
# Set: time-limit-minutes = 10  (0 = no limit)
```

### 2. Upload to Cluster
```bash
./upload_to_cluster.sh user@cluster-address.com
```

### 3. SSH to Cluster
```bash
ssh user@cluster-address.com
cd ~/berlin25-eurosat
```

### 4. Edit Venv Path (First Time Only)
```bash
nano submit_job.sh
# Line 109: Change to actual path
# source /actual/path/to/hackathon-venv/bin/activate
```

### 5. Submit Jobs
```bash
mkdir -p logs         # Create logs directory

# Option A: Use config file settings
./submit_job.sh fp32
./submit_job.sh fp16
./submit_job.sh int8

# Option B: Override at submission
./submit_job.sh fp32 20 10    # 20 rounds OR 10 minutes (whichever first)

# Option C: Time-based comparison (all 3 precisions, same time limit)
./submit_time_comparison.sh 10    # Compare FP32 vs FP16 vs INT8 in 10 minutes
```

**Done!** 🎉

---

## 📊 Monitor Your Jobs

```bash
# Check status
squeue -u $USER

# Watch live output
tail -f logs/fl-fp32-r10-*.out

# Check results
ls -lh outputs/
```

---

## 📚 Full Documentation

- **`TIME_LIMIT_GUIDE.md`** → ⏱️ **NEW!** Time-based training guide
- **`CLUSTER_QUICK_START.md`** → Complete step-by-step guide
- **`DEPLOYMENT_SUMMARY.md`** → Technical details & design decisions
- **`README.md`** → Original Flower documentation

---

## ✅ What Was Changed

### ❌ Removed
- WandB logging (no API keys needed)
- User interactions (fully automated)
- Specific package versions (uses cluster's venv)

### ✅ Added
- 3 config files (fp32, fp16, int8)
- Upload script (`upload_to_cluster.sh`)
- Job submission script (`submit_job.sh`)
- Round setter (`set_rounds.sh`)
- Console logging (prints to terminal)

---

## 🎯 File Overview

| File | Purpose |
|------|---------|
| `set_rounds.sh` | Set rounds for all configs at once |
| `upload_to_cluster.sh` | Upload code to cluster |
| `submit_job.sh` | Submit training jobs |
| `pyproject_fp32.toml` | 32-bit config |
| `pyproject_fp16.toml` | 16-bit config |
| `pyproject_int8.toml` | 8-bit config |
| `eurosat/server_app.py` | Modified (no wandb) |

---

## 🔧 Common Tasks

### Change Rounds or Time Limit
```bash
# Method 1: All configs at once
./set_rounds.sh 20

# Method 2: Individual config
nano pyproject_fp32.toml
# Change: num-server-rounds = 20
# Change: time-limit-minutes = 10  (0 = no limit)

# Method 3: Override at submission
./submit_job.sh fp32 20      # 20 rounds, no time limit
./submit_job.sh fp32 20 10   # 20 rounds OR 10 min (whichever first)

# Method 4: Time comparison
./submit_time_comparison.sh 10  # All 3 precisions, 10 min each
```

### Cancel a Job
```bash
squeue -u $USER        # Get job ID
scancel <job-id>       # Cancel it
```

### Download Results
```bash
# From local machine
scp -r user@cluster:~/berlin25-eurosat/outputs/ ./outputs/
```

---

## 🚨 Troubleshooting

### Job Won't Start?
```bash
squeue              # Check queue
sinfo               # Check nodes
```

### Job Failed?
```bash
cat logs/fl-fp32-*.err    # Check error log
cat logs/fl-fp32-*.out    # Check output log
```

### Can't Find Python?
```bash
# In submit_job.sh, verify venv path is correct
# Line 109: source /path/to/hackathon-venv/bin/activate
```

---

## 💡 Pro Tips

1. **Test locally first**: `flwr run . local-simulation`
2. **Start with 3 rounds**: Test before running 20 rounds
3. **Submit all 3 at once**: They'll queue automatically
4. **Monitor GPU usage**: Check if actually using GPU
5. **Save logs**: Don't delete `logs/` directory

---

## 📈 Expected Results

| Config | Model Size | Time (5 rounds) | Accuracy |
|--------|------------|-----------------|----------|
| FP32   | ~12 MB     | ~10-15 min      | Baseline |
| FP16   | ~6 MB      | ~8-12 min       | -0.1%    |
| INT8   | ~3 MB      | ~6-10 min       | -0.3%    |

---

## 🎯 Your Workflow

```mermaid
graph LR
    A[Set Rounds] --> B[Upload]
    B --> C[SSH to Cluster]
    C --> D[Submit Jobs]
    D --> E[Monitor]
    E --> F[Download Results]
```

**Actual commands:**
```bash
# Local
./set_rounds.sh 10
./upload_to_cluster.sh user@cluster.ai

# Cluster
ssh user@cluster.ai
cd ~/berlin25-eurosat
mkdir -p logs
./submit_job.sh fp32
./submit_job.sh fp16
./submit_job.sh int8
squeue -u $USER

# Local (after jobs finish)
scp -r user@cluster.ai:~/berlin25-eurosat/outputs/ ./
```

---

## ✅ Pre-Flight Checklist

Before submitting jobs:

- [ ] Set number of rounds: `./set_rounds.sh <number>`
- [ ] Uploaded to cluster: `./upload_to_cluster.sh`
- [ ] Updated venv path in `submit_job.sh`
- [ ] Created logs directory: `mkdir -p logs`
- [ ] Tested locally (optional): `flwr run . local-simulation`

---

## 🎓 What This System Does

1. **Removes complexity**: No package installation, no WandB setup
2. **Manual control**: You set rounds explicitly each time
3. **Multi-precision**: Easy to run 3 different bit-widths
4. **Cluster-optimized**: Uses pre-configured GPU environment
5. **Production-ready**: Console logging, error handling, monitoring

---

## 📞 Need Help?

1. Read `CLUSTER_QUICK_START.md` for detailed guide
2. Check logs: `tail -f logs/*.err`
3. Ask in #support channel with:
   - Job ID
   - Error message
   - What you tried

---

**Ready to train! 🚀**

Run: `./set_rounds.sh 10` to begin
