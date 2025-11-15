# ⏱️ Time-Based Training Guide

## Overview

You can now **limit training by time** instead of just rounds. This is perfect for:
- **Fair comparisons** between different precisions (FP32, FP16, INT8)
- **Simulating real-world constraints** (e.g., "train for 10 minutes max")
- **Quick experiments** when you don't know ideal round count

---

## 🎯 How It Works

Training stops when **EITHER** condition is met:
1. **Max rounds completed** (e.g., 20 rounds)
2. **Time limit reached** (e.g., 10 minutes)

**Whichever comes first!**

---

## ⚙️ Setting Time Limits

### Method 1: Config File (Before Upload)

Edit the config files:
```toml
[tool.flwr.app.config]
num-server-rounds = 20     # Max rounds
time-limit-minutes = 10    # Time limit (0 = no limit)
```

Then upload and submit:
```bash
./upload_to_cluster.sh user@cluster.ai
ssh user@cluster.ai
./submit_job.sh fp32       # Uses 20 rounds OR 10 min (whichever first)
```

### Method 2: Command-Line Override

```bash
./submit_job.sh fp32 20 10
#                    │   └─ Time limit: 10 minutes
#                    └───── Max rounds: 20
```

### Method 3: Time-Based Comparison Script

Run **all 3 precisions with same time limit**:
```bash
./submit_time_comparison.sh 10
```

This submits:
- FP32: 100 max rounds OR 10 min
- FP16: 100 max rounds OR 10 min
- INT8: 100 max rounds OR 10 min

Perfect for comparing: "How many rounds can each precision complete in 10 minutes?"

---

## 📊 Example Use Cases

### 1. Fair Precision Comparison
**Goal**: Compare FP32 vs FP16 vs INT8 in same time

```bash
# All 3 jobs run for exactly 10 minutes
./submit_time_comparison.sh 10

# Results will show:
# - FP32: Completed 8 rounds in 10 min → 85% accuracy
# - FP16: Completed 10 rounds in 10 min → 84.9% accuracy
# - INT8: Completed 12 rounds in 10 min → 84.7% accuracy
```

### 2. Quick Experiment
**Goal**: Test if model works, don't wait for 20 rounds

```bash
# Stop after 5 minutes regardless of rounds
./submit_job.sh fp32 100 5
```

### 3. Long Training with Safety
**Goal**: Train as much as possible but stop by deadline

```bash
# Run up to 50 rounds but guarantee finish in 30 min
./submit_job.sh fp32 50 30
```

### 4. No Time Limit (Traditional)
**Goal**: Run exact number of rounds

```bash
# No time limit - runs until 20 rounds complete
./submit_job.sh fp32 20 0

# Or just use rounds
./submit_job.sh fp32 20
```

---

## 📈 What You'll See

### During Training

```
⏱️  Time limit: 10 minutes (600 seconds)
⏱️  Training will stop after 10 min OR 20 rounds (whichever comes first)

Round 1 | Loss: 1.2345 | Accuracy: 0.7500 | Elapsed: 1.23 min | Remaining: 8.77 min
Round 2 | Loss: 0.9876 | Accuracy: 0.8200 | Elapsed: 2.45 min | Remaining: 7.55 min
Round 3 | Loss: 0.8765 | Accuracy: 0.8450 | Elapsed: 3.67 min | Remaining: 6.33 min
...
Round 8 | Loss: 0.5432 | Accuracy: 0.8800 | Elapsed: 9.87 min | Remaining: 0.13 min

⏱️  TIME LIMIT REACHED!
   Elapsed: 10.02 min / Limit: 10.0 min
   Stopping training after round 8

Training completed!
Total time: 10.02 minutes (601.2 seconds)
⏱️  Stopped due to time limit (10 minutes)
```

### Without Time Limit

```
🔄 No time limit - will run for 20 rounds

Round 1 | Loss: 1.2345 | Accuracy: 0.7500 | Elapsed: 1.23 min
Round 2 | Loss: 0.9876 | Accuracy: 0.8200 | Elapsed: 2.45 min
...
Round 20 | Loss: 0.4321 | Accuracy: 0.8900 | Elapsed: 24.50 min

Training completed!
Total time: 24.50 minutes (1470.0 seconds)
```

---

## 🔧 Configuration Matrix

| Command | Max Rounds | Time Limit | Behavior |
|---------|------------|------------|----------|
| `./submit_job.sh fp32` | From config | From config | Uses both limits from file |
| `./submit_job.sh fp32 20` | 20 | From config | Override rounds only |
| `./submit_job.sh fp32 20 10` | 20 | 10 min | Override both |
| `./submit_job.sh fp32 20 0` | 20 | None | Only rounds limit |
| `./submit_time_comparison.sh 10` | 100 | 10 min | All 3 precisions |

---

## 💡 Pro Tips

### 1. Set High Rounds for Time-Limited Runs
```bash
# Guarantees training stops at time limit, not rounds
./submit_job.sh fp32 100 10    # Will stop at 10 min (probably ~8-12 rounds)
```

### 2. Estimate Time Per Round
First run without time limit to measure:
```bash
./submit_job.sh fp32 5
# Check logs: "Round 5 | ... | Elapsed: 6.23 min"
# = ~1.25 min/round for FP32
```

Then set time limit accordingly:
```bash
# Want 20 rounds? Need ~25 minutes
./submit_job.sh fp32 20 30    # 30 min safety buffer
```

### 3. Compare Efficiency
```bash
# Same time limit for all precisions
./submit_time_comparison.sh 15

# Results show efficiency:
# FP32: 10 rounds in 15 min = 1.5 min/round
# FP16: 12 rounds in 15 min = 1.25 min/round (faster!)
# INT8: 15 rounds in 15 min = 1.0 min/round (fastest!)
```

### 4. Quick Testing
```bash
# Test code changes quickly
./submit_job.sh fp32 100 2    # Just 2 minutes to verify it works
```

---

## 🎯 Recommended Workflows

### Workflow 1: Initial Testing
```bash
# Quick 3-minute test
./submit_job.sh fp32 10 3
```

### Workflow 2: Fair Comparison
```bash
# Compare all precisions with 10-minute limit
./submit_time_comparison.sh 10
```

### Workflow 3: Full Training
```bash
# No time limit, run full rounds
./submit_job.sh fp32 20 0
./submit_job.sh fp16 20 0
./submit_job.sh int8 20 0
```

### Workflow 4: Hybrid Approach
```bash
# Run rounds but with safety timeout
./submit_job.sh fp32 20 30    # Stop at 20 rounds or 30 min
```

---

## 📊 Expected Timing

Based on typical GPU performance:

| Precision | Time/Round | 10 Rounds | 20 Rounds |
|-----------|-----------|-----------|-----------|
| FP32      | ~1.5 min  | ~15 min   | ~30 min   |
| FP16      | ~1.2 min  | ~12 min   | ~24 min   |
| INT8      | ~1.0 min  | ~10 min   | ~20 min   |

*Times vary based on GPU, dataset size, and network*

---

## 🚨 Important Notes

1. **Time limit is NOT a hard deadline**
   - Checks happen between rounds (not mid-round)
   - May exceed limit by ~1 round duration
   - Example: 10 min limit might finish at 11.5 min

2. **Time includes everything**
   - Data loading
   - Training
   - Evaluation
   - Communication

3. **Logs show remaining time**
   - Updated each round
   - Helps predict completion

4. **Job name includes time limit**
   - `fl-fp32-r20-t10m.slurm` = FP32, 20 rounds, 10 min limit
   - `fl-fp16-r10.slurm` = FP16, 10 rounds, no time limit

---

## ✅ Quick Reference

```bash
# Config-based (edit before upload)
nano pyproject_fp32.toml     # Set: time-limit-minutes = 10
./submit_job.sh fp32

# Command-line override
./submit_job.sh fp32 20 10   # 20 rounds OR 10 min

# Compare all 3 precisions
./submit_time_comparison.sh 10

# No time limit
./submit_job.sh fp32 20 0
```

---

**Happy training!** ⏱️🚀
