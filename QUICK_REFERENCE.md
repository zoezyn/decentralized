# ⚡ Quick Reference Card

## 🚀 Fast Commands

```bash
# 1. Set parameters (pick one)
./set_rounds.sh 10                          # Just rounds
nano pyproject_fp32.toml                    # Rounds + time limit

# 2. Upload
./upload_to_cluster.sh user@cluster.ai

# 3. Submit (pick one)
./submit_job.sh fp32                        # Config settings
./submit_job.sh fp32 20                     # 20 rounds
./submit_job.sh fp32 20 10                  # 20 rounds OR 10 min
./submit_time_comparison.sh 10              # All 3 jobs, 10 min each

# 4. Monitor
squeue -u $USER                             # Job status
tail -f logs/fl-*.out                       # Live output
```

---

## 📋 Command Matrix

| Command | Rounds | Time | Usage |
|---------|--------|------|-------|
| `./submit_job.sh fp32` | Config | Config | Use file settings |
| `./submit_job.sh fp32 20` | 20 | Config | Override rounds |
| `./submit_job.sh fp32 20 10` | 20 | 10 min | Override both |
| `./submit_job.sh fp32 20 0` | 20 | None | Only rounds |
| `./submit_time_comparison.sh 10` | 100 | 10 min | All 3 precisions |

---

## ⚙️ Config Parameters

```toml
[tool.flwr.app.config]
num-server-rounds = 20       # Max rounds
time-limit-minutes = 10      # Time limit (0 = no limit)
fraction-train = 1.0         # Fraction of clients (1.0 = all 10)
local-epochs = 2             # Epochs per round
lr = 0.001                   # Learning rate
```

---

## 📊 Monitoring

```bash
# Check queue
squeue -u $USER

# Watch logs
tail -f logs/fl-fp32-r20-*.out

# Check outputs
ls -lh outputs/

# Cancel job
scancel <job-id>
```

---

## 🎯 Common Scenarios

### Quick Test (2 minutes)
```bash
./submit_job.sh fp32 100 2
```

### Fair Comparison
```bash
./submit_time_comparison.sh 10
```

### Full Training
```bash
./submit_job.sh fp32 20 0
./submit_job.sh fp16 20 0
./submit_job.sh int8 20 0
```

### Deadline Training
```bash
./submit_job.sh fp32 50 30    # 50 rounds max, 30 min deadline
```

---

## 📁 File Structure

```
berlin25-eurosat/
├── eurosat/              # Source code
├── pyproject_*.toml      # Configs (fp32, fp16, int8)
├── upload_to_cluster.sh  # Upload script
├── submit_job.sh         # Job submission
├── submit_time_comparison.sh  # Time comparison
├── set_rounds.sh         # Quick config tool
├── validate_setup.sh     # Validation
└── logs/                 # Job outputs
```

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| `START_HERE.md` | Quick start |
| `TIME_LIMIT_GUIDE.md` | ⏱️ Time limits |
| `CLUSTER_QUICK_START.md` | Detailed guide |
| `DEPLOYMENT_SUMMARY.md` | Technical details |

---

## 🔧 Troubleshooting

```bash
# Validate setup
./validate_setup.sh

# Check job error
cat logs/fl-*.err

# Check job output
cat logs/fl-*.out

# Check cluster
sinfo
squeue
```

---

## ⏱️ Time Limit Quick Guide

```bash
# No limit - run 20 rounds
./submit_job.sh fp32 20 0

# 10 min limit - run up to 20 rounds
./submit_job.sh fp32 20 10

# Compare all 3 in 10 minutes
./submit_time_comparison.sh 10
```

**Behavior**: Stops at **time limit OR max rounds** (whichever first)

---

## ✅ Pre-Flight Checklist

- [ ] Set rounds: `./set_rounds.sh 10`
- [ ] Uploaded: `./upload_to_cluster.sh`
- [ ] Updated venv path in `submit_job.sh`
- [ ] Created logs: `mkdir -p logs`
- [ ] Validated: `./validate_setup.sh`

**Ready to go!** 🚀
