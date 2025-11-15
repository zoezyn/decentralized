# 🚀 Cluster Deployment - Ultra Quick Start

## 📝 Summary
Train 3 Federated Learning models (FP32, FP16, INT8) on cluster with **zero installation** - uses pre-configured `hackathon-venv`.

---

## 🎯 Super Fast Workflow

### 1️⃣ Set Number of Rounds (Before Upload)

Edit the config files to set how many rounds you want:

```bash
# Edit these files locally BEFORE uploading:
nano pyproject_fp32.toml   # Change: num-server-rounds = 5
nano pyproject_fp16.toml    # Change: num-server-rounds = 5
nano pyproject_int8.toml    # Change: num-server-rounds = 5
```

### 2️⃣ Upload to Cluster (From Local Machine)

```bash
./upload_to_cluster.sh <your-username>@<cluster-address>
```

**Example:**
```bash
./upload_to_cluster.sh marty@gpu-cluster.hackathon.ai
```

### 3️⃣ SSH Into Cluster

```bash
ssh <your-username>@<cluster-address>
cd ~/berlin25-eurosat
```

### 4️⃣ Update Venv Path (First Time Only)

Edit `submit_job.sh` and replace `/path/to/hackathon-venv` with actual path:

```bash
nano submit_job.sh
# Change line 109 to actual path:
# source /actual/path/to/hackathon-venv/bin/activate
```

### 5️⃣ Submit Jobs

```bash
# Submit FP32 job
./submit_job.sh fp32

# Submit FP16 job
./submit_job.sh fp16

# Submit INT8 job
./submit_job.sh int8
```

**Or override rounds on the fly:**
```bash
./submit_job.sh fp32 10    # Run FP32 with 10 rounds instead of config value
```

---

## 📊 Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f logs/fl-fp32-r5-<job-id>.out

# Check all logs
ls -lh logs/
```

---

## 📁 Get Results

```bash
# List outputs
ls -lh outputs/

# Download results to local machine (from local terminal)
scp -r <username>@<cluster>:~/berlin25-eurosat/outputs/ ./outputs/
```

---

## 🔧 Config Files Explained

### `pyproject_fp32.toml` - 32-bit Baseline
- **Best accuracy**, largest model size
- Standard floating point training

### `pyproject_fp16.toml` - 16-bit Half Precision
- **~50% smaller** than FP32
- Minimal accuracy loss (~0.1%)

### `pyproject_int8.toml` - 8-bit Quantized
- **~75% smaller** than FP32
- Small accuracy loss (~0.3%)
- **Best trade-off for satellite communication**

---

## ⚙️ Quick Config Changes

Want to change parameters? Edit the config files:

```toml
[tool.flwr.app.config]
num-server-rounds = 5      # ← Number of FL rounds (3, 5, 10, 20)
fraction-train = 1.0       # ← Fraction of clients (0.5 = 50%, 1.0 = 100%)
local-epochs = 2           # ← Local training epochs per round
lr = 0.001                 # ← Learning rate
```

After editing, re-upload and submit!

---

## 🎯 Complete Example

```bash
# LOCAL MACHINE
# 1. Edit rounds
nano pyproject_fp32.toml  # Set: num-server-rounds = 10

# 2. Upload
./upload_to_cluster.sh user@cluster.ai

# CLUSTER
# 3. SSH in
ssh user@cluster.ai
cd ~/berlin25-eurosat

# 4. Submit all 3 jobs
./submit_job.sh fp32
./submit_job.sh fp16
./submit_job.sh int8

# 5. Monitor
squeue -u $USER
tail -f logs/fl-fp32-r10-*.out

# 6. Check results
ls -lh outputs/
```

---

## 🚨 Troubleshooting

### Job stuck in queue?
```bash
squeue              # Check cluster load
sinfo               # Check node availability
```

### Job failed?
```bash
# Check error log
cat logs/fl-fp32-r5-<job-id>.err

# Check output log
cat logs/fl-fp32-r5-<job-id>.out
```

### Need to cancel a job?
```bash
scancel <job-id>
```

---

## 📈 Expected Results

| Precision | Model Size | Accuracy Drop | Training Time (5 rounds) |
|-----------|------------|---------------|-------------------------|
| FP32      | ~12 MB     | 0% (baseline) | ~10-15 min              |
| FP16      | ~6 MB      | ~0.1%         | ~8-12 min               |
| INT8      | ~3 MB      | ~0.3%         | ~6-10 min               |

---

## ✅ Checklist

- [ ] Edited config files with desired rounds
- [ ] Uploaded to cluster using `upload_to_cluster.sh`
- [ ] Updated venv path in `submit_job.sh`
- [ ] Created `logs/` directory: `mkdir -p logs`
- [ ] Submitted 3 jobs (fp32, fp16, int8)
- [ ] Monitored with `squeue -u $USER`
- [ ] Downloaded results from `outputs/`

---

## 💡 Pro Tips

1. **Test locally first**: Run `flwr run . local-simulation` before uploading
2. **Start small**: Use 3 rounds first, then scale to 10-20
3. **Monitor GPU usage**: Check if jobs are actually using GPU
4. **Save bandwidth**: Only upload changed files (rsync does this automatically)
5. **Parallel jobs**: Submit all 3 precision jobs at once - they'll queue

---

## 📞 Need Help?

1. Check logs first: `tail -f logs/*.err`
2. Ask in #support channel
3. Share job ID and error message

Good luck! 🚀
