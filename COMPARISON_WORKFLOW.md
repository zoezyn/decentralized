# 📊 Quantization Comparison Workflow

## Goal

Compare three approaches:
1. **Training comparison**: How far does each precision (FP32, FP16, INT8) get in 10 minutes?
2. **Post-Training Quantization (PTQ)**: Real compression of FP32 model to FP16/INT8
3. **Visualization**: Plots showing accuracy, size, compression, and cost

---

## Quick Start

### Option 1: Automated Workflow (Recommended)

Run everything automatically:

```bash
./run_comparison.sh
```

This will:
- Submit 3 jobs (FP32, FP16, INT8) with 10-minute time limits
- Wait for completion
- Run PTQ comparison on best FP32 model
- Generate comparison plots

### Option 2: Manual Step-by-Step

**Step 1: Run time-limited training**

```bash
./submit_job.sh fp32 100 10  # Train FP32 for max 10 minutes
./submit_job.sh fp16 100 10  # Train FP16 for max 10 minutes
./submit_job.sh int8 100 10  # Train INT8 for max 10 minutes
```

**Step 2: Wait and monitor**

```bash
squeue -u $USER                         # Check job status
tail -f logs/fl-fp32-r100-t10m-*.out   # Watch FP32 progress
```

**Step 3: Compare quantized versions**

```bash
# Find your FP32 model
ls -lh outputs/fp32/*/final_model.pt

# Run PTQ comparison
python compare_quantizations.py outputs/fp32/2025-11-15/14-30-00/final_model.pt
```

**Step 4: Generate plots**

```bash
python create_comparison_plots.py
```

---

## What Gets Compared?

### 1. Training Performance (10-minute limit)

Each precision trains for exactly 10 minutes:
- **FP32**: Baseline, highest accuracy, slowest
- **FP16**: Half precision, faster training
- **INT8**: 8-bit quantized, fastest training

**Question answered**: *Which precision trains fastest?*

### 2. Post-Training Quantization (PTQ)

Take the FP32 model and quantize it:
- **FP16**: Convert weights to half precision
- **INT8**: Dynamic quantization to 8-bit integers

**Question answered**: *How much can we compress without losing accuracy?*

### 3. Results

You get:

**JSON file** (`quantization_comparison.json`):
```json
{
  "precisions": {
    "fp32": {
      "accuracy": 0.8520,
      "file_size_mb": 1.66,
      "compression_ratio": 1.0
    },
    "fp16": {
      "accuracy": 0.8518,
      "file_size_mb": 0.83,
      "compression_ratio": 2.00,
      "accuracy_drop_percent": -0.02
    },
    "int8": {
      "accuracy": 0.8495,
      "file_size_mb": 0.52,
      "compression_ratio": 3.19,
      "accuracy_drop_percent": -0.25
    }
  }
}
```

**Plots** (`comparison_plots.png`):
- Accuracy vs Model Size scatter
- Model Size bars
- Compression Ratio bars
- Accuracy Drop bars
- Communication Cost comparison
- Summary table

---

## Understanding the Results

### Key Metrics

**Accuracy**: How well the model classifies satellite images
- Higher is better
- FP32 is baseline (100%)
- Check how much FP16/INT8 lose

**Model Size**: Actual file size on disk (MB)
- Smaller is better for transmission
- INT8 should be ~3x smaller than FP32

**Compression Ratio**: How much smaller vs FP32
- INT8 target: ~3x compression
- FP16 target: ~2x compression

**Accuracy Drop**: Percentage points lost vs FP32
- Smaller is better
- Acceptable: <0.5% for INT8, <0.1% for FP16

**Communication Cost**: Money spent transmitting models
- Calculated: (model_size × 2 × num_clients × num_rounds × $/MB)
- 10 clients, 10 rounds, $5/MB
- INT8 should save ~70% vs FP32

### Cost Calculation Example

For 10 clients, 10 rounds, $5/MB:

```
FP32: 1.66 MB × 2 (down+up) × 10 clients × 10 rounds × $5/MB = $1,660
INT8: 0.52 MB × 2 (down+up) × 10 clients × 10 rounds × $5/MB = $520

Savings: $1,140 (69% reduction)
```

---

## Interpreting Trade-offs

### Good Quantization Result
✅ INT8 loses <0.5% accuracy
✅ 3x+ compression ratio
✅ 65%+ cost savings

### Acceptable Result
⚠️ INT8 loses 0.5-1.0% accuracy
⚠️ 2.5-3x compression
⚠️ 60-65% cost savings

### Poor Result (investigate!)
❌ INT8 loses >1.0% accuracy
❌ <2.5x compression
❌ <60% cost savings

---

## Output Directory Structure

```
outputs/
├── fp32/
│   └── 2025-11-15/
│       └── 14-30-00/
│           ├── final_model.pt              # FP32 trained model
│           ├── model_fp32.pt               # PTQ baseline
│           ├── model_fp16.pt               # PTQ FP16
│           ├── model_int8.pt               # PTQ INT8
│           ├── quantization_comparison.json
│           ├── comparison_plots.png
│           └── comparison_plots_highres.png
├── fp16/
│   └── 2025-11-15/
│       └── 14-35-00/
│           └── final_model.pt              # FP16 trained model
└── int8/
    └── 2025-11-15/
        └── 14-40-00/
            └── final_model.pt              # INT8 trained model
```

---

## Tips

1. **10-minute limit is strict**: Jobs will stop exactly at 10 minutes
2. **PTQ uses FP32 model**: Always quantize from the best FP32 model
3. **Plots are publication-ready**: 300 DPI PNG, high-res version at 600 DPI
4. **Multiple runs**: Can repeat with different time limits (5min, 15min, etc.)

---

## Troubleshooting

**No plots generated?**
```bash
# Check if matplotlib is available
python -c "import matplotlib; print('OK')"

# If missing, it should be in hackathon-venv
source ../hackathon-venv/bin/activate
```

**Can't find FP32 model?**
```bash
# List all models
find outputs -name "final_model.pt"

# Use the latest FP32 one
python compare_quantizations.py outputs/fp32/2025-11-15/*/final_model.pt
```

**Jobs not starting?**
```bash
# Check queue
squeue -u $USER

# Check logs for errors
ls -lh logs/
cat logs/fl-fp32-*.err
```

---

Ready to compare! 🚀
