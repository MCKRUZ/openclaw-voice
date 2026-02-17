# RTX 5090 Compatibility Blocker

**Date:** February 16, 2026
**GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM, Blackwell sm_120)
**Status:** ❌ **BLOCKED - No PyTorch Support**

---

## Critical Finding

The **RTX 5090 is too new** for current PyTorch builds. Both stable and nightly releases fail with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device

NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**Tested Versions:**
- ❌ PyTorch 2.6.0+cu124 (Stable) - No sm_120 support
- ❌ PyTorch 2.7.0.dev20250310+cu124 (Nightly) - No sm_120 support

---

## Impact on Your Voice Bot

### Currently Affected

All GPU-accelerated components are **non-functional**:

| Component | Current Status | Impact |
|-----------|---------------|--------|
| **faster-whisper STT** | CPU-only | 3-5x slower (550ms → ~2s) |
| **Coqui XTTS v2 TTS** | CPU-only | 2-3x slower (1.6s → ~4-5s) |
| **Kani-TTS-2 testing** | Blocked | Cannot evaluate |
| **Total latency** | ~10-15s | vs target 3.5s ❌ |

### What Still Works

- ✅ Discord bot (voice receiving/sending)
- ✅ OpenClaw Gateway (LLM inference)
- ✅ VAD (Silero, CPU-based)
- ✅ Smart Turn v3 (ONNX, CPU-based)
- ⚠️ STT/TTS (fallback to CPU, very slow)

---

## Solutions

### Option 1: Wait for PyTorch Support (Recommended)

**Timeline:** 1-3 months (estimated)

**Reason:** RTX 5090 released Jan 2025, PyTorch typically adds new GPU support within 2-4 months.

**Monitor:**
- [PyTorch Releases](https://github.com/pytorch/pytorch/releases)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)

**Action:**
- Check weekly for PyTorch updates
- Subscribe to PyTorch announcements
- Test with: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124`

### Option 2: Build PyTorch from Source (Advanced)

**Difficulty:** High
**Time:** 4-8 hours
**Risk:** May not work if CUDA Toolkit doesn't support sm_120

**Steps:**
1. Install CUDA Toolkit 12.8+ (if available with sm_120 support)
2. Clone PyTorch:
   ```bash
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   ```
3. Build with sm_120:
   ```bash
   export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
   python setup.py install
   ```
4. Test

**Resources:**
- [Building PyTorch from Source](https://github.com/pytorch/pytorch#from-source)

### Option 3: Use Different GPU

**If available**, use older GPU for development:

| GPU | CUDA Capability | PyTorch Support | Recommendation |
|-----|-----------------|-----------------|----------------|
| RTX 4090 | sm_89 | ✅ Full support | ✅ Ideal for development |
| RTX 4080 | sm_89 | ✅ Full support | ✅ Good alternative |
| RTX 4070 Ti | sm_89 | ✅ Full support | ✅ Sufficient for voice bot |
| RTX 3090 | sm_86 | ✅ Full support | ✅ Works well |

**Action:**
- Check if you have access to RTX 40-series or 30-series GPU
- Use for development until RTX 5090 support lands

### Option 4: Run in Cloud with Supported GPU

**Platforms:**
- **RunPod** - RTX 4090 @ $0.79/hr
- **Vast.ai** - RTX 4090 @ $0.40-0.60/hr
- **Google Colab Pro** - A100/V100 @ $10/month

**Pros:**
- Immediate GPU access
- Supported hardware
- Test optimizations quickly

**Cons:**
- Ongoing cost
- Need to upload code/data
- Network latency for Discord bot

### Option 5: CPU-Only (Temporary Workaround)

**Use case:** Testing logic while waiting for GPU support

**Current setup** (already done):
```bash
pip install torch torchvision torchaudio  # CPU version
```

**Performance:**
- STT: ~2-3s (vs 0.3s target)
- TTS: ~4-5s (vs 0.9s target)
- Total: ~10-15s (vs 3.5s target)

**Acceptable for:**
- Testing conversation flow
- Debugging bot logic
- Development (not production)

---

## Recommended Action Plan

### Immediate (This Week)

1. ✅ **Rollback to CPU PyTorch** for development:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. ✅ **Focus on non-GPU optimizations**:
   - Query routing (Haiku vs Sonnet vs Opus)
   - TTS caching
   - Sentence-level streaming
   - Response filtering

3. ✅ **Test bot functionality** with CPU (slow but works)

### Short-term (Next 2-4 Weeks)

4. 🔄 **Monitor PyTorch releases** for sm_120 support

5. 🧪 **Evaluate cloud GPU** options:
   - Test on RunPod/Vast.ai with RTX 4090
   - Measure actual performance gains
   - Compare cost vs waiting

6. 📊 **Benchmark CPU baseline** to quantify GPU improvement later

### Long-term (Next 1-3 Months)

7. ⏳ **Wait for PyTorch sm_120 support**

8. 🚀 **Deploy with GPU** when support lands

9. 🔍 **Re-evaluate Kani-TTS-2** once GPU works

---

## Current Bot Configuration

**For now, use CPU-only mode:**

```yaml
# config.yaml
pipeline:
  stt:
    model_size: "small"  # Smaller = faster on CPU
    device: "cpu"        # Force CPU
    beam_size: 1         # Faster decoding

  tts:
    device: "cpu"        # Force CPU
```

**.env overrides:**
```bash
PIPELINE__STT__DEVICE=cpu
PIPELINE__STT__MODEL_SIZE=small
PIPELINE__TTS__DEVICE=cpu
```

---

## When PyTorch Supports sm_120

**Test with:**
```bash
# Uninstall current
pip uninstall torch torchaudio torchvision -y

# Install latest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Test computation
python -c "import torch; x=torch.rand(100,100,device='cuda'); print('GPU OK')"
```

**Then update config:**
```yaml
pipeline:
  stt:
    device: "cuda"
    model_size: "medium"  # Can use larger model on GPU
    beam_size: 5          # Better quality

  tts:
    device: "cuda"
```

**Expected improvement:**
- STT: ~2s → ~0.35s (6x faster)
- TTS: ~4-5s → ~0.9s (5x faster)
- Total: ~10-15s → ~4s (3x faster, near 3.5s target!)

---

## Resources

- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [RTX 5090 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- [RunPod Cloud GPU](https://www.runpod.io/)
- [Vast.ai GPU Marketplace](https://vast.ai/)

---

**Summary:** RTX 5090 support is coming, but not here yet. Use CPU mode for development now, monitor for PyTorch updates, or use cloud GPU for testing in the meantime.
