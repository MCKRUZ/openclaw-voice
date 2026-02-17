# Kani-TTS-2 Evaluation Report

**Date:** February 16, 2026
**System:** Windows 11, RTX 5090 (32GB VRAM)

---

## Summary

**Status:** ❌ **Cannot test Kani-TTS-2 on Windows** (compilation issues)

Attempted installation of Kani-TTS-2 encountered critical dependency compilation errors on Windows. Additionally, current environment has PyTorch CPU-only installation despite having RTX 5090.

---

## Issues Discovered

### 1. PyTorch CPU-Only Installation

**Current Status:**
```
PyTorch: 2.10.0+cpu
CUDA available: False
CUDA version: N/A
```

**Impact:**
- Current TTS (Coqui XTTS v2) may not be using GPU acceleration
- Kani-TTS-2 requires CUDA-enabled PyTorch
- STT (faster-whisper) may not be using GPU acceleration

**Required:** PyTorch with CUDA 12.x support

### 2. Kani-TTS-2 Installation Failure

**Error:**
```
Failed building wheel for pynini
error: command 'cl.exe' failed with exit code 2
```

**Root Cause:**
- `nemo-toolkit` dependency requires `pynini`
- `pynini` compilation uses GCC/Clang flags (`-Wno-register`) incompatible with MSVC compiler
- No pre-built Windows wheels available for `pynini==2.1.6.post1`

**Dependency Chain:**
```
kani-tts-2 → nemo-toolkit[tts]==2.4.0 → pynini → [COMPILATION FAILED]
```

---

## Kani-TTS-2 Pros & Cons (Based on Documentation)

### Potential Benefits

✅ **3-4x faster generation** - RTF of 0.2 vs current 0.78
✅ **Zero-shot voice cloning** - No fine-tuning needed
✅ **Lower VRAM usage** - 3GB vs current 2-3GB (similar)
✅ **Simple API** - Clean Python interface
✅ **Commercial license** - Apache 2.0
✅ **Fast training** - 10k hours in 6 hours on 8x H100

### Challenges

❌ **Windows compatibility** - Compilation issues with dependencies
❌ **Requires nemo-toolkit** - Heavy dependency with C++ compilation
❌ **English-only** - Current version limited to English
❓ **Quality unknown** - Cannot test without successful installation
❓ **Streaming support** - Not documented, unclear if supported

---

## Alternative Solutions

### Option 1: Fix PyTorch CUDA Installation (Recommended)

**Goal:** Get current system using GPU properly + enable future testing

**Steps:**
1. Uninstall CPU PyTorch:
   ```bash
   pip uninstall torch torchaudio torchvision
   ```

2. Install CUDA PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show RTX 5090
   ```

**Impact:**
- Current Coqui XTTS v2 will use GPU (faster)
- faster-whisper STT will use GPU (faster)
- Enables future Kani-TTS-2 testing

### Option 2: Use WSL2 or Docker (Linux Environment)

**Goal:** Run Kani-TTS-2 in Linux where dependencies compile properly

**Setup WSL2:**
```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-24.04

# Install CUDA in WSL
# Follow: https://docs.nvidia.com/cuda/wsl-user-guide/

# Clone repo and test in WSL
cd /mnt/c/Users/kruz7/...
python test_kani_tts.py
```

**Pros:**
- Native Linux environment, better compatibility
- Access to Windows GPU via WSL-CUDA
- Can test Kani-TTS-2 properly

**Cons:**
- Additional setup complexity
- Need to manage two environments

### Option 3: Wait for Windows Support

**Goal:** Wait for Kani-TTS-2 to release Windows pre-built wheels

**Timeline:**
- Kani-TTS-2 is very new (Feb 2025)
- Windows wheels may be released in future versions
- Monitor: https://pypi.org/project/kani-tts-2/

**Meanwhile:**
- Stick with current Coqui XTTS v2
- Focus on other optimizations (query routing, caching, streaming)

### Option 4: Alternative TTS Engines

Consider other fast TTS options with better Windows support:

**A. Piper TTS**
- Very fast (RTF ~0.1)
- Lightweight, runs on CPU
- Pre-built Windows binaries
- Good quality
- Con: Limited voice cloning

**B. Bark**
- High quality
- Good voice cloning
- Con: Slower than current setup

**C. StyleTTS2**
- Excellent quality
- Zero-shot voice cloning
- Con: Slower, complex setup

---

## Recommendation

### Immediate Action: Fix PyTorch CUDA

**Priority: HIGH** - This affects current system performance

```bash
# From project root with venv activated
pip uninstall torch torchaudio torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected Improvement:**
- Current TTS latency: 1.63s → ~0.8-1.0s (using GPU)
- STT latency: 0.55s → ~0.3-0.4s (faster on GPU)
- Total: ~5.5s → ~4.0s (closer to 3.5s target)

### Kani-TTS-2 Strategy

**Short-term (Next Week):**
- Focus on optimizing current Coqui XTTS v2 with GPU
- Implement additional TTS caching
- Optimize streaming chunk size

**Medium-term (Next Month):**
- Monitor Kani-TTS-2 for Windows wheel releases
- Test in WSL2 if critical for evaluation
- Evaluate Piper TTS as alternative

**Long-term (Next Quarter):**
- Revisit Kani-TTS-2 when Windows support matures
- Consider migration to Linux host if TTS performance critical

---

## Current Performance Baseline

Based on README.md:

| Stage | Current | Target | Status |
|-------|---------|--------|--------|
| VAD silence detection | 800ms | 800ms | ✅ |
| STT (medium) | 550ms | 300ms | ⚠️ (CPU-only) |
| OpenClaw/LLM | 2470ms | 2000ms | ✅ |
| TTS first chunk | 1630ms | 300ms | ❌ (CPU-only?) |
| **Total** | **~5.5s** | **~3.5s** | ⚠️ |

**With GPU PyTorch (estimated):**

| Stage | With CUDA | Improvement |
|-------|-----------|-------------|
| STT | ~350ms | 1.6x faster |
| TTS | ~900ms | 1.8x faster |
| **Total** | **~4.0s** | **1.4x faster** |

Still short of 3.5s target, but closer. Kani-TTS-2 could bridge the gap if Windows support improves.

---

## Next Steps

1. ✅ **Fix PyTorch CUDA** (see Option 1 above)
2. 🔄 **Re-benchmark current system** with GPU acceleration
3. 📊 **Measure actual improvement** in TTS latency
4. 🔍 **Evaluate if 4.0s total latency** is acceptable
5. 🕐 **Monitor Kani-TTS-2** for Windows support
6. 🧪 **Test Piper TTS** as lightweight alternative

---

## References

- [Kani-TTS-2 GitHub](https://github.com/nineninesix-ai/kani-tts-2)
- [Kani-TTS-2 HuggingFace](https://huggingface.co/nineninesix/kani-tts-2-en)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [WSL CUDA Setup](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Piper TTS](https://github.com/rhasspy/piper)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2)

---

**Conclusion:** Kani-TTS-2 shows promise (3-4x faster) but Windows compatibility issues prevent testing. **Immediate priority should be fixing PyTorch CUDA** to improve current system performance, then revisit Kani-TTS-2 when Windows support improves or via WSL2.
