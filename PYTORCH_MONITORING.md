# PyTorch RTX 5090 Support Monitoring Guide

**Goal:** Get notified when PyTorch adds Blackwell (sm_120) support

---

## Quick Check (Weekly)

### Test PyTorch Nightly

```bash
# From project root
cd "C:\Users\kruz7\OneDrive\Documents\Code Repos\MCKRUZ\openclaw-voice"
source venv/Scripts/activate

# Test nightly build
pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Quick test
python -c "import torch; x=torch.rand(10,10,device='cuda'); print('✓ RTX 5090 WORKS!' if x.device.type=='cuda' else '✗ Not yet')"
```

If you see **"✓ RTX 5090 WORKS!"** → GPU support is here! Run `fix_pytorch_cuda.bat`

---

## Automated Monitoring

### Option 1: GitHub Watch (Recommended)

1. Go to: https://github.com/pytorch/pytorch
2. Click **"Watch"** → **"Custom"**
3. Check **"Releases"** only
4. Get email when new PyTorch releases

### Option 2: RSS Feed

Subscribe to PyTorch releases:
```
https://github.com/pytorch/pytorch/releases.atom
```

Use RSS reader (Feedly, Inoreader) or browser extension

### Option 3: Weekly Calendar Reminder

Set recurring calendar event:
- **When:** Every Monday 9am
- **What:** Check PyTorch RTX 5090 support
- **How:** Run quick test above

---

## What to Look For

### In Release Notes

Keywords indicating sm_120 support:
- ✅ "Blackwell" or "sm_120"
- ✅ "RTX 5090" or "50-series"
- ✅ "CUDA capability 12.0"
- ✅ "Hopper+Blackwell" or "H100+B100"

### Example Release Note:
```
PyTorch 2.X.0 Release Notes
- Added support for NVIDIA Blackwell architecture (sm_120)
- RTX 50-series GPUs now fully supported
```

---

## When Support Lands

### 1. Update PyTorch

```bash
cd "C:\Users\kruz7\OneDrive\Documents\Code Repos\MCKRUZ\openclaw-voice"

# Run the fix script
fix_pytorch_cuda.bat

# Or manually:
source venv/Scripts/activate
pip uninstall torch torchaudio torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Verify GPU Works

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); x=torch.rand(100,100,device='cuda'); print('GPU OK!')"
```

Expected output:
```
NVIDIA GeForce RTX 5090
GPU OK!
```

### 3. Update Config

Edit `config.yaml`:
```yaml
pipeline:
  stt:
    device: "cuda"        # Was: cpu
    model_size: "medium"  # Can increase from small
    beam_size: 5          # Can increase from 1

  tts:
    device: "cuda"        # Was: cpu
```

### 4. Test Performance

```bash
# Start the bot
python run.py

# In Discord
/join
# Say: "Hey Jarvis, test response time"
/status  # Check latency stats

# Expected improvement:
# - STT: ~2s → ~0.35s (6x faster)
# - TTS: ~4s → ~0.9s (4x faster)
# - Total: ~10s → ~4s (near 3.5s target!)
```

### 5. Re-test Kani-TTS-2

```bash
python test_kani_tts.py

# If successful:
# - Compare quality with current Coqui XTTS v2
# - Check if RTF ~0.2 achieved
# - Decide if worth integrating
```

---

## Estimated Timeline

Based on historical GPU support addition:

| GPU Architecture | Release Date | PyTorch Support Added | Time Gap |
|------------------|--------------|----------------------|----------|
| Ampere (RTX 30) | Sep 2020 | Nov 2020 | 2 months |
| Ada Lovelace (RTX 40) | Oct 2022 | Dec 2022 | 2 months |
| Hopper (H100) | Mar 2023 | May 2023 | 2 months |
| **Blackwell (RTX 50)** | **Jan 2025** | **Est: Mar-Apr 2025** | **2-3 months** |

**Conservative estimate:** March 2025 (1 month from now)
**Optimistic estimate:** Late February 2025 (2 weeks)
**Pessimistic estimate:** May 2025 (3 months)

---

## While You Wait

### Optimize Non-GPU Components

Focus on improvements that work on CPU:

1. **Query Routing** (already implemented)
   - Haiku for simple queries
   - Sonnet for medium
   - Opus for complex

2. **TTS Caching** (already implemented)
   - Pre-generate common phrases
   - Cache by hash

3. **Response Filtering**
   - Improve relevance detection
   - Reduce unnecessary responses

4. **Streaming Optimization**
   - Sentence-level playback
   - Parallel processing where possible

### Test Bot Logic

Even with slow performance, you can:
- Test conversation flow
- Debug agent personalities
- Refine prompt engineering
- Test Discord commands
- Verify OpenClaw integration

### Prepare for GPU

- Read KANI_TTS_EVALUATION.md
- Plan integration strategy
- Review current TTS implementation
- Identify optimization opportunities

---

## Contact/Support

**PyTorch Issues:** https://github.com/pytorch/pytorch/issues
**PyTorch Forums:** https://discuss.pytorch.org/
**NVIDIA Developer:** https://forums.developer.nvidia.com/

**Search for:** "RTX 5090 support" or "sm_120" or "Blackwell"

---

## Summary

✅ **Weekly:** Run quick test or check GitHub releases
✅ **When ready:** Run `fix_pytorch_cuda.bat`
✅ **Then:** Update config, test performance, evaluate Kani-TTS-2
✅ **Expected:** March 2025 (1-2 months)

**Bookmark this file** and check weekly until GPU support lands!
