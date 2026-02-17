# Discord Voice Bot - Optimization Testing Guide

**Goal:** Verify the 3-10x latency improvements from Phase 1 optimizations

---

## Pre-Flight Checklist

### ✅ Requirements

1. **Discord Bot Token** - Set in `.env` file
2. **OpenClaw Gateway** - Running at `http://192.168.50.9:18789` (or update `.env`)
3. **Voice Files** - `server/voices/jarvis.wav` (or `.mp3`)
4. **GPU** - CUDA-capable GPU available
5. **Discord Server** - Bot invited with Voice permissions

### ✅ Configuration Check

**Verify these settings in `config.yaml`:**

```yaml
pipeline:
  stt:
    model_size: "medium"
    device: "cuda"
    beam_size: 1  # ✅ Should be 1 (was 5)
```

**Verify `.env` file exists:**
```bash
# Check if .env is configured
cat .env | grep -E "(DISCORD_TOKEN|OPENCLAW_BASE_URL|OPENCLAW_AUTH_TOKEN)"
```

---

## Starting the Bot

### 1. Activate Environment

**Windows:**
```cmd
activate.bat
```

**If venv not found:**
```cmd
setup.bat
```

### 2. Start Bot

```cmd
python run.py
```

### 3. Expected Startup Output

**Watch for these critical logs:**

```
======================================================================
Jarvis Voice Bot Starting
======================================================================
Loading configuration...
✓ Discord token configured
✓ OpenClaw Gateway configured

Initializing TTS and STT engines...
Loading Chatterbox-Turbo on cuda...
Model loaded. Sample rate: 24000Hz
✓ TTS engine initialized (cuda)

🔥 NEW: Warming up TTS engine and caching common phrases...
Pre-generating 15 phrases for jarvis...
Cached phrase for jarvis: 'Yes, sir.'
Cached phrase for jarvis: 'Right away, sir.'
...
Warmup complete: cached 27 phrases in 8.3s (3.3 phrases/sec)
✓ TTS warmup complete (27 phrases cached)

Loading faster-whisper model: medium (device: cuda, compute: float16)
Whisper model loaded successfully: medium
✓ STT engine initialized (medium on cuda)

🔥 NEW: Query router initialized (default: sonnet)

✓ Discord bot started
✓ API server started on 0.0.0.0:8880

All services running. Press Ctrl+C to stop.
```

**🚨 If you don't see "TTS warmup complete" and "Query router initialized", the optimizations didn't load!**

---

## Discord Commands

### Join Voice Channel

In Discord server, type:
```
/join
```

**Or specify channel:**
```
/join channel:General Voice
```

**Expected Response:**
```
✅ Joined voice channel: General Voice
🎤 Listening for voice...
```

**Server Logs:**
```
Created pipeline for user: YourName (123456789)
Voice connection established
Audio bridge ready
```

---

## Testing the Optimizations

### Test 1: Simple Query + Cache Hit (Fastest)

**Goal:** Verify TTS cache is working (should be near-instant)

**Say:** "Hey Jarvis"

**Expected Behavior:**
- Response in ~400-700ms
- Router → Haiku
- TTS → Cache hit

**Server Logs to Watch:**
```
Speech started: YourName (123456789)
Speech ended: YourName (silence: 0.32s)
Turn complete for YourName (latency: 0.051s)

Transcribed (YourName): "Hey Jarvis" (latency: 0.287s)  ✅ Faster than before!
Added to transcript: YourName said "Hey Jarvis"

Responding to YourName: "Hey Jarvis" (latency: 0.113s)

🔥 NEW: Routed to haiku (confidence: 0.90, reason: matched_simple_pattern)

🔥 NEW: First sentence from LLM in 0.124s: "Yes, sir."

🔥 NEW: Cache hit for jarvis: 'Yes, sir.' (hit rate: 100.0%)

🔥 NEW: First audio playing in 0.154s (LLM: 0.124s, TTS: 0.030s)

Streaming response complete (jarvis, haiku): "Yes, sir."
Pipeline complete for YourName: total latency 0.673s

✅ SUCCESS: <1 second total latency!
```

**What This Tests:**
- ✅ STT beam_size=1 optimization
- ✅ Smart Model Router (Haiku selection)
- ✅ TTS phrase caching
- ✅ Total latency <1s

---

### Test 2: Simple Query + Cache Miss (Still Fast)

**Goal:** Verify Haiku routing for simple queries

**Say:** "Thank you Jarvis"

**Expected Behavior:**
- Response in ~700-1200ms
- Router → Haiku
- TTS → Cache miss (generate on-the-fly)

**Server Logs to Watch:**
```
Transcribed (YourName): "Thank you Jarvis" (latency: 0.312s)

🔥 NEW: Routed to haiku (confidence: 0.90, reason: matched_simple_pattern)

🔥 NEW: First sentence from LLM in 0.183s: "You're welcome, sir."

Cache miss  ← Phrase not in cache
Generating TTS for 'jarvis': "You're welcome, sir." (0 emotion tags)
Generated 1.24s audio in 0.38s (RTF: 0.31)

🔥 NEW: First audio playing in 0.612s (LLM: 0.183s, TTS: 0.429s)

Pipeline complete for YourName: total latency 1.087s

✅ SUCCESS: Just over 1 second!
```

**What This Tests:**
- ✅ Haiku routing for greetings/thanks
- ✅ Streaming TTS (generates while LLM streams)
- ✅ Total latency ~1s

---

### Test 3: Medium Query (Sonnet)

**Goal:** Verify Sonnet routing for medium complexity

**Say:** "What's the weather like today?"

**Expected Behavior:**
- Response in ~1-2s
- Router → Sonnet
- Sentence-level streaming TTS

**Server Logs to Watch:**
```
Transcribed (YourName): "What's the weather like today?" (latency: 0.341s)

🔥 NEW: Routed to sonnet (confidence: 0.80, reason: matched_medium_pattern)

🔥 NEW: First sentence from LLM in 0.423s: "Let me check the weather for you."

Extracted sentence #0: "Let me check the weather for you."
Cache miss
Generating TTS for 'jarvis': "Let me check the weather for you."
Generated 1.89s audio in 0.52s (RTF: 0.27)

🔥 NEW: First audio playing in 0.987s (LLM: 0.423s, TTS: 0.564s)

Extracted sentence #1: "Currently, it's partly cloudy with a temperature..."
Played sentence #0 (1.89s audio)
Generating TTS for sentence #1...
Played sentence #1 (2.34s audio)

Streaming response complete (jarvis, sonnet): "Let me check... Currently..."
Pipeline complete for YourName: total latency 2.134s

✅ SUCCESS: Under 2.5 seconds target!
```

**What This Tests:**
- ✅ Sonnet routing for information queries
- ✅ Sentence-level streaming (first audio while rest generates)
- ✅ Total latency <2.5s

---

### Test 4: Complex Query (Opus)

**Goal:** Verify Opus routing for complex analysis

**Say:** "Analyze the pros and cons of using Pipecat versus a custom voice pipeline"

**Expected Behavior:**
- Response in ~1.5-3s
- Router → Opus
- Multiple sentences streaming

**Server Logs to Watch:**
```
Transcribed (YourName): "Analyze the pros and cons of using Pipecat..." (latency: 0.387s)

🔥 NEW: Routed to opus (confidence: 0.85, reason: matched_complex_pattern)

🔥 NEW: First sentence from LLM in 0.892s: "That's an excellent question, sir."

Cache miss
Generating TTS...

🔥 NEW: First audio playing in 1.476s (LLM: 0.892s, TTS: 0.584s)

Extracted sentence #1: "Pipecat offers several advantages including..."
Extracted sentence #2: "On the other hand, a custom pipeline gives you..."
Extracted sentence #3: "In terms of performance, Pipecat claims..."

Streaming response complete (jarvis, opus): "That's an excellent... [full response]"
Pipeline complete for YourName: total latency 2.876s

✅ SUCCESS: Under 3 seconds for complex query!
```

**What This Tests:**
- ✅ Opus routing for analysis/complex queries
- ✅ Multi-sentence streaming
- ✅ Total latency <3s (acceptable for complex queries)

---

### Test 5: Barge-In (Interruption)

**Goal:** Verify barge-in support still works

**Say:** "Hey Jarvis, tell me a really long story about—"
**Then interrupt:** "Never mind"

**Expected Behavior:**
- Bot stops current response
- Processes new query immediately

**Server Logs:**
```
Responding to YourName: "Hey Jarvis, tell me..."
First audio playing in 1.123s
Playing sentence #0...

🔥 Barge-in detected: YourName spoke during response
Pipeline cancelled for YourName
Speech started: YourName (123456789)

Transcribed (YourName): "Never mind" (latency: 0.298s)
Routed to haiku (confidence: 0.90)
```

**What This Tests:**
- ✅ Barge-in detection works with streaming
- ✅ Pipeline cancellation
- ✅ Immediate processing of new query

---

## Performance Monitoring

### Real-Time Stats

**In Discord, type:**
```
/status
```

**Expected Response:**
```
📊 Jarvis Voice Bot Status

🎯 Active Agent: Jarvis
🔊 Sensitivity: medium
👥 Active Users: 1
💬 Total Utterances: 12
🤖 Total Responses: 8
🚫 Cancellations: 1

⚡ Performance (Average):
├─ STT: 0.31s  ✅ (was ~1-2s)
├─ Routing: 0.01s  🆕
├─ Relevance: 0.11s
├─ LLM (first sentence): 0.38s  🆕
├─ TTS (first chunk): 0.29s  🆕
├─ Time to First Audio: 0.89s  ⭐ KEY METRIC!
└─ Total: 1.87s  ✅ (was ~4-11s)

🧠 Model Usage:
├─ Haiku: 67% (8 queries)  ← Fast responses
├─ Sonnet: 25% (3 queries)  ← Medium complexity
└─ Opus: 8% (1 query)  ← Deep reasoning

💾 TTS Cache:
├─ Size: 27 phrases
├─ Hits: 5 (42%)  ← 42% instant responses!
└─ Misses: 7 (58%)
```

**🎯 Target Metrics:**
- **Time to First Audio:** <1.5s (was 4-11s)
- **Total Latency:** <2.5s (was 4-11s)
- **STT:** <500ms (was 1-2s)
- **Cache Hit Rate:** 30-50% (higher over time)

### API Stats Endpoint

**From another terminal:**
```bash
curl http://localhost:8880/stats | python -m json.tool
```

**Response:**
```json
{
  "active_users": 1,
  "current_agent": "jarvis",
  "total_utterances": 12,
  "total_responses": 8,
  "avg_time_to_first_audio_latency": 0.893,  ⭐ <1s!
  "avg_llm_first_sentence_latency": 0.382,
  "avg_tts_first_chunk_latency": 0.294,
  "avg_stt_latency": 0.314,
  "avg_total_latency": 1.872,  ⭐ <2s!

  "router_stats": {
    "total_routes": 12,
    "routes_by_model": {
      "haiku": 8,
      "sonnet": 3,
      "opus": 1
    },
    "distribution": {
      "haiku": 0.667,
      "sonnet": 0.250,
      "opus": 0.083
    }
  }
}
```

---

## Optimization Verification Checklist

After running all 5 tests, verify:

- [ ] **STT is faster:** Latency ~300ms (was 1-2s)
- [ ] **Router is working:** See "Routed to haiku/sonnet/opus" in logs
- [ ] **Cache is hitting:** See "Cache hit" for common phrases
- [ ] **Streaming is working:** See "First sentence from LLM" and "First audio playing"
- [ ] **Time to first audio:** <1.5s average
- [ ] **Total latency:** <2.5s for most queries
- [ ] **Model distribution:** ~60-70% Haiku, ~20-30% Sonnet, ~10% Opus

---

## Troubleshooting

### Problem: No "TTS warmup complete" log

**Cause:** TTS synthesizer not calling warmup

**Fix:**
```bash
# Check run.py has warmup call
grep "warmup" run.py
```

Should see:
```python
await tts_synthesizer.warmup()
```

**Restart bot after confirming.**

---

### Problem: No "Routed to" logs

**Cause:** Router not integrated into orchestrator

**Fix:**
```bash
# Check orchestrator has router
grep "query_router" pipeline/orchestrator.py
```

**Verify orchestrator initialization includes router.**

---

### Problem: Still slow (>3s latency)

**Check each stage:**

1. **STT slow (>1s)?**
   - Verify `beam_size: 1` in config
   - Check GPU is being used: `nvidia-smi`

2. **LLM slow (>2s first sentence)?**
   - Check OpenClaw Gateway is responding
   - Verify model routing is working (should use Haiku for simple queries)
   - Test Gateway directly:
     ```bash
     curl http://192.168.50.9:18789/health
     ```

3. **TTS slow (>1s)?**
   - Check GPU utilization
   - Verify Chatterbox-Turbo is loaded (not Coqui)
   - Check cache is enabled in tts.py

4. **Cache not hitting?**
   - Check exact LLM responses in logs
   - Add common variations to `TTSSynthesizer.COMMON_PHRASES`

---

### Problem: Router always uses Sonnet

**Cause:** Queries don't match patterns

**Debug:**
```python
# Test router manually
from pipeline.query_router import QueryRouter

router = QueryRouter()
print(router.route("Hey Jarvis"))
# Should show: model='haiku', reason='matched_simple_pattern'
```

**Fix:** Add custom patterns to `pipeline/query_router.py`

---

### Problem: Cache hit rate is 0%

**Cause:** Phrase normalization mismatch

**Debug:** Check logs for exact LLM responses. Example:

```
LLM response: "Yes sir."  ← Missing comma!
Cache key: "yes, sir"     ← Has comma
```

**Fix:** Add variation to COMMON_PHRASES or update normalization.

---

## Expected Results Summary

| Test | Before | After | Improvement |
|------|--------|-------|-------------|
| **Simple (cached)** | 4-7s | 0.4-0.7s | **6-10x faster** ✅ |
| **Simple (uncached)** | 4-7s | 0.7-1.2s | **4-6x faster** ✅ |
| **Medium** | 5-9s | 1-2s | **3-5x faster** ✅ |
| **Complex** | 6-11s | 1.5-3s | **2-4x faster** ✅ |

**🎯 All queries should be under 2.5 seconds!**

---

## Next Steps

### If Everything Works:

1. **Test with multiple users** in voice channel
2. **Monitor cache hit rate** over time (should increase as common responses are cached)
3. **Tune router patterns** for your specific use cases
4. **Add more cached phrases** based on actual usage logs

### If You Want Even Faster (<1s):

See `OPTIMIZATION_SUMMARY.md` for Phase 2 options:
- Kani-TTS-2 evaluation (faster TTS engine)
- Full Pipecat integration (500-800ms target)

---

## Recording Your Results

Create a results log:

```bash
# Run test session
echo "=== Optimization Test Results ===" > test_results.txt
echo "Date: $(date)" >> test_results.txt
echo "" >> test_results.txt

# Test each scenario and record
echo "Simple Query (cached): Hey Jarvis" >> test_results.txt
# ... copy latency from logs

echo "Simple Query (uncached): Thank you" >> test_results.txt
# ... copy latency from logs

# etc.
```

**Share your results!** Compare before/after latencies to verify the 3-10x improvement.

---

*Testing the optimizations is the fun part — enjoy the speed boost!* 🚀
