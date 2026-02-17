"""
Kani-TTS-2 Testing Script
Compare Kani-TTS-2 with current Coqui XTTS v2 implementation
"""

import time
import wave
from pathlib import Path
import numpy as np

print("=" * 70)
print("Kani-TTS-2 Testing Script")
print("=" * 70)

# Test configuration
TEST_PHRASES = [
    "Yes, sir. I am at your service.",  # Short, simple (cache test)
    "The weather today is partly cloudy with a high of 72 degrees.",  # Medium
    "I've analyzed the data and found several interesting patterns that warrant further investigation.",  # Long
]

VOICE_FILES = {
    "jarvis": "server/voices/jarvis.mp3",
    "sage": "server/voices/sage.wav",
}

# Step 1: Check dependencies
print("\n[1/6] Checking dependencies...")
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError:
    print("[ERROR] PyTorch not installed")
    exit(1)

try:
    from kani_tts import KaniTTS, SpeakerEmbedder
    print("[OK] Kani-TTS-2 installed")
except ImportError:
    print("[WARN] Kani-TTS-2 not installed. Installing now...")
    import subprocess
    subprocess.run(["pip", "install", "kani-tts-2"], check=True)
    subprocess.run(["pip", "install", "-U", "transformers==4.56.0"], check=True)
    from kani_tts import KaniTTS, SpeakerEmbedder
    print("[OK] Kani-TTS-2 installed successfully")

# Step 2: Check voice files
print("\n[2/6] Checking voice reference files...")
available_voices = {}
for agent, voice_path in VOICE_FILES.items():
    if Path(voice_path).exists():
        print(f"[OK] {agent}: {voice_path}")
        available_voices[agent] = voice_path
    else:
        print(f"[WARN] {agent}: {voice_path} not found")

if not available_voices:
    print("[ERROR] No voice files found. Please add voice samples to server/voices/")
    exit(1)

# Step 3: Initialize Kani-TTS-2
print("\n[3/6] Initializing Kani-TTS-2 model...")
init_start = time.time()
try:
    model = KaniTTS('nineninesix/kani-tts-2-en')
    embedder = SpeakerEmbedder()
    init_time = time.time() - init_start
    print(f"[OK] Model loaded in {init_time:.2f}s")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Step 4: Generate speaker embeddings
print("\n[4/6] Generating speaker embeddings...")
speaker_embeddings = {}
for agent, voice_path in available_voices.items():
    try:
        embed_start = time.time()
        speaker_emb = embedder.embed_audio_file(voice_path)
        embed_time = time.time() - embed_start
        speaker_embeddings[agent] = speaker_emb
        print(f"[OK] {agent}: {speaker_emb.shape} in {embed_time:.2f}s")
    except Exception as e:
        print(f"[ERROR] {agent}: {e}")

# Step 5: Run latency benchmarks
print("\n[5/6] Running latency benchmarks...")
print("-" * 70)

results = []

for i, text in enumerate(TEST_PHRASES, 1):
    print(f"\n[Test {i}/3] \"{text[:50]}...\"")

    for agent, speaker_emb in speaker_embeddings.items():
        try:
            # Generate audio
            start = time.time()
            audio, processed_text = model(
                text,
                speaker_emb=speaker_emb,
                temperature=0.75,
                top_p=0.85
            )
            generation_time = time.time() - start

            # Calculate metrics
            audio_duration = len(audio) / 22050  # 22kHz sample rate
            rtf = generation_time / audio_duration

            # Save output
            output_path = f"test_outputs/kani_{agent}_test{i}.wav"
            Path("test_outputs").mkdir(exist_ok=True)
            model.save_audio(audio, output_path)

            print(f"  {agent}:")
            print(f"    Generation: {generation_time:.2f}s")
            print(f"    Audio length: {audio_duration:.2f}s")
            print(f"    RTF: {rtf:.2f}")
            print(f"    Output: {output_path}")

            results.append({
                "test": i,
                "agent": agent,
                "text_length": len(text),
                "generation_time": generation_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "output": output_path
            })

        except Exception as e:
            print(f"  {agent}: [ERROR] {e}")

# Step 6: Generate report
print("\n[6/6] Performance Summary")
print("=" * 70)

if results:
    avg_generation = np.mean([r["generation_time"] for r in results])
    avg_rtf = np.mean([r["rtf"] for r in results])

    print(f"\nAverage Metrics:")
    print(f"  Generation Time: {avg_generation:.2f}s")
    print(f"  RTF: {avg_rtf:.2f}")
    print(f"  Expected RTF from docs: ~0.2")

    print(f"\nPer-Test Breakdown:")
    for i in range(1, 4):
        test_results = [r for r in results if r["test"] == i]
        if test_results:
            test_rtf = np.mean([r["rtf"] for r in test_results])
            test_gen = np.mean([r["generation_time"] for r in test_results])
            print(f"  Test {i} ('{TEST_PHRASES[i-1][:30]}...')")
            print(f"    Avg Generation: {test_gen:.2f}s")
            print(f"    Avg RTF: {test_rtf:.2f}")

    print(f"\nOutput files saved to: test_outputs/")
    print(f"   Listen to samples and compare quality with current TTS")

    print(f"\n[OK] Testing complete!")
    print(f"\nNext steps:")
    print(f"  1. Listen to generated audio samples in test_outputs/")
    print(f"  2. Compare quality with current Coqui XTTS v2")
    print(f"  3. If quality is acceptable and RTF < 0.3, consider integration")
    print(f"  4. See KANI_TTS_INTEGRATION.md for implementation guide")

else:
    print("[ERROR] No successful tests - check errors above")

print("=" * 70)
