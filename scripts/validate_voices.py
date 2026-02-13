"""Validate voice reference files for TTS."""

import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed")
    print("Run: pip install soundfile")
    sys.exit(1)


def validate_voice_file(file_path: Path) -> bool:
    """
    Validate a voice reference file.
    
    Args:
        file_path: Path to voice file
        
    Returns:
        True if valid, False otherwise
    """
    print(f"\nValidating: {file_path.name}")
    print("-" * 50)
    
    # Check if file exists
    if not file_path.exists():
        print("❌ File not found")
        return False
    
    print(f"✓ File exists")
    
    # Check file size
    file_size = file_path.stat().st_size
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    if file_size < 100_000:
        print("❌ File too small (should be at least 100KB)")
        return False
    
    print("✓ File size acceptable")
    
    try:
        # Read audio file
        audio, sample_rate = sf.read(str(file_path))
        
        # Duration
        if len(audio.shape) > 1:
            # Stereo
            duration = len(audio) / sample_rate
            channels = audio.shape[1]
        else:
            # Mono
            duration = len(audio) / sample_rate
            channels = 1
        
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels} ({'stereo' if channels > 1 else 'mono'})")
        print(f"  Duration: {duration:.2f} seconds")
        
        # Validate sample rate
        if sample_rate < 22050:
            print(f"⚠️  Sample rate is low (recommended: 22-48kHz)")
        else:
            print("✓ Sample rate acceptable")
        
        # Validate duration
        if duration < 10.0:
            print(f"❌ Duration too short (need at least 10 seconds, got {duration:.1f}s)")
            return False
        elif duration > 30.0:
            print(f"⚠️  Duration is long (recommended: 10-30 seconds, got {duration:.1f}s)")
        else:
            print("✓ Duration acceptable")
        
        # Check for silence
        import numpy as np
        audio_flat = audio.flatten() if len(audio.shape) > 1 else audio
        max_amplitude = np.abs(audio_flat).max()
        
        if max_amplitude < 0.01:
            print(f"❌ Audio seems to be silent (max amplitude: {max_amplitude:.4f})")
            return False
        
        print(f"  Max amplitude: {max_amplitude:.4f}")
        print("✓ Audio contains sound")
        
        print("\n✅ Voice file is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")
        return False


def main():
    """Main validation function."""
    print("=" * 70)
    print("Jarvis Voice Bot - Voice Reference Validation")
    print("=" * 70)
    
    # Get voices directory
    voices_dir = Path(__file__).parent.parent / "server" / "voices"
    
    if not voices_dir.exists():
        print(f"\nERROR: Voices directory not found: {voices_dir}")
        print("Run setup.bat first to create directory structure")
        sys.exit(1)
    
    print(f"\nVoices directory: {voices_dir}")
    
    # Check for required voice files
    required_voices = ["jarvis.wav", "sage.wav"]
    results = {}
    
    for voice_name in required_voices:
        voice_path = voices_dir / voice_name
        results[voice_name] = validate_voice_file(voice_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_valid = all(results.values())
    
    for voice_name, is_valid in results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID/MISSING"
        print(f"  {voice_name}: {status}")
    
    if all_valid:
        print("\n🎉 All voice files are valid!")
        print("\nYou can now start the bot with:")
        print("  activate.bat")
        print("  python run.py")
        return 0
    else:
        print("\n⚠️  Some voice files are missing or invalid")
        print("\nPlease add voice reference files to server/voices/:")
        print("  - Format: WAV")
        print("  - Sample rate: 22-48kHz")
        print("  - Duration: 10-30 seconds")
        print("  - Quality: Clean speech, minimal background noise")
        return 1


if __name__ == "__main__":
    sys.exit(main())
