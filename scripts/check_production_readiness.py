"""Production readiness checklist for Jarvis Voice Bot."""

import sys
from pathlib import Path


def check_env_file():
    """Check if .env file exists and is configured."""
    env_path = Path(__file__).parent.parent / ".env"
    
    if not env_path.exists():
        return False, ".env file not found (copy from .env.example)"
    
    # Check for placeholder values
    content = env_path.read_text()
    
    if "your_discord_bot_token_here" in content:
        return False, "Discord token not configured in .env"
    
    if "your-synology-nas" in content:
        return False, "OpenClaw URL not configured in .env"
    
    return True, ".env file configured"


def check_voice_files():
    """Check if voice reference files exist."""
    voices_dir = Path(__file__).parent.parent / "server" / "voices"
    
    required = ["jarvis.wav", "sage.wav"]
    missing = []
    
    for voice in required:
        if not (voices_dir / voice).exists():
            missing.append(voice)
    
    if missing:
        return False, f"Missing voice files: {', '.join(missing)}"
    
    return True, "Voice files present"


def check_models():
    """Check if models directory exists."""
    models_dir = Path(__file__).parent.parent / "models"
    
    if not models_dir.exists():
        return False, "Models directory not found"
    
    return True, "Models directory exists"


def check_python_version():
    """Check Python version."""
    import sys
    
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        return False, f"Python 3.12+ required (found {version.major}.{version.minor})"
    
    return True, f"Python {version.major}.{version.minor}.{version.micro}"


def main():
    """Run production readiness checks."""
    print("=" * 70)
    print("Jarvis Voice Bot - Production Readiness Checklist")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment Variables", check_env_file),
        ("Voice Reference Files", check_voice_files),
        ("Models Directory", check_models),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            passed, message = check_func()
            results.append((name, passed, message))
        except Exception as e:
            results.append((name, False, f"Check failed: {e}"))
    
    # Print results
    print()
    for name, passed, message in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {message}")
    
    # Summary
    total = len(results)
    passed_count = sum(1 for _, p, _ in results if p)
    
    print("\n" + "=" * 70)
    print(f"Results: {passed_count}/{total} checks passed")
    print("=" * 70)
    
    if passed_count == total:
        print("\n🎉 System is ready for production!")
        print("\nNext steps:")
        print("  1. Activate virtual environment: activate.bat")
        print("  2. Run the bot: python run.py")
        print("  3. Invite bot to Discord server")
        print("  4. Use /join command in voice channel")
        return 0
    else:
        print("\n⚠️  Please address the issues above before production use")
        return 1


if __name__ == "__main__":
    sys.exit(main())
