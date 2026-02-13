# Stubs, TODOs, and Temporary Items

This document tracks all temporary implementations, placeholders, and items that need to be replaced with real implementations.

## Phase 5: Smart Turn v3

### Mock ONNX Model
- **File:** `scripts/create_mock_turn_model.py`
- **File:** `models/smart_turn_v3.onnx` (generated mock, 164 bytes)
- **Status:** TEMPORARY - Mock model for testing
- **TODO:** Replace with actual Smart Turn v3 model from HuggingFace
  - Download from: `pipecat-ai/smart-turn-v3`
  - Expected file: `model.onnx` (~8MB)
  - Will need `huggingface_hub` package installed
- **Action:** Delete mock model and script once real model is downloaded
- **Command to download real model:**
  ```python
  from huggingface_hub import hf_hub_download
  downloaded_path = hf_hub_download(
      repo_id="pipecat-ai/smart-turn-v3",
      filename="model.onnx",
      cache_dir="models/",
  )
  ```

## Phase 9: OpenClaw Client

### Base URL Configuration
- **File:** `openclaw_client/client.py`
- **Line:** OpenClawConfig.base_url
- **Current:** `"http://your-synology-nas:port"`
- **Status:** PLACEHOLDER
- **TODO:** Replace with actual Synology NAS URL and port
  - Get actual URL/IP from user
  - Get actual port number
  - Example: `"http://192.168.1.100:8080"` or `"http://synology.local:8080"`

### Auth Token
- **File:** `openclaw_client/client.py`
- **Line:** OpenClawConfig.auth_token
- **Current:** `None`
- **Status:** PLACEHOLDER
- **TODO:** Get actual authentication token from OpenClaw instance
  - May need to generate API key in OpenClaw
  - Store in environment variable or config

### LLM Client Stub
- **File:** `openclaw_client/client.py`
- **Method:** `_send_request()`
- **Current:** Stubbed implementation with fallback placeholder response
- **Status:** STUB - For testing before OpenClaw integration
- **TODO:** Replace with actual OpenClaw API calls
  - Determine OpenClaw API endpoints
  - Implement proper request/response handling
  - May need session management
  - May need streaming support

### Agent Personalities
- **File:** `openclaw_client/client.py`
- **Constant:** AGENT_PERSONALITIES
- **Status:** TEMPORARY - Hardcoded for stub
- **TODO:**
  - Verify these match OpenClaw's agent definitions
  - May need to be fetched from OpenClaw API
  - May need to be configurable per deployment

## Phase 10: Chatterbox TTS

### TTS Engine Stub
- **File:** `server/tts.py`
- **Class:** ChatterboxTTS
- **Status:** STUB - Returns silence for testing
- **TODO:** Replace with actual Chatterbox TTS implementation
  - Verify Chatterbox TTS availability and installation
  - Alternative: Coqui XTTS v2 if Chatterbox unavailable
  - Install with: `pip install chatterbox-tts` (verify package name)
  - May need GPU support packages

### Voice Reference Files
- **Directory:** `server/voices/`
- **Files needed:**
  - `jarvis.wav` - Voice reference for Jarvis agent
  - `sage.wav` - Voice reference for Sage agent
- **Status:** MISSING - User must provide
- **TODO:**
  - Get 10-30 seconds of clean speech for each agent
  - Format: WAV, 22-48kHz sample rate
  - Place in `server/voices/` directory
  - Validate with: Check file size > 100KB

### Emotion Tag Support
- **File:** `server/tts.py`
- **Supported tags:** `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[whisper]`, `[excited]`, `[sad]`
- **Status:** Parsed but not used in stub
- **TODO:** Verify emotion tag support in actual Chatterbox TTS
  - May need different tag format
  - May need different tag names
  - Implement actual emotion control when real TTS integrated

## General Configuration Items

### Config File Settings
- **File:** `config.yaml`
- **Section:** `openclaw`
- **Fields to configure:**
  - `base_url`: Synology NAS URL
  - `auth_token`: From environment variable
  - `timeout`: May need tuning based on actual performance
  - `agent_personalities`: May need to match OpenClaw

### Environment Variables Needed
Create `.env` file with:
```
OPENCLAW_BASE_URL=http://your-synology-nas:port
OPENCLAW_AUTH_TOKEN=your-actual-token
DISCORD_BOT_TOKEN=your-discord-token
```

## Testing Items

### Mock LLM Classifier (Relevance Filter)
- **Used in:** `pipeline/relevance_filter.py` tests
- **Status:** Mock for unit testing only
- **TODO:** Integration tests will need real LLM or OpenClaw API

### Mock Whisper Model (STT)
- **Used in:** `server/stt.py` tests
- **Status:** Mocked in tests with `patch("server.stt.WhisperModel")`
- **TODO:** Integration tests will need actual model download
  - First run will download model (~500MB-5GB depending on size)
  - Configure model cache directory

## Cleanup Commands

Once real implementations are in place:

```bash
# Remove mock Smart Turn model
rm models/smart_turn_v3.onnx
rm scripts/create_mock_turn_model.py

# Verify real model exists
ls -lh models/  # Should show ~8MB model.onnx

# Update config.yaml with real values
# Update .env with real credentials
```

## Phase Completion Checklist

Before going to production:
- [ ] Download real Smart Turn v3 model from HuggingFace
- [ ] Remove mock ONNX model and script
- [ ] Configure Synology NAS URL in config
- [ ] Get OpenClaw auth token and configure
- [ ] Replace OpenClaw stub with real API integration
- [ ] Test with actual OpenClaw instance
- [ ] Download faster-whisper models (first run)
- [ ] Configure Discord bot token
- [ ] Set up voice reference files (jarvis.wav, sage.wav)
- [ ] Test end-to-end voice flow

## Implementation Progress

**Completed Phases (14/14 - 100% COMPLETE!):**
- [x] Phase 1: Project Scaffolding ✅
- [x] Phase 2: Audio Utilities & Format Conversion ✅
- [x] Phase 3: Discord Bot Foundation ✅
- [x] Phase 4: VAD & Audio Buffering ✅
- [x] Phase 5: Smart Turn v3 Integration ✅ (using mock model)
- [x] Phase 6: Speech-to-Text (STT) ✅
- [x] Phase 7: Transcript Management ✅
- [x] Phase 8: Relevance Filter ✅
- [x] Phase 9: OpenClaw Client (Stubbed) ✅
- [x] Phase 10: Text-to-Speech (Chatterbox TTS) ✅ (using stub)
- [x] Phase 11: Pipeline Orchestration ✅
- [x] Phase 12: FastAPI Server (TTS/STT API) ✅
- [x] Phase 13: Configuration & Environment Setup ✅
- [x] Phase 14: Testing & Polish ✅

**Remaining Phases:** NONE - PROJECT COMPLETE! 🎉

**Total Tests Passing:** 318 tests (as of Phase 14)
