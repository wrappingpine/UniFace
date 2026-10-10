# UniFace Unlock

Production-quality, fully local face authentication system for Pop!_OS 24.04 using UniFace library.

## Overview

This system provides local face authentication with:
- Face detection
- Face recognition  
- Anti-spoofing / liveness detection
- **No cloud services, no remote APIs, fully offline**

Built for AMD Ryzen 3 3250U with CPU inference optimization.

## Architecture

```
Webcam → Face Detection → Exactly one face? → Anti-Spoofing → Real person? 
→ Face Recognition → Match enrolled identity? → Multiple consecutive frames 
→ AUTHENTICATED
```

## Installation

```bash
# Clone repository
git clone <repo>
cd uniface-unlock

# Install dependencies
pip install -e ".[dev]"

# Or with requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Check camera
```bash
uniface-unlock status
```

### 2. Enroll a face
```bash
uniface-unlock enroll
```

### 3. Test recognition
```bash
uniface-unlock recognize
```

### 4. Test liveness
```bash
uniface-unlock liveness
```

### 5. Authenticate
```bash
uniface-unlock authenticate
```

### 6. Benchmark performance
```bash
uniface-unlock benchmark
```

## Commands

- `uniface-unlock enroll` - Enroll a new face
- `uniface-unlock recognize` - Test face recognition
- `uniface-unlock liveness` - Test anti-spoofing
- `uniface-unlock authenticate` - Full authentication
- `uniface-unlock benchmark` - Performance benchmark
- `uniface-unlock status` - System status
- `uniface-unlock test` - Run tests

## Configuration

Config file: `~/.config/uniface-unlock/config.json`

Default settings:
- Camera: `/dev/video0` at 640x480, 5-10 FPS
- Detection threshold: 0.7
- Recognition threshold: calibrated per user
- Liveness threshold: 0.5
- Required consecutive frames: 3-5

## Hardware Requirements

- Pop!_OS 24.04 LTS with COSMIC desktop
- Webcam (V4L2 compatible)
- CPU with ONNX Runtime support
- ~500MB RAM available
- ~1GB disk for models

## Security

- Face embeddings stored locally only
- Never stores photographs
- Filesystem permissions protected
- Fail-closed design
- Password authentication remains primary
- No cloud services

## PAM Integration

**IMPORTANT**: Do NOT modify PAM until standalone system is fully tested.

See `docs/pam.md` for PAM integration guide after validation.

## Documentation

- `docs/architecture.md` - System architecture
- `docs/security.md` - Security model and limitations
- `docs/pam.md` - PAM integration guide

## Development

See `docs/architecture.md` for technical details.
