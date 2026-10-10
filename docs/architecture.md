# UniFace Unlock - Architecture Documentation

## Overview

This project implements a production-quality, fully local face authentication system using UniFace library v4.0.0 for Pop!_OS 24.04 LTS with COSMIC desktop.

## Repository Inspection Results

### UniFace Installation
- **Package**: `pip install "uniface[cpu]"` (CPU and Apple Silicon)
- **Version**: 4.0.0
- **Python**: >=3.10,<3.15
- **Core dependencies**: numpy, opencv-python, scikit-image, scipy, requests, tqdm
- **CPU backend**: onnxruntime>=1.16.0

### Architecture

```
Webcam
   ↓
Face Detection
   ↓
Exactly one face validation
   ↓
Anti-Spoofing / Liveness (MiniFASNet)
   ↓
Real person confirmation
   ↓
Face Recognition (MobileFace)
   ↓
Match against enrolled identity
   ↓
Multiple consecutive frames validation
   ↓
AUTHENTICATED
```

## Selected Models and APIs

### Face Detection

**Module**: `uniface.detection`

**Available detectors**:
- SCRFD (recommended for balance)
- RetinaFace
- CenterFace
- BlazeFace (lightweight)
- YOLOv5-Face
- YOLOv8-Face

**Base API**: `BaseDetector`
```python
from uniface.detection import SCRFD, RetinaFace

detector = SCRFD(config={...})
faces = detector.detect(image)  # Returns List[Face]

for face in faces:
    bbox = face.bbox  # np.ndarray (4,) [x1, y1, x2, y2]
    confidence = face.confidence  # float
    landmarks = face.landmarks  # np.ndarray (K, 2)
```

**Detection requirements**:
- Exactly one face must be detected
- Confidence threshold configurable
- Return bounding boxes and optionally landmarks
- Measure inference latency

### Face Recognition

**Module**: `uniface.recognition`

**Available models**:
- MobileFace (recommended for CPU, lightweight)
- AdaFace
- ArcFace
- EdgeFace
- SphereFace

**Base API**: `BaseRecognizer`
```python
from uniface.recognition import MobileFace

# MobileFace expects input size (112, 112)
recognizer = MobileFace(model_path=..., providers=['CPUExecutionProvider'])

# Get embedding
embedding = recognizer.get_normalized_embedding(image, landmarks)
# Returns: np.ndarray shape (512,)

# Comparison
from uniface.face_utils import compute_similarity
similarity = compute_similarity(embedding1, embedding2)
```

**Recognition workflow**:
1. Detect face and get bounding box + landmarks
2. Extract aligned face crop using landmarks
3. Preprocess (resize to 112x112, normalize)
4. Generate embedding
5. Compare with enrolled embedding using cosine similarity

### Anti-Spoofing / Liveness Detection

**Module**: `uniface.spoofing`

**Available models**:
- MiniFASNet (recommended, lightweight)

**Base API**: `BaseSpoofer`
```python
from uniface.spoofing import MiniFASNet

spoofer = MiniFASNet(providers=['CPUExecutionProvider'])

# Predict liveness
result = spoofer.predict(image, bbox)  # SpoofingResult
# Returns:
#   is_real: bool
#   confidence: float
```

**Liveness workflow**:
1. Face detected (bbox from detector)
2. Crop face region
3. Run MiniFASNet inference
4. Return SpoofingResult
5. Handle UNKNOWN conservatively (fail closed)

### Face Object Structure

```python
@dataclass
class Face:
    bbox: np.ndarray           # [x1, y1, x2, y2]
    confidence: float          # Detection confidence
    landmarks: np.ndarray      # (K, 2) landmarks
    embedding: np.ndarray | None
    gender: int | None
    age: int | None
    # ... optional attributes

@dataclass(frozen=True)
class SpoofingResult:
    is_real: bool
    confidence: float
```

## Hardware Constraints Compliance

**Target**: AMD Ryzen 3 3250U, 2 cores/4 threads, 5.7 GiB RAM, Radeon Vega 3

**Optimization strategy**:
- Use CPU-only ONNX Runtime
- MobileFace for recognition (small model)
- SCRFD or RetinaFace for detection (lighter than YOLO)
- MiniFASNet for anti-spoofing (lightweight)
- 640x480 camera resolution
- 5-10 FPS processing
- Frame skipping
- Model reuse (load once per session)
- ONNX Runtime CPU execution provider

## Security Considerations

**Local-only**:
- No cloud APIs
- No network requests for inference
- Models downloaded once, verified by SHA-256

**Data protection**:
- Store embeddings only, not photos
- File permissions: ~/.config/uniface-unlock/
- Fail closed on uncertain predictions
- Password remains primary authentication

## File Paths

```
~/.config/uniface-unlock/
├── enrollment.dat          # Enrolled embedding (binary)
├── config.json             # User configuration
└── logs/                   # Optional logs
```

## Performance Targets

- **Camera FPS**: 5-10 FPS
- **Detection latency**: <100ms
- **Recognition latency**: <150ms
- **Liveness latency**: <100ms
- **Total auth**: <500ms per frame
- **CPU usage**: <50% sustained
- **RAM usage**: <500MB total

## Configuration

Key parameters:
- detection_threshold: float (default 0.7)
- recognition_threshold: float (calibrated)
- liveness_threshold: float (default 0.5)
- required_consecutive_frames: int (default 3-5)
- camera_device: str (default /dev/video0)
- camera_resolution: tuple (default 640x480)
- camera_fps: int (default 5-10)

## PAM Integration Strategy

**Phase 1**: Standalone authentication system (no PAM modifications)

**Phase 2**: PAM integration after validation
- Investigate COSMIC Greeter PAM config
- Create small PAM helper daemon
- Never load ML models directly in PAM process
- Keep password authentication enabled
- Fail closed design
