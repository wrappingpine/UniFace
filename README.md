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
uniface-unlock benchmark --frames 20   # more inference samples
```

The benchmark deliberately separates things that are easy to conflate:

- **Raw camera capture FPS** -- reads frames for 2 s with *no* inference.
  This is what the sensor/link can actually deliver.
- **Camera init** and **model load** -- reported as fixed setup costs rather
  than being smeared into a per-frame rate.
- **Cold vs warm latency** per stage (detection, recognition, liveness) plus
  end-to-end latency, measured on a frame that actually contains a face. The
  benchmark searches up to 60 frames for one, because recognition and liveness
  need a crop and a single arbitrary frame often has no face -- which used to
  leave those stages silently unmeasured.
- **Processed vs intentionally skipped vs dropped frames** over a 3 s window.
  "Intentional skips" come from `detection.frame_skip`; "camera drops" are
  frames the driver could not supply. Only processed frames feed the reported
  inference rate.
- **Process RSS** at baseline, after model load, and at exit, plus system RAM
  and swap.

An earlier revision reported an "Effective FPS" derived from wall-clock reads,
which counted frames discarded by frame skipping as if the pipeline had
processed them. That number is gone on purpose; it overstated throughput.

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
- Detection input size: 640 (see below)
- Detection threshold: 0.7
- Recognition threshold: calibrated per user
- Liveness threshold: 0.5
- Required consecutive frames: 3
- Authentication timeout: 30 seconds
- Frame skip: 2 (`authentication.frame_skip`)

### `detection.input_size`

Sets the SCRFD detection input resolution. Accepts an integer (square input)
or a `[width, height]` pair. A malformed value raises `ValueError` at detector
initialisation rather than being coerced, so a typo fails loudly at startup
instead of quietly changing recognition behaviour.

```json
{ "detection": { "input_size": 480 } }
```

**The default is left at 640, matching the upstream SCRFD default.** Smaller
inputs were measured on live frames with
`scripts/probe_scrfd_input_size.py` and `scripts/probe_recognition_vs_enrolled.py`.
Detection gets cheaper as the input shrinks, but recognition similarity against
the *existing enrolled template* also moves with detector input size. Two
independent probe sessions agreed on the direction and on the magnitude of the
shift (480 loses about 0.01-0.015 mean similarity versus 640, 320 loses about
0.05), and neither produced a single decision flip between sizes. Because the
enrolled template was produced through the default 640 path, changing input
size silently changes recognition behaviour, so it stays opt-in per machine.
Re-run both probes against your own enrollment before adopting a smaller value
(see Performance notes for the absolute similarity numbers, which are far less
stable than these relative deltas).

### `authentication.frame_skip`

Previously this key was read from config and silently ignored. It is now
enforced: after a frame that is processed without yielding a match, the next
`frame_skip` frames are read from the camera and discarded, which reduces the
duty cycle of the CPU-bound pipeline. A failed camera read counts as a skip and
resets accumulated progress, so skipping cannot be used to hold a stale partial
state open. Skips are reported by `benchmark` and included in the timeout
summary.

## Performance notes and limitations

Measurements from `benchmark` on the development machine (4 logical CPUs, ONNX
Runtime with the CPU execution provider only):

- **Latency is dominated by system memory pressure, not by the models.**
  Two consecutive `benchmark` runs of identical code, minutes apart, measured
  detection warm mean of 550 ms and 827 ms (n=15 each) while the system sat at
  ~92% RAM and 72-76% swap used. End-to-end per frame was 632 ms and 1091 ms.
  The spread between runs is larger than any code-level difference we are
  chasing, so treat any single number as an environment reading: close
  memory-heavy applications and re-measure before judging a change.
- **The camera is not the bottleneck; inference is.** Raw capture ran at 16.6
  and 20.0 FPS with 0 dropped frames, while inference-processed throughput was
  1.6 and 1.0 FPS. The gap is CPU inference plus intentional frame skipping.
- **Liveness performs one ONNX inference per call, saving roughly half the
  cost.** It previously ran two: a `predict()` call for the decision, then a
  second manual preprocess + `session.run()` used only to read logits for the
  diagnostics dict. The redundant inference was removed.
  `scripts/probe_liveness_double_inference.py` measures both paths on identical
  inputs with per-input warm-up, alternating execution order and median-of-reps
  timing. Three live runs on real camera frames with a detected face:

  | run | inputs | old (2 inferences) | new (1 inference) | saving |
  |-----|--------|--------------------|-------------------|--------|
  | 1   | 9      | 15.9 ms            | 7.4 ms            | 8.5 ms (53.3%) |
  | 2   | 12     | 17.5 ms            | 7.5 ms            | 9.9 ms (56.9%) |
  | 3   | 7      | 16.6 ms            | 9.0 ms            | 7.6 ms (45.6%) |

  A seeded synthetic run (no camera needed, seed 20261010) gives 11.9 ms ->
  5.6 ms, saving 6.2 ms (52.5%).

  Equivalence was checked, not assumed: `uniface.MiniFASNet.predict()` is
  `preprocess -> session.run -> softmax -> argmax` returning
  `is_real = (argmax == 1)`, so the single-pass path consumes identical math.
  Across all runs the two paths produced **0 decision mismatches** and a
  maximum probability difference of **0.000e+00** (bitwise identical). The
  benchmark corroborates this independently: `Liveness (MiniFASNet)` warm mean
  7.6 ms and 9.4 ms, matching the probe's single-pass figures.
- **Global ONNX thread-pool tuning was evaluated and rejected.** Setting
  `intra_op_num_threads` via the process-wide ONNX thread pool looked faster in
  isolation but broke sessions created inside the `uniface` package, producing
  provider/session errors. It is deliberately not configured here.
- **Recognition similarity is not reproducible across sessions, and this is the
  main open risk.** Similarity against the *same, unmodified* enrolled template
  varies enormously between probe sessions on the same machine:

  | probe session | SCRFD input 640 | input 480 | input 320 |
  |---------------|-----------------|-----------|-----------|
  | earlier session, 16 frames | mean 0.526 (min 0.457) | mean 0.525 (min 0.443) | mean 0.494 (min 0.391) |
  | this session, 8 frames     | mean 0.283 (min 0.218) | mean 0.271 (min 0.203) | mean 0.230 (min 0.167) |

  The *relative* ordering is stable (640 >= 480 > 320, with 0 decision flips
  between sizes in both sessions), but the *absolute* level moved by roughly
  0.24 between sessions. In the worst sample every frame sat far below the 0.70
  threshold (match rate 0.0%), so face unlock would time out even though
  detection and liveness were working. The threshold and the enrolled template
  were deliberately **not** changed to make any measurement pass. This must be
  diagnosed as a capture-pipeline problem (exposure, focus, distance, pose,
  alignment) before face unlock can be considered reliable; lowering the
  threshold would trade a reliability failure for a security failure.
- **`scripts/probe_recognition_vs_enrolled.py` is the tool for this.** It prints
  only aggregate similarity statistics -- never images, crops, landmarks or
  embeddings -- so it is safe to run and to quote.
- **MiniFASNet is not a strong anti-spoof guarantee.** It is a single-frame
  presentation-attack classifier. It raises the cost of a naive photo replay;
  it is not a defence against a determined attacker with a screen or a
  rendered model, and it should not be treated as a sole authentication factor
  for anything sensitive.
- **Diagnostics stay non-sensitive.** No face crops, embeddings, landmarks or
  passwords are written to logs or to disk by production code; benchmark and
  probe scripts report timings, rates, sizes and aggregate similarity
  statistics only. Probe scripts hold frames in memory and never write images.
- **The 0.7 recognition threshold is a starting point, not a calibrated
  operating point.** It has not been fitted against a impostor distribution for
  this camera/user pair; without that, neither the false-accept nor the
  false-reject rate is known.

## Hardware Requirements

- Pop!_OS 24.04 LTS with COSMIC desktop
- Webcam (V4L2 compatible)
- CPU with ONNX Runtime support
- ~500MB RAM available
- ~1GB disk for models

## Security

- Face embeddings stored locally only
- Never stores photographs
- Filesystem permissions protected (enrollment files `0600`, directory `0700`)
- Fail-closed design
- Enrollment templates are validated on load: a file that is not a finite,
  non-degenerate float vector of the expected dimension is rejected rather than
  compared, so a truncated or hand-edited template cannot match
- `numpy.load` is called without `allow_pickle`, so a crafted `.npy` payload is
  not deserialised
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
