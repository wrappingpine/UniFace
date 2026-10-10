"""CLI interface for UniFace Unlock."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .camera import Camera, check_camera, list_cameras
from .config import get_config
from .enrollment import FaceEnrollment
from .authenticator import FaceAuthenticator, AuthenticationError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def cli(verbose: bool) -> None:
    """UniFace Unlock - Local face authentication."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
def enroll() -> None:
    """Enroll a new face for authentication."""
    try:
        # Check camera
        config = get_config()
        device = config.get("camera.device", "/dev/video0")
        
        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            click.echo(f"Available cameras: {list_cameras()}", err=True)
            sys.exit(2)

        click.echo("📷 Camera detected")
        click.echo("👤 Starting enrollment...")
        click.echo("Please look at the camera naturally")

        enrollment = FaceEnrollment()
        enrollment.enroll()

        click.echo("✅ Enrollment completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        click.echo(f"❌ Enrollment failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def authenticate() -> None:
    """Authenticate using face recognition."""
    try:
        config = get_config()
        device = config.get("camera.device", "/dev/video0")
        
        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            sys.exit(2)

        authenticator = FaceAuthenticator()
        
        if not authenticator.enrollment.is_enrolled():
            click.echo("❌ No face enrolled. Run 'uniface-unlock enroll' first", err=True)
            sys.exit(3)

        click.echo("🔍 Looking for face...")
        click.echo("Please look at the camera")

        success = authenticator.authenticate()

        if success:
            click.echo("✅ AUTHENTICATED")
            sys.exit(0)
        else:
            click.echo("❌ Authentication failed", err=True)
            sys.exit(1)

    except AuthenticationError as e:
        if e.code == 2:
            click.echo(f"❌ Camera failure: {e}", err=True)
            sys.exit(2)
        elif e.code == 3:
            click.echo(f"❌ Configuration error: {e}", err=True)
            sys.exit(3)
        elif e.code == 5:
            click.echo(f"❌ Timeout: {e}", err=True)
            sys.exit(5)
        else:
            click.echo(f"❌ Authentication failed: {e}", err=True)
            sys.exit(1)
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def recognize() -> None:
    """Test face recognition."""
    try:
        from .recognizer import FaceRecognizer
        from .camera import Camera
        from .detector import FaceDetector

        config = get_config()
        device = config.get("camera.device", "/dev/video0")

        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            sys.exit(2)

        enrollment = FaceEnrollment()
        if not enrollment.is_enrolled():
            click.echo("❌ No face enrolled", err=True)
            sys.exit(3)

        detector = FaceDetector()
        recognizer = FaceRecognizer()
        enrolled_embedding = enrollment.load_enrollment()

        with Camera(device=device) as cam:
            click.echo("📷 Camera opened, detecting face...")
            frame = cam.read()

            if frame is None:
                click.echo("❌ Failed to read frame", err=True)
                sys.exit(1)

            try:
                face = detector.validate_exactly_one_face(frame)
                is_match, similarity = recognizer.verify(
                    frame, face.landmarks, enrolled_embedding
                )

                click.echo(f"Similarity: {similarity:.3f}")
                if is_match:
                    click.echo("✅ Face recognized")
                    sys.exit(0)
                else:
                    click.echo("❌ Face not recognized")
                    sys.exit(1)
            except Exception as e:
                click.echo(f"❌ Recognition failed: {e}", err=True)
                sys.exit(1)

    except Exception as e:
        logger.error(f"Recognition error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def liveness() -> None:
    """Test anti-spoofing liveness detection."""
    try:
        from .liveness import LivenessDetector
        from .camera import Camera
        from .detector import FaceDetector

        config = get_config()
        device = config.get("camera.device", "/dev/video0")

        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            sys.exit(2)

        detector = FaceDetector()
        liveness = LivenessDetector()

        with Camera(device=device) as cam:
            click.echo("📷 Camera opened, detecting face...")
            frame = cam.read()

            if frame is None:
                click.echo("❌ Failed to read frame", err=True)
                sys.exit(1)

            try:
                face = detector.validate_exactly_one_face(frame)
                status, confidence, details = liveness.check_liveness(frame, face.bbox, raw_output=True)

                click.echo(f"Liveness status: {status}")
                click.echo(f"Confidence: {confidence:.3f}")
                click.echo(f"Fake probability: {details['fake_probability']:.3f}")
                click.echo(f"Real probability: {details['real_probability']:.3f}")
                click.echo(f"Latency: {details['latency_ms']:.1f}ms")

                if status == "REAL":
                    click.echo("✅ Real person detected")
                    sys.exit(0)
                else:
                    click.echo(f"❌ Liveness check failed: {status}")
                    sys.exit(1)
            except Exception as e:
                click.echo(f"❌ Liveness check failed: {e}", err=True)
                sys.exit(1)

    except Exception as e:
        logger.error(f"Liveness error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def liveness_test() -> None:
    """Diagnostic mode for liveness detection with detailed output."""
    try:
        from .liveness import LivenessDetector
        from .camera import Camera
        from .detector import FaceDetector
        import time

        config = get_config()
        device = config.get("camera.device", "/dev/video0")

        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            sys.exit(2)

        detector = FaceDetector()
        liveness = LivenessDetector()

        click.echo("🔬 Liveness Diagnostic Mode")
        click.echo("Press Ctrl+C to exit")
        click.echo()
        click.echo(f"{'Frame':>5} | {'Status':>7} | {'Real':>7} | {'Fake':>7} | {'Mask':>7} | {'Class':>5} | {'Latency':>8} | {'FPS':>5}")
        click.echo("-" * 75)

        with Camera(device=device) as cam:
            frame_count = 0
            start_time = time.time()

            try:
                while True:
                    frame = cam.read()
                    if frame is None:
                        continue

                    frame_count += 1

                    try:
                        face = detector.validate_exactly_one_face(frame)
                        status, confidence, details = liveness.check_liveness(
                            frame, face.bbox, raw_output=True
                        )

                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0

                        click.echo(f"\r{frame_count:>5} | {status:>7} | "
                                  f"{details['real_probability']:>7.3f} | "
                                  f"{details['fake_probability']:>7.3f} | "
                                  f"{details['mask_probability']:>7.3f} | "
                                  f"{details['predicted_class']:>5} | "
                                  f"{details['latency_ms']:>7.1f}ms | "
                                  f"{fps:>5.1f}", nl=False)

                    except Exception:
                        continue

            except KeyboardInterrupt:
                click.echo("\n\n✅ Diagnostic stopped")
                sys.exit(0)

    except Exception as e:
        logger.error(f"Liveness test error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def recognition_test() -> None:
    """Diagnostic mode for face recognition with statistics."""
    try:
        from .recognizer import FaceRecognizer
        from .camera import Camera
        from .detector import FaceDetector
        from .enrollment import FaceEnrollment
        import time
        import numpy as np

        config = get_config()
        device = config.get("camera.device", "/dev/video0")

        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            sys.exit(2)

        enrollment = FaceEnrollment()
        if not enrollment.is_enrolled():
            click.echo("❌ No face enrolled", err=True)
            sys.exit(3)

        detector = FaceDetector()
        recognizer = FaceRecognizer()
        enrolled_embedding = enrollment.load_enrollment()
        threshold = recognizer.threshold

        click.echo("👤 Recognition Diagnostic Mode")
        click.echo("Press Ctrl+C to exit")
        click.echo()
        click.echo(f"{'Frame':>5} | {'Similarity':>9} | {'Threshold':>9} | {'Match':>5} | {'Latency':>8} | {'FPS':>5}")
        click.echo("-" * 65)

        similarities = []
        with Camera(device=device) as cam:
            frame_count = 0
            start_time = time.time()

            try:
                while True:
                    frame = cam.read()
                    if frame is None:
                        continue

                    frame_count += 1

                    try:
                        face = detector.validate_exactly_one_face(frame)
                        is_match, similarity = recognizer.verify(
                            frame, face.landmarks, enrolled_embedding
                        )
                        similarities.append(similarity)

                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0

                        click.echo(f"\r{frame_count:>5} | {similarity:>9.3f} | "
                                  f"{threshold:>9.3f} | {'YES' if is_match else 'NO':>5} | "
                                  f"{0:>7.1f}ms | {fps:>5.1f}", nl=False)

                    except Exception:
                        continue

            except KeyboardInterrupt:
                click.echo()
                click.echo()
                click.echo("=== Recognition Statistics ===")
                if len(similarities) >= 5:
                    sims = np.array(similarities)
                    click.echo(f"Samples collected: {len(similarities)}")
                    click.echo(f"Mean similarity:   {np.mean(sims):.3f}")
                    click.echo(f"Std deviation:     {np.std(sims):.3f}")
                    click.echo(f"Min similarity:    {np.min(sims):.3f}")
                    click.echo(f"Max similarity:    {np.max(sims):.3f}")
                    click.echo(f"Median similarity: {np.median(sims):.3f}")
                    click.echo(f"Match rate:        {np.mean(sims >= threshold)*100:.1f}%")
                else:
                    click.echo("❌ Insufficient samples for statistics")
                sys.exit(0)

    except Exception as e:
        logger.error(f"Recognition test error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status() -> None:
    """Show system status."""
    config = get_config()
    enrollment = FaceEnrollment()
    authenticator = FaceAuthenticator()

    click.echo("=== UniFace Unlock Status ===")
    click.echo()

    # Camera
    device = config.get("camera.device", "/dev/video0")
    cameras = list_cameras()
    click.echo(f"Camera: {device}")
    click.echo(f"Available: {device in cameras}")
    click.echo(f"Working: {check_camera(device)}")
    click.echo()

    # Enrollment
    click.echo(f"Enrolled: {enrollment.is_enrolled()}")
    if enrollment.is_enrolled():
        info = enrollment.get_info()
        click.echo(f"Path: {info['path']}")
    click.echo()

    # Config
    click.echo("Configuration:")
    click.echo(f"  Detection: {config.get('detection.model')} (threshold {config.get('detection.threshold')})")
    click.echo(f"  Recognition: {config.get('recognition.model')} (threshold {config.get('recognition.threshold')})")
    click.echo(f"  Liveness: {config.get('liveness.model')} (threshold {config.get('liveness.threshold')})")
    click.echo(f"  Required frames: {config.get('authentication.required_consecutive_frames')}")
    click.echo()

    click.echo("✅ System ready")


@cli.command()
@click.option("--frames", "-n", default=6, help="Inference samples per stage (warm samples = frames-1)")
def benchmark(frames: int) -> None:
    """Benchmark performance with honest metric separation.

    Distinguishes:
      - Raw camera capture FPS (no inference)
      - Detection-only latency (SCRFD), cold vs warm
      - Recognition latency (MobileFace)
      - Liveness latency (MiniFASNet, single pass)
      - End-to-end per-frame pipeline latency
      - Intentional frames skipped (authentication.frame_skip)
      - Dropped/failed camera reads
      - Process RSS before/after model load and at exit
      - System RAM and swap usage

    Does NOT enroll faces, modify thresholds, or touch PAM.
    """
    try:
        import statistics
        import time as _time
        import numpy as np
        import psutil
        from .camera import Camera
        from .detector import FaceDetector
        from .recognizer import FaceRecognizer
        from .liveness import LivenessDetector

        config = get_config()
        device = config.get("camera.device", "/dev/video0")

        if not check_camera(device):
            click.echo(f"\u274c Camera not available: {device}", err=True)
            sys.exit(2)

        def _rss_mb():
            try:
                with open("/proc/self/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024.0
            except OSError:
                pass
            return float("nan")

        click.echo("\U0001F4CA Running UniFace Unlock benchmark...")
        rss_base = _rss_mb()
        click.echo(f"Process RSS baseline: {rss_base:.1f} MiB")
        click.echo()

        # 1. Camera initialization
        _t_cam_start = _time.perf_counter()
        with Camera(device=device) as cam:
            cam_init_ms = (_time.perf_counter() - _t_cam_start) * 1000

            warm_frame = cam.read()
            if warm_frame is None:
                click.echo("\u274c Failed to read camera frame", err=True)
                sys.exit(1)
            # The first frames from most V4L2 devices are under-exposed while
            # auto-exposure settles, so discard a few before measuring.
            for _ in range(5):
                fr = cam.read()
                if fr is not None:
                    warm_frame = fr

            # 2. Raw camera capture rate (NO inference -- measures the camera,
            #    not the pipeline; the old benchmark misleadingly labelled
            #    this "Effective FPS")
            click.echo("Measuring raw camera capture rate (no inference)...")
            cap_count = 0
            cap_dropped = 0
            t_cap = _time.perf_counter()
            while _time.perf_counter() - t_cap < 2.0:
                if cam.read() is not None:
                    cap_count += 1
                else:
                    cap_dropped += 1
            cap_elapsed = _time.perf_counter() - t_cap
            raw_fps = cap_count / cap_elapsed if cap_elapsed > 0 else 0.0

            # 3. Model initialization (cold)
            click.echo("Loading models...")
            t_load = _time.perf_counter()
            detector = FaceDetector()
            recognizer = FaceRecognizer()
            liveness = LivenessDetector()
            model_init_ms = (_time.perf_counter() - t_load) * 1000
            rss_models = _rss_mb()

            def _lat(fn, n):
                out = []
                for _ in range(n):
                    t0 = _time.perf_counter()
                    fn()
                    out.append((_time.perf_counter() - t0) * 1000)
                return out

            # 4. Detection latency (cold sample is index 0)
            click.echo("Measuring detection latency (SCRFD)...")
            det_lats = _lat(lambda: detector.detect(warm_frame), frames)

            # Recognition and liveness need a crop, so their latency is only
            # meaningful on a frame that actually contains a face. Search a
            # bounded number of frames rather than gambling on one arbitrary
            # frame, which used to leave these stages silently unmeasured.
            # Frames are held in memory only and never written to disk.
            click.echo("Searching for a frame containing a face...")
            face_frame = None
            face = None
            search_reads = 0
            while search_reads < 60:
                search_reads += 1
                fr = cam.read()
                if fr is None:
                    continue
                fs = detector.detect(fr)
                if fs:
                    face_frame, face = fr, fs[0]
                    break

            rec_lats: list = []
            live_lats: list = []
            e2e_lats: list = []
            if face_frame is not None:
                click.echo("Measuring recognition latency (MobileFace)...")
                rec_lats = _lat(
                    lambda: recognizer.extract_embedding(face_frame, face.landmarks), frames
                )
                click.echo("Measuring liveness latency (MiniFASNet)...")
                live_lats = _lat(
                    lambda: liveness.check_liveness(face_frame, face.bbox), frames
                )
                click.echo("Measuring end-to-end pipeline latency (detect+liveness+recognize)...")

                def _e2e():
                    # Uses the face-bearing frame so all three stages actually
                    # run; on a faceless frame detect() short-circuits and the
                    # measurement silently degrades to detection only.
                    fs = detector.detect(face_frame)
                    if fs:
                        liveness.check_liveness(face_frame, fs[0].bbox)
                        recognizer.extract_embedding(face_frame, fs[0].landmarks)

                e2e_lats = _lat(_e2e, max(3, frames // 2))
            else:
                click.echo("\u26a0\ufe0f  No face found in "
                           f"{search_reads} frames; downstream stages not measured")

            # 5. Pipeline duty cycle: processed vs intentionally skipped vs dropped
            click.echo("Measuring pipeline duty cycle over 3 seconds...")
            frame_skip = int(config.get("authentication.frame_skip", 2))
            processed = 0
            intentional_skips = 0
            camera_drops = 0
            t_pipe = _time.perf_counter()
            while _time.perf_counter() - t_pipe < 3.0:
                fr = cam.read()
                if fr is None:
                    camera_drops += 1
                    continue
                detector.detect(fr)
                processed += 1
                for _ in range(frame_skip):
                    if _time.perf_counter() - t_pipe >= 3.0:
                        break
                    if cam.read() is None:
                        camera_drops += 1
                    else:
                        intentional_skips += 1
            pipe_elapsed = _time.perf_counter() - t_pipe
            eff_fps = processed / pipe_elapsed if pipe_elapsed > 0 else 0.0

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        rss_end = _rss_mb()
        cpu_percent = psutil.cpu_percent(interval=0.5)

        def _stat(name, lats):
            if not lats:
                return
            warm = lats[1:] or lats
            click.echo(
                f"{name:<26} cold={lats[0]:>7.1f}  "
                f"warm mean={statistics.fmean(warm):>7.1f}  "
                f"med={statistics.median(warm):>7.1f}  "
                f"max={max(warm):>7.1f}  n={len(lats)}"
            )

        click.echo()
        click.echo("=== Latency profile (ms) ===")
        click.echo(f"Camera init:                 {cam_init_ms:>7.1f}")
        click.echo(f"Model load (3 models):       {model_init_ms:>7.1f}")
        _stat("Detection (SCRFD)", det_lats)
        _stat("Recognition (MobileFace)", rec_lats)
        _stat("Liveness (MiniFASNet)", live_lats)
        _stat("End-to-end per frame", e2e_lats)

        click.echo()
        click.echo("=== Frame rates ===")
        click.echo(f"Raw camera capture:  {raw_fps:>6.1f} FPS  "
                   f"({cap_count} reads, {cap_dropped} dropped)")
        click.echo(f"Inference-processed: {eff_fps:>6.1f} FPS  ({processed} frames)")
        click.echo(f"Intentional skips:   {intentional_skips} (frame_skip={frame_skip})")
        click.echo(f"Camera drops:        {camera_drops}")

        click.echo()
        click.echo("=== Memory ===")
        click.echo(f"Process RSS baseline:   {rss_base:>7.1f} MiB")
        click.echo(f"Process RSS after load: {rss_models:>7.1f} MiB (+{rss_models - rss_base:.1f})")
        click.echo(f"Process RSS at exit:    {rss_end:>7.1f} MiB")
        click.echo(f"System RAM: {mem.percent:.1f}% used, {mem.available / 1024**2:.0f} MiB available")
        click.echo(f"System swap: {swap.percent:.1f}% used ({swap.used / 1024**2:.0f} / {swap.total / 1024**2:.0f} MiB)")
        click.echo(f"CPU (system-wide, 0.5s sample): {cpu_percent:.0f}%")
        click.echo()
        click.echo("\u2705 Benchmark complete")

    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        click.echo(f"\u274c Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def calibration() -> None:
    """Calibrate recognition threshold using collected samples."""
    try:
        from .recognizer import FaceRecognizer
        from .camera import Camera
        from .detector import FaceDetector
        from .enrollment import FaceEnrollment
        import time

        config = get_config()
        device = config.get("camera.device", "/dev/video0")

        if not check_camera(device):
            click.echo(f"❌ Camera not available: {device}", err=True)
            sys.exit(2)

        enrollment = FaceEnrollment()
        if not enrollment.is_enrolled():
            click.echo("❌ No face enrolled", err=True)
            sys.exit(3)

        detector = FaceDetector()
        recognizer = FaceRecognizer()
        enrolled_embedding = enrollment.load_enrollment()

        click.echo("📊 Recognition Calibration")
        click.echo("Collect samples to determine optimal threshold")
        click.echo("Press Ctrl+C to finish collection")
        click.echo()

        similarities = []
        with Camera(device=device) as cam:
            try:
                while len(similarities) < 50:
                    frame = cam.read()
                    if frame is None:
                        continue

                    try:
                        face = detector.validate_exactly_one_face(frame)
                        is_match, similarity = recognizer.verify(
                            frame, face.landmarks, enrolled_embedding
                        )
                        similarities.append(similarity)
                        
                        click.echo(f"\rCollected {len(similarities)} samples | "
                                  f"Similarity: {similarity:.3f} | "
                                  f"Match: {'YES' if is_match else 'NO'}", nl=False)
                        
                    except Exception:
                        continue
                        
                click.echo()
                
            except KeyboardInterrupt:
                pass

        if len(similarities) < 5:
            click.echo("❌ Insufficient samples for calibration", err=True)
            sys.exit(1)

        import numpy as np
        similarities = np.array(similarities)
        
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))
        median_sim = float(np.median(similarities))
        
        # Suggest threshold at mean - 2*std (with minimum)
        suggested_threshold = max(0.5, mean_sim - 2 * std_sim)
        
        click.echo()
        click.echo("=== Calibration Results ===")
        click.echo(f"Samples collected: {len(similarities)}")
        click.echo(f"Mean similarity: {mean_sim:.3f}")
        click.echo(f"Std deviation: {std_sim:.3f}")
        click.echo(f"Min similarity: {min_sim:.3f}")
        click.echo(f"Max similarity: {max_sim:.3f}")
        click.echo(f"Median similarity: {median_sim:.3f}")
        click.echo()
        click.echo(f"Suggested threshold: {suggested_threshold:.3f}")
        click.echo(f"Current threshold: {config.get('recognition.threshold')}")
        
        if suggested_threshold != config.get('recognition.threshold'):
            click.echo()
            click.echo("⚠️  Threshold adjustment recommended")
            click.echo("Update config to use suggested threshold")
        else:
            click.echo()
            click.echo("✅ Threshold is optimal")

    except Exception as e:
        logger.error(f"Calibration error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--seconds', '-s', default=20, help='Seconds to show the preview window')
@click.option('--save-dir', default=None, help=
    'Opt-in only: directory to save snapshots into. Nothing is written '
    'unless this flag is passed explicitly.')
def preview(seconds: int, save_dir: str | None) -> None:
    """LOCAL-ONLY visual preview of the exact model input crops.

    Shows one local window with: the raw frame (bbox + 5 landmarks), the
    aligned 112x112 crop fed to MobileFace, and the 80x80 crop fed to
    MiniFASNet. Press 'q' to quit early.

    Privacy: nothing is written to disk unless --save-dir is explicitly
    passed. No data leaves the machine (no network I/O anywhere).
    """
    import os
    from .preview import compose_preview, pick_qt_backend, set_message_canvas
    # Pick the Qt platform plugin that actually exists on this machine before
    # cv2 is imported, so the Wayland->xcb font fallback never fires.
    os.environ.setdefault("QT_QPA_PLATFORM", pick_qt_backend())
    import time
    import cv2
    import numpy as np
    from uniface.face_utils import face_alignment
    from .detector import FaceDetector, DetectionError
    from .recognizer import FaceRecognizer
    from .liveness import LivenessDetector
    from .enrollment import FaceEnrollment
    from .quality import assess_face

    config = get_config()
    device = config.get("camera.device", "/dev/video0")
    if not check_camera(device):
        click.echo(f"\u274c Camera not available: {device}", err=True)
        sys.exit(1)

    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        click.echo("\u274c No display found. Run 'uniface-unlock preview' from a desktop "
                   "session (or use 'diagnose -n N' for headless numbers).", err=True)
        sys.exit(1)

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    liveness = LivenessDetector()
    enrollment = FaceEnrollment()
    tmpl = enrollment.load_enrollment() if enrollment.is_enrolled() else None

    out_dir = Path(save_dir).expanduser() if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Snapshots will be saved to {out_dir}")
    click.echo("Local preview window open. Press 'q' to quit.")

    WIN = "uniface-unlock crop preview (local only)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    start = time.time()
    snap_count = 0
    try:
        with Camera(device=device, fps=10) as cam:
            while time.time() - start < seconds:
                frame = cam.read()
                if frame is None:
                    # No frame at all: show a placeholder so the loop keeps running.
                    cv2.imshow(WIN, set_message_canvas("camera read failed"))
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                    continue
                try:
                    face = detector.validate_exactly_one_face(frame)
                except DetectionError as e:
                    canvas = cv2.resize(frame, (640, 480))
                    cv2.putText(canvas, str(e), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(WIN, canvas)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                    continue

                # Exact MobileFace input: aligned 112x112 similarity warp
                aligned, _ = face_alignment(frame, face.landmarks, recognizer.recognizer.input_size)
                # Exact MiniFASNet input: scaled-bbox crop resized to model size
                mf_crop = liveness.spoofer._crop_face(
                    frame, liveness.spoofer._xyxy_to_xywh(face.bbox)
                )

                status, _conf, details = liveness.check_liveness(frame, face.bbox)
                verdict = assess_face(frame, face.bbox, face.landmarks, aligned_crop=aligned)
                sim_txt = "n/a"
                if tmpl is not None:
                    try:
                        _m, sim = recognizer.verify(frame, face.landmarks, tmpl)
                        sim_txt = f"{sim:.3f}"
                    except Exception:
                        pass

                # Presentation-only panel: draw bbox + landmarks on a copy of the
                # raw frame, then let compose_preview fit it proportionally to a
                # common display height. The model-input crops (aligned, mf_crop)
                # are passed through untouched.
                raw = frame.copy()
                b = face.bbox.astype(int)
                cv2.rectangle(raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
                for i, (x, y) in enumerate(face.landmarks.astype(int)):
                    cv2.circle(raw, (x, y), 2, (0, 0, 255), -1)
                    cv2.putText(raw, str(i), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

                q_txt = "OK" if verdict["ok"] else "REJECT: " + ", ".join(verdict["problems"])[:44]
                thr = config.get("recognition.threshold", 0.7)
                canvas = compose_preview(
                    raw=raw,
                    aligned=aligned,
                    fas=mf_crop,
                    lines=[
                        "RAW            ALIGNED-112(MobileFace)   FAS-80(MiniFASNet)",
                        f"sim={sim_txt}/{thr:.2f}  liveness={status} "
                        f"real={details['real_probability']:.2f} "
                        f"fake={details['fake_probability']:.2f} "
                        f"mask={details['mask_probability']:.2f}",
                        f"quality: {q_txt}",
                    ],
                )

                cv2.imshow(WIN, canvas)

                if out_dir and (snap_count % 10) == 0:
                    p = out_dir / f"preview_{snap_count:04d}.png"
                    cv2.imwrite(str(p), canvas)
                    logger.info(f"Snapshot saved: {p}")
                snap_count += 1

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()


@cli.command()
@click.option('--live-frames', '-n', default=0, help='Also report live similarity stats over N frames')
def diagnose(live_frames: int) -> None:
    """Safe diagnostic: model identity, preprocessing, template integrity, similarity.

    Reports model/config identity, embedding shape/dtype/finite/norm, file hashes
    and template file stats, plus bbox dimensions and similarity when --live-frames
    is given. Never prints biometric embedding values.
    """
    import hashlib
    import time
    import numpy as np

    try:
        from .detector import FaceDetector, DetectionError
        from .recognizer import FaceRecognizer, RecognitionError

        config = get_config()
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        enrollment = FaceEnrollment()

        click.echo("=== Model identity / configuration ===")
        rec = recognizer.recognizer
        rec_path = Path(rec.model_path)
        det = detector.detector
        det_path = Path(getattr(det, 'model_path', 'unknown'))

        def sha(p):
            try:
                return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
            except Exception:
                return 'unknown'

        click.echo(f"Recognition class:   {rec.__class__.__name__}")
        click.echo(f"Recognition weights: {rec_path.name}  sha256(16)={sha(rec_path)}  bytes={rec_path.stat().st_size if rec_path.exists() else 'missing'}")
        click.echo(f"Recognition input:   size={rec.input_size} mean={rec.input_mean} std={rec.input_std} swapRB=True (BGR->RGB) align=similarity(5pt)")
        click.echo(f"Detection class:     {det.__class__.__name__}")
        click.echo(f"Detection weights:   {det_path.name}  sha256(16)={sha(det_path)}")
        click.echo(f"Recognition threshold (config): {recognizer.threshold}")
        click.echo(f"Similarity metric: cosine (uniface.face_utils.compute_similarity)")

        click.echo()
        click.echo("=== Enrollment template integrity (no biometric values) ===")
        if not enrollment.is_enrolled():
            click.echo("No enrollment present.")
            tmpl = None
        else:
            f = enrollment.embedding_path.with_suffix('.npy')
            st = f.stat()
            tmpl = enrollment.load_enrollment()
            click.echo(f"File:              {f}")
            click.echo(f"File sha256(16):   {sha(f)}")
            click.echo(f"File mtime:        {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")
            click.echo(f"Template sha256(16):   {hashlib.sha256(np.ascontiguousarray(tmpl).tobytes()).hexdigest()[:16]}")
            click.echo(f"Shape:             {tmpl.shape}")
            click.echo(f"Dtype:             {tmpl.dtype}")
            click.echo(f"Finite:            {bool(np.isfinite(tmpl).all())}")
            click.echo(f"L2 norm:           {float(np.linalg.norm(tmpl)):.6f}")
            if tmpl.ndim != 1 or tmpl.shape[0] != 512 or tmpl.dtype != np.float32:
                click.echo("WARNING: unexpected template shape/dtype for MobileFace (expected float32 (512,))")

        if live_frames and tmpl is not None:
            device = config.get("camera.device", "/dev/video0")
            if not check_camera(device):
                click.echo("Camera unavailable, skipping live check.")
            else:
                click.echo()
                click.echo(f"=== Live similarity over {live_frames} frames ===")
                sims = []
                with Camera(device=device, fps=10) as cam:
                    i = 0
                    guard = 0
                    while len(sims) < live_frames and guard < live_frames * 10:
                        guard += 1
                        frame = cam.read()
                        if frame is None:
                            continue
                        try:
                            face = detector.validate_exactly_one_face(frame)
                        except DetectionError:
                            continue
                        try:
                            is_match, sim = recognizer.verify(frame, face.landmarks, tmpl)
                        except RecognitionError:
                            continue
                        w = float(face.bbox[2] - face.bbox[0])
                        h = float(face.bbox[3] - face.bbox[1])
                        sims.append(sim)
                        click.echo(f"frame {len(sims):3d}: bbox {w:.0f}x{h:.0f}px  sim={sim:.3f}  match={is_match}")
                        time.sleep(0.2)
                if sims:
                    a = np.array(sims)
                    click.echo(f"n={len(sims)} mean={a.mean():.3f} min={a.min():.3f} max={a.max():.3f} std={a.std():.3f}")

    except Exception as e:
        logger.error(f"Diagnose error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
