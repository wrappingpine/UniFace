#!/usr/bin/env python3
"""
Self-test script for UniFace Unlock system.
Run this to test all components individually.

NOTE: interactive hardware self-test; run directly with
`python scripts/selftest.py`. Functions are named check_* (not test_*)
so pytest does not collect this manual script as part of the
automated suite.
"""

import sys
sys.path.insert(0, '/home/shubham/uniface-unlock/src')

import numpy as np
from uniface_unlock.camera import Camera
from uniface_unlock.detector import FaceDetector
from uniface_unlock.liveness import LivenessDetector
from uniface_unlock.recognizer import FaceRecognizer
from uniface_unlock.enrollment import FaceEnrollment
from uniface_unlock.authenticator import FaceAuthenticator

def check_camera():
    """Test camera access."""
    print("📸 Testing Camera...")
    try:
        with Camera(device='/dev/video0', resolution=(640, 480)) as cam:
            frame = cam.read()
            if frame is not None:
                print(f"   ✅ Camera working: {frame.shape}, {frame.dtype}")
                return frame
            else:
                print("   ❌ Failed to read frame")
                return None
    except Exception as e:
        print(f"   ❌ Camera error: {e}")
        return None

def check_detector(frame):
    """Test face detection."""
    print("\n🔍 Testing Face Detector...")
    try:
        detector = FaceDetector()
        faces = detector.detect(frame)
        print(f"   ✅ Detector initialized: {detector.__class__.__name__}")
        print(f"   Found {len(faces)} faces")
        
        if faces:
            face = faces[0]
            bbox = face.bbox
            print(f"   ✅ Face detected: bbox={bbox}")
            print(f"   Confidence: {face.confidence:.3f}")
            return face
        else:
            print("   ⚠️  No faces found")
            return None
    except Exception as e:
        print(f"   ❌ Detector error: {e}")
        return None

def check_liveness(frame, face):
    """Test liveness detection."""
    print("\n🛡️  Testing Liveness Detection...")
    try:
        liveness = LivenessDetector()
        status, confidence, details = liveness.check_liveness(
            frame, face.bbox, raw_output=True
        )
        print(f"   ✅ Liveness initialized")
        print(f"   Status: {status}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Real probability: {details['real_probability']:.3f}")
        print(f"   Fake probability: {details['fake_probability']:.3f}")
        return status == "REAL"
    except Exception as e:
        print(f"   ❌ Liveness error: {e}")
        return False

def check_recognition(frame, face):
    """Test face recognition."""
    print("\n👤 Testing Face Recognition...")
    try:
        enrollment = FaceEnrollment()
        if not enrollment.is_enrolled():
            print("   ⚠️  No face enrolled")
            return False
            
        recognizer = FaceRecognizer()
        embedding = enrollment.load_enrollment()
        
        is_match, similarity = recognizer.verify(
            frame, face.landmarks, embedding
        )
        
        print(f"   ✅ Recognizer initialized")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Match: {'✅ Yes' if is_match else '❌ No'}")
        return is_match
    except Exception as e:
        print(f"   ❌ Recognition error: {e}")
        return False

def check_full_authentication():
    """Test full authentication flow."""
    print("\n🔐 Testing Full Authentication...")
    try:
        auth = FaceAuthenticator()
        print(f"   ✅ Authenticator initialized")
        print(f"   Required frames: {auth.required_frames}")
        print(f"   Timeout: {auth.timeout_seconds}s")
        print("   (Full authentication requires camera interaction)")
        return True
    except Exception as e:
        print(f"   ❌ Authenticator error: {e}")
        return False

def main():
    print("="*70)
    print("UNIFACE UNLOCK - SELF TEST")
    print("="*70)
    
    # Test camera
    frame = check_camera()
    if frame is None:
        print("\n❌ Cannot continue without camera")
        sys.exit(1)
    
    # Test detector
    face = check_detector(frame)
    if face is None:
        print("\n⚠️  Continuing without face detection...")
    
    if face is not None:
        # Test liveness
        liveness_ok = check_liveness(frame, face)
        
        # Test recognition
        recognition_ok = check_recognition(frame, face)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Camera:        ✅ Working")
        print(f"Detector:      ✅ Working")
        print(f"Liveness:      {'✅ Working' if liveness_ok else '⚠️  Check lighting'}")
        print(f"Recognition:   {'✅ Working' if recognition_ok else '⚠️  May need calibration'}")
    
    # Test authenticator
    check_full_authentication()
    
    print("\n" + "="*70)
    print("✅ Self-test complete!")
    print("\nRun authentication with:")
    print("  uniface-unlock authenticate")
    print("="*70)

if __name__ == "__main__":
    main()
