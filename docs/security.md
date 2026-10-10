# Security Documentation

## Threat Model

### Assumptions
- Webcam is physically secure
- User is present at laptop
- No network access to face data
- Models are downloaded from trusted source (SHA-256 verified)
- Physical access to laptop requires authentication

### Threats Addressed

1. **Photo/video spoofing**: Mitigated by MiniFASNet liveness detection
2. **Unauthorized face enrollment**: Enrollment requires physical presence
3. **Embedding extraction**: Filesystem permissions protect enrollment data
4. **Replay attacks**: Consecutive frame requirement prevents replay
5. **Multiple faces**: Exactly one face requirement
6. **Poor quality samples**: Enrollment rejects low-quality samples
7. **Model evasion**: Fail-closed design

### Threats NOT addressed

1. **High-quality masks**: MiniFASNet may not detect sophisticated masks
2. **Deepfake injection**: No protection against direct video feed manipulation
3. **Insider threats**: Physical access to config files allows tampering
4. **Shoulder surfing**: Visual feedback during authentication
5. **Brute force**: No rate limiting built in

## Security Principles

### Fail Closed
- UNKNOWN liveness → reject
- Detection failure → reject
- Recognition failure → reject
- Camera failure → reject
- Timeout → reject

### Local Only
- No cloud APIs
- No network requests during inference
- All processing on-device
- Models cached locally

### Data Minimization
- Store embeddings only, never photos
- Embeddings are irreversible (one-way)
- No biometric metadata stored
- Configurable cleanup

### Defense in Depth
1. Exactly one face required
2. Liveness detection
3. Recognition threshold
4. Consecutive frames
5. Secure file permissions

## File Permissions

```
~/.config/uniface-unlock/
├── enrollment.npz (0o600)  # Only owner can read
├── config.json (0o600)
└── directory (0o700)
```

## Limitations

### MiniFASNet Limitations
- May fail with extreme lighting
- May fail with occlusions (masks, glasses)
- Uncertain with printed photos
- Video replay may be detected but not guaranteed

### Face Recognition Limitations
- Appearance changes (beard, glasses) affect matching
- Pose variations affect accuracy
- Lighting changes affect accuracy
- Threshold calibration required per user

### Hardware Limitations
- CPU inference is slower than GPU
- Thermal throttling possible
- Camera quality affects detection
- 5-10 FPS processing limit

## Security Best Practices

1. **Never disable password authentication**
2. **Keep UniFace models updated**
3. **Protect enrollment files**
4. **Test liveness regularly**
5. **Monitor logs for anomalies**
6. **Re-enroll if appearance changes significantly**
7. **Use strong password backup**
8. **Physical security of device**

## Incident Response

### If enrollment is compromised
```bash
uniface-unlock enroll  # Re-enroll
```

### If camera is compromised
- Use password authentication
- Check /dev/video* permissions
- Review system logs

### If authentication fails repeatedly
1. Check camera status
2. Verify enrollment exists
3. Check lighting conditions
4. Run benchmark
5. Re-enroll if needed

## PAM Integration Safety

**IMPORTANT**: Before modifying PAM:

1. Backup PAM files
2. Test standalone system thoroughly
3. Keep password auth enabled
4. Never make face auth mandatory
5. Provide rollback plan
6. Test recovery procedures

See `docs/pam.md` for PAM-specific security.
