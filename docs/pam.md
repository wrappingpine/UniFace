# UniFace Unlock - PAM Integration Guide

**WARNING**: Do NOT modify PAM until standalone system is fully tested and validated.

## Overview

This document outlines the safe approach for integrating face authentication with Pop!_OS 24.04 COSMIC Greeter PAM.

## Current State

Pop!_OS 24.04 uses COSMIC desktop with greetd for authentication. Investigation required before integration.

## Investigation Checklist

### 1. Determine Greeter Type
```bash
cat /etc/pam.d/cosmic-greeter
cat /etc/pam.d/common-auth
systemctl status cosmic-greeter
systemctl status greetd
```

Check if using:
- greetd
- PAM directly
- cosmic-greeter PAM configuration

### 2. Identify PAM Configuration
```bash
ls -la /etc/pam.d/
grep -r "pam_unix" /etc/pam.d/
grep -r "pam_succeed_if" /etc/pam.d/
```

### 3. Understand Authentication Flow
- Boot/login authentication
- Lock screen/unlock authentication
- Session management

## Safe Integration Strategy

### Phase 1: Standalone Service (CURRENT)
- Face authentication runs as standalone service
- CLI commands work independently
- No PAM modifications

### Phase 2: Helper Daemon (RECOMMENDED)
Create a small authentication helper that:
1. Accepts authentication requests from PAM
2. Communicates with face authentication service via IPC
3. Returns success/failure to PAM
4. Never loads ML models in PAM process

### Phase 3: PAM Integration (OPTIONAL)
If Phase 2 is sufficient, PAM config:
1. Keep pam_unix.so as primary
2. Add pam_exec.so for face auth as optional
3. Use pam_fail_delay for security
4. Ensure password fallback always available

## PAM Safety Rules

### DO NOT:
- Modify /etc/pam.d/common-auth without backup
- Remove password authentication
- Make face auth mandatory
- Load models in PAM process
- Disable existing auth methods

### ALWAYS:
- Backup PAM files first
- Test in VM or separate session
- Keep rollback script ready
- Maintain password fallback
- Log authentication attempts
- Provide recovery instructions

## Backup and Rollback

### Backup Script
```bash
#!/bin/bash
# scripts/backup-pam.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/pam-$TIMESTAMP"
mkdir -p "$BACKUP_DIR"
cp /etc/pam.d/* "$BACKUP_DIR/"
echo "PAM backed up to $BACKUP_DIR"
```

### Restore Script
```bash
#!/bin/bash
# scripts/restore-pam.sh
# Usage: restore-pam.sh <backup_dir>
cp "$1"/* /etc/pam.d/
```

## Testing PAM Integration

### Safe Testing Sequence
1. Test in VM first
2. Keep SSH access open
3. Test authentication in separate TTY
4. Have live USB ready
5. Test rollback procedure
6. Verify password auth still works

### Test Cases
- Successful face authentication
- Face authentication failure
- Camera disconnected
- Enrollment not found
- Liveness detection failure
- Multiple users
- Timeout scenarios
- Password fallback

## Lock Screen vs Login

**Important**: These are separate environments.

- **Login**: Authentication at boot/greeter
- **Lock Screen**: Authentication after session lock

Test both independently. Do not assume login works = lock screen works.

## Recommendation

**Wait for full standalone validation before PAM integration.**

Minimum validation criteria:
1. ✅ Camera works reliably
2. ✅ Detection works reliably
3. ✅ Enrollment works
4. ✅ Recognition works
5. ✅ Liveness works
6. ✅ Combined auth works
7. ✅ Benchmarks show acceptable performance
8. ✅ Security review complete
9. ✅ Tests pass
10. ✅ Documentation complete

**Estimated timeline**: 2-4 weeks of testing before PAM integration consideration.

## Emergency Recovery

If PAM is broken:
1. Boot to recovery mode
2. Restore from backup: `/var/backups/pam-*/`
3. Or use live USB to restore
4. Never leave system without password access

## Next Steps

1. Complete Stage 1-12 validation
2. Document test results
3. Review security implications
4. Create PAM integration proposal
5. Get explicit user approval
6. Execute with rollback plan ready
