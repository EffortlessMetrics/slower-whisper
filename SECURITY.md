# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.0.x   | :white_check_mark: | Current stable release |
| < 1.0   | :x:                | Pre-release, not supported |

**Recommendation:** Always use the latest stable release for security fixes.

---

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow responsible disclosure practices.

### How to Report

**DO NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security issues via one of these methods:

1. **GitHub Security Advisories (Preferred):**
   - Go to the [Security tab](https://github.com/EffortlessMetrics/slower-whisper/security)
   - Click "Report a vulnerability"
   - Fill out the private advisory form

2. **Email (if GitHub Security Advisories unavailable):**
   - Create an issue on the repository with minimal details
   - Tag as "security" if not sensitive
   - For sensitive issues, contact via GitHub profile email

### What to Include

Please provide as much information as possible:

- **Description:** Clear description of the vulnerability
- **Impact:** What an attacker could achieve
- **Steps to Reproduce:** Detailed reproduction steps
- **Affected Versions:** Which versions are affected
- **Proposed Fix:** If you have a suggestion (optional)
- **Disclosure Timeline:** Your preferred disclosure timeline

### Response Timeline

We aim to respond within:
- **24 hours:** Initial acknowledgment
- **7 days:** Assessment and severity classification
- **30 days:** Fix development and testing
- **60 days:** Public disclosure (coordinated with you)

### Severity Levels

We classify vulnerabilities using CVSS scores:

- **Critical (9.0-10.0):** Remote code execution, privilege escalation
- **High (7.0-8.9):** Data exposure, authentication bypass
- **Medium (4.0-6.9):** Denial of service, information disclosure
- **Low (0.1-3.9):** Minor information leaks, low-impact bugs

### Bounty Program

Currently, we do not offer a bug bounty program. However, we will:
- Acknowledge your contribution in release notes
- Credit you in the security advisory (if you wish)
- List you in our CONTRIBUTORS.md file

---

## Security Best Practices for Users

### 1. Installation Security

#### Verify Package Integrity

Always install from official sources:

```bash
# Install from PyPI (official)
pip install slower-whisper

# Verify package signature (future enhancement)
# pip install slower-whisper --require-hashes

# Install from source (verify git tag)
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper
git verify-tag v1.0.0  # Verify GPG signature
pip install -e .
```

#### Use Virtual Environments

Never install system-wide. Always use a virtual environment:

```bash
# Create isolated environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install within venv
pip install slower-whisper
```

---

### 2. Audio File Security

#### Trusted Sources Only

**⚠️ WARNING:** Only process audio files from trusted sources.

Malicious audio files could potentially:
- Exploit vulnerabilities in ffmpeg or audio codecs
- Contain metadata designed to trigger parsing bugs
- Exhaust system resources (very large files)

**Best Practices:**

```bash
# Scan files before processing (Linux/Mac)
clamscan audio_file.wav

# Check file size before processing
ls -lh audio_file.wav

# Verify file type
file audio_file.wav
```

#### File Size Limits

Recommended maximum file sizes:
- **Raw audio:** 2GB per file
- **Normalized audio:** 1GB per file
- **Total batch:** 10GB

To check file sizes:

```bash
# Check single file
du -h audio.wav

# Check directory total
du -sh raw_audio/
```

---

### 3. Model Security

#### Trusted Model Sources

Models are automatically downloaded from:
- **Whisper models:** `Systran/faster-whisper-*` (HuggingFace)
- **Emotion models:** `audeering/*`, `ehcalabres/*` (HuggingFace)
- **Diarization models:** `pyannote/speaker-diarization-3.1` (HuggingFace)

**Security Measures:**
- Models downloaded via HTTPS
- Handled by trusted libraries (transformers, faster-whisper, pyannote.audio)
- Cached locally after first download

**Model Cache Locations:**

By default, slower-whisper configures:
- `SLOWER_WHISPER_CACHE_ROOT` (default: `~/.cache/slower-whisper`)
- `HF_HOME` → `$SLOWER_WHISPER_CACHE_ROOT/hf`
- `TORCH_HOME` → `$SLOWER_WHISPER_CACHE_ROOT/torch`
- `HF_HUB_CACHE` → `$HF_HOME/hub`

Specific model caches:
```bash
~/.cache/slower-whisper/
  ├── hf/          # HuggingFace hub cache (HF_HOME)
  ├── torch/       # PyTorch cache (TORCH_HOME)
  ├── whisper/     # Whisper model weights
  ├── emotion/     # Emotion recognition models
  └── diarization/ # Pyannote diarization models
```

**To verify and manage models:**

```bash
# Inspect cache locations and sizes
slower-whisper cache --show

# List downloaded HuggingFace models
ls -lh ~/.cache/slower-whisper/hf/hub/

# Clear specific cache (will re-download on next use)
slower-whisper cache --clear whisper
slower-whisper cache --clear emotion
slower-whisper cache --clear diarization

# Clear all caches
slower-whisper cache --clear all
```

**Custom cache location:**
```bash
# Set before running to override default location
export SLOWER_WHISPER_CACHE_ROOT=/secure/models/cache
slower-whisper transcribe
```

See [`docs/MODEL_CACHE.md`](docs/MODEL_CACHE.md) for detailed cache management and security considerations.

---

### 4. API Key Management (LLM Examples)

If using the LLM integration examples:

#### Environment Variables

**✅ CORRECT:**
```bash
# Set in environment
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run application
python examples/llm_integration/llm_integration_demo.py
```

**❌ NEVER DO THIS:**
```python
# ❌ WRONG - Hardcoded in code
api_key = "sk-abc123..."

# ❌ WRONG - Committed to git
with open("api_key.txt") as f:
    api_key = f.read()
```

#### .env Files

If using `.env` files (with python-dotenv):

```bash
# .env (NOT committed to git)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Ensure .env is in .gitignore:**
```gitignore
.env
.env.*
*.key
secrets.json
```

---

### 5. File System Security

#### Directory Permissions

Set appropriate permissions on working directories:

```bash
# Linux/Mac: Restrict access to user only
chmod 700 raw_audio/ input_audio/ whisper_json/ transcripts/

# Verify permissions
ls -ld raw_audio/
# Should show: drwx------ (700)
```

#### Sensitive Audio Data

If processing sensitive audio:

1. **Encryption at rest:**
   ```bash
   # Encrypt directory (Linux)
   sudo cryptsetup luksFormat /dev/sdX
   sudo cryptsetup open /dev/sdX encrypted
   sudo mkfs.ext4 /dev/mapper/encrypted
   sudo mount /dev/mapper/encrypted /mnt/secure
   ```

2. **Secure deletion:**
   ```bash
   # Securely delete after processing (Linux)
   shred -vfz -n 3 audio_file.wav

   # On Mac
   rm -P audio_file.wav
   ```

3. **Memory security:**
   - Transcription happens in memory
   - Python doesn't guarantee memory wiping
   - Restart application after processing sensitive data

---

### 6. Network Security

#### Offline Operation

For maximum security, run completely offline:

```bash
# 1. Download models on trusted machine
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"

# 2. Copy cache to offline machine
rsync -av ~/.cache/huggingface/ offline-machine:~/.cache/huggingface/

# 3. Disconnect network and run transcription
# (Models will be used from cache)
```

#### Firewall Configuration

If running online, restrict network access:

```bash
# Linux: Allow only HuggingFace
sudo ufw allow out to huggingface.co port 443

# Block all other outbound (except DNS)
sudo ufw default deny outgoing
```

---

### 7. Dependency Security

#### Regular Updates

Keep dependencies updated:

```bash
# Check for updates
pip list --outdated

# Update all dependencies
pip install --upgrade slower-whisper

# Upgrade with caution (test first)
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

#### Vulnerability Scanning

Scan for known vulnerabilities:

```bash
# Install security tools
pip install pip-audit safety

# Scan with pip-audit (official)
pip-audit

# Scan with safety
safety check

# Example output:
# ┌─────────────────────────────────────────────────────────────┐
# │ Found 2 known vulnerabilities in 1 package                  │
# └─────────────────────────────────────────────────────────────┘
```

**Recommended Schedule:**
- **Weekly:** Automated scans in CI/CD
- **Monthly:** Manual review of results
- **Immediate:** After vulnerability announcements

---

### 8. Operational Security

#### Least Privilege

Run with minimal permissions:

```bash
# Create dedicated user (Linux)
sudo useradd -m -s /bin/bash transcriber
sudo -u transcriber slower-whisper

# Use containers for isolation
docker run --rm -v $(pwd):/data transcription-app
```

#### Process Isolation

Isolate from other processes:

```bash
# Use systemd for service isolation (Linux)
systemd-run --user --scope -p MemoryMax=8G slower-whisper

# Use nice for CPU priority
nice -n 19 slower-whisper  # Lower priority
```

#### Logging

Monitor for security events:

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
slower-whisper --verbose

# Monitor logs for errors
tail -f transcription.log | grep -i "error\|warn\|security"
```

---

## Known Security Considerations

### 1. GPL Dependencies

**praat-parselmouth** is licensed under GPL-3.0.

**Implications:**
- If you distribute modified versions, you must provide source code
- Linking in proprietary software may require compliance review
- Contact a lawyer if unsure about GPL obligations

**Alternatives:**
- Use librosa-only mode (no Parselmouth)
- Implement custom pitch extraction

### 2. Large Model Files

Models are large (2-6GB) and downloaded from HuggingFace.

**Risks:**
- Network interception (mitigated by HTTPS)
- Disk space exhaustion
- Supply chain attacks on model files

**Mitigations:**
- Models verified by transformers library
- Use trusted model sources only
- Monitor model checksums (future enhancement)

### 3. Local-Only Processing

**Good for Privacy:**
- No data sent to external services
- All processing happens on your machine
- You control your data

**Responsibilities:**
- Secure your own machine
- Manage your own backups
- Handle sensitive data appropriately

---

## Security Checklist

Before deploying in production:

- [ ] Install from verified PyPI package
- [ ] Use virtual environment
- [ ] Set up automated vulnerability scanning
- [ ] Configure appropriate file permissions
- [ ] Validate audio file sources
- [ ] Use environment variables for any API keys
- [ ] Enable logging and monitoring
- [ ] Test with non-sensitive data first
- [ ] Document your security procedures
- [ ] Train users on security best practices

---

## Incident Response

If you suspect a security breach:

1. **Contain:**
   - Stop the affected process
   - Isolate the machine from network
   - Preserve logs and evidence

2. **Assess:**
   - Determine scope of exposure
   - Identify affected data
   - Document timeline

3. **Remediate:**
   - Update to patched version
   - Rotate any exposed credentials
   - Review and update security controls

4. **Report:**
   - Notify affected parties
   - Report to security team
   - Document lessons learned

---

## Security Resources

### Documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Secure development practices
- [README.md](README.md) - Project overview and setup
- [docs/INDEX.md](docs/INDEX.md) - Complete documentation index

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [HuggingFace Security](https://huggingface.co/docs/hub/security)

### Tools
- [pip-audit](https://github.com/pypa/pip-audit) - Dependency vulnerability scanner
- [bandit](https://github.com/PyCQA/bandit) - Python security linter
- [safety](https://github.com/pyupio/safety) - Dependency checker
- [semgrep](https://semgrep.dev/) - Static analysis

---

## Security Contacts

- **GitHub Security:** https://github.com/EffortlessMetrics/slower-whisper/security/advisories
- **General Issues:** https://github.com/EffortlessMetrics/slower-whisper/issues
- **Maintainers:** See GitHub profile for contact options

---

## License

This security policy is part of the slower-whisper project and follows the same Apache 2.0 license.

**Last Updated:** 2025-11-17
**Policy Version:** 1.0
