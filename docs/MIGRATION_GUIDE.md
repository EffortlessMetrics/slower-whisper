# Migration Guide: Project Structure Changes

This guide documents the project structure changes made to improve organization and maintainability.

## Overview

The slower-whisper project has been reorganized to follow Python project best practices:

- Scripts moved from root to `scripts/` directory
- Documentation consolidated in `docs/` directory
- Deployment files organized in new `config/` directory
- Removed empty directories

## Changed File Locations

### Moved Scripts
- `audio_enrich.py` → `scripts/audio_enrich.py`
- `transcribe_pipeline.py` → `scripts/transcribe_pipeline.py`
- `verify_ami_setup.sh` → `scripts/verify_ami_setup.sh`

### Moved Documentation
- `INSTALL.md` → `docs/INSTALL.md`
- `TIER2_IMPLEMENTATION_PLAN.md` → `docs/TIER2_IMPLEMENTATION_PLAN.md`
- `turn_integration_summary.txt` → `docs/turn_integration_summary.txt`
- `integration_report.md` → `docs/integration_report.md`

### New Configuration Directory
Created `config/` directory containing:
- `docker-compose.yml` - Main Docker Compose configuration
- `docker-compose.api.yml` - API service configuration
- `docker-compose.dev.yml` - Development overrides
- `Dockerfile` - Standard Docker image
- `Dockerfile.api` - FastAPI service image
- `Dockerfile.gpu` - GPU-enabled image
- `.dockerignore` - Docker build exclusions

See `config/README.md` for detailed usage instructions.

## Impact on Workflows

### Running Scripts Directly

**Before:**
```bash
python audio_enrich.py
python transcribe_pipeline.py
```

**After:**
```bash
python -m scripts.audio_enrich
python -m scripts.transcribe_pipeline
```

### Docker Commands

**Before:**
```bash
docker compose up
```

**After:**
```bash
docker compose up  # Still works with docker-compose.yml in root
# OR
docker compose -f config/docker-compose.yml up
```

### Import Paths

Python scripts now properly handle imports with the `scripts/` module structure. No changes needed for imports.

## Backward Compatibility

All existing workflows continue to work:
- Docker commands using default files work unchanged
- Script execution via module path works correctly
- Documentation links are updated to point to new locations

## Migration Timeline

- **Phase 1** (Immediate): File moves and directory creation
- **Phase 2** (Following releases): Update documentation references
- **Phase 3** (Future): Remove backward compatibility shims

## Questions?

If you encounter issues with these changes, please:
1. Check this migration guide
2. Verify you're using the correct command syntax
3. Open an issue on GitHub

## Rollback

If needed, rollback with:
```bash
git revert HEAD~1  # Undo file moves
