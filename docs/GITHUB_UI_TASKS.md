# GitHub UI Tasks Checklist

This file contains tasks that must be completed manually via the GitHub web interface.

Repository metadata (description, topics, labels, branch protection) is now managed in [`.github/settings.yml`](../.github/settings.yml). Use this checklist only for UI-only actions that cannot be encoded in repository files.

## Priority 1: Critical (Do First)

### Fix GitHub Actions Billing Issue

**Status:** ⚠️ **BLOCKING** - CI/CD workflows are failing

**Issue:** GitHub Actions workflows are failing with billing error:
```
The job was not started because recent account payments have failed or your
spending limit needs to be increased. Please check the 'Billing & plans'
section in your settings
```

**Action Required:**
1. Go to: https://github.com/settings/billing
2. Check payment method and billing status
3. Verify GitHub Actions spending limit
4. Update payment information if needed
5. Increase spending limit if using free tier

**Impact:**
- All CI/CD workflows are blocked (lint, tests, docker builds)
- Cannot verify code quality automatically
- Cannot push Docker images to registry

---

## Priority 2: Release (After billing is fixed)

### Create GitHub Release for Current Stable Version

**Status:** 📋 Ready to execute

**Steps:**
1. Go to: https://github.com/EffortlessMetrics/slower-whisper/releases/new
2. **Tag:** latest version tag (for example, `v2.0.0`)
3. **Title:** `slower-whisper vX.Y.Z`
4. **Description:** Summarize from `CHANGELOG.md` and `docs/releases/RELEASE.md`
5. Check: ✅ **Set as the latest release**
6. Click **Publish release**

**Verification:**
- Check that release appears at: https://github.com/EffortlessMetrics/slower-whisper/releases
- Verify release badge shows the new tag
- Confirm download links work

---

## Priority 3: Repository Settings

### Update Repository About Section

**Status:** ✅ Managed as code

**Primary path:**
1. Edit [`.github/settings.yml`](../.github/settings.yml).
2. Apply settings via your configured settings automation.
3. Verify repo About section and topics on the main page.

**Manual fallback (if settings automation is unavailable):**
1. Go to: https://github.com/EffortlessMetrics/slower-whisper
2. Click **⚙️** next to **About**
3. Copy values from [`.github/settings.yml`](../.github/settings.yml)
4. Save changes

---

### Add ROADMAP.md Link to Repo Description

**Status:** 📋 Optional enhancement

**Steps:**
1. Edit the main README.md pinned section (if you have one)
2. Or add a notice at the top:
   ```markdown
   > 📋 See [ROADMAP.md](ROADMAP.md) for planned features and development timeline
   ```

**Alternative:**
- Pin ROADMAP.md as a GitHub Discussion or Project board

---

## Priority 4: Documentation & Community

### Set Up GitHub Discussions (Optional)

**Status:** 💭 Optional

**Steps:**
1. Go to: https://github.com/EffortlessMetrics/slower-whisper/settings
2. Scroll to **Features** section
3. Check: ✅ **Discussions**
4. Click **Set up discussions**
5. Create initial categories:
   - 💬 General
   - 🙏 Q&A
   - 💡 Ideas / Feature Requests
   - 📣 Announcements
   - 🐛 Bug Reports (if not using Issues)

**Benefits:**
- Community can discuss features before opening issues
- Q&A section reduces duplicate issues
- Centralized place for user conversations

---

### Enable Security Advisories

**Status:** ✅ Already enabled (referenced in SECURITY.md)

**Verification:**
- Check: https://github.com/EffortlessMetrics/slower-whisper/security/advisories
- Should show "Report a vulnerability" button

**If not enabled:**
1. Go to: https://github.com/EffortlessMetrics/slower-whisper/settings
2. Scroll to **Security** section
3. Check: ✅ **Private vulnerability reporting**

---

### Add Issue Templates

**Status:** 📋 Nice to have

**Steps:**
1. Go to: https://github.com/EffortlessMetrics/slower-whisper/issues/templates/edit
2. Add templates for:
   - 🐛 Bug Report
   - ✨ Feature Request
   - 📖 Documentation Improvement
   - ❓ Question

**Or use existing templates in `.github/ISSUE_TEMPLATE/` if present**

---

## Priority 5: CI/CD Configuration (After billing fix)

### Verify GitHub Actions After Billing Fix

**Status:** ⏳ Blocked by billing issue

**Steps:**
1. After fixing billing, manually trigger workflow:
   - Go to: https://github.com/EffortlessMetrics/slower-whisper/actions
   - Select "CI" workflow
   - Click **Run workflow** → **Run workflow**

2. Monitor workflow execution:
   - Verify all jobs complete successfully
   - Check that tests pass
   - Confirm Docker builds succeed

3. If failures occur:
   - Review logs for errors
   - Fix any path issues from recent reorganization
   - Re-run workflow

**Expected Results:**
- ✅ Lint (ruff check) - Pass
- ✅ Format check (ruff format) - Pass
- ✅ Type check (mypy) - Pass
- ✅ Test (Python 3.12, 3.13) - Pass
- ✅ Integration tests - Pass
- ✅ Docker builds - Pass (4 variants)

---

### Configure Branch Protection Rules

**Status:** 📋 Recommended for production

**Primary path:**
1. Keep [`.github/settings.yml`](../.github/settings.yml) updated.
2. Ensure your repository settings automation applies it.

**Manual fallback (if automation is unavailable):**
1. Go to: https://github.com/EffortlessMetrics/slower-whisper/settings/branches
2. Configure rule for `main` with:
   - Required status checks: `CI Success`, `Verify (quick)`
   - Require branches to be up to date
   - Require pull request with one approval
   - Require CODEOWNERS review
   - Apply to admins

**Benefits:**
- Prevents direct pushes to main
- Ensures all code passes CI before merge
- Forces code review process

---

## Completion Checklist

Mark items as you complete them:

**Priority 1: Critical**
- [ ] Fix GitHub Actions billing issue
- [ ] Verify CI/CD workflows run successfully

**Priority 2: Release**
- [ ] Create GitHub Release for current stable version
- [ ] Verify release appears correctly
- [ ] Test download links

**Priority 3: Repository Settings**
- [ ] Update About section (description + topics)
- [ ] Verify topics are visible
- [ ] (Optional) Add ROADMAP link to README

**Priority 4: Documentation**
- [ ] (Optional) Enable GitHub Discussions
- [ ] Verify Security Advisories are enabled
- [ ] (Optional) Add issue templates

**Priority 5: CI/CD**
- [ ] Manually trigger CI workflow after billing fix
- [ ] Verify all workflows pass
- [ ] (Optional) Configure branch protection rules

---

## Notes

- **Billing issue is blocking:** Fix this first before anything else
- **Release metadata:** Source from `CHANGELOG.md` + `docs/releases/RELEASE.md`
- **Workflows are configured correctly:** The failures are due to billing, not code
- **Repository is clean:** All code changes have been committed and pushed

---

## Support

If you encounter issues with any of these tasks:

1. Check GitHub's documentation: https://docs.github.com
2. Check GitHub Actions status: https://www.githubstatus.com
3. For billing: https://github.com/settings/billing
4. For repository settings: https://github.com/EffortlessMetrics/slower-whisper/settings

---

**Last Updated:** 2026-02-16
**Created By:** Automated during repository cleanup and polish phase
