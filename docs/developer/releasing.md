# Releasing

This document describes the release process for Underthesea.

## Version Scheme

Underthesea follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 9.0.0)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Release Checklist

### Pre-release

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Version bumped in `__init__.py`

### Release Steps

1. **Update version**

   ```python
   # underthesea/__init__.py
   __version__ = "9.1.0"
   ```

   ```toml
   # pyproject.toml
   [project]
   version = "9.1.0"
   ```

2. **Update CHANGELOG**

   Add entry to `docs/history.md`:

   ```markdown
   ## 9.1.0 (2024-XX-XX)

   ### New Features
   - Added X feature

   ### Bug Fixes
   - Fixed Y issue

   ### Documentation
   - Updated Z docs
   ```

3. **Run tests**

   ```bash
   tox -e lint
   tox -e core
   ```

4. **Create release commit**

   ```bash
   git add -A
   git commit -m "Release version 9.1.0"
   git tag v9.1.0
   ```

5. **Push to GitHub**

   ```bash
   git push origin main
   git push origin v9.1.0
   ```

6. **Publish to PyPI**

   GitHub Actions will automatically publish when a tag is pushed.

   Or manually:

   ```bash
   uv pip install build twine
   python -m build
   twine upload dist/*
   ```

7. **Create GitHub Release**

   - Go to [Releases](https://github.com/undertheseanlp/underthesea/releases)
   - Click "Draft a new release"
   - Select the tag
   - Add release notes
   - Publish

## Versioning Guidelines

### When to bump MAJOR

- Removing a public function
- Changing function signatures
- Dropping Python version support
- Breaking changes to output format

### When to bump MINOR

- Adding new functions
- Adding new parameters (with defaults)
- New model support
- New optional features

### When to bump PATCH

- Bug fixes
- Performance improvements
- Documentation updates
- Dependency updates

## Hotfix Process

For urgent bug fixes:

1. Create branch from latest release tag:
   ```bash
   git checkout -b hotfix/9.0.1 v9.0.0
   ```

2. Fix the bug and add tests

3. Bump PATCH version

4. Merge to main and tag:
   ```bash
   git checkout main
   git merge hotfix/9.0.1
   git tag v9.0.1
   git push origin main v9.0.1
   ```

## Release Automation

GitHub Actions workflow (`.github/workflows/publish.yml`):

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

## Post-release

- [ ] Verify package on PyPI
- [ ] Test installation: `pip install underthesea==9.1.0`
- [ ] Update ReadTheDocs if needed
- [ ] Announce on social media
- [ ] Close related GitHub issues
