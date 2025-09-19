# Underthesea Core Documentation

## Release Workflow

1. Change version in `Cargo.toml` and `pyproject.toml`
2. Push to branch `core` with commit `Publish Underthesea Core`
  * This will trigger `release-pypi-core` action
3. Check latest version in [pypi](https://pypi.org/project/underthesea_core/)

Note*: Run a self-hosted for building `macos-arm`