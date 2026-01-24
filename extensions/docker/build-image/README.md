# Docker Image for Build Rust

Base from AlmaLinux:8 (supports both x86_64 and aarch64/ARM64)

Environments:

* gcc
* Cargo (Rust)
* Python: 3.10, 3.11, 3.12, 3.13, 3.14

# Steps to build docker

Rebase

```
git rebase origin/core
```

Update Dockerfile & Push to github registry

```
$ git checkout origin/core
$ git checkout -B core
$ git commit -m 'build docker'
$ git push origin core
```

**Note**: Commit messages contains 'build docker' will trigger `build-core-docker` actions.

# Usage

```
docker run -it ghcr.io/undertheseanlp/underthesea/build_rust:0.0.1a13 bash
```
