# Docker Image for Build Rust

Base from CentOS:7

Environments:

* gcc
* glibc 2.17
* Cargo
* Python: 3.7, 3.8, 3.9, 3.10, 3.11

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
