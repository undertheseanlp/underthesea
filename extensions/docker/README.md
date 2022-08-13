# Docker Image for Build Rust

Base from CentOS:7

Environments:

* gcc
* glibc 2.17
* Cargo
* Python: 3.6, 3.7, 3.8, 3.9, 3.10

# Usage

Update Dockerfile & Push to github registry

```
$ git checkout origin/core
$ git checkout -B core
$ git commit -m 'build docker'
$ git push origin core
```

**Note**: Commit messages contains 'build docker' will trigger `build-core-docker` actions.
