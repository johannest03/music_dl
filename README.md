# Music Transformer

### Development:

The commands are for windows docker.

#### Build:

```sh
docker build -t music_dl_dev -f Dockerfile.dev .
```

#### Run:

```sh
docker run --rm -it --gpus all `
  -v ${PWD}/src:/workspace `
  -v ${PWD}/output_samples:/output_samples `
  -v ${PWD}/datasets:/datasets `
  music_dl_dev
```

#### Run tests:

```sh
docker run --rm -it --gpus all `
  -v ${PWD}/src:/workspace/src `
  -v ${PWD}/output_samples:/output_samples `
  -v ${PWD}/datasets:/datasets `
  -v ${PWD}/tests:/workspace/tests `
  music_dl_dev `
  bash -c "PYTHONPATH=/workspace/src pytest -v /workspace/tests"
```

### Deployment:

#### Build:

```sh
docker build -t music_dl -f Dockerfile .
```

#### Run:

```sh
docker run --rm -it --gpus all `
 -v ${PWD}/output_samples:/output_samples `
music_dl
```
