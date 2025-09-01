# music_dl

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
 -v ${PWD}/output_samples:/workspace/output_samples `
music_dl_dev
```

### Deployment:

#### Build:

```sh
docker build -t music_dl -f Dockerfile .
```

#### Run:

```sh
docker run --rm -it --gpus all `
 -v ${PWD}/output_samples:/workspace/output_samples `
music_dl
```
