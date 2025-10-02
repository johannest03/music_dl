# Music Transformer

This project uses the Piano Aria dataset for training and evaluation of music generation models.

## Structure

Following the provided commands following structure is expected.

root

-   src
-   output
-   datasets

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
  -v ${PWD}/output:/output `
  -v ${PWD}/datasets:/datasets `
  music_dl_dev
```
