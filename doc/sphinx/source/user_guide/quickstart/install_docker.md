# Shamrock install (Docker)

## Install docker

::::{tab-set}
:::{tab-item} Linux (debian)

See [Docker documentation](https://docs.docker.com/engine/install/debian/#installation-methods).

For convenience you can add your user to the docker group (to avoid having to use sudo everytime).
See [post installation steps](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

:::
:::{tab-item} MacOS

```bash
brew install --cask docker
open /Applications/Docker.app
```

:::
::::

## Starting Shamrock docker container

Shamrock CI automatically generate the `ghcr.io/shamrock-code/shamrock:latest-oneapi` docker
container with the last commit. It is convenient if you want to test it right away
but do not want to clone/compile and so on...

Note however that while convenient it prevents you from displaying windows, so a jupyter
notebook, terminal apps, and normal runs are fine but you can not display a matplotlib window this way.

```bash
docker run -it --platform=linux/amd64 ghcr.io/shamrock-code/shamrock:latest-oneapi
```
