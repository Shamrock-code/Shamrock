# Shamrock install (Docker)

## Install docker

=== "Linux (debian)"

    See [Docker documentation](https://docs.docker.com/engine/install/debian/#installation-methods).

    For convenience you can add your user to the docker group (to avoid having to use sudo everytime).
    See [post installation steps](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).
    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```

=== "MacOS"

    ```bash
    brew install --cask docker
    open /Applications/Docker.app
    ```

## Starting Shamrock docker container

```bash
docker run -it --platform=linux/amd64 ghcr.io/shamrock-code/shamrock:latest-oneapi
```
