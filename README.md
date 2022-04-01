# Introduction

# Quickstart

```bash
############################
### INSTALL DEPENDENCIES ###
############################
# install docker
curl https://get.docker.com | sh && sudo systemctl --now enable docker

# install nvidia-docker and docker-compose
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt install -y nvidia-docker2 docker-compose
sudo systemctl restart docker

# add current user to the docker group (for convience, restart to take effect) this will allow the user to not have to run docker commands with `sudo`
sudo usermod -aG docker $USER

# (optional) verify that the GPU can be seen with docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# clone repo
git clone git@github.com:LilDataMonster/Lego-CNN.git

############################
### RUN DOCKER SETUP #######
############################
# start up environment
cd Lego-CNN/
docker-compose up

# navigate to jupyterlab url provided by docker container
```

# Dependencies

The environment used for the project is developed using Docker to keep stability and consistency of the development environment across multiple machines. As a result the main requirement will be support for Docker, ideally on an amd64/x86_64 platform. The environment has been tested with linux, docker availability via WSL2 on windows may vary as support is still under quite heavy development.

- [`docker`](https://docs.docker.com/get-docker/)
- [`docker-compose`](https://docs.docker.com/compose/install/)
- [`nvidia-docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#)

# Docker Images

It's important to note that changes made within the docker container will not persist after shutting down the container. In order to persist data, a volume is recommended to be mounted to the container (which is performed within the docker-compose configuration).

### Building the docker image

Building the docker image can take quite a bit of time. The definition of the docker image is stored within the `environment/Dockerfile` file which contains the setup of the image itself. If the `Dockerfile` is modified, the image will need to be rebuilt.

To rebuild the docker image using docker-compose, run: `docker-compose build`. Note that the `docker-compose.yml` file must be defined to `build:` instead of `image` in order for the image to be built, otherwise it will pull a pre-built image from the container registry.

### Pulling the pre-built image from the container registry

Pre-built images will be stored on the github project's container registry which are pre-built images that can simply be downloaded and executed as docker containers. To pull the image, run `docker pull `. To run the image using docker-compose, simply run docker-compose as normal and it will pull the image if it is missing. 

### Docker Environment

The docker environment currently contains the following significant programs/libraries:

- Ubuntu 20.04 LTS
- Git
- Vim
- CUDA 11.3
- CUDNN 8.2.0.53-1
- NodeJS
- Ruby
- TensorRT
- Python 3.8 (Ubuntu 20.04 default)
- Python libraries:
  - jupyterlab
  - blenderproc
  - beautifulsoup4
  - ffmpeg-python
  - matplotlib
  - numpy
  - pandas
  - optuna
  - onnx
  - opencv
  - plotly
  - seaborn
  - scipy
  - sympy
  - tensorflow 2.5.3
  - tensorboard
  - tqdm
  - torch
  - torchvision
  - pillow
  - scikit-learn
  - fiftyone

# Running Docker Setup

The setup utilizes docker-compose which is an orchestration tool built ontop of docker. It's similar to Kubernetes without the super advanced bells and whistles. There really isn't a use for orchestration at the moment but it does provide some nice parameterization methods for running the image such as volume mounts, adding GPGPU support, setting environment variables, networking, etc.

To run the docker-compose setup, navigate to the location of the `docker-compose.yml` file and run `docker-compose up`. By default, the container will launch jupyterlab with the network attached to the host network.