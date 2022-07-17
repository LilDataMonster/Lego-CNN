
FROM nvcr.io/nvidia/tensorflow:22.06-tf2-py3

ARG TZ=America/New_York
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

RUN apt-get update -yq && apt-get install -yq \
        build-essential pkg-config software-properties-common \
        unzip curl gnupg ca-certificates libgl1-mesa-glx ffmpeg git vim \
        postgresql-client x11-apps \
        python3 python3-venv python3-pip python3-dev && \
    curl -L https://deb.nodesource.com/setup_16.x | bash && \
    apt-get update -yq && \
    apt-get install -yq \
        nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -s $(which python3) /usr/local/bin/python

# Create user
ENV USER=tf
RUN useradd -rm -p "$(openssl passwd -6 lildatamonster)" -d /home/${USER} -s /bin/bash -g root -G sudo -u 1000 ${USER}
USER ${USER}
WORKDIR /home/${USER}

# Create python venv and setup blenderproc
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    python -m venv /home/${USER}/tf_env && \
    source /home/${USER}/tf_env/bin/activate && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir blenderproc

# Initalize/install blender
RUN source /home/${USER}/tf_env/bin/activate && \
    blenderproc pip install && \
    echo "import blenderproc; blenderproc.init()" > /home/${USER}/init_blenderproc.py && \
    blenderproc run /home/${USER}/init_blenderproc.py && rm /home/${USER}/init_blenderproc.py

# Install python packages
COPY ./requirements.txt /
RUN source /home/${USER}/tf_env/bin/activate && \
    pip install --no-cache-dir -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116 && \
    jupyter lab build

ENV PATH=$PATH:/home/${USER}/blender/blender-3.1.0-linux-x64/

#CMD ["python", "samples/lego/lego.py", "--dataset=datasets/lego", "--logs=snapshots", "--enable-augmentation", "--weights=last", "--epochs=40"]
CMD jupyter lab --allow-root --ip 0.0.0.0 --port 8080 --no-browser
