#FROM python:3.9-alpine
FROM tensorflow/tensorflow:2.5.1-gpu-jupyter

RUN apt-get update -yq && apt-get install -yq curl gnupg ca-certificates libgl1-mesa-glx ffmpeg \
    && curl -L https://deb.nodesource.com/setup_16.x | bash \
    && apt-get update -yq \
    && apt-get install -yq \
        dh-autoreconf \
        ruby \
        ruby-dev \
        nodejs \
    && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN jupyter labextension install jupyterlab-plotly@4.14.3 \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3 \
    && jupyter lab build

#COPY ./ /root/
#WORKDIR /root/

#CMD ["python", "samples/lego/lego.py", "--dataset=datasets/lego", "--logs=snapshots", "--enable-augmentation", "--weights=last", "--epochs=40"]
CMD jupyter lab --allow-root --ip 0.0.0.0 --port 8080 --no-browser
