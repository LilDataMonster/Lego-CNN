#FROM python:3.9-alpine
FROM tensorflow/tensorflow:2.5.1-gpu-jupyter

COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y libgl1-mesa-glx ffmpeg \
    && rm -rf /var/lib/apt/lists/*

#COPY ./ /root/
#WORKDIR /root/

#CMD ["python", "samples/lego/lego.py", "--dataset=datasets/lego", "--logs=snapshots", "--enable-augmentation", "--weights=last", "--epochs=40"]
CMD jupyter lab --allow-root --ip 0.0.0.0 --port 8080 --no-browser
