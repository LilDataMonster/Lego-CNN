FROM python:3.8-alpine

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./ /root/
WORKDIR /root/

CMD ["python", "samples/lego/lego.py", "--dataset=datasets/lego", "--logs=snapshots", "--enable-augmentation", "--weights=last", "--epochs=40"]
