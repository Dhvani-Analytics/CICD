FROM python:3.8

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy scipy scikit-learn scikit-image setuptools==41.0.0 gunicorn==19.9.0 gevent flask Pillow nvidia-pyindex tritonclient[all] attrdict boto3 sagemaker flask-cors torch==1.8.0 torchvision==0.9.0 && \
        rm -rf /root/.cache

RUN pip install pycocotools-fix

#set a directory for the app
WORKDIR /usr/src/app

#copy all the files to the container
COPY . .

# tell the port number the container should be expose

EXPOSE 5000

# run the command
CMD ["python", "./app.py"]

