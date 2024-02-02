FROM python:3.7-alpine
WORKDIR /code
ADD object_detection.py /code
COPY requirements.txt requirements.txt
COPY uploads /code/uploads
COPY yolo-coco /code/yolo-coco
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt
EXPOSE 80/tcp
CMD ["python","/code/object_detection.py"]