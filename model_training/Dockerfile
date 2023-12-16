#
FROM python:3.11
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./train_weather_predictor.py /code/train_weather_predictor.py
COPY ./models.py /code/models.py
COPY ./weather_dataset.py /code/weather_dataset.py

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#RUN pip install --upgrade -r /code/requirements.txt




