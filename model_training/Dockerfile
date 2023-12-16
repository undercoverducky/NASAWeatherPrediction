#
FROM python:3.11
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./collect_training_data.py /code/collect_training_data.py
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


