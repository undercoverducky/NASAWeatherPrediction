#
FROM python:3.11
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./collect_metrics.py /code/collect_metrics.py
COPY ./collect_30_days_metrics.py /code/collect_30_days_metrics.py
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


