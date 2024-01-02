
import boto3
import csv
import os
from owslib.wms import WebMapService
from datetime import datetime, timedelta
import time
import base64
import requests
import math
import json
from PIL import Image
import numpy as np
from io import BytesIO
import io
# Assuming fetch_model_inference and fetch_sequence_model_inference are defined as above
# Also assuming you have AWS credentials set up for boto3 to use

# S3 Configuration
s3 = boto3.client('s3')
bucket_name = 'austin-weather-prediction-metrics'
csv_file_name = 'weather_prediction_metrics.csv'


def download_from_s3(bucket, s3_file, local_file):
    try:
        s3.download_file(bucket, s3_file, local_file)
    except:
        print(f"File {s3_file} not found in bucket {bucket}. Creating a new one.")


# Existing function definitions...
def fetch_weather_data(date, max_retries=3, backoff_factor=2):
    # Read NOAA API Token from a file
    api_token = "bByvWkCzCbQwvXkaDoqXsGApUblGDdQE"

    # NOAA CDO URL
    base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'

    # Station for Austin, Texas Bergstrom Airport
    station_id = 'GHCND:USW00013904'
    headers = {
        'token': api_token
    }

    retries = 0
    while retries < max_retries:
        try:
            params = {
                'datasetid': 'GHCND',  # Global Historical Climatology Network Daily
                'stationid': station_id,
                'startdate': date.strftime('%Y-%m-%d'),
                'enddate': date.strftime('%Y-%m-%d'),
                'units': 'metric',
                'limit': 1000
            }

            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP error codes
            return response.json()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            retries += 1
            time.sleep(backoff_factor ** retries)  # Exponential backoff

    raise Exception(f"Failed to fetch data after {max_retries} retries.")
def fetch_sattelite_image(date):
    austin_bbox = (-98.2, 29.85, -97.4, 30.7)
    gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    wms = WebMapService(gibs_url)
    layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
    img_format = "image/jpeg"  # Format of the image to retrieve
    img_size = (512, 512)  # Size of the image (width, height)
    # TODO train the next model version to predict next day's high and low
    response = wms.getmap(layers=[layer],
                          styles=[''],
                          srs='EPSG:4326',
                          bbox=austin_bbox,
                          size=img_size,
                          format=img_format,
                          time=date.strftime("%Y-%m-%d"))

    image_bytes = BytesIO(response.read())
    image = Image.open(image_bytes)
    image.save("output.jpg", "JPEG")
    image_array = np.array(image)
    image_list = image_array.flatten().tolist()
    return image_list, image

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fetch_model_inference(image_list, date):
    day_sin = math.sin(2 * math.pi * date.day / 31)
    day_cos = math.cos(2 * math.pi * date.day / 31)
    month_sin = math.sin(2 * math.pi * date.month / 12)
    month_cos = math.cos(2 * math.pi * date.month / 12)

    image_list.extend([day_sin, day_cos, month_sin, month_cos])
    batch = {"data": {"ndarray": image_list}}
    endpoint = "http://weather-prediction-deployment-weather-regression-component:8000/api/v1.0/predictions"
    headers = {'Content-Type': 'application/json'}

    # Make the prediction by sending a POST request
    response = requests.post(endpoint, headers=headers, data=json.dumps(batch))
    content = json.loads(response.content.decode('utf-8'))
    # {'data': {'names': [], 'ndarray': [24.93512725830078, 8.416102409362793]}, 'meta': {'requestPath': {'weather-regression': 'undercoverducky/weatherpredictionmodel:0.0.5-amd64'}}}
    return content
def retrieve_sat_img(date):
    austin_bbox = (-98.2, 29.85, -97.4, 30.7)
    gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    wms = WebMapService(gibs_url)
    layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
    img_format = "image/jpeg"  # Format of the image to retrieve
    img_size = (512, 512)  # Size of the image (width, height)


    response = wms.getmap(layers=[layer],
                              styles=[''],
                              srs='EPSG:4326',
                              bbox=austin_bbox,
                              size=img_size,
                              format=img_format,
                              time=date.strftime("%Y-%m-%d"))

    encoded_image = base64.b64encode(response.read()).decode('utf-8')
    image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    return image
def fetch_sequence_model_inference(date):
    seq_len = 20
    image_list_sequence = []

    days_collected = 0
    while days_collected < seq_len:
        image = retrieve_sat_img(date)
        if image is not None:  # Check if image is available for the date
            image_array = np.array(image)
            image_list = image_array.flatten().tolist()

            day_sin = math.sin(2 * math.pi * date.day / 31)
            day_cos = math.cos(2 * math.pi * date.day / 31)
            month_sin = math.sin(2 * math.pi * date.month / 12)
            month_cos = math.cos(2 * math.pi * date.month / 12)
            date_features = [day_sin, day_cos, month_sin, month_cos]

            image_list.extend(date_features)
            image_list_sequence.extend(image_list)

            days_collected += 1

        # Move back one day
        date -= timedelta(days=1)
    # Prepare batch for the new endpoint
    batch = {"data": {"ndarray": image_list_sequence}}
    endpoint = "http://weather-seq-deployment-weather-seq-component:8000/api/v1.0/predictions"
    headers = {'Content-Type': 'application/json'}

    # Make the prediction by sending a POST request
    response = requests.post(endpoint, headers=headers, data=json.dumps(batch))
    content = json.loads(response.content.decode('utf-8'))
    return content
def append_to_csv(file_name, row):
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Date', 'Non-Sequential High', 'Non-Sequential Low', 'Sequential High', 'Sequential Low', 'Ground Truth High', 'Ground Truth Low'])  # Header
        writer.writerow(row)


def upload_to_s3(local_file, bucket, s3_file):
    s3.upload_file(local_file, bucket, s3_file)

def main():
    today_date = datetime.now() - timedelta(days=5)
    today_date_str = today_date.strftime('%Y-%m-%d')
    yesterday_date = today_date - timedelta(days=1)
    print(f"Collecting metrics for {today_date_str}")
    # Try to download the existing CSV file from S3
    download_from_s3(bucket_name, csv_file_name, csv_file_name)
    print(f"Downloaded metrics from s3")
    # Fetch the satellite image and model predictions
    image_list, _ = fetch_sattelite_image(yesterday_date)
    print("Fetched single image")
    non_seq_pred = fetch_model_inference(image_list, today_date)
    print("Fetched single date inference")
    seq_pred = fetch_sequence_model_inference(yesterday_date)
    print("Fetched transformer inference")
    # Extract predictions
    non_seq_high = round(celsius_to_fahrenheit(non_seq_pred['data']['ndarray'][0]))
    non_seq_low = round(celsius_to_fahrenheit(non_seq_pred['data']['ndarray'][1]))
    seq_high = round(celsius_to_fahrenheit(seq_pred['data']['ndarray'][-1][0]))
    seq_low = round(celsius_to_fahrenheit(seq_pred['data']['ndarray'][-1][1]))

    # Fetch ground truth data

    weather_data = fetch_weather_data(today_date)
    daily_high_temp = None
    daily_low_temp = None
    for item in weather_data.get('results', []):
        if item['datatype'] == 'TMAX':
            daily_high_temp = round(celsius_to_fahrenheit(float(item['value'])))
        if item['datatype'] == 'TMIN':
            daily_low_temp = round(celsius_to_fahrenheit(float(item['value'])))
        if daily_high_temp is not None and daily_low_temp is not None:
            break
    print("Fetched Ground Truth")
    # Append to CSV with ground truth
    row = [today_date_str, non_seq_high, non_seq_low, seq_high, seq_low, daily_high_temp, daily_low_temp]
    append_to_csv(csv_file_name, row)
    print("Appended to CSV")

    # Upload to S3
    upload_to_s3(csv_file_name, bucket_name, csv_file_name)
    print("Uploaded to s3")

def main():
    start_date = datetime.now() - timedelta(days=5)
    num_days = 30

    # Try to download the existing CSV file from S3
    download_from_s3(bucket_name, csv_file_name, csv_file_name)
    print("Downloaded metrics from s3")

    for day in reversed(range(num_days)):
        current_date = start_date - timedelta(days=day)
        current_date_str = current_date.strftime('%Y-%m-%d')
        print(f"Collecting metrics for {current_date_str}")

        # Fetch the satellite image and model predictions for the day before the current date
        image_list, _ = fetch_sattelite_image(current_date - timedelta(days=1))
        non_seq_pred = fetch_model_inference(image_list, current_date)
        seq_pred = fetch_sequence_model_inference(current_date - timedelta(days=1))

        # Extract predictions
        non_seq_high = round(celsius_to_fahrenheit(non_seq_pred['data']['ndarray'][0]))
        non_seq_low = round(celsius_to_fahrenheit(non_seq_pred['data']['ndarray'][1]))
        seq_high = round(celsius_to_fahrenheit(seq_pred['data']['ndarray'][-1][0]))
        seq_low = round(celsius_to_fahrenheit(seq_pred['data']['ndarray'][-1][1]))

        # Fetch ground truth data
        weather_data = fetch_weather_data(current_date)
        daily_high_temp = None
        daily_low_temp = None
        for item in weather_data.get('results', []):
            if item['datatype'] == 'TMAX':
                daily_high_temp = round(celsius_to_fahrenheit(float(item['value'])))
            if item['datatype'] == 'TMIN':
                daily_low_temp = round(celsius_to_fahrenheit(float(item['value'])))
            if daily_high_temp is not None and daily_low_temp is not None:
                break

        # Append to CSV with ground truth
        row = [current_date_str, non_seq_high, non_seq_low, seq_high, seq_low, daily_high_temp, daily_low_temp]
        append_to_csv(csv_file_name, row)
        print(f"Appended data for {current_date_str}")

    # Upload to S3 after collecting all data
    upload_to_s3(csv_file_name, bucket_name, csv_file_name)
    print("Uploaded to s3")

if __name__ == "__main__":
    main()
