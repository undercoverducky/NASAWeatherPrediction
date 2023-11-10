import boto3
import datetime
import requests
import csv
from owslib.wms import WebMapService
from io import BytesIO
import time

s3 = boto3.client('s3')
bucket_name = "austin-gibs-images"

end_date = datetime.datetime.now() - datetime.timedelta(days=10)
start_date = end_date - datetime.timedelta(days=31)
def fetch_and_upload_sattelite_images():
      # Replace with your S3 bucket name
    # Define geographic coordinates for Austin, Texas
    austin_bbox = (-98.2, 29.85, -97.4, 30.7)  # (west, south, east, north)

    # URL for NASA GIBS WMS service
    gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

    # Connect to GIBS WMS service
    wms = WebMapService(gibs_url)

    # Choose a layer
    layer = "MODIS_Terra_CorrectedReflectance_TrueColor"

    # Define image parameters
    img_format = "image/jpeg"  # Format of the image to retrieve
    img_size = (512, 512)  # Size of the image (width, height)

    # Iterate through each day of the past month


    current_date = start_date
    num_uploaded = 0
    while current_date <= end_date:
        max_attempts = 3
        attempt = 0
        backoff_factor = 2
        wait_time = 1  # initial wait time in seconds
        while attempt < max_attempts:
            try:
                response = wms.getmap(layers=[layer],
                                      styles=[''],
                                      srs='EPSG:4326',
                                      bbox=austin_bbox,
                                      size=img_size,
                                      format=img_format,
                                      time=current_date.strftime("%Y-%m-%d"))

                # Filename in S3
                file_name = f"satellite_images/austin_satellite_{current_date.strftime('%Y-%m-%d')}.jpg"

                # Upload to S3
                s3.upload_fileobj(BytesIO(response.read()), bucket_name, file_name)
                print(f"Uploaded {file_name}")
                num_uploaded += 1
                break
            except Exception as e:
                attempt += 1
                if attempt < max_attempts:
                    time.sleep(wait_time)
                    wait_time *= backoff_factor  # Exponential backoff
                else:
                    print(f"Failed image pull and upload with exception {e}")
                    pass
        current_date += datetime.timedelta(days=1)
    print(f"finished uploading {num_uploaded} images")

def fetch_weather_data(date, max_retries=3, backoff_factor=2):
    api_token = "bByvWkCzCbQwvXkaDoqXsGApUblGDdQE"
    headers = {
        'token': api_token
    }
    base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
    station_id = 'GHCND:USW00013904'

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
            pass

    print(f"Failed to fetch weather data after {max_retries} retries.")
def upload_weather_data():

    csv_file = f'temperature_data_{end_date.strftime("%b-%Y")}.csv'
    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'High Temperature (°C)', 'Low Temperature (°C)'])

        # Iterate through the date range and fetch data
        current_date = start_date
        while current_date <= end_date:
            data = fetch_weather_data(current_date)

            # Find the daily high and low temperatures
            daily_high_temp = None
            daily_low_temp = None
            for item in data.get('results', []):
                if item['datatype'] == 'TMAX':
                    daily_high_temp = item['value']
                if item['datatype'] == 'TMIN':
                    daily_low_temp = item['value']
                if daily_high_temp is not None and daily_low_temp is not None:
                    break

            # Write data to CSV
            writer.writerow([current_date.strftime('%Y-%m-%d'), daily_high_temp, daily_low_temp])

            # Print status
            print(f"Data for {current_date.strftime('%Y-%m-%d')} written to CSV.")

            # Move to the next day
            current_date += datetime.timedelta(days=1)

    s3_key = f'weather_data/{csv_file}'  # The path and file name in the S3 bucket

    try:
        s3.upload_file(csv_file, bucket_name, s3_key)
        print(f"File {csv_file} uploaded to {bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")

fetch_and_upload_sattelite_images()
upload_weather_data()