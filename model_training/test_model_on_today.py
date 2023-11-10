from model_training import models
from owslib.wms import WebMapService
from io import BytesIO
from PIL import Image
from torchvision import transforms
import requests
from datetime import datetime, timedelta
import time
import math
import torch
# Read NOAA API Token from a file
with open('../data_collection/NOAA_api_key.txt', 'r') as file:
    api_token = file.read().strip()

# NOAA CDO URL
base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'

# Station for Austin, Texas Bergstrom Airport
station_id = 'GHCND:USW00013904'
headers = {
    'token': api_token
}
def fetch_data(date, max_retries=3, backoff_factor=2):
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

austin_bbox = (-98.2, 29.85, -97.4, 30.7)
gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
wms = WebMapService(gibs_url)
layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
img_format = "image/jpeg"  # Format of the image to retrieve
img_size = (512, 512)  # Size of the image (width, height)
date = datetime.now() - timedelta(days=20)

response = wms.getmap(layers=[layer],
                          styles=[''],
                          srs='EPSG:4326',
                          bbox=austin_bbox,
                          size=img_size,
                          format=img_format,
                          time=date.strftime("%Y-%m-%d"))
image_bytes = BytesIO(response.read())
image = Image.open(image_bytes)

# Convert the image to a PyTorch tensor
transform = transforms.ToTensor()
tensor = transform(image)

data = fetch_data(date)

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

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

print(celsius_to_fahrenheit(daily_high_temp))

print(celsius_to_fahrenheit(daily_low_temp))
day_sin = math.sin(2 * math.pi * date.day / 31)
day_cos = math.cos(2 * math.pi * date.day / 31)
month_sin = math.sin(2 * math.pi * date.month / 12)
month_cos = math.cos(2 * math.pi * date.month / 12)
temp_tensor = torch.tensor([daily_high_temp, daily_low_temp], dtype=torch.float32)
date_tensor = torch.tensor([day_sin, day_cos, month_sin, month_cos], dtype=torch.float32)
print(tensor)
current_time = "Nov-01" #datetime.now().strftime('%b-%d')
model = models.load_model([32, 64], current_time)
logits = model(tensor.unsqueeze(0), date_tensor.unsqueeze(0))

pred_high, pred_low = celsius_to_fahrenheit(logits[0][0]), celsius_to_fahrenheit(logits[0][1])
print((pred_high, pred_low) )
