import os
import pandas as pd
import boto3
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import math
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import logging
import glob

# Define the custom dataset
class WeatherDataset(Dataset):
    def __init__(self, bucket_name, transform=None):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.transform = transform

        # Download CSV file from S3 and read it
        obj = self.s3.get_object(Bucket=bucket_name, Key='temperature_data.csv')
        self.temperature_data = pd.read_csv(BytesIO(obj['Body'].read()))

        # List all images in the bucket
        self.image_keys = [item['Key'] for item in self.s3.list_objects(Bucket=bucket_name).get('Contents', []) if item.get('Key', '').endswith('.jpg')]

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # Get image key
        image_key = self.image_keys[idx]

        # Download image from S3
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=image_key)
        image = Image.open(BytesIO(obj['Body'].read()))

        # Extract date from the image filename
        date_string = image_key.split('_')[-1].split('.')[0]

        date = datetime.strptime(date_string, '%Y-%m-%d')

        # Cyclical encoding
        day_sin = math.sin(2 * math.pi * date.day / 31)
        day_cos = math.cos(2 * math.pi * date.day / 31)
        month_sin = math.sin(2 * math.pi * date.month / 12)
        month_cos = math.cos(2 * math.pi * date.month / 12)

        # Find the corresponding temperature data
        temperature_row = self.temperature_data[self.temperature_data['Date'] == date_string]

        # Extract temperature data
        high_temp = temperature_row['High Temperature (°C)'].values[0]
        low_temp = temperature_row['Low Temperature (°C)'].values[0]

        # Apply transformations to the image if any
        if self.transform:
            image = self.transform(image)

        temp_tensor = torch.tensor([high_temp, low_temp], dtype=torch.float32)
        date_tensor = torch.tensor([day_sin, day_cos, month_sin, month_cos], dtype=torch.float32)

        return image, date_tensor, temp_tensor
class LocalWeatherDataset(Dataset):
    def __init__(self, local_dir, transform=None):
        self.local_dir = local_dir
        self.transform = transform

        temperature_data_dir = local_dir + 'temperatures/'
        csv_files = glob.glob(os.path.join(temperature_data_dir, 'temperature_data*.csv'))
        data_frames = [pd.read_csv(file) for file in csv_files]
        # Read CSV file from local directory
        self.temperature_data = pd.concat(data_frames, ignore_index=True)

        # List all images in the local directory
        self.image_keys = [file for file in os.listdir(os.path.join(local_dir, 'satellite_images/')) if file.endswith('.jpg')]


    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # Get image key
        image_key = self.image_keys[idx]

        # Read image from local directory
        image_path = os.path.join(os.path.join(self.local_dir, 'satellite_images/'), image_key)
        image = Image.open(image_path)

        # Extract date from the image filename
        date_string = image_key.split('_')[-1].split('.')[0]

        date = datetime.strptime(date_string, '%Y-%m-%d')

        # Cyclical encoding
        day_sin = math.sin(2 * math.pi * date.day / 31)
        day_cos = math.cos(2 * math.pi * date.day / 31)
        month_sin = math.sin(2 * math.pi * date.month / 12)
        month_cos = math.cos(2 * math.pi * date.month / 12)

        # Find the corresponding temperature data
        temperature_row = self.temperature_data[self.temperature_data['Date'] == date_string]

        if temperature_row.empty:
            print("wrong")

        # Extract temperature data
        high_temp = temperature_row['High Temperature (°C)'].values[0]
        low_temp = temperature_row['Low Temperature (°C)'].values[0]

        # Apply transformations to the image if any
        if self.transform:
            image = self.transform(image)

        temp_tensor = torch.tensor([high_temp, low_temp], dtype=torch.float32)
        date_tensor = torch.tensor([day_sin, day_cos, month_sin, month_cos], dtype=torch.float32)

        return image, date_tensor, temp_tensor

class LocalWeatherSequenceDataset(Dataset):
    def __init__(self, local_dir, seq_len, transform=None):
        self.local_dir = local_dir
        self.seq_len = seq_len
        self.transform = transform

        temperature_data_dir = local_dir + 'temperatures/'
        csv_files = glob.glob(os.path.join(temperature_data_dir, 'temperature_data*.csv'))
        data_frames = [pd.read_csv(file) for file in csv_files]
        self.temperature_data = pd.concat(data_frames, ignore_index=True)

        # List all images in the local directory and sort them by date
        image_files = [file for file in os.listdir(os.path.join(local_dir, 'satellite_images/')) if file.endswith('.jpg')]
        self.image_keys = sorted(image_files, key=lambda x: datetime.strptime(x.split('_')[-1].split('.')[0], '%Y-%m-%d'))

    def __len__(self):
        return len(self.image_keys) - self.seq_len # subtract 1 to account for using the next day instead of current

    def find_closest_temperature(self, target_date):
        max_diff_days = 15  # set a reasonable limit to how far you want to search
        for diff in range(1, max_diff_days + 1):
            # Check previous date
            prev_date = target_date - timedelta(days=diff)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            if prev_date_str in self.temperature_data['Date'].values:
                return self.temperature_data[self.temperature_data['Date'] == prev_date_str]

            # Check next date
            next_date = target_date + timedelta(days=diff)
            next_date_str = next_date.strftime('%Y-%m-%d')
            if next_date_str in self.temperature_data['Date'].values:
                return self.temperature_data[self.temperature_data['Date'] == next_date_str]

        return None  # No data found within the range

    def __getitem__(self, idx):
        images = []
        date_tensors = []
        temp_tensors = []

        for i in range(self.seq_len):
            image_key = self.image_keys[idx + i]
            image_path = os.path.join(self.local_dir, 'satellite_images/', image_key)
            image = Image.open(image_path)

            # Apply transformations to the image if any
            if self.transform:
                image = self.transform(image)

            date_string = image_key.split('_')[-1].split('.')[0]
            date = datetime.strptime(date_string, '%Y-%m-%d')
            next_day = date + timedelta(days=1)
            next_day_string = next_day.strftime('%Y-%m-%d')

            # Cyclical encoding for date
            day_sin = math.sin(2 * math.pi * date.day / 31)
            day_cos = math.cos(2 * math.pi * date.day / 31)
            month_sin = math.sin(2 * math.pi * date.month / 12)
            month_cos = math.cos(2 * math.pi * date.month / 12)

            temperature_row = self.temperature_data[self.temperature_data['Date'] == next_day_string]
            if temperature_row.empty:
                temperature_row = self.find_closest_temperature(next_day)
                if temperature_row is None:
                    raise ValueError(f"No close temperature data found for date: {next_day_string}")

            high_temp = temperature_row['High Temperature (°C)'].values[0]
            low_temp = temperature_row['Low Temperature (°C)'].values[0]

            images.append(torch.tensor(image, dtype=torch.float32))
            date_tensors.append(torch.tensor([day_sin, day_cos, month_sin, month_cos], dtype=torch.float32))
            temp_tensors.append(torch.tensor([high_temp, low_temp], dtype=torch.float32))

        image_tensor = torch.stack(images)
        date_tensor = torch.stack(date_tensors)
        temp_tensor = torch.stack(temp_tensors)

        return image_tensor, date_tensor, temp_tensor
def download_dataset_from_s3():
    bucket_name = 'austin-gibs-images'
    end_date = datetime.now() - timedelta(days=11)
    start_date = end_date - timedelta(days=365*4)

    s3 = boto3.client('s3')

    # Local directories
    satellite_images_dir = './austin_weather_data/satellite_images'
    temperature_data_dir = './austin_weather_data/temperatures'

    # Create directories if they don't exist
    if not os.path.exists(satellite_images_dir):
        os.makedirs(satellite_images_dir)
    if not os.path.exists(temperature_data_dir):
        os.makedirs(temperature_data_dir)

    # Download satellite images
    current_date = start_date
    while current_date <= end_date:
        file_name = f"austin_satellite_{current_date.strftime('%Y-%m-%d')}.jpg"
        local_file_path = os.path.join(satellite_images_dir, file_name)
        try:
            s3.download_file(bucket_name, f'satellite_images/{file_name}', local_file_path)
        except ClientError as e:
            logging.error(f"Could not download {file_name}: {e}")
        current_date += timedelta(days=1)

    # Download temggggperature data

    file_name = f"temperature_data.csv"
    local_file_path = os.path.join(temperature_data_dir, file_name)
    try:
        s3.download_file(bucket_name, f'weather_data/{file_name}', local_file_path)
    except ClientError as e:
        logging.error(f"Could not download {file_name}: {e}")

    current_month = start_date
    while current_month <= end_date:
        file_name = f"temperature_data_{current_month.strftime('%b-%Y')}.csv"
        local_file_path = os.path.join(temperature_data_dir, file_name)
        try:
            s3.download_file(bucket_name, f'weather_data/{file_name}', local_file_path)
        except ClientError as e:
            logging.error(f"Could not download {file_name}: {e}")
        current_month += timedelta(days=32)
        current_month = current_month.replace(day=1)