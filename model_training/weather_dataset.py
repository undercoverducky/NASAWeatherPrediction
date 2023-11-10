import os
import pandas as pd
import boto3
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import math
import datetime

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

        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')

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

        # Read CSV file from local directory
        self.temperature_data = pd.read_csv(os.path.join(local_dir, '../temperature_data.csv'))

        # List all images in the local directory
        self.image_keys = [file for file in os.listdir(local_dir) if file.endswith('.jpg')]


    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # Get image key
        image_key = self.image_keys[idx]

        # Read image from local directory
        image_path = os.path.join(self.local_dir, image_key)
        image = Image.open(image_path)

        # Extract date from the image filename
        date_string = image_key.split('_')[-1].split('.')[0]

        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')

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
