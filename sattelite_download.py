import datetime
import matplotlib.pyplot as plt
from owslib.wms import WebMapService
import boto3
from io import BytesIO

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = "austin-gibs-images"  # Replace with your S3 bucket name

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

# Function to fetch and upload image
def fetch_and_upload_image(date):
    # Request the image
    response = wms.getmap(layers=[layer],
                          styles=[''],
                          srs='EPSG:4326',
                          bbox=austin_bbox,
                          size=img_size,
                          format=img_format,
                          time=date.strftime("%Y-%m-%d"))

    # Filename in S3
    file_name = f"austin_satellite_{date.strftime('%Y-%m-%d')}.jpg"

    # Upload to S3
    s3.upload_fileobj(BytesIO(response.read()), bucket_name, file_name)
    print(f"Uploaded {file_name}")

# Iterate through each day of the past 3 years
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=4*365)

current_date = start_date
num_uploaded = 0
while current_date <= end_date:
    try:
        fetch_and_upload_image(current_date)
        num_uploaded += 1
    except:
        pass
    current_date += datetime.timedelta(days=1)
print(f"finished uploading {num_uploaded} images")