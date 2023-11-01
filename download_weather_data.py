import requests
import csv
from datetime import datetime, timedelta
import time

# Read NOAA API Token from a file
with open('NOAA_api_key.txt', 'r') as file:
    api_token = file.read().strip()

# NOAA CDO URL
base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'

# Station for Austin, Texas Bergstrom Airport
station_id = 'GHCND:USW00013904'

# Desired date range (modify as needed)
start_date = datetime.now() - timedelta(days=4*365)  # approximately 4 years ago
end_date = datetime.now()

# Prepare headers
headers = {
    'token': api_token
}

# Function to fetch data for a specific date
# Function to fetch data for a specific date with retry behavior
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
# CSV file to store data
csv_file = 'temperature_data.csv'

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'High Temperature (°C)', 'Low Temperature (°C)'])

    # Iterate through the date range and fetch data
    current_date = start_date
    while current_date <= end_date:
        data = fetch_data(current_date)

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
        current_date += timedelta(days=1)