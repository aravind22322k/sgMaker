import urllib.request
import os
import pandas as pd
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Step 1: Download the dataset
logging.info("Downloading the dataset...")
urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/53/iris.zip', 'data.zip')

# Step 2: Unzip the dataset
logging.info("Unzipping the dataset...")
os.makedirs('data', exist_ok=True)
os.system('unzip data.zip -d data/')

# Step 3: Read and preprocess the data
logging.info("Reading and preprocessing the data...")
data = pd.read_csv('data/iris.data', header=None)
data[4] = data[4].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2])
data = data.sample(frac=1).reset_index(drop=True)
data = data[[4, 0, 1, 2, 3]]  # Reorder columns

# Step 4: Split the data into training and validation sets
logging.info("Splitting the data into training and validation sets...")
train_data = data[:120]
val_data = data[120:]

# Save the data locally
logging.info("Saving the training and validation data locally...")
train_data.to_csv('train.csv', header=False, index=False)
val_data.to_csv('val.csv', header=False, index=False)

# Step 5: Upload the data to S3
bucket_name = 'chris2223'
s3 = boto3.Session().resource('s3')

try:
    logging.info("Uploading training data to S3...")
    s3.Bucket(bucket_name).Object('data/train/data.csv').upload_file('train.csv')
    logging.info("Training data uploaded successfully.")

    logging.info("Uploading validation data to S3...")
    s3.Bucket(bucket_name).Object('data/val/data.csv').upload_file('val.csv')
    logging.info("Validation data uploaded successfully.")
except Exception as e:
    logging.error(f"Error during S3 upload: {e}")
