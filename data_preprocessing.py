import urllib.request
import os
import pandas as pd
import boto3

import boto3

# Directly pass AWS credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id='your_access_key_id',
    aws_secret_access_key='your_secret_access_key'
)


# Step 1: Download the dataset
urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/53/iris.zip', 'data.zip')

# Step 2: Unzip the dataset
os.makedirs('data', exist_ok=True)
os.system('unzip data.zip -d data/')

# Step 3: Read and preprocess the data
data = pd.read_csv('iris.data', header=None)
data[4] = data[4].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2])
data = data.sample(frac=1).reset_index(drop=True)
data = data[[4, 0, 1, 2, 3]]  # Reorder columns

# Step 4: Split the data into training and validation sets
train_data = data[:120]
val_data = data[120:]

# Step 5: Upload the data to S3
bucket_name = 'sagemaker-build-and-deploy-model-sagemaker'
train_data.to_csv('train.csv', header=False, index=False)
val_data.to_csv('val.csv', header=False, index=False)

s3 = boto3.Session().resource('s3')
s3.Bucket(chris2223).Object('data/train/data.csv').upload_file('train.csv')
s3.Bucket(chris2223).Object('data/val/data.csv').upload_file('val.csv')
