import boto3
import sagemaker
from sagemaker import image_uris

# Set environment variables (if needed)
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA2UC3CX7ON4KNN4YV'
os.environ['AWS_SECRET_ACCESS_KEY'] = '1dlP7hdulAkoAw19RJIG0tmeVId1flV6mkgQJ6T6'

# Explicitly set the AWS region
region = boto3.Session().region_name
if region is None:
    region = 'us-east-1'  # Set your default region here

# Create a SageMaker session with the specified region
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

# Bucket and data paths
bucket_name = 'chris2223'
train_data = f's3://{bucket_name}/data/train/data.csv'
val_data = f's3://{bucket_name}/data/val/data.csv'
s3_output_location = f's3://{bucket_name}/model/xgb_model'

# Retrieve the image URI for XGBoost
xgboost_image = image_uris.retrieve(framework='xgboost', region=region, version='latest')

# Manually set the role ARN (if needed)
role_arn = 'arn:aws:iam::your-account-id:role/your-role-name'

# Create the XGBoost Estimator
xgb_model = sagemaker.estimator.Estimator(
    xgboost_image,
    role_arn,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker_session
)

# Set hyperparameters
xgb_model.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    silent=0,
    objective='multi:softmax',
    num_class=3,
    num_round=10
)

# Define input channels
train_channel = sagemaker.inputs.TrainingInput(train_data, content_type='text/csv')
val_channel = sagemaker.inputs.TrainingInput(val_data, content_type='text/csv')

# Train the model
xgb_model.fit({'train': train_channel, 'validation': val_channel})
