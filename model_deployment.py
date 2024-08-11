from sagemaker import Session
from sagemaker.xgboost import XGBoost

# Define the session and S3 bucket
sagemaker_session = Session()
bucket = 'chris2223'

# Set up the estimator
xgb_estimator = XGBoost(entry_point='train.py',
                        role=role,
                        instance_count=1,
                        instance_type='ml.m4.xlarge',
                        framework_version='1.3-1',
                        output_path=f's3://{bucket}/output',
                        sagemaker_session=sagemaker_session)

# Train the model
xgb_estimator.fit({'train': 's3://your-bucket-name/path-to-your-data/train.csv'})

# Deploy the model
xgb_predictor = xgb_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
