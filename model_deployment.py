import sagemaker
from sagemaker import Session

# Set the region
region = 'us-west-2'  # Replace with your desired AWS region

# Initialize the SageMaker session with the specified region
sagemaker_session = Session(boto_session=sagemaker.Session(region_name=region))

# Rest of your deployment code
xgb_model = ...  # Your model definition or loading
xgb_predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    sagemaker_session=sagemaker_session
)
