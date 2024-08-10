import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker import get_execution_role

bucket_name = 'sagemaker-build-and-deploy-model-sagemaker'
train_data = f's3://chris2223/data/train/data.csv'
val_data = f's3://chris2223/data/val/data.csv'

s3_output_location = f's3://chris2223/model/xgb_model'

xgb_model = sagemaker.estimator.Estimator(
    get_image_uri(boto3.Session().region_name, 'xgboost'),
    get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session()
)

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

train_channel = sagemaker.inputs.TrainingInput(train_data, content_type='text/csv')
val_channel = sagemaker.inputs.TrainingInput(val_data, content_type='text/csv')

xgb_model.fit({'train': train_channel, 'validation': val_channel})
