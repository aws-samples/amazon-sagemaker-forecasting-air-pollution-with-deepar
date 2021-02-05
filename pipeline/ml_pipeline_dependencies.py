import uuid
import time
import boto3, json
import argparse
import os, urllib.request

import stepfunctions
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps.sagemaker import *
from stepfunctions.steps.states import *
from stepfunctions.workflow import Workflow
from stepfunctions.steps import *
from IPython.display import display, HTML, Javascript

import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.model import Model
from  sagemaker.transformer import Transformer

session = boto3.Session()
region = session.region_name
account_id = session.client('sts').get_caller_identity().get('Account')
bucket_name = f'openaq-forecasting-{account_id}-{region}'

sagemaker_session = sagemaker.Session()
role = get_execution_role()

S3_KEY_TRAINED_MODEL = "sagemaker/model/model.tar.gz"
EXISTING_MODEL_URI = f"s3://{bucket_name}/{S3_KEY_TRAINED_MODEL}"

def display_state_machine_advice(workflow_name, execution_id):
    display(HTML(f'''<br>The Step Function workflow "{workflow_name}" is now executing... 
            <br>To view state machine in the console click 
            <a target="_blank" href="https://{region}.console.aws.amazon.com/states/home?region={region}#/statemachines/view/arn:aws:states:ap-southeast-2:{account_id}:stateMachine:{workflow_name}">State Machine</a> 
            <br>To view execution in the console click 
            <a target="_blank" href="https://{region}.console.aws.amazon.com/states/home?region={region}#/executions/details/arn:aws:states:ap-southeast-2:{account_id}:execution:{workflow_name}:{execution_id}">Execution</a>.
        '''))

def display_training_job_advice(training_job_name):
    display(HTML(f'''<br>The training job "{training_job_name}" is now running. 
        To view it in the console click 
        <a target="_blank" href="https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs">here</a>.
    '''))  
    
def display_hpo_tuner_advice(hpo_job_name):
    display(HTML(f'''<br>The hyperparameter tuning job "{hpo_job_name}" is now running. 
            To view it in the console click 
            <a target="_blank" href="https://console.aws.amazon.com/sagemaker/home?region={region}#/hyper-tuning-jobs">here</a>.
        '''))

def display_processing_advice(processing_job_name):
    display(HTML(f'''<br>The processing job "{processing_job_name}" is now running. 
            To view it in the console click 
            <a target="_blank" href="https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/processing-jobs">here</a>.
        '''))


def setup_trained_model(bucket_name, s3_key_trained_model):
    # upload existing model artifact to working bucket
    s3 = boto3.client('s3')

    os.makedirs('model', exist_ok=True)
    urllib.request.urlretrieve('https://d8pl0xx4oqh22.cloudfront.net/model.tar.gz', 'model/model.tar.gz')
    s3.upload_file('model/model.tar.gz', bucket_name, s3_key_trained_model)
    
if __name__ == "__main__":
    setup_trained_model(bucket_name, S3_KEY_TRAINED_MODEL)